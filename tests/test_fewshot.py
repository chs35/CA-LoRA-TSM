"""The 10-image, tensor-core training path: subset pinning, precision, latent crop."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from calora.config import DataConfig
from calora.data import apply_training_subset, build_dataset
from calora.models.projections import ProjectionTable
from calora.train.lora import lora_parameters
from catsm.cli import restrict_to_training_subset
from catsm.config import CatsmConfig
from catsm.intervals import Interval
from catsm.models.router import Router
from catsm.train.foster import train_expert
from conftest import ROOT, make_adapter, random_head_selection, train_cfg


def dummy_dataset(items: int = 8):
    cfg = DataConfig(
        kind="dummy",
        num_dummy_items=items,
        augmentation_file=str(ROOT / "configs" / "prompt_augmentations.json"),
        caption_template="photorealistic first-person urban street view",
        hflip=True,
    )
    return build_dataset(cfg, resolution=64, seed=0)


class FullFrameDataset(Sequence):
    """A random-crop dataset whose items must not be read once latents are cached."""

    crop = "random"
    hflip = True
    resolution = 64

    def __init__(self, items: int = 3):
        self.items = items
        self.reads = 0

    def __len__(self) -> int:
        return self.items

    def __getitem__(self, index: int) -> dict:
        self.reads += 1
        raise AssertionError("the loop read an item despite the latent cache")

    def full_frame(self, index: int) -> torch.Tensor:
        return torch.full((3, 64, 128), index / self.items)

    def caption_at(self, index: int) -> str:
        return "photorealistic first-person urban street view"


def run_cfg(tmp_path, *, size: int | None = 4, subset_seed: int = 0) -> CatsmConfig:
    cfg = CatsmConfig(name="run", output_dir=str(tmp_path))
    cfg.train.subset_size = size
    cfg.train.subset_seed = subset_seed
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_subset_is_pinned_and_repeatable(tmp_path):
    cfg = run_cfg(tmp_path)
    first = dummy_dataset()
    record = restrict_to_training_subset(cfg, first)
    assert len(first) == 4 and record["source_size"] == 8
    assert (cfg.run_dir / "training_subset.json").exists()

    second = dummy_dataset()
    assert restrict_to_training_subset(cfg, second) == record
    assert second.item_keys() == first.item_keys()


def test_a_different_draw_in_the_same_run_is_refused(tmp_path):
    cfg = run_cfg(tmp_path)
    restrict_to_training_subset(cfg, dummy_dataset(16))
    cfg.train.subset_seed = 1
    with pytest.raises(RuntimeError, match="differs"):
        restrict_to_training_subset(cfg, dummy_dataset(16))


def test_no_subset_leaves_the_dataset_alone(tmp_path):
    cfg = run_cfg(tmp_path, size=None)
    dataset = dummy_dataset()
    assert restrict_to_training_subset(cfg, dataset) is None
    assert len(dataset) == 8
    assert not (cfg.run_dir / "training_subset.json").exists()


def test_subset_draw_does_not_depend_on_the_training_seed():
    a, b = dummy_dataset(16), dummy_dataset(16)
    assert apply_training_subset(a, 4, seed=0) == apply_training_subset(b, 4, seed=0)


def test_bf16_fostering_keeps_fp32_lora_weights(tmp_path):
    adapter = make_adapter()
    table = ProjectionTable.from_model(adapter.unet, granularity="head")
    selection = random_head_selection(table, seed=21)
    state = train_expert(
        adapter, dummy_dataset(), selection, train_cfg(max_steps=2, mixed_precision="bf16"),
        interval=Interval(0, 124), expert="n4_i0", output_dir=tmp_path / "e",
    )
    params = lora_parameters(adapter.unet)
    assert state.step == 2 and torch.isfinite(torch.tensor(state.loss))
    assert (tmp_path / "e" / "lora-final.safetensors").exists()
    # train_expert freezes nothing further, so the masters are still the trainable set.
    assert params and all(p.dtype == torch.float32 for p in params)


def test_fostering_takes_the_random_crop_in_latent_space(tmp_path):
    adapter = make_adapter()
    table = ProjectionTable.from_model(adapter.unet, granularity="head")
    selection = random_head_selection(table, seed=22)
    dataset = FullFrameDataset()
    state = train_expert(
        adapter, dataset, selection, train_cfg(max_steps=2, cache_latents=True),
        interval=Interval(0, 999), expert="n1_i0", output_dir=tmp_path / "e",
    )
    assert state.step == 2 and dataset.reads == 0


def test_router_gate_stays_fp32_under_autocast():
    router = Router(["n1"], ["m0", "m1"], latent_channels=4, num_train_timesteps=1000)
    z, t = torch.zeros(2, 4, 8, 8), torch.tensor([3, 900])
    seen = []
    router.fc.register_forward_hook(lambda _m, _i, out: seen.append(out.dtype))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert router(z, t).dtype == torch.float32
    # The sum alone would promote a bf16 fc output, so look at fc itself.
    assert seen == [torch.float32]
