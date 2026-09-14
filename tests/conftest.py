"""Shared toy-model fixtures. Everything runs on CPU with no downloads."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calora.ca.select import Selection, select_topk  # noqa: E402
from calora.config import DataConfig, ModelConfig, TrainConfig  # noqa: E402
from calora.config import LoraConfig_  # noqa: E402
from calora.data import build_dataset  # noqa: E402
from calora.models.loader import load_adapter  # noqa: E402
from calora.models.projections import ProjectionTable  # noqa: E402
from calora.utils.seeding import derive_generator  # noqa: E402
from catsm.select import selection_sha  # noqa: E402

torch.set_num_threads(2)

TOY = {
    "base_channels": 32,
    "context_dim": 32,
    "latent_channels": 4,
    "vae_scale_factor": 8,
    "text_seq_len": 8,
    "seed": 0,
}


def make_adapter():
    """A fresh toy adapter (fresh UNet) every call; deterministic weights."""
    return load_adapter(ModelConfig(kind="toy", device="cpu", resolution=64, toy=dict(TOY)))


@pytest.fixture(scope="session")
def dataset():
    cfg = DataConfig(
        kind="dummy",
        num_dummy_items=6,
        augmentation_file=str(ROOT / "configs" / "prompt_augmentations.json"),
        caption_template="photorealistic first-person urban street view",
        hflip=True,
    )
    return build_dataset(cfg, resolution=64, seed=0)


@pytest.fixture(scope="session")
def head_table() -> ProjectionTable:
    return ProjectionTable.from_model(make_adapter().unet, granularity="head")


def random_head_selection(table: ProjectionTable, *, seed: int, fraction: float = 0.15) -> Selection:
    """A distinct head-granularity selection per seed (random CA scores)."""
    scores = torch.rand(len(table), generator=derive_generator(seed, "fake-ca"), dtype=torch.float64)
    selection = select_topk(
        scores, table, fraction=fraction, budget_basis="unit_count", strategy=f"random-{seed}",
        meta={"axis": "style"},
    )
    selection.meta["selection_sha"] = selection_sha(selection.names)
    return selection


def train_cfg(**overrides) -> TrainConfig:
    base = dict(
        lora=LoraConfig_(rank=4, alpha=4),
        max_steps=2,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=0,
        log_every=1,
        stability_every=0,
    )
    base.update(overrides)
    return TrainConfig(**base)
