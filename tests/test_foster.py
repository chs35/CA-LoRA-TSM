from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from calora.config import LoraConfig_
from calora.models.projections import ProjectionTable
from calora.train.lora import load_lora, lora_state_dict, train_lora
from catsm.intervals import Interval
from catsm.train import foster
from catsm.train.foster import check_timesteps, sample_timesteps, train_expert
from conftest import make_adapter, random_head_selection, train_cfg


def test_full_range_expert_is_bitwise_identical_to_train_lora(dataset, tmp_path):
    cfg = train_cfg(max_steps=3)
    table = ProjectionTable.from_model(make_adapter().unet, granularity="head")
    selection = random_head_selection(table, seed=11)

    ref = make_adapter()
    train_lora(ref, dataset, selection, cfg, output_dir=tmp_path / "ref")
    ours = make_adapter()
    state = train_expert(ours, dataset, selection, cfg, interval=Interval(0, 999),
                         expert="n1_i0", output_dir=tmp_path / "n1_i0")

    a, b = lora_state_dict(ref.unet), lora_state_dict(ours.unet)
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert state.t_min_seen is not None and 0 <= state.t_min_seen <= state.t_max_seen <= 999
    with safe_open(str(tmp_path / "n1_i0" / "lora-final.safetensors"), framework="pt") as h:
        meta = h.metadata()
    assert (meta["t_lo"], meta["t_hi"], meta["expert"]) == ("0", "999", "n1_i0")
    assert meta["granularity"] == "head" and meta["lora_alpha"] == "4"


def test_sampled_timesteps_stay_inside_the_interval(dataset, tmp_path, monkeypatch):
    interval = Interval(250, 499)
    for step in range(1, 50):
        t = sample_timesteps(interval, 4, seed=0, step=step, micro=0)
        assert int(t.min()) >= 250 and int(t.max()) <= 499
    # The same generator stream as the reference draw when the interval is everything.
    ref = torch.randint(0, 1000, (4,), generator=foster.derive_generator(0, "train-t", 3, 0))
    assert torch.equal(sample_timesteps(Interval(0, 999), 4, seed=0, step=3, micro=0), ref)

    table = ProjectionTable.from_model(make_adapter().unet, granularity="head")
    selection = random_head_selection(table, seed=12)
    adapter = make_adapter()
    state = train_expert(adapter, dataset, selection, train_cfg(max_steps=3, batch_size=4),
                         interval=interval, expert="n4_i2", output_dir=tmp_path / "ok")
    assert 250 <= state.t_min_seen <= state.t_max_seen <= 499
    assert state.num_timesteps_drawn == 12
    summary = json.loads((tmp_path / "ok" / "train_summary.json").read_text())
    assert summary["final"]["t_min_seen"] >= 250 and summary["final"]["t_max_seen"] <= 499
    assert summary["metadata"]["t_min_seen"] == str(state.t_min_seen)

    # A draw that escapes the interval is a hard failure, not a warning.
    monkeypatch.setattr(foster, "sample_timesteps",
                        lambda interval, batch, **kw: torch.full((batch,), 999))
    with pytest.raises(RuntimeError, match="leaves the expert's interval"):
        train_expert(make_adapter(), dataset, selection, train_cfg(max_steps=1),
                     interval=interval, expert="n4_i2", output_dir=tmp_path / "bad")
    with pytest.raises(RuntimeError):
        check_timesteps(torch.tensor([249, 300]), interval)
    assert check_timesteps(torch.tensor([300, 250]), interval) == (250, 300)


def test_expert_file_records_interval_selection_and_reloads_alone(dataset, tmp_path):
    table = ProjectionTable.from_model(make_adapter().unet, granularity="head")
    selection = random_head_selection(table, seed=13)
    adapter = make_adapter()
    train_expert(adapter, dataset, selection, train_cfg(),
                 interval=Interval(125, 249), expert="n4_i1", output_dir=tmp_path,
                 metadata={"selection_sha": selection.meta["selection_sha"],
                           "interval_scheme": "n4", "ca_timesteps": "[201]"})
    path = tmp_path / "lora-final.safetensors"
    with safe_open(str(path), framework="pt") as h:
        meta = h.metadata()
    assert meta["selection_sha"] == selection.meta["selection_sha"]
    assert meta["interval_scheme"] == "n4" and meta["ca_timesteps"] == "[201]"
    assert float(meta["logsnr_lo"]) > float(meta["logsnr_hi"])
    assert 125 <= int(meta["t_min_seen"]) <= int(meta["t_max_seen"]) <= 249
    # Single-adapter format: calora's own loader accepts it as-is.
    fresh = make_adapter()
    loaded = load_lora(fresh.unet, path, LoraConfig_(rank=4, alpha=4), expect_granularity="head")
    assert set(loaded.names) == set(selection.names)
    state = load_file(str(path))
    assert all(".experts." not in k for k in state)


def test_different_intervals_train_different_weights_under_one_seed(dataset, tmp_path):
    table = ProjectionTable.from_model(make_adapter().unet, granularity="head")
    selection = random_head_selection(table, seed=14)
    weights = {}
    for name, interval in (("n2_i0", Interval(0, 499)), ("n2_i1", Interval(500, 999))):
        adapter = make_adapter()
        train_expert(adapter, dataset, selection, train_cfg(), interval=interval,
                     expert=name, output_dir=tmp_path / name)
        weights[name] = lora_state_dict(adapter.unet)
    changed = [k for k in weights["n2_i0"] if ".lora_B" in k
               and not torch.equal(weights["n2_i0"][k], weights["n2_i1"][k])]
    assert changed, "the two intervals saw different timesteps and must diverge"


def test_train_expert_refuses_bad_inputs(dataset, tmp_path):
    table = ProjectionTable.from_model(make_adapter().unet, granularity="head")
    selection = random_head_selection(table, seed=15)
    with pytest.raises(ValueError):
        train_expert(make_adapter(), dataset, selection, train_cfg(), interval=Interval(0, 1000),
                     expert="n1_i0", output_dir=tmp_path)
    with pytest.raises(ValueError):
        train_expert(make_adapter(), dataset, selection, train_cfg(), interval=Interval(0, 999),
                     expert="n1.i0", output_dir=tmp_path)
