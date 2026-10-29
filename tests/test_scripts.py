"""scripts/generate_images.py and scripts/evaluate.py on the toy model (offline)."""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from catsm.bundle import load_bundle, save_bundle
from catsm.config import load_config
from catsm.train.assemble import build_assembly
from calora.models.projections import ProjectionTable
from catsm.intervals import Interval
from catsm.train.foster import train_expert
from conftest import ROOT, make_adapter, random_head_selection, train_cfg

TINY = ROOT / "configs" / "tiny_cpu.yaml"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generate_images():
    return load_script("generate_images")


@pytest.fixture(scope="module")
def evaluate():
    return load_script("evaluate")


@pytest.fixture(scope="module")
def routed_bundle(dataset, tmp_path_factory):
    """A routed n1+n2 bundle from three 2-step toy fosters, with a non-trivial router."""
    root = tmp_path_factory.mktemp("scripts")
    table = ProjectionTable.from_model(make_adapter().unet, granularity="head")
    for name, interval, seed in (
        ("n1_i0", Interval(0, 999), 51), ("n2_i0", Interval(0, 499), 52), ("n2_i1", Interval(500, 999), 53),
    ):
        selection = random_head_selection(table, seed=seed)
        train_expert(make_adapter(), dataset, selection, train_cfg(), interval=interval,
                     expert=name, output_dir=root / "experts" / name,
                     metadata={"selection_sha": selection.meta["selection_sha"]})
    cfg = load_config(TINY)
    assembly = build_assembly(make_adapter(), cfg, experts_dir=root / "experts")
    with torch.no_grad():  # make the routing visibly non-trivial
        assembly.router.fc.bias.copy_(torch.linspace(-1.5, 1.5, assembly.router.fc.bias.numel()))
    path = root / "bundle.safetensors"
    save_bundle(assembly, path, extra_metadata={"steps": "0"})
    assembly.detach()
    return path


def test_gate_hook_matches_assembly_predict_for_scalar_and_batched_timesteps(generate_images, routed_bundle):
    """The sampler hands the UNet a scalar t (diffusers) or a (B,) t (toy); both
    must produce exactly what ``assembly.predict`` gives for that step."""
    adapter = make_adapter()
    assembly, hook = generate_images.attach_bundle(adapter, routed_bundle)
    gen = torch.Generator().manual_seed(3)
    z = torch.randn(adapter.latent_shape(2, adapter.resolution), generator=gen)
    cond = adapter.encode_prompts(["a", "b"])
    with torch.no_grad():
        via_hook = adapter.unet(z, torch.tensor(700), cond["prompt_embeds"])  # scalar t, like diffusers
        assert hook.calls == 1
        via_hook_b = adapter.unet(z, torch.tensor([700, 700]), cond["prompt_embeds"])
    hook.remove()  # gate cleared and unbound
    reference = load_bundle(make_adapter(), routed_bundle)
    with torch.no_grad():
        expected = reference.predict(z, torch.tensor([700, 700]), cond)
        base = make_adapter().predict(z, torch.tensor([700, 700]), cond)
    torch.testing.assert_close(via_hook, expected)
    torch.testing.assert_close(via_hook_b, expected)
    assert not torch.allclose(expected, base)  # the experts actually did something
    with torch.no_grad():
        after = adapter.unet(z, torch.tensor(700), cond["prompt_embeds"])
    torch.testing.assert_close(after, base)  # no gate after remove(): containers fall back to W0
    with pytest.raises(ValueError, match="does not match"):
        generate_images._timesteps_for(z, torch.tensor([1, 2, 3]))


def test_generate_and_evaluate_end_to_end(generate_images, evaluate, routed_bundle, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)  # data.augmentation_file is repo-relative
    common = ["--config", str(TINY), "--output-dir", str(tmp_path), "--name", "tiny"]
    run_dir = tmp_path / "tiny"
    generate_images.main([*common, "--bundle", str(routed_bundle), "--run-name", "arm"])
    generate_images.main([*common, "--run-name", "base"])
    arm = json.loads((run_dir / "arm" / "generated" / "manifest.json").read_text())
    base = json.loads((run_dir / "base" / "generated" / "manifest.json").read_text())
    assert len(arm["images"]) == 6 and {r["condition"] for r in arm["images"]} == {"in_domain", "foggy"}
    assert arm["gate_hook_calls"] > 0 and base["gate_hook_calls"] == 0
    assert arm["routing"]["sets"] == ["n1", "n2"] and arm["routing"]["core"] == "n2"
    assert all((run_dir / r["path"]).exists() for r in arm["images"])
    pixels = lambda m, i: np.asarray(Image.open(run_dir / m["images"][i]["path"]))  # noqa: E731
    assert not np.array_equal(pixels(arm, 0), pixels(base, 0))  # routed experts change the image
    assert [r["prompt"] for r in arm["images"]] == [r["prompt"] for r in base["images"]]  # paired prompts

    # Resume: nothing redrawn, identical manifest.
    calls = []
    real = generate_images.load_adapter

    def counting(cfg):
        adapter = real(cfg)
        inner = adapter.generate
        adapter.generate = lambda prompts, **kw: (calls.append(len(prompts)), inner(prompts, **kw))[1]
        return adapter

    monkeypatch.setattr(generate_images, "load_adapter", counting)
    generate_images.main([*common, "--bundle", str(routed_bundle), "--run-name", "arm", "--skip-existing"])
    assert calls == []
    resumed = json.loads((run_dir / "arm" / "generated" / "manifest.json").read_text())
    assert resumed.pop("gate_hook_calls") == 0  # nothing was sampled, so the hook never fired
    arm.pop("gate_hook_calls")
    assert resumed == arm  # key for key, including every image's path

    evaluate.main([*common, "--run-name", "arm", "--embedder", "pixel"])
    report = json.loads((run_dir / "arm" / "eval.json").read_text())
    assert report["reportable"] is False and report["miou"] is None
    assert report["cmmd"]["num_generated"] == 4 and report["cmmd"]["num_reference"] == 4
    assert set(report["clip_score"]["per_condition"]) == {"in_domain", "foggy"}
    assert report["routing"]["core"] == "n2" and report["bundle"] == str(routed_bundle)
    assert (run_dir / "reference" / "00003.png").exists()


def test_make_results_row_reads_routing_and_weather_sem():
    make_results = load_script("make_results")
    cond = {c: {"clip_score": 20.0 + i, "clip_score_std": 2.0, "num_pairs": 100}
            for i, c in enumerate(make_results.CONDITIONS)}
    cond["in_domain"] = {"clip_score": 22.0, "clip_score_std": 2.0, "num_pairs": 500}
    d = {"routing": {"sets": ["n1", "n4"], "core": "n4", "disabled": ["n4_i3"], "router": {"type": "x"},
                     "experts": [{"name": n} for n in ("n1_i0", "n4_i0", "n4_i1", "n4_i2", "n4_i3")]},
         "protocol": {"seed": 0}, "num_images": 900, "cmmd": {"cmmd": 1.0},
         "clip_score": {"per_condition": cond, "weather_mean": 21.5}}
    r = make_results.row_from_eval("c2", d, label="C2", desc="", group="control")
    assert (r["routing"], r["n_experts"], r["nominal_head_pct"], r["active_head_pct_per_t"]) == ("router", 5, 10.0, 4.0)
    assert r["active_note"] == "2% on [500,999]" and r["fostering_steps_total"] == 20000 and r["router_steps"] == 1000
    assert r["clip_weather_sem"] == pytest.approx((4 * 4 / 100) ** 0.5 / 4)   # sqrt(sum s^2/n) / 4


def test_router_gate_helpers_pick_active_expert_and_scaled_delta_norm():
    rg = load_script("router_gate_vs_t")
    meta = {"experts": [{"name": "a0", "set": "a", "t_lo": 0, "t_hi": 499, "rank": 2, "alpha": 4.0},
                        {"name": "a1", "set": "a", "t_lo": 500, "t_hi": 999, "rank": 2, "alpha": 4.0}]}
    assert rg.active_expert(meta, "a", 499) == "a0" and rg.active_expert(meta, "a", 500) == "a1"
    A, B = torch.randn(2, 3), torch.randn(5, 2)
    state = {"m.experts.a0.lora_A": A, "m.experts.a0.lora_B": B}
    norms = rg.delta_norms(state, meta, ["m", "other"])
    assert norms["a0"][0].item() == pytest.approx(2.0 * float((B.double() @ A.double()).norm()))
    assert norms["a0"][1].item() == 0.0 and norms["a1"].tolist() == [0.0, 0.0]
