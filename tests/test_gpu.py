"""GPU checks: the whole toy pipeline on CUDA, plus what only a GPU exercises.

Every test here is skipped without CUDA, so the CPU suite is unchanged. On the
GPU pod::

    source scripts/env.sh
    python -m pytest tests/test_gpu.py -q                   # toy model on CUDA, ~1-2 min
    CATSM_SDXL_SMOKE=1 python -m pytest tests/test_gpu.py -q -k sdxl   # real SDXL, weights cached

Covered:

1. the four CLI stages end to end with ``model.device=cuda`` — the exact commands
   the pod runs — and the bundle they produce loads on both devices;
2. CA measurement on CUDA agrees with the CPU measurement of the same samples;
3. an expert fostered on CUDA loads into a CPU model and predicts the same;
4. bf16 and fp16 mixed precision (GradScaler path) for fostering and routing;
5. bundle round trip CUDA -> file -> CPU and CUDA;
6. per-sample gate tensors on CUDA under non-reentrant gradient checkpointing;
7. (opt-in) the SDXL config at 256 px on the dummy dataset, one step per stage.

Cross-device tolerances assume fp32 without TF32 (the ``no_tf32`` fixture); a
toy forward differs between CPU and CUDA at the 1e-6 level, gradient statistics
(CA) at about 1e-4.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from calora.ca.measure import measure_concept_awareness, prepare_samples
from calora.config import ModelConfig, SelectConfig
from calora.models.loader import load_adapter
from calora.models.projections import ProjectionTable
from calora.train.lora import lora_parameters, lora_state_dict
from catsm.bundle import load_bundle, save_bundle
from catsm.config import AssembleConfig, load_config
from catsm.intervals import Interval, build_scheme
from catsm.models.multi_sliced_lora import MultiSlicedLoRALinear, iter_multi_sliced_lora
from catsm.models.router import Router
from catsm.select import select_scheme
from catsm.train.assemble import build_assembly, load_expert_into, train_router
from catsm.train.foster import train_expert
from conftest import ROOT, TOY, make_adapter, random_head_selection, train_cfg

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

CUDA = torch.device("cuda")
TINY = ROOT / "configs" / "tiny_cpu.yaml"
PROMPT = "photorealistic first-person urban street view"


def make_cuda_adapter():
    """The conftest toy model, built on CUDA (same seed, same weights)."""
    return load_adapter(ModelConfig(kind="toy", device="cuda", resolution=64, toy=dict(TOY)))


@pytest.fixture
def no_tf32():
    """fp32 matmuls/convs stay fp32 so CPU and CUDA can be compared tightly."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def batch(adapter, timesteps):
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(adapter.latent_shape(len(timesteps), adapter.resolution), generator=gen)
    cond = adapter.encode_prompts([PROMPT] * len(timesteps))
    t = torch.tensor(timesteps)
    z, t = z.to(adapter.device, adapter.dtype), t.to(adapter.device)
    return z, t, {k: v.to(adapter.device, adapter.dtype) for k, v in cond.items()}


def run_cli(args: list[str], cwd: Path = ROOT) -> str:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    env.setdefault("OMP_NUM_THREADS", "2")
    proc = subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"{' '.join(args)} failed ({proc.returncode})\n--- stdout ---\n{proc.stdout[-4000:]}"
            f"\n--- stderr ---\n{proc.stderr[-4000:]}"
        )
    return proc.stdout + proc.stderr


def all_on(module: nn.Module, device: torch.device) -> bool:
    return all(p.device.type == device.type for p in module.parameters()) and all(
        b.device.type == device.type for b in module.buffers()
    )


# --------------------------------------------------------------------------------------
# 1. The CLI pipeline on CUDA
# --------------------------------------------------------------------------------------


def test_cli_pipeline_runs_on_cuda(tmp_path, no_tf32):
    """measure -> select -> foster x3 -> assemble with model.device=cuda, as on the pod."""
    common = ["--config", str(TINY), "--output-dir", str(tmp_path), "--name", "gpu"]
    device = ["--set", "model.device=cuda", "measure.accum_device=auto"]
    run_cli(["scripts/measure_ca.py", *common, *device])
    run_dir = tmp_path / "gpu"
    log = (run_dir / "run.log").read_text()
    assert "Building toy adapter on cuda" in log, "the CLI did not put the model on CUDA"
    assert sorted(p.name for p in (run_dir / "measurements").glob("t*.safetensors")) == [
        f"t{t:04d}{h}.safetensors" for t in (1, 81, 201, 481, 750) for h in ("", "_half0", "_half1")
    ]

    for scheme in ("n1", "n2"):
        run_cli(["-m", "catsm.cli", "select", *common, "--scheme", scheme, *device])
    assert (run_dir / "selections" / "n2" / "n2_i1.json").exists()

    # ``--set train.mixed_precision=no`` would YAML-parse to False; "no" is the default.
    for expert, precision in (("n1_i0", None), ("n2_i0", "bf16"), ("n2_i1", "fp16")):
        override = [f"train.mixed_precision={precision}"] if precision else []
        run_cli(["-m", "catsm.cli", "foster", *common, "--expert", expert, *device, *override])
        summary = json.loads((run_dir / "experts" / expert / "train_summary.json").read_text())
        lo, hi = summary["interval"]
        assert lo <= summary["final"]["t_min_seen"] <= summary["final"]["t_max_seen"] <= hi
        assert summary["final"]["step"] == 2 and summary["final"]["loss"] == summary["final"]["loss"]

    run_cli(["-m", "catsm.cli", "assemble", *common, "--sets", "n1", "n2", "--core", "n2",
             *device, "assemble.mixed_precision=bf16"])
    bundle = run_dir / "bundles" / "n1+n2-core-n2" / "bundle.safetensors"
    assert bundle.exists()
    summary = json.loads((bundle.parent / "assemble_summary.json").read_text())
    assert summary["final"]["step"] == 2 and summary["core"] == "n2"

    # Hard routing needs no training and must also run on CUDA.
    run_cli(["-m", "catsm.cli", "assemble", *common, "--sets", "n2", "--core", "n2",
             "--no-router", *device])
    assert (run_dir / "bundles" / "n2-core-n2" / "bundle.safetensors").exists()

    # The bundle the pod writes must load on the GPU *and* on a CPU box, and agree.
    on_cuda = load_bundle(make_cuda_adapter(), bundle)
    on_cpu = load_bundle(make_adapter(), bundle)
    assert all_on(on_cuda.unet, CUDA) and all_on(on_cuda.router, CUDA)
    z, t, cond = batch(on_cuda.adapter, [10, 700, 300])
    with torch.no_grad():
        got = on_cuda.predict(z, t, cond)
        ref = on_cpu.predict(z.cpu(), t.cpu(), {k: v.cpu() for k, v in cond.items()})
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------------------
# 2. CA measurement
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def measure_dataset():
    from calora.config import DataConfig
    from calora.data import build_dataset

    cfg = DataConfig(
        kind="dummy", num_dummy_items=4, concept_axes=["style", "viewpoint"],
        augmentation_file=str(ROOT / "configs" / "prompt_augmentations.json"),
        caption_template=PROMPT,
    )
    return build_dataset(cfg, resolution=64, seed=0, crop="center", hflip=False, augs_per_sample=2)


def measure(adapter, dataset, *, accum_device, timestep=81):
    table = ProjectionTable.from_model(adapter.unet, granularity="head")
    samples = prepare_samples(adapter, dataset, axes=["style", "viewpoint"], num_samples=4)
    result = measure_concept_awareness(
        adapter, table, samples, timestep=timestep, axes=["style", "viewpoint"],
        granularity="matrix", accum_device=accum_device, seed=0, split_half=True,
    )
    return table, result


def test_ca_measurement_on_cuda_matches_cpu(measure_dataset, no_tf32):
    cpu_table, cpu = measure(make_adapter(), measure_dataset, accum_device="cpu")
    cuda_table, cuda = measure(make_cuda_adapter(), measure_dataset, accum_device=CUDA)
    assert cuda.names == cpu.names and len(cuda.names) == 80
    assert cuda.num_samples == cpu.num_samples == 8
    for axis in ("style", "viewpoint"):
        torch.testing.assert_close(cuda.ca(axis).cpu(), cpu.ca(axis).cpu(), rtol=2e-3, atol=1e-5)
    torch.testing.assert_close(cuda.gd_norm_mean.cpu(), cpu.gd_norm_mean.cpu(), rtol=2e-3, atol=1e-6)
    assert torch.isfinite(cuda.similarity("style", "viewpoint").cpu()).all()
    assert cuda.halves is not None and len(cuda.halves) == 2

    # Accumulating on the GPU and on the CPU is the same statistic.
    _, cuda_cpu_accum = measure(make_cuda_adapter(), measure_dataset, accum_device="cpu")
    torch.testing.assert_close(cuda_cpu_accum.ca("style").cpu(), cuda.ca("style").cpu(), rtol=1e-4, atol=1e-6)

    # Selection from either device picks the same heads.
    select = SelectConfig(granularity="head", fraction=0.1, budget_basis="unit_count", axis="style")
    scheme = build_scheme("n1", 1000)
    picked_cpu = select_scheme({81: cpu}, cpu_table, scheme, mode="aggregate", cfg=select)["n1_i0"]
    picked_cuda = select_scheme({81: cuda}, cuda_table, scheme, mode="aggregate", cfg=select)["n1_i0"]
    assert sorted(picked_cpu.names) == sorted(picked_cuda.names)


# --------------------------------------------------------------------------------------
# 3-4. Fostering on CUDA
# --------------------------------------------------------------------------------------


def test_expert_fostered_on_cuda_loads_on_cpu_and_predicts_the_same(dataset, tmp_path, no_tf32):
    table = ProjectionTable.from_model(make_adapter().unet, granularity="head")
    selection = random_head_selection(table, seed=31)
    adapter = make_cuda_adapter()
    state = train_expert(adapter, dataset, selection, train_cfg(max_steps=3),
                         interval=Interval(0, 499), expert="n2_i0", output_dir=tmp_path,
                         metadata={"selection_sha": selection.meta["selection_sha"]})
    assert 0 <= state.t_min_seen <= state.t_max_seen <= 499
    assert torch.isfinite(torch.tensor(state.loss))
    params = lora_parameters(adapter.unet)
    assert params and all(p.device.type == "cuda" and p.dtype == torch.float32 for p in params)
    # B is zero-initialised: after three steps it must have moved, on the GPU.
    assert any(k.endswith(".lora_B") and v.abs().sum() > 0 for k, v in lora_state_dict(adapter.unet).items())

    path = tmp_path / "lora-final.safetensors"
    cpu = make_adapter()
    info = load_expert_into(cpu.unet, path)
    assert info.name == "n2_i0" and info.selection_sha == selection.meta["selection_sha"]
    cuda = make_cuda_adapter()
    load_expert_into(cuda.unet, path)
    assert all_on(cuda.unet, CUDA)
    for a in (cpu, cuda):  # the lone expert under a unit gate
        for _, module in iter_multi_sliced_lora(a.unet):
            module.gate = {"n2_i0": 1.0}
    z, t, cond = batch(cuda, [10, 400])
    with torch.no_grad():
        got = cuda.predict(z, t, cond)
        ref = cpu.predict(z.cpu(), t.cpu(), {k: v.cpu() for k, v in cond.items()})
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_mixed_precision_fostering_keeps_fp32_weights(dataset, tmp_path, precision):
    table = ProjectionTable.from_model(make_adapter().unet, granularity="head")
    selection = random_head_selection(table, seed=32)
    adapter = make_cuda_adapter()
    cfg = train_cfg(max_steps=3, mixed_precision=precision)
    state = train_expert(adapter, dataset, selection, cfg, interval=Interval(500, 999),
                         expert="n2_i1", output_dir=tmp_path)
    assert torch.isfinite(torch.tensor(state.loss)) and 500 <= state.t_min_seen
    for p in lora_parameters(adapter.unet):
        assert p.dtype == torch.float32 and p.device.type == "cuda"
    moved = [k for k, v in lora_state_dict(adapter.unet).items() if k.endswith(".lora_B") and v.abs().sum() > 0]
    assert moved, f"{precision}: no LoRA B matrix changed (loss scaling swallowed the update?)"
    meta_path = tmp_path / "lora-final.safetensors"
    assert meta_path.exists()


# --------------------------------------------------------------------------------------
# 5. Assembling on CUDA
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cuda_experts_dir(dataset, tmp_path_factory):
    """n1_i0, n2_i0, n2_i1 fostered on CUDA (2 steps each)."""
    root = tmp_path_factory.mktemp("cuda_experts")
    table = ProjectionTable.from_model(make_adapter().unet, granularity="head")
    for name, interval, seed in (
        ("n1_i0", Interval(0, 999), 41),
        ("n2_i0", Interval(0, 499), 42),
        ("n2_i1", Interval(500, 999), 43),
    ):
        selection = random_head_selection(table, seed=seed)
        train_expert(make_cuda_adapter(), dataset, selection, train_cfg(), interval=interval,
                     expert=name, output_dir=root / name,
                     metadata={"selection_sha": selection.meta["selection_sha"]})
    return root


@pytest.fixture(scope="module")
def tiny_cfg():
    return load_config(TINY)


@pytest.mark.parametrize("precision", ["no", "bf16", "fp16"])
def test_router_trains_on_cuda_with_frozen_experts(cuda_experts_dir, tiny_cfg, dataset, tmp_path, precision):
    adapter = make_cuda_adapter()
    assembly = build_assembly(adapter, tiny_cfg, experts_dir=cuda_experts_dir)
    assert assembly.router is not None and all_on(assembly.router, CUDA) and all_on(adapter.unet, CUDA)
    z, t, cond = batch(adapter, [10, 700])
    gate = assembly.gate_for(t, z)
    assert all(g.device.type == "cuda" and g.shape == (2,) for g in gate.values())
    assert gate["n2_i0"].tolist() == [1.0, 0.0] and gate["n2_i1"].tolist() == [0.0, 1.0]

    before = {k: v.clone() for k, v in lora_state_dict(adapter.unet).items()}
    router_before = {k: v.clone() for k, v in assembly.router.state_dict().items()}
    cfg = AssembleConfig(sets=["n1", "n2"], core="n2", max_steps=3, batch_size=2,
                         learning_rate=1e-2, mixed_precision=precision, log_every=1)
    state = train_router(assembly, dataset, cfg, output_dir=tmp_path)
    assert state.step == 3 and torch.isfinite(torch.tensor(state.loss))
    after = lora_state_dict(adapter.unet)
    assert all(torch.equal(before[k], after[k]) for k in before), "an expert moved during assembling"
    assert lora_parameters(adapter.unet) == []
    changed = [k for k, v in assembly.router.state_dict().items() if not torch.equal(v, router_before[k])]
    assert "fc.bias" in changed, f"{precision}: the router did not learn"
    assert all(p.dtype == torch.float32 for p in assembly.router.parameters())
    assert assembly.gate == {}
    assert (tmp_path / "bundle.safetensors").exists()


def test_bundle_round_trip_across_devices(cuda_experts_dir, tiny_cfg, tmp_path, no_tf32):
    adapter = make_cuda_adapter()
    assembly = build_assembly(adapter, tiny_cfg, experts_dir=cuda_experts_dir)
    with torch.no_grad():
        assembly.router.fc.bias.copy_(torch.linspace(-1, 1, assembly.router.fc.bias.numel()))
        assembly.router.t_embed.weight[700].fill_(-1.3)
    path = tmp_path / "bundle.safetensors"
    save_bundle(assembly, path, extra_metadata={"steps": "0"})
    z, t, cond = batch(adapter, [10, 700, 300])
    with torch.no_grad():
        expected = assembly.predict(z, t, cond)
    assembly.detach()

    reloaded_cuda = load_bundle(make_cuda_adapter(), path)
    reloaded_cpu = load_bundle(make_adapter(), path)
    assert torch.equal(reloaded_cuda.router.fc.bias.cpu(), assembly.router.fc.bias.cpu())
    with torch.no_grad():
        got_cuda = reloaded_cuda.predict(z, t, cond)
        got_cpu = reloaded_cpu.predict(z.cpu(), t.cpu(), {k: v.cpu() for k, v in cond.items()})
    torch.testing.assert_close(got_cuda, expected)
    torch.testing.assert_close(got_cpu, expected.cpu(), rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------------------
# 6. Gate tensors on CUDA under gradient checkpointing (the SDXL training path)
# --------------------------------------------------------------------------------------


def test_cuda_gate_gradient_survives_non_reentrant_checkpoint():
    heads, head_dim, d_in = 4, 8, 16
    torch.manual_seed(0)
    base = nn.Linear(d_in, heads * head_dim).to(CUDA)
    multi = MultiSlicedLoRALinear(base, num_heads=heads, head_dim=head_dim, slice_dim="row")
    multi.add_expert("n1_i0", head_indices=[1, 3], rank=2, alpha=4.0)
    with torch.no_grad():
        multi.experts["n1_i0"].lora_B.normal_()
    multi.requires_grad_(False)
    multi.module_name = "blk.to_q"
    assert all_on(multi, CUDA)

    router = Router(["n1"], ["blk.to_q"], latent_channels=4, num_train_timesteps=1000).to(CUDA)
    z = torch.randn(2, 4, 8, 8, device=CUDA)
    t = torch.tensor([10, 700], device=CUDA)
    weights = router(z, t)  # (B, M, K) on CUDA
    assert weights.device.type == "cuda" and weights.shape == (2, 1, 1)
    multi.gate = {("blk.to_q", "n1_i0"): weights[:, 0, 0]}
    x = torch.randn(2, 5, d_in, device=CUDA)
    out = checkpoint(multi, x, use_reentrant=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_bf16 = multi(x)
    assert out_bf16.dtype == torch.bfloat16 and torch.isfinite(out_bf16).all()
    out.pow(2).sum().backward()
    assert router.fc.bias.grad is not None and router.fc.bias.grad.abs().sum() > 0
    assert router.t_embed.weight.grad is not None and router.t_embed.weight.grad[700].abs().sum() > 0
    assert router.t_embed.weight.grad[5].abs().sum() == 0  # only the visited timesteps
    assert all(p.grad is None for p in multi.parameters())

    # Merging the expert (hard routing) is the same forward on CUDA.
    multi.gate = {"n1_i0": 1.0}
    with torch.no_grad():
        gated = multi(x)
        multi.merge("n1_i0")
        multi.gate = None
        merged = multi(x)
        multi.unmerge("n1_i0")
    torch.testing.assert_close(gated, merged, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------------------
# 7. Opt-in: the real SDXL config, one step per stage, dummy data, 256 px
# --------------------------------------------------------------------------------------


@pytest.mark.sdxl
@pytest.mark.skipif(
    os.environ.get("CATSM_SDXL_SMOKE") != "1",
    reason="set CATSM_SDXL_SMOKE=1 (SDXL weights must be in HF_HOME; ~7 GB, several minutes)",
)
def test_sdxl_pipeline_smoke_on_cuda(tmp_path):
    """configs/sdxl_cityscapes.yaml as shipped, shrunk to something that finishes:
    dummy data instead of Cityscapes, 256 px, two measured timesteps, n1+n2,
    one fostering/router step. Exercises the real UNet under gradient
    checkpointing with the gate dict, bf16 autocast and the text-encoder parking.
    """
    config = ROOT / "configs" / "sdxl_cityscapes.yaml"
    common = ["--config", str(config), "--output-dir", str(tmp_path), "--name", "sdxl_smoke"]
    shrink = [
        "--set",
        "model.resolution=256",
        "data.kind=dummy", "data.num_dummy_items=2", "data.crop=center",
        "measure.timesteps=[81,750]", "measure.num_samples=1", "measure.augs_per_sample=1",
        "measure.split_half=false", "measure.log_every=0",
        "intervals.schemes=[1,2]",
        "train.max_steps=1", "train.checkpoint_every=0", "train.log_every=1",
        "train.mixed_precision=bf16",
        "assemble.sets=[n1,n2]", "assemble.core=n2", "assemble.max_steps=1", "assemble.log_every=1",
        "assemble.mixed_precision=bf16",
    ]
    run_cli(["scripts/measure_ca.py", *common, *shrink])
    run_dir = tmp_path / "sdxl_smoke"
    log = (run_dir / "run.log").read_text()
    assert "StableDiffusionXLPipeline" in log and "cuda" in log
    for scheme in ("n1", "n2"):
        run_cli(["-m", "catsm.cli", "select", *common, "--scheme", scheme, *shrink])
    intervals = json.loads((run_dir / "selections" / "n2" / "intervals.json").read_text())
    assert [iv["timesteps"] for iv in intervals["intervals"]] == [[81], [750]]
    for expert in ("n1_i0", "n2_i0", "n2_i1"):
        run_cli(["-m", "catsm.cli", "foster", *common, "--expert", expert, *shrink])
        summary = json.loads((run_dir / "experts" / expert / "train_summary.json").read_text())
        assert summary["final"]["step"] == 1 and summary["final"]["loss"] == summary["final"]["loss"]
    run_cli(["-m", "catsm.cli", "assemble", *common, *shrink])
    bundle = run_dir / "bundles" / "n1+n2-core-n2" / "bundle.safetensors"
    assert bundle.exists()
    summary = json.loads((bundle.parent / "assemble_summary.json").read_text())
    assert summary["final"]["step"] == 1 and summary["router"]["modules"]
    # 2% of 10,400 head units: the selection ratio survived the shrink.
    selection = json.loads((run_dir / "experts" / "n2_i0" / "selection.json").read_text())
    assert 190 <= len(selection["names"]) <= 220
