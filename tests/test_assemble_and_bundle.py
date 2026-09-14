from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from calora.config import LoraConfig_
from calora.models.projections import ProjectionTable
from calora.train.lora import load_lora, lora_parameters, lora_state_dict
from catsm.models.multi_sliced_lora import MultiSlicedLoRALinear
from catsm.bundle import BUNDLE_FORMAT, load_bundle, save_bundle
from catsm.config import AssembleConfig, load_config
from catsm.intervals import Interval, build_scheme
from catsm.models.router import Router, RoutingTable
from catsm.train.assemble import TSMAssembly, build_assembly, load_expert_into, train_router
from catsm.train.foster import train_expert
from conftest import ROOT, make_adapter, random_head_selection, train_cfg

T = 1000


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "tiny_cpu.yaml")


@pytest.fixture(scope="module")
def experts_dir(dataset, tmp_path_factory):
    """Foster n1_i0, n2_i0, n2_i1 once (2 steps each) into <dir>/<expert>/."""
    root = tmp_path_factory.mktemp("experts")
    table = ProjectionTable.from_model(make_adapter().unet, granularity="head")
    for name, interval, seed in (
        ("n1_i0", Interval(0, 999), 21),
        ("n2_i0", Interval(0, 499), 22),
        ("n2_i1", Interval(500, 999), 23),
    ):
        selection = random_head_selection(table, seed=seed)
        train_expert(make_adapter(), dataset, selection, train_cfg(), interval=interval,
                     expert=name, output_dir=root / name,
                     metadata={"selection_sha": selection.meta["selection_sha"]})
    return root


def batch(adapter, timesteps):
    gen = torch.Generator().manual_seed(0)
    b = len(timesteps)
    z = torch.randn(adapter.latent_shape(b, adapter.resolution), generator=gen)
    cond = adapter.encode_prompts(["photorealistic first-person urban street view"] * b)
    return z, torch.tensor(timesteps), cond


def test_expert_files_load_into_containers_and_match_their_selection_sha(experts_dir):
    adapter = make_adapter()
    info = load_expert_into(adapter.unet, experts_dir / "n2_i0" / "lora-final.safetensors")
    assert (info.name, info.set, info.t_lo, info.t_hi, info.rank, info.alpha) == ("n2_i0", "n2", 0, 499, 4, 4.0)
    with safe_open(str(experts_dir / "n2_i0" / "lora-final.safetensors"), framework="pt") as h:
        assert h.metadata()["selection_sha"] == info.selection_sha
    # Wrong name or interval is refused; a tampered head set is refused.
    with pytest.raises(ValueError):
        load_expert_into(make_adapter().unet, experts_dir / "n2_i0" / "lora-final.safetensors", name="n2_i1")
    with pytest.raises(ValueError):
        load_expert_into(make_adapter().unet, experts_dir / "n2_i0" / "lora-final.safetensors",
                         expect_interval=Interval(0, 124))


def test_hard_routing_activates_exactly_the_interval_owner():
    schemes = {"n1": build_scheme("n1", T), "n2": build_scheme("n2", T)}
    routing = RoutingTable(schemes, core="n2", num_train_timesteps=T)
    assert routing.context_sets == ["n1"] and routing.expert_names == ["n1_i0", "n2_i0", "n2_i1"]
    gate = routing.gate(torch.tensor([10, 700]))
    assert gate["n1_i0"].tolist() == [1.0, 1.0]
    assert gate["n2_i0"].tolist() == [1.0, 0.0] and gate["n2_i1"].tolist() == [0.0, 1.0]
    modules = ["blk.attn1.to_q", "blk.attn1.to_out.0"]
    router = Router(["n1"], modules, latent_channels=4, num_train_timesteps=T, init_gate=1.0)
    weights = router(torch.randn(2, 4, 8, 8), torch.tensor([10, 700]))
    assert weights.shape == (2, 2, 1)
    assert torch.allclose(weights, torch.full((2, 2, 1), 1.0))  # gate == init_gate at start
    with torch.no_grad():  # module 1 at t=700 goes negative (subtracts), module 0 at t=10 amplifies
        router.t_embed.weight[700, 1] = -4.0
        router.t_embed.weight[10, 0] = 3.0
    weights = router(torch.zeros(2, 4, 8, 8), torch.tensor([10, 700]))
    soft = routing.gate(torch.tensor([10, 700]), weights, modules=modules)
    assert "n1_i0" not in soft  # context experts are gated per module, never globally
    assert soft[("blk.attn1.to_q", "n1_i0")].tolist() == pytest.approx([4.0, 1.0])
    assert soft[("blk.attn1.to_out.0", "n1_i0")].tolist() == pytest.approx([1.0, -3.0])  # raw, any sign
    assert soft["n2_i0"].tolist() == [1.0, 0.0]  # core stays global
    with pytest.raises(ValueError):
        routing.gate(torch.tensor([10, 700]), weights)  # module list required
    with pytest.raises(ValueError):
        RoutingTable(schemes, core="n3", num_train_timesteps=T)


def test_assembled_single_set_forward_equals_the_lone_expert_model(experts_dir, cfg):
    adapter = make_adapter()
    assembly = build_assembly(adapter, cfg, experts_dir=experts_dir, sets=["n2"], core="n2")
    assert assembly.router is None and assembly.expert_names == ["n2_i0", "n2_i1"]
    for t, expert in ((10, "n2_i0"), (700, "n2_i1")):
        z, ts, cond = batch(adapter, [t])
        with torch.no_grad():
            ours = assembly.predict(z, ts, cond)
        lone = make_adapter()
        load_lora(lone.unet, experts_dir / expert / "lora-final.safetensors", LoraConfig_(rank=4, alpha=4))
        with torch.no_grad():
            ref = lone.predict(z, ts, cond)
        torch.testing.assert_close(ours, ref)
    assembly.clear()
    with pytest.raises(ValueError):
        train_router(assembly, [], AssembleConfig(sets=["n2"], core="n2", max_steps=1), output_dir=experts_dir / "x")


def test_router_training_updates_only_the_router(experts_dir, cfg, dataset, tmp_path):
    adapter = make_adapter()
    assembly = build_assembly(adapter, cfg, experts_dir=experts_dir)
    assert assembly.router is not None and assembly.context_sets == ["n1"]
    # One router row per matrix that carries the context expert n1_i0.
    holders = sorted(n for n, m in adapter.unet.named_modules()
                     if isinstance(m, MultiSlicedLoRALinear) and "n1_i0" in m.experts)
    assert list(assembly.router.modules_gated) == holders == assembly.gated_modules
    z, ts, cond = batch(adapter, [10, 700])
    gate = assembly.gate_for(ts, z)
    assert all(g.shape == (2,) for g in gate.values())
    assert {k for k in gate if isinstance(k, tuple)} == {(m, "n1_i0") for m in holders}
    before = {k: v.clone() for k, v in lora_state_dict(adapter.unet).items()}
    router_before = {k: v.clone() for k, v in assembly.router.state_dict().items()}
    state = train_router(assembly, dataset, cfg.assemble, output_dir=tmp_path)
    assert state.step == cfg.assemble.max_steps and torch.isfinite(torch.tensor(state.loss))
    after = lora_state_dict(adapter.unet)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert lora_parameters(adapter.unet) == []
    changed = [k for k, v in assembly.router.state_dict().items() if not torch.equal(v, router_before[k])]
    assert "fc.bias" in changed
    assert assembly.gate == {}  # cleared after the run
    assert (tmp_path / "bundle.safetensors").exists() and (tmp_path / "assemble_summary.json").exists()
    summary = json.loads((tmp_path / "assemble_summary.json").read_text())
    assert summary["core"] == "n2" and [e["name"] for e in summary["experts"]] == ["n1_i0", "n2_i0", "n2_i1"]


def test_bundle_round_trip_reproduces_predictions_and_metadata(experts_dir, cfg, tmp_path):
    adapter = make_adapter()
    assembly = build_assembly(adapter, cfg, experts_dir=experts_dir)
    with torch.no_grad():  # per-module, per-t pattern that must survive the round trip
        assembly.router.fc.bias.copy_(torch.linspace(-1, 1, assembly.router.fc.bias.numel()))
        assembly.router.t_embed.weight[700].fill_(-1.3)
    path = tmp_path / "bundle.safetensors"
    save_bundle(assembly, path, extra_metadata={"steps": "0"})
    z, ts, cond = batch(adapter, [10, 700, 300])
    with torch.no_grad():
        expected = assembly.predict(z, ts, cond)
    assembly.detach()

    fresh = make_adapter()
    loaded = load_bundle(fresh, path)
    assert loaded.sets == ["n1", "n2"] and loaded.core == "n2" and loaded.expert_names == assembly.expert_names
    assert [e.to_dict() for e in loaded.experts] == [e.to_dict() for e in assembly.experts]
    assert torch.equal(loaded.router.fc.bias, assembly.router.fc.bias)
    with torch.no_grad():
        got = loaded.predict(z, ts, cond)
    torch.testing.assert_close(got, expected)
    with safe_open(str(path), framework="pt") as h:
        meta = h.metadata()
    assert meta["format"] == BUNDLE_FORMAT and meta["axis"] == "style"
    assert json.loads(meta["intervals"]) == {"n1": [[0, 999]], "n2": [[0, 499], [500, 999]]}
    keys = load_file(str(path)).keys()
    assert any(k.endswith(".experts.n2_i1.lora_head_indices") for k in keys)
    assert {k for k in keys if k.startswith("router.")} == {"router.fc.weight", "router.fc.bias", "router.t_embed.weight"}


def test_bundle_loader_refuses_gaps_missing_experts_and_stray_keys(experts_dir, cfg, tmp_path):
    adapter = make_adapter()
    assembly = build_assembly(adapter, cfg, experts_dir=experts_dir)
    path = tmp_path / "bundle.safetensors"
    save_bundle(assembly, path)
    state = load_file(str(path))
    with safe_open(str(path), framework="pt") as h:
        meta = dict(h.metadata())

    def tampered(name, state=state, meta=meta):
        p = tmp_path / f"{name}.safetensors"
        save_file(state, str(p), metadata=meta)
        return p

    gap = dict(meta, intervals=json.dumps({"n1": [[0, 999]], "n2": [[0, 498], [500, 999]]}))
    with pytest.raises(ValueError, match="contiguous"):
        load_bundle(make_adapter(), tampered("gap", meta=gap))

    missing = {k: v for k, v in state.items() if ".experts.n2_i1." not in k}
    with pytest.raises(ValueError, match="tensors cover experts"):
        load_bundle(make_adapter(), tampered("missing", state=missing))

    stray = dict(state, **{"down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A": torch.zeros(1)})
    with pytest.raises(ValueError, match="Unexpected tensor"):
        load_bundle(make_adapter(), tampered("stray", state=stray))

    no_router_meta = dict(meta, router="null")
    with pytest.raises(ValueError, match="declares no router"):
        load_bundle(make_adapter(), tampered("norouter", meta=no_router_meta))

    wrong_format = dict(meta, format="something-else")
    with pytest.raises(ValueError, match="not a"):
        load_bundle(make_adapter(), tampered("fmt", meta=wrong_format))

    wrong_core = dict(meta, core="n7")
    with pytest.raises(ValueError, match="core"):
        load_bundle(make_adapter(), tampered("core", meta=wrong_core))


def test_single_set_bundle_has_no_router_and_reloads(experts_dir, cfg, tmp_path):
    adapter = make_adapter()
    assembly = build_assembly(adapter, cfg, experts_dir=experts_dir, sets=["n2"], core="n2")
    path = tmp_path / "hard.safetensors"
    save_bundle(assembly, path)
    keys = load_file(str(path)).keys()
    assert not any(k.startswith("router.") for k in keys)
    loaded = load_bundle(make_adapter(), path)
    assert loaded.router is None and loaded.sets == ["n2"]
    gate = loaded.gate_for(torch.tensor([600]))
    assert gate["n2_i1"].tolist() == [1.0] and gate["n2_i0"].tolist() == [0.0]


def test_assembly_rejects_experts_that_disagree_with_their_scheme(experts_dir, cfg):
    adapter = make_adapter()
    infos = [load_expert_into(adapter.unet, experts_dir / n / "lora-final.safetensors")
             for n in ("n1_i0", "n2_i0", "n2_i1")]
    schemes = {"n1": build_scheme("n1", T), "n2": build_scheme("n2", T, cfg.intervals)}
    TSMAssembly(adapter, schemes=schemes, core="n2", experts=infos, router=None)  # fine
    bad = {"n1": schemes["n1"], "n2": build_scheme("n2", T, type(cfg.intervals)(schemes=[2], boundaries={"n2": [400]}))}
    with pytest.raises(ValueError, match="scheme says"):
        TSMAssembly(adapter, schemes=bad, core="n2", experts=infos, router=None)
    wrong_router = Router(["n2"], ["x"], latent_channels=4, num_train_timesteps=T)
    with pytest.raises(ValueError, match="Router covers"):
        TSMAssembly(adapter, schemes=schemes, core="n2", experts=infos, router=wrong_router)
    wrong_modules = Router(["n1"], ["x"], latent_channels=4, num_train_timesteps=T)
    with pytest.raises(ValueError, match="gates 1 module"):
        TSMAssembly(adapter, schemes=schemes, core="n2", experts=infos, router=wrong_modules)


def test_disabled_expert_is_stored_but_never_fires(experts_dir, cfg, tmp_path):
    """Control C2: the high-interval expert stays in the bundle with weight 0 everywhere."""
    schemes = {"n1": build_scheme("n1", T), "n2": build_scheme("n2", T)}
    routing = RoutingTable(schemes, core="n2", num_train_timesteps=T, disabled=["n2_i1"])
    gate = routing.gate(torch.tensor([10, 700]))
    assert gate["n2_i1"].tolist() == [0.0, 0.0] and gate["n2_i0"].tolist() == [1.0, 0.0]
    assert gate["n1_i0"].tolist() == [1.0, 1.0]
    with pytest.raises(ValueError, match="not in"):
        RoutingTable(schemes, core="n2", num_train_timesteps=T, disabled=["n9_i0"])
    with pytest.raises(ValueError, match="Every expert"):
        RoutingTable(schemes, core="n2", num_train_timesteps=T, disabled=["n1_i0", "n2_i0", "n2_i1"])

    adapter = make_adapter()
    assembly = build_assembly(adapter, cfg, experts_dir=experts_dir, disabled=["n2_i1"])
    assert assembly.disabled == ["n2_i1"] and assembly.expert_names == ["n1_i0", "n2_i0", "n2_i1"]
    z, ts, cond = batch(adapter, [700])
    with torch.no_grad():
        ours = assembly.predict(z, ts, cond)
    # At t=700 only n1_i0 (under its init gate 1.0) is left: same as the lone n1 model.
    lone = make_adapter()
    load_lora(lone.unet, experts_dir / "n1_i0" / "lora-final.safetensors", LoraConfig_(rank=4, alpha=4))
    with torch.no_grad():
        ref = lone.predict(z, ts, cond)
    torch.testing.assert_close(ours, ref)

    path = tmp_path / "c2.safetensors"
    save_bundle(assembly, path)
    with safe_open(str(path), framework="pt") as h:
        assert json.loads(h.metadata()["disabled"]) == ["n2_i1"]
    assert any(k.startswith("down_blocks") or k.startswith("up_blocks") or k.startswith("mid")
               for k in load_file(str(path)) if ".experts.n2_i1." in k)  # still stored
    assembly.detach()
    loaded = load_bundle(make_adapter(), path)
    assert loaded.disabled == ["n2_i1"]
    with torch.no_grad():
        again = loaded.predict(z, ts, cond)
    torch.testing.assert_close(again, ours)

    # Disabling the only context expert leaves the router nothing to gate.
    with pytest.raises(ValueError, match="no active context expert"):
        build_assembly(make_adapter(), cfg, experts_dir=experts_dir, disabled=["n1_i0"])
    hard = build_assembly(make_adapter(), cfg, experts_dir=experts_dir, disabled=["n1_i0"], with_router=False)
    assert hard.router is None and hard.context_experts == []
