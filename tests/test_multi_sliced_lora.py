from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from calora.models.projections import ProjectionTable
from calora.models.sliced_lora import HeadLoraPlan, SlicedLoRALinear
from calora.train.lora import lora_parameters, lora_state_dict, plan_sliced_lora
from catsm.models.multi_sliced_lora import (
    MultiSlicedLoRALinear,
    SlicedLoRABranch,
    attach_multi_sliced_lora,
    clear_gate,
    expert_names,
    gated_modules,
    has_multi_sliced_lora,
    set_gate,
)
from conftest import make_adapter, random_head_selection

H, DH, DIN = 4, 8, 16  # heads, head dim, input width for a q-like matrix


def linear(out_f: int, in_f: int, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(in_f, out_f)


def single_and_multi(slice_dim: str, heads=(1, 3), rank=2, alpha=4.0):
    """The same matrix wrapped both ways with identical LoRA initialisation."""
    out_f, in_f = (H * DH, DIN) if slice_dim == "row" else (DIN, H * DH)
    base_a, base_b = linear(out_f, in_f), linear(out_f, in_f)
    torch.manual_seed(1)
    single = SlicedLoRALinear(base_a, head_indices=heads, num_heads=H, head_dim=DH,
                              slice_dim=slice_dim, rank=rank, alpha=alpha)
    multi = MultiSlicedLoRALinear(base_b, num_heads=H, head_dim=DH, slice_dim=slice_dim)
    torch.manual_seed(1)
    multi.add_expert("e0", head_indices=heads, rank=rank, alpha=alpha)
    # A trained LoRA has a non-zero B; give both the same one.
    with torch.no_grad():
        single.lora_B.normal_()
        multi.experts["e0"].lora_B.copy_(single.lora_B)
    return single, multi


@pytest.mark.parametrize("slice_dim", ["row", "col"])
def test_one_expert_with_unit_gate_is_bitwise_a_sliced_lora(slice_dim):
    single, multi = single_and_multi(slice_dim)
    x = torch.randn(3, 5, single.in_features)
    expected = single(x)
    multi.gate = {"e0": 1.0}
    assert torch.equal(multi(x), expected)
    multi.gate = {"e0": torch.ones(3)}
    assert torch.equal(multi(x), expected)
    assert torch.equal(multi.experts["e0"].delta_weight(), single.delta_weight())


def test_no_gate_or_zero_gate_returns_the_base_output():
    single, multi = single_and_multi("row")
    x = torch.randn(2, 7, single.in_features)
    base = multi.base_layer(x)
    multi.gate = None
    assert torch.equal(multi(x), base)
    multi.gate = {}
    assert torch.equal(multi(x), base)
    multi.gate = {"e0": 0.0}
    assert torch.equal(multi(x), base)
    multi.gate = {"other": 1.0}  # an expert this matrix does not carry
    assert torch.equal(multi(x), base)


def test_two_experts_sum_their_slices_and_per_sample_gates_scale_rows():
    _, multi = single_and_multi("row", heads=(0, 1))
    torch.manual_seed(5)
    multi.add_expert("e1", head_indices=(1, 3), rank=2, alpha=4.0)
    with torch.no_grad():
        multi.experts["e1"].lora_B.normal_()
    x = torch.randn(3, 5, multi.in_features)
    base = multi.base_layer(x)

    def padded(name):
        br = multi.experts[name]
        full = torch.zeros_like(base)
        return full.index_add(-1, br._slice_index, br.delta(x))

    multi.gate = {"e0": 1.0, "e1": 1.0}
    torch.testing.assert_close(multi(x), base + padded("e0") + padded("e1"))
    multi.gate = {"e0": torch.tensor([1.0, 0.0, 0.5]), "e1": torch.tensor([0.0, 1.0, 0.5])}
    out = multi(x)
    torch.testing.assert_close(out[0], (base + padded("e0"))[0])
    torch.testing.assert_close(out[1], (base + padded("e1"))[1])
    torch.testing.assert_close(out[2], (base + 0.5 * padded("e0") + 0.5 * padded("e1"))[2])
    with pytest.raises(ValueError):  # a CFG-doubled batch must not pass silently
        multi.gate = {"e0": torch.ones(6)}
        multi(x)


def test_per_module_gate_entry_beats_the_global_one():
    _, multi = single_and_multi("row")
    multi.module_name = "blk.attn1.to_q"
    x = torch.randn(2, 3, multi.in_features)
    base = multi.base_layer(x)
    multi.gate = {"e0": 1.0}
    full = multi(x)
    multi.gate = {"e0": 1.0, ("blk.attn1.to_q", "e0"): 0.0}
    assert torch.equal(multi(x), base)  # this matrix's own entry wins
    multi.gate = {("other.to_q", "e0"): 0.0, "e0": 1.0}
    assert torch.equal(multi(x), full)  # another matrix's entry is ignored
    multi.gate = {("blk.attn1.to_q", "e0"): torch.tensor([1.0, 0.0])}
    out = multi(x)
    assert torch.equal(out[0], full[0]) and torch.equal(out[1], base[1])


def test_gate_gradient_reaches_a_router_parameter_and_frozen_experts_get_none():
    _, multi = single_and_multi("col")
    for p in multi.parameters():
        p.requires_grad_(False)
    weight = nn.Parameter(torch.full((3,), 0.5))  # a raw router output for this matrix
    multi.gate = {"e0": weight * 1.0}
    x = torch.randn(3, 5, multi.in_features)
    multi(x).pow(2).sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in multi.parameters())


def test_gate_gradient_survives_non_reentrant_checkpoint():
    _, multi = single_and_multi("row")
    for p in multi.parameters():
        p.requires_grad_(False)
    weight = nn.Parameter(torch.full((2,), 0.5))
    multi.gate = {"e0": weight * 1.0}
    x = torch.randn(2, 4, multi.in_features)
    out = checkpoint(multi, x, use_reentrant=False)
    out.pow(2).sum().backward()
    assert weight.grad is not None and weight.grad.abs().sum() > 0


def test_add_expert_rejects_duplicates_dotted_names_and_unsorted_heads():
    _, multi = single_and_multi("row")
    with pytest.raises(RuntimeError):
        multi.add_expert("e0", head_indices=(0,), rank=2, alpha=4.0)
    with pytest.raises(ValueError):
        multi.add_expert("n4.i0", head_indices=(0,), rank=2, alpha=4.0)
    with pytest.raises(ValueError):
        multi.add_expert("e1", head_indices=(3, 1), rank=2, alpha=4.0)
    with pytest.raises(ValueError):
        SlicedLoRABranch(DIN, H * DH, head_indices=(0,), num_heads=H, head_dim=DH + 1,
                         slice_dim="row", rank=2, alpha=4.0)


def test_merge_matches_the_gated_forward_and_refuses_soft_gates():
    for slice_dim in ("row", "col"):
        single, multi = single_and_multi(slice_dim)
        x = torch.randn(2, 3, multi.in_features)
        multi.gate = {"e0": 1.0}
        gated = multi(x)
        original = multi.base_layer.weight.detach().clone()
        multi.merge("e0")
        assert "e0" in multi.merged
        torch.testing.assert_close(multi(x), gated)  # forward skips merged experts
        multi.unmerge("e0")
        torch.testing.assert_close(multi.base_layer.weight, original)
        multi.gate = {"e0": torch.full((2,), 0.5)}
        with pytest.raises(RuntimeError):
            multi.merge("e0")


def test_attach_reuses_containers_keeps_projection_names_and_filters_see_branches():
    adapter = make_adapter()
    unet = adapter.unet
    table = ProjectionTable.from_model(unet, granularity="head")
    names_before = table.names
    sel_a = random_head_selection(table, seed=1)
    sel_b = random_head_selection(table, seed=2)
    assert set(sel_a.names) != set(sel_b.names)

    attach_multi_sliced_lora(unet, plan_sliced_lora(unet, sel_a), expert="n2_i0", rank=4, alpha=4)
    # Planning on the wrapped model resolves to the containers, and the unit
    # names the CA pipeline sees are unchanged.
    plans_b = plan_sliced_lora(unet, sel_b)
    attach_multi_sliced_lora(unet, plans_b, expert="n2_i1", rank=4, alpha=4)
    assert ProjectionTable.from_model(unet, granularity="head").names == names_before
    assert has_multi_sliced_lora(unet) and expert_names(unet) == ["n2_i0", "n2_i1"]

    shared = set(sel_a.module_names()) & set(sel_b.module_names())
    containers = dict(unet.named_modules())
    for module in shared:
        assert isinstance(containers[module], MultiSlicedLoRALinear)
        assert set(containers[module].experts) == {"n2_i0", "n2_i1"}
    only_b = set(sel_b.module_names()) - set(sel_a.module_names())
    for module in only_b:
        assert set(containers[module].experts) == {"n2_i1"}

    keys = lora_state_dict(unet)
    assert all(".experts." in k and (".lora_A" in k or ".lora_B" in k or k.endswith(".lora_head_indices"))
               for k in keys)
    assert not any(k.startswith("router.") for k in keys)
    n_matrices = len(set(sel_a.module_names()) | set(sel_b.module_names()))
    n_branches = len(sel_a.module_names()) + len(sel_b.module_names())
    assert len(lora_parameters(unet)) == 2 * n_branches
    assert sum(1 for _ in unet.modules() if isinstance(_, MultiSlicedLoRALinear)) == n_matrices

    with pytest.raises(RuntimeError):  # same expert twice
        attach_multi_sliced_lora(unet, plans_b, expert="n2_i1", rank=4, alpha=4)

    # set_gate binds one object everywhere and stamps module names; clear_gate unbinds it.
    gate: dict = {}
    set_gate(unet, gate)
    assert all(m.gate is gate and m.module_name == n
               for n, m in unet.named_modules() if isinstance(m, MultiSlicedLoRALinear))
    assert gated_modules(unet, ["n2_i1"]) == sorted(sel_b.module_names())
    clear_gate(unet)
    assert all(m.gate is None for _, m in unet.named_modules() if isinstance(m, MultiSlicedLoRALinear))


def test_attach_refuses_a_single_adapter_model():
    from calora.train.lora import attach_lora
    from calora.config import LoraConfig_

    adapter = make_adapter()
    table = ProjectionTable.from_model(adapter.unet, granularity="head")
    sel = random_head_selection(table, seed=3)
    attach_lora(adapter.unet, sel, LoraConfig_(rank=4, alpha=4))
    with pytest.raises(TypeError):
        attach_multi_sliced_lora(adapter.unet, plan_sliced_lora(adapter.unet, sel),
                                 expert="n1_i0", rank=4, alpha=4)


def test_bf16_autocast_forward_is_finite_with_fp32_branches():
    _, multi = single_and_multi("row")
    multi.gate = {"e0": torch.full((2,), 0.5)}
    x = torch.randn(2, 3, multi.in_features)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = multi(x)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()


def test_plan_head_dim_matches_container():
    plan = HeadLoraPlan(module_name="x", kind="q", head_indices=(0,), num_heads=H, head_dim=DH,
                        slice_dim="row")
    assert plan.lora_type == "B" and plan.sliced_dim_size == DH
