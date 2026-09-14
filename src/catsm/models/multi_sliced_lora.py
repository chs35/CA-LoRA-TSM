"""Several sliced LoRA experts on one attention matrix.

:class:`calora.models.sliced_lora.SlicedLoRALinear` holds exactly one adapter
per matrix and refuses a second. Interval experts break that assumption: the CA
selection of interval *i* and of interval *j* may pick *different heads* of the
same ``to_q``, so each expert needs its own ``lora_head_indices``. The minimal
unit is therefore a **branch** — ``(A, B, head_indices)`` without the base
layer — and the module wrapping the matrix is a container of branches.

Three principles carry over from the single-adapter module unchanged:

* the wrapped ``nn.Linear`` keeps the submodule name ``base_layer``, so
  ``PROJECTION_PATTERN`` enumerates the same unit names on an adapted UNet;
* the branch parameters are named ``lora_A`` / ``lora_B`` / ``lora_head_indices``,
  so ``lora_parameters`` / ``lora_state_dict`` see them and a checkpoint key
  ``<module>.experts.<expert>.lora_head_indices`` states which heads it covers;
* attaching the same expert twice to a matrix is refused.

Routing is a **gate**: one mapping ``{expert: weight}`` shared by reference
across every container in the model. A weight is a float (hard routing) or a
``(B,)`` tensor (per-sample soft routing; gradients flow through it into the
router). Experts absent from the gate contribute nothing.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from calora.models.sliced_lora import (
    LORA_TYPE_BY_SLICE_DIM,
    HeadLoraPlan,
    SlicedLoRALinear,
    _replace_submodule,
    head_slice_index,
)

#: ``nn.ModuleDict`` rejects keys containing ``.``; keep names path-safe too.
EXPERT_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def validate_expert_name(name: str) -> str:
    if not isinstance(name, str) or EXPERT_NAME_PATTERN.match(name) is None:
        raise ValueError(
            f"Expert name must match {EXPERT_NAME_PATTERN.pattern} (no dots), got {name!r}"
        )
    return name


class SlicedLoRABranch(nn.Module):
    """One expert's slice of one matrix: the LoRA half of ``SlicedLoRALinear``.

    Shapes, validation and initialisation are those of
    :class:`~calora.models.sliced_lora.SlicedLoRALinear` so a single branch under
    a unit gate is numerically the single-adapter module (pinned by tests).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        head_indices: Sequence[int],
        num_heads: int,
        head_dim: int,
        slice_dim: str,
        rank: int,
        alpha: float,
        init_lora_weights: str | bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if slice_dim not in LORA_TYPE_BY_SLICE_DIM:
            raise ValueError(f"slice_dim must be 'row' or 'col', got {slice_dim!r}")
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        heads = tuple(int(h) for h in head_indices)
        if not heads:
            raise ValueError("A sliced LoRA needs at least one head")
        if sorted(set(heads)) != list(heads):
            raise ValueError(f"head_indices must be sorted and unique, got {heads}")
        if heads[-1] >= num_heads or heads[0] < 0:
            raise ValueError(f"head_indices {heads} outside [0, {num_heads})")
        sliced_axis = out_features if slice_dim == "row" else in_features
        if sliced_axis != num_heads * head_dim:
            raise ValueError(
                f"{slice_dim}-sliced axis is {sliced_axis} but heads*d_h = "
                f"{num_heads}*{head_dim} = {num_heads * head_dim}"
            )

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.slice_dim = slice_dim
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)

        width = len(heads) * head_dim
        if slice_dim == "row":  # type B: full A, row-sliced B
            a_shape, b_shape = (rank, in_features), (width, rank)
        else:  # type A: column-sliced A, full B
            a_shape, b_shape = (rank, width), (out_features, rank)
        self.lora_A = nn.Parameter(torch.empty(a_shape, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(b_shape, device=device, dtype=dtype))
        self.register_buffer(
            "lora_head_indices", torch.tensor(heads, dtype=torch.long, device=device)
        )
        self.register_buffer(
            "_slice_index", head_slice_index(heads, head_dim).to(device), persistent=False
        )
        self.reset_lora_parameters(init_lora_weights)

    def reset_lora_parameters(self, init_lora_weights: str | bool = True) -> None:
        if init_lora_weights == "gaussian":
            nn.init.normal_(self.lora_A, std=1.0 / self.rank)
        else:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def head_indices(self) -> tuple[int, ...]:
        return tuple(int(h) for h in self.lora_head_indices.tolist())

    @property
    def lora_type(self) -> str:
        return LORA_TYPE_BY_SLICE_DIM[self.slice_dim]

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """``s · B A x`` shaped like the *slice*: ``[..., d_out]`` (type A) or
        ``[..., |sel|·d_h]`` (type B)."""
        if self.slice_dim == "col":
            x = x.index_select(-1, self._slice_index)
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def extra_repr(self) -> str:
        return (
            f"type={self.lora_type}, heads={list(self.head_indices)}/{self.num_heads}, "
            f"d_h={self.head_dim}, r={self.rank}, alpha={self.alpha:g}, slice={self.slice_dim}"
        )


class MultiSlicedLoRALinear(nn.Module):
    """``nn.Linear`` plus any number of gated :class:`SlicedLoRABranch` experts.

    ``out = W0 x + Σ_e gate[e] · Δ_e(x)``, each ``Δ_e`` written only into its own
    head slice. The gate is set from outside (:func:`set_gate`) and read at
    forward time; it must stay set until ``backward()`` has run when the UNet
    uses (non-reentrant) gradient checkpointing, because the recomputation reads
    it again.
    """

    def __init__(self, base_layer: nn.Linear, *, num_heads: int, head_dim: int, slice_dim: str):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"base_layer must be nn.Linear, got {type(base_layer).__name__}")
        if slice_dim not in LORA_TYPE_BY_SLICE_DIM:
            raise ValueError(f"slice_dim must be 'row' or 'col', got {slice_dim!r}")
        sliced_axis = base_layer.out_features if slice_dim == "row" else base_layer.in_features
        if sliced_axis != num_heads * head_dim:
            raise ValueError(
                f"{slice_dim}-sliced axis is {sliced_axis} but heads*d_h = "
                f"{num_heads}*{head_dim} = {num_heads * head_dim}"
            )
        self.base_layer = base_layer  # name fixed: PROJECTION_PATTERN matches ".base_layer"
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.slice_dim = slice_dim
        self.experts = nn.ModuleDict()
        #: Shared routing context, assigned by :func:`set_gate`; not part of the state.
        self.gate: Mapping[object, float | torch.Tensor] | None = None
        #: This matrix's path in the UNet; set by :func:`attach_multi_sliced_lora`
        #: / :func:`set_gate` and used to look up per-module gate entries.
        self.module_name: str = ""
        #: Experts whose delta is currently folded into ``base_layer.weight``.
        self.merged: set[str] = set()

    # -- experts -----------------------------------------------------------------------

    def add_expert(
        self,
        name: str,
        *,
        head_indices: Sequence[int],
        rank: int,
        alpha: float,
        init_lora_weights: str | bool = True,
    ) -> SlicedLoRABranch:
        validate_expert_name(name)
        if name in self.experts:
            raise RuntimeError(f"Expert {name!r} is already attached here; attach once per matrix")
        weight = self.base_layer.weight
        branch = SlicedLoRABranch(
            self.base_layer.in_features,
            self.base_layer.out_features,
            head_indices=head_indices,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            slice_dim=self.slice_dim,
            rank=rank,
            alpha=alpha,
            init_lora_weights=init_lora_weights,
            device=weight.device,
            dtype=weight.dtype,
        )
        self.experts[name] = branch
        return branch

    # -- nn.Linear surface -------------------------------------------------------------

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def lora_type(self) -> str:
        return LORA_TYPE_BY_SLICE_DIM[self.slice_dim]

    # -- maths ---------------------------------------------------------------------------

    def gate_weight(self, name: str) -> float | torch.Tensor | None:
        """This matrix's weight for expert ``name``: the ``(module, expert)`` entry
        if the gate has one, else the expert's global entry, else ``None``."""
        gate = self.gate
        if not gate:
            return None
        w = gate.get((self.module_name, name))
        return gate.get(name) if w is None else w

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = self.base_layer(x, *args, **kwargs)
        if not self.gate:
            return out
        for name, branch in self.experts.items():
            if name in self.merged:
                continue  # already in the weights; adding it again would double it
            w = self.gate_weight(name)
            if w is None:
                continue
            if torch.is_tensor(w):
                if w.dim() == 0:
                    w = w.reshape(1).expand(x.shape[0])
                if w.dim() != 1 or w.shape[0] != x.shape[0]:
                    raise ValueError(
                        f"Gate for {name!r} must be a scalar or a (B,)={x.shape[0]} tensor, got "
                        f"{tuple(w.shape)}. (Classifier-free guidance doubles the batch: "
                        "repeat the gate for the unconditional half.)"
                    )
                delta = branch.delta(x)
                delta = delta * w.to(device=delta.device, dtype=delta.dtype).view(
                    -1, *([1] * (delta.dim() - 1))
                )
            else:
                w = float(w)
                if w == 0.0:
                    continue
                delta = branch.delta(x)
                if w != 1.0:
                    delta = delta * w
            delta = delta.to(out.dtype)
            if branch.slice_dim == "col":
                out = out + delta
            else:
                out = out.index_add(-1, branch._slice_index, delta)
        return out

    # -- merge / unmerge (hard routing only) ---------------------------------------------

    def _check_mergeable(self, name: str) -> SlicedLoRABranch:
        if name not in self.experts:
            raise KeyError(f"No expert {name!r} on this matrix; have {list(self.experts)}")
        w = self.gate_weight(name)
        if w is not None and (torch.is_tensor(w) or float(w) != 1.0):
            raise RuntimeError(
                f"Expert {name!r} is under a soft gate ({w!r}); merging is only defined for "
                "hard (weight 1) routing"
            )
        return self.experts[name]

    def merge(self, name: str) -> None:
        """Fold ``s·B A`` of one expert into ``base_layer.weight`` (idempotent)."""
        if name in self.merged:
            return
        branch = self._check_mergeable(name)
        weight = self.base_layer.weight
        axis = 0 if branch.slice_dim == "row" else 1
        with torch.no_grad():
            weight.data.index_add_(
                axis, branch._slice_index, branch.delta_weight().detach().to(weight.dtype)
            )
        self.merged.add(name)

    def unmerge(self, name: str) -> None:
        if name not in self.merged:
            return
        branch = self.experts[name]
        weight = self.base_layer.weight
        axis = 0 if branch.slice_dim == "row" else 1
        with torch.no_grad():
            weight.data.index_add_(
                axis, branch._slice_index, -branch.delta_weight().detach().to(weight.dtype)
            )
        self.merged.discard(name)

    def extra_repr(self) -> str:
        return (
            f"type={self.lora_type}, heads={self.num_heads}, d_h={self.head_dim}, "
            f"experts={list(self.experts)}"
        )


# --------------------------------------------------------------------------------------
# Model-level helpers
# --------------------------------------------------------------------------------------


def iter_multi_sliced_lora(model: nn.Module) -> Iterator[tuple[str, MultiSlicedLoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, MultiSlicedLoRALinear):
            yield name, module


def has_multi_sliced_lora(model: nn.Module) -> bool:
    return any(True for _ in iter_multi_sliced_lora(model))


def expert_names(model: nn.Module) -> list[str]:
    """Every expert attached anywhere in ``model``, sorted."""
    names: set[str] = set()
    for _, module in iter_multi_sliced_lora(model):
        names.update(module.experts.keys())
    return sorted(names)


def set_gate(model: nn.Module, gate: Mapping[object, float | torch.Tensor] | None) -> None:
    """Bind one mapping object to every container (mutate it, do not replace it).

    Also (re)stamps each container's ``module_name`` from ``named_modules`` so
    per-module gate entries resolve against the live model, not a stale name.
    """
    for name, module in iter_multi_sliced_lora(model):
        module.module_name = name
        module.gate = gate


def gated_modules(model: nn.Module, experts: Sequence[str]) -> list[str]:
    """Sorted paths of the containers carrying any of ``experts`` (the router's rows)."""
    wanted = set(experts)
    return sorted(
        name for name, module in iter_multi_sliced_lora(model) if wanted & set(module.experts)
    )


def clear_gate(model: nn.Module) -> None:
    set_gate(model, None)


def attach_multi_sliced_lora(
    model: nn.Module,
    plans: Sequence[HeadLoraPlan],
    *,
    expert: str,
    rank: int,
    alpha: float,
    init_lora_weights: str | bool = True,
) -> list[SlicedLoRABranch]:
    """Add ``expert``'s branches to the matrices named by ``plans``.

    A matrix that is still a plain ``nn.Linear`` is frozen and wrapped; one that
    already carries a container just gains a branch. A matrix under the
    single-adapter :class:`SlicedLoRALinear` is refused: experts must be loaded
    into a pristine UNet, never on top of a fostering-stage adapter.
    """
    validate_expert_name(expert)
    if not plans:
        raise ValueError("Selection is empty; nothing to adapt")
    branches: list[SlicedLoRABranch] = []
    for plan in plans:
        module = model.get_submodule(plan.module_name)
        if isinstance(module, SlicedLoRALinear):
            raise TypeError(
                f"{plan.module_name} carries a single-adapter SlicedLoRALinear; load experts "
                "into a pristine UNet"
            )
        if isinstance(module, nn.Linear):
            module.requires_grad_(False)
            container = MultiSlicedLoRALinear(
                module,
                num_heads=plan.num_heads,
                head_dim=plan.head_dim,
                slice_dim=plan.slice_dim,
            )
            container.module_name = plan.module_name
            _replace_submodule(model, plan.module_name, container)
        elif isinstance(module, MultiSlicedLoRALinear):
            container = module
            if (container.num_heads, container.head_dim, container.slice_dim) != (
                plan.num_heads, plan.head_dim, plan.slice_dim
            ):
                raise ValueError(f"{plan.module_name}: plan disagrees with the attached container")
        else:
            raise TypeError(
                f"{plan.module_name} is a {type(module).__name__}, not nn.Linear; "
                "head-slice LoRA adapts the fused projection itself"
            )
        branches.append(
            container.add_expert(
                expert,
                head_indices=plan.head_indices,
                rank=rank,
                alpha=alpha,
                init_lora_weights=init_lora_weights,
            )
        )
    return branches
