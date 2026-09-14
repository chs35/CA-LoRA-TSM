"""LoRA attached to a *slice* of an attention matrix — the paper's type A / type B.

``peft`` can only adapt a whole ``nn.Linear``. Head-granularity selection
(``select.granularity: head``) generally buys only some of a matrix's heads, and
adapting the whole matrix would train parameters the budget never paid for, so
those arms need their own module. This is it.

The parameterisation follows the paper (Appendix A.2, Algorithms 3-4) exactly,
and the type name is a statement about the *slicing axis*, not about how much of
the matrix is covered:

**Type B** — ``to_q``/``to_k``/``to_v``, whose heads own output *rows*::

    A in R^{r x d_in}                     (full: every head reads the whole input)
    B in R^{(|sel| d_h) x r}              (only the selected heads' rows are written)
    out = W0 x ;  out[..., rows] += s * B A x

**Type A** — ``to_out.0``, whose heads own input *columns*::

    A in R^{r x (|sel| d_h)}              (only the selected heads' columns are read)
    B in R^{d_out x r}                    (full: the write covers the whole output)
    out = W0 x + s * B A x[..., cols]

``s = alpha / r`` as in standard LoRA, ``r`` is the module-level rank (64 in the
paper) — not a per-head rank, so there is no degeneracy at small ``d_h``.

Three properties are load-bearing and are pinned by tests:

* **Equivalence.** When every head of a matrix is selected the slice is the whole
  matrix and the module is numerically a standard full LoRA, forward and
  backward.
* **Locality.** ``ΔW`` is exactly zero outside the selected slice, before and
  after training: the unselected rows/columns of ``B`` (type B) simply do not
  exist, rather than being trained and hoped to stay small.
* **Identity preservation.** The wrapped ``nn.Linear`` keeps the submodule name
  ``base_layer``, which is what ``peft`` calls it too. ``PROJECTION_PATTERN``
  already matches a trailing ``.base_layer``, so CA re-measurement during
  training (the drift callback) enumerates the *same* unit names on an adapted model
  as on the pretrained one — which is the whole basis of the drift comparison.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from calora.models.projections import HEAD_SLICE_DIMS

#: Slicing axis -> the paper's LoRA type. Derived from
#: :data:`~calora.models.projections.HEAD_SLICE_DIMS` rather than from the
#: projection kind, so the "which axis does a head own" rule stays defined in
#: exactly one place.
LORA_TYPE_BY_SLICE_DIM: dict[str, str] = {"row": "B", "col": "A"}


def lora_type_for_kind(kind: str) -> str:
    """``"q"|"k"|"v" -> "B"``, ``"out" -> "A"``."""
    return LORA_TYPE_BY_SLICE_DIM[HEAD_SLICE_DIMS[kind]]


@dataclass(frozen=True)
class HeadLoraPlan:
    """One module's share of a head-granularity selection.

    Exactly one LoRA is attached per attention matrix, covering the *union* of
    that matrix's selected heads (paper: one adapter per module, not per head).
    """

    module_name: str
    kind: str
    head_indices: tuple[int, ...]
    num_heads: int
    head_dim: int
    slice_dim: str

    @property
    def lora_type(self) -> str:
        return LORA_TYPE_BY_SLICE_DIM[self.slice_dim]

    @property
    def is_full(self) -> bool:
        """All ``H`` heads selected: the slice is the whole matrix."""
        return len(self.head_indices) == self.num_heads

    @property
    def sliced_dim_size(self) -> int:
        return len(self.head_indices) * self.head_dim


def head_slice_index(head_indices: Sequence[int], head_dim: int) -> torch.Tensor:
    """Element indices of the selected heads, concatenated in head order.

    ``([0, 2], d_h=3) -> [0,1,2, 6,7,8]``. This is the ``indices`` vector of the
    paper's Algorithm 4; keeping the heads in ascending order is what makes the
    ``B`` rows (type B) / ``A`` columns (type A) correspond to heads in a fixed,
    checkpoint-stable way.
    """
    return torch.cat(
        [
            torch.arange(head * head_dim, (head + 1) * head_dim, dtype=torch.long)
            for head in head_indices
        ]
    )


class SlicedLoRALinear(nn.Module):
    """``nn.Linear`` plus a LoRA branch restricted to a set of attention heads.

    Not a ``peft`` layer: it mirrors the parts of that interface the rest of the
    project uses (``base_layer``, ``merge``/``unmerge``, ``lora_A``/``lora_B``
    parameter names) so that checkpointing, ``merged_lora`` and the
    ``"lora_"``-prefix conventions work identically for both paths.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        head_indices: Sequence[int],
        num_heads: int,
        head_dim: int,
        slice_dim: str,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init_lora_weights: str | bool = True,
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

        sliced_axis = base_layer.out_features if slice_dim == "row" else base_layer.in_features
        if sliced_axis != num_heads * head_dim:
            raise ValueError(
                f"{slice_dim}-sliced axis is {sliced_axis} but heads*d_h = "
                f"{num_heads}*{head_dim} = {num_heads * head_dim}"
            )

        self.base_layer = base_layer
        self.slice_dim = slice_dim
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.merged = False
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        width = len(heads) * head_dim
        device, dtype = base_layer.weight.device, base_layer.weight.dtype
        if slice_dim == "row":  # type B: full A, row-sliced B
            a_shape = (rank, base_layer.in_features)
            b_shape = (width, rank)
        else:  # type A: column-sliced A, full B
            a_shape = (rank, width)
            b_shape = (base_layer.out_features, rank)
        self.lora_A = nn.Parameter(torch.empty(a_shape, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(b_shape, device=device, dtype=dtype))

        # Persistent: the checkpoint has to state which heads it covers, and the
        # tensors are the ground truth for that — never a selection JSON that may
        # have moved on.
        self.register_buffer(
            "lora_head_indices", torch.tensor(heads, dtype=torch.long, device=device)
        )
        # Derived, so not persisted: the element indices of those heads.
        self.register_buffer(
            "_slice_index", head_slice_index(heads, head_dim).to(device), persistent=False
        )
        self.reset_lora_parameters(init_lora_weights)

    # -- construction helpers ----------------------------------------------------------

    def reset_lora_parameters(self, init_lora_weights: str | bool = True) -> None:
        """Standard LoRA init: ``B = 0`` so attaching does not change any output.

        ``A``'s distribution follows ``train.lora.init_lora_weights`` with the
        same meaning ``peft`` gives it, so the head path and the module path are
        initialised alike under one config.
        """
        if init_lora_weights == "gaussian":
            nn.init.normal_(self.lora_A, std=1.0 / self.rank)
        else:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def head_indices(self) -> tuple[int, ...]:
        return tuple(int(h) for h in self.lora_head_indices.tolist())

    # ``peft``'s LoraLayer forwards these to its own base layer, and some
    # diffusers attention processors read ``attn.to_q.weight.dtype``. Forwarding
    # them keeps a wrapped projection substitutable for the ``nn.Linear`` it
    # replaced; the *parameter* named ``weight`` still lives at
    # ``<module>.base_layer.weight``, which is what ProjectionTable binds to.
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

    @property
    def is_full(self) -> bool:
        return len(self.lora_head_indices) == self.num_heads

    # -- maths ---------------------------------------------------------------------------

    def delta_weight(self) -> torch.Tensor:
        """``s * B A`` — the update, shaped like the *slice*, not the matrix."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        base = self.base_layer(x, *args, **kwargs)
        if self.merged:
            # Weights already carry the delta; applying it again would double it.
            return base
        if self.slice_dim == "col":  # type A
            sliced = x.index_select(-1, self._slice_index)
            delta = F.linear(F.linear(self.lora_dropout(sliced), self.lora_A), self.lora_B)
            return base + delta.to(base.dtype) * self.scaling
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base.index_add(-1, self._slice_index, delta.to(base.dtype) * self.scaling)

    # -- merge / unmerge -------------------------------------------------------------------

    def merge(self) -> None:
        """Fold the delta into ``base_layer.weight`` (idempotent per state).

        Written through ``.data`` like ``peft`` does: during CA re-measurement the
        base weight is the parameter gradients are read from, and an in-place
        autograd op on a leaf that requires grad would raise.
        """
        if self.merged:
            return
        delta = self.delta_weight().detach()
        weight = self.base_layer.weight
        axis = 0 if self.slice_dim == "row" else 1
        with torch.no_grad():
            weight.data.index_add_(axis, self._slice_index, delta.to(weight.dtype))
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        delta = self.delta_weight().detach()
        weight = self.base_layer.weight
        axis = 0 if self.slice_dim == "row" else 1
        with torch.no_grad():
            weight.data.index_add_(axis, self._slice_index, -delta.to(weight.dtype))
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"type={self.lora_type}, heads={list(self.head_indices)}/{self.num_heads}, "
            f"d_h={self.head_dim}, r={self.rank}, alpha={self.alpha:g}, "
            f"slice={self.slice_dim}"
        )


# --------------------------------------------------------------------------------------
# Attaching to a UNet
# --------------------------------------------------------------------------------------


def iter_sliced_lora(model: nn.Module) -> Iterator[tuple[str, SlicedLoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, SlicedLoRALinear):
            yield name, module


def has_sliced_lora(model: nn.Module) -> bool:
    return any(True for _ in iter_sliced_lora(model))


def sliced_lora_head_groups(model: nn.Module) -> dict[str, list[int]]:
    """Module path -> the head indices its LoRA covers."""
    return {name: list(module.head_indices) for name, module in iter_sliced_lora(model)}


def _replace_submodule(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    """Swap ``root.<qualified_name>`` for ``replacement``.

    ``to_out.0`` lives inside an ``nn.ModuleList``, whose children are keyed by
    the string ``"0"``; ``setattr`` would work there too but going through
    ``__setitem__`` keeps the list's own bookkeeping intact.
    """
    parent_path, _, attribute = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and attribute.isdigit():
        parent[int(attribute)] = replacement
    else:
        setattr(parent, attribute, replacement)


def attach_sliced_lora(
    model: nn.Module,
    plans: Sequence[HeadLoraPlan],
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    init_lora_weights: str | bool = True,
) -> list[SlicedLoRALinear]:
    """Wrap each planned matrix in a :class:`SlicedLoRALinear`.

    The wrapped weights are frozen, as ``peft`` freezes the base layers it
    injects into: only ``A`` and ``B`` are trainable, whether or not the caller
    remembered to freeze the model first. (CA measurement re-enables the
    projection gradients it needs and restores them afterwards, so the drift
    path is unaffected.)

    Refuses to attach twice: a second adapter over the first would silently
    stack two deltas, and the checkpoint would then be unreproducible.
    """
    if not plans:
        raise ValueError("Selection is empty; nothing to adapt")
    attached: list[SlicedLoRALinear] = []
    for plan in plans:
        base = model.get_submodule(plan.module_name)
        if isinstance(base, SlicedLoRALinear):
            raise RuntimeError(
                f"{plan.module_name} already carries a sliced LoRA; attach once per model"
            )
        if not isinstance(base, nn.Linear):
            raise TypeError(
                f"{plan.module_name} is a {type(base).__name__}, not nn.Linear; "
                "head-slice LoRA adapts the fused projection itself"
            )
        base.requires_grad_(False)
        module = SlicedLoRALinear(
            base,
            head_indices=plan.head_indices,
            num_heads=plan.num_heads,
            head_dim=plan.head_dim,
            slice_dim=plan.slice_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            init_lora_weights=init_lora_weights,
        )
        _replace_submodule(model, plan.module_name, module)
        attached.append(module)
    return attached
