"""Enumerate the attention units of a UNet, at module or head granularity.

Every downstream stage keys off the identifiers produced here:

* CA measurement accumulates one statistic per :class:`ProjectionSpec`.
* Selection ranks those statistics and emits a subset of ``ProjectionSpec.name``.
* LoRA training feeds exactly those names to ``peft`` as ``target_modules``.

Two granularities exist, chosen with ``select.granularity``:

``module`` (the original, kept for the granularity ablation)
    One unit per fused ``nn.Linear``. ``name`` is the *module* path (e.g.
    ``down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q``), not the
    parameter path, because that is what ``peft`` matches against. The
    corresponding parameter is ``name + ".weight"``.

``head`` (default; the paper's unit)
    One unit per *theoretical attention head*: a row-slice of the fused matrix
    for ``to_q``/``to_k``/``to_v`` and a column-slice for ``to_out.0``. Those are
    exactly the axes along which the gradient separates per head. ``name`` is
    ``<module path>@h<index>``; ``@`` cannot occur in a module path, so a head
    identifier is never mistaken for a ``peft`` target and
    :func:`parse_unit_name` recovers the module unambiguously.

The head count ``H`` is always introspected from the attention module's
``heads`` attribute (diffusers ``Attention.heads``; the toy model mirrors it),
never hardcoded, and ``d_h = to_q.out_features // H`` must divide exactly.

Parameter counting keeps the module-granularity convention: weights only, no
biases. A head unit therefore costs ``weight.numel() // H``, and the head units
of a matrix sum to exactly the matrix's own count — which is what makes a
parameter budget comparable across the two granularities. (``to_out.0``'s bias
cannot be attributed to a head at all; under this convention it is counted at
neither granularity, so nothing has to be split.)
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import nn

#: ``to_out`` is an ``nn.ModuleList`` in diffusers (``[Linear, Dropout]``), so the
#: linear layer lives at ``to_out.0``. The toy model mirrors that layout.
#:
#: The optional trailing ``.base_layer`` matches a peft-wrapped module: after
#: LoRA injection the original ``...to_q`` is a ``lora.Linear`` and the real
#: ``nn.Linear`` moves one level down. Matching it lets the drift callback re-measure CA on
#: an adapted model while keeping the *same* projection identifiers, which is
#: what makes the drift comparison well defined.
PROJECTION_PATTERN = re.compile(
    r"(?:^|\.)attn(?P<attn_index>\d+)\.to_(?P<kind>q|k|v|out)(?:\.0)?(?P<wrapper>\.base_layer)?$"
)

PROJECTION_KINDS: tuple[str, ...] = ("q", "k", "v", "out")

#: Unit granularities. ``module`` = one fused ``nn.Linear``; ``head`` = one
#: theoretical attention head's slice of it.
GRANULARITIES: tuple[str, ...] = ("module", "head")

#: Separates a module path from a head index in a unit name. Chosen because it
#: cannot appear in a ``named_modules`` path, so ``peft`` target matching and a
#: head identifier can never be confused for one another.
HEAD_SEPARATOR = "@h"

#: Which axis of the fused weight a head owns, per projection kind. ``to_q``,
#: ``to_k`` and ``to_v`` all write into ``inner_dim`` (rows); ``to_out.0`` reads
#: from it (columns). Cross-attention is no exception: ``to_k``/``to_v`` take a
#: wider input but still produce ``inner_dim`` rows.
HEAD_SLICE_DIMS: dict[str, str] = {"q": "row", "k": "row", "v": "row", "out": "col"}

_BLOCK_PATTERN = re.compile(r"^(?P<block>down_blocks|up_blocks|mid_block)(?:\.(?P<index>\d+))?")


def head_unit_name(module_name: str, head_index: int) -> str:
    """``("...attn1.to_q", 3) -> "...attn1.to_q@h3"``."""
    return f"{module_name}{HEAD_SEPARATOR}{head_index}"


def parse_unit_name(name: str) -> tuple[str, int | None]:
    """Split a unit name into ``(module path, head index or None)``.

    The single place unit names are decoded. Selection JSON, drift reports and
    the LoRA wiring all route through it, so a change to the naming convention
    has exactly one edit site.
    """
    module, separator, suffix = name.rpartition(HEAD_SEPARATOR)
    if not separator:
        return name, None
    if not suffix.isdigit():
        raise ValueError(
            f"Malformed head unit name {name!r}: expected '<module>{HEAD_SEPARATOR}<index>'"
        )
    return module, int(suffix)


@dataclass(frozen=True)
class ProjectionSpec:
    """Static metadata for one unit: a whole projection, or one head of it."""

    name: str
    """Canonical identifier.

    At module granularity this is also a valid ``peft`` ``target_modules``
    entry. At head granularity it is ``<module path>@h<index>`` — see
    :func:`parse_unit_name`. Either way it is stable across LoRA injection, so
    statistics measured before and after training are directly comparable.
    """

    kind: str
    """One of ``q``, ``k``, ``v``, ``out``."""

    attn_index: int
    """1 for self-attention (``attn1``), 2 for cross-attention (``attn2``)."""

    block: str
    """``down``, ``mid``, ``up`` or ``other``."""

    block_index: int | None
    shape: tuple[int, ...]
    """Shape of *this unit's* slice of the weight (the whole weight at module
    granularity)."""

    numel: int
    """``shape`` product: the unit's weight element count, and its budget cost."""

    module_path: str = ""
    """Where the ``nn.Linear`` actually lives; differs from the module part of
    ``name`` only under a peft wrapper (``<name>.base_layer``)."""

    head_index: int | None = None
    """``None`` at module granularity; otherwise this head's index in ``[0, H)``."""

    num_heads: int = 1
    """``H`` for the owning attention module, introspected from ``Attention.heads``."""

    slice_dim: str = ""
    """``row`` or ``col`` for a head unit (which weight axis it owns); ``""``
    at module granularity."""

    param_count: int = 0
    """Base parameters attributed to this unit. Equals ``numel`` — weights only,
    matching the module-granularity convention; see the module docstring on
    biases. Carried explicitly so a future convention change is visible in the
    serialised table rather than implied by ``shape``."""

    def __post_init__(self) -> None:
        if not self.module_path:
            object.__setattr__(self, "module_path", self.module_name)
        if not self.param_count:
            object.__setattr__(self, "param_count", self.numel)
        if self.head_index is None:
            if self.slice_dim:
                raise ValueError(f"{self.name}: slice_dim is meaningless without a head_index")
            return
        if self.slice_dim not in ("row", "col"):
            raise ValueError(
                f"{self.name}: head units need slice_dim 'row' or 'col', got {self.slice_dim!r}"
            )
        if not 0 <= self.head_index < self.num_heads:
            raise ValueError(
                f"{self.name}: head_index {self.head_index} outside [0, {self.num_heads})"
            )

    @property
    def module_name(self) -> str:
        """The canonical ``nn.Linear`` path, with any head suffix removed."""
        return parse_unit_name(self.name)[0]

    @property
    def is_head(self) -> bool:
        return self.head_index is not None

    @property
    def weight_name(self) -> str:
        return f"{self.module_path}.weight"

    @property
    def is_cross_attention(self) -> bool:
        return self.attn_index == 2

    def slice_weight(self, tensor: torch.Tensor) -> torch.Tensor:
        """This unit's view of a weight-shaped tensor (a gradient, typically).

        Identity at module granularity. At head granularity it is the row or
        column band this head owns — the only slicing rule in the project, so
        measurement, scoring and drift cannot disagree about it.
        """
        if self.head_index is None:
            return tensor
        if tensor.ndim != 2:
            raise ValueError(
                f"{self.name}: expected a 2-D weight-shaped tensor, got shape {tuple(tensor.shape)}"
            )
        if self.slice_dim == "row":
            width = self.shape[0]
            start = self.head_index * width
            if tensor.shape[0] != width * self.num_heads or tensor.shape[1] != self.shape[1]:
                raise ValueError(
                    f"{self.name}: tensor {tuple(tensor.shape)} does not match "
                    f"{self.num_heads} x {self.shape}"
                )
            return tensor[start : start + width]
        width = self.shape[1]
        start = self.head_index * width
        if tensor.shape[1] != width * self.num_heads or tensor.shape[0] != self.shape[0]:
            raise ValueError(
                f"{self.name}: tensor {tuple(tensor.shape)} does not match "
                f"{self.shape} x {self.num_heads}"
            )
        return tensor[:, start : start + width]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data


def _classify_block(name: str) -> tuple[str, int | None]:
    match = _BLOCK_PATTERN.match(name)
    if match is None:
        return "other", None
    block = match.group("block")
    index = match.group("index")
    short = {"down_blocks": "down", "up_blocks": "up", "mid_block": "mid"}[block]
    return short, int(index) if index is not None else None


def enumerate_attention_projections(
    model: nn.Module,
    *,
    kinds: Sequence[str] = PROJECTION_KINDS,
    attn_indices: Sequence[int] | None = None,
    pattern: re.Pattern[str] = PROJECTION_PATTERN,
) -> list[ProjectionSpec]:
    """Collect every ``nn.Linear`` attention projection in ``model``.

    Args:
        model: A UNet (diffusers ``UNet2DConditionModel``) or any module using the
            same ``attn{n}.to_{q,k,v,out.0}`` naming.
        kinds: Restrict to a subset of ``q``/``k``/``v``/``out``.
        attn_indices: Restrict to e.g. ``[1]`` for self-attention only.

    Returns:
        Specs in ``named_modules`` order, which is stable for a given model class.
    """
    kinds = tuple(kinds)
    unknown = set(kinds) - set(PROJECTION_KINDS)
    if unknown:
        raise ValueError(f"Unknown projection kind(s): {sorted(unknown)}")

    specs: list[ProjectionSpec] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        match = pattern.search(name)
        if match is None:
            continue
        kind = match.group("kind")
        attn_index = int(match.group("attn_index"))
        if kind not in kinds:
            continue
        if attn_indices is not None and attn_index not in attn_indices:
            continue
        canonical = name[: -len(match.group("wrapper"))] if match.group("wrapper") else name
        block, block_index = _classify_block(canonical)
        weight = module.weight
        specs.append(
            ProjectionSpec(
                name=canonical,
                kind=kind,
                attn_index=attn_index,
                block=block,
                block_index=block_index,
                shape=tuple(weight.shape),
                numel=weight.numel(),
                module_path=name,
            )
        )

    if not specs:
        raise ValueError(
            "No attention projections found. Check that the model uses the "
            "diffusers 'attn{n}.to_{q,k,v,out.0}' naming, or pass a custom pattern."
        )
    return specs


def _attention_path(module_name: str) -> str:
    """``"...attn1.to_out.0" -> "...attn1"`` (and the same for ``to_q/k/v``)."""
    return module_name.rsplit(".to_", 1)[0]


def attention_head_counts(model: nn.Module) -> dict[str, int]:
    """Attention module path -> its number of heads, read off the module itself.

    diffusers stores ``H`` on ``Attention.heads``; the toy model mirrors that
    attribute deliberately, so both go through this one path and nothing in the
    project ever hardcodes a head count. Modules without a ``to_q`` are ignored,
    which keeps unrelated ``heads`` attributes from being picked up.
    """
    counts: dict[str, int] = {}
    for name, module in model.named_modules():
        heads = getattr(module, "heads", None)
        if isinstance(heads, int) and not isinstance(heads, bool) and hasattr(module, "to_q"):
            if heads < 1:
                raise ValueError(f"Attention module {name!r} reports heads={heads}")
            counts[name] = heads
    return counts


def enumerate_attention_heads(
    model: nn.Module,
    *,
    specs: Sequence[ProjectionSpec] | None = None,
    **kwargs,
) -> list[ProjectionSpec]:
    """Expand every attention projection of ``model`` into its ``H`` head units.

    ``d_h`` is defined by ``to_q`` (``out_features // H``) and every other
    projection of the same attention module must agree with it along its sliced
    axis; anything else means the layout is not the one this slicing assumes, so
    it raises rather than producing silently mis-attributed statistics.

    Order is ``named_modules`` order with the heads of one matrix consecutive,
    which is the canonical index order for every ``[U]``-shaped statistic.
    """
    module_specs = list(specs) if specs is not None else enumerate_attention_projections(
        model, **kwargs
    )
    counts = attention_head_counts(model)

    head_dims: dict[str, int] = {}
    for spec in module_specs:
        if spec.kind != "q":
            continue
        attn = _attention_path(spec.name)
        heads = counts.get(attn)
        if heads is None:
            raise ValueError(
                f"No head count for attention module {attn!r}: it exposes no int 'heads' "
                "attribute. Head granularity needs one (diffusers Attention.heads)."
            )
        if spec.shape[0] % heads:
            raise ValueError(
                f"{spec.name}: out_features {spec.shape[0]} is not divisible by heads={heads}"
            )
        head_dims[attn] = spec.shape[0] // heads

    units: list[ProjectionSpec] = []
    for spec in module_specs:
        attn = _attention_path(spec.name)
        heads = counts.get(attn)
        if heads is None:
            raise ValueError(
                f"No head count for attention module {attn!r}: it exposes no int 'heads' "
                "attribute. Head granularity needs one (diffusers Attention.heads)."
            )
        head_dim = head_dims.get(attn)
        if head_dim is None:
            raise ValueError(
                f"Attention module {attn!r} has no to_q projection, so d_h is undefined. "
                "Restrict `kinds` consistently or measure at module granularity."
            )
        slice_dim = HEAD_SLICE_DIMS[spec.kind]
        axis = 0 if slice_dim == "row" else 1
        if spec.shape[axis] != heads * head_dim:
            raise ValueError(
                f"{spec.name}: sliced axis {axis} is {spec.shape[axis]}, expected "
                f"heads*d_h = {heads}*{head_dim} = {heads * head_dim}"
            )
        unit_shape = (
            (head_dim, spec.shape[1]) if slice_dim == "row" else (spec.shape[0], head_dim)
        )
        unit_numel = unit_shape[0] * unit_shape[1]
        for head in range(heads):
            units.append(
                ProjectionSpec(
                    name=head_unit_name(spec.name, head),
                    kind=spec.kind,
                    attn_index=spec.attn_index,
                    block=spec.block,
                    block_index=spec.block_index,
                    shape=unit_shape,
                    numel=unit_numel,
                    module_path=spec.module_path,
                    head_index=head,
                    num_heads=heads,
                    slice_dim=slice_dim,
                    param_count=unit_numel,
                )
            )
    return units


class ProjectionTable:
    """An ordered, indexable view over units plus their live parameters.

    The order defined here is the canonical index order for every ``[U]``-shaped
    statistic tensor in the project (``[P]`` at module granularity, where a unit
    *is* a projection).

    A table is homogeneous: either every unit is a whole module or every unit is
    a head. Mixing the two would make a budget, a Jaccard or a Spearman a
    comparison between incommensurable populations, so it is rejected at
    construction.
    """

    def __init__(self, specs: Sequence[ProjectionSpec], model: nn.Module | None = None):
        self.specs: tuple[ProjectionSpec, ...] = tuple(specs)
        self._index = {spec.name: i for i, spec in enumerate(self.specs)}
        if len(self._index) != len(self.specs):
            raise ValueError("Duplicate projection names in table")
        kinds = {spec.is_head for spec in self.specs}
        if len(kinds) > 1:
            raise ValueError(
                "Projection table mixes module-granularity and head-granularity units; "
                "every statistic keyed off it would compare incommensurable populations."
            )
        self.granularity: str = "head" if kinds == {True} else "module"
        self._parameters: dict[str, nn.Parameter] | None = None
        if model is not None:
            self.bind(model)

    # -- construction ----------------------------------------------------------------

    @classmethod
    def from_model(
        cls, model: nn.Module, *, granularity: str = "module", **kwargs
    ) -> ProjectionTable:
        """Build a table for ``model``.

        ``granularity`` defaults to ``module`` — the API-level default is the
        conservative one, while ``select.granularity`` (default ``head``) is what
        the pipeline actually passes.
        """
        if granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")
        specs = enumerate_attention_projections(model, **kwargs)
        if granularity == "head":
            specs = enumerate_attention_heads(model, specs=specs)
        return cls(specs, model=model)

    def bind(self, model: nn.Module) -> ProjectionTable:
        """Attach the live ``.weight`` parameters of ``model`` to this table."""
        params = dict(model.named_parameters())
        bound: dict[str, nn.Parameter] = {}
        missing: list[str] = []
        for spec in self.specs:
            param = params.get(spec.weight_name)
            if param is None:
                missing.append(spec.weight_name)
            else:
                bound[spec.name] = param
        if missing:
            raise KeyError(f"Model is missing {len(missing)} projection weights, e.g. {missing[:3]}")
        self._parameters = bound
        return self

    # -- container protocol ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Iterator[ProjectionSpec]:
        return iter(self.specs)

    def __getitem__(self, key: int | str) -> ProjectionSpec:
        if isinstance(key, str):
            return self.specs[self._index[key]]
        return self.specs[key]

    def __contains__(self, name: object) -> bool:
        return name in self._index

    # -- accessors -------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    @property
    def module_names(self) -> list[str]:
        """The distinct ``nn.Linear`` paths behind the units, in table order."""
        return list(dict.fromkeys(spec.module_name for spec in self.specs))

    @property
    def head_counts(self) -> dict[str, int]:
        """Module path -> ``H``; ``{}`` at module granularity."""
        return {spec.module_name: spec.num_heads for spec in self.specs if spec.is_head}

    @property
    def numels(self) -> list[int]:
        return [spec.numel for spec in self.specs]

    @property
    def total_params(self) -> int:
        return sum(spec.numel for spec in self.specs)

    def index_of(self, name: str) -> int:
        return self._index[name]

    @property
    def parameters(self) -> dict[str, nn.Parameter]:
        if self._parameters is None:
            raise RuntimeError("ProjectionTable is not bound to a model; call bind(model)")
        return self._parameters

    def iter_parameters(self) -> Iterator[tuple[ProjectionSpec, nn.Parameter]]:
        params = self.parameters
        for spec in self.specs:
            yield spec, params[spec.name]

    def requires_grad_(self, flag: bool = True) -> ProjectionTable:
        for param in self.parameters.values():
            param.requires_grad_(flag)
        return self

    def zero_grad(self) -> None:
        for param in self.parameters.values():
            param.grad = None

    # -- filtering / grouping --------------------------------------------------------

    def filter(
        self,
        *,
        kinds: Iterable[str] | None = None,
        attn_indices: Iterable[int] | None = None,
        blocks: Iterable[str] | None = None,
    ) -> ProjectionTable:
        kinds_set = set(kinds) if kinds is not None else None
        attn_set = set(attn_indices) if attn_indices is not None else None
        blocks_set = set(blocks) if blocks is not None else None
        kept = [
            spec
            for spec in self.specs
            if (kinds_set is None or spec.kind in kinds_set)
            and (attn_set is None or spec.attn_index in attn_set)
            and (blocks_set is None or spec.block in blocks_set)
        ]
        table = ProjectionTable(kept)
        if self._parameters is not None:
            table._parameters = {spec.name: self._parameters[spec.name] for spec in kept}
        return table

    def group_indices(self, key: str) -> dict[str, list[int]]:
        """Bucket projection indices by ``kind``, ``attn``, ``block`` or ``block_id``."""
        getters = {
            "kind": lambda s: s.kind,
            "attn": lambda s: f"attn{s.attn_index}",
            "block": lambda s: s.block,
            "block_id": lambda s: f"{s.block}{'' if s.block_index is None else s.block_index}",
        }
        if key not in getters:
            raise ValueError(f"Unknown grouping key {key!r}; expected one of {sorted(getters)}")
        getter = getters[key]
        groups: dict[str, list[int]] = {}
        for i, spec in enumerate(self.specs):
            groups.setdefault(getter(spec), []).append(i)
        return groups

    def numel_tensor(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.tensor(self.numels, dtype=torch.float64, device=device)

    def summary(self) -> dict[str, object]:
        by_kind = {k: len(v) for k, v in sorted(self.group_indices("kind").items())}
        by_block = {k: len(v) for k, v in sorted(self.group_indices("block_id").items())}
        out: dict[str, object] = {
            "granularity": self.granularity,
            "num_projections": len(self),
            "total_params": self.total_params,
            "by_kind": by_kind,
            "by_block": by_block,
        }
        if self.granularity == "head":
            out["num_modules"] = len(self.module_names)
            counts = self.head_counts
            out["heads_per_module"] = {
                str(h): sum(1 for v in counts.values() if v == h)
                for h in sorted(set(counts.values()))
            }
        return out

    def to_records(self) -> list[dict]:
        return [spec.to_dict() for spec in self.specs]

    @classmethod
    def from_records(cls, records: Sequence[dict]) -> ProjectionTable:
        specs = [
            ProjectionSpec(
                name=r["name"],
                kind=r["kind"],
                attn_index=int(r["attn_index"]),
                block=r["block"],
                block_index=None if r["block_index"] is None else int(r["block_index"]),
                shape=tuple(r["shape"]),
                numel=int(r["numel"]),
                # Older records predate head granularity: an absent module_path
                # meant "same as name", which is still right for a module unit.
                module_path=r.get("module_path", "") or parse_unit_name(r["name"])[0],
                head_index=None if r.get("head_index") is None else int(r["head_index"]),
                num_heads=int(r.get("num_heads", 1)),
                slice_dim=str(r.get("slice_dim", "")),
                param_count=int(r.get("param_count", 0)),
            )
            for r in records
        ]
        return cls(specs)
