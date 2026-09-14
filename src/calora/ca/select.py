r"""Unit selection and timestep scoring.

Two things happen here.

**Selection (the paper).** Rank units by CA and keep the top ``fraction``.
``select_topk`` implements that, plus the ``random``/``all`` baselines the
ablation needs.

A *unit* is whatever the injected :class:`~calora.models.projections.ProjectionTable`
says it is — an attention head (``select.granularity: head``, the paper's unit
and the default) or a whole fused projection (``module``, the granularity
ablation). Nothing in this module branches on that: the scores are ``[U]``, the
table supplies ``numel`` and the grouping keys, and the only visible difference
is which ``budget_basis`` is the sensible default. The paper's budget is a
*count* of units ("top k% of projection layers") — ``unit_count`` — while the
module-unit runs already on disk used the parameter budget ``attn_params``; see
:data:`DEFAULT_BUDGET_BASIS`. Parameter statistics are reported under every
basis, because the report needs both readings.

**Timestep scoring (this project).** The paper picks ``t=81`` from four
candidates because that is where the style-CA and viewpoint-CA *distributions*
differ most — a heuristic over magnitudes only. Since
``CA = E[||g_c|| / ||g_d||]`` discards the direction of ``g_c``, two axes can
have very different CA magnitudes while pushing a projection the same way, or
identical magnitudes while pulling it apart. The direction statistics from
:mod:`calora.ca.stats` let us score a timestep by what we actually want: many
high-CA projections whose style and viewpoint gradients *disagree*.

Several scores are reported, deliberately, because they fail in different ways.
The two **primary** ones are ``expanded_weighted_similarity`` and
``conflict_count``; the others are diagnostics.

``expanded_weighted_similarity`` (primary)
    CA-weighted mean similarity over the top-``n·k`` projections, where ``k`` is
    the budget that would actually be trained and ``n`` is
    ``tscore_selection_multiplier``. Reading only the top-k makes the metric
    hostage to exactly where the budget boundary falls; widening to ``n·k`` and
    weighting by CA (normalised to sum to 1 within the set) keeps the extra
    projections in view without letting them dominate. Lower is better.

``conflict_count`` (primary)
    ``|{top-k by CA} ∩ {sim < tau}|`` where ``tau`` is a quantile of the
    similarities *pooled across all timesteps*. This is the project proposal's
    hypothesis stated literally. The pooling matters: a per-``t`` similarity
    quantile would make the count identical at every ``t`` by construction.
    ``tau`` is swept over ``conflict_sim_quantiles`` and the whole sweep is
    reported: if the timestep order flips between quantiles, the metric is too
    fragile to headline, and that has to be visible.

``selected_mean_similarity``
    The unweighted top-k reading. Kept as the direct answer to "will the layers I
    fine-tune here be concept-discriminative?" and as the control that shows what
    the ``n·k`` widening changed. Lower is better.

``score`` (:math:`S(t)`)
    :math:`\sum_\theta \widetilde{CA}(\theta,t)\,(1-\mathrm{sim}(\theta,t))` with
    :math:`\widetilde{CA}` summing to 1 within each ``t``. Smooth and uses every
    projection, but it keeps CA magnitude as a continuous weight while the
    decision it informs is binary (a projection is trained or it is not), so it
    is demoted to a diagnostic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from calora.ca.stats import MeasurementResult
from calora.models.projections import ProjectionTable, parse_unit_name

NORMALIZATIONS = ("zscore", "rank", "sum", "max", "none")
AGGREGATIONS = ("normalized_mean", "rank_mean", "single")
AXIS_COMBINERS = ("mean", "min", "max")
BUDGET_BASES = ("unit_count", "attn_params", "unet_params", "projection_count")
DEFAULT_CONFLICT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5)
DEFAULT_PRIMARY_QUANTILE = 0.3

#: What ``select.budget_basis: auto`` resolves to, per unit granularity.
#:
#: The paper's budget is a *count* of units ("top-k% of projection layers"), so
#: head granularity takes ``unit_count``. The module-granularity path keeps
#: ``attn_params``, which is what the preserved module-unit runs were selected
#: under; changing it would make the granularity ablation a comparison of two
#: things at once.
DEFAULT_BUDGET_BASIS: dict[str, str] = {"head": "unit_count", "module": "attn_params"}


# --------------------------------------------------------------------------------------
# Score preparation
# --------------------------------------------------------------------------------------


def normalize_scores(values: torch.Tensor, method: str = "sum") -> torch.Tensor:
    """Put per-timestep CA vectors on a comparable scale.

    ``sum`` (default) makes the vector a weight distribution, so any weighted
    average built from it is directly comparable across timesteps. ``rank`` is
    scale-free and robust to the heavy tail of CA values.
    """
    if method not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got {method!r}")
    values = values.to(torch.float64)
    if method == "none":
        return values
    if method == "zscore":
        std = values.std()
        return (values - values.mean()) / (std if std > 0 else 1.0)
    if method == "rank":
        order = values.argsort()
        ranks = torch.empty_like(values)
        ranks[order] = torch.arange(len(values), dtype=values.dtype)
        return ranks / max(len(values) - 1, 1)
    if method == "sum":
        total = values.sum()
        return values / (total if total > 0 else 1.0)
    maximum = values.abs().max()
    return values / (maximum if maximum > 0 else 1.0)


def axis_scores(
    result: MeasurementResult, axis: str = "style", combine: str | None = None
) -> torch.Tensor:
    """CA for one axis, or a combination across all axes.

    ``axis`` may name a concrete concept axis, or be one of ``mean``/``min``/``max``.
    """
    key = combine or axis
    if key in AXIS_COMBINERS:
        stacked = torch.stack([result.ca(a).to(torch.float64) for a in result.axes])
        if key == "mean":
            return stacked.mean(dim=0)
        return stacked.min(dim=0).values if key == "min" else stacked.max(dim=0).values
    if key not in result.per_axis:
        raise KeyError(
            f"Unknown selection axis {key!r}; measured axes are {list(result.axes)} "
            f"(or one of {AXIS_COMBINERS})"
        )
    return result.ca(key).to(torch.float64)


def aggregate_across_timesteps(
    results: Mapping[int, MeasurementResult],
    *,
    axis: str = "style",
    method: str = "normalized_mean",
    normalization: str = "sum",
    timesteps: Sequence[int] | None = None,
) -> torch.Tensor:
    """Combine per-timestep CA into a single ranking score.

    ``normalized_mean`` averages within-``t`` normalised CA; ``rank_mean``
    averages within-``t`` ranks, which ignores magnitude entirely and is the
    robust variant when one timestep has a much heavier CA tail than the others.
    """
    if method not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {method!r}")
    chosen = list(timesteps) if timesteps is not None else sorted(results)
    if not chosen:
        raise ValueError("No timesteps to aggregate")
    missing = [t for t in chosen if t not in results]
    if missing:
        raise KeyError(f"No measurement for timestep(s) {missing}")

    if method == "single":
        if len(chosen) != 1:
            raise ValueError(f"aggregation='single' needs exactly one timestep, got {chosen}")
        return axis_scores(results[chosen[0]], axis)

    per_t = [
        normalize_scores(
            axis_scores(results[t], axis),
            "rank" if method == "rank_mean" else normalization,
        )
        for t in chosen
    ]
    return torch.stack(per_t).mean(dim=0)


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------


@dataclass
class Selection:
    """The units to attach LoRA to, plus why.

    At ``granularity="module"`` the names are module paths and go straight into
    ``peft``. At ``granularity="head"`` they are ``<module>@h<index>``, a matrix
    may be only partly selected, and training goes through the sliced type A/B
    modules of :mod:`calora.models.sliced_lora` instead of ``peft`` — see
    :meth:`head_groups` and :func:`calora.train.lora.plan_sliced_lora`.
    """

    names: list[str]
    indices: list[int]
    scores: list[float]
    fraction_requested: float
    fraction_achieved: float
    params_selected: int
    params_total: int
    budget_basis: str
    strategy: str
    breakdown: dict[str, dict[str, int]] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)
    granularity: str = "module"
    units_total: int = 0
    """Size of the table this was selected from; ``0`` when unknown (a loaded
    selection that predates the field)."""

    def __len__(self) -> int:
        return len(self.names)

    @property
    def unit_fraction_achieved(self) -> float:
        """Selected units / total units — the quantity a ``unit_count`` budget caps.

        ``fraction_achieved`` stays a *parameter* fraction under every basis, so
        both readings are always available in the JSON; under ``unit_count`` it
        is a report, not the constraint.
        """
        return len(self.names) / self.units_total if self.units_total else 0.0

    def head_groups(self) -> dict[str, list[int]]:
        """Module path -> the selected head indices, sorted.

        ``{}`` at module granularity (nothing is sliced), so callers can branch
        on ``granularity`` alone.
        """
        if self.granularity != "head":
            return {}
        groups: dict[str, list[int]] = {}
        for name in self.names:
            module, head = parse_unit_name(name)
            if head is None:
                raise ValueError(
                    f"Selection declares head granularity but {name!r} carries no head index"
                )
            groups.setdefault(module, []).append(head)
        return {module: sorted(heads) for module, heads in groups.items()}

    def module_names(self) -> list[str]:
        """The distinct matrices touched, in first-appearance order."""
        return list(dict.fromkeys(parse_unit_name(name)[0] for name in self.names))

    def as_target_modules(self, head_counts: Mapping[str, int] | None = None) -> list[str]:
        """Exactly what ``peft.LoraConfig(target_modules=...)`` expects.

        Module granularity: the names themselves.

        Head granularity: the matrices *all* of whose heads were selected, for
        which a head-level and a matrix-level LoRA coincide. Deciding that needs
        ``H`` per matrix, so ``head_counts`` is required, and any partly-selected
        matrix raises — ``peft`` has no way to adapt part of a matrix, and
        widening it to the whole one would train parameters the budget never
        bought. Partly-selected matrices are trained through the sliced modules
        of :mod:`calora.models.sliced_lora` instead, which is what
        :func:`calora.train.lora.attach_lora` routes head selections to; this
        method stays the strict ``peft``-only view.
        """
        if self.granularity != "head":
            return list(self.names)
        groups = self.head_groups()
        if head_counts is None:
            raise ValueError(
                "A head-granularity selection needs head_counts to tell fully-selected "
                "matrices (type A) from partly-selected ones (type B)."
            )
        missing = [module for module in groups if module not in head_counts]
        if missing:
            raise KeyError(
                f"No head count for {len(missing)} selected matrices, e.g. {missing[:3]}"
            )
        partial = [module for module, heads in groups.items() if len(heads) != head_counts[module]]
        if partial:
            raise NotImplementedError(
                f"{len(partial)} of {len(groups)} selected matrices are only partly selected "
                f"(e.g. {partial[:3]}), and peft cannot express an adapter over part of a "
                "matrix. Use calora.train.lora.attach_lora, which routes head-granularity "
                "selections to the sliced type A/B modules."
            )
        return sorted(groups)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "granularity": self.granularity,
            "num_projections": len(self.names),
            "num_modules": len(self.module_names()),
            "units_total": self.units_total,
            "unit_fraction_achieved": self.unit_fraction_achieved,
            "params_selected": self.params_selected,
            "params_total": self.params_total,
            "fraction_requested": self.fraction_requested,
            "fraction_achieved": self.fraction_achieved,
            "budget_basis": self.budget_basis,
            "breakdown": self.breakdown,
            "meta": self.meta,
            "names": self.names,
            "scores": self.scores,
        }

    def jaccard(self, other: Selection) -> float:
        a, b = set(self.names), set(other.names)
        union = a | b
        return len(a & b) / len(union) if union else 1.0

    @classmethod
    def from_dict(cls, data: Mapping[str, object], table: ProjectionTable | None = None) -> Selection:
        """Rebuild a selection written by :meth:`to_dict`.

        ``indices`` are not serialised (they depend on the table ordering), so
        they are recovered from ``table`` when one is supplied. ``granularity``
        is read from the payload when present and otherwise inferred from the
        names, so selections written before head granularity existed load as the
        module selections they are.
        """
        names = list(data["names"])  # type: ignore[arg-type]
        indices = [table.index_of(n) for n in names] if table is not None else []
        granularity = data.get("granularity")
        if granularity is None:
            granularity = (
                "head" if any(parse_unit_name(n)[1] is not None for n in names) else "module"
            )
        return cls(
            names=names,
            indices=indices,
            scores=list(data.get("scores", [])),  # type: ignore[arg-type]
            fraction_requested=float(data.get("fraction_requested", 0.0)),  # type: ignore[arg-type]
            fraction_achieved=float(data.get("fraction_achieved", 0.0)),  # type: ignore[arg-type]
            params_selected=int(data.get("params_selected", 0)),  # type: ignore[arg-type]
            params_total=int(data.get("params_total", 0)),  # type: ignore[arg-type]
            budget_basis=str(data.get("budget_basis", "attn_params")),
            strategy=str(data.get("strategy", "loaded")),
            breakdown=dict(data.get("breakdown", {})),  # type: ignore[arg-type]
            meta=dict(data.get("meta", {})),  # type: ignore[arg-type]
            granularity=str(granularity),
            units_total=int(data.get("units_total", 0) or 0),  # type: ignore[arg-type]
        )


def _breakdown(table: ProjectionTable, indices: Sequence[int]) -> dict[str, dict[str, int]]:
    chosen = set(indices)
    out: dict[str, dict[str, int]] = {}
    for key in ("kind", "attn", "block_id"):
        counts: dict[str, int] = {}
        for group, members in sorted(table.group_indices(key).items()):
            counts[group] = sum(1 for i in members if i in chosen)
        out[key] = counts
    return out


#: Bases whose budget counts units rather than parameters.
_COUNTING_BASES = ("projection_count", "unit_count")


def _budget(table: ProjectionTable, fraction: float, basis: str, unet_params: int | None) -> float:
    if basis == "attn_params":
        return fraction * table.total_params
    if basis == "unet_params":
        if unet_params is None:
            raise ValueError("budget_basis='unet_params' requires unet_params")
        return fraction * unet_params
    if basis == "unit_count":
        # floor: the paper's "top 2%" of U units is |_0.02 U_|. Never zero, so a
        # tiny table still yields a selection rather than an empty arm.
        return float(max(1, math.floor(fraction * len(table))))
    if basis == "projection_count":
        # The older count basis, kept because the module-granularity results were
        # produced with its rounding. Differs from unit_count only in the ceil.
        return float(max(1, math.ceil(fraction * len(table))))
    raise ValueError(f"budget_basis must be one of {BUDGET_BASES}, got {basis!r}")


def select_topk(
    scores: torch.Tensor,
    table: ProjectionTable,
    *,
    fraction: float = 0.02,
    budget_basis: str = "attn_params",
    unet_params: int | None = None,
    strategy: str = "topk_ca",
    meta: dict[str, object] | None = None,
) -> Selection:
    """Greedily take the highest-scoring units that fit the budget.

    Under a *count* basis (``unit_count``, the paper's "top k% of projection
    layers", and the older ``projection_count``) every unit costs 1 and the
    budget is a number of units. Under a *parameter* basis (``attn_params``,
    ``unet_params``) a few wide projections consume it faster than many narrow
    ones, and a unit that would overflow the budget is skipped rather than
    ending the scan — a smaller, slightly lower-scoring one may still fit, and
    letting the widest layer through unchecked would blow past the very budget
    that makes the ablation arms comparable. Only if *nothing* fits (the budget
    is smaller than any single unit) is the top-scoring one returned alone, so
    the result is never empty.
    """
    if len(scores) != len(table):
        raise ValueError(f"scores has {len(scores)} entries but table has {len(table)}")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    budget = _budget(table, fraction, budget_basis, unet_params)
    order = torch.argsort(scores.to(torch.float64), descending=True).tolist()

    chosen: list[int] = []
    spent = 0.0
    counting_units = budget_basis in _COUNTING_BASES
    for index in order:
        cost = 1.0 if counting_units else float(table[index].numel)
        if spent + cost > budget:
            continue
        chosen.append(index)
        spent += cost
        if spent >= budget:
            break

    if not chosen:
        chosen = [order[0]]
    chosen.sort()
    params_selected = sum(table[i].numel for i in chosen)
    denominator = (
        table.total_params if budget_basis != "unet_params" else (unet_params or table.total_params)
    )
    return Selection(
        names=[table[i].name for i in chosen],
        indices=chosen,
        scores=[float(scores[i]) for i in chosen],
        fraction_requested=fraction,
        fraction_achieved=params_selected / denominator,
        params_selected=params_selected,
        params_total=denominator,
        budget_basis=budget_basis,
        strategy=strategy,
        breakdown=_breakdown(table, chosen),
        meta=dict(meta or {}),
        granularity=table.granularity,
        units_total=len(table),
    )


def select_random(
    table: ProjectionTable,
    *,
    fraction: float = 0.02,
    budget_basis: str = "attn_params",
    unet_params: int | None = None,
    seed: int = 0,
) -> Selection:
    """Random baseline: same parameter budget, no CA signal."""
    generator = torch.Generator().manual_seed(seed)
    scores = torch.rand(len(table), generator=generator, dtype=torch.float64)
    selection = select_topk(
        scores,
        table,
        fraction=fraction,
        budget_basis=budget_basis,
        unet_params=unet_params,
        strategy="random",
        meta={"seed": seed},
    )
    return selection


def select_all(table: ProjectionTable) -> Selection:
    """Full-attention LoRA baseline."""
    indices = list(range(len(table)))
    return Selection(
        names=table.names,
        indices=indices,
        scores=[0.0] * len(indices),
        fraction_requested=1.0,
        fraction_achieved=1.0,
        params_selected=table.total_params,
        params_total=table.total_params,
        budget_basis="attn_params",
        strategy="all",
        breakdown=_breakdown(table, indices),
        granularity=table.granularity,
        units_total=len(table),
    )


# --------------------------------------------------------------------------------------
# Timestep scoring
# --------------------------------------------------------------------------------------


@dataclass
class TimestepScore:
    timestep: int
    score: float
    selected_mean_similarity: float
    """Unweighted mean similarity over the top-k that would be trained."""

    expanded_mean_similarity: float
    """Unweighted mean over the top-n·k, for comparison with the weighted form."""

    expanded_weighted_similarity: float
    """CA-weighted mean over the top-n·k. The primary similarity metric."""

    conflict_count: int
    conflict_fraction: float
    mean_similarity: float
    num_selected: int
    num_expanded: int
    mean_ca: dict[str, float]
    axis_pair: tuple[str, str]
    similarity_source: str
    sim_threshold: float
    """``sim_thresholds[primary_quantile]``, repeated for convenience."""

    primary_quantile: float
    sim_thresholds: dict[float, float] = field(default_factory=dict)
    conflict_counts: dict[float, int] = field(default_factory=dict)
    conflict_fractions: dict[float, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestep": self.timestep,
            "score": self.score,
            "selected_mean_similarity": self.selected_mean_similarity,
            "expanded_mean_similarity": self.expanded_mean_similarity,
            "expanded_weighted_similarity": self.expanded_weighted_similarity,
            "conflict_count": self.conflict_count,
            "conflict_fraction": self.conflict_fraction,
            "mean_similarity": self.mean_similarity,
            "num_selected": self.num_selected,
            "num_expanded": self.num_expanded,
            "mean_ca": self.mean_ca,
            "axis_pair": list(self.axis_pair),
            "similarity_source": self.similarity_source,
            "sim_threshold": self.sim_threshold,
            "primary_quantile": self.primary_quantile,
            # JSON object keys must be strings; str(float) round-trips exactly.
            "sim_thresholds": {str(q): v for q, v in self.sim_thresholds.items()},
            "conflict_counts": {str(q): v for q, v in self.conflict_counts.items()},
            "conflict_fractions": {str(q): v for q, v in self.conflict_fractions.items()},
        }


#: metric getter and whether "best" means "largest", for every ``rank_by`` value.
RANK_KEYS: dict[str, tuple[Callable[[TimestepScore], float], bool]] = {
    "expanded_weighted_similarity": (lambda s: s.expanded_weighted_similarity, False),
    "conflict_count": (lambda s: float(s.conflict_count), True),
    "selected_mean_similarity": (lambda s: s.selected_mean_similarity, False),
    "score": (lambda s: s.score, True),
}


def _rank_key(rank_by: str) -> tuple[Callable[[TimestepScore], float], bool]:
    try:
        return RANK_KEYS[rank_by]
    except KeyError:
        raise ValueError(
            f"rank_by must be one of {sorted(RANK_KEYS)}, got {rank_by!r}"
        ) from None


def sort_timestep_scores(scores: Sequence[TimestepScore], rank_by: str) -> list[TimestepScore]:
    """Order scored timesteps best-first under ``rank_by``."""
    getter, descending = _rank_key(rank_by)
    return sorted(scores, key=getter, reverse=descending)


def pooled_similarity_thresholds(
    results: Mapping[int, MeasurementResult],
    axis_pair: tuple[str, str],
    *,
    quantiles: Sequence[float] = DEFAULT_CONFLICT_QUANTILES,
    source: str = "cos_of_means",
) -> dict[float, float]:
    """``tau`` per quantile, from the similarities of *all* timesteps pooled.

    Using a per-timestep quantile instead would make ``conflict_count`` constant
    across timesteps and the metric vacuous. The whole sweep is returned from one
    pooling pass so that every quantile sees the identical population.
    """
    if not quantiles:
        raise ValueError("At least one quantile is required")
    outside = [q for q in quantiles if not 0.0 <= q <= 1.0]
    if outside:
        raise ValueError(f"Quantiles must lie in [0, 1]; got {outside}")

    pooled = torch.cat(
        [results[t].similarity(*axis_pair, source=source).to(torch.float64) for t in sorted(results)]
    )
    ordered = sorted(quantiles)
    values = torch.quantile(pooled, torch.tensor(ordered, dtype=torch.float64))
    return {float(q): float(v) for q, v in zip(ordered, values)}


def score_timestep(
    result: MeasurementResult,
    table: ProjectionTable,
    *,
    axis_pair: tuple[str, str],
    sim_thresholds: Mapping[float, float],
    primary_quantile: float = DEFAULT_PRIMARY_QUANTILE,
    similarity_source: str = "cos_of_means",
    ca_weighting: str = "mean",
    ca_normalization: str = "sum",
    fraction: float = 0.02,
    selection_multiplier: int = 5,
    budget_basis: str = "attn_params",
    unet_params: int | None = None,
    selection_axis: str = "style",
) -> TimestepScore:
    """Score one timestep for how concept-discriminative its top-CA projections are."""
    if ca_weighting not in AXIS_COMBINERS:
        raise ValueError(f"ca_weighting must be one of {AXIS_COMBINERS}, got {ca_weighting!r}")
    if selection_multiplier < 1:
        raise ValueError(f"selection_multiplier must be >= 1, got {selection_multiplier}")
    if primary_quantile not in sim_thresholds:
        raise KeyError(
            f"primary quantile {primary_quantile} is not among the swept quantiles "
            f"{sorted(sim_thresholds)}"
        )

    similarity = result.similarity(*axis_pair, source=similarity_source).to(torch.float64)
    weights = normalize_scores(axis_scores(result, ca_weighting), ca_normalization)
    score = float((weights * (1.0 - similarity)).sum())

    ca = axis_scores(result, selection_axis)
    selection = select_topk(
        ca,
        table,
        fraction=fraction,
        budget_basis=budget_basis,
        unet_params=unet_params,
        strategy=f"topk_ca@t{result.timestep}",
    )
    # The same ranking, n times the budget. Cheap: nothing is trained from it.
    expanded = select_topk(
        ca,
        table,
        fraction=min(fraction * selection_multiplier, 1.0),
        budget_basis=budget_basis,
        unet_params=unet_params,
        strategy=f"topk_ca@t{result.timestep}x{selection_multiplier}",
    )

    selected_sim = similarity[torch.tensor(selection.indices, dtype=torch.long)]
    expanded_index = torch.tensor(expanded.indices, dtype=torch.long)
    expanded_sim = similarity[expanded_index]
    # Weights are the selection axis's CA renormalised *within* the expanded set,
    # so the metric is a proper weighted mean and stays comparable across t.
    expanded_ca = ca[expanded_index].clamp(min=0.0)
    total = float(expanded_ca.sum())
    if total > 0.0:
        expanded_weights = expanded_ca / total
    else:  # a degenerate timestep with no CA mass at all; fall back to uniform
        expanded_weights = torch.full_like(expanded_ca, 1.0 / len(expanded_ca))

    conflict_counts = {
        q: int((selected_sim < tau).sum()) for q, tau in sorted(sim_thresholds.items())
    }
    conflict_fractions = {
        q: count / max(len(selection), 1) for q, count in conflict_counts.items()
    }

    return TimestepScore(
        timestep=result.timestep,
        score=score,
        selected_mean_similarity=float(selected_sim.mean()),
        expanded_mean_similarity=float(expanded_sim.mean()),
        expanded_weighted_similarity=float((expanded_weights * expanded_sim).sum()),
        conflict_count=conflict_counts[primary_quantile],
        conflict_fraction=conflict_fractions[primary_quantile],
        mean_similarity=float(similarity.mean()),
        num_selected=len(selection),
        num_expanded=len(expanded),
        mean_ca={axis: float(result.ca(axis).mean()) for axis in result.axes},
        axis_pair=axis_pair,
        similarity_source=similarity_source,
        sim_threshold=sim_thresholds[primary_quantile],
        primary_quantile=primary_quantile,
        sim_thresholds=dict(sorted(sim_thresholds.items())),
        conflict_counts=conflict_counts,
        conflict_fractions=conflict_fractions,
    )


def rank_timesteps(
    results: Mapping[int, MeasurementResult],
    table: ProjectionTable,
    *,
    axis_pair: tuple[str, str] | None = None,
    similarity_source: str = "cos_of_means",
    conflict_quantiles: Sequence[float] = DEFAULT_CONFLICT_QUANTILES,
    primary_quantile: float = DEFAULT_PRIMARY_QUANTILE,
    ca_weighting: str = "mean",
    ca_normalization: str = "sum",
    fraction: float = 0.02,
    selection_multiplier: int = 5,
    budget_basis: str = "attn_params",
    unet_params: int | None = None,
    selection_axis: str = "style",
    rank_by: str = "expanded_weighted_similarity",
) -> list[TimestepScore]:
    """Score every measured timestep and return them best-first.

    The conflict thresholds are pooled over exactly the ``results`` passed in, so
    scoring a split half ranks it against *its own* pooled population rather than
    inheriting the full run's — otherwise the halves would share a threshold and
    their agreement would be flattered.
    """
    if not results:
        raise ValueError("No measurements to rank")
    _rank_key(rank_by)  # fail before doing the work if the key is unknown
    if axis_pair is None:
        first = results[next(iter(sorted(results)))]
        if len(first.axes) < 2:
            raise ValueError("Timestep scoring needs at least two concept axes")
        axis_pair = (first.axes[0], first.axes[1])

    thresholds = pooled_similarity_thresholds(
        results, axis_pair, quantiles=conflict_quantiles, source=similarity_source
    )
    scores = [
        score_timestep(
            results[t],
            table,
            axis_pair=axis_pair,
            sim_thresholds=thresholds,
            primary_quantile=primary_quantile,
            similarity_source=similarity_source,
            ca_weighting=ca_weighting,
            ca_normalization=ca_normalization,
            fraction=fraction,
            selection_multiplier=selection_multiplier,
            budget_basis=budget_basis,
            unet_params=unet_params,
            selection_axis=selection_axis,
        )
        for t in sorted(results)
    ]
    return sort_timestep_scores(scores, rank_by)


def timestep_rank_agreement(
    scores_a: Sequence[TimestepScore],
    scores_b: Sequence[TimestepScore],
    *,
    rank_by: str,
    top_n: int = 5,
) -> dict:
    """How much two independently-scored halves agree on the timestep ranking.

    Feed it the output of :func:`rank_timesteps` run separately on each half of a
    split-half measurement. Spearman covers the whole ordering; the top-``n``
    Jaccard covers the part that is actually acted on (the aggregated selection
    uses the top ``select.top_timesteps``). A high Spearman with a low overlap means the
    halves agree about the broad shape but not about the winner, which is the
    case that should stop a single-``t`` claim.
    """
    # Imported late: calora.analysis.stability imports this module for Selection.
    from calora.analysis.stability import jaccard, spearman

    getter, descending = _rank_key(rank_by)
    a = {s.timestep: s for s in scores_a}
    b = {s.timestep: s for s in scores_b}
    common = sorted(set(a) & set(b))

    def top(store: Mapping[int, TimestepScore]) -> list[int]:
        return sorted(common, key=lambda t: getter(store[t]), reverse=descending)[:top_n]

    top_a, top_b = top(a), top(b)
    rho = float("nan")
    if len(common) >= 2:
        rho = spearman(
            torch.tensor([getter(a[t]) for t in common], dtype=torch.float64),
            torch.tensor([getter(b[t]) for t in common], dtype=torch.float64),
        )
    return {
        "rank_by": rank_by,
        "num_common": len(common),
        "timesteps": common,
        "spearman": rho,
        "top_n": top_n,
        "top_a": top_a,
        "top_b": top_b,
        "top_overlap": jaccard(top_a, top_b),
        "num_top_shared": len(set(top_a) & set(top_b)),
    }
