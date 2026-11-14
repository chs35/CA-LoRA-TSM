"""Does the CA-derived projection set stay valid during training?

CA-LoRA measures Concept-Awareness once, on the *pretrained* weights, and then
trains only the top-k projections for 10k steps. That is sound only if the CA
ranking is stable under the update — if the top-2% set drifts, the layers being
trained are no longer the concept-aware ones, and the method's premise weakens.

This module re-measures CA on the current (LoRA-adapted) weights at intervals
and reports:

``jaccard``
    Overlap of the top-k set with the step-0 set. The headline number: how much
    of what we chose to train would we still choose now.
``spearman``
    Rank correlation over *all* units, which catches gradual reordering that a
    set-overlap metric misses until it crosses the top-k boundary.
``kind_mass`` / ``block_mass``
    Where the CA mass sits across q/k/v/out and across UNet blocks. A shift here
    is a qualitative change in which part of attention carries the concept.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

from calora.ca.select import Selection
from calora.models.projections import ProjectionTable

#: Allowed values of ``measure.drift_autocast``.
DRIFT_AUTOCAST_MODES = ("no", "bf16")


def drift_autocast_context(device: torch.device, mode: str):
    """Autocast for the CA *drift* measurement only (``measure.drift_autocast``).

    ``"bf16"`` roughly halves the snapshot cost, which is ~50% of a training
    arm's wall clock. The accumulators are unaffected: CA is read off
    ``param.grad`` of fp32 parameters, so the sums stay fp32 whatever dtype the
    forward ran in — only the measured value moves, by the amount the
    equivalence gate bounds. ``"no"`` returns the null context rather than a
    disabled ``autocast`` so the default path is bit-identical to code that never
    knew about this option.
    """
    if mode == "no":
        return nullcontext()
    if mode not in DRIFT_AUTOCAST_MODES:
        raise ValueError(
            f"Unknown drift_autocast {mode!r}; expected one of {list(DRIFT_AUTOCAST_MODES)}"
        )
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Ranks with ties averaged, as Spearman's rho requires."""
    values = values.to(torch.float64)
    n = len(values)
    order = values.argsort()
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = torch.arange(n, dtype=torch.float64)

    sorted_values = values[order]
    start = 0
    for i in range(1, n + 1):
        if i == n or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                tied = order[start:i]
                ranks[tied] = ranks[tied].mean()
            start = i
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two score vectors."""
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float(torch.dot(ra, rb) / denom) if denom > 0 else float("nan")


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def kind_mass(scores: torch.Tensor, table: ProjectionTable, key: str = "kind") -> dict[str, float]:
    """Fraction of total CA mass carried by each q/k/v/out (or block) group."""
    scores = scores.to(torch.float64).clamp(min=0.0)
    total = float(scores.sum())
    if total <= 0:
        return {group: 0.0 for group in table.group_indices(key)}
    return {
        group: float(scores[torch.tensor(members, dtype=torch.long)].sum()) / total
        for group, members in sorted(table.group_indices(key).items())
    }


@dataclass
class DriftSnapshot:
    """CA state at one training step, compared against the step-0 baseline."""

    step: int
    timestep: int
    jaccard_vs_baseline: float
    spearman_vs_baseline: float
    jaccard_vs_previous: float
    jaccard_vs_trained: float
    mean_ca: float
    kind_mass: dict[str, float]
    block_mass: dict[str, float]
    num_selected: int
    entered: list[str] = field(default_factory=list)
    left: list[str] = field(default_factory=list)
    #: Wall clock of the re-measurement behind this snapshot. Recorded so a
    #: drift-tracking run's cost can be separated from its optimisation steps.
    #: One measurement serves every axis, so each axis's snapshot carries the same
    #: value; do not add them up across axes.
    seconds: float = 0.0
    #: Which concept axis's CA these numbers are about.
    axis: str | None = None
    #: Part of ``seconds`` spent regenerating x0 from the current weights
    #: (``measure.drift_source: selfgen_current``); 0 when x0 is fixed.
    generation_seconds: float = 0.0
    #: sha256 prefix of the latents this snapshot measured, so a reader can tell
    #: a regenerated x0 from the fixed one.
    x0_fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "step": self.step,
            "timestep": self.timestep,
            "jaccard_vs_baseline": self.jaccard_vs_baseline,
            "spearman_vs_baseline": self.spearman_vs_baseline,
            "jaccard_vs_previous": self.jaccard_vs_previous,
            "jaccard_vs_trained": self.jaccard_vs_trained,
            "mean_ca": self.mean_ca,
            "kind_mass": self.kind_mass,
            "block_mass": self.block_mass,
            "num_selected": self.num_selected,
            "num_entered": len(self.entered),
            "num_left": len(self.left),
            "entered": self.entered,
            "left": self.left,
            "seconds": self.seconds,
            "generation_seconds": self.generation_seconds,
            "x0_fingerprint": self.x0_fingerprint,
        }


class StabilityTracker:
    """Accumulates :class:`DriftSnapshot`s across a training run.

    Two baselines are tracked, and the distinction matters:

    ``baseline_selection``
        The top-k re-derived at step 0 from the *same* sample subset the
        snapshots use. Comparing against this isolates genuine drift, because
        estimator noise is held constant.

    ``trained_selection``
        The set that training actually adapted, derived from the full
        (``measure.num_samples``) run. ``jaccard_vs_trained`` therefore mixes
        drift with the noise of a smaller subset — but it is the number that
        answers "are the layers I am training still the concept-aware ones?".
    """

    def __init__(
        self,
        table: ProjectionTable,
        baseline_scores: torch.Tensor,
        baseline_selection: Selection,
        *,
        timestep: int,
        trained_selection: Selection | None = None,
        axis: str | None = None,
    ):
        if len(baseline_scores) != len(table):
            raise ValueError("baseline_scores length does not match the projection table")
        self.table = table
        self.timestep = timestep
        self.axis = axis
        self.baseline_scores = baseline_scores.detach().to(torch.float64).cpu()
        self.baseline_selection = baseline_selection
        self.trained_selection = trained_selection or baseline_selection
        self.snapshots: list[DriftSnapshot] = []
        self._previous_names: list[str] = list(baseline_selection.names)

    def record(
        self,
        step: int,
        scores: torch.Tensor,
        selection: Selection,
        *,
        seconds: float = 0.0,
        generation_seconds: float = 0.0,
        x0_fingerprint: str = "",
    ) -> DriftSnapshot:
        scores = scores.detach().to(torch.float64).cpu()
        baseline_names = self.baseline_selection.names
        current = list(selection.names)
        snapshot = DriftSnapshot(
            step=step,
            timestep=self.timestep,
            jaccard_vs_baseline=jaccard(baseline_names, current),
            spearman_vs_baseline=spearman(self.baseline_scores, scores),
            jaccard_vs_previous=jaccard(self._previous_names, current),
            jaccard_vs_trained=jaccard(self.trained_selection.names, current),
            mean_ca=float(scores.mean()),
            kind_mass=kind_mass(scores, self.table, "kind"),
            block_mass=kind_mass(scores, self.table, "block_id"),
            num_selected=len(current),
            entered=sorted(set(current) - set(baseline_names)),
            left=sorted(set(baseline_names) - set(current)),
            seconds=seconds,
            axis=self.axis,
            generation_seconds=generation_seconds,
            x0_fingerprint=x0_fingerprint,
        )
        self.snapshots.append(snapshot)
        self._previous_names = current
        return snapshot

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "timestep": self.timestep,
            "granularity": self.table.granularity,
            "num_units": len(self.table),
            "baseline_selection": self.baseline_selection.to_dict(),
            "trained_selection": self.trained_selection.to_dict(),
            "baseline_vs_trained_jaccard": self.baseline_selection.jaccard(self.trained_selection),
            "snapshots": [s.to_dict() for s in self.snapshots],
            "summary": self.summary(),
        }

    def summary(self) -> dict:
        if not self.snapshots:
            return {}
        jaccards = [s.jaccard_vs_baseline for s in self.snapshots]
        spearmans = [s.spearman_vs_baseline for s in self.snapshots]
        return {
            "num_snapshots": len(self.snapshots),
            "final_jaccard": jaccards[-1],
            "min_jaccard": min(jaccards),
            "final_spearman": spearmans[-1],
            "min_spearman": min(spearmans),
            "final_jaccard_vs_trained": self.snapshots[-1].jaccard_vs_trained,
            "ever_left": sorted({n for s in self.snapshots for n in s.left}),
            "total_seconds": sum(s.seconds for s in self.snapshots),
            "total_generation_seconds": sum(s.generation_seconds for s in self.snapshots),
        }


class MultiAxisStabilityTracker:
    """One :class:`StabilityTracker` per concept axis, fed by the same measurements.

    Every CA measurement already covers all axes, so tracking them all costs no
    extra gradient. ``primary_axis`` is the axis the run's selection was made on,
    and the one its drift should be read on: a viewpoint adapter is asked whether
    its heads stay in the *viewpoint* top-k. A control selected on no axis has
    ``primary_axis=None`` and no axis is privileged over another.
    """

    def __init__(
        self, trackers: Mapping[str, StabilityTracker], *, primary_axis: str | None = None
    ):
        if not trackers:
            raise ValueError("At least one axis is required")
        if primary_axis is not None and primary_axis not in trackers:
            raise ValueError(
                f"Primary axis {primary_axis!r} is not tracked; tracked: {sorted(trackers)}"
            )
        self.trackers = dict(trackers)
        self.primary_axis = primary_axis
        first = next(iter(self.trackers.values()))
        self.table = first.table
        self.timestep = first.timestep

    def to_dict(self) -> dict:
        return {
            "timestep": self.timestep,
            "granularity": self.table.granularity,
            "num_units": len(self.table),
            "primary_axis": self.primary_axis,
            "axes": {axis: tracker.to_dict() for axis, tracker in self.trackers.items()},
        }

    def summary(self) -> dict:
        return {
            "primary_axis": self.primary_axis,
            "axes": {axis: tracker.summary() for axis, tracker in self.trackers.items()},
        }


def drift_view(payload: Mapping, axis: str | None = None) -> Mapping:
    """One axis's drift out of a ``stability.json``; the primary axis by default.

    Files from before per-axis tracking hold a single axis at the top level and do
    not say which (it was ``select.axis``, whatever the run was selected on), so
    they are returned whole and cannot be asked for an axis by name.
    """
    if "axes" not in payload:
        if axis is not None:
            raise ValueError(f"This stability.json predates per-axis tracking; cannot read {axis!r}")
        return payload
    name = payload["primary_axis"] if axis is None else axis
    if name is None:
        raise ValueError(
            f"This run has no primary axis; name one of {sorted(payload['axes'])} explicitly"
        )
    return payload["axes"][name]


def compare_selections(selections: Mapping[str, Selection]) -> dict[str, dict[str, float]]:
    """Pairwise Jaccard between named selections (e.g. the ablation arms)."""
    keys = list(selections)
    return {
        a: {b: selections[a].jaccard(selections[b]) for b in keys}
        for a in keys
    }


# --------------------------------------------------------------------------------------
# Re-measuring CA on a LoRA-adapted model
# --------------------------------------------------------------------------------------


@contextmanager
def merged_lora(unet: nn.Module) -> Iterator[None]:
    """Temporarily fold LoRA deltas into the base weights.

    Needed because CA is a statement about the *effective* projection weight
    ``W = W0 + BA``. Inside this context ``<name>.base_layer.weight`` holds ``W``
    and the LoRA branch is bypassed, so ``dL/dW`` is exactly what we want. Any
    layer that fails to merge is unmerged again on exit.

    Covers both adapter kinds: ``peft`` layers (module granularity) and the
    sliced type A/B modules (head granularity). A head adapter merges into
    exactly its own rows/columns, so the enclosing measurement sees the same
    ``W`` an unadapted model would have outside the selected slice.
    """
    from calora.models.sliced_lora import SlicedLoRALinear

    sliced = [m for m in unet.modules() if isinstance(m, SlicedLoRALinear)]
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError as exc:  # pragma: no cover - environment dependent
        if not sliced:
            raise ImportError("peft is required to re-measure CA on an adapted model") from exc
        layers: list[nn.Module] = list(sliced)
    else:
        layers = [m for m in unet.modules() if isinstance(m, BaseTunerLayer)] + sliced

    merged: list[nn.Module] = []
    try:
        for layer in layers:
            layer.merge()
            merged.append(layer)
        yield
    finally:
        for layer in reversed(merged):
            layer.unmerge()


def build_stability_tracker(
    adapter,
    samples,
    *,
    axes: Sequence[str],
    timestep: int,
    selection_axis: str = "style",
    fraction: float = 0.02,
    budget_basis: str = "attn_params",
    seed: int = 0,
    trained_selection: Selection | None = None,
    drift_autocast: str = "no",
    granularity: str = "module",
) -> StabilityTracker:
    """Baseline the drift tracker by re-measuring CA at step 0.

    Crucially this uses the *same* ``samples`` the snapshots will use. Baselining
    against the full ``measure.num_samples`` run instead would make the step-0
    Jaccard less than 1 purely from estimator noise, and that noise would be
    indistinguishable from real drift in every later snapshot.

    ``drift_autocast`` has to match the snapshots' for the same reason: a step-0
    baseline measured in fp32 and snapshots measured in bf16 would differ by
    precision alone, and that difference would be reported as drift.

    ``granularity`` (``select.granularity``) has to match the trained selection's
    for ``jaccard_vs_trained`` to compare like with like — Jaccard and Spearman
    are computed over unit *names*, which differ between the two.

    Must be called before LoRA is attached.
    """
    table, result = _measure_baseline(
        adapter,
        samples,
        axes=axes,
        timestep=timestep,
        seed=seed,
        trained_selection=trained_selection,
        drift_autocast=drift_autocast,
        granularity=granularity,
    )
    return _tracker_from_baseline(
        result,
        table,
        selection_axis,
        fraction=fraction,
        budget_basis=budget_basis,
        timestep=timestep,
        trained_selection=trained_selection,
    )


def build_multi_axis_stability_tracker(
    adapter,
    samples,
    *,
    axes: Sequence[str],
    timestep: int,
    primary_axis: str | None = None,
    fraction: float = 0.02,
    budget_basis: str = "attn_params",
    seed: int = 0,
    trained_selection: Selection | None = None,
    drift_autocast: str = "no",
    granularity: str = "module",
) -> MultiAxisStabilityTracker:
    """The step-0 baseline of :func:`build_stability_tracker`, on every axis at once.

    A single measurement, so the axes stay paired on the same samples and noise.
    Must be called before LoRA is attached.
    """
    table, result = _measure_baseline(
        adapter,
        samples,
        axes=axes,
        timestep=timestep,
        seed=seed,
        trained_selection=trained_selection,
        drift_autocast=drift_autocast,
        granularity=granularity,
    )
    return MultiAxisStabilityTracker(
        {
            axis: _tracker_from_baseline(
                result,
                table,
                axis,
                fraction=fraction,
                budget_basis=budget_basis,
                timestep=timestep,
                trained_selection=trained_selection,
            )
            for axis in axes
        },
        primary_axis=primary_axis,
    )


def _measure_baseline(
    adapter,
    samples,
    *,
    axes: Sequence[str],
    timestep: int,
    seed: int,
    trained_selection: Selection | None,
    drift_autocast: str,
    granularity: str,
):
    from calora.ca.measure import measure_concept_awareness

    table = ProjectionTable.from_model(adapter.unet, granularity=granularity)
    if trained_selection is not None and trained_selection.granularity != table.granularity:
        raise ValueError(
            f"Drift granularity {table.granularity!r} does not match the trained selection's "
            f"{trained_selection.granularity!r}; jaccard_vs_trained would compare "
            "different populations."
        )
    with drift_autocast_context(adapter.device, drift_autocast):
        result = measure_concept_awareness(
            adapter,
            table,
            samples,
            timestep=timestep,
            axes=axes,
            accumulate_direction=False,
            compute_paired_cosine=False,
            seed=seed,
        )
    return table, result


def _tracker_from_baseline(
    result,
    table: ProjectionTable,
    axis: str,
    *,
    fraction: float,
    budget_basis: str,
    timestep: int,
    trained_selection: Selection | None,
) -> StabilityTracker:
    from calora.ca.select import axis_scores, select_topk

    scores = axis_scores(result, axis)
    selection = select_topk(
        scores,
        table,
        fraction=fraction,
        budget_basis=budget_basis,
        strategy="drift-baseline@step0",
    )
    return StabilityTracker(
        table,
        scores,
        selection,
        timestep=timestep,
        trained_selection=trained_selection,
        axis=axis,
    )


def make_stability_callback(
    adapter,
    samples,
    tracker: StabilityTracker | MultiAxisStabilityTracker,
    *,
    axes: Sequence[str],
    every: int,
    max_steps: int,
    selection_axis: str = "style",
    fraction: float = 0.02,
    budget_basis: str = "attn_params",
    seed: int = 0,
    drift_autocast: str = "no",
    on_snapshot: Callable[[DriftSnapshot], None] | None = None,
    sample_provider: Callable[[int], Sequence] | None = None,
):
    """A ``train_lora`` callback that re-measures CA every ``every`` steps.

    ``sample_provider`` (``measure.drift_source: selfgen_current``) is called with
    the step inside the merged-LoRA context and returns that snapshot's samples --
    x0 regenerated by the current weights (see
    :class:`calora.analysis.drift_data.SelfGenDriftProvider`). Without it every
    snapshot measures the fixed ``samples``. Generation runs outside the drift
    autocast: the base set it is paired with was sampled the same way.

    Re-uses :func:`calora.ca.measure.measure_concept_awareness` unchanged, so the
    snapshots and the step-0 baseline come from the same estimator with the same
    samples and seed. Runs with the LoRA deltas merged; see :func:`merged_lora`.

    ``drift_autocast`` (``measure.drift_autocast``) covers this measurement only —
    never the training forward and never scripts/measure_ca.py, whose output is
    the selection every arm is defined by. Pass the same value to
    :func:`build_stability_tracker`.

    With a :class:`MultiAxisStabilityTracker` every tracked axis is recorded and
    ``selection_axis`` is not consulted; ``on_snapshot`` fires once per axis.
    """
    from calora.ca.measure import measure_concept_awareness
    from calora.ca.select import axis_scores, select_topk

    trackers = (
        tracker.trackers
        if isinstance(tracker, MultiAxisStabilityTracker)
        else {selection_axis: tracker}
    )

    def callback(state) -> None:
        if every <= 0:
            return
        if state.step % every != 0 and state.step != max_steps:
            return

        started = time.perf_counter()
        was_training = adapter.unet.training
        with merged_lora(adapter.unet):
            table = ProjectionTable.from_model(
                adapter.unet, granularity=tracker.table.granularity
            )
            if table.names != tracker.table.names:
                raise RuntimeError(
                    "Projection identities changed after LoRA injection; drift numbers "
                    "would compare different layers."
                )
            snapshot_samples = samples if sample_provider is None else sample_provider(state.step)
            with drift_autocast_context(adapter.device, drift_autocast):
                result = measure_concept_awareness(
                    adapter,
                    table,
                    snapshot_samples,
                    timestep=tracker.timestep,
                    axes=axes,
                    accumulate_direction=False,
                    compute_paired_cosine=False,
                    seed=seed,
                )
        adapter.unet.train(was_training)
        if torch.cuda.is_available():
            # The merged-LoRA measurement graph leaves the allocator fragmented;
            # releasing the cached blocks keeps the next training step from
            # failing to find a contiguous one (the tsweep OOM lesson).
            torch.cuda.empty_cache()

        readings = {
            axis: axis_scores(result, axis) for axis in trackers
        }
        selections = {
            axis: select_topk(
                scores,
                tracker.table,
                fraction=fraction,
                budget_basis=budget_basis,
                strategy=f"drift@step{state.step}",
            )
            for axis, scores in readings.items()
        }
        seconds = time.perf_counter() - started
        generation_seconds = float(getattr(sample_provider, "last_generation_seconds", 0.0))
        fingerprint = str(getattr(sample_provider, "last_fingerprint", "")) or _fingerprint(
            snapshot_samples
        )
        for axis, single in trackers.items():
            snapshot = single.record(
                state.step,
                readings[axis],
                selections[axis],
                seconds=seconds,
                generation_seconds=generation_seconds,
                x0_fingerprint=fingerprint,
            )
            if on_snapshot is not None:
                on_snapshot(snapshot)

    return callback


def _fingerprint(samples) -> str:
    from calora.analysis.drift_data import latents_fingerprint

    return latents_fingerprint(samples)
