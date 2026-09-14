r"""Online accumulation of Concept-Awareness statistics.

Two families of statistics are produced per attention *unit* :math:`\theta` — a
whole fused projection or one attention head's slice of it, depending on the
:class:`~calora.models.projections.ProjectionTable` handed in. Nothing here
knows which: the table's specs supply the slice
(:meth:`~calora.models.projections.ProjectionSpec.slice_weight`) and every
buffer below is sized from ``spec.numel``, so the head path costs the same
memory as the module path (the head buffers of one matrix partition the matrix
buffer) and produces the same statistics evaluated one level down.

**Magnitude (the paper's CA).**

.. math::
    \mathrm{CA}(\theta) = \mathbb{E}\left[
        \frac{\lVert \nabla_\theta \mathcal{L}_{Concept} \rVert}
             {\lVert \nabla_\theta \mathcal{L}_{Diff} \rVert}\right]

**Direction (this project's addition).** Normalising by a *scalar* norm throws
away the direction of the concept gradient, which is exactly the information the
paper's timestep search needs. We therefore also accumulate

.. math::
    \hat{g}(\theta) = \mathbb{E}\left[
        \frac{\nabla_\theta \mathcal{L}_{Concept}}
             {\lVert \nabla_\theta \mathcal{L}_{Diff} \rVert}\right]

per concept axis, and report two cross-axis similarities:

``cos_of_means``
    :math:`\cos(\mathbb{E}[\hat g_{a}], \mathbb{E}[\hat g_{b}])` — the quantity the
    project proposal asks for. Requires the full mean vectors.

``mean_of_cos``
    :math:`\mathbb{E}[\cos(\hat g_{a}, \hat g_{b})]` — per-sample agreement, which
    does not wash out under sampling noise. Cheap, and a useful cross-check:
    a projection where the two axes genuinely conflict should score low on both.

They are not interchangeable. ``mean_of_cos`` is invariant to the ``L_Diff``
normalisation sample-by-sample (it is a positive scalar), so it is pure
direction agreement. ``cos_of_means`` averages *before* taking the cosine, which
implicitly weights each sample by :math:`1/\lVert g_d \rVert` — so it carries
some timestep-dependent magnitude information even when every sample's concept
direction is identical. Report both when interpreting S(t).

Both similarities compare *two* axes. Within one axis,
:meth:`MeasurementResult.direction_concentration` reports
:math:`\lVert \mathbb{E}[\hat g] \rVert / \mathbb{E}[\lVert \hat g \rVert] \in [0,1]`,
whose denominator is CA itself — so it costs nothing extra to measure. It says
how much the axis's own ``c_Aug`` prompts agree with each other, and a low value
is a warning label on ``cos_of_means``: a cosine between two nearly-cancelled
mean vectors is noise. It is derived from stored quantities and therefore not
serialised.

A sample may contribute **several augmentation slots** per axis
(``measure.augs_per_sample``). The grad-enabled forward and the ``L_Diff``
backward are shared within a sample, so each extra ``c_Aug`` prompt is a cheap
extra draw of the concept gradient. Every slot is an independent ratio sample for
CA and an independent term in the direction sum; slot ``j`` of one axis is paired
with slot ``j`` of every other, which is what keeps ``mean_of_cos`` a
like-for-like comparison.

Memory is independent of the sample count: only per-projection sums are kept.
See :meth:`ConceptAwarenessAccumulator.estimate_memory_bytes`.

Every one of those buffers is a plain sum, so two accumulators fed disjoint
samples add exactly (:meth:`ConceptAwarenessAccumulator.merge_`). That is what
makes split-half stability free: measure once into two accumulators, finalise
each for the half results, then merge for the full one — no third set of
direction buffers, which are as large as the attention parameters.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from calora.models.projections import ProjectionTable

GRANULARITIES = ("matrix", "elementwise")


def _pair_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def _fp32_norms(flats: Sequence[torch.Tensor]) -> torch.Tensor:
    """Norms of a list of flat tensors, computed in fp32 with a single host sync.

    ``torch.linalg.vector_norm(..., dtype=float32)`` accumulates in fp32 without
    materialising a cast copy, which matters when the model runs in fp16/bf16.
    """
    if not flats:
        return torch.zeros(0, dtype=torch.float64)
    stacked = torch.stack([torch.linalg.vector_norm(f, dtype=torch.float32) for f in flats])
    return stacked.to(device="cpu", dtype=torch.float64)


@dataclass
class AxisStats:
    """Finalised per-projection statistics for one concept axis."""

    ca_mean: torch.Tensor
    """[P] E[||g_c|| / ||g_d||] — the paper's Concept-Awareness."""

    ca_std: torch.Tensor
    """[P] sample standard deviation of the same ratio."""

    gc_norm_mean: torch.Tensor
    """[P] E[||g_c||], for diagnosing degenerate projections."""

    direction_mean: dict[str, torch.Tensor] | None
    """name -> flat E[g_c / ||g_d||]; ``None`` when direction accumulation is off."""

    def ca_sem(self, count: int) -> torch.Tensor:
        """Standard error of the CA estimate — use it to check ``num_samples``."""
        return self.ca_std / max(count, 1) ** 0.5


@dataclass
class MeasurementResult:
    """Everything one (model, timestep) measurement run produces."""

    timestep: int
    num_samples: int
    """Number of *contributions* — ``num_items * augs_per_sample``. This is the
    count CA and the direction statistics are averaged over, not the number of
    images; see :attr:`num_items` for that."""

    axes: tuple[str, ...]
    names: tuple[str, ...]
    per_axis: dict[str, AxisStats]
    gd_norm_mean: torch.Tensor
    valid_count: torch.Tensor
    cos_of_means: dict[str, torch.Tensor]
    mean_of_cos: dict[str, torch.Tensor]
    granularity: str = "matrix"
    """How the per-unit CA ratio was read: ``matrix`` (norm ratio) or
    ``elementwise``. Orthogonal to :attr:`unit_granularity`."""

    unit_granularity: str = "module"
    """What a unit *is*: ``module`` (fused ``nn.Linear``) or ``head``. Recorded so
    that measurements of the two kinds cannot be pooled by accident; files written
    before head granularity existed carry no such field and read back as
    ``module``, which is what they are."""

    num_items: int = 0
    """Number of distinct (x0, noise) draws behind :attr:`num_samples`."""

    augs_per_sample: int = 1
    table_records: list[dict] | None = None
    halves: tuple[MeasurementResult, MeasurementResult] | None = None
    """The even/odd sample halves this result was merged from, when
    ``measure.split_half`` is on. Deliberately not serialised — the halves are
    written as their own files so that a reader never has to know that a
    measurement is a container for two more."""

    @property
    def num_projections(self) -> int:
        return len(self.names)

    def ca(self, axis: str) -> torch.Tensor:
        return self.per_axis[axis].ca_mean

    def similarity(self, axis_a: str, axis_b: str, source: str = "cos_of_means") -> torch.Tensor:
        """Cross-axis similarity vector [P]; accepts the pair in either order."""
        try:
            store = {"cos_of_means": self.cos_of_means, "mean_of_cos": self.mean_of_cos}[source]
        except KeyError:
            raise ValueError(
                f"Unknown similarity source {source!r}; expected 'cos_of_means' or 'mean_of_cos'"
            ) from None
        for key in (_pair_key(axis_a, axis_b), _pair_key(axis_b, axis_a)):
            if key in store:
                return store[key]
        raise KeyError(
            f"No {source} for ({axis_a}, {axis_b}); available: {sorted(store)}. "
            "Check measure.accumulate_direction / measure.compute_paired_cosine."
        )

    def direction_concentration(self, axis: str, *, eps: float = 1e-12) -> torch.Tensor:
        r"""[P] within-axis direction agreement, in ``[0, 1]``. Free — nothing extra is measured.

        .. math::
            \frac{\lVert \mathbb{E}[\hat g] \rVert}{\mathbb{E}[\lVert \hat g \rVert]}
            = \frac{\lVert \texttt{direction\_mean} \rVert}{\texttt{ca\_mean}}

        The denominator *is* CA: :math:`\lVert \hat g \rVert =
        \lVert g_c \rVert / \lVert g_d \rVert` because the normalisation is a
        positive scalar. Both averages run over the same (sample, slot)
        contributions and the same valid mask, so the ratio is bounded by Jensen.

        1.0 means every ``c_Aug`` prompt of the axis pushes this projection the
        same way; near 0 means the axis's templates disagree among themselves and
        ``E[\hat g]`` is a cancellation artefact. That is a warning label on
        ``cos_of_means``, which is a cosine *between* two such means: a
        cross-axis similarity computed from two nearly-cancelled vectors is noise,
        not evidence. Read it per axis before reading the similarity.

        Returned in fp64 (the accumulators' precision) even though the stored
        vectors are fp32. Not serialised: it is a function of values
        :func:`save_measurement` already writes.
        """
        try:
            stats = self.per_axis[axis]
        except KeyError:
            raise KeyError(
                f"No statistics for axis {axis!r}; measured axes: {list(self.axes)}"
            ) from None
        if stats.direction_mean is None:
            raise ValueError(
                f"Axis {axis!r} carries no direction mean, so its concentration is undefined. "
                "Set measure.accumulate_direction=True (and measure.save_direction_vectors=True "
                "if this result was loaded from disk)."
            )
        missing = [name for name in self.names if name not in stats.direction_mean]
        if missing:
            raise ValueError(
                f"Direction mean for axis {axis!r} covers only part of the projection table: "
                f"{len(missing)} of {len(self.names)} missing, e.g. {missing[:3]}"
            )

        norms = torch.stack(
            [
                torch.linalg.vector_norm(stats.direction_mean[name], dtype=torch.float64)
                for name in self.names
            ]
        ).to(device="cpu")
        ca = stats.ca_mean.to(device="cpu", dtype=torch.float64)
        # A projection with CA ~ 0 has no concept gradient to have a direction;
        # report 0 rather than a ratio of two numerical zeros.
        return torch.where(ca > eps, norms / ca.clamp(min=eps), ca.new_zeros(1))

    def table(self) -> ProjectionTable:
        if self.table_records is None:
            raise ValueError("This result carries no projection metadata")
        table = ProjectionTable.from_records(self.table_records)
        if table.granularity != self.unit_granularity:
            raise ValueError(
                f"Measurement declares unit_granularity={self.unit_granularity!r} but its "
                f"table records are {table.granularity!r} units"
            )
        return table

    def summary(self) -> dict:
        out: dict = {
            "timestep": self.timestep,
            "num_samples": self.num_samples,
            "num_items": self.num_items,
            "augs_per_sample": self.augs_per_sample,
            "num_projections": self.num_projections,
            "granularity": self.granularity,
            "unit_granularity": self.unit_granularity,
            "axes": list(self.axes),
        }
        for axis, stats in self.per_axis.items():
            ca = stats.ca_mean
            out[f"ca/{axis}"] = {
                "mean": float(ca.mean()),
                "std": float(ca.std()),
                "min": float(ca.min()),
                "max": float(ca.max()),
                "mean_sem": float(stats.ca_sem(self.num_samples).mean()),
            }
        concentration = {
            axis: float(self.direction_concentration(axis).mean())
            for axis, stats in self.per_axis.items()
            if stats.direction_mean is not None
        }
        if concentration:
            # Averaged over projections: a per-axis health check on the c_Aug
            # template set, cheap enough to carry in every timestep summary.
            out["direction_concentration_mean"] = concentration
        for source, store in (
            ("cos_of_means", self.cos_of_means),
            ("mean_of_cos", self.mean_of_cos),
        ):
            for key, values in store.items():
                out[f"{source}/{key}"] = {
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
        return out


class _SampleScope:
    """Accumulates the gradients of a single (x0, noise) draw.

    Created by :meth:`ConceptAwarenessAccumulator.sample`. The caller must invoke
    :meth:`set_denominator` before any :meth:`add_axis`, and :meth:`commit` once
    at the end (automatic when used as a context manager).

    ``add_axis`` is called ``augs_per_sample`` times per concept axis — once per
    ``c_Aug`` prompt — and the calls for one axis must arrive in slot order,
    because slot ``j`` of every axis is paired for the cross-axis cosine.

    Nothing an axis contributes reaches the parent accumulator until
    :meth:`commit`. That matters because each ``add_axis`` is preceded by a
    backward pass that can fail (OOM, NaN guard, a keyboard interrupt): a scope
    abandoned half-way must leave the accumulator exactly as it found it, or the
    axes end up averaged over different sample sets and every cross-axis
    similarity is quietly wrong.
    """

    def __init__(self, parent: ConceptAwarenessAccumulator):
        self._parent = parent
        self._gd_norm: torch.Tensor | None = None
        self._gd_abs: dict[str, torch.Tensor] | None = None
        #: axis -> one unit-vector dict per augmentation slot.
        self._units: dict[str, list[dict[str, torch.Tensor]]] = {}
        #: axis -> one (gc_norm, ratio) pair per augmentation slot.
        self._pending: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        #: axis -> running sum of ``g_c / ||g_d||`` over this sample's slots.
        self._pending_directions: dict[str, dict[str, torch.Tensor]] = {}
        self._committed = False

    # -- inputs ------------------------------------------------------------------------

    def set_denominator(self, grads: dict[str, torch.Tensor]) -> None:
        r"""Record :math:`\nabla_\theta \mathcal{L}_{Diff}` for this sample."""
        parent = self._parent
        indices, flats = parent._collect(grads)
        self._gd_norm = parent._scatter(indices, _fp32_norms(flats))
        if parent.granularity == "elementwise":
            self._gd_abs = {
                parent.names[i]: flat.abs().to(device=parent.device, dtype=torch.float32)
                for i, flat in zip(indices, flats)
            }

    def add_axis(self, axis: str, grads: dict[str, torch.Tensor]) -> None:
        r"""Record :math:`\nabla_\theta \mathcal{L}_{Concept}` for one augmentation slot.

        Called once per ``(axis, c_Aug prompt)``; repeated calls for the same
        axis append slots rather than replacing the previous one.
        """
        parent = self._parent
        if self._gd_norm is None:
            raise RuntimeError("set_denominator() must be called before add_axis()")
        if axis not in parent.axes:
            raise KeyError(f"Unknown concept axis {axis!r}; expected one of {parent.axes}")
        slot = len(self._pending.get(axis, ()))
        if slot >= parent.augs_per_sample:
            raise RuntimeError(
                f"Axis {axis!r} already added {parent.augs_per_sample} augmentation slot(s) to "
                "this sample, which is what the accumulator was configured for"
            )

        indices, flats = parent._collect(grads)
        gc_norm = parent._scatter(indices, _fp32_norms(flats))
        gd_norm = self._gd_norm
        valid = gd_norm > parent.eps

        if parent.granularity == "matrix":
            ratio = torch.where(valid, gc_norm / gd_norm.clamp(min=parent.eps), gd_norm.new_zeros(1))
        else:
            ratio = self._elementwise_ratio(indices, flats)

        needs_vectors = parent.accumulate_direction or parent.compute_paired_cosine
        units: dict[str, torch.Tensor] = {}
        directions = (
            self._pending_directions.setdefault(axis, {}) if parent.accumulate_direction else {}
        )
        if needs_vectors:
            for i, flat in zip(indices, flats):
                name = parent.names[i]
                vec = flat.to(device=parent.device, dtype=torch.float32)
                if parent.accumulate_direction and bool(valid[i]):
                    # Scaled here (the gradient may be freed by the caller as soon
                    # as add_axis returns) but only summed into the parent by
                    # commit(), so an aborted sample contributes nothing. Slots of
                    # the same axis are summed in place, so the transient cost
                    # does not grow with augs_per_sample.
                    contribution = vec / float(gd_norm[i])
                    if name in directions:
                        directions[name].add_(contribution)
                    else:
                        directions[name] = contribution
                if parent.compute_paired_cosine:
                    norm = float(gc_norm[i])
                    if norm > 0.0:
                        # Unit vector; cos(ghat_a, ghat_b) == cos(g_a, g_b) because
                        # the L_Diff normalisation is a positive scalar.
                        units[name] = vec / norm

        self._pending.setdefault(axis, []).append((gc_norm, ratio))
        if parent.compute_paired_cosine:
            self._units.setdefault(axis, []).append(units)

    def _elementwise_ratio(
        self, indices: Sequence[int], flats: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """``mean_j |g_c,j| / (|g_d,j| + eps)`` — the "average per-weight CA" reading.

        Slower than the matrix path (one host sync per projection); it exists for
        the ablation that tests which reading of the paper's wording matters.
        """
        parent = self._parent
        assert self._gd_abs is not None
        ratio = torch.zeros(len(parent.names), dtype=torch.float64)
        for i, flat in zip(indices, flats):
            denom = self._gd_abs.get(parent.names[i])
            if denom is None:
                continue
            numer = flat.abs().to(device=parent.device, dtype=torch.float32)
            ratio[i] = float((numer / (denom + parent.eps)).mean())
        return ratio

    # -- finalisation ------------------------------------------------------------------

    def commit(self) -> None:
        if self._committed:
            raise RuntimeError("Sample already committed")
        parent = self._parent
        if self._gd_norm is None:
            raise RuntimeError("Sample has no denominator; nothing to commit")
        missing = set(parent.axes) - set(self._pending)
        if missing:
            raise RuntimeError(f"Sample is missing concept axes {sorted(missing)}")

        slot_counts = {axis: len(contributions) for axis, contributions in self._pending.items()}
        distinct = set(slot_counts.values())
        if len(distinct) != 1:
            raise RuntimeError(
                f"Concept axes contributed different numbers of augmentation slots {slot_counts}. "
                "Slot j of one axis is paired with slot j of every other, so the counts must match."
            )
        num_slots = distinct.pop()
        if num_slots != parent.augs_per_sample:
            raise RuntimeError(
                f"Sample contributed {num_slots} augmentation slot(s) per axis but the "
                f"accumulator was built for {parent.augs_per_sample}"
            )

        valid = self._gd_norm > parent.eps
        # Counted in contributions, not images: CA and the direction mean are
        # averaged over every (sample, slot) pair.
        parent._valid_count += valid.to(torch.float64) * num_slots
        parent._gd_norm_sum += self._gd_norm
        parent._num_items += 1
        parent._num_samples += num_slots

        for axis, contributions in self._pending.items():
            for gc_norm, ratio in contributions:
                masked = torch.where(valid, ratio, ratio.new_zeros(1))
                parent._ratio_sum[axis] += masked
                parent._ratio_sq_sum[axis] += masked * masked
                parent._gc_norm_sum[axis] += gc_norm

        for axis, directions in self._pending_directions.items():
            target = parent._direction_sum[axis]
            for name, contribution in directions.items():
                target[name].add_(contribution)

        if parent.compute_paired_cosine:
            self._commit_pairwise_cosines()

        self._committed = True
        self._units.clear()
        self._pending.clear()
        self._pending_directions.clear()
        self._gd_abs = None

    def _commit_pairwise_cosines(self) -> None:
        """One cosine per (pair, slot): slot j of axis a against slot j of axis b.

        Pairing by slot is what makes the number interpretable — both sides then
        come from the same position in each axis's augmentation draw, so the
        cosine measures axis disagreement rather than which prompts happened to
        line up.
        """
        parent = self._parent
        for a, b in parent.axis_pairs:
            for units_a, units_b in zip(self._units.get(a, ()), self._units.get(b, ())):
                shared = [name for name in parent.names if name in units_a and name in units_b]
                if not shared:
                    continue
                # One stack + one host sync for the whole pair rather than P syncs.
                dots = torch.stack([torch.dot(units_a[n], units_b[n]) for n in shared]).cpu()
                key = _pair_key(a, b)
                idx = torch.tensor([parent._index[n] for n in shared], dtype=torch.long)
                parent._cos_sum[key].index_add_(0, idx, dots.to(torch.float64))
                parent._cos_count[key].index_add_(
                    0, idx, torch.ones(len(shared), dtype=torch.float64)
                )

    def __enter__(self) -> _SampleScope:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None and not self._committed:
            self.commit()


class ConceptAwarenessAccumulator:
    """Streams per-sample gradients into CA and direction statistics."""

    def __init__(
        self,
        table: ProjectionTable,
        axes: Sequence[str],
        *,
        accumulate_direction: bool = True,
        compute_paired_cosine: bool = True,
        granularity: str = "matrix",
        device: str | torch.device = "cpu",
        eps: float = 1e-12,
        augs_per_sample: int = 1,
    ):
        if granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")
        if not axes:
            raise ValueError("At least one concept axis is required")
        if len(set(axes)) != len(axes):
            raise ValueError(f"Duplicate concept axes: {axes}")
        if augs_per_sample < 1:
            raise ValueError(f"augs_per_sample must be >= 1, got {augs_per_sample}")

        self.table = table
        self.names: tuple[str, ...] = tuple(table.names)
        self._index = {name: i for i, name in enumerate(self.names)}
        self.axes: tuple[str, ...] = tuple(axes)
        self.accumulate_direction = accumulate_direction
        self.compute_paired_cosine = compute_paired_cosine and len(self.axes) >= 2
        self.granularity = granularity
        self.unit_granularity = table.granularity
        self.device = torch.device(device)
        self.eps = eps
        self.augs_per_sample = augs_per_sample

        p = len(self.names)

        def zeros() -> torch.Tensor:
            return torch.zeros(p, dtype=torch.float64)

        self._ratio_sum = {axis: zeros() for axis in self.axes}
        self._ratio_sq_sum = {axis: zeros() for axis in self.axes}
        self._gc_norm_sum = {axis: zeros() for axis in self.axes}
        self._gd_norm_sum = zeros()
        self._valid_count = zeros()
        self._num_samples = 0
        self._num_items = 0

        self._direction_sum: dict[str, dict[str, torch.Tensor]] = {}
        if accumulate_direction:
            for axis in self.axes:
                self._direction_sum[axis] = {
                    spec.name: torch.zeros(spec.numel, dtype=torch.float32, device=self.device)
                    for spec in table
                }

        self.axis_pairs: tuple[tuple[str, str], ...] = tuple(
            (self.axes[i], self.axes[j])
            for i in range(len(self.axes))
            for j in range(i + 1, len(self.axes))
        )
        self._cos_sum = {_pair_key(a, b): zeros() for a, b in self.axis_pairs}
        self._cos_count = {_pair_key(a, b): zeros() for a, b in self.axis_pairs}

    # -- internals -------------------------------------------------------------------------

    def _collect(self, grads: dict[str, torch.Tensor]) -> tuple[list[int], list[torch.Tensor]]:
        """Flatten the gradients present in ``grads``, in canonical unit order.

        At head granularity every unit of one matrix reads the *same* weight
        gradient and takes its own row/column band out of it
        (:meth:`ProjectionSpec.slice_weight`). That is what makes ``CA_head`` the
        paper's ``E[||g_c||/||g_d||]`` evaluated at head level, rather than a
        post-hoc decomposition of the matrix-level ratio — the two differ,
        because a mean of ratios is not a ratio of means.
        """
        indices: list[int] = []
        flats: list[torch.Tensor] = []
        for i, spec in enumerate(self.table.specs):
            grad = grads.get(spec.name)
            if grad is None:
                continue
            indices.append(i)
            flats.append(spec.slice_weight(grad.detach()).reshape(-1))
        return indices, flats

    def _scatter(self, indices: Sequence[int], values: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(len(self.names), dtype=torch.float64)
        if indices:
            out[torch.tensor(list(indices), dtype=torch.long)] = values
        return out

    # -- introspection -----------------------------------------------------------------------

    @property
    def num_samples(self) -> int:
        """Contributions accumulated so far (``items * augs_per_sample``)."""
        return self._num_samples

    @property
    def num_items(self) -> int:
        """Distinct (x0, noise) draws accumulated so far."""
        return self._num_items

    def estimate_memory_bytes(self) -> dict[str, int]:
        """Projected peak footprint on ``self.device`` from fp32 vector buffers."""
        attn_bytes = self.table.total_params * 4
        persistent = attn_bytes * len(self.axes) if self.accumulate_direction else 0
        # Both per-axis buffers live until the sample commits: the unit vectors
        # for the pairwise cosines, and the direction contributions that commit()
        # folds into the persistent sums. Only the former scales with
        # augs_per_sample — slot j's unit vector has to survive until axis b
        # reaches slot j, whereas the directions are summed in place.
        transient = (
            attn_bytes * len(self.axes) * self.augs_per_sample if self.compute_paired_cosine else 0
        )
        if self.accumulate_direction:
            transient += attn_bytes * len(self.axes)
        if self.granularity == "elementwise":
            transient += attn_bytes
        return {
            "attention_params": self.table.total_params,
            "persistent_accumulators": persistent,
            "transient_per_sample": transient,
            "total": persistent + transient,
        }

    # -- accumulation -------------------------------------------------------------------------

    def sample(self) -> _SampleScope:
        """Open a scope for one (x0, noise) draw. Usable as a context manager."""
        return _SampleScope(self)

    def merge_(self, other: ConceptAwarenessAccumulator) -> ConceptAwarenessAccumulator:
        """Fold ``other``'s statistics into this accumulator, in place.

        Every buffer is a sum over samples, so an accumulator fed samples A and
        one fed samples B add to exactly the accumulator that would have been fed
        ``A ∪ B``. Split-half measurement relies on that: the two halves *are*
        the full run, and merging one into the other avoids allocating a third
        set of direction buffers.

        Both accumulators must be configured identically; anything else would
        silently average incomparable statistics. :meth:`finalize` does not
        mutate state, so the halves can be finalised before this is called.
        """
        if other is self:
            raise ValueError("Cannot merge an accumulator into itself")
        mismatches = [
            field
            for field, mine, theirs in (
                ("names", self.names, other.names),
                ("axes", self.axes, other.axes),
                ("granularity", self.granularity, other.granularity),
                ("unit_granularity", self.unit_granularity, other.unit_granularity),
                ("accumulate_direction", self.accumulate_direction, other.accumulate_direction),
                ("compute_paired_cosine", self.compute_paired_cosine, other.compute_paired_cosine),
                ("augs_per_sample", self.augs_per_sample, other.augs_per_sample),
                ("eps", self.eps, other.eps),
            )
            if mine != theirs
        ]
        if mismatches:
            raise ValueError(
                f"Cannot merge accumulators configured differently: {sorted(mismatches)} differ"
            )
        if self.accumulate_direction and self.device != other.device:
            raise ValueError(
                f"Cannot merge direction sums held on {self.device} and {other.device}"
            )

        for axis in self.axes:
            self._ratio_sum[axis] += other._ratio_sum[axis]
            self._ratio_sq_sum[axis] += other._ratio_sq_sum[axis]
            self._gc_norm_sum[axis] += other._gc_norm_sum[axis]
        self._gd_norm_sum += other._gd_norm_sum
        self._valid_count += other._valid_count
        self._num_samples += other._num_samples
        self._num_items += other._num_items

        for axis, buffers in self._direction_sum.items():
            for name, vec in buffers.items():
                vec.add_(other._direction_sum[axis][name])
        for key in self._cos_sum:
            self._cos_sum[key] += other._cos_sum[key]
            self._cos_count[key] += other._cos_count[key]
        return self

    def finalize(self, timestep: int) -> MeasurementResult:
        # Contributions for the statistics averaged per (sample, slot); items for
        # the ones that exist once per image, i.e. the L_Diff gradient norm.
        n = max(self._num_samples, 1)
        items = max(self._num_items, 1)
        denom = self._valid_count.clamp(min=1.0)

        per_axis: dict[str, AxisStats] = {}
        for axis in self.axes:
            mean = self._ratio_sum[axis] / denom
            second = self._ratio_sq_sum[axis] / denom
            var = (second - mean * mean).clamp(min=0.0)
            direction = None
            if self.accumulate_direction:
                # Divide by the per-projection valid count, matching ca_mean.
                direction = {
                    name: (vec / max(float(denom[self._index[name]]), 1.0)).clone()
                    for name, vec in self._direction_sum[axis].items()
                }
            per_axis[axis] = AxisStats(
                ca_mean=mean.to(torch.float32),
                ca_std=var.sqrt().to(torch.float32),
                gc_norm_mean=(self._gc_norm_sum[axis] / n).to(torch.float32),
                direction_mean=direction,
            )

        cos_of_means: dict[str, torch.Tensor] = {}
        if self.accumulate_direction:
            for a, b in self.axis_pairs:
                da, db = per_axis[a].direction_mean, per_axis[b].direction_mean
                assert da is not None and db is not None
                sims = torch.stack(
                    [
                        torch.nn.functional.cosine_similarity(
                            da[name], db[name], dim=0, eps=1e-20
                        )
                        for name in self.names
                    ]
                )
                cos_of_means[_pair_key(a, b)] = sims.to(device="cpu", dtype=torch.float32)

        mean_of_cos = {
            key: (self._cos_sum[key] / self._cos_count[key].clamp(min=1.0)).to(torch.float32)
            for key in self._cos_sum
        }

        return MeasurementResult(
            timestep=timestep,
            num_samples=self._num_samples,
            axes=self.axes,
            names=self.names,
            per_axis=per_axis,
            gd_norm_mean=(self._gd_norm_sum / items).to(torch.float32),
            valid_count=self._valid_count.to(torch.float32),
            cos_of_means=cos_of_means,
            mean_of_cos=mean_of_cos,
            granularity=self.granularity,
            unit_granularity=self.unit_granularity,
            num_items=self._num_items,
            augs_per_sample=self.augs_per_sample,
            table_records=self.table.to_records(),
        )


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def save_measurement(
    result: MeasurementResult, path: str | Path, *, save_direction_vectors: bool = False
) -> None:
    """Write a measurement to safetensors, with JSON metadata in the header.

    Direction vectors are as large as the attention parameters and are only
    needed to recompute ``cos_of_means`` (already stored), so they are excluded
    unless explicitly requested.
    """
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {
        "gd_norm_mean": result.gd_norm_mean,
        "valid_count": result.valid_count,
    }
    for axis, stats in result.per_axis.items():
        tensors[f"ca_mean/{axis}"] = stats.ca_mean
        tensors[f"ca_std/{axis}"] = stats.ca_std
        tensors[f"gc_norm_mean/{axis}"] = stats.gc_norm_mean
        if save_direction_vectors and stats.direction_mean is not None:
            for name, vec in stats.direction_mean.items():
                tensors[f"direction/{axis}/{name}"] = vec
    for key, values in result.cos_of_means.items():
        tensors[f"cos_of_means/{key}"] = values
    for key, values in result.mean_of_cos.items():
        tensors[f"mean_of_cos/{key}"] = values

    metadata = {
        "timestep": str(result.timestep),
        "num_samples": str(result.num_samples),
        "num_items": str(result.num_items),
        "augs_per_sample": str(result.augs_per_sample),
        "granularity": result.granularity,
        "unit_granularity": result.unit_granularity,
        "axes": json.dumps(list(result.axes)),
        "names": json.dumps(list(result.names)),
        "table_records": json.dumps(result.table_records or []),
    }
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in tensors.items()},
        str(path),
        metadata=metadata,
    )


def load_measurement(
    path: str | Path, *, expect_granularity: str | None = None
) -> MeasurementResult:
    """Read a measurement back.

    ``expect_granularity`` (``module``/``head``) rejects a file whose units are
    not the ones the caller is working in. Head and module measurements have the
    same shape and file name and differ only in what their entries *mean*, so
    without this check a leftover file from the other granularity would be pooled
    into a ranking and quietly corrupt it. Every pipeline entry point passes
    ``select.granularity``.
    """
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as fh:  # type: ignore[no-untyped-call]
        metadata = fh.metadata() or {}
        for key in fh.keys():
            tensors[key] = fh.get_tensor(key)

    unit_granularity = metadata.get("unit_granularity", "module")
    if expect_granularity is not None and unit_granularity != expect_granularity:
        raise ValueError(
            f"{path} was measured at unit granularity {unit_granularity!r} but "
            f"{expect_granularity!r} was expected. Head- and module-unit measurements are not "
            "comparable; re-measure, or point select.granularity at the matching set."
        )

    axes = tuple(json.loads(metadata["axes"]))
    names = tuple(json.loads(metadata["names"]))
    records = json.loads(metadata.get("table_records", "[]")) or None

    directions: dict[str, dict[str, torch.Tensor]] = {axis: {} for axis in axes}
    for key, value in tensors.items():
        if key.startswith("direction/"):
            _, axis, name = key.split("/", 2)
            directions[axis][name] = value

    per_axis = {
        axis: AxisStats(
            ca_mean=tensors[f"ca_mean/{axis}"],
            ca_std=tensors[f"ca_std/{axis}"],
            gc_norm_mean=tensors[f"gc_norm_mean/{axis}"],
            direction_mean=directions[axis] or None,
        )
        for axis in axes
    }

    return MeasurementResult(
        timestep=int(metadata["timestep"]),
        num_samples=int(metadata["num_samples"]),
        axes=axes,
        names=names,
        per_axis=per_axis,
        gd_norm_mean=tensors["gd_norm_mean"],
        valid_count=tensors["valid_count"],
        cos_of_means={
            k.split("/", 1)[1]: v for k, v in tensors.items() if k.startswith("cos_of_means/")
        },
        mean_of_cos={
            k.split("/", 1)[1]: v for k, v in tensors.items() if k.startswith("mean_of_cos/")
        },
        granularity=metadata.get("granularity", "matrix"),
        unit_granularity=unit_granularity,
        num_items=int(metadata.get("num_items", metadata["num_samples"])),
        augs_per_sample=int(metadata.get("augs_per_sample", 1)),
        table_records=records,
    )
