"""Per-interval CA head selection (design §3.2, modes (A)/(B), control C1).

The CA measurements themselves are calora's (``scripts/measure_ca.py`` →
``<run_dir>/measurements/t{t:04d}.safetensors``); this module only decides which
measured timesteps feed each interval's top-k and stamps the result so the
expert trained from it can be traced back (``selection_sha``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch

from calora.ca.select import Selection, aggregate_across_timesteps, axis_scores, select_topk
from calora.ca.stats import MeasurementResult, load_measurement
from calora.cli import is_half_measurement
from calora.config import SelectConfig
from calora.models.projections import ProjectionTable
from calora.utils.io import dump_json, load_json
from catsm.config import SELECT_MODES
from catsm.intervals import Interval, IntervalScheme, scheme_record


def load_measurements(
    directory: str | Path, *, granularity: str = "head"
) -> dict[int, MeasurementResult]:
    """Every full (non-half) measurement in ``directory``, keyed by timestep."""
    directory = Path(directory)
    paths = sorted(p for p in directory.glob("t*.safetensors") if not is_half_measurement(p))
    if not paths:
        raise FileNotFoundError(f"No measurements (t*.safetensors) under {directory}")
    results: dict[int, MeasurementResult] = {}
    for path in paths:
        result = load_measurement(path, expect_granularity=granularity)
        if result.timestep in results:
            raise ValueError(f"Two measurements for t={result.timestep} under {directory}")
        results[result.timestep] = result
    names = {r.names for r in results.values()}
    if len(names) != 1:
        raise ValueError(
            f"Measurements under {directory} disagree on their unit list; they were made "
            "against different models or granularities"
        )
    return results


def selection_sha(names: Iterable[str]) -> str:
    """Order-independent fingerprint of a selected unit set."""
    payload = "\n".join(sorted(names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def timesteps_in(results: Mapping[int, MeasurementResult], interval: Interval) -> list[int]:
    return sorted(t for t in results if t in interval)


def select_for_interval(
    results: Mapping[int, MeasurementResult],
    table: ProjectionTable,
    interval: Interval,
    *,
    mode: str,
    cfg: SelectConfig,
    representative_t: int | None = None,
    fixed_t: int | None = None,
) -> Selection:
    """Top-k head selection for one interval.

    ``representative`` (A): the CA of one measured ``representative_t`` inside the
    interval. ``aggregate`` (B): calora's multi-t aggregation over every measured
    t inside the interval. ``fixed`` (C1): the CA of ``fixed_t`` regardless of the
    interval — the control that keeps the selection at t=81 and varies only the
    training range.
    """
    if mode not in SELECT_MODES:
        raise ValueError(f"mode must be one of {SELECT_MODES}, got {mode!r}")
    if mode == "representative":
        if representative_t is None:
            raise ValueError("mode='representative' needs representative_t")
        if representative_t not in interval:
            raise ValueError(
                f"Representative t={representative_t} lies outside the interval "
                f"[{interval.lo}, {interval.hi}]"
            )
        if representative_t not in results:
            raise KeyError(f"No measurement for representative t={representative_t}")
        timesteps = [int(representative_t)]
        scores = axis_scores(results[representative_t], cfg.axis)
    elif mode == "aggregate":
        timesteps = timesteps_in(results, interval)
        if not timesteps:
            raise ValueError(
                f"No measured timestep inside [{interval.lo}, {interval.hi}]; measured: "
                f"{sorted(results)}"
            )
        scores = aggregate_across_timesteps(
            results,
            axis=cfg.axis,
            method=cfg.aggregation if len(timesteps) > 1 else "single",
            normalization=cfg.normalization,
            timesteps=timesteps,
        )
    else:  # fixed
        if fixed_t is None:
            raise ValueError("mode='fixed' needs fixed_t")
        if fixed_t not in results:
            raise KeyError(f"No measurement for fixed t={fixed_t}")
        timesteps = [int(fixed_t)]
        scores = axis_scores(results[fixed_t], cfg.axis)

    selection = select_topk(
        scores,
        table,
        fraction=cfg.fraction,
        budget_basis=cfg.budget_basis,
        strategy=f"interval:{mode}",
        meta={
            "axis": cfg.axis,
            "timesteps": timesteps,
            "interval": interval.as_list(),
            "mode": mode,
            "normalization": cfg.normalization,
            "aggregation": cfg.aggregation,
        },
    )
    selection.meta["selection_sha"] = selection_sha(selection.names)
    return selection


def select_scheme(
    results: Mapping[int, MeasurementResult],
    table: ProjectionTable,
    scheme: IntervalScheme,
    *,
    mode: str,
    cfg: SelectConfig,
    representative: Sequence[int] | None = None,
    fixed_t: int | None = None,
) -> dict[str, Selection]:
    """Expert name -> its selection, for every interval of ``scheme``."""
    if mode == "representative":
        if representative is None or len(representative) != scheme.n:
            raise ValueError(
                f"mode='representative' needs one timestep per interval of {scheme.name} "
                f"({scheme.n}), got {representative}"
            )
    selections: dict[str, Selection] = {}
    for i, interval in enumerate(scheme.intervals):
        selection = select_for_interval(
            results,
            table,
            interval,
            mode=mode,
            cfg=cfg,
            representative_t=representative[i] if representative is not None else None,
            fixed_t=fixed_t,
        )
        selection.meta["expert"] = scheme.expert_name(i)
        selection.meta["scheme"] = scheme.name
        selections[scheme.expert_name(i)] = selection
    return selections


def write_scheme_selections(
    selections: Mapping[str, Selection],
    scheme: IntervalScheme,
    out_dir: str | Path,
    *,
    alphas_cumprod: torch.Tensor | None = None,
) -> Path:
    """``<out_dir>/<expert>.json`` per expert plus ``<out_dir>/intervals.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, selection in selections.items():
        dump_json(selection.to_dict(), out_dir / f"{name}.json")
    record = scheme_record(scheme, alphas_cumprod)
    for entry in record["intervals"]:
        selection = selections[entry["expert"]]
        entry["selection_sha"] = selection.meta["selection_sha"]
        entry["timesteps"] = selection.meta["timesteps"]
        entry["mode"] = selection.meta["mode"]
        entry["num_units"] = len(selection)
    dump_json(record, out_dir / "intervals.json")
    return out_dir


def load_selection(path: str | Path, table: ProjectionTable | None = None) -> Selection:
    selection = Selection.from_dict(load_json(path), table)
    if selection.granularity != "head":
        raise ValueError(f"{path}: interval experts need a head-granularity selection")
    selection.meta.setdefault("selection_sha", selection_sha(selection.names))
    return selection
