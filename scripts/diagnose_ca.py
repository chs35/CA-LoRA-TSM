"""Timestep diagnostics for the per-interval t decision (design §4, runbook step 2).

    python scripts/diagnose_ca.py --config configs/sdxl_cityscapes.yaml
    python scripts/diagnose_ca.py --config configs/sdxl_cityscapes.yaml \
        --measurements runs/sdxl_cityscapes/measurements --timesteps 1 41 61 81

Reads every full measurement ``t*.safetensors`` and writes ``<run_dir>/diagnosis/``.
The CA-LoRA visual heuristic is reproduced as *figures only* — no score decides
anything here; the person reading them does:

* ``ca_distribution.png``    — per t, histogram of the sum-normalised style-CA and
  viewpoint-CA over all units (log scale).
* ``ca_heatmap_<axis>_layers[_shared].png`` — attention layer × {q,k,v,out}
  normalised CA, one panel per t (the paper's Fig. 11 layout), per-panel or
  shared colour scale.
* ``ca_heatmap_<axis>_blocks.png`` — the same summed to block groups, for a
  coarse first look.
* ``jaccard_matrix.png``     — t × t Jaccard of the top-``select.fraction`` sets on
  ``select.axis`` (design §4: block structure → interval boundaries).
* ``diagnosis.json``         — the numbers behind the figures plus, per t, the
  split-half Jaccard (sampling stability, design §4.3) and cross-axis Jaccard.
* ``DECISION_TEMPLATE.md``   — every interval of every scheme with its measured
  candidate t and an empty ``decision:`` line.

No training, no adapter: only the measurement files are read.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from calora import cli  # noqa: E402
from calora.ca.select import Selection, normalize_scores, select_topk  # noqa: E402
from calora.models.projections import parse_unit_name  # noqa: E402
from calora.ca.stats import MeasurementResult, load_measurement  # noqa: E402
from calora.utils.io import dump_json  # noqa: E402
from calora.utils.logging import get_logger  # noqa: E402
from catsm import cli as catsm_cli  # noqa: E402
from catsm.intervals import build_scheme  # noqa: E402
from catsm.select import load_measurements  # noqa: E402

logger = get_logger("diagnose_ca")

KINDS = ("q", "k", "v", "out")
REFERENCE_T = 81  # the paper's choice; every t's selection is also compared against it
#: Scheduler length of every backbone this project targets (SDXL, SD1.5, toy).
#: Hard-coded so the diagnosis never has to load a model.
NUM_TRAIN_TIMESTEPS = 1000


def select(result: MeasurementResult, table, axis: str, fraction: float, tag: str) -> Selection:
    return select_topk(
        result.ca(axis).to(torch.float64), table, fraction=fraction, budget_basis="unit_count",
        strategy=f"diag-{tag}", meta={"axis": axis},
    )


def group_kind_table(dist: torch.Tensor, groups: dict, groups_kind: dict) -> dict[str, dict[str, float]]:
    """Summed normalised CA per (group, kind); rows in model order."""
    kind_of = {i: k for k, idx in groups_kind.items() for i in idx}
    out: dict[str, dict[str, float]] = {}
    for group, indices in groups.items():
        row = {k: 0.0 for k in KINDS}
        for i in indices:
            row[kind_of[i]] += float(dist[i])
        out[group] = row
    return out


def layer_groups(table) -> dict[str, list[int]]:
    """Unit indices per attention layer (``...attn1`` / ``...attn2``), in model order."""
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(table.names):
        module, _ = parse_unit_name(name)
        layer = module.rsplit(".to_", 1)[0]
        groups.setdefault(layer, []).append(i)
    return groups


def short_layer(layer: str) -> str:
    """``down_blocks.1.attentions.0.transformer_blocks.3.attn2`` -> ``d1.a0.t3.attn2``."""
    return (layer.replace("down_blocks.", "d").replace("up_blocks.", "u").replace("mid_block", "m")
            .replace(".attentions.", ".a").replace(".transformer_blocks.", ".t"))


def half_paths(directory: Path, t: int) -> tuple[Path, Path]:
    return directory / f"t{t:04d}_half0.safetensors", directory / f"t{t:04d}_half1.safetensors"


def diagnose(results: dict[int, MeasurementResult], directory: Path, cfg) -> dict:
    table = next(iter(results.values())).table()
    axes = list(cfg.data.concept_axes)
    if len(axes) < 2:
        raise ValueError(f"The heuristic compares two concept axes; data.concept_axes={axes}")
    pairs = list(itertools.combinations(axes, 2))
    main_axis, fraction = cfg.select.axis, cfg.select.fraction
    groups_block, groups_kind = table.group_indices("block_id"), table.group_indices("kind")
    groups_layer = layer_groups(table)
    timesteps = sorted(results)

    per_t: dict[str, dict] = {}
    selections: dict[int, Selection] = {}
    for t in timesteps:
        result = results[t]
        dists = {a: normalize_scores(result.ca(a), "sum") for a in axes}
        entry: dict = {"timestep": t, "num_items": result.num_items}
        entry["ca_normalised"] = {a: dists[a].tolist() for a in axes}
        chosen = {a: select(result, table, a, fraction, f"t{t}") for a in axes}
        selections[t] = chosen[main_axis]
        entry["num_units"] = {a: len(s) for a, s in chosen.items()}
        entry["cross_axis_jaccard"] = {f"{a}|{b}": chosen[a].jaccard(chosen[b]) for a, b in pairs}
        h0, h1 = half_paths(directory, t)
        if h0.exists() and h1.exists():
            halves = [load_measurement(p, expect_granularity=cfg.select.granularity) for p in (h0, h1)]
            entry["split_half_jaccard"] = select(halves[0], table, main_axis, fraction, f"t{t}h0").jaccard(
                select(halves[1], table, main_axis, fraction, f"t{t}h1"))
        else:
            entry["split_half_jaccard"] = None
        entry["block_kind"] = {a: group_kind_table(dists[a], groups_block, groups_kind) for a in axes}
        entry["layer_kind"] = {a: group_kind_table(dists[a], groups_layer, groups_kind) for a in axes}
        per_t[str(t)] = entry

    matrix = [[selections[a].jaccard(selections[b]) for b in timesteps] for a in timesteps]
    ref = selections.get(REFERENCE_T)
    for t in timesteps:
        per_t[str(t)]["jaccard_vs_81"] = selections[t].jaccard(ref) if ref is not None else None

    return {
        "axes": axes, "select_axis": main_axis, "fraction": fraction,
        "granularity": cfg.select.granularity, "num_units_total": len(table),
        "timesteps": timesteps,
        "per_timestep": per_t,
        "jaccard_matrix": {"axis": main_axis, "timesteps": timesteps, "values": matrix},
        "blocks": list(groups_block), "layers": list(groups_layer), "kinds": list(KINDS),
    }


def decision_template(diag: dict, cfg) -> str:
    lines = ["# Interval t decision", "",
             "Look at `ca_distribution.png`, `ca_heatmap_<axis>_layers.png` and `jaccard_matrix.png`; "
             "no score is computed. J(half) = split-half Jaccard of the top set (sampling stability), "
             "J(81) = Jaccard with the t=81 set.",
             f"Measured t: {diag['timesteps']}. Fill every `decision:` line, then set "
             "`interval_select.representative` in the config.", ""]
    for n in cfg.intervals.schemes:
        scheme = build_scheme(f"n{n}", NUM_TRAIN_TIMESTEPS, cfg.intervals)
        lines.append(f"## {scheme.name}")
        for i, iv in enumerate(scheme.intervals):
            inside = [t for t in diag["timesteps"] if t in iv]
            lines.append(f"- **{scheme.expert_name(i)}** [{iv.lo}, {iv.hi}]")
            for t in inside:
                e = diag["per_timestep"][str(t)]
                j81 = e["jaccard_vs_81"]; jh = e["split_half_jaccard"]
                lines.append(f"  - t={t}: J(half) {'-' if jh is None else f'{jh:.3f}'}, "
                             f"J(81) {'-' if j81 is None else f'{j81:.3f}'}")
            if not inside:
                lines.append("  - (no measured t inside; measure one before deciding)")
            lines.append("  - decision: ")
        lines.append("")
    return "\n".join(lines)


def plot(diag: dict, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        logger.warning("matplotlib unavailable (%s); skipping figures", exc)
        return
    ts = diag["timesteps"]
    per = [diag["per_timestep"][str(t)] for t in ts]
    axes = diag["axes"]

    # Normalised-CA histograms, style vs viewpoint, one panel per t (log10 of the weight).
    import numpy as np
    cols = min(len(ts), 5); rows = -(-len(ts) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False, sharex=True, sharey=True)
    logs = {a: [np.log10(np.clip(np.asarray(e["ca_normalised"][a]), 1e-12, None)) for e in per] for a in axes}
    lo = min(v.min() for a in axes for v in logs[a]); hi = max(v.max() for a in axes for v in logs[a])
    bins = np.linspace(lo, hi, 60)
    for k, (a, t) in enumerate(zip(axs.flat, ts)):
        for axis in axes:
            a.hist(logs[axis][k], bins=bins, histtype="step", linewidth=1.2, label=axis)
        a.set_title(f"t={t}", fontsize=9); a.grid(alpha=0.3)
    for a in list(axs.flat)[len(ts):]:
        a.axis("off")
    axs[0][0].legend(fontsize=8)
    fig.supxlabel("log10 normalised CA (per unit)"); fig.supylabel("units")
    fig.suptitle("CA distribution per timestep — " + " vs ".join(axes))
    fig.tight_layout(); fig.savefig(out / "ca_distribution.png", dpi=120); plt.close(fig)

    m = diag["jaccard_matrix"]["values"]
    fig, ax = plt.subplots(figsize=(0.55 * len(ts) + 2, 0.55 * len(ts) + 1.5))
    im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(ts))); ax.set_xticklabels(ts, rotation=90); ax.set_yticks(range(len(ts))); ax.set_yticklabels(ts)
    for i in range(len(ts)):
        for j in range(len(ts)):
            ax.text(j, i, f"{m[i][j]:.2f}", ha="center", va="center", fontsize=6, color="w" if m[i][j] < 0.6 else "k")
    ax.set_title(f"top-{diag['fraction']:.0%} Jaccard ({diag['select_axis']})"); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out / "jaccard_matrix.png", dpi=120); plt.close(fig)

    kinds = diag["kinds"]

    def heatmaps(key: str, rows: list[str], labels: list[str], suffix: str, shared_too: bool) -> None:
        for axis in axes:
            grids = [[[e[key][axis][r][k] for k in kinds] for r in rows] for e in per]
            vmax_shared = max(v for g in grids for row in g for v in row) or 1.0
            for shared in ((False, True) if shared_too else (False,)):
                fig, axs = plt.subplots(1, len(ts), figsize=(1.5 * len(ts) + 1.5, 0.12 * len(rows) + 1.8), squeeze=False)
                for a, t, g in zip(axs[0], ts, grids):
                    vmax = vmax_shared if shared else (max(max(r) for r in g) or 1.0)
                    a.imshow(g, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
                    a.set_title(f"t={t}", fontsize=8); a.set_xticks(range(len(kinds))); a.set_xticklabels(kinds, fontsize=7)
                    a.set_yticks(range(len(rows))); a.set_yticklabels(labels if a is axs[0][0] else [], fontsize=5)
                fig.suptitle(f"{axis} CA (sum-normalised) per {suffix} × kind — {'shared' if shared else 'per-panel'} scale")
                fig.tight_layout(rect=(0, 0, 1, 0.985))  # keep the suptitle clear of the panel titles
                fig.savefig(out / f"ca_heatmap_{axis}_{suffix}{'_shared' if shared else ''}.png", dpi=120); plt.close(fig)

    heatmaps("layer_kind", diag["layers"], [short_layer(l) for l in diag["layers"]], "layers", True)
    heatmaps("block_kind", diag["blocks"], diag["blocks"], "blocks", False)


def main() -> None:
    parser = cli.base_parser(__doc__ or "")
    parser.add_argument("--measurements", default=None, help="Directory of t*.safetensors")
    parser.add_argument("--timesteps", nargs="*", type=int, default=None, help="Subset of measured t")
    parser.add_argument("--out", default=None, help="Output directory (default <run_dir>/diagnosis)")
    args = parser.parse_args()
    cfg = catsm_cli.resolve(args)
    directory = Path(args.measurements) if args.measurements else cfg.run_dir / "measurements"
    out = Path(args.out) if args.out else cfg.run_dir / "diagnosis"
    out.mkdir(parents=True, exist_ok=True)

    results = load_measurements(directory, granularity=cfg.select.granularity)
    if args.timesteps:
        missing = sorted(set(args.timesteps) - set(results))
        if missing:
            raise FileNotFoundError(f"No measurement for t={missing} under {directory}")
        results = {t: results[t] for t in args.timesteps}
    logger.info("Diagnosing %d timestep(s) %s, %d units", len(results), sorted(results), len(next(iter(results.values())).names))

    diag = diagnose(results, directory, cfg)
    dump_json(diag, out / "diagnosis.json")
    (out / "DECISION_TEMPLATE.md").write_text(decision_template(diag, cfg), encoding="utf-8")
    plot(diag, out)
    logger.info("Wrote %s: diagnosis.json, DECISION_TEMPLATE.md, ca_distribution.png, "
                "ca_heatmap_<axis>_{layers,layers_shared,blocks}.png, jaccard_matrix.png", out)


if __name__ == "__main__":
    main()
