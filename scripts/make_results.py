"""README results: evidence copies, summary table, Jaccard and trade-off figures.

    python scripts/make_results.py --refresh   # copy evidence from runs/ (+ Improving-CA-LoRA) into results/
    python scripts/make_results.py             # rebuild everything below from results/ only

``--refresh`` copies, verbatim, every reportable ``<arm>/eval.json`` to
``results/evals/<arm>.json`` and the Improving-CA-LoRA 10K-step reference evals to
``results/reference_10k/<arm>.json``, and extracts the Jaccard part of
``runs/sdxl_cityscapes/diagnosis/diagnosis.json`` (written by ``scripts/diagnose_ca.py``;
the full file is 15 MB of per-layer CA) to ``results/diagnosis_jaccard.json``.

The build step then writes, from ``results/`` alone:

* ``results/summary.csv`` and ``results/summary.md`` — one row per configuration;
* ``figures/jaccard_matrix.png`` — t × t Jaccard of the top-2 % style-CA head sets;
* ``figures/tradeoff_scatter.png`` — CMMD vs weather CLIP-Score, same axes as
  Improving-CA-LoRA's ``figures/tradeoff_scatter.png``;
* ``results/MANIFEST.md`` — sha256 of every file in ``results/`` and ``figures/``.

Weather CLIP SEM is the per-image sampling error of the mean of the four condition means,
``sqrt(Σ_c s_c² / n_c) / 4``, from the stds stored in eval.json. It is *not* a seed-to-seed
spread: every configuration was generated with one seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
RUNS = ROOT / "runs"
UPSTREAM = Path("/workspace/Optimizing-CA-LoRA/Improving-CA-LoRA/results/sdxl_cityscapes")

HEAD_PCT = 2.0          # 208 of 10,400 attention-head units per expert
FOSTER_STEPS = 4000     # per expert
ROUTER_STEPS = 1000     # batch 8

# arm -> (label, source eval.json, description)
ARMS = {
    "no1": ("No.1", "sdxl_cityscapes/no1", "n1 (CA-LoRA, t=81), 4K"),
    "no2": ("No.2", "sdxl_cityscapes/no2", "n4"),
    "no3": ("No.3", "sdxl_cityscapes/no3", "n1+n4, core n4"),
    "no4": ("No.4", "sdxl_cityscapes/no4", "n1+n3+n4, core n4"),
    "no5": ("No.5", "sdxl_cityscapes/no5", "n1+n4+n5, core n4"),
    "no6": ("No.6", "sdxl_cityscapes/no6", "n3+n4+n5, core n4"),
    "no7": ("No.7", "sdxl_cityscapes/no7", "n1+n4+n5, core n5"),
    "no8": ("No.8", "sdxl_cityscapes/no8", "n1+n3+n4+n5, core n4"),
    "no9": ("No.9", "sdxl_cityscapes/no9", "n1+n3+n4+n5, core n5"),
    "c1": ("C1", "sdxl_c1/c1", "n4 intervals, every selection at t=81"),
    "c2n2": ("C2 (No.2)", "sdxl_cityscapes/c2n2", "No.2 with n4_i3 [500,999] off"),
    "c2": ("C2 (No.3)", "sdxl_cityscapes/c2", "No.3 with n4_i3 [500,999] off, router retrained"),
}
REFERENCE = {
    "control_base": ("base SDXL", "no LoRA (0%)"),
    "A_t81_style": ("10K t=81", "single LoRA, style CA t=81, 10K"),
    "C_t41_style": ("10K t=41", "single LoRA, style CA t=41, 10K"),
    "D_t1_style": ("10K t=1", "single LoRA, style CA t=1, 10K"),
    "B_top3_style": ("10K multi-t", "single LoRA, style CA top-3 aggregate, 10K"),
    "E_random": ("10K random 2%", "single LoRA, random 2% heads, 10K"),
    "E_all": ("10K full attention", "LoRA on all attention, 10K"),
}
CONDITIONS = ("foggy", "night-time", "rainy", "snowy")
# Halving boundaries actually used (design §3.3 / DECISION.md): interval starts.
BOUNDARIES = {"n5": 62, "n4/n5": 125, "n3/n4/n5": 250, "all m>1": 500}


def refresh() -> None:
    (RES / "evals").mkdir(parents=True, exist_ok=True)
    (RES / "reference_10k").mkdir(parents=True, exist_ok=True)
    for arm, (_, src, _) in ARMS.items():
        shutil.copy2(RUNS / src / "eval.json", RES / "evals" / f"{arm}.json")
    for arm in REFERENCE:
        shutil.copy2(UPSTREAM / arm / "eval.json", RES / "reference_10k" / f"{arm}.json")
    diag_path = RUNS / "sdxl_cityscapes/diagnosis/diagnosis.json"
    diag = json.loads(diag_path.read_text())
    keep = {
        "source": "runs/sdxl_cityscapes/diagnosis/diagnosis.json (scripts/diagnose_ca.py)",
        "source_sha256": sha256(diag_path),
        "axis": diag["select_axis"], "fraction": diag["fraction"], "granularity": diag["granularity"],
        "num_units_total": diag["num_units_total"],
        "jaccard_matrix": diag["jaccard_matrix"],
        "per_timestep": {t: {k: v[k] for k in ("num_items", "split_half_jaccard", "jaccard_vs_81", "cross_axis_jaccard")}
                         for t, v in diag["per_timestep"].items()},
    }
    (RES / "diagnosis_jaccard.json").write_text(json.dumps(keep, indent=1))
    print("refreshed results/evals, results/reference_10k, results/diagnosis_jaccard.json")


def row_from_eval(arm: str, d: dict, *, label: str, desc: str, group: str) -> dict:
    pc = d["clip_score"]["per_condition"]
    sem = math.sqrt(sum(pc[c]["clip_score_std"] ** 2 / pc[c]["num_pairs"] for c in CONDITIONS)) / len(CONDITIONS)
    routing = d.get("routing") or {}
    experts = routing.get("experts", [])
    disabled = routing.get("disabled", [])
    sets = routing.get("sets", [])
    if group == "reference":
        pct = {"control_base": 0.0, "E_all": 100.0}.get(arm, HEAD_PCT)
        n_exp, active, nominal = (0 if arm == "control_base" else 1), pct, pct
        foster, router = (0, 0) if arm == "control_base" else (10000, 0)
    else:
        n_exp = len(experts)
        nominal = HEAD_PCT * n_exp
        active = HEAD_PCT * len(sets)            # one active expert per set at every t
        foster = FOSTER_STEPS * n_exp
        router = ROUTER_STEPS if routing.get("router") else 0
    if disabled:
        active_note = f"{active - HEAD_PCT * len(disabled):g}% on [500,999]"
    else:
        active_note = ""
    return {
        "arm": arm, "label": label, "group": group, "description": desc,
        "sets": "+".join(sets), "core": routing.get("core") or "",
        "routing": (("router" if routing.get("router") else "hard") if len(experts) > 1 else "single") if routing else "",
        "disabled": "+".join(disabled),
        "n_experts": n_exp, "nominal_head_pct": nominal, "active_head_pct_per_t": active,
        "active_note": active_note,
        "fostering_steps_total": foster, "router_steps": router,
        "gen_seed": d["protocol"]["seed"], "num_images": d["num_images"],
        "cmmd": d["cmmd"]["cmmd"],
        "clip_in_domain": pc["in_domain"]["clip_score"],
        "clip_weather": d["clip_score"]["weather_mean"], "clip_weather_sem": sem,
        **{f"clip_{c}": pc[c]["clip_score"] for c in CONDITIONS},
    }


def load_rows() -> list[dict]:
    rows = []
    for arm, (label, _, desc) in ARMS.items():
        d = json.loads((RES / "evals" / f"{arm}.json").read_text())
        assert d.get("reportable", True) and d["num_images"] == 900 and d["protocol"]["seed"] == 0, arm
        rows.append(row_from_eval(arm, d, label=label, desc=desc, group="control" if arm.startswith("c") else "main"))
    for arm, (label, desc) in REFERENCE.items():
        d = json.loads((RES / "reference_10k" / f"{arm}.json").read_text())
        rows.append(row_from_eval(arm, d, label=label, desc=desc, group="reference"))
    return rows


def write_tables(rows: list[dict]) -> None:
    with open(RES / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})

    def pct(r):
        if r["active_head_pct_per_t"] is None:
            return "—"
        s = f"{r['active_head_pct_per_t']:g}% / {r['nominal_head_pct']:g}%"
        return s + (f" ({r['active_note']})" if r["active_note"] else "")

    head = "| Config | Experts | Routing | Active / nominal heads | CMMD ↓ | In-domain CLIP | Weather CLIP ↑ | foggy | night | rainy | snowy |"
    sep = "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    best_cmmd = min(r["cmmd"] for r in rows if r["group"] != "reference")
    best_w = max(r["clip_weather"] for r in rows if r["group"] == "main")

    def line(r):
        cm = f"{r['cmmd']:.3f}"
        wc = f"{r['clip_weather']:.2f} ± {r['clip_weather_sem']:.2f}"
        if r["group"] != "reference" and r["cmmd"] == best_cmmd:
            cm = f"**{cm}**"
        if r["group"] == "main" and r["clip_weather"] == best_w:
            wc = f"**{wc}**"
        routing = r["routing"] or ("—" if r["arm"] == "control_base" else "single LoRA")
        return (f"| {r['label']} | {r['description']} | {routing} | {pct(r)} | {cm} | {r['clip_in_domain']:.2f} | {wc} | "
                + " | ".join(f"{r[f'clip_{c}']:.2f}" for c in CONDITIONS) + " |")

    main = [r for r in rows if r["group"] != "reference"]
    ref = [r for r in rows if r["group"] == "reference"]
    text = "\n".join([head, sep, *map(line, main), "", "Reference only — Improving-CA-LoRA, single LoRA trained 10K steps; "
                      "not a direct comparison (design §5.1):", "", head, sep, *map(line, ref)])
    (RES / "summary.md").write_text(text + "\n")
    print(text)


def plot_jaccard() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = json.loads((RES / "diagnosis_jaccard.json").read_text())
    ts = d["jaccard_matrix"]["timesteps"]
    m = np.array(d["jaccard_matrix"]["values"])
    sh = [d["per_timestep"][str(t)]["split_half_jaccard"] for t in ts]
    n = len(ts)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6.4), dpi=150, gridspec_kw={"width_ratios": [1.25, 1]})
    im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{m[i, j]:.2f}"[1:] if m[i, j] < 1 else "1", ha="center", va="center", fontsize=5.5,
                    color="black" if m[i, j] > 0.55 else "white")
    ax.set_xticks(range(n), ts, rotation=90, fontsize=7)
    ax.set_yticks(range(n), ts, fontsize=7)
    ax.set_xlabel("measured t"); ax.set_ylabel("measured t")
    # Interval starts actually used, drawn between the measured t on either side.
    for start in BOUNDARIES.values():
        k = next(i for i, t in enumerate(ts) if t >= start) - 0.5
        ax.axhline(k, color="white", lw=1.2, ls="--"); ax.axvline(k, color="white", lw=1.2, ls="--")
    ax.set_title("Jaccard of top-2% style-CA head sets (dashed: interval\nstarts used, 62 | 125 | 250 | 500)",
                 loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    adj = [m[i, i + 1] for i in range(n - 1)]
    floor = [min(sh[i], sh[i + 1]) for i in range(n - 1)]
    x = np.arange(n - 1)
    navy, mid = "#08306b", "#6baed6"
    bx.plot(x, adj, color=navy, lw=2, marker="o", ms=5, label="J(adjacent measured t)")
    bx.plot(x, floor, color=mid, lw=2, ls="--", marker="s", ms=4, label="split-half J (two halves of the 256 images, min of the pair)")
    bx.set_xticks(x, [f"{ts[i]}|{ts[i + 1]}" for i in range(n - 1)], rotation=90, fontsize=7)
    bx.set_ylim(0, 1); bx.set_ylabel("Jaccard")
    for start in BOUNDARIES.values():
        k = next(i for i, t in enumerate(ts) if t >= start) - 1
        bx.axvline(k, color="#999999", lw=0.8, ls=":")
    bx.set_title("Adjacent-t similarity vs re-sampling the same t\n(dotted: pairs straddling an interval start)",
                 loc="left", fontsize=10)
    bx.legend(frameon=False, fontsize=8, loc="lower left")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    fig.text(0.01, 0.01, f"head granularity, {d['num_units_total']:,} units, top {d['fraction']:.0%} = 208; "
             "256 images × 4 prompt augs per t, one seed (0), same images and noise at every t",
             fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "jaccard_matrix.png")
    print(f"wrote {FIG / 'jaccard_matrix.png'}")


def plot_tradeoff(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference categorical palette (dataviz slots 1, 2, 7, 4) + neutral grey; marker shape is a second channel.
    groups = {
        "single": ("#2a78d6", "o", "single CA-LoRA, 4K (No.1)"),
        "hard": ("#eb6834", "s", "interval experts, hard routing (No.2, C1)"),
        "router": ("#6250d6", "^", "interval experts + router (No.3–9)"),
        "c2": ("#eda100", "D", "C2: [500,999] expert disabled"),
        "ref": ("#8c8b85", "o", "reference: Improving-CA-LoRA, 10K (not a direct comparison)"),
    }

    def group(r):
        if r["group"] == "reference":
            return "ref"
        if r["arm"] == "no1":
            return "single"
        if r["arm"].startswith("c2"):
            return "c2"
        return "router" if r["routing"] == "router" else "hard"

    offsets = {"No.1": (8, -10), "No.2": (8, 3), "C1": (8, -4), "No.6": (8, 3), "C2 (No.2)": (-62, 4),
               "C2 (No.3)": (8, 4), "No.3": (-34, 7), "No.8": (8, -9), "10K t=81": (8, -4), "10K t=41": (8, 2),
               "10K t=1": (8, -4), "10K multi-t": (8, -9), "10K random 2%": (8, 2), "10K full attention": (-12, -13)}
    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=160)
    base = None
    for r in rows:
        if r["arm"] == "control_base":
            base = r
            continue
        g = group(r)
        c, mk, _ = groups[g]
        face = "white" if g == "ref" else c
        ax.scatter(r["cmmd"], r["clip_weather"], s=70, marker=mk, facecolors=face, edgecolors=c if g == "ref" else "white",
                   linewidths=1.6, zorder=3)
        if r["label"] in offsets or g != "router":
            ax.annotate(r["label"], (r["cmmd"], r["clip_weather"]), xytext=offsets.get(r["label"], (8, 2)),
                        textcoords="offset points", fontsize=8.5, color="#3d3d3a")
    # the five overlapping n1-context routers get one shared label
    rt = [r for r in rows if r["arm"] in ("no4", "no5", "no7", "no9")]
    ax.annotate("No.4/5/7/9", (min(r["cmmd"] for r in rt), min(r["clip_weather"] for r in rt)), xytext=(-6, -18),
                textcoords="offset points", fontsize=8.5, color="#3d3d3a")
    for g, (c, mk, lab) in groups.items():
        ax.scatter([], [], s=60, marker=mk, facecolors="white" if g == "ref" else c, edgecolors=c, label=lab)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.annotate(f"0% control off scale: CMMD {base['cmmd']:.2f}, weather {base['clip_weather']:.1f} →",
                (0.99, 0.02), xycoords="axes fraction", ha="right", fontsize=8, color="#6b6a63")
    ax.set_xlabel("CMMD  (lower = better domain fit)", fontsize=11)
    ax.set_ylabel("Weather CLIP-Score  (higher = better prompt control)", fontsize=11)
    ax.set_title("Domain fit vs. weather controllability (one point per configuration)", fontsize=11.5)
    ax.grid(True, color="#e6e5df", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.005, "one generation seed (0), 900 images per point; weather = mean of foggy/night-time/rainy/snowy "
             "(100 images each); ±SEM ≈ 0.12", fontsize=7, color="#6b6a63")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "tradeoff_scatter.png")
    print(f"wrote {FIG / 'tradeoff_scatter.png'}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest() -> None:
    files = sorted(p for d in (RES, FIG) for p in d.rglob("*") if p.is_file() and p.name != "MANIFEST.md")
    lines = [
        "# Results manifest",
        "",
        "sha256 of every file in `results/` and `figures/` (this file excluded). Regenerate with",
        "`python scripts/make_results.py` (and `scripts/router_gate_vs_t.py` for `router_gate/`).",
        "",
        "Provenance: `evals/<arm>.json` are verbatim copies of `runs/<run>/<arm>/eval.json`;",
        "`reference_10k/<arm>.json` are verbatim copies of Improving-CA-LoRA",
        "`results/sdxl_cityscapes/<arm>/eval.json` (identical to its own MANIFEST);",
        "`diagnosis_jaccard.json` is extracted from `runs/sdxl_cityscapes/diagnosis/diagnosis.json`",
        "(source sha256 inside the file). Checkpoints, measurements and generated images stay in the",
        "gitignored `runs/` and are not part of this repository.",
        "",
        "| File | Bytes | sha256 |",
        "|---|---:|---|",
    ]
    for p in files:
        lines.append(f"| `{p.relative_to(ROOT)}` | {p.stat().st_size:,} | `{sha256(p)}` |")
    lines += ["", "Source eval.json in `runs/` (not committed):", "", "| Source | sha256 |", "|---|---|"]
    for arm, (_, src, _) in ARMS.items():
        p = RUNS / src / "eval.json"
        if p.exists():
            lines.append(f"| `runs/{src}/eval.json` | `{sha256(p)}` |")
    (RES / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {RES / 'MANIFEST.md'} ({len(files)} files)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="Copy evidence from runs/ and Improving-CA-LoRA first")
    args = parser.parse_args(argv)
    if args.refresh:
        refresh()
    rows = load_rows()
    write_tables(rows)
    plot_jaccard()
    plot_tradeoff(rows)
    write_manifest()


if __name__ == "__main__":
    main()
