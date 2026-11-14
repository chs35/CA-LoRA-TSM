"""Trade-off figure for REPORT.md, with reader-facing names instead of arm ids.

Reads the eval.json copies under ``results/selfgen_fewshot10/evals`` and writes
``results/figures/report_tradeoff.png``. CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "results" / "selfgen_fewshot10" / "evals"
OUT = ROOT / "results" / "figures" / "report_tradeoff.png"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"

# arm -> (group, direct label, label offset in points, horizontal alignment)
POINTS = {
    "no1": ("calora", "CA-LoRA\n(one LoRA, heads chosen at t=81)", (10, 0), "left"),
    "c1": ("interval", "interval experts,\nevery selection at t=81", (0, -20), "center"),
    "no2": ("interval", "interval experts,\nselection at each interval's t", (10, -4), "left"),
    "no3": ("mix", "TSM mix", (12, -12), "left"),
    "c2": ("mixcut", "TSM mix without the\n[500, 999] interval expert", (10, 2), "left"),
}
MORE_SETS = ["no4", "no5", "no7", "no8", "no9"]
STYLE = {
    "calora": (BLUE, "o", "CA-LoRA"),
    "interval": (ORANGE, "s", "interval experts, no mixing"),
    "mix": (AQUA, "^", "TSM mix (filled: basic, open: more interval partitions)"),
    "mixcut": (YELLOW, "D", "TSM mix without the [500, 999] interval expert"),
}


def metrics(arm: str) -> tuple[float, float]:
    d = json.loads((EVALS / f"{arm}.json").read_text())
    return d["cmmd"]["cmmd"], d["clip_score"]["weather_mean"]


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=160)
    ax.set_facecolor("white")

    # the two no-mixing configurations differ only in the selection
    (x0, y0), (x1, y1) = metrics("c1"), metrics("no2")
    ax.plot([x0, x1], [y0, y1], color=ORANGE, lw=1.2, ls=(0, (4, 3)), zorder=1)

    for arm in MORE_SETS:
        x, y = metrics(arm)
        ax.scatter(x, y, s=56, marker="^", facecolors="white", edgecolors=AQUA, linewidths=1.6, zorder=3)
    for arm, (group, label, offset, ha) in POINTS.items():
        color, marker, _ = STYLE[group]
        x, y = metrics(arm)
        ax.scatter(x, y, s=92, marker=marker, color=color, edgecolors="white", linewidths=1.2, zorder=4)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", ha=ha, va="center",
                    fontsize=8.5, color=INK, linespacing=1.25)

    for group, (color, marker, name) in STYLE.items():
        ax.scatter([], [], s=70, marker=marker, color=color, label=name)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK, handletextpad=0.4)

    ax.set_xlim(1.10, 1.63)
    ax.set_ylim(19.3, 27.6)
    ax.set_xlabel("CMMD (lower = closer to Cityscapes)", fontsize=10, color=INK)
    ax.set_ylabel("weather CLIP-Score (higher = weather prompt followed)", fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.text(0.99, 0.98, "off scale: base SDXL without LoRA (4.08, 31.68);\nmix without the full-range expert (3.22, 22.58)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
