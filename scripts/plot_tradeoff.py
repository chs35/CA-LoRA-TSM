"""Alignment-vs-controllability scatter (CMMD vs weather CLIP-Score) from the eval.json files.

    python scripts/plot_tradeoff.py            # writes docs/figures/cmmd_vs_weather_clip.png
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs/sdxl_cityscapes"
U = Path("/workspace/Optimizing-CA-LoRA/runs/sdxl_cityscapes")
ARMS = [("base", U / "control_base/eval.json", "ref"), ("No.1", R / "no1/eval.json", "single"), ("No.2", R / "no2/eval.json", "hard"),
        *[(f"No.{i}", R / f"no{i}/eval.json", "router") for i in range(3, 10)],
        ("C1", ROOT / "runs/sdxl_c1/c1/eval.json", "hard"), ("C2-n2", R / "c2n2/eval.json", "ctrl"), ("C2-n3", R / "c2/eval.json", "ctrl")]
GROUPS = {"single": ("#2a78d6", "single LoRA (No.1)", "o"), "hard": ("#eb6834", "hard routing (No.2, C1)", "s"),
          "router": ("#1baf7a", "router (No.3-9)", "^"), "ctrl": ("#eda100", "C2 ablation controls", "D"),
          "ref": ("#6b6a63", "base SDXL", "X")}
OFFSET = {"No.1": (-6, -15), "C2-n3": (-14, 9), "C1": (9, -3), "No.2": (9, 3), "No.6": (9, 3), "C2-n2": (9, 3), "base": (-34, -4)}


def draw(ax, pts, labels):
    for k, (x, y, g) in pts.items():
        c, _, m = GROUPS[g]
        ax.scatter(x, y, s=80, c=c, marker=m, edgecolors="white", linewidths=1.5, zorder=3)
        if k in labels:
            ax.annotate(k, (x, y), xytext=labels[k], textcoords="offset points", fontsize=9, color="#1a1a19")
    ax.grid(True, color="#e6e5df", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#6b6a63", labelsize=9)


def main():
    pts = {}
    for k, p, g in ARMS:
        d = json.loads(p.read_text())
        pts[k] = (d["cmmd"]["cmmd"], d["clip_score"]["weather_mean"], g)
    fig, (ax, zx) = plt.subplots(1, 2, figsize=(13, 6), dpi=150, gridspec_kw={"width_ratios": [1.5, 1]})
    draw(ax, pts, {k: OFFSET.get(k, (7, 4)) for k in pts if pts[k][2] != "router" or k == "No.6"})
    ax.set_xlabel("CMMD  (lower = closer to Cityscapes)", color="#1a1a19")
    ax.set_ylabel("weather CLIP-Score  (higher = prompt still obeyed)", color="#1a1a19")
    ax.set_title("Alignment vs controllability, all arms (seed 0, 900 images each)", loc="left", fontsize=12)
    ax.annotate("better", (0.03, 0.93), xycoords="axes fraction", fontsize=9, color="#6b6a63")
    ax.annotate("", (0.015, 0.985), xytext=(0.075, 0.90), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": "#6b6a63"})
    for g, (c, l, m) in GROUPS.items():
        ax.scatter([], [], c=c, marker=m, s=70, label=l)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    zoom = {k: v for k, v in pts.items() if v[0] < 1.65 and v[1] < 24}
    zoff = {"No.1": (8, -4), "C2-n3": (8, 2), "C1": (8, 2), "No.3": (8, 2), "No.4": (8, 2), "No.5": (-10, -15), "No.7": (-38, -3),
            "No.8": (8, -4), "No.9": (8, 0)}
    draw(zx, zoom, zoff)
    zx.set_xlim(0.9, 1.68)
    zx.set_xlabel("CMMD", color="#1a1a19")
    zx.set_title("Zoom: t=81-selection cluster", loc="left", fontsize=12)
    fig.tight_layout()
    out = ROOT / "docs/figures/cmmd_vs_weather_clip.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
