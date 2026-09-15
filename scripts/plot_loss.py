"""Loss curves of fostered experts (one panel per expert, shared step axis).

    python scripts/plot_loss.py                      # -> runs/loss_curves.png
    python scripts/plot_loss.py --out path.png

Completed experts are read from runs/<run>/experts/<expert>/train_summary.json
(`history`, one entry per log_every=50 steps); an expert still training is read
from the tail of runs/<run>/run.log (its last "Fostering <expert>" block).
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEP_RE = re.compile(r"\| (\S+) step (\d+)/(\d+)\s+loss=([0-9.eE+-]+|nan)")
RUNS = [("sdxl_cityscapes", "runs/sdxl_cityscapes"), ("sdxl_c1", "runs/sdxl_c1")]

def completed(run_dir: Path):
    out = {}
    for p in sorted(run_dir.glob("experts/*/train_summary.json")):
        s = json.loads(p.read_text())
        h = s["history"]
        out[s["expert"]] = dict(steps=[e["step"] for e in h], loss=[e["loss"] for e in h],
                                interval=s["interval"], done=True)
    return out

def in_progress(run_dir: Path, skip: set[str]):
    log = run_dir / "run.log"
    if not log.exists():
        return {}
    blocks: dict[str, list] = {}
    for line in log.read_text().splitlines():
        if "Fostering " in line and " on [" in line:
            name = line.split("Fostering ")[1].split(" ")[0]
            blocks[name] = []            # restart -> drop the earlier (killed) attempt
            continue
        m = STEP_RE.search(line)
        if m and m.group(1) in blocks:
            blocks[m.group(1)].append((int(m.group(2)), float(m.group(4))))
    return {n: dict(steps=[s for s, _ in v], loss=[l for _, l in v], interval=None, done=False)
            for n, v in blocks.items() if n not in skip and v}

def smooth(y, k=10):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/loss_curves.png")
    a = ap.parse_args()
    panels = []
    for run_name, rd in RUNS:
        rd = Path(rd)
        done = completed(rd)
        prog = in_progress(rd, set(done))
        for name, d in list(done.items()) + list(prog.items()):
            panels.append((run_name, name, d))
    if not panels:
        sys.exit("no expert histories found")
    n = len(panels); cols = min(3, n); rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.2 * rows), squeeze=False,
                             facecolor="#fcfcfb")
    blue = "#2a78d6"
    for ax, (run_name, name, d) in zip(axes.flat, panels):
        s, y = np.asarray(d["steps"]), np.asarray(d["loss"])
        ax.set_facecolor("#fcfcfb")
        ax.plot(s, y, color=blue, alpha=0.28, lw=1.0)
        ax.plot(s, smooth(y), color=blue, lw=2.0)
        iv = d["interval"]
        label = f"{run_name} / {name}" + (f"  t∈[{iv[0]},{iv[1]}]" if iv else "") + \
                ("" if d["done"] else f"  (in progress, step {s[-1]})")
        ax.set_title(label, fontsize=10, color="#0b0b0b", loc="left")
        ax.set_xlim(0, 4000); ax.set_ylim(0, max(0.05, float(np.nanmax(y)) * 1.05))
        ax.grid(True, color="#e6e5e2", lw=0.6); ax.tick_params(colors="#52514e", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#d8d7d3")
        ax.annotate(f"last {y[-1]:.3f}  mean(last 10) {smooth(y)[-1]:.3f}", xy=(0.02, 0.92),
                    xycoords="axes fraction", fontsize=8, color="#52514e")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("step", fontsize=9, color="#52514e")
    for r in axes:
        r[0].set_ylabel("loss (batch 1)", fontsize=9, color="#52514e")
    fig.suptitle("Fostering loss — raw per-50-step value (light) and 10-point running mean (dark)",
                 fontsize=11, color="#0b0b0b", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out} ({n} panels)")

if __name__ == "__main__":
    main()
