"""Collect every <run>/<arm>/eval.json into one table (stage 6, docs/GPU_RUNBOOK.md).

    python scripts/collect_results.py                                  # runs/sdxl_cityscapes + runs/sdxl_c1
    python scripts/collect_results.py --extra /workspace/Optimizing-CA-LoRA/runs/sdxl_cityscapes/control_base/eval.json
    python scripts/collect_results.py --md docs/results_table.md --json runs/results.json

Rows: arm, bundle sets/core/disabled, generation seed, CMMD (lower is better), in-domain
CLIP-Score, weather CLIP-Score mean and per condition, image count. Arms whose name ends in
``_smoke`` or whose ``reportable`` flag is false are listed separately and never in the main table.
Seed replicates (``<arm>_s<k>``) are grouped with their base arm and summarised as mean ± std.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        d = json.loads(path.read_text())
        routing = d.get("routing") or {}
        pc = (d.get("clip_score") or {}).get("per_condition", {})
        rows.append({
            "run": d.get("run"), "arm": d.get("arm"), "path": str(path),
            "sets": "+".join(routing.get("sets", [])) if routing else ("lora" if d.get("lora") else "base"),
            "core": routing.get("core"), "disabled": "+".join(routing.get("disabled", [])) or "",
            "seed": (d.get("protocol") or {}).get("seed"),
            "reportable": d.get("reportable", True),
            "cmmd": (d.get("cmmd") or {}).get("cmmd"),
            "clip_in": pc.get("in_domain", {}).get("clip_score"),
            "clip_weather": (d.get("clip_score") or {}).get("weather_mean"),
            "per_condition": {c: v.get("clip_score") for c, v in pc.items() if c != "in_domain"},
            "n": d.get("num_images"),
        })
    return rows


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=["runs/sdxl_cityscapes", "runs/sdxl_c1"])
    parser.add_argument("--extra", nargs="*", default=[], help="Additional eval.json files (e.g. upstream base control)")
    parser.add_argument("--md", default=None, help="Write the markdown table here")
    parser.add_argument("--json", default=None, help="Write all rows here")
    args = parser.parse_args(argv)

    paths = sorted(p for run in args.runs for p in Path(run).glob("*/eval.json"))
    paths += [Path(p) for p in args.extra]
    rows = load_rows(paths)
    main_rows = [r for r in rows if r["reportable"] and not r["arm"].endswith("_smoke")]
    side_rows = [r for r in rows if r not in main_rows]

    conditions = sorted({c for r in rows for c in r["per_condition"]})
    head = ["arm", "run", "sets", "core", "disabled", "seed", "n", "CMMD↓", "CLIP in", "CLIP weather↑", *conditions]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in sorted(main_rows, key=lambda r: (r["run"], r["arm"])):
        lines.append("| " + " | ".join([
            r["arm"], r["run"], r["sets"], r["core"] or "—", r["disabled"] or "—", str(r["seed"]), str(r["n"]),
            fmt(r["cmmd"]), fmt(r["clip_in"], 2), fmt(r["clip_weather"], 2),
            *[fmt(r["per_condition"].get(c), 2) for c in conditions],
        ]) + " |")

    # seed replicates: base arm = name without _s<k>
    groups: dict[tuple, list[dict]] = {}
    for r in main_rows:
        base = re.sub(r"_s\d+$", "", r["arm"])
        groups.setdefault((r["run"], base), []).append(r)
    agg = []
    for (run, base), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        def ms(key):
            vals = [m[key] for m in members if m[key] is not None]
            return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0) if vals else (None, None)
        c, ci, cw = ms("cmmd"), ms("clip_in"), ms("clip_weather")
        agg.append(f"| {base} | {run} | {len(members)} | {fmt(c[0])} ± {fmt(c[1])} | {fmt(ci[0], 2)} ± {fmt(ci[1], 2)} | {fmt(cw[0], 2)} ± {fmt(cw[1], 2)} |")
    if agg:
        lines += ["", "Seed replicates (mean ± std over generation seeds):", "",
                  "| arm | run | seeds | CMMD↓ | CLIP in | CLIP weather↑ |", "|---|---|---|---|---|---|", *agg]
    if side_rows:
        lines += ["", "Not reportable / smoke:", ""] + [f"- {r['run']}/{r['arm']} ({r['path']})" for r in side_rows]

    text = "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
