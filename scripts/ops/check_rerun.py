"""Does one finished stage of the few-shot rerun have sound artifacts? Exit 0 if so.

    python scripts/ops/check_rerun.py expert   --run sdxl_cityscapes_fewshot10 --name n4_i0
    python scripts/ops/check_rerun.py bundle   --run sdxl_cityscapes_fewshot10 --name n4-core-n4
    python scripts/ops/check_rerun.py generate --run sdxl_cityscapes_fewshot10 --name no2
    python scripts/ops/check_rerun.py evaluate --run sdxl_cityscapes_fewshot10 --name no2

The network volume corrupts and loses writes, so this reads files back rather than
trusting that they exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_IMAGES = 900
SUBSET_SIZE = 10
#: Below this per-image pixel std (0-255) the image is flat: what a NaN latent decodes to.
MIN_PIXEL_STD = 1.0


def finite_tensors(path: Path) -> tuple[dict, list[str]]:
    import torch
    from safetensors import safe_open

    problems = []
    with safe_open(str(path), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                problems.append(f"non-finite tensor {key}")
    return metadata, problems


def check_expert(run_dir: Path, name: str) -> list[str]:
    final = run_dir / "experts" / name / "lora-final.safetensors"
    if not final.exists():
        return [f"missing {final}"]
    metadata, problems = finite_tensors(final)
    keys = [key for key in metadata.get("training_subset", "").split(",") if key]
    if len(keys) != SUBSET_SIZE:
        problems.append(f"training_subset names {len(keys)} images, expected {SUBSET_SIZE}")
    recorded = json.loads((run_dir / "training_subset.json").read_text())
    if keys != recorded["keys"]:
        problems.append("training_subset differs from training_subset.json")
    if "shared_from" not in metadata:
        lo, hi = int(metadata["t_lo"]), int(metadata["t_hi"])
        if not lo <= int(metadata["t_min_seen"]) <= int(metadata["t_max_seen"]) <= hi:
            problems.append(f"timesteps seen left [{lo}, {hi}]")
    selection = json.loads((run_dir / "experts" / name / "selection.json").read_text())
    if selection["meta"]["selection_sha"] != metadata["selection_sha"]:
        problems.append("selection sha differs from selection.json")
    return problems


def check_bundle(run_dir: Path, name: str) -> list[str]:
    path = run_dir / "bundles" / name / "bundle.safetensors"
    if not path.exists():
        return [f"missing {path}"]
    return finite_tensors(path)[1]


def check_generate(run_dir: Path, name: str) -> list[str]:
    import numpy as np
    from PIL import Image

    manifest = run_dir / name / "generated" / "manifest.json"
    if not manifest.exists():
        return [f"missing {manifest}"]
    records = [r for r in json.loads(manifest.read_text())["images"] if "path" in r]
    if len(records) != EXPECTED_IMAGES:
        return [f"{len(records)} images in the manifest, expected {EXPECTED_IMAGES}"]

    def problem(record: dict) -> str | None:
        path = run_dir / record["path"]
        try:
            with Image.open(path) as image:
                std = float(np.asarray(image.convert("RGB")).std())
        except Exception as exc:  # PIL raises SyntaxError on a corrupt PNG, not only OSError
            return f"{path}: {exc}"
        return None if std >= MIN_PIXEL_STD else f"{path}: flat image (std={std:.2f})"

    with ThreadPoolExecutor(8) as pool:
        return [found for found in pool.map(problem, records) if found]


def check_evaluate(run_dir: Path, name: str) -> list[str]:
    path = run_dir / name / "eval.json"
    if not path.exists():
        return [f"missing {path}"]
    payload = json.loads(path.read_text())
    problems = []
    if not payload.get("reportable"):
        problems.append("eval.json is not reportable")
    if "cmmd" not in json.dumps(payload):
        problems.append("eval.json has no CMMD")
    return problems


CHECKS = {"expert": check_expert, "bundle": check_bundle, "generate": check_generate, "evaluate": check_evaluate}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=sorted(CHECKS))
    parser.add_argument("--run", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    problems = CHECKS[args.stage](ROOT / "runs" / args.run, args.name)
    for line in problems:
        print(f"{args.stage} {args.run}/{args.name}: {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
