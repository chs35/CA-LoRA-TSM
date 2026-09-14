"""Score a generated arm: CMMD against real data, CLIP-Score against the prompts.

    python scripts/evaluate.py --config configs/sdxl_cityscapes.yaml --run-name no3

Reads ``<run_dir>/<arm>/generated/manifest.json`` from ``scripts/generate_images.py``,
exports the CMMD reference set once per run directory, and writes ``<arm>/eval.json``.

Copied from Optimizing-CA-LoRA/scripts/evaluate.py; the config is resolved
through ``catsm.cli`` so this project's extra sections are accepted, and the
manifest's ``routing`` block (bundle sets / core / disabled) is carried into
``eval.json``. mIoU is deliberately not computed (no label generator; same
decision as Improving CA-LoRA) — CMMD and per-condition CLIP-Score are the
reported metrics. ``--embedder pixel`` is the offline plumbing check; its
numbers are not reportable.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calora import cli  # noqa: E402
from calora.data import build_dataset, to_pil  # noqa: E402
from calora.eval.embeddings import PixelEmbedder  # noqa: E402
from calora.eval.metrics import CMMD_SCALE, CMMD_SIGMA, clip_score, cmmd  # noqa: E402
from calora.utils.io import dump_json, load_json  # noqa: E402
from calora.utils.io_images import save_image_verified, verify_image  # noqa: E402
from calora.utils.logging import get_logger  # noqa: E402
from catsm import cli as catsm_cli  # noqa: E402
from catsm.config import CatsmConfig  # noqa: E402

logger = get_logger("evaluate")


def export_reference(cfg: CatsmConfig) -> Path:
    """Write the CMMD reference images once (centre crop, un-mirrored); resumable per image."""
    directory = cfg.run_dir / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    expected = [directory / f"{index:05d}.png" for index in range(cfg.eval.reference_num_images)]
    missing = [index for index, path in enumerate(expected) if not _is_intact(path)]
    if not missing:
        logger.info("Reusing %d verified reference images in %s", len(expected), directory)
        return directory
    dataset = build_dataset(
        cfg.data, resolution=cfg.model.resolution, seed=cfg.eval.seed,
        crop="center", hflip=False, split=cfg.eval.reference_split,
    )
    count = min(cfg.eval.reference_num_images, len(dataset))
    todo = [index for index in missing if index < count]
    for index in todo:
        save_image_verified(to_pil(dataset[index]["image"]), expected[index])
    logger.info("Exported %d reference images (%d reused) from split %r to %s",
                len(todo), count - len(todo), cfg.eval.reference_split, directory)
    return directory


def _is_intact(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        verify_image(path)
    except OSError as exc:
        logger.warning("Reference image %s does not decode (%s); re-exporting", path, exc)
        with contextlib.suppress(OSError):
            path.unlink()
        return False
    return True


def main(argv: list[str] | None = None) -> None:
    parser = cli.base_parser(__doc__ or "")
    parser.add_argument("--run-name", required=True, help="Arm subdirectory under the run dir")
    parser.add_argument("--embedder", choices=("clip", "pixel"), default="clip",
                        help="'pixel' is the offline stand-in; its numbers are not reportable")
    args = parser.parse_args(argv)
    cfg = catsm_cli.resolve(args)

    arm_dir = cfg.run_dir / args.run_name
    manifest = load_json(arm_dir / "generated" / "manifest.json")
    images = [record for record in manifest["images"] if "path" in record]
    if not images:
        raise FileNotFoundError(
            f"{arm_dir / 'generated' / 'manifest.json'} lists no rendered images; "
            "run scripts/generate_images.py first"
        )

    if args.embedder == "pixel":
        logger.warning("Using PixelEmbedder: plumbing check only; CMMD/CLIP-Score are NOT reportable.")
        embedder = PixelEmbedder()
    else:
        from calora.eval.embeddings import CLIPEmbedder

        embedder = CLIPEmbedder(cfg.eval.clip_model)

    results: dict[str, object] = {}
    in_domain = [r for r in images if r["condition"] == "in_domain"]
    if in_domain:
        reference_dir = export_reference(cfg)
        generated_dir = (cfg.run_dir / in_domain[0]["path"]).parent
        results["cmmd"] = cmmd(reference_dir, generated_dir, embedder=embedder)
        logger.info("CMMD = %.4f (lower is better)", results["cmmd"]["cmmd"])
    else:
        logger.warning("No in-domain images in the manifest; skipping CMMD")

    per_condition: dict[str, dict] = {}
    for condition in sorted({r["condition"] for r in images}):
        group = [r for r in images if r["condition"] == condition]
        per_condition[condition] = clip_score(
            [cfg.run_dir / r["path"] for r in group], [r["prompt"] for r in group], embedder=embedder
        )
        logger.info("CLIP-Score[%s] = %.3f (n=%d)", condition, per_condition[condition]["clip_score"], len(group))
    overall = clip_score([cfg.run_dir / r["path"] for r in images], [r["prompt"] for r in images], embedder=embedder)
    weather = [c for c in per_condition if c != "in_domain"]
    results["clip_score"] = {
        "overall": overall,
        "per_condition": per_condition,
        "weather_mean": (sum(per_condition[c]["clip_score"] for c in weather) / len(weather)) if weather else None,
    }

    payload = {
        "run": cfg.name,
        "arm": args.run_name,
        "bundle": manifest.get("bundle"),
        "lora": manifest.get("lora"),
        "routing": manifest.get("routing"),
        "embedder": args.embedder,
        "reportable": args.embedder == "clip",
        "miou": None,  # not computed in this project (no label generator)
        "protocol": {
            **manifest.get("protocol", {}),
            "clip_model": cfg.eval.clip_model if args.embedder == "clip" else "pixel-stub",
            "reference_split": cfg.eval.reference_split,
            "reference_num_images": cfg.eval.reference_num_images,
            "cmmd_sigma": CMMD_SIGMA,
            "cmmd_scale": CMMD_SCALE,
        },
        "num_images": len(images),
        **results,
    }
    dump_json(payload, arm_dir / "eval.json")
    logger.info("Wrote %s", arm_dir / "eval.json")


if __name__ == "__main__":
    main()
