"""Generate (or complete, or just verify) the self-generated CA measurement set.

    python scripts/generate_selfgen.py --config configs/ca_selfgen.yaml
    python scripts/generate_selfgen.py --config configs/tiny_cpu.yaml --verify-only

The paper measures CA on images the base model generated from the fixed prompt c
(Sec. 3.2: "x0 = Phi_T2I(c; theta)"). This script samples
``measure.selfgen.num_images`` of them from c = ``data.caption_template`` with
the base weights (no LoRA is ever loaded here) into ``measure.selfgen.dir``, and
VAE-encodes them once into the latent cache next to them. Re-running it is
cheap: images that are present and pass their sha256 are not generated again,
so it doubles as the resume path after a crash and as the integrity check.

Copied from Improving-CA-LoRA/scripts/generate_selfgen.py; the config is
resolved through ``catsm.cli``. The set used for this project's selections is
the one made in Improving-CA-LoRA (runs/ca_selfgen_images).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calora import cli  # noqa: E402
from catsm import cli as catsm_cli  # noqa: E402
from calora.data.selfgen import (  # noqa: E402
    build_selfgen_dataset,
    generate_selfgen_set,
    manifest_sha256,
    selfgen_latents,
    spec_from_config,
    verify_selfgen_set,
)
from calora.models.loader import load_adapter  # noqa: E402
from calora.utils.io import dump_json  # noqa: E402
from calora.utils.logging import get_logger  # noqa: E402

logger = get_logger("generate_selfgen")


def main() -> None:
    parser = cli.base_parser(__doc__ or "")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Check the set on disk against the config without loading the model",
    )
    parser.add_argument(
        "--no-encode", action="store_true", help="Skip building the latent cache"
    )
    args = parser.parse_args()
    cfg = catsm_cli.resolve(args)
    sg = cfg.measure.selfgen
    directory = Path(sg.dir)

    if args.verify_only:
        manifest = verify_selfgen_set(directory, spec_from_config(cfg), num_images=sg.num_images)
        logger.info("OK: %d images at %s (manifest sha256 %s)",
                    manifest["num_images"], directory, manifest_sha256(directory))
        return

    adapter = load_adapter(cfg.model)
    started = time.perf_counter()
    manifest = generate_selfgen_set(
        adapter,
        spec_from_config(cfg, adapter),
        directory,
        num_images=sg.num_images,
        batch_size=sg.batch_size,
        extra_manifest={"config_name": cfg.name},
    )
    generation_seconds = time.perf_counter() - started
    verify_selfgen_set(directory, spec_from_config(cfg, adapter), num_images=sg.num_images)

    encode_seconds = 0.0
    if not args.no_encode:
        started = time.perf_counter()
        dataset = build_selfgen_dataset(
            cfg, num_items=sg.num_images, augs_per_sample=cfg.measure.augs_per_sample
        )
        selfgen_latents(adapter, dataset, cfg.model)
        encode_seconds = time.perf_counter() - started

    summary = {
        "dir": str(directory),
        "num_images": manifest["num_images"],
        "manifest_sha256": manifest_sha256(directory),
        "prompt": manifest["prompt"],
        "generation_seconds": generation_seconds,
        "encode_seconds": encode_seconds,
    }
    dump_json(summary, cfg.run_dir / "selfgen_summary.json")
    logger.info("Self-generated set ready: %s", summary)


if __name__ == "__main__":
    main()
