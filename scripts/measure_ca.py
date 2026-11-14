"""Measure Concept-Awareness across a grid of timesteps.

    python scripts/generate_selfgen.py --config configs/ca_selfgen.yaml   # once
    python scripts/measure_ca.py --config configs/ca_selfgen.yaml --skip-existing

x0 is, by default (``measure.source: selfgen``), the set of images the base model
generated from c -- the paper's protocol (Sec. 3.2, Algorithm 1). The set is made
by scripts/generate_selfgen.py and only *verified* here; its VAE latents are
encoded once and cached on disk next to it, and the prepared samples are reused
at every timestep, so differences between timesteps are not confounded by
resampling and a restart re-encodes nothing. ``measure.source: dataset`` reads
real images as every run before 2026-11-07 did.

One safetensors file per timestep is written to ``<run_dir>/measurements/``, with
the x0 source and numeric regime in its ``provenance`` header.

Copied from Improving-CA-LoRA/scripts/measure_ca.py; the only change is that
the config is resolved through ``catsm.cli`` so this project's extra sections
(``intervals``, ``interval_select``, ``assemble``) are accepted. This project does
not measure on its own any more: the self-generated measurement is made once in
Improving-CA-LoRA (runs/ca_selfgen) and imported with
scripts/ops/import_measurements.sh. The copy is kept in step so a local
re-measurement would follow the same protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from calora import cli  # noqa: E402
from catsm import cli as catsm_cli  # noqa: E402
from calora.ca.measure import measure_concept_awareness, prepare_samples  # noqa: E402
from calora.ca.precision import measurement_precision  # noqa: E402
from calora.ca.stats import save_measurement  # noqa: E402
from calora.data import build_measurement_dataset  # noqa: E402
from calora.data.selfgen import provenance_for, selfgen_latents, spec_from_config  # noqa: E402
from calora.models.loader import load_adapter  # noqa: E402
from calora.models.projections import ProjectionTable  # noqa: E402
from calora.utils.io import dump_json, load_json  # noqa: E402
from calora.utils.logging import get_logger  # noqa: E402

logger = get_logger("measure_ca")


def main() -> None:
    parser = cli.base_parser(__doc__ or "")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Leave already-measured timesteps alone"
    )
    args = parser.parse_args()
    cfg = catsm_cli.resolve(args)

    adapter = load_adapter(cfg.model)
    # The measurement keeps its own numeric and memory regime whatever the
    # training settings of the same config say: measure.precision (fp32 unless
    # the pre-fixed gate of scripts/ops/check_measure_precision.py adopted
    # tf32/bf16) and measure.gradient_checkpointing (on by default -- the
    # direction accumulators alone take ~36 GiB on SDXL).
    precision = cfg.measure.precision
    if cfg.model.allow_tf32 and precision == "fp32":
        logger.info("model.allow_tf32 is for training; measure.precision=fp32 keeps TF32 off")
    if cfg.measure.gradient_checkpointing:
        adapter.enable_gradient_checkpointing()
    logger.info(
        "Measurement regime: precision=%s, gradient checkpointing=%s",
        precision, cfg.measure.gradient_checkpointing,
    )
    # select.granularity decides what a unit is (head slice vs fused matrix) for
    # the whole pipeline, measurement included: the saved statistics are per unit
    # and cannot be converted between the two after the fact.
    table = ProjectionTable.from_model(adapter.unet, granularity=cfg.select.granularity)
    logger.info("Projection table: %s", table.summary())

    # crop="center" and hflip=False regardless of data.*: every timestep and both
    # concept axes must see the identical x0, which is the whole basis of the
    # pairing. A mirrored frame is as much a different x0 as a shifted crop is.
    dataset = build_measurement_dataset(
        cfg,
        num_items=cfg.measure.num_samples,
        crop="center",
        hflip=False,
        augs_per_sample=cfg.measure.augs_per_sample,
    )
    latents = None
    if cfg.measure.source == "selfgen":
        # Re-verified with the adapter's scheduler name, which the config alone
        # cannot supply: a set sampled by another scheduler is another set.
        from calora.data.selfgen import verify_selfgen_set

        verify_selfgen_set(
            dataset.directory, spec_from_config(cfg, adapter),
            num_images=len(dataset), check_files=False,
        )
        latents = selfgen_latents(adapter, dataset, cfg.model)
    provenance = {**provenance_for(cfg), "precision": precision}
    logger.info("x0 provenance: %s", provenance)

    axes = cfg.data.concept_axes
    logger.info(
        "Preparing %d samples over axes %s, %d augmentation(s) each",
        cfg.measure.num_samples,
        axes,
        cfg.measure.augs_per_sample,
    )
    samples = prepare_samples(
        adapter, dataset, axes=axes, num_samples=cfg.measure.num_samples, latents=latents
    )

    offload = getattr(adapter, "offload_encoders", None)
    if callable(offload):
        offload()

    summaries: dict[str, dict] = {}
    for timestep in cfg.measure.timesteps:
        path = cli.measurement_path(cfg, timestep)
        if args.skip_existing and path.exists():
            _check_existing_provenance(path, provenance)
            logger.info("t=%d already measured at %s; skipping", timestep, path)
            continue

        with measurement_precision(adapter.device, precision):
            result = measure_concept_awareness(
                adapter,
                table,
                samples,
                timestep=timestep,
                axes=axes,
                batch_size=cfg.measure.batch_size,
                granularity=cfg.measure.granularity,
                accumulate_direction=cfg.measure.accumulate_direction,
                compute_paired_cosine=cfg.measure.compute_paired_cosine,
                accum_device=(
                    adapter.device
                    if cfg.measure.accum_device == "auto"
                    else cfg.measure.accum_device
                ),
                eps=cfg.measure.eps,
                seed=cfg.measure.seed,
                paired_across_timesteps=cfg.measure.paired_across_timesteps,
                split_half=cfg.measure.split_half,
                log_every=cfg.measure.log_every,
            )
        result.provenance = dict(provenance)
        for half_result in result.halves or ():
            half_result.provenance = dict(provenance)
        save_measurement(
            result, path, save_direction_vectors=cfg.measure.save_direction_vectors
        )
        summary = result.summary()
        summaries[str(timestep)] = summary
        logger.info("t=%d -> %s", timestep, path)
        # ||E[ghat]|| / CA per axis: how much the axis's own c_Aug prompts agree
        # with each other. A low value makes this timestep's cos_of_means a
        # cosine between two cancelled-out vectors — check it before the ranking.
        concentration = summary.get("direction_concentration_mean")
        if concentration:
            logger.info(
                "t=%d direction concentration %s",
                timestep,
                {axis: round(value, 4) for axis, value in concentration.items()},
            )

        # The halves are written alongside the full result rather than inside it:
        # score_timesteps.py ranks each half independently, so it wants them as
        # ordinary measurements it can load and pool on their own.
        half_result = None  # bound even with no halves, for the del below
        for half, half_result in enumerate(result.halves or ()):
            half_path = cli.measurement_path(cfg, timestep, half=half)
            save_measurement(
                half_result,
                half_path,
                save_direction_vectors=cfg.measure.save_direction_vectors,
            )
            logger.info(
                "t=%d half%d (%d samples) -> %s",
                timestep, half, half_result.num_items, half_path,
            )

        # Two timesteps' accumulators must never be alive at once: the direction
        # vectors this result (and each half) carries on the accum device are the
        # transient half of the budget logged above, and 14.2 persistent + 2x35.6
        # transient does not fit an 80GB A100. `result` and the loop's
        # `half_result` are the only names still holding them, so drop both before
        # the next measure_concept_awareness() starts allocating, and return the
        # freed blocks to the pool that allocation draws from — the OOM crashed
        # with 6.85 GiB reserved but unallocated.
        del result, half_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = cfg.run_dir / "measure_summary.json"
    # With --skip-existing only the freshly measured timesteps are in `summaries`,
    # so writing it verbatim would drop the summaries of the ones we skipped —
    # whose safetensors files are still sitting in measurements/. Merge instead,
    # letting this run's numbers win for any timestep it did measure.
    merged = dict(_existing_summaries(summary_path))
    merged.update(summaries)

    dump_json(
        {
            "run": cfg.name,
            "timesteps": cfg.measure.timesteps,
            "axes": axes,
            "granularity": cfg.select.granularity,
            "provenance": provenance,
            "projections": table.summary(),
            "summaries": merged,
        },
        summary_path,
    )
    logger.info("Wrote %s (%d timestep summaries)", summary_path, len(merged))


def _check_existing_provenance(path: Path, provenance: dict[str, str]) -> None:
    """--skip-existing may only keep a file measured on the same x0 in the same regime.

    Otherwise a run directory holding pre-2026-11-07 real-image files would be
    silently completed with self-generated ones and ranked as one sweep.
    """
    from safetensors import safe_open

    from calora.ca.stats import PROVENANCE_IDENTITY_KEYS, read_provenance

    with safe_open(str(path), framework="pt") as handle:  # type: ignore[no-untyped-call]
        existing = read_provenance(handle.metadata() or {})
    for key in PROVENANCE_IDENTITY_KEYS:
        if existing.get(key) != provenance.get(key):
            raise RuntimeError(
                f"{path} was measured with {key}={existing.get(key)!r}, this run uses "
                f"{provenance.get(key)!r}. Use a fresh run name rather than mixing them."
            )


def _existing_summaries(path: Path) -> dict[str, dict]:
    """Per-timestep summaries already on disk; empty if the file is absent or unreadable."""
    if not path.exists():
        return {}
    try:
        previous = load_json(path)
    except (OSError, ValueError) as exc:  # a truncated file must not lose this run
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return {}
    summaries = previous.get("summaries")
    return summaries if isinstance(summaries, dict) else {}


if __name__ == "__main__":
    main()
