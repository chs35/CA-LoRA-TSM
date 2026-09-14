"""Shared CLI plumbing for the ``scripts/`` entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from calora.config import Config, load_config, parse_cli_overrides, save_config
from calora.utils.logging import setup_logging


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config overrides, e.g. measure.num_samples=32 model.device=cpu",
    )
    parser.add_argument("--output-dir", default=None, help="Override output_dir")
    parser.add_argument("--name", default=None, help="Override the run name")
    parser.add_argument("--log-level", default="INFO")
    return parser


def resolve(args: argparse.Namespace) -> Config:
    """Load the config, apply overrides, set up logging, and persist the result."""
    overrides = parse_cli_overrides(args.set)
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.name:
        overrides["name"] = args.name

    cfg = load_config(args.config, overrides)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, log_file=run_dir / "run.log")
    save_config(cfg, run_dir / "config.yaml")
    return cfg


def measurement_dir(cfg: Config) -> Path:
    path = cfg.run_dir / "measurements"
    path.mkdir(parents=True, exist_ok=True)
    return path


def measurement_path(cfg: Config, timestep: int, half: int | None = None) -> Path:
    """Where one timestep's measurement lives.

    ``half`` names one side of a ``measure.split_half`` run. The suffix keeps the
    halves in the same directory as the full result while staying distinguishable
    from it, so readers that only want full measurements can filter on it.
    """
    suffix = "" if half is None else f"_half{half}"
    return measurement_dir(cfg) / f"t{timestep:04d}{suffix}.safetensors"


def is_half_measurement(path: Path) -> bool:
    return "_half" in path.stem
