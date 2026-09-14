"""``python -m catsm.cli {select,foster,assemble} --config ...``

    select   --scheme n4              measurements -> selections/n4/<expert>.json
    foster   --expert n4_i0           selection    -> experts/n4_i0/lora-final.safetensors
    assemble --sets n1 n4 --core n4   experts      -> bundles/n1+n4-core-n4/bundle.safetensors
    assemble ... --disable n4_i3      control C2   -> bundles/n1+n4-core-n4-disable-n4_i3/

CA measurement itself is ``scripts/measure_ca.py`` (calora's script).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from calora.ca.select import Selection
from calora.config import parse_cli_overrides, save_config
from calora.data import build_dataset
from calora.models.loader import load_adapter
from calora.utils.io import dump_json
from calora.utils.logging import get_logger, setup_logging
from catsm.config import CatsmConfig, load_config
from catsm.intervals import build_scheme, parse_expert_name
from catsm.select import load_measurements, load_selection, select_scheme, write_scheme_selections
from catsm.train.assemble import build_assembly, train_router
from catsm.train.foster import train_expert

logger = get_logger("catsm")


def add_base_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Dotted config overrides, e.g. train.max_steps=10")
    parser.add_argument("--output-dir", default=None, help="Override output_dir")
    parser.add_argument("--name", default=None, help="Override the run name")
    parser.add_argument("--log-level", default="INFO")


def resolve(args: argparse.Namespace) -> CatsmConfig:
    """Load the config, apply overrides, set up logging, persist the resolved config."""
    overrides = parse_cli_overrides(args.set)
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.name:
        overrides["name"] = args.name
    cfg = load_config(args.config, overrides)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, log_file=cfg.run_dir / "run.log")
    save_config(cfg, cfg.run_dir / "config.yaml")
    return cfg


def selections_dir(cfg: CatsmConfig, scheme: str) -> Path:
    return cfg.run_dir / "selections" / scheme


def experts_dir(cfg: CatsmConfig) -> Path:
    return cfg.run_dir / "experts"


def bundle_dir(cfg: CatsmConfig, sets: list[str], core: str, disabled: list[str] = ()) -> Path:
    name = f"{'+'.join(sorted(sets))}-core-{core}"
    if disabled:
        name += f"-disable-{'+'.join(sorted(disabled))}"
    return cfg.run_dir / "bundles" / name


# --------------------------------------------------------------------------------------


def cmd_select(args: argparse.Namespace) -> None:
    cfg = resolve(args)
    mode = args.mode or cfg.interval_select.mode
    measurements = Path(args.measurements) if args.measurements else cfg.run_dir / "measurements"
    results = load_measurements(measurements, granularity=cfg.select.granularity)
    table = next(iter(results.values())).table()
    adapter = load_adapter(cfg.model)  # only for T and the log-SNR record
    scheme = build_scheme(args.scheme, adapter.num_train_timesteps, cfg.intervals)
    logger.info("Scheme %s: %s (mode %s, measured t=%s)",
                scheme.name, [iv.as_list() for iv in scheme.intervals], mode, sorted(results))
    selections = select_scheme(
        results, table, scheme,
        mode=mode,
        cfg=cfg.select,
        representative=cfg.interval_select.representative.get(scheme.name),
        fixed_t=cfg.interval_select.fixed_timestep,
    )
    out = write_scheme_selections(
        selections, scheme, selections_dir(cfg, scheme.name), alphas_cumprod=adapter.alphas_cumprod
    )
    for name, selection in selections.items():
        logger.info("%s: %d unit(s) from t=%s, sha %s", name, len(selection),
                    selection.meta["timesteps"], selection.meta["selection_sha"][:12])
    logger.info("Wrote %s", out)


def cmd_foster(args: argparse.Namespace) -> None:
    cfg = resolve(args)
    set_name, index = parse_expert_name(args.expert)
    adapter = load_adapter(cfg.model)
    scheme = build_scheme(set_name, adapter.num_train_timesteps, cfg.intervals)
    interval = scheme.intervals[index]
    path = Path(args.selection) if args.selection else selections_dir(cfg, set_name) / f"{args.expert}.json"
    selection = load_selection(path)
    dataset = build_dataset(cfg.data, resolution=cfg.model.resolution, seed=cfg.train.seed)
    output = experts_dir(cfg) / args.expert
    logger.info("Fostering %s on [%d, %d] from %s (%d units) -> %s",
                args.expert, interval.lo, interval.hi, path, len(selection), output)
    train_expert(
        adapter, dataset, selection, cfg.train,
        interval=interval,
        expert=args.expert,
        output_dir=output,
        metadata={
            "interval_scheme": scheme.name,
            "select_mode": str(selection.meta.get("mode", "")),
            "ca_timesteps": str(selection.meta.get("timesteps", "")),
            "selection_sha": str(selection.meta["selection_sha"]),
        },
    )
    dump_json(selection.to_dict(), output / "selection.json")
    logger.info("Expert %s written to %s", args.expert, output / "lora-final.safetensors")


def cmd_assemble(args: argparse.Namespace) -> None:
    cfg = resolve(args)
    sets = list(args.sets) if args.sets else list(cfg.assemble.sets)
    core = args.core or cfg.assemble.core
    adapter = load_adapter(cfg.model)
    disabled = list(args.disable or [])
    assembly = build_assembly(
        adapter, cfg,
        experts_dir=Path(args.experts_dir) if args.experts_dir else experts_dir(cfg),
        sets=sets, core=core, with_router=not args.no_router, disabled=disabled,
    )
    output = bundle_dir(cfg, sets, core, disabled)
    if assembly.router is None:
        from catsm.bundle import save_bundle

        save_bundle(assembly, output / "bundle.safetensors", extra_metadata={"steps": "0"})
        logger.info("Hard-routed bundle (no router) written to %s", output)
        return
    dataset = build_dataset(cfg.data, resolution=cfg.model.resolution, seed=cfg.assemble.seed)
    train_router(assembly, dataset, cfg.assemble, output_dir=output)
    logger.info("Bundle written to %s", output / "bundle.safetensors")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("select", help="Per-interval CA head selection for one scheme")
    add_base_args(p)
    p.add_argument("--scheme", required=True, help="Expert set, e.g. n4")
    p.add_argument("--mode", choices=("representative", "aggregate", "fixed"), default=None)
    p.add_argument("--measurements", default=None, help="Directory of t*.safetensors")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("foster", help="Train one expert on its interval")
    add_base_args(p)
    p.add_argument("--expert", required=True, help="Expert name, e.g. n4_i0")
    p.add_argument("--selection", default=None, help="Selection JSON (default: selections/<set>/<expert>.json)")
    p.set_defaults(func=cmd_foster)

    p = sub.add_parser("assemble", help="Load experts and train the router")
    add_base_args(p)
    p.add_argument("--sets", nargs="*", default=None, help="Expert sets (default: assemble.sets)")
    p.add_argument("--core", default=None, help="Core set (default: assemble.core)")
    p.add_argument("--experts-dir", default=None)
    p.add_argument("--no-router", action="store_true", help="Hard routing only; no training")
    p.add_argument("--disable", nargs="*", default=None, metavar="EXPERT",
                   help="Experts to keep in the bundle but route with weight 0 (control C2), "
                        "e.g. --disable n4_i3")
    p.set_defaults(func=cmd_assemble)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
