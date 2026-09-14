"""The assembled checkpoint: every expert's branches plus the router, one file.

Tensor keys::

    <module>.experts.<expert>.lora_A | lora_B | lora_head_indices
    router.<parameter>

Metadata ``disabled`` (JSON list) names experts that are stored but routed with
weight 0 (control C2); the loader restores that routing.

The keys are the ground truth (which experts, which heads); the metadata is what
a reader checks cheaply, and the loader refuses any disagreement between the two
— the same stance calora takes with its single-adapter files. ``save_lora`` is
not used here on purpose: it derives ``granularity`` from ``has_sliced_lora``,
which is false for the containers and would label the bundle ``module``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from calora.models.adapter import DenoiserAdapter
from calora.train.lora import lora_state_dict
from calora.utils.logging import get_logger
from catsm.intervals import Interval, IntervalScheme, validate_partition
from catsm.models.router import Router
from catsm.train.assemble import ExpertInfo, TSMAssembly, attach_expert_from_state

logger = get_logger(__name__)

BUNDLE_FORMAT = "tsm-sliced-v1"
ROUTER_PREFIX = "router."
EXPERTS_INFIX = ".experts."


def bundle_state_dict(assembly: TSMAssembly) -> dict[str, torch.Tensor]:
    state = lora_state_dict(assembly.unet)
    if assembly.router is not None:
        state.update(
            {ROUTER_PREFIX + k: v.detach().cpu() for k, v in assembly.router.state_dict().items()}
        )
    return state


def bundle_metadata(
    assembly: TSMAssembly, *, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    axes = {e.axis for e in assembly.experts if e.axis}
    meta = {
        "format": BUNDLE_FORMAT,
        "num_train_timesteps": str(assembly.adapter.num_train_timesteps),
        "sets": json.dumps(assembly.sets),
        "core": assembly.core,
        "intervals": json.dumps(
            {name: [iv.as_list() for iv in scheme.intervals]
             for name, scheme in assembly.schemes.items()}
        ),
        "experts": json.dumps([e.to_dict() for e in assembly.experts], sort_keys=True),
        "router": json.dumps(assembly.router.describe()) if assembly.router else "null",
        "disabled": json.dumps(assembly.disabled),
    }
    if len(axes) == 1:
        meta["axis"] = axes.pop()
    meta.update({k: str(v) for k, v in (extra or {}).items() if v not in (None, "")})
    return meta


def save_bundle(
    assembly: TSMAssembly, path: str | Path, *, extra_metadata: Mapping[str, str] | None = None
) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(bundle_state_dict(assembly), str(path), metadata=bundle_metadata(assembly, extra=extra_metadata))
    logger.info(
        "Saved bundle %s: %d expert(s) over sets %s, router=%s",
        path, len(assembly.experts), assembly.sets, assembly.router is not None,
    )


def split_expert_keys(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """``{expert: {"<module>.lora_X": tensor}}`` plus the router tensors (prefix stripped)."""
    experts: dict[str, dict[str, torch.Tensor]] = {}
    router: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.startswith(ROUTER_PREFIX):
            router[key[len(ROUTER_PREFIX):]] = tensor
            continue
        if EXPERTS_INFIX not in key:
            raise ValueError(f"Unexpected tensor {key!r}: neither an expert branch nor the router")
        module, _, rest = key.partition(EXPERTS_INFIX)
        expert, _, suffix = rest.partition(".")
        if not suffix.startswith("lora_"):
            raise ValueError(f"Unexpected tensor {key!r} under expert {expert!r}")
        experts.setdefault(expert, {})[f"{module}.{suffix}"] = tensor
    return experts, router


def load_bundle(adapter: DenoiserAdapter, path: str | Path) -> TSMAssembly:
    """Rebuild an assembly on a pristine UNet, refusing anything inconsistent."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    state = load_file(str(path))
    with safe_open(str(path), framework="pt") as handle:
        meta = handle.metadata() or {}
    if meta.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{path} is not a {BUNDLE_FORMAT} bundle (format={meta.get('format')!r})")
    T = adapter.num_train_timesteps
    if int(meta.get("num_train_timesteps", -1)) != T:
        raise ValueError(
            f"{path} was assembled for {meta.get('num_train_timesteps')} timesteps, the model has {T}"
        )
    sets = json.loads(meta["sets"])
    core = meta["core"]
    if core not in sets:
        raise ValueError(f"{path}: core {core!r} is not one of the sets {sets}")
    intervals = json.loads(meta["intervals"])
    schemes: dict[str, IntervalScheme] = {}
    for name in sets:
        ivs = tuple(Interval(int(lo), int(hi)) for lo, hi in intervals[name])
        validate_partition(ivs, T)
        schemes[name] = IntervalScheme(name, ivs)
    infos = [ExpertInfo.from_dict(d) for d in json.loads(meta["experts"])]
    declared = sorted(e.name for e in infos)
    expected = sorted(e for s in schemes.values() for e in s.expert_names())
    if declared != expected:
        raise ValueError(f"{path}: metadata lists experts {declared}, the sets need {expected}")

    per_expert, router_state = split_expert_keys(state)
    found = sorted(per_expert)
    if found != declared:
        raise ValueError(f"{path}: tensors cover experts {found}, metadata declares {declared}")

    by_name = {e.name: e for e in infos}
    for name in declared:
        info = by_name[name]
        rank, names = attach_expert_from_state(
            adapter.unet, per_expert[name], name=name, alpha=info.alpha, path=path, axis=info.axis
        )
        if rank != info.rank:
            raise ValueError(f"{path}: expert {name} has rank {rank}, metadata says {info.rank}")

    router = None
    router_meta = json.loads(meta.get("router", "null"))
    if router_meta is None:
        if router_state:
            raise ValueError(f"{path}: carries router tensors but declares no router")
    else:
        if not router_state:
            raise ValueError(f"{path}: declares a router but carries no router tensors")
        router = Router(
            router_meta["context_sets"],
            router_meta["modules"],
            latent_channels=int(router_meta["latent_channels"]),
            num_train_timesteps=int(router_meta["num_train_timesteps"]),
            init_gate=float(router_meta.get("init_gate", 1.0)),
        )
        missing, unexpected = router.load_state_dict(router_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{path}: router load mismatch ({len(missing)} missing, {len(unexpected)} unexpected)"
            )
    disabled = json.loads(meta.get("disabled", "[]"))
    assembly = TSMAssembly(
        adapter, schemes=schemes, core=core, experts=infos, router=router, disabled=disabled
    )
    logger.info(
        "Loaded bundle %s: sets %s, core %s, router=%s", path, sets, core, router is not None
    )
    return assembly
