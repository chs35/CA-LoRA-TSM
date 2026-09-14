"""Assembling stage: load fostered experts into containers, train the router.

Expert files are the single-adapter checkpoints the fostering stage writes.
Each is read back through calora's own key parsing (the ``lora_head_indices``
tensors are the ground truth for which heads it covers), attached as one
branch per matrix, and loaded strictly per branch — so a file whose keys do
not all land somewhere is refused rather than half-loaded.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from calora.data.transforms import hflip
from calora.models.adapter import DenoiserAdapter
from calora.train.lora import (
    StepCallback,
    TrainingState,
    _autocast,
    _checkpoint_granularity,
    _draw_flips,
    _draw_indices,
    _format_duration,
    _head_selection_from_state,
    _InputCache,
    _is_deterministic,
    _ItemPrefetcher,
    _micro_batch_plan,
    _park_text_encoders,
    build_lr_lambda,
    lora_parameters,
    plan_sliced_lora,
    resolve_mixed_precision,
)
from calora.utils.io import dump_json
from calora.utils.logging import get_logger
from calora.utils.seeding import derive_generator, seed_everything
from catsm.config import AssembleConfig, CatsmConfig
from catsm.intervals import Interval, IntervalScheme, build_scheme, parse_expert_name
from catsm.models.multi_sliced_lora import (
    attach_multi_sliced_lora,
    clear_gate,
    expert_names,
    gated_modules,
    set_gate,
)
from catsm.models.router import Router, RoutingTable
from catsm.select import selection_sha

logger = get_logger(__name__)

BRANCH_KEYS = ("lora_A", "lora_B", "lora_head_indices")


@dataclass(frozen=True)
class ExpertInfo:
    name: str
    set: str
    t_lo: int
    t_hi: int
    rank: int
    alpha: float
    selection_sha: str
    steps: int | None = None
    source: str = ""
    axis: str = ""
    logsnr_lo: float | None = None
    logsnr_hi: float | None = None

    @property
    def interval(self) -> Interval:
        return Interval(self.t_lo, self.t_hi)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> ExpertInfo:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


# --------------------------------------------------------------------------------------
# Loading expert files into containers
# --------------------------------------------------------------------------------------


def read_expert_file(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    state = load_file(str(path))
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    if not state:
        raise ValueError(f"{path} contains no tensors")
    if _checkpoint_granularity(state, metadata) != "head":
        raise ValueError(f"{path} is not a head-granularity (sliced) LoRA checkpoint")
    for key in ("lora_alpha", "t_lo", "t_hi"):
        if key not in metadata:
            raise ValueError(
                f"{path} records no {key!r}; an expert file must carry its interval and alpha "
                "(a wrong alpha silently rescales the expert)"
            )
    return state, metadata


def attach_expert_from_state(
    unet: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    name: str,
    alpha: float,
    path: Path,
    axis: str = "",
) -> tuple[int, list[str]]:
    """Attach ``name`` from single-adapter keys ``<module>.lora_*`` and load them strictly.

    Returns the rank and the selected unit names. Every ``.lora_`` key of
    ``state`` must be consumed, or the file described something the model
    cannot express and loading is refused.
    """
    selection = _head_selection_from_state(state, path=path, axis=axis)
    plans = plan_sliced_lora(unet, selection)
    ranks = {int(v.shape[0]) for k, v in state.items() if k.endswith(".lora_A")}
    if len(ranks) != 1:
        raise ValueError(f"{path} mixes LoRA ranks {sorted(ranks)}; one expert has one rank")
    rank = ranks.pop()
    branches = attach_multi_sliced_lora(unet, plans, expert=name, rank=rank, alpha=alpha)
    consumed: set[str] = set()
    for plan, branch in zip(plans, branches):
        payload = {}
        for suffix in BRANCH_KEYS:
            key = f"{plan.module_name}.{suffix}"
            if key not in state:
                raise RuntimeError(f"{path}: missing tensor {key}")
            payload[suffix] = state[key]
            consumed.add(key)
        branch.load_state_dict(payload, strict=True)
    stray = sorted(k for k in state if ".lora_" in k and k not in consumed)
    if stray:
        raise RuntimeError(f"{path}: {len(stray)} LoRA tensor(s) not consumed, e.g. {stray[:3]}")
    return rank, list(selection.names)


def load_expert_into(
    unet: nn.Module,
    path: str | Path,
    *,
    name: str | None = None,
    expect_interval: Interval | None = None,
) -> ExpertInfo:
    """Attach the expert stored at ``path`` to ``unet`` and describe it."""
    path = Path(path)
    state, meta = read_expert_file(path)
    recorded = meta.get("expert")
    if name is None:
        if not recorded:
            raise ValueError(f"{path} records no expert name; pass name=")
        name = recorded
    elif recorded and recorded != name:
        raise ValueError(f"{path} was trained as expert {recorded!r}, not {name!r}")
    set_name, _ = parse_expert_name(name)
    interval = Interval(int(meta["t_lo"]), int(meta["t_hi"]))
    if expect_interval is not None and interval != expect_interval:
        raise ValueError(
            f"{path}: expert {name} was trained on [{interval.lo}, {interval.hi}] but the "
            f"scheme puts it at [{expect_interval.lo}, {expect_interval.hi}]"
        )
    alpha = float(meta["lora_alpha"])
    rank, names = attach_expert_from_state(
        unet, state, name=name, alpha=alpha, path=path, axis=meta.get("axis", "")
    )
    sha = selection_sha(names)
    if meta.get("selection_sha") and meta["selection_sha"] != sha:
        raise ValueError(
            f"{path}: the heads in the file hash to {sha[:12]} but the metadata says "
            f"{meta['selection_sha'][:12]}; the file does not match its selection"
        )
    steps = meta.get("steps")
    logger.info(
        "Loaded expert %s from %s: %d head unit(s), rank %d, alpha %g, t∈[%d,%d]",
        name, path, len(names), rank, alpha, interval.lo, interval.hi,
    )
    return ExpertInfo(
        name=name,
        set=set_name,
        t_lo=interval.lo,
        t_hi=interval.hi,
        rank=rank,
        alpha=alpha,
        selection_sha=sha,
        steps=int(steps) if steps else None,
        source=str(path),
        axis=meta.get("axis", ""),
        logsnr_lo=float(meta["logsnr_lo"]) if "logsnr_lo" in meta else None,
        logsnr_hi=float(meta["logsnr_hi"]) if "logsnr_hi" in meta else None,
    )


# --------------------------------------------------------------------------------------
# The assembled model
# --------------------------------------------------------------------------------------


class TSMAssembly:
    """A UNet carrying every expert of several sets, plus the routing between them."""

    def __init__(
        self,
        adapter: DenoiserAdapter,
        *,
        schemes: Mapping[str, IntervalScheme],
        core: str,
        experts: Sequence[ExpertInfo],
        router: Router | None,
        disabled: Sequence[str] = (),
    ):
        self.adapter = adapter
        self.routing = RoutingTable(
            schemes, core=core, num_train_timesteps=adapter.num_train_timesteps, disabled=disabled
        )
        self.schemes = self.routing.schemes
        self.core = core
        self.experts = sorted(experts, key=lambda e: e.name)
        expected = self.routing.expert_names
        got = sorted(e.name for e in self.experts)
        if got != expected:
            raise ValueError(f"Experts {got} do not match the schemes' experts {expected}")
        attached = expert_names(adapter.unet)
        if attached != expected:
            raise ValueError(f"UNet carries experts {attached}, expected {expected}")
        for info in self.experts:
            scheme_interval = self.schemes[info.set].interval_of(info.name)
            if info.interval != scheme_interval:
                raise ValueError(
                    f"{info.name} trained on {info.interval} but its scheme says {scheme_interval}"
                )
        # Router rows exist only for matrices carrying an *active* context expert:
        # a disabled context expert never contributes, so it needs no weight.
        self.gated_modules = gated_modules(adapter.unet, self.context_experts)
        if self.routing.context_sets and not self.gated_modules and router is not None:
            raise ValueError("Every context expert is disabled; assemble with --no-router")
        if router is not None:
            if tuple(router.context_sets) != tuple(self.routing.context_sets):
                raise ValueError(
                    f"Router covers {router.context_sets}, the assembly needs "
                    f"{self.routing.context_sets}"
                )
            if list(router.modules_gated) != self.gated_modules:
                raise ValueError(
                    f"Router gates {len(router.modules_gated)} module(s) but the loaded context "
                    f"experts occupy {len(self.gated_modules)}; the lists must match exactly"
                )
            if router.num_train_timesteps != adapter.num_train_timesteps:
                raise ValueError("Router and model disagree on num_train_timesteps")
            router.to(adapter.device)
        elif self.routing.context_sets:
            logger.info(
                "No router: context sets %s ride with weight 1 (hard routing)",
                self.routing.context_sets,
            )
        self.router = router
        self.gate: dict[str, torch.Tensor] = {}
        set_gate(adapter.unet, self.gate)

    @property
    def unet(self) -> nn.Module:
        return self.adapter.unet

    @property
    def sets(self) -> list[str]:
        return self.routing.sets

    @property
    def context_sets(self) -> list[str]:
        return self.routing.context_sets

    @property
    def expert_names(self) -> list[str]:
        return [e.name for e in self.experts]

    @property
    def disabled(self) -> list[str]:
        return list(self.routing.disabled)

    @property
    def context_experts(self) -> list[str]:
        """Context-set experts that can actually fire (disabled ones excluded)."""
        return [
            e.name for e in self.experts if e.set != self.core and e.name not in self.routing.disabled
        ]

    def gate_for(
        self, t: torch.Tensor, z_t: torch.Tensor | None = None
    ) -> dict[str | tuple[str, str], torch.Tensor]:
        if self.router is None:
            return self.routing.gate(t)
        if z_t is None:
            raise ValueError("A routed assembly needs z_t to compute the gate")
        return self.routing.gate(t, self.router(z_t, t), modules=self.router.modules_gated)

    def set_gate_for(self, t: torch.Tensor, z_t: torch.Tensor | None = None) -> None:
        """Refresh the shared gate in place (the containers hold the same dict)."""
        weights = self.gate_for(t, z_t)
        self.gate.clear()
        self.gate.update(weights)

    def clear(self) -> None:
        self.gate.clear()

    def predict(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, cond: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """One denoiser call with the gate set for ``timesteps`` (left set for backward)."""
        self.set_gate_for(timesteps, noisy)
        return self.adapter.predict(noisy, timesteps, cond)

    def freeze_experts(self) -> None:
        for name, param in self.unet.named_parameters():
            if "lora_" in name:
                param.requires_grad_(False)

    def detach(self) -> None:
        """Unbind the gate from the containers (e.g. before reusing the UNet)."""
        self.gate.clear()
        clear_gate(self.unet)


def build_assembly(
    adapter: DenoiserAdapter,
    cfg: CatsmConfig,
    *,
    experts_dir: str | Path,
    sets: Sequence[str] | None = None,
    core: str | None = None,
    with_router: bool = True,
    disabled: Sequence[str] = (),
) -> TSMAssembly:
    """Load ``<experts_dir>/<expert>/lora-final.safetensors`` for every expert of ``sets``.

    ``disabled`` experts are still loaded (the bundle stays complete and
    re-loadable) but routed with weight 0 everywhere — control C2.
    """
    sets = list(sets) if sets is not None else list(cfg.assemble.sets)
    core = core if core is not None else cfg.assemble.core
    if core not in sets:
        raise ValueError(f"core={core!r} is not one of {sets}")
    T = adapter.num_train_timesteps
    schemes = {name: build_scheme(name, T, cfg.intervals) for name in sets}
    experts_dir = Path(experts_dir)
    infos: list[ExpertInfo] = []
    for name in sorted(schemes):
        scheme = schemes[name]
        for i, interval in enumerate(scheme.intervals):
            expert = scheme.expert_name(i)
            path = experts_dir / expert / "lora-final.safetensors"
            if not path.exists():
                raise FileNotFoundError(f"Expert {expert} has not been fostered: {path} is missing")
            infos.append(load_expert_into(adapter.unet, path, name=expert, expect_interval=interval))
    context = [name for name in sorted(schemes) if name != core]
    router = None
    if with_router and context:
        context_experts = [i.name for i in infos if i.set != core and i.name not in set(disabled)]
        modules = gated_modules(adapter.unet, context_experts)
        if not modules:
            raise ValueError(
                f"disabled={sorted(disabled)} leaves no active context expert; use with_router=False"
            )
        router = Router(
            context,
            modules,
            latent_channels=adapter.latent_shape(1, adapter.resolution)[1],
            num_train_timesteps=T,
            init_gate=cfg.assemble.init_gate,
        )
    if disabled:
        logger.info("Disabled expert(s) %s: routed with weight 0 at every t", sorted(disabled))
    return TSMAssembly(
        adapter, schemes=schemes, core=core, experts=infos, router=router, disabled=disabled
    )


# --------------------------------------------------------------------------------------
# Router training
# --------------------------------------------------------------------------------------


def train_router(
    assembly: TSMAssembly,
    dataset: Sequence[dict],
    cfg: AssembleConfig,
    *,
    output_dir: str | Path,
    callbacks: Sequence[StepCallback] = (),
) -> TrainingState:
    """Train the router on ``L_Diff`` over all timesteps with every expert frozen."""
    from catsm.bundle import save_bundle

    if assembly.router is None:
        raise ValueError("This assembly has no router (single set or --no-router); nothing to train")
    adapter = assembly.adapter
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    autocast_dtype, scaler = resolve_mixed_precision(cfg.mixed_precision, adapter.device)

    unet = adapter.unet
    unet.requires_grad_(False)
    assembly.freeze_experts()
    if lora_parameters(unet):
        raise RuntimeError("Experts must be frozen during assembling")
    router = assembly.router.to(adapter.device)
    params = [p for p in router.parameters() if p.requires_grad]
    logger.info(
        "Router over context sets %s (core %s), %d gated module(s): %s trainable parameters",
        assembly.context_sets, assembly.core, router.num_modules,
        f"{sum(p.numel() for p in params):,}",
    )
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(cfg))

    device, dtype = adapter.device, adapter.dtype
    use_hflip = bool(getattr(dataset, "hflip", False))
    cache = _InputCache(adapter)
    if cfg.cache_latents and _is_deterministic(dataset):
        cache.precompute_latents(dataset, with_hflip=use_hflip)
    if device.type == "cuda":
        _park_text_encoders(adapter, cache, dataset)
    reads_dataset = cache.latents is None or cache.captions is None
    prefetcher = (
        _ItemPrefetcher(dataset, _micro_batch_plan(len(dataset), cfg))
        if reads_dataset and cfg.prefetch
        else None
    )

    unet.train()
    router.train()
    state = TrainingState(step=0, loss=float("nan"), learning_rate=cfg.learning_rate,
                          elapsed=0.0, adapter=adapter)
    started = time.perf_counter()
    T = adapter.num_train_timesteps
    try:
        for step in range(1, cfg.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for micro in range(cfg.gradient_accumulation_steps):
                indices = _draw_indices(len(dataset), cfg, step, micro)
                flips = (
                    _draw_flips(len(indices), cfg, step, micro) if use_hflip
                    else [False] * len(indices)
                )
                if cache.latents is not None and cache.captions is not None:
                    pools = (cache.latents, cache.latents_flipped or cache.latents)
                    latents = torch.cat(
                        [pools[int(flip)][i] for i, flip in zip(indices, flips)], dim=0
                    )
                    captions = [cache.captions[i] for i in indices]
                else:
                    batch = (
                        prefetcher.next() if prefetcher is not None
                        else [dataset[i] for i in indices]
                    )
                    images = [
                        hflip(item["image"]) if flip else item["image"]
                        for item, flip in zip(batch, flips)
                    ]
                    latents = adapter.encode_images(torch.stack(images))
                    captions = [item["caption"] for item in batch]
                latents = latents.to(device=device, dtype=dtype)
                cond = {
                    k: v.to(device=device, dtype=dtype)
                    for k, v in cache.conditioning(captions).items()
                }
                noise = torch.randn(
                    latents.shape,
                    generator=derive_generator(cfg.seed, "train-noise", step, micro),
                    dtype=torch.float32,
                ).to(device=device, dtype=dtype)
                # The router must see every timestep: the full range, as in train_lora.
                timesteps = torch.randint(
                    0, T, (latents.shape[0],),
                    generator=derive_generator(cfg.seed, "train-t", step, micro),
                ).to(device)

                noisy = adapter.add_noise(latents, noise, timesteps)
                target = adapter.get_target(latents, noise, timesteps)
                with _autocast(device.type, autocast_dtype):
                    prediction = assembly.predict(noisy, timesteps, cond)
                    loss = (prediction.float() - target.float()).pow(2).mean()
                scaled = loss / cfg.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                # Only now: gradient checkpointing re-reads the gate during backward.
                assembly.clear()
                total_loss += float(loss.detach()) / cfg.gradient_accumulation_steps

            if scaler is not None:
                scaler.unscale_(optimizer)
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            state.step = step
            state.loss = total_loss
            state.learning_rate = float(scheduler.get_last_lr()[0])
            state.elapsed = time.perf_counter() - started
            if cfg.log_every and step % cfg.log_every == 0:
                state.history.append(state.to_dict())
                per_step = state.elapsed / step
                logger.info(
                    "router step %d/%d  loss=%.5f  lr=%.2e  (%.1fs, %.2fs/step, ETA %s)",
                    step, cfg.max_steps, state.loss, state.learning_rate, state.elapsed,
                    per_step, _format_duration(per_step * (cfg.max_steps - step)),
                )
            if cfg.checkpoint_every and step % cfg.checkpoint_every == 0 and step < cfg.max_steps:
                save_bundle(
                    assembly, output_dir / f"bundle-step{step:06d}.safetensors",
                    extra_metadata={"steps": str(step)},
                )
            for callback in callbacks:
                callback(state)
    finally:
        assembly.clear()
        if prefetcher is not None:
            prefetcher.close()

    router.eval()
    unet.eval()
    save_bundle(
        assembly, output_dir / "bundle.safetensors", extra_metadata={"steps": str(cfg.max_steps)}
    )
    dump_json(
        {
            "sets": assembly.sets,
            "core": assembly.core,
            "context_sets": assembly.context_sets,
            "disabled": assembly.disabled,
            "experts": [e.to_dict() for e in assembly.experts],
            "router": router.describe(),
            "final": state.to_dict(),
            "history": state.history,
        },
        output_dir / "assemble_summary.json",
    )
    return state


def experts_json(experts: Sequence[ExpertInfo]) -> str:
    return json.dumps([e.to_dict() for e in experts], sort_keys=True)
