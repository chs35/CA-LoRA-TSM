"""Fostering stage: train one sliced LoRA expert on one timestep interval.

This is :func:`calora.train.lora.train_lora` with a single substantive change:
the training timestep is drawn from the expert's interval ``[t_lo, t_hi]``
instead of ``[0, T)``, and every draw is checked against it. Everything else —
the sliced type A/B adapter, the input cache, the seed derivation, the
optimiser, the checkpoint format — is the reference implementation, reused.
With the full interval ``[0, T-1]`` the draw is
``torch.randint(0, T, ...)`` under the same generator, so the n=1 expert is
bit-identical to a reference ``train_lora`` run (pinned by tests): No.1 in the
draft's matrix is a faithful same-step baseline.

The expert file keeps the single-adapter format (``<module>.lora_A`` …), so it
loads with ``load_lora`` on its own and the Assembling stage renames its keys
into the multi-expert containers.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from calora.ca.select import Selection
from calora.config import TrainConfig
from calora.data.transforms import hflip
from calora.models.adapter import DenoiserAdapter
from calora.train.lora import (
    StepCallback,
    TrainingState,
    _autocast,
    _draw_flips,
    _draw_indices,
    _format_duration,
    _InputCache,
    _is_deterministic,
    _ItemPrefetcher,
    _micro_batch_plan,
    _park_text_encoders,
    attach_lora,
    build_lr_lambda,
    lora_parameters,
    resolve_mixed_precision,
    save_lora,
)
from calora.utils.io import dump_json
from calora.utils.logging import get_logger
from calora.utils.seeding import derive_generator, seed_everything
from catsm.intervals import Interval, logsnr
from catsm.models.multi_sliced_lora import validate_expert_name

logger = get_logger(__name__)


@dataclass
class FosterState(TrainingState):
    """Training state plus the timestep range actually visited."""

    interval: Interval | None = None
    t_min_seen: int | None = None
    t_max_seen: int | None = None
    num_timesteps_drawn: int = 0

    def to_dict(self) -> dict:
        out = super().to_dict()
        out.update(
            {
                "t_lo": self.interval.lo if self.interval else None,
                "t_hi": self.interval.hi if self.interval else None,
                "t_min_seen": self.t_min_seen,
                "t_max_seen": self.t_max_seen,
                "num_timesteps_drawn": self.num_timesteps_drawn,
            }
        )
        return out

    def observe(self, timesteps: torch.Tensor) -> None:
        lo, hi = check_timesteps(timesteps, self.interval)
        self.t_min_seen = lo if self.t_min_seen is None else min(self.t_min_seen, lo)
        self.t_max_seen = hi if self.t_max_seen is None else max(self.t_max_seen, hi)
        self.num_timesteps_drawn += int(timesteps.numel())


def sample_timesteps(
    interval: Interval, batch: int, *, seed: int, step: int, micro: int
) -> torch.Tensor:
    """``batch`` timesteps uniform on ``[lo, hi]`` from the reference ``"train-t"`` stream."""
    return torch.randint(
        interval.lo,
        interval.hi + 1,
        (batch,),
        generator=derive_generator(seed, "train-t", step, micro),
    )


def check_timesteps(timesteps: torch.Tensor, interval: Interval | None) -> tuple[int, int]:
    """Raise unless every drawn timestep lies in ``interval``; return (min, max)."""
    if timesteps.numel() == 0:
        raise RuntimeError("Empty timestep batch")
    lo, hi = int(timesteps.min()), int(timesteps.max())
    if interval is not None and (lo < interval.lo or hi > interval.hi):
        raise RuntimeError(
            f"Timestep draw [{lo}, {hi}] leaves the expert's interval "
            f"[{interval.lo}, {interval.hi}]; the fostering stage must never train "
            "outside it"
        )
    return lo, hi


def _interval_metadata(
    adapter: DenoiserAdapter, interval: Interval, expert: str, state: FosterState | None
) -> dict[str, str]:
    meta = {
        "expert": expert,
        "t_lo": str(interval.lo),
        "t_hi": str(interval.hi),
        "num_train_timesteps": str(adapter.num_train_timesteps),
        "logsnr_lo": repr(logsnr(adapter.alphas_cumprod, interval.lo)),
        "logsnr_hi": repr(logsnr(adapter.alphas_cumprod, interval.hi)),
    }
    if state is not None and state.t_min_seen is not None:
        meta["t_min_seen"] = str(state.t_min_seen)
        meta["t_max_seen"] = str(state.t_max_seen)
    return meta


def train_expert(
    adapter: DenoiserAdapter,
    dataset: Sequence[dict],
    selection: Selection,
    cfg: TrainConfig,
    *,
    interval: Interval,
    expert: str,
    output_dir: str | Path,
    metadata: Mapping[str, str] | None = None,
    callbacks: Sequence[StepCallback] = (),
) -> FosterState:
    """Fine-tune ``selection``'s heads on ``L_Diff`` at timesteps drawn from ``interval``.

    ``metadata`` (e.g. ``ca_timesteps``, ``select_mode``, ``selection_sha``,
    ``interval_scheme``) is written into every checkpoint next to the interval
    itself, so an expert file states on its own where it was trained.
    """
    validate_expert_name(expert)
    if selection.granularity != "head":
        raise ValueError("Interval experts are sliced LoRAs; the selection must be head-granularity")
    if interval.hi >= adapter.num_train_timesteps:
        raise ValueError(
            f"Interval [{interval.lo}, {interval.hi}] exceeds the model's "
            f"{adapter.num_train_timesteps} training timesteps"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    autocast_dtype, scaler = resolve_mixed_precision(cfg.mixed_precision, adapter.device)

    unet = adapter.unet
    unet.requires_grad_(False)
    attach_lora(unet, selection, cfg.lora)
    params = lora_parameters(unet)
    if not params:
        raise RuntimeError("No trainable LoRA parameters after injection")
    logger.info(
        "Expert %s: interval [%d, %d] (%d timesteps), %s trainable LoRA parameters",
        expert, interval.lo, interval.hi, interval.length, f"{sum(p.numel() for p in params):,}",
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
    if cfg.cache_latents:
        if _is_deterministic(dataset):
            cache.precompute_latents(dataset, with_hflip=use_hflip)
        else:
            logger.warning(
                "train.cache_latents=true ignored: the dataset uses a random crop; "
                "falling back to per-step VAE encoding."
            )
    if device.type == "cuda":
        _park_text_encoders(adapter, cache, dataset)
    reads_dataset = cache.latents is None or cache.captions is None
    prefetcher = (
        _ItemPrefetcher(dataset, _micro_batch_plan(len(dataset), cfg))
        if reads_dataset and cfg.prefetch
        else None
    )

    base_meta = {"strategy": selection.strategy}
    axis = str(selection.meta.get("axis", "") or "")
    if axis:
        base_meta["axis"] = axis
    base_meta.update({k: str(v) for k, v in (metadata or {}).items()})

    unet.train()
    state = FosterState(
        step=0, loss=float("nan"), learning_rate=cfg.learning_rate, elapsed=0.0,
        adapter=adapter, interval=interval,
    )
    started = time.perf_counter()

    def checkpoint_metadata(step: int) -> dict[str, str]:
        return {**base_meta, **_interval_metadata(adapter, interval, expert, state), "steps": str(step)}

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
                # The one line that makes this the fostering stage.
                timesteps = sample_timesteps(
                    interval, latents.shape[0], seed=cfg.seed, step=step, micro=micro
                )
                state.observe(timesteps)
                timesteps = timesteps.to(device)

                noisy = adapter.add_noise(latents, noise, timesteps)
                target = adapter.get_target(latents, noise, timesteps)
                with _autocast(device.type, autocast_dtype):
                    prediction = adapter.predict(noisy, timesteps, cond)
                    loss = (prediction.float() - target.float()).pow(2).mean()

                scaled = loss / cfg.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
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
                    "%s step %d/%d  loss=%.5f  lr=%.2e  t∈[%d,%d]  (%.1fs, %.2fs/step, ETA %s)",
                    expert, step, cfg.max_steps, state.loss, state.learning_rate,
                    state.t_min_seen, state.t_max_seen, state.elapsed, per_step,
                    _format_duration(per_step * (cfg.max_steps - step)),
                )
            if cfg.checkpoint_every and step % cfg.checkpoint_every == 0 and step < cfg.max_steps:
                save_lora(
                    unet,
                    output_dir / f"lora-step{step:06d}.safetensors",
                    metadata=checkpoint_metadata(step),
                    cfg=cfg.lora,
                )
            for callback in callbacks:
                callback(state)
    finally:
        if prefetcher is not None:
            prefetcher.close()

    save_lora(
        unet,
        output_dir / "lora-final.safetensors",
        metadata=checkpoint_metadata(cfg.max_steps),
        cfg=cfg.lora,
    )
    unet.eval()
    dump_json(
        {
            "expert": expert,
            "interval": interval.as_list(),
            "metadata": checkpoint_metadata(cfg.max_steps),
            "final": state.to_dict(),
            "history": state.history,
        },
        output_dir / "train_summary.json",
    )
    return state
