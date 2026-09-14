"""Build a :class:`~calora.models.adapter.DenoiserAdapter` from a config."""

from __future__ import annotations

import torch

from calora.config import ModelConfig
from calora.models.adapter import DenoiserAdapter
from calora.utils.device import resolve_device, resolve_dtype
from calora.utils.logging import get_logger

logger = get_logger(__name__)


def load_adapter(cfg: ModelConfig) -> DenoiserAdapter:
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)

    if cfg.kind == "toy":
        from calora.models.toy import ToyAdapter

        logger.info("Building toy adapter on %s (%s)", device, dtype)
        # model.resolution is what every other adapter samples at, so the toy one
        # follows it too unless model.toy overrides it explicitly.
        return ToyAdapter(device=device, dtype=dtype, **{"resolution": cfg.resolution, **cfg.toy})

    if cfg.kind not in {"sd", "sdxl"}:
        raise ValueError(f"Unknown model.kind {cfg.kind!r}; expected 'toy', 'sd' or 'sdxl'")
    if not cfg.name:
        raise ValueError("model.name is required for kind 'sd'/'sdxl'")

    try:
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "diffusers is required for model.kind='sd'/'sdxl'. "
            "Install with: pip install 'calora[diffusion]'"
        ) from exc

    from calora.models.diffusion import StableDiffusionAdapter, StableDiffusionXLAdapter

    pipeline_cls = StableDiffusionXLPipeline if cfg.kind == "sdxl" else StableDiffusionPipeline
    adapter_cls = StableDiffusionXLAdapter if cfg.kind == "sdxl" else StableDiffusionAdapter

    kwargs: dict[str, object] = {"torch_dtype": dtype, "safety_checker": None}
    if cfg.kind == "sdxl":
        kwargs.pop("safety_checker")
    if cfg.revision:
        kwargs["revision"] = cfg.revision
    if cfg.variant:
        kwargs["variant"] = cfg.variant

    logger.info("Loading %s from %s (%s, %s)", pipeline_cls.__name__, cfg.name, device, dtype)
    pipe = pipeline_cls.from_pretrained(cfg.name, **kwargs)
    pipe.to(device)

    vae_dtype = resolve_dtype(cfg.vae_dtype)
    if vae_dtype != dtype:
        pipe.vae.to(dtype=vae_dtype)

    pipe.unet.eval()
    pipe.unet.requires_grad_(False)

    if cfg.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("xformers unavailable (%s); continuing without it", exc)

    adapter = adapter_cls(pipe, device=device, dtype=dtype, resolution=cfg.resolution)
    if cfg.enable_gradient_checkpointing:
        adapter.enable_gradient_checkpointing()
    logger.info("Adapter ready: %s", adapter.describe())
    return adapter


def maybe_free(adapter: DenoiserAdapter) -> None:
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
