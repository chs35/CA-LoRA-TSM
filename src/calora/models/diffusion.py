"""diffusers-backed adapters for Stable Diffusion 1.x/2.x and SDXL.

Imported lazily by :mod:`calora.models.loader`; ``diffusers`` is an optional
dependency so that the toy path stays importable on a machine without it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from calora.models.adapter import Conditioning, DenoiserAdapter
from calora.utils.logging import get_logger

logger = get_logger(__name__)


def _unwrap(output: Any) -> torch.Tensor:
    return output.sample if hasattr(output, "sample") else output


class _DiffusersAdapterBase(DenoiserAdapter):
    def __init__(self, pipe: Any, *, device: torch.device, dtype: torch.dtype, resolution: int):
        self._pipe = pipe
        self._device = device
        self._dtype = dtype
        self.resolution = resolution

        scheduler = pipe.scheduler
        self.num_train_timesteps = int(scheduler.config.num_train_timesteps)
        self.prediction_type = str(getattr(scheduler.config, "prediction_type", "epsilon"))
        self._alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
        self._vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        self._latent_channels = int(pipe.vae.config.latent_channels)
        self._scaling_factor = float(pipe.vae.config.scaling_factor)
        self.scheduler_name = type(scheduler).__name__

        self.freeze_non_unet()

    # -- model access ------------------------------------------------------------------

    @property
    def pipe(self) -> Any:
        return self._pipe

    @property
    def unet(self) -> nn.Module:
        return self._pipe.unet

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        return self._alphas_cumprod

    # -- lifecycle ---------------------------------------------------------------------

    def freeze_non_unet(self) -> None:
        for attr in ("vae", "text_encoder", "text_encoder_2"):
            module = getattr(self._pipe, attr, None)
            if isinstance(module, nn.Module):
                module.requires_grad_(False)
                module.eval()

    #: Everything :meth:`offload_encoders` can park, VAE first.
    ENCODER_ATTRS = ("vae", "text_encoder", "text_encoder_2")

    def offload_encoders(self, *, include_vae: bool = True) -> None:
        """Move the VAE and text encoders to CPU.

        Call this after :func:`~calora.ca.measure.prepare_samples`: the
        measurement loop only touches the UNet, so ~2 GB of SDXL encoder weights
        can leave the GPU before the gradient accumulators are allocated.

        ``include_vae=False`` is the training-loop case: the captions come from a
        fixed template so the text encoders (~3.3 GB fp32 for SDXL) are finished
        once the conditioning cache is warm, but the VAE still encodes one image
        per step whenever ``data.crop="random"`` defeats the latent cache.
        """
        attrs = self.ENCODER_ATTRS if include_vae else self.ENCODER_ATTRS[1:]
        for attr in attrs:
            module = getattr(self._pipe, attr, None)
            if isinstance(module, nn.Module):
                module.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def restore_encoders(self) -> None:
        """Bring every offloaded encoder back onto the compute device."""
        for attr in self.ENCODER_ATTRS:
            module = getattr(self._pipe, attr, None)
            if isinstance(module, nn.Module):
                module.to(self._device)

    # -- encoding ----------------------------------------------------------------------

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        vae = self._pipe.vae
        images = images.to(device=self._device, dtype=vae.dtype)
        latent_dist = vae.encode(images).latent_dist
        # Deterministic: the mode, not a sample, so repeated measurements of the
        # same image are identical across timesteps and concept axes.
        latents = latent_dist.mode() * self._scaling_factor
        return latents.to(self._dtype)

    # -- sampling ----------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompts,
        *,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        generator: torch.Tensor | None = None,
    ) -> list:
        """Sample through the already-loaded pipeline — nothing is re-downloaded.

        ``self._pipe`` holds the very UNet that :func:`calora.train.lora.attach_lora`
        injected into, so the LoRA under evaluation is active by construction; a
        freshly constructed pipeline would silently sample the base model.

        The scheduler is whatever the checkpoint shipped with (recorded as
        :attr:`scheduler_name`): swapping in a "better" sampler would improve
        every arm's numbers by an amount that has nothing to do with the
        projection choice under test.
        """
        prompts = list(prompts)
        # prepare_samples may have pushed the encoders to CPU to make room for the
        # gradient accumulators; sampling needs them back.
        self._pipe.to(self._device)
        was_training = self.unet.training
        self.unet.eval()
        try:
            output = self._pipe(
                prompt=prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=self.resolution,
                width=self.resolution,
                output_type="pil",
            )
        finally:
            if was_training:
                self.unet.train()
        return list(output.images)

    # -- shapes -------------------------------------------------------------------------

    def latent_shape(self, batch_size: int, resolution: int) -> tuple[int, ...]:
        side = resolution // self._vae_scale_factor
        return (batch_size, self._latent_channels, side, side)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "pipeline": type(self._pipe).__name__,
                "vae_scale_factor": self._vae_scale_factor,
                "latent_channels": self._latent_channels,
                "scaling_factor": self._scaling_factor,
                "resolution": self.resolution,
            }
        )
        return info


class StableDiffusionAdapter(_DiffusersAdapterBase):
    """SD 1.x / 2.x: a single CLIP text encoder, no extra conditioning."""

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]) -> Conditioning:
        prompt_embeds, _negative = self._pipe.encode_prompt(
            prompt=prompts,
            device=self._device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return {"prompt_embeds": prompt_embeds.to(self._dtype)}

    def predict(
        self, noisy_latents: torch.Tensor, timesteps: torch.Tensor, cond: Conditioning
    ) -> torch.Tensor:
        return _unwrap(
            self.unet(noisy_latents, timesteps, encoder_hidden_states=cond["prompt_embeds"])
        )


class StableDiffusionXLAdapter(_DiffusersAdapterBase):
    """SDXL: dual text encoders plus pooled embeddings and micro-conditioning ids."""

    def __init__(self, pipe: Any, **kwargs):
        super().__init__(pipe, **kwargs)
        self._time_ids = torch.tensor(
            [self.resolution, self.resolution, 0, 0, self.resolution, self.resolution],
            dtype=torch.float32,
            device=self._device,
        )

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]) -> Conditioning:
        prompt_embeds, _neg, pooled, _neg_pooled = self._pipe.encode_prompt(
            prompt=prompts,
            device=self._device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return {
            "prompt_embeds": prompt_embeds.to(self._dtype),
            "text_embeds": pooled.to(self._dtype),
        }

    def predict(
        self, noisy_latents: torch.Tensor, timesteps: torch.Tensor, cond: Conditioning
    ) -> torch.Tensor:
        batch = noisy_latents.shape[0]
        added = {
            "text_embeds": cond["text_embeds"],
            "time_ids": self._time_ids.to(self._dtype).expand(batch, -1),
        }
        return _unwrap(
            self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=cond["prompt_embeds"],
                added_cond_kwargs=added,
            )
        )
