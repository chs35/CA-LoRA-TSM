"""A thin abstraction over "a latent diffusion denoiser".

The CA engine only needs five things from a model: encode images to latents,
encode prompts to conditioning, add noise at a timestep, predict, and know what
the regression target is. Hiding SD/SDXL/toy differences behind this interface
is what lets every unit test run on CPU with no downloads while the exact same
measurement code runs against SDXL on an A100.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

#: Adapter-specific conditioning. ``prompt_embeds`` is always present; SDXL adds
#: ``text_embeds`` (pooled) and ``time_ids``.
Conditioning = dict[str, torch.Tensor]


class DenoiserAdapter(ABC):
    """Uniform interface over a latent diffusion denoiser."""

    #: Number of timesteps in the training schedule (1000 for SD/SDXL).
    num_train_timesteps: int = 1000
    #: ``"epsilon"``, ``"v_prediction"`` or ``"sample"``.
    prediction_type: str = "epsilon"
    #: Side length in pixels that this adapter samples and encodes at.
    resolution: int = 512

    # -- model access ------------------------------------------------------------------

    @property
    @abstractmethod
    def unet(self) -> nn.Module:
        """The module whose attention projections are measured / adapted."""

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    # -- encoding ----------------------------------------------------------------------

    @abstractmethod
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """``[B,3,H,W]`` in ``[-1,1]`` -> latents ``[B,C,h,w]`` (no grad)."""

    @abstractmethod
    def encode_prompts(self, prompts: list[str]) -> Conditioning:
        """Text -> conditioning tensors (no grad)."""

    @abstractmethod
    def predict(
        self, noisy_latents: torch.Tensor, timesteps: torch.Tensor, cond: Conditioning
    ) -> torch.Tensor:
        """Denoiser forward pass; gradients flow to the UNet parameters."""

    # -- diffusion schedule --------------------------------------------------------------

    @property
    @abstractmethod
    def alphas_cumprod(self) -> torch.Tensor:
        """``[num_train_timesteps]`` cumulative alphas on the model device."""

    def add_noise(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        r"""``x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon``."""
        alphas = self.alphas_cumprod.to(device=latents.device, dtype=torch.float32)
        a = alphas[timesteps].view(-1, *([1] * (latents.ndim - 1)))
        sqrt_a = a.sqrt().to(latents.dtype)
        sqrt_1ma = (1.0 - a).sqrt().to(latents.dtype)
        return sqrt_a * latents + sqrt_1ma * noise

    def get_target(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """The regression target of ``L_Diff`` for this model's parameterisation.

        The paper writes ``L_Diff = ||eps_theta - eps||^2``, which assumes
        epsilon-prediction (true for SD 1.x and SDXL). For a v-prediction
        checkpoint the analogous target is ``v``, and using ``eps`` there would
        make the CA denominator meaningless, so we follow the scheduler.
        """
        if self.prediction_type == "epsilon":
            return noise
        if self.prediction_type == "sample":
            return latents
        if self.prediction_type == "v_prediction":
            alphas = self.alphas_cumprod.to(device=latents.device, dtype=torch.float32)
            a = alphas[timesteps].view(-1, *([1] * (latents.ndim - 1)))
            sqrt_a = a.sqrt().to(latents.dtype)
            sqrt_1ma = (1.0 - a).sqrt().to(latents.dtype)
            return sqrt_a * noise - sqrt_1ma * latents
        raise ValueError(f"Unsupported prediction_type {self.prediction_type!r}")

    # -- sampling ---------------------------------------------------------------------------

    #: Name of the sampler used by :meth:`generate`, recorded in the generation
    #: manifest: two arms compared under different schedulers are not comparable,
    #: so the scheduler has to be part of the artefact, not of the run log only.
    scheduler_name: str = "unknown"

    def generate(
        self,
        prompts: Sequence[str],
        *,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        generator: torch.Generator | None = None,
    ) -> list[Any]:
        """Sample one PIL image per prompt.

        Whatever LoRA is currently attached to :attr:`unet` is what gets used —
        sampling deliberately goes through the same module the measurement and
        the training touched, rather than re-loading a clean pipeline.

        Implementations must be a deterministic function of ``generator``: the
        ablation compares arms image-for-image, so a run has to be repeatable.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot generate images")

    # -- shapes ---------------------------------------------------------------------------

    @abstractmethod
    def latent_shape(self, batch_size: int, resolution: int) -> tuple[int, ...]: ...

    # -- lifecycle -------------------------------------------------------------------------

    def freeze_non_unet(self) -> None:
        """Disable gradients everywhere except the UNet (default: nothing to do)."""

    def offload_encoders(self, *, include_vae: bool = True) -> None:
        """Move the encoders to CPU to free VRAM (default: nothing to do).

        ``include_vae=False`` is for the training loop: with ``data.crop="random"``
        the VAE runs once per step and has to stay resident, while the text
        encoders are done after the conditioning cache is warm.
        """

    def restore_encoders(self) -> None:
        """Undo :meth:`offload_encoders` (default: nothing to do)."""

    def enable_gradient_checkpointing(self) -> None:
        fn = getattr(self.unet, "enable_gradient_checkpointing", None)
        if callable(fn):
            fn()

    def describe(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "num_train_timesteps": self.num_train_timesteps,
            "prediction_type": self.prediction_type,
            "scheduler": self.scheduler_name,
        }
