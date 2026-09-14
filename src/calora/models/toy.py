"""An offline, CPU-sized stand-in for a latent diffusion model.

This exists so the whole pipeline — CA measurement, selection, t-scoring, LoRA
wiring, drift analysis — can be developed and unit-tested on a laptop with no
GPU and no Hugging Face downloads. It deliberately mirrors the diffusers module
naming (``down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q``) so that
:mod:`calora.models.projections` and ``peft`` target-module matching exercise the
same code paths they will on SDXL.

It is *not* a good generative model and is never used for real experiments.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from calora.models.adapter import Conditioning, DenoiserAdapter
from calora.utils.seeding import stable_hash


def _to(cond: Conditioning, device: torch.device, dtype: torch.dtype) -> Conditioning:
    return {k: v.to(device=device, dtype=dtype) for k, v in cond.items()}


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ToyAttention(nn.Module):
    """Matches the diffusers ``Attention`` parameter layout (incl. ``to_out.0``)."""

    def __init__(self, query_dim: int, context_dim: int | None = None, heads: int = 2):
        super().__init__()
        self.heads = heads
        kv_dim = context_dim if context_dim is not None else query_dim
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, query_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, query_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(query_dim, query_dim), nn.Dropout(0.0)])

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.view(b, n, self.heads, c // self.heads).transpose(1, 2)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        ctx = x if context is None else context
        q = self._split(self.to_q(x))
        k = self._split(self.to_k(ctx))
        v = self._split(self.to_v(ctx))
        out = F.scaled_dot_product_attention(q, k, v)
        b, h, n, d = out.shape
        out = out.transpose(1, 2).reshape(b, n, h * d)
        return self.to_out[1](self.to_out[0](out))


class ToyTransformerBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = ToyAttention(dim, None, heads)  # self-attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = ToyAttention(dim, context_dim, heads)  # cross-attention
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        return x + self.ff(self.norm3(x))


class ToySpatialTransformer(nn.Module):
    """Stand-in for ``Transformer2DModel``: NCHW in, NCHW out."""

    def __init__(self, channels: int, context_dim: int, depth: int = 1, heads: int = 2):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = nn.ModuleList(
            [ToyTransformerBlock(channels, context_dim, heads) for _ in range(depth)]
        )
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, h, w = x.shape
        y = self.norm(x).permute(0, 2, 3, 1).reshape(b, h * w, c)
        y = self.proj_in(y)
        for block in self.transformer_blocks:
            y = block(y, context)
        y = self.proj_out(y)
        return residual + y.reshape(b, h, w, c).permute(0, 3, 1, 2)


class ToyBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        context_dim: int,
        num_attentions: int = 1,
        depth: int = 1,
        heads: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.attentions = nn.ModuleList(
            [
                ToySpatialTransformer(out_channels, context_dim, depth, heads)
                for _ in range(num_attentions)
            ]
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = F.silu(self.norm(h))
        for attn in self.attentions:
            h = attn(h, context)
        return h


class ToyUNet(nn.Module):
    """A tiny UNet with diffusers-compatible attention module names."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 32,
        context_dim: int = 32,
        time_dim: int = 32,
        num_attentions: int = 1,
        depth: int = 1,
        heads: int = 2,
    ):
        super().__init__()
        c1, c2 = base_channels, base_channels * 2
        self.context_dim = context_dim
        self.in_channels = in_channels
        self.time_dim = time_dim

        self.time_embedding = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.down_blocks = nn.ModuleList(
            [
                ToyBlock(c1, c1, time_dim, context_dim, num_attentions, depth, heads),
                ToyBlock(c1, c2, time_dim, context_dim, num_attentions, depth, heads),
            ]
        )
        self.downsamplers = nn.ModuleList([nn.Conv2d(c1, c1, 3, stride=2, padding=1)])
        self.mid_block = ToyBlock(c2, c2, time_dim, context_dim, num_attentions, depth, heads)
        self.up_blocks = nn.ModuleList(
            [
                ToyBlock(c2 + c2, c2, time_dim, context_dim, num_attentions, depth, heads),
                ToyBlock(c2 + c1, c1, time_dim, context_dim, num_attentions, depth, heads),
            ]
        )
        self.norm_out = nn.GroupNorm(min(8, c1), c1)
        self.conv_out = nn.Conv2d(c1, out_channels, 3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **_ignored,
    ) -> torch.Tensor:
        if timestep.ndim == 0:
            timestep = timestep.expand(sample.shape[0])
        t_emb = self.time_embedding(timestep_embedding(timestep, self.time_dim).to(sample.dtype))

        h = self.conv_in(sample)
        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.down_blocks):
            h = block(h, t_emb, encoder_hidden_states)
            skips.append(h)
            if i < len(self.downsamplers):
                h = self.downsamplers[i](h)

        h = self.mid_block(h, t_emb, encoder_hidden_states)

        for block in self.up_blocks:
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), t_emb, encoder_hidden_states)

        return self.conv_out(F.silu(self.norm_out(h)))


class ToyAdapter(DenoiserAdapter):
    """A fully offline :class:`DenoiserAdapter`.

    The "VAE" is a fixed random strided conv and the "text encoder" hashes the
    prompt string into a deterministic embedding. Both are frozen: the point is
    that identical prompts give identical conditioning and different prompts
    give different conditioning, which is all the CA measurement needs.
    """

    #: :meth:`generate` runs a plain deterministic DDIM (eta=0), not a diffusers
    #: scheduler. Recorded in the manifest like any other sampler so a toy run is
    #: never mistaken for a real one.
    scheduler_name = "ToyDDIM"

    def __init__(
        self,
        *,
        base_channels: int = 32,
        context_dim: int = 32,
        latent_channels: int = 4,
        vae_scale_factor: int = 8,
        text_seq_len: int = 8,
        num_train_timesteps: int = 1000,
        prediction_type: str = "epsilon",
        resolution: int = 64,
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        num_attentions: int = 1,
        depth: int = 1,
        heads: int = 2,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.resolution = resolution
        self._device = torch.device(device)
        self._dtype = dtype
        self._latent_channels = latent_channels
        self._vae_scale_factor = vae_scale_factor
        self._text_seq_len = text_seq_len
        self._context_dim = context_dim
        self._seed = seed

        generator = torch.Generator().manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self._unet = ToyUNet(
                in_channels=latent_channels,
                out_channels=latent_channels,
                base_channels=base_channels,
                context_dim=context_dim,
                num_attentions=num_attentions,
                depth=depth,
                heads=heads,
            )
        self._unet.to(device=self._device, dtype=self._dtype).eval()

        # Frozen "VAE": a fixed random strided convolution.
        self._vae_weight = (
            torch.randn(
                latent_channels, 3, vae_scale_factor, vae_scale_factor, generator=generator
            )
            / (3 * vae_scale_factor**2) ** 0.5
        ).to(device=self._device, dtype=self._dtype)

        betas = torch.linspace(1e-4, 0.02, num_train_timesteps, dtype=torch.float32)
        self._alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).to(self._device)

    # -- model access ------------------------------------------------------------------

    @property
    def unet(self) -> nn.Module:
        return self._unet

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        return self._alphas_cumprod

    # -- encoding ----------------------------------------------------------------------

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(device=self._device, dtype=self._dtype)
        return F.conv2d(images, self._vae_weight, stride=self._vae_scale_factor)

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]) -> Conditioning:
        embeds = []
        for prompt in prompts:
            generator = torch.Generator().manual_seed(stable_hash("toy-text", prompt) % (2**31 - 1))
            embeds.append(
                torch.randn(self._text_seq_len, self._context_dim, generator=generator)
            )
        stacked = torch.stack(embeds).to(device=self._device, dtype=self._dtype)
        return {"prompt_embeds": stacked}

    def predict(
        self, noisy_latents: torch.Tensor, timesteps: torch.Tensor, cond: Conditioning
    ) -> torch.Tensor:
        return self._unet(noisy_latents, timesteps, cond["prompt_embeds"])

    # -- sampling -----------------------------------------------------------------------

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """``[B,C,h,w]`` -> ``[B,3,H,W]`` in ``[-1,1]``: the encoder run backwards.

        The "VAE" is a single strided convolution, so its adjoint (a transposed
        convolution with the same kernel) is the natural stand-in for a decoder.
        It is not an inverse and the output is not a photograph of anything; the
        contrast normalisation below only exists so the PNGs are not a uniform
        grey field that every metric would score identically. Per-image and
        deterministic, so equal seeds still give equal pixels.
        """
        decoded = F.conv_transpose2d(
            latents.to(device=self._device, dtype=self._dtype),
            self._vae_weight,
            stride=self._vae_scale_factor,
        )
        peak = decoded.flatten(1).abs().amax(dim=1).clamp(min=1e-6)
        return (decoded / peak.view(-1, 1, 1, 1)).clamp(-1.0, 1.0)

    @torch.no_grad()
    def generate(
        self,
        prompts,
        *,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        generator: torch.Generator | None = None,
    ) -> list:
        """Deterministic DDIM (eta=0) with classifier-free guidance, then decode.

        Mirrors the real samplers structurally — same guidance formula, same
        eps-to-x0 algebra, same LoRA-adapted UNet — so the generation script has
        one code path rather than a toy branch. The images are meaningless; what
        the CPU tests check is that the same seed reproduces the same pixels and
        that a different seed does not.
        """
        from calora.data.transforms import to_pil

        prompts = list(prompts)
        cond = _to(self.encode_prompts(prompts), self._device, self._dtype)
        uncond = _to(self.encode_prompts([""] * len(prompts)), self._device, self._dtype)

        shape = self.latent_shape(len(prompts), self.resolution)
        latents = torch.randn(shape, generator=generator, dtype=torch.float32).to(
            device=self._device, dtype=self._dtype
        )

        # Evenly spaced, strictly descending; the last step lands on t=0.
        step = max(self.num_train_timesteps // num_inference_steps, 1)
        schedule = list(range(self.num_train_timesteps - 1, -1, -step))[:num_inference_steps]
        alphas = self._alphas_cumprod.to(torch.float32)

        for position, timestep in enumerate(schedule):
            batched = torch.full((len(prompts),), timestep, device=self._device, dtype=torch.long)
            eps = self.predict(latents, batched, cond)
            if guidance_scale != 1.0:
                eps_uncond = self.predict(latents, batched, uncond)
                eps = eps_uncond + guidance_scale * (eps - eps_uncond)

            alpha = alphas[timestep]
            previous = schedule[position + 1] if position + 1 < len(schedule) else None
            alpha_prev = alphas[previous] if previous is not None else torch.ones_like(alpha)
            # x0 = (x_t - sqrt(1-a_t) eps) / sqrt(a_t), then re-noise to t-1.
            predicted_x0 = (latents - (1 - alpha).sqrt() * eps) / alpha.sqrt().clamp(min=1e-8)
            latents = alpha_prev.sqrt() * predicted_x0 + (1 - alpha_prev).sqrt() * eps

        return [to_pil(image) for image in self.decode_latents(latents)]

    # -- shapes -------------------------------------------------------------------------

    def latent_shape(self, batch_size: int, resolution: int) -> tuple[int, ...]:
        side = resolution // self._vae_scale_factor
        return (batch_size, self._latent_channels, side, side)
