"""Image loading and normalisation shared by every dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


#: ``center`` is deterministic and is what every *measurement* uses; ``random``
#: exists because a 2048x1024 Cityscapes frame centre-cropped to a square throws
#: away half of it, including the pavements and building facades that carry most
#: of the domain signal.
CROP_MODES = ("center", "random")


def load_image(
    path: str | Path,
    resolution: int,
    *,
    crop: str = "center",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Load an RGB image as ``[3,resolution,resolution]`` in ``[-1,1]``.

    Aspect-preserving resize on the short side, then a crop. ``crop="center"``
    keeps the middle of the frame — road surface, vehicles and the horizon.
    ``crop="random"`` slides that window horizontally, which is only sound for
    *training*: anything that pairs samples across timesteps or concept axes
    needs the same pixels every time and must stay on ``center``.

    Args:
        generator: Source of randomness for ``crop="random"``. Derive it with
            :func:`calora.utils.seeding.derive_generator` so the run stays
            reproducible; ``None`` falls back to the global torch RNG.
    """
    if crop not in CROP_MODES:
        raise ValueError(f"Unknown crop mode {crop!r}; expected one of {CROP_MODES}")
    with Image.open(path) as img:
        img = img.convert("RGB")
        resized = resize_short_side(img, resolution)
        if crop == "center":
            return to_tensor(center_crop(resized, resolution))
        return to_tensor(random_crop(resized, resolution, generator=generator))


def resize_short_side(img: Image.Image, size: int) -> Image.Image:
    width, height = img.size
    scale = size / min(width, height)
    new_size = (max(size, round(width * scale)), max(size, round(height * scale)))
    return img.resize(new_size, Image.BICUBIC)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    width, height = img.size
    left = (width - size) // 2
    top = (height - size) // 2
    return img.crop((left, top, left + size, top + size))


def random_crop(
    img: Image.Image, size: int, *, generator: torch.Generator | None = None
) -> Image.Image:
    """Crop a ``size x size`` window at a uniformly sampled horizontal offset.

    Only the horizontal position moves: after :func:`resize_short_side` a
    landscape frame has slack along the width alone, and that is exactly the
    part a centre crop discards. The vertical window stays centred so the
    horizon sits where the centre crop would have put it.
    """
    width, height = img.size
    span = max(width - size, 0)
    left = 0 if span == 0 else int(torch.randint(0, span + 1, (1,), generator=generator))
    top = (height - size) // 2
    return img.crop((left, top, left + size, top + size))


def hflip(image: torch.Tensor) -> torch.Tensor:
    """Mirror a ``[...,H,W]`` image tensor left-to-right.

    Applied to *pixels*, never to latents. ``VAE(flip(x)) != flip(VAE(x))``: the
    encoder is a stack of strided convolutions whose padding and downsampling
    grid are not mirror-symmetric, so flipping a latent produces something no
    real image maps to. Anything that caches latents therefore has to encode
    both orientations rather than flip the cached tensor.
    """
    return image.flip(-1)


def to_tensor(img: Image.Image) -> torch.Tensor:
    array = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def to_pil(tensor: torch.Tensor) -> Image.Image:
    """Inverse of :func:`to_tensor` for a single ``[3,H,W]`` image."""
    array = ((tensor.detach().float().cpu().permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255)
    return Image.fromarray(array.numpy().astype(np.uint8))
