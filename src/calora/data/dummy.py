"""A synthetic stand-in for Cityscapes, for CPU tests and smoke runs.

Images are deterministic low-frequency gradients plus noise rather than pure
``randn``: a VAE (real or toy) maps flat noise to near-degenerate latents, which
would make gradient norms unrepresentative and hide bugs in the CA maths.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from calora.data.prompts import PromptAugmenter
from calora.utils.seeding import derive_generator

FAKE_CITIES = (
    "aachen",
    "bochum",
    "bremen",
    "cologne",
    "darmstadt",
    "erfurt",
    "hamburg",
    "jena",
)

#: Stand-in for the class names read off a real ``gtFine`` label map, so the
#: generation script's "... with [Class names]" prompt path is exercised on CPU
#: without Cityscapes. A subset of the real 19, kept in trainId order.
DUMMY_CLASS_NAMES = (
    "road",
    "sidewalk",
    "building",
    "pole",
    "vegetation",
    "sky",
    "person",
    "car",
)


def synthetic_image(index: int, resolution: int, seed: int = 0) -> torch.Tensor:
    """A deterministic ``[3,R,R]`` image in ``[-1,1]`` with real spatial structure."""
    generator = derive_generator(seed, "dummy-image", index)
    coords = torch.linspace(-1.0, 1.0, resolution)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

    channels = []
    for channel in range(3):
        freq = 1.0 + 1.7 * channel + (index % 4)
        phase = float(torch.rand(1, generator=generator)) * 2 * math.pi
        base = torch.sin(freq * math.pi * grid_x + phase) * torch.cos(
            (freq * 0.7) * math.pi * grid_y - phase
        )
        noise = torch.randn(resolution, resolution, generator=generator) * 0.15
        channels.append((0.85 * base + noise).clamp(-1.0, 1.0))
    return torch.stack(channels)


class DummyConceptDataset(Sequence):
    """Same item schema as :class:`~calora.data.cityscapes.CityscapesConceptDataset`."""

    #: Synthetic images are generated at the requested resolution, so nothing is
    #: ever cropped. Declared anyway so consumers can ask any dataset whether
    #: ``__getitem__`` is deterministic without special-casing the class.
    crop = "center"

    def __init__(
        self,
        augmenter: PromptAugmenter,
        *,
        num_items: int = 8,
        resolution: int = 64,
        caption_template: str = "a photo of a city street scene in {city}",
        hflip: bool = False,
        augs_per_sample: int = 1,
        seed: int = 0,
    ):
        self.augmenter = augmenter
        self.num_items = num_items
        self.resolution = resolution
        self.caption_template = caption_template
        self.hflip = hflip
        self.augs_per_sample = augs_per_sample
        self.seed = seed

    def __len__(self) -> int:
        return self.num_items

    def caption_for_index(self, index: int) -> str:
        city = FAKE_CITIES[index % len(FAKE_CITIES)]
        return self.caption_template.format(city=city, key=f"dummy_{index:06d}")

    def distinct_captions(self) -> list[str]:
        """Every caption, without synthesising an image — same contract as
        :class:`~calora.data.cityscapes.CityscapesConceptDataset`."""
        return sorted({self.caption_for_index(i) for i in range(self.num_items)})

    def __getitem__(self, index: int) -> dict:  # type: ignore[override]
        if not 0 <= index < self.num_items:
            raise IndexError(index)
        city = FAKE_CITIES[index % len(FAKE_CITIES)]
        caption = self.caption_for_index(index)
        return {
            "index": index,
            "image": synthetic_image(index, self.resolution, self.seed),
            "caption": caption,
            "augmented": self.augmenter.augment_many(caption, index, self.augs_per_sample),
            "city": city,
            "key": f"dummy_{index:06d}",
            "image_path": None,
            "label_path": None,
        }
