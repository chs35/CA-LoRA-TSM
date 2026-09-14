"""Datasets and concept-axis prompt augmentation."""

from __future__ import annotations

from typing import Sequence

import torch

from calora.config import DataConfig
from calora.data.cityscapes import (
    CityscapesConceptDataset,
    extract_class_names,
    scan_cityscapes,
)
from calora.data.dummy import DUMMY_CLASS_NAMES, DummyConceptDataset
from calora.data.prompts import ConceptAxis, PromptAugmenter
from calora.data.transforms import hflip, load_image, to_pil, to_tensor
from calora.utils.seeding import derive_generator

__all__ = [
    "CityscapesConceptDataset",
    "ClassNameSampler",
    "ConceptAxis",
    "DummyConceptDataset",
    "PromptAugmenter",
    "build_augmenter",
    "build_dataset",
    "extract_class_names",
    "hflip",
    "load_image",
    "scan_cityscapes",
    "to_pil",
    "to_tensor",
]


def build_augmenter(cfg: DataConfig, seed: int = 0) -> PromptAugmenter:
    return PromptAugmenter.from_file(cfg.augmentation_file, axes=cfg.concept_axes, seed=seed)


def build_dataset(
    cfg: DataConfig,
    *,
    resolution: int,
    seed: int = 0,
    crop: str | None = None,
    hflip: bool | None = None,
    split: str | None = None,
    augs_per_sample: int = 1,
) -> Sequence[dict]:
    """Instantiate the dataset named by ``cfg.kind``.

    Args:
        crop: Overrides ``cfg.crop``. Every *measurement* call site passes
            ``"center"`` explicitly: CA is compared across timesteps and concept
            axes on the assumption that they saw identical pixels, and a random
            crop would silently break that pairing while still producing
            plausible-looking numbers. Training is the only caller that may
            honour ``cfg.crop``.
        hflip: Overrides ``cfg.hflip``, and carries the same invariant for the
            same reason — a mirrored frame is a different x0, so the three
            measurement call sites pass ``False`` explicitly. The flag only
            *permits* flipping; the draw happens in the training loop, so
            ``__getitem__`` stays deterministic either way.
        split: Overrides ``cfg.split``. Used by the evaluation script to export
            the CMMD reference set from ``eval.reference_split`` without
            disturbing the split training reads.
        augs_per_sample: How many ``c_Aug`` prompts each item carries per axis
            (``measure.augs_per_sample``). It lives on the measurement config
            rather than on ``DataConfig`` because it is a property of the CA
            estimator, not of the data; training reads only ``caption``.
    """
    augmenter = build_augmenter(cfg, seed=seed)
    crop = cfg.crop if crop is None else crop
    hflip = cfg.hflip if hflip is None else hflip
    split = cfg.split if split is None else split

    if cfg.kind == "dummy":
        return DummyConceptDataset(
            augmenter,
            num_items=cfg.num_dummy_items if cfg.max_items is None else cfg.max_items,
            resolution=resolution,
            caption_template=cfg.caption_template,
            hflip=hflip,
            augs_per_sample=augs_per_sample,
            seed=seed,
        )
    if cfg.kind == "cityscapes":
        if not cfg.root:
            raise ValueError("data.root is required for data.kind='cityscapes'")
        return CityscapesConceptDataset(
            cfg.root,
            augmenter,
            split=split,
            resolution=resolution,
            caption_mode=cfg.caption_mode,
            caption_template=cfg.caption_template,
            caption_file=cfg.caption_file or None,
            max_items=cfg.max_items,
            crop=crop,
            hflip=hflip,
            augs_per_sample=augs_per_sample,
            seed=seed,
        )
    raise ValueError(f"Unknown data.kind {cfg.kind!r}; expected 'dummy' or 'cityscapes'")


class ClassNameSampler:
    """Draws one ``gtFine`` label map per generated image and lists its classes.

    Table 12's training prompt ends in "with [Class names]", so generation has to
    name a plausible set of objects. The paper gets them from the label map it
    *conditions* on; we have no label generator (see
    :func:`calora.eval.metrics.segmentation_miou`), so we reproduce the prompt
    only — a real label map is drawn from the train split and read for which of
    the 19 classes it contains.

    The scan happens once here rather than per image: 3k directory entries per
    generated image would dominate the sampling cost.
    """

    def __init__(self, cfg: DataConfig, *, seed: int = 0, split: str = "train"):
        self.seed = seed
        self.kind = cfg.kind
        self._labels: list[str] = []
        if cfg.kind == "cityscapes":
            if not cfg.root:
                raise ValueError("data.root is required for data.kind='cityscapes'")
            self._labels = [
                str(entry.label_path)
                for entry in scan_cityscapes(cfg.root, split)
                if entry.label_path is not None
            ]
            if not self._labels:
                raise FileNotFoundError(
                    f"No gtFine label maps under {cfg.root} for split {split!r}. "
                    "Download gtFine_trainvaltest.zip, or set eval.include_class_names=false."
                )

    def __call__(self, index: int) -> list[str]:
        generator = derive_generator(self.seed, "gen-classes", index)
        if self.kind != "cityscapes":
            # No label maps: draw a subset of the placeholder names so the prompt
            # still varies per image, which is what the smoke path is checking.
            count = int(torch.randint(3, len(DUMMY_CLASS_NAMES) + 1, (1,), generator=generator))
            picks = torch.randperm(len(DUMMY_CLASS_NAMES), generator=generator)[:count]
            return [DUMMY_CLASS_NAMES[i] for i in sorted(picks.tolist())]
        pick = int(torch.randint(0, len(self._labels), (1,), generator=generator))
        return extract_class_names(self._labels[pick])
