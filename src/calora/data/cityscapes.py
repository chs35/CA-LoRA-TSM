"""Cityscapes image/caption dataset.

The dataset itself is not redistributable: register at
https://www.cityscapes-dataset.com/register/ and download
``leftImg8bit_trainvaltest.zip`` (~11 GB) and, if you later run the segmentation
evaluation, ``gtFine_trainvaltest.zip`` (~241 MB). Expected layout::

    <root>/leftImg8bit/<split>/<city>/<city>_<seq>_<frame>_leftImg8bit.png
    <root>/gtFine/<split>/<city>/<city>_<seq>_<frame>_gtFine_labelIds.png

Captioning: the paper defers implementation details to its appendix, so both a
deterministic template and an external caption file are supported. Whichever is
used is recorded in the run config, because ``c`` and ``c_Aug`` together define
what "concept" means for the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from calora.data.prompts import PromptAugmenter
from calora.data.transforms import CROP_MODES, load_image
from calora.utils.io import load_json
from calora.utils.logging import get_logger
from calora.utils.seeding import derive_generator

logger = get_logger(__name__)

IMAGE_SUFFIX = "_leftImg8bit.png"
LABEL_SUFFIX = "_gtFine_labelIds.png"

#: The official Cityscapes ``labelIds`` table (``cityscapesScripts/helpers/
#: labels.py``). Indexed by the raw id stored in a ``*_gtFine_labelIds.png``.
CITYSCAPES_LABEL_NAMES: dict[int, str] = {
    0: "unlabeled",
    1: "ego vehicle",
    2: "rectification border",
    3: "out of roi",
    4: "static",
    5: "dynamic",
    6: "ground",
    7: "road",
    8: "sidewalk",
    9: "parking",
    10: "rail track",
    11: "building",
    12: "wall",
    13: "fence",
    14: "guard rail",
    15: "bridge",
    16: "tunnel",
    17: "pole",
    18: "polegroup",
    19: "traffic light",
    20: "traffic sign",
    21: "vegetation",
    22: "terrain",
    23: "sky",
    24: "person",
    25: "rider",
    26: "car",
    27: "truck",
    28: "bus",
    29: "caravan",
    30: "trailer",
    31: "train",
    32: "motorcycle",
    33: "bicycle",
}

#: The 19 evaluation classes, in trainId order. The void-ish ids (0-6, 9, 10,
#: 14-16, 18, 29, 30) are excluded: they are either ignore regions or classes
#: Cityscapes itself does not score, and naming them in a prompt would ask SDXL
#: to draw "out of roi".
CITYSCAPES_TRAIN_LABEL_IDS: tuple[int, ...] = (
    7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,
)


def extract_class_names(label_path: str | Path) -> list[str]:
    """Names of the 19 evaluation classes present in one ``labelIds`` map.

    The order is the fixed trainId order, never the order the ids happen to
    appear in the array: the result goes straight into a generation prompt, and
    a prompt whose word order depended on pixel layout would make two runs over
    the same label map produce different conditioning.
    """
    present = set(load_label(label_path).unique().tolist())
    return [
        CITYSCAPES_LABEL_NAMES[label_id]
        for label_id in CITYSCAPES_TRAIN_LABEL_IDS
        if label_id in present
    ]


@dataclass(frozen=True)
class CityscapesEntry:
    image_path: Path
    label_path: Path | None
    city: str
    key: str


def scan_cityscapes(root: str | Path, split: str = "train") -> list[CityscapesEntry]:
    """Enumerate a Cityscapes split in a stable, sorted order."""
    root = Path(root)
    image_root = root / "leftImg8bit" / split
    if not image_root.is_dir():
        raise FileNotFoundError(
            f"Cityscapes split not found at {image_root}. Set data.root to the directory "
            "that contains 'leftImg8bit/' (and 'gtFine/')."
        )
    label_root = root / "gtFine" / split

    entries: list[CityscapesEntry] = []
    for image_path in sorted(image_root.rglob(f"*{IMAGE_SUFFIX}")):
        city = image_path.parent.name
        key = image_path.name[: -len(IMAGE_SUFFIX)]
        label_path = label_root / city / f"{key}{LABEL_SUFFIX}"
        entries.append(
            CityscapesEntry(
                image_path=image_path,
                label_path=label_path if label_path.exists() else None,
                city=city,
                key=key,
            )
        )
    if not entries:
        raise FileNotFoundError(f"No '*{IMAGE_SUFFIX}' files under {image_root}")
    logger.info("Cityscapes %s: %d images across %d cities", split, len(entries),
                len({e.city for e in entries}))
    return entries


class CityscapesConceptDataset(Sequence):
    """Yields ``{image, caption, augmented, ...}`` dicts for the CA pipeline.

    ``crop`` picks the framing: ``"center"`` makes ``__getitem__`` a pure
    function of the index (required by anything that measures CA, and what lets
    training pre-compute latents once), ``"random"`` slides the crop window
    horizontally each time an item is read so a 10k-step run sees more than the
    middle square of every frame.

    ``hflip`` is a *declaration*, not a transform: the flip is drawn and applied
    by the training loop, not here, so ``__getitem__`` stays a pure function of
    the index and the latent cache can hold both orientations. See
    :func:`calora.train.lora.train_lora`.
    """

    def __init__(
        self,
        root: str | Path,
        augmenter: PromptAugmenter,
        *,
        split: str = "train",
        resolution: int = 1024,
        caption_mode: str = "template",
        caption_template: str = "a photo of a city street scene in {city}",
        caption_file: str | Path | None = None,
        max_items: int | None = None,
        crop: str = "center",
        hflip: bool = False,
        augs_per_sample: int = 1,
        seed: int = 0,
    ):
        if crop not in CROP_MODES:
            raise ValueError(f"Unknown crop mode {crop!r}; expected one of {CROP_MODES}")
        self.entries = scan_cityscapes(root, split)
        if max_items is not None:
            self.entries = self.entries[:max_items]
        self.augmenter = augmenter
        self.resolution = resolution
        self.caption_mode = caption_mode
        self.caption_template = caption_template
        self.crop = crop
        self.hflip = hflip
        self.augs_per_sample = augs_per_sample
        self.seed = seed
        self._crop_draws = 0
        self._captions: dict[str, str] = {}

        if caption_mode == "file":
            if not caption_file:
                raise ValueError("data.caption_file is required when caption_mode='file'")
            self._captions = load_json(caption_file)
            missing = [e.key for e in self.entries if e.key not in self._captions]
            if missing:
                raise KeyError(
                    f"{len(missing)} images have no caption in {caption_file}, "
                    f"e.g. {missing[:3]}"
                )
        elif caption_mode != "template":
            raise ValueError(f"Unknown caption_mode {caption_mode!r}; use 'template' or 'file'")

    def __len__(self) -> int:
        return len(self.entries)

    def caption_for(self, entry: CityscapesEntry) -> str:
        if self.caption_mode == "file":
            return self._captions[entry.key]
        return self.caption_template.format(city=entry.city.replace("_", " "), key=entry.key)

    def distinct_captions(self) -> list[str]:
        """Every caption this dataset can yield, without decoding a single image.

        Cheap because captions come from ``entries`` metadata (or the caption
        file), never from pixels: the template collapses ~3k frames onto one
        caption per city. That is what lets training pre-encode all of them and
        then push the text encoders off the GPU.
        """
        return sorted({self.caption_for(entry) for entry in self.entries})

    def _crop_generator(self, index: int) -> torch.Generator | None:
        """Per-read RNG for ``crop='random'``, derived rather than global.

        The stream is keyed on (seed, item index, how many crops this dataset has
        already handed out), so a run is reproducible end to end while successive
        reads of the same image still get different windows. Using Python's
        global ``random`` here would couple the crops to every other consumer of
        that RNG and make reruns unrepeatable.
        """
        if self.crop != "random":
            return None
        generator = derive_generator(self.seed, "cityscapes-crop", index, self._crop_draws)
        self._crop_draws += 1
        return generator

    def __getitem__(self, index: int) -> dict:  # type: ignore[override]
        entry = self.entries[index]
        caption = self.caption_for(entry)
        return {
            "index": index,
            "image": load_image(
                entry.image_path,
                self.resolution,
                crop=self.crop,
                generator=self._crop_generator(index),
            ),
            "caption": caption,
            "augmented": self.augmenter.augment_many(caption, index, self.augs_per_sample),
            "city": entry.city,
            "key": entry.key,
            "image_path": str(entry.image_path),
            "label_path": None if entry.label_path is None else str(entry.label_path),
        }

    def collate_paths(self) -> list[str]:
        return [str(e.image_path) for e in self.entries]


def load_label(path: str | Path) -> torch.Tensor:
    """Read a ``gtFine`` label map as ``[H,W]`` int64 (raw Cityscapes ids)."""
    from PIL import Image

    with Image.open(path) as img:
        import numpy as np

        return torch.from_numpy(np.asarray(img.convert("L"), dtype=np.int64))
