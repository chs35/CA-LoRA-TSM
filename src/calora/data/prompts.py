"""Concept-axis prompt augmentation (``c_Aug`` in the paper).

``L_Concept`` compares the denoiser's prediction under the original caption ``c``
against its prediction under an augmented caption ``c_Aug``. Which augmentations
you pick *defines* the concept axis, so the sets live in a versioned JSON file
(``configs/prompt_augmentations.json``) rather than in code.

Selection is deterministic in ``(seed, axis, item_index)``: the same item always
gets the same augmented prompt, at every timestep and in every run. Without that,
cross-timestep and cross-axis comparisons would be confounded by prompt sampling.
The key deliberately excludes the timestep *and* the augmentation slot, so
drawing several prompts per item (``measure.augs_per_sample``) still leaves every
timestep looking at the identical set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from calora.utils.io import load_json
from calora.utils.logging import get_logger
from calora.utils.seeding import derive_seed

logger = get_logger(__name__)

DEFAULT_AUGMENTATION_FILE = "configs/prompt_augmentations.json"


@dataclass(frozen=True)
class ConceptAxis:
    name: str
    description: str
    templates: tuple[str, ...]

    def render(self, caption: str, index: int) -> str:
        return self.templates[index % len(self.templates)].format(caption=caption)


class PromptAugmenter:
    """Deterministically maps ``(caption, item_index)`` to one ``c_Aug`` per axis."""

    def __init__(self, axes: dict[str, ConceptAxis], seed: int = 0):
        if not axes:
            raise ValueError("PromptAugmenter needs at least one concept axis")
        self.axes = axes
        self.seed = seed

    @classmethod
    def from_file(
        cls,
        path: str | Path = DEFAULT_AUGMENTATION_FILE,
        *,
        axes: Sequence[str] | None = None,
        seed: int = 0,
    ) -> PromptAugmenter:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt augmentation file not found: {path}. "
                f"Expected the repo default at {DEFAULT_AUGMENTATION_FILE}."
            )
        payload = load_json(path)
        available = payload.get("axes", {})
        wanted = list(axes) if axes is not None else list(available)
        missing = [a for a in wanted if a not in available]
        if missing:
            raise KeyError(f"Axes {missing} not in {path}; available: {sorted(available)}")
        built = {
            name: ConceptAxis(
                name=name,
                description=available[name].get("description", ""),
                templates=tuple(available[name]["templates"]),
            )
            for name in wanted
        }
        for axis in built.values():
            if not axis.templates:
                raise ValueError(f"Concept axis {axis.name!r} has no templates")
        return cls(built, seed=seed)

    @property
    def axis_names(self) -> list[str]:
        return list(self.axes)

    def augment_many(self, caption: str, index: int, m: int = 1) -> dict[str, list[str]]:
        """``m`` augmented captions per axis for item ``index``.

        The draw is keyed on ``(seed, axis, index)`` only — not on the timestep
        and not on the slot ``j`` — so a single RNG produces the whole set at
        once and every timestep sees exactly the same prompts. Keying on ``j``
        would work too, but keying on ``t`` would silently unpair the timesteps.

        Sampling is without replacement, so the ``m`` prompts of one axis are
        distinct. If more are asked for than the axis has templates, every
        template is used once and the remainder is drawn with replacement.
        """
        if m < 1:
            raise ValueError(f"augment_many needs m >= 1, got {m}")
        out: dict[str, list[str]] = {}
        for name, axis in self.axes.items():
            rng = random.Random(derive_seed(self.seed, "c_aug", name, index))
            available = len(axis.templates)
            picks = rng.sample(range(available), min(m, available))
            if m > available:
                logger.warning(
                    "Concept axis %r has %d templates but %d augmentations per sample were "
                    "requested; the remaining %d are drawn with replacement, so those "
                    "contributions are not independent.",
                    name,
                    available,
                    m,
                    m - available,
                )
                picks += rng.choices(range(available), k=m - available)
            out[name] = [axis.render(caption, pick) for pick in picks]
        return out

    def augment(self, caption: str, index: int) -> dict[str, str]:
        """One augmented caption per axis for item ``index``."""
        return {name: prompts[0] for name, prompts in self.augment_many(caption, index, 1).items()}

    def describe(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "description": axis.description,
                "num_templates": len(axis.templates),
                "templates": list(axis.templates),
            }
            for name, axis in self.axes.items()
        }
