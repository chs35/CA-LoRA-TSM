"""Generation-quality metrics for the run-matrix comparisons.

``cmmd``
    CLIP-space Maximum Mean Discrepancy (Jayasumana et al., 2024), the paper's
    domain-alignment metric. Unlike FID it is unbiased at small sample counts,
    which matters because the ablation compares many arms on a few thousand
    generated images each. Lower is better.

``clip_score``
    Image-text agreement, used in the paper to check that the generated scene
    actually has the requested weather condition. Higher is better.

``segmentation_miou``
    The paper's headline downstream metric. Deliberately left unimplemented —
    see its docstring.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from calora.eval.embeddings import CLIPEmbedder, PixelEmbedder, list_images
from calora.utils.logging import get_logger

logger = get_logger(__name__)

#: Anything with CLIPEmbedder's two methods. Injectable so the CPU tests and the
#: smoke run can substitute :class:`~calora.eval.embeddings.PixelEmbedder`.
Embedder = CLIPEmbedder | PixelEmbedder

#: Bandwidth and output scaling from the reference CMMD implementation. Changing
#: either makes numbers incomparable with published values.
CMMD_SIGMA = 10.0
CMMD_SCALE = 1000.0


def mmd_rbf(
    x: torch.Tensor, y: torch.Tensor, *, sigma: float = CMMD_SIGMA, scale: float = CMMD_SCALE
) -> float:
    r"""Squared MMD with a Gaussian RBF kernel.

    .. math:: \mathrm{MMD}^2 = \overline{k(x,x')} + \overline{k(y,y')} - 2\overline{k(x,y)}

    Means include the diagonal, matching the reference implementation (the
    biased V-statistic).
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Embedding dim mismatch: {x.shape[1]} vs {y.shape[1]}")

    x = x.to(torch.float64)
    y = y.to(torch.float64)
    x_sq = x.pow(2).sum(dim=1)
    y_sq = y.pow(2).sum(dim=1)
    gamma = 1.0 / (2.0 * sigma**2)

    k_xx = torch.exp(-gamma * (x_sq[:, None] + x_sq[None, :] - 2.0 * (x @ x.T))).mean()
    k_yy = torch.exp(-gamma * (y_sq[:, None] + y_sq[None, :] - 2.0 * (y @ y.T))).mean()
    k_xy = torch.exp(-gamma * (x_sq[:, None] + y_sq[None, :] - 2.0 * (x @ y.T))).mean()
    return float(scale * (k_xx + k_yy - 2.0 * k_xy))


def cmmd(
    reference_dir: str | Path,
    generated_dir: str | Path,
    *,
    embedder: Embedder | None = None,
    max_images: int | None = None,
) -> dict[str, float]:
    """CMMD between two image directories. Lower means better domain alignment."""
    embedder = embedder or CLIPEmbedder()
    reference = list_images(reference_dir)[:max_images]
    generated = list_images(generated_dir)[:max_images]
    logger.info("CMMD over %d reference and %d generated images", len(reference), len(generated))
    value = mmd_rbf(embedder.embed_images(reference), embedder.embed_images(generated))
    return {"cmmd": value, "num_reference": len(reference), "num_generated": len(generated)}


def clip_score(
    images: Sequence[Path | str],
    prompts: Sequence[str],
    *,
    embedder: Embedder | None = None,
    weight: float = 100.0,
) -> dict[str, float]:
    """``w * max(0, cos(image, text))`` averaged over pairs (Hessel et al., 2021)."""
    if len(images) != len(prompts):
        raise ValueError(f"Got {len(images)} images but {len(prompts)} prompts")
    embedder = embedder or CLIPEmbedder()
    image_features = embedder.embed_images(images)
    text_features = embedder.embed_texts(prompts)
    cosine = (image_features * text_features).sum(dim=-1).clamp(min=0.0)
    scores = weight * cosine
    return {
        "clip_score": float(scores.mean()),
        "clip_score_std": float(scores.std()) if len(scores) > 1 else 0.0,
        "num_pairs": len(scores),
    }


def segmentation_miou(*args, **kwargs):
    """Downstream mIoU of a segmentation model trained on the generated dataset.

    Not implemented: the project scope stops at generation quality (CMMD /
    CLIP-Score), because this metric requires training a full segmentation model
    (DAFormer / HRDA in the paper) per ablation arm — several GPU-days on top of
    the study's ~10 LoRA runs.

    To add it later:

    1. Generate a synthetic Cityscapes-style set per arm, pairing each image with
       a ``gtFine`` label map (the paper conditions generation on real labels, so
       the label map is reused as ground truth).
    2. Train the segmentation model on real + synthetic data with the mmseg
       configs from DAFormer / HRDA.
    3. Evaluate mIoU on Cityscapes val and the ACDC / Dark Zurich / BDD /
       Mapillary target sets, mirroring Tables 1-2 of the paper.

    The generation half of step 1 is already covered by
    :func:`calora.train.lora.train_lora` plus a sampling script; only the
    segmentation training harness is missing.
    """
    raise NotImplementedError(
        "Downstream segmentation mIoU is out of scope for the current phase; "
        "see the docstring for the intended implementation path."
    )
