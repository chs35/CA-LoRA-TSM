"""CLIP embeddings backing both CMMD and CLIP-Score.

Both metrics are defined on CLIP ViT-L/14 features, so they share one embedder:
using different backbones for the two would make the numbers incomparable with
the literature and with each other.

:class:`PixelEmbedder` is a second implementation of the same interface that
needs no download. It exists only so the metric plumbing can be exercised
offline — see its docstring for why its numbers must never be reported.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from calora.utils.device import resolve_device
from calora.utils.io_images import load_image_rgb
from calora.utils.logging import get_logger
from calora.utils.seeding import derive_generator

logger = get_logger(__name__)

DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _features(output) -> torch.Tensor:
    """The projected CLIP embedding, whichever shape ``get_*_features`` returns.

    transformers 4.x returns a plain ``Tensor`` from
    ``CLIPModel.get_image_features`` / ``get_text_features``. transformers 5.x
    returns a ``BaseModelOutputWithPooling`` instead, whose ``pooler_output``
    *is* the projected embedding — verified on ``openai/clip-vit-large-patch14``,
    where it comes out at 768 dims, the projection width, not the 1024-dim raw
    vision-tower hidden state. On that version ``image_embeds`` / ``text_embeds``
    are absent or ``None``, so ``pooler_output`` is the only handle.

    The tensor passthrough has to stay: the project runs on both environments
    (CPU box on 4.x, pod on 5.x) and the metrics must be identical on each, so
    this cannot be narrowed to the 5.x shape alone. ``image_embeds`` /
    ``text_embeds`` are preferred when present and non-``None`` in case a later
    transformers restores them as the canonical field.
    """
    if isinstance(output, torch.Tensor):
        return output
    for name in ("image_embeds", "text_embeds"):
        value = getattr(output, name, None)
        if value is not None:
            return value
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    raise TypeError(
        f"Cannot read CLIP features from {type(output).__name__}: expected a Tensor or an "
        "object with image_embeds/text_embeds/pooler_output."
    )


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images under {directory}")
    return paths


class CLIPEmbedder:
    """Batched CLIP image/text embeddings, L2-normalised."""

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        *,
        device: str | torch.device = "auto",
        dtype: torch.dtype = torch.float32,
        batch_size: int = 32,
    ):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "transformers is required for CLIP-based metrics. "
                "Install with: pip install 'calora[diffusion]'"
            ) from exc

        self.device = resolve_device(device)
        self.dtype = dtype
        self.batch_size = batch_size
        logger.info("Loading CLIP %s on %s", model_name, self.device)
        # ``dtype=`` not ``torch_dtype=``: the latter is a backwards-compatibility
        # alias in transformers 5.x and is on its way out.
        self.model = CLIPModel.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_images(self, images: Sequence[Path | str] | Iterable) -> torch.Tensor:
        """``[N, D]`` normalised image embeddings from paths or PIL images."""
        items = list(images)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            # Paths go through load_image_rgb: it retries the transient OSErrors
            # the RunPod volume throws mid-eval, and it closes the handle it
            # opened, so a 500-image directory leaks no file descriptors. PIL
            # objects handed in by the caller are left alone; they are not ours.
            pil = [
                load_image_rgb(item) if isinstance(item, (str, Path)) else item
                for item in chunk
            ]
            inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
            features = _features(self.model.get_image_features(**inputs))
            outputs.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """``[N, D]`` normalised text embeddings."""
        outputs: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            inputs = self.processor(
                text=chunk, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            features = _features(self.model.get_text_features(**inputs))
            outputs.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
        return torch.cat(outputs, dim=0)


class PixelEmbedder:
    """An offline stand-in for :class:`CLIPEmbedder`. **Plumbing tests only.**

    Images become downsampled raw pixels; text becomes a hash-seeded random
    vector. Both are L2-normalised and share one dimensionality, so
    :func:`~calora.eval.metrics.cmmd` and :func:`~calora.eval.metrics.clip_score`
    run end to end without a single download.

    **Never report a number produced with this embedder.** CMMD over raw pixels
    is a comparison of colour histograms, and the "text" side is unrelated to
    the string beyond being a deterministic function of it, so a CLIP-Score
    computed here measures nothing at all — an arm could win it by being beige.
    It exists so the CPU smoke test and the unit tests can prove the pipeline
    wiring (paths, pairing, grouping, JSON) without a GPU or a 1.7 GB checkpoint.
    """

    def __init__(self, *, side: int = 16, batch_size: int = 32, seed: int = 0):
        self.side = side
        self.batch_size = batch_size
        self.seed = seed
        self.dim = 3 * side * side

    @torch.no_grad()
    def embed_images(self, images: Sequence[Path | str] | Iterable) -> torch.Tensor:
        from PIL import Image

        from calora.data.transforms import to_tensor

        outputs: list[torch.Tensor] = []
        for item in list(images):
            # Same retrying reader as the CLIP path, so the stub and the real
            # embedder fail (or recover) identically on a flaky volume.
            if isinstance(item, (str, Path)):
                pil = load_image_rgb(item)
            else:
                pil = item.convert("RGB")
            small = pil.resize((self.side, self.side), Image.BILINEAR)
            outputs.append(to_tensor(small).reshape(-1))
        stacked = torch.stack(outputs)
        return torch.nn.functional.normalize(stacked.float(), dim=-1)

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        # Keyed on the string itself, so the same prompt always lands on the same
        # point and two runs over one manifest agree exactly.
        vectors = [
            torch.randn(self.dim, generator=derive_generator(self.seed, "pixel-text", text))
            for text in texts
        ]
        return torch.nn.functional.normalize(torch.stack(vectors), dim=-1)
