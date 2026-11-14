"""Where the CA drift snapshots get their x0 (``measure.drift_source``).

The selection is measured on images the base model generated itself (the
paper's protocol; :mod:`calora.data.selfgen`). Drift asks how that measurement
changes as the weights move, and there are three readings of "the same
measurement" at step N:

``selfgen_current``
    x0 is regenerated from c by the *current* (LoRA-merged) model before every
    snapshot, with the same per-image seeds as the base set. The snapshot then
    measures the model at step N on its own samples -- the self-generated
    protocol applied to the model being measured -- while the shared initial
    noise keeps snapshot i paired with baseline image i. Step 0 is the base set
    itself (the base model's own samples under the same seeds). Costs one
    generation of ``train.stability_num_samples`` images per snapshot.
``selfgen_fixed``
    The base set's first ``train.stability_num_samples`` images throughout.
    Isolates weight change from data change, at the price of x0 no longer being
    the measured model's own sample once the weights move.
``dataset``
    Real images (centre crop, no flip), as every run before 2026-11-07.

:func:`drift_samples` builds the baseline samples and, for ``selfgen_current``,
the provider the stability callback calls at each snapshot.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from calora.ca.measure import MeasurementSample, prepare_samples
from calora.utils.logging import get_logger

logger = get_logger(__name__)


def latents_fingerprint(samples: Sequence[MeasurementSample]) -> str:
    """Short sha256 of the samples' latents: which x0 a snapshot measured."""
    digest = hashlib.sha256()
    for sample in samples:
        array = sample.latents.detach().to(torch.float32).cpu().contiguous().numpy()
        digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def pil_to_x0(image, resolution: int) -> torch.Tensor:
    """The tensor :func:`calora.data.transforms.load_image` would give for ``image``
    saved and read back: generated images are already at the model resolution,
    so the resize and centre crop are identities and PNG is lossless."""
    from calora.data.transforms import center_crop, resize_short_side, to_tensor

    return to_tensor(center_crop(resize_short_side(image.convert("RGB"), resolution), resolution))


@dataclass
class SelfGenDriftProvider:
    """Regenerates the drift x0 from the current weights, image by image paired.

    Called by the stability callback inside :func:`calora.analysis.stability.merged_lora`,
    so the sampler runs the effective weights ``W0 + BA``. Prompt conditionings
    are reused from the baseline samples: only the UNet is trained, so c and
    c_Aug encode identically at every step.
    """

    adapter: object
    template: Sequence[MeasurementSample]
    prompt: str
    seed: int
    num_inference_steps: int
    guidance_scale: float
    resolution: int
    batch_size: int = 4
    last_generation_seconds: float = 0.0
    last_fingerprint: str = ""

    def __call__(self, step: int) -> list[MeasurementSample]:
        from calora.data.selfgen import image_generator

        started = time.perf_counter()
        count = len(self.template)
        latents: list[torch.Tensor] = []
        for offset in range(0, count, self.batch_size):
            # Always a full batch on the base set's boundaries, even past `count`
            # (the extras are dropped): batched kernels are not bit-identical
            # across batch compositions, and with unchanged weights a snapshot
            # must reproduce the base set exactly -- step 0 is that set.
            indices = list(range(offset, offset + self.batch_size))
            images = self.adapter.generate(
                [self.prompt] * len(indices),
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=[image_generator(self.seed, i) for i in indices],
            )
            for index, image in zip(indices, images):
                if index >= count:
                    break
                x0 = pil_to_x0(image, self.resolution).unsqueeze(0)
                latents.append(self.adapter.encode_images(x0).detach().cpu())
        samples = [replace(sample, latents=latent) for sample, latent in zip(self.template, latents)]
        self.last_generation_seconds = time.perf_counter() - started
        self.last_fingerprint = latents_fingerprint(samples)
        logger.info(
            "drift x0 @ step %d: regenerated %d images from the current weights in %.1fs (%s)",
            step, count, self.last_generation_seconds, self.last_fingerprint,
        )
        return samples


def drift_samples(cfg, adapter):
    """Baseline drift samples, the per-snapshot provider (or ``None``), and provenance.

    Must be called before LoRA is attached: for both self-generated modes the
    baseline is the base model's set.
    """
    source = cfg.measure.drift_source
    count = cfg.train.stability_num_samples
    axes = cfg.data.concept_axes
    if source == "dataset":
        from calora.data import build_dataset

        dataset = build_dataset(
            cfg.data,
            resolution=cfg.model.resolution,
            seed=cfg.measure.seed,
            crop="center",
            hflip=False,
            augs_per_sample=cfg.measure.augs_per_sample,
        )
        samples = prepare_samples(adapter, dataset, axes=axes, num_samples=count)
        return samples, None, {"data_source": "dataset", "drift_source": source}

    from calora.data.selfgen import (
        build_selfgen_dataset,
        manifest_sha256,
        selfgen_latents,
        spec_from_config,
    )

    dataset = build_selfgen_dataset(
        cfg,
        num_items=count,
        augs_per_sample=cfg.measure.augs_per_sample,
        spec=spec_from_config(cfg, adapter),
    )
    latents = selfgen_latents(adapter, dataset, cfg.model)
    samples = prepare_samples(adapter, dataset, axes=axes, num_samples=count, latents=latents)
    provenance = {
        "data_source": "selfgen",
        "drift_source": source,
        "selfgen_manifest_sha256": manifest_sha256(dataset.directory),
    }
    if source == "selfgen_fixed":
        return samples, None, provenance

    sg = cfg.measure.selfgen
    provider = SelfGenDriftProvider(
        adapter=adapter,
        template=samples,
        prompt=dataset.caption,
        seed=sg.seed,
        num_inference_steps=sg.num_inference_steps,
        guidance_scale=sg.guidance_scale,
        resolution=cfg.model.resolution,
        batch_size=sg.batch_size,
    )
    return samples, provider, provenance
