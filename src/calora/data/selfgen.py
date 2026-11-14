"""The CA measurement set the paper prescribes: images the model generated itself.

CA-LoRA (arXiv:2503.22172) Sec. 3.2: "we generate a few images x0 using the T2I
model Phi_T2I parameterized by theta and the text prompt c (i.e.
x0 = Phi_T2I(c; theta))"; Algorithm 1 then VAE-encodes each image and
forward-noises it with ``scheduler.add_noise``. Every run of this project before
2026-11-07 measured on real Cityscapes frames instead; this module is the fix.

Three pieces:

* :func:`generate_selfgen_set` samples ``num_images`` images from ``c`` and
  writes them with a manifest that pins the spec (model, prompt, sampler, seed)
  and every image's sha256. It is resumable per image and generates nothing when
  a matching, intact set is already on disk -- the set is made once and reused
  by every timestep, every restart and every downstream project.
* :class:`SelfGeneratedConceptDataset` serves the set in the item schema of the
  other datasets (``image``, ``caption``, ``augmented``), refusing to run if the
  caption it is asked to anchor ``L_Concept`` on is not the prompt the images
  were generated from.
* :func:`selfgen_latents` VAE-encodes the set once and keeps the latents on disk
  next to it (:class:`calora.train.latent_store.LatentStore`, sha256-checked).

Per-image generators (:func:`image_generator`) make an image's initial noise a
function of its index alone, and images are sampled in full batches on fixed
boundaries, so a resume point or a repair never changes the batch an image was
sampled in (batched kernels are not bit-identical across compositions). The same seeds regenerate the drift images from the
current weights (``measure.drift_source: selfgen_current``), which keeps those
snapshots paired with the base set image by image.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from calora.data.prompts import PromptAugmenter
from calora.utils.io import dump_json, load_json
from calora.utils.logging import get_logger
from calora.utils.seeding import derive_generator

logger = get_logger(__name__)

MANIFEST_NAME = "manifest.json"
#: Bumped whenever what a stored image means changes, so old sets miss.
FORMAT_VERSION = "1"
#: Seed-derivation key. Not shared with any other stream (eval generation uses
#: "gen"), so the measurement set never coincides with an evaluated image.
SEED_KEY = "ca-selfgen"

#: Spec fields that decide the pixels. A stored set whose spec differs in any of
#: them is a different set; ``num_images`` is checked separately (a larger set
#: serves a smaller request).
SPEC_KEYS = (
    "format",
    "prompt",
    "num_inference_steps",
    "guidance_scale",
    "seed",
    "seed_key",
    "resolution",
    "model",
    "scheduler",
    "batch_size",
)


class SelfGenMismatchError(RuntimeError):
    """A set on disk was generated under a different spec than the one requested."""


def image_generator(seed: int, index: int) -> torch.Generator:
    """The generator image ``index`` of a set with base ``seed`` draws its noise from."""
    return derive_generator(seed, SEED_KEY, index)


def image_name(index: int) -> str:
    return f"{index:05d}.png"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_sha256(directory: str | Path) -> str:
    """sha256 of the manifest bytes: the identity of a set, recorded in provenance."""
    return sha256_file(Path(directory) / MANIFEST_NAME)


def check_prompt(prompt: str) -> str:
    if not prompt or "{" in prompt:
        raise ValueError(
            f"Self-generated CA data needs one fixed prompt c, got {prompt!r}: the images "
            "are sampled from it and L_Concept is anchored on it, so a per-item template "
            "({city}, {key}) has no single value to generate from."
        )
    return prompt


def build_spec(
    *,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    resolution: int,
    model: Mapping[str, str],
    scheduler: str,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "prompt": check_prompt(prompt),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "seed": int(seed),
        "seed_key": SEED_KEY,
        "resolution": int(resolution),
        "model": {k: str(v) for k, v in model.items()},
        "scheduler": str(scheduler),
        # Part of the spec: it fixes the batch composition each image was sampled
        # in, and batched kernels are not bit-identical across compositions.
        "batch_size": int(batch_size),
    }


def spec_from_config(cfg, adapter=None) -> dict[str, Any]:
    """The spec a :class:`calora.config.Config` asks for.

    ``scheduler`` is the adapter's sampler name when an adapter is at hand, and
    otherwise left out of the comparison by being read back from the manifest
    (see :func:`verify_selfgen_set`) -- checking a set must not require loading
    the model.
    """
    from calora.train.latent_store import model_identity

    sg = cfg.measure.selfgen
    return build_spec(
        prompt=cfg.data.caption_template,
        num_inference_steps=sg.num_inference_steps,
        guidance_scale=sg.guidance_scale,
        seed=sg.seed,
        resolution=cfg.model.resolution,
        model=model_identity(cfg.model),
        scheduler=getattr(adapter, "scheduler_name", "") if adapter is not None else "",
        batch_size=sg.batch_size,
    )


def _spec_diff(stored: Mapping[str, Any], wanted: Mapping[str, Any]) -> dict[str, tuple]:
    diff = {}
    for key in SPEC_KEYS:
        if key == "scheduler" and not wanted.get(key):
            continue  # caller had no adapter to ask; see spec_from_config
        if stored.get(key) != wanted.get(key):
            diff[key] = (stored.get(key), wanted.get(key))
    return diff


def load_manifest(directory: str | Path) -> dict[str, Any] | None:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        return None
    return load_json(path)


def _write_manifest(directory: Path, spec: Mapping[str, Any], images: Sequence[dict],
                    extra: Mapping[str, Any]) -> None:
    dump_json(
        {
            **spec,
            "num_images": len(images),
            "images": list(images),
            **dict(extra),
        },
        directory / MANIFEST_NAME,
    )


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def generate_selfgen_set(
    adapter,
    spec: Mapping[str, Any],
    directory: str | Path,
    *,
    num_images: int,
    batch_size: int | None = None,
    save_image: Callable | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Make sure ``directory`` holds ``num_images`` intact images of ``spec``.

    Only what is missing or fails its sha256 is generated; a complete, matching
    set costs a manifest read and one hash per image. A set made under any other
    spec raises :class:`SelfGenMismatchError` rather than being mixed into or
    overwritten -- delete it, or point ``measure.selfgen.dir`` elsewhere.

    Returns the manifest as written.
    """
    from calora.utils.io_images import save_image_verified

    save = save_image or save_image_verified
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = dict(spec)
    batch_size = int(spec["batch_size"]) if batch_size is None else int(batch_size)
    if batch_size != int(spec["batch_size"]):
        raise ValueError(
            f"batch_size={batch_size} disagrees with the spec's {spec['batch_size']}; the "
            "batch composition is part of what an image is"
        )
    if not spec.get("scheduler"):
        spec["scheduler"] = str(getattr(adapter, "scheduler_name", ""))

    manifest = load_manifest(directory)
    records: dict[int, dict] = {}
    if manifest is not None:
        diff = _spec_diff(manifest, spec)
        if diff:
            raise SelfGenMismatchError(
                f"{directory} holds a self-generated set of another spec {diff} "
                "(stored, requested). Use another measure.selfgen.dir or delete it."
            )
        records = {int(r["index"]): dict(r) for r in manifest.get("images", [])}

    todo = []
    for index in range(num_images):
        record = records.get(index)
        path = directory / image_name(index)
        if record is not None and path.exists() and sha256_file(path) == record["sha256"]:
            continue
        if record is not None:
            logger.warning("Self-generated image %s is missing or corrupt; regenerating", path)
        todo.append(index)

    extra = {
        "torch": torch.__version__,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        **dict(extra_manifest or {}),
    }
    if not todo:
        logger.info("Self-generated set %s is complete (%d images); nothing to generate",
                    directory, num_images)
        if manifest is not None and int(manifest.get("num_images", 0)) >= num_images:
            return manifest
    else:
        logger.info("Generating %d of %d self-generated images into %s",
                    len(todo), num_images, directory)

    started = time.perf_counter()
    # Batches sit on fixed boundaries (indices [k*b, (k+1)*b)) and are always
    # full, so an image is sampled in the same batch composition whether the set
    # is made in one go, resumed, or repaired -- and the drift provider, which
    # regenerates on the same boundaries, reproduces it exactly at step 0.
    # Batched kernels are not bit-identical across batch compositions.
    batches = sorted({index // batch_size for index in todo})
    done = 0
    for batch in batches:
        chunk = list(range(batch * batch_size, (batch + 1) * batch_size))
        images = adapter.generate(
            [spec["prompt"]] * len(chunk),
            num_inference_steps=spec["num_inference_steps"],
            guidance_scale=spec["guidance_scale"],
            generator=[image_generator(spec["seed"], index) for index in chunk],
        )
        for index, image in zip(chunk, images):
            if index not in todo:
                continue
            path = save(image, directory / image_name(index))
            records[index] = {"index": index, "file": image_name(index),
                              "sha256": sha256_file(path)}
        # The manifest is rewritten after every batch so a crash costs at most one
        # batch; it only ever lists images whose bytes were verified on disk.
        _write_manifest(directory, spec, [records[i] for i in sorted(records)], extra)
        done += sum(1 for index in chunk if index in todo)
        elapsed = time.perf_counter() - started
        logger.info("self-gen %d/%d (%.2f s/image)", done, len(todo), elapsed / done)

    ordered = [records[i] for i in sorted(records)]
    _write_manifest(directory, spec, ordered, extra)
    return load_manifest(directory)  # type: ignore[return-value]


def verify_selfgen_set(
    directory: str | Path, spec: Mapping[str, Any], *, num_images: int, check_files: bool = True
) -> dict[str, Any]:
    """The manifest of a complete set matching ``spec``, or an error saying why not.

    Measurement never generates on its own (scripts/generate_selfgen.py does), so
    a missing or partial set is a hard error here rather than a silent detour.
    """
    directory = Path(directory)
    manifest = load_manifest(directory)
    if manifest is None:
        raise FileNotFoundError(
            f"No self-generated CA set at {directory}. Run scripts/generate_selfgen.py "
            "with the same config first."
        )
    diff = _spec_diff(manifest, spec)
    if diff:
        raise SelfGenMismatchError(
            f"{directory} was generated under another spec {diff} (stored, requested)."
        )
    records = {int(r["index"]): r for r in manifest.get("images", [])}
    missing = [i for i in range(num_images) if i not in records]
    if missing:
        raise FileNotFoundError(
            f"{directory} holds {len(records)} images, {num_images} requested "
            f"(first missing: {missing[:5]}). Re-run scripts/generate_selfgen.py."
        )
    if check_files:
        for index in range(num_images):
            path = directory / records[index]["file"]
            if not path.exists() or sha256_file(path) != records[index]["sha256"]:
                raise OSError(
                    f"{path} is missing or fails its sha256; re-run scripts/generate_selfgen.py "
                    "to regenerate it."
                )
    return manifest


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class SelfGeneratedConceptDataset(Sequence):
    """A verified self-generated set in the item schema of the other datasets.

    ``caption`` is always the prompt the images were generated from. Being asked
    for any other caption is an error: ``L_Concept`` compares the prediction under
    c against c_Aug, and c_Aug is a rewrite of exactly that string -- a different
    c would make the measurement mix the caption change into the concept change.
    """

    #: Generated at the model resolution, so nothing is ever cropped; declared so
    #: consumers can ask any dataset whether ``__getitem__`` is deterministic.
    crop = "center"

    def __init__(
        self,
        directory: str | Path,
        augmenter: PromptAugmenter,
        *,
        caption: str,
        num_items: int,
        resolution: int,
        augs_per_sample: int = 1,
        manifest: Mapping[str, Any] | None = None,
    ):
        self.directory = Path(directory)
        self.manifest = dict(manifest) if manifest is not None else load_manifest(self.directory)
        if self.manifest is None:
            raise FileNotFoundError(f"No self-generated CA set at {self.directory}")
        if caption != self.manifest["prompt"]:
            raise ValueError(
                f"Caption {caption!r} differs from the prompt the self-generated set was "
                f"sampled from ({self.manifest['prompt']!r}). x0 and the L_Concept anchor "
                "must share one c."
            )
        if int(self.manifest["resolution"]) != resolution:
            raise ValueError(
                f"Set {self.directory} is {self.manifest['resolution']}px, the model runs at "
                f"{resolution}px"
            )
        records = {int(r["index"]): r for r in self.manifest["images"]}
        if any(i not in records for i in range(num_items)):
            raise FileNotFoundError(
                f"{self.directory} holds {len(records)} images, {num_items} requested"
            )
        self.records = [records[i] for i in range(num_items)]
        self.augmenter = augmenter
        self.caption = caption
        self.resolution = resolution
        self.augs_per_sample = augs_per_sample
        self.seed = int(self.manifest["seed"])
        self.source_indices = list(range(num_items))

    def __len__(self) -> int:
        return len(self.source_indices)

    def restrict(self, indices: Sequence[int]) -> None:
        self.source_indices = [self.source_indices[i] for i in indices]

    def item_keys(self) -> list[str]:
        return [f"selfgen_{s:05d}" for s in self.source_indices]

    def collate_paths(self) -> list[str]:
        """Identity for :func:`calora.train.latent_store.dataset_identity`: the file
        list *and* every file's sha256, so a regenerated image is a cache miss."""
        return [
            f"{self.records[s]['file']}:{self.records[s]['sha256']}" for s in self.source_indices
        ]

    def caption_at(self, index: int) -> str:
        return self.caption

    def distinct_captions(self) -> list[str]:
        return [self.caption]

    def image_path(self, index: int) -> Path:
        return self.directory / self.records[self.source_indices[index]]["file"]

    def __getitem__(self, index: int) -> dict:  # type: ignore[override]
        from calora.data.transforms import load_image

        if not 0 <= index < len(self):
            raise IndexError(index)
        source = self.source_indices[index]
        path = self.image_path(index)
        return {
            "index": index,
            # Generated at the model resolution: load_image's resize + centre crop
            # is the identity, kept so the tensor convention is the shared one.
            "image": load_image(path, self.resolution, crop="center"),
            "caption": self.caption,
            # Keyed by the image index, like every dataset, never by t or slot.
            "augmented": self.augmenter.augment_many(self.caption, source, self.augs_per_sample),
            "city": "selfgen",
            "key": f"selfgen_{source:05d}",
            "image_path": str(path),
            "label_path": None,
        }


def build_selfgen_dataset(cfg, *, num_items: int, augs_per_sample: int, verify: bool = True,
                          spec: Mapping[str, Any] | None = None) -> SelfGeneratedConceptDataset:
    """The verified set ``cfg.measure.selfgen`` names, ``num_items`` long."""
    from calora.data import build_augmenter

    directory = Path(cfg.measure.selfgen.dir)
    wanted = dict(spec) if spec is not None else spec_from_config(cfg)
    manifest = (
        verify_selfgen_set(directory, wanted, num_images=num_items)
        if verify
        else load_manifest(directory)
    )
    return SelfGeneratedConceptDataset(
        directory,
        build_augmenter(cfg.data, seed=cfg.measure.seed),
        caption=cfg.data.caption_template,
        num_items=num_items,
        resolution=cfg.model.resolution,
        augs_per_sample=augs_per_sample,
        manifest=manifest,
    )


# --------------------------------------------------------------------------------------
# Latents
# --------------------------------------------------------------------------------------


def selfgen_latents(adapter, dataset: SelfGeneratedConceptDataset, model_cfg) -> list[torch.Tensor]:
    """VAE latents of ``dataset``, encoded once and kept on disk next to the images.

    The cache key is the model identity plus the dataset's file list *with* each
    image's sha256, so a regenerated image can never be served a stale latent.
    """
    from calora.train.latent_store import LatentStore

    store = LatentStore.for_run(dataset.directory, model_cfg, dataset)
    cached = store.load("center", count=len(dataset), with_hflip=False)
    if cached is not None:
        logger.info("Self-generated latents read from %s", store.path_for("center", False))
        return [t.clone() for t in cached[0]]
    started = time.perf_counter()
    latents = []
    for index in range(len(dataset)):
        image = dataset[index]["image"].unsqueeze(0)
        latents.append(adapter.encode_images(image).detach().cpu())
    logger.info("Encoded %d self-generated images in %.1fs", len(latents),
                time.perf_counter() - started)
    store.save("center", latents, None)
    return latents


def provenance_for(cfg, *, manifest: Mapping[str, Any] | None = None,
                   directory: str | Path | None = None) -> dict[str, str]:
    """What a measurement's x0 was, as flat strings for a safetensors header."""
    if cfg.measure.source == "dataset":
        return {
            "data_source": "dataset",
            "dataset_kind": str(cfg.data.kind),
            "dataset_split": str(cfg.data.split),
            "dataset_root": str(cfg.data.root),
        }
    directory = Path(directory or cfg.measure.selfgen.dir)
    manifest = manifest or load_manifest(directory) or {}
    return {
        "data_source": "selfgen",
        "selfgen_manifest_sha256": manifest_sha256(directory),
        "selfgen_prompt": str(manifest.get("prompt", "")),
        "selfgen_num_inference_steps": str(manifest.get("num_inference_steps", "")),
        "selfgen_guidance_scale": str(manifest.get("guidance_scale", "")),
        "selfgen_seed": str(manifest.get("seed", "")),
        "selfgen_model": json.dumps(manifest.get("model", {}), sort_keys=True),
    }
