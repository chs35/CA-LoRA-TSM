"""On-disk copy of the training latent cache, shared by every arm.

Every arm of the ablation trains on the same images through the same frozen VAE,
so the latents are identical across arms. Encoding them costs roughly what a
10k-step run's per-step encodes cost (2,975 uncropped Cityscapes frames x 2
orientations at 2048x1024), so an in-memory cache alone only moves that GPU time
to the start of the run. Written once to disk, it is paid by the first arm only.

Two failure modes shape the format:

* **A stale file must never be read as a fresh one.** The file name carries a
  digest of everything that determines the latents (model, VAE dtype,
  resolution, image list, layout, flip), and the same identity is stored in the
  file and compared on load. Anything that differs is a miss, never a reuse.
* **The storage can corrupt writes silently** (the RunPod network volume did,
  more than once). The file records the sha256 of its tensor bytes; a write is
  read back and checked before it counts, and a read that does not reproduce the
  digest is discarded and re-encoded. A bad cache costs time, never numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from calora.utils.logging import get_logger

logger = get_logger(__name__)

#: Bumped whenever the meaning of a stored latent changes, so old files miss.
FORMAT_VERSION = "1"

LatentPools = tuple[list[torch.Tensor], list[torch.Tensor] | None]


def latents_digest(latents: Sequence[torch.Tensor], flipped: Sequence[torch.Tensor] | None) -> str:
    """sha256 over every tensor's raw bytes, in index order (flipped after)."""
    digest = hashlib.sha256()
    for tensor in [*latents, *(flipped or ())]:
        array = tensor.detach().to("cpu").contiguous().numpy()
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.data)
    return digest.hexdigest()


def dataset_identity(dataset: Sequence[dict]) -> str:
    """What images a dataset serves, cheaply: a digest of its file list.

    Datasets without files on disk (the synthetic one) are identified by their
    class, length and seed instead, which is what determines their pixels.
    """
    lister = getattr(dataset, "collate_paths", None)
    if callable(lister):
        paths = "\n".join(lister())
        return "files:" + hashlib.sha256(paths.encode()).hexdigest()
    sources = getattr(dataset, "source_indices", None)
    if sources is not None:
        digest = hashlib.sha256(json.dumps(list(sources)).encode()).hexdigest()
        return f"{type(dataset).__name__}:sources={digest}:seed={getattr(dataset, 'seed', None)}"
    return f"{type(dataset).__name__}:n={len(dataset)}:seed={getattr(dataset, 'seed', None)}"


def model_identity(model_cfg) -> dict[str, str]:
    """The parts of a ``ModelConfig`` that change what the VAE produces."""
    return {
        "model_kind": str(model_cfg.kind),
        "model_name": str(model_cfg.name),
        "revision": str(model_cfg.revision),
        "variant": str(model_cfg.variant),
        "dtype": str(model_cfg.dtype),
        "vae_dtype": str(model_cfg.vae_dtype),
        "resolution": str(model_cfg.resolution),
    }


@dataclass(frozen=True)
class LatentStore:
    """Where the cache lives and which run configuration it belongs to.

    ``identity`` comes from the caller (it knows the model config); the layout
    (``center`` crop vs ``full`` frame) and the flip are added here, because
    those are decided inside the training loop.
    """

    directory: Path
    identity: Mapping[str, str]

    @classmethod
    def for_run(cls, directory: str | Path, model_cfg, dataset: Sequence[dict]) -> "LatentStore":
        identity = {**model_identity(model_cfg), "dataset": dataset_identity(dataset)}
        return cls(Path(directory), identity)

    def _key(self, layout: str, with_hflip: bool) -> dict[str, str]:
        return {
            **{k: str(v) for k, v in self.identity.items()},
            "layout": layout,
            "hflip": str(bool(with_hflip)),
            "format": FORMAT_VERSION,
        }

    def path_for(self, layout: str, with_hflip: bool) -> Path:
        key = json.dumps(self._key(layout, with_hflip), sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.directory / f"latents-{layout}-{digest}.safetensors"

    # -- read ------------------------------------------------------------------------------

    def load(self, layout: str, *, count: int, with_hflip: bool) -> LatentPools | None:
        """The stored pools, or ``None`` on any miss (absent, stale, or corrupt)."""
        from safetensors import safe_open

        path = self.path_for(layout, with_hflip)
        if not path.exists():
            return None
        expected = self._key(layout, with_hflip)
        try:
            with safe_open(str(path), framework="pt") as handle:
                metadata = handle.metadata() or {}
                stored = json.loads(metadata.get("identity", "{}"))
                if stored != expected:
                    logger.warning("Latent cache %s belongs to another configuration; "
                                   "re-encoding", path)
                    return None
                if int(metadata.get("count", -1)) != count:
                    logger.warning("Latent cache %s holds %s latents, dataset has %d; "
                                   "re-encoding", path, metadata.get("count"), count)
                    return None
                latents = [handle.get_tensor(f"{i:06d}") for i in range(count)]
                flipped = (
                    [handle.get_tensor(f"{i:06d}.flip") for i in range(count)]
                    if with_hflip else None
                )
        except Exception as exc:  # noqa: BLE001 - a broken header is a miss, not a crash
            logger.warning("Latent cache %s is unreadable (%s); re-encoding", path, exc)
            return None
        if latents_digest(latents, flipped) != metadata.get("sha256"):
            logger.warning("Latent cache %s fails its sha256 check (storage corruption?); "
                           "re-encoding", path)
            return None
        return latents, flipped

    # -- write -----------------------------------------------------------------------------

    def save(
        self,
        layout: str,
        latents: Sequence[torch.Tensor],
        flipped: Sequence[torch.Tensor] | None,
    ) -> Path | None:
        """Write, read back, verify. Returns the path, or ``None`` if it did not stick.

        A failed save is logged and swallowed: the run already holds the latents
        in memory, so the only cost of a bad write is that the next arm encodes
        again.
        """
        from safetensors.torch import save_file

        with_hflip = flipped is not None
        path = self.path_for(layout, with_hflip)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tensors = {f"{i:06d}": t.contiguous() for i, t in enumerate(latents)}
        if flipped is not None:
            tensors.update({f"{i:06d}.flip": t.contiguous() for i, t in enumerate(flipped)})
        metadata = {
            "identity": json.dumps(self._key(layout, with_hflip), sort_keys=True),
            "count": str(len(latents)),
            "sha256": latents_digest(latents, flipped),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(tmp), metadata=metadata)
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("Could not write latent cache %s (%s); continuing in memory", path, exc)
            tmp.unlink(missing_ok=True)
            return None
        if self.load(layout, count=len(latents), with_hflip=with_hflip) is None:
            logger.warning("Latent cache %s did not read back intact; deleting it", path)
            path.unlink(missing_ok=True)
            return None
        logger.info("Latent cache written and verified: %s (%.1f MiB)",
                    path, path.stat().st_size / 2**20)
        return path
