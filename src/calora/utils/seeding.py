"""Deterministic seeding helpers.

Reproducibility matters more than usual here: the whole project compares CA
statistics across timesteps and concept axes, so those comparisons are only
meaningful if the underlying (x0, noise, caption) draws are identical.
``derive_generator`` gives a per-sample generator whose stream depends on the
sample index but *not* on the timestep, which is what makes cross-t comparisons
paired.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_algorithms: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def stable_hash(*parts: object) -> int:
    """A process-independent hash (``hash()`` is salted per process)."""
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def derive_seed(base_seed: int, *parts: object) -> int:
    return (base_seed + stable_hash(*parts)) % (2**31 - 1)


def derive_generator(
    base_seed: int, *parts: object, device: str | torch.device = "cpu"
) -> torch.Generator:
    """A ``torch.Generator`` seeded from ``base_seed`` and a stable key."""
    generator = torch.Generator(device=device)
    generator.manual_seed(derive_seed(base_seed, *parts))
    return generator
