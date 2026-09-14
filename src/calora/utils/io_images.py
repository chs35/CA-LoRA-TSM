"""Image reads and writes that survive a flaky network volume.

The experiments run against an mfs volume on RunPod which, on 2026-07-30, both
returned ``EIO`` mid-write (killing a 500-image generation arm at image 82) and
hung/failed on reads during evaluation. Every helper here exists for that one
reason: a transient volume error must cost seconds of retry, not a re-queued
multi-hour job, and it must never leave a half-written PNG that a later
``--skip-existing`` or "reference already exported" check would accept.

Lives in the package rather than in ``scripts/`` because three callers need it:
``scripts/generate_images.py`` (saving samples), ``scripts/evaluate.py``
(exporting the CMMD reference set) and :mod:`calora.eval.embeddings` (reading
both sets back). ``scripts/`` is not an importable package, so a script-level
home would not have reached the third.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Callable

from calora.utils.logging import get_logger

logger = get_logger(__name__)

#: Save attempts per image and the first backoff, doubling per retry. A couple
#: of seconds of retry is cheaper than re-queueing a generation job.
SAVE_ATTEMPTS = 3
SAVE_BACKOFF_SECONDS = 0.5

#: Read attempts and the first backoff, doubling per retry (1s then 2s). Reads
#: are bounded more patiently than writes because a stalled read is the volume
#: recovering, and the alternative is losing a whole arm's evaluation.
READ_ATTEMPTS = 3
READ_BACKOFF_SECONDS = 1.0


def verify_image(path: str | Path) -> None:
    """Fully decode ``path``. ``load()``, not ``verify()``: only a real decode
    catches a file whose header survived but whose pixel data was cut short."""
    from PIL import Image

    with Image.open(path) as handle:
        handle.load()


def save_image_verified(
    image,
    path: Path,
    *,
    attempts: int = SAVE_ATTEMPTS,
    backoff: float = SAVE_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Write ``image`` to ``path`` and prove the file is readable afterwards.

    Two failure modes, both observed on the RunPod network volume:

    * the write raises ``OSError``/EIO — retried with exponential backoff;
    * the write "succeeds" but leaves a truncated PNG — which ``--skip-existing``
      would later accept as a finished image, quietly poisoning the arm's
      evaluation set. So the file is re-opened and fully decoded, and a partial
      one is deleted rather than left behind.

    Every PIL decode error is an ``OSError`` subclass, so one except clause
    covers both.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            image.save(path)
            verify_image(path)
            return path
        except OSError as exc:
            with contextlib.suppress(OSError):
                if path.exists():
                    path.unlink()
            if attempt == attempts:
                raise OSError(f"Giving up on {path} after {attempts} attempt(s): {exc}") from exc
            delay = backoff * 2 ** (attempt - 1)
            logger.warning(
                "Saving %s failed (%s); retrying in %.1fs (attempt %d/%d)",
                path, exc, delay, attempt + 1, attempts,
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def load_image_rgb(
    path: str | Path,
    *,
    attempts: int = READ_ATTEMPTS,
    backoff: float = READ_BACKOFF_SECONDS,
    sleep: Callable[[float], None] | None = None,
) -> "object":
    """Open ``path``, decode it fully and return it as RGB, retrying on ``OSError``.

    ``Image.open`` is lazy, so the retry has to wrap the ``convert`` as well —
    that is where the bytes are actually pulled off the volume and where a
    truncated or unreadable file raises. The returned image owns its own buffer
    and the source handle is already closed, so a caller iterating thousands of
    paths leaks no file descriptors.

    The original error is re-raised once the budget is spent: a file that is
    genuinely corrupt must fail the evaluation loudly rather than be skipped.

    ``sleep`` defaults to ``time.sleep`` resolved at call time rather than as a
    default argument, so a test can neutralise the backoff of a caller several
    frames up (the embedders do not thread the parameter through).
    """
    from PIL import Image

    sleeper = time.sleep if sleep is None else sleep
    for attempt in range(1, attempts + 1):
        try:
            with Image.open(path) as handle:
                return handle.convert("RGB")
        except OSError as exc:
            if attempt == attempts:
                raise
            delay = backoff * 2 ** (attempt - 1)
            logger.warning(
                "Reading %s failed (%s); retrying in %.1fs (attempt %d/%d)",
                path, exc, delay, attempt + 1, attempts,
            )
            sleeper(delay)
    raise AssertionError("unreachable")  # pragma: no cover
