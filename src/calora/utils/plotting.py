"""Plots of the artefacts a run leaves behind. Currently only the loss curve.

Deliberately outside the training loop: a plot is a reporting convenience, and a
missing or broken matplotlib must never be able to fail a 10k-step arm whose
weights and ``train_summary.json`` are already on disk. Every entry point here
therefore degrades to ``None`` rather than raising.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from calora.utils.logging import get_logger

logger = get_logger(__name__)

#: Slots 1 and 2 of the reference categorical palette (light surface).
_LOSS_COLOR = "#2a78d6"
_LR_COLOR = "#eb6834"


def plot_loss_curve(history: Sequence[dict], path: str | Path) -> Path | None:
    """Render loss vs step from :attr:`TrainingState.history` into a PNG.

    ``history`` entries are ``{"step", "loss", "learning_rate", "elapsed"}``.
    Returns the written path, or ``None`` when there is nothing to draw or
    matplotlib is unavailable.

    The learning rate gets its own panel only when it actually varies: under the
    constant scheduler a flat line is noise. It is a second panel rather than a
    second y-axis on the loss plot because two scales sharing one frame invite
    reading a crossing point as meaningful.
    """
    if not history:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: the pod has no display, nor does CI
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping the loss curve")
        return None

    steps = [int(entry["step"]) for entry in history]
    losses = [float(entry["loss"]) for entry in history]
    rates = [float(entry.get("learning_rate", 0.0)) for entry in history]
    lr_varies = len(set(rates)) > 1

    if lr_varies:
        figure, (loss_axis, lr_axis) = plt.subplots(
            2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": (3, 1)}
        )
    else:
        figure, loss_axis = plt.subplots(figsize=(8, 4))
        lr_axis = None

    subtitle = f"{steps[-1]} steps" if lr_varies else f"{steps[-1]} steps, lr {rates[-1]:.2e}"
    loss_axis.plot(steps, losses, color=_LOSS_COLOR, linewidth=1.5)
    loss_axis.set_ylabel("diffusion loss")
    loss_axis.set_title(f"Training loss ({subtitle})")

    if lr_axis is not None:
        lr_axis.plot(steps, rates, color=_LR_COLOR, linewidth=1.5)
        lr_axis.set_ylabel("learning rate")

    bottom = lr_axis if lr_axis is not None else loss_axis
    bottom.set_xlabel("step")
    for axis in (loss_axis, lr_axis):
        if axis is None:
            continue
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=150)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        logger.warning("Could not write the loss curve to %s: %s", path, exc)
        return None
    finally:
        plt.close(figure)
    return path
