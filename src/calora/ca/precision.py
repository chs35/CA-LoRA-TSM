"""The numeric regime of a CA measurement (``measure.precision``).

Every earlier selection was measured in full fp32 with TF32 off -- on an A100
that leaves the tensor cores idle. This module names the three regimes so a
measurement states which one it ran in:

``fp32``  TF32 off. The reference.
``tf32``  TF32 tensor-core matmuls; weights, activations and grads stay fp32.
``bf16``  bf16 autocast over the forward (TF32 for what autocast leaves fp32).
          The CA statistic is still read off the fp32 ``param.grad``.

Nothing but ``fp32`` is the default: switching is decided by the pre-fixed gate
of scripts/ops/check_measure_precision.py, and the regime is recorded in every
measurement's provenance.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from collections.abc import Iterator

import torch

from calora.config import MEASURE_PRECISIONS
from calora.utils.device import matmul_tf32_enabled, set_matmul_tf32


@contextmanager
def measurement_precision(device: torch.device | str, precision: str) -> Iterator[None]:
    """Run the enclosed measurement in ``precision``; restore the TF32 flag after."""
    if precision not in MEASURE_PRECISIONS:
        raise ValueError(
            f"Unknown measure.precision {precision!r}; expected one of {list(MEASURE_PRECISIONS)}"
        )
    device = torch.device(device)
    previous = matmul_tf32_enabled()
    set_matmul_tf32(precision != "fp32")
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )
    try:
        with autocast:
            yield
    finally:
        set_matmul_tf32(previous)
