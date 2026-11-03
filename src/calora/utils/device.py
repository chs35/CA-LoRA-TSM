from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps

import torch
from torch import nn

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_device(spec: str | torch.device) -> torch.device:
    if isinstance(spec, torch.device):
        return spec
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def resolve_dtype(spec: str | torch.dtype) -> torch.dtype:
    if isinstance(spec, torch.dtype):
        return spec
    try:
        return _DTYPES[spec.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown dtype {spec!r}; expected one of {sorted(_DTYPES)}") from exc


def set_matmul_tf32(enabled: bool) -> None:
    """Allow (or forbid) TF32 tensor cores for fp32 CUDA matmuls, process-wide.

    Only the *matmul* flag is touched. cuDNN convolutions already run in TF32
    under PyTorch's default (``torch.backends.cudnn.allow_tf32 = True``), and
    every run this project has reported ran with that default, so forcing it
    off here would change the numbers of paths that never asked for a change.
    Harmless without CUDA: the flag is simply never consulted.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)


def matmul_tf32_enabled() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32)


def _cast_output(output, dtype: torch.dtype):
    """``output``'s floating tensors in ``dtype``: a tensor, a tuple of them
    (``return_dict=False``) or a diffusers output object with ``.sample``."""
    if isinstance(output, torch.Tensor):
        return output.to(dtype) if output.is_floating_point() else output
    if isinstance(output, tuple):
        return type(output)(_cast_output(item, dtype) for item in output)
    if isinstance(getattr(output, "sample", None), torch.Tensor):
        output.sample = _cast_output(output.sample, dtype)
    return output


@contextmanager
def module_forward_autocast(module: nn.Module, device: torch.device, mode: str) -> Iterator[None]:
    """Run ``module``'s forward under autocast, and nothing else in the block.

    For sampling, where a pipeline call runs the text encoders, the UNet and the
    VAE decode in one go and only the UNet should change precision. Wrapping the
    forward reaches the UNet inside the pipeline without re-implementing its loop.
    ``"no"`` touches nothing, so that path is bit-identical to not calling this.

    The output is cast back to the input's dtype. Otherwise the bf16 noise
    prediction leaks: the Euler scheduler returns latents in the model output's
    dtype, so the whole trajectory turns bf16 and the fp32 VAE refuses to decode it.
    """
    if mode == "no":
        yield
        return
    if mode != "bf16":
        raise ValueError(f"Unknown autocast mode {mode!r}; expected 'no' or 'bf16'")

    original = module.forward

    @wraps(original)
    def forward(*args, **kwargs):
        inputs = [*args, *kwargs.values()]
        dtype = next(item.dtype for item in inputs if isinstance(item, torch.Tensor))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = original(*args, **kwargs)
        return _cast_output(output, dtype)

    module.forward = forward
    try:
        yield
    finally:
        del module.forward
