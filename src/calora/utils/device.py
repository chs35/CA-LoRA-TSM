from __future__ import annotations

import torch

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
