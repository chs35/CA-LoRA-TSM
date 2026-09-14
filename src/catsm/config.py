"""Configuration: calora's schema plus the three sections this project adds.

``calora.config.load_config`` is bound to ``calora.config.Config`` and rejects
unknown keys, so the extended dataclass gets its own loader. Everything else —
``_base_`` chains, dotted CLI overrides, ``save_config`` — is reused as is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from calora.config import Config, _deep_merge, build_dataclass, load_config_dict

SELECT_MODES = ("representative", "aggregate", "fixed")
#: Expert-set names: ``n<number of intervals>``.
SET_NAME = re.compile(r"^n([1-9]\d*)$")


def set_size(name: str) -> int:
    """``"n4" -> 4``."""
    match = SET_NAME.match(name)
    if match is None:
        raise ValueError(f"Expert-set name must look like 'n4', got {name!r}")
    return int(match.group(1))


@dataclass
class IntervalsConfig:
    """Which interval schemes exist and how their boundaries are defined.

    Boundaries default to halving from the top (draft §3.3); ``boundaries``
    overrides a scheme's interior interval starts, e.g. ``{"n4": [125, 250, 500]}``,
    once the §4 diagnosis has fixed them.
    """

    schemes: list[int] = field(default_factory=lambda: [1, 3, 4, 5])
    boundaries: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.schemes:
            raise ValueError("intervals.schemes must not be empty")
        bad = [n for n in self.schemes if not isinstance(n, int) or isinstance(n, bool) or n < 1]
        if bad:
            raise ValueError(f"intervals.schemes must be positive integers, got {bad}")
        if len(set(self.schemes)) != len(self.schemes):
            raise ValueError(f"intervals.schemes contains duplicates: {self.schemes}")
        for name, starts in self.boundaries.items():
            n = set_size(name)
            if len(starts) != n - 1:
                raise ValueError(
                    f"intervals.boundaries[{name!r}] needs {n - 1} interior start(s), got {starts}"
                )


@dataclass
class IntervalSelectConfig:
    """How each interval's CA selection is made (draft §3.2 (A)/(B) and control C1)."""

    #: ``representative`` = one measured t per interval (A); ``aggregate`` = pool
    #: every measured t inside the interval (B); ``fixed`` = every interval uses
    #: ``fixed_timestep`` (control C1: selection fixed, training per interval).
    mode: str = "aggregate"
    #: Scheme name -> one t per interval, lowest interval first (``representative``).
    representative: dict[str, list[int]] = field(default_factory=dict)
    fixed_timestep: int = 81

    def __post_init__(self) -> None:
        if self.mode not in SELECT_MODES:
            raise ValueError(
                f"interval_select.mode must be one of {SELECT_MODES}, got {self.mode!r}"
            )
        for name, ts in self.representative.items():
            n = set_size(name)
            if len(ts) != n:
                raise ValueError(
                    f"interval_select.representative[{name!r}] needs {n} timestep(s), got {ts}"
                )


@dataclass
class AssembleConfig:
    """Router training over frozen experts (draft §3.2 step 4).

    The optimisation fields mirror :class:`calora.config.TrainConfig` by name so
    the reference loop helpers (index/flip draws, LR schedule, latent cache) can
    take this object where they expect a ``TrainConfig``.
    """

    sets: list[str] = field(default_factory=lambda: ["n1", "n4"])
    #: The finest scheme; its expert carries fixed weight 1, the others are gated.
    core: str = "n4"
    max_steps: int = 1000
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    #: Initial gate value of every context expert (the router's bias; no activation).
    #: 1.0 = start from the plain sum of every active expert; 0.0 = start core-only.
    init_gate: float = 1.0
    mixed_precision: str = "no"
    cache_latents: bool = True
    prefetch: bool = True
    seed: int = 0
    checkpoint_every: int = 0
    log_every: int = 50

    def __post_init__(self) -> None:
        if not self.sets:
            raise ValueError("assemble.sets must not be empty")
        for name in self.sets:
            set_size(name)
        if len(set(self.sets)) != len(self.sets):
            raise ValueError(f"assemble.sets contains duplicates: {self.sets}")
        if self.core not in self.sets:
            raise ValueError(f"assemble.core={self.core!r} is not one of assemble.sets={self.sets}")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1 or self.max_steps < 1:
            raise ValueError("assemble.max_steps, batch_size, gradient_accumulation_steps must be >= 1")


@dataclass
class CatsmConfig(Config):
    intervals: IntervalsConfig = field(default_factory=IntervalsConfig)
    interval_select: IntervalSelectConfig = field(default_factory=IntervalSelectConfig)
    assemble: AssembleConfig = field(default_factory=AssembleConfig)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> CatsmConfig:
    data = load_config_dict(path)
    if overrides:
        data = _deep_merge(data, overrides)
    return build_dataclass(CatsmConfig, data)
