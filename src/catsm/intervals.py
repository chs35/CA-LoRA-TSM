"""Timestep intervals and expert naming (design §3.2–3.3).

An expert set ``n<k>`` partitions the training timesteps ``[0, T-1]`` into ``k``
contiguous, inclusive intervals, lowest t first. Boundaries are raw timesteps;
the log-SNR of each boundary is recorded alongside so the same definition can be
restated on another backbone (design §3.3), but nothing routes on it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from catsm.config import IntervalsConfig, set_size

#: ``<set>_i<index>``: ``n4_i0`` is the lowest interval of the four-way scheme.
EXPERT_NAME = re.compile(r"^(n[1-9]\d*)_i(\d+)$")


@dataclass(frozen=True)
class Interval:
    """Inclusive raw-timestep range ``[lo, hi]``."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if not (0 <= self.lo <= self.hi):
            raise ValueError(f"Interval needs 0 <= lo <= hi, got [{self.lo}, {self.hi}]")

    def __contains__(self, t: object) -> bool:
        return isinstance(t, int) and self.lo <= t <= self.hi

    @property
    def length(self) -> int:
        return self.hi - self.lo + 1

    def as_list(self) -> list[int]:
        return [self.lo, self.hi]


@dataclass(frozen=True)
class IntervalScheme:
    name: str
    intervals: tuple[Interval, ...]

    def __post_init__(self) -> None:
        set_size(self.name)
        if len(self.intervals) != set_size(self.name):
            raise ValueError(
                f"Scheme {self.name!r} declares {set_size(self.name)} intervals but has "
                f"{len(self.intervals)}"
            )

    @property
    def n(self) -> int:
        return len(self.intervals)

    def expert_name(self, index: int) -> str:
        if not 0 <= index < self.n:
            raise IndexError(f"{self.name} has intervals 0..{self.n - 1}, not {index}")
        return f"{self.name}_i{index}"

    def expert_names(self) -> list[str]:
        return [self.expert_name(i) for i in range(self.n)]

    def interval_of(self, expert: str) -> Interval:
        set_name, index = parse_expert_name(expert)
        if set_name != self.name:
            raise ValueError(f"Expert {expert!r} does not belong to scheme {self.name!r}")
        return self.intervals[index]

    def owner(self, t: int) -> int:
        """Index of the interval containing ``t``."""
        for i, interval in enumerate(self.intervals):
            if t in interval:
                return i
        raise ValueError(f"t={t} is outside every interval of {self.name}")

    def lookup(self, num_train_timesteps: int) -> torch.Tensor:
        """``[T]`` int64: interval index per timestep, for batched routing."""
        validate_partition(self.intervals, num_train_timesteps)
        table = torch.empty(num_train_timesteps, dtype=torch.long)
        for i, interval in enumerate(self.intervals):
            table[interval.lo : interval.hi + 1] = i
        return table


def parse_expert_name(name: str) -> tuple[str, int]:
    """``"n4_i2" -> ("n4", 2)``."""
    match = EXPERT_NAME.match(name)
    if match is None:
        raise ValueError(f"Expert name must look like 'n4_i2', got {name!r}")
    set_name, index = match.group(1), int(match.group(2))
    if index >= set_size(set_name):
        raise ValueError(f"{name!r}: interval index {index} out of range for {set_name}")
    return set_name, index


def validate_partition(intervals: Sequence[Interval], num_train_timesteps: int) -> None:
    """Contiguous, non-overlapping, and covering exactly ``[0, T-1]``."""
    if not intervals:
        raise ValueError("A scheme needs at least one interval")
    if intervals[0].lo != 0:
        raise ValueError(f"First interval must start at 0, got {intervals[0]}")
    if intervals[-1].hi != num_train_timesteps - 1:
        raise ValueError(
            f"Last interval must end at {num_train_timesteps - 1}, got {intervals[-1]}"
        )
    for prev, nxt in zip(intervals, intervals[1:]):
        if nxt.lo != prev.hi + 1:
            raise ValueError(f"Intervals {prev} and {nxt} are not contiguous")


def halving_intervals(n: int, num_train_timesteps: int) -> tuple[Interval, ...]:
    """Split ``[0, T-1]`` by halving from the top, ``n`` pieces (design §3.3).

    ``T=1000``: n=3 -> (500, 250, 250), n=4 -> (500, 250, 125, 125),
    n=5 -> (500, 250, 125, 63, 62) — the lowest two intervals are ``[0, 61]``
    and ``[62, 124]``, which is what separates t=41 from t=81.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    starts = [num_train_timesteps >> k for k in range(1, n)]
    if any(s < 1 for s in starts) or len(set(starts)) != len(starts):
        raise ValueError(f"Cannot halve {num_train_timesteps} timesteps into {n} intervals")
    return intervals_from_starts(sorted(starts), num_train_timesteps)


def intervals_from_starts(starts: Sequence[int], num_train_timesteps: int) -> tuple[Interval, ...]:
    """Interior interval starts (ascending) -> the full partition."""
    starts = [int(s) for s in starts]
    if any(not 1 <= s <= num_train_timesteps - 1 for s in starts):
        raise ValueError(f"Interval starts must lie in [1, {num_train_timesteps - 1}], got {starts}")
    if starts != sorted(set(starts)):
        raise ValueError(f"Interval starts must be strictly increasing, got {starts}")
    edges = [0, *starts, num_train_timesteps]
    intervals = tuple(Interval(lo, hi - 1) for lo, hi in zip(edges, edges[1:]))
    validate_partition(intervals, num_train_timesteps)
    return intervals


def build_scheme(
    set_name: str, num_train_timesteps: int, cfg: IntervalsConfig | None = None
) -> IntervalScheme:
    """The scheme for ``set_name``: a config override if present, else halving."""
    n = set_size(set_name)
    if cfg is not None and set_name in cfg.boundaries:
        intervals = intervals_from_starts(cfg.boundaries[set_name], num_train_timesteps)
    else:
        intervals = halving_intervals(n, num_train_timesteps)
    return IntervalScheme(set_name, intervals)


def logsnr(alphas_cumprod: torch.Tensor, t: int) -> float:
    r"""``log(ᾱ_t / (1 − ᾱ_t))`` in float64. Informational only."""
    a = float(alphas_cumprod.detach().to("cpu", torch.float64)[t])
    return math.log(a) - math.log1p(-a)


def scheme_record(scheme: IntervalScheme, alphas_cumprod: torch.Tensor | None = None) -> dict:
    """JSON-ready description of a scheme, with log-SNR at every boundary."""
    intervals = []
    for i, interval in enumerate(scheme.intervals):
        entry: dict = {
            "expert": scheme.expert_name(i),
            "t_lo": interval.lo,
            "t_hi": interval.hi,
            "length": interval.length,
        }
        if alphas_cumprod is not None:
            entry["logsnr_lo"] = logsnr(alphas_cumprod, interval.lo)
            entry["logsnr_hi"] = logsnr(alphas_cumprod, interval.hi)
        intervals.append(entry)
    return {"scheme": scheme.name, "n": scheme.n, "intervals": intervals}
