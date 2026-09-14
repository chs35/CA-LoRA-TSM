"""Assembling-stage routing (draft §3.2 step 4).

At timestep ``t`` every expert set contributes exactly one expert — the one
whose interval contains ``t``. The finest set (``core``) enters with weight 1;
each other set (``context``) is scaled by the raw router output
``FC(mean z_t) + E(t)`` — no activation, so a context expert can be attenuated,
amplified or even subtracted per matrix. Not softmax: with a single context set
(e.g. No.3 = n1 + n4) a softmax would be identically 1. No bound or sign
constraint either: that would only encode a prior the experiment is meant to test.

The gate is **per projection matrix**, as in TimeStep Master: every attention
projection that carries a context expert has its own weight for that set, so
``to_q`` of one block can lean on the coarse expert while ``to_out.0`` of another
ignores it. The router therefore knows the ordered list of gated modules and
emits ``(B, M, K)`` gate weights; the gate mapping is keyed ``(module, expert)`` for
context experts and ``expert`` (global) for the core set and for hard routing.

The router is a separate module, never a UNet submodule, so it can neither leak
into ``lora_state_dict`` nor match ``PROJECTION_PATTERN``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from catsm.intervals import IntervalScheme, validate_partition


class Router(nn.Module):
    """``gate[b, m, k] = fc(mean_hw(z_t))[m, k] + t_embed[t][m, k]`` in fp32.

    ``modules`` are the projection matrices that carry at least one context
    expert, in a fixed order; ``k`` indexes ``context_sets``.
    """

    def __init__(
        self,
        context_sets: Sequence[str],
        modules: Sequence[str],
        *,
        latent_channels: int,
        num_train_timesteps: int,
        init_gate: float = 1.0,
    ):
        super().__init__()
        self.context_sets = tuple(context_sets)
        self.modules_gated = tuple(modules)
        if not self.context_sets:
            raise ValueError("A router needs at least one context expert set")
        if len(set(self.context_sets)) != len(self.context_sets):
            raise ValueError(f"Duplicate context sets: {self.context_sets}")
        if not self.modules_gated:
            raise ValueError("A router needs at least one gated module")
        if len(set(self.modules_gated)) != len(self.modules_gated):
            raise ValueError("Duplicate module names in the router")
        self.latent_channels = int(latent_channels)
        self.num_train_timesteps = int(num_train_timesteps)
        self.init_gate = float(init_gate)
        self.num_modules = len(self.modules_gated)
        self.num_sets = len(self.context_sets)
        width = self.num_modules * self.num_sets
        self.fc = nn.Linear(self.latent_channels, width)
        self.t_embed = nn.Embedding(self.num_train_timesteps, width)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, self.init_gate)
        nn.init.zeros_(self.t_embed.weight)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``(B, C, ...)``, ``(B,)`` -> ``(B, M, K)`` gate weights (raw, any sign)."""
        if z_t.dim() < 2 or z_t.shape[1] != self.latent_channels:
            raise ValueError(
                f"z_t must be (B, {self.latent_channels}, ...), got {tuple(z_t.shape)}"
            )
        pooled = z_t.float().mean(dim=tuple(range(2, z_t.dim()))) if z_t.dim() > 2 else z_t.float()
        flat = self.fc(pooled) + self.t_embed(t.long().to(pooled.device))
        return flat.view(-1, self.num_modules, self.num_sets)

    def describe(self) -> dict:
        return {
            "type": "fc_z_plus_e_t",
            "gate": "identity",
            "scope": "per_module",
            "context_sets": list(self.context_sets),
            "modules": list(self.modules_gated),
            "latent_channels": self.latent_channels,
            "num_train_timesteps": self.num_train_timesteps,
            "init_gate": self.init_gate,
        }


class RoutingTable:
    """Which expert of each set is active at ``t``, and with what weight."""

    def __init__(
        self,
        schemes: Mapping[str, IntervalScheme],
        *,
        core: str,
        num_train_timesteps: int,
        disabled: Sequence[str] = (),
    ):
        if core not in schemes:
            raise ValueError(f"core={core!r} is not one of the sets {sorted(schemes)}")
        for name, scheme in schemes.items():
            if scheme.name != name:
                raise ValueError(f"Scheme {scheme.name!r} registered under {name!r}")
            validate_partition(scheme.intervals, num_train_timesteps)
        self.schemes = {name: schemes[name] for name in sorted(schemes)}
        self.core = core
        self.num_train_timesteps = int(num_train_timesteps)
        self.sets = list(self.schemes)
        self.context_sets = [name for name in self.sets if name != core]
        self._lookup = {
            name: scheme.lookup(num_train_timesteps) for name, scheme in self.schemes.items()
        }
        #: Experts whose weight is forced to 0 at every t (draft §5.1 control C2:
        #: "high-interval expert disabled"). Their interval is then covered by
        #: whatever other sets are active there, or by the base model alone.
        unknown = sorted(set(disabled) - set(self.expert_names))
        if unknown:
            raise ValueError(f"disabled experts {unknown} are not in {self.expert_names}")
        if len(set(disabled)) == len(self.expert_names) and self.expert_names:
            raise ValueError("Every expert is disabled; nothing would be routed")
        self.disabled = sorted(set(disabled))

    @property
    def expert_names(self) -> list[str]:
        return sorted(e for scheme in self.schemes.values() for e in scheme.expert_names())

    def expert_of(self, set_name: str, t: int) -> str:
        scheme = self.schemes[set_name]
        return scheme.expert_name(scheme.owner(t))

    def active_experts(self, t: int) -> dict[str, str]:
        """Set name -> the expert active at ``t``."""
        return {name: self.expert_of(name, t) for name in self.sets}

    def gate(
        self,
        t: torch.Tensor,
        weights: torch.Tensor | None = None,
        *,
        modules: Sequence[str] | None = None,
    ) -> dict[str | tuple[str, str], torch.Tensor]:
        """Per-sample weights ``(B,)`` for every expert (zeros where inactive).

        Hard routing (``weights is None``): one global entry per expert, keyed by
        its name. A disabled expert gets an all-zero entry either way. Soft routing: the core set stays global; every context expert
        gets one entry per gated module, keyed ``(module, expert)`` and scaled
        by the router output ``weights[:, m, k]`` as is. ``modules`` must then be
        the router's ordered module list.
        """
        t = t.long()
        if t.dim() != 1:
            raise ValueError(f"t must be a (B,) tensor, got {tuple(t.shape)}")
        if weights is not None:
            if modules is None:
                raise ValueError("Soft routing needs the router's module list")
            expected = (t.shape[0], len(modules), len(self.context_sets))
            if tuple(weights.shape) != expected:
                raise ValueError(
                    f"router weights must be (B, M, K)={expected}, got {tuple(weights.shape)}"
                )
            weights = weights.float().to(t.device)
        out: dict[str | tuple[str, str], torch.Tensor] = {}
        for name, scheme in self.schemes.items():
            owners = self._lookup[name].to(t.device)[t]
            soft = weights is not None and name != self.core
            for i in range(scheme.n):
                expert = scheme.expert_name(i)
                mask = (owners == i).to(torch.float32)
                if expert in self.disabled:
                    mask = torch.zeros_like(mask)
                if not soft:
                    out[expert] = mask
                    continue
                scaled = mask[:, None] * weights[:, :, self.context_sets.index(name)]  # (B, M)
                for m, module in enumerate(modules):
                    out[(module, expert)] = scaled[:, m]
        return out
