#!/usr/bin/env python3
"""Occupancy trigger + K selection for progressive / 2-wave demote.

Defaults (2-wave): B=12 GiB, α=0.92, β=0.80; K = clamp(ceil((used−βB)/save), K_min, K_max).
Soft path (hf_mixed_demote): target ≈ W8_alloc + headroom; small K; fire on projected usage.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


OCC_ALPHA = 0.92
OCC_BETA = 0.80
OCC_K_MIN = 8
OCC_K_MAX = 24
OCC_EPS_GIB = 0.0  # folded into θ=αB; kept for logging


@dataclass
class LayerRank:
    demote_order: list[int]
    save_per_layer_mib: dict[int, float]
    save_mean_mib: float
    metric: str
    path: str

    @classmethod
    def load(cls, path: str | Path) -> "LayerRank":
        p = Path(path)
        raw = json.loads(p.read_text())
        order = [int(x) for x in raw["demote_order"]]
        save_raw = raw.get("save_per_layer_mib") or {}
        save = {int(k): float(v) for k, v in save_raw.items()}
        mean = float(raw.get("save_per_layer_mib_mean") or 0.0)
        if mean <= 0 and save:
            mean = sum(save.values()) / len(save)
        return cls(
            demote_order=order,
            save_per_layer_mib=save,
            save_mean_mib=mean if mean > 0 else 91.3,
            metric=str(raw.get("metric") or "unknown"),
            path=str(p),
        )


@dataclass
class OccupancyCtrl:
    """One-way lever sequencer driven by occupancy pressure."""

    budget_gib: float = 12.0
    alpha: float = OCC_ALPHA
    beta: float = OCC_BETA
    k_min: int = OCC_K_MIN
    k_max: int = OCC_K_MAX
    waves: int = 2
    rank: LayerRank | None = None
    min_fire_ctx: int = 0
    complete_wave2: bool = False
    soft: bool = False
    target_gib: float | None = None
    waves_done: int = 0
    wave1_k: int = 0
    wave1_layers: list[int] = field(default_factory=list)
    demoted_layers: list[int] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    cooldown: bool = False

    @property
    def theta_gib(self) -> float:
        return self.alpha * self.budget_gib

    @property
    def beta_gib(self) -> float:
        return self.beta * self.budget_gib

    @property
    def hold_gib(self) -> float:
        """Occupancy ceiling used for soft pressure / K planning."""
        if self.target_gib is not None:
            return float(self.target_gib)
        return self.beta_gib

    @property
    def weight_is_w4(self) -> bool:
        """True when every ranked layer has been demoted (or legacy wave done)."""
        if self.soft:
            order = self.rank.demote_order if self.rank else list(range(36))
            return len(self.demoted_layers) >= len(order)
        need = 1 if self.waves <= 1 else 2
        return self.waves_done >= need

    @property
    def short_phase(self) -> bool:
        return not self.weight_is_w4

    def pressure(self, used_gib: float) -> bool:
        if self.soft:
            return used_gib > self.hold_gib
        return used_gib >= self.theta_gib

    def plan_k(self, used_gib: float) -> int:
        save = self.rank.save_mean_mib if self.rank else 91.3
        if save <= 0:
            save = 91.3
        ceiling = self.hold_gib if (self.soft or self.target_gib is not None) else self.beta_gib
        deficit_mib = max(0.0, (used_gib - ceiling) * 1024.0)
        k = int(math.ceil(deficit_mib / save)) if deficit_mib > 0 else self.k_min
        n = len(self.rank.demote_order) if self.rank else 36
        rem = max(0, n - len(self.demoted_layers))
        hi = min(self.k_max, max(self.k_min, rem if rem > 0 else self.k_min))
        return max(self.k_min, min(hi, k, rem if rem > 0 else k))

    def wave1_layers_for(self, k: int) -> list[int]:
        order = self.rank.demote_order if self.rank else list(range(36))
        return list(order[:k])

    def residual_layers(self) -> list[int]:
        order = self.rank.demote_order if self.rank else list(range(36))
        done = set(self.wave1_layers) | set(self.demoted_layers)
        return [lid for lid in order if lid not in done]

    def remaining_layers(self) -> list[int]:
        order = self.rank.demote_order if self.rank else list(range(36))
        done = set(self.demoted_layers)
        return [lid for lid in order if lid not in done]

    def maybe_fire(self, used_gib: float, *, ctx_len: int) -> dict[str, Any] | None:
        """Fire at most one lever. Returns event dict or None."""
        if ctx_len < self.min_fire_ctx:
            return None

        if self.soft:
            return self._maybe_fire_soft(used_gib, ctx_len=ctx_len)

        if self.complete_wave2 and self.waves >= 2 and self.waves_done == 1:
            rest = self.residual_layers()
            self.waves_done = 2
            if not rest:
                return None
            self.demoted_layers.extend(rest)
            ev = {
                "lever": "inplace_wave2_sched",
                "wave": 2,
                "at_ctx": ctx_len,
                "used_gib": round(used_gib, 3),
                "theta_gib": round(self.theta_gib, 3),
                "k": len(rest),
                "layers": rest,
            }
            self.events.append(ev)
            return ev
        if self.cooldown:
            self.cooldown = False
            return None
        if not self.pressure(used_gib):
            return None
        if self.waves <= 1:
            if self.waves_done >= 1:
                return None
            self.waves_done = 1
            order = self.rank.demote_order if self.rank else list(range(36))
            self.wave1_k = len(order)
            self.wave1_layers = list(order)
            self.demoted_layers = list(order)
            ev = {
                "lever": "inplace_full",
                "wave": 1,
                "at_ctx": ctx_len,
                "used_gib": round(used_gib, 3),
                "theta_gib": round(self.theta_gib, 3),
                "k": self.wave1_k,
                "layers": list(self.wave1_layers),
            }
            self.events.append(ev)
            self.cooldown = True
            return ev

        if self.waves_done == 0:
            k = self.plan_k(used_gib)
            layers = self.wave1_layers_for(k)
            self.wave1_k = k
            self.wave1_layers = layers
            self.demoted_layers.extend(layers)
            self.waves_done = 1
            ev = {
                "lever": "inplace_wave1",
                "wave": 1,
                "at_ctx": ctx_len,
                "used_gib": round(used_gib, 3),
                "theta_gib": round(self.theta_gib, 3),
                "beta_gib": round(self.beta_gib, 3),
                "k": k,
                "layers": layers,
                "save_mean_mib": round(self.rank.save_mean_mib, 2) if self.rank else None,
            }
            self.events.append(ev)
            self.cooldown = True
            return ev

        if self.waves_done == 1:
            rest = self.residual_layers()
            self.waves_done = 2
            self.demoted_layers.extend(rest)
            ev = {
                "lever": "inplace_wave2",
                "wave": 2,
                "at_ctx": ctx_len,
                "used_gib": round(used_gib, 3),
                "theta_gib": round(self.theta_gib, 3),
                "k": len(rest),
                "layers": rest,
            }
            self.events.append(ev)
            self.cooldown = True
            return ev

        return None

    def _maybe_fire_soft(self, used_gib: float, *, ctx_len: int) -> dict[str, Any] | None:
        """Small demotes until projected usage returns toward hold_gib."""
        rem = self.remaining_layers()
        if not rem:
            self.waves_done = 2
            return None
        if not self.pressure(used_gib):
            return None
        k = self.plan_k(used_gib)
        k = max(1, min(k, len(rem)))
        layers = rem[:k]
        self.demoted_layers.extend(layers)
        if not self.wave1_layers:
            self.wave1_layers = list(layers)
            self.wave1_k = len(layers)
        self.waves_done = 2 if not self.remaining_layers() else 1
        ev = {
            "lever": "soft_demote",
            "wave": self.waves_done,
            "at_ctx": ctx_len,
            "used_gib": round(used_gib, 3),
            "target_gib": round(self.hold_gib, 3),
            "theta_gib": round(self.theta_gib, 3),
            "k": len(layers),
            "layers": layers,
            "n_demoted_total": len(self.demoted_layers),
            "n_remaining": len(self.remaining_layers()),
            "save_mean_mib": round(self.rank.save_mean_mib, 2) if self.rank else None,
        }
        self.events.append(ev)
        return ev

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_gib": self.budget_gib,
            "alpha": self.alpha,
            "beta": self.beta,
            "theta_gib": self.theta_gib,
            "beta_gib": self.beta_gib,
            "target_gib": self.target_gib,
            "hold_gib": self.hold_gib,
            "soft": self.soft,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "waves": self.waves,
            "min_fire_ctx": self.min_fire_ctx,
            "complete_wave2": self.complete_wave2,
            "waves_done": self.waves_done,
            "wave1_k": self.wave1_k,
            "wave1_layers": list(self.wave1_layers),
            "demoted_layers": list(self.demoted_layers),
            "events": list(self.events),
            "rank_path": self.rank.path if self.rank else None,
            "rank_metric": self.rank.metric if self.rank else None,
            "save_mean_mib": self.rank.save_mean_mib if self.rank else None,
        }
