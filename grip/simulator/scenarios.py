"""Scenarios mutate each SimGpu's health as a deterministic function of sim time.

healthy                - nothing drifts
gpu_local_degradation  - one GPU slowly loses clock + perf/watt, runs hotter
shared_env             - a power/cooling domain drifts together
hard_failure           - one GPU throws XID + ECC-DBE + HW throttle (contrast)
"""

from __future__ import annotations

from collections.abc import Callable

from .model import SimGpu


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _ramp(t_s: float, start: float, end: float) -> float:
    return _smoothstep((t_s - start) / max(1.0, end - start))


def healthy(gpus: list[SimGpu], t_s: float, p: dict) -> None:
    return


def gpu_local_degradation(gpus: list[SimGpu], t_s: float, p: dict) -> None:
    target = int(p.get("target", 6))
    prog = _ramp(t_s, p.get("ramp_start_s", 300), p.get("ramp_end_s", 3600))
    h = gpus[target].health
    h.perf_mult = 1.0 - 0.11 * prog
    h.efficiency_mult = 1.0 - 0.14 * prog
    h.thermal_offset_c = 9.0 * prog
    h.nccl_mult = 1.0 - 0.05 * prog


def shared_env(gpus: list[SimGpu], t_s: float, p: dict) -> None:
    domain = set(p.get("domain", [4, 5, 6, 7]))
    prog = _ramp(t_s, p.get("ramp_start_s", 300), p.get("ramp_end_s", 3000))
    for g in gpus:
        h = g.health
        if g.index in domain:
            h.thermal_offset_c = 12.0 * prog
            h.efficiency_mult = 1.0 - 0.09 * prog
            h.perf_mult = 1.0 - 0.03 * prog
            h.fan_pct = 60.0 + 40.0 * prog
        else:
            h.thermal_offset_c = 1.5 * prog  # shared inlet, no real drift


def hard_failure(gpus: list[SimGpu], t_s: float, p: dict) -> None:
    target = int(p.get("target", 3))
    at = float(p.get("at_s", 1200))
    g = gpus[target]
    if t_s >= at:
        g.health.force_throttle = ["hw_slowdown"]
        g.health.perf_mult = 0.55
        if g.dbe_total == 0 and not g.health.xid_burst:
            g.health.xid_burst = [79]
            g.health.dbe_events = 2


SCENARIOS: dict[str, Callable[[list[SimGpu], float, dict], None]] = {
    "healthy": healthy,
    "gpu_local_degradation": gpu_local_degradation,
    "shared_env": shared_env,
    "hard_failure": hard_failure,
}
