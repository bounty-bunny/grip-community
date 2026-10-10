"""Synthetic GPU model. A scenario mutates GpuHealth; sample() renders telemetry.

Design intent for the demo: in gradual scenarios the *work* falls (clock, TFLOPS)
while *power* stays flat -> perf/watt drifts down. Nothing crosses a hard-failure
threshold, so conventional monitoring stays green.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..schema import GpuSample, GpuSignals, ProbeKind, ProbeResult, Source, ThrottleReason


@dataclass
class GpuNominal:
    sm_clock_mhz: float = 1830.0
    mem_clock_mhz: float = 2619.0
    temp_c: float = 61.0
    power_w: float = 430.0
    power_limit_w: float = 700.0
    util_pct: float = 98.0
    sm_active_pct: float = 92.0
    mem_total_mb: float = 81920.0
    mem_used_mb: float = 68000.0
    gemm_tflops: float = 750.0
    nccl_gbps: float = 220.0


@dataclass
class GpuHealth:
    perf_mult: float = 1.0  # scales clock and TFLOPS
    efficiency_mult: float = 1.0  # extra TFLOPS loss at unchanged power
    thermal_offset_c: float = 0.0
    nccl_mult: float = 1.0
    extra_sbe_per_min: float = 0.0
    extra_replay_per_min: float = 0.0
    force_throttle: list[str] = field(default_factory=list)
    fan_pct: float | None = None
    xid_burst: list[int] = field(default_factory=list)
    dbe_events: int = 0


@dataclass
class SimGpu:
    index: int
    uuid: str
    nominal: GpuNominal = field(default_factory=GpuNominal)
    health: GpuHealth = field(default_factory=GpuHealth)
    sbe_total: int = 0
    dbe_total: int = 0
    replay_total: int = 0


def _n(rng: np.random.Generator, sigma: float) -> float:
    return float(rng.normal(0.0, sigma))


def sample(
    g: SimGpu,
    node_id: str,
    ts: datetime,
    dt_s: float,
    rng: np.random.Generator,
    with_probe: bool,
) -> GpuSample:
    nom, h = g.nominal, g.health

    util = float(np.clip(nom.util_pct + _n(rng, 0.6), 0, 100))
    sm_active = float(np.clip(nom.sm_active_pct + _n(rng, 1.0), 0, 100))
    temp = nom.temp_c + h.thermal_offset_c + _n(rng, 0.35)

    throttle = list(h.force_throttle)
    if temp > 86 and "sw_thermal" not in throttle:
        throttle.append("sw_thermal")
    if temp > 90 and "thermal" not in throttle:
        throttle.append("thermal")

    clock = nom.sm_clock_mhz * h.perf_mult + _n(rng, 4.0)
    if "thermal" in throttle or "hw_slowdown" in throttle:
        clock *= 0.82
    elif "sw_thermal" in throttle:
        clock *= 0.94

    power = nom.power_w * (0.85 + 0.15 * util / 100.0) + _n(rng, 4.0)
    if "power" in throttle:
        power = min(power, nom.power_w * 0.80)
    if "hw_slowdown" in throttle:
        power *= 0.5
    power = float(np.clip(power, 40.0, nom.power_limit_w))

    # monotonic counters
    g.sbe_total += int(rng.poisson(max(0.0, h.extra_sbe_per_min) * dt_s / 60.0))
    g.replay_total += int(rng.poisson(max(0.0, h.extra_replay_per_min) * dt_s / 60.0))

    xids: list[int] = []
    if h.xid_burst and g.dbe_total == 0:
        xids = list(h.xid_burst)
        g.dbe_total += h.dbe_events
        h.xid_burst = []
        h.dbe_events = 0

    fan = h.fan_pct if h.fan_pct is not None else float(np.clip(35 + (temp - 60) * 2.5, 20, 100))

    signals = GpuSignals(
        sm_clock_mhz=round(clock, 1),
        mem_clock_mhz=round(nom.mem_clock_mhz + _n(rng, 2.0), 1),
        temperature_c=round(temp, 2),
        memory_temperature_c=round(temp + 8 + _n(rng, 0.5), 2),
        power_draw_w=round(power, 1),
        power_limit_w=nom.power_limit_w,
        gpu_utilization_pct=round(util, 1),
        sm_activity_pct=round(sm_active, 1),
        memory_used_mb=round(nom.mem_used_mb + _n(rng, 120), 0),
        fan_speed_pct=round(fan, 1),
        pcie_replay_count=g.replay_total,
        nvlink_crc_errors=0,
        ecc_sbe_total=g.sbe_total,
        ecc_dbe_total=g.dbe_total,
        xid_errors=xids,
        throttle_reasons=[ThrottleReason(t) for t in throttle],
    )

    probe = None
    if with_probe:
        eff = h.perf_mult * h.efficiency_mult
        tflops = nom.gemm_tflops * eff + _n(rng, 4.0)
        if "thermal" in throttle or "hw_slowdown" in throttle:
            tflops *= 0.80
        elif "sw_thermal" in throttle:
            tflops *= 0.93
        gbps = nom.nccl_gbps * h.nccl_mult + _n(rng, 3.0)
        probe = ProbeResult(
            kind=ProbeKind.gemm,
            duration_ms=round(900.0 * nom.gemm_tflops / max(tflops, 1.0), 1),
            gemm_tflops=round(max(tflops, 1.0), 1),
            nccl_ring_gbps=round(max(gbps, 1.0), 1),
            matrix_n=16384,
            dtype="bf16",
        )

    return GpuSample(
        source=Source.simulator,
        node_id=node_id,
        gpu_index=g.index,
        gpu_uuid=g.uuid,
        ts=ts,
        signals=signals,
        probe=probe,
    )
