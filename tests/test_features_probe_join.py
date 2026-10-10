"""Probe-to-power join (F-003).

A probe measures over about a second. Pairing it with a single *nearest* power
sample - unbounded in time, and possibly landing on a transient - produced
perf/watt swings of tens of percent between healthy peers on real hardware,
which is the highest-weighted signal in the score.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from grip.engine.features import build_window

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _tel(offsets_s, power_w):
    """Telemetry rows at the given second-offsets before T0."""
    return [
        {
            "ts": T0 - timedelta(seconds=s),
            "sm_clock_mhz": 1400.0,
            "temperature_c": 60.0,
            "power_draw_w": p,
            "sm_activity_pct": 100.0,
            "ecc_sbe_total": 0,
            "ecc_dbe_total": 0,
            "pcie_replay_count": 0,
            "gpu_uuid": "GPU-x",
            "throttle_reasons": [],
            "xid_errors": [],
        }
        for s, p in zip(offsets_s, power_w, strict=True)
    ]


def _probe(offset_s, tflops):
    return {"ts": T0 - timedelta(seconds=offset_s), "gemm_tflops": tflops, "nccl_ring_gbps": None}


def _eff(rows, probes):
    w = build_window("n", 0, rows, probes, T0)
    assert w is not None
    return w.series.get("power_efficiency"), w.efficiency_is_proxy


def test_power_denominator_is_the_median_around_the_probe():
    """One transient power reading must not set the whole ratio."""
    secs = list(range(60, 0, -10))
    # a single 40 W spike among 400 W samples, right where a probe lands
    watts = [400.0, 400.0, 40.0, 400.0, 400.0, 400.0]
    rows = _tel(secs, watts)
    probes = [_probe(40, 280.0), _probe(30, 280.0)]
    (t, v), proxy = _eff(rows, probes)
    assert proxy is False
    # 280 / 400 = 0.70; a nearest-sample join against the 40 W spike gives 7.0
    for x in v:
        assert 0.6 < x < 0.8, f"transient leaked into the denominator: {x}"


def test_probe_with_no_nearby_telemetry_is_discarded():
    """Better no measurement than one divided by an unrelated power reading."""
    rows = _tel([10, 5], [400.0, 400.0])
    # probes an hour away from any telemetry
    far = [_probe(3600, 280.0), _probe(3500, 280.0)]
    series, proxy = _eff(rows, far)
    assert proxy is True, "must fall back to the proxy, not fabricate perf/watt"


def test_good_probes_still_produce_real_efficiency():
    secs = list(range(60, 0, -10))
    rows = _tel(secs, [400.0] * 6)
    probes = [_probe(50, 280.0), _probe(30, 280.0), _probe(10, 280.0)]
    (t, v), proxy = _eff(rows, probes)
    assert proxy is False
    assert len(v) == 3
    assert all(abs(x - 0.7) < 1e-6 for x in v)


def test_zero_and_nan_power_are_excluded_from_the_median():
    secs = list(range(60, 0, -10))
    rows = _tel(secs, [400.0, 0.0, 400.0, None, 400.0, 400.0])
    probes = [_probe(40, 280.0), _probe(20, 280.0)]
    (t, v), proxy = _eff(rows, probes)
    assert proxy is False
    for x in v:
        assert 0.6 < x < 0.8
