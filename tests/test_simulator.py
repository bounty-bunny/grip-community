"""The simulator's two-phase run: backfill healthy history, then drift live."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from grip.schema import TelemetryBatch
from grip.simulator import run as run_mod


class _CaptureClient:
    def __init__(self, *a, **k):
        self.batches: list[TelemetryBatch] = []

    def post(self, _path, content=None):
        self.batches.append(TelemetryBatch.model_validate_json(content))
        return self

    def raise_for_status(self):
        return None

    def close(self):
        return None


def _run(monkeypatch, scenario, **kw):
    cap = _CaptureClient()
    monkeypatch.setattr(run_mod.httpx, "Client", lambda *a, **k: cap)
    run_mod.run(
        scenario=scenario,
        n_gpus=8,
        tick_s=15.0,
        speed=100_000.0,  # collapse the live phase to a handful of ticks
        duration_s=6000.0,
        prefill_minutes=kw.get("prefill", 10.0),
        params=kw.get("params"),
    )
    return cap.batches


def test_backfill_is_healthy_and_in_the_past(monkeypatch):
    batches = _run(monkeypatch, "gpu_local_degradation", prefill=10.0)
    now = datetime.now(timezone.utc)
    pre = batches[: int(10 * 60 / 15)]
    assert len(pre) == 40
    for b in pre:
        assert b.samples[0].ts < now
        for s in b.samples:
            assert not s.signals.xid_errors
            assert s.signals.temperature_c < 75  # no thermal drift in baseline


def test_live_phase_drifts_the_target_only(monkeypatch):
    batches = _run(monkeypatch, "gpu_local_degradation", prefill=5.0, params={"target": 6})
    last = batches[-1]
    by_idx = {s.gpu_index: s for s in last.samples}
    # by end of the ramp, GPU 6 has lost clock + TFLOPS, peers have not
    assert by_idx[6].signals.sm_clock_mhz < by_idx[0].signals.sm_clock_mhz - 80
    if by_idx[6].probe and by_idx[0].probe:
        assert by_idx[6].probe.gemm_tflops < by_idx[0].probe.gemm_tflops - 40


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
