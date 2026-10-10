"""Drive a synthetic fleet and ship telemetry to the ingestion API.

Two phases:
  1. Backfill `prefill_minutes` of healthy baseline at past timestamps, so the
     engine has a full analysis window the moment it starts.
  2. Live: emit at wall-clock time while the *scenario* clock advances at
     `speed`x. So a degradation authored to unfold over an hour of scenario time
     plays out in ~(3600 / speed) real seconds, on a real-time-stamped series
     the engine can trend against.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np

from ..schema import Source, TelemetryBatch, WorkloadBatch
from .model import GpuHealth, GpuNominal, SimGpu, sample
from .scenarios import SCENARIOS
from .workload_model import SimJob


def run(
    scenario: str,
    node_id: str = "sim-node-1",
    n_gpus: int = 8,
    tick_s: float = 15.0,
    speed: float = 6.0,
    duration_s: float = 5400.0,
    api: str = "http://localhost:8000",
    token: str = "dev",
    probe_every: int = 4,
    seed: int = 7,
    stdout: bool = False,
    params: dict | None = None,
    prefill_minutes: float = 35.0,
    with_job: bool = True,
) -> None:
    if scenario not in SCENARIOS:
        raise SystemExit(f"unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
    fn = SCENARIOS[scenario]
    params = params or {}
    rng = np.random.default_rng(seed)

    fleet = [
        SimGpu(index=i, uuid=f"GPU-sim-{node_id}-{i:02d}", nominal=GpuNominal(), health=GpuHealth())
        for i in range(n_gpus)
    ]
    job = (
        SimJob(
            job_id=f"{node_id}-llama70b",
            name="llama-70b-pretrain",
            node_id=node_id,
            gpu_indices=list(range(n_gpus)),
        )
        if with_job and scenario != "healthy"
        else (
            SimJob(
                job_id=f"{node_id}-bert",
                name="bert-large-finetune",
                node_id=node_id,
                gpu_indices=list(range(n_gpus)),
            )
            if with_job
            else None
        )
    )

    client = (
        None
        if stdout
        else httpx.Client(
            base_url=api,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=10.0,
        )
    )

    def _send(path: str, body) -> None:
        if stdout:
            return
        try:
            client.post(path, content=body.model_dump_json()).raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"[sim] post {path} failed: {e}", flush=True)

    def emit(ts: datetime, sim_t: float, tick_idx: int) -> None:
        fn(fleet, sim_t, params)
        with_probe = tick_idx % max(1, probe_every) == 0
        samples = [sample(g, node_id, ts, tick_s, rng, with_probe) for g in fleet]
        batch = TelemetryBatch(
            source=Source.simulator, node_id=node_id, sent_at=ts, samples=samples
        )
        if stdout:
            print(json.dumps(batch.model_dump(mode="json"))[:2000])
        else:
            _send("/api/telemetry", batch)
        if job is not None:
            steps = job.generate(fleet, tick_s, ts, rng)
            if steps:
                _send("/api/workload", WorkloadBatch(steps=steps))

    if job is not None:
        print(f"[sim] registering job {job.name} across {len(job.gpu_indices)} GPUs", flush=True)
        _send("/api/workload", WorkloadBatch(job=job.spec()))

    # -- phase 1: backfill healthy baseline
    n_pre = int(prefill_minutes * 60 / tick_s) if prefill_minutes > 0 else 0
    if n_pre:
        print(
            f"[sim] backfilling {prefill_minutes:.0f} min baseline ({n_pre} ticks)...", flush=True
        )
        anchor = datetime.now(timezone.utc)
        for k in range(n_pre):
            emit(anchor - timedelta(seconds=(n_pre - k) * tick_s), 0.0, k)
        for g in fleet:  # scrub any sim_t=0 scenario touches
            g.health = GpuHealth()
            g.sbe_total = g.dbe_total = g.replay_total = 0

    # -- phase 2: live, scenario clock at `speed`x wall time
    real_sleep = max(0.0, tick_s / max(speed, 1e-6))
    print(
        f"[sim] live: scenario={scenario} speed={speed}x "
        f"(scenario hour -> ~{3600 / speed / 60:.0f} real min), "
        f"duration={duration_s / 60:.0f} scenario-min",
        flush=True,
    )
    wall_start = datetime.now(timezone.utc)
    tick = n_pre
    while True:
        ts = datetime.now(timezone.utc)
        sim_t = (ts - wall_start).total_seconds() * speed
        if sim_t > duration_s:
            break
        emit(ts, sim_t, tick)

        if (tick - n_pre) % 20 == 0:
            affected = [
                g.index
                for g in fleet
                if g.health.perf_mult < 0.995
                or g.health.efficiency_mult < 0.995
                or g.health.thermal_offset_c > 1.5
                or g.health.force_throttle
            ]
            print(
                f"[sim] scenario t={sim_t / 60:5.1f}min  affected={affected}",
                flush=True,
            )

        tick += 1
        if real_sleep:
            time.sleep(real_sleep)

    print("[sim] done", flush=True)
    if client:
        client.close()
