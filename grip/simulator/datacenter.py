"""Deterministic 4,096-GPU demo: 512 x 8-GPU H100 nodes running four training
workloads. One workload (llama3-70b-pretrain) has a single subtle GPU-local
degradation on dc1-n0007 / GPU 3 that conventional monitoring keeps reporting as
PASS.

Full telemetry streams for a representative slice of nodes (including the
affected one); the rest of the 512-node inventory is enrolled and heartbeats.
The engine, impact math, incident and recommendation are all real - nothing here
writes incident text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np

from ..schema import (
    DiscoveredGpu,
    JobSpec,
    RankPlacement,
    Source,
    StepMetric,
    TelemetryBatch,
    WorkloadBatch,
)
from .model import GpuHealth, GpuNominal, SimGpu, sample

N_NODES = 512
GPUS_PER_NODE = 8
STREAM_NODES = 24
AFFECTED_NODE_IX = 7
AFFECTED_GPU = 3
NODE = lambda i: f"dc1-n{i:04d}"  # noqa: E731

_H100 = GpuNominal(
    sm_clock_mhz=1830.0,
    mem_clock_mhz=2619.0,
    temp_c=61.0,
    power_w=560.0,
    power_limit_w=700.0,
    util_pct=99.0,
    sm_active_pct=94.0,
    mem_total_mb=81920.0,
    mem_used_mb=71000.0,
    gemm_tflops=780.0,
    nccl_gbps=340.0,
)


@dataclass
class DcJob:
    job_id: str
    name: str
    world_size: int
    tokens_per_step: float
    checkpoint_interval: int
    ranks: list[tuple[int, str, int]]  # (rank, node_id, gpu_index) - sampled
    base_compute_ms: float = 640.0
    collective_ms: float = 110.0
    step: int = 0
    steps_since_ckpt: int = 0
    carry_s: float = field(default=0.0, repr=False)

    def spec(self) -> JobSpec:
        return JobSpec(
            job_id=self.job_id,
            name=self.name,
            world_size=self.world_size,
            tokens_per_step=self.tokens_per_step,
            checkpoint_interval_steps=self.checkpoint_interval,
            placement=[RankPlacement(rank=r, node_id=n, gpu_index=g) for r, n, g in self.ranks],
        )

    def gen(
        self,
        streamed: dict[str, SimGpu],
        dt_s: float,
        ts: datetime,
        rng: np.random.Generator,
        cap: int = 6,
    ) -> list[StepMetric]:
        out: list[StepMetric] = []
        self.carry_s += dt_s
        made = 0
        while made < cap:
            comps = {}
            for r, n, g in self.ranks:
                h = streamed.get(f"{n}:{g}")
                # compute time tracks clock (perf_mult); perf/watt is a separate signal
                pm = h.health.perf_mult if h else 1.0
                comps[r] = self.base_compute_ms / max(0.6, pm) + float(rng.normal(0, 4))
            mx = max(comps.values())
            step_s = (mx + self.collective_ms + float(rng.normal(0, 3))) / 1000.0
            if self.carry_s < step_s:
                break
            self.carry_s -= step_s
            self.step += 1
            self.steps_since_ckpt += 1
            if self.steps_since_ckpt >= self.checkpoint_interval:
                self.steps_since_ckpt = 0
            for r, _n, _g in self.ranks:
                c = comps[r]
                wait = (mx - c) + self.collective_ms + float(rng.normal(0, 2))
                out.append(
                    StepMetric(
                        job_id=self.job_id,
                        rank=r,
                        step=self.step,
                        ts=ts,
                        step_time_ms=round(c + max(wait, 0.0), 2),
                        compute_ms=round(c, 2),
                        collective_wait_ms=round(max(wait, 0.0), 2),
                        tokens=self.tokens_per_step / self.world_size,
                        steps_since_checkpoint=self.steps_since_ckpt,
                    )
                )
            made += 1
        if self.carry_s > 3.0:
            self.carry_s = 0.5
        return out


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _build_jobs(rng: np.random.Generator) -> tuple[list[DcJob], tuple[int, str, int]]:
    stream_nodes = [NODE(i) for i in range(STREAM_NODES)]
    aff = NODE(AFFECTED_NODE_IX)

    def sample_ranks(world: int, k: int, force: tuple[int, str, int] | None = None):
        stride = max(1, world // k)
        ranks = list(range(0, world, stride))[:k]
        placed = []
        for j, r in enumerate(ranks):
            n = stream_nodes[(r // GPUS_PER_NODE) % STREAM_NODES]
            g = (r + j) % GPUS_PER_NODE
            if n == aff and g == AFFECTED_GPU:
                g = (g + 1) % GPUS_PER_NODE
            placed.append((r, n, g))
        if force:
            placed = [force] + [p for p in placed if p[0] != force[0]][: k - 1]
        return placed

    straggler_rank = 500  # off-stride so it is never a duplicate of a sampled rank
    jobs = [
        DcJob(
            "dc1-llama3-70b",
            "llama3-70b-pretrain",
            2048,
            8_388_608,
            500,
            sample_ranks(2048, 64, force=(straggler_rank, aff, AFFECTED_GPU)),
        ),
        DcJob(
            "dc1-mixtral",
            "mixtral-8x22b-sft",
            1024,
            4_194_304,
            400,
            sample_ranks(1024, 24),
            base_compute_ms=610.0,
        ),
        DcJob(
            "dc1-sdxl",
            "sdxl-refiner-train",
            512,
            1_048_576,
            300,
            sample_ranks(512, 16),
            base_compute_ms=430.0,
            collective_ms=70.0,
        ),
        DcJob(
            "dc1-eval",
            "eval-harness-nightly",
            256,
            2_097_152,
            1000,
            sample_ranks(256, 12),
            base_compute_ms=520.0,
        ),
    ]
    return jobs, (straggler_rank, aff, AFFECTED_GPU)


def _degrade(g: SimGpu, prog: float) -> None:
    # subtle: nothing here crosses a DCGM threshold
    g.health.perf_mult = 1.0 - 0.045 * prog  # -4.5% SM clock
    g.health.efficiency_mult = 1.0 - 0.06 * prog  # -6% perf/watt
    g.health.thermal_offset_c = 4.5 * prog  # +4.5 C
    g.health.nccl_mult = 1.0 - 0.02 * prog


def run_datacenter(
    api: str = "http://localhost:8000",
    token: str = "dev",
    speed: float = 6.0,
    duration_s: float = 36000.0,
    prefill_minutes: float = 30.0,
    seed: int = 20,
    stdout: bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    base = api.rstrip("/")

    # --- streamed GPUs (SimGpu per node:gpu)
    streamed: dict[str, SimGpu] = {}
    for ni in range(STREAM_NODES):
        for gi in range(GPUS_PER_NODE):
            streamed[f"{NODE(ni)}:{gi}"] = SimGpu(
                index=gi,
                uuid=f"GPU-dc1-{ni:04d}-{gi:02d}",
                nominal=_H100,
                health=GpuHealth(),
            )
    affected = streamed[f"{NODE(AFFECTED_NODE_IX)}:{AFFECTED_GPU}"]
    jobs, _straggler = _build_jobs(rng)

    if stdout:
        print(
            f"[dc] would run {N_NODES} nodes / {N_NODES * GPUS_PER_NODE} GPUs, "
            f"{len(jobs)} jobs, affected {NODE(AFFECTED_NODE_IX)}/GPU {AFFECTED_GPU}",
            flush=True,
        )
        return

    # --- provision a real cluster + machine credential
    ctrl = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {token}"}, timeout=30.0)
    machine_token = token
    try:
        existing = ctrl.get("/api/clusters")
        existing.raise_for_status()
        c = next((c for c in existing.json() if c["name"] == "dc1-production"), None)
        if c:
            rotated = ctrl.post(f"/api/clusters/{c['cluster_id']}/rotate-enrollment")
            rotated.raise_for_status()
            c["enrollment_token"] = rotated.json()["enrollment_token"]
        else:
            c = ctrl.post(
                "/api/clusters",
                json={
                    "name": "dc1-production",
                    "environment": "bare_metal",
                    "region": "us-central-tx",
                },
            ).json()
        e = ctrl.post("/api/enroll", json={"enrollment_token": c["enrollment_token"]}).json()
        machine_token = e["machine_token"]
        print(
            f"[dc] cluster {c['cluster_id']} enrolled; machine credential {e['credential_id']}",
            flush=True,
        )
    except Exception as ex:  # noqa: BLE001
        print(f"[dc] cluster provisioning skipped ({ex}) - using token directly", flush=True)

    agent = httpx.Client(
        base_url=base,
        headers={"Authorization": f"Bearer {machine_token}", "Content-Type": "application/json"},
        timeout=30.0,
    )

    def hellos(indices) -> list[dict]:
        return [
            {
                "cluster_token": "",
                "node_id": NODE(i),
                "agent_version": "0.3.0",
                "driver_version": "560.35.03",
                "cuda_version": "12060",
                "dcgm_available": True,
                "gpus": [
                    DiscoveredGpu(
                        index=g,
                        uuid=f"GPU-dc1-{i:04d}-{g:02d}",
                        model="NVIDIA H100 80GB HBM3",
                        memory_mb=81920.0,
                    ).model_dump()
                    for g in range(GPUS_PER_NODE)
                ],
            }
            for i in indices
        ]

    def bulk_enroll() -> None:
        for start in range(0, N_NODES, 128):
            batch = hellos(range(start, min(start + 128, N_NODES)))
            try:
                agent.post("/api/agent/bulk", json=batch).raise_for_status()
            except Exception as ex:  # noqa: BLE001
                print(f"[dc] bulk {start} failed: {ex}", flush=True)

    print(f"[dc] enrolling {N_NODES} nodes / {N_NODES * GPUS_PER_NODE} GPUs...", flush=True)
    bulk_enroll()
    for j in jobs:
        agent.post("/api/workload", content=WorkloadBatch(job=j.spec()).model_dump_json())
    print(f"[dc] registered jobs: {', '.join(j.name for j in jobs)}", flush=True)

    def emit(ts: datetime, tick_idx: int, dt_s: float) -> None:
        for ni in range(STREAM_NODES):
            with_probe = tick_idx % 4 == 0
            samples = [
                sample(streamed[f"{NODE(ni)}:{gi}"], NODE(ni), ts, dt_s, rng, with_probe)
                for gi in range(GPUS_PER_NODE)
            ]
            b = TelemetryBatch(
                source=Source.simulator, node_id=NODE(ni), sent_at=ts, samples=samples
            )
            try:
                agent.post("/api/telemetry", content=b.model_dump_json()).raise_for_status()
            except Exception as ex:  # noqa: BLE001
                print(f"[dc] telemetry {NODE(ni)} failed: {ex}", flush=True)
        steps: list[StepMetric] = []
        for j in jobs:
            steps += j.gen(streamed, dt_s, ts, rng)
        for k in range(0, len(steps), 400):
            agent.post(
                "/api/workload",
                content=WorkloadBatch(steps=steps[k : k + 400]).model_dump_json(),
            )

    # --- phase 1: backfill healthy history
    tick_s = 30.0
    n_pre = int(prefill_minutes * 60 / tick_s)
    print(f"[dc] backfilling {prefill_minutes:.0f} min ({n_pre} ticks)...", flush=True)
    anchor = datetime.now(timezone.utc)
    for k in range(n_pre):
        emit(anchor - timedelta(seconds=(n_pre - k) * tick_s), k, tick_s)
    affected.health = GpuHealth()

    # --- phase 2: live, scenario clock at `speed`x
    real_sleep = max(0.0, tick_s / max(speed, 1e-6))
    ramp_start, ramp_end = 180.0, 2700.0
    print(
        f"[dc] live: speed={speed}x, subtle degradation on {NODE(AFFECTED_NODE_IX)}/GPU "
        f"{AFFECTED_GPU} ramps over ~{(ramp_end - ramp_start) / speed / 60:.0f} real min",
        flush=True,
    )
    wall_start = datetime.now(timezone.utc)
    tick = n_pre
    last_heartbeat = time.time()
    while True:
        ts = datetime.now(timezone.utc)
        sim_t = (ts - wall_start).total_seconds() * speed
        if sim_t > duration_s:
            break
        _degrade(affected, _smoothstep((sim_t - ramp_start) / (ramp_end - ramp_start)))
        emit(ts, tick, tick_s)
        if time.time() - last_heartbeat > 150:
            bulk_enroll()  # keep the full inventory "reporting"
            last_heartbeat = time.time()
        if (tick - n_pre) % 10 == 0:
            print(
                f"[dc] scenario t={sim_t / 60:5.1f}min  "
                f"GPU3 perf_mult={affected.health.perf_mult:.3f} "
                f"thermal+{affected.health.thermal_offset_c:.1f}C",
                flush=True,
            )
        tick += 1
        if real_sleep:
            time.sleep(real_sleep)

    print("[dc] done", flush=True)
    agent.close()
    ctrl.close()
