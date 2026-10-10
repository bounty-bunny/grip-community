"""A synthetic distributed training job laid over the simulated fleet.

Every step: each rank does fwd+bwd (compute_ms, scaled up when its GPU is
degraded), then a collective barrier. Rank r's wall step time is
max(compute) + collective, so a single slow GPU drags the whole job - which is
exactly what the impact engine picks up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..schema import JobSpec, RankPlacement, StepMetric
from .model import SimGpu


@dataclass
class SimJob:
    job_id: str
    name: str
    node_id: str
    gpu_indices: list[int]
    tokens_per_step: float = 4_000_000.0
    base_compute_ms: float = 620.0
    collective_ms: float = 90.0
    checkpoint_interval_steps: int = 400
    step: int = 0
    steps_since_ckpt: int = 0
    _carry_s: float = field(default=0.0, repr=False)

    def spec(self) -> JobSpec:
        return JobSpec(
            job_id=self.job_id,
            name=self.name,
            placement=[
                RankPlacement(rank=r, node_id=self.node_id, gpu_index=g)
                for r, g in enumerate(self.gpu_indices)
            ],
            tokens_per_step=self.tokens_per_step,
            checkpoint_interval_steps=self.checkpoint_interval_steps,
        )

    def generate(
        self,
        fleet: list[SimGpu],
        elapsed_s: float,
        ts: datetime,
        rng: np.random.Generator,
    ) -> list[StepMetric]:
        out: list[StepMetric] = []
        self._carry_s += elapsed_s
        n = len(self.gpu_indices)
        guard = 0
        while guard < 400:
            guard += 1
            comps = {}
            for rank, g_idx in enumerate(self.gpu_indices):
                h = fleet[g_idx].health
                slow = 1.0 / max(0.5, h.perf_mult * h.efficiency_mult)
                comps[rank] = self.base_compute_ms * slow + float(rng.normal(0, 4))
            max_comp = max(comps.values())
            step_time_s = (max_comp + self.collective_ms + float(rng.normal(0, 3))) / 1000.0
            if self._carry_s < step_time_s:
                break
            self._carry_s -= step_time_s
            self.step += 1
            self.steps_since_ckpt += 1
            if (
                self.checkpoint_interval_steps
                and self.steps_since_ckpt >= self.checkpoint_interval_steps
            ):
                self.steps_since_ckpt = 0
            for rank in range(n):
                compute = comps[rank]
                wait = (max_comp - compute) + self.collective_ms + float(rng.normal(0, 2))
                out.append(
                    StepMetric(
                        job_id=self.job_id,
                        rank=rank,
                        step=self.step,
                        ts=ts,
                        step_time_ms=round(compute + max(wait, 0.0), 2),
                        compute_ms=round(compute, 2),
                        collective_wait_ms=round(max(wait, 0.0), 2),
                        tokens=self.tokens_per_step / n,
                        steps_since_checkpoint=self.steps_since_ckpt,
                    )
                )
        return out
