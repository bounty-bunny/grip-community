"""The wire + engine contract. Shared by agent, simulator, backend, engine.

Collection types (GpuSignals, ProbeResult, GpuSample, TelemetryBatch) are what the
agent/simulator send. Assessment types are what the engine produces.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


class Source(str, Enum):
    agent = "agent"
    simulator = "simulator"


class ThrottleReason(str, Enum):
    thermal = "thermal"  # HW thermal slowdown
    sw_thermal = "sw_thermal"  # SW thermal cap
    power = "power"  # power cap / board limit
    hw_slowdown = "hw_slowdown"  # HW slowdown (usually a fault)
    sync_boost = "sync_boost"
    board_limit = "board_limit"


# ---------------------------------------------------------------- collection


class FaultSeverity(str, Enum):
    fatal = "fatal"  # device lost / reset required
    uncorrectable = "uncorrectable"  # data corruption, job must be considered poisoned
    correctable = "correctable"  # recovered by hardware; a rate, not an event
    info = "info"


class FaultEvent(BaseModel):
    """One discrete hardware fault reported by the vendor stack since the previous
    sample. `vendor_code` is the vendor's own identifier, verbatim (`xid:79`,
    `ras:umc:uncorrectable`); the engine reads only `severity`."""

    vendor_code: str
    severity: FaultSeverity
    description: str = ""


class GpuSignals(BaseModel):
    sm_clock_mhz: float
    mem_clock_mhz: float
    temperature_c: float
    memory_temperature_c: float | None = None
    power_draw_w: float
    power_limit_w: float
    gpu_utilization_pct: float
    # Fraction of time the compute units were busy, 0-100. The NVML provider fills
    # this from utilization.gpu - NOT from DCGM PROF_SM_ACTIVE, which would be the
    # better source and is not collected today. vendor_evidence["sm_active_source"]
    # records which one a sample actually came from.
    sm_activity_pct: float
    memory_used_mb: float
    fan_speed_pct: float | None = None
    # monotonic counters - engine differentiates them into rates
    pcie_replay_count: int = 0
    nvlink_crc_errors: int = 0
    ecc_sbe_total: int = 0
    ecc_dbe_total: int = 0
    # events since last sample
    xid_errors: list[int] = Field(default_factory=list)  # legacy NVIDIA-only view
    throttle_reasons: list[ThrottleReason] = Field(default_factory=list)
    # Typed fault events, any vendor. The NVML provider emits one per Xid it sees
    # AND mirrors the code into xid_errors, so both views agree for one release.
    fault_events: list[FaultEvent] = Field(default_factory=list)
    # Everything the vendor said that does not fit the normalized fields, verbatim:
    # raw throttle bitmasks, which sensor a value came from, DCGM field ids. Stored,
    # never read by the engine. This is where vendor evidence is preserved.
    vendor_evidence: dict = Field(default_factory=dict)


class ProbeKind(str, Enum):
    gemm = "gemm"
    nccl_ring = "nccl_ring"


class ProbeResult(BaseModel):
    kind: ProbeKind
    duration_ms: float
    gemm_tflops: float | None = None
    nccl_ring_gbps: float | None = None
    matrix_n: int | None = None
    dtype: str | None = None


class GpuSample(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source: Source
    node_id: str
    gpu_index: int
    gpu_uuid: str = ""
    ts: datetime
    signals: GpuSignals
    probe: ProbeResult | None = None


class TelemetryBatch(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source: Source
    node_id: str
    sent_at: datetime
    samples: list[GpuSample]


# ---------------------------------------------------------------- engine output


class WorkloadBucket(str, Enum):
    idle = "idle"
    low = "low"
    med = "med"
    high = "high"


class DriftDirection(str, Enum):
    up = "up"
    down = "down"


class DiagnosisClass(str, Enum):
    none = "none"
    gpu_local = "gpu_local"
    shared_env = "shared_env"
    indeterminate = "indeterminate"


class HealthBand(str, Enum):
    healthy = "healthy"
    watch = "watch"
    degrading = "degrading"
    critical = "critical"


class ScoreTrend(str, Enum):
    improving = "improving"
    stable = "stable"
    declining = "declining"


class SignalDeviation(BaseModel):
    signal: str
    label: str
    unit: str
    value: float | None = None
    peer_median: float | None = None
    peer_mad: float | None = None
    peer_robust_z: float | None = None
    peer_pct_delta: float | None = None
    hist_median: float | None = None
    hist_robust_z: float | None = None
    trend_pct_per_hour: float | None = None
    trend_r2: float | None = None


class DriftFlag(BaseModel):
    signal: str
    label: str
    direction: DriftDirection
    severity: float  # 0..1
    persistence_ticks: int
    since: datetime
    detail: str


class GpuAssessment(BaseModel):
    ts: datetime
    node_id: str
    gpu_index: int
    gpu_uuid: str = ""
    workload_bucket: WorkloadBucket
    peer_count: int

    deviations: list[SignalDeviation]
    drift_flags: list[DriftFlag]

    efficiency_tflops_per_w: float | None = None
    efficiency_peer_pct_delta: float | None = None
    efficiency_trend_per_hour: float | None = None
    efficiency_is_proxy: bool = True

    gpu_local_score: float
    shared_env_score: float
    diagnosis_class: DiagnosisClass
    diagnosis_confidence: float
    evidence: list[str]

    health_score: float
    health_band: HealthBand
    health_trend: ScoreTrend


class ConventionalCheck(BaseModel):
    name: str
    status: str  # pass | warn | fail
    detail: str


class ConventionalVerdict(BaseModel):
    ts: datetime
    node_id: str
    gpu_index: int
    status: str  # pass | warn | fail  (worst of checks)
    checks: list[ConventionalCheck]


# ---------------------------------------------------------------- enrollment


class DiscoveredGpu(BaseModel):
    index: int
    uuid: str = ""
    vendor: str = "nvidia"  # nvidia | amd | ... - set by the vendor's collector
    model: str = "unknown"
    memory_mb: float | None = None


class AgentHello(BaseModel):
    """Sent by grip-agent on startup. GRIP maps the node + GPUs automatically."""

    schema_version: str = SCHEMA_VERSION
    cluster_token: str
    node_id: str
    agent_version: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    dcgm_available: bool = False
    gpus: list[DiscoveredGpu] = Field(default_factory=list)


class AgentHelloResponse(BaseModel):
    ok: bool
    cluster_id: str
    cluster_name: str
    node_id: str
    gpus_registered: int
    message: str = ""


# ---------------------------------------------------------------- workload


class RankPlacement(BaseModel):
    rank: int
    node_id: str
    gpu_index: int


class JobSpec(BaseModel):
    """Registered once (or on change) by grip.workload / the simulator."""

    job_id: str
    name: str
    framework: str = "pytorch"
    placement: list[RankPlacement]
    # Declared job size. May exceed len(placement) when only a sample of ranks
    # reports step metrics (normal at multi-thousand-GPU scale).
    world_size: int | None = None
    tokens_per_step: float | None = None  # global tokens per optimizer step
    expected_tokens_per_s: float | None = None  # from a healthy baseline, if known
    checkpoint_interval_steps: int | None = None
    started_at: datetime | None = None


class StepMetric(BaseModel):
    job_id: str
    rank: int
    step: int
    ts: datetime
    step_time_ms: float  # wall time for the whole optimizer step
    compute_ms: float | None = None  # fwd+bwd before the first collective
    collective_wait_ms: float | None = None  # time in / waiting on collectives
    tokens: float | None = None  # tokens this rank processed this step
    steps_since_checkpoint: int | None = None


class WorkloadBatch(BaseModel):
    schema_version: str = SCHEMA_VERSION
    job: JobSpec | None = None  # present to (re)register
    steps: list[StepMetric] = Field(default_factory=list)


# ---------------------------------------------------------------- job impact


class JobImpact(BaseModel):
    ts: datetime
    job_id: str
    window_minutes: float
    gpu_count: int  # declared job size (world_size)
    measured_ranks: int  # how many ranks actually contributed step metrics
    actual_step_time_ms: float  # MEASURED
    counterfactual_step_time_ms: float  # MEASURED (same-job median)
    actual_tokens_per_s: float | None = None  # MEASURED
    ideal_tokens_per_s: float | None = None  # MEASURED
    throughput_loss_pct: float  # MEASURED, 0..1
    straggler_ranks: list[int]
    straggler_gpus: list[dict]  # [{node_id, gpu_index, rank}]
    effective_gpus_lost: float  # ESTIMATED: throughput_loss_pct * gpu_count
    est_cost_per_hour: float  # ESTIMATED: effective_gpus_lost * cost_per_gpu_hour
    degraded: bool


class RemediationAction(str, Enum):
    monitor = "monitor"
    allow_checkpoint = "allow_checkpoint"
    run_diagnostic = "run_diagnostic"
    drain_node = "drain_node"
    replace_node = "replace_node"
    reschedule_job = "reschedule_job"
    quarantine_gpu = "quarantine_gpu"
    recommend_rma = "recommend_rma"
    check_rack_cooling = "check_rack_cooling"
    investigate_fabric = "investigate_fabric"


class Risk(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class RecommendedStep(BaseModel):
    order: int
    action: RemediationAction
    target: str
    detail: str


class Recommendation(BaseModel):
    summary: str
    steps: list[RecommendedStep]
    expected_recovery_pct: float  # 0..1 of the lost throughput
    risk_of_continuing: Risk


class OutcomeAction(str, Enum):
    """What the operator did.

    A controlled vocabulary, because free text across many fleets is unjoinable -
    and the outcome label is what makes an episode learnable rather than merely
    recorded. Free-text detail still travels in `action_taken`.
    """

    none = "none"  # observed only, no action taken
    reset = "reset"
    rebooted = "rebooted"
    drained = "drained"
    reseated = "reseated"
    replaced = "replaced"
    firmware_updated = "firmware_updated"
    workload_moved = "workload_moved"
    other = "other"


class OutcomeResolution(str, Enum):
    """What actually happened afterwards.

    `pending` is the default on purpose: RMA results arrive weeks later, and a
    resolution nobody has confirmed must never be stored as one that was.
    `rma_no_fault_found` is as valuable a label as a confirmed fault - it is the
    only way to learn which patterns are false alarms.
    """

    pending = "pending"
    recovered = "recovered"
    persisted = "persisted"
    recurred = "recurred"
    rma_fault_confirmed = "rma_fault_confirmed"
    rma_no_fault_found = "rma_no_fault_found"
    unknown = "unknown"


class IncidentOutcome(BaseModel):
    action: OutcomeAction
    resolution: OutcomeResolution = OutcomeResolution.pending
    action_taken: str | None = None  # free-text detail alongside the enum
    throughput_before_pct_loss: float | None = None
    throughput_after_pct_loss: float | None = None
    notes: str | None = None
    recorded_at: datetime | None = None
