"""NVIDIA provider: NVML in, normalized GpuSignals out - never a verdict.

This is the one module that imports a vendor SDK. Everything it knows that does
not fit the normalized fields goes into `vendor_evidence`, verbatim, so nothing
NVIDIA said is lost on the way to a vendor-neutral engine.

Requires `pip install 'grip[gpu]'` (pynvml) on the GPU host.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from ..schema import (
    AgentHello,
    DiscoveredGpu,
    FaultEvent,
    FaultSeverity,
    GpuSample,
    GpuSignals,
    Source,
    ThrottleReason,
)

# nvmlClocksEventReasons bits -> normalized reason. The mask itself is also
# recorded in vendor_evidence["nvml_throttle_bits"] so the mapping can be
# re-derived later if it ever turns out to be wrong.
_THROTTLE_BITS = {
    0x0000000000000004: ThrottleReason.power,  # SwPowerCap - a power reason, not thermal
    0x0000000000000008: ThrottleReason.sync_boost,  # SyncBoost
    0x0000000000000020: ThrottleReason.sw_thermal,  # SwThermalSlowdown
    0x0000000000000040: ThrottleReason.hw_slowdown,  # HwSlowdown
    0x0000000000000080: ThrottleReason.thermal,  # HwThermalSlowdown
    0x0000000000000100: ThrottleReason.power,  # HwPowerBrakeSlowdown
}

# What the sm_activity_pct field is actually filled from on this provider. DCGM's
# PROF_SM_ACTIVE would be the better source and is not collected today.
_SM_ACTIVE_SOURCE = "nvml_utilization_gpu"

# Per tick, never drain more than this many queued events - a wedged driver must
# not turn collect() into an infinite loop.
_MAX_EVENTS_PER_TICK = 64

# Read-only GPU *configuration* fields for the install/uninstall snapshot.
# Nothing here changes from tick to tick on its own.
CONFIG_SNAPSHOT_FIELDS = (
    "index",
    "uuid",
    "persistence_mode",
    "compute_mode",
    "power.limit",
    "clocks.applications.graphics",
    "clocks.applications.memory",
    "ecc.mode.current",
    "mig.mode.current",
)


class NvmlProvider:
    vendor = "nvidia"

    def __init__(self) -> None:
        import pynvml  # noqa: PLC0415

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.count = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.count)]
        self._unreadable: set[int] = set()  # warn once per GPU, not every tick
        self._tick_xids: dict[int, list[int]] = {}
        self._events = None
        self._xid_source = "unsupported"
        self._register_xid_events()

    # ------------------------------------------------------------ lifecycle

    def shutdown(self) -> None:
        if self._events is not None:
            try:
                self.nvml.nvmlEventSetFree(self._events)
            except Exception:  # noqa: BLE001
                pass
        try:
            self.nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ helpers

    def _str(self, fn, *a, default: str = "") -> str:
        try:
            v = fn(*a)
            return v.decode() if isinstance(v, bytes) else str(v)
        except Exception:  # noqa: BLE001
            return default

    def _int(self, fn, *a, default=0):
        try:
            return int(fn(*a))
        except Exception:  # noqa: BLE001
            return default

    def _float(self, fn, *a, scale: float = 1.0, default=None):
        """A signal NVML refuses on this SKU is missing, not zero - the engine
        treats None as absent and zero as an observation."""
        try:
            return float(fn(*a)) * scale
        except Exception:  # noqa: BLE001
            return default

    def _cuda_version(self) -> str:
        """NVML returns CUDA as an int like 12040 -> '12.4'."""
        try:
            v = int(self.nvml.nvmlSystemGetCudaDriverVersion())
        except Exception:  # noqa: BLE001
            return ""
        return f"{v // 1000}.{(v % 1000) // 10}"

    # ------------------------------------------------------------ discovery

    def discover(self) -> list[DiscoveredGpu]:
        nv = self.nvml
        return [
            DiscoveredGpu(
                index=i,
                uuid=self._str(nv.nvmlDeviceGetUUID, h),
                vendor=self.vendor,
                model=self._str(nv.nvmlDeviceGetName, h, default="unknown"),
                memory_mb=self._float(lambda x: nv.nvmlDeviceGetMemoryInfo(x).total, h, scale=1e-6),
            )
            for i, h in enumerate(self.handles)
        ]

    def runtime(self) -> dict:
        return {
            "driver_version": self._str(self.nvml.nvmlSystemGetDriverVersion),
            "runtime_name": "cuda",
            "runtime_version": self._cuda_version(),
            "vendor_tools": [],  # DCGM presence is not probed by this provider today
        }

    def hello(self, cluster_token: str, node_id: str, agent_version: str = "") -> AgentHello:
        rt = self.runtime()
        return AgentHello(
            cluster_token=cluster_token,
            node_id=node_id,
            agent_version=agent_version,
            driver_version=rt["driver_version"],
            cuda_version=rt["runtime_version"],
            dcgm_available=False,
            gpus=self.discover(),
        )

    # ------------------------------------------------------------ Xid events

    def _register_xid_events(self) -> None:
        """Subscribe to critical Xid events for every device. Not every driver,
        SKU or virtualised host allows this; when it does not, say so in the
        evidence rather than pretending there were no faults."""
        nv = self.nvml
        try:
            es = nv.nvmlEventSetCreate()
            for h in self.handles:
                nv.nvmlDeviceRegisterEvents(h, nv.nvmlEventTypeXidCriticalError, es)
        except Exception as e:  # noqa: BLE001
            print(f"[agent] Xid events unavailable ({e}) - fault_events will be empty", flush=True)
            self._events = None
            self._xid_source = "unsupported"
            return
        self._events = es
        self._xid_source = "nvml_events"

    def _device_index(self, handle) -> int | None:
        try:
            return int(self.nvml.nvmlDeviceGetIndex(handle))
        except Exception:  # noqa: BLE001
            pass
        for i, h in enumerate(self.handles):
            if h == handle:
                return i
        return None

    def _drain_xids(self) -> dict[int, list[int]]:
        """Every critical Xid queued since the last tick, by device index."""
        out: dict[int, list[int]] = defaultdict(list)
        if self._events is None:
            return out
        nv = self.nvml
        wait = getattr(nv, "nvmlEventSetWait_v2", None) or getattr(nv, "nvmlEventSetWait", None)
        if wait is None:
            return out
        for _ in range(_MAX_EVENTS_PER_TICK):
            try:
                data = wait(self._events, 0)  # timeout 0: raise when the queue is empty
            except Exception:  # noqa: BLE001 - NVMLError_Timeout is the normal exit
                break
            idx = self._device_index(getattr(data, "device", None))
            if idx is None:
                continue
            try:
                out[idx].append(int(getattr(data, "eventData", 0)))
            except (TypeError, ValueError):
                continue
        return out

    # ------------------------------------------------------------ sampling

    def _throttle(self, h) -> tuple[list[ThrottleReason], int | None]:
        """Normalized reasons plus the raw mask they were derived from."""
        # pynvml renamed this call; older builds only have the "Throttle" spelling.
        for name in (
            "nvmlDeviceGetCurrentClocksEventReasons",
            "nvmlDeviceGetCurrentClocksThrottleReasons",
        ):
            fn = getattr(self.nvml, name, None)
            if fn is None:
                continue
            try:
                bits = int(fn(h))
            except Exception:  # noqa: BLE001
                return [], None
            active = {r for mask, r in _THROTTLE_BITS.items() if bits & mask}
            return sorted(active, key=lambda r: r.value), bits
        return [], None

    def _power_limit_w(self, h) -> float | None:
        # Enforced limit reflects an active cap (what the lab harness sets); the
        # management limit is the fallback on SKUs that refuse the enforced read.
        nv = self.nvml
        for fn in (nv.nvmlDeviceGetEnforcedPowerLimit, nv.nvmlDeviceGetPowerManagementLimit):
            v = self._float(fn, h, scale=1e-3)
            if v is not None:
                return v
        return None

    def _sample(self, node_id: str, i: int, h, now: datetime) -> GpuSample | None:
        nv = self.nvml
        core = {
            "sm_clock_mhz": self._float(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM),
            "mem_clock_mhz": self._float(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_MEM),
            "temperature_c": self._float(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU),
            "power_draw_w": self._float(nv.nvmlDeviceGetPowerUsage, h, scale=1e-3),
            "power_limit_w": self._power_limit_w(h),
        }
        util = self._float(lambda x: nv.nvmlDeviceGetUtilizationRates(x).gpu, h)
        mem_used = self._float(lambda x: nv.nvmlDeviceGetMemoryInfo(x).used, h, scale=1e-6)
        missing = [k for k, v in core.items() if v is None]
        if util is None:
            missing.append("gpu_utilization_pct")
        if mem_used is None:
            missing.append("memory_used_mb")
        if missing:
            # Emitting a partial sample would feed the baseline a value NVML never
            # reported. Skip the GPU for this tick and say so once.
            if i not in self._unreadable:
                self._unreadable.add(i)
                print(f"[agent] gpu{i}: NVML has no {', '.join(missing)} - skipping", flush=True)
            return None
        self._unreadable.discard(i)

        throttle, bits = self._throttle(h)
        xids = self._tick_xids.get(i, [])
        evidence: dict = {
            "sm_active_source": _SM_ACTIVE_SOURCE,
            "xid_source": self._xid_source,
        }
        if bits is not None:
            evidence["nvml_throttle_bits"] = bits
        if xids:
            evidence["xid"] = list(xids)

        signals = GpuSignals(
            **core,
            gpu_utilization_pct=util,
            sm_activity_pct=util,  # see _SM_ACTIVE_SOURCE - this is utilization.gpu
            memory_used_mb=mem_used,
            fan_speed_pct=self._float(nv.nvmlDeviceGetFanSpeed, h),
            pcie_replay_count=self._int(nv.nvmlDeviceGetPcieReplayCounter, h),
            nvlink_crc_errors=0,
            ecc_sbe_total=self._int(
                nv.nvmlDeviceGetTotalEccErrors,
                h,
                nv.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                nv.NVML_VOLATILE_ECC,
            ),
            ecc_dbe_total=self._int(
                nv.nvmlDeviceGetTotalEccErrors,
                h,
                nv.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                nv.NVML_VOLATILE_ECC,
            ),
            xid_errors=list(xids),  # legacy view, mirrors fault_events
            throttle_reasons=throttle,
            fault_events=[
                FaultEvent(
                    vendor_code=f"xid:{x}",
                    severity=FaultSeverity.fatal,  # the event set is XidCriticalError only
                    description=f"NVML critical Xid {x}",
                )
                for x in xids
            ],
            vendor_evidence=evidence,
        )
        return GpuSample(
            source=Source.agent,
            node_id=node_id,
            gpu_index=i,
            gpu_uuid=self._str(nv.nvmlDeviceGetUUID, h),
            ts=now,
            signals=signals,
        )

    def collect(self, node_id: str) -> list[GpuSample]:
        now = datetime.now(timezone.utc)
        self._tick_xids = self._drain_xids()
        out: list[GpuSample] = []
        for i, h in enumerate(self.handles):
            try:
                s = self._sample(node_id, i, h, now)
            except Exception as e:  # noqa: BLE001
                # One sick device must not cost us telemetry for its healthy peers.
                print(f"[agent] gpu{i}: collect failed ({e})", flush=True)
                continue
            if s is not None:
                out.append(s)
        return out

    # ------------------------------------------------------------ configuration snapshot

    @staticmethod
    def config_snapshot(sh, which) -> list[dict] | None:
        """Read-only GPU configuration via nvidia-smi, for install/uninstall
        proof. Needs no NVML session, so `grip uninstall` can take it after the
        package's Python dependencies are gone. Returns None rather than guessing
        when nvidia-smi is unavailable."""
        exe = which("nvidia-smi")
        if not exe:
            return None
        r = sh(
            [
                exe,
                f"--query-gpu={','.join(CONFIG_SNAPSHOT_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
        )
        if r.returncode != 0:
            return None
        rows = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(CONFIG_SNAPSHOT_FIELDS):
                continue
            rows.append(dict(zip(CONFIG_SNAPSHOT_FIELDS, parts, strict=True)))
        return rows


# The name this class had through rc8. Kept so nothing importing it breaks.
NvmlCollector = NvmlProvider
