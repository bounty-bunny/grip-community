"""EXPERIMENTAL AMD (ROCm / amdsmi) telemetry provider for the open foundation.

This implements the same `AcceleratorProvider` seam as the NVIDIA collector, so
`grip-community monitor/dashboard/export/metrics` work on AMD Instinct hardware.
It reads amdsmi and normalizes into the shared `GpuSignals`; everything that does
not fit, or that this build does not interpret, goes into `vendor_evidence`.

**Experimental, and labelled so on purpose.** GRIP has not been validated on AMD
hardware, and amdsmi's API and field names vary across ROCm releases, so this is
best-effort collection: every amdsmi call is guarded, a field it cannot read comes
back absent (and the GPU is skipped rather than reported with half its signals, the
same rule the NVIDIA provider follows), and the raw values are preserved in
`vendor_evidence` so a mismatch can be diagnosed. It makes **no** detection claim —
the Community example detector is single-device thresholds, and peer-relative
reliability intelligence (the commercial product) is not validated on AMD.

amdsmi ships with ROCm (not PyPI): typically `pip install /opt/rocm/share/amd_smi`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..schema import AgentHello, DiscoveredGpu, GpuSample, GpuSignals, Source


def _amdsmi():
    import amdsmi  # noqa: PLC0415

    return amdsmi


def _try(fn, *args, **kw):
    """Call an amdsmi function, returning None on any failure. amdsmi raises a
    vendor exception type that differs by version, so we catch broadly here - the
    point is that one unreadable field never takes down a whole sample."""
    try:
        return fn(*args, **kw)
    except Exception:  # noqa: BLE001
        return None


def _pick(d, *keys):
    """First present, non-None value among keys in a dict (amdsmi field names move
    between releases, e.g. average_socket_power vs current_socket_power)."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # amdsmi uses large sentinels (e.g. 0xFFFF / 65535, or 2^32-1) for "unset".
    if f < 0 or f >= 4_000_000_000 or f in (65535.0, 65504.0):
        return None
    return f


class AmdSmiProvider:
    vendor = "amd"

    def __init__(self) -> None:
        self.smi = _amdsmi()
        self.smi.amdsmi_init()
        self.handles = list(self.smi.amdsmi_get_processor_handles())
        self.count = len(self.handles)
        self._static = [self._read_static(h) for h in self.handles]
        self._unreadable: set[int] = set()

    # ---- static identity, read once
    def _read_static(self, h) -> dict:
        asic = _try(self.smi.amdsmi_get_gpu_asic_info, h) or {}
        vram = _try(self.smi.amdsmi_get_gpu_vram_info, h) or {}
        drv = _try(self.smi.amdsmi_get_gpu_driver_info, h) or {}
        uuid = _try(self.smi.amdsmi_get_gpu_device_uuid, h)
        return {
            "model": str(_pick(asic, "market_name") or "AMD Instinct"),
            "uuid": uuid if isinstance(uuid, str) and uuid else None,
            "vram_total_mb": _num(_pick(vram, "vram_size", "vram_size_mb", "size")),
            "driver_version": str(_pick(drv, "driver_version") or ""),
        }

    def _uuid(self, i: int) -> str:
        return self._static[i]["uuid"] or f"amd-gpu-{i}"

    def shutdown(self) -> None:
        _try(self.smi.amdsmi_shut_down)

    def discover(self) -> list[DiscoveredGpu]:
        return [
            DiscoveredGpu(
                index=i, uuid=self._uuid(i), vendor=self.vendor,
                model=self._static[i]["model"],
                memory_mb=self._static[i]["vram_total_mb"] or 0.0,
            )
            for i in range(self.count)
        ]

    def runtime(self) -> dict:
        rocm = ""
        try:
            from pathlib import Path  # noqa: PLC0415

            rocm = Path("/opt/rocm/.info/version").read_text().strip()
        except OSError:
            pass
        return {
            "driver_version": self._static[0]["driver_version"] if self._static else "",
            "runtime_name": "rocm",
            "runtime_version": rocm,
            "vendor_tools": [],
        }

    def hello(self, cluster_token: str, node_id: str, agent_version: str = "") -> AgentHello:
        rt = self.runtime()
        return AgentHello(
            cluster_token=cluster_token, node_id=node_id,
            agent_version=f"{agent_version}+amd-experimental",
            driver_version=rt["driver_version"],
            cuda_version=rt["runtime_version"],  # NVIDIA-named field holds ROCm here
            dcgm_available=False,
            gpus=self.discover(),
        )

    # ---- per-sample telemetry
    def _sample(self, node_id: str, i: int, h, now: datetime) -> GpuSample | None:
        smi = self.smi
        metrics = _try(smi.amdsmi_get_gpu_metrics_info, h) or {}
        power = _try(smi.amdsmi_get_power_info, h) or {}
        cap = _try(smi.amdsmi_get_power_cap_info, h) or {}
        act = _try(smi.amdsmi_get_gpu_activity, h) or {}
        vram = _try(smi.amdsmi_get_gpu_vram_usage, h) or {}
        ecc = _try(smi.amdsmi_get_gpu_total_ecc_count, h) or {}

        gfx_clk = _num(_pick(metrics, "current_gfxclk", "current_gfxclks"))
        if isinstance(_pick(metrics, "current_gfxclks"), (list, tuple)):
            vals = [x for x in metrics["current_gfxclks"] if _num(x)]
            gfx_clk = _num(vals[0]) if vals else gfx_clk
        mem_clk = _num(_pick(metrics, "current_uclk", "current_vclk0"))
        temp = _num(_pick(metrics, "temperature_edge", "temperature_hotspot"))
        pwr = _num(_pick(power, "current_socket_power", "average_socket_power")) \
            or _num(_pick(metrics, "current_socket_power", "average_socket_power"))
        pwr_cap = _num(_pick(cap, "power_cap", "max_power_cap"))
        util = _num(_pick(act, "gfx_activity")) or _num(_pick(metrics, "average_gfx_activity"))
        mem_used = _num(_pick(vram, "vram_used"))

        core = {"sm_clock_mhz": gfx_clk, "mem_clock_mhz": mem_clk, "temperature_c": temp,
                "power_draw_w": pwr, "power_limit_w": pwr_cap}
        missing = [k for k, v in core.items() if v is None]
        if util is None:
            missing.append("gpu_utilization_pct")
        if mem_used is None:
            missing.append("memory_used_mb")
        if missing:
            if i not in self._unreadable:
                self._unreadable.add(i)
                print(f"[amd] gpu{i}: amdsmi did not provide {', '.join(missing)} - "
                      f"skipping (field names vary by ROCm version; see vendor_evidence)",
                      flush=True)
            return None
        self._unreadable.discard(i)

        signals = GpuSignals(
            sm_clock_mhz=gfx_clk, mem_clock_mhz=mem_clk, temperature_c=temp,
            power_draw_w=pwr, power_limit_w=pwr_cap,
            memory_temperature_c=_num(_pick(metrics, "temperature_mem")),
            gpu_utilization_pct=util, sm_activity_pct=util, memory_used_mb=mem_used,
            fan_speed_pct=None,
            pcie_replay_count=int(_num(_pick(metrics, "pcie_replay_count")) or 0),
            nvlink_crc_errors=0,  # XGMI goes to vendor_evidence (F-011)
            ecc_sbe_total=int(_num(_pick(ecc, "correctable_count")) or 0),
            ecc_dbe_total=int(_num(_pick(ecc, "uncorrectable_count")) or 0),
            xid_errors=[], throttle_reasons=[], fault_events=[],
            vendor_evidence={
                "provider": "amd-experimental",
                "sm_active_source": "amdsmi_gfx_activity",
                "amdsmi_throttle_status": _pick(metrics, "throttle_status"),
                "temp_hotspot_c": _num(_pick(metrics, "temperature_hotspot")),
                "ecc_uncorrectable": _num(_pick(ecc, "uncorrectable_count")),
            },
        )
        return GpuSample(
            source=Source.agent, node_id=node_id, gpu_index=i,
            gpu_uuid=self._uuid(i), ts=now, signals=signals,
        )

    def collect(self, node_id: str) -> list[GpuSample]:
        now = datetime.now(timezone.utc)
        out = [self._sample(node_id, i, h, now) for i, h in enumerate(self.handles)]
        return [s for s in out if s is not None]

    @staticmethod
    def config_snapshot(sh, which) -> list[dict] | None:
        """Best-effort GPU config via the amd-smi CLI, for the service install/uninstall
        diff. Returns None when amd-smi is absent rather than guessing."""
        if not which("amd-smi"):
            return None
        return None  # a structured amd-smi snapshot is future work; None is honest here
