"""NvmlCollector against a fake pynvml.

This is the one module that only ever runs on a real GPU host, so its failure
modes (pynvml API renames, byte-vs-str returns, a SKU that refuses a signal)
have to be pinned here rather than discovered on rented hardware.
"""

from __future__ import annotations

import sys

import pytest

from grip.agent.collect import NvmlCollector
from grip.schema import ThrottleReason


class _NvmlError(Exception):
    pass


class _Rates:
    def __init__(self, gpu: float) -> None:
        self.gpu = gpu


class _Mem:
    def __init__(self, used: float, total: float) -> None:
        self.used = used
        self.total = total


class FakeNvml:
    """`style="old"` = pre-rename pynvml: bytes strings + ClocksThrottleReasons."""

    NVML_CLOCK_SM = 0
    NVML_CLOCK_MEM = 1
    NVML_TEMPERATURE_GPU = 0
    NVML_MEMORY_ERROR_TYPE_CORRECTED = 0
    NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
    NVML_VOLATILE_ECC = 0

    nvmlEventTypeXidCriticalError = 0x8

    def __init__(
        self,
        n: int = 4,
        style: str = "new",
        fail: dict[int, set[str]] | None = None,
        events: str = "supported",
        throttle_bits: dict[int, int] | None = None,
    ):
        self.n = n
        self.style = style
        self.fail = fail or {}
        self.shutdown_calls = 0
        self.events_mode = events  # supported | unsupported
        self.queued: list[tuple[int, int]] = []  # (device index, xid)
        self.registered: list[int] = []
        self.event_set_freed = 0
        self.throttle_bits = throttle_bits if throttle_bits is not None else {1: 0x80}
        name = "nvmlDeviceGetCurrentClocksEventReasons"
        if style == "old":
            name = "nvmlDeviceGetCurrentClocksThrottleReasons"
        setattr(self, name, self._throttle_bits)

    # --- helpers -------------------------------------------------------
    def _guard(self, h: int, call: str) -> None:
        if call in self.fail.get(h, ()):
            raise _NvmlError(f"{call} not supported on gpu{h}")

    def _s(self, v: str):
        return v.encode() if self.style == "old" else v

    def _throttle_bits(self, h: int) -> int:
        self._guard(h, "throttle")
        return self.throttle_bits.get(h, 0)  # default: gpu1 is thermally throttled

    # --- event API (Xid) -------------------------------------------------
    def nvmlEventSetCreate(self):
        if self.events_mode == "unsupported":
            raise _NvmlError("Not Supported")
        return "eventset"

    def nvmlDeviceRegisterEvents(self, h: int, types: int, es) -> None:
        self.registered.append(h)

    def nvmlEventSetWait_v2(self, es, timeout_ms: int):
        if not self.queued:
            raise _NvmlError("Timeout")
        dev, xid = self.queued.pop(0)

        class _Data:
            pass

        d = _Data()
        d.device, d.eventData = dev, xid
        return d

    def nvmlDeviceGetIndex(self, h: int) -> int:
        return h

    def nvmlEventSetFree(self, es) -> None:
        self.event_set_freed += 1

    # --- NVML surface --------------------------------------------------
    def nvmlInit(self) -> None:
        pass

    def nvmlShutdown(self) -> None:
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self) -> int:
        return self.n

    def nvmlDeviceGetHandleByIndex(self, i: int) -> int:
        return i

    def nvmlSystemGetDriverVersion(self):
        return self._s("550.90.07")

    def nvmlSystemGetCudaDriverVersion(self) -> int:
        return 12040

    def nvmlDeviceGetUUID(self, h: int):
        return self._s(f"GPU-fake-{h:04d}")

    def nvmlDeviceGetName(self, h: int):
        return self._s("NVIDIA H100 80GB HBM3")

    def nvmlDeviceGetClockInfo(self, h: int, clock: int) -> int:
        self._guard(h, "clock")
        return 1980 if clock == self.NVML_CLOCK_SM else 2619

    def nvmlDeviceGetTemperature(self, h: int, sensor: int) -> int:
        self._guard(h, "temperature")
        return 62

    def nvmlDeviceGetPowerUsage(self, h: int) -> int:
        self._guard(h, "power")
        return 412_000  # milliwatts

    def nvmlDeviceGetEnforcedPowerLimit(self, h: int) -> int:
        self._guard(h, "enforced_limit")
        return 700_000

    def nvmlDeviceGetPowerManagementLimit(self, h: int) -> int:
        self._guard(h, "mgmt_limit")
        return 700_000

    def nvmlDeviceGetUtilizationRates(self, h: int) -> _Rates:
        self._guard(h, "util")
        return _Rates(97.0)

    def nvmlDeviceGetMemoryInfo(self, h: int) -> _Mem:
        self._guard(h, "memory")
        return _Mem(used=68_000e6, total=80_000e6)

    def nvmlDeviceGetFanSpeed(self, h: int) -> int:
        self._guard(h, "fan")
        return 55

    def nvmlDeviceGetPcieReplayCounter(self, h: int) -> int:
        self._guard(h, "pcie")
        return 3

    def nvmlDeviceGetTotalEccErrors(self, h: int, etype: int, scope: int) -> int:
        self._guard(h, "ecc")
        return 11 if etype == self.NVML_MEMORY_ERROR_TYPE_CORRECTED else 0


@pytest.fixture
def collector(monkeypatch):
    def build(**kw) -> NvmlCollector:
        fake = FakeNvml(**kw)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        c = NvmlCollector()
        c.fake = fake  # type: ignore[attr-defined]
        return c

    return build


@pytest.mark.parametrize("style", ["new", "old"])
def test_collect_normalizes_both_pynvml_generations(collector, style):
    """Old pynvml hands back bytes and spells throttle reasons differently."""
    c = collector(n=4, style=style)
    samples = c.collect("node-a")

    assert len(samples) == 4
    s = samples[0]
    assert s.gpu_uuid == "GPU-fake-0000"  # str, never b"GPU-..."
    assert isinstance(s.gpu_uuid, str)
    assert s.node_id == "node-a"
    assert s.signals.sm_clock_mhz == 1980.0
    assert s.signals.mem_clock_mhz == 2619.0
    assert s.signals.temperature_c == 62.0
    assert s.signals.power_draw_w == pytest.approx(412.0)  # mW -> W
    assert s.signals.power_limit_w == pytest.approx(700.0)
    assert s.signals.memory_used_mb == pytest.approx(68_000.0)  # bytes -> MB
    assert s.signals.fan_speed_pct == 55.0
    assert s.signals.ecc_sbe_total == 11
    assert s.signals.ecc_dbe_total == 0
    assert s.signals.pcie_replay_count == 3
    # the throttle mask is read through whichever spelling this pynvml exposes
    assert samples[1].signals.throttle_reasons == [ThrottleReason.thermal]
    assert samples[0].signals.throttle_reasons == []


def test_hello_reports_str_identifiers_and_decoded_cuda_version(collector):
    c = collector(n=8, style="old")
    hello = c.hello("grpm_x", "node-a", agent_version="0.3.0")

    assert hello.driver_version == "550.90.07"
    assert hello.cuda_version == "12.4"  # 12040, not "12040"
    assert len(hello.gpus) == 8
    g = hello.gpus[0]
    assert g.index == 0
    assert g.uuid == "GPU-fake-0000"
    assert g.model == "NVIDIA H100 80GB HBM3"
    assert g.memory_mb == pytest.approx(80_000.0)


def test_enforced_power_limit_falls_back_to_management_limit(collector):
    """Some SKUs refuse the enforced read; the sample must survive."""
    c = collector(n=4, fail={2: {"enforced_limit"}})
    samples = c.collect("node-a")

    assert len(samples) == 4
    assert samples[2].signals.power_limit_w == pytest.approx(700.0)


def test_unreadable_core_signal_skips_only_that_gpu(collector, capsys):
    """A GPU NVML won't report must not take its healthy peers' telemetry with it,
    and must not be invented as a zero."""
    c = collector(n=8, fail={5: {"temperature"}})
    samples = c.collect("node-a")

    assert [s.gpu_index for s in samples] == [0, 1, 2, 3, 4, 6, 7]
    assert "gpu5" in capsys.readouterr().out

    # warned once, not once per tick
    c.collect("node-a")
    assert capsys.readouterr().out == ""


def test_gpu_recovering_is_collected_again(collector):
    c = collector(n=4, fail={1: {"util"}})
    assert [s.gpu_index for s in c.collect("node-a")] == [0, 2, 3]

    c.fake.fail = {}  # type: ignore[attr-defined]
    assert [s.gpu_index for s in c.collect("node-a")] == [0, 1, 2, 3]


def test_unexpected_error_on_one_gpu_does_not_lose_the_batch(collector, monkeypatch):
    c = collector(n=4)
    real = c._sample

    def boom(node_id, i, h, now):
        if i == 2:
            raise RuntimeError("pynvml ABI mismatch")
        return real(node_id, i, h, now)

    monkeypatch.setattr(c, "_sample", boom)
    assert [s.gpu_index for s in c.collect("node-a")] == [0, 1, 3]


def test_optional_signals_are_absent_not_zero(collector):
    """fan_speed_pct is None when NVML has no fan - a real 0% would mean a dead fan."""
    c = collector(n=2, fail={0: {"fan", "pcie"}})
    s = c.collect("node-a")[0]

    assert s.signals.fan_speed_pct is None
    assert s.signals.pcie_replay_count == 0  # counter: absent == none seen


def test_all_samples_share_one_timestamp(collector):
    """Peer-relative baselining compares GPUs within a tick."""
    c = collector(n=8)
    assert len({s.ts for s in c.collect("node-a")}) == 1


# ---------------------------------------------------------------- provider seam, evidence, faults
# Added with the vendor-neutral seam: the throttle mapping is pinned bit by bit
# (0x04 is a POWER reason - it was mislabelled sw_thermal through rc8), real Xid
# events become typed fault events, and everything NVIDIA said that does not fit
# the normalized fields lands in vendor_evidence.

from grip.agent.collect import _THROTTLE_BITS, NvmlProvider  # noqa: E402
from grip.schema import FaultSeverity  # noqa: E402


def test_nvmlcollector_is_an_alias_of_the_provider():
    assert NvmlCollector is NvmlProvider
    assert NvmlProvider.vendor == "nvidia"


@pytest.mark.parametrize(
    "bit, reason",
    [
        (0x04, ThrottleReason.power),  # SwPowerCap - the rc8 mislabel, corrected
        (0x08, ThrottleReason.sync_boost),
        (0x20, ThrottleReason.sw_thermal),
        (0x40, ThrottleReason.hw_slowdown),
        (0x80, ThrottleReason.thermal),
        (0x100, ThrottleReason.power),  # HwPowerBrake
    ],
)
def test_every_throttle_bit_maps_to_its_meaning(collector, bit, reason):
    c = collector(n=2, throttle_bits={0: bit})
    s = c.collect("node-a")[0]
    assert s.signals.throttle_reasons == [reason]
    assert s.signals.vendor_evidence["nvml_throttle_bits"] == bit
    assert _THROTTLE_BITS[bit] is reason


def test_sw_power_cap_and_sw_thermal_are_distinct_reasons(collector):
    """Both bits set: the sample carries both, in enum order, and the raw mask."""
    c = collector(n=1, throttle_bits={0: 0x04 | 0x20})
    s = c.collect("node-a")[0]
    assert s.signals.throttle_reasons == [ThrottleReason.power, ThrottleReason.sw_thermal]
    assert s.signals.vendor_evidence["nvml_throttle_bits"] == 0x24


def test_vendor_evidence_states_where_sm_activity_came_from(collector):
    c = collector(n=1)
    s = c.collect("node-a")[0]
    assert s.signals.vendor_evidence["sm_active_source"] == "nvml_utilization_gpu"
    assert s.signals.sm_activity_pct == s.signals.gpu_utilization_pct


def test_real_xid_events_become_typed_faults_and_the_legacy_list(collector):
    c = collector(n=4)
    assert c.fake.registered == [0, 1, 2, 3], "every device subscribed"
    c.fake.queued = [(2, 79), (2, 13), (0, 31)]
    samples = {s.gpu_index: s for s in c.collect("node-a")}

    g2 = samples[2].signals
    assert g2.xid_errors == [79, 13]
    assert [(e.vendor_code, e.severity) for e in g2.fault_events] == [
        ("xid:79", FaultSeverity.fatal),
        ("xid:13", FaultSeverity.fatal),
    ]
    assert g2.vendor_evidence["xid"] == [79, 13]
    assert g2.vendor_evidence["xid_source"] == "nvml_events"
    assert samples[0].signals.xid_errors == [31]
    assert samples[1].signals.fault_events == [] and samples[1].signals.xid_errors == []

    # drained: the next tick carries nothing
    assert all(s.signals.fault_events == [] for s in c.collect("node-a"))


def test_unsupported_event_api_is_stated_not_hidden(collector, capsys):
    c = collector(n=2, events="unsupported")
    s = c.collect("node-a")[0]
    assert s.signals.fault_events == [] and s.signals.xid_errors == []
    assert s.signals.vendor_evidence["xid_source"] == "unsupported"
    assert "Xid events unavailable" in capsys.readouterr().out


def test_event_drain_is_bounded_per_tick(collector):
    c = collector(n=1)
    c.fake.queued = [(0, 79)] * 500
    s = c.collect("node-a")[0]
    assert len(s.signals.fault_events) == 64
    assert len(c.fake.queued) == 436  # the rest wait for the next tick


def test_shutdown_frees_the_event_set(collector):
    c = collector(n=1)
    c.shutdown()
    assert c.fake.event_set_freed == 1 and c.fake.shutdown_calls == 1


def test_discover_and_runtime_are_vendor_tagged(collector):
    c = collector(n=2, style="old")
    gpus = c.discover()
    assert [g.vendor for g in gpus] == ["nvidia", "nvidia"]
    assert gpus[0].uuid == "GPU-fake-0000"
    rt = c.runtime()
    assert rt == {
        "driver_version": "550.90.07",
        "runtime_name": "cuda",
        "runtime_version": "12.4",
        "vendor_tools": [],
    }


def test_config_snapshot_is_a_readonly_query(monkeypatch):
    calls = []

    class R:
        returncode = 0
        stdout = "0, GPU-a, Enabled, Default, 400.00, 1410, 1593, Enabled, Disabled\n"

    def sh(cmd, check=True):
        calls.append(cmd)
        return R()

    rows = NvmlProvider.config_snapshot(sh=sh, which=lambda n: "nvidia-smi")
    assert rows == [
        {
            "index": "0",
            "uuid": "GPU-a",
            "persistence_mode": "Enabled",
            "compute_mode": "Default",
            "power.limit": "400.00",
            "clocks.applications.graphics": "1410",
            "clocks.applications.memory": "1593",
            "ecc.mode.current": "Enabled",
            "mig.mode.current": "Disabled",
        }
    ]
    assert calls[0][1].startswith("--query-gpu=") and len(calls[0]) == 3
    assert NvmlProvider.config_snapshot(sh=sh, which=lambda n: None) is None
