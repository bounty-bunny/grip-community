"""Mock-level tests for the experimental AMD provider.

No AMD hardware or real amdsmi here, so these fake the amdsmi module and verify
the translation into GpuSignals, the skip-on-missing-field rule, and that nothing
imports amdsmi at module load.
"""

import sys
import types

import pytest

from grip.schema import Source


def _fake_amdsmi(metrics_override=None):
    m = types.ModuleType("amdsmi")
    handles = [10, 11]

    def metrics(h):
        base = {
            "current_gfxclk": 1700, "current_uclk": 1300,
            "temperature_edge": 55, "temperature_hotspot": 65, "temperature_mem": 58,
            "average_gfx_activity": 90, "throttle_status": 0, "pcie_replay_count": 0,
        }
        if metrics_override is not None:
            base.update(metrics_override)
        return base

    m.amdsmi_init = lambda: None
    m.amdsmi_shut_down = lambda: None
    m.amdsmi_get_processor_handles = lambda: handles
    m.amdsmi_get_gpu_asic_info = lambda h: {"market_name": "AMD Instinct MI300X"}
    m.amdsmi_get_gpu_vram_info = lambda h: {"vram_size": 196608}
    m.amdsmi_get_gpu_driver_info = lambda h: {"driver_version": "6.8.5"}
    m.amdsmi_get_gpu_device_uuid = lambda h: f"amd-uuid-{h}"
    m.amdsmi_get_gpu_metrics_info = metrics
    m.amdsmi_get_power_info = lambda h: {"current_socket_power": 350}
    m.amdsmi_get_power_cap_info = lambda h: {"power_cap": 560}
    m.amdsmi_get_gpu_activity = lambda h: {"gfx_activity": 90}
    m.amdsmi_get_gpu_vram_usage = lambda h: {"vram_used": 40000}
    m.amdsmi_get_gpu_total_ecc_count = lambda h: {"correctable_count": 0, "uncorrectable_count": 0}
    return m


@pytest.fixture
def amd(monkeypatch):
    monkeypatch.setitem(sys.modules, "amdsmi", _fake_amdsmi())
    from grip.agent.amd import AmdSmiProvider
    return AmdSmiProvider()


def test_discover(amd):
    gpus = amd.discover()
    assert len(gpus) == 2
    assert gpus[0].vendor == "amd"
    assert "MI300X" in gpus[0].model
    assert gpus[0].uuid == "amd-uuid-10"


def test_collect_translates_amdsmi_to_signals(amd):
    samples = amd.collect("mi300x-node")
    assert len(samples) == 2
    s = samples[0]
    assert s.source == Source.agent and s.node_id == "mi300x-node"
    sig = s.signals
    assert sig.sm_clock_mhz == 1700
    assert sig.temperature_c == 55
    assert sig.power_draw_w == 350
    assert sig.power_limit_w == 560
    assert sig.gpu_utilization_pct == 90
    assert sig.memory_used_mb == 40000
    assert sig.vendor_evidence["provider"] == "amd-experimental"
    # Deliberate AMD omissions, honestly empty:
    assert sig.throttle_reasons == [] and sig.fault_events == [] and sig.nvlink_crc_errors == 0


def test_skips_gpu_when_a_core_field_is_missing(monkeypatch):
    # No temperature anywhere -> partial sample must be skipped, not emitted.
    monkeypatch.setitem(sys.modules, "amdsmi",
                        _fake_amdsmi(metrics_override={"temperature_edge": None,
                                                       "temperature_hotspot": None}))
    from grip.agent.amd import AmdSmiProvider
    assert AmdSmiProvider().collect("n") == []


def test_hello_reports_amd_experimental(amd):
    hello = amd.hello("tok", "node", agent_version="0.1.0")
    assert "amd-experimental" in hello.agent_version
    assert hello.dcgm_available is False
    assert len(hello.gpus) == 2


def test_importable_without_amdsmi():
    # The module must import even where amdsmi is absent (lazy import).
    import importlib

    import grip.agent.amd as amd_mod
    importlib.reload(amd_mod)
    assert hasattr(amd_mod, "AmdSmiProvider")


def test_detect_vendor_none_without_any_sdk(monkeypatch):
    import grip.agent.provider as prov
    monkeypatch.setattr(prov.importlib.util, "find_spec", lambda name: None)
    assert prov.detect_vendor(which=lambda name: None) is None
    # amd-smi present -> amd
    def which_amd(name):
        return "/usr/bin/amd-smi" if name == "amd-smi" else None
    assert prov.detect_vendor(which=which_amd) == "amd"
