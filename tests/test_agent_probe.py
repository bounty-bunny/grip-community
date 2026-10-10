"""GemmProbe against a fake torch.

The bug this pins: CUDA events belong to whichever device is *current* when they
are created. The probe set the tensors' device but never the context, so on an
8-GPU node every probe after device 0 timed device 0's idle stream. TFLOPS came
back ~1000x high and unstable, which poisoned power_efficiency - the highest
weighted signal - for every GPU on the node. Invisible on one GPU, invisible in
simulation.
"""

from __future__ import annotations

import sys

import pytest

from grip.agent.probe import GemmProbe


class _Tensor:
    def __init__(self, device: str, fake):
        self.device = device
        self.fake = fake

    def __matmul__(self, other):
        # record which device the work actually ran on
        self.fake.work_devices.append(int(self.device.split(":")[1]))
        return _Tensor(self.device, self.fake)


class _Event:
    def __init__(self, fake):
        self.fake = fake
        # an event is bound to the device that was current at creation
        self.created_on = fake.current
        self.recorded_on = None

    def record(self):
        self.recorded_on = self.fake.current

    def elapsed_time(self, end):
        # If the event was created on a device other than the one doing the work,
        # it timed an idle stream: near-zero, exactly as the real bug behaved.
        did_work = self.fake.work_devices[-1] if self.fake.work_devices else None
        if self.created_on != did_work:
            return 0.004
        return self.fake.real_ms


class _DeviceCtx:
    def __init__(self, fake, device):
        self.fake, self.device, self.prev = fake, device, None

    def __enter__(self):
        self.prev = self.fake.current
        self.fake.current = self.device
        self.fake.entered.append(self.device)
        return self

    def __exit__(self, *exc):
        self.fake.current = self.prev
        return False


class _Cuda:
    def __init__(self, fake):
        self.fake = fake

    def is_available(self):
        return True

    def device(self, d):
        return _DeviceCtx(self.fake, d)

    def synchronize(self, d=None):
        pass

    def Event(self, enable_timing=False):  # noqa: N802 - mirrors torch's API
        return _Event(self.fake)

    def mem_get_info(self, d=None):
        return self.fake.free_bytes, self.fake.total_bytes

    def empty_cache(self):
        self.fake.empty_cache_calls += 1


class FakeTorch:
    """Models the two behaviours that matter: events bind to the current device,
    and memory is something the probe has to ask for and give back."""

    bfloat16 = "bf16"

    def __init__(self, real_ms: float = 1127.0, free_gib: float = 40.0):
        self.current = 0  # torch's default current device
        self.entered: list[int] = []
        self.work_devices: list[int] = []
        self.real_ms = real_ms
        self.free_bytes = int(free_gib * 2**30)
        self.total_bytes = 40 * 2**30
        self.allocations = 0
        self.empty_cache_calls = 0
        self.cuda = _Cuda(self)

    def randn(self, *shape, device=None, dtype=None):
        self.allocations += 1
        return _Tensor(device, self)


@pytest.fixture
def probe(monkeypatch):
    def build(real_ms: float = 1127.0, free_gib: float = 40.0) -> tuple[GemmProbe, FakeTorch]:
        fake = FakeTorch(real_ms, free_gib)
        monkeypatch.setitem(sys.modules, "torch", fake)
        return GemmProbe(), fake

    return build


def test_probe_enters_the_target_device_context(probe):
    """The fix. Without it, events for device != 0 time the wrong stream."""
    p, fake = probe()
    p.run(3)
    assert fake.entered == [3], "run() must enter the target device's context"


def test_every_gpu_reports_a_plausible_result_not_just_device_zero(probe):
    """The exact failure seen on 8x A100: gpu0 sane, gpu1-7 ~1000x high."""
    p, fake = probe()
    results = [p.run(d) for d in range(8)]
    assert all(r is not None for r in results), "no GPU may be silently dropped"
    tflops = [r.gemm_tflops for r in results]
    assert len(set(tflops)) == 1, f"all devices must measure alike, got {tflops}"
    assert 300 < tflops[0] < 320, tflops[0]


def test_implausible_tflops_are_discarded_not_reported(probe):
    """A mis-timed probe must not reach the engine at all - power_efficiency is
    the highest-weighted signal and one bad value skews the whole node's peer
    median."""
    p, _ = probe(real_ms=0.002)
    assert p.run(0) is None


def test_non_positive_elapsed_time_is_discarded(probe):
    p, _ = probe(real_ms=0.0)
    assert p.run(0) is None


def test_result_shape_is_unchanged(probe):
    p, _ = probe()
    r = p.run(0)
    assert r.kind.value == "gemm"
    assert r.dtype == "bf16" and r.matrix_n == 16384
    assert r.duration_ms == pytest.approx(1127.0, abs=0.1)
    assert r.nccl_ring_gbps is None


def test_probe_is_a_noop_without_cuda(monkeypatch):
    fake = FakeTorch()
    fake.cuda.is_available = lambda: False
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert GemmProbe().run(0) is None


# ---------------------------------------------------------------- footprint
# The agent is not read-only: the probe does real work on the customer's GPU.
# These pin the promises made to an operator about exactly what that costs.


def test_footprint_is_three_matrices_and_is_stated(probe):
    """a, b and a@b in bf16. The number the pilot brief quotes must be the code's."""
    p, _ = probe()
    assert p.footprint_bytes == 3 * 16384 * 16384 * 2  # 1.5 GiB
    assert "1536 MiB" in p.describe(probe_every=8, interval_s=15.0)
    assert "120 s" in p.describe(probe_every=8, interval_s=15.0)


def test_nothing_stays_resident_between_probes(probe):
    """The original probe cached its matrices forever - ~1.5 GiB parked on every
    GPU for the life of the agent, an OOM on a 40 GB card running a job sized to
    fit. Now the buffers are dropped AND the cache is returned to the driver."""
    p, fake = probe()
    p.run(0)
    assert not hasattr(p, "_buffers"), "no buffer cache may exist"
    assert fake.empty_cache_calls == 1, "blocks must go back to the driver, not the cache"
    p.run(0)
    assert fake.allocations == 4, "each probe allocates fresh and frees"
    assert fake.empty_cache_calls == 2


def test_busy_gpu_is_skipped_not_contended_with(probe):
    """On a synchronous job a 1.3 s stall on one rank stalls every rank at the
    next collective. GRIP must not manufacture the straggler it measures."""
    p, fake = probe()
    assert p.run(2, busy_pct=85.0) is None
    assert fake.work_devices == [], "no GEMM may run on a busy GPU"
    assert fake.allocations == 0, "nothing may be allocated on a busy GPU"


def test_idle_gpu_is_probed(probe):
    p, fake = probe()
    assert p.run(2, busy_pct=2.0) is not None
    assert fake.work_devices, "an idle GPU is measured"


def test_idle_threshold_is_configurable(monkeypatch):
    fake = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake)
    p = GemmProbe(idle_max_pct=20.0)
    assert p.run(0, busy_pct=15.0) is not None


def test_insufficient_free_memory_skips_before_allocating(probe):
    """The probe must never be the reason a job OOMs. Needs footprint + 10%."""
    p, fake = probe(free_gib=1.0)  # 1 GiB free < 1.5 GiB * 1.1
    assert p.run(0) is None
    assert fake.allocations == 0, "must check before allocating, not after failing"
    assert fake.work_devices == []


def test_skip_is_logged_once_per_reason_not_every_tick(probe, capsys):
    """A busy node probed every 120 s must not produce a log line per GPU per tick."""
    p, _ = probe()
    for _ in range(5):
        p.run(0, busy_pct=90.0)
    out = capsys.readouterr().out
    assert out.count("skipped") == 1
    p.run(0, busy_pct=1.0)  # recovers -> next skip logs again
    p.run(0, busy_pct=90.0)
    assert capsys.readouterr().out.count("skipped") == 1


def test_no_busy_pct_means_no_gate(probe):
    """Callers without utilisation data (tests, tooling) still get a measurement."""
    p, _ = probe()
    assert p.run(0) is not None
