"""On-device microbenchmark probe: a fixed-size GEMM per GPU.

Gives a repeatable TFLOPS number that, divided by concurrent power draw, is a
trustworthy perf/watt. Requires `pip install 'grip[probe]'` (torch). If torch or
CUDA is missing, probing is silently skipped and the engine falls back to the
sm_active/W proxy.

This is the one thing the agent does that is not passive, so its footprint is
stated here rather than discovered by an operator:

  VRAM   three n x n bf16 matrices (a, b, a@b) exist only while a probe runs -
         ~1.5 GiB at the default n=16384 - and are released to the driver
         afterwards, not left in PyTorch's cache. Nothing stays resident.
  SM     ~1.3 s at ~265 TFLOPS per probe, once every `probe_every` ticks.
  When   only on an idle GPU (utilisation <= idle_max_pct) with enough free
         VRAM. A busy GPU is skipped, never contended with: on a synchronous
         job a 1.3 s stall on one rank stalls every rank at the next collective,
         and GRIP must not manufacture the straggler it exists to measure.

NCCL ring bandwidth needs a multi-process launcher (torchrun); left as a hook -
`nccl_ring_gbps` stays None from the single-process agent for now.
"""

from __future__ import annotations

from ..schema import ProbeKind, ProbeResult

_N = 16384  # matrix dimension
_ITERS = 40
_IDLE_MAX_PCT = 5.0
_FREE_MARGIN = 1.10  # headroom over the transient footprint before we allocate

# No accelerator reaches this even at the lowest precision. A probe reporting
# more has mis-timed rather than measured, and must never reach the engine:
# power_efficiency is the highest-weighted signal, so one bad probe poisons the
# peer median for every GPU on the node.
_MAX_PLAUSIBLE_TFLOPS = 10_000.0


class GemmProbe:
    def __init__(
        self,
        matrix_n: int = _N,
        iters: int = _ITERS,
        idle_max_pct: float = _IDLE_MAX_PCT,
    ) -> None:
        import torch  # noqa: PLC0415

        self.torch = torch
        self.n = matrix_n
        self.iters = iters
        self.idle_max_pct = idle_max_pct
        self.available = torch.cuda.is_available()
        self._last_skip: dict[int, str] = {}

    @property
    def footprint_bytes(self) -> int:
        """Peak VRAM a probe transiently needs: a, b and their product, bf16."""
        return 3 * self.n * self.n * 2

    def describe(self, probe_every: int, interval_s: float) -> str:
        return (
            f"{self.n}^2 bf16 GEMM x{self.iters}, ~{self.footprint_bytes >> 20} MiB "
            f"transient per GPU, every {probe_every} ticks ({probe_every * interval_s:.0f} s), "
            f"only when utilisation <= {self.idle_max_pct:.0f}%"
        )

    def _skip(self, device: int, reason: str) -> None:
        # Log a skip only when its reason changes; a busy node must not be a log storm.
        if self._last_skip.get(device) != reason:
            print(f"[probe] gpu{device}: skipped - {reason}", flush=True)
            self._last_skip[device] = reason
        return None

    def run(self, device: int, *, busy_pct: float | None = None) -> ProbeResult | None:
        if not self.available:
            return None
        if busy_pct is not None and busy_pct > self.idle_max_pct:
            return self._skip(device, f"busy ({busy_pct:.0f}% > {self.idle_max_pct:.0f}%)")

        t = self.torch
        # CUDA events belong to whichever device is CURRENT when they are created.
        # Without this context every probe after device 0 recorded its events on
        # device 0's idle stream while the GEMM ran elsewhere, so elapsed_time
        # measured nothing and TFLOPS came back ~1000x high - and unstable.
        with t.cuda.device(device):
            free, _total = t.cuda.mem_get_info(device)
            need = int(self.footprint_bytes * _FREE_MARGIN)
            if free < need:
                return self._skip(device, f"{free >> 20} MiB free, probe needs {need >> 20} MiB")
            self._last_skip.pop(device, None)

            a = b = c = None
            try:
                a = t.randn(self.n, self.n, device=f"cuda:{device}", dtype=t.bfloat16)
                b = t.randn(self.n, self.n, device=f"cuda:{device}", dtype=t.bfloat16)
                for _ in range(3):  # warmup
                    (a @ b)
                t.cuda.synchronize(device)
                start = t.cuda.Event(enable_timing=True)
                end = t.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(self.iters):
                    c = a @ b
                end.record()
                t.cuda.synchronize(device)
                ms = start.elapsed_time(end)
            finally:
                # Dropping the references returns the blocks to PyTorch's cache,
                # where the training job would still see them as used. Hand them
                # back to the driver so the GPU is exactly as we found it.
                del a, b, c
                t.cuda.empty_cache()

        if ms <= 0.0:
            print(f"[probe] gpu{device}: non-positive elapsed time - discarding", flush=True)
            return None
        flop = 2.0 * self.n**3 * self.iters
        tflops = flop / (ms / 1000.0) / 1e12
        if tflops > _MAX_PLAUSIBLE_TFLOPS:
            print(
                f"[probe] gpu{device}: implausible {tflops:.0f} TFLOPS in {ms:.3f} ms "
                "- discarding rather than feeding the detector",
                flush=True,
            )
            return None
        return ProbeResult(
            kind=ProbeKind.gemm,
            duration_ms=round(ms, 1),
            gemm_tflops=round(tflops, 1),
            nccl_ring_gbps=None,
            matrix_n=self.n,
            dtype="bf16",
        )
