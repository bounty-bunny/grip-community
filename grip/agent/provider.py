"""The seam between a vendor's SDK and everything else.

An AcceleratorProvider is the only object in the agent that knows which vendor
made the GPUs. It turns the vendor's telemetry into normalized `GpuSignals`,
keeps whatever does not fit in `vendor_evidence`, and answers the two
operational questions the service commands ask: what is installed here, and
what is its configuration right now.

Today there is one implementation (NVML). The protocol exists so the second one
plugs in without touching the engine, and so `run.py` and `service.py` stop
constructing a vendor class by name.
"""

from __future__ import annotations

import importlib.util
import shutil
from typing import Protocol, runtime_checkable

from ..schema import AgentHello, DiscoveredGpu, GpuSample


@runtime_checkable
class AcceleratorProvider(Protocol):
    vendor: str
    count: int

    def discover(self) -> list[DiscoveredGpu]: ...
    def runtime(self) -> dict: ...
    def hello(self, cluster_token: str, node_id: str, agent_version: str = "") -> AgentHello: ...
    def collect(self, node_id: str) -> list[GpuSample]: ...
    def shutdown(self) -> None: ...

    @staticmethod
    def config_snapshot(sh, which) -> list[dict] | None: ...


def detect_vendor(which=shutil.which) -> str | None:
    """Which vendor's stack is present on this host, or None.

    NVIDIA takes priority because it is the validated adapter; AMD (amdsmi /
    amd-smi) is detected as the experimental second provider. On the rare host
    with both SDKs present, NVIDIA wins - set GRIP so one is visible if you need
    the other.
    """
    if importlib.util.find_spec("pynvml") is not None or which("nvidia-smi"):
        return "nvidia"
    if importlib.util.find_spec("amdsmi") is not None or which("amd-smi"):
        return "amd"
    return None


def provider_class(vendor: str):
    if vendor == "nvidia":
        from .collect import NvmlProvider  # noqa: PLC0415

        return NvmlProvider
    if vendor == "amd":
        from .amd import AmdSmiProvider  # noqa: PLC0415

        return AmdSmiProvider
    raise RuntimeError(f"no accelerator provider for vendor {vendor!r}")


def provider_for_host() -> AcceleratorProvider:
    """Construct the provider for whatever this host has. Raises with a plain
    sentence when there is nothing supported to talk to."""
    vendor = detect_vendor()
    if vendor is None:
        raise RuntimeError(
            "no supported accelerator SDK found on this host "
            "(NVIDIA needs pynvml: pip install 'grip[gpu]'; "
            "AMD needs amdsmi from ROCm, e.g. pip install /opt/rocm/share/amd_smi)"
        )
    return provider_class(vendor)()
