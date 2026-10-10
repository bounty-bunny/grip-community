"""Public detector interface for the GRIP open telemetry foundation.

This is the stable, versioned contract between telemetry and detection. The open
foundation ships this interface plus one transparent, single-device example
detector (`ExampleThresholdDetector`).

GRIP's commercial product provides a *peer-relative reliability detector* that
implements this same `Detector` interface — it compares each GPU against
comparable peers to surface gray degradation that single-device thresholds miss.
That algorithm, and its tuned weights and thresholds, are **not** part of this
foundation and are not revealed here. The example below is deliberately simple and
reveals nothing proprietary.

Write your own by implementing `Detector`: take a list of `GpuSample` (the public
telemetry schema) and return a list of `Finding`. Nothing in this module imports
the commercial engine, so the open foundation runs without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from .schema import FaultSeverity, GpuSample, ThrottleReason

# Bump the minor version for additive, backward-compatible changes to Finding or
# Detector; the major version for a breaking change. A detector reports which
# version it was written against so a host can refuse an incompatible one.
DETECTOR_API_VERSION = "1.0"


class Severity(str, Enum):
    """Generic severity. Deliberately NOT the commercial product's health bands —
    a foundation detector states a severity, not a GRIP reliability verdict."""

    ok = "ok"
    notice = "notice"
    warn = "warn"
    critical = "critical"


_RANK = {Severity.ok: 0, Severity.notice: 1, Severity.warn: 2, Severity.critical: 3}


@dataclass
class Finding:
    """One detector observation about one GPU. `evidence` holds the raw values
    behind the call so a UI can show why, never a fabricated probability."""

    gpu_index: int
    severity: Severity
    signal: str
    message: str
    gpu_uuid: str = ""
    evidence: dict = field(default_factory=dict)


@runtime_checkable
class Detector(Protocol):
    """A detector turns telemetry into findings. Implement this to plug any
    detection logic into the foundation's CLI and exporters."""

    name: str
    version: str
    api_version: str

    def assess(self, samples: Sequence[GpuSample]) -> list[Finding]:
        """Return findings for the most recent reading of each GPU. An empty list
        means nothing notable. A detector that lacks the evidence to judge should
        abstain (return nothing) rather than guess."""
        ...


@dataclass
class ExampleThresholdDetector:
    """Transparent single-device checks. No peer comparison, no history, no tuning
    to hide — every rule is visible right here. It catches the obvious
    (over-temperature, active hardware slowdown, uncorrectable faults) and will, by
    design, MISS a GPU that is quietly slow while every absolute reading looks fine.
    That gap is exactly what the commercial peer-relative detector is for; this
    example is an honest floor, not a substitute.
    """

    temp_warn_c: float = 87.0
    temp_critical_c: float = 92.0
    name: str = "example-threshold"
    version: str = "1.0"
    api_version: str = DETECTOR_API_VERSION

    def assess(self, samples: Sequence[GpuSample]) -> list[Finding]:
        out: list[Finding] = []
        for s in samples:
            sig = s.signals
            base = {"gpu_index": s.gpu_index, "gpu_uuid": s.gpu_uuid}

            if sig.temperature_c >= self.temp_critical_c:
                out.append(Finding(severity=Severity.critical, signal="temperature_c",
                    message=f"temperature {sig.temperature_c:.0f}C >= {self.temp_critical_c:.0f}C",
                    evidence={"temperature_c": sig.temperature_c}, **base))
            elif sig.temperature_c >= self.temp_warn_c:
                out.append(Finding(severity=Severity.warn, signal="temperature_c",
                    message=f"temperature {sig.temperature_c:.0f}C >= {self.temp_warn_c:.0f}C",
                    evidence={"temperature_c": sig.temperature_c}, **base))

            # Hardware slowdown is usually a fault; thermal/power throttling is the
            # device protecting itself and is reported, not alarmed.
            if ThrottleReason.hw_slowdown in sig.throttle_reasons:
                out.append(Finding(severity=Severity.critical, signal="throttle",
                    message="hardware slowdown active (possible fault)",
                    evidence={"throttle_reasons": [t.value for t in sig.throttle_reasons]}, **base))
            elif ThrottleReason.thermal in sig.throttle_reasons:
                out.append(Finding(severity=Severity.warn, signal="throttle",
                    message="thermal throttling active",
                    evidence={"throttle_reasons": [t.value for t in sig.throttle_reasons]}, **base))

            for fe in sig.fault_events:
                if fe.severity in (FaultSeverity.fatal, FaultSeverity.uncorrectable):
                    out.append(Finding(severity=Severity.critical, signal="fault_event",
                        message=f"{fe.severity.value} fault: {fe.vendor_code}",
                        evidence={"vendor_code": fe.vendor_code,
                                  "severity": fe.severity.value}, **base))

            if sig.ecc_dbe_total > 0:
                out.append(Finding(severity=Severity.warn, signal="ecc_dbe_total",
                    message=f"{sig.ecc_dbe_total} uncorrectable ECC error(s) counted",
                    evidence={"ecc_dbe_total": sig.ecc_dbe_total}, **base))
        return out


def worst(findings: Sequence[Finding]) -> Severity:
    """Highest severity across findings, or ok if there are none."""
    return max((f.severity for f in findings), key=lambda s: _RANK[s], default=Severity.ok)
