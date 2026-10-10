"""Install, stop and remove the agent as a systemd service - and prove what it touched.

An operator letting GRIP onto a node needs two things to be true and checkable:
that they can turn it off in one command, and that removing it leaves the node
exactly as it was found. So this module is deliberately narrow:

  - every path it creates is listed in `MANAGED_PATHS`, and uninstall removes
    exactly those and nothing else
  - the only nvidia-smi invocation anywhere in this module is a read-only
    `--query-gpu`; there is no code path that passes -pl, -lgc, -rgc, -ac, -pm,
    -e, -c or -mig, and a test greps for that
  - at install it snapshots each GPU's *configuration* (persistence mode, compute
    mode, power limit, application clocks, ECC and MIG modes); at uninstall it
    takes the same snapshot again and prints the diff, which should be empty

Linux/systemd only. On anything else every command refuses before touching a file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .collect import CONFIG_SNAPSHOT_FIELDS
from .provider import detect_vendor, provider_class

UNIT_NAME = "grip-agent"
UNIT_PATH = Path("/etc/systemd/system") / f"{UNIT_NAME}.service"
CONF_DIR = Path("/etc/grip")
ENV_PATH = CONF_DIR / "agent.env"
STATE_DIR = Path("/var/lib/grip")
SNAPSHOT_PATH = STATE_DIR / "gpu-config-at-install.json"

# Everything install() may create. Uninstall removes exactly this set.
MANAGED_PATHS: tuple[Path, ...] = (UNIT_PATH, ENV_PATH, CONF_DIR, SNAPSHOT_PATH, STATE_DIR)

# GPU *configuration* only - nothing that changes on its own from tick to tick.
# If any of these differ between install and uninstall, something other than the
# agent changed them (the agent has no code that can), and the operator should
# know either way. The field list and the read live with the vendor provider.
_CONFIG_FIELDS = CONFIG_SNAPSHOT_FIELDS


class ServiceError(RuntimeError):
    """Anything the operator should read as a sentence, not a traceback."""


class NotSupported(ServiceError):
    pass


def _require_linux_root() -> None:
    if sys.platform != "linux":
        raise NotSupported("grip service commands manage a systemd unit and need Linux")
    if os.geteuid() != 0:  # type: ignore[attr-defined]
        raise NotSupported("run as root (or with sudo): this writes /etc and a systemd unit")


def _sh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Single choke point for every external command, so a test can see all of them."""
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise ServiceError(f"{cmd[0]} not found - is this a systemd host?") from e
    except subprocess.CalledProcessError as e:
        raise ServiceError(f"{' '.join(cmd)} failed: {(e.stderr or e.stdout or '').strip()}") from e


# ---------------------------------------------------------------- GPU config snapshot


def gpu_config_snapshot() -> list[dict] | None:
    """Read-only, through the vendor provider. Returns None when no supported
    vendor tooling is present rather than guessing."""
    vendor = detect_vendor(which=shutil.which)
    if vendor is None:
        return None
    return provider_class(vendor).config_snapshot(sh=_sh, which=shutil.which)


def diff_snapshots(before: list[dict] | None, after: list[dict] | None) -> list[str]:
    """Human-readable differences, keyed by GPU uuid. Empty means untouched."""
    if before is None or after is None:
        return ["GPU configuration could not be compared: nvidia-smi unavailable"]
    b = {r["uuid"]: r for r in before}
    a = {r["uuid"]: r for r in after}
    out: list[str] = []
    for uuid in sorted(set(b) | set(a)):
        if uuid not in a:
            out.append(f"{uuid}: present at install, absent now")
            continue
        if uuid not in b:
            out.append(f"{uuid}: absent at install, present now")
            continue
        for k in _CONFIG_FIELDS:
            if k in ("index", "uuid"):
                continue
            if b[uuid].get(k) != a[uuid].get(k):
                out.append(
                    f"gpu {a[uuid]['index']} ({uuid}): {k} {b[uuid].get(k)} -> {a[uuid].get(k)}"
                )
    return out


# ---------------------------------------------------------------- unit + env


@dataclass(frozen=True)
class AgentConfig:
    api: str
    machine_token: str
    node_id: str
    interval_s: float = 15.0
    probe_every: int = 8
    probe_idle_max_pct: float = 5.0
    probe_n: int = 16384
    no_probe: bool = False

    def env_file(self) -> str:
        lines = [
            "# GRIP agent configuration. Written by `grip install-service`; removed by",
            "# `grip uninstall`. The machine token is revocable from the GRIP dashboard.",
            f"GRIP_API={self.api}",
            f"GRIP_MACHINE_TOKEN={self.machine_token}",
            f"GRIP_NODE_ID={self.node_id}",
            f"GRIP_AGENT_INTERVAL={self.interval_s:g}",
            f"GRIP_AGENT_PROBE_EVERY={self.probe_every}",
            f"GRIP_AGENT_PROBE_IDLE_MAX={self.probe_idle_max_pct:g}",
            f"GRIP_AGENT_PROBE_N={self.probe_n}",
            f"GRIP_AGENT_NO_PROBE={'true' if self.no_probe else 'false'}",
        ]
        return "\n".join(lines) + "\n"


def render_unit(grip_exe: str) -> str:
    """The systemd unit. Hardened so the unit itself says what the agent cannot do:
    no new privileges, read-only system, no home access. NVML needs only /dev."""
    return f"""[Unit]
Description=GRIP GPU reliability agent (passive NVML telemetry + bounded idle-only probe)
Documentation=https://griplab.dev
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile={ENV_PATH}
# No arguments: the agent reads GRIP_* from the environment file, so the machine
# token never appears in `ps` or `systemctl show`.
ExecStart={grip_exe} agent
Restart=on-failure
RestartSec=10
# The agent reads NVML and posts JSON. It has no reason to write anywhere.
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadWritePaths={STATE_DIR}

[Install]
WantedBy=multi-user.target
"""


# ---------------------------------------------------------------- commands


def install(cfg: AgentConfig, grip_exe: str | None = None) -> list[str]:
    """Create the unit, the env file and the install-time GPU snapshot; start it.
    Returns the list of paths created, in order, for the operator's record."""
    _require_linux_root()
    exe = grip_exe or shutil.which("grip") or f"{sys.executable} -m grip.cli"
    created: list[str] = []

    CONF_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    created.append(str(CONF_DIR))
    ENV_PATH.write_text(cfg.env_file())
    os.chmod(ENV_PATH, 0o600)
    created.append(str(ENV_PATH))

    STATE_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
    created.append(str(STATE_DIR))
    SNAPSHOT_PATH.write_text(
        json.dumps(
            {
                "taken_at": datetime.now(timezone.utc).isoformat(),
                "fields": list(_CONFIG_FIELDS),
                "gpus": gpu_config_snapshot(),
            },
            indent=2,
        )
    )
    created.append(str(SNAPSHOT_PATH))

    UNIT_PATH.write_text(render_unit(exe))
    created.append(str(UNIT_PATH))

    _sh(["systemctl", "daemon-reload"])
    _sh(["systemctl", "enable", "--now", UNIT_NAME])
    return created


def start() -> None:
    _require_linux_root()
    _sh(["systemctl", "start", UNIT_NAME])


def stop() -> None:
    """Stops collection and probing. Touches nothing else - the unit, config and
    snapshot stay so `grip start` resumes exactly as before."""
    _require_linux_root()
    _sh(["systemctl", "stop", UNIT_NAME])


def status() -> dict:
    _require_linux_root()
    active = _sh(["systemctl", "is-active", UNIT_NAME], check=False).stdout.strip()
    enabled = _sh(["systemctl", "is-enabled", UNIT_NAME], check=False).stdout.strip()
    return {
        "unit": UNIT_NAME,
        "active": active,
        "enabled": enabled,
        "installed": UNIT_PATH.exists(),
    }


def uninstall(purge_package: bool = False) -> dict:
    """Stop, disable, remove every managed path, then prove the GPUs are as found.

    Returns what was removed and the GPU-config diff against the install-time
    snapshot. The diff is computed BEFORE the snapshot file is deleted and is
    the operator's evidence that nothing but GRIP's own files changed.
    """
    _require_linux_root()
    removed: list[str] = []

    if UNIT_PATH.exists():
        _sh(["systemctl", "disable", "--now", UNIT_NAME], check=False)
    before = None
    if SNAPSHOT_PATH.exists():
        try:
            before = json.loads(SNAPSHOT_PATH.read_text()).get("gpus")
        except (ValueError, OSError):
            before = None
    diff = (
        diff_snapshots(before, gpu_config_snapshot())
        if SNAPSHOT_PATH.exists()
        else ["no install-time snapshot found; GPU configuration not compared"]
    )

    for p in (UNIT_PATH, ENV_PATH, SNAPSHOT_PATH):
        if p.exists():
            p.unlink()
            removed.append(str(p))
    for d in (CONF_DIR, STATE_DIR):
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
            removed.append(str(d))
    _sh(["systemctl", "daemon-reload"], check=False)

    if purge_package:
        _sh([sys.executable, "-m", "pip", "uninstall", "-y", "grip"], check=False)
        removed.append("python package 'grip'")

    return {"removed": removed, "gpu_config_diff": diff}
