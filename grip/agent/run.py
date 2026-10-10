"""grip-agent: sample NVML every interval, run the GEMM probe every probe_every
ticks, ship batches to the ingestion API.
"""

from __future__ import annotations

import socket
import time
from datetime import datetime, timezone

import httpx

from .. import __version__
from ..schema import Source, TelemetryBatch
from .provider import provider_for_host


def run(
    api: str = "http://localhost:8000",
    token: str = "dev",
    cluster_id: str | None = None,
    node_id: str | None = None,
    interval_s: float = 15.0,
    probe_every: int = 8,
    no_probe: bool = False,
    probe_n: int = 16384,
    probe_idle_max_pct: float = 5.0,
) -> None:
    node_id = node_id or socket.gethostname()
    collector = provider_for_host()
    print(
        f"[agent] node={node_id} vendor={collector.vendor} gpus={collector.count} api={api}",
        flush=True,
    )

    client = httpx.Client(
        base_url=api,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=15.0,
    )

    # enroll + auto-discover
    try:
        hello = collector.hello(token, node_id, agent_version=__version__)
        r = client.post("/api/agent/hello", content=hello.model_dump_json())
        if r.status_code == 200:
            body = r.json()
            print(
                f"[agent] enrolled to cluster {body.get('cluster_id')} - "
                f"{body.get('gpus_registered')} GPUs discovered",
                flush=True,
            )
        else:
            print(f"[agent] hello -> {r.status_code} {r.text[:120]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[agent] hello failed ({e}) - continuing with telemetry only", flush=True)

    probe = None
    if not no_probe:
        try:
            from .probe import GemmProbe  # noqa: PLC0415

            probe = GemmProbe(matrix_n=probe_n, idle_max_pct=probe_idle_max_pct)
            if not probe.available:
                print("[agent] CUDA not available - probe disabled", flush=True)
                probe = None
            else:
                # The one non-passive thing the agent does. Say exactly what it costs.
                print(f"[agent] probe: {probe.describe(probe_every, interval_s)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[agent] probe unavailable ({e}) - continuing telemetry-only", flush=True)

    tick = 0
    try:
        while True:
            t0 = time.time()
            samples = collector.collect(node_id)
            if probe is not None and tick % max(1, probe_every) == 0:
                for s in samples:
                    try:
                        s.probe = probe.run(s.gpu_index, busy_pct=s.signals.gpu_utilization_pct)
                    except Exception as e:  # noqa: BLE001
                        print(f"[agent] probe gpu{s.gpu_index} failed: {e}", flush=True)

            batch = TelemetryBatch(
                source=Source.agent,
                node_id=node_id,
                sent_at=datetime.now(timezone.utc),
                samples=samples,
            )
            try:
                client.post("/api/telemetry", content=batch.model_dump_json()).raise_for_status()
            except Exception as e:  # noqa: BLE001
                print(f"[agent] post failed: {e}", flush=True)

            tick += 1
            time.sleep(max(0.5, interval_s - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        collector.shutdown()
        client.close()
