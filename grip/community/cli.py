"""grip-community: monitor GPUs locally, export telemetry, serve Prometheus metrics.

    python -m grip.community.cli discover
    python -m grip.community.cli monitor            # live table (needs a GPU)
    python -m grip.community.cli monitor --demo      # synthetic, no GPU needed
    python -m grip.community.cli export out.json
    python -m grip.community.cli metrics --port 9835

Imports only the open foundation. No commercial module is referenced, so this runs
with the engine/backend absent.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import typer

from ..detect import ExampleThresholdDetector, Severity, worst
from ..schema import GpuSample

app = typer.Typer(add_completion=False, help="GRIP Community Edition — open GPU telemetry.")
_DET = ExampleThresholdDetector()


def _demo_samples(node_id: str = "demo-node", n: int = 8) -> list[GpuSample]:
    """Synthetic healthy samples via the open simulator, so the tool demos with no
    GPU. Clearly synthetic — never present this as a measurement."""
    import numpy as np  # noqa: PLC0415

    from ..simulator.model import GpuNominal, SimGpu, sample  # noqa: PLC0415

    rng = np.random.default_rng(0)
    gpus = [SimGpu(index=i, uuid=f"GPU-demo-{i}", nominal=GpuNominal()) for i in range(n)]
    now = datetime.now(timezone.utc)
    return [sample(g, node_id, now, 1.0, rng, with_probe=False) for g in gpus]


def _live_samples(node_id: str) -> list[GpuSample]:
    from ..agent.provider import provider_for_host  # noqa: PLC0415

    collector = provider_for_host()
    try:
        return list(collector.collect(node_id))
    finally:
        collector.shutdown()


def _collect(demo: bool, node_id: str) -> list[GpuSample]:
    if demo:
        return _demo_samples(node_id)
    try:
        return _live_samples(node_id)
    except Exception as e:  # noqa: BLE001
        raise typer.BadParameter(
            f"could not read GPUs ({e}). No NVML/driver here? Try --demo for synthetic output."
        ) from e


@app.command()
def discover(node_id: str = typer.Option("local", help="node label")) -> None:
    """List the GPUs this host exposes."""
    samples = _collect(demo=False, node_id=node_id)
    typer.echo(f"{len(samples)} GPU(s) on {node_id}:")
    for s in samples:
        typer.echo(f"  [{s.gpu_index}] {s.gpu_uuid or '(no uuid)'}  "
                   f"{s.signals.temperature_c:.0f}C  {s.signals.power_draw_w:.0f}/"
                   f"{s.signals.power_limit_w:.0f}W")


@app.command()
def monitor(
    interval: float = typer.Option(5.0, help="seconds between samples"),
    once: bool = typer.Option(False, help="one pass then exit"),
    demo: bool = typer.Option(False, help="synthetic data, no GPU needed"),
    node_id: str = typer.Option("local"),
) -> None:
    """Live single-device health using the transparent example detector."""
    hdr = f"{'gpu':>3}  {'temp':>5}  {'power':>11}  {'util':>5}  {'clock':>6}  status"
    while True:
        samples = _collect(demo, node_id)
        findings = _DET.assess(samples)
        by_gpu: dict[int, list] = {}
        for f in findings:
            by_gpu.setdefault(f.gpu_index, []).append(f)
        typer.echo(f"\n{datetime.now().strftime('%H:%M:%S')}  node={node_id}"
                   + ("  [DEMO - synthetic]" if demo else ""))
        typer.echo(hdr)
        typer.echo("-" * len(hdr))
        for s in sorted(samples, key=lambda x: x.gpu_index):
            g = by_gpu.get(s.gpu_index, [])
            st = worst(g).value if g else "ok"
            note = ("  " + "; ".join(f.message for f in g)) if g else ""
            typer.echo(f"{s.gpu_index:>3}  {s.signals.temperature_c:>4.0f}C  "
                       f"{s.signals.power_draw_w:>4.0f}/{s.signals.power_limit_w:>4.0f}W  "
                       f"{s.signals.gpu_utilization_pct:>4.0f}%  {s.signals.sm_clock_mhz:>5.0f}  "
                       f"{st}{note}")
        if once:
            raise typer.Exit(0 if worst(findings) != Severity.critical else 2)
        time.sleep(max(0.5, interval))


@app.command()
def export(
    path: str = typer.Argument(..., help="output file; .json or .csv"),
    samples_n: int = typer.Option(1, "--samples", help="how many passes to capture"),
    interval: float = typer.Option(5.0),
    demo: bool = typer.Option(False),
    node_id: str = typer.Option("local"),
) -> None:
    """Capture telemetry to JSON or CSV for your own tooling."""
    rows: list[GpuSample] = []
    for i in range(samples_n):
        rows.extend(_collect(demo, node_id))
        if i < samples_n - 1:
            time.sleep(max(0.5, interval))
    if path.endswith(".csv"):
        import csv  # noqa: PLC0415

        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ts", "node_id", "gpu_index", "gpu_uuid", "temperature_c",
                        "power_draw_w", "power_limit_w", "gpu_utilization_pct", "sm_clock_mhz"])
            for s in rows:
                g = s.signals
                w.writerow([s.ts.isoformat(), s.node_id, s.gpu_index, s.gpu_uuid,
                            g.temperature_c, g.power_draw_w, g.power_limit_w,
                            g.gpu_utilization_pct, g.sm_clock_mhz])
    else:
        with open(path, "w") as fh:
            json.dump([json.loads(s.model_dump_json()) for s in rows], fh, indent=2)
    typer.echo(f"wrote {len(rows)} sample(s) to {path}")


def _prometheus(samples: list[GpuSample], findings) -> str:
    lines = [
        "# HELP grip_gpu_temperature_celsius GPU temperature.",
        "# TYPE grip_gpu_temperature_celsius gauge",
    ]
    def g(name, help_, typ="gauge"):
        lines.append(f"# HELP {name} {help_}")
        lines.append(f"# TYPE {name} {typ}")
    metrics = {
        "grip_gpu_power_watts": ("Power draw (W).", lambda s: s.signals.power_draw_w),
        "grip_gpu_power_limit_watts": ("Power limit (W).", lambda s: s.signals.power_limit_w),
        "grip_gpu_utilization_ratio": (
            "Utilization 0-1.", lambda s: s.signals.gpu_utilization_pct / 100.0),
        "grip_gpu_sm_clock_mhz": ("SM clock (MHz).", lambda s: s.signals.sm_clock_mhz),
        "grip_gpu_memory_used_mb": ("Memory used (MB).", lambda s: s.signals.memory_used_mb),
    }
    for s in samples:
        lab = f'gpu="{s.gpu_index}",uuid="{s.gpu_uuid}",node="{s.node_id}"'
        lines.append(f"grip_gpu_temperature_celsius{{{lab}}} {s.signals.temperature_c}")
    for name, (help_, fn) in metrics.items():
        g(name, help_)
        for s in samples:
            lab = f'gpu="{s.gpu_index}",uuid="{s.gpu_uuid}",node="{s.node_id}"'
            lines.append(f"{name}{{{lab}}} {fn(s)}")
    g("grip_gpu_finding", "1 per active example-detector finding.")
    for f in findings:
        lines.append(f'grip_gpu_finding{{gpu="{f.gpu_index}",severity="{f.severity.value}",'
                     f'signal="{f.signal}"}} 1')
    return "\n".join(lines) + "\n"


@app.command()
def metrics(
    port: int = typer.Option(9835, help="Prometheus scrape port"),
    demo: bool = typer.Option(False),
    node_id: str = typer.Option("local"),
) -> None:
    """Serve Prometheus-compatible metrics at http://<host>:<port>/metrics (stdlib only)."""
    from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: PLC0415

    cli = {"demo": demo, "node_id": node_id}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.rstrip("/") not in ("/metrics", ""):
                self.send_response(404)
                self.end_headers()
                return
            try:
                samples = _collect(cli["demo"], cli["node_id"])
                body = _prometheus(samples, _DET.assess(samples)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:  # noqa: BLE001
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"# scrape failed: {e}\n".encode())

    typer.echo(f"serving metrics on :{port}/metrics" + ("  [DEMO]" if demo else ""))
    HTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    app()
