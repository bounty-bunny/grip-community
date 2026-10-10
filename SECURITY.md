# Security & privileges — GRIP Community Edition

## What it reads

Per GPU, through NVML: utilization, SM/memory clocks, temperature, power draw and
limit, memory used, fan, throttle reasons, ECC counters, PCIe replay counter, and
fault/XID events. Plus host identity: hostname, GPU model, UUID, driver version.
It does not read your job scripts, your data, your filesystem, or other users'
processes.

## What it does to the device

Nothing that changes it. Collection is passive: NVML reads, and `nvidia-smi` only
with read-only `--query-*`. **There is no code path in this edition that sets a
power limit, clock, persistence/compute mode, ECC or MIG setting.**

It is not strictly "read-only," and we say so plainly: the broader GRIP agent can
run a small, idle-gated GEMM benchmark (~1% of GPU time) because measuring
achieved throughput needs real work on the device. The Community
`monitor`/`export`/`metrics` commands do passive collection only and run no probe.

## What leaves your machine

Nothing. Community Edition is single-host and local. `metrics` serves Prometheus
on a port you choose; `export` writes a file you name. No account, no telemetry to
us, no cloud.

## Privileges

Everything above runs as an ordinary user. No root, no special capabilities. (The
systemd service installer in the full agent needs root to write a unit, and the
installed agent then runs unprivileged — not part of this edition's commands.)

## Verify it yourself

It is open source — read `grip/agent/collect.py` for exactly what is sampled, and
`grip/detect.py` for every rule the example detector applies. Report a security
issue by [opening a private advisory / emailing the maintainer — fill in before
release].
