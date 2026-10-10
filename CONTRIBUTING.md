# Contributing to GRIP Community Edition

Thanks for your interest. Community Edition is the open telemetry foundation of GRIP:
the agent/collector, the telemetry schema, the vendor seam, the `Detector` interface,
and a transparent example detector. The proprietary reliability engine is not part of
this project.

## Developer Certificate of Origin (DCO)

This project uses the **DCO**, not a CLA. By contributing, you certify the
[Developer Certificate of Origin 1.1](https://developercertificate.org/): that you
wrote the change or have the right to submit it under the project's license.

Sign off every commit with `-s`:

```bash
git commit -s -m "your message"
```

This adds a `Signed-off-by: Your Name <your@email>` line. Commits without a sign-off
cannot be merged.

## Scope

In scope: collection, exporters, the `Detector` interface and alternative detectors,
docs, tests, platform/vendor adapters. Out of scope: anything that reproduces or
approximates the proprietary peer-relative detection method.

## Ground rules

- Keep the foundation independent — it must not import a commercial module.
- The agent stays passive by default and makes no GPU/driver configuration changes.
- Don't claim validated detection or failure prediction that the evidence doesn't
  support.
- Run `ruff check` and `pytest` before opening a pull request.

## License

Contributions are accepted under the [Apache License 2.0](LICENSE).
