"""GRIP Community Edition — the open GPU telemetry foundation.

Collection, a transparent example detector, local monitoring, data export and a
Prometheus endpoint. This package imports only the open foundation (grip.agent,
grip.schema, grip.detect, grip.simulator) and never the commercial engine,
backend or auth — so it runs standalone.

What it is NOT: peer-relative reliability intelligence, fleet analytics or failure
prediction. Those are the commercial product. The example detector does
single-device threshold checks and will miss a GPU that is quietly slow while its
absolute readings look fine.
"""
