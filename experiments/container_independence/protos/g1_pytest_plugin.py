"""pytest plugin: run sktime tests with the G1 trusted path monkeypatched in.

G1_MODE env: off (baseline) | schema_items | vectorize_est.
At session end, prints how often the trusted path ran vs fell back.
"""
import os

import g1_trusted as g1

MODE = os.environ.get("G1_MODE", "schema_items")


def pytest_configure(config):
    if MODE != "off":
        g1.install(MODE)


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(
        f"G1_MODE={MODE} trusted_calls={g1.STATS['trusted']} "
        f"fallbacks={g1.STATS['fallback_reasons']}"
    )
