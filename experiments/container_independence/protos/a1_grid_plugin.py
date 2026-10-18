"""pytest plugin for a1: install EstimatorGrid patches when A1_GRID_MODE is set.

Usage: PYTHONPATH=<protos dir>:... A1_GRID_MODE=strict|compat pytest -p a1_grid_plugin
A1_GRID_MODE=off (default) installs nothing (baseline).
"""

import os


def pytest_configure(config):
    mode = os.environ.get("A1_GRID_MODE", "off")
    if mode in ("strict", "compat"):
        from a1_estimator_grid import install

        install(mode)
        print(f"\n[a1_grid_plugin] EstimatorGrid installed, mode={mode}")
