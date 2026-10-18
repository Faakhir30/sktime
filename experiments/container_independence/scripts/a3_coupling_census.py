"""a3: census of time-index / forecasting-horizon coupling in sktime (static, regex).

Counts (non-test) lines and files that use pandas time-index semantics, split
into the forecasting module vs the rest. Output: results/a3_coupling_census.json

Run from WT root:
    PYTHONPATH=.:experiments/container_independence $EXP_PY \
        experiments/container_independence/scripts/a3_coupling_census.py
"""

import json
import os
import re
from collections import defaultdict

ROOT = "sktime"
PATTERNS = {
    # ForecastingHorizon API (framework-level, already abstracted behind fh)
    "fh.to_absolute(": r"\.to_absolute\(",
    "fh.to_absolute_index(": r"\.to_absolute_index\(",
    "fh.to_relative(": r"\.to_relative\(",
    "fh.to_absolute_int(": r"\.to_absolute_int\(",
    "fh.to_indexer(": r"\.to_indexer\(",
    "fh.get_expected_pred_idx(": r"get_expected_pred_idx\(",
    "fh.to_pandas()": r"\.to_pandas\(\)",
    # cutoff
    "self.cutoff / self._cutoff": r"self\._?cutoff\b",
    # pandas-only time semantics
    ".freq / .freqstr": r"\.freq(str)?\b",
    "pd.infer_freq / infer_freq": r"infer_freq\(",
    "to_offset / DateOffset / pd.offsets": r"to_offset\(|DateOffset|pd\.offsets\.",
    "PeriodIndex / pd.Period": r"PeriodIndex|pd\.Period\b|period_range\(",
    ".to_period( / .to_timestamp(": r"\.to_period\(|\.to_timestamp\(",
    "DatetimeIndex / date_range": r"DatetimeIndex|date_range\(",
    "RangeIndex": r"RangeIndex",
    "TimedeltaIndex / pd.Timedelta": r"TimedeltaIndex|pd\.Timedelta\b",
    "MultiIndex get_level_values(-1)/levels[-1]": r"get_level_values\(-1\)|levels\[-1\]",
    "sktime time utils (_shift/_get_duration/_coerce_duration_to_int)": (
        r"\b_shift\(|_get_duration\(|_coerce_duration_to_int\("
    ),
}


def walk():
    for d, _, files in os.walk(ROOT):
        if "/tests" in d or d.endswith("tests"):
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(d, f)


def main():
    res = {}
    comp = {k: re.compile(v) for k, v in PATTERNS.items()}
    lines = defaultdict(lambda: {"forecasting": 0, "other": 0})
    files = defaultdict(lambda: {"forecasting": set(), "other": set()})
    forecaster_files_any = set()
    for path in walk():
        area = "forecasting" if path.startswith("sktime/forecasting") else "other"
        with open(path, encoding="utf-8") as fh:
            txt = fh.read().splitlines()
        for ln in txt:
            s = ln.strip()
            if s.startswith("#"):
                continue
            for k, rx in comp.items():
                if rx.search(ln):
                    lines[k][area] += 1
                    files[k][area].add(path)
                    if area == "forecasting" and k not in (
                        "fh.to_pandas()",
                        "RangeIndex",
                    ):
                        forecaster_files_any.add(path)
    for k in PATTERNS:
        res[k] = {
            "lines_forecasting": lines[k]["forecasting"],
            "files_forecasting": len(files[k]["forecasting"]),
            "lines_other": lines[k]["other"],
            "files_other": len(files[k]["other"]),
        }
    n_fc_files = sum(
        1 for p in walk() if p.startswith("sktime/forecasting")
    )
    res["_summary"] = {
        "forecasting_py_files_nontest": n_fc_files,
        "forecasting_files_touching_any_time_semantics": len(forecaster_files_any),
    }
    # which forecasting files use pandas-only semantics (freq/period/offset)
    pandas_only_keys = [
        ".freq / .freqstr",
        "pd.infer_freq / infer_freq",
        "to_offset / DateOffset / pd.offsets",
        "PeriodIndex / pd.Period",
        ".to_period( / .to_timestamp(",
    ]
    po = set()
    for k in pandas_only_keys:
        po |= files[k]["forecasting"]
    res["_summary"]["forecasting_files_pandas_only_time_semantics"] = len(po)
    res["_summary"]["forecasting_files_pandas_only_list"] = sorted(po)
    os.makedirs("experiments/container_independence/results", exist_ok=True)
    out = "experiments/container_independence/results/a3_coupling_census.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    w = max(len(k) for k in PATTERNS)
    print(f"{'pattern':<{w}} | fc_lines fc_files | other_lines other_files")
    for k in PATTERNS:
        r = res[k]
        print(
            f"{k:<{w}} | {r['lines_forecasting']:8d} {r['files_forecasting']:8d} |"
            f" {r['lines_other']:11d} {r['files_other']:11d}"
        )
    s = res["_summary"]
    print({k: v for k, v in s.items() if not k.endswith("_list")})


if __name__ == "__main__":
    main()
