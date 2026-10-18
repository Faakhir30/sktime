"""final_run: serial runner for the final benchmark phase.

Runs one benchmark command (given after ``--``) with a wall-clock timeout,
records machine load (uptime) before/after, wall time, exit code and the max RSS
of the largest waited-for descendant (getrusage RUSAGE_CHILDREN), and appends a
record to results/final_runs.jsonl. Child stdout/stderr go to results/final_<name>.log.

Usage (from the worktree root)::

    $EXP_PY experiments/container_independence/scripts/final_run.py NAME TIMEOUT_S -- cmd...
"""

import json
import os
import resource
import subprocess
import sys
import time

ED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ED, "results")


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def main():
    name, timeout = sys.argv[1], float(sys.argv[2])
    cmd = sys.argv[sys.argv.index("--") + 1 :]
    log = os.path.join(RES, f"final_{name}.log")
    env = dict(os.environ)
    env["PYTHONPATH"] = ".:experiments/container_independence:experiments/container_independence/protos"
    rec = {"name": name, "cmd": " ".join(cmd), "uptime_before": uptime(), "start": time.strftime("%H:%M:%S")}
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write(f"# {rec['uptime_before']}\n# {rec['cmd']}\n")
        fh.flush()
        try:
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout)
            rec["exit"] = p.returncode
        except subprocess.TimeoutExpired:
            rec["exit"] = "timeout"
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["max_child_rss_mb"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)
    rec["uptime_after"] = uptime()
    with open(os.path.join(RES, "final_runs.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
