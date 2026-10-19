"""G1: markdown tables from g1_trusted_bench_*.json (ratios = stock / trusted)."""

import glob
import json
import os
import sys

R = os.path.join(os.path.dirname(__file__), "..", "results")
PH = ["fit", "predict", "fit+predict", "predict_interval", "update", "predict_after_update", "total"]


def main(pattern):
    for p in sorted(glob.glob(os.path.join(R, pattern))):
        d = json.load(open(p))
        print(f"\n### {os.path.basename(p)}\n")
        modes = [k[6:] for k in d["rows"][0] if k.startswith("ratio_")] if d["rows"] else []
        for m in modes:
            print(f"mode `{m}`\n")
            print("| est | sc | cont | n | T | stock total s | " + " | ".join(PH)
                  + " | equal (pred,int,upd,cutoff,state) |")
            print("|" + "---|" * (6 + len(PH) + 1))
            for r in d["rows"]:
                if "error" in r:
                    print(f"| {r['est']} | {r['scitype']} | {r['cont']} | {r['n']} | "
                          f"{r['T']} | ERROR {r['error'][:60]} |")
                    continue
                rt = dict(r[f"ratio_{m}"])
                ts, tm = r["times"]["stock"], r["times"][m]
                rt["fit+predict"] = (ts["fit"] + ts["predict"]) / (tm["fit"] + tm["predict"])
                eq = r["equal"][m]
                ok = all(v is True for k, v in eq.items() if k != "state_mismatch")
                sm = eq.get("state_mismatch", "n/a")
                eqs = ("all True" if ok else str(eq)) + (
                    "" if sm == "n/a" else ("; state ok" if sm == [] else f"; {sm}"))
                tot = sum(r["times"]["stock"].values())
                print(f"| {r['est']} | {r['scitype'][:4]} | {r['cont']} | {r['n']} | "
                      f"{r['T']} | {tot:.2f} | "
                      + " | ".join(f"{rt[k]:.2f}" if k in rt else "-" for k in PH)
                      + f" | {eqs} |")
            print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "g1_trusted_bench_*.json")
