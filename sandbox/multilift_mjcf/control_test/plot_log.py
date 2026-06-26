# Copyright (c) 2026, The direct_rl Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize a control-test CSV log (setpoint vs. achieved) — saves a PNG.

Works for both the single-drone and multilift logs (long format with an ``env``
column; multilift positions are the payload). No Isaac Sim needed.

    python control_test/plot_log.py --log control_test/logs/single_circle.csv
    python control_test/plot_log.py --log .../multilift_circle.csv --cruise 0.65
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    by_env = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            e = int(row["env"]) if "env" in row and row["env"] != "" else 0
            for k, v in row.items():
                if k == "env":
                    continue
                try:
                    by_env[e][k].append(float(v))
                except (ValueError, TypeError):
                    pass
    return {e: {k: np.asarray(v) for k, v in cols.items()} for e, cols in by_env.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="", help="Output PNG (default: alongside the CSV).")
    ap.add_argument(
        "--cruise", type=float, default=0.0, help="Cruise speed [m/s] for the lag annotation."
    )
    a = ap.parse_args()
    if not os.path.exists(a.log):
        raise SystemExit(f"log not found: {a.log}")
    out = a.out or os.path.splitext(a.log)[0] + ".png"

    data = load(a.log)
    envs = sorted(data)
    e0 = data[envs[0]]
    tilt_key = (
        "tilt_deg" if "tilt_deg" in e0 else ("max_tilt_deg" if "max_tilt_deg" in e0 else None)
    )
    thr_key = "thrust" if "thrust" in e0 else ("thrust_mean" if "thrust_mean" in e0 else None)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{os.path.basename(a.log)}  ({len(envs)} env(s))", fontsize=12)

    # A: XY trajectory — setpoint (dashed) + each env's actual (thin)
    axA = ax[0, 0]
    axA.plot(e0["x_sp"], e0["y_sp"], "k--", lw=2, label="setpoint")
    for e in envs:
        d = data[e]
        axA.plot(d["x"], d["y"], lw=0.8, alpha=0.7)
    axA.set_title("XY trajectory (env-local)")
    axA.set_xlabel("x [m]")
    axA.set_ylabel("y [m]")
    axA.axis("equal")
    axA.legend(loc="upper right")
    axA.grid(alpha=0.3)

    # B: position error vs time — mean across envs + min/max band
    axB = ax[0, 1]
    t = e0["t"]
    errs = np.vstack([data[e]["err_pos"][: len(t)] for e in envs])
    axB.plot(t, errs.mean(0), "b", lw=1.6, label="mean |pos err|")
    if len(envs) > 1:
        axB.fill_between(t, errs.min(0), errs.max(0), color="b", alpha=0.15, label="min–max")
    if "drone_err" in e0:
        de = np.vstack([data[e]["drone_err"][: len(t)] for e in envs]).mean(0)
        axB.plot(t, de, "g", lw=1.2, label="mean drone-formation err")
    title = "position tracking error"
    if a.cruise > 0:
        steady = errs[:, len(t) // 2 :].mean()
        title += f"  (steady≈{steady:.3f} m ≈ {steady / a.cruise * 1000:.0f} ms lag)"
    axB.set_title(title)
    axB.set_xlabel("t [s]")
    axB.set_ylabel("error [m]")
    axB.legend(loc="upper right")
    axB.grid(alpha=0.3)

    # C: per-axis position vs time (env 0): setpoint vs actual
    axC = ax[1, 0]
    for k, c in (("x", "tab:red"), ("y", "tab:green"), ("z", "tab:blue")):
        axC.plot(t, e0[k + "_sp"], c, ls="--", lw=1.0)
        axC.plot(t, e0[k], c, lw=1.2, label=k)
    axC.set_title("position vs time — env 0 (dashed = setpoint)")
    axC.set_xlabel("t [s]")
    axC.set_ylabel("pos [m]")
    axC.legend(loc="upper right")
    axC.grid(alpha=0.3)

    # D: tilt + thrust vs time (env 0)
    axD = ax[1, 1]
    if tilt_key:
        axD.plot(t, e0[tilt_key], "tab:orange", lw=1.2, label="tilt [deg]")
    axD.set_xlabel("t [s]")
    axD.set_ylabel("tilt [deg]")
    if thr_key:
        axt = axD.twinx()
        axt.plot(t, e0[thr_key], "tab:purple", lw=1.0, alpha=0.8, label="thrust [norm]")
        axt.set_ylabel("thrust [norm]")
    axD.set_title("attitude tilt & collective thrust — env 0")
    axD.grid(alpha=0.3)
    axD.legend(loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=120)
    print(f"saved plot -> {out}")


if __name__ == "__main__":
    main()
