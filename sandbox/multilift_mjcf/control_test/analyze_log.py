# Copyright (c) 2026, The direct_rl Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline analysis of a control-test CSV log: setpoint vs. achieved tracking.

    python control_test/analyze_log.py --log control_test/logs/single_circle.csv --cruise 1.05

Prints per-axis RMS / max position error, the steady-state error, and an
effective tracking-delay estimate (steady error / cruise speed) — the quantity
that explains circular-tracking lag in a cascaded position→attitude→rate→motor
controller (no Isaac Sim needed).
"""

import argparse
import os

from helpers import summarize_tracking

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to a tracking CSV (from the circle tests).")
    ap.add_argument(
        "--cruise", type=float, default=0.0, help="Cruise speed [m/s] for the delay estimate."
    )
    a = ap.parse_args()
    if not os.path.exists(a.log):
        raise SystemExit(f"log not found: {a.log}")
    print(summarize_tracking(a.log, cruise_speed=a.cruise or None))
