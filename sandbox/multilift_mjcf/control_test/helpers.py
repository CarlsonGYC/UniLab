# Copyright (c) 2026, The direct_rl Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the control tests: YAML config loading, CSV setpoint/actual
logging, and a tracking-error summary used to compare the commanded trajectory
against what the simulation actually achieved."""

from __future__ import annotations

import csv
import math
import os


def grid_origins(n: int, spacing: float) -> list[tuple[float, float, float]]:
    """``n`` env origins on a centered square grid in the x-y plane (z = 0)."""
    cols = int(math.ceil(math.sqrt(max(1, n))))
    off = 0.5 * (cols - 1) * spacing
    out = []
    for e in range(n):
        r, c = divmod(e, cols)
        out.append((c * spacing - off, r * spacing - off, 0.0))
    return out


def apply_drone_config(px4_cfg, drone_cfg, d: dict):
    """Align the sim controller with a drone's PX4 params (in place).

    Maps a ``charpi.yaml``-style dict onto the outer ``PX4ControlCfg``
    (``attitude`` + ``position``) and the rate-loop ``DroneControlCfg`` (``rate``).
    Unspecified keys keep their existing value.
    """
    if not d:
        return
    att = px4_cfg.attitude
    for k in (
        "roll_p",
        "pitch_p",
        "yaw_p",
        "yaw_weight",
        "rollrate_max_deg",
        "pitchrate_max_deg",
        "yawrate_max_deg",
    ):
        if k in d.get("attitude", {}):
            setattr(att, k, float(d["attitude"][k]))

    pos = px4_cfg.position
    p = d.get("position", {})
    for k in ("gain_pos_p", "gain_vel_p", "gain_vel_i", "gain_vel_d"):
        if k in p:
            setattr(pos, k, tuple(p[k]))
    for k in (
        "lim_vel_horizontal",
        "lim_vel_up",
        "lim_vel_down",
        "lim_thr_min",
        "lim_thr_max",
        "lim_thr_xy_margin",
        "tilt_max_deg",
        "hover_thrust",
    ):
        if k in p:
            setattr(pos, k, float(p[k]))

    r = d.get("rate", {})
    for k in ("kp", "ki", "kd", "kk", "k_ff"):
        if k in r:
            setattr(drone_cfg, k, tuple(r[k]))
    if "gyro_cutoff_hz" in r and "dgyro_cutoff_hz" in r:
        drone_cfg.cutoff_hz = (float(r["gyro_cutoff_hz"]), float(r["dgyro_cutoff_hz"]))


def load_config(path: str) -> dict:
    """Load a YAML config; returns {} if the file is missing."""
    if not path or not os.path.exists(path):
        print(f"[cfg] no config at '{path}', using built-in defaults")
        return {}
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


class CsvLogger:
    """Append-style CSV writer for per-control-step setpoint vs. actual rows."""

    def __init__(self, path: str, columns: list[str]):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.columns = columns
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(columns)
        self.n = 0

    def write(self, **kw):
        self._w.writerow([kw.get(c, "") for c in self.columns])
        self.n += 1

    def close(self):
        self._f.close()


def summarize_tracking(path: str, cruise_speed: float | None = None) -> str:
    """Read a tracking CSV and report position-error RMS/max + a phase-lag estimate.

    Expects columns ``t, x_sp,y_sp,z_sp, x,y,z`` (extra columns ignored). The lag
    estimate divides the steady-state error by the cruise speed → an effective
    delay (the dominant cause of circular-tracking error in a cascaded controller).
    """
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return f"[summary] {path}: empty"

    def g(row, k):
        return float(row[k])

    n = len(rows)
    se = [0.0, 0.0, 0.0]
    max_e = 0.0
    # use the second half (steady portion, after ramp-up) for the lag estimate
    half = rows[n // 2 :]
    se_h = 0.0
    for row in rows:
        ex = g(row, "x") - g(row, "x_sp")
        ey = g(row, "y") - g(row, "y_sp")
        ez = g(row, "z") - g(row, "z_sp")
        se[0] += ex * ex
        se[1] += ey * ey
        se[2] += ez * ez
        max_e = max(max_e, math.sqrt(ex * ex + ey * ey + ez * ez))
    for row in half:
        ex = g(row, "x") - g(row, "x_sp")
        ey = g(row, "y") - g(row, "y_sp")
        ez = g(row, "z") - g(row, "z_sp")
        se_h += ex * ex + ey * ey + ez * ez
    rms = [math.sqrt(s / n) for s in se]
    rms_xyz = math.sqrt(sum(se) / n)
    rms_steady = math.sqrt(se_h / max(1, len(half)))

    lines = [
        f"[summary] {path}  ({n} steps)",
        f"  pos-error RMS  x={rms[0]:.3f}  y={rms[1]:.3f}  z={rms[2]:.3f}  |xyz|={rms_xyz:.3f} m",
        f"  pos-error max  |xyz|={max_e:.3f} m",
        f"  steady (2nd half) RMS |xyz|={rms_steady:.3f} m",
    ]
    if cruise_speed and cruise_speed > 1e-6:
        lines.append(
            f"  ≈ effective tracking delay = steadyRMS/cruise = "
            f"{rms_steady / cruise_speed * 1000:.0f} ms  (cruise {cruise_speed:.2f} m/s)"
        )
    return "\n".join(lines)
