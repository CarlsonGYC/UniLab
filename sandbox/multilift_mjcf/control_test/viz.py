#!/usr/bin/env python
"""Live in-viewer overlay for the circle tests: draws the commanded circle and the
current setpoint(s) into the passive viewer's user scene, so the GUI shows the
commanded trajectory vs. the achieved motion (the 3-D analogue of plot_log.py's
"setpoint vs achieved" panel)."""

from __future__ import annotations

import mujoco
import numpy as np


def _add_sphere(user_scn, pos, size, rgba) -> None:
    if user_scn.ngeom >= user_scn.maxgeom:
        return
    g = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([size, size, size], dtype=float),
        np.asarray(pos, dtype=float),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    user_scn.ngeom += 1


def draw_overlay(user_scn, circle_center, circle_radius, targets, n_ring: int = 72) -> None:
    """Reset the user scene and draw the commanded circle (green ring) + target markers.

    targets: list of ``(pos[3], size, rgba)`` spheres (e.g. payload + per-drone setpoints).
    """
    user_scn.ngeom = 0
    cx, cy, cz = (float(c) for c in circle_center)
    for k in range(n_ring if circle_radius > 1e-6 else 0):
        th = 2.0 * np.pi * k / n_ring
        _add_sphere(
            user_scn,
            (cx + circle_radius * np.cos(th), cy + circle_radius * np.sin(th), cz),
            0.012,
            (0.1, 0.85, 0.1, 1.0),
        )
    for pos, size, rgba in targets:
        _add_sphere(user_scn, pos, size, rgba)
