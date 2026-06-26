# Copyright (c) 2026, The direct_rl Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal quaternion→matrix helper, used to drop the single ``isaaclab.utils.math``
dependency in the ported control stack (``air_drag_effect.py``, ``position_controller.py``).

``matrix_from_quat`` is byte-for-byte the IsaacLab implementation (the same
PyTorch3D-style formula with ``two_s = 2/|q|^2`` so numerics are identical), so the
ported controller behaves exactly as it did under Isaac.
"""

from __future__ import annotations

import torch


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Rotation matrices from (w, x, y, z) quaternions. Shape (..., 4) -> (..., 3, 3)."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
