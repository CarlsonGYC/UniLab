# Copyright (c) 2026, The direct_rl Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Drone dynamics + low-level control, ported verbatim from isaac_dreamer_drone
(`isaac_drone_dynamics`). The numerics/control are byte-for-byte the same as the
isaac_dreamer 5_in_drone stack — only the internal import paths were changed.

The only piece NOT ported is the env-coupled action term (`envs/mdp/actions.py`,
which depends on ManagerBasedRLEnv + the isaac_drone_racer logger). Its compute
logic is reproduced framework-agnostically in :mod:`control_stack` so it can be
dropped into a DirectRLEnv `_apply_action` or a standalone script.
"""

from .actuators.air_drag_effect import AirDragEffect  # noqa: F401
from .actuators.allocation import Allocation  # noqa: F401
from .actuators.motor import Motor  # noqa: F401
from .base.first_order_response_zoh import FirstOrderResponseZOH  # noqa: F401
from .base.second_order_response_zoh import SecondOrderResponseZOH  # noqa: F401
from .base.time_delay import IntegerTimeDelay  # noqa: F401
from .control_stack import CTBRControlStack, DroneControlCfg  # noqa: F401
from .filter.low_pass_filter import ButterworthFilter  # noqa: F401
from .low_level_control.body_rate_controller import BodyRateController  # noqa: F401
from .low_level_control.control_allocator import ControlAllocator  # noqa: F401
from .position_controller import ButterworthPositionController, PositionControllerCfg  # noqa: F401
from .px4_control import (  # noqa: F401
    PX4AttitudeControl,
    PX4AttitudeControlCfg,
    PX4ControlCfg,
    PX4PositionAttitudeController,
    PX4PositionControl,
    PX4PositionControlCfg,
    hover_thrust_for_thrust,
)

__all__ = [
    "IntegerTimeDelay",
    "FirstOrderResponseZOH",
    "SecondOrderResponseZOH",
    "ButterworthFilter",
    "Allocation",
    "Motor",
    "AirDragEffect",
    "BodyRateController",
    "ControlAllocator",
    "CTBRControlStack",
    "DroneControlCfg",
    "ButterworthPositionController",
    "PositionControllerCfg",
    # PX4 position + attitude controllers (PVA → CTBR command)
    "PX4PositionControl",
    "PX4AttitudeControl",
    "PX4PositionAttitudeController",
    "PX4PositionControlCfg",
    "PX4AttitudeControlCfg",
    "PX4ControlCfg",
    "hover_thrust_for_thrust",
]
