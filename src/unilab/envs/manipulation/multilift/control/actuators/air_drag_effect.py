# Copyright (c) 2025, Tianchen Sun
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch

from .. import quat_math as math_utils


class AirDragEffect:
    def __init__(
        self,
        num_envs,
        rotor_drag_const,
        rolling_moment_const,
        max_v_para_z,
        device="cpu",
        dtype=torch.float32,
    ):
        """
        Initializes the linear velocity effort model for a quadrotor, this includes:
        - rotor_axis force reduction due to linear velocity parallel to the rotor axis (body z-axis)
        - air drag model:

        Args:
        - num_envs (int): Number of environments
        - rotor_drag_const (float): Rotor drag coefficient
        - rolling_moment_const (float): Rolling moment coefficient
        - max_v_para_z (float): Maximum linear velocity parallel to the body z-axis for lift force computation
        - device (str): 'cpu' or 'cuda'
        - dtype (torch.dtype): Desired tensor dtype

        Parameters:
        - _max_v_para_z (float): Maximum linear velocity parallel to the body z-axis for lift force computation
        - _rotor_dir (tensor): Direction of each rotor, shape (num_envs, num_motors)
        - _scaling_factor (tensor): Scaling factor for rotor thrust based on linear velocity, shape (num_envs, 1)
        - _dyn_rotor_thrust (tensor): Dynamic rotor thrust after considering linear velocity, shape (num_envs, num_motors)
        - _air_drag_force_each_rotor (tensor): Air drag force on each rotor, shape (num_envs, num_motors, 3)
        - _rolling_moment (tensor): Rolling moment due to linear velocity, shape (num_envs, 3)
        """

        self._rotor_drag_const = (
            torch.ones(num_envs, 1, device=device, dtype=dtype) * rotor_drag_const
        )  # (num_envs, 1)
        self._rolling_moment_const = (
            torch.ones(num_envs, 1, device=device, dtype=dtype) * rolling_moment_const
        )  # (num_envs, 1)
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype

        # max linear velocity parallel to the body z-axis for lift force computation
        self._max_v_para_z = max_v_para_z

        self._rotor_dir = torch.tensor([1, -1, -1, 1], device=device, dtype=dtype).expand(
            num_envs, -1
        )  # (num_envs, num_motors)
        self._scaling_factor = torch.ones(num_envs, 1, device=device, dtype=dtype)  # (num_envs, 1)
        self._dyn_rotor_thrust = torch.zeros(
            num_envs, 4, device=device, dtype=dtype
        )  # (num_envs, num_motors)
        self._rolling_moment = torch.zeros(num_envs, 3, device=device, dtype=dtype)  # (num_envs, 3)
        self._air_drag_force_each_rotor = torch.zeros(
            num_envs, 4, 3, device=device, dtype=dtype
        )  # (num_envs, num_motors, 3)

        # temp tensor to avoid re-allocation
        self._temp_relative_wind_b = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._temp_omega_abs = torch.zeros(num_envs, 4, device=device, dtype=dtype)

        # velocity parallel and perpendicular to the body z-axis
        self._v_para_z = torch.zeros(num_envs, 1, device=device, dtype=dtype)
        self._v_norm_z = torch.zeros(num_envs, 3, device=device, dtype=dtype)

    def compute(self, linear_vel_w, root_quat_w, omega, wind_vel_w, static_rotor_thrust):
        """
        Compute the air drag, rolling moment based on the rotor angular velocities, and the drone orientation.

        Args:
        - linear_vel_w (tensor): linear velocity of the drone in world frame, shape (num_envs, 3)
        - root_quat_w (tensor): drone orientation quaternion (w, x, y, z) under the world frame, shape (num_envs, 4)
        - omega (tensor): actual rotor angular velocities, shape (num_envs, num_motors)
        - wind_vel_w (tensor): wind velocity in world frame, shape (num_envs, 3)
        - static_rotor_thrust (tensor): static rotor thrust forces, without considering the linear velocity effect shape (num_envs, num_motors)

        """
        # 1. Calculate the rotation matrix in the world frame, convert the body frame vector to world frame
        rotmat_w = math_utils.matrix_from_quat(root_quat_w)  # (num_envs, 3, 3)
        rotmat_b = rotmat_w.transpose(
            1, 2
        )  # transmition from world frame to body frame (num_envs, 3, 3)

        # 2. Calculate the linear velocity parallel and perpendicular to the body z-axis
        torch.sub(linear_vel_w, wind_vel_w, out=self._temp_relative_wind_b)  # (num_envs, 3)
        relative_wind_vel_b = torch.bmm(rotmat_b, self._temp_relative_wind_b.unsqueeze(-1)).squeeze(
            -1
        )  # (num_envs, 3)

        self._v_para_z = relative_wind_vel_b[:, 2:3]  # (num_envs, 1)
        self._v_norm_z[:, :2] = relative_wind_vel_b[:, :2]  # (num_envs, 3)

        # 3. Scale down the static rotor forces based on the linear velocity parallel to the body z-axis
        self._scaling_factor = torch.clamp(
            1
            - torch.norm(self._v_para_z, dim=-1, keepdim=True)
            / self._max_v_para_z,  # (num_envs, 1)
            min=0.0,
        )  # (num_envs, 1)
        self._dyn_rotor_thrust = (
            static_rotor_thrust * self._scaling_factor
        )  # (num_envs, num_motors)

        # 4. Calculate the air drag force in the body frame
        self._air_drag_force_each_rotor = (-torch.abs(omega) * self._rotor_drag_const)[
            :, :, None
        ] * self._v_norm_z[
            :, None, :
        ]  # (num_envs, num_motors, 1) * (num_envs, 1, 3) = (num_envs, num_motors, 3)

        # 5. Calculate the rolling moment
        self._rolling_moment = (
            torch.sum(
                -torch.abs(omega) * self._rotor_dir * self._rolling_moment_const,
                dim=-1,
                keepdim=True,
            )
            * self._v_norm_z
        )  # (num_envs, 1) * (num_envs, 3)

    @property
    def dynamic_rotor_thrust(self):
        """rotor thrust after considering the linear velocity effect"""
        return self._dyn_rotor_thrust  # (num_envs, num_motors)

    @property
    def air_drag_force_each_rotor(self):
        """air drag force on each rotor"""
        return self._air_drag_force_each_rotor  # (num_envs, num_motors, 3)

    @property
    def rolling_moment(self):
        """rolling moment due to the linear velocity, for the rigid body"""
        return self._rolling_moment  # (num_envs, 3)

    @property
    def rotor_drag_const(self):
        """Get the rotor drag coefficient"""
        return self._rotor_drag_const

    @rotor_drag_const.setter
    def rotor_drag_const(self, value: torch.Tensor):
        """Set the rotor drag coefficient"""
        self._rotor_drag_const = value
