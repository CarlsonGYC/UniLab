# Copyright (c) 2025, Tianchen Sun
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch


class ControlAllocator:
    def __init__(
        self,
        num_envs,
        rotor_pos_com,
        ctl_thrust_const,
        ctl_moment_const,
        device="cpu",
        dtype=torch.float32,
    ):
        """Control allocator for quadcopters.
        Control allocator computes the normalized PWM and thrust from the normalized wrench.
        Allocation module, instead, simulates the motor speed to the ACTUAL wrench.

        1. Transfer the normalized torque and thrust to the normalized PWM.
        2. Normalized PWM to the Normalized Thrust based on the thrust curve.

        Args:
            num_envs (int): Number of environments.
            rotor_pos_com(float): The position of the rotors in xy axes w.r.t the CoM.
                           This value may not be the true geometry value, due to the PX4 wrong configuration.
            ctl_thrust_const (float): Thrust coefficient for the rotor control, thrust = thrust_const * u^2.
            ctl_moment_const (float): Moment coefficient for the rotor control, moment = moment_coeff * thrust.
                           This value may not be the true geometry value, due to the PX4 wrong configuration.
            device (str): Device to run the computations on.
            dtype (torch.dtype): Data type for the computations.

        """
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        pos_com = rotor_pos_com
        ct = ctl_thrust_const
        km = ctl_moment_const

        # construction of the actuator effectiveness matrix with the px4 controller configuration
        # for quad_x_typical_config:
        eff_matrix = torch.tensor(
            [
                [ct, ct, ct, ct],
                [-pos_com * ct, pos_com * ct, pos_com * ct, -pos_com * ct],
                [-pos_com * ct, pos_com * ct, -pos_com * ct, pos_com * ct],
                [-km * ct, -km * ct, km * ct, km * ct],
            ],
            dtype=dtype,
            device=device,
        )
        # pseudo-inverse of the allocation matrix
        ctl_allocator = get_control_allocator(eff_matrix)

        # scale the control allocator
        ctl_allocator = normalize_control_allocator(ctl_allocator)

        # Repeat the ctl_allocator for each environment
        self._ctl_allocators = ctl_allocator.unsqueeze(0).repeat(num_envs, 1, 1)  # (num_envs, 4, 4)

    def compute(self, norm_thrust, norm_torque):
        return compute_control_allocation(self._ctl_allocators, norm_thrust, norm_torque)


def get_control_allocator(eff_matrix):
    return torch.linalg.pinv(eff_matrix)


def normalize_control_allocator(ctl_allocator):
    """
    This method is called since we choose CA_METHOD = 0 Pseudo-inverse with output clipping
    Normalize the control allocator matrix. for stable inner loop control, this follows
    the PX4 logic src/modules/control_allocator/ControlAllocation/ControlAllocationPseudoInverse.cpp
    """
    num_non_zero_roll_torque = 4
    num_non_zero_pitch_torque = 4
    num_non_zero_thrust = 4

    # scale for thrust
    thrust_norm_scale = torch.sum(ctl_allocator[:, 0]) / num_non_zero_thrust

    # scale for roll and pitch torque
    roll_norm_scale = torch.sqrt(
        torch.sum(ctl_allocator[:, 1] ** 2) / (num_non_zero_roll_torque / 2)
    )
    pitch_norm_scale = torch.sqrt(
        torch.sum(ctl_allocator[:, 2] ** 2) / (num_non_zero_pitch_torque / 2)
    )

    # yaw scale is the larget element absolute value in the fourth column
    yaw_norm_scale = torch.max(torch.abs(ctl_allocator[:, 3]))

    rp_scale = max(roll_norm_scale, pitch_norm_scale)
    ctl_allocator[:, 0] = ctl_allocator[:, 0] / thrust_norm_scale
    ctl_allocator[:, 1] = ctl_allocator[:, 1] / rp_scale
    ctl_allocator[:, 2] = ctl_allocator[:, 2] / rp_scale
    ctl_allocator[:, 3] = ctl_allocator[:, 3] / yaw_norm_scale

    return ctl_allocator


@torch.jit.script
def compute_control_allocation(ctl_allocators, norm_thrust, norm_torque):
    """Compute the control allocation based on the normalized thrust and torque.
    Args:
        ctl_allocators (torch.Tensor): Control allocation matrix (num_envs, 4, 4).
        norm_thrust (torch.Tensor): Normalized thrust in [ 0, 1] (num_envs, 1).
        norm_torque (torch.Tensor): Normalized torque in [-1, 1] (num_envs, 3).
    Returns:
        torch.Tensor: Computed actuator setpoint u (num_envs, 4).
    """
    wrench_vec = torch.cat([norm_thrust, norm_torque], dim=1).unsqueeze(-1)  # (num_envs, 4, 1)
    u_vec = torch.bmm(ctl_allocators, wrench_vec).squeeze(-1)  # (num_envs, 4)
    return torch.clamp(u_vec, min=0.0, max=1.0)  # (num_envs, 4)
