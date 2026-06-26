# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch


class Allocation:
    def __init__(self, num_envs, arm_length, moment_const, device="cpu", dtype=torch.float32):
        """
        Initializes the allocation matrix for a quadrotor for multiple environments.
        Control allocator computes the normalized PWM and thrust from the normalized wrench.
        Allocation module, instead, compute the ACTUAL wrench from the simulated rotor speed.

        Parameters:
        - num_envs (int): Number of environments
        - arm_length (float): Distance from the center to the rotor
        - thrust_const (float): Rotor thrust constant
        - moment_const (float): Rotor torque constant
        - device (str): 'cpu' or 'cuda'
        - dtype (torch.dtype): Desired tensor dtype
        """
        self.sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        self.arm_length = arm_length
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        moment_const_list = []
        for i in range(len(moment_const)):
            moment_const_list.append(torch.ones(num_envs, 1, device=device) * moment_const[i])
        self._moment_const = torch.cat(moment_const_list, dim=1)  # (num_envs, 4)

        # Build per-environment allocation matrices
        self._build_allocation_matrix()

    def _build_allocation_matrix(self):
        """Build batch allocation matrix from current parameters."""
        sqrt2_inv = self.sqrt2_inv
        arm_length = self.arm_length

        # Initialize allocation matrix for all environments
        # Shape: (num_envs, 4, 4)
        self._allocation_matrix = torch.zeros(
            (self.num_envs, 4, 4), dtype=self.dtype, device=self.device
        )

        # Fill in the constant rows (same for all environments)
        self._allocation_matrix[:, 0, :] = 1.0  # Thrust row
        self._allocation_matrix[:, 1, 0] = -arm_length * sqrt2_inv
        self._allocation_matrix[:, 1, 1] = arm_length * sqrt2_inv
        self._allocation_matrix[:, 1, 2] = arm_length * sqrt2_inv
        self._allocation_matrix[:, 1, 3] = -arm_length * sqrt2_inv

        self._allocation_matrix[:, 2, 0] = -arm_length * sqrt2_inv
        self._allocation_matrix[:, 2, 1] = arm_length * sqrt2_inv
        self._allocation_matrix[:, 2, 2] = -arm_length * sqrt2_inv
        self._allocation_matrix[:, 2, 3] = arm_length * sqrt2_inv

        # Fill in the moment_const row (per-environment)
        # moment_const shape: (num_envs, 4)
        self._allocation_matrix[:, 3, 0] = -self._moment_const[:, 0]
        self._allocation_matrix[:, 3, 1] = -self._moment_const[:, 1]
        self._allocation_matrix[:, 3, 2] = self._moment_const[:, 2]
        self._allocation_matrix[:, 3, 3] = self._moment_const[:, 3]

    def compute(self, rotor_thrust):
        """
        Computes the total thrust and body torques given the dynamics rotor thrust.

        Parameters:
        - rotor_thrust (torch.Tensor): Tensor of shape (num_envs, 4) representing dynamics rotor thrust

        Returns:
        - thrust_torque (torch.Tensor): Tensor of shape (num_envs, 4)
        """
        thrust_torque = torch.bmm(self._allocation_matrix, rotor_thrust.unsqueeze(-1)).squeeze(-1)
        return thrust_torque

    @property
    def moment_const(self) -> torch.Tensor:
        """Get the moment coefficient for the rotor control."""
        return self._moment_const

    @moment_const.setter
    def moment_const(self, value: torch.Tensor):
        """Set the moment coefficient for the rotor control."""
        self._moment_const = value
        # Build per-environment allocation matrices
        self._build_allocation_matrix()
