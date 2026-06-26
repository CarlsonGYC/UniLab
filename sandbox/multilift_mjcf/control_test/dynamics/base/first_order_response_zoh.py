# Copyright (c) 2025, Tianchen Sun
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch


class FirstOrderResponseZOH:
    """First-order system response with zero-order hold (ZOH) for discrete-time simulation.

    The first-order system is defined by the following differential equation:
        tau * dx/dt + x = u
    where:
        x: system output
        u: system input
        tau: time constant
    The discrete-time update is computed using the analytical solution of the first-order system.
    Args:
        num_envs (int): Number of parallel environments.
        dt (float): Time step duration.
        tau (float or list of float): Time constant(s) for each axis.
        device (str or torch.device): Device to store tensors.
        dtype (torch.dtype): Data type for tensors.
    """

    def __init__(self, num_envs, num_axes, dt, device="cpu", dtype=torch.float32):
        self.num_envs = num_envs
        self.num_axes = num_axes
        self.dt = dt

        # State vector: [x]
        self.state = torch.zeros(num_envs, self.num_axes, device=device, dtype=dtype)

    def step(self, u, tau):
        """Update the system state based on the input using ZOH assumption.

        Args:
            u (torch.Tensor): Input tensor of shape (num_envs, num_axes).
            tau (torch.Tensor): Time constant(s) for each axis. (num_envs, num_axes)

        Returns:
            torch.Tensor: Updated state tensor of shape (num_envs, num_axes).
        """
        beta = 1 - torch.exp(-self.dt / tau)  # (num_envs, num_axes)

        # Update the state using the discrete-time first-order response equation
        self.state = beta * u + (1 - beta) * self.state

        return self.state

    def reset(self, env_ids=None):
        """Reset the internal state to zero."""
        self.state[env_ids] = 0.0
