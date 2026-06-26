# Copyright (c) 2025, Tianchen Sun
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch


class SecondOrderResponseZOH:
    """Second-order system response with zero-order hold (ZOH) for discrete-time simulation.

    The second-order system is defined by the following differential equation:
        d2x/dt2 + 2*zeta*wn*dx/dt + wn^2*x = K*wn^2*u
    where:
        x: system output
        u: system input
        zeta: damping ratio
        wn: natural frequency (rad/s)
        K: system gain

    The discrete-time update is computed using the state-space representation and matrix exponentiation.

    Args:
        num_envs (int): Number of parallel environments.
        dt (float): Time step duration.
        damping (float): Damping ratio (zeta).
        natural_freq (float): Natural frequency (wn) in rad/s.
        gain (float): System gain (K).
        device (str or torch.device): Device to store tensors.
        dtype (torch.dtype): Data type for tensors.
    """

    def __init__(
        self, num_envs, dt, damping, natural_freq, gain=1.0, device="cpu", dtype=torch.float32
    ):
        self.num_envs = num_envs
        self.num_axes = len(damping)
        self.dt = dt
        self.damping = torch.tensor(damping, device=device, dtype=dtype)  # (num_axes,)
        self.natural_freq = torch.tensor(natural_freq, device=device, dtype=dtype)  # (num_axes,)
        self.gain = torch.tensor(gain, device=device, dtype=dtype)  # (num_axes,)

        # State vector: [x, dx/dt]
        self.state = torch.zeros(num_envs, self.num_axes, 2, device=device, dtype=dtype)

        self.Ad = torch.zeros(
            self.num_axes, 2, 2, device=device, dtype=dtype
        )  # State transition matrix
        self.Bd = torch.zeros(self.num_axes, 2, 1, device=device, dtype=dtype)  # Input matrix

        # Precompute the state transition matrix A and input matrix B for each axis
        for i in range(self.num_axes):
            wn = self.natural_freq[i].item()
            zeta = self.damping[i].item()
            K = self.gain[i].item()

            A = torch.tensor([[0.0, 1.0], [-(wn**2), -2 * zeta * wn]], device=device, dtype=dtype)

            B = torch.tensor([[0.0], [K * wn**2]], device=device, dtype=dtype)

            M = torch.zeros(3, 3, device=device, dtype=dtype)
            M[0:2, 0:2] = A * self.dt
            M[0:2, 2:3] = B * self.dt
            M[2, 2] = 1.0

            expM = torch.matrix_exp(M)
            self.Ad[i] = expM[0:2, 0:2]  # State transition matrix #(num_axes, 2, 2)
            self.Bd[i] = expM[0:2, 2:3]  # Input matrix #(num_axes, 2, 1)

    def reset(self, env_ids=None):
        """Reset the state of the specified environments to zero.

        Args:
            env_ids (torch.Tensor or None): Indices of environments to reset. If None, reset all.
        """
        self.state[env_ids] = 0.0

    def step(self, u):
        """Update the state based on the input using the discrete-time second-order system model.

        Args:
            u (torch.Tensor): Input tensor of shape (num_envs, num_axes).

        Returns:
            torch.Tensor: Updated output tensor of shape (num_envs, num_axes)
        """
        self.state = torch.einsum(
            "eai,aij->eaj", self.state, self.Ad.transpose(-1, -2)
        ) + torch.einsum("ea, aij->eaj", u, self.Bd.transpose(-1, -2))
        return self.state[:, :, 0]
