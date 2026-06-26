# Copyright (c) 2025, Tianchen Sun
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch


class IntegerTimeDelay:
    def __init__(self, num_envs, input_dim, delay_steps, device="cpu", dtype=torch.float32):
        """
        Integer time delay module using a FIFO queue to simulate time delay in control systems.

        Args:
            num_envs (int): Number of environments.
            input_dim (int): Dimension of the input tensor.
            delay_steps (int): Number of time steps to delay the input.
            device (str): Device to run the computations on.
            dtype (torch.dtype): Data type for the computations.
        """
        self.num_envs = num_envs
        self.input_dim = input_dim
        self.delay_steps = delay_steps
        self.device = device
        self.dtype = dtype

        # Initialize a FIFO queue to store past inputs
        self._queue = torch.zeros(num_envs, input_dim, delay_steps, device=device, dtype=dtype)
        self._index = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )  # Current index for each environment
        self._env_indices = torch.arange(num_envs, dtype=torch.long, device=device)

    def step(self, u):
        """
        Step the time delay module with the given u.
        Args:
            u (torch.Tensor): Input tensor of shape (num_envs, input_dim).
                              The input will be delayed by delay_steps.
        Returns:
            torch.Tensor: Delayed output tensor of shape (num_envs, input_dim).
        """
        if self.delay_steps <= 0:
            return u

        # Get the current index for each environment
        idx = self._index

        # Get the delayed output
        output = self._queue[self._env_indices, :, idx]

        # Update the queue with the new input
        self._queue[self._env_indices, :, idx] = u

        # Update the index for the next step
        self._index = (idx + 1) % self.delay_steps

        return output

    def reset(self, env_ids=None):
        """Reset the time delay module."""
        if env_ids is not None:
            self._queue[env_ids] = 0.0
            self._index[env_ids] = 0.0
        else:
            self._queue[:] = 0.0
            self._index[:] = 0.0
