# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch

from ..base.first_order_response_zoh import FirstOrderResponseZOH


class Motor:
    def __init__(
        self,
        num_envs,
        taus_up,
        taus_down,
        init,
        max_rotor_acc,
        min_rotor_acc,
        dt,
        use,
        device="cpu",
        dtype=torch.float32,
    ):
        """
        Initializes the motor model.

        Parameters:
        - num_envs: Number of envs.
        - init: (4,) Tensor or list specifying the initial omega per motor. (rad/s)
        - taus_up: (4,) Tensor or list specifying time constants per motor.
        - taus_down: (4,) Tensor or list specifying time constants per motor.
        - beta_up: (4,) Tensor or list specifying beta values per motor. (derived from taus_up)
        - beta_down: (4,) Tensor or list specifying beta values per motor. (derived from taus_down)
        - max_rotor_acc: (4,) Tensor or list specifying max rate of change of omega per motor. (rad/s^2)
        - min_rotor_acc: (4,) Tensor or list specifying min rate of change of omega per motor. (rad/s^2)
        - dt: Time step for integration.
        - use: Boolean indicating whether to use motor dynamics.
        - device: 'cpu' or 'cuda' for tensor operations.
        - dtype: Data type for tensors.
        """
        self.num_envs = num_envs
        self.num_motors = 4
        self.dt = dt
        self.use = use
        self.init = init
        self.device = device
        self.dtype = dtype

        self.omega = (
            torch.tensor(init, device=device).expand(num_envs, -1).clone()
        )  # (num_envs, num_motors)

        # Convert to tensors and expand for all drones
        self.taus_up = torch.ones(num_envs, 4, device=device) * torch.tensor(
            taus_up, device=device
        )  # (num_envs, num_motors)
        self.taus_down = torch.ones(num_envs, 4, device=device) * torch.tensor(
            taus_down, device=device
        )  # (num_envs, num_motors)

        # Create FirstOrderResponseZOH instance for motor dynamics
        self.response = FirstOrderResponseZOH(
            num_envs, self.num_motors, dt, device=device, dtype=dtype
        )
        # start the first-order lag state at idle speed (NOT 0) so the motor does not ramp
        # up from 0 rad/s on the first step / after every reset (would cause a thrust dip).
        self.response.state[:] = self.omega

        # Set the max and min rotor acceleration limits
        self.max_rotor_acc = torch.tensor(max_rotor_acc, device=device).expand(
            num_envs, -1
        )  # (num_envs, num_motors)
        self.min_rotor_acc = torch.tensor(min_rotor_acc, device=device).expand(
            num_envs, -1
        )  # (num_envs, num_motors)

    def compute(self, omega_ref):
        """
        Computes the actual omega based on reference omega and motor dynamics.
        via the zero-order hold method, instead of Euler integration.
        Zero-order hold method: omega[k+1] = omega[k] + beta * (omega_ref[k] - omega[k])
        where beta = 1 - exp(-dt/tau)
        Parameters:
        - omega_ref (Tensor): Tensor of reference omega values. shape (num_envs, num_motors)

        Returns:
        - omega (Tensor): Tensor of updated omega values. shape (num_envs, num_motors)
        """
        if not self.use:
            self.omega = omega_ref
            return self.omega

        # Distinguish between up and down phases and choose response instance accordingly
        tau = torch.where(omega_ref > self.omega, self.taus_up, self.taus_down)
        self.omega = self.response.step(omega_ref, tau)

        return self.omega

    def reset(self, env_ids):
        """
        Resets the motor model to initial conditions.
        """
        idle = torch.tensor(self.init, device=self.device, dtype=self.dtype).expand(
            len(env_ids), -1
        )
        self.omega[env_ids] = idle
        # reset the first-order lag state to idle speed (NOT 0 — response.reset() would zero it,
        # making the motor ramp up from 0 rad/s each episode and dip the thrust at reset).
        self.response.state[env_ids] = idle

    @property
    def tau_up(self):
        """Get the time constants for motor speed up."""
        return self._tau_up

    @property
    def tau_down(self):
        """Get the time constants for motor speed down."""
        return self._tau_down

    @tau_up.setter
    def tau_up(self, value):
        """Set the time constants for motor speed up."""
        self._tau_up = value

    @tau_down.setter
    def tau_down(self, value):
        """Set the time constants for motor speed down."""
        self._tau_down = value
