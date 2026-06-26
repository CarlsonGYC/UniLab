# Copyright (c) 2025, Tianchen Sun
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch

from ..filter.low_pass_filter import ButterworthFilter


class BodyRateController:
    def __init__(
        self, num_envs, dt, kp, ki, kd, kk, k_ff, cutoff_hz, device="cpu", dtype=torch.float32
    ):
        """ Body rate PID controller for quadcopters.
        A batch parallel implementation of the PX4 body rate controller.

        Args:
        - num_envs (int): Number of environments.
        - dt (float): Time step duration, should be 1 KHz (0.001 seconds) for the PX4 controller.
        - kp (torch.Tensor): Proportional gain for the controller.
        - ki (torch.Tensor): Integral gain for the controller.
        - kd (torch.Tensor): Derivative gain for the controller.
        - k_ff (torch.Tensor): Feedforward gain for the controller.
        - cutoff_hz list(float): Butterworth Low-pass filter cutoff frequency [Hz] \
                                for the rate and rate derivate term.
        """

        self.num_envs = num_envs
        self.dt = dt
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kk = kk
        self.k_ff = k_ff

        self._integral = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._prev_rate = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._prev_error = torch.zeros(num_envs, 3, device=device, dtype=dtype)

        # body rate butterworth filter
        self.rate_filter = ButterworthFilter(
            num_envs=num_envs,
            dt=dt,
            cutoff_hz=cutoff_hz[0],  # Typical cutoff frequency for body rate control
            device=device,
            dtype=dtype,
        )

        # body rate derivate butterworth filter
        self.rate_derivate_filter = ButterworthFilter(
            num_envs=num_envs,
            dt=dt,
            cutoff_hz=cutoff_hz[1],  # Typical cutoff frequency for body rate control
            device=device,
            dtype=dtype,
        )

        """Initialize the integral and previous error terms."""

        # Ensure gains are tensors
        if not isinstance(self.kp, torch.Tensor):
            self.kp = torch.tensor(self.kp, device=device, dtype=dtype)
        if not isinstance(self.ki, torch.Tensor):
            self.ki = torch.tensor(self.ki, device=device, dtype=dtype)
        if not isinstance(self.kd, torch.Tensor):
            self.kd = torch.tensor(self.kd, device=device, dtype=dtype)

        # Ensure gains are broadcastable to the number of environments
        if self.kp.dim() == 0:
            self.kp = self.kp.expand(num_envs, 3)
        if self.ki.dim() == 0:
            self.ki = self.ki.expand(num_envs, 3)
        if self.kd.dim() == 0:
            self.kd = self.kd.expand(num_envs, 3)

    def compute(self, rate_ref, rate, debug=False):
        """Compute the control action based on the reference and current angular velocities.

        Args:
        - rate_ref (torch.Tensor): Reference body rate (rad/s) (num_envs, 3).
        - rate (torch.Tensor): Current body rate (rad/s) (num_envs, 3).
        - debug (bool): If True, return additional debug information.

        Returns:
        - torque (torch.Tensor): torque to be applied (num_envs, 3).

        Remarks:
        - No anti-windup is applied to the control action.
        - The control output, named `torque`, follows the PX4 body rate controller convention.
            However, since the typical MC_ROLLRATE_P = 0.05, it means the so called `torque`
            is actually just a quanity within [-1,1] for the follwing mixer.

        - normalized wrench -> [mixer (unit scale)] -> normalized PWM
          normalized PWM -> [thrust curve] -> normalized ACTUAL thrust -> normalized ACTUAL wrench
            If the [thrust curve] is approx linear (Y=X),
            Then
                normalized wrench = normalized ACTUAL wrench
        """
        # 1. Compute the rate derivative
        d_rate = (rate - self._prev_rate) / self.dt

        # 2. Apply low-pass filter to the rate and the rate derivative
        rate_filtered = self.rate_filter(rate)
        d_rate_filtered = self.rate_derivate_filter(d_rate)

        # 3. Compute the error, error integral, and rate derivate term based on
        #    the reference and filter body rate
        error = rate_ref - rate_filtered
        self._integral += error * self.dt

        # 4. PID with the feedforward term
        torque = self.kk * (
            self.kp * error
            + self.ki * self._integral
            - self.kd * d_rate_filtered
            + self.k_ff * rate_ref
        )

        # 5. Update with original (unfiltered) rate
        self._prev_rate = rate.clone()

        if debug:
            debug_info = {
                "error": error.clone(),
                "integral": self._integral.clone(),
                "derivative": d_rate.clone(),
                "torque": torque.clone(),
            }
            return torque, debug_info
        else:
            return torque

    def reset(self, env_ids):
        """Reset the controller state."""
        self._integral[env_ids] = 0.0
        self._prev_error[env_ids] = 0.0
        self._prev_rate[env_ids] = 0.0

        # reset the low-pass filters
        self.rate_filter.reset(env_ids)
        self.rate_derivate_filter.reset(env_ids)
