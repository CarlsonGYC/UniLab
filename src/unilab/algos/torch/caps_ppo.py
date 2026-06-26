from __future__ import annotations

import math
from typing import Any

import torch

from unilab.algos.torch.rsl_rl_ppo import FinalObservationAwarePPO


class CapsPPO(FinalObservationAwarePPO):
    """PPO + CAPS spatial action-smoothness regularizer (Mysore, Mabsout, Mancuso & Saenko, 2021).

    Adds to the policy loss::

        L_S = lambda_s(progress) * || mu(s) - mu(s + eps) ||^2,   eps ~ N(0, caps_sigma * I)

    where the noise is applied in the **normalized** observation space (after
    ``actor.obs_normalizer``). It forces the mean action to be ~invariant to small obs
    perturbations, directly suppressing the high-frequency "buzzing" a policy shows once it
    settles at a setpoint. Complements the env's 1st/2nd-order action-smoothness reward terms
    (``smooth`` / ``smooth2``) with a learning-level regularizer.

    ``lambda_s`` is **ramped** 0 -> ``caps_lambda_s`` linearly over
    ``[caps_warmup_frac, caps_full_frac]`` of training progress (= update count /
    ``caps_total_iters``). With ``caps_warmup_frac`` aligned to the point the env curriculum
    reaches full difficulty, smoothing only kicks in once the task is mostly learned — early
    exploration is not suppressed (direct_rl's CAPS schedule).

    Optionally caps the policy action std at ``max_action_std`` (projected after each update),
    mirroring direct_rl's ``clip_log_std`` / ``max_log_std`` — large std past the action clamp
    only buys saturation + setpoint jitter, not useful exploration.

    Enable via the PPO owner YAML::

        algo.algorithm.class_name:      unilab.algos.torch.caps_ppo:CapsPPO
        algo.algorithm.caps_lambda_s:   0.5
        algo.algorithm.caps_sigma:      0.05
        algo.algorithm.caps_warmup_frac: 0.40
        algo.algorithm.caps_full_frac:   0.80
        algo.algorithm.caps_total_iters: ${algo.max_iterations}
        algo.algorithm.max_action_std:   0.6
    """

    def __init__(
        self,
        *args: Any,
        caps_lambda_s: float = 0.0,
        caps_sigma: float = 0.05,
        caps_warmup_frac: float = 0.0,
        caps_full_frac: float = 0.0,
        caps_total_iters: int = 0,
        max_action_std: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._caps_lambda_s_full = float(caps_lambda_s)
        self._caps_sigma = float(caps_sigma)
        self._caps_warmup_frac = float(caps_warmup_frac)
        self._caps_full_frac = float(caps_full_frac)
        self._caps_total_iters = int(caps_total_iters)
        self._max_action_std = float(max_action_std)
        self._update_count = 0
        # current (ramped) weight used inside the minibatch loss; starts at full when no schedule.
        scheduled = self._caps_total_iters > 0 and self._caps_full_frac > self._caps_warmup_frac
        self._caps_lambda_s = 0.0 if scheduled else self._caps_lambda_s_full
        # CAPS lives in the per-minibatch tensor loss, which the parent only uses on its compiled
        # fast path. Force the (uncompiled) tensor path so the regularizer is actually applied.
        self._force_tensor_path = self._caps_lambda_s_full > 0.0

    @staticmethod
    def _ramp(p: float, start: float, full: float) -> float:
        if full <= start:
            return 1.0 if p >= full else 0.0
        return min(1.0, max(0.0, (p - start) / (full - start)))

    def update(self) -> Any:
        # advance the CAPS curriculum before collecting the minibatch loss
        if self._caps_lambda_s_full > 0.0 and self._caps_total_iters > 0:
            progress = self._update_count / float(self._caps_total_iters)
            self._caps_lambda_s = self._caps_lambda_s_full * self._ramp(
                progress, self._caps_warmup_frac, self._caps_full_frac
            )
        self._update_count += 1
        out = super().update()
        self._clamp_action_std()
        return out

    def _clamp_action_std(self) -> None:
        if self._max_action_std <= 0.0:
            return
        dist = getattr(self.actor, "distribution", None)
        if dist is None:
            return
        std_type = getattr(dist, "std_type", None)
        with torch.no_grad():
            if std_type == "scalar" and hasattr(dist, "std_param"):
                dist.std_param.clamp_(min=1.0e-4, max=self._max_action_std)
            elif std_type == "log" and hasattr(dist, "log_std_param"):
                dist.log_std_param.clamp_(max=math.log(self._max_action_std))

    def _supports_compiled_update_path(self) -> bool:
        if not getattr(self, "_force_tensor_path", False):
            return super()._supports_compiled_update_path()
        # same structural requirements as the parent, but NOT gated on enable_compile
        if self.rnd or self.symmetry or self.is_multi_gpu:
            return False
        if self.actor.is_recurrent or self.critic.is_recurrent:
            return False
        distribution: Any = getattr(self.actor, "distribution", None)
        if distribution is None or not hasattr(distribution, "std_type"):
            return False
        if distribution.std_type == "scalar":
            return hasattr(distribution, "std_param")
        if distribution.std_type == "log":
            return hasattr(distribution, "log_std_param")
        return False

    def _minibatch_loss_tensors(
        self,
        actor_obs,
        critic_obs,
        actions,
        target_values,
        advantages,
        old_actions_log_prob,
        old_values,
        old_mu,
        old_sigma,
    ):
        loss, surrogate_loss, value_loss, entropy, kl_mean = super()._minibatch_loss_tensors(
            actor_obs,
            critic_obs,
            actions,
            target_values,
            advantages,
            old_actions_log_prob,
            old_values,
            old_mu,
            old_sigma,
        )
        if self._caps_lambda_s > 0.0:
            normalized = self.actor.obs_normalizer(actor_obs)
            mu = self.actor.mlp(normalized)
            mu_noisy = self.actor.mlp(normalized + self._caps_sigma * torch.randn_like(normalized))
            loss = loss + self._caps_lambda_s * (mu - mu_noisy).pow(2).mean()
        return loss, surrogate_loss, value_loss, entropy, kl_mean
