#!/usr/bin/env python
"""CPU-only unit test of the PX4 position/attitude port — no simulator required.

Identical to direct_rl/control_test/test_px4_control.py; only the module path points
at the copied ``dynamics/px4_control.py`` (which depends only on ``torch`` / ``math``).

    uv run python control_test/test_px4_control.py     # exits non-zero on any failure
"""

import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.join(_HERE, "dynamics", "px4_control.py")
spec = importlib.util.spec_from_file_location("px4_control", MOD)
px4 = importlib.util.module_from_spec(spec)
sys.modules["px4_control"] = px4  # so dataclass can resolve nested annotations
spec.loader.exec_module(px4)

torch.set_printoptions(precision=4, sci_mode=False)
fails = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)


# 1. frame adapters are involutions
v = torch.randn(5, 3)
check("enu_ned involution", torch.allclose(px4.enu_ned(px4.enu_ned(v)), v, atol=1e-6))
check("flu_frd involution", torch.allclose(px4.flu_frd(px4.flu_frd(v)), v, atol=1e-6))

# 2. quaternion utils
qid = torch.tensor([[1.0, 0, 0, 0]])
check("quat_mul identity", torch.allclose(px4.quat_mul(qid, qid), qid))
check(
    "quat_dcm_z identity",
    torch.allclose(px4.quat_dcm_z(qid), torch.tensor([[0.0, 0, 1]]), atol=1e-6),
)
q2 = px4.quat_from_two_vectors(torch.tensor([[0.0, 0, 1]]), torch.tensor([[0.0, 0, 1]]))
check(
    "two_vec same dir ~ identity",
    torch.allclose(px4.quat_dcm_z(q2), torch.tensor([[0.0, 0, 1]]), atol=1e-6),
)

# 3. ENU/FLU identity attitude → body-z is NED down
q_nd = px4.quat_enu_flu_to_ned_frd(qid)
bz = px4.quat_dcm_z(q_nd)
check(
    "identity ENU → body-z NED down",
    torch.allclose(bz, torch.tensor([[0.0, 0, 1]]), atol=1e-5),
    f"body_z_ned={bz.tolist()}",
)


def desired_body_up_enu(ctrl):
    """Desired attitude's body-UP axis (FLU +z), expressed in ENU."""
    return px4.enu_ned(-px4.quat_dcm_z(ctrl._q_d_ned))


N = 1
g = px4.CONSTANTS_ONE_G
support_mass = 1.59
hover = px4.hover_thrust_for_thrust(support_mass * g)
cfg = px4.PX4ControlCfg()
cfg.position.hover_thrust = hover
ctrl = px4.PX4PositionAttitudeController(N, dt=0.01, cfg=cfg)

# 4. hover: level, on target, zero velocity
ctrl.reset()
p = torch.tensor([[1.0, 2.0, 1.5]])
for _ in range(3):
    thr, rate = ctrl.compute(p, torch.zeros(N, 3), qid, p, torch.zeros(N, 3), torch.zeros(N, 3))
check(
    "hover thrust ≈ hover_thrust",
    abs(float(thr[0, 0]) - hover) < 0.02,
    f"thrust={float(thr[0, 0]):.3f} hover={hover:.3f}",
)
check("hover body rate ≈ 0", float(rate.abs().max()) < 1e-3)
up = desired_body_up_enu(ctrl)
check("hover desired body-up ≈ +Z(ENU)", torch.allclose(up, torch.tensor([[0.0, 0, 1]]), atol=1e-3))

# 5. climb (+z ENU target) → more thrust
ctrl.reset()
for _ in range(3):
    thr_up, _ = ctrl.compute(
        torch.tensor([[0.0, 0, 1.0]]),
        torch.zeros(N, 3),
        qid,
        torch.tensor([[0.0, 0, 2.0]]),
        torch.zeros(N, 3),
        torch.zeros(N, 3),
    )
check(
    "climb command increases thrust",
    float(thr_up[0, 0]) > hover + 0.02,
    f"thrust={float(thr_up[0, 0]):.3f}",
)

# 6/7. lateral targets → desired body-up tilts toward target
p0 = torch.tensor([[0.0, 0.0, 1.5]])
for axis, sp in (("x", torch.tensor([[1.0, 0, 1.5]])), ("y", torch.tensor([[0.0, 1.0, 1.5]]))):
    ctrl.reset()
    for _ in range(3):
        _, rate_f = ctrl.compute(
            p0, torch.zeros(N, 3), qid, sp, torch.zeros(N, 3), torch.zeros(N, 3)
        )
    u = desired_body_up_enu(ctrl)[0]
    idx = 0 if axis == "x" else 1
    check(f"+{axis} target → body-up leans +{axis}", float(u[idx]) > 0.05, f"up={u.tolist()}")
    check(f"+{axis} target → body-up stays mostly +z", float(u[2]) > 0.7)
check("lateral target → finite nonzero rate cmd", 1e-4 < float(rate_f.abs().max()) < 10.0)

# 8. batch of 1000 envs: shapes + finite
M = 1000
cb = px4.PX4PositionAttitudeController(M, dt=0.01, cfg=cfg)
pp = torch.randn(M, 3) * 0.1 + torch.tensor([0.0, 0, 1.5])
qq = qid.repeat(M, 1)
t8, r8 = cb.compute(
    pp,
    torch.zeros(M, 3),
    qq,
    torch.randn(M, 3) * 0.2 + torch.tensor([0.0, 0, 1.5]),
    torch.zeros(M, 3),
    torch.randn(M, 3) * 0.5,
)
check("batch shapes", tuple(t8.shape) == (M, 1) and tuple(r8.shape) == (M, 3))
check("batch all finite", torch.isfinite(t8).all().item() and torch.isfinite(r8).all().item())

# 9. acceleration feed-forward enters the thrust
ctrl.reset()
for _ in range(3):
    thr_a0, _ = ctrl.compute(p0, torch.zeros(N, 3), qid, p0, torch.zeros(N, 3), torch.zeros(N, 3))
thr_a0 = thr_a0.clone()  # returned tensor aliases an internal buffer reset() zeroes in place
ctrl.reset()
for _ in range(3):
    thr_az, _ = ctrl.compute(
        p0, torch.zeros(N, 3), qid, p0, torch.zeros(N, 3), torch.tensor([[0.0, 0, 3.0]])
    )
check(
    "+z accel FF raises thrust",
    float(thr_az[0, 0]) > float(thr_a0[0, 0]) + 0.01,
    f"thr_az={float(thr_az[0, 0]):.3f} thr_a0={float(thr_a0[0, 0]):.3f}",
)

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
