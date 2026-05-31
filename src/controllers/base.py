from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config.params import RobotParams


@dataclass(frozen=True)
class ControlContext:
    t: float
    v_cmd: float
    y_ref: float
    external_force_x: float


class Controller:
    name = "base"

    def reset(self) -> None:
        pass

    def compute(self, state: np.ndarray, context: ControlContext) -> np.ndarray:
        raise NotImplementedError

    def diagnostics(self) -> dict[str, float]:
        return {}


def equilibrium_torque(v_cmd: float, params: RobotParams) -> np.ndarray:
    tau_wheel = params.x_damping * v_cmd * params.wheel_radius
    contact_tangent = tau_wheel / params.wheel_radius
    contact_moment_arm = params.contact_pitch_coupling * (
        params.nominal_leg_length + params.wheel_radius
    )
    knee, hip = distribute_leg_torque(
        active_leg_force=params.mass * params.gravity,
        joint_pitch_moment=-contact_moment_arm * contact_tangent,
        params=params,
    )
    return np.array([tau_wheel, knee, hip], dtype=float)


def clip_tau(tau: np.ndarray, params: RobotParams) -> np.ndarray:
    return np.clip(np.asarray(tau, dtype=float), -params.tau_limits, params.tau_limits)


def residual_torque_action(
    action: np.ndarray,
    state: np.ndarray,
    v_cmd: float,
    y_ref: float,
    params: RobotParams,
    external_force_x: float = 0.0,
) -> np.ndarray:
    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    base = stabilizing_baseline_torque(state, v_cmd, y_ref, params, external_force_x)
    tau = base + action * params.rl_residual_scale
    return clip_tau(tau, params)


def direct_torque_action(action: np.ndarray, params: RobotParams) -> np.ndarray:
    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    tau_wheel = action[0] * params.tau_limits[0]
    # The knee actuator is modeled as the main leg-extension actuator, so the
    # direct policy controls its full usable range [0, tau_max] instead of a
    # symmetric pull/push range. It still outputs the complete torque command.
    tau_knee = 0.5 * (action[1] + 1.0) * params.tau_limits[1]
    tau_hip = action[2] * params.tau_limits[2]
    return clip_tau(np.array([tau_wheel, tau_knee, tau_hip], dtype=float), params)


def distribute_leg_torque(
    active_leg_force: float,
    joint_pitch_moment: float,
    params: RobotParams,
) -> tuple[float, float]:
    matrix = np.array([[1.0, 0.6], [0.15, 1.0]], dtype=float)
    rhs = np.array(
        [params.leg_moment_arm * active_leg_force, joint_pitch_moment],
        dtype=float,
    )
    knee, hip = np.linalg.solve(matrix, rhs)
    return float(knee), float(hip)


def stabilizing_baseline_torque(
    state: np.ndarray,
    v_cmd: float,
    y_ref: float,
    params: RobotParams,
    external_force_x: float = 0.0,
) -> np.ndarray:
    _x, y, theta, vx, vy, omega = np.asarray(state, dtype=float)

    ax_cmd = 3.0 * (v_cmd - vx) - external_force_x / max(params.mass, 1.0)
    tau_wheel = params.wheel_radius * (params.x_damping * v_cmd + params.mass * ax_cmd)

    ay_cmd = 70.0 * (y_ref - y) - 16.0 * vy
    f_leg = params.mass * (params.gravity + ay_cmd)
    f_leg = max(f_leg, 0.0)

    alpha_cmd = 85.0 * (0.0 - theta) - 18.0 * omega
    contact_tangent = tau_wheel / params.wheel_radius
    contact_moment_arm = params.contact_pitch_coupling * max(y, 0.1)
    joint_pitch_moment = (
        params.body_inertia * alpha_cmd
        + params.theta_stiffness * theta
        + params.theta_damping * omega
        - contact_moment_arm * contact_tangent
    )
    knee, hip = distribute_leg_torque(f_leg, joint_pitch_moment, params)
    return clip_tau(np.array([tau_wheel, knee, hip], dtype=float), params)
