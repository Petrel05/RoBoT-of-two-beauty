from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config.params import RobotParams, SafetyLimits
from src.model.kinematics import inverse_vertical_leg, wheel_center_y
from src.model.terrain import Terrain


@dataclass(frozen=True)
class DynamicsInfo:
    ax: float
    ay: float
    alpha: float
    contact_normal: float
    contact_tangent: float
    leg_length: float
    leg_feasible: bool
    saturated: np.ndarray


def initial_state(y_ref: float = 0.82) -> np.ndarray:
    return np.array([0.0, y_ref, 0.0, 0.0, 0.0, 0.0], dtype=float)


def clip_control(u: np.ndarray, params: RobotParams) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=float)
    clipped = np.clip(u, -params.tau_limits, params.tau_limits)
    saturated = np.isclose(clipped, params.tau_limits, rtol=0.0, atol=1e-9) | np.isclose(
        clipped, -params.tau_limits, rtol=0.0, atol=1e-9
    )
    return clipped, saturated


def accelerations(
    state: np.ndarray,
    control: np.ndarray,
    terrain: Terrain,
    params: RobotParams,
    external_force_x: float = 0.0,
) -> tuple[np.ndarray, DynamicsInfo]:
    x, y, theta, vx, vy, omega = np.asarray(state, dtype=float)
    tau, saturated = clip_control(control, params)
    tau_wheel, tau_knee, tau_hip = tau

    wheel_y = wheel_center_y(x, terrain, params)
    leg = inverse_vertical_leg(y, wheel_y, params)
    leg_length = max(leg.leg_length, 1e-3)

    f_leg = (tau_knee + 0.6 * tau_hip) / params.leg_moment_arm
    f_drive_cmd = tau_wheel / params.wheel_radius
    contact_normal = max(f_leg, 0.0)
    friction_limit = params.friction_coeff * max(contact_normal, 1.0)
    contact_tangent = float(np.clip(f_drive_cmd, -friction_limit, friction_limit))

    terrain_slope = terrain.slope(x)
    slope_drag = params.mass * params.gravity * terrain_slope
    ax = (
        contact_tangent
        + external_force_x
        - params.x_damping * vx
        - slope_drag
    ) / params.mass

    terrain_feed = terrain.curvature(x) * vx * vx
    ay = (
        f_leg
        - params.mass * params.gravity
        - params.y_damping * vy
    ) / params.mass
    ay -= 0.25 * terrain_feed

    com_height = max(y - wheel_y, 0.1)
    external_moment = 0.25 * external_force_x * com_height
    alpha = (
        tau_hip
        + 0.15 * tau_knee
        + external_moment
        - params.theta_stiffness * theta
        - params.theta_damping * omega
    ) / params.body_inertia

    qdd = np.array([ax, ay, alpha], dtype=float)
    info = DynamicsInfo(
        ax=float(ax),
        ay=float(ay),
        alpha=float(alpha),
        contact_normal=float(contact_normal),
        contact_tangent=float(contact_tangent),
        leg_length=float(leg_length),
        leg_feasible=bool(leg.feasible),
        saturated=saturated,
    )
    return qdd, info


def state_derivative(
    state: np.ndarray,
    control: np.ndarray,
    terrain: Terrain,
    params: RobotParams,
    external_force_x: float = 0.0,
) -> tuple[np.ndarray, DynamicsInfo]:
    ax_ay_alpha, info = accelerations(
        state, control, terrain, params, external_force_x
    )
    x, y, theta, vx, vy, omega = np.asarray(state, dtype=float)
    deriv = np.array([vx, vy, omega, *ax_ay_alpha], dtype=float)
    return deriv, info


def rk4_step(
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
    terrain: Terrain,
    params: RobotParams,
    external_force_x: float = 0.0,
) -> tuple[np.ndarray, DynamicsInfo]:
    k1, info = state_derivative(state, control, terrain, params, external_force_x)
    k2, _ = state_derivative(state + 0.5 * dt * k1, control, terrain, params, external_force_x)
    k3, _ = state_derivative(state + 0.5 * dt * k2, control, terrain, params, external_force_x)
    k4, _ = state_derivative(state + dt * k3, control, terrain, params, external_force_x)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_state, info


def is_failure(
    state: np.ndarray,
    info: DynamicsInfo,
    y_ref: float,
    params: RobotParams,
    safety: SafetyLimits,
) -> tuple[bool, str]:
    _x, y, theta, _vx, _vy, _omega = state
    if abs(theta) > safety.theta_fail:
        return True, "theta_fail"
    if abs(y - y_ref) > safety.height_fail:
        return True, "height_fail"
    if not info.leg_feasible:
        return True, "leg_infeasible"
    if info.leg_length < safety.min_leg_length:
        return True, "leg_collapsed"
    return False, ""

