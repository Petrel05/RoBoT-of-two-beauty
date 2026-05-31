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


@dataclass(frozen=True)
class WBCAffineDynamics:
    """Affine reduced-order dynamics used by the constrained WBC QP.

    The QP decision vector is:
    [ax, ay, alpha, contact_tangent, contact_normal,
     tau_wheel, tau_knee, tau_hip].
    """

    equality_matrix: np.ndarray
    equality_rhs: np.ndarray
    wheel_y: float
    leg_length: float
    leg_rate: float
    passive_leg_force: float
    contact_moment_arm: float
    terrain_slope: float
    terrain_curvature: float


def initial_state(y_ref: float = 0.82) -> np.ndarray:
    return np.array([0.0, y_ref, 0.0, 0.0, 0.0, 0.0], dtype=float)


def clip_control(u: np.ndarray, params: RobotParams) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=float)
    clipped = np.clip(u, -params.tau_limits, params.tau_limits)
    saturated = np.isclose(clipped, params.tau_limits, rtol=0.0, atol=1e-9) | np.isclose(
        clipped, -params.tau_limits, rtol=0.0, atol=1e-9
    )
    return clipped, saturated


def wbc_affine_dynamics(
    state: np.ndarray,
    terrain: Terrain,
    params: RobotParams,
    external_force_x: float = 0.0,
) -> WBCAffineDynamics:
    """Build the exact affine equations for the reduced-order WBC QP.

    Contact kinematics are eliminated analytically by defining the wheel-center
    height as terrain.height(x) + wheel_radius. The remaining dynamics and
    actuator/contact maps are imposed as hard equalities in the QP.
    """

    x, y, theta, vx, vy, omega = np.asarray(state, dtype=float)
    wheel_y = wheel_center_y(x, terrain, params)
    terrain_slope = float(terrain.slope(x))
    leg_length = float(y - wheel_y)
    leg_rate = float(vy - terrain_slope * vx)
    passive_leg_force = float(
        params.leg_stiffness * (params.nominal_leg_length - leg_length)
        - params.leg_damping * leg_rate
    )
    contact_moment_arm = params.contact_pitch_coupling * max(
        float(y - terrain.height(x)),
        0.1,
    )

    slope_drag = params.mass * params.gravity * terrain_slope

    # Decision vector:
    # [ax, ay, alpha, contact_tangent, contact_normal,
    #  tau_wheel, tau_knee, tau_hip]
    equality_matrix = np.zeros((5, 8), dtype=float)
    equality_rhs = np.zeros(5, dtype=float)

    # Centroidal translation along x and y.
    equality_matrix[0, [0, 3]] = [params.mass, -1.0]
    equality_rhs[0] = external_force_x - params.x_damping * vx - slope_drag

    equality_matrix[1, [1, 4]] = [params.mass, -1.0]
    equality_rhs[1] = (
        -params.mass * params.gravity
        - params.y_damping * vy
    )

    # Body pitch dynamics.
    equality_matrix[2, [2, 3, 6, 7]] = [
        params.body_inertia,
        -contact_moment_arm,
        -0.15,
        -1.0,
    ]
    equality_rhs[2] = (
        -params.theta_stiffness * theta
        - params.theta_damping * omega
    )

    # Rolling drive and leg-extension actuator maps.
    equality_matrix[3, [3, 5]] = [-params.wheel_radius, 1.0]
    equality_matrix[4, [4, 6, 7]] = [
        -params.leg_moment_arm,
        1.0,
        0.6,
    ]
    equality_rhs[4] = -params.leg_moment_arm * passive_leg_force

    return WBCAffineDynamics(
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        wheel_y=float(wheel_y),
        leg_length=leg_length,
        leg_rate=leg_rate,
        passive_leg_force=passive_leg_force,
        contact_moment_arm=contact_moment_arm,
        terrain_slope=terrain_slope,
        terrain_curvature=float(terrain.curvature(x)),
    )


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
    leg_rate = float(vy - terrain.slope(x) * vx)

    f_leg_active = (tau_knee + 0.6 * tau_hip) / params.leg_moment_arm
    f_leg_passive = (
        params.leg_stiffness * (params.nominal_leg_length - leg_length)
        - params.leg_damping * leg_rate
    )
    f_leg = f_leg_active + f_leg_passive
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

    ay = (
        f_leg
        - params.mass * params.gravity
        - params.y_damping * vy
    ) / params.mass

    contact_moment_arm = params.contact_pitch_coupling * max(
        y - terrain.height(x),
        0.1,
    )
    alpha = (
        tau_hip
        + 0.15 * tau_knee
        + contact_moment_arm * contact_tangent
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
    _, info = state_derivative(next_state, control, terrain, params, external_force_x)
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

