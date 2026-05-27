from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config.params import RobotParams
from src.model.terrain import Terrain


@dataclass(frozen=True)
class LegKinematics:
    leg_length: float
    hip_angle: float
    knee_angle: float
    feasible: bool


def wheel_center_y(x: float, terrain: Terrain, params: RobotParams) -> float:
    return terrain.height(x) + params.wheel_radius


def inverse_vertical_leg(
    hip_y: float, wheel_y: float, params: RobotParams
) -> LegKinematics:
    length = float(hip_y - wheel_y)
    l1 = params.thigh_length
    l2 = params.shank_length
    feasible = params.wheel_radius <= length <= l1 + l2
    clipped = float(np.clip(length, 1e-6, l1 + l2 - 1e-6))
    cos_knee_internal = (l1 * l1 + l2 * l2 - clipped * clipped) / (2.0 * l1 * l2)
    cos_knee_internal = float(np.clip(cos_knee_internal, -1.0, 1.0))
    knee_internal = float(np.arccos(cos_knee_internal))
    knee_flexion = float(np.pi - knee_internal)
    hip_angle = 0.5 * knee_flexion
    return LegKinematics(
        leg_length=length,
        hip_angle=hip_angle,
        knee_angle=knee_flexion,
        feasible=feasible,
    )


def hip_position(state: np.ndarray) -> np.ndarray:
    return np.array([state[0], state[1]], dtype=float)


def state_to_observation(
    state: np.ndarray,
    v_cmd: float,
    y_ref: float,
    terrain: Terrain,
    params: RobotParams,
) -> np.ndarray:
    x, y, theta, vx, vy, omega = state
    wheel_y = wheel_center_y(x, terrain, params)
    leg = inverse_vertical_leg(y, wheel_y, params)
    return np.array(
        [
            y - y_ref,
            theta,
            vx - v_cmd,
            vy,
            omega,
            leg.leg_length,
            terrain.height(x),
            terrain.slope(x),
            v_cmd,
        ],
        dtype=np.float32,
    )

