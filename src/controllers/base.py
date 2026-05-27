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


def equilibrium_torque(v_cmd: float, params: RobotParams) -> np.ndarray:
    tau_wheel = params.x_damping * v_cmd * params.wheel_radius
    knee = params.mass * params.gravity * params.leg_moment_arm / 0.91
    hip = -0.15 * knee
    return np.array([tau_wheel, knee, hip], dtype=float)


def clip_tau(tau: np.ndarray, params: RobotParams) -> np.ndarray:
    return np.clip(np.asarray(tau, dtype=float), -params.tau_limits, params.tau_limits)

