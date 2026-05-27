from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RobotParams:
    mass: float = 50.0
    body_inertia: float = 5.0
    thigh_length: float = 0.5
    shank_length: float = 0.5
    wheel_radius: float = 0.1
    gravity: float = 9.81
    friction_coeff: float = 0.8
    x_damping: float = 10.0
    y_damping: float = 35.0
    theta_damping: float = 2.8
    theta_stiffness: float = 18.0
    leg_moment_arm: float = 0.25
    tau_limits: np.ndarray = field(
        default_factory=lambda: np.array([30.0, 160.0, 120.0], dtype=float)
    )


@dataclass(frozen=True)
class SimParams:
    dt: float = 0.01
    duration: float = 8.0
    y_ref: float = 0.82
    max_steps: int = 800


@dataclass(frozen=True)
class SafetyLimits:
    theta_fail: float = np.deg2rad(30.0)
    height_fail: float = 0.15
    min_leg_length: float = 0.18
    saturation_fraction_limit: float = 0.35


ROBOT = RobotParams()
SIM = SimParams()
SAFETY = SafetyLimits()

