from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.config.params import RobotParams
from src.controllers.base import ControlContext, Controller, clip_tau, equilibrium_torque
from src.model.dynamics import accelerations
from src.model.terrain import Terrain


class WBCQPController(Controller):
    name = "wbc_qp"

    def __init__(self, params: RobotParams, terrain: Terrain):
        self.params = params
        self.terrain = terrain
        self._last_tau: np.ndarray | None = None

    def reset(self) -> None:
        self._last_tau = None

    def _desired_accel(self, state: np.ndarray, context: ControlContext) -> np.ndarray:
        x, y, theta, vx, vy, omega = state
        terrain_ff = self.terrain.curvature(x) * vx * vx
        ax_des = 4.0 * (context.v_cmd - vx)
        ay_des = 85.0 * (context.y_ref - y) - 18.0 * vy + 0.35 * terrain_ff
        alpha_des = 95.0 * (0.0 - theta) - 18.0 * omega
        return np.array([ax_des, ay_des, alpha_des], dtype=float)

    def compute(self, state: np.ndarray, context: ControlContext) -> np.ndarray:
        tau0 = (
            self._last_tau.copy()
            if self._last_tau is not None
            else equilibrium_torque(context.v_cmd, self.params)
        )
        desired = self._desired_accel(state, context)
        weights = np.array([1.0, 8.0, 12.0], dtype=float)
        tau_scale = np.maximum(self.params.tau_limits, 1.0)

        def objective(tau: np.ndarray) -> float:
            acc, info = accelerations(
                state, tau, self.terrain, self.params, context.external_force_x
            )
            task_error = weights * (acc - desired)
            energy = 0.015 * np.sum((tau / tau_scale) ** 2)
            smooth = 0.01 * np.sum(((tau - tau0) / tau_scale) ** 2)
            friction_margin = max(abs(info.contact_tangent) - self.params.friction_coeff * max(info.contact_normal, 1.0), 0.0)
            return float(np.sum(task_error * task_error) + energy + smooth + 10.0 * friction_margin)

        bounds = [(-lim, lim) for lim in self.params.tau_limits]
        result = minimize(
            objective,
            x0=clip_tau(tau0, self.params),
            bounds=bounds,
            method="SLSQP",
            options={"maxiter": 40, "ftol": 1e-5, "disp": False},
        )
        tau = result.x if result.success else tau0
        tau = clip_tau(tau, self.params)
        self._last_tau = tau
        return tau

