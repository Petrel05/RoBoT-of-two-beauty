from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from src.config.params import RobotParams
from src.controllers.base import ControlContext, Controller, clip_tau, equilibrium_torque
from src.model.dynamics import state_derivative
from src.model.terrain import FlatTerrain


class LQRController(Controller):
    name = "lqr"

    def __init__(self, params: RobotParams, y_ref: float, eps: float = 1e-5):
        self.params = params
        self.y_ref = y_ref
        self.eps = eps
        self.terrain = FlatTerrain()
        self._cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def _f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        dx, _ = state_derivative(x, u, self.terrain, self.params, 0.0)
        return dx

    def _linearize(self, v_cmd: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(float(v_cmd), 2)
        if key in self._cache:
            return self._cache[key]

        x0 = np.array([0.0, self.y_ref, 0.0, v_cmd, 0.0, 0.0], dtype=float)
        u0 = equilibrium_torque(v_cmd, self.params)
        n = x0.size
        m = u0.size
        A = np.zeros((n, n), dtype=float)
        B = np.zeros((n, m), dtype=float)

        for i in range(n):
            dx = np.zeros(n)
            dx[i] = self.eps
            A[:, i] = (self._f(x0 + dx, u0) - self._f(x0 - dx, u0)) / (2.0 * self.eps)
        for j in range(m):
            du = np.zeros(m)
            du[j] = self.eps
            B[:, j] = (self._f(x0, u0 + du) - self._f(x0, u0 - du)) / (2.0 * self.eps)

        Q = np.diag([0.0, 600.0, 1800.0, 30.0, 120.0, 280.0])
        R = np.diag([0.03, 0.06, 0.05])
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)
        self._cache[key] = (K, u0)
        return K, u0

    def compute(self, state: np.ndarray, context: ControlContext) -> np.ndarray:
        K, u0 = self._linearize(context.v_cmd)
        x_ref = np.array(
            [state[0], context.y_ref, 0.0, context.v_cmd, 0.0, 0.0], dtype=float
        )
        tau = u0 - K @ (state - x_ref)
        return clip_tau(tau, self.params)

