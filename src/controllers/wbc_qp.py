from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from src.config.params import RobotParams, SafetyLimits
from src.controllers.base import (
    ControlContext,
    Controller,
    clip_tau,
    stabilizing_baseline_torque,
)
from src.model.dynamics import WBCAffineDynamics, wbc_affine_dynamics
from src.model.terrain import Terrain


@dataclass(frozen=True)
class WBCQPWeights:
    task_accel: tuple[float, float, float] = (8.0, 180.0, 120.0)
    torque: float = 0.03
    torque_rate: float = 0.12
    contact_force: float = 0.002
    reachability_slack: float = 4.0e5
    height_slack: float = 6.0e5


@dataclass(frozen=True)
class QPSolveResult:
    x: np.ndarray
    success: bool
    iterations: int
    status: str
    status_val: int
    objective: float
    primal_residual: float
    dual_residual: float
    solve_time: float


class WBCQPController(Controller):
    """Constrained reduced-order whole-body controller.

    The QP explicitly solves for task acceleration, wheel-ground contact force,
    and actuator torque. Dynamics, rolling drive, leg-extension actuation,
    unilateral contact, friction cone, torque bounds, and predictive
    reachability bounds are represented directly in the optimization problem.
    """

    name = "wbc_qp"

    # Core decision variables.
    AX = 0
    AY = 1
    ALPHA = 2
    FT = 3
    FN = 4
    TAU_WHEEL = 5
    TAU_KNEE = 6
    TAU_HIP = 7

    # Soft-constraint slack variables.
    LEG_LOW_SLACK = 8
    LEG_HIGH_SLACK = 9
    HEIGHT_LOW_SLACK = 10
    HEIGHT_HIGH_SLACK = 11
    NVAR = 12
    OSQP_SOLVED_STATUS_VALUES = {1, 2}

    def __init__(
        self,
        params: RobotParams,
        terrain: Terrain,
        safety: SafetyLimits,
        dt: float = 0.01,
        prediction_horizon: float = 0.30,
        minimum_normal_force_fraction: float = 0.35,
        weights: WBCQPWeights = WBCQPWeights(),
    ):
        self.params = params
        self.terrain = terrain
        self.safety = safety
        self.dt = dt
        self.prediction_horizon = prediction_horizon
        self.minimum_normal_force_fraction = minimum_normal_force_fraction
        self.weights = weights
        self._last_tau: np.ndarray | None = None
        self._last_solution: np.ndarray | None = None
        self._speed_error_integral = 0.0
        self._diagnostics: dict[str, float] = {}

    def reset(self) -> None:
        self._last_tau = None
        self._last_solution = None
        self._speed_error_integral = 0.0
        self._diagnostics = {}

    def diagnostics(self) -> dict[str, float]:
        return dict(self._diagnostics)

    def _desired_accel(self, state: np.ndarray, context: ControlContext) -> np.ndarray:
        _x, y, theta, vx, vy, omega = state
        speed_error = context.v_cmd - vx
        self._speed_error_integral = float(
            np.clip(self._speed_error_integral + self.dt * speed_error, -2.0, 2.0)
        )
        ax_des = 5.0 * speed_error + 2.0 * self._speed_error_integral
        ay_des = 115.0 * (context.y_ref - y) - 24.0 * vy
        alpha_des = -125.0 * theta - 24.0 * omega
        return np.array([ax_des, ay_des, alpha_des], dtype=float)

    def _prediction_rows(
        self,
        state: np.ndarray,
        context: ControlContext,
        model: WBCAffineDynamics,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        horizon = self.prediction_horizon
        x, y, _theta, vx, vy, _omega = np.asarray(state, dtype=float)

        # Preview the terrain along the current velocity, then linearize the
        # effect of the optimized horizontal acceleration around that preview.
        preview_x = x + horizon * vx
        preview_wheel_y = (
            self.terrain.height(preview_x) + self.params.wheel_radius
        )
        preview_slope = self.terrain.slope(preview_x)

        # l_pred = y_pred - h(x_pred) - r
        leg_row = np.zeros(self.NVAR, dtype=float)
        leg_row[self.AX] = -0.5 * horizon * horizon * preview_slope
        leg_row[self.AY] = 0.5 * horizon * horizon
        leg_constant = y + horizon * vy - preview_wheel_y

        height_row = np.zeros(self.NVAR, dtype=float)
        height_row[self.AY] = 0.5 * horizon * horizon
        height_constant = y + horizon * vy

        return leg_row, np.array([leg_constant]), height_row, np.array([height_constant])

    def _constraints(
        self,
        state: np.ndarray,
        context: ControlContext,
        model: WBCAffineDynamics,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        equality_matrix = np.zeros((5, self.NVAR), dtype=float)
        equality_matrix[:, :8] = model.equality_matrix
        equality_rhs = model.equality_rhs.copy()

        leg_row, leg_constant, height_row, height_constant = self._prediction_rows(
            state, context, model
        )
        leg_constant_value = float(leg_constant[0])
        height_constant_value = float(height_constant[0])

        min_leg = self.safety.min_leg_length + 0.02
        max_leg = self.params.thigh_length + self.params.shank_length - 0.02
        min_height = context.y_ref - self.safety.height_fail + 0.01
        max_height = context.y_ref + self.safety.height_fail - 0.01

        rows: list[np.ndarray] = []
        rhs: list[float] = []

        # Linearized Coulomb friction cone: |Ft| <= mu Fn.
        row = np.zeros(self.NVAR, dtype=float)
        row[self.FT] = 1.0
        row[self.FN] = -self.params.friction_coeff
        rows.append(row)
        rhs.append(0.0)

        row = np.zeros(self.NVAR, dtype=float)
        row[self.FT] = -1.0
        row[self.FN] = -self.params.friction_coeff
        rows.append(row)
        rhs.append(0.0)

        # Predictive leg reachability with penalized slacks.
        row = -leg_row.copy()
        row[self.LEG_LOW_SLACK] = -1.0
        rows.append(row)
        rhs.append(leg_constant_value - min_leg)

        row = leg_row.copy()
        row[self.LEG_HIGH_SLACK] = -1.0
        rows.append(row)
        rhs.append(max_leg - leg_constant_value)

        # Predictive absolute-height safety corridor with penalized slacks.
        row = -height_row.copy()
        row[self.HEIGHT_LOW_SLACK] = -1.0
        rows.append(row)
        rhs.append(height_constant_value - min_height)

        row = height_row.copy()
        row[self.HEIGHT_HIGH_SLACK] = -1.0
        rows.append(row)
        rhs.append(max_height - height_constant_value)

        return equality_matrix, equality_rhs, np.vstack(rows), np.array(rhs, dtype=float)

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.NVAR, -np.inf, dtype=float)
        upper = np.full(self.NVAR, np.inf, dtype=float)

        lower[self.FN] = (
            self.minimum_normal_force_fraction
            * self.params.mass
            * self.params.gravity
        )
        lower[self.TAU_WHEEL : self.TAU_HIP + 1] = -self.params.tau_limits
        upper[self.TAU_WHEEL : self.TAU_HIP + 1] = self.params.tau_limits
        lower[self.LEG_LOW_SLACK :] = 0.0
        return lower, upper

    def _objective_matrices(
        self,
        desired_accel: np.ndarray,
        last_tau: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        hessian = np.zeros((self.NVAR, self.NVAR), dtype=float)
        gradient = np.zeros(self.NVAR, dtype=float)

        def add_weighted_target(index: int, weight: float, target: float, scale: float = 1.0):
            scaled_weight = weight / (scale * scale)
            hessian[index, index] += scaled_weight
            gradient[index] -= scaled_weight * target

        for index, weight, target in zip(
            [self.AX, self.AY, self.ALPHA],
            self.weights.task_accel,
            desired_accel,
        ):
            add_weighted_target(index, weight, float(target))

        for offset, limit in enumerate(self.params.tau_limits):
            index = self.TAU_WHEEL + offset
            add_weighted_target(index, self.weights.torque, 0.0, float(limit))
            add_weighted_target(
                index,
                self.weights.torque_rate,
                float(last_tau[offset]),
                float(limit),
            )

        add_weighted_target(
            self.FT,
            self.weights.contact_force,
            0.0,
            self.params.tau_limits[0] / self.params.wheel_radius,
        )
        add_weighted_target(
            self.FN,
            self.weights.contact_force,
            self.params.mass * self.params.gravity,
            self.params.mass * self.params.gravity,
        )

        for index in [self.LEG_LOW_SLACK, self.LEG_HIGH_SLACK]:
            add_weighted_target(index, self.weights.reachability_slack, 0.0)
        for index in [self.HEIGHT_LOW_SLACK, self.HEIGHT_HIGH_SLACK]:
            add_weighted_target(index, self.weights.height_slack, 0.0)

        # OSQP uses 0.5 * z.T P z + q.T z.
        return hessian, gradient

    @staticmethod
    def _solve_acceleration_from_equalities(
        z: np.ndarray,
        equality_matrix: np.ndarray,
        equality_rhs: np.ndarray,
    ) -> None:
        accel_matrix = equality_matrix[:3, :3]
        rhs = equality_rhs[:3] - equality_matrix[:3, 3:] @ z[3:]
        z[:3] = np.linalg.solve(accel_matrix, rhs)

    def _soft_constraint_slacks(
        self,
        z: np.ndarray,
        state: np.ndarray,
        context: ControlContext,
        model: WBCAffineDynamics,
    ) -> None:
        leg_row, leg_constant, height_row, height_constant = self._prediction_rows(
            state, context, model
        )
        predicted_leg = float(leg_constant[0] + leg_row @ z)
        predicted_height = float(height_constant[0] + height_row @ z)

        min_leg = self.safety.min_leg_length + 0.02
        max_leg = self.params.thigh_length + self.params.shank_length - 0.02
        min_height = context.y_ref - self.safety.height_fail + 0.01
        max_height = context.y_ref + self.safety.height_fail - 0.01

        z[self.LEG_LOW_SLACK] = max(min_leg - predicted_leg, 0.0)
        z[self.LEG_HIGH_SLACK] = max(predicted_leg - max_leg, 0.0)
        z[self.HEIGHT_LOW_SLACK] = max(min_height - predicted_height, 0.0)
        z[self.HEIGHT_HIGH_SLACK] = max(predicted_height - max_height, 0.0)

    def _initial_guess(
        self,
        state: np.ndarray,
        context: ControlContext,
        model: WBCAffineDynamics,
        equality_matrix: np.ndarray,
        equality_rhs: np.ndarray,
    ) -> np.ndarray:
        z = np.zeros(self.NVAR, dtype=float)
        tau = stabilizing_baseline_torque(
            state,
            context.v_cmd,
            context.y_ref,
            self.params,
            context.external_force_x,
        )
        tau = clip_tau(tau, self.params)
        min_contact_normal = (
            self.minimum_normal_force_fraction
            * self.params.mass
            * self.params.gravity
        )
        active_leg_force = (
            tau[1] + 0.6 * tau[2]
        ) / self.params.leg_moment_arm
        contact_normal = active_leg_force + model.passive_leg_force
        if contact_normal < min_contact_normal:
            tau[1] = np.clip(
                self.params.leg_moment_arm
                * (min_contact_normal - model.passive_leg_force)
                - 0.6 * tau[2],
                -self.params.tau_limits[1],
                self.params.tau_limits[1],
            )
            active_leg_force = (
                tau[1] + 0.6 * tau[2]
            ) / self.params.leg_moment_arm
            contact_normal = active_leg_force + model.passive_leg_force
        contact_normal = max(float(contact_normal), min_contact_normal)

        contact_tangent = float(
            np.clip(
                tau[0] / self.params.wheel_radius,
                -self.params.friction_coeff * contact_normal,
                self.params.friction_coeff * contact_normal,
            )
        )
        tau[0] = self.params.wheel_radius * contact_tangent

        z[self.FT] = contact_tangent
        z[self.FN] = contact_normal
        z[self.TAU_WHEEL : self.TAU_HIP + 1] = tau
        self._solve_acceleration_from_equalities(z, equality_matrix, equality_rhs)
        self._soft_constraint_slacks(z, state, context, model)
        return z

    @staticmethod
    def _constraint_residuals(
        z: np.ndarray,
        equality_matrix: np.ndarray,
        equality_rhs: np.ndarray,
        inequality_matrix: np.ndarray,
        inequality_rhs: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> tuple[float, float]:
        equality_residual = float(
            np.max(np.abs(equality_matrix @ z - equality_rhs))
        )
        inequality_violation = float(max(np.max(inequality_matrix @ z - inequality_rhs), 0.0))
        finite_lower = np.isfinite(lower_bounds)
        finite_upper = np.isfinite(upper_bounds)
        if np.any(finite_lower):
            inequality_violation = max(
                inequality_violation,
                float(np.max(lower_bounds[finite_lower] - z[finite_lower])),
            )
        if np.any(finite_upper):
            inequality_violation = max(
                inequality_violation,
                float(np.max(z[finite_upper] - upper_bounds[finite_upper])),
            )
        inequality_violation = max(inequality_violation, 0.0)
        return equality_residual, inequality_violation

    def _solve_qp(
        self,
        hessian: np.ndarray,
        gradient: np.ndarray,
        equality_matrix: np.ndarray,
        equality_rhs: np.ndarray,
        inequality_matrix: np.ndarray,
        inequality_rhs: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        initial_guess: np.ndarray,
    ) -> QPSolveResult:
        constraint_matrix = sparse.vstack(
            [
                sparse.csc_matrix(equality_matrix),
                sparse.csc_matrix(inequality_matrix),
                sparse.eye(self.NVAR, format="csc"),
            ],
            format="csc",
        )
        lower = np.concatenate(
            [
                equality_rhs,
                np.full(inequality_rhs.shape, -np.inf, dtype=float),
                lower_bounds,
            ]
        )
        upper = np.concatenate([equality_rhs, inequality_rhs, upper_bounds])

        try:
            solver = osqp.OSQP()
            solver.setup(
                P=sparse.csc_matrix(0.5 * (hessian + hessian.T)),
                q=gradient,
                A=constraint_matrix,
                l=lower,
                u=upper,
                verbose=False,
                polishing=True,
                warm_starting=True,
                eps_abs=1e-8,
                eps_rel=1e-8,
                max_iter=4000,
            )
            solver.warm_start(x=initial_guess)
            raw_result = solver.solve(raise_error=False)
        except Exception as exc:
            return QPSolveResult(
                x=initial_guess.copy(),
                success=False,
                iterations=0,
                status=f"setup_error:{type(exc).__name__}",
                status_val=-1,
                objective=float("nan"),
                primal_residual=float("nan"),
                dual_residual=float("nan"),
                solve_time=float("nan"),
            )

        info = raw_result.info
        candidate = (
            np.asarray(raw_result.x, dtype=float)
            if raw_result.x is not None
            else initial_guess.copy()
        )
        status_val = int(getattr(info, "status_val", -1))
        solve_time = float(
            getattr(info, "solve_time", getattr(info, "run_time", float("nan")))
        )
        return QPSolveResult(
            x=candidate,
            success=status_val in self.OSQP_SOLVED_STATUS_VALUES,
            iterations=int(getattr(info, "iter", 0)),
            status=str(getattr(info, "status", "unknown")),
            status_val=status_val,
            objective=float(getattr(info, "obj_val", float("nan"))),
            primal_residual=float(getattr(info, "prim_res", float("nan"))),
            dual_residual=float(getattr(info, "dual_res", float("nan"))),
            solve_time=solve_time,
        )

    def _diagnostic_values(
        self,
        z: np.ndarray,
        result: QPSolveResult,
        used_fallback: bool,
        state: np.ndarray,
        context: ControlContext,
        model: WBCAffineDynamics,
        equality_matrix: np.ndarray,
        equality_rhs: np.ndarray,
        inequality_matrix: np.ndarray,
        inequality_rhs: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> dict[str, float]:
        equality_residual, inequality_violation = self._constraint_residuals(
            z,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
            lower_bounds,
            upper_bounds,
        )
        leg_row, leg_constant, height_row, height_constant = self._prediction_rows(
            state, context, model
        )
        predicted_leg = float(leg_constant[0] + leg_row @ z)
        predicted_height = float(height_constant[0] + height_row @ z)
        friction_ratio = abs(float(z[self.FT])) / max(
            self.params.friction_coeff * float(z[self.FN]),
            1.0,
        )
        return {
            "qp_success": float(not used_fallback),
            "qp_solver_success": float(result.success),
            "qp_fallback": float(used_fallback),
            "qp_iterations": float(result.iterations),
            "qp_solver_status_val": float(result.status_val),
            "qp_objective": float(result.objective),
            "qp_primal_residual": float(result.primal_residual),
            "qp_dual_residual": float(result.dual_residual),
            "qp_solve_time_ms": 1000.0 * float(result.solve_time),
            "qp_eq_residual": equality_residual,
            "qp_ineq_violation": inequality_violation,
            "qp_friction_ratio": friction_ratio,
            "qp_leg_slack": float(
                max(z[self.LEG_LOW_SLACK], z[self.LEG_HIGH_SLACK])
            ),
            "qp_height_slack": float(
                max(z[self.HEIGHT_LOW_SLACK], z[self.HEIGHT_HIGH_SLACK])
            ),
            "qp_predicted_leg_length": predicted_leg,
            "qp_predicted_height": predicted_height,
            "qp_planned_contact_normal": float(z[self.FN]),
            "qp_planned_contact_tangent": float(z[self.FT]),
        }

    def compute(self, state: np.ndarray, context: ControlContext) -> np.ndarray:
        model = wbc_affine_dynamics(
            state,
            self.terrain,
            self.params,
            context.external_force_x,
        )
        equality_matrix, equality_rhs, inequality_matrix, inequality_rhs = (
            self._constraints(state, context, model)
        )
        lower_bounds, upper_bounds = self._bounds()
        last_tau = (
            self._last_tau.copy()
            if self._last_tau is not None
            else stabilizing_baseline_torque(
                state,
                context.v_cmd,
                context.y_ref,
                self.params,
                context.external_force_x,
            )
        )
        hessian, gradient = self._objective_matrices(
            self._desired_accel(state, context),
            last_tau,
        )
        initial_guess = self._initial_guess(
            state,
            context,
            model,
            equality_matrix,
            equality_rhs,
        )
        result = self._solve_qp(
            hessian,
            gradient,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
            lower_bounds,
            upper_bounds,
            initial_guess,
        )
        candidate = np.asarray(result.x, dtype=float)
        equality_residual, inequality_violation = self._constraint_residuals(
            candidate,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
            lower_bounds,
            upper_bounds,
        )
        used_fallback = (
            not result.success
            or not np.all(np.isfinite(candidate))
            or equality_residual > 1e-5
            or inequality_violation > 1e-6
        )
        solution = initial_guess if used_fallback else candidate

        tau = clip_tau(solution[self.TAU_WHEEL : self.TAU_HIP + 1], self.params)
        self._last_tau = tau
        self._last_solution = solution.copy()
        self._diagnostics = self._diagnostic_values(
            solution,
            result,
            used_fallback,
            state,
            context,
            model,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
            lower_bounds,
            upper_bounds,
        )
        return tau
