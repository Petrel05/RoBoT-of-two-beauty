from __future__ import annotations

import numpy as np

from src.config.params import RobotParams, SafetyLimits
from src.config.scenarios import Scenario
from src.controllers.base import ControlContext, Controller
from src.model.dynamics import accelerations, initial_state, is_failure, rk4_step
from src.model.kinematics import inverse_vertical_leg, wheel_center_y
from src.simulation.logger import SimulationLog


class SimulationRunner:
    def __init__(
        self,
        params: RobotParams,
        safety: SafetyLimits,
        dt: float = 0.01,
        y_ref: float = 0.82,
        provide_force_measurement: bool = False,
    ):
        self.params = params
        self.safety = safety
        self.dt = dt
        self.y_ref = y_ref
        self.provide_force_measurement = provide_force_measurement

    def run(
        self,
        controller: Controller,
        scenario: Scenario,
        stop_on_failure: bool = True,
    ) -> SimulationLog:
        controller.reset()
        state = initial_state(self.y_ref)
        log = SimulationLog()
        steps = int(np.ceil(scenario.duration / self.dt))

        def context_at(t: float) -> ControlContext:
            force_x = scenario.force_x(t)
            measured_force_x = force_x if self.provide_force_measurement else 0.0
            return ControlContext(
                t=t,
                v_cmd=scenario.v_cmd(t),
                y_ref=self.y_ref,
                external_force_x=measured_force_x,
            )

        def append_row(
            t: float,
            tau: np.ndarray,
            info,
            diagnostics: dict[str, float],
            failed: bool,
        ) -> None:
            v_cmd = scenario.v_cmd(t)
            force_x = scenario.force_x(t)
            wheel_y = wheel_center_y(state[0], scenario.terrain, self.params)
            leg = inverse_vertical_leg(state[1], wheel_y, self.params)
            log.append(
                t=t,
                x=state[0],
                y=state[1],
                theta=state[2],
                vx=state[3],
                vy=state[4],
                omega=state[5],
                v_cmd=v_cmd,
                y_ref=self.y_ref,
                force_x=force_x,
                terrain_h=scenario.terrain.height(state[0]),
                terrain_slope=scenario.terrain.slope(state[0]),
                wheel_y=wheel_y,
                leg_length=leg.leg_length,
                tau_wheel=tau[0],
                tau_knee=tau[1],
                tau_hip=tau[2],
                contact_normal=info.contact_normal,
                contact_tangent=info.contact_tangent,
                ax=info.ax,
                ay=info.ay,
                alpha=info.alpha,
                saturated=float(np.any(info.saturated)),
                failed=float(failed),
                qp_success=float(diagnostics.get("qp_success", np.nan)),
                qp_solver_success=float(
                    diagnostics.get("qp_solver_success", np.nan)
                ),
                qp_fallback=float(diagnostics.get("qp_fallback", np.nan)),
                qp_iterations=float(diagnostics.get("qp_iterations", np.nan)),
                qp_eq_residual=float(diagnostics.get("qp_eq_residual", np.nan)),
                qp_ineq_violation=float(
                    diagnostics.get("qp_ineq_violation", np.nan)
                ),
                qp_friction_ratio=float(
                    diagnostics.get("qp_friction_ratio", np.nan)
                ),
                qp_leg_slack=float(diagnostics.get("qp_leg_slack", np.nan)),
                qp_height_slack=float(diagnostics.get("qp_height_slack", np.nan)),
                qp_predicted_leg_length=float(
                    diagnostics.get("qp_predicted_leg_length", np.nan)
                ),
                qp_predicted_height=float(
                    diagnostics.get("qp_predicted_height", np.nan)
                ),
                qp_planned_contact_normal=float(
                    diagnostics.get("qp_planned_contact_normal", np.nan)
                ),
                qp_planned_contact_tangent=float(
                    diagnostics.get("qp_planned_contact_tangent", np.nan)
                ),
            )

        context = context_at(0.0)
        tau = controller.compute(state.copy(), context)
        diagnostics = controller.diagnostics()
        _, info = accelerations(
            state,
            tau,
            scenario.terrain,
            self.params,
            external_force_x=scenario.force_x(0.0),
        )
        append_row(0.0, tau, info, diagnostics, failed=False)

        for k in range(steps):
            t = k * self.dt
            force_x = scenario.force_x(t)
            if k > 0:
                context = context_at(t)
                tau = controller.compute(state.copy(), context)
                diagnostics = controller.diagnostics()
            next_state, info = rk4_step(
                state,
                tau,
                self.dt,
                scenario.terrain,
                self.params,
                external_force_x=force_x,
            )
            failed, reason = is_failure(
                next_state, info, self.y_ref, self.params, self.safety
            )

            state = next_state
            append_row((k + 1) * self.dt, tau, info, diagnostics, failed)
            if failed and not log.failure_reason:
                log.failure_reason = reason
            if failed and stop_on_failure:
                break

        return log
