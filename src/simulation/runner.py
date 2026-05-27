from __future__ import annotations

import numpy as np

from src.config.params import RobotParams, SafetyLimits
from src.config.scenarios import Scenario
from src.controllers.base import ControlContext, Controller
from src.model.dynamics import initial_state, is_failure, rk4_step
from src.model.kinematics import inverse_vertical_leg, wheel_center_y
from src.simulation.logger import SimulationLog


class SimulationRunner:
    def __init__(
        self,
        params: RobotParams,
        safety: SafetyLimits,
        dt: float = 0.01,
        y_ref: float = 0.82,
    ):
        self.params = params
        self.safety = safety
        self.dt = dt
        self.y_ref = y_ref

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

        for k in range(steps + 1):
            t = k * self.dt
            v_cmd = scenario.v_cmd(t)
            force_x = scenario.force_x(t)
            context = ControlContext(
                t=t, v_cmd=v_cmd, y_ref=self.y_ref, external_force_x=force_x
            )
            tau = controller.compute(state.copy(), context)
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
            )

            state = next_state
            if failed and not log.failure_reason:
                log.failure_reason = reason
            if failed and stop_on_failure:
                break

        return log
