from __future__ import annotations

import unittest

import numpy as np

from src.config.params import RobotParams, SafetyLimits
from src.config.scenarios import (
    Scenario,
    constant_force,
    constant_speed,
    default_scenarios,
    no_force,
)
from src.controllers.base import ControlContext, Controller, equilibrium_torque
from src.controllers.wbc_qp import WBCQPController
from src.model.dynamics import accelerations, clip_control, wbc_affine_dynamics
from src.model.terrain import FlatTerrain, NoiseTerrain, SineTerrain
from src.simulation.runner import SimulationRunner


class ZeroController(Controller):
    name = "zero"

    def compute(self, state: np.ndarray, context: ControlContext) -> np.ndarray:
        return np.zeros(3, dtype=float)


class ContextCaptureController(Controller):
    name = "capture"

    def __init__(self, params: RobotParams):
        self.params = params
        self.measured_forces: list[float] = []

    def reset(self) -> None:
        self.measured_forces = []

    def compute(self, state: np.ndarray, context: ControlContext) -> np.ndarray:
        self.measured_forces.append(context.external_force_x)
        return equilibrium_torque(context.v_cmd, self.params)


class DynamicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = RobotParams()

    def test_com_force_does_not_create_pitch_acceleration(self) -> None:
        state = np.array([0.0, 0.82, 0.0, 0.0, 0.0, 0.0], dtype=float)
        tau = equilibrium_torque(0.0, self.params)

        qdd_without_force, _ = accelerations(
            state, tau, FlatTerrain(), self.params, external_force_x=0.0
        )
        qdd_with_force, _ = accelerations(
            state, tau, FlatTerrain(), self.params, external_force_x=100.0
        )

        self.assertAlmostEqual(qdd_without_force[2], qdd_with_force[2], places=12)
        self.assertAlmostEqual(qdd_with_force[0] - qdd_without_force[0], 2.0)

    def test_wbc_affine_equations_match_forward_dynamics(self) -> None:
        terrain = SineTerrain(amplitude=0.04, wavelength=1.3)
        state = np.array([0.37, 0.84, 0.03, 1.2, -0.08, 0.04], dtype=float)
        control = np.array([4.0, 122.0, -18.0], dtype=float)
        external_force_x = 37.0

        qdd, info = accelerations(
            state, control, terrain, self.params, external_force_x
        )
        clipped_control, _ = clip_control(control, self.params)
        decision = np.array(
            [
                qdd[0],
                qdd[1],
                qdd[2],
                info.contact_tangent,
                info.contact_normal,
                *clipped_control,
            ],
            dtype=float,
        )
        model = wbc_affine_dynamics(
            state, terrain, self.params, external_force_x
        )

        np.testing.assert_allclose(
            model.equality_matrix @ decision,
            model.equality_rhs,
            atol=1e-10,
            rtol=0.0,
        )

    def test_wbc_qp_solves_with_osqp(self) -> None:
        controller = WBCQPController(
            self.params,
            FlatTerrain(),
            SafetyLimits(),
            dt=0.01,
        )
        state = np.array([0.0, 0.82, 0.0, 0.4, 0.0, 0.0], dtype=float)
        context = ControlContext(
            t=0.0,
            v_cmd=1.0,
            y_ref=0.82,
            external_force_x=0.0,
        )

        tau = controller.compute(state, context)
        diagnostics = controller.diagnostics()

        self.assertTrue(np.all(np.isfinite(tau)))
        self.assertEqual(diagnostics["qp_solver_success"], 1.0)
        self.assertEqual(diagnostics["qp_fallback"], 0.0)
        self.assertIn(diagnostics["qp_solver_status_val"], {1.0, 2.0})
        self.assertLess(diagnostics["qp_eq_residual"], 1e-5)
        self.assertLess(diagnostics["qp_ineq_violation"], 1e-6)


class TerrainTests(unittest.TestCase):
    def test_noise_terrain_has_smooth_bounded_curvature(self) -> None:
        terrain = NoiseTerrain(amplitude=0.04, seed=7)
        xs = np.linspace(0.0, 20.0, 2001)
        curvature = np.array([terrain.curvature(float(x)) for x in xs])

        self.assertLess(float(np.max(np.abs(curvature))), 1.0)


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = RobotParams()
        self.safety = SafetyLimits()

    def test_runner_logs_exact_horizon(self) -> None:
        scenario = Scenario(
            name="hold",
            duration=8.0,
            v_cmd=constant_speed(0.0),
            force_x=no_force,
            terrain=FlatTerrain(),
            description="Static horizon check.",
        )
        runner = SimulationRunner(self.params, self.safety)
        log = runner.run(ContextCaptureController(self.params), scenario)

        self.assertEqual(len(log.rows), 801)
        self.assertAlmostEqual(log.rows[0]["t"], 0.0)
        self.assertAlmostEqual(log.rows[-1]["t"], 8.0)

    def test_runner_logs_the_failed_next_state(self) -> None:
        scenario = Scenario(
            name="fall",
            duration=8.0,
            v_cmd=constant_speed(0.0),
            force_x=no_force,
            terrain=FlatTerrain(),
            description="Failure-state logging check.",
        )
        runner = SimulationRunner(self.params, self.safety)
        log = runner.run(ZeroController(), scenario)

        self.assertEqual(log.failure_reason, "height_fail")
        self.assertEqual(log.rows[-1]["failed"], 1.0)
        self.assertGreater(
            abs(log.rows[-1]["y"] - runner.y_ref),
            self.safety.height_fail,
        )

    def test_disturbance_measurement_is_opt_in(self) -> None:
        scenario = Scenario(
            name="force",
            duration=0.02,
            v_cmd=constant_speed(0.0),
            force_x=constant_force(80.0),
            terrain=FlatTerrain(),
            description="Force-measurement fairness check.",
        )

        blind_controller = ContextCaptureController(self.params)
        SimulationRunner(self.params, self.safety).run(blind_controller, scenario)
        self.assertEqual(blind_controller.measured_forces, [0.0, 0.0])

        sensed_controller = ContextCaptureController(self.params)
        SimulationRunner(
            self.params,
            self.safety,
            provide_force_measurement=True,
        ).run(sensed_controller, scenario)
        self.assertEqual(sensed_controller.measured_forces, [80.0, 80.0])


class ScenarioTests(unittest.TestCase):
    def test_fixed_scenarios_cover_requirement_boundary(self) -> None:
        scenarios = {scenario.name: scenario for scenario in default_scenarios()}

        self.assertEqual(len(scenarios), 8)
        self.assertAlmostEqual(scenarios["F_high_speed_flat"].v_cmd(8.0), 5.0)
        boundary = scenarios["G_requirement_boundary"]
        self.assertAlmostEqual(boundary.v_cmd(8.0), 5.0)
        self.assertAlmostEqual(boundary.force_x(4.0), 100.0)
        self.assertNotIsInstance(boundary.terrain, FlatTerrain)


if __name__ == "__main__":
    unittest.main()
