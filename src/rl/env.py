from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - imported only in RL setup
    raise ImportError(
        "gymnasium is required for RL. Install it into the same interpreter "
        "you use to run this script, for example: python -m pip install gymnasium"
    ) from exc

from src.config.params import ROBOT, SAFETY, SIM, RobotParams, SafetyLimits, SimParams
from src.config.random_scenarios import ScenarioSampler
from src.config.scenarios import Scenario, default_scenarios
from src.controllers.base import direct_torque_action, residual_torque_action
from src.model.dynamics import accelerations, initial_state, is_failure, rk4_step
from src.model.kinematics import state_to_observation


class WheelLegRobotEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: list[Scenario] | None = None,
        scenario_sampler: ScenarioSampler | None = None,
        params: RobotParams = ROBOT,
        sim: SimParams = SIM,
        safety: SafetyLimits = SAFETY,
        seed: int | None = None,
        action_mode: str = "residual",
    ):
        super().__init__()
        if action_mode not in {"residual", "direct"}:
            raise ValueError("action_mode must be 'residual' or 'direct'")
        self.params = params
        self.sim = sim
        self.safety = safety
        self.action_mode = action_mode
        self.scenarios = scenarios or default_scenarios()
        self.scenario_sampler = scenario_sampler
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        high = np.array([2.0, np.pi, 8.0, 5.0, 10.0, 1.2, 0.3, 1.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.scenario = self.scenarios[0]
        self.state = initial_state(sim.y_ref)
        self.t = 0.0
        self.steps = 0

    def _obs(self) -> np.ndarray:
        return state_to_observation(
            self.state,
            self.scenario.v_cmd(self.t),
            self.sim.y_ref,
            self.scenario.terrain,
            self.params,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.scenario_sampler is not None:
            self.scenario = self.scenario_sampler.sample(self.rng)
        else:
            self.scenario = self.rng.choice(self.scenarios)
        self.state = initial_state(self.sim.y_ref)
        self.state[1] += self.rng.normal(0.0, 0.01)
        self.state[2] = self.rng.normal(0.0, np.deg2rad(0.8))
        self.state[3] = self.rng.normal(0.0, 0.04)
        self.state[4] = self.rng.normal(0.0, 0.02)
        self.state[5] = self.rng.normal(0.0, np.deg2rad(1.0))
        self.t = 0.0
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -1.0, 1.0)
        v_cmd = self.scenario.v_cmd(self.t)
        force_x = self.scenario.force_x(self.t)
        if self.action_mode == "residual":
            tau = residual_torque_action(
                action, self.state, v_cmd, self.sim.y_ref, self.params, force_x
            )
        else:
            tau = direct_torque_action(action, self.params)

        next_state, info = rk4_step(
            self.state,
            tau,
            self.sim.dt,
            self.scenario.terrain,
            self.params,
            external_force_x=force_x,
        )
        acc, _ = accelerations(
            self.state, tau, self.scenario.terrain, self.params, external_force_x=force_x
        )
        failed, reason = is_failure(
            next_state, info, self.sim.y_ref, self.params, self.safety
        )

        y_err = next_state[1] - self.sim.y_ref
        theta = next_state[2]
        v_err = next_state[3] - v_cmd
        reward = 0.5
        reward += 3.5 * np.exp(-130.0 * y_err * y_err)
        reward += 2.5 * np.exp(-45.0 * theta * theta)
        reward += 1.2 * np.exp(-0.7 * v_err * v_err)
        reward -= 35.0 * float(y_err * y_err)
        reward -= 3.0 * float(theta * theta)
        action_cost = 0.03 if self.action_mode == "residual" else 0.008
        reward -= action_cost * float(np.sum(action * action))
        reward -= 0.001 * float(np.sum(acc * acc))
        if self.action_mode == "direct":
            tau_ratio = tau / np.maximum(self.params.tau_limits, 1.0)
            reward -= 0.004 * float(np.sum(tau_ratio * tau_ratio))
        if np.any(np.abs(action) > 0.98):
            reward -= 0.5
        if failed:
            reward -= 50.0

        self.state = next_state
        self.t += self.sim.dt
        self.steps += 1
        terminated = bool(failed)
        truncated = bool(self.steps >= self.sim.max_steps)
        info_dict = {"failure_reason": reason, "scenario": self.scenario.name}
        return self._obs(), float(reward), terminated, truncated, info_dict
