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
from src.config.scenarios import Scenario, default_scenarios
from src.model.dynamics import accelerations, initial_state, is_failure, rk4_step
from src.model.kinematics import state_to_observation


class WheelLegRobotEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: list[Scenario] | None = None,
        params: RobotParams = ROBOT,
        sim: SimParams = SIM,
        safety: SafetyLimits = SAFETY,
        seed: int | None = None,
    ):
        super().__init__()
        self.params = params
        self.sim = sim
        self.safety = safety
        self.scenarios = scenarios or default_scenarios()
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
        self.scenario = self.rng.choice(self.scenarios)
        self.state = initial_state(self.sim.y_ref)
        self.state[2] = self.rng.normal(0.0, np.deg2rad(1.0))
        self.state[3] = self.rng.normal(0.0, 0.05)
        self.t = 0.0
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -1.0, 1.0)
        tau = action * self.params.tau_limits
        v_cmd = self.scenario.v_cmd(self.t)
        force_x = self.scenario.force_x(self.t)

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
        reward = 3.0 * np.exp(-30.0 * theta * theta)
        reward += 3.0 * np.exp(-55.0 * y_err * y_err)
        reward += 1.5 * np.exp(-0.6 * v_err * v_err)
        reward -= 0.01 * float(np.sum(action * action))
        reward -= 0.002 * float(np.sum(acc * acc))
        if np.any(np.abs(action) > 0.98):
            reward -= 0.3
        if failed:
            reward -= 20.0

        self.state = next_state
        self.t += self.sim.dt
        self.steps += 1
        terminated = bool(failed)
        truncated = bool(self.steps >= self.sim.max_steps)
        info_dict = {"failure_reason": reason, "scenario": self.scenario.name}
        return self._obs(), float(reward), terminated, truncated, info_dict
