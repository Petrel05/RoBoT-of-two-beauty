from __future__ import annotations

import numpy as np

from src.config.params import RobotParams
from src.controllers.base import ControlContext, Controller, clip_tau
from src.model.kinematics import state_to_observation
from src.model.terrain import Terrain


class RandomPolicyController(Controller):
    name = "rl_random"

    def __init__(self, params: RobotParams, seed: int = 0):
        self.params = params
        self.rng = np.random.default_rng(seed)

    def compute(self, state: np.ndarray, context: ControlContext) -> np.ndarray:
        action = self.rng.uniform(-0.25, 0.25, size=3)
        return action * self.params.tau_limits


class TrainedPPOController(Controller):
    name = "rl_ppo"

    def __init__(self, model_path: str, params: RobotParams, terrain: Terrain):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required to load a trained PPO controller."
            ) from exc
        self.model = PPO.load(model_path)
        self.params = params
        self.terrain = terrain

    def compute(self, state: np.ndarray, context: ControlContext) -> np.ndarray:
        obs = state_to_observation(
            state, context.v_cmd, context.y_ref, self.terrain, self.params
        )
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=float)
        return clip_tau(action * self.params.tau_limits, self.params)

