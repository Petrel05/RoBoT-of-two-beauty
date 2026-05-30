from __future__ import annotations

import numpy as np

from src.config.params import RobotParams
from src.controllers.base import (
    ControlContext,
    Controller,
    direct_torque_action,
    residual_torque_action,
)
from src.model.kinematics import state_to_observation
from src.model.terrain import Terrain


class RandomPolicyController(Controller):
    name = "rl_random"

    def __init__(self, params: RobotParams, seed: int = 0):
        self.params = params
        self.rng = np.random.default_rng(seed)

    def compute(self, state: np.ndarray, context: ControlContext) -> np.ndarray:
        action = self.rng.uniform(-0.25, 0.25, size=3)
        return residual_torque_action(
            action,
            state,
            context.v_cmd,
            context.y_ref,
            self.params,
            context.external_force_x,
        )


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
        return residual_torque_action(
            action,
            state,
            context.v_cmd,
            context.y_ref,
            self.params,
            context.external_force_x,
        )


class DirectPPOController(Controller):
    name = "rl_ppo_direct"

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
        return direct_torque_action(action, self.params)
