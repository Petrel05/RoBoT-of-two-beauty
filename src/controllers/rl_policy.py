from __future__ import annotations

import sys

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


def ppo_space_custom_objects() -> dict:
    try:
        from gymnasium import spaces
    except ImportError as exc:
        raise ImportError(
            "gymnasium is required to construct the PPO policy spaces."
        ) from exc

    observation_high = np.array(
        [2.0, np.pi, 8.0, 5.0, 10.0, 1.2, 0.3, 1.0, 5.0],
        dtype=np.float32,
    )
    return {
        "action_space": spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        ),
        "observation_space": spaces.Box(
            low=-observation_high, high=observation_high, dtype=np.float32
        ),
    }


def install_numpy_pickle_compat() -> None:
    # Models saved with NumPy 2 refer to the private numpy._core path. NumPy 1.x
    # exposes the same implementation through numpy.core.
    try:
        import numpy._core.numeric  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError:
        import numpy.core.numeric as numeric

        sys.modules.setdefault("numpy._core.numeric", numeric)


def load_ppo_model(model_path: str, *, env=None, device: str = "cpu"):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required to load a trained PPO controller."
        ) from exc

    install_numpy_pickle_compat()
    return PPO.load(
        model_path,
        env=env,
        custom_objects=ppo_space_custom_objects(),
        device=device,
    )


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
        self.model = load_ppo_model(model_path)
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
        self.model = load_ppo_model(model_path)
        self.params = params
        self.terrain = terrain

    def compute(self, state: np.ndarray, context: ControlContext) -> np.ndarray:
        obs = state_to_observation(
            state, context.v_cmd, context.y_ref, self.terrain, self.params
        )
        action, _ = self.model.predict(obs, deterministic=True)
        return direct_torque_action(action, self.params)
