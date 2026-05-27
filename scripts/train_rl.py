from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.env import WheelLegRobotEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--save-path", default="outputs/models/ppo_wheel_leg")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError(
            "Install RL dependencies into the same interpreter first: "
            "python -m pip install gymnasium stable-baselines3 torch"
        ) from exc

    def make_env():
        return Monitor(WheelLegRobotEnv(seed=args.seed))

    env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        device="auto",
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Saved PPO model to {save_path}")


if __name__ == "__main__":
    main()
