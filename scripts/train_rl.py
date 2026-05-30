from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.random_scenarios import training_sampler
from src.config.scenarios import training_scenarios
from src.rl.env import WheelLegRobotEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--load-path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--action-mode", choices=["residual", "direct"], default="residual")
    parser.add_argument(
        "--scenario-set",
        choices=[
            "easy",
            "default",
            "all",
            "random_easy",
            "random_force",
            "random_full",
        ],
        default="default",
    )
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

    save_path = Path(
        args.save_path
        or (
            "outputs/models/ppo_wheel_leg_residual"
            if args.action_mode == "residual"
            else "outputs/models/ppo_wheel_leg_direct"
        )
    )
    sampler = training_sampler(args.scenario_set)
    scenarios = None if sampler is not None else training_scenarios(args.scenario_set)

    def make_env(rank: int):
        def _init():
            return Monitor(
                WheelLegRobotEnv(
                    scenarios=scenarios,
                    scenario_sampler=sampler,
                    seed=args.seed + rank,
                    action_mode=args.action_mode,
                )
            )

        return _init

    env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    if args.load_path:
        model = PPO.load(args.load_path, env=env, device=args.device)
        model.verbose = 1
    else:
        ent_coef = 0.003 if args.action_mode == "residual" else 0.008
        learning_rate = 2e-4 if args.action_mode == "residual" else 3e-4
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Saved PPO model to {save_path}")


if __name__ == "__main__":
    main()
