from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.random_scenarios import training_sampler
from src.config.scenarios import training_scenarios
from src.controllers.rl_policy import load_ppo_model
from src.rl.env import WheelLegRobotEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--load-path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--action-cost", type=float, default=None)
    parser.add_argument("--direct-torque-cost", type=float, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument(
        "--provide-force-measurement",
        action="store_true",
        help="Provide exact horizontal disturbances to residual baselines during training.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["residual", "feedforward_residual", "direct"],
        default="residual",
    )
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

    default_save_paths = {
        "residual": "outputs/models/ppo_wheel_leg_residual",
        "feedforward_residual": "outputs/models/ppo_wheel_leg_feedforward_residual",
        "direct": "outputs/models/ppo_wheel_leg_direct",
    }
    save_path = Path(args.save_path or default_save_paths[args.action_mode])
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
                    provide_force_measurement=args.provide_force_measurement,
                    action_cost=args.action_cost,
                    direct_torque_cost=args.direct_torque_cost,
                )
            )

        return _init

    env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    if args.load_path:
        model = load_ppo_model(args.load_path, env=env, device=args.device)
        model.verbose = args.verbose
    else:
        ent_coef = (
            args.ent_coef
            if args.ent_coef is not None
            else (0.003 if args.action_mode != "direct" else 0.008)
        )
        learning_rate = (
            args.learning_rate
            if args.learning_rate is not None
            else (2e-4 if args.action_mode != "direct" else 3e-4)
        )
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
            verbose=args.verbose,
            seed=args.seed,
            device=args.device,
        )
    model.learn(total_timesteps=args.timesteps, progress_bar=not args.no_progress_bar)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Saved PPO model to {save_path}")


if __name__ == "__main__":
    main()
