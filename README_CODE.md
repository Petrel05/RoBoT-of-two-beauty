# Wheel-Leg Robot Control Code

This code provides a lightweight 2D simulation for comparing LQR, QP/WBC, and RL-style controllers for the report task.

## Quick Run

```bash
cd /Users/apple/Desktop/code/python/robot2
python3 scripts/run_compare.py --controllers lqr wbc_qp rl_random
```

Outputs are written to:

- `outputs/metrics.csv`
- `outputs/figures/*.png`
- `outputs/logs/*.npz`

## PPO Training Order

Create or activate your M1 PyTorch environment first, then install dependencies. On macOS, prefer `python -m pip` inside the conda environment instead of `python3 -m pip`; `python3` may point to Homebrew's externally managed Python and trigger PEP 668 errors.

```bash
cd /Users/apple/Desktop/code/python/robot2
conda activate m1torch
which python
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements.txt
```

If the printed Python path is not inside your conda environment, use the conda environment Python explicitly, for example:

```bash
/opt/anaconda3/envs/m1torch/bin/python -m pip install -r requirements.txt
```

If `m1torch` is not a real conda environment or still points to Homebrew Python, create a clean environment:

```bash
conda create -n robot2 python=3.11 -y
conda activate robot2
python -m pip install -r requirements.txt
```

Check whether PyTorch can see Apple Silicon MPS:

```bash
python -c "import torch; print('mps:', torch.backends.mps.is_available())"
```

Smoke-test the environment:

```bash
python -c "from src.rl.env import WheelLegRobotEnv; env=WheelLegRobotEnv(); obs,_=env.reset(); print(obs.shape); print(env.step(env.action_space.sample())[1])"
```

Train PPO. The current RL setup uses residual torque control, so the policy learns a correction around gravity compensation rather than the full torque from scratch. For a fairer generalization test, train on randomized smooth scenarios and keep `run_compare.py` fixed benchmark scenarios for testing.

```bash
python scripts/train_rl.py \
  --action-mode residual \
  --scenario-set random_full \
  --timesteps 300000 \
  --n-envs 4 \
  --save-path outputs/models/ppo_wheel_leg_residual_random
```

Evaluate the trained policy:

```bash
python scripts/eval_rl.py --model-path outputs/models/ppo_wheel_leg_residual_random --out-dir outputs/eval_rl
```

Compare trained PPO against model-based controllers:

```bash
python scripts/run_compare.py --controllers lqr wbc_qp rl_ppo --model-path outputs/models/ppo_wheel_leg_residual_random --out-dir outputs/compare_trained
```

The older non-residual model path `outputs/models/ppo_wheel_leg` is not compatible with the current residual-policy controller. Retrain or use a residual model such as `outputs/models/ppo_wheel_leg_residual_random`.

Random training scenario sets:

- `random_easy`: random speeds on flat ground without external forces.
- `random_force`: random speeds with smooth random force disturbances on flat ground.
- `random_full`: random speeds, smooth random terrains, and smooth random force disturbances.

Fixed benchmark scenarios are still only used by `run_compare.py`: impulse push, constant push, sine terrain, and smooth noise terrain.

## Direct PPO From Scratch

`rl_ppo_direct` is the fourth controller variant. It outputs the full torque command directly instead of a residual around a PD/gravity-compensation controller. This is intentionally harder and should be trained with a curriculum.

Stage 1: learn to stand and move on easy flat-ground tasks.

```bash
python scripts/train_rl.py \
  --action-mode direct \
  --scenario-set random_easy \
  --timesteps 1000000 \
  --n-envs 8 \
  --save-path outputs/models/ppo_wheel_leg_direct_stage1
```

Stage 2: continue from stage 1 on random force tasks.

```bash
python scripts/train_rl.py \
  --action-mode direct \
  --scenario-set random_force \
  --load-path outputs/models/ppo_wheel_leg_direct_stage1 \
  --timesteps 1000000 \
  --n-envs 8 \
  --save-path outputs/models/ppo_wheel_leg_direct_stage2
```

Stage 3: continue from stage 2 on random terrain + random force tasks.

```bash
python scripts/train_rl.py \
  --action-mode direct \
  --scenario-set random_full \
  --load-path outputs/models/ppo_wheel_leg_direct_stage2 \
  --timesteps 1500000 \
  --n-envs 8 \
  --save-path outputs/models/ppo_wheel_leg_direct
```

Evaluate direct PPO alone:

```bash
python scripts/run_compare.py \
  --controllers rl_ppo_direct \
  --direct-model-path outputs/models/ppo_wheel_leg_direct \
  --out-dir outputs/eval_direct
```

Compare all four controllers:

```bash
python scripts/run_compare.py \
  --controllers lqr wbc_qp rl_ppo rl_ppo_direct \
  --residual-model-path outputs/models/ppo_wheel_leg_residual \
  --direct-model-path outputs/models/ppo_wheel_leg_direct \
  --out-dir outputs/compare_four
```

Expected behavior: direct PPO may need millions of steps. A very short smoke run is expected to fail because the policy has not yet learned to support the body.

If you want plots to continue after a controller has already failed, add:

```bash
python scripts/run_compare.py --controllers lqr wbc_qp rl_random --continue-after-failure
```

## Notes

- The default simulation is intentionally lightweight. It is meant to support the report-level comparison and provide a stable code scaffold.
- `rl_random` is only a placeholder controller for checking the pipeline before PPO training.
- `tau_limits` are defined in `src/config/params.py` as self-defined actuator limits, not as task-given constants.
