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

Train PPO:

```bash
python scripts/train_rl.py --timesteps 300000 --save-path outputs/models/ppo_wheel_leg
```

Evaluate the trained policy:

```bash
python scripts/eval_rl.py --model-path outputs/models/ppo_wheel_leg --out-dir outputs/eval_rl
```

Compare trained PPO against model-based controllers:

```bash
python scripts/run_compare.py --controllers lqr wbc_qp rl_ppo --model-path outputs/models/ppo_wheel_leg --out-dir outputs/compare_trained
```

If you want plots to continue after a controller has already failed, add:

```bash
python scripts/run_compare.py --controllers lqr wbc_qp rl_random --continue-after-failure
```

## Notes

- The default simulation is intentionally lightweight. It is meant to support the report-level comparison and provide a stable code scaffold.
- `rl_random` is only a placeholder controller for checking the pipeline before PPO training.
- `tau_limits` are defined in `src/config/params.py` as self-defined actuator limits, not as task-given constants.
