from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.params import ROBOT, SAFETY, SIM
from src.config.scenarios import default_scenarios
from src.controllers.lqr import LQRController
from src.controllers.rl_policy import RandomPolicyController, TrainedPPOController
from src.controllers.wbc_qp import WBCQPController
from src.evaluation.metrics import compute_metrics, format_metrics_table
from src.evaluation.plots import plot_log
from src.simulation.runner import SimulationRunner


def build_controller(name: str, scenario, model_path: str | None = None):
    if name == "lqr":
        return LQRController(ROBOT, SIM.y_ref)
    if name == "wbc_qp":
        return WBCQPController(ROBOT, scenario.terrain)
    if name == "rl_random":
        return RandomPolicyController(ROBOT, seed=0)
    if name == "rl_ppo":
        if not model_path:
            raise ValueError("--model-path is required for rl_ppo")
        return TrainedPPOController(model_path, ROBOT, scenario.terrain)
    raise ValueError(f"Unknown controller: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["lqr", "wbc_qp", "rl_random"],
        choices=["lqr", "wbc_qp", "rl_random", "rl_ppo"],
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="Keep simulating after a failure is first detected.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    log_dir = out_dir / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = SimulationRunner(ROBOT, SAFETY, dt=SIM.dt, y_ref=SIM.y_ref)
    rows = []
    for scenario in default_scenarios():
        for controller_name in args.controllers:
            controller = build_controller(controller_name, scenario, args.model_path)
            log = runner.run(
                controller,
                scenario,
                stop_on_failure=not args.continue_after_failure,
            )
            metrics = compute_metrics(log, ROBOT)
            row = {
                "controller": controller.name,
                "scenario": scenario.name,
                **metrics,
            }
            rows.append(row)
            stem = f"{scenario.name}__{controller.name}"
            log.save_npz(str(log_dir / f"{stem}.npz"))
            plot_log(log, f"{scenario.name} / {controller.name}", fig_dir / f"{stem}.png")
            print(f"finished {stem}: success={row['success']} reason={row['failure_reason']}")

    table = format_metrics_table(rows)
    (out_dir / "metrics.csv").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
