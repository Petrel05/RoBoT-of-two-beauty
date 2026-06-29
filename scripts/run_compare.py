from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.params import ROBOT, SAFETY, SIM
from src.config.scenarios import default_scenarios, static_push_scenarios
from src.controllers.lqr import LQRController
from src.controllers.rl_policy import (
    DirectPPOController,
    FeedforwardPPOController,
    RandomPolicyController,
    TrainedPPOController,
)
from src.controllers.wbc_qp import WBCQPController
from src.evaluation.metrics import compute_metrics, format_metrics_table
from src.evaluation.plots import (
    plot_controller_comparison,
    plot_log,
    plot_wbc_diagnostics,
)
from src.simulation.runner import SimulationRunner


def build_controller(
    name: str,
    scenario,
    model_path: str | None = None,
    residual_model_path: str | None = None,
    feedforward_model_path: str | None = None,
    direct_model_path: str | None = None,
):
    if name == "lqr":
        return LQRController(ROBOT, SIM.y_ref)
    if name == "wbc_qp":
        return WBCQPController(ROBOT, scenario.terrain, SAFETY, dt=SIM.dt)
    if name == "rl_random":
        return RandomPolicyController(ROBOT, seed=0)
    if name == "rl_ppo":
        path = residual_model_path or model_path
        if not path:
            raise ValueError("--model-path is required for rl_ppo")
        return TrainedPPOController(path, ROBOT, scenario.terrain)
    if name == "rl_ppo_feedforward":
        path = feedforward_model_path or model_path
        if not path:
            raise ValueError(
                "--feedforward-model-path or --model-path is required for rl_ppo_feedforward"
            )
        return FeedforwardPPOController(path, ROBOT, scenario.terrain)
    if name == "rl_ppo_direct":
        path = direct_model_path or model_path
        if not path:
            raise ValueError("--direct-model-path or --model-path is required for rl_ppo_direct")
        return DirectPPOController(path, ROBOT, scenario.terrain)
    raise ValueError(f"Unknown controller: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["lqr", "wbc_qp", "rl_random"],
        choices=[
            "lqr",
            "wbc_qp",
            "rl_random",
            "rl_ppo",
            "rl_ppo_feedforward",
            "rl_ppo_direct",
        ],
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--residual-model-path", default=None)
    parser.add_argument("--feedforward-model-path", default=None)
    parser.add_argument("--direct-model-path", default=None)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument(
        "--scenario-set",
        choices=["default", "static_push"],
        default="default",
    )
    parser.add_argument(
        "--provide-force-measurement",
        action="store_true",
        help="Provide the exact horizontal disturbance to controllers that support it.",
    )
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

    runner = SimulationRunner(
        ROBOT,
        SAFETY,
        dt=SIM.dt,
        y_ref=SIM.y_ref,
        provide_force_measurement=args.provide_force_measurement,
    )
    rows = []
    scenarios = (
        static_push_scenarios()
        if args.scenario_set == "static_push"
        else default_scenarios()
    )
    for scenario in scenarios:
        scenario_logs = []
        for controller_name in args.controllers:
            controller = build_controller(
                controller_name,
                scenario,
                model_path=args.model_path,
                residual_model_path=args.residual_model_path,
                feedforward_model_path=args.feedforward_model_path,
                direct_model_path=args.direct_model_path,
            )
            log = runner.run(
                controller,
                scenario,
                stop_on_failure=not args.continue_after_failure,
            )
            metrics = compute_metrics(log, ROBOT)
            scenario_logs.append((controller.name, log))
            row = {
                "controller": controller.name,
                "scenario": scenario.name,
                **metrics,
            }
            rows.append(row)
            stem = f"{scenario.name}__{controller.name}"
            log.save_npz(str(log_dir / f"{stem}.npz"))
            plot_log(log, f"{scenario.name} / {controller.name}", fig_dir / f"{stem}.png")
            plot_wbc_diagnostics(
                log,
                ROBOT,
                SAFETY,
                f"{scenario.name} / {controller.name} constraints",
                fig_dir / f"{stem}__diagnostics.png",
            )
            print(f"finished {stem}: success={row['success']} reason={row['failure_reason']}")
        if len(scenario_logs) > 1:
            plot_controller_comparison(
                scenario_logs,
                f"{scenario.name} controller comparison",
                fig_dir / f"{scenario.name}__controller_comparison.png",
            )

    table = format_metrics_table(rows)
    (out_dir / "metrics.csv").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
