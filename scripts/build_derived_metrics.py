from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.params import ROBOT


DERIVED_HEADERS = [
    "rmse_v_steady_t_ge_2",
    "sat_ratio_wheel",
    "sat_ratio_knee",
    "sat_ratio_hip",
    "jerk_proxy",
    "torque_energy",
]


def _trapz(y: np.ndarray, t: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    return float(np.trapezoid(y, t))


def _derived_from_log(path: Path) -> dict[str, float]:
    data = np.load(path)
    t = data["t"].astype(float)
    vx = data["vx"].astype(float)
    v_cmd = data["v_cmd"].astype(float)
    tau = np.vstack(
        [
            data["tau_wheel"].astype(float),
            data["tau_knee"].astype(float),
            data["tau_hip"].astype(float),
        ]
    ).T

    steady = t >= 2.0
    if np.any(steady):
        v_err = vx[steady] - v_cmd[steady]
    else:
        v_err = vx - v_cmd

    tau_abs_ratio = np.abs(tau) / ROBOT.tau_limits
    jerk_trans = (
        np.gradient(data["ax"].astype(float), t, edge_order=1)
        if t.size > 2
        else np.zeros_like(t)
    )
    theta_acc = data["alpha"].astype(float)

    return {
        "rmse_v_steady_t_ge_2": float(np.sqrt(np.mean(v_err * v_err))),
        "sat_ratio_wheel": float(np.mean(tau_abs_ratio[:, 0] >= 0.98)),
        "sat_ratio_knee": float(np.mean(tau_abs_ratio[:, 1] >= 0.98)),
        "sat_ratio_hip": float(np.mean(tau_abs_ratio[:, 2] >= 0.98)),
        "jerk_proxy": _trapz(jerk_trans * jerk_trans + 0.1 * theta_acc * theta_acc, t),
        "torque_energy": _trapz(np.sum(tau * tau, axis=1), t),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--logs-dir", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()

    with args.metrics_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
        headers = list(rows[0].keys()) if rows else []

    for row in rows:
        log_path = args.logs_dir / f"{row['scenario']}__{row['controller']}.npz"
        derived = _derived_from_log(log_path)
        for key, value in derived.items():
            row[key] = f"{value:.6g}"

    out_headers = headers + [h for h in DERIVED_HEADERS if h not in headers]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_headers)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
