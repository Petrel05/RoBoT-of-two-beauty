from __future__ import annotations

import numpy as np

from src.config.params import RobotParams
from src.simulation.logger import SimulationLog


def _trapz(y: np.ndarray, t: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    return float(np.trapz(y, t))


def compute_metrics(log: SimulationLog, params: RobotParams) -> dict[str, float | str]:
    data = log.to_arrays()
    if not data:
        return {"success": 0.0, "failure_reason": "empty_log"}

    t = data["t"]
    duration = max(float(t[-1] - t[0]), 1e-9)
    y_err = data["y"] - data["y_ref"]
    theta = data["theta"]
    v_err = data["vx"] - data["v_cmd"]
    tau = np.vstack([data["tau_wheel"], data["tau_knee"], data["tau_hip"]]).T
    tau_abs_ratio = np.abs(tau) / params.tau_limits
    sat = tau_abs_ratio >= 0.98
    jerk_trans = np.gradient(data["ax"], t, edge_order=1) if t.size > 2 else np.zeros_like(t)
    theta_acc = data["alpha"]
    jerk_score = _trapz(jerk_trans * jerk_trans + 0.1 * theta_acc * theta_acc, t)

    return {
        "success": 1.0 if not log.failure_reason else 0.0,
        "failure_reason": log.failure_reason or "none",
        "duration": float(duration),
        "rmse_h": float(np.sqrt(np.mean(y_err * y_err))),
        "max_abs_h": float(np.max(np.abs(y_err))),
        "rmse_theta_deg": float(np.rad2deg(np.sqrt(np.mean(theta * theta)))),
        "max_abs_theta_deg": float(np.rad2deg(np.max(np.abs(theta)))),
        "rmse_v": float(np.sqrt(np.mean(v_err * v_err))),
        "jerk": float(jerk_score),
        "energy": _trapz(np.sum(tau * tau, axis=1), t),
        "sat_ratio": float(np.mean(sat)),
        "mean_tau_norm": float(np.mean(np.linalg.norm(tau, axis=1))),
    }


def format_metrics_table(rows: list[dict[str, float | str]]) -> str:
    headers = [
        "controller",
        "scenario",
        "success",
        "rmse_h",
        "max_abs_h",
        "rmse_theta_deg",
        "max_abs_theta_deg",
        "rmse_v",
        "sat_ratio",
        "failure_reason",
    ]
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                values.append(f"{v:.6g}")
            else:
                values.append(str(v))
        lines.append(",".join(values))
    return "\n".join(lines)

