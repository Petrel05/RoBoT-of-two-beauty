from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np

from src.config.params import RobotParams, SafetyLimits
from src.simulation.logger import SimulationLog


def plot_log(log: SimulationLog, title: str, out_path: str | Path) -> None:
    data = log.to_arrays()
    if not data:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = data["t"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(title)

    axes[0].plot(t, data["y"], label="y_hip")
    axes[0].plot(t, data["y_ref"], "--", label="y_ref")
    axes[0].plot(t, data["wheel_y"], ":", label="wheel_y")
    axes[0].set_ylabel("height (m)")
    axes[0].legend(loc="best")

    axes[1].plot(t, np.rad2deg(data["theta"]), label="theta")
    axes[1].set_ylabel("theta (deg)")
    axes[1].legend(loc="best")

    axes[2].plot(t, data["vx"], label="vx")
    axes[2].plot(t, data["v_cmd"], "--", label="v_cmd")
    axes[2].set_ylabel("speed (m/s)")
    axes[2].legend(loc="best")

    axes[3].plot(t, data["tau_wheel"], label="wheel")
    axes[3].plot(t, data["tau_knee"], label="knee")
    axes[3].plot(t, data["tau_hip"], label="hip")
    axes[3].set_ylabel("torque (Nm)")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="best")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_wbc_diagnostics(
    log: SimulationLog,
    params: RobotParams,
    safety: SafetyLimits,
    title: str,
    out_path: str | Path,
) -> None:
    data = log.to_arrays()
    if not data or "qp_success" not in data:
        return
    valid = np.isfinite(data["qp_success"])
    if not np.any(valid):
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = data["t"]
    friction_limit = params.friction_coeff * data["contact_normal"]
    max_leg_length = params.thigh_length + params.shank_length

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(title)

    axes[0].plot(t, data["contact_normal"], label="normal")
    axes[0].plot(t, data["contact_tangent"], label="tangent")
    axes[0].plot(t, friction_limit, "--", label="+mu normal")
    axes[0].plot(t, -friction_limit, "--", label="-mu normal")
    axes[0].set_ylabel("contact force (N)")
    axes[0].legend(loc="best")

    axes[1].plot(t, data["leg_length"], label="actual leg length")
    axes[1].plot(t, data["qp_predicted_leg_length"], "--", label="predicted leg length")
    axes[1].axhline(safety.min_leg_length, color="tab:red", linestyle=":", label="minimum")
    axes[1].axhline(max_leg_length, color="tab:gray", linestyle=":", label="maximum")
    axes[1].set_ylabel("leg length (m)")
    axes[1].legend(loc="best")

    axes[2].plot(t, data["qp_eq_residual"], label="equality residual")
    axes[2].plot(t, data["qp_ineq_violation"], label="inequality violation")
    axes[2].plot(t, data["qp_leg_slack"], label="leg slack")
    axes[2].plot(t, data["qp_height_slack"], label="height slack")
    axes[2].set_yscale("symlog", linthresh=1e-10)
    axes[2].set_ylabel("QP residual")
    axes[2].legend(loc="best")

    axes[3].plot(t, data["qp_friction_ratio"], label="friction utilization")
    axes[3].plot(t, data["qp_success"], label="QP success")
    axes[3].plot(t, data["qp_solver_success"], label="solver converged")
    axes[3].plot(t, data["qp_fallback"], label="fallback")
    axes[3].axhline(1.0, color="tab:red", linestyle=":", label="friction boundary")
    axes[3].set_ylabel("ratio / flag")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="best")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
