from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np

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
