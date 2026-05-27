from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationLog:
    rows: list[dict] = field(default_factory=list)
    failure_reason: str = ""

    def append(self, **kwargs) -> None:
        self.rows.append(kwargs)

    def to_arrays(self) -> dict[str, np.ndarray]:
        if not self.rows:
            return {}
        keys = self.rows[0].keys()
        return {key: np.array([row[key] for row in self.rows]) for key in keys}

    def save_npz(self, path: str) -> None:
        arrays = self.to_arrays()
        arrays["failure_reason"] = np.array(self.failure_reason)
        np.savez(path, **arrays)

