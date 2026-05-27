from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Terrain:
    def height(self, x: float) -> float:
        raise NotImplementedError

    def slope(self, x: float) -> float:
        eps = 1e-4
        return (self.height(x + eps) - self.height(x - eps)) / (2.0 * eps)

    def curvature(self, x: float) -> float:
        eps = 1e-3
        return (self.height(x + eps) - 2.0 * self.height(x) + self.height(x - eps)) / (
            eps * eps
        )


@dataclass(frozen=True)
class FlatTerrain(Terrain):
    z: float = 0.0

    def height(self, x: float) -> float:
        return self.z

    def slope(self, x: float) -> float:
        return 0.0

    def curvature(self, x: float) -> float:
        return 0.0


@dataclass(frozen=True)
class SineTerrain(Terrain):
    amplitude: float = 0.05
    wavelength: float = 1.0

    def height(self, x: float) -> float:
        return float(self.amplitude * np.sin(2.0 * np.pi * x / self.wavelength))

    def slope(self, x: float) -> float:
        k = 2.0 * np.pi / self.wavelength
        return float(self.amplitude * k * np.cos(k * x))

    def curvature(self, x: float) -> float:
        k = 2.0 * np.pi / self.wavelength
        return float(-self.amplitude * k * k * np.sin(k * x))


@dataclass(frozen=True)
class NoiseTerrain(Terrain):
    amplitude: float = 0.04
    seed: int = 1
    spacing: float = 0.25

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        xs = np.arange(-20.0, 80.0 + self.spacing, self.spacing)
        values = rng.normal(0.0, self.amplitude, size=xs.shape)
        kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = kernel / kernel.sum()
        for _ in range(4):
            values = np.convolve(values, kernel, mode="same")
        object.__setattr__(self, "_xs", xs)
        object.__setattr__(self, "_values", values)

    def height(self, x: float) -> float:
        return float(np.interp(x, self._xs, self._values))

