from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config.scenarios import (
    ForceFn,
    Scenario,
    constant_speed,
    no_force,
    smooth_speed,
)
from src.model.terrain import FlatTerrain, MultiSineTerrain, NoiseTerrain, SineTerrain


class ScenarioSampler:
    def sample(self, rng: np.random.Generator) -> Scenario:
        raise NotImplementedError


def smooth_impulse_force(
    start: float, duration: float, magnitude: float
) -> ForceFn:
    def fn(t: float) -> float:
        if t < start or t > start + duration:
            return 0.0
        phase = (t - start) / duration
        return float(magnitude * 0.5 * (1.0 - np.cos(2.0 * np.pi * phase)))

    return fn


def smooth_constant_force(start: float, magnitude: float, tau: float) -> ForceFn:
    def fn(t: float) -> float:
        return float(magnitude * 0.5 * (1.0 + np.tanh((t - start) / tau)))

    return fn


def random_force(rng: np.random.Generator, level: str) -> ForceFn:
    if level == "none":
        return no_force
    force_type = rng.choice(["none", "impulse", "constant"], p=[0.25, 0.45, 0.30])
    if force_type == "none":
        return no_force
    if force_type == "impulse":
        return smooth_impulse_force(
            start=float(rng.uniform(1.0, 6.0)),
            duration=float(rng.uniform(0.15, 1.0)),
            magnitude=float(rng.uniform(-100.0, 100.0)),
        )
    return smooth_constant_force(
        start=float(rng.uniform(0.5, 2.5)),
        magnitude=float(rng.uniform(-80.0, 80.0)),
        tau=float(rng.uniform(0.2, 0.6)),
    )


def random_terrain(rng: np.random.Generator, level: str):
    if level == "flat":
        return FlatTerrain()
    terrain_type = rng.choice(["flat", "sine", "multi_sine", "noise"], p=[0.20, 0.35, 0.25, 0.20])
    if terrain_type == "flat":
        return FlatTerrain()
    if terrain_type == "sine":
        return SineTerrain(
            amplitude=float(rng.uniform(0.0, 0.06)),
            wavelength=float(rng.uniform(0.8, 2.5)),
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if terrain_type == "multi_sine":
        n = int(rng.integers(2, 5))
        return MultiSineTerrain(
            amplitudes=rng.uniform(0.004, 0.025, size=n),
            wavelengths=rng.uniform(0.8, 3.0, size=n),
            phases=rng.uniform(0.0, 2.0 * np.pi, size=n),
            max_abs_height=0.06,
        )
    return NoiseTerrain(
        amplitude=float(rng.uniform(0.015, 0.05)),
        seed=int(rng.integers(0, 1_000_000)),
        spacing=float(rng.uniform(0.3, 0.7)),
    )


@dataclass(frozen=True)
class RandomScenarioSampler(ScenarioSampler):
    name: str
    speed_range: tuple[float, float]
    terrain_level: str
    force_level: str
    duration: float = 8.0

    def sample(self, rng: np.random.Generator) -> Scenario:
        v = float(rng.uniform(*self.speed_range))
        if v < 1e-6:
            v_cmd = constant_speed(0.0)
        else:
            v_cmd = smooth_speed(v, ramp_time=float(rng.uniform(0.6, 1.5)))
        return Scenario(
            name=f"{self.name}_sample",
            duration=self.duration,
            v_cmd=v_cmd,
            force_x=random_force(rng, self.force_level),
            terrain=random_terrain(rng, self.terrain_level),
            description="Random smooth training scenario.",
        )


def training_sampler(kind: str) -> ScenarioSampler | None:
    if kind == "random_easy":
        return RandomScenarioSampler(
            name="random_easy",
            speed_range=(0.0, 2.0),
            terrain_level="flat",
            force_level="none",
        )
    if kind == "random_force":
        return RandomScenarioSampler(
            name="random_force",
            speed_range=(0.0, 3.0),
            terrain_level="flat",
            force_level="smooth",
        )
    if kind == "random_full":
        return RandomScenarioSampler(
            name="random_full",
            speed_range=(0.0, 5.0),
            terrain_level="smooth",
            force_level="smooth",
        )
    return None
