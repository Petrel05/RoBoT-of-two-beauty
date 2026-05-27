from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.model.terrain import FlatTerrain, NoiseTerrain, SineTerrain, Terrain


ForceFn = Callable[[float], float]
CommandFn = Callable[[float], float]


@dataclass(frozen=True)
class Scenario:
    name: str
    duration: float
    v_cmd: CommandFn
    force_x: ForceFn
    terrain: Terrain
    description: str


def constant_speed(v: float) -> CommandFn:
    return lambda _t: float(v)


def smooth_speed(v: float, ramp_time: float = 1.0) -> CommandFn:
    def fn(t: float) -> float:
        s = np.clip(t / ramp_time, 0.0, 1.0)
        s = s * s * (3.0 - 2.0 * s)
        return float(v * s)

    return fn


def no_force(_t: float) -> float:
    return 0.0


def pulse_force(start: float, duration: float, magnitude: float) -> ForceFn:
    def fn(t: float) -> float:
        return float(magnitude if start <= t <= start + duration else 0.0)

    return fn


def constant_force(magnitude: float) -> ForceFn:
    return lambda _t: float(magnitude)


def default_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="A_impulse_push",
            duration=8.0,
            v_cmd=smooth_speed(2.0),
            force_x=pulse_force(start=5.0, duration=0.1, magnitude=100.0),
            terrain=FlatTerrain(),
            description="2 m/s flat ground with a 100 N, 0.1 s horizontal impulse.",
        ),
        Scenario(
            name="B_constant_push",
            duration=8.0,
            v_cmd=smooth_speed(3.0),
            force_x=constant_force(80.0),
            terrain=FlatTerrain(),
            description="3 m/s flat ground with a persistent 80 N horizontal force.",
        ),
        Scenario(
            name="C_rough_terrain",
            duration=8.0,
            v_cmd=smooth_speed(2.0),
            force_x=no_force,
            terrain=SineTerrain(amplitude=0.05, wavelength=1.0),
            description="2 m/s over a 5 cm amplitude, 1 m wavelength sine terrain.",
        ),
        Scenario(
            name="C_noise_terrain",
            duration=8.0,
            v_cmd=smooth_speed(2.0),
            force_x=no_force,
            terrain=NoiseTerrain(amplitude=0.04, seed=7),
            description="2 m/s over smooth pseudo-random rough terrain.",
        ),
    ]

