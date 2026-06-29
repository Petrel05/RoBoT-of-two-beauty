from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.model.terrain import FlatTerrain, MultiSineTerrain, NoiseTerrain, SineTerrain, Terrain


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


def combined_force(*force_fns: ForceFn) -> ForceFn:
    def fn(t: float) -> float:
        return float(sum(force(t) for force in force_fns))

    return fn


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
        Scenario(
            name="D_large_irregular_terrain",
            duration=8.0,
            v_cmd=smooth_speed(2.0),
            force_x=no_force,
            terrain=MultiSineTerrain(
                amplitudes=np.array([0.040, 0.026, 0.018], dtype=float),
                wavelengths=np.array([1.60, 0.95, 0.55], dtype=float),
                phases=np.array([0.20, 1.70, 3.10], dtype=float),
                max_abs_height=0.085,
            ),
            description="2 m/s over larger non-single-period terrain, max height about 8.5 cm.",
        ),
        Scenario(
            name="E_combined_stress",
            duration=8.0,
            v_cmd=smooth_speed(3.0),
            force_x=combined_force(
                constant_force(60.0),
                pulse_force(start=4.0, duration=0.15, magnitude=100.0),
            ),
            terrain=MultiSineTerrain(
                amplitudes=np.array([0.045, 0.030, 0.020], dtype=float),
                wavelengths=np.array([1.35, 0.85, 0.50], dtype=float),
                phases=np.array([0.60, 2.10, 4.20], dtype=float),
                max_abs_height=0.090,
            ),
            description="3 m/s with 60 N constant push, 100 N impulse, and large irregular terrain.",
        ),
        Scenario(
            name="F_high_speed_flat",
            duration=8.0,
            v_cmd=smooth_speed(5.0),
            force_x=no_force,
            terrain=FlatTerrain(),
            description="Upper-bound 5 m/s tracking on flat ground.",
        ),
        Scenario(
            name="G_requirement_boundary",
            duration=8.0,
            v_cmd=smooth_speed(5.0),
            force_x=constant_force(100.0),
            terrain=SineTerrain(amplitude=0.05, wavelength=1.0),
            description="Requirement-boundary test: 5 m/s, 100 N push, and uneven terrain.",
        ),
    ]


def static_push_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="H0_static_impulse_push",
            duration=8.0,
            v_cmd=constant_speed(0.0),
            force_x=pulse_force(start=4.0, duration=0.1, magnitude=100.0),
            terrain=FlatTerrain(),
            description="Standing on flat ground with a 100 N, 0.1 s horizontal impulse.",
        ),
        Scenario(
            name="H1_static_constant_push",
            duration=8.0,
            v_cmd=constant_speed(0.0),
            force_x=constant_force(100.0),
            terrain=FlatTerrain(),
            description="Standing on flat ground with a persistent 100 N horizontal force.",
        ),
    ]


def easy_training_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="easy_stand",
            duration=8.0,
            v_cmd=constant_speed(0.0),
            force_x=no_force,
            terrain=FlatTerrain(),
            description="Standing on flat ground without external force.",
        ),
        Scenario(
            name="easy_slow",
            duration=8.0,
            v_cmd=smooth_speed(1.0),
            force_x=no_force,
            terrain=FlatTerrain(),
            description="Slow 1 m/s tracking on flat ground without external force.",
        ),
        Scenario(
            name="easy_cruise",
            duration=8.0,
            v_cmd=smooth_speed(2.0),
            force_x=no_force,
            terrain=FlatTerrain(),
            description="2 m/s tracking on flat ground without external force.",
        ),
    ]


def training_scenarios(kind: str) -> list[Scenario]:
    if kind == "easy":
        return easy_training_scenarios()
    if kind == "default":
        return default_scenarios()
    if kind == "all":
        return easy_training_scenarios() + default_scenarios()
    raise ValueError(f"Unknown scenario set: {kind}")
