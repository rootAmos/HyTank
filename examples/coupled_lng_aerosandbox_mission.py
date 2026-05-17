"""
Closed-loop AeroSandbox mission and LNG tank example.

This example uses AeroSandbox's 2D point-mass speed/gamma dynamics stack with
three mission phases: climb, cruise, and descent. The LNG tank is solved inside
the same AeroSandbox optimization problem. Atmospheric temperature from the
flight trajectory drives the tank heat leak, engine fuel flow draws liquid LNG
from the tank, and the remaining tank fluid mass feeds back into aircraft
weight.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import numpy as onp

from lngtank.aerosandbox_tank import (
    MissionInputs,
    TankDesign,
    initial_gas_density_from_pressure,
    liquid_volume_from_height_fraction,
    tank_rhs,
    tank_volume,
)
from lngtank.asb_properties_interpolants import CoolPropGridInterpolants


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
DEFAULT_SCHEDULE_PATH = Path(__file__).resolve().parent / "mission_debug_stage_07_subsonic_single_climb.json"
G = 9.80665
M_TO_FT = 3.280839895
MPS_TO_KT = 1.943844492
M_TO_NMI = 1 / 1852.0
KG_TO_LBM = 2.2046226218
KGPS_TO_LBHR = KG_TO_LBM * 3600.0
PA_TO_PSIA = 1 / 6894.757293
K_TO_R = 1.8


@dataclass(frozen=True)
class AircraftModel:
    dry_mass: float = 3500.0  # kg, excluding LNG fluid
    tank_hardware_mass: float = 450.0  # kg
    wing_area: float = 45.0  # m^2
    aspect_ratio: float = 9.0
    oswald_efficiency: float = 0.82
    cd0: float = 0.028
    max_sea_level_shaft_power: float = 1.2e6  # W
    power_lapse_exponent: float = 0.8
    propeller_efficiency: float = 0.85
    psfc: float = 7.6e-8  # kg / W / s, roughly 0.45 lb / hp / hr


def load_mission(path=DEFAULT_SCHEDULE_PATH):
    # Normalizes SI schedules and human-readable reference schedules into one solver input shape.
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    solver = data.get("solver", {})
    tank = data.get("tank", {})
    aircraft = data.get("aircraft_model", {})
    tank_design = data.get("tank_design", {})
    settings = {
        "max_accel_m_s2": float(solver.get("max_accel_m_s2", 2.0)),
        "max_gamma_rate_rad_s": onp.radians(float(solver.get("max_gamma_rate_deg_s", 0.05))),
        "constrain_speed_rate": bool(solver.get("constrain_speed_rate", True)),
        "constrain_gamma_rate": bool(solver.get("constrain_gamma_rate", True)),
        "min_lift_coefficient": float(solver.get("min_lift_coefficient", 0.05)),
        "max_lift_coefficient": float(solver.get("max_lift_coefficient", 1.4)),
        "min_power_fraction": float(solver.get("min_power_fraction", 0.05)),
        "max_throttle_rate_per_s": (
            float(solver["max_throttle_rate_per_s"]) if solver.get("max_throttle_rate_per_s") is not None else None
        ),
        "lift_coefficient_step_limit": float(solver.get("lift_coefficient_step_limit", 0.12)),
        "p_heater_w": float(tank.get("p_heater_w", 1000.0)),
        "initial_fill": float(tank.get("initial_fill", 0.82)),
        "initial_pressure_pa": float(tank.get("initial_pressure_pa", 1.064e6)),
        "initial_gas_temperature_k": float(tank.get("initial_gas_temperature_k", 151.8)),
        "initial_liquid_temperature_k": float(tank.get("initial_liquid_temperature_k", 145.8)),
        "aircraft_model": aircraft,
        "tank_design": tank_design,
    }

    segments = []
    if "range_m" in data:
        for segment in data["segments"]:
            constraints = segment.get("constraints", {})
            minimum = dict(constraints.get("min", constraints.get("minimum", {})))
            if "power_fraction" in minimum:
                minimum["power_fraction"] = settings["min_power_fraction"]
            segments.append(
                {
                    "name": segment["name"],
                    "type": segment.get("segment_type", segment.get("type", "segment")),
                    "end": dict(segment["end"]),
                    "fix": dict(constraints.get("fix", {})),
                    "min": minimum,
                    "max": dict(constraints.get("max", constraints.get("maximum", {}))),
                }
            )

        num_nodes = data.get("num_nodes")
        if num_nodes is None and all("num_nodes" in segment["end"] for segment in segments):
            num_nodes = 1 + sum(segment["end"]["num_nodes"] - 1 for segment in segments)
        duration_s = data.get("duration_s")
        duration_guess_s = data.get("duration_guess_s", duration_s)
        if duration_guess_s is None:
            raise ValueError("Mission must define either duration_s or duration_guess_s.")
        range_m = data.get("range_m")
        range_guess_m = data.get("range_guess_m", range_m)
        if range_guess_m is None:
            raise ValueError("Mission must define either range_m or range_guess_m.")
        initial = data.get("initial", {})
        final = data.get("final", {})
        return {
            "name": data.get("mission_name", Path(path).stem),
            "duration_s": float(duration_s) if duration_s is not None else None,
            "duration_guess_s": float(duration_guess_s),
            "duration_bounds_s": tuple(data["duration_bounds_s"]) if data.get("duration_bounds_s") else None,
            "range_m": float(range_m) if range_m is not None else None,
            "range_guess_m": float(range_guess_m),
            "num_nodes": int(num_nodes or 61),
            "initial_altitude_m": float(initial.get("altitude_m", 0.0)),
            "initial_speed_m_s": float(initial["speed_m_s"]),
            "final_altitude_m": float(final.get("altitude_m", 0.0)),
            "final_speed_m_s": float(final["speed_m_s"]) if "speed_m_s" in final else None,
            "segments": segments,
            **settings,
        }

    summary = data["summary"]
    for segment in data["segments"]:
        end = {
            "condition": "time",
            "node_fraction": float(segment["end_time_min"]) / summary["duration_min"],
            "altitude_m": segment["end_altitude_ft"] / M_TO_FT,
        }
        if "end_distance_nmi" in segment:
            end["distance_m"] = segment["end_distance_nmi"] / M_TO_NMI
        if "end_speed_kt" in segment:
            end["speed_m_s"] = segment["end_speed_kt"] / MPS_TO_KT
        if "end_gamma_deg" in segment:
            end["gamma_deg"] = float(segment["end_gamma_deg"])

        segments.append(
            {
                "name": segment["name"],
                "type": segment.get("segment_type", "segment"),
                "end": end,
                "fix": {"power_fraction": float(segment["power_fraction"])} if "power_fraction" in segment else {},
                "min": {
                    **({"power_fraction": float(segment["power_min_fraction"])} if "power_min_fraction" in segment else {}),
                    **({"speed_m_s": segment["speed_min_kt"] / MPS_TO_KT} if "speed_min_kt" in segment else {}),
                },
                "max": {
                    **({"power_fraction": float(segment["power_max_fraction"])} if "power_max_fraction" in segment else {}),
                    **({"speed_m_s": segment["speed_max_kt"] / MPS_TO_KT} if "speed_max_kt" in segment else {}),
                },
            }
        )

    return {
        "name": data.get("mission_name", Path(path).stem),
        "duration_s": 60.0 * summary["duration_min"],
        "duration_guess_s": 60.0 * summary["duration_min"],
        "duration_bounds_s": None,
        "range_m": summary["total_distance_nmi"] / M_TO_NMI,
        "range_guess_m": summary["total_distance_nmi"] / M_TO_NMI,
        "num_nodes": 61,
        "initial_altitude_m": 0.0,
        "initial_speed_m_s": summary["initial_speed_kt"] / MPS_TO_KT,
        "final_altitude_m": 0.0,
        "final_speed_m_s": summary["final_speed_kt"] / MPS_TO_KT,
        "segments": segments,
        **settings,
    }


def interpolate_schedule_guess(schedule, key, default_end_value):
    # Builds initial guesses for distance, altitude, and speed from the same schedule endpoints.
    indices = [0]
    values = [
        {
            "altitude_m": schedule["initial_altitude_m"],
            "speed_m_s": schedule["initial_speed_m_s"],
            "distance_m": 0.0,
        }[key]
    ]
    for segment, index in zip(
        schedule["segments"],
        schedule_end_node_indices(schedule, onp.linspace(0.0, schedule["duration_guess_s"], schedule["num_nodes"])),
    ):
        value = segment["end"].get(key)
        if value is not None:
            indices.append(index)
            values.append(value)
    if indices[-1] != schedule["num_nodes"] - 1:
        indices.append(schedule["num_nodes"] - 1)
        values.append(default_end_value)
    return onp.interp(onp.arange(schedule["num_nodes"]), indices, values)


def schedule_end_node_indices(schedule, time):
    # Maps flight-segment endpoints to solver node numbers for guesses, constraints, and plot markers.
    end_indices = []
    last_index = 0
    elapsed = 0.0
    for i, segment in enumerate(schedule["segments"]):
        end = segment["end"]
        if end.get("num_nodes") is not None:
            index = last_index + int(end["num_nodes"]) - 1
        elif end.get("node_fraction") is not None:
            index = int(round(end["node_fraction"] * (schedule["num_nodes"] - 1)))
        elif end.get("condition") == "duration":
            elapsed += float(end["duration_s"])
            index = int(onp.searchsorted(time, elapsed, side="left"))
        elif end.get("condition") == "distance" and end.get("distance_m") is not None:
            if schedule["range_m"] is None:
                raise ValueError("Distance-based segment endpoints require fixed range_m.")
            index = int(round(end["distance_m"] / schedule["range_m"] * (schedule["num_nodes"] - 1)))
        else:
            remaining_segments = len(schedule["segments"]) - i
            remaining_nodes = schedule["num_nodes"] - 1 - last_index
            index = last_index + max(1, int(round(remaining_nodes / remaining_segments)))

        if i == len(schedule["segments"]) - 1:
            index = schedule["num_nodes"] - 1
        index = min(max(index, last_index + 1), schedule["num_nodes"] - 1)
        end_indices.append(index)
        last_index = index
    return end_indices


def apply_schedule_constraints(opti, dyn, throttle, schedule, time, rho=None, rho0=None, dyn_derivatives=None):
    # Applies all schedule endpoint, fixed, and min/max band constraints in one solver-facing pass.
    end_indices = schedule_end_node_indices(schedule, time)
    opti.subject_to(
        [
            dyn.altitude[0] == schedule["initial_altitude_m"],
            dyn.speed[0] == schedule["initial_speed_m_s"],
            dyn.altitude[-1] == schedule["final_altitude_m"],
        ]
    )
    if schedule["range_m"] is not None:
        opti.subject_to(dyn.x_e[-1] == schedule["range_m"])
    if schedule["final_speed_m_s"] is not None:
        opti.subject_to(dyn.speed[-1] == schedule["final_speed_m_s"])

    start = 0
    for segment, end_index in zip(schedule["segments"], end_indices):
        end = segment["end"]
        endpoint = end_index
        if end.get("altitude_m") is not None:
            opti.subject_to(dyn.altitude[endpoint] == end["altitude_m"])
        if end.get("distance_m") is not None:
            opti.subject_to(dyn.x_e[endpoint] == end["distance_m"])
        if end.get("speed_m_s") is not None:
            opti.subject_to(dyn.speed[endpoint] == end["speed_m_s"])
        if end.get("gamma_deg") is not None:
            opti.subject_to(dyn.gamma[endpoint] == onp.radians(end["gamma_deg"]))

        for rule, op in (("fix", "=="), ("min", ">="), ("max", "<=")):
            for key, value in segment[rule].items():
                if key in {
                    "gamma_rate_deg_s",
                    "climb_angle_rate_deg_s",
                    "flight_path_angle_rate_deg_s",
                    "gamma_rate_rad_s",
                    "climb_angle_rate_rad_s",
                    "flight_path_angle_rate_rad_s",
                    "speed_rate_m_s2",
                    "speed_rate",
                    "accel_m_s2",
                    "acceleration_m_s2",
                    "altitude_rate_m_s",
                    "altitude_rate",
                    "climb_rate_m_s",
                }:
                    first_index = start
                else:
                    first_index = start if start == 0 and key in {"power_fraction", "throttle"} else start + 1
                if first_index > end_index:
                    continue
                last_index = end_index + 1
                if (
                    rule in {"min", "max"}
                    and key in {"gamma_deg", "climb_angle_deg", "flight_path_angle_deg"}
                    and end.get("gamma_deg") is not None
                ):
                    last_index = end_index
                if first_index >= last_index:
                    continue
                selection = slice(first_index, last_index)
                if key in {"speed_m_s", "speed"}:
                    expr = dyn.speed[selection]
                elif key in {"equivalent_speed_m_s", "eas_m_s", "eas"}:
                    if rho is None or rho0 is None:
                        raise ValueError("Equivalent airspeed constraints require atmosphere density.")
                    expr = dyn.speed[selection] * (rho[selection] / rho0) ** 0.5
                elif key in {"altitude_m", "altitude"}:
                    expr = dyn.altitude[selection]
                elif key in {"distance_m", "range_m", "x_m", "x_e"}:
                    expr = dyn.x_e[selection]
                elif key in {"power_fraction", "throttle"}:
                    expr = throttle[selection]
                elif key in {"gamma_deg", "climb_angle_deg", "flight_path_angle_deg"}:
                    expr = dyn.gamma[selection]
                    value = onp.radians(value)
                elif key in {"gamma_rate_deg_s", "climb_angle_rate_deg_s", "flight_path_angle_rate_deg_s"}:
                    if dyn_derivatives is None:
                        raise ValueError("Gamma-rate constraints require dynamics derivatives.")
                    expr = dyn_derivatives["gamma"][selection]
                    value = onp.radians(value)
                elif key in {"gamma_rate_rad_s", "climb_angle_rate_rad_s", "flight_path_angle_rate_rad_s"}:
                    if dyn_derivatives is None:
                        raise ValueError("Gamma-rate constraints require dynamics derivatives.")
                    expr = dyn_derivatives["gamma"][selection]
                elif key in {"speed_rate_m_s2", "speed_rate", "accel_m_s2", "acceleration_m_s2"}:
                    if dyn_derivatives is None:
                        raise ValueError("Speed-rate constraints require dynamics derivatives.")
                    expr = dyn_derivatives["speed"][selection]
                elif key in {"altitude_rate_m_s", "altitude_rate", "climb_rate_m_s"}:
                    if dyn_derivatives is None:
                        raise ValueError("Altitude-rate constraints require dynamics derivatives.")
                    expr = -dyn_derivatives["z_e"][selection]
                else:
                    raise ValueError(f"Unsupported segment constraint key '{key}'.")

                if op == "==":
                    opti.subject_to(expr == value)
                elif op == ">=":
                    opti.subject_to(expr >= value)
                else:
                    opti.subject_to(expr <= value)

        start = end_index
    return end_indices


def build_coupled_problem(
    aircraft: AircraftModel = AircraftModel(),
    tank_design: TankDesign = TankDesign(radius=1.15, length=2.5, n_layers=20.0, heat_multiplier=2.0),
    schedule=None,
):
    # Builds the coupled AeroSandbox aircraft and LNG tank optimization problem.
    schedule = load_mission(DEFAULT_SCHEDULE_PATH) if schedule is None else schedule
    aircraft = AircraftModel(**{**aircraft.__dict__, **schedule["aircraft_model"]})
    tank_design = TankDesign(**{**tank_design.__dict__, **schedule["tank_design"]})

    props = CoolPropGridInterpolants()
    opti = asb.Opti()

    n = schedule["num_nodes"]
    duration_guess = schedule["duration_guess_s"]
    range_guess_m = schedule["range_guess_m"]
    max_altitude = max(segment["end"].get("altitude_m", 0.0) for segment in schedule["segments"])
    if schedule["duration_s"] is None:
        duration_lower, duration_upper = schedule["duration_bounds_s"] or (0.5 * duration_guess, 2.0 * duration_guess)
        duration = opti.variable(
            init_guess=duration_guess,
            lower_bound=duration_lower,
            upper_bound=duration_upper,
            scale=duration_guess,
        )
    else:
        duration = duration_guess
    tau = onp.linspace(0.0, 1.0, n)
    time = tau * duration
    time_guess = tau * duration_guess

    volume = float(tank_volume(tank_design.radius, tank_design.length))
    v_gas0 = volume * (1 - schedule["initial_fill"])
    m_liq0 = (volume - v_gas0) * float(props.liquid_density(schedule["initial_liquid_temperature_k"]))
    m_gas0 = (
        initial_gas_density_from_pressure(
            props,
            schedule["initial_pressure_pa"],
            schedule["initial_gas_temperature_k"],
        )
        * v_gas0
    )

    x_guess = interpolate_schedule_guess(schedule, "distance_m", range_guess_m)
    z_guess = interpolate_schedule_guess(schedule, "altitude_m", schedule["final_altitude_m"])
    v_guess = interpolate_schedule_guess(
        schedule,
        "speed_m_s",
        schedule["final_speed_m_s"] if schedule["final_speed_m_s"] is not None else schedule["initial_speed_m_s"],
    )
    gamma_guess = onp.arctan2(onp.gradient(z_guess, time_guess), onp.maximum(v_guess, 1.0))
    fuel_guess = onp.linspace(
        0.0,
        min(0.8 * m_liq0, aircraft.psfc * aircraft.max_sea_level_shaft_power * 0.55 * duration_guess),
        n,
    )

    x = opti.variable(init_guess=x_guess, n_vars=n, lower_bound=0.0, scale=range_guess_m)
    z_e = opti.variable(
        init_guess=-z_guess,
        n_vars=n,
        lower_bound=-max_altitude,
        upper_bound=0.0,
        scale=max(max_altitude, 1.0),
    )
    schedule_speeds = [schedule["initial_speed_m_s"]]
    if schedule["final_speed_m_s"] is not None:
        schedule_speeds.append(schedule["final_speed_m_s"])
    for segment in schedule["segments"]:
        if segment["end"].get("speed_m_s") is not None:
            schedule_speeds.append(segment["end"]["speed_m_s"])
        for rules in (segment["fix"], segment["min"], segment["max"]):
            schedule_speeds.extend(
                float(value)
                for key, value in rules.items()
                if key in {"speed_m_s", "speed", "equivalent_speed_m_s", "eas_m_s", "eas"}
            )
    speed_upper_bound = max([125.0] + [1.1 * speed for speed in schedule_speeds])
    v = opti.variable(
        init_guess=v_guess,
        n_vars=n,
        lower_bound=45.0,
        upper_bound=speed_upper_bound,
        scale=max(schedule["initial_speed_m_s"], speed_upper_bound / 2),
    )
    gamma = opti.variable(init_guess=gamma_guess, n_vars=n, lower_bound=-0.12, upper_bound=0.12, scale=0.05)

    throttle = opti.variable(init_guess=0.55, n_vars=n, lower_bound=schedule["min_power_fraction"], upper_bound=1.0)
    throttle_rate = opti.variable(init_guess=0.0, n_vars=n, scale=0.01)
    cl = opti.variable(
        init_guess=0.65,
        n_vars=n,
        lower_bound=schedule["min_lift_coefficient"],
        upper_bound=schedule["max_lift_coefficient"],
    )

    m_gas = opti.variable(init_guess=m_gas0, n_vars=n, lower_bound=1e-3, scale=max(m_gas0, 1.0))
    m_liq = opti.variable(
        init_guess=m_liq0 - fuel_guess,
        n_vars=n,
        lower_bound=100.0,
        scale=max(m_liq0, 1.0),
    )
    t_gas = opti.variable(
        init_guess=schedule["initial_gas_temperature_k"],
        n_vars=n,
        lower_bound=92.0,
        upper_bound=230.0,
        scale=150.0,
    )
    t_liq = opti.variable(
        init_guess=schedule["initial_liquid_temperature_k"],
        n_vars=n,
        lower_bound=90.0,
        upper_bound=190.0,
        scale=150.0,
    )
    v_gas = opti.variable(init_guess=v_gas0, n_vars=n, lower_bound=1e-4, upper_bound=0.98 * volume, scale=volume)
    q_add = opti.variable(init_guess=0.0, n_vars=n, lower_bound=0.0, scale=schedule["p_heater_w"])
    h_liq_frac = opti.variable(init_guess=schedule["initial_fill"], n_vars=n, lower_bound=1e-3, upper_bound=1 - 1e-3)

    mass = aircraft.dry_mass + aircraft.tank_hardware_mass + m_gas + m_liq
    dyn = asb.DynamicsPointMass2DSpeedGamma(
        mass_props=asb.MassProperties(mass=mass),
        x_e=x,
        z_e=z_e,
        speed=v,
        gamma=gamma,
    )

    opti.subject_to(
        [
            m_gas[0] == m_gas0,
            m_liq[0] == m_liq0,
            t_gas[0] == schedule["initial_gas_temperature_k"],
            t_liq[0] == schedule["initial_liquid_temperature_k"],
            v_gas[0] == v_gas0,
            q_add[0] == 0.0,
        ]
    )

    rho0 = float(asb.Atmosphere(altitude=0.0).density())
    rho = dyn.op_point.atmosphere.density()
    t_env = dyn.op_point.atmosphere.temperature()
    density_ratio = rho / rho0

    q_dyn = dyn.op_point.dynamic_pressure()
    induced_factor = 1 / (np.pi * aircraft.aspect_ratio * aircraft.oswald_efficiency)
    cd0 = min(aircraft.cd0, 0.03)
    cd = cd0 + induced_factor * cl**2
    lift = q_dyn * aircraft.wing_area * cl
    drag = q_dyn * aircraft.wing_area * cd
    shaft_power = throttle * aircraft.max_sea_level_shaft_power * density_ratio**aircraft.power_lapse_exponent
    thrust = aircraft.propeller_efficiency * shaft_power / dyn.speed
    m_dot_fuel = aircraft.psfc * shaft_power

    dyn.add_gravity_force(g=G)
    dyn.add_force(Fx=thrust - drag, Fz=-lift, axes="wind")
    dyn_derivatives = dyn.state_derivatives()

    segment_indices = apply_schedule_constraints(
        opti, dyn, throttle, schedule, time, rho=rho, rho0=rho0, dyn_derivatives=dyn_derivatives
    )
    segment_labels = [
        f"climb {i + 1}" if segment["type"].replace("_", " ").strip() == "climb"
        else segment["type"].replace("_", " ").strip() or f"segment {i + 1}"
        for i, segment in enumerate(schedule["segments"])
    ]
    opti.subject_to(dyn.gamma[0] == 0.0)
    if schedule["constrain_speed_rate"]:
        opti.subject_to(dyn_derivatives["speed"] <= schedule["max_accel_m_s2"])
        opti.subject_to(dyn_derivatives["speed"] >= -schedule["max_accel_m_s2"])
    if schedule["constrain_gamma_rate"]:
        opti.subject_to(dyn_derivatives["gamma"] <= schedule["max_gamma_rate_rad_s"])
        opti.subject_to(dyn_derivatives["gamma"] >= -schedule["max_gamma_rate_rad_s"])
    dyn.constrain_derivatives(opti, time)

    tank_states = [m_gas, m_liq, t_gas, t_liq, v_gas, q_add]
    tank_inputs = MissionInputs(
        duration=duration_guess,
        t_env=t_env,
        p_heater=schedule["p_heater_w"],
        m_dot_liq_out=m_dot_fuel,
        m_dot_gas_out=0.0,
    )
    tank_rhs_values, tank_aux = tank_rhs(
        tank_states,
        tank_design,
        tank_inputs,
        props,
        h_liq_frac=h_liq_frac,
    )
    for state, derivative in zip(tank_states, tank_rhs_values):
        opti.constrain_derivative(
            derivative=derivative,
            variable=state,
            with_respect_to=time,
            method="trapezoidal",
        )

    opti.subject_to(tank_aux["pressure"] <= schedule["initial_pressure_pa"])
    opti.subject_to(tank_aux["pressure"] >= 2.0e5)
    opti.subject_to(tank_aux["fill_level"] >= 0.05)
    opti.subject_to(tank_aux["fill_level"] <= 0.95)
    opti.subject_to(
        liquid_volume_from_height_fraction(tank_design.radius, tank_design.length, h_liq_frac)
        == volume * tank_aux["fill_level"]
    )
    aux = {
        "pressure": tank_aux["pressure"],
        "fill_level": tank_aux["fill_level"],
        "m_dot_fuel": m_dot_fuel,
        "mass": mass,
        "q_gas": tank_aux["q_gas"],
        "q_liq": tank_aux["q_liq"],
        "t_env": t_env,
        "rho": rho,
        "rho0": rho0,
        "accel": dyn_derivatives["speed"],
        "gamma_rate": dyn_derivatives["gamma"],
        "throttle_rate": throttle_rate,
    }

    opti.constrain_derivative(
        derivative=throttle_rate,
        variable=throttle,
        with_respect_to=time,
        method="trapezoidal",
    )
    if schedule["max_throttle_rate_per_s"] is not None:
        opti.subject_to(throttle_rate <= schedule["max_throttle_rate_per_s"])
        opti.subject_to(throttle_rate >= -schedule["max_throttle_rate_per_s"])
    opti.subject_to(cl[1:] - cl[:-1] <= schedule["lift_coefficient_step_limit"])
    opti.subject_to(cl[1:] - cl[:-1] >= -schedule["lift_coefficient_step_limit"])

    opti.minimize(
        1000.0 * (m_liq[0] - m_liq[-1])
        + 5000.0 * np.sum(throttle_rate**2)
        + 100.0 * np.sum((cl[1:] - cl[:-1]) ** 2)
        + 5000.0 * np.sum(dyn_derivatives["speed"] ** 2)
        + 1000.0 * np.sum(dyn_derivatives["gamma"] ** 2)
    )

    return {
        "opti": opti,
        "time": time,
        "duration": duration,
        "mission_name": schedule["name"],
        "segment_indices": segment_indices,
        "segment_labels": segment_labels,
        "states": {
            "x": dyn.x_e,
            "altitude": dyn.altitude,
            "V": dyn.speed,
            "gamma": dyn.gamma,
            "m_gas": m_gas,
            "m_liq": m_liq,
            "T_gas": t_gas,
            "T_liq": t_liq,
            "V_gas": v_gas,
            "Q_add": q_add,
            "throttle": throttle,
            "CL": cl,
        },
        "aux": aux,
    }


def plot_solution(problem, sol, output_path):
    # Converts the solved trajectory into engineering units, plots it, and returns summary arrays.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        time = onp.array(sol.value(problem["time"]), dtype=float)
    except Exception:
        time = onp.array(problem["time"], dtype=float)
    time_min = time / 60
    states = problem["states"]
    aux = problem["aux"]
    segment_indices = problem.get("segment_indices", [])
    segment_labels = problem.get("segment_labels", [])

    values = {
        "altitude_ft": onp.array(sol.value(states["altitude"])) * M_TO_FT,
        "range_nmi": onp.array(sol.value(states["x"])) * M_TO_NMI,
        "speed_kt": onp.array(sol.value(states["V"])) * MPS_TO_KT,
        "gamma_deg": onp.degrees(onp.array(sol.value(states["gamma"]))),
        "throttle": onp.array(sol.value(states["throttle"])),
        "mass_lbm": onp.array(sol.value(aux["mass"])) * KG_TO_LBM,
        "fuel_flow_lb_hr": onp.array(sol.value(aux["m_dot_fuel"])) * KGPS_TO_LBHR,
        "pressure_psia": onp.array(sol.value(aux["pressure"])) * PA_TO_PSIA,
        "fill_level": onp.array(sol.value(aux["fill_level"])),
        "t_gas_R": onp.array(sol.value(states["T_gas"])) * K_TO_R,
        "t_liq_R": onp.array(sol.value(states["T_liq"])) * K_TO_R,
        "t_env_R": onp.array(sol.value(aux["t_env"])) * K_TO_R,
    }
    values["equivalent_speed_kt"] = values["speed_kt"] * onp.sqrt(
        onp.array(sol.value(aux["rho"])) / float(aux["rho0"])
    )

    fig, axes = plt.subplots(5, 2, figsize=(12, 15), sharex=True)
    axes = axes.ravel()
    plots = [
        ("altitude_ft", "Altitude [ft]"),
        ("speed_kt", "Speed [kt]"),
        ("gamma_deg", "Flight path angle [deg]"),
        ("throttle", "Throttle [-]"),
        ("fuel_flow_lb_hr", "Fuel flow [lb/hr]"),
        ("mass_lbm", "Aircraft mass [lbm]"),
        ("pressure_psia", "Tank pressure [psia]"),
        ("fill_level", "Tank fill level [-]"),
        ("t_gas_R", "Tank temperatures [R]"),
    ]

    for ax, (key, ylabel) in zip(axes, plots):
        ax.plot(time_min, values[key], linewidth=1.8, marker="o", markersize=2.5)
        if key == "speed_kt":
            ax.lines[0].set_label("True")
            ax.plot(time_min, values["equivalent_speed_kt"], linewidth=1.8, marker="o", markersize=2.5, label="Equivalent")
            ax.legend(loc="best")
        if key == "t_gas_R":
            ax.plot(time_min, values["t_liq_R"], linewidth=1.8, marker="o", markersize=2.5, label="Liquid")
            ax.plot(time_min, values["t_env_R"], linewidth=1.8, marker="o", markersize=2.5, label="Atmosphere")
            ax.lines[0].set_label("Ullage")
            ax.legend(loc="best")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [min]")
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
        ax.grid(True, alpha=0.3)
        for segment_index in segment_indices[:-1]:
            ax.axvline(time_min[segment_index], color="0.55", linewidth=1.0, alpha=0.75)

    for ax in axes[len(plots) :]:
        ax.set_visible(False)
    if segment_indices:
        for label, segment_index in zip(segment_labels[1:], segment_indices[:-1]):
            axes[0].text(
                time_min[segment_index],
                0.98,
                label,
                transform=axes[0].get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                color="0.35",
                fontsize=8,
            )
    fig.suptitle(f"Coupled LNG Tank and AeroSandbox Mission ({problem['mission_name']})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return values


def main():
    # Runs the reference mission and writes the plot.
    problem = build_coupled_problem()
    sol = problem["opti"].solve(verbose=False)
    output_path = OUTPUT_DIR / "lng_coupled_aerosandbox_mission_reference.png"
    values = plot_solution(problem, sol, output_path)

    fuel_used = values["mass_lbm"][0] - values["mass_lbm"][-1]
    try:
        duration_s = float(sol.value(problem["duration"]))
    except Exception:
        duration_s = float(problem["duration"])
    print(f"Coupled LNG AeroSandbox mission solved ({problem['mission_name']})")
    print(f"Final range: {values['range_nmi'][-1]:.1f} nmi")
    print(f"Final duration: {duration_s / 60:.2f} min")
    print(f"Fuel and boil-off mass reduction: {fuel_used:.2f} lbm")
    print(f"Final tank pressure: {values['pressure_psia'][-1]:.3f} psia")
    print(f"Final fill level: {values['fill_level'][-1]:.4f}")
    print(f"Saved plot: {output_path}")
    return problem, sol, values


if __name__ == "__main__":
    main()
