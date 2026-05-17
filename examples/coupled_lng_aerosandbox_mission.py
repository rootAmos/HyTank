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
from dataclasses import dataclass, field
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
DEFAULT_SCHEDULE_PATH = Path(__file__).resolve().parent / "mission_lng_4seg.json"
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
    propeller_efficiency: float = 0.85
    psfc: float = 7.6e-8  # kg / W / s, roughly 0.45 lb / hp / hr


@dataclass(frozen=True)
class MissionSpec:
    num_nodes: int = 61
    duration: float = 2400.0  # s
    range: float = 170e3  # m
    cruise_altitude: float = 2500.0  # m
    cruise_speed: float = 72.0  # m/s
    end_of_cruise_range_fraction: float = 0.75
    initial_speed: float = 78.0  # m/s
    final_speed: float = 78.0  # m/s
    max_accel: float = 2.0  # m/s^2
    max_gamma_rate: float = onp.radians(0.05)  # rad/s, 0.05 deg/s
    p_heater: float = 1000.0  # W
    initial_fill: float = 0.82
    initial_pressure: float = 1.064e6  # Pa
    initial_gas_temperature: float = 151.8  # K
    initial_liquid_temperature: float = 145.8  # K


@dataclass(frozen=True)
class SegmentEnd:
    condition: str
    node_fraction: float | None = None
    duration_s: float | None = None
    distance_m: float | None = None
    altitude_m: float | None = None
    speed_m_s: float | None = None
    gamma_deg: float | None = None

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass(frozen=True)
class SegmentRules:
    fix: dict = field(default_factory=dict)
    minimum: dict = field(default_factory=dict)
    maximum: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            fix=data.get("fix", {}),
            minimum=data.get("min", data.get("minimum", {})),
            maximum=data.get("max", data.get("maximum", {})),
        )


@dataclass(frozen=True)
class MissionSegment:
    name: str
    segment_type: str
    end: SegmentEnd
    constraints: SegmentRules = field(default_factory=SegmentRules)

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            segment_type=data.get("segment_type", data.get("type", "segment")),
            end=SegmentEnd.from_dict(data["end"]),
            constraints=SegmentRules.from_dict(data.get("constraints", {})),
        )


@dataclass(frozen=True)
class MissionSchedule:
    mission_name: str
    duration_s: float | None
    duration_guess_s: float
    duration_bounds_s: tuple[float, float] | None
    range_m: float
    num_nodes: int
    initial_altitude_m: float
    initial_speed_m_s: float
    final_altitude_m: float
    final_speed_m_s: float | None
    segments: tuple[MissionSegment, ...]

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        initial = data.get("initial", {})
        final = data.get("final", {})
        duration_s = data.get("duration_s")
        duration_guess_s = data.get("duration_guess_s", duration_s)
        if duration_guess_s is None:
            raise ValueError("Schedule must define either duration_s or duration_guess_s.")
        duration_bounds_s = data.get("duration_bounds_s")
        return cls(
            mission_name=data.get("mission_name", Path(path).stem),
            duration_s=float(duration_s) if duration_s is not None else None,
            duration_guess_s=float(duration_guess_s),
            duration_bounds_s=tuple(float(value) for value in duration_bounds_s) if duration_bounds_s else None,
            range_m=float(data["range_m"]),
            num_nodes=int(data.get("num_nodes", 61)),
            initial_altitude_m=float(initial.get("altitude_m", 0.0)),
            initial_speed_m_s=float(initial["speed_m_s"]),
            final_altitude_m=float(final.get("altitude_m", 0.0)),
            final_speed_m_s=float(final["speed_m_s"]) if "speed_m_s" in final else None,
            segments=tuple(MissionSegment.from_dict(segment) for segment in data["segments"]),
        )


def smooth_piecewise_altitude(time, duration, cruise_altitude):
    """Initial guess only; constraints define the actual phase endpoints."""
    tau = time / duration
    altitude = onp.empty_like(time)
    for i, tau_i in enumerate(tau):
        if tau_i <= 0.25:
            altitude[i] = cruise_altitude * tau_i / 0.25
        elif tau_i <= 0.75:
            altitude[i] = cruise_altitude
        else:
            altitude[i] = cruise_altitude * (1 - (tau_i - 0.75) / 0.25)
    return altitude


def schedule_to_mission_spec(schedule: MissionSchedule, base: MissionSpec = MissionSpec()):
    return MissionSpec(
        num_nodes=schedule.num_nodes,
        duration=schedule.duration_guess_s,
        range=schedule.range_m,
        cruise_altitude=max(
            [base.cruise_altitude]
            + [segment.end.altitude_m for segment in schedule.segments if segment.end.altitude_m is not None]
        ),
        cruise_speed=base.cruise_speed,
        end_of_cruise_range_fraction=base.end_of_cruise_range_fraction,
        initial_speed=schedule.initial_speed_m_s,
        final_speed=schedule.final_speed_m_s if schedule.final_speed_m_s is not None else base.final_speed,
        max_accel=base.max_accel,
        max_gamma_rate=base.max_gamma_rate,
        p_heater=base.p_heater,
        initial_fill=base.initial_fill,
        initial_pressure=base.initial_pressure,
        initial_gas_temperature=base.initial_gas_temperature,
        initial_liquid_temperature=base.initial_liquid_temperature,
    )


def segment_end_indices(schedule: MissionSchedule, time):
    end_indices = []
    last_index = 0
    elapsed = 0.0
    for i, segment in enumerate(schedule.segments):
        end = segment.end
        if end.node_fraction is not None:
            index = int(round(end.node_fraction * (schedule.num_nodes - 1)))
        elif end.condition == "duration":
            elapsed += float(end.duration_s)
            index = int(onp.searchsorted(time, elapsed, side="left"))
        elif end.condition == "distance" and end.distance_m is not None:
            index = int(round(end.distance_m / schedule.range_m * (schedule.num_nodes - 1)))
        else:
            remaining_segments = len(schedule.segments) - i
            remaining_nodes = schedule.num_nodes - 1 - last_index
            index = last_index + max(1, int(round(remaining_nodes / remaining_segments)))

        if i == len(schedule.segments) - 1:
            index = schedule.num_nodes - 1
        index = min(max(index, last_index + 1), schedule.num_nodes - 1)
        end_indices.append(index)
        last_index = index
    return end_indices


def apply_value_constraint(opti, dyn, throttle, key, selection, value, equality=True, rho=None, rho0=None):
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
    else:
        raise ValueError(f"Unsupported segment constraint key '{key}'.")

    if equality:
        opti.subject_to(expr == value)
    else:
        return expr, value


def segment_constraint_slice(key, start, end_index):
    first_index = start if start == 0 and key in {"power_fraction", "throttle"} else start + 1
    return slice(first_index, end_index + 1)


def segment_plot_label(segment, index):
    segment_type = segment.segment_type.replace("_", " ").strip()
    if segment_type == "climb":
        return f"climb {index + 1}"
    return segment_type or f"segment {index + 1}"


def apply_schedule_constraints(opti, dyn, throttle, schedule: MissionSchedule, time, rho=None, rho0=None):
    end_indices = segment_end_indices(schedule, time)
    opti.subject_to(
        [
            dyn.altitude[0] == schedule.initial_altitude_m,
            dyn.speed[0] == schedule.initial_speed_m_s,
            dyn.altitude[-1] == schedule.final_altitude_m,
            dyn.x_e[-1] == schedule.range_m,
        ]
    )
    if schedule.final_speed_m_s is not None:
        opti.subject_to(dyn.speed[-1] == schedule.final_speed_m_s)

    start = 0
    for segment, end_index in zip(schedule.segments, end_indices):
        end = segment.end
        endpoint = end_index
        if end.altitude_m is not None:
            opti.subject_to(dyn.altitude[endpoint] == end.altitude_m)
        if end.distance_m is not None:
            opti.subject_to(dyn.x_e[endpoint] == end.distance_m)
        if end.speed_m_s is not None:
            opti.subject_to(dyn.speed[endpoint] == end.speed_m_s)
        if end.gamma_deg is not None:
            opti.subject_to(dyn.gamma[endpoint] == onp.radians(end.gamma_deg))

        for key, value in segment.constraints.fix.items():
            segment_slice = segment_constraint_slice(key, start, end_index)
            if segment_slice.stop > segment_slice.start:
                apply_value_constraint(
                    opti,
                    dyn,
                    throttle,
                    key,
                    segment_slice,
                    value,
                    equality=True,
                    rho=rho,
                    rho0=rho0,
                )
        for key, value in segment.constraints.minimum.items():
            segment_slice = segment_constraint_slice(key, start, end_index)
            if segment_slice.stop > segment_slice.start:
                expr, converted = apply_value_constraint(
                    opti,
                    dyn,
                    throttle,
                    key,
                    segment_slice,
                    value,
                    equality=False,
                    rho=rho,
                    rho0=rho0,
                )
                opti.subject_to(expr >= converted)
        for key, value in segment.constraints.maximum.items():
            segment_slice = segment_constraint_slice(key, start, end_index)
            if segment_slice.stop > segment_slice.start:
                expr, converted = apply_value_constraint(
                    opti,
                    dyn,
                    throttle,
                    key,
                    segment_slice,
                    value,
                    equality=False,
                    rho=rho,
                    rho0=rho0,
                )
                opti.subject_to(expr <= converted)

        start = end_index
    return end_indices


def build_coupled_problem(
    mission: MissionSpec = MissionSpec(),
    aircraft: AircraftModel = AircraftModel(),
    tank_design: TankDesign = TankDesign(radius=1.15, length=2.5, n_layers=20.0, heat_multiplier=2.0),
    mode: str = "segmented",
    schedule: MissionSchedule | None = None,
):
    if mode not in {"segmented", "free", "schedule"}:
        raise ValueError("mode must be 'segmented', 'free', or 'schedule'")
    if mode == "schedule":
        if schedule is None:
            schedule = MissionSchedule.from_json(DEFAULT_SCHEDULE_PATH)
        mission = schedule_to_mission_spec(schedule, mission)

    props = CoolPropGridInterpolants()
    opti = asb.Opti()

    n = mission.num_nodes
    if mode == "schedule" and schedule.duration_s is None:
        duration_lower, duration_upper = schedule.duration_bounds_s or (0.5 * mission.duration, 2.0 * mission.duration)
        duration = opti.variable(
            init_guess=mission.duration,
            lower_bound=duration_lower,
            upper_bound=duration_upper,
            scale=mission.duration,
        )
    else:
        duration = mission.duration
    tau = onp.linspace(0.0, 1.0, n)
    time = tau * duration
    time_guess = tau * mission.duration
    climb_end = int(0.25 * (n - 1))
    descent_start = int(0.75 * (n - 1))

    volume = float(tank_volume(tank_design.radius, tank_design.length))
    v_gas0 = volume * (1 - mission.initial_fill)
    m_liq0 = (volume - v_gas0) * float(props.liquid_density(mission.initial_liquid_temperature))
    m_gas0 = (
        initial_gas_density_from_pressure(
            props,
            mission.initial_pressure,
            mission.initial_gas_temperature,
        )
        * v_gas0
    )

    x_guess = onp.linspace(0.0, mission.range, n)
    z_guess = smooth_piecewise_altitude(time_guess, mission.duration, mission.cruise_altitude)
    v_guess = onp.full(n, mission.initial_speed)
    gamma_guess = onp.gradient(z_guess, time_guess) / mission.initial_speed
    fuel_guess = onp.linspace(0.0, 140.0, n)
    altitude_upper_bound = mission.cruise_altitude if mode in {"segmented", "schedule"} else 1.8 * mission.cruise_altitude

    x = opti.variable(init_guess=x_guess, n_vars=n, lower_bound=0.0, scale=mission.range)
    z_e = opti.variable(
        init_guess=-z_guess,
        n_vars=n,
        lower_bound=-altitude_upper_bound,
        upper_bound=0.0,
        scale=mission.cruise_altitude,
    )
    v = opti.variable(init_guess=v_guess, n_vars=n, lower_bound=45.0, upper_bound=125.0, scale=mission.initial_speed)
    gamma = opti.variable(init_guess=gamma_guess, n_vars=n, lower_bound=-0.12, upper_bound=0.12, scale=0.05)

    throttle = opti.variable(init_guess=0.55, n_vars=n, lower_bound=0.05, upper_bound=1.0)
    cl = opti.variable(init_guess=0.65, n_vars=n, lower_bound=0.05, upper_bound=1.4)

    m_gas = opti.variable(init_guess=m_gas0, n_vars=n, lower_bound=1e-3, scale=max(m_gas0, 1.0))
    m_liq = opti.variable(
        init_guess=m_liq0 - fuel_guess,
        n_vars=n,
        lower_bound=100.0,
        scale=max(m_liq0, 1.0),
    )
    t_gas = opti.variable(
        init_guess=mission.initial_gas_temperature,
        n_vars=n,
        lower_bound=92.0,
        upper_bound=230.0,
        scale=150.0,
    )
    t_liq = opti.variable(
        init_guess=mission.initial_liquid_temperature,
        n_vars=n,
        lower_bound=90.0,
        upper_bound=190.0,
        scale=150.0,
    )
    v_gas = opti.variable(init_guess=v_gas0, n_vars=n, lower_bound=1e-4, upper_bound=0.98 * volume, scale=volume)
    q_add = opti.variable(init_guess=0.0, n_vars=n, lower_bound=0.0, scale=mission.p_heater)
    h_liq_frac = opti.variable(init_guess=mission.initial_fill, n_vars=n, lower_bound=1e-3, upper_bound=1 - 1e-3)

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
            t_gas[0] == mission.initial_gas_temperature,
            t_liq[0] == mission.initial_liquid_temperature,
            v_gas[0] == v_gas0,
            q_add[0] == 0.0,
        ]
    )

    rho0 = float(asb.Atmosphere(altitude=0.0).density())
    rho = dyn.op_point.atmosphere.density()
    t_env = dyn.op_point.atmosphere.temperature()
    density_ratio = rho / rho0

    if mode == "schedule":
        segment_indices = apply_schedule_constraints(opti, dyn, throttle, schedule, time, rho=rho, rho0=rho0)
        segment_labels = [segment_plot_label(segment, i) for i, segment in enumerate(schedule.segments)]
        opti.subject_to(dyn.gamma[0] == 0.0)
    else:
        opti.subject_to(
            [
                dyn.x_e[0] == 0.0,
                dyn.x_e[-1] == mission.range,
                dyn.altitude[0] == 0.0,
                dyn.altitude[-1] == 0.0,
                dyn.speed[0] == mission.initial_speed,
                dyn.speed[-1] == mission.final_speed,
                dyn.gamma[0] == 0.0,
                dyn.gamma[-1] == 0.0,
            ]
        )
        if mode == "segmented":
            opti.subject_to(
                [
                    dyn.altitude[climb_end] == mission.cruise_altitude,
                    dyn.altitude[descent_start] == mission.cruise_altitude,
                    dyn.speed[climb_end] == mission.cruise_speed,
                    dyn.speed[descent_start] == mission.cruise_speed,
                    dyn.x_e[descent_start] == mission.end_of_cruise_range_fraction * mission.range,
                    dyn.gamma[climb_end] == 0.0,
                    dyn.gamma[descent_start] == 0.0,
                ]
            )
        segment_indices = []
        segment_labels = []

    q_dyn = dyn.op_point.dynamic_pressure()
    induced_factor = 1 / (np.pi * aircraft.aspect_ratio * aircraft.oswald_efficiency)
    cd = aircraft.cd0 + induced_factor * cl**2
    lift = q_dyn * aircraft.wing_area * cl
    drag = q_dyn * aircraft.wing_area * cd
    shaft_power = throttle * aircraft.max_sea_level_shaft_power * density_ratio**0.8
    thrust = aircraft.propeller_efficiency * shaft_power / dyn.speed
    m_dot_fuel = aircraft.psfc * shaft_power

    dyn.add_gravity_force(g=G)
    dyn.add_force(Fx=thrust - drag, Fz=-lift, axes="wind")
    dyn_derivatives = dyn.state_derivatives()
    opti.subject_to(dyn_derivatives["speed"] <= mission.max_accel)
    opti.subject_to(dyn_derivatives["speed"] >= -mission.max_accel)
    opti.subject_to(dyn_derivatives["gamma"] <= mission.max_gamma_rate)
    opti.subject_to(dyn_derivatives["gamma"] >= -mission.max_gamma_rate)
    dyn.constrain_derivatives(opti, time)

    tank_states = [m_gas, m_liq, t_gas, t_liq, v_gas, q_add]
    tank_inputs = MissionInputs(
        duration=mission.duration,
        t_env=t_env,
        p_heater=mission.p_heater,
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

    opti.subject_to(tank_aux["pressure"] <= mission.initial_pressure)
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
    }

    for k in range(n - 1):
        opti.subject_to(throttle[k + 1] - throttle[k] <= 0.12)
        opti.subject_to(throttle[k + 1] - throttle[k] >= -0.12)
        opti.subject_to(cl[k + 1] - cl[k] <= 0.12)
        opti.subject_to(cl[k + 1] - cl[k] >= -0.12)
        opti.subject_to(v[k + 1] - v[k] <= 3.0)
        opti.subject_to(v[k + 1] - v[k] >= -3.0)

    opti.minimize(
        1000.0 * (m_liq[0] - m_liq[-1])
        + 5000.0 * np.sum((throttle[1:] - throttle[:-1]) ** 2)
        + 100.0 * np.sum((cl[1:] - cl[:-1]) ** 2)
        + 10.0 * np.sum((v[1:] - v[:-1]) ** 2)
        + 5000.0 * np.sum(dyn_derivatives["speed"] ** 2)
        + 1000.0 * np.sum(dyn_derivatives["gamma"] ** 2)
    )

    return {
        "opti": opti,
        "time": time,
        "duration": duration,
        "mode": mode,
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
        ax.plot(time_min, values[key], linewidth=2)
        if key == "speed_kt":
            ax.lines[0].set_label("True")
            ax.plot(time_min, values["equivalent_speed_kt"], linewidth=2, label="Equivalent")
            ax.legend(loc="best")
        if key == "t_gas_R":
            ax.plot(time_min, values["t_liq_R"], linewidth=2, label="Liquid")
            ax.plot(time_min, values["t_env_R"], linewidth=2, label="Atmosphere")
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
    fig.suptitle(f"Coupled LNG Tank and AeroSandbox Mission ({problem['mode']})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return values


def run_case(mode: str):
    problem = build_coupled_problem(mode=mode)
    sol = problem["opti"].solve(verbose=False)
    output_path = OUTPUT_DIR / f"lng_coupled_aerosandbox_mission_{mode}.png"
    values = plot_solution(problem, sol, output_path)

    fuel_used = values["mass_lbm"][0] - values["mass_lbm"][-1]
    try:
        duration_s = float(sol.value(problem["duration"]))
    except Exception:
        duration_s = float(problem["duration"])
    print(f"Coupled LNG AeroSandbox mission solved ({mode})")
    print(f"Final range: {values['range_nmi'][-1]:.1f} nmi")
    print(f"Final duration: {duration_s / 60:.2f} min")
    print(f"Fuel and boil-off mass reduction: {fuel_used:.2f} lbm")
    print(f"Final tank pressure: {values['pressure_psia'][-1]:.3f} psia")
    print(f"Final fill level: {values['fill_level'][-1]:.4f}")
    print(f"Saved plot: {output_path}")
    return problem, sol, values


def main():
    for mode in ("schedule", "segmented", "free"):
        run_case(mode)
        print()


if __name__ == "__main__":
    main()
