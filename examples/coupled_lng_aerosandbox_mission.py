"""
Closed-loop AeroSandbox mission and LNG tank example.

This example uses a simple powered 2D point-mass aircraft model with three
mission phases: climb, cruise, and descent. The LNG tank is solved inside the
same AeroSandbox optimization problem. Atmospheric temperature from the flight
trajectory drives the tank heat leak, engine fuel flow draws liquid LNG from
the tank, and the remaining tank fluid mass feeds back into aircraft weight.
"""

from dataclasses import dataclass
from pathlib import Path

import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import numpy as onp

from lngtank.aerosandbox_tank import (
    InitialState,
    MissionInputs,
    TankDesign,
    initial_gas_density_from_pressure,
    liquid_volume_from_height_fraction,
    tank_rhs,
    tank_volume,
)
from lngtank.asb_properties_interpolants import CoolPropGridInterpolants


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
G = 9.80665


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
    initial_speed: float = 78.0  # m/s
    final_speed: float = 78.0  # m/s
    p_heater: float = 1000.0  # W
    initial_fill: float = 0.82
    initial_pressure: float = 1.064e6  # Pa
    initial_gas_temperature: float = 151.8  # K
    initial_liquid_temperature: float = 145.8  # K


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


def build_coupled_problem(
    mission: MissionSpec = MissionSpec(),
    aircraft: AircraftModel = AircraftModel(),
    tank_design: TankDesign = TankDesign(radius=1.15, length=2.5, n_layers=20.0, heat_multiplier=2.0),
):
    props = CoolPropGridInterpolants()
    opti = asb.Opti()

    n = mission.num_nodes
    time = onp.linspace(0.0, mission.duration, n)
    dt = mission.duration / (n - 1)
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
    z_guess = smooth_piecewise_altitude(time, mission.duration, mission.cruise_altitude)
    v_guess = onp.full(n, mission.initial_speed)
    gamma_guess = onp.gradient(z_guess, time) / mission.initial_speed
    fuel_guess = onp.linspace(0.0, 140.0, n)

    x = opti.variable(init_guess=x_guess, n_vars=n, lower_bound=0.0, scale=mission.range)
    z = opti.variable(
        init_guess=z_guess,
        n_vars=n,
        lower_bound=0.0,
        upper_bound=mission.cruise_altitude,
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

    opti.subject_to([x[0] == 0.0, x[-1] == mission.range])
    opti.subject_to([z[0] == 0.0, z[climb_end] == mission.cruise_altitude, z[descent_start] == mission.cruise_altitude, z[-1] == 0.0])
    opti.subject_to(z[climb_end : descent_start + 1] == mission.cruise_altitude)
    opti.subject_to([v[0] == mission.initial_speed, v[-1] == mission.final_speed])
    opti.subject_to([gamma[0] == 0.0, gamma[climb_end] == 0.0, gamma[descent_start] == 0.0, gamma[-1] == 0.0])

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

    states = [x, z, v, gamma, m_gas, m_liq, t_gas, t_liq, v_gas, q_add]
    rhs = []
    aux = {key: [] for key in ["pressure", "fill_level", "m_dot_fuel", "mass", "q_gas", "q_liq", "t_env"]}

    rho0 = float(asb.Atmosphere(altitude=0.0).density())
    induced_factor = 1 / (np.pi * aircraft.aspect_ratio * aircraft.oswald_efficiency)

    def node_rhs(k):
        atmosphere = asb.Atmosphere(altitude=z[k])
        rho = atmosphere.density()
        t_env = atmosphere.temperature()
        density_ratio = rho / rho0
        q_dyn = 0.5 * rho * v[k] ** 2
        cd = aircraft.cd0 + induced_factor * cl[k] ** 2
        lift = q_dyn * aircraft.wing_area * cl[k]
        drag = q_dyn * aircraft.wing_area * cd
        shaft_power = throttle[k] * aircraft.max_sea_level_shaft_power * density_ratio**0.8
        thrust = aircraft.propeller_efficiency * shaft_power / v[k]
        m_dot_fuel = aircraft.psfc * shaft_power

        tank_inputs = MissionInputs(
            duration=mission.duration,
            t_env=t_env,
            p_heater=mission.p_heater,
            m_dot_liq_out=m_dot_fuel,
            m_dot_gas_out=0.0,
        )
        tank_f, tank_aux = tank_rhs(
            [m_gas[k], m_liq[k], t_gas[k], t_liq[k], v_gas[k], q_add[k]],
            tank_design,
            tank_inputs,
            props,
            h_liq_frac=h_liq_frac[k],
        )

        mass = aircraft.dry_mass + aircraft.tank_hardware_mass + m_gas[k] + m_liq[k]
        x_dot = v[k] * np.cos(gamma[k])
        z_dot = v[k] * np.sin(gamma[k])
        v_dot = (thrust - drag) / mass - G * np.sin(gamma[k])
        gamma_dot = (lift - mass * G * np.cos(gamma[k])) / (mass * v[k])

        aux["pressure"].append(tank_aux["pressure"])
        aux["fill_level"].append(tank_aux["fill_level"])
        aux["m_dot_fuel"].append(m_dot_fuel)
        aux["mass"].append(mass)
        aux["q_gas"].append(tank_aux["q_gas"])
        aux["q_liq"].append(tank_aux["q_liq"])
        aux["t_env"].append(t_env)
        return [x_dot, z_dot, v_dot, gamma_dot, *tank_f]

    for k in range(n):
        rhs_k = node_rhs(k)
        rhs.append(rhs_k)

        opti.subject_to(aux["pressure"][k] <= mission.initial_pressure)
        opti.subject_to(aux["pressure"][k] >= 2.0e5)
        opti.subject_to(aux["fill_level"][k] >= 0.05)
        opti.subject_to(aux["fill_level"][k] <= 0.95)
        opti.subject_to(
            liquid_volume_from_height_fraction(tank_design.radius, tank_design.length, h_liq_frac[k])
            == volume * aux["fill_level"][k]
        )

    for k in range(n - 1):
        for i, state in enumerate(states):
            opti.subject_to(state[k + 1] - state[k] == 0.5 * dt * (rhs[k][i] + rhs[k + 1][i]))
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
    )

    return {
        "opti": opti,
        "time": time,
        "states": {
            "x": x,
            "z": z,
            "V": v,
            "gamma": gamma,
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
    time_min = problem["time"] / 60
    states = problem["states"]
    aux = problem["aux"]

    values = {
        "altitude_m": onp.array(sol.value(states["z"])),
        "range_km": onp.array(sol.value(states["x"])) / 1000,
        "speed_m_s": onp.array(sol.value(states["V"])),
        "throttle": onp.array(sol.value(states["throttle"])),
        "mass_kg": onp.array([sol.value(v) for v in aux["mass"]]),
        "fuel_flow_kg_s": onp.array([sol.value(v) for v in aux["m_dot_fuel"]]),
        "pressure_bar": onp.array([sol.value(v) for v in aux["pressure"]]) / 1e5,
        "fill_level": onp.array([sol.value(v) for v in aux["fill_level"]]),
        "t_gas_K": onp.array(sol.value(states["T_gas"])),
        "t_liq_K": onp.array(sol.value(states["T_liq"])),
        "t_env_K": onp.array([sol.value(v) for v in aux["t_env"]]),
    }

    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
    axes = axes.ravel()
    plots = [
        ("altitude_m", "Altitude [m]"),
        ("speed_m_s", "Speed [m/s]"),
        ("throttle", "Throttle [-]"),
        ("fuel_flow_kg_s", "Fuel flow [kg/s]"),
        ("mass_kg", "Aircraft mass [kg]"),
        ("pressure_bar", "Tank pressure [bar]"),
        ("fill_level", "Tank fill level [-]"),
        ("t_gas_K", "Tank temperatures [K]"),
    ]

    for ax, (key, ylabel) in zip(axes, plots):
        ax.plot(time_min, values[key], linewidth=2)
        if key == "t_gas_K":
            ax.plot(time_min, values["t_liq_K"], linewidth=2, label="Liquid")
            ax.plot(time_min, values["t_env_K"], linewidth=2, label="Atmosphere")
            ax.lines[0].set_label("Ullage")
            ax.legend(loc="best")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    for ax in axes[-2:]:
        ax.set_xlabel("Time [min]")
    fig.suptitle("Coupled LNG Tank and AeroSandbox Mission")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return values


def main():
    problem = build_coupled_problem()
    sol = problem["opti"].solve(verbose=False)
    output_path = OUTPUT_DIR / "lng_coupled_aerosandbox_mission.png"
    values = plot_solution(problem, sol, output_path)

    fuel_used = values["mass_kg"][0] - values["mass_kg"][-1]
    print("Coupled LNG AeroSandbox mission solved")
    print(f"Final range: {values['range_km'][-1]:.1f} km")
    print(f"Fuel and boil-off mass reduction: {fuel_used:.2f} kg")
    print(f"Final tank pressure: {values['pressure_bar'][-1]:.3f} bar")
    print(f"Final fill level: {values['fill_level'][-1]:.4f}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
