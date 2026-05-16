"""
AeroSandbox/CasADi-native LNG tank trajectory model.

This is an optimization-oriented transient surrogate for the OpenMDAO LNG tank
model. It keeps the state structure and path constraints needed for trajectory
optimization, but uses simple methane property fits/constants so the problem is
smooth and CasADi-compatible.
"""

from dataclasses import dataclass

import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as onp


@dataclass(frozen=True)
class LNGSurrogateProperties:
    """Methane-like LNG properties for a first-pass CasADi-compatible model."""

    gas_constant: float = 518.28  # J/kg/K, methane
    liquid_density_ref: float = 422.3  # kg/m^3 near 111.7 K
    liquid_temp_ref: float = 111.7  # K
    liquid_beta: float = 3.5e-3  # 1/K, rough volumetric expansion
    liquid_cp: float = 3500.0  # J/kg/K
    gas_cv: float = 1700.0  # J/kg/K
    latent_heat: float = 5.1e5  # J/kg

    def liquid_density(self, T_liq):
        return self.liquid_density_ref * (1 - self.liquid_beta * (T_liq - self.liquid_temp_ref))


@dataclass(frozen=True)
class TankDesign:
    radius: float = 2.75  # m
    length: float = 2.0  # m, cylindrical section only
    n_layers: float = 20.0
    heat_multiplier: float = 2.0


@dataclass(frozen=True)
class MissionInputs:
    duration: float = 3600.0  # s
    t_env: float = 350.0  # K
    p_heater: float = 1000.0  # W
    m_dot_liq_out: float = 300.0 / 3600.0  # kg/s
    m_dot_gas_out: float = 0.0  # kg/s


@dataclass(frozen=True)
class InitialState:
    ullage_pressure: float = 1.064e6  # Pa
    ullage_temperature: float = 151.8  # K
    liquid_temperature: float = 145.8  # K
    fill_level: float = 0.9
    q_add: float = 0.0  # W


def tank_volume(radius, length):
    return 4 / 3 * np.pi * radius**3 + np.pi * radius**2 * length


def wetted_area(radius, length, fill_level):
    total_area = 4 * np.pi * radius**2 + 2 * np.pi * radius * length
    return fill_level * total_area


def dry_area(radius, length, fill_level):
    total_area = 4 * np.pi * radius**2 + 2 * np.pi * radius * length
    return (1 - fill_level) * total_area


def mli_heat_flux(t_hot, t_cold, n_layers):
    """Same Keller-style MLI correlation form used by lngtank.heat_leak."""
    layer_density = 30.0
    solid_cond_coeff = 8.95e-8
    gas_cond_coeff = 1.46e4
    rad_coeff = 5.39e-10
    emittance = 0.031
    vacuum_pressure = 1e-6

    q_rad = rad_coeff * emittance / n_layers * (t_hot**4.67 - t_cold**4.67)
    q_solid = solid_cond_coeff * layer_density**2.56 / n_layers * (t_hot + t_cold) / 2 * (t_hot - t_cold)
    q_gas = gas_cond_coeff * vacuum_pressure / n_layers * (t_hot**0.52 - t_cold**0.52)
    return q_rad + q_solid + q_gas


def heat_leak(radius, length, fill_level, t_env, t_liq, t_gas, n_layers, heat_multiplier):
    q_liq = heat_multiplier * mli_heat_flux(t_env, t_liq, n_layers) * wetted_area(radius, length, fill_level)
    q_gas = heat_multiplier * mli_heat_flux(t_env, t_gas, n_layers) * dry_area(radius, length, fill_level)
    return q_liq, q_gas


def tank_rhs(
    state,
    design: TankDesign,
    inputs: MissionInputs,
    props: LNGSurrogateProperties = LNGSurrogateProperties(),
    heater_rate_const: float = 1e-3,
    heater_boil_frac: float = 0.75,
):
    """Return state derivatives and auxiliary values for the transient tank surrogate."""
    m_gas, m_liq, t_gas, t_liq, v_gas, q_add = state

    fill_level = 1 - v_gas / tank_volume(design.radius, design.length)
    rho_liq = props.liquid_density(t_liq)
    pressure = m_gas / v_gas * props.gas_constant * t_gas
    q_liq, q_gas = heat_leak(
        design.radius,
        design.length,
        fill_level,
        inputs.t_env,
        t_liq,
        t_gas,
        design.n_layers,
        design.heat_multiplier,
    )

    m_dot_boil = (q_liq + heater_boil_frac * q_add) / props.latent_heat
    m_gas_dot = m_dot_boil - inputs.m_dot_gas_out
    m_liq_dot = -m_dot_boil - inputs.m_dot_liq_out
    v_gas_dot = -m_liq_dot / rho_liq
    q_add_dot = heater_rate_const * (inputs.p_heater - q_add)

    t_liq_dot = ((1 - heater_boil_frac) * q_add) / (m_liq * props.liquid_cp)
    t_gas_dot = q_gas / (m_gas * props.gas_cv)

    xdot = [m_gas_dot, m_liq_dot, t_gas_dot, t_liq_dot, v_gas_dot, q_add_dot]
    aux = {
        "pressure": pressure,
        "fill_level": fill_level,
        "q_liq": q_liq,
        "q_gas": q_gas,
        "m_dot_boil": m_dot_boil,
    }
    return xdot, aux


def build_trajectory_problem(
    n_nodes: int = 41,
    design: TankDesign = TankDesign(),
    inputs: MissionInputs = MissionInputs(),
    initial: InitialState = InitialState(),
    p_max: float = 1.064e6,
):
    """
    Build a direct-transcription AeroSandbox problem for LNG tank dynamics.

    Uses trapezoidal defects with fixed design and mission inputs. Later
    iterations can unfreeze design/input variables and connect them to the
    aircraft trajectory.
    """
    props = LNGSurrogateProperties()
    opti = asb.Opti()
    time = onp.linspace(0, inputs.duration, n_nodes)
    dt = inputs.duration / (n_nodes - 1)

    volume = float(tank_volume(design.radius, design.length))
    v_gas0 = volume * (1 - initial.fill_level)
    rho_liq0 = props.liquid_density(initial.liquid_temperature)
    m_liq0 = (volume - v_gas0) * rho_liq0
    m_gas0 = initial.ullage_pressure * v_gas0 / (props.gas_constant * initial.ullage_temperature)

    m_gas = opti.variable(init_guess=m_gas0, n_vars=n_nodes, lower_bound=1e-4, scale=max(m_gas0, 1.0))
    m_liq = opti.variable(
        init_guess=onp.linspace(m_liq0, m_liq0 - inputs.m_dot_liq_out * inputs.duration, n_nodes),
        n_vars=n_nodes,
        lower_bound=1.0,
        scale=max(m_liq0, 1.0),
    )
    t_gas = opti.variable(init_guess=initial.ullage_temperature, n_vars=n_nodes, lower_bound=92, upper_bound=300, scale=150)
    t_liq = opti.variable(init_guess=initial.liquid_temperature, n_vars=n_nodes, lower_bound=90, upper_bound=190, scale=150)
    v_gas = opti.variable(init_guess=v_gas0, n_vars=n_nodes, lower_bound=1e-5, upper_bound=volume * 0.99, scale=volume)
    q_add = opti.variable(init_guess=initial.q_add, n_vars=n_nodes, lower_bound=0, scale=max(inputs.p_heater, 1.0))

    states = [m_gas, m_liq, t_gas, t_liq, v_gas, q_add]

    opti.subject_to(m_gas[0] == m_gas0)
    opti.subject_to(m_liq[0] == m_liq0)
    opti.subject_to(t_gas[0] == initial.ullage_temperature)
    opti.subject_to(t_liq[0] == initial.liquid_temperature)
    opti.subject_to(v_gas[0] == v_gas0)
    opti.subject_to(q_add[0] == initial.q_add)

    pressure = []
    fill_level = []
    q_liq = []
    q_gas = []
    m_dot_boil = []

    rhs = []
    for k in range(n_nodes):
        x_k = [state[k] for state in states]
        f_k, aux_k = tank_rhs(x_k, design, inputs, props)
        rhs.append(f_k)
        pressure.append(aux_k["pressure"])
        fill_level.append(aux_k["fill_level"])
        q_liq.append(aux_k["q_liq"])
        q_gas.append(aux_k["q_gas"])
        m_dot_boil.append(aux_k["m_dot_boil"])

        opti.subject_to(aux_k["pressure"] <= p_max)
        opti.subject_to(aux_k["fill_level"] >= 0.05)
        opti.subject_to(aux_k["fill_level"] <= 0.95)

    for k in range(n_nodes - 1):
        for i, state in enumerate(states):
            opti.subject_to(state[k + 1] - state[k] == 0.5 * dt * (rhs[k][i] + rhs[k + 1][i]))

    opti.minimize(m_liq[0] - m_liq[-1])

    return {
        "opti": opti,
        "time": time,
        "states": {
            "m_gas": m_gas,
            "m_liq": m_liq,
            "T_gas": t_gas,
            "T_liq": t_liq,
            "V_gas": v_gas,
            "Q_add": q_add,
        },
        "aux": {
            "pressure": pressure,
            "fill_level": fill_level,
            "Q_liq": q_liq,
            "Q_gas": q_gas,
            "m_dot_boil": m_dot_boil,
        },
    }


def solve_demo(n_nodes: int = 41):
    problem = build_trajectory_problem(n_nodes=n_nodes)
    sol = problem["opti"].solve(verbose=False)
    return problem, sol


if __name__ == "__main__":
    problem, sol = solve_demo()
    time = problem["time"]
    pressure = onp.array([sol.value(p) for p in problem["aux"]["pressure"]])
    fill_level = onp.array([sol.value(f) for f in problem["aux"]["fill_level"]])
    m_liq = onp.array(sol.value(problem["states"]["m_liq"]))

    print(f"Final pressure: {pressure[-1] / 1e5:.3f} bar")
    print(f"Final fill level: {fill_level[-1]:.4f}")
    print(f"Liquid used: {m_liq[0] - m_liq[-1]:.2f} kg over {time[-1] / 3600:.2f} hr")
