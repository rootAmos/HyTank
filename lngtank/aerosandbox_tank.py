"""
AeroSandbox/CasADi-native LNG tank trajectory model.

This is an optimization-oriented transient surrogate for the OpenMDAO LNG tank
model. It keeps the state structure and path constraints needed for trajectory
optimization, with swappable CasADi-compatible property backends.
"""

from dataclasses import dataclass

import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as onp

GRAV_CONST = 9.80665
UNIVERSAL_GAS_CONST = 8.3145
MOLEC_WEIGHT_LNG = 16.0425e-3


@dataclass(frozen=True)
class LNGSurrogateProperties:
    """Local methane/LNG property fits for a CasADi-compatible OpenMDAO physics port."""

    gas_constant: float = 518.28  # J/kg/K, methane
    gas_compressibility_ref: float = 0.8223  # near 10.64 bar, 151.8 K
    liquid_density_ref: float = 366.1  # kg/m^3 near 145.8 K
    liquid_temp_ref: float = 145.8  # K
    liquid_beta: float = 3.5e-3  # 1/K, rough volumetric expansion
    liquid_cp_ref: float = 3936.0  # J/kg/K
    gas_cv_ref: float = 1758.0  # J/kg/K
    latent_heat: float = 4.334e5  # J/kg
    p_ref: float = 1.064e6
    t_gas_ref: float = 151.8
    t_liq_ref: float = 145.8
    t_sat_ref: float = 150.50665599987252

    def linear_2d(self, value_ref, d_dP, d_dT, P, T):
        return value_ref + d_dP * (P - self.p_ref) + d_dT * (T - self.t_gas_ref)

    def linear_1d(self, value_ref, d_dT, T, T_ref=None):
        if T_ref is None:
            T_ref = self.t_liq_ref
        return value_ref + d_dT * (T - T_ref)

    def liquid_density(self, T_liq):
        return self.linear_1d(366.12501939887187, -1.9105294002843627, T_liq)

    def gas_pressure(self, m_gas, v_gas, T_gas):
        return m_gas / v_gas * self.gas_constant * T_gas * self.gas_compressibility_ref

    def gas_density(self, P, T_gas):
        return P / (self.gas_constant * T_gas * self.gas_compressibility_ref)

    def gas_h(self, P, T_gas):
        return self.linear_2d(559203.37926667, -0.04974959, 2874.04318806, P, T_gas)

    def gas_u(self, P, T_gas):
        return self.linear_2d(494506.88734827, -0.03340543, 2097.34588861, P, T_gas)

    def gas_cv(self, P, T_gas):
        return self.linear_2d(1758.18262736, 0.00038059, -12.69779818, P, T_gas)

    def liquid_h(self, T_liq):
        return self.linear_1d(125807.47523201918, 3958.6908265189977, T_liq)

    def liquid_u(self, T_liq):
        return self.linear_1d(123470.91295579074, 3834.8441754564506, T_liq)

    def liquid_cp_value(self, T_liq):
        return self.linear_1d(3935.9212725242774, 24.03107812804126, T_liq)

    def liquid_pressure(self, T_liq):
        return self.linear_1d(855473.9087183854, 40879.2869023012, T_liq)

    def liquid_pressure_dT(self, T_liq):
        return 40879.2869023012 + 0 * T_liq

    def liquid_beta_value(self, T_liq):
        return self.linear_1d(0.005218243235294378, 8.657974245432964e-05, T_liq)

    def liquid_viscosity(self, T_liq):
        return self.linear_1d(6.165300797425853e-05, -1.0546978565888305e-06, T_liq)

    def liquid_k(self, T_liq):
        return self.linear_1d(0.13526878447997115, -0.0014223165932328115, T_liq)

    def sat_gas_T(self, P):
        return self.t_sat_ref + 2.0904625871326975e-05 * (P - self.p_ref)

    def sat_gas_T_dP(self, P):
        return 2.0904625871326975e-05 + 0 * P

    def sat_gas_cp(self, T_sat):
        return self.linear_1d(2927.6859767045003, 38.796800036445035, T_sat, self.t_sat_ref)

    def sat_gas_viscosity(self, T_sat):
        return self.linear_1d(5.915267643534557e-06, 5.255969282737378e-08, T_sat, self.t_sat_ref)

    def sat_gas_k(self, T_sat):
        return self.linear_1d(0.0185757687618166, 0.00025121652757426993, T_sat, self.t_sat_ref)

    def sat_gas_beta(self, T_sat):
        # Use magnitude here because the OpenMDAO component clips negative
        # Grashof values to zero; a smooth positive beta keeps the NLP usable.
        return np.fabs(self.linear_1d(-0.04532377141136093, 0.0003314694696493343, T_sat, self.t_sat_ref))

    def sat_gas_rho(self, T_sat):
        return self.linear_1d(16.707497068001675, 0.757246780857093, T_sat, self.t_sat_ref)


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


def initial_gas_density_from_pressure(props, pressure, temperature):
    """Find a numeric gas density that is consistent with props.gas_pressure()."""
    if isinstance(props, LNGSurrogateProperties):
        return props.gas_density(pressure, temperature)

    rho_center = float(props.gas_density(pressure, temperature))
    rho_low = max(1e-8, 0.5 * rho_center)
    rho_high = max(2.0 * rho_center, rho_low * 2)

    def residual(rho):
        return float(props.gas_pressure(rho, 1.0, temperature)) - pressure

    while residual(rho_low) > 0:
        rho_high = rho_low
        rho_low *= 0.5
    while residual(rho_high) < 0:
        rho_low = rho_high
        rho_high *= 2.0

    for _ in range(60):
        rho_mid = 0.5 * (rho_low + rho_high)
        if residual(rho_mid) < 0:
            rho_low = rho_mid
        else:
            rho_high = rho_mid
    return 0.5 * (rho_low + rho_high)


def tank_volume(radius, length):
    return 4 / 3 * np.pi * radius**3 + np.pi * radius**2 * length


def liquid_volume_from_height_fraction(radius, length, h_liq_frac, end_cap_depth_ratio=1.0):
    h = h_liq_frac * 2 * radius
    v_caps = np.pi * h**2 / 3 * (3 * radius - h) * end_cap_depth_ratio
    theta = 2 * np.arccos(1 - h / radius)
    v_cyl = radius**2 / 2 * (theta - np.sin(theta)) * length
    return v_caps + v_cyl


def tank_areas_from_height_fraction(radius, length, h_liq_frac, end_cap_depth_ratio=1.0):
    h = h_liq_frac * 2 * radius

    e = (1 - end_cap_depth_ratio**2) ** 0.5
    if e == 0.0:
        e += 1e-9
    elif e == 1.0:
        e -= 1e-9
    a_spheroid = radius**2 * np.pi * (
        2 + end_cap_depth_ratio**2 / e * np.log((1 + e) / (1 - e))
    )
    a_tank = a_spheroid + 2 * np.pi * radius * length

    chord = 2 * np.sqrt(2 * radius * h - h**2)
    theta = 2 * np.arccos(1 - h / radius)

    a_cap_wet = a_spheroid * (
        0.5 * (1 - np.cos(theta / 2)) * end_cap_depth_ratio
        + (theta - np.sin(theta)) / (2 * np.pi) * (1 - end_cap_depth_ratio)
    )
    a_wet = a_cap_wet + theta * radius * length
    a_dry = a_tank - a_wet
    return a_wet, a_dry


def tank_interface_geometry(radius, length, h_liq_frac, end_cap_depth_ratio=1.0):
    h = h_liq_frac * 2 * radius
    chord = 2 * np.sqrt(2 * radius * h - h**2)
    a_interface = np.pi * (chord / 2) ** 2 * end_cap_depth_ratio + chord * length
    return a_interface, chord


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


def heat_leak(radius, length, h_liq_frac, t_env, t_liq, t_gas, n_layers, heat_multiplier):
    a_wet, a_dry = tank_areas_from_height_fraction(radius, length, h_liq_frac)
    q_liq = heat_multiplier * mli_heat_flux(t_env, t_liq, n_layers) * a_wet
    q_gas = heat_multiplier * mli_heat_flux(t_env, t_gas, n_layers) * a_dry
    return q_liq, q_gas


def tank_rhs(
    state,
    design: TankDesign,
    inputs: MissionInputs,
    props: LNGSurrogateProperties = LNGSurrogateProperties(),
    h_liq_frac=None,
    heater_rate_const: float = 1e-3,
    heater_boil_frac: float = 0.1,
    heat_transfer_C_gas_const: float = 0.27 / 4,
    heat_transfer_n_gas_const: float = 0.25,
    heat_transfer_C_liq_const: float = 0.27 / 20,
    heat_transfer_n_liq_const: float = 0.25,
    sigmoid_fac: float = 100.0,
):
    """Return state derivatives and auxiliary values for the transient tank surrogate."""
    m_gas, m_liq, t_gas, t_liq, v_gas, q_add = state

    fill_level = 1 - v_gas / tank_volume(design.radius, design.length)
    if h_liq_frac is None:
        h_liq_frac = fill_level
    rho_liq = props.liquid_density(t_liq)
    pressure = props.gas_pressure(m_gas, v_gas, t_gas)
    q_liq, q_gas = heat_leak(
        design.radius,
        design.length,
        h_liq_frac,
        inputs.t_env,
        t_liq,
        t_gas,
        design.n_layers,
        design.heat_multiplier,
    )

    a_interface, l_interface = tank_interface_geometry(design.radius, design.length, h_liq_frac)

    h_gas = props.gas_h(pressure, t_gas)
    u_gas = props.gas_u(pressure, t_gas)
    cv_gas = props.gas_cv(pressure, t_gas)
    h_liq = props.liquid_h(t_liq)
    u_liq = props.liquid_u(t_liq)
    cp_liq = props.liquid_cp_value(t_liq)
    p_liq = props.liquid_pressure(t_liq)
    beta_liq = props.liquid_beta_value(t_liq)
    visc_liq = props.liquid_viscosity(t_liq)
    k_liq = props.liquid_k(t_liq)

    t_int = props.sat_gas_T(pressure)
    cp_sat_gas = props.sat_gas_cp(t_int)
    visc_sat_gas = props.sat_gas_viscosity(t_int)
    k_sat_gas = props.sat_gas_k(t_int)
    beta_sat_gas = props.sat_gas_beta(t_int)
    rho_sat_gas = props.sat_gas_rho(t_int)

    eps = 1e-12
    prandtl_gas = cp_sat_gas * visc_sat_gas / k_sat_gas
    grashof_gas = (
        GRAV_CONST
        * beta_sat_gas
        * rho_sat_gas**2
        * np.sqrt((t_gas - t_int) ** 2 + eps)
        * l_interface**3
        / visc_sat_gas**2
    )
    nusselt_gas = heat_transfer_C_gas_const * (prandtl_gas * grashof_gas) ** heat_transfer_n_gas_const
    htc_gas_int = k_sat_gas / l_interface * nusselt_gas
    q_gas_int = htc_gas_int * a_interface * (t_gas - t_int)

    prandtl_liq = cp_liq * visc_liq / k_liq
    grashof_liq = (
        GRAV_CONST
        * beta_liq
        * rho_liq**2
        * np.sqrt((t_liq - t_int) ** 2 + eps)
        * l_interface**3
        / visc_liq**2
    )
    nusselt_liq = heat_transfer_C_liq_const * (prandtl_liq * grashof_liq) ** heat_transfer_n_liq_const
    htc_liq_int = k_liq / l_interface * nusselt_liq
    q_liq_int = htc_liq_int * a_interface * (t_liq - t_int)

    m_dot_boil = (q_gas_int + q_liq_int + q_add * heater_boil_frac) / (h_gas - h_liq)
    m_liq_dot = -m_dot_boil - inputs.m_dot_liq_out
    m_gas_dot = m_dot_boil - inputs.m_dot_gas_out
    v_liq_dot = m_liq_dot / rho_liq
    v_gas_dot = -v_liq_dot

    t_liq_dot = (
        q_liq
        - q_liq_int
        + q_add * (1 - heater_boil_frac)
        - pressure * v_liq_dot
        + m_liq_dot * (h_liq - u_liq)
    ) / (m_liq * cp_liq)
    t_gas_dot = (q_gas - q_gas_int - pressure * v_gas_dot + m_gas_dot * (h_gas - u_gas)) / (m_gas * cv_gas)

    d_p_liq_d_t = props.liquid_pressure_dT(t_liq)
    m_dot_bulk_boil = (
        MOLEC_WEIGHT_LNG
        * v_gas
        / (UNIVERSAL_GAS_CONST * t_gas)
        * (
            d_p_liq_d_t * t_liq_dot
            - UNIVERSAL_GAS_CONST
            / MOLEC_WEIGHT_LNG
            * (
                m_gas_dot * t_gas / v_gas
                + m_gas * t_gas_dot / v_gas
                - m_gas * t_gas * v_gas_dot / v_gas**2
            )
        )
    )
    m_dot_bulk_boil = np.sqrt(m_dot_bulk_boil**2 + eps)
    bulk_boil_multiplier = 1 / (1 + np.exp(sigmoid_fac * (pressure - p_liq) / p_liq))

    d_t_sat_d_p = props.sat_gas_T_dP(pressure)
    m_dot_cloud_cond = (
        pressure
        * v_gas
        * MOLEC_WEIGHT_LNG
        / (UNIVERSAL_GAS_CONST * t_gas**2)
        * (
            d_t_sat_d_p
            * UNIVERSAL_GAS_CONST
            / MOLEC_WEIGHT_LNG
            * (
                m_gas_dot * t_gas / v_gas
                + m_gas * t_gas_dot / v_gas
                - m_gas * t_gas * v_gas_dot / v_gas**2
            )
            - t_gas_dot
        )
    )
    m_dot_cloud_cond = np.sqrt(m_dot_cloud_cond**2 + eps)
    cloud_cond_multiplier = 1 / (1 + np.exp(sigmoid_fac * (t_gas - t_int) / t_int))

    m_dot_bbcc = m_dot_bulk_boil * bulk_boil_multiplier - m_dot_cloud_cond * cloud_cond_multiplier
    m_liq_dot = m_liq_dot - m_dot_bbcc
    m_gas_dot = m_gas_dot + m_dot_bbcc
    v_gas_dot = v_gas_dot + m_dot_bbcc / rho_liq

    t_liq_dot = t_liq_dot + (pressure * m_dot_bbcc / rho_liq - m_dot_bbcc * (h_liq - u_liq)) / (m_liq * cp_liq)
    t_gas_dot = t_gas_dot + (-pressure * m_dot_bbcc / rho_liq + m_dot_bbcc * (h_gas - u_gas)) / (m_gas * cv_gas)

    q_liq_bulk_boil = -(h_gas - h_liq) * m_dot_bulk_boil * bulk_boil_multiplier
    t_liq_dot = t_liq_dot + q_liq_bulk_boil / (m_liq * cp_liq)

    q_gas_cloud_cond = (h_gas - h_liq) * m_dot_cloud_cond * cloud_cond_multiplier
    t_gas_dot = t_gas_dot + q_gas_cloud_cond / (m_gas * cv_gas)

    q_add_dot = heater_rate_const * (inputs.p_heater - q_add)

    xdot = [m_gas_dot, m_liq_dot, t_gas_dot, t_liq_dot, v_gas_dot, q_add_dot]
    aux = {
        "pressure": pressure,
        "fill_level": fill_level,
        "q_liq": q_liq,
        "q_gas": q_gas,
        "q_liq_int": q_liq_int,
        "q_gas_int": q_gas_int,
        "m_dot_boil": m_dot_boil,
    }
    return xdot, aux


def build_trajectory_problem(
    n_nodes: int = 41,
    design: TankDesign = TankDesign(),
    inputs: MissionInputs = MissionInputs(),
    initial: InitialState = InitialState(),
    p_max: float = 1.064e6,
    props=None,
):
    """
    Build a direct-transcription AeroSandbox problem for LNG tank dynamics.

    Uses trapezoidal defects with fixed design and mission inputs. Later
    iterations can unfreeze design/input variables and connect them to the
    aircraft trajectory.
    """
    if props is None:
        props = LNGSurrogateProperties()
    opti = asb.Opti()
    time = onp.linspace(0, inputs.duration, n_nodes)
    dt = inputs.duration / (n_nodes - 1)

    volume = float(tank_volume(design.radius, design.length))
    v_gas0 = volume * (1 - initial.fill_level)
    rho_liq0 = props.liquid_density(initial.liquid_temperature)
    m_liq0 = (volume - v_gas0) * rho_liq0
    m_gas0 = initial_gas_density_from_pressure(
        props,
        initial.ullage_pressure,
        initial.ullage_temperature,
    ) * v_gas0

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
    h_liq_frac = opti.variable(init_guess=initial.fill_level, n_vars=n_nodes, lower_bound=1e-3, upper_bound=1 - 1e-3)

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
        f_k, aux_k = tank_rhs(x_k, design, inputs, props, h_liq_frac=h_liq_frac[k])
        rhs.append(f_k)
        pressure.append(aux_k["pressure"])
        fill_level.append(aux_k["fill_level"])
        q_liq.append(aux_k["q_liq"])
        q_gas.append(aux_k["q_gas"])
        m_dot_boil.append(aux_k["m_dot_boil"])

        opti.subject_to(aux_k["pressure"] <= p_max)
        opti.subject_to(aux_k["fill_level"] >= 0.05)
        opti.subject_to(aux_k["fill_level"] <= 0.95)
        opti.subject_to(
            liquid_volume_from_height_fraction(design.radius, design.length, h_liq_frac[k])
            == tank_volume(design.radius, design.length) * aux_k["fill_level"]
        )

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
