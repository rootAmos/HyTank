"""
AeroSandbox/CasADi-compatible LNG property fits from Mendez Ramos.

This module ports the explicit equations from
``lngtank/LNG_properties_MendezRamos.py`` for use inside AeroSandbox/CasADi
optimization graphs. It avoids NumPy array mutation and type checks that do not
work on symbolic variables.
"""

from dataclasses import dataclass

import aerosandbox.numpy as np

from lngtank.utilities.constants import MOLEC_WEIGHT_LNG, UNIVERSAL_GAS_CONST


_T_SHIFT = 27.6691
_SAT_GNG_T_P_CAP = 1235172.0
_SAT_GNG_T_CAP = 32.459


def _check_deriv_bool(deriv):
    if not isinstance(deriv, bool):
        raise ValueError("deriv input must be a boolean")


def _check_deriv_bool_or_2(deriv):
    if not (isinstance(deriv, bool) or deriv == 2):
        raise ValueError("deriv input must be a boolean or 2")


def _tanh_fit_2d(P, T, base, terms, deriv=False):
    """Evaluate y = 1e3 * (base + sum(w_i * tanh(b_i + p_i * P_MPa + t_i * T)))."""
    P_MPa = P * 1e-6

    if deriv:
        y_P_MPa = 0.0
        y_T = 0.0
        for b_i, p_i, t_i, w_i in terms:
            arg_i = b_i + p_i * P_MPa + t_i * T
            sech2_i = 1 - np.tanh(arg_i) ** 2
            y_P_MPa += w_i * p_i * sech2_i
            y_T += w_i * t_i * sech2_i
        return y_P_MPa * 1e-3, y_T * 1e3

    y = base
    for b_i, p_i, t_i, w_i in terms:
        y += w_i * np.tanh(b_i + p_i * P_MPa + t_i * T)
    return y * 1e3


_GNG_CV_TERMS = (
    (0.3990654435825, 0.275976950953691, -0.0364541001467881, -70.3037835837473),
    (-4.86826687998567, -2.62940079620686, 0.216404289884235, -3.65980723192549),
    (1.12548975067467, 0.269571399186571, -0.0349250996335972, 16.4087880702894),
    (-5.57528770123298, -3.1347851189393, 0.276378205247149, -3.5451526649205),
    (-0.768555052808218, -0.415187001771689, 0.0303660811195935, 1.59237668513045),
    (0.271020942398391, -1.67288064390926, 0.0673489649704582, 16.3356631788945),
    (-1.10142968268782, 0.0083174017874869, 0.0108662672630239, 2.87332235554376),
    (1.90064900956529, 1.17456681024276, -0.0902336799991085, -32.0695317395356),
    (-0.673064219823259, -0.943525497821755, 0.0734160926798799, -162.92972771138),
    (-1.6061497898911, -1.99739736255923, 0.108199114343877, -12.804152874995),
    (1.73534543350612, 0.388202467775416, -0.0616088045146524, 8.56885046215908),
    (2.70166390032434, 1.3326759135354, -0.118211564739528, 16.8828056394326),
    (-9.32984593347698, 9.42488755971789, 0.0784916020511085, 0.000773883721821057),
    (1.2207418073194, 1.13189377766378, -0.0841043542652104, -11.6665802642623),
    (2.58665026763982, 1.84291010959674, -0.13970364770731, -43.8044401310621),
)

_GNG_U_TERMS = (
    (0.233539747755315, 0.948951297504785, -0.0499745330828293, -1446.64509367172),
    (2.05471470504367, 2.13093505892378, -0.119572061260962, 399.518468943879),
    (0.398892133197, 0.587703772663913, -0.00336363426965824, -333.71693491459),
    (0.143237719165029, 0.992347038797521, -0.0143964824374372, -228.44619912214),
    (-3.96616131435837, -2.74657042229242, 0.186453818992668, 30.515033222315),
    (1.44405264544866, -0.645489048931109, -0.0104476083443945, 74.6722187603707),
    (2.02006732389655, 3.10126879273836, -0.132381179154892, -58.2024392498572),
    (0.455915191247191, -0.295895155010423, -0.00270733720731719, -1594.44683837146),
    (-1.42121486333193, 0.135982980631872, 0.0473542542463267, 69.542870910495),
    (-0.662630708916202, -0.206113999100671, 0.00414455453741901, 1174.42287038803),
    (0.56740916395517, 1.67371525829477, -0.0218895513528701, 26.0663709626282),
    (0.237517777100042, 0.1589500143037, -0.0106519272801902, 478.510727358678),
    (1.54182270996619, 0.754447306182754, -0.0541064232047049, 213.76235040309),
    (2.62921984539398, 1.72079624611917, -0.0971068814297633, 47.1707429768787),
    (-1.04212680836281, -0.484939056845666, 0.0116518536560464, -14.8049823561287),
)

_GNG_H_TERMS = (
    (1.73579390297524, -0.511704824190866, -0.00328063298246929, -245.023690422356),
    (-0.17506458742044, -0.0787067450602815, 0.0043943295092826, -762.587308746834),
    (-1.49793344088779, -2.75869130445277, 0.110698824574587, -117.086388339974),
    (-1.37669485784553, 0.133730746273731, 0.00649135107315252, 2084.0234411658),
    (-1.1405914153055, -1.88672855275111, 0.0981301522587858, 3494.85395981366),
    (0.446124093572671, 0.429585607469454, -0.0285787427235227, -79.8462693619194),
    (-2.49683893292792, 0.398927054757526, 0.0128111795517157, -257.346057454727),
    (-1.34037612046496, -1.82172820931808, 0.0975510622080853, -2180.32225879755),
    (-0.00454329936396595, -0.0840846337708169, 0.00192700682148055, 3232.88193434466),
    (3.25989647210865, -0.817575328835318, -0.00790145526484188, 143.039857861931),
    (1.2138688950769, -5.5187467484401, 0.115937604475061, 57.936905487342),
    (-4.2841473959043, -3.44784008416792, 0.225314720754115, 141.765833322109),
    (-0.749714454876193, 0.59904718534549, 0.015852118328226, 91.6603988756128),
    (-3.56659915773571, -3.26552973737454, 0.201884080139387, -354.856864121115),
    (-0.368559368031245, -4.14838409559996, 0.1417710518596, -407.462486717321),
)


def gng_P(rho, T, deriv=False):
    """Pressure of gaseous natural gas from the ideal gas law, in Pa."""
    _check_deriv_bool(deriv)
    if deriv:
        return (
            T * UNIVERSAL_GAS_CONST / MOLEC_WEIGHT_LNG + rho * 0,
            rho * UNIVERSAL_GAS_CONST / MOLEC_WEIGHT_LNG + T * 0,
        )
    return rho * T * UNIVERSAL_GAS_CONST / MOLEC_WEIGHT_LNG


def gng_rho(P, T, deriv=False):
    """Density of gaseous natural gas from the ideal gas law, in kg/m^3."""
    _check_deriv_bool(deriv)
    if deriv:
        return (
            1 / T / UNIVERSAL_GAS_CONST * MOLEC_WEIGHT_LNG + P * 0,
            -P / T**2 / UNIVERSAL_GAS_CONST * MOLEC_WEIGHT_LNG,
        )
    return P / T / UNIVERSAL_GAS_CONST * MOLEC_WEIGHT_LNG


def gng_cv(P, T, deriv=False):
    """Specific heat at constant volume of gaseous natural gas, in J/(kg-K)."""
    _check_deriv_bool(deriv)
    return _tanh_fit_2d(P, T, 56.0992565764207, _GNG_CV_TERMS, deriv=deriv)


def gng_u(P, T, deriv=False):
    """Internal energy of gaseous natural gas, in J/kg."""
    _check_deriv_bool(deriv)
    return _tanh_fit_2d(P, T, 673.611983193655, _GNG_U_TERMS, deriv=deriv)


def gng_h(P, T, deriv=False):
    """Enthalpy of gaseous natural gas, in J/kg."""
    _check_deriv_bool(deriv)
    return _tanh_fit_2d(P, T, 1281.75444728572, _GNG_H_TERMS, deriv=deriv)


def lng_P(T, deriv=False):
    """Pressure of saturated LNG, in Pa."""
    _check_deriv_bool_or_2(deriv)
    if deriv == 2:
        return 4.2644 * 5.2644 * 0.0138 * T**3.2644
    if deriv:
        return 5.2644 * 0.0138 * T**4.2644
    return 0.0138 * T**5.2644


def lng_h(T, deriv=False):
    """Enthalpy of saturated LNG, in J/kg."""
    _check_deriv_bool(deriv)
    dT = T - _T_SHIFT
    if deriv:
        return 16864.749 + 2 * 893.59208 * dT + 3 * 103.63758 * dT**2 + 4 * 7.756004 * dT**3
    return -371985.2 + 16864.749 * T + 893.59208 * dT**2 + 103.63758 * dT**3 + 7.756004 * dT**4


def lng_u(T, deriv=False):
    """Internal energy of saturated LNG, in J/kg."""
    _check_deriv_bool(deriv)
    dT = T - _T_SHIFT
    if deriv:
        return (
            15183.043
            + 2 * 614.10133 * dT
            + 3 * 40.845478 * dT**2
            + 4 * 9.1394916 * dT**3
            + 5 * 1.8297788 * dT**4
            + 6 * 0.1246228 * dT**5
        )
    return (
        -334268
        + 15183.043 * T
        + 614.10133 * dT**2
        + 40.845478 * dT**3
        + 9.1394916 * dT**4
        + 1.8297788 * dT**5
        + 0.1246228 * dT**6
    )


def lng_cp(T, deriv=False):
    """Specific heat at constant pressure of saturated LNG, in J/(kg-K)."""
    _check_deriv_bool(deriv)
    dT = T - _T_SHIFT
    denom = 0.0002684 - 7.6143e-6 * T - 2.5759e-7 * dT**2
    if deriv:
        return -(-7.6143e-6 - 2 * 2.5759e-7 * dT) / denom**2
    return 1 / denom


def lng_rho(T, deriv=False):
    """Density of saturated LNG, in kg/m^3."""
    _check_deriv_bool(deriv)
    dT = T - _T_SHIFT
    if deriv:
        return (
            -2.0067591
            - 2 * 0.1067411 * dT
            - 3 * 0.0085915 * dT**2
            - 4 * 0.0019879 * dT**3
            - 5 * 0.0003988 * dT**4
            - 6 * 2.7179e-5 * dT**5
        )
    return (
        115.53291
        - 2.0067591 * T
        - 0.1067411 * dT**2
        - 0.0085915 * dT**3
        - 0.0019879 * dT**4
        - 0.0003988 * dT**5
        - 2.7179e-5 * dT**6
    )


def lng_k(T, deriv=False):
    """Thermal conductivity of saturated LNG, in W/(m-K)."""
    _check_deriv_bool(deriv)
    if deriv:
        return 2 * -0.00011780751646212103 * T + 0.005025141732149419
    return -0.00011780751646212103 * T**2 + 0.005025141732149419 * T + 0.05028285917065491


def lng_viscosity(T, deriv=False):
    """Dynamic viscosity of saturated LNG, in Pa-s."""
    _check_deriv_bool(deriv)
    if deriv:
        return 2 * 6.12354624060653e-8 * T - 3.7677099409542915e-6
    return 6.12354624060653e-8 * T**2 - 3.7677099409542915e-6 * T + 6.501471081626786e-5


def lng_beta(T, deriv=False):
    """Coefficient of thermal expansion of saturated LNG, in 1/K."""
    _check_deriv_bool(deriv)
    if deriv:
        return 2 * 0.0002876573504834155 * T - 0.009922769421340946
    return 0.0002876573504834155 * T**2 - 0.009922769421340946 * T + 0.09728151340667823


def sat_gng_rho(T, deriv=False):
    """Density of saturated gaseous natural gas, in kg/m^3."""
    _check_deriv_bool(deriv)
    dT = T - _T_SHIFT
    if deriv:
        return (
            1.2864736
            + 2 * 0.1140157 * dT
            + 3 * 0.0086723 * dT**2
            + 4 * 0.0019006 * dT**3
            + 5 * 0.0003805 * dT**4
            + 6 * 2.5918e-5 * dT**5
        )
    return (
        -28.97599
        + 1.2864736 * T
        + 0.1140157 * dT**2
        + 0.0086723 * dT**3
        + 0.0019006 * dT**4
        + 0.0003805 * dT**5
        + 2.5918e-5 * dT**6
    )


def sat_gng_cp(T, deriv=False):
    """Specific heat at constant pressure of saturated gaseous natural gas, in J/(kg-K)."""
    _check_deriv_bool(deriv)
    dT = T - _T_SHIFT
    exponent = 6.445199 + 0.1249361 * T + 0.0125811 * dT**2 + 0.0027137 * dT**3 + 0.0006249 * dT**4 + 4.8352e-5 * dT**5
    if deriv:
        return np.exp(exponent) * (
            0.1249361 + 2 * 0.0125811 * dT + 3 * 0.0027137 * dT**2 + 4 * 0.0006249 * dT**3 + 5 * 4.8352e-5 * dT**4
        )
    return np.exp(exponent)


def sat_gng_k(T, deriv=False):
    """Thermal conductivity of saturated gaseous natural gas, in W/(m-K)."""
    _check_deriv_bool(deriv)
    dT = T - _T_SHIFT
    denom = 110.21937 - 2.6596443 * T - 0.0153377 * dT**2 - 0.0088632 * dT**3
    if deriv:
        return -(-2.6596443 - 2 * 0.0153377 * dT - 3 * 0.0088632 * dT**2) / denom**2
    return 1 / denom


def sat_gng_viscosity(T, deriv=False):
    """Dynamic viscosity of saturated gaseous natural gas, in Pa-s."""
    _check_deriv_bool(deriv)
    dT = T - _T_SHIFT
    denom = 1582670.2 - 34545.242 * T - 211.73722 * dT**2 - 283.70972 * dT**3 - 18.848797 * dT**4
    if deriv:
        return -(-34545.242 - 2 * 211.73722 * dT - 3 * 283.70972 * dT**2 - 4 * 18.848797 * dT**3) / denom**2
    return 1 / denom


def sat_gng_beta(T, deriv=False):
    """Coefficient of thermal expansion of saturated gaseous natural gas, in 1/K."""
    _check_deriv_bool(deriv)
    if deriv:
        return -1 / T**2
    return 1 / T


def sat_gng_T(P, deriv=False):
    """Temperature of saturated gaseous natural gas at pressure P, in K."""
    _check_deriv_bool_or_2(deriv)
    dP = P - 598825.0
    if deriv == 2:
        val = -2 * 5.85e-12 + 6 * 3.292e-18 * dP - 12 * 1.246e-24 * dP**2 + 20 * 2.053e-29 * dP**3 - 30 * 3.463e-35 * dP**4
        return np.where(P > _SAT_GNG_T_P_CAP, 0.0, val)
    if deriv:
        val = 9.5791e-6 - 2 * 5.85e-12 * dP + 3 * 3.292e-18 * dP**2 - 4 * 1.246e-24 * dP**3 + 5 * 2.053e-29 * dP**4 - 6 * 3.463e-35 * dP**5
        return np.where(P > _SAT_GNG_T_P_CAP, 0.0, val)
    val = 22.509518 + 9.5791e-6 * P - 5.85e-12 * dP**2 + 3.292e-18 * dP**3 - 1.246e-24 * dP**4 + 2.053e-29 * dP**5 - 3.463e-35 * dP**6
    return np.where(P > _SAT_GNG_T_P_CAP, _SAT_GNG_T_CAP, val)


@dataclass(frozen=True)
class MendezRamosAeroSandboxProperties:
    """Adapter with the method names used by ``lngtank.aerosandbox_tank``."""

    def gas_pressure(self, m_gas, v_gas, T_gas):
        return gng_P(m_gas / v_gas, T_gas)

    def gas_h(self, P, T_gas):
        return gng_h(P, T_gas)

    def gas_u(self, P, T_gas):
        return gng_u(P, T_gas)

    def gas_cv(self, P, T_gas):
        return gng_cv(P, T_gas)

    def liquid_h(self, T_liq):
        return lng_h(T_liq)

    def liquid_u(self, T_liq):
        return lng_u(T_liq)

    def liquid_cp_value(self, T_liq):
        return lng_cp(T_liq)

    def liquid_density(self, T_liq):
        return lng_rho(T_liq)

    def liquid_pressure(self, T_liq):
        return lng_P(T_liq)

    def liquid_pressure_dT(self, T_liq):
        return lng_P(T_liq, deriv=True)

    def liquid_beta_value(self, T_liq):
        return lng_beta(T_liq)

    def liquid_viscosity(self, T_liq):
        return lng_viscosity(T_liq)

    def liquid_k(self, T_liq):
        return lng_k(T_liq)

    def sat_gas_T(self, P):
        return sat_gng_T(P)

    def sat_gas_T_dP(self, P):
        return sat_gng_T(P, deriv=True)

    def sat_gas_cp(self, T_sat):
        return sat_gng_cp(T_sat)

    def sat_gas_viscosity(self, T_sat):
        return sat_gng_viscosity(T_sat)

    def sat_gas_k(self, T_sat):
        return sat_gng_k(T_sat)

    def sat_gas_beta(self, T_sat):
        return sat_gng_beta(T_sat)

    def sat_gas_rho(self, T_sat):
        return sat_gng_rho(T_sat)


def _smoke_test():
    import lngtank.LNG_properties_MendezRamos as ref

    P = 1.064e6
    Tg = 151.8
    Tl = 145.8
    rho_g = gng_rho(P, Tg)
    T_sat = sat_gng_T(P)

    cases = (
        ("gng_P", gng_P(rho_g, Tg), ref.gng_P(float(rho_g), Tg)),
        ("gng_rho", rho_g, ref.gng_rho(P, Tg)),
        ("gng_cv", gng_cv(P, Tg), ref.gng_cv(P, Tg)),
        ("gng_u", gng_u(P, Tg), ref.gng_u(P, Tg)),
        ("gng_h", gng_h(P, Tg), ref.gng_h(P, Tg)),
        ("lng_P", lng_P(Tl), ref.lng_P(Tl)),
        ("lng_h", lng_h(Tl), ref.lng_h(Tl)),
        ("lng_u", lng_u(Tl), ref.lng_u(Tl)),
        ("lng_cp", lng_cp(Tl), ref.lng_cp(Tl)),
        ("lng_rho", lng_rho(Tl), ref.lng_rho(Tl)),
        ("lng_k", lng_k(Tl), ref.lng_k(Tl)),
        ("lng_viscosity", lng_viscosity(Tl), ref.lng_viscosity(Tl)),
        ("lng_beta", lng_beta(Tl), ref.lng_beta(Tl)),
        ("sat_gng_T", T_sat, ref.sat_gng_T(P)),
        ("sat_gng_cp", sat_gng_cp(T_sat), ref.sat_gng_cp(float(T_sat))),
        ("sat_gng_k", sat_gng_k(T_sat), ref.sat_gng_k(float(T_sat))),
        ("sat_gng_viscosity", sat_gng_viscosity(T_sat), ref.sat_gng_viscosity(float(T_sat))),
        ("sat_gng_beta", sat_gng_beta(T_sat), ref.sat_gng_beta(float(T_sat))),
        ("sat_gng_rho", sat_gng_rho(T_sat), ref.sat_gng_rho(float(T_sat))),
    )

    for name, value, ref_value in cases:
        print(f"{name:17s} value={float(value): .12e} ref={float(ref_value): .12e} delta={float(value - ref_value): .3e}")


if __name__ == "__main__":
    _smoke_test()
