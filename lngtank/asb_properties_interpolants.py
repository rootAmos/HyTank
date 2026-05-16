"""
CasADi-compatible methane property interpolants for the AeroSandbox LNG tank.

CoolProp is used only while constructing the grids. Runtime evaluations are
CasADi interpolants, so the same methods can be used with numeric values or
CasADi/AeroSandbox symbolic expressions.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

try:
    import CoolProp.CoolProp as CP
except ImportError:  # pragma: no cover - exercised by import environment
    CP = None


@dataclass(frozen=True)
class GridSpec:
    """Default grid sized for the current AeroSandbox LNG tank operating point."""

    fluid: str = "Methane"
    p_min_gas: float = 5.0e4
    p_max_gas: float = 1.064e6
    t_min_gas: float = 151.0
    t_max_gas: float = 220.0
    t_min_sat: float = 92.0
    t_max_sat: float = 185.0
    n_p_gas: int = 28
    n_t_gas: int = 34
    n_rho_gas: int = 28
    n_sat: int = 80


class CoolPropGridInterpolants:
    """
    Methane property backend with the LNGProperties public method names.

    The gas pressure grid is built on ``(rho, T)``. Other gas properties are
    built on ``(P, T)``. Saturated liquid/vapor properties are one-dimensional
    grids in saturation temperature except ``sat_gng_T(P)``.
    """

    def __init__(self, spec: GridSpec | None = None, interpolant_type: str = "linear"):
        if CP is None:
            raise ImportError("CoolProp is required to build LNG property grids.")

        self.spec = spec or GridSpec()
        self.interpolant_type = interpolant_type
        self._gas_funcs = {}
        self._gas_grad_funcs = {}
        self._sat_funcs = {}
        self._sat_grad_funcs = {}

        self._build_gas_interpolants()
        self._build_saturated_interpolants()

    def _propssi(self, output, input_1, value_1, input_2, value_2):
        value_1_arr, value_2_arr = np.broadcast_arrays(np.asarray(value_1), np.asarray(value_2))
        out = np.empty(value_1_arr.shape, dtype=float)

        it = np.nditer(
            [value_1_arr, value_2_arr, out],
            flags=["multi_index", "refs_ok", "zerosize_ok"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for val_1, val_2, out_val in it:
            out_val[...] = CP.PropsSI(
                output,
                input_1,
                float(val_1),
                input_2,
                float(val_2),
                self.spec.fluid,
            )
        return out

    def _make_interpolant(self, name, grids, values):
        func = ca.interpolant(
            name,
            self.interpolant_type,
            [np.asarray(grid, dtype=float) for grid in grids],
            np.asarray(values, dtype=float).ravel(order="F"),
        )
        x = ca.MX.sym(f"{name}_x", len(grids))
        y = func(x)
        grad = ca.Function(f"{name}_grad", [x], [ca.jacobian(y, x)])
        return func, grad

    def _build_gas_interpolants(self):
        spec = self.spec
        p_grid = np.linspace(spec.p_min_gas, spec.p_max_gas, spec.n_p_gas)
        t_grid = np.linspace(spec.t_min_gas, spec.t_max_gas, spec.n_t_gas)
        pp, tt = np.meshgrid(p_grid, t_grid, indexing="ij")

        gas_outputs = {
            "rho": "Dmass",
            "cv": "Cvmass",
            "cp": "Cpmass",
            "u": "Umass",
            "h": "Hmass",
        }
        for name, output in gas_outputs.items():
            values = self._propssi(output, "P", pp, "T", tt)
            self._gas_funcs[name], self._gas_grad_funcs[name] = self._make_interpolant(
                f"gng_{name}_pt",
                [p_grid, t_grid],
                values,
            )

        rho_min = float(CP.PropsSI("Dmass", "P", spec.p_min_gas, "T", spec.t_max_gas, spec.fluid))
        rho_max = float(CP.PropsSI("Dmass", "P", spec.p_max_gas, "T", spec.t_min_gas, spec.fluid))
        rho_grid = np.linspace(max(1e-4, 0.95 * rho_min), rho_max, spec.n_rho_gas)
        rr, tt_rho = np.meshgrid(rho_grid, t_grid, indexing="ij")
        values = self._propssi("P", "Dmass", rr, "T", tt_rho)
        self._gas_funcs["P"], self._gas_grad_funcs["P"] = self._make_interpolant(
            "gng_P_rhot",
            [rho_grid, t_grid],
            values,
        )

    def _build_saturated_interpolants(self):
        spec = self.spec
        t_triple = float(CP.PropsSI("Ttriple", spec.fluid))
        t_crit = float(CP.PropsSI("Tcrit", spec.fluid))
        t_min = max(spec.t_min_sat, t_triple + 1e-3)
        t_max = min(spec.t_max_sat, t_crit - 1e-3)
        t_sat = np.linspace(t_min, t_max, spec.n_sat)

        p_sat = self._propssi("P", "T", t_sat, "Q", 0)
        rho_liq = self._propssi("Dmass", "T", t_sat, "Q", 0)
        rho_vap = self._propssi("Dmass", "T", t_sat, "Q", 1)

        sat_values = {
            "lng_P": p_sat,
            "lng_h": self._propssi("Hmass", "T", t_sat, "Q", 0),
            "lng_u": self._propssi("Umass", "T", t_sat, "Q", 0),
            "lng_cp": self._propssi("Cpmass", "T", t_sat, "Q", 0),
            "lng_rho": rho_liq,
            "lng_k": self._propssi("conductivity", "T", t_sat, "Q", 0),
            "lng_viscosity": self._propssi("viscosity", "T", t_sat, "Q", 0),
            "lng_beta": self._thermal_expansion_from_density(t_sat, rho_liq),
            "sat_gng_rho": rho_vap,
            "sat_gng_h": self._propssi("Hmass", "T", t_sat, "Q", 1),
            "sat_gng_cp": self._propssi("Cpmass", "T", t_sat, "Q", 1),
            "sat_gng_k": self._propssi("conductivity", "T", t_sat, "Q", 1),
            "sat_gng_viscosity": self._propssi("viscosity", "T", t_sat, "Q", 1),
            "sat_gng_beta": self._thermal_expansion_from_density(t_sat, rho_vap),
        }

        for name, values in sat_values.items():
            self._sat_funcs[name], self._sat_grad_funcs[name] = self._make_interpolant(
                name,
                [t_sat],
                values,
            )

        self._sat_funcs["sat_gng_T"], self._sat_grad_funcs["sat_gng_T"] = self._make_interpolant(
            "sat_gng_T",
            [p_sat],
            t_sat,
        )

    @staticmethod
    def _thermal_expansion_from_density(t_values, rho_values):
        return -np.gradient(rho_values, t_values, edge_order=2) / rho_values

    @staticmethod
    def _is_symbolic(value):
        return isinstance(value, (ca.MX, ca.SX))

    @staticmethod
    def _scalar_or_array(value, shape):
        if shape == ():
            return float(np.asarray(value).reshape(-1)[0])
        return value.reshape(shape)

    def _eval_1d(self, name, x, deriv=False):
        if deriv not in [False, True, 1]:
            raise ValueError("Only first derivatives are supported by the CasADi interpolant backend.")

        func = self._sat_grad_funcs[name] if deriv else self._sat_funcs[name]
        if self._is_symbolic(x):
            if x.numel() > 1:
                return ca.vertcat(*[func(x[i]) for i in range(x.numel())])
            return func(x)

        x_arr = np.asarray(x, dtype=float)
        out = np.array([float(func(float(v))) for v in x_arr.reshape(-1)])
        return self._scalar_or_array(out, x_arr.shape)

    def _eval_2d(self, name, x0, x1, deriv=False):
        if deriv not in [False, True, 1]:
            raise ValueError("Only first derivatives are supported by the CasADi interpolant backend.")

        if self._is_symbolic(x0) or self._is_symbolic(x1):
            if not self._is_symbolic(x0):
                x0 = ca.DM(x0)
            if not self._is_symbolic(x1):
                x1 = ca.DM(x1)
            if x0.numel() > 1 or x1.numel() > 1:
                if x0.numel() != x1.numel():
                    raise ValueError("Symbolic interpolant inputs must have the same length.")
                if deriv:
                    grads = [self._gas_grad_funcs[name](ca.vertcat(x0[i], x1[i])) for i in range(x0.numel())]
                    return ca.vertcat(*[grad[0] for grad in grads]), ca.vertcat(*[grad[1] for grad in grads])
                return ca.vertcat(*[self._gas_funcs[name](ca.vertcat(x0[i], x1[i])) for i in range(x0.numel())])

            x = ca.vertcat(x0, x1)
            if deriv:
                grad = self._gas_grad_funcs[name](x)
                return grad[0], grad[1]
            return self._gas_funcs[name](x)

        x0_arr, x1_arr = np.broadcast_arrays(np.asarray(x0, dtype=float), np.asarray(x1, dtype=float))
        if deriv:
            out0 = np.empty(x0_arr.shape, dtype=float)
            out1 = np.empty(x0_arr.shape, dtype=float)
            for idx in np.ndindex(x0_arr.shape):
                grad = self._gas_grad_funcs[name]([float(x0_arr[idx]), float(x1_arr[idx])])
                out0[idx] = float(grad[0])
                out1[idx] = float(grad[1])
            return self._scalar_or_array(out0, x0_arr.shape), self._scalar_or_array(out1, x0_arr.shape)

        out = np.empty(x0_arr.shape, dtype=float)
        for idx in np.ndindex(x0_arr.shape):
            out[idx] = float(self._gas_funcs[name]([float(x0_arr[idx]), float(x1_arr[idx])]))
        return self._scalar_or_array(out, x0_arr.shape)

    def gng_P(self, rho, T, deriv=False):
        return self._eval_2d("P", rho, T, deriv=deriv)

    def gng_rho(self, P, T, deriv=False):
        return self._eval_2d("rho", P, T, deriv=deriv)

    def gng_cv(self, P, T, deriv=False):
        return self._eval_2d("cv", P, T, deriv=deriv)

    def gng_cp(self, P, T, deriv=False):
        return self._eval_2d("cp", P, T, deriv=deriv)

    def gng_u(self, P, T, deriv=False):
        return self._eval_2d("u", P, T, deriv=deriv)

    def gng_h(self, P, T, deriv=False):
        return self._eval_2d("h", P, T, deriv=deriv)

    def lng_P(self, T, deriv=False):
        return self._eval_1d("lng_P", T, deriv=deriv)

    def lng_h(self, T, deriv=False):
        return self._eval_1d("lng_h", T, deriv=deriv)

    def lng_u(self, T, deriv=False):
        return self._eval_1d("lng_u", T, deriv=deriv)

    def lng_cp(self, T, deriv=False):
        return self._eval_1d("lng_cp", T, deriv=deriv)

    def lng_rho(self, T, deriv=False):
        return self._eval_1d("lng_rho", T, deriv=deriv)

    def lng_k(self, T, deriv=False):
        return self._eval_1d("lng_k", T, deriv=deriv)

    def lng_viscosity(self, T, deriv=False):
        return self._eval_1d("lng_viscosity", T, deriv=deriv)

    def lng_beta(self, T, deriv=False):
        return self._eval_1d("lng_beta", T, deriv=deriv)

    def sat_gng_T(self, P, deriv=False):
        return self._eval_1d("sat_gng_T", P, deriv=deriv)

    def sat_gng_rho(self, T, deriv=False):
        return self._eval_1d("sat_gng_rho", T, deriv=deriv)

    def sat_gng_h(self, T, deriv=False):
        return self._eval_1d("sat_gng_h", T, deriv=deriv)

    def sat_gng_cp(self, T, deriv=False):
        return self._eval_1d("sat_gng_cp", T, deriv=deriv)

    def sat_gng_k(self, T, deriv=False):
        return self._eval_1d("sat_gng_k", T, deriv=deriv)

    def sat_gng_viscosity(self, T, deriv=False):
        return self._eval_1d("sat_gng_viscosity", T, deriv=deriv)

    def sat_gng_beta(self, T, deriv=False):
        return self._eval_1d("sat_gng_beta", T, deriv=deriv)

    def gas_pressure(self, m_gas, v_gas, T_gas):
        return self.gng_P(m_gas / v_gas, T_gas)

    def gas_density(self, P, T_gas):
        return self.gng_rho(P, T_gas)

    def gas_h(self, P, T_gas):
        return self.gng_h(P, T_gas)

    def gas_u(self, P, T_gas):
        return self.gng_u(P, T_gas)

    def gas_cv(self, P, T_gas):
        return self.gng_cv(P, T_gas)

    def liquid_density(self, T_liq):
        return self.lng_rho(T_liq)

    def liquid_h(self, T_liq):
        return self.lng_h(T_liq)

    def liquid_u(self, T_liq):
        return self.lng_u(T_liq)

    def liquid_cp_value(self, T_liq):
        return self.lng_cp(T_liq)

    def liquid_pressure(self, T_liq):
        return self.lng_P(T_liq)

    def liquid_pressure_dT(self, T_liq):
        return self.lng_P(T_liq, deriv=True)

    def liquid_beta_value(self, T_liq):
        return self.lng_beta(T_liq)

    def liquid_viscosity(self, T_liq):
        return self.lng_viscosity(T_liq)

    def liquid_k(self, T_liq):
        return self.lng_k(T_liq)

    def sat_gas_T(self, P):
        return self.sat_gng_T(P)

    def sat_gas_T_dP(self, P):
        return self.sat_gng_T(P, deriv=True)

    def sat_gas_cp(self, T_sat):
        return self.sat_gng_cp(T_sat)

    def sat_gas_viscosity(self, T_sat):
        return self.sat_gng_viscosity(T_sat)

    def sat_gas_k(self, T_sat):
        return self.sat_gng_k(T_sat)

    def sat_gas_beta(self, T_sat):
        beta = self.sat_gng_beta(T_sat)
        if self._is_symbolic(beta):
            return ca.fabs(beta)
        return np.abs(beta)

    def sat_gas_rho(self, T_sat):
        return self.sat_gng_rho(T_sat)


def _rel_err(model, reference):
    return abs(model - reference) / max(abs(reference), 1e-30)


def _smoke_test():
    p_gas = 1.064e6
    t_gas = 151.8
    t_liq = 145.8
    props = CoolPropGridInterpolants()
    fluid = props.spec.fluid

    rho_gas_ref = CP.PropsSI("Dmass", "P", p_gas, "T", t_gas, fluid)
    t_sat_ref = CP.PropsSI("T", "P", p_gas, "Q", 1, fluid)
    dt_beta = 1e-3
    rho_liq_low = CP.PropsSI("Dmass", "T", t_liq - dt_beta, "Q", 0, fluid)
    rho_liq_high = CP.PropsSI("Dmass", "T", t_liq + dt_beta, "Q", 0, fluid)
    rho_sat_low = CP.PropsSI("Dmass", "T", t_sat_ref - dt_beta, "Q", 1, fluid)
    rho_sat_high = CP.PropsSI("Dmass", "T", t_sat_ref + dt_beta, "Q", 1, fluid)
    beta_liq_ref = -(rho_liq_high - rho_liq_low) / (2 * dt_beta) / CP.PropsSI(
        "Dmass",
        "T",
        t_liq,
        "Q",
        0,
        fluid,
    )
    beta_sat_ref = -(rho_sat_high - rho_sat_low) / (2 * dt_beta) / CP.PropsSI(
        "Dmass",
        "T",
        t_sat_ref,
        "Q",
        1,
        fluid,
    )
    checks = [
        ("gng_P", props.gng_P(rho_gas_ref, t_gas), p_gas),
        ("gng_rho", props.gng_rho(p_gas, t_gas), rho_gas_ref),
        ("gng_cv", props.gng_cv(p_gas, t_gas), CP.PropsSI("Cvmass", "P", p_gas, "T", t_gas, fluid)),
        ("gng_u", props.gng_u(p_gas, t_gas), CP.PropsSI("Umass", "P", p_gas, "T", t_gas, fluid)),
        ("gng_h", props.gng_h(p_gas, t_gas), CP.PropsSI("Hmass", "P", p_gas, "T", t_gas, fluid)),
        ("lng_P", props.lng_P(t_liq), CP.PropsSI("P", "T", t_liq, "Q", 0, fluid)),
        ("lng_h", props.lng_h(t_liq), CP.PropsSI("Hmass", "T", t_liq, "Q", 0, fluid)),
        ("lng_u", props.lng_u(t_liq), CP.PropsSI("Umass", "T", t_liq, "Q", 0, fluid)),
        ("lng_cp", props.lng_cp(t_liq), CP.PropsSI("Cpmass", "T", t_liq, "Q", 0, fluid)),
        ("lng_rho", props.lng_rho(t_liq), CP.PropsSI("Dmass", "T", t_liq, "Q", 0, fluid)),
        ("lng_k", props.lng_k(t_liq), CP.PropsSI("conductivity", "T", t_liq, "Q", 0, fluid)),
        ("lng_viscosity", props.lng_viscosity(t_liq), CP.PropsSI("viscosity", "T", t_liq, "Q", 0, fluid)),
        ("lng_beta", props.lng_beta(t_liq), beta_liq_ref),
    ]

    t_sat = props.sat_gng_T(p_gas)
    checks.extend(
        [
            ("sat_gng_T", t_sat, CP.PropsSI("T", "P", p_gas, "Q", 1, fluid)),
            ("sat_gng_cp", props.sat_gng_cp(t_sat), CP.PropsSI("Cpmass", "T", t_sat, "Q", 1, fluid)),
            ("sat_gng_k", props.sat_gng_k(t_sat), CP.PropsSI("conductivity", "T", t_sat, "Q", 1, fluid)),
            (
                "sat_gng_viscosity",
                props.sat_gng_viscosity(t_sat),
                CP.PropsSI("viscosity", "T", t_sat, "Q", 1, fluid),
            ),
            ("sat_gng_rho", props.sat_gng_rho(t_sat), CP.PropsSI("Dmass", "T", t_sat, "Q", 1, fluid)),
            ("sat_gng_beta", props.sat_gng_beta(t_sat), beta_sat_ref),
        ]
    )

    x = ca.MX.sym("x")
    symbolic_value = props.lng_h(x)
    print(f"Symbolic check: {symbolic_value}")
    for name, model, reference in checks:
        print(f"{name:18s} model={model: .8e}  ref={reference: .8e}  rel_err={_rel_err(model, reference):.3e}")


if __name__ == "__main__":
    _smoke_test()
