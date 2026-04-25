"""
@File    :   LNG_properties.py
@Date    :   2023/10/10
@Author  :   Eytan Adler
@Description : Surrogate models to compute thermophysical properties of LNG
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import pickle
from pathlib import Path
from time import time

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import scipy.interpolate as interp

try:
    import CoolProp.CoolProp as CP
except ImportError:
    CP = None

class LNGProperties:
    """
    Class for computing LNG properties using surrogate models.

    The default ``backend="coolprop"`` generates the surrogates from CoolProp at
    runtime and caches them in a user-writable location. CoolProp does not expose
    a generic LNG pseudo-fluid, so this model uses methane as the LNG proxy.
    """

    _FLUID = "Methane"
    _T_MIN_GAS = 92.0
    _T_MAX_GAS = 300.0
    _P_MIN_GAS = 1.0e4
    _P_MAX_GAS = 12.5e5
    _N_SAT = 400
    _N_P_GAS = 60
    _N_T_GAS = 160

    def __init__(self, print_output=False, backend="coolprop"):
        self._print_output = print_output
        self.backend = backend

        if self.backend not in ["coolprop", "data"]:
            raise ValueError(f'Unsupported LNG property backend "{self.backend}"')
        if self.backend == "data":
            raise ValueError("backend='data' is not supported for LNG; use backend='coolprop'.")
        if self.backend == "coolprop" and CP is None:
            raise ImportError("CoolProp is required for backend='coolprop'. Install the 'CoolProp' package.")

        cache_dir = self._get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        fluid_tag = self._FLUID.lower()
        sat_dump_file = cache_dir / f"{fluid_tag}_{self.backend}_saturated_property_surrogate_models.pkl"
        gas_dump_file = cache_dir / f"{fluid_tag}_{self.backend}_real_gas_property_surrogate_models.pkl"

        if sat_dump_file.exists():
            if self._print_output:
                print("Reading saturated property surrogate models from cache...", end="")
            t_start = time()
            with sat_dump_file.open("rb") as f:
                self.sat_surrogates = pickle.load(f)
            if self._print_output:
                print(f"done in {time() - t_start} sec")
        else:
            if self._print_output:
                print(f"Training saturated property surrogate models from {self.backend}...", end="")
            t_start = time()
            self.sat_surrogates = self._build_saturated_surrogates()
            if self._print_output:
                print(f"done in {time() - t_start} sec")

            t_start = time()
            with sat_dump_file.open("wb") as f:
                pickle.dump(self.sat_surrogates, f, protocol=pickle.HIGHEST_PROTOCOL)
            if self._print_output:
                print(f"    ...cached surrogate models in {time() - t_start} sec")

        if gas_dump_file.exists():
            if self._print_output:
                print("Reading real gas property surrogate models from cache...", end="")
            t_start = time()
            with gas_dump_file.open("rb") as f:
                self.gas_surrogates = pickle.load(f)
            if self._print_output:
                print(f"done in {time() - t_start} sec")
        else:
            if self._print_output:
                print(f"Training real gas property surrogate models from {self.backend}...", end="")
            t_start = time()
            self.gas_surrogates = self._build_gas_surrogates()
            if self._print_output:
                print(f"done in {time() - t_start} sec")

            t_start = time()
            with gas_dump_file.open("wb") as f:
                pickle.dump(self.gas_surrogates, f, protocol=pickle.HIGHEST_PROTOCOL)
            if self._print_output:
                print(f"    ...cached surrogate models in {time() - t_start} sec")

        self.fd_step_P = 1e-1
        self.fd_step_T = 1e-6
        self.fd_step_rho = 1e-6

    def _get_cache_dir(self):
        cache_root = os.getenv("LNGTANK_CACHE_DIR")
        if cache_root:
            return Path(cache_root)

        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "lngtank"

        return Path.home() / ".cache" / "lngtank"

    def _build_saturated_surrogates(self):
        T_trip = CP.PropsSI("Ttriple", self._FLUID)
        T_crit = CP.PropsSI("Tcrit", self._FLUID)
        T_sat = np.linspace(T_trip + 1e-3, T_crit - 1e-3, self._N_SAT)

        P_sat = self._propssi("P", "T", T_sat, "Q", 0)
        h_liq = self._propssi("Hmass", "T", T_sat, "Q", 0)
        u_liq = self._propssi("Umass", "T", T_sat, "Q", 0)
        cp_liq = self._propssi("Cpmass", "T", T_sat, "Q", 0)
        rho_liq = self._propssi("Dmass", "T", T_sat, "Q", 0)
        k_liq = self._propssi("conductivity", "T", T_sat, "Q", 0)
        viscosity_liq = self._propssi("viscosity", "T", T_sat, "Q", 0)

        rho_vap = self._propssi("Dmass", "T", T_sat, "Q", 1)
        h_vap = self._propssi("Hmass", "T", T_sat, "Q", 1)
        cp_vap = self._propssi("Cpmass", "T", T_sat, "Q", 1)
        k_vap = self._propssi("conductivity", "T", T_sat, "Q", 1)
        viscosity_vap = self._propssi("viscosity", "T", T_sat, "Q", 1)

        sat_surrogates = {
            "lng_P": {"x": T_sat, "y": P_sat},
            "lng_h": {"x": T_sat, "y": h_liq},
            "lng_u": {"x": T_sat, "y": u_liq},
            "lng_cp": {"x": T_sat, "y": cp_liq},
            "lng_rho": {"x": T_sat, "y": rho_liq},
            "lng_k": {"x": T_sat, "y": k_liq},
            "lng_viscosity": {"x": T_sat, "y": viscosity_liq},
            "lng_beta": {"x": T_sat, "y": self._thermal_expansion_from_density(T_sat, rho_liq)},
            "sat_gng_rho": {"x": T_sat, "y": rho_vap},
            "sat_gng_h": {"x": T_sat, "y": h_vap},
            "sat_gng_cp": {"x": T_sat, "y": cp_vap},
            "sat_gng_k": {"x": T_sat, "y": k_vap},
            "sat_gng_viscosity": {"x": T_sat, "y": viscosity_vap},
            "sat_gng_beta": {"x": T_sat, "y": self._thermal_expansion_from_density(T_sat, rho_vap)},
            "sat_gng_T": {"x": P_sat, "y": T_sat},
        }

        for key, val in sat_surrogates.items():
            sat_surrogates[key]["surrogate"] = interp.CubicSpline(val["x"], val["y"], extrapolate=True)
            sat_surrogates[key]["surrogate_deriv"] = sat_surrogates[key]["surrogate"].derivative()
            sat_surrogates[key]["surrogate_second_deriv"] = sat_surrogates[key]["surrogate_deriv"].derivative()

        return sat_surrogates

    def _build_gas_surrogates(self):
        T_crit = CP.PropsSI("Tcrit", self._FLUID)
        T_vals = np.linspace(self._T_MIN_GAS, self._T_MAX_GAS, self._N_T_GAS)
        P_vals = np.linspace(self._P_MIN_GAS, self._P_MAX_GAS, self._N_P_GAS)
        PP, TT = np.meshgrid(P_vals, T_vals, indexing="ij")

        Psat_limit = np.full_like(TT, np.inf, dtype=float)
        subcritical = TT < T_crit
        if np.any(subcritical):
            Psat_limit[subcritical] = self._propssi("P", "T", TT[subcritical], "Q", 1)

        vapor_mask = (~subcritical) | (PP <= Psat_limit * 0.999)
        P = PP[vapor_mask]
        T = TT[vapor_mask]

        vals = {
            "P": P,
            "T": T,
            "rho": self._propssi("Dmass", "P", P, "T", T),
            "cv": self._propssi("Cvmass", "P", P, "T", T),
            "cp": self._propssi("Cpmass", "P", P, "T", T),
            "u": self._propssi("Umass", "P", P, "T", T),
            "h": self._propssi("Hmass", "P", P, "T", T),
        }

        surr_keys = {
            "P": ["rho", "T"],
            "rho": ["P", "T"],
            "cv": ["P", "T"],
            "cp": ["P", "T"],
            "u": ["P", "T"],
            "h": ["P", "T"],
        }

        gas_surrogates = {}
        for key, value_keys in surr_keys.items():
            gas_surrogates[key] = interp.CloughTocher2DInterpolator(
                np.vstack((vals[value_keys[0]], vals[value_keys[1]])).T,
                vals[key],
            )

        return gas_surrogates

    def _thermal_expansion_from_density(self, T, rho):
        rho_spline = interp.CubicSpline(T, rho, extrapolate=True)
        return -rho_spline.derivative()(T) / rho

    def _propssi(self, output, input_1, value_1, input_2, value_2):
        value_1_arr, value_2_arr = np.broadcast_arrays(np.asarray(value_1, dtype=float), np.asarray(value_2, dtype=float))
        out = np.empty(value_1_arr.shape, dtype=float)

        it = np.nditer(
            [value_1_arr, value_2_arr, out],
            flags=["multi_index", "refs_ok", "zerosize_ok"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for val_1, val_2, out_val in it:
            out_val[...] = CP.PropsSI(output, input_1, float(val_1), input_2, float(val_2), self._FLUID)

        if out.shape == ():
            return float(out)
        return out

    def _eval_surrogate(self, name, x, deriv=False):
        """
        Evaluate the surrogate models. If name corresponds to a real gas property, x must be passed
        as a numpy array with dimension (num eval points, 2). If name corresponds to a saturated
        property, x must be passed as either a scalar or 1D numpy array.
        """
        if name in ["P", "rho", "cv", "cp", "u", "h"]:
            if deriv not in [1, True, False]:
                raise ValueError(f"deriv value of {deriv} not supported, must be True, False, or 1")

            x_eval = np.array(x, copy=True)

            if name != "P":
                P_cutoff = 12.4e5
                x_eval[:, 0][x_eval[:, 0] > P_cutoff] = P_cutoff

            val = self.gas_surrogates[name](x_eval)
            if deriv:
                step_first = self.fd_step_rho if name == "P" else self.fd_step_P

                x_first = np.array(x_eval, copy=True)
                x_first[:, 0] += step_first
                val_first_step = self.gas_surrogates[name](x_first)

                x_temp = np.array(x_eval, copy=True)
                x_temp[:, 1] += self.fd_step_T
                val_T_step = self.gas_surrogates[name](x_temp)

                return ((val_first_step - val) / step_first, (val_T_step - val) / self.fd_step_T)
            return val

        if deriv not in [1, 2, True, False]:
            raise ValueError(f"deriv value of {deriv} not supported, must be True, False, 1, or 2")

        is_float = not isinstance(x, np.ndarray)
        if deriv == 1:
            val = self.sat_surrogates[name]["surrogate_deriv"](x)
        elif deriv == 2:
            val = self.sat_surrogates[name]["surrogate_second_deriv"](x)
        else:
            val = self.sat_surrogates[name]["surrogate"](x)
        return val.item() if is_float else val

    def gng_P(self, rho, T, deriv=False):
        if isinstance(rho, np.ndarray) and isinstance(T, np.ndarray) and rho.shape != T.shape:
            raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")
        return self._eval_surrogate("P", np.vstack((rho, T)).T, deriv=deriv)

    def gng_rho(self, P, T, deriv=False):
        if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
            raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")
        return self._eval_surrogate("rho", np.vstack((P, T)).T, deriv=deriv)

    def gng_cv(self, P, T, deriv=False):
        if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
            raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")
        return self._eval_surrogate("cv", np.vstack((P, T)).T, deriv=deriv)

    def gng_cp(self, P, T, deriv=False):
        if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
            raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")
        return self._eval_surrogate("cp", np.vstack((P, T)).T, deriv=deriv)

    def gng_u(self, P, T, deriv=False):
        if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
            raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")
        return self._eval_surrogate("u", np.vstack((P, T)).T, deriv=deriv)

    def gng_h(self, P, T, deriv=False):
        if isinstance(P, np.ndarray) and isinstance(T, np.ndarray) and P.shape != T.shape:
            raise ValueError("Pressure and temperature must have the same shape if they are both numpy arrays")
        return self._eval_surrogate("h", np.vstack((P, T)).T, deriv=deriv)

    def lng_P(self, T, deriv=False):
        return self._eval_surrogate("lng_P", T, deriv=deriv)

    def lng_h(self, T, deriv=False):
        return self._eval_surrogate("lng_h", T, deriv=deriv)

    def lng_u(self, T, deriv=False):
        return self._eval_surrogate("lng_u", T, deriv=deriv)

    def lng_cp(self, T, deriv=False):
        return self._eval_surrogate("lng_cp", T, deriv=deriv)

    def lng_rho(self, T, deriv=False):
        return self._eval_surrogate("lng_rho", T, deriv=deriv)

    def lng_k(self, T, deriv=False):
        return self._eval_surrogate("lng_k", T, deriv=deriv)

    def lng_viscosity(self, T, deriv=False):
        return self._eval_surrogate("lng_viscosity", T, deriv=deriv)

    def lng_beta(self, T, deriv=False):
        return self._eval_surrogate("lng_beta", T, deriv=deriv)

    def sat_gng_rho(self, T, deriv=False):
        return self._eval_surrogate("sat_gng_rho", T, deriv=deriv)

    def sat_gng_h(self, T, deriv=False):
        return self._eval_surrogate("sat_gng_h", T, deriv=deriv)

    def sat_gng_cp(self, T, deriv=False):
        return self._eval_surrogate("sat_gng_cp", T, deriv=deriv)

    def sat_gng_k(self, T, deriv=False):
        return self._eval_surrogate("sat_gng_k", T, deriv=deriv)

    def sat_gng_viscosity(self, T, deriv=False):
        return self._eval_surrogate("sat_gng_viscosity", T, deriv=deriv)

    def sat_gng_beta(self, T, deriv=False):
        return self._eval_surrogate("sat_gng_beta", T, deriv=deriv)

    def sat_gng_T(self, P, deriv=False):
        return self._eval_surrogate("sat_gng_T", P, deriv=deriv)
