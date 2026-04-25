"""
@File    :   test_LNG_properties.py
@Date    :   2023/10/10
@Author  :   Eytan Adler
@Description : Test the code in LNG_properties.py and LNG_properties_MendezRamos.py
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from parameterized import parameterized_class
from openmdao.utils.assert_utils import assert_near_equal

# ==============================================================================
# Extension modules
# ==============================================================================
from lngtank.LNG_properties_MendezRamos import *
from lngtank.LNG_properties import LNGProperties


LNG_prop = LNGProperties()

# For some reason need to wrap function handle in list to properly call it
real_gas_funcs = [
    {"gas_func": [gng_P], "cmplx": True},
    {"gas_func": [gng_rho], "cmplx": True},
    {"gas_func": [gng_cv], "cmplx": True},
    {"gas_func": [gng_cp], "cmplx": True},
    {"gas_func": [gng_u], "cmplx": True},
    {"gas_func": [gng_h], "cmplx": True},
    {"gas_func": [LNG_prop.gng_P], "cmplx": False},
    {"gas_func": [LNG_prop.gng_rho], "cmplx": False},
    {"gas_func": [LNG_prop.gng_cv], "cmplx": False},
    {"gas_func": [LNG_prop.gng_cp], "cmplx": False},
    {"gas_func": [LNG_prop.gng_u], "cmplx": False},
    {"gas_func": [LNG_prop.gng_h], "cmplx": False},
]

sat_funcs = [
    # Function to test    func takes T not P   test second deriv      func is complex safe
    {"sat_func": [lng_P], "temp_input": True, "test_2_deriv": True, "cmplx": True},
    {"sat_func": [lng_h], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [lng_u], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [lng_cp], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [lng_rho], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [lng_k], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [lng_viscosity], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [lng_beta], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [sat_gng_rho], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [sat_gng_h], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [sat_gng_cp], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [sat_gng_k], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [sat_gng_viscosity], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [sat_gng_beta], "temp_input": True, "test_2_deriv": False, "cmplx": True},
    {"sat_func": [sat_gng_T], "temp_input": False, "test_2_deriv": True, "cmplx": True},
    {"sat_func": [LNG_prop.lng_P], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.lng_h], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.lng_u], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.lng_cp], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.lng_rho], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.lng_k], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.lng_viscosity], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.lng_beta], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.sat_gng_rho], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.sat_gng_h], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.sat_gng_cp], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.sat_gng_k], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.sat_gng_viscosity], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.sat_gng_beta], "temp_input": True, "test_2_deriv": True, "cmplx": False},
    {"sat_func": [LNG_prop.sat_gng_T], "temp_input": False, "test_2_deriv": True, "cmplx": False},
]


@parameterized_class(real_gas_funcs)
class RealGasPropertyTestCase(unittest.TestCase):
    def test_scalars(self):
        func = self.gas_func[0]
        if func in [gng_P, LNG_prop.gng_P]:
            first_input = 1.0  # kg/m^3, density
        else:
            first_input = 1e5  # Pa, pressure
        T = 20.1
        out = func(first_input, T)
        out_first, out_T = func(first_input, T, deriv=True)

        self.assertTrue(isinstance(out, float) or len(out) == 1)
        self.assertTrue(isinstance(out_first, float) or len(out_first) == 1)
        self.assertTrue(isinstance(out_T, float) or len(out_T) == 1)

    def test_vectors(self):
        func = self.gas_func[0]
        n = 3
        if func in [gng_P, LNG_prop.gng_P]:
            first_input = np.linspace(1, 2, n)  # kg/m^3, density
        else:
            first_input = np.linspace(1e4, 1e6, n)  # Pa, pressure
        T = np.linspace(10, 30, n)
        out = func(first_input, T)
        out_first, out_T = func(first_input, T, deriv=True)

        self.assertEqual(out.shape, (n,))
        self.assertEqual(out_first.shape, (n,))
        self.assertEqual(out_T.shape, (n,))

    def test_invalid_vector_shapes(self):
        func = self.gas_func[0]

        with self.assertRaises(ValueError):
            _ = func(np.zeros(3), np.zeros(4))

        with self.assertRaises(ValueError):
            _ = func(np.zeros(4), np.zeros(3))

    def test_derivatives(self):
        func = self.gas_func[0]
        dtype = complex if self.cmplx else float
        if func in [gng_P, LNG_prop.gng_P]:
            first_input = np.linspace(1, 2, 3, dtype=dtype)  # kg/m^3, density
        else:
            first_input = np.linspace(1e4, 1e6, 3, dtype=dtype)  # Pa, pressure
        T = np.linspace(20, 30, 3, dtype=dtype)
        out_first, out_T = func(first_input, T, deriv=True)

        step = 1e-200 * 1j if self.cmplx else (1e-7 if func in [gng_P, LNG_prop.gng_P] else 1e-2)
        tol = 1e-12 if self.cmplx else 1e-5
        out_orig = func(first_input, T)

        for i in range(first_input.size):
            first_input[i] += step
            out = func(first_input, T)
            first_input[i] -= step

            deriv = np.imag(out[i]) / np.imag(step) if self.cmplx else (out[i] - out_orig[i]) / step

            assert_near_equal(deriv, np.real(out_first[i]), tolerance=tol)

        step = 1e-200 * 1j if self.cmplx else 1e-7

        for i in range(T.size):
            T[i] += step
            out = func(first_input, T)
            T[i] -= step

            deriv = np.imag(out[i]) / np.imag(step) if self.cmplx else (out[i] - out_orig[i]) / step

            assert_near_equal(deriv, np.real(out_T[i]), tolerance=tol)


@parameterized_class(sat_funcs)
class SaturatedPropertyTestCase(unittest.TestCase):
    def test_scalar(self):
        func = self.sat_func[0]
        if self.temp_input:
            input = 20.1
        else:
            input = 1e5
        out = func(input)
        out_deriv = func(input, deriv=True)

        self.assertTrue(isinstance(out, float))
        self.assertTrue(isinstance(out_deriv, float))

    def test_derivatives(self):
        func = self.sat_func[0]
        if self.temp_input:
            input = np.linspace(10, 30, 3, dtype=complex if self.cmplx else float)
        else:
            input = np.linspace(1e4, 1e6, 3, dtype=complex if self.cmplx else float)
        out_deriv = func(input, deriv=True)

        step = 1e-200 if self.cmplx else 1e-6
        tol = 1e-13 if self.cmplx else 5e-4
        out_orig = func(input)

        for i in range(input.size):
            if self.cmplx:
                input[i] += step * 1j
                out = func(input)
                input[i] -= step * 1j
                deriv_check = np.imag(out[i]) / step
            else:
                input[i] += step
                out_step = func(input)
                input[i] -= step
                deriv_check = (out_step[i] - out_orig[i]) / step

            assert_near_equal(deriv_check, np.real(out_deriv[i]), tolerance=tol)

    def test_second_derivatives(self):
        if not self.test_2_deriv:
            return

        func = self.sat_func[0]
        if self.temp_input:
            input = np.linspace(10, 30, 3, dtype=complex if self.cmplx else float)
        else:
            input = np.linspace(1e4, 1e6, 3, dtype=complex if self.cmplx else float)
        out_deriv = func(input, deriv=2)

        step = 1e-200 if self.cmplx else 1e-6
        tol = 1e-13 if self.cmplx else 5e-4
        out_orig = func(input, deriv=True)

        for i in range(input.size):
            if self.cmplx:
                input[i] += step * 1j
                out = func(input, deriv=True)
                input[i] -= step * 1j
                deriv_check = np.imag(out[i]) / step
            else:
                input[i] += step
                out_step = func(input, deriv=True)
                input[i] -= step
                deriv_check = (out_step[i] - out_orig[i]) / step

            assert_near_equal(deriv_check, np.real(out_deriv[i]), tolerance=tol)


if __name__ == "__main__":
    unittest.main()
