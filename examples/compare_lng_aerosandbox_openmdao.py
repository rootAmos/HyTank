"""
Compare the AeroSandbox LNG trajectory surrogate against the OpenMDAO LNG tank.

Both models are run with the same geometry, initial pressure/temperature/fill,
mission duration, heat environment, heater power, and extraction flow rates.
The printed metrics make it clear where the current AeroSandbox surrogate does
and does not reproduce the OpenMDAO transient.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om

from lngtank.LNG_tank import LNGTank
from lngtank.asb_properties_interpolants import CoolPropGridInterpolants
from lngtank.aerosandbox_tank import (
    InitialState,
    LNGSurrogateProperties,
    MissionInputs,
    TankDesign,
    build_trajectory_problem,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


@dataclass(frozen=True)
class ComparisonCase:
    duration_hr: float = 10.0
    num_nodes: int = 101
    radius: float = 2.75
    length: float = 2.0
    fill_level_init: float = 0.9
    ullage_pressure_init: float = 1.064e6
    ullage_temperature_init: float = 151.8
    liquid_temperature_init: float = 145.8
    p_heater: float = 1000.0
    m_dot_gas_out_kg_hr: float = 0.0
    m_dot_liq_out_kg_hr: float = 700.0
    t_env: float = 350.0
    n_layers: float = 20.0
    max_expected_operating_pressure_bar: float = 10.64


def run_openmdao(case: ComparisonCase):
    p = om.Problem(reports=False)
    p.model.add_subsystem(
        "tank",
        LNGTank(
            num_nodes=case.num_nodes,
            fill_level_init=case.fill_level_init,
            ullage_P_init=case.ullage_pressure_init,
            ullage_T_init=case.ullage_temperature_init,
            liquid_T_init=case.liquid_temperature_init,
        ),
        promotes=["*"],
    )
    p.model.nonlinear_solver = om.NewtonSolver(iprint=0, solve_subsystems=True, maxiter=10)
    p.model.nonlinear_solver.options["err_on_non_converge"] = True
    p.model.linear_solver = om.DirectSolver()

    p.setup()

    p.set_val("thermals.boil_off.integ.duration", case.duration_hr, units="h")
    p.set_val("radius", case.radius, units="m")
    p.set_val("length", case.length, units="m")
    p.set_val("P_heater", case.p_heater, units="W")
    p.set_val("m_dot_gas_out", case.m_dot_gas_out_kg_hr, units="kg/h")
    p.set_val("m_dot_liq_out", case.m_dot_liq_out_kg_hr, units="kg/h")
    p.set_val("T_env", case.t_env, units="K")
    p.set_val("N_layers", case.n_layers)
    p.set_val("environment_design_pressure", 1, units="atm")
    p.set_val("max_expected_operating_pressure", case.max_expected_operating_pressure_bar, units="bar")
    p.set_val("vacuum_gap", 0.1, units="m")

    p.run_model()

    return {
        "time_hr": np.linspace(0, case.duration_hr, case.num_nodes),
        "P_bar": p.get_val("P", units="bar"),
        "fill_level": p.get_val("fill_level"),
        "T_gas_K": p.get_val("T_gas", units="K"),
        "T_liq_K": p.get_val("T_liq", units="K"),
        "m_gas_kg": p.get_val("m_gas", units="kg"),
        "m_liq_kg": p.get_val("m_liq", units="kg"),
        "Q_add_W": p.get_val("thermals.boil_off.ode.Q_add", units="W"),
        "Q_gas_W": p.get_val("thermals.heat_leak.Q_gas", units="W"),
        "Q_liq_W": p.get_val("thermals.heat_leak.Q_liq", units="W"),
    }


def run_aerosandbox(case: ComparisonCase, props=None):
    design = TankDesign(
        radius=case.radius,
        length=case.length,
        n_layers=case.n_layers,
        heat_multiplier=2.0,
    )
    inputs = MissionInputs(
        duration=case.duration_hr * 3600,
        t_env=case.t_env,
        p_heater=case.p_heater,
        m_dot_liq_out=case.m_dot_liq_out_kg_hr / 3600,
        m_dot_gas_out=case.m_dot_gas_out_kg_hr / 3600,
    )
    initial = InitialState(
        ullage_pressure=case.ullage_pressure_init,
        ullage_temperature=case.ullage_temperature_init,
        liquid_temperature=case.liquid_temperature_init,
        fill_level=case.fill_level_init,
    )

    problem = build_trajectory_problem(
        n_nodes=case.num_nodes,
        design=design,
        inputs=inputs,
        initial=initial,
        p_max=case.max_expected_operating_pressure_bar * 1e5,
        props=props,
    )
    sol = problem["opti"].solve(verbose=False)

    return {
        "time_hr": problem["time"] / 3600,
        "P_bar": np.array([sol.value(v) for v in problem["aux"]["pressure"]]) / 1e5,
        "fill_level": np.array([sol.value(v) for v in problem["aux"]["fill_level"]]),
        "T_gas_K": np.array(sol.value(problem["states"]["T_gas"])),
        "T_liq_K": np.array(sol.value(problem["states"]["T_liq"])),
        "m_gas_kg": np.array(sol.value(problem["states"]["m_gas"])),
        "m_liq_kg": np.array(sol.value(problem["states"]["m_liq"])),
        "Q_add_W": np.array(sol.value(problem["states"]["Q_add"])),
        "Q_gas_W": np.array([sol.value(v) for v in problem["aux"]["Q_gas"]]),
        "Q_liq_W": np.array([sol.value(v) for v in problem["aux"]["Q_liq"]]),
    }


def summarize(openmdao_results, aerosandbox_results, label):
    variables = [
        ("P_bar", "bar"),
        ("fill_level", "-"),
        ("T_gas_K", "K"),
        ("T_liq_K", "K"),
        ("m_gas_kg", "kg"),
        ("m_liq_kg", "kg"),
        ("Q_add_W", "W"),
        ("Q_gas_W", "W"),
        ("Q_liq_W", "W"),
    ]

    print(f"LNG Tank OpenMDAO vs AeroSandbox ({label})")
    print("=" * 72)
    print(
        f"{'Variable':<14} {'OM init':>12} {'ASB init':>12} "
        f"{'OM final':>12} {'ASB final':>12} {'final err':>12} {'RMS err':>12}"
    )
    print("-" * 72)

    for name, units in variables:
        om_val = np.asarray(openmdao_results[name], dtype=float)
        asb_val = np.asarray(aerosandbox_results[name], dtype=float)
        diff = asb_val - om_val
        print(
            f"{name + ' [' + units + ']':<14} "
            f"{om_val[0]:>12.6g} {asb_val[0]:>12.6g} "
            f"{om_val[-1]:>12.6g} {asb_val[-1]:>12.6g} "
            f"{diff[-1]:>12.6g} {np.sqrt(np.mean(diff**2)):>12.6g}"
        )


def plot_evolution(openmdao_results, aerosandbox_results, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variables = [
        ("P_bar", "Pressure [bar]"),
        ("fill_level", "Fill level [-]"),
        ("T_gas_K", "Ullage temperature [K]"),
        ("T_liq_K", "Liquid temperature [K]"),
        ("m_gas_kg", "Gas mass [kg]"),
        ("m_liq_kg", "Liquid mass [kg]"),
        ("Q_gas_W", "Gas-side heat leak [W]"),
        ("Q_liq_W", "Liquid-side heat leak [W]"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
    axes = axes.ravel()
    for ax, (name, ylabel) in zip(axes, variables):
        ax.plot(openmdao_results["time_hr"], openmdao_results[name], label="OpenMDAO", linewidth=2)
        ax.plot(
            aerosandbox_results["time_hr"],
            aerosandbox_results[name],
            "--",
            label="AeroSandbox interpolants",
            linewidth=2,
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    for ax in axes[-2:]:
        ax.set_xlabel("Time [hr]")
    axes[0].legend(loc="best")
    fig.suptitle("LNG Tank Transient Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    case = ComparisonCase()
    openmdao_results = run_openmdao(case)
    interpolant_results = None
    backends = [
        ("local linear property fit", LNGSurrogateProperties()),
        ("CoolProp CasADi interpolants", CoolPropGridInterpolants()),
    ]
    for label, props in backends:
        print()
        try:
            aerosandbox_results = run_aerosandbox(case, props=props)
        except Exception as exc:
            print(f"LNG Tank OpenMDAO vs AeroSandbox ({label})")
            print("=" * 72)
            print(f"FAILED: {type(exc).__name__}: {exc}")
            continue
        summarize(openmdao_results, aerosandbox_results, label)
        if label == "CoolProp CasADi interpolants":
            interpolant_results = aerosandbox_results

    if interpolant_results is not None:
        output_path = plot_evolution(
            openmdao_results,
            interpolant_results,
            OUTPUT_DIR / "lng_openmdao_vs_aerosandbox_interpolants.png",
        )
        print()
        print(f"Saved transient comparison plot: {output_path}")


if __name__ == "__main__":
    main()
