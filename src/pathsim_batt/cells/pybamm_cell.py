from __future__ import annotations

from typing import Any

import numpy as np
import pybamm
from pathsim.blocks._block import Block


class _CellBase(Block):
    """Shared base for PyBaMM cell blocks.

    Handles model construction, parameter setup, lazy initialisation, and the
    PathSim dynamic-block protocol.  Subclasses define port labels and
    implement ``update()`` to read specific PyBaMM output variables.
    """

    initial_value = 0.0  # sentinel — makes PathSim call step() each cycle

    def __init__(
        self,
        model: pybamm.BaseBatteryModel | None,
        parameter_values: pybamm.ParameterValues | None,
        initial_soc: float,
        solver: pybamm.BaseSolver | None,
        output_variables: list[str] | None,
        thermal_option: str,
    ) -> None:
        super().__init__()

        self._initial_soc = float(initial_soc)
        self._extra_var_names = list(output_variables or [])
        self.extra_outputs: dict[str, float] = {}
        self._q_nominal: float | None = None

        if model is None:
            model = pybamm.lithium_ion.SPMe(options={"thermal": thermal_option})
        self._model = model

        if parameter_values is None:
            parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values = parameter_values.copy()
        parameter_values["Current function [A]"] = "[input]"
        parameter_values["Ambient temperature [K]"] = "[input]"
        self._parameter_values = parameter_values

        if solver is None:
            solver = pybamm.CasadiSolver(mode="safe")
        self._pybamm_solver = solver

        self._sim: pybamm.Simulation | None = None
        self._initialized = False
        self._stepped = False

    # --- PathSim protocol --------------------------------------------------

    def __len__(self) -> int:
        return 0

    def set_solver(self, Solver: Any, parent: Any, **solver_args: Any) -> None:
        # Create a dummy 1-state engine so PathSim adds this block to its
        # dynamic set and calls step() each cycle.  The engine state is unused.
        if Solver is not None:
            self.engine = Solver(np.array([0.0]), **solver_args)

    def reset(self) -> None:
        self.inputs.reset()
        self.outputs.reset()
        self._sim = None
        self._initialized = False
        self._stepped = False
        self.extra_outputs = {}

    def buffer(self, dt: float) -> None:
        super().buffer(dt)
        self._stepped = False

    def step(self, t: float, dt: float) -> tuple[bool, float, float]:
        if not self._initialized:
            self._initialize()
        if not self._stepped:
            self._sim.step(dt, inputs=self._build_inputs())  # type: ignore[union-attr]
            self._stepped = True
        return True, 0.0, 1.0

    # --- helpers -----------------------------------------------------------

    def _build_inputs(self) -> dict[str, float]:
        T = float(self.inputs[1]) or 298.15
        return {
            "Current function [A]": float(self.inputs[0]),
            "Ambient temperature [K]": T,
        }

    def _compute_soc(self, sol: pybamm.Solution) -> float:
        q_dis = float(sol["Discharge capacity [A.h]"].entries[-1])
        return float(max(0.0, min(1.0, self._initial_soc - q_dis / self._q_nominal)))  # type: ignore[operator]

    def _solution_ready(self) -> bool:
        return (
            self._initialized
            and self._sim is not None
            and self._sim.solution is not None
        )

    def _initialize(self) -> None:
        self._sim = pybamm.Simulation(
            self._model,
            parameter_values=self._parameter_values,
            solver=self._pybamm_solver,
        )
        self._sim.build(initial_soc=self._initial_soc, inputs=self._build_inputs())
        self._q_nominal = float(self._parameter_values["Nominal cell capacity [A.h]"])
        self._initialized = True


class CellElectrical(_CellBase):
    """Cell block — electrical outputs only, external thermal coupling.

    PyBaMM runs with an isothermal assumption; PathSim is responsible for
    integrating the cell temperature ODE.  Wire ``Q_heat`` to a
    ``LumpedThermalModel`` (or similar) and feed its temperature output back
    to ``T_cell``.

    Parameters
    ----------
    model :
        PyBaMM lithium-ion model.  Defaults to ``SPMe(thermal="isothermal")``.
    parameter_values :
        PyBaMM parameter set.  Defaults to ``Chen2020``.
    initial_soc :
        Initial state of charge (0–1).  Default 1.0.
    solver :
        PyBaMM solver.  Defaults to ``CasadiSolver(mode="safe")``.
    output_variables :
        Extra PyBaMM variable names stored in ``block.extra_outputs`` after
        each step.

    Inputs
    ------
    I (0) : current [A], positive = discharge
    T_cell (1) : cell temperature [K] from external PathSim thermal block

    Outputs
    -------
    V (0) : terminal voltage [V]
    Q_heat (1) : X-averaged volumetric heat generation [W m⁻³]
    SOC (2) : state of charge (0–1)
    """

    input_port_labels = {"I": 0, "T_cell": 1}
    output_port_labels = {"V": 0, "Q_heat": 1, "SOC": 2}

    def __init__(
        self,
        model: pybamm.BaseBatteryModel | None = None,
        parameter_values: pybamm.ParameterValues | None = None,
        initial_soc: float = 1.0,
        solver: pybamm.BaseSolver | None = None,
        output_variables: list[str] | None = None,
    ) -> None:
        super().__init__(
            model,
            parameter_values,
            initial_soc,
            solver,
            output_variables,
            thermal_option="isothermal",
        )

    def update(self, t: float) -> None:
        if not self._solution_ready():
            return
        sol = self._sim.solution  # type: ignore[union-attr]
        self.outputs[0] = float(sol["Terminal voltage [V]"].entries[-1])
        self.outputs[1] = float(sol["X-averaged total heating [W.m-3]"].entries[-1])
        self.outputs[2] = self._compute_soc(sol)
        for name in self._extra_var_names:
            self.extra_outputs[name] = float(sol[name].entries[-1])


class CellElectrothermal(_CellBase):
    """Cell block — coupled electrical and thermal model.

    PyBaMM's built-in lumped thermal submodel integrates the cell temperature
    ODE internally.  Supply a time-varying ambient / coolant temperature via
    ``T_amb`` to couple to a pack-level thermal model.

    Parameters
    ----------
    model :
        PyBaMM lithium-ion model.  Defaults to ``SPMe(thermal="lumped")``.
    parameter_values :
        PyBaMM parameter set.  Defaults to ``Chen2020``.
    initial_soc :
        Initial state of charge (0–1).  Default 1.0.
    solver :
        PyBaMM solver.  Defaults to ``CasadiSolver(mode="safe")``.
    output_variables :
        Extra PyBaMM variable names stored in ``block.extra_outputs`` after
        each step.

    Inputs
    ------
    I (0) : current [A], positive = discharge
    T_amb (1) : ambient / coolant temperature [K]

    Outputs
    -------
    V (0) : terminal voltage [V]
    T (1) : cell temperature [K] (computed by PyBaMM)
    Q_heat (2) : X-averaged volumetric heat generation [W m⁻³]
    SOC (3) : state of charge (0–1)
    """

    input_port_labels = {"I": 0, "T_amb": 1}
    output_port_labels = {"V": 0, "T": 1, "Q_heat": 2, "SOC": 3}

    def __init__(
        self,
        model: pybamm.BaseBatteryModel | None = None,
        parameter_values: pybamm.ParameterValues | None = None,
        initial_soc: float = 1.0,
        solver: pybamm.BaseSolver | None = None,
        output_variables: list[str] | None = None,
    ) -> None:
        super().__init__(
            model,
            parameter_values,
            initial_soc,
            solver,
            output_variables,
            thermal_option="lumped",
        )

    def update(self, t: float) -> None:
        if not self._solution_ready():
            return
        sol = self._sim.solution  # type: ignore[union-attr]
        self.outputs[0] = float(sol["Terminal voltage [V]"].entries[-1])
        self.outputs[1] = float(sol["X-averaged cell temperature [K]"].entries[-1])
        self.outputs[2] = float(sol["X-averaged total heating [W.m-3]"].entries[-1])
        self.outputs[3] = self._compute_soc(sol)
        for name in self._extra_var_names:
            self.extra_outputs[name] = float(sol[name].entries[-1])


Cell = CellElectrothermal
