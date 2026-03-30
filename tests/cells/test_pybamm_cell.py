import unittest

import numpy as np
import pybamm
from pathsim import Connection, Simulation
from pathsim.blocks import Constant
from pathsim.solvers import ESDIRK43

from pathsim_batt.cells import (
    Cell,
    CellElectrical,
    CellElectrothermal,
)


class TestPorts(unittest.TestCase):
    def test_electrical_input_labels(self):
        self.assertEqual(CellElectrical.input_port_labels["I"], 0)
        self.assertEqual(CellElectrical.input_port_labels["T_cell"], 1)

    def test_electrical_output_labels(self):
        self.assertEqual(CellElectrical.output_port_labels["V"], 0)
        self.assertEqual(CellElectrical.output_port_labels["Q_heat"], 1)
        self.assertEqual(CellElectrical.output_port_labels["SOC"], 2)

    def test_electrothermal_input_labels(self):
        self.assertEqual(CellElectrothermal.input_port_labels["I"], 0)
        self.assertEqual(CellElectrothermal.input_port_labels["T_amb"], 1)

    def test_electrothermal_output_labels(self):
        self.assertEqual(CellElectrothermal.output_port_labels["V"], 0)
        self.assertEqual(CellElectrothermal.output_port_labels["T"], 1)
        self.assertEqual(CellElectrothermal.output_port_labels["Q_heat"], 2)
        self.assertEqual(CellElectrothermal.output_port_labels["SOC"], 3)

    def test_alias(self):
        self.assertIs(Cell, CellElectrothermal)

    def test_is_dynamic(self):
        self.assertTrue(hasattr(CellElectrical(), "initial_value"))
        self.assertTrue(hasattr(CellElectrothermal(), "initial_value"))

    def test_len_zero(self):
        cell_e = CellElectrical()
        cell_e.set_solver(ESDIRK43, None)
        self.assertEqual(len(cell_e), 3)  # V, Q_heat, SOC
        cell_et = CellElectrothermal()
        cell_et.set_solver(ESDIRK43, None)
        self.assertEqual(len(cell_et), 4)  # V, T, Q_heat, SOC

    def test_current_always_input(self):
        pv = pybamm.ParameterValues("Chen2020")
        for cls in (CellElectrical, CellElectrothermal):
            cell = cls(parameter_values=pv)
            self.assertIsInstance(
                cell._parameter_values["Current function [A]"],
                pybamm.InputParameter,
            )

    def test_custom_soc(self):
        self.assertAlmostEqual(CellElectrical(initial_soc=0.5)._initial_soc, 0.5)
        self.assertAlmostEqual(CellElectrothermal(initial_soc=0.8)._initial_soc, 0.8)

    def test_reset_clears_extra_outputs(self):
        for cls in (CellElectrical, CellElectrothermal):
            cell = cls()
            cell.extra_outputs = {"foo": 1.0}
            cell.reset()
            self.assertEqual(cell.extra_outputs, {})

    def test_initial_value_is_numpy_array(self):
        for cls in (CellElectrical, CellElectrothermal):
            cell = cls()
            self.assertIsInstance(cell.initial_value, np.ndarray)
            self.assertGreater(len(cell.initial_value), 1)

    def test_has_casadi_rhs(self):
        """CasADi RHS is compiled and callable at construction time."""
        for cls in (CellElectrical, CellElectrothermal):
            cell = cls()
            self.assertIsNotNone(cell._casadi_rhs)


class TestElectrical(unittest.TestCase):
    """Integration tests for CellElectrical — PathSim integrates the PyBaMM ODE."""

    def _make_simulation(self, cell, I, T_cell):
        """Create a Simulation with the cell and constant inputs."""
        I_src = Constant(I)
        T_src = Constant(T_cell)
        return Simulation(
            blocks=[I_src, T_src, cell],
            connections=[
                Connection(I_src, cell["I"]),
                Connection(T_src, cell["T_cell"]),
            ],
            dt=1.0,
            Solver=ESDIRK43,
        )

    def setUp(self):
        self.cell = CellElectrical(initial_soc=1.0)
        self.sim = self._make_simulation(self.cell, 1.0, 298.15)

    def test_outputs_in_range(self):
        self.sim.run(1)
        self.assertGreater(self.cell.outputs[0], 3.0)  # V
        self.assertLess(self.cell.outputs[0], 4.3)
        self.assertGreaterEqual(self.cell.outputs[1], 0.0)  # Q_heat
        self.assertGreater(self.cell.outputs[2], 0.0)  # SOC
        self.assertLessEqual(self.cell.outputs[2], 1.0)

    def test_step_returns_success(self):
        self.sim.run(1)
        # Simulation completed without error
        self.assertIsNotNone(self.cell.outputs)

    def test_soc_decreases_on_discharge(self):
        self.sim.run(1)
        soc_0 = self.cell.outputs[2]
        self.sim.run(60)
        self.assertLess(self.cell.outputs[2], soc_0)

    def test_pathsim_state_advances(self):
        """The PathSim engine state changes after a step (not a stub)."""
        self.sim.run(1)
        state_before = self.cell.engine.state.copy()
        self.sim.run(2)
        self.assertFalse(np.allclose(self.cell.engine.state, state_before))


class TestElectrothermal(unittest.TestCase):
    """Integration tests for CellElectrothermal — PathSim integrates the PyBaMM ODE."""

    def _make_simulation(self, cell, I, T_amb):
        """Create a Simulation with the cell and constant inputs."""
        I_src = Constant(I)
        T_src = Constant(T_amb)
        return Simulation(
            blocks=[I_src, T_src, cell],
            connections=[
                Connection(I_src, cell["I"]),
                Connection(T_src, cell["T_amb"]),
            ],
            dt=1.0,
            Solver=ESDIRK43,
        )

    def setUp(self):
        self.cell = CellElectrothermal(initial_soc=1.0)
        self.sim = self._make_simulation(self.cell, 1.0, 298.15)

    def test_outputs_in_range(self):
        self.sim.run(1)
        self.assertGreater(self.cell.outputs[0], 3.0)  # V
        self.assertLess(self.cell.outputs[0], 4.3)
        self.assertGreater(self.cell.outputs[1], 250.0)  # T
        self.assertLess(self.cell.outputs[1], 400.0)
        self.assertGreaterEqual(self.cell.outputs[2], 0.0)  # Q_heat
        self.assertGreater(self.cell.outputs[3], 0.0)  # SOC
        self.assertLessEqual(self.cell.outputs[3], 1.0)

    def test_step_returns_success(self):
        self.sim.run(1)
        # Simulation completed without error
        self.assertIsNotNone(self.cell.outputs)

    def test_soc_decreases_on_discharge(self):
        self.sim.run(1)
        soc_0 = self.cell.outputs[3]
        self.sim.run(60)
        self.assertLess(self.cell.outputs[3], soc_0)

    def test_pathsim_state_advances(self):
        """The PathSim engine state changes after a step (not a stub)."""
        self.sim.run(1)
        state_before = self.cell.engine.state.copy()
        self.sim.run(2)
        self.assertFalse(np.allclose(self.cell.engine.state, state_before))


if __name__ == "__main__":
    unittest.main()
