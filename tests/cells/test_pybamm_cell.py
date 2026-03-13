import unittest
import numpy as np
import pybamm

from pathsim_batt.cells import PyBaMMCell


class TestPyBaMMCellInit(unittest.TestCase):

    def test_port_labels(self):
        self.assertEqual(PyBaMMCell.input_port_labels["I"], 0)
        self.assertEqual(PyBaMMCell.input_port_labels["T_ext"], 1)
        self.assertEqual(PyBaMMCell.output_port_labels["V"], 0)
        self.assertEqual(PyBaMMCell.output_port_labels["Q_heat"], 1)
        self.assertEqual(PyBaMMCell.output_port_labels["SOC"], 2)

    def test_is_dynamic(self):
        """Block must have initial_value so PathSim treats it as dynamic."""
        self.assertTrue(hasattr(PyBaMMCell, "initial_value"))

    def test_len_zero(self):
        cell = PyBaMMCell()
        self.assertEqual(len(cell), 0)

    def test_default_construction(self):
        cell = PyBaMMCell()
        self.assertIsNone(cell._sim)
        self.assertFalse(cell._initialized)
        self.assertAlmostEqual(cell._initial_soc, 1.0)

    def test_custom_initial_soc(self):
        cell = PyBaMMCell(initial_soc=0.5)
        self.assertAlmostEqual(cell._initial_soc, 0.5)

    def test_current_always_input(self):
        """parameter_values must have current as '[input]' regardless of what was passed."""
        pv = pybamm.ParameterValues("Chen2020")
        cell = PyBaMMCell(parameter_values=pv)
        self.assertEqual(cell._parameter_values["Current function [A]"], "[input]")

    def test_reset_clears_state(self):
        cell = PyBaMMCell()
        cell._initialized = True
        cell._sim = object()
        cell.reset()
        self.assertFalse(cell._initialized)
        self.assertIsNone(cell._sim)


class TestPyBaMMCellSimulation(unittest.TestCase):
    """Integration tests — these actually run PyBaMM, so are slower."""

    def setUp(self):
        self.cell = PyBaMMCell(initial_soc=1.0)
        self.cell.inputs["I"]     = 1.0   # 1 A discharge
        self.cell.inputs["T_ext"] = 298.15

    def test_set_solver_initializes_pybamm(self):
        self.cell.set_solver(None, parent=None)
        self.assertTrue(self.cell._initialized)
        self.assertIsNotNone(self.cell._sim)

    def test_update_populates_outputs(self):
        self.cell.set_solver(None, parent=None)
        self.cell.update(0.0)
        V      = self.cell.outputs["V"]
        Q_heat = self.cell.outputs["Q_heat"]
        SOC    = self.cell.outputs["SOC"]
        # Sanity ranges for a Chen2020 LG M50 at full charge
        self.assertGreater(V, 3.0)
        self.assertLess(V, 4.3)
        self.assertGreaterEqual(Q_heat, 0.0)
        self.assertGreater(SOC, 0.0)
        self.assertLessEqual(SOC, 1.0)

    def test_step_advances_state(self):
        self.cell.set_solver(None, parent=None)
        self.cell.update(0.0)
        V_before = self.cell.outputs["V"]
        self.cell.step(0.0, 10.0)   # 10-second step
        self.cell.update(10.0)
        V_after = self.cell.outputs["V"]
        # Discharging at 1 A should lower the voltage slightly
        self.assertLess(V_after, V_before + 1e-6)

    def test_step_returns_success(self):
        self.cell.set_solver(None, parent=None)
        success, error, scale = self.cell.step(0.0, 1.0)
        self.assertTrue(success)

    def test_soc_decreases_on_discharge(self):
        self.cell.set_solver(None, parent=None)
        self.cell.update(0.0)
        soc_0 = self.cell.outputs["SOC"]
        for _ in range(5):
            self.cell.step(0.0, 60.0)   # 5 × 1-minute steps
        self.cell.update(300.0)
        soc_1 = self.cell.outputs["SOC"]
        self.assertLess(soc_1, soc_0)


if __name__ == "__main__":
    unittest.main()
