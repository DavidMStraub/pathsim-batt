import unittest

import pybamm

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
        self.assertTrue(hasattr(CellElectrical, "initial_value"))
        self.assertTrue(hasattr(CellElectrothermal, "initial_value"))

    def test_len_zero(self):
        self.assertEqual(len(CellElectrical()), 0)
        self.assertEqual(len(CellElectrothermal()), 0)

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

    def test_reset_clears_state(self):
        for cls in (CellElectrical, CellElectrothermal):
            cell = cls()
            cell._sim = object()
            cell.reset()
            self.assertIsNone(cell._sim)


def _advance(cell, dt):
    """Simulate one PathSim timestep: buffer → step → update."""
    cell.buffer(dt)
    cell.step(0.0, dt)
    cell.update(dt)


class TestElectrical(unittest.TestCase):
    """Integration tests for CellElectrical — runs PyBaMM."""

    def setUp(self):
        self.cell = CellElectrical(initial_soc=1.0)
        self.cell.inputs[0] = 1.0  # I
        self.cell.inputs[1] = 298.15  # T_cell

    def test_lazy_init_on_first_step(self):
        self.assertIsNone(self.cell._sim)
        _advance(self.cell, 1.0)
        self.assertIsNotNone(self.cell._sim)

    def test_outputs_in_range(self):
        _advance(self.cell, 1.0)
        self.assertGreater(self.cell.outputs[0], 3.0)  # V
        self.assertLess(self.cell.outputs[0], 4.3)
        self.assertGreaterEqual(self.cell.outputs[1], 0.0)  # Q_heat
        self.assertGreater(self.cell.outputs[2], 0.0)  # SOC
        self.assertLessEqual(self.cell.outputs[2], 1.0)

    def test_step_returns_success(self):
        self.cell.set_solver(None, parent=None)
        self.cell.buffer(1.0)
        success, error, scale = self.cell.step(0.0, 1.0)
        self.assertTrue(success)

    def test_soc_decreases_on_discharge(self):
        _advance(self.cell, 1.0)
        soc_0 = self.cell.outputs[2]
        for _ in range(5):
            _advance(self.cell, 60.0)
        self.assertLess(self.cell.outputs[2], soc_0)


class TestElectrothermal(unittest.TestCase):
    """Integration tests for CellElectrothermal — runs PyBaMM."""

    def setUp(self):
        self.cell = CellElectrothermal(initial_soc=1.0)
        self.cell.inputs[0] = 1.0  # I
        self.cell.inputs[1] = 298.15  # T_amb

    def test_lazy_init_on_first_step(self):
        self.assertIsNone(self.cell._sim)
        _advance(self.cell, 1.0)
        self.assertIsNotNone(self.cell._sim)

    def test_outputs_in_range(self):
        _advance(self.cell, 1.0)
        self.assertGreater(self.cell.outputs[0], 3.0)  # V
        self.assertLess(self.cell.outputs[0], 4.3)
        self.assertGreater(self.cell.outputs[1], 250.0)  # T
        self.assertLess(self.cell.outputs[1], 400.0)
        self.assertGreaterEqual(self.cell.outputs[2], 0.0)  # Q_heat
        self.assertGreater(self.cell.outputs[3], 0.0)  # SOC
        self.assertLessEqual(self.cell.outputs[3], 1.0)

    def test_step_returns_success(self):
        self.cell.set_solver(None, parent=None)
        self.cell.buffer(1.0)
        success, error, scale = self.cell.step(0.0, 1.0)
        self.assertTrue(success)

    def test_soc_decreases_on_discharge(self):
        _advance(self.cell, 1.0)
        soc_0 = self.cell.outputs[3]
        for _ in range(5):
            _advance(self.cell, 60.0)
        self.assertLess(self.cell.outputs[3], soc_0)


if __name__ == "__main__":
    unittest.main()
