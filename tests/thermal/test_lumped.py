import unittest
import numpy as np

from pathsim.solvers import SSPRK22

from pathsim_batt.thermal import LumpedThermalModel


class TestLumpedThermalModel(unittest.TestCase):

    def test_port_labels(self):
        self.assertEqual(LumpedThermalModel.input_port_labels["Q_dot"], 0)
        self.assertEqual(LumpedThermalModel.input_port_labels["T_amb"], 1)
        self.assertEqual(LumpedThermalModel.output_port_labels["T"], 0)

    def test_default_init(self):
        blk = LumpedThermalModel()
        self.assertAlmostEqual(blk.initial_value[0], 298.15)

    def test_custom_init(self):
        blk = LumpedThermalModel(mass=0.1, Cp=1000.0, UA=1.0, T0=310.0)
        self.assertAlmostEqual(blk.initial_value[0], 310.0)
        self.assertAlmostEqual(blk.mass, 0.1)
        self.assertAlmostEqual(blk.UA, 1.0)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            LumpedThermalModel(mass=0.0)
        with self.assertRaises(ValueError):
            LumpedThermalModel(Cp=-1.0)
        with self.assertRaises(ValueError):
            LumpedThermalModel(UA=-0.1)

    def test_output_equals_state_at_init(self):
        blk = LumpedThermalModel(T0=300.0)
        blk.set_solver(SSPRK22, parent=None)
        blk.update(0.0)
        self.assertAlmostEqual(blk.outputs["T"], 300.0)

    def test_no_heating_no_cooling_temperature_constant(self):
        """With Q_dot=0 and T == T_amb, temperature should not change."""
        blk = LumpedThermalModel(T0=298.15, UA=1.0)
        blk.set_solver(SSPRK22, parent=None)
        blk.inputs["Q_dot"] = 0.0
        blk.inputs["T_amb"] = 298.15
        blk.step(0.0, 1.0)
        blk.update(1.0)
        self.assertAlmostEqual(blk.outputs["T"], 298.15, places=6)

    def test_heating_raises_temperature(self):
        """Positive Q_dot with no cooling should raise temperature."""
        blk = LumpedThermalModel(mass=0.065, Cp=750.0, UA=0.0, T0=298.15)
        blk.set_solver(SSPRK22, parent=None)
        blk.inputs["Q_dot"] = 10.0   # 10 W
        blk.inputs["T_amb"] = 298.15
        blk.step(0.0, 10.0)
        blk.update(10.0)
        self.assertGreater(blk.outputs["T"], 298.15)

    def test_cooling_lowers_temperature(self):
        """Starting above ambient with no heat source should cool down."""
        blk = LumpedThermalModel(mass=0.065, Cp=750.0, UA=1.0, T0=320.0)
        blk.set_solver(SSPRK22, parent=None)
        blk.inputs["Q_dot"] = 0.0
        blk.inputs["T_amb"] = 298.15
        blk.step(0.0, 10.0)
        blk.update(10.0)
        self.assertLess(blk.outputs["T"], 320.0)

    def test_energy_balance(self):
        """Without cooling, ΔT ≈ Q_dot * dt / (m * Cp)."""
        mass, Cp, Q_dot, dt = 0.1, 1000.0, 5.0, 2.0
        blk = LumpedThermalModel(mass=mass, Cp=Cp, UA=0.0, T0=300.0)
        blk.set_solver(SSPRK22, parent=None)
        blk.inputs["Q_dot"] = Q_dot
        blk.inputs["T_amb"] = 300.0
        blk.step(0.0, dt)
        blk.update(dt)
        expected_dT = Q_dot * dt / (mass * Cp)
        self.assertAlmostEqual(blk.outputs["T"] - 300.0, expected_dT, places=4)


if __name__ == "__main__":
    unittest.main()
