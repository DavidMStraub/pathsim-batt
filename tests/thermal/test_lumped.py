import unittest

from pathsim import Connection, Simulation
from pathsim.blocks import Constant
from pathsim.solvers import SSPRK22

from pathsim_batt.thermal import LumpedThermal


class TestLumpedThermalInit(unittest.TestCase):
    def test_port_labels(self):
        self.assertEqual(LumpedThermal.input_port_labels["Q_dot"], 0)
        self.assertEqual(LumpedThermal.input_port_labels["T_amb"], 1)
        self.assertEqual(LumpedThermal.output_port_labels["T"], 0)

    def test_default_init(self):
        blk = LumpedThermal()
        self.assertAlmostEqual(blk.initial_value[0], 298.15)

    def test_custom_init(self):
        blk = LumpedThermal(mass=0.1, Cp=1000.0, UA=1.0, T0=310.0)
        self.assertAlmostEqual(blk.initial_value[0], 310.0)
        self.assertAlmostEqual(blk.mass, 0.1)
        self.assertAlmostEqual(blk.UA, 1.0)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            LumpedThermal(mass=0.0)
        with self.assertRaises(ValueError):
            LumpedThermal(Cp=-1.0)
        with self.assertRaises(ValueError):
            LumpedThermal(UA=-0.1)

    def test_output_equals_state_at_init(self):
        blk = LumpedThermal(T0=300.0)
        blk.set_solver(SSPRK22, parent=None)
        blk.update(0.0)
        self.assertAlmostEqual(blk.outputs[0], 300.0)


class TestLumpedThermalSimulation(unittest.TestCase):
    """Physics tests run via a full PathSim Simulation."""

    def _run(self, blk, t_end, dt=0.1):
        """Helper: wire block to constant sources and run."""
        Q_src = Constant(blk.inputs[0])
        T_src = Constant(blk.inputs[1])
        sim = Simulation(
            blocks=[Q_src, T_src, blk],
            connections=[
                Connection(Q_src, blk[0]),
                Connection(T_src, blk[1]),
            ],
            dt=dt,
            Solver=SSPRK22,
        )
        sim.run(t_end)
        return blk.outputs[0]

    def test_no_heat_no_gradient_stays_constant(self):
        """Q_dot=0, T == T_amb → temperature must not change."""
        blk = LumpedThermal(T0=298.15, UA=1.0)
        blk.inputs[0] = 0.0
        blk.inputs[1] = 298.15
        T = self._run(blk, t_end=10.0)
        self.assertAlmostEqual(T, 298.15, places=4)

    def test_heating_raises_temperature(self):
        """Q_dot > 0 with no cooling should raise temperature."""
        blk = LumpedThermal(mass=0.065, Cp=750.0, UA=0.0, T0=298.15)
        blk.inputs[0] = 10.0  # 10 W
        blk.inputs[1] = 298.15
        T = self._run(blk, t_end=10.0)
        self.assertGreater(T, 298.15)

    def test_cooling_lowers_temperature(self):
        """T > T_amb, Q_dot=0 → temperature decreases."""
        blk = LumpedThermal(mass=0.065, Cp=750.0, UA=1.0, T0=320.0)
        blk.inputs[0] = 0.0
        blk.inputs[1] = 298.15
        T = self._run(blk, t_end=10.0)
        self.assertLess(T, 320.0)

    def test_energy_balance(self):
        """Without cooling: ΔT = Q_dot * t / (m * Cp)."""
        mass, Cp, Q_dot, t_end = 0.1, 1000.0, 5.0, 2.0
        blk = LumpedThermal(mass=mass, Cp=Cp, UA=0.0, T0=300.0)
        blk.inputs[0] = Q_dot
        blk.inputs[1] = 300.0
        T = self._run(blk, t_end=t_end, dt=0.01)
        expected = 300.0 + Q_dot * t_end / (mass * Cp)
        self.assertAlmostEqual(T, expected, places=2)


if __name__ == "__main__":
    unittest.main()
