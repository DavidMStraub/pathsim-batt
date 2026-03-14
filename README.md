# pathsim-batt

Battery simulation blocks for [PathSim](https://github.com/pathsim/pathsim), using [PyBaMM](https://pybamm.org) as the electrochemical backend.

## Installation

```bash
pip install pathsim-batt
```

## Blocks

### `CellElectrothermal`

Coupled electrical + thermal cell. PyBaMM's built-in lumped thermal model integrates the cell temperature ODE internally. Supply ambient temperature to couple to a pack-level thermal model.

```
Inputs:  I [A], T_amb [K]
Outputs: V [V], T [K], Q_heat [W/m³], SOC
```

`Cell` is an alias for `CellElectrothermal`.

### `CellElectrical`

Electrical outputs only. PyBaMM runs isothermal at the temperature you supply. Use this when PathSim owns the thermal dynamics (via `LumpedThermal` or a more complex thermal network).

```
Inputs:  I [A], T_cell [K]
Outputs: V [V], Q_heat [W/m³], SOC
```

### `LumpedThermal`

Single-node thermal model: `m·Cₚ·dT/dt = Q̇ − UA·(T − T_amb)`

```
Inputs:  Q_dot [W], T_amb [K]
Outputs: T [K]
```

## Examples

### 1 — Simple discharge (`CellElectrothermal`)

```python
import pybamm
from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Scope
from pathsim.solvers import SSPRK22
from pathsim_batt import CellElectrothermal

cell  = CellElectrothermal(parameter_values=pybamm.ParameterValues("Chen2020"))
I_app = Constant(5.0)     # 1 C
T_amb = Constant(298.15)
scope = Scope(labels=["V [V]", "SOC"])

Sim = Simulation(
    blocks=[cell, I_app, T_amb, scope],
    connections=[
        Connection(I_app, cell["I"]),
        Connection(T_amb, cell["T_amb"]),
        Connection(cell["V"],   scope[0]),
        Connection(cell["SOC"], scope[1]),
    ],
    dt=10.0, Solver=SSPRK22,
)

Sim.run(3600)
scope.plot()
```

### 2 — External thermal coupling (`CellElectrical` + `LumpedThermal`)

When you want PathSim to own the thermal ODE — for example to couple multiple cells to a shared cooling model:

```python
import numpy as np, pybamm
from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Scope, Amplifier
from pathsim.solvers import SSPRK22
from pathsim_batt import CellElectrical
from pathsim_batt.thermal import LumpedThermal

cell_volume = np.pi * 0.0105**2 * 0.070   # m³ (LG M50 21700)

cell    = CellElectrical(parameter_values=pybamm.ParameterValues("Chen2020"))
thermal = LumpedThermal(mass=0.065, Cp=750.0, UA=0.5, T0=298.15)
gain    = Amplifier(cell_volume)           # W/m³ → W
I_app   = Constant(5.0)
T_amb   = Constant(298.15)
scope   = Scope(labels=["V [V]", "T [K]", "SOC"])

Sim = Simulation(
    blocks=[cell, thermal, gain, I_app, T_amb, scope],
    connections=[
        Connection(I_app,          cell["I"]),
        Connection(thermal["T"],   cell["T_cell"]),   # feedback
        Connection(cell["Q_heat"], gain),
        Connection(gain,           thermal["Q_dot"]),
        Connection(T_amb,          thermal["T_amb"]),
        Connection(cell["V"],      scope[0]),
        Connection(thermal["T"],   scope[1]),
        Connection(cell["SOC"],    scope[2]),
    ],
    dt=10.0, Solver=SSPRK22,
)

Sim.run(3600)
scope.plot()
```

## Thermal coupling modes

| Mode | Block | Owns cell temperature | Use when |
|---|---|---|---|
| Internal | `CellElectrothermal` | PyBaMM | Single-cell simulations, quick setup |
| External | `CellElectrical` + `LumpedThermal` | PathSim | Multi-cell packs, custom cooling models, different thermal timestep |

## Model and parameter set selection

Any PyBaMM battery model and parameter set can be injected:

```python
model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
params = pybamm.ParameterValues("Mohtat2020")
cell = CellElectrothermal(model=model, parameter_values=params)
```

Extra PyBaMM output variables are accessible via `block.extra_outputs`:

```python
cell = CellElectrothermal(
    output_variables=["X-averaged negative particle surface concentration [mol.m-3]"]
)
# after stepping:
c_neg = cell.extra_outputs["X-averaged negative particle surface concentration [mol.m-3]"]
```
