
<p align="center">
  <strong>Battery simulation blocks for PathSim</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/pathsim/pathsim-batt" alt="License">
</p>

<p align="center">
  <a href="https://docs.pathsim.org/batt">Documentation</a> &bull;
  <a href="https://pathsim.org">PathSim Homepage</a> &bull;
  <a href="https://github.com/pathsim/pathsim-batt">GitHub</a>
</p>

---

PathSim-Batt extends the [PathSim](https://github.com/pathsim/pathsim) simulation framework with battery cell blocks using [PyBaMM](https://pybamm.org) as the electrochemical backend. All blocks follow the standard PathSim block interface and can be connected into simulation diagrams.

## Blocks

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| `CellElectrothermal` | Coupled electrical + thermal cell (PyBaMM owns temperature ODE) | `model`, `parameter_values`, `initial_soc` |
| `CellElectrical` | Electrical only, isothermal (PathSim owns temperature ODE) | `model`, `parameter_values`, `initial_soc` |
| `LumpedThermal` | Single-node thermal model for external thermal coupling | `mass`, `Cp`, `UA`, `T0` |

`Cell` is an alias for `CellElectrothermal`.

## PyBaMM integration

The cell blocks wrap [PyBaMM](https://pybamm.org) models behind the PathSim block interface. PyBaMM handles the electrochemistry (SPMe, DFN, ...) while PathSim handles the system-level simulation loop, connections, and time stepping.

- **Any PyBaMM model** can be injected via the `model` parameter
- **Any parameter set** can be used via `parameter_values` (defaults to `Chen2020`)
- **Extra output variables** from PyBaMM are accessible via `block.extra_outputs`
- **Lazy initialisation** — the PyBaMM simulation is only built on the first timestep

```python
import pybamm

model  = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
params = pybamm.ParameterValues("Mohtat2020")
cell   = CellElectrothermal(model=model, parameter_values=params)
```

## Thermal coupling modes

| Mode | Block | Owns cell temperature | Use when |
|---|---|---|---|
| Internal | `CellElectrothermal` | PyBaMM | Single-cell simulations, quick setup |
| External | `CellElectrical` + `LumpedThermal` | PathSim | Multi-cell packs, custom cooling models |

## Install

```bash
pip install pathsim-batt
```

## License

MIT
