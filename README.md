
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

Any PyBaMM battery model and parameter set can be injected. Extra PyBaMM output variables are accessible via `block.extra_outputs`.

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
