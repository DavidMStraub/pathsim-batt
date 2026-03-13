###############################################################################
##
##                          PYBAMM CELL BLOCK
##                      (cells/pybamm_cell.py)
##
###############################################################################

import numpy as np
import pybamm

from pathsim.blocks._block import Block


class PyBaMMCell(Block):
    """A PathSim block wrapping a PyBaMM electrochemical cell model.

    Drives a PyBaMM ``Simulation`` forward one PathSim timestep at a time,
    exposing terminal voltage, heat generation and state of charge as output
    signals.  The full electrochemical model (SPM, SPMe, DFN, …) and its
    parameter set are configured externally and injected at construction time,
    keeping this block chemistry-agnostic.

    The thermal submodel is intentionally externalised: PyBaMM is configured
    with ``"external submodels": ["thermal"]`` so that it receives the cell
    temperature as an input signal rather than computing it internally.  This
    lets a ``LumpedThermalModel`` (or any other PathSim thermal block) own the
    temperature dynamics and feed them back, enabling arbitrary thermal
    networks at the pack level.

    Parameters
    ----------
    model : pybamm.lithium_ion.BaseModel, optional
        A PyBaMM lithium-ion model instance.  If *None*, defaults to
        ``pybamm.lithium_ion.SPMe`` with a lumped, externally-driven thermal
        submodel.
    parameter_values : pybamm.ParameterValues, optional
        PyBaMM parameter set.  If *None*, defaults to ``Chen2020``.  The
        ``"Current function [A]"`` entry is always overridden to ``"[input]"``
        so that PathSim can supply current as a signal at every timestep.
    initial_soc : float
        Initial state of charge (0–1).  Default 1.0 (fully charged).
    solver : pybamm.BaseSolver, optional
        PyBaMM solver to use.  If *None*, defaults to
        ``pybamm.CasadiSolver(mode="safe")``.
    output_variables : list[str], optional
        Extra PyBaMM variable names to cache after each step.  Accessible via
        ``block.extra_outputs`` as a dict after the simulation runs.

    Inputs
    ------
    I : float
        Applied current [A].  Positive = discharge (PyBaMM convention).
    T_ext : float
        External cell temperature [K].  Only used when the thermal submodel is
        external (default).  Ignored if ``model`` was built without an external
        thermal submodel.

    Outputs
    -------
    V : float
        Terminal voltage [V].
    Q_heat : float
        X-averaged volumetric heat generation [W m⁻³].
    SOC : float
        State of charge (0–1) as reported by PyBaMM.

    Notes
    -----
    PathSim identifies a block as *dynamic* (i.e. requires timestepping) by
    the presence of an ``initial_value`` attribute.  This block sets
    ``initial_value = 0.0`` as a sentinel so PathSim calls ``step()`` each
    cycle, but the actual state is owned entirely by the PyBaMM
    ``Simulation`` object.  PathSim's ODE solver is never attached: the
    ``set_solver`` override initialises PyBaMM instead.
    """

    input_port_labels  = {"I": 0, "T_ext": 1}
    output_port_labels = {"V": 0, "Q_heat": 1, "SOC": 2}

    # Sentinel so PathSim treats this as a dynamic (state) block and calls step().
    initial_value = 0.0

    def __init__(
        self,
        model=None,
        parameter_values=None,
        initial_soc=1.0,
        solver=None,
        output_variables=None,
    ):
        super().__init__()

        self._initial_soc = float(initial_soc)
        self._extra_var_names = list(output_variables or [])
        self.extra_outputs = {}

        # --- build model ---------------------------------------------------------
        if model is None:
            model = pybamm.lithium_ion.SPMe(
                options={
                    "thermal": "lumped",
                    "external submodels": ["thermal"],
                }
            )
        self._model = model

        # --- parameter values ----------------------------------------------------
        if parameter_values is None:
            parameter_values = pybamm.ParameterValues("Chen2020")
        # Current must always be an input so PathSim can drive it each step.
        parameter_values = parameter_values.copy()
        parameter_values["Current function [A]"] = "[input]"
        self._parameter_values = parameter_values

        # --- solver --------------------------------------------------------------
        if solver is None:
            solver = pybamm.CasadiSolver(mode="safe")
        self._pybamm_solver = solver

        # Simulation created lazily in set_solver / _initialize so that the
        # block can be constructed before PyBaMM is fully configured.
        self._sim = None
        self._initialized = False

    # ------------------------------------------------------------------
    # PathSim interface
    # ------------------------------------------------------------------

    def __len__(self):
        # No algebraic passthrough: outputs depend only on internal PyBaMM state.
        return 0

    def set_solver(self, Solver, parent, **solver_args):
        """Override: do not attach PathSim's ODE solver; initialise PyBaMM instead."""
        self._initialize()

    def reset(self):
        """Reset I/O registers and reinitialise PyBaMM from initial SOC."""
        self.inputs.reset()
        self.outputs.reset()
        self._sim = None
        self._initialized = False
        self.extra_outputs = {}

    def update(self, t):
        """Read latest PyBaMM solution into output ports."""
        if not self._initialized:
            self._initialize()
        sol = self._sim.solution
        self.outputs["V"]      = float(sol["Terminal voltage [V]"].entries[-1])
        self.outputs["Q_heat"] = float(sol["X-averaged total heating [W.m-3]"].entries[-1])
        self.outputs["SOC"]    = float(sol["State of Charge"].entries[-1])
        for name in self._extra_var_names:
            self.extra_outputs[name] = float(sol[name].entries[-1])

    def step(self, t, dt):
        """Advance PyBaMM by one PathSim timestep.

        Returns
        -------
        success : bool
        error : float
        scale : float
        """
        if not self._initialized:
            self._initialize()

        I = float(self.inputs["I"])
        inputs = {"Current function [A]": I}

        # Feed external temperature if the model uses an external thermal submodel.
        if "external submodels" in getattr(self._model, "options", {}):
            T_ext = float(self.inputs["T_ext"])
            inputs["Volume-averaged cell temperature [K]"] = T_ext

        self._sim.step(dt, inputs=inputs, save=False)

        # Return (success, error, scale) — no adaptive control from PyBaMM side.
        return True, 0.0, 1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize(self):
        """Create and warm-start the PyBaMM Simulation at initial_soc."""
        self._sim = pybamm.Simulation(
            self._model,
            parameter_values=self._parameter_values,
            solver=self._pybamm_solver,
        )
        # Solve a tiny interval to initialise internal state vectors.
        I0 = float(self.inputs["I"]) if self.inputs["I"] != 0.0 else 1e-6
        inputs = {"Current function [A]": I0}
        if "external submodels" in getattr(self._model, "options", {}):
            inputs["Volume-averaged cell temperature [K]"] = float(self.inputs["T_ext"]) or 298.15
        self._sim.solve(
            [0, 1e-6],
            initial_soc=self._initial_soc,
            inputs=inputs,
            save=False,
        )
        self._initialized = True
