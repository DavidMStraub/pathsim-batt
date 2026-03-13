###############################################################################
##
##                        LUMPED THERMAL MODEL BLOCK
##                        (thermal/lumped.py)
##
###############################################################################

import numpy as np

from pathsim.blocks import DynamicalSystem


class LumpedThermalModel(DynamicalSystem):
    """Single-node (lumped) thermal model for a battery cell or pack.

    Models the cell as a single thermal mass exchanging heat with an ambient
    reservoir through a fixed heat-transfer coefficient:

    .. math::

        m C_p \\frac{dT}{dt} = \\dot{Q} - U A (T - T_{\\mathrm{amb}})

    where :math:`\\dot{Q}` [W] is the total heat input from electrochemical
    and ohmic sources, and :math:`UA` [W K⁻¹] is the overall thermal
    conductance to the ambient.

    This block is designed to be paired with ``PyBaMMCell``: connect
    ``PyBaMMCell["Q_heat"]`` through a ``VolumetricToWatts`` gain (multiply by
    cell volume [m³]) to ``LumpedThermalModel["Q_dot"]``, and feed the
    temperature output back to ``PyBaMMCell["T_ext"]``.

    Parameters
    ----------
    mass : float
        Cell mass [kg].
    Cp : float
        Specific heat capacity [J kg⁻¹ K⁻¹].
    UA : float
        Overall heat transfer coefficient times cooling area [W K⁻¹].
    T0 : float
        Initial cell temperature [K].  Default 298.15 K (25 °C).

    Inputs
    ------
    Q_dot : float
        Heat generation rate [W].
    T_amb : float
        Ambient (coolant) temperature [K].

    Outputs
    -------
    T : float
        Cell temperature [K].
    """

    input_port_labels  = {"Q_dot": 0, "T_amb": 1}
    output_port_labels = {"T": 0}

    def __init__(self, mass=0.065, Cp=750.0, UA=0.5, T0=298.15):
        if mass <= 0:
            raise ValueError(f"mass must be positive, got {mass}")
        if Cp <= 0:
            raise ValueError(f"Cp must be positive, got {Cp}")
        if UA < 0:
            raise ValueError(f"UA must be non-negative, got {UA}")

        self.mass = float(mass)
        self.Cp   = float(Cp)
        self.UA   = float(UA)

        def _fn_d(x, u, t):
            T,     = x
            Q_dot, T_amb = u
            dT = (Q_dot - self.UA * (T - T_amb)) / (self.mass * self.Cp)
            return np.array([dT])

        def _fn_a(x, u, t):
            return x.copy()  # output is the state directly

        super().__init__(
            func_dyn=_fn_d,
            func_alg=_fn_a,
            initial_value=np.array([float(T0)]),
        )
