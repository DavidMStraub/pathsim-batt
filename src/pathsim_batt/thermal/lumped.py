from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pathsim.blocks import DynamicalSystem


class LumpedThermal(DynamicalSystem):
    """Single-node lumped thermal model.

    .. math::

        m C_p \\frac{dT}{dt} = \\dot{Q} - U A (T - T_{\\mathrm{amb}})

    Parameters
    ----------
    mass : float
        Thermal mass [kg].
    Cp : float
        Specific heat capacity [J kg⁻¹ K⁻¹].
    UA : float
        Overall heat transfer conductance [W K⁻¹].
    T0 : float
        Initial temperature [K].  Default 298.15 K.

    Inputs
    ------
    Q_dot (0) : heat generation rate [W]
    T_amb (1) : ambient temperature [K]

    Outputs
    -------
    T (0) : cell temperature [K]
    """

    input_port_labels = {"Q_dot": 0, "T_amb": 1}
    output_port_labels = {"T": 0}

    def __init__(
        self,
        mass: float = 0.065,
        Cp: float = 750.0,
        UA: float = 0.5,
        T0: float = 298.15,
    ) -> None:
        if mass <= 0:
            raise ValueError(f"mass must be positive, got {mass}")
        if Cp <= 0:
            raise ValueError(f"Cp must be positive, got {Cp}")
        if UA < 0:
            raise ValueError(f"UA must be non-negative, got {UA}")

        self.mass = float(mass)
        self.Cp = float(Cp)
        self.UA = float(UA)

        def _fn_d(x: NDArray, u: NDArray, _t: float) -> NDArray:
            (T,) = x
            Q_dot, T_amb = u
            return np.array([(Q_dot - self.UA * (T - T_amb)) / (self.mass * self.Cp)])

        def _fn_a(x: NDArray, _u: NDArray, _t: float) -> NDArray:
            return x.copy()

        super().__init__(
            func_dyn=_fn_d,
            func_alg=_fn_a,
            initial_value=np.array([float(T0)]),
        )
