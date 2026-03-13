"""
PathSim-Batt: Battery Simulation Blocks for PathSim

A toolbox providing battery simulation blocks for the PathSim framework,
using PyBaMM as the electrochemical backend.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["__version__"]

from .cells import *
from .thermal import *
