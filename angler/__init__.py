# used for setup.py
name = "angler"

# import the main classes
from .optimization import Optimization
from .simulation import Simulation

# import the various utilities
from .constants import *
from .plot import *
from .structures import *
from .utils import *