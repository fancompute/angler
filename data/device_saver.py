import numpy as np
from copy import deepcopy
import warnings

class Device:
    """ Stores the data from a finished structure optimization.  
        Stores the simulation and optimization object.
        Then stores the structure type and geometry.
        If data does not exist (like fields, freq scan, power scan), generates them.

        USAGE:
            D = Device(simulation, optimization)
            D.set_parameters(lambda0, dl, NPML, chi3, eps_m, source_amp)
            D.set_geometry("three_port", L, H, w, d, l, spc, dl, NPML, eps_start)
            D.save(fname='data/figs/device3.p')
    """

    def __init__(self, simulation, optimization):
        """ Loads the main objects and does some processing to compute plot-ables"""

        # main objects (deepcopy as to not mess up originals)
        self.simulation = deepcopy(simulation)
        self.optimization = deepcopy(optimization)
        self.L0 = self.simulation.L0

        # electric fields
        self.simulation.solve_fields()
        self.simulation.solve_fields_nl()
        self.Ez = self.simulation.fields['Ez']
        self.Ez_nl = self.simulation.fields_nl['Ez']        

        # spatial coordinates
        (Nx, Ny) = self.Ez.shape
        nx, ny = Nx//2, Ny//2
        self.x_range = np.linspace(-Nx/2*dl, Nx/2*dl, Nx)
        self.y_range = np.linspace(-Ny/2*dl, Ny/2*dl, Ny)


        # whether other important things have been stored
        self.parameters_set = False
        self.geometry_set = False

        """ Solve fields, index shift, frequency scan (Q), power scan, etc. """

    def set_parameters(self, lambda0, dl, NPML, chi3, eps_m, source_amp):
        """ sets the general simulation parameters not stored in the other objects"""

        # store main parameters
        self.lambda0 = lambda0
        self.dl = dl
        self.NPML = NPML
        self.chi3 = chi3
        self.eps_m = eps_m
        self.source_amp = source_amp

        # compute nice-to-haves
        self.grids_per_lamda = self.lambda0 / self.dl / self.L0

        self.parameters_set = True

    def set_geometry(self, structure_type, *structure_params):
        """ Takes the 'type' of structure {three_port, two_port, ortho_port} and
            the parameters used in the construction of this structure from structures.py
            Saves these in the object for plotting later
        """

        if structure_type == 'three_port':
            self.L, self.H, self.w, self.d, self.l, self.spc, self.dl, self.NPML, self.eps_start = structure_params
        elif structure_type == 'two_port':
            self.L, self.H, self.w, self.l, self.spc, self.dl, self.NPML, self.eps_start = structure_params
        elif structure_type == 'ortho_port':
            self.L, self.L2, self.H, self.H2, self.w, self.l, self.dl, self.NPML, self.eps_start = structure_params
        else:
            raise ValueError("Need to specify a 'structure_type' argument as one of '{three_port, two_port, ortho_port}'")

        self.structure_type = structure_type
        self.geometry_set = True

    def save(filename):
        """ Pickle this object and save it to file 
            NOTE: filename should probably end in '.p'
            for reference, to load this object: `device = pickle.load( open(fname, 'rb'))`
        """

        # warn if parameters are not set.
        if not self.parameters_set:
            warnings.warn('parameters not set')
        if not self.geometry_set:
            warnings.warn('WARNING: geometry not set')

        # save to filename
        with open(filename, "wb" ) as f:
            pickle.dump(self, f)

