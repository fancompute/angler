import numpy as np
from copy import deepcopy
import warnings
import dill as pickle
import matplotlib.pylab as plt
from collections import namedtuple
import sys
sys.path.append('../angler')
from angler import Optimization


def load_device(fname):
    """ Loads the pickled Device object """
    D_dict = pickle.load(open(fname, "rb"))
    D = namedtuple('Device', D_dict.keys())(*D_dict.values())
    return D


class Device:
    """ Stores the data from a finished structure optimization.  
        Stores the simulation and optimization object.
        Then stores the structure type and geometry.
        If data does not exist (like fields, freq scan, power scan), generates them.

        USAGE:
            D = Device(simulation, optimization)
            D.set_parameters(lambda0, dl, NPML, chi3, eps_m, source_amp)
            D.set_geometry("three_port", L, H, w, d, l, spc, dl, NPML, eps_start)
            D.generate_data()
            D.save(fname='data/figs/device3.p')
    """

    def __init__(self, simulation, optimization):
        """ Loads the main objects and does some processing to compute plot-ables"""

        print('loading in simulation and optimization...')

        # main objects (deepcopy as to not mess up originals)
        self.simulation = deepcopy(simulation)
        self.optimization = deepcopy(optimization)
        self.L0 = self.simulation.L0

        # electric fields
        print('    solving fields')
        self.simulation.solve_fields()
        self.simulation.solve_fields_nl()
        self.Ez = self.simulation.fields['Ez']
        self.Ez_nl = self.simulation.fields_nl['Ez']        

        # whether other important things have been stored
        self.are_parameters_set = False
        self.is_geometry_set = False
        self.is_data_generated = False

    def set_parameters(self, lambda0, dl, NPML, chi3, eps_m, source_amp):
        """ sets the general simulation parameters not stored in the other objects"""

        print('setting parameters...')

        # store main parameters
        self.lambda0 = lambda0
        c0 = 3e8
        self.omega = 2*np.pi*c0/lambda0
        self.dl = dl
        self.NPML = NPML
        self.chi3 = chi3
        self.eps_m = eps_m
        self.source_amp = source_amp

        # compute nice-to-haves
        self.grids_per_lamda = self.lambda0 / self.dl / self.L0
        (self.Nx, self.Ny) = self.Ez.shape
        self.nx, self.ny = self.Nx//2, self.Ny//2
        self.x_range = np.linspace(-self.Nx/2*self.dl, self.Nx/2*self.dl, self.Nx)
        self.y_range = np.linspace(-self.Ny/2*self.dl, self.Ny/2*self.dl, self.Ny)

        self.are_parameters_set = True

    def set_geometry(self, structure_type, *structure_params):
        """ Takes the 'type' of structure {three_port, two_port, ortho_port} and
            the parameters used in the construction of this structure from structures.py
            Saves these in the object for plotting later
        """

        print('setting geometry parameters...')

        if structure_type == 'three_port':
            self.L, self.H, self.w, self.d, self.l, self.spc, self.dl, self.NPML, self.eps_start = structure_params
        elif structure_type == 'two_port':
            self.L, self.H, self.w, self.l, self.spc, self.dl, self.NPML, self.eps_start = structure_params
        elif structure_type == 'ortho_port':
            self.L, self.L2, self.H, self.H2, self.w, self.l, self.dl, self.NPML, self.eps_start = structure_params
        else:
            raise ValueError("Need to specify a 'structure_type' argument as one of '{three_port, two_port, ortho_port}'")

        self.structure_type = structure_type
        self.is_geometry_set = True

    def generate_data(self, Nf=100, Np=100, s_min=1e-1, s_max=1e3):
        """ Solve fields, index shift, frequency scan (Q), power scan, S_matrix etc. """

        print('generating data...')

        print('    computing index shift')
        self.compute_index_shift()

        print('    computing power transmission')
        self.calc_transmissions()

        print('    computing frequency scan ({} points)'.format(Nf))
        self.freq_scan(Nf=Nf)

        print('    computing power scan ({} points)'.format(Np))
        self.power_scan(Np=Np, s_min=s_min, s_max=s_max)

        self.is_data_generated = True

    def compute_index_shift(self):
        """ compute the refractive index shift """
        self.index_shift = self.simulation.compute_index_shift()
        print("        -> max index shift = {}".format(self.index_shift))

    def calc_transmissions(self):
        """ Gets the transmission data """

        if self.structure_type == 'two_port':
            self._calc_trans_two()                
        elif self.structure_type == 'three_port':
            self._calc_trans_three()    
        elif self.structure_type == 'ortho_port':
            self._calc_trans_ortho()

    def _calc_trans_two(self):

        # input power
        self.W_in = self.simulation.W_in
        print("        -> W_in = {}".format(self.W_in))

        # linear powers
        self.W_out_lin = self.simulation.flux_probe('x', [-self.NPML[0]-int(self.l/2/self.dl), self.ny], int(self.Ny/2))
        self.W_in_lin = self.simulation.flux_probe('x', [self.NPML[0]+int(2*self.l/3/self.dl), self.ny], int(self.Ny/2))

        self.T_lin = self.W_out_lin/self.W_in

        # nonlinear powers
        self.W_out_nl = self.simulation.flux_probe('x', [-self.NPML[0]-int(self.l/2/self.dl), self.ny], int(self.Ny/2), nl=True)
        self.W_in_nl = self.simulation.flux_probe('x', [self.NPML[0]+int(2*self.l/3/self.dl), self.ny], int(self.Ny/2), nl=True)

        self.T_nl = self.W_out_nl/self.W_in

        print('        -> linear transmission              = {:.4f}'.format(self.T_lin))
        print('        -> nonlinear transmission           = {:.4f}'.format(self.T_nl))

    def _calc_trans_three(self):
        # input power
        self.W_in = self.simulation.W_in
        print("        -> W_in = {}".format(self.W_in))

        # linear powers
        self.W_top_lin = self.simulation.flux_probe('x', [-self.NPML[0]-int(self.l/2/self.dl), self.ny+int(self.d/2/self.dl)], int(self.H/2/self.dl))
        self.W_bot_lin = self.simulation.flux_probe('x', [-self.NPML[0]-int(self.l/2/self.dl), self.ny-int(self.d/2/self.dl)], int(self.H/2/self.dl))

        # nonlinear powers
        self.W_top_nl = self.simulation.flux_probe('x', [-self.NPML[0]-int(self.l/2/self.dl), self.ny+int(self.d/2/self.dl)], int(self.H/2/self.dl), nl=True)
        self.W_bot_nl = self.simulation.flux_probe('x', [-self.NPML[0]-int(self.l/2/self.dl), self.ny-int(self.d/2/self.dl)], int(self.H/2/self.dl), nl=True)


        print('        -> linear transmission (top)        = {:.4f}'.format(self.W_top_lin / self.W_in))
        print('        -> linear transmission (bottom)     = {:.4f}'.format(self.W_bot_lin / self.W_in))
        print('        -> nonlinear transmission (top)     = {:.4f}'.format(self.W_top_nl / self.W_in))
        print('        -> nonlinear transmission (bottom)  = {:.4f}'.format(self.W_bot_nl / self.W_in))

        self.S = [[self.W_top_lin / self.W_in, self.W_top_nl / self.W_in],
                  [self.W_bot_lin / self.W_in, self.W_bot_nl / self.W_in]]

        plt.imshow(self.S, cmap='magma')
        plt.colorbar()
        plt.title('power matrix')
        plt.show()

    def _calc_trans_ortho(self):
        # input power
        self.W_in = self.simulation.W_in
        print("        -> W_in = {}".format(self.W_in))

        # linear powers
        self.W_right_lin = self.simulation.flux_probe('x', [-self.NPML[0]-int(self.l/2/self.dl), self.ny], int(self.H/2/self.dl))
        self.W_top_lin  = self.simulation.flux_probe('y', [self.nx, -self.NPML[1]-int(self.l/2/self.dl)], int(self.H/2/self.dl))

        # nonlinear powers
        self.W_right_nl = self.simulation.flux_probe('x', [-self.NPML[0]-int(self.l/2/self.dl), self.ny], int(self.H/2/self.dl), nl=True)
        self.W_top_nl  = self.simulation.flux_probe('y', [self.nx, -self.NPML[1]-int(self.l/2/self.dl)], int(self.H/2/self.dl), nl=True)


        print('        -> linear transmission (right)      = {:.4f}'.format(self.W_right_lin / self.W_in))
        print('        -> linear transmission (top)        = {:.4f}'.format(self.W_top_lin / self.W_in))
        print('        -> nonlinear transmission (right)   = {:.4f}'.format(self.W_right_nl / self.W_in))
        print('        -> nonlinear transmission (top)     = {:.4f}'.format(self.W_top_nl / self.W_in))

        self.S = [[self.W_top_lin / self.W_in, self.W_right_lin / self.W_in],
                  [self.W_top_nl / self.W_in,  self.W_right_nl / self.W_in]]

        plt.imshow(self.S, cmap='magma')
        plt.colorbar()
        plt.title('power matrix')
        plt.show()        

    def freq_scan(self, Nf=100):
        """ Scans frequency and saves results"""

        freqs, objs, FWHM = self.optimization.scan_frequency(Nf=Nf, df=1/400, pbar=False)

        self.freqs = freqs
        self.objs = objs
        self.FWHM = FWHM
        self.Q = self.omega/2/np.pi/FWHM

        plt.plot([(f-150e12)/1e9 for f in freqs], objs)
        plt.xlabel('frequency difference (GHz)')
        plt.ylabel('objective function')
        plt.show()

        print('        -> computed FWHM of {} (GHz):'.format(FWHM/1e9))
        print('        -> Q factor of {0:.2E}'.format(self.Q))        

    def power_scan(self, Np=50, s_min=1e-1, s_max=1e3):
        """ Gets the transmission data """

        if self.structure_type == 'two_port':
            probe_out = lambda simulation: simulation.flux_probe('x', [-self.NPML[0]-int(self.l/2/self.dl), self.ny], int(self.Ny/2), nl=True)
            probes = [probe_out]            
            self._power_scan_all(probes, Np=Np, s_min=s_min, s_max=s_max)

        elif self.structure_type == 'three_port':
            probe_top = lambda simulation: simulation.flux_probe('x', [-self.NPML[0]-int(self.l/2/self.dl), self.ny+int(self.d/2/self.dl)], int(self.H/2/self.dl), nl=True)
            probe_bot = lambda simulation: simulation.flux_probe('x', [-self.NPML[0]-int(self.l/2/self.dl), self.ny-int(self.d/2/self.dl)], int(self.H/2/self.dl), nl=True)
            probes = [probe_top, probe_bot]            
            self._power_scan_all(probes, Np=Np, s_min=s_min, s_max=s_max)

        elif self.structure_type == 'ortho_port':
            probe_right = lambda simulation: simulation.flux_probe('x', [-self.NPML[0]-int(self.l/2/self.dl), self.ny], int(self.H/2/self.dl), nl=True)
            probe_top = lambda simulation: simulation.flux_probe('y', [self.nx, -self.NPML[1]-int(self.l/2/self.dl)], int(self.H/2/self.dl), nl=True)
            probes = [probe_right, probe_top]            
            self._power_scan_all(probes, Np=Np, s_min=s_min, s_max=s_max)

    def _power_scan_all(self, probes, Np=50, s_min=1e-1, s_max=1e1):
        self.powers, self.transmissions = self.optimization.scan_power(probes=probes, Ns=Np, s_min=s_min, s_max=s_max, solver='hybrid')
        for probe_index, _ in enumerate(probes):
            plt.plot(self.powers, self.transmissions[probe_index])
        plt.xscale('log')
        plt.xlabel('input power (W / mum)')
        plt.ylabel('transmission')
        plt.show()        

    def save(self, filename):
        """ Pickle this object and save it to file 
            NOTE: filename should probably end in '.p'
            for reference, to load this object: `device = pickle.load( open(fname, 'rb'))`
        """

        print('pickling device...')

        # warn if parameters are not set.
        if not self.are_parameters_set:
            warnings.warn('parameters not set.  Run "device.set_parameters()" to fix')
        if not self.is_geometry_set:
            warnings.warn('WARNING: geometry not set. Run "device.set_geometry()" to fix')
        if not self.is_data_generated:
            warnings.warn('WARNING: plotting data not generated.  Run "device.generate_data()" to fix')

        # save to filename
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)

        print('done')


