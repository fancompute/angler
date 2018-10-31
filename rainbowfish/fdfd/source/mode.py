import numpy as np
import scipy.sparse as sp
from copy import deepcopy

from fdfdpy.constants import *
from fdfdpy.linalg import *


class mode:

    def __init__(self, neff, direction_normal, center, width, scale, order=1):
        self.neff = neff
        self.direction_normal = direction_normal
        self.center = center
        self.width = width
        self.order = order
        self.scale = scale

    def setup_src(self, simulation, matrix_format=DEFAULT_MATRIX_FORMAT):
        # compute the input power here using an only waveguide simulation
        self.compute_normalization(simulation, matrix_format=matrix_format)

        # insert the mode into the waveguide
        self.insert_mode(simulation, simulation.src, matrix_format=matrix_format)

    def compute_normalization(self, simulation, matrix_format=DEFAULT_MATRIX_FORMAT):
        # creates a single waveguide simulation, solves the source, computes the power

        # get some information from the permittivity
        original_eps = simulation.eps_r
        (Nx, Ny) = original_eps.shape
        eps_max = np.max(np.abs(original_eps))
        norm_eps = np.ones((Nx, Ny))

        # make a new simulation and get a new probe center
        simulation_norm = deepcopy(simulation)
        new_center = list(self.center)

        # compute where the source and waveguide should be
        if self.direction_normal == "x":
            inds_y = original_eps[self.center[0], :] > 1
            norm_eps[:, inds_y] = eps_max
            new_center[0] = Nx - new_center[0]
        elif self.direction_normal == "y":
            inds_x = original_eps[:, self.center[1]] > 1
            norm_eps[inds_x, :] = eps_max
            new_center[1] = Ny - new_center[1]
        else:
            raise ValueError("The value of direction_normal is not x or y!")

        # reset the permittivity to be a straight waveguide, solve fields, compute power
        simulation_norm.eps_r = norm_eps
        self.insert_mode(simulation_norm, simulation_norm.src, matrix_format=matrix_format)
        simulation_norm.solve_fields()
        W_in = simulation_norm.flux_probe(self.direction_normal, new_center, self.width)

        # save this value in the original simulation
        simulation.W_in = W_in
        simulation.E2_in = np.sum(np.square(np.abs(
                        simulation_norm.fields['Ez']))*np.abs(simulation_norm.src))

    def insert_mode(self, simulation, destination, matrix_format=DEFAULT_MATRIX_FORMAT):
        EPSILON_0_ = EPSILON_0*simulation.L0
        MU_0_ = MU_0*simulation.L0

        # first extract the slice of the permittivity
        if self.direction_normal == "x":
            inds_x = [self.center[0], self.center[0]+1]
            inds_y = [int(self.center[1]-self.width/2), int(self.center[1]+self.width/2)]
        elif self.direction_normal == "y":
            inds_x = [int(self.center[0]-self.width/2), int(self.center[0]+self.width/2)]
            inds_y = [self.center[1], self.center[1]+1]
        else:
            raise ValueError("The value of direction_normal is not x or y!")

        eps_r = simulation.eps_r[inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]]
        N = eps_r.size

        Dxb = createDws('x', 'b', [simulation.dl], [N], matrix_format=matrix_format)
        Dxf = createDws('x', 'f', [simulation.dl], [N], matrix_format=matrix_format)

        vector_eps = EPSILON_0_*eps_r.reshape((-1,))
        vector_eps_x = EPSILON_0_*grid_average(eps_r, 'x').reshape((-1,))
        T_eps = sp.spdiags(vector_eps, 0, N, N, format=matrix_format)
        T_epsxinv = sp.spdiags(vector_eps_x**(-1), 0, N, N, format=matrix_format)

        if simulation.pol == 'Ez':
            A = np.square(simulation.omega)*MU_0_*T_eps + Dxf.dot(Dxb)

        elif simulation.pol == 'Hz':
            A = np.square(simulation.omega)*MU_0_*T_eps + T_eps.dot(Dxf).dot(T_epsxinv).dot(Dxb)

        est_beta = simulation.omega*np.sqrt(MU_0_*EPSILON_0_)*self.neff
        (vals, vecs) = solver_eigs(A, self.order, guess_value=np.square(est_beta))

        if self.order == 1:
            src = vecs
        else:
            src = vecs[:, self.order-1]

        src *= self.scale

        if self.direction_normal == 'x':
            src = src.reshape((1, -1))
            destination[inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]] = np.abs(src)*np.sign(np.real(src))
        else:
            src = src.reshape((-1, 1))
            destination[inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]] = np.abs(src)*np.sign(np.real(src))
