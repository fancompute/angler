import numpy as np
import scipy.sparse as sp
from copy import deepcopy

from angler.linalg import construct_A, solver_direct, grid_average
from angler.derivatives import unpack_derivs
from angler.nonlinear_solvers import born_solve, newton_solve, newton_krylov_solve
from angler.source.mode import mode
from angler.nonlinearity import Nonlinearity
from angler.constants import (DEFAULT_LENGTH_SCALE, DEFAULT_MATRIX_FORMAT,
                              DEFAULT_SOLVER, EPSILON_0, MU_0)
from angler.filter import eps2rho
from angler.plot import plt_base_eps, plt_base

class Simulation:

    def __init__(self, omega, eps_r, dl, NPML, pol, L0=DEFAULT_LENGTH_SCALE, use_dirichlet_bcs=False):
        # initializes Fdfd object

        self.L0 = L0
        self.omega = omega
        self.NPML = NPML
        self.pol = pol
        self.dl = dl
        self.use_dirichlet_bcs = use_dirichlet_bcs

        self._check_inputs()

        grid_shape = eps_r.shape
        if len(grid_shape) == 1:
            self.flatten = True
            grid_shape = (grid_shape[0], 1)
            eps_r = np.reshape(eps_r, grid_shape)
        else:
            self.flatten = False

        (Nx, Ny) = grid_shape

        self.Nx = Nx
        self.Ny = Ny

        self.mu_r = np.ones((self.Nx, self.Ny))
        self.src = np.zeros((self.Nx, self.Ny), dtype=np.complex64)

        self.xrange = [0, float(self.Nx*self.dl)]
        self.yrange = [0, float(self.Ny*self.dl)]

        self.NPML = [int(n) for n in self.NPML]
        self.omega = float(self.omega)
        self.dl = float(dl)

        # construct the system matrix
        self.eps_r = eps_r

        self.modes = []
        self.nonlinearity = []
        self.eps_nl = np.zeros(eps_r.shape)
        self.dnl_de = np.zeros(eps_r.shape)
        self.dnl_deps = np.zeros(eps_r.shape)

    def setup_modes(self):
        # calculates
        for modei in self.modes:
            modei.setup_src(self)

    def add_mode(self, neff, direction_normal, center, width,
                 scale=1, order=1):
        # adds a mode definition to the simulation
        new_mode = mode(neff, direction_normal, center, width,
                        scale=scale, order=order)
        self.modes.append(new_mode)

    def compute_nl(self, e, matrix_format=DEFAULT_MATRIX_FORMAT):
        # evaluates the nonlinear functions for a field e
        self.eps_nl = np.zeros(self.eps_r.shape)
        self.dnl_de = np.zeros(self.eps_r.shape)
        self.dnl_deps = np.zeros(self.eps_r.shape)
        for nli in self.nonlinearity:
            self.eps_nl = self.eps_nl + nli.eps_nl(e, self.eps_r)
            self.dnl_de = self.dnl_de + nli.dnl_de(e, self.eps_r)
            self.dnl_deps = self.dnl_deps + nli.dnl_deps(e, self.eps_r)
        Nbig = self.Nx*self.Ny
        Anl = sp.spdiags(self.omega**2*EPSILON_0*self.L0*self.eps_nl.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)
        self.Anl = Anl

    def add_nl(self, chi, nl_region, nl_type='kerr', eps_scale=False, eps_max=None):
        # adds a nonlinearity to the simulation
        new_nl = Nonlinearity(chi/np.square(self.L0), nl_region, nl_type, eps_scale, eps_max)
        self.nonlinearity.append(new_nl)

    @property
    def eps_r(self):
        return self.__eps_r

    @eps_r.setter
    def eps_r(self, new_eps):

        grid_shape = new_eps.shape
        if len(grid_shape) == 1:
            grid_shape = (grid_shape[0], 1)
            new_eps.reshape(grid_shape)

        (Nx, Ny) = grid_shape
        self.Nx = Nx
        self.Ny = Ny

        self.__eps_r = new_eps
        (A, derivs) = construct_A(self.omega, self.xrange, self.yrange,
                                  self.eps_r, self.NPML, self.pol, self.L0,
                                  matrix_format=DEFAULT_MATRIX_FORMAT,
                                  use_dirichlet_bcs=self.use_dirichlet_bcs,
                                  timing=False)
        self.A = A
        self.derivs = derivs
        self.fields = {f: None for f in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']}
        self.fields_nl = {f: None for f in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']}

    def solve_fields(self, include_nl=False, timing=False, averaging=False,
                     matrix_format=DEFAULT_MATRIX_FORMAT):
        # performs direct solve for A given source

        EPSILON_0_ = EPSILON_0*self.L0
        MU_0_ = MU_0*self.L0

        if include_nl==False:
            eps_tot = self.eps_r
            X = solver_direct(self.A, self.src*1j*self.omega, timing=timing)
        else:
            eps_tot = self.eps_r + self.eps_nl
            X = solver_direct(self.A + self.Anl, self.src*1j*self.omega, timing=timing)


        (Nx, Ny) = self.src.shape
        M = Nx*Ny
        (Dyb, Dxb, Dxf, Dyf) = unpack_derivs(self.derivs)

        if self.pol == 'Hz':
            if averaging:
                eps_x = grid_average(EPSILON_0_*(eps_tot), 'x')
                vector_eps_x = eps_x.reshape((-1,))
                eps_y = grid_average(EPSILON_0_*(eps_tot), 'y')
                vector_eps_y = eps_y.reshape((-1,))
            else:
                vector_eps_x = EPSILON_0_*(eps_tot).reshape((-1,))
                vector_eps_y = EPSILON_0_*(eps_tot).reshape((-1,))

            T_eps_x_inv = sp.spdiags(1/vector_eps_x, 0, M, M,
                                  format=matrix_format)
            T_eps_y_inv = sp.spdiags(1/vector_eps_y, 0, M, M,
                                  format=matrix_format)

            ex =  1/1j/self.omega * T_eps_y_inv.dot(Dyb).dot(X)
            ey = -1/1j/self.omega * T_eps_x_inv.dot(Dxb).dot(X)

            if self.flatten:
                Ex = ex.flatten()
                Ey = ey.flatten()
                Hz = X.flatten()
            else:
                Ex = ex.reshape((Nx, Ny))
                Ey = ey.reshape((Nx, Ny))
                Hz = X.reshape((Nx, Ny))

            if include_nl==False:
                self.fields['Ex'] = Ex
                self.fields['Ey'] = Ey
                self.fields['Hz'] = Hz

            return (Ex, Ey, Hz)

        elif self.pol == 'Ez':
            hx = -1/1j/self.omega/MU_0_ * Dyb.dot(X)
            hy = 1/1j/self.omega/MU_0_ * Dxb.dot(X)

            if self.flatten:
                Hx = hx.flatten()
                Hy = hy.flatten()
                Ez = X.flatten()
            else:
                Hx = hx.reshape((Nx, Ny))
                Hy = hy.reshape((Nx, Ny))
                Ez = X.reshape((Nx, Ny))

            if include_nl==False:
                self.fields['Hx'] = Hx
                self.fields['Hy'] = Hy
                self.fields['Ez'] = Ez

            return (Hx, Hy, Ez)

        else:
            raise ValueError('Invalid polarization: {}'.format(str(self.pol)))

    def solve_fields_nl(self,
                        timing=False, averaging=False,
                        Estart=None, solver_nl='hybrid', conv_threshold=1e-10,
                        max_num_iter=50,
                        matrix_format=DEFAULT_MATRIX_FORMAT):
        # solves for the nonlinear fields of the simulation.

        # note: F just stands for the field component (could be H or E depending on polarization)
        if solver_nl == 'born':
            (Fx, Fy, Fz, conv_array) = born_solve(self, Estart,
                                                  conv_threshold,
                                                  max_num_iter,
                                                  averaging=averaging)
        elif solver_nl == 'newton':
            (Fx, Fy, Fz, conv_array) = newton_solve(self, Estart,
                                                    conv_threshold,
                                                    max_num_iter,
                                                    averaging=averaging)
        elif solver_nl == 'LM':
            (Fx, Fy, Fz, conv_array) = LM_solve(self, Estart,
                                                conv_threshold,
                                                max_num_iter,
                                                averaging=averaging)
        elif solver_nl == 'krylov':
            (Fx, Fy, Fz, conv_array) = newton_krylov_solve(self, Estart,
                                                conv_threshold,
                                                max_num_iter,
                                                averaging=averaging)
        elif solver_nl == 'hybrid':
            (Fx, Fy, Fz, conv_array) = born_solve(self, Estart,
                                              conv_threshold, 20,
                                              averaging=averaging) 
            if conv_array[-1] > 1e-10:
                (Fx, Fy, Fz, conv_array) = newton_solve(self, Fz,
                                                    conv_threshold, 20,
                                                    averaging=averaging)                                       
        else:
            raise AssertionError("solver must be one of "
                                 "{'born', 'newton', 'LM'}")

            # return final nonlinear fields and an array of the convergences

        if self.pol == 'Ez':

            self.fields_nl['Hx'] = Fx
            self.fields_nl['Hy'] = Fy
            self.fields_nl['Ez'] = Fz

        elif self.pol == 'Hz':

            self.fields_nl['Ex'] = Fx
            self.fields_nl['Ey'] = Fy
            self.fields_nl['Hz'] = Fz

        else:
            raise ValueError('Invalid polarization: {}'.format(str(self.pol)))

        return (Fx, Fy, Fz, conv_array)

    def _check_inputs(self):
        # checks the inputs and makes sure they are kosher

        assert self.L0 > 0, "L0 must be a positive number, was supplied {},".format(str(self.L0))
        assert self.omega > 0, "omega must be a positive number, was supplied {},".format(str(self.omega))
        assert isinstance(self.NPML, list), "NPML must be a list of integers"
        assert len(self.NPML) == 2, "yrange must be a list of length 2, was supplied {}, which is of length {}".format(str(self.NPML), len(self.NPML))
        assert self.NPML[0] >= 0 and self.NPML[1] >= 0, "both elements of NPML must be >= 0"
        assert self.pol in ['Ez', 'Hz'], "pol must be one of 'Ez' or 'Hz'"
        # to do, check for correct types as well.

    def flux_probe(self, direction_normal, center, width, nl=False):
        # computes the total flux across the plane (line in 2D) defined by direction_normal, center, width

        # first extract the slice of the permittivity
        if direction_normal == "x":
            inds_x = [center[0], center[0]+1]
            inds_y = [int(center[1]-width/2), int(center[1]+width/2)]
        elif direction_normal == "y":
            inds_x = [int(center[0]-width/2), int(center[0]+width/2)]
            inds_y = [center[1], center[1]+1]
        else:
            raise ValueError("The value of direction_normal is neither x nor y!")

        if self.pol == 'Ez':

            if nl:
                field_val_Ez = self.fields_nl['Ez']
                field_val_Hy = self.fields_nl['Hy']
                field_val_Hx = self.fields_nl['Hx']
            else:
                field_val_Ez = self.fields['Ez']
                field_val_Hy = self.fields['Hy']
                field_val_Hx = self.fields['Hx']

            Ez_x = grid_average(field_val_Ez[inds_x[0]:inds_x[1]+1, inds_y[0]:inds_y[1]+1], 'x')[:-1,:-1]
            Ez_y = grid_average(field_val_Ez[inds_x[0]:inds_x[1]+1, inds_y[0]:inds_y[1]+1], 'y')[:-1,:-1]
            # NOTE: Last part drops the extra rows/cols used for grid_average

            if direction_normal == "x":
                Sx = -1/2*np.real(Ez_x*np.conj(field_val_Hy[inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]]))
                return self.dl*np.sum(Sx)
            elif direction_normal == "y":
                Sy = 1/2*np.real(Ez_y*np.conj(field_val_Hx[inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]]))
                return self.dl*np.sum(Sy)

        elif self.pol == 'Hz':

            if nl:
                field_val_Hz = self.fields_nl['Hz']
                field_val_Ey = self.fields_nl['Ey']
                field_val_Ex = self.fields_nl['Ex']
            else:
                field_val_Hz = self.fields['Hz']
                field_val_Ey = self.fields['Ey']
                field_val_Ex = self.fields['Ex']

            Hz_x = grid_average(field_val_Hz[inds_x[0]:inds_x[1]+1, inds_y[0]:inds_y[1]+1], 'x')[:-1, :-1]
            Hz_y = grid_average(field_val_Hz[inds_x[0]:inds_x[1]+1, inds_y[0]:inds_y[1]+1], 'y')[:-1, :-1]
            # NOTE: Last part drops the extra rows/cols used for grid_average

            if direction_normal == "x":
                Sx = 1/2*np.real(field_val_Ey[inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]]*np.conj(Hz_x))
                return self.dl*np.sum(Sx)
            elif direction_normal == "y":
                Sy = -1/2*np.real(field_val_Ex[inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]]*np.conj(Hz_y))
                return self.dl*np.sum(Sy)

    def init_design_region(self, design_region, eps_m, style=''):
        """ Initializes the design_region permittivity depending on style"""

        if style == 'full':
            # eps_m filled in design region
            eps_full = self.eps_r.copy()
            eps_full[design_region == 1] = eps_m
            self.eps_r = eps_full

        elif style == 'halfway':
            # halfway between 1 and eps_m in design region
            eps_halfway = self.eps_r
            eps_halfway[design_region == 1] = eps_m/2 + 1/2
            self.eps_r = eps_halfway

        elif style == 'empty':
            # nothing in design region
            eps_empty = self.eps_r.copy()
            eps_empty[design_region == 1] = 1
            self.eps_r = eps_empty

        elif style == 'random':
            # random pixels in design region
            eps_random = (eps_m-1)*np.random.random(self.eps_r.shape)+1
            eps_random[design_region == 0] = self.eps_r[design_region == 0]
            self.eps_r = eps_random

        elif style == 'random_sym':
            # random pixels in design region.  Symmetric across y=0
            R = np.random.random(self.eps_r.shape)
            R_sym = 1/2*(np.fliplr(R) + R)
            eps_random = (eps_m-1)*R_sym+1
            eps_random[design_region == 0] = self.eps_r[design_region == 0]
            self.eps_r = eps_random

        self.rho = eps2rho(self.eps_r, eps_m)

    def compute_index_shift(self):
        """ Computes array of nonlinear refractive index shift"""

        _ = self.solve_fields()
        _ = self.solve_fields_nl()
        index_nl = np.sqrt(np.real(self.eps_r + self.eps_nl))
        index_lin = np.sqrt(np.real(self.eps_r))
        return np.abs(index_nl - index_lin)

    def plt_abs(self, nl=False, cbar=True, outline=True, ax=None, vmin=None, vmax=None, logscale=False, tiled_y=1):
        # plot np.absolute value of primary field (e.g. Ez/Hz)

        if self.fields[self.pol] is None:
            raise ValueError("need to solve the simulation first")

        eps_r = deepcopy(self.eps_r)
        eps_r = np.hstack(tiled_y*[eps_r])

        if nl:
            field_val = np.abs(self.fields_nl[self.pol])
        else:
            field_val = np.abs(self.fields[self.pol])

        field_val = np.hstack(tiled_y*[field_val])

        outline_val = np.abs(eps_r)
        
        if vmax is None:
            vmax = field_val.max()
        if vmin is None:
            vmin = 0.0

        cmap = "magma"

        return plt_base(field_val, outline_val, cmap, vmin, vmax, self.pol,
                        cbar=cbar, outline=outline, ax=ax, logscale=logscale)

    def plt_re(self, nl=False, cbar=True, outline=True, ax=None, tiled_y=1):
        """ Plots the real part of primary field (e.g. Ez/Hz)"""

        eps_r = deepcopy(self.eps_r)
        eps_r = np.hstack(tiled_y*[eps_r])

        if self.fields[self.pol] is None:
            raise ValueError("need to solve the simulation first")

        if nl:
            field_val = np.real(self.fields_nl[self.pol])
        else:
            field_val = np.real(self.fields[self.pol])

        field_val = np.hstack(tiled_y*[field_val])

        outline_val = np.abs(eps_r)
        vmin = -np.abs(field_val).max()
        vmax = +np.abs(field_val).max()
        cmap = "RdBu"

        return plt_base(field_val, outline_val, cmap, vmin, vmax, self.pol,
                        cbar=cbar, outline=outline, ax=ax)

    def plt_diff(self, cbar=True, outline=True, ax=None, vmax=None, tiled_y=1,
                 normalize=True):
        """ Plots the difference between |E| and |E_nl|"""

        # get the outline value
        eps_r = deepcopy(self.eps_r)
        eps_r = np.hstack(tiled_y*[eps_r])
        outline_val = np.abs(eps_r)

        # get the fields and tile them
        field_lin = np.abs(self.fields[self.pol])
        field_lin = np.hstack(tiled_y*[field_lin])
        field_nl = np.abs(self.fields_nl[self.pol])
        field_nl = np.hstack(tiled_y*[field_nl])

        # take the difference, normalize by the max E_lin field if desired
        field_diff = field_lin - field_nl
        if normalize:
            field_diff = field_diff/np.abs(field_lin).max()

        # set limits
        if vmax is None:
            vmax = np.abs(field_diff).max()
        vmin = -vmax

        return plt_base(field_diff, outline_val, 'RdYlBu', vmin, vmax,
                        self.pol, cbar=cbar, outline=outline, ax=ax)

    def plt_eps(self, cbar=True, outline=True, ax=None, tiled_y=1):
        # plot the permittivity distribution

        eps_r = deepcopy(self.eps_r)
        eps_r = np.hstack(tiled_y*[eps_r])

        eps_val = np.abs(eps_r)
        outline_val = np.abs(eps_r)
        vmin = np.abs(eps_r).min()
        vmax = np.abs(eps_r).max()
        cmap = "Greys"

        return plt_base_eps(eps_val, outline_val, cmap, vmin, vmax, cbar=cbar,
                            outline=outline, ax=ax)
