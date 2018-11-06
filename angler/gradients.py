import numpy as np
from inspect import signature

from angler.adjoint import adjoint_linear, adjoint_kerr
from angler.filter import (eps2rho, rho2eps, get_W, deps_drhob, drhob_drhot,
                    drhot_drho, rho2rhot, drhot_drho, rhot2rhob)
from angler.constants import *
from angler.derivatives import unpack_derivs

""" This is where the gradients are defined
    These are selected when you define an objective function
"""

def grad_linear_Ez(optimization, dJ, Ez, args):
    """gives the linear field gradient: partial J/ partial * E_lin dE_lin / deps"""

    b_aj = -dJ(*args)
    Ez_aj = adjoint_linear(optimization.simulation, b_aj)

    EPSILON_0_ = EPSILON_0*optimization.simulation.L0
    omega = optimization.simulation.omega
    dAdeps = optimization.design_region*omega**2*EPSILON_0_

    rho = optimization.simulation.rho
    rho_t = rho2rhot(rho, optimization.W)
    rho_b = rhot2rhob(rho_t, eta=optimization.eta, beta=optimization.beta)
    eps_mat = (optimization.eps_m - 1)

    filt_mat = drhot_drho(optimization.W)
    proj_mat = drhob_drhot(rho_t, eta=optimization.eta, beta=optimization.beta)

    Ez_vec = np.reshape(Ez, (-1,))

    dAdeps_vec = np.reshape(dAdeps, (-1,))
    dfdrho = eps_mat*filt_mat.multiply(Ez_vec*proj_mat*dAdeps_vec)
    Ez_aj_vec = np.reshape(Ez_aj, (-1,))
    sensitivity_vec = dfdrho.dot(Ez_aj_vec)        

    return 1*np.real(np.reshape(sensitivity_vec, rho.shape))

def grad_linear_Hx(optimization, dJ, Ez, args):
    """gives the linear field gradient: partial J/ partial * E_lin dE_lin / deps"""

    # get the adjoint Ez corresponding to Hx
    Dyb = optimization.simulation.derivs['Dyb']
    partial = -dJ(*args).T
    partial_vec = partial.reshape((-1,)).T

    MU_0_ = MU_0*optimization.simulation.L0
    omega = optimization.simulation.omega

    b_aj_vec = -1/1j/omega/MU_0_ * (Dyb.T.dot(partial_vec.T)).T
    b_aj_vec = b_aj_vec.T
    b_aj = b_aj_vec.reshape(Ez.shape)

    # rest is the same
    Ez_aj = adjoint_linear(optimization.simulation, b_aj)

    EPSILON_0_ = EPSILON_0*optimization.simulation.L0
    omega = optimization.simulation.omega
    dAdeps = optimization.design_region*omega**2*EPSILON_0_

    rho = optimization.simulation.rho
    rho_t = rho2rhot(rho, optimization.W)
    rho_b = rhot2rhob(rho_t, eta=optimization.eta, beta=optimization.beta)
    eps_mat = (optimization.eps_m - 1)

    filt_mat = drhot_drho(optimization.W)
    proj_mat = drhob_drhot(rho_t, eta=optimization.eta, beta=optimization.beta)

    Ez_vec = np.reshape(Ez, (-1,))

    dAdeps_vec = np.reshape(dAdeps, (-1,))
    dfdrho = eps_mat*filt_mat.multiply(Ez_vec*proj_mat*dAdeps_vec)
    Ez_aj_vec = np.reshape(Ez_aj, (-1,))
    sensitivity_vec = dfdrho.dot(Ez_aj_vec)        

    return 1*np.real(np.reshape(sensitivity_vec, rho.shape))

def grad_kerr_Ez(optimization, dJ, Ez_nl, args):
    """gives the linear field gradient: partial J/ partial * E_lin dE_lin / deps"""

    b_aj = -dJ(*args)
    Ez_aj = adjoint_kerr(optimization.simulation, b_aj)
    optimization.simulation.compute_nl(Ez_nl)

    EPSILON_0_ = EPSILON_0*optimization.simulation.L0
    omega = optimization.simulation.omega
    dAdeps = optimization.design_region*omega**2*EPSILON_0_
    dAnldeps = dAdeps + optimization.design_region*omega**2*EPSILON_0_*optimization.simulation.dnl_deps

    rho = optimization.simulation.rho
    rho_t = rho2rhot(rho, optimization.W)
    rho_b = rhot2rhob(rho_t, eta=optimization.eta, beta=optimization.beta)
    eps_mat = (optimization.eps_m - 1)

    filt_mat = drhot_drho(optimization.W)
    proj_mat = drhob_drhot(rho_t, eta=optimization.eta, beta=optimization.beta)

    Ez_vec = np.reshape(Ez_nl, (-1,))

    dfdrho = eps_mat*filt_mat.multiply(Ez_vec*proj_mat*np.reshape(dAnldeps, (-1,)))
    return 1*np.real(np.reshape(dfdrho.dot(np.reshape(Ez_aj, (-1,))), rho.shape))

