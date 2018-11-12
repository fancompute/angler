import numpy as np
import scipy.sparse as sp
from inspect import signature

from angler.adjoint import adjoint_linear_Ez, adjoint_kerr_Ez, adjoint_linear_Hz
from angler.filter import (eps2rho, rho2eps, get_W, deps_drhob, drhob_drhot,
                    drhot_drho, rho2rhot, drhot_drho, rhot2rhob)
from angler.constants import *
from angler.derivatives import unpack_derivs
from angler.linalg import grid_average

""" This is where the gradients are defined
    These are selected when you define an objective function
"""

def grad_linear_Ez(optimization, dJ, Ez, args):
    """gives the linear field gradient: partial J/ partial * E_lin dE_lin / deps"""

    b_aj = -dJ(*args)
    Ez_aj = adjoint_linear_Ez(optimization.simulation, b_aj)

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
    partial = -dJ(*args)
    partial_vec = partial.reshape((-1,))

    MU_0_ = MU_0*optimization.simulation.L0
    omega = optimization.simulation.omega

    b_aj_vec = -1/1j/omega/MU_0_ * Dyb.T.dot(partial_vec)
    b_aj = b_aj_vec.reshape(Ez.shape)

    # rest is the same
    Ez_aj = adjoint_linear_Ez(optimization.simulation, b_aj)

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

def grad_linear_Hy(optimization, dJ, Ez, args):
    """gives the linear field gradient: partial J/ partial * E_lin dE_lin / deps"""

    # get the adjoint Ez corresponding to Hx
    Dxb = optimization.simulation.derivs['Dxb']
    partial = -dJ(*args)
    partial_vec = partial.reshape((-1,))

    MU_0_ = MU_0*optimization.simulation.L0
    omega = optimization.simulation.omega

    b_aj_vec = 1/1j/omega/MU_0_ * Dxb.T.dot(partial_vec)
    b_aj = b_aj_vec.reshape(Ez.shape)

    # rest is the same
    Ez_aj = adjoint_linear_Ez(optimization.simulation, b_aj)

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
    Ez_aj = adjoint_kerr_Ez(optimization.simulation, b_aj)
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


def grad_kerr_Hx(optimization, dJ, Ez_nl, args):

    # get the adjoint source corresponding to Hx
    Dyb = optimization.simulation.derivs['Dyb']
    partial = -dJ(*args)
    partial_vec = partial.reshape((-1,))

    MU_0_ = MU_0*optimization.simulation.L0
    omega = optimization.simulation.omega

    b_aj_vec = -1/1j/omega/MU_0_ * Dyb.T.dot(partial_vec)
    b_aj = b_aj_vec.reshape(Ez.shape)

    # everything else is the same
    Ez_aj = adjoint_kerr_Ez(optimization.simulation, b_aj)
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



def grad_kerr_Hy(optimization, dJ, Ez_nl, args):
    """gives the linear field gradient: partial J/ partial * E_lin dE_lin / deps"""

    # get the adjoint source corresponding to Hy
    Dxb = optimization.simulation.derivs['Dxb']
    partial = -dJ(*args)
    partial_vec = partial.reshape((-1,))

    MU_0_ = MU_0*optimization.simulation.L0
    omega = optimization.simulation.omega

    b_aj_vec = 1/1j/omega/MU_0_ * Dxb.T.dot(partial_vec)
    b_aj = b_aj_vec.reshape(Ez.shape)

    # everything else is the same
    Ez_aj = adjoint_kerr_Ez(optimization.simulation, b_aj)
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




####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################


def grad_linear_Hz(optimization, dJ, Hz, args, averaging=False):

    EPSILON_0_ = EPSILON_0 * optimization.simulation.L0
    MU_0_ = MU_0*optimization.simulation.L0
    omega = optimization.simulation.omega
    Dxb = optimization.simulation.derivs['Dxf']
    Dyb = optimization.simulation.derivs['Dyf']

    eps_tot = optimization.simulation.eps_r
    M = eps_tot.size

    if averaging:
        eps_x = grid_average(EPSILON_0_*(eps_tot), 'x')
        vector_eps_x = eps_x.reshape((-1,))
        eps_y = grid_average(EPSILON_0_*(eps_tot), 'y')
        vector_eps_y = eps_y.reshape((-1,))
    else:
        vector_eps_x = EPSILON_0_*(eps_tot).reshape((-1,))
        vector_eps_y = EPSILON_0_*(eps_tot).reshape((-1,))

    T_eps_x_inv = sp.spdiags(1/vector_eps_x, 0, M, M,
                          format=DEFAULT_MATRIX_FORMAT)
    T_eps_y_inv = sp.spdiags(1/vector_eps_y, 0, M, M,
                          format=DEFAULT_MATRIX_FORMAT)

    # get fields
    Hz_vec = np.reshape(Hz, (-1,))
    Ex_vec =  1/1j/omega * T_eps_y_inv.dot(Dyb).dot(Hz_vec)
    Ey_vec = -1/1j/omega * T_eps_x_inv.dot(Dxb).dot(Hz_vec) 

    # adjoint
    # partial = -dJ(*args)
    # partial_vec = partial.reshape((-1,))

    b_aj = -dJ(*args)
    (Ex_aj, Ey_aj) = adjoint_linear_Hz(optimization.simulation, b_aj, averaging=averaging)   
    Ex_aj_vec = np.reshape(Ex_aj, (-1,)).T
    Ey_aj_vec = np.reshape(Ey_aj, (-1,)).T

    # set up filtering bits
    rho = optimization.simulation.rho
    rho_t = rho2rhot(rho, optimization.W)
    rho_b = rhot2rhob(rho_t, eta=optimization.eta, beta=optimization.beta)
    eps_mat = (optimization.eps_m - 1)

    filt_mat = drhot_drho(optimization.W)
    proj_mat = drhob_drhot(rho_t, eta=optimization.eta, beta=optimization.beta)

    if averaging:

        design_region_x = grid_average(optimization.design_region, 'x')
        dAdeps_x = design_region_x*omega**2*EPSILON_0_            
        design_region_y = grid_average(optimization.design_region, 'y')
        dAdeps_y = design_region_y*omega**2*EPSILON_0_
        dAdeps_vec_x = np.reshape(dAdeps_x, (-1,))
        dAdeps_vec_y = np.reshape(dAdeps_y, (-1,))
        dfdrho_x = eps_mat*filt_mat.multiply(Ex_vec*proj_mat*dAdeps_vec_x)
        dfdrho_y = eps_mat*filt_mat.multiply(Ey_vec*proj_mat*dAdeps_vec_x)
        sensitivity_vec = dfdrho_x.dot(Ex_aj_vec) + dfdrho_y.dot(Ey_aj_vec)    

    else:

        dAdeps = optimization.design_region*omega**2*EPSILON_0_    # Note: physical constants go here if need be!
        dAdeps_vec = np.reshape(dAdeps, (-1,))
        dfdrho_x = eps_mat*filt_mat.multiply(Ex_vec*proj_mat*dAdeps_vec)
        dfdrho_y = eps_mat*filt_mat.multiply(Ey_vec*proj_mat*dAdeps_vec)
        sensitivity_vec = dfdrho_x.dot(Ex_aj_vec) + dfdrho_y.dot(Ey_aj_vec)    

    return 1*np.real(np.reshape(sensitivity_vec, rho.shape))    

def grad_linear_Ex(optimization, dJ, Hz, args, averaging=False):

    EPSILON_0_ = EPSILON_0 * optimization.simulation.L0
    MU_0_ = MU_0*optimization.simulation.L0
    omega = optimization.simulation.omega
    Dxb = optimization.simulation.derivs['Dxf']
    Dyb = optimization.simulation.derivs['Dyf']

    eps_tot = optimization.simulation.eps_r
    M = eps_tot.size

    if averaging:
        eps_x = grid_average(EPSILON_0_*(eps_tot), 'x')
        vector_eps_x = eps_x.reshape((-1,))
        eps_y = grid_average(EPSILON_0_*(eps_tot), 'y')
        vector_eps_y = eps_y.reshape((-1,))
    else:
        vector_eps_x = EPSILON_0_*(eps_tot).reshape((-1,))
        vector_eps_y = EPSILON_0_*(eps_tot).reshape((-1,))

    T_eps_x_inv = sp.spdiags(1/vector_eps_x, 0, M, M,
                          format=DEFAULT_MATRIX_FORMAT)
    T_eps_y_inv = sp.spdiags(1/vector_eps_y, 0, M, M,
                          format=DEFAULT_MATRIX_FORMAT)

    # get fields
    Hz_vec = np.reshape(Hz, (-1,))
    Ex_vec =  1/1j/omega * T_eps_y_inv.dot(Dyb).dot(Hz_vec)
    Ey_vec = -1/1j/omega * T_eps_x_inv.dot(Dxb).dot(Hz_vec) 

    partial = -dJ(*args)
    partial_vec = partial.reshape((-1,))

    b_aj_vec =  1/1j/omega * T_eps_y_inv.dot(Dyb.T).dot(partial_vec)
    b_aj = b_aj_vec.reshape(Hz.shape)

    (Ex_aj, Ey_aj) = adjoint_linear_Hz(optimization.simulation, b_aj, averaging=averaging)   
    Ex_aj_vec = np.reshape(Ex_aj, (-1,)).T
    Ey_aj_vec = np.reshape(Ey_aj, (-1,)).T

    # set up filtering bits
    rho = optimization.simulation.rho
    rho_t = rho2rhot(rho, optimization.W)
    rho_b = rhot2rhob(rho_t, eta=optimization.eta, beta=optimization.beta)
    eps_mat = (optimization.eps_m - 1)

    filt_mat = drhot_drho(optimization.W)
    proj_mat = drhob_drhot(rho_t, eta=optimization.eta, beta=optimization.beta)

    if averaging:

        design_region_x = grid_average(optimization.design_region, 'x')
        dAdeps_x = design_region_x*omega**2*EPSILON_0_            
        design_region_y = grid_average(optimization.design_region, 'y')
        dAdeps_y = design_region_y*omega**2*EPSILON_0_
        dAdeps_vec_x = np.reshape(dAdeps_x, (-1,))
        dAdeps_vec_y = np.reshape(dAdeps_y, (-1,))
        dfdrho_x = eps_mat*filt_mat.multiply(Ex_vec*proj_mat*dAdeps_vec_x)
        dfdrho_y = eps_mat*filt_mat.multiply(Ey_vec*proj_mat*dAdeps_vec_x)
        sensitivity_vec = dfdrho_x.dot(Ex_aj_vec) + dfdrho_y.dot(Ey_aj_vec)    

    else:

        dAdeps = optimization.design_region*omega**2*EPSILON_0_    # Note: physical constants go here if need be!
        dAdeps_vec = np.reshape(dAdeps, (-1,))
        dfdrho_x = eps_mat*filt_mat.multiply(Ex_vec*proj_mat*dAdeps_vec)
        dfdrho_y = eps_mat*filt_mat.multiply(Ey_vec*proj_mat*dAdeps_vec)
        sensitivity_vec = dfdrho_x.dot(Ex_aj_vec) + dfdrho_y.dot(Ey_aj_vec)    

    return 1*np.real(np.reshape(sensitivity_vec, rho.shape)) 

def grad_linear_Ey(optimization, dJ, Hz, args, averaging=False):

    EPSILON_0_ = EPSILON_0 * optimization.simulation.L0
    MU_0_ = MU_0*optimization.simulation.L0
    omega = optimization.simulation.omega
    Dxb = optimization.simulation.derivs['Dxf']
    Dyb = optimization.simulation.derivs['Dyf']

    eps_tot = optimization.simulation.eps_r
    M = eps_tot.size

    if averaging:
        eps_x = grid_average(EPSILON_0_*(eps_tot), 'x')
        vector_eps_x = eps_x.reshape((-1,))
        eps_y = grid_average(EPSILON_0_*(eps_tot), 'y')
        vector_eps_y = eps_y.reshape((-1,))
    else:
        vector_eps_x = EPSILON_0_*(eps_tot).reshape((-1,))
        vector_eps_y = EPSILON_0_*(eps_tot).reshape((-1,))

    T_eps_x_inv = sp.spdiags(1/vector_eps_x, 0, M, M,
                          format=DEFAULT_MATRIX_FORMAT)
    T_eps_y_inv = sp.spdiags(1/vector_eps_y, 0, M, M,
                          format=DEFAULT_MATRIX_FORMAT)

    # get fields
    Hz_vec = np.reshape(Hz, (-1,))
    Ex_vec =  1/1j/omega * T_eps_y_inv.dot(Dyb).dot(Hz_vec)
    Ey_vec = -1/1j/omega * T_eps_x_inv.dot(Dxb).dot(Hz_vec) 

    partial = -dJ(*args)
    partial_vec = partial.reshape((-1,))

    b_aj_vec = -1/1j/omega * T_eps_x_inv.dot(Dxb.T).dot(partial_vec)
    b_aj = b_aj_vec.reshape(Hz.shape)

    (Ex_aj, Ey_aj) = adjoint_linear_Hz(optimization.simulation, b_aj, averaging=averaging)   
    Ex_aj_vec = np.reshape(Ex_aj, (-1,)).T
    Ey_aj_vec = np.reshape(Ey_aj, (-1,)).T

    # set up filtering bits
    rho = optimization.simulation.rho
    rho_t = rho2rhot(rho, optimization.W)
    rho_b = rhot2rhob(rho_t, eta=optimization.eta, beta=optimization.beta)
    eps_mat = (optimization.eps_m - 1)

    filt_mat = drhot_drho(optimization.W)
    proj_mat = drhob_drhot(rho_t, eta=optimization.eta, beta=optimization.beta)

    if averaging:

        design_region_x = grid_average(optimization.design_region, 'x')
        dAdeps_x = design_region_x*omega**2*EPSILON_0_            
        design_region_y = grid_average(optimization.design_region, 'y')
        dAdeps_y = design_region_y*omega**2*EPSILON_0_
        dAdeps_vec_x = np.reshape(dAdeps_x, (-1,))
        dAdeps_vec_y = np.reshape(dAdeps_y, (-1,))
        dfdrho_x = eps_mat*filt_mat.multiply(Ex_vec*proj_mat*dAdeps_vec_x)
        dfdrho_y = eps_mat*filt_mat.multiply(Ey_vec*proj_mat*dAdeps_vec_x)
        sensitivity_vec = dfdrho_x.dot(Ex_aj_vec) + dfdrho_y.dot(Ey_aj_vec)    

    else:

        dAdeps = optimization.design_region*omega**2*EPSILON_0_    # Note: physical constants go here if need be!
        dAdeps_vec = np.reshape(dAdeps, (-1,))
        dfdrho_x = eps_mat*filt_mat.multiply(Ex_vec*proj_mat*dAdeps_vec)
        dfdrho_y = eps_mat*filt_mat.multiply(Ey_vec*proj_mat*dAdeps_vec)
        sensitivity_vec = dfdrho_x.dot(Ex_aj_vec) + dfdrho_y.dot(Ey_aj_vec)    

    return 1*np.real(np.reshape(sensitivity_vec, rho.shape))

def grad_kerr_Hz(optimization, dJ, Ez_nl, args):
    raise NotImplementedError("need to write gradient for kerr Hz")

def grad_kerr_Ex(optimization, dJ, Ez_nl, args):
    raise NotImplementedError("need to write gradient for kerr Ex")

def grad_kerr_Ey(optimization, dJ, Ez_nl, args):
    raise NotImplementedError("need to write gradient for kerr Ey")

