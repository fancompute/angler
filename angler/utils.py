import numpy as np
import autograd.numpy as npa
import matplotlib.pylab as plt
from functools import wraps
from autograd.scipy.signal import convolve

""" This is meant to be a place to write convenience functions and objects to
    help with the optimization
"""

def eps2rho_bar(eps, eps_m):
    return (eps - 1) / (eps_m - 1)

def rho_bar2eps(rho_bar, eps_m):
    return rho_bar * (eps_m - 1) + 1

def deps_drho_bar(eps, eps_m):
    return 1 / (eps_m - 1)

def rho_bar(rho, eta=0.5, beta=100):
    num = npa.tanh(beta*eta) + npa.tanh(beta*(rho - eta))
    den = npa.tanh(beta*eta) + npa.tanh(beta*(1 - eta))
    return num / den

def drho_bar_drho(rho, eta=0.5, beta=100):
    rho_bar = partial(rho_bar, eta=eta, beta=beta)
    grad_rho_bar = grad(rho_bar)
    grad_rho_bar = np.vectorize(grad_rho_bar)
    return grad_rho_bar(rho)


class Binarizer():
    """
    Warning: Experimental Feature!!

        Takes a normal objective function and adds a binarization component to it
        For example:

            binarizer = Binarizer(design_region, eps_m)

            J = lambda e, e_nl, eps: npa.sum(npa.abs(e)) + npa.sum(npa.abs(e_nl))
            J_binarized = binarizer.density(J)

    where now J_binarized has multiplied J by some binary dependence on eps defined below in the density method.
    This can also be done in a fancy way like

        binarizer = Binarizer(design_region, eps_m)

        @binarizer.density
        J = lambda e, e_nl, eps: npa.sum(npa.abs(e)) + npa.sum(npa.abs(e_nl))

    In this case, J is changed in place with the decorator @binarizer.density
    """

    def __init__(self, design_region, eps_m, exp_const=1):

        self.design_region = design_region
        self.eps_m = eps_m
        self.exp_const = exp_const

    def density(self, J):
        """ Multiplies objective function by the density of eps_pixels near boundaries"""

        def J_bin(eps):
            """ Gives a number between 0 and 1 incidating how close each eps_r
                pixel is to being binarized, used to multiply with J()
            """

            # material density in design region
            rho = (eps - 1) / (self.eps_m - 1) * self.design_region

            # number of cells in design region
            N = npa.sum(self.design_region)

            # gray level map
            M_nd = 4 * rho * (1 - rho)

            # gray level indicator
            M_nd_scalar = npa.sum(M_nd) / N

            # average over each cell
            return 1 - M_nd_scalar

        # note, this is the actual generator being returned
        # defines how to combine J_bin (binarization function) with original J
        def J_new(*args, **kwargs):

            eps = args[2]
            return J(*args) * J_bin(eps)

        return J_new

    def density_exp(self, J):
        """ Multiplies objective function by the density of eps_pixels near boundaries"""

        def J_bin(eps):
            """ Gives a number between 0 and 1 incidating how close each eps_r
                pixel is to being binarized, used to multiply with J()
            """

            # material density in design region
            rho = (eps - 1) / (self.eps_m - 1) * self.design_region

            # number of cells in design region
            N = npa.sum(self.design_region)

            # gray level map
            M_nd = 4 * rho * (1 - rho)

            # gray level indicator
            M_nd_scalar = npa.sum(M_nd) / N

            # average over each cell
            # return 1 - M_nd_scalar
            # average over each cell
            return npa.exp(-self.exp_const*npa.abs(M_nd_scalar))

        # note, this is the actual generator being returned
        # defines how to combine J_bin (binarization function) with original J
        def J_new(*args, **kwargs):

            eps = args[2]
            return J(*args) * J_bin(eps)

        return J_new

    def smoothness(self, J):
        """ Multiplies objective function by the measure of smoothness of permittivity distribution"""

        def J_bin(eps):
            """ Gives a number between 0 and 1 incidating how close each eps_r
                pixel is to being binarized, used to multiply with J()
            """
            # material density in design region
            rho = (eps - 1) / (self.eps_m - 1) * self.design_region

            # number of cells in design region
            N = npa.sum(self.design_region)

            # gray level map
            M_nd = 4 * rho * (1 - rho)

            (Nx, Ny) = eps.shape

            width = 1
            N_conv = min(Nx, Ny)
            k = np.zeros((N_conv))
            k[N_conv//2:N_conv//2 + width] = 1/width

            N_conv = 2
            k = np.zeros((N_conv))
            k[1] = 1

            penalty = 0.0
            for i in range(Nx):
                strip_i = M_nd[i,:]
                output_i = convolve(k, strip_i)
                # penalty = penalty + 4*npa.sum(output_i*(1-output_i))
                penalty = penalty + npa.sum(npa.abs(output_i))

            for j in range(Ny):
                strip_j = M_nd[:,j]
                output_j = convolve(k, strip_j.T)
                # penalty = penalty + 4*npa.sum(output_j*(1-output_j))
                penalty = penalty + npa.sum(npa.abs(output_j))

            # print(penalty, N)
            penalty = penalty / N / 2
            return 1 - penalty

        # note, this is the actual generator being returned
        # defines how to combine J_bin (binarization function) with original J
        def J_new(*args, **kwargs):

            eps = args[2]
            # print(J_bin(eps))

            return J(*args) * J_bin(eps)

        return J_new

    # Implement more below vvvv if you want
