import numpy as np
import autograd.numpy as npa
import matplotlib.pylab as plt
from functools import wraps

""" This is meant to be a place to write convenience functions and objects to
    help with the optimization
"""


class Binarizer():
    """
    Warning: Experimental Feature!!

        Basically takes a normal objective function and adds a binarization component to it
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

    def __init__(self, design_region, eps_m, strength=1):

        self.design_region = design_region
        self.eps_m = eps_m

        # some methods below may require a tunable strength parameter
        self.strength = strength

    def density(self, J):
        """ Multiplies objective function by the density of eps_pixels near boundaries"""

        def J_bin(eps):
            """ Gives a number between 0 and 1 incidating how close each eps_r
                pixel is to being binarized, used to multiply with J()
            """

            A = npa.power((self.eps_m - 1) / 2, -2)

            # number of cells to average over
            N = npa.sum(self.design_region)

            # eps in design region
            eps_masked = eps * self.design_region

            # midpoint permittivity map
            eps_mid = (self.eps_m + 1) / 2 * self.design_region

            f_eps = A * npa.square(eps_masked - eps_mid)

            # average over each cell
            return npa.sum(f_eps) / N

        # note, this is the actual generator being returned
        # defines how to combine J_bin (binarization function) with original J
        def J_new(*args, **kwargs):

            eps = args[2]
            return J(*args) * J_bin(eps)

        return J_new

    # Implement more below vvvv if you want