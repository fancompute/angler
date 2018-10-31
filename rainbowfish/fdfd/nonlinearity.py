import numpy as np

from fdfdpy.linalg import *


class Nonlinearity:

    def __init__(self, chi, nl_region, nl_type='kerr', eps_scale=False, eps_max=None):
        self.chi = chi
        self.nl_region = nl_region
        self.nl_type = nl_type
        self.eps_scale = eps_scale
        self.eps_max = eps_max
        self.eps_nl = []
        self.dnl_de = []
        self.dnl_deps = []

        if self.nl_type == 'kerr':
            if self.eps_scale == True:
                if self.eps_max == None:
                    raise AssertionError("Must provide eps_max when eps_scale is True") 

                else:
                    kerr_nonlinearity     = lambda e, eps_r:3*chi*nl_region*np.square(np.abs(e))*((eps_r-1)/(eps_max - 1))
                    kerr_nl_de            = lambda e, eps_r:3*chi*nl_region*np.conj(e)*((eps_r-1)/(eps_max - 1))
                    kerr_nl_deps          = lambda e, eps_r:3*chi*nl_region*np.square(np.abs(e))*(1/(eps_max - 1))

            else:
                kerr_nonlinearity     = lambda e, eps_r:3*chi*nl_region*np.square(np.abs(e))
                kerr_nl_de            = lambda e, eps_r:3*chi*nl_region*np.conj(e)
                kerr_nl_deps          = lambda e, eps_r:0
            self.eps_nl = kerr_nonlinearity
            self.dnl_de = kerr_nl_de
            self.dnl_deps = kerr_nl_deps

        else:
            raise AssertionError("Only 'kerr' type nonlinearity is supported") 
