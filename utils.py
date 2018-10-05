import numpy as np
import autograd.numpy as npa
import matplotlib.pylab as plt

""" This is meant to be a place to write convenience functions and objects to
    help with the optimization
"""

def J_bin(eps, eps_max, design_region):
    # normalize so that on boundaries, get 1
    A = npa.power((eps_max - 1) / 2, -2)
    N = npa.sum(design_region)
    eps_masked = eps * design_region
    eps_mid = (eps_max + 1) / 2 * design_region
    squared_diff = A * npa.square(eps_masked - eps_mid)
    return npa.sum(squared_diff) / N

if __name__ == '__main__':
    from numpy.random import random
    eps_m = 6
    N = 1000
    eps = random((N, N))*(eps_m - 1) + 1.0
    plt.imshow(eps)
    plt.colorbar()
    plt.show()    
    design_region = np.zeros((N, N))
    design_region[N//4:3*N//4, N//4:3*N//4] = 1

    penalty = J_bin(eps, eps_m, design_region)

    assert (0 < penalty and penalty < 1)
