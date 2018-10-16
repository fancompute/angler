import matplotlib.pylab as plt
import numpy as np
import copy

from fdfdpy import Simulation
from structures import two_port, three_port

eps = np.load('data/eps_r_final.npy')

lambda0 = 2e-6              # free space wavelength (m)
c0 = 3e8                    # speed of light in vacuum (m/s)
omega = 2*np.pi*c0/lambda0  # angular frequency (2pi/s)
dl = 0.8e-1                 # grid size (L0)
NPML = [15, 15]             # number of pml grid points on x and y borders
pol = 'Ez'                  # polarization (either 'Hz' or 'Ez')
source_amp = 1              # amplitude of modal source (A/L0^2?)    L = 7           # length of box (L0)

L = 5         # length of box (L0)
H = 5           # height of box (L0)
w = .3          # width of waveguides (L0)
d = H/1.5       # distance between waveguides (L0)
l = 5           # length of waveguide from PML to box (L0)
spc = 3         # space between box and PML (L0)

n_index = 2.44              # refractive index
eps_m = n_index**2          # relative permittivity
chi3 = 4.1*1e-19            # Al2S3 from Boyd (m^2/V^2)

# poor man's binarization
# eps = eps_m*(eps > (eps_m/2 + 1/2)) + 1*(eps <= (eps_m/2 + 1/2))

_, design_region = three_port(L, H, w, d, l, spc, dl, NPML, eps_m)

(Nx, Ny) = eps.shape

nx, ny = Nx//2, Ny//2

# make the simulation
simulation = Simulation(omega, eps, dl, NPML, pol)
simulation.add_mode(np.sqrt(eps_m), 'x', [NPML[0]+int(l/2/dl), ny], int(H/2/dl), scale=source_amp)
simulation.setup_modes()

# compute straight line simulation
eps_r_wg, _ = two_port(L, H, w, l, spc, dl, NPML, eps_start=eps_m)
(Nx_wg, Ny_wg) = eps_r_wg.shape
nx_wg, ny_wg = int(Nx_wg/2), int(Ny_wg/2)            # halfway grid points     
simulation_wg = Simulation(omega, eps_r_wg, dl, NPML, 'Ez')
simulation_wg.add_mode(np.sqrt(eps_m), 'x', [NPML[0]+int(l/2/dl), ny_wg], int(H/2/dl), scale=source_amp)
simulation_wg.setup_modes()

# compute normalization
sim_out = Simulation(omega, eps_r_wg, dl, NPML, 'Ez')
sim_out.add_mode(np.sqrt(eps_m), 'x', [-NPML[0]-int(l/2/dl), ny], int(H/2/dl))
sim_out.setup_modes()
J_out = np.abs(sim_out.src)
(_, _, Ez_wg) = simulation_wg.solve_fields()
SCALE = np.sum(np.square(np.abs(Ez_wg))*J_out)
print('computed a scale of {} in units of E^2 J_out'.format(SCALE))

# add nonlinearity
nl_region = copy.deepcopy(design_region)
simulation.nonlinearity = []  # This is needed in case you re-run this cell, for example (or you can re-initialize simulation every time)
simulation.add_nl(chi3, nl_region, eps_scale=True, eps_max=eps_m)

import autograd.numpy as npa
def J(e, e_nl, eps):
    """ objective function of BOOST means perfect separation between linear and nonlinear
        note: need BOOST if the objective function is too low (source amplitude is low).
        In this case LBFGS just converges and returns before it's actually done.
        Working on this.
    """
    BOOST = 1e7
    linear_out =     1*npa.sum(npa.square(npa.abs(e))*J_out)
    nonlinear_out = -1*npa.sum(npa.square(npa.abs(e_nl))*J_out)
    objfn = linear_out + nonlinear_out
    objfn_binary = objfn
    return objfn_binary / SCALE * BOOST

Nb = 100
Js = []
eps_range = np.linspace(1, eps_m, Nb)
for e in eps_range:
    eps_test = copy.deepcopy(simulation.eps_r)
    eps_test[simulation.eps_r<e] = 1
    eps_test[simulation.eps_r>=e] = eps_m
    sim_test = copy.deepcopy(simulation)
    sim_test.eps_r = eps_test
    (_, _, Ez) = sim_test.solve_fields()
    (_, _, Ez_nl, _) = sim_test.solve_fields_nl()
    objfn = J(Ez, Ez_nl, eps_test)
    print("threshold: {:.2f} -> {:.2f}".format(e, objfn))
    Js.append(objfn)
plt.plot(eps_range, Js)
plt.show()

