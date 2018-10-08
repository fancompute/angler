
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pylab as plt
import copy
import autograd.numpy as npa

import sys
sys.path.append("..")

from utils import Binarizer
from fdfdpy import Simulation
from structures import three_port, two_port
from optimization import Optimization

lambda0 = 2e-6              # free space wavelength (m)
c0 = 3e8                    # speed of light in vacuum (m/s)
omega = 2*np.pi*c0/lambda0  # angular frequency (2pi/s)
dl = 1.5e-1                 # grid size (L0)
NPML = [15, 15]             # number of pml grid points on x and y borders
pol = 'Ez'                  # polarization (either 'Hz' or 'Ez')
source_amp = np.sqrt(2)     # amplitude of modal source (A/L0^2?)

# material constants
n_index = 2.44              # refractive index
eps_m = n_index**2          # relative permittivity
chi3 = 4.1*1e-19            # Al2S3 from Boyd (m^2/V^2)
max_ind_shift = 5.8e-3      # maximum allowed nonlinear refractive index shift (computed from damage threshold)

# geometric parameters
L = 7         # length of box (L0)
H = 5         # height of box (L0)
w = .2        # width of waveguides (L0)
d = H/1.5     # distance between waveguides (L0)
l = 5         # length of waveguide from PML to box (L0)
spc = 3       # space between box and PML (L0)

# define permittivity of three port system
eps_r, design_region = three_port(L, H, w, d, l, spc, dl, NPML, eps_m)
(Nx, Ny) = eps_r.shape
nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points

simulation = Simulation(omega,eps_r,dl,NPML,pol)
# compute the grid size the total grid size
print("Computed a domain with {} grids in x and {} grids in y".format(Nx,Ny))
print("The simulation has {} grids per free space wavelength".format(int(lambda0/dl/simulation.L0)))

# check to make sure the number of points in each waveguide is the same
eps = simulation.eps_r
pts_in = np.sum(eps[NPML[0]+3,:] > 1)
pts_top = np.sum(eps[-NPML[0]-3,:ny] > 1)
pts_bot = np.sum(eps[-NPML[0]-3,ny:] > 1)

print('waveguide has {} points in in port'.format(pts_in))
print('waveguide has {} points in top port'.format(pts_top))
print('waveguide has {} points in bottom port'.format(pts_bot))
assert pts_in == pts_top == pts_bot, "number of grid points in each waveguide is not consistent"

# set the modal source and probes
simulation = Simulation(omega, eps_r, dl, NPML, 'Ez')
simulation.add_mode(np.sqrt(eps_m), 'x', [NPML[0]+int(l/2/dl), ny], int(H/2/dl), scale=source_amp)
simulation.setup_modes()

# top modal profile
top = Simulation(omega, eps_r, dl, NPML, 'Ez')
top.add_mode(np.sqrt(eps_m), 'x', [-NPML[0]-int(l/2/dl), ny+int(d/2/dl)], int(H/2/dl))
top.setup_modes()
J_top = np.abs(top.src)

# bottom modal profile
bot = Simulation(omega, eps_r, dl, NPML, 'Ez')
bot.add_mode(np.sqrt(eps_m), 'x', [-NPML[0]-int(l/2/dl), ny-int(d/2/dl)], int(H/2/dl))
bot.setup_modes()
J_bot = np.abs(bot.src)

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

# changes design region. 'style' can be in {'full', 'empty', 'halfway', 'random'}
simulation.init_design_region(design_region, eps_m, style='halfway')

# add nonlinearity
nl_region = copy.deepcopy(design_region)
simulation.nonlinearity = []  # This is needed in case you re-run this cell, for example (or you can re-initialize simulation every time)
simulation.add_nl(chi3, nl_region, eps_scale=True, eps_max=eps_m)

# define objective function
def J(e, e_nl, eps):
    """ objective function of BOOST means perfect separation between linear and nonlinear
        note: need BOOST if the objective function is too low (source amplitude is low).
        In this case LBFGS just converges and returns before it's actually done.
        Working on this.
    """
    BOOST = 1e10
    linear_top =     1*npa.sum(npa.square(npa.abs(e))*J_top)
    linear_bot =     1*npa.sum(npa.square(npa.abs(e))*J_bot)
    nonlinear_top =  1*npa.sum(npa.square(npa.abs(e_nl))*J_top)
    nonlinear_bot =  1*npa.sum(npa.square(npa.abs(e_nl))*J_bot)
    objfn = ( (linear_top - nonlinear_top) + (nonlinear_bot - linear_bot) ) / 2
    return objfn / SCALE * BOOST

binarizer = Binarizer(design_region, eps_m)
J_bin = binarizer.density(J)

# make optimization object and check derivatives
optimization = Optimization(J=J, Nsteps=5, eps_max=eps_m, field_start='linear', nl_solver='newton', max_ind_shift=None)

# # check the derivatives (note, full derivatives are checked, linear and nonlinear no longer separate)
# (grad_avm, grad_num) = optimization.check_deriv(simulation, design_region, Npts=4)
# print('adjoint gradient   = {}\nnumerical gradient = {}'.format(grad_avm, grad_num))

# first run the simulation with LBFGS and binary enforcing constraints
optimization.run(simulation, design_region, method='lbfgs', verbose=False)
optimization.J = J_bin
optimization.Nsteps = 10
optimization.run(simulation, design_region, method='lbfgs', verbose=False)

# plot optimization results
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))

simulation.plt_eps(ax=ax1)
ax1.set_title('final permittivity distribution')

optimization.plt_objs(ax=ax2, norm='power')
ax2.set_yscale('linear')
plt.savefig('../data/test/opt_results', dpi=400)
plt.clf()


# compare the linear and nonlinear fields

# setup subplots
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,5))

# linear fields
(Hx,Hy,Ez) = simulation.solve_fields()
simulation.plt_abs(ax=ax1)
ax1.set_title('linear field')

# nonlinear fields
(Hx_nl,Hy_nl,Ez_nl,_) = simulation.solve_fields_nl()
simulation.plt_abs(ax=ax2)
ax2.set_title('nonlinear field')

# difference
simulation.plt_diff(ax=ax3)
ax3.set_title('|Ez| for linear - nonlinear')

plt.savefig('../data/test/fields', dpi=400)
plt.clf()

# compute the refractive index shift
index_shift = simulation.compute_index_shift()
print('maximum refractive index shift of {}'.format(np.max(index_shift)))
plt.imshow(index_shift.T, cmap='magma', origin='lower')
plt.colorbar()
plt.title('refractive index shift')
plt.savefig('../data/test/shift', dpi=400)
plt.clf()

# compute powers

# input power
W_in = simulation.W_in

# linear powers
(Hx,Hy,Ez) = simulation.solve_fields()
W_top_lin =  simulation.flux_probe('x', [-NPML[0]-int(l/2/dl), ny + int(d/2/dl)], int(H/2/dl))
W_bot_lin  = simulation.flux_probe('x', [-NPML[0]-int(l/2/dl), ny - int(d/2/dl)], int(H/2/dl))

# nonlinear powers
(Hx_nl,Hy_nl,Ez_nl,_) = simulation.solve_fields_nl()
W_top_nl =  simulation.flux_probe('x', [-NPML[0]-int(l/2/dl), ny + int(d/2/dl)], int(H/2/dl))
W_bot_nl  = simulation.flux_probe('x', [-NPML[0]-int(l/2/dl), ny - int(d/2/dl)], int(H/2/dl))


print('linear transmission (top)                 = {:.4f}'.format(W_top_lin/W_in))
print('linear transmission (bottom)              = {:.4f}'.format(W_bot_lin/W_in))
print('nonlinear transmission (top)              = {:.4f}'.format(W_top_nl/W_in))
print('nonlinear transmission (bottom)           = {:.4f}'.format(W_bot_nl/W_in))
print('relative power difference (top)    = {:.2f} %'.format(100*abs(W_top_lin - W_top_nl) / W_top_lin))
print('relative power difference (bottom) = {:.2f} %'.format(100*abs(W_bot_lin - W_bot_nl)  / W_bot_nl))

S = [[W_bot_lin/W_in, W_top_lin/W_in],
     [W_bot_nl/W_in,  W_top_nl/W_in]]
plt.imshow(S, cmap='magma')
plt.colorbar()
plt.title('power matrix')
plt.savefig('../data/test/matrix', dpi=400)
plt.clf()

print('scanning frequency')
freqs, objs, FWHM = optimization.scan_frequency(Nf=50, df=1/100)
optimization.simulation.omega = omega
plt.plot([(f-150e12)/1e9 for f in freqs], objs)
plt.xlabel('frequency difference (GHz)')
plt.ylabel('objective function')
print('computed FWHM of {} (GHz):'.format(FWHM/1e9))

plt.savefig('../data/test/freqs', dpi=400)
plt.clf()

# plot optimization results
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))
simulation.plt_eps(ax=ax1)
ax1.set_title('permittivity distribution after first step')
optimization.plt_objs(ax=ax2, norm='power')
ax2.set_yscale('linear')

np.save('../data/test/eps_r_5_5', simulation.eps_r)

