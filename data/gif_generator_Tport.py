import numpy as np
import matplotlib.pylab as plt
import copy

# add angler to path (not necessary if pip installed)
import sys
sys.path.append("..")

# import the main simulation and optimization classes
from angler import Simulation, Optimization
from angler.plot import Temp_plt

# import some structure generators
from angler.structures import three_port, two_port, ortho_port

lambda0 = 2e-6              # free space wavelength (m)
c0 = 3e8                    # speed of light in vacuum (m/s)
omega = 2*np.pi*c0/lambda0  # angular frequency (2pi/s)
dl = 0.4e-1                 # grid size (L0)
NPML = [25, 25]             # number of pml grid points on x and y borders
pol = 'Ez'                  # polarization (either 'Hz' or 'Ez')
source_amp = 6             # amplitude of modal source (A/L0^2?)

# material constants
n_index = 2.44              # refractive index
eps_m = n_index**2          # relative permittivity
chi3 = 4.1*1e-19            # Al2S3 from Boyd (m^2/V^2)
# max_ind_shift = 5.8e-3      # maximum allowed nonlinear refractive index shift (computed from damage threshold)

# geometric parameters
L1 = 6         # length waveguides in design region (L0)
L2 = 6          # width of box (L0)
H1 = 6          # height waveguides in design region (L0)
H2 = 6          # height of box (L0)
w = .3          # width of waveguides (L0)
l = 3           # length of waveguide from PML to box (L0)
spc = 2         # space between box and PML (L0)

# define permittivity of three port system
eps_r, design_region = ortho_port(L1, L2, H1, H2, w, l, dl, NPML, eps_m)
(Nx, Ny) = eps_r.shape
nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points

simulation = Simulation(omega,eps_r,dl,NPML,pol)

# set the modal source and probes
simulation = Simulation(omega, eps_r, dl, NPML, 'Ez')
simulation.add_mode(np.sqrt(eps_m), 'x', [NPML[0]+int(l/2/dl), ny], int(H1/2/dl), scale=source_amp)
simulation.setup_modes()

# left modal profile
right = Simulation(omega, eps_r, dl, NPML, 'Ez')
right.add_mode(np.sqrt(eps_m), 'x', [-NPML[0]-int(l/2/dl), ny], int(H1/2/dl))
right.setup_modes()
J_right = np.abs(right.src)

# top modal profile
top = Simulation(omega, eps_r, dl, NPML, 'Ez')
top.add_mode(np.sqrt(eps_m), 'y', [nx, -NPML[1]-int(l/2/dl)], int(L1/2/dl))
top.setup_modes()
J_top = np.abs(top.src)

# compute straight line simulation
eps_r_wg, _ = two_port(L1, H1, w, l, spc, dl, NPML, eps_start=eps_m)
(Nx_wg, Ny_wg) = eps_r_wg.shape
nx_wg, ny_wg = int(Nx_wg/2), int(Ny_wg/2)            # halfway grid points     
simulation_wg = Simulation(omega, eps_r_wg, dl, NPML, 'Ez')
simulation_wg.add_mode(np.sqrt(eps_m), 'x', [NPML[0]+int(l/2/dl), ny_wg], int(Ny/3), scale=source_amp)
simulation_wg.setup_modes()

# compute normalization
sim_out = Simulation(omega, eps_r_wg, dl, NPML, 'Ez')
sim_out.add_mode(np.sqrt(eps_m), 'x', [-NPML[0]-int(l/2/dl), ny], int(Ny/3))
sim_out.setup_modes()
J_out = np.abs(sim_out.src)
(_, _, Ez_wg) = simulation_wg.solve_fields()
SCALE = np.sum(np.square(np.abs(Ez_wg))*J_out)
J_out = J_out

J_right = J_right / SCALE
J_top = J_top / SCALE


# changes design region. 'style' can be in {'full', 'empty', 'halfway', 'random'}
np.random.seed(0)
simulation.init_design_region(design_region, eps_m, style='halfway')

# add nonlinearity
nl_region = copy.deepcopy(design_region)

simulation.nonlinearity = []  # This is needed in case you re-run this cell, for example (or you can re-initialize simulation every time)
simulation.add_nl(chi3, nl_region, eps_scale=True, eps_max=eps_m)

# define objective function
import autograd.numpy as npa
def J(e, e_nl):
    linear_right =     1*npa.sum(npa.square(npa.abs(e))*J_right)
    linear_top =     -1*npa.sum(npa.square(npa.abs(e))*J_top)
    nonlinear_right = -1*npa.sum(npa.square(npa.abs(e_nl))*J_right)
    nonlinear_top =   1*npa.sum(npa.square(npa.abs(e_nl))*J_top)
    objfn = (linear_right + linear_top + nonlinear_right + nonlinear_top)/2
    return objfn

# make optimization object
R = None       # filter radius of curvature (pixels)  (takes a while to set up as R > 5-10)
beta = 1e-8     # projection strength
eta= 0.50      # projection halfway

temp_plt = Temp_plt(it_plot=1, plot_what=('eps', 'elin', 'enl'), folder='../data/figs/data/temp_im/', figsize=(15,4))
optimization = Optimization(J=J, simulation=simulation, design_region=design_region, eps_m=eps_m, R=R, beta=beta, eta=eta)
(grad_avm, grad_num) = optimization.check_deriv(Npts=5, d_rho=5e-4)
# print('adjoint gradient   = {}\nnumerical gradient = {}'.format(grad_avm, grad_num))
optimization.run(method='adam', Nsteps=500, temp_plt=temp_plt)