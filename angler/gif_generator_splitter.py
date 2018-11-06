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
from angler.structures import three_port, two_port

# define the similation constants
lambda0 = 2e-6              # free space wavelength (m)
c0 = 3e8                    # speed of light in vacuum (m/s)
omega = 2*np.pi*c0/lambda0  # angular frequency (2pi/s)
dl = 0.6e-1                   # grid size (L0)
NPML = [20, 20]             # number of pml grid points on x and y borders
pol = 'Ez'                  # polarization (either 'Hz' or 'Ez')
source_amp = 1e-9           # amplitude of modal source (make around 1 for nonlinear effects)

# define material constants
n_index = 2.44              # refractive index
eps_m = n_index**2          # relative permittivity

# geometric parameters for a 1 -> 2 port device
L = 6         # length of box (L0)
H = 4         # height of box (L0)
w = .3        # width of waveguides (L0)
d = H/1.5     # distance between waveguides (L0)
l = 4         # length of waveguide from PML to box (L0)
spc = 2       # space between box and PML (L0)

# define permittivity of three port system
eps_r, design_region = three_port(L, H, w, d, l, spc, dl, NPML, eps_m)
(Nx, Ny) = eps_r.shape
nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points

# make a new simulation object
simulation = Simulation(omega, eps_r, dl, NPML, pol)

# set the input waveguide modal source
simulation.add_mode(neff=np.sqrt(eps_m), direction_normal='x', center=[NPML[0]+int(l/2/dl), ny], width=int(H/2/dl), scale=source_amp)
simulation.setup_modes()

# make a new simulation to get the modal profile of the top output port
top = Simulation(omega, eps_r, dl, NPML, 'Ez')
top.add_mode(neff=np.sqrt(eps_m), direction_normal='x', center=[-NPML[0]-int(l/2/dl), ny+int(d/2/dl)], width=int(H/2/dl))
top.setup_modes()
J_top = np.abs(top.src)

# make a new simulation to get the modal profile of the bottom output port
bot = Simulation(omega, eps_r, dl, NPML, 'Ez')
bot.add_mode(neff=np.sqrt(eps_m), direction_normal='x', center=[-NPML[0]-int(l/2/dl), ny-int(d/2/dl)], width=int(H/2/dl))
bot.setup_modes()
J_bot = np.abs(bot.src)

# compute straight line simulation
eps_r_wg, _ = two_port(L, H, w, l, spc, dl, NPML, eps_start=eps_m)
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
SCALE = np.sum(np.square(np.abs(Ez_wg*J_out)))
J_out = J_out

simulation_wg.plt_abs(outline=True, cbar=True);

J_top = J_top / np.sqrt(SCALE)
J_bot = J_bot / np.sqrt(SCALE)

# changes design region. 'style' can be one of {'full', 'empty', 'halfway', 'random', 'random_sym'}.
np.random.seed(0) 
simulation.init_design_region(design_region, eps_m, style='random')

# define objective function  (equal power transmission to bottom and top)
import autograd.numpy as npa
def J(e, e_nl):
    linear_top =     1*npa.sum(npa.square(npa.abs(e*J_top)))
    linear_bot =     1*npa.sum(npa.square(npa.abs(e*J_bot)))
    objfn = linear_top * linear_bot * 4
    return objfn

# make optimization object
R = 3	       # filter radius of curvature (pixels)  (takes a while to set up as R > 5-10)
beta = 100     # projection strength
eta= 0.50      # projection halfway

temp_plt = Temp_plt(it_plot=1, plot_what=('eps', 'elin'), folder='../data/figs/data/temp_im/')
optimization = Optimization(J=J, simulation=simulation, design_region=design_region, eps_m=eps_m, R=R, beta=beta, eta=eta)
(grad_avm, grad_num) = optimization.check_deriv(Npts=5, d_rho=5e-4)
# print('adjoint gradient   = {}\nnumerical gradient = {}'.format(grad_avm, grad_num))
optimization.run(method='adam', step_size=0.0001, Nsteps=200, temp_plt=temp_plt)