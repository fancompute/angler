import numpy as np


def get_grid(shape, dl):
	# computes the coordinates in the grid

	(Nx,Ny) = shape

	# coordinate vectors
	x_coord = np.linspace(-Nx/2*dl, Nx/2*dl, Nx)
	y_coord = np.linspace(-Ny/2*dl, Ny/2*dl, Ny)

	# x and y coordinate arrays
	xs, ys = np.meshgrid(x_coord, y_coord)
	return (xs, ys)


def apply_regions(reg_list, xs, ys, eps_start):
	# constructs the permittivity given a list of regions

	# if it's not a list, make it one
	if not isinstance(reg_list, list):
		reg_list = [reg_list]

	# initialize permittivity
	eps_r = np.ones(xs.shape)

	# loop through lambdas and apply masks
	for reg in reg_list:
		reg_vec = np.vectorize(reg)
		material_mask = reg_vec(xs, ys)
		eps_r[material_mask] = eps_start

	return eps_r


def three_port(L, H, w, d, dl, shape, eps_start):

	# CONSTRUCTS A ONE IN TWO OUT PORT DEVICE
	# L         : box length in L0
	# H         : box width  in L0
	# w         : waveguide widths in L0
	# d         : distance between out waveguide (centers) in L0
	# dl        : grid size in L0
	# shape     : shape of the permittivity output
	# eps_start : starting relative permittivity

	# x and y coordinate arrays
	xs, ys = get_grid(shape, dl)

	# define regions
	box    = lambda x, y : (np.abs(x) < L/2) * (np.abs(y) < H/2)
	wg_in  = lambda x, y : (x < 0)           * (np.abs(y) < w/2)
	wg_top = lambda x, y : (x > 0)           * (np.abs(y-d/2) < w/2)
	wg_bot = lambda x, y : (x > 0)           * (np.abs(y+d/2) < w/2)

	reg_list = [box, wg_in, wg_top, wg_bot]

	eps_r = apply_regions(reg_list, xs, ys, eps_start)

	return eps_r

