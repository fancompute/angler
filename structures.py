import numpy as np


def get_grid(shape, dl):
    # computes the coordinates in the grid

    (Nx, Ny) = shape

    # coordinate vectors
    x_coord = np.linspace(-Nx/2*dl, Nx/2*dl, Nx)
    y_coord = np.linspace(-Ny/2*dl, Ny/2*dl, Ny)

    # x and y coordinate arrays
    xs, ys = np.meshgrid(x_coord, y_coord, indexing='ij')
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


def three_port(L, H, w, d, l, spc, dl, NPML, eps_start):

    # CONSTRUCTS A ONE IN TWO OUT PORT DEVICE
    # L         : box length in L0
    # H         : box width  in L0
    # w         : waveguide widths in L0
    # d         : distance between out waveguide (centers) in L0
    # dl        : grid size in L0
    # shape     : shape of the permittivity output
    # eps_start : starting relative permittivity

    Nx = 2*NPML[0] + int((2*l + L)/dl)       # num. grids in horizontal
    Ny = 2*NPML[1] + int((H + 2*spc)/dl)   # num. grids in vertical
    nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points
    shape = (Nx, Ny)                          # shape of domain (in num. grids)

    y_mid = dl*int(Ny/2-ny)
    y_bot = dl*int(Ny/2-ny-d/2/dl)
    y_top = dl*int(Ny/2-ny+d/2/dl)
    wg_width_px = int(w/dl)

    # x and y coordinate arrays
    xs, ys = get_grid(shape, dl)

    # define regions
    box    = lambda x, y: (np.abs(x) < L/2) * (np.abs(y-y_mid) < H/2)
    wg_in  = lambda x, y: (x < 0)           * (np.abs(y-y_mid) < dl*wg_width_px/2)  # note, this slight offset is to fix gridding issues
    wg_top = lambda x, y: (x > 0)           * (np.abs(y-y_bot) < dl*wg_width_px/2)
    wg_bot = lambda x, y: (x > 0)           * (np.abs(y-y_top) < dl*wg_width_px/2)

    reg_list = [box, wg_in, wg_top, wg_bot]

    eps_r = apply_regions(reg_list, xs, ys, eps_start)
    design_region = apply_regions([box], xs, ys, 2) - 1

    return eps_r, design_region


def two_port(L, H, w, l, spc, dl, NPML, eps_start):

    # CONSTRUCTS A ONE IN ONE OUT PORT DEVICE
    # L         : design region length in L0
    # H         : design region width  in L0
    # w         : waveguide widths in L0
    # l         : distance between waveguide and design region in L0 (x)
    # spc       : spc bewtween PML and top/bottom of design region
    # dl        : grid size in L0
    # NPML      : number of PML grids in [x, y]
    # eps_start : starting relative permittivity

    Nx = 2*NPML[0] + int((2*l + L)/dl)       # num. grids in horizontal
    Ny = 2*NPML[1] + int((H + 2*spc)/dl)   # num. grids in vertical
    nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points
    shape = (Nx, Ny)                          # shape of domain (in num. grids)

    # x and y coordinate arrays
    xs, ys = get_grid(shape, dl)

    # define regions
    box = lambda x, y: (np.abs(x) < L/2) * (np.abs(y) < H/2)
    wg  = lambda x, y: (np.abs(y) < w/2)

    eps_r = apply_regions([wg], xs, ys, eps_start=eps_start)
    design_region = apply_regions([box], xs, ys, eps_start=2) - 1

    return eps_r, design_region


def ortho_port(L, L2, H, H2, w, l, dl, NPML, eps_start):

    # CONSTRUCTS A TOP DOWN, LEFT RIGHT OUT PORT DEVICE
    # L         : waveguide design region length in L0
    # L2        : width of design region in L0
    # H         : waveguide design region height in L0
    # H2        : design region height in L0
    # w         : waveguide widths in L0
    # l         : distance between waveguide and design region in L0 (x)
    # spc       : spc bewtween PML and bottom of design region
    # dl        : grid size in L0
    # NPML      : number of PML grids in [x, y]
    # eps_start : starting relative permittivity

    Nx = 2*NPML[0] + int((2*l + L)/dl)       # num. grids in horizontal
    Ny = 2*NPML[1] + int((2*l + H)/dl)     # num. grids in vertical
    nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points
    shape = (Nx, Ny)                         # shape of domain (in num. grids)

    # x and y coordinate arrays
    xs, ys = get_grid(shape, dl)

    # define regions\n",
    box  = lambda x, y: (np.abs(x) < L/2) * (np.abs(y) < w/2)
    box2 = lambda x, y: (np.abs(x) < L2/2) * (np.abs(y) < H2/2)
    box3 = lambda x, y: (-w/2 < x < w/2+dl/2) * (-H2/2 < y < H/2)
    wg   = lambda x, y: (np.abs(y) < w/2)
    wg2  = lambda x, y: (-w/2 < x < w/2+dl/2) * (y > 0)

    eps_r         = apply_regions([wg, wg2], xs, ys, eps_start=eps_start)
    design_region = apply_regions([box, box2, box3], xs, ys, eps_start=2)

    design_region = design_region - 1

    return eps_r, design_region


def accelerator(beta, gap, lambda0, L, spc, dl, NPML, eps_start):

    # CONSTRUCTS A DIELECTRIC LASER ACCELERATOR STRUCTURE
    # beta      : electron speed / speed of light
    # gap       : gap size in L0
    # lambda0   : free space wavelength in L0
    # L         : length of design region in L0
    # spc       : distance between PML and src.  src and design region
    # dl        : grid size in L0
    # NPML      : number of pml points in x and y
    # eps_start : starting relative permittivity

    Nx = 2*NPML[0] + int((4*spc + 2*L + gap)/dl)
    Ny = int(beta*lambda0/dl)
    nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points
    shape = (Nx, Ny)                          # shape of domain (in num. grids)

    # x and y coordinate arrays
    xs, ys = get_grid(shape, dl)

    des = lambda x, y: (np.abs(x) > gap/2) * (np.abs(x) < gap/2 + L)

    eps_r = apply_regions(des, xs, ys, eps_start)

    return eps_r