import numpy as np
from device_saver import load_device

""" Opens a device and prints its stored stats for the paper"""

def get_stats(fname):
    print("\n============================================================")

    D = load_device(fname)

    print('input power of {:.4f} mW/um'.format(D.W_in*1000))

    if hasattr(D, 'index_shift'):
        index_shift = D.index_shift
    else:
        index_shift = D.simulation.compute_index_shift()

    print('index shift: {:.2E}'.format(np.max(index_shift)))

    print('Q-factor: {:.2E}'.format(D.Q))
    print('bandwidth: {:.1f} GHz'.format(D.FWHM / 1e9))

    if D.structure_type == 'two_port':
        print('linear transmission: {:.4f}'.format(D.T_lin))
        print('nonlinear transmission: {:.4f}'.format(D.T_nl))
    elif D.structure_type == 'ortho_port':
        print('linear transmission (right)      = {:.4f} %'.format(100*D.W_right_lin / D.W_in))
        print('linear transmission (top)        = {:.4f} %'.format(100*D.W_top_lin / D.W_in))
        print('nonlinear transmission (right)   = {:.4f} %'.format(100*D.W_right_nl / D.W_in))
        print('nonlinear transmission (top)     = {:.4f} %'.format(100*D.W_top_nl / D.W_in)) 
    print("============================================================\n")

if __name__ == '__main__':

    fname2 = 'data/figs/devices/2_port.p'
    get_stats(fname2)

    fnameT = 'data/figs/devices/T_port.p'
    get_stats(fnameT)    