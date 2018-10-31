import unittest

from numpy import pi, ones, zeros, square, conj, logspace, array, append, unwrap, angle
from numpy.testing import assert_allclose
from fdfdpy import Simulation

class Test_Chi3(unittest.TestCase):

    def test_chi3(self):
        """Tests whether we get a nonlinear phase shift (self phase modulation) which agrees with analytical predictions"""

        # Set simulation parameters
        n0 = 3.4
        omega = 2*pi*200e12
        dl = 0.01
        chi3 = 2.8E-18

        width  = 1 # WG width
        L      = 5 # WG length
        L_chi3 = 4 # WG nonlinear length

        # Convert to voxels
        width_voxels  = int(width/dl)
        L_chi3_voxels = int(L_chi3/dl)
        Nx = int(L/dl)
        Ny = int(3.5*width/dl)

        # Setup
        eps_r = ones((Nx, Ny))
        eps_r[:,int(Ny/2-width_voxels/2):int(Ny/2+width_voxels/2)] = square(n0)
        nl_region = zeros(eps_r.shape)
        nl_region[int(Nx/2-L_chi3_voxels/2):int(Nx/2+L_chi3_voxels/2), int(Ny/2-width_voxels/2):int(Ny/2+width_voxels/2)] = 1
        simulation = Simulation(omega, eps_r, dl, [15, 15], 'Ez')
        simulation.add_mode(n0, 'x', [17, int(Ny/2)], width_voxels*3)
        simulation.setup_modes()
        simulation.solve_fields()

        # Probe field from linear simulation to get phase
        fld0 = simulation.fields['Ez'][20, int(Ny/2)]
        fld1 = simulation.fields['Ez'][Nx-20, int(Ny/2)]
        T_linear = fld1/fld0

        # Set nonlinear functions
        kerr_nonlinearity = lambda e: 3*chi3/square(simulation.L0)*square(abs(e))
        dkerr_de = lambda e: 3*chi3/square(simulation.L0)*conj(e)

        # Sweep source power and record nonlinear phase accumulation
        srcval_vec = logspace(1, 3, 3)
        pwr_vec = array([])
        T_vec = array([])
        for srcval in srcval_vec:
            simulation.setup_modes()
            simulation.src *= srcval
            simulation.solve_fields_nl(kerr_nonlinearity, nl_region,
                                       dnl_de=dkerr_de, timing=False, averaging=True,
                                       Estart=None, solver_nl='newton')
            fld0 = simulation.fields['Ez'][20, int(Ny/2)]
            fld1 = simulation.fields['Ez'][Nx-20, int(Ny/2)]
            T_vec = append(T_vec, fld1/fld0)
            pwr = simulation.flux_probe('x', [Nx-20, int(Ny/2)], width_voxels*3)
            pwr_vec = append(pwr_vec, pwr)

        # Analytically calculate the expected nonlinear phase accumulation
        n2  = 12*square(pi)*chi3*1e4/square(n0)
        n2 *= 1e-4/square(simulation.L0)

        width = dl*width_voxels
        height = width
        Aeff = width*height

        L = dl*L_chi3_voxels
        gamma_spm = (omega/299792458*simulation.L0)*n2/Aeff

        P            = pwr_vec*height # Power
        Phi_fdfd     = -unwrap(angle(T_vec)-angle(T_linear))/pi # Nonlinear phase in FDFD
        Phi_analytic = (pwr_vec*height)*L*gamma_spm/pi # Analytic nonlinear phase 

        # If our simulation is correct, these values should be equal
        assert_allclose(Phi_fdfd, Phi_analytic, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
