from numpy import sqrt

EPSILON_0 = 8.85418782e-12
MU_0 = 1.25663706e-6
C_0 = sqrt(1/EPSILON_0/MU_0)
ETA_0 = sqrt(MU_0/EPSILON_0)

DEFAULT_MATRIX_FORMAT = 'csr'
DEFAULT_SOLVER = 'pardiso'
DEFAULT_LENGTH_SCALE = 1e-6  # microns
