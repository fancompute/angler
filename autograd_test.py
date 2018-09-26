import autograd.numpy as np
from autograd import grad

# number of grid points in X and Y
N = 5

# measuring point
PT = np.zeros((N, N))
PT[N//2, N//2] = 1


# define objective function
def J(e_lin, e_nl, eps):
    return np.sum(np.square(np.abs(e_lin))*PT) \
         + np.sum(np.square(np.abs(e_nl))*PT) \
         - np.max(eps)


# compute partial derivatives with autograd
dJde_lin = grad(J, 0)  # gradient wrt first argument
dJde_nl = grad(J, 1)   # gradient wrt second argument
dJdeps = grad(J, 2)    # gradient wrt third argument


# random linear electric fields
e_lin = np.random.rand(N, N) + 1j*np.random.rand(N, N)
# random nonlinear electric fields
e_nl = np.random.rand(N, N) + 1j*np.random.rand(N, N)
# random permittivity
eps = np.random.rand(N, N)

print('objective function : \n{}\n'.format(J(e_lin, e_nl, eps)))

print('gradient wrt e_lin : \n{}\n'.format(dJde_lin(e_lin, e_nl, eps)))
print('gradient wrt e_nl  : \n{}\n'.format(dJde_nl(e_lin, e_nl, eps)))
print('gradient wrt eps   : \n{}\n'.format(dJdeps(e_lin, e_nl, eps)))
