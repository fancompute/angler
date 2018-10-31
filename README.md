# Rainbowfish

`rainbowfish` is a package for simulating and optimizing optical structures.

It provides a frequency-domain solver for simulating for linear and nonlinear devices.

	"A finite-difference makes an infinite difference"

It also provides tools for inverse design and optimization of linear and nonlinear devices.

This package is released as part of a paper `Adjoint method and inverse design for nonlinear optical devices`, which can be downloaded [here](broken_link).  If you use this package, please cite us as:

	BibTeX Citation

## Installation

	pip install rainbowfish

## Examples / Quickstart

There are several jupyter notebook examples in the `Notebooks/` directory:

### Electromagnetic simulations

For modeling linear devices with our finite-difference frequency-domain solver, see

	Notebooks/Simple.ipynb

For modeling nonlinear devices wuth FDFD, see 

	Notebooks/Nonlinear_system.ipynb

### Inverse design & optimization

For an example of optimizing a linear device, see 

	Notebooks/accelerator.ipynb

For several examples of optimizing nonlinear devices, the notebooks

	Notebooks/2_port.ipynb
	Notebooks/3_port.ipynb
	Notebooks/T_port.ipynb

were used for the devices in the paper.

## Package Structure

`rainbowfish` provides two main classes, `Simulation` and `Optimization`, which perform most of the functionality.

Generally, `Simulation` objects are used to perform FDFD simulations, and `Optimization` classes run inverse design and optimization algorithms over `Simulation`s.

Here we will go over some of the features in each class.

### Simulation

`Simulation` objects are at the core of `rainbowfish` and provide methods for modeling your electromagentic system (solving fields, adding sources and nonlinearities, plotting).

A `Simulation` object is initialized as

```python
simulation = Simulation(omega, eps_r, dl, NPML, pol)
```

`omega` is the angular frequency (in radians / second)
`eps_r` is a numpy array specifying the relative permittivity
`dl` is the grid size (in units of micron by default)
`NPML` is a list containing the number of PML grids in x and y directions
`pol` is the polarization (`'Ez'` or `'Hz'`)

Note:  a keyword argument `L0` may be supplied as the default length scale.  All length parameters (including `dl`) are be specified in units of L0.

Note: a reciprocal and non-magnetic system is assumed.  If you want to extend this, feel free to submit a pull request.

Note: once initialized, the FDFD system matrix is constructed and ready to solve given a source.  If the `simulation.eps_r` is changed, the system matrix will automatically be re-constructed given the mew permittivity.

For convenience, one may set the permittivity within some region, `design_region`, using the following method:

```python
simulation.init_design_region(design_region, eps_m, style='')
```
where `eps_m` is the maximum permittivity and `style` is one of `('full', 'halfway', 'empty', 'random', 'random_sym')`.  

Current sources may be specified by assigning a numpy array to `S.src`.  This is assumed to be a `Jz` source for `Ez` polarization or `Mz` source for `Hz` polarization.

Modal sources for waveguides may also be specified by using the `Simulation.add_mode()` method.  See the `Simple.ipynb` notebook for an example.

The power flux through a line in the simulation domain may be computed using 

```python
simulation.flux_probe(direction_normal='x', center=[0,0], width=10, nl=False)
```
where `direction_normal` is one of `x`, `y` and `center` / `width` define the line in terms of pixels.  If `nl=True`, this will compute and use the nonlinear fields (more on this later).

With the source in place, one can solve for the electric and magnetic fields as 

```python
(Hx, Hy, Ez) = simulation.solve_fields()
```
if `Ez` polarization and
```python
(Ex, Ey, Hz) = simulation.solve_fields()
```
if `Hz` polarization.

Adding nonlinearity to the system can be done simply as

```python
simulation.add_nl(chi3, nl_region, eps_scale=True, eps_max=5)
```

`chi3` is the (scalar) third order nonlinear susceptibility in units of m^2/V^2
`nl_region` is a boolean array indicating the spatial extent of the nonlinearity in the domain.
if `eps_scale` is `True`, the nonlinearity will be proportional to the density of material in the `nl_region` where `eps_max` is the maximum relative permittivity.

Note:  Right now only self-frequency, Kerr nonlinearity is currently supported, but feel free to add your own according to the template in the file `rainbowfish/nonlinearity.py`.

With nonlinearity added, one can solve for the nonlinear fields as 

```python
(Hx, Hy, Ez, conv) = simulation.solve_fields_nl()
```

`rainbowfish` supports a few iterative methods for solving the nonlinear system, including Born/Picard iterations and Newton-Raphson method.  These can be changed with the `solver_nl` keyword argument.

`conv` is a list of the convergence over each iteration of the solver.

With nonlinear regions defined, one may compute the refractive index shift distribution `dn_map` by calling

```python
dn_map = simulation.compute_index_shift()
```

Finally, `Simulation` objects offer the following methods for plotting field patterns and permittivities

```python
simulation.plt_abs()   # || of field z component
simulation.plt_re()    # Re{} of field z component
simulation.plt_diff()  # diff between linear and nonlinear fields
simulation.plt_eps()   # plots relative permittivity
```
more plotting utilities are defined in `rainbowfish/plot.py`

For more info on `Simulation`s, see `rainbowfish/simulation.py`

### Optimization

Whereas `Simulation` objects define the physical system being solved, `Optimization` objects allow one to perform inverse design on top of these `Simulation`s.

Before defining an `Optimization` object one needs to construct a `Simulation` object, define a `design_region` (boolean array) and an objective function `J`.

The objective function must be defined using the `autograd` wrapper for `numpy`.  This package allows us to automatically take partial derivatives of even complicated objective functions, greatly simplifying the process of solving for adjoint sensitivities.  For an example, we can define the objective funcion corresponding to concentration of `Ez` at a single point `P` with the following

```python
import autograd.numpy as npa
def J(Ez, Ez_nl):
	E_P = Ez * P                # Ez at point P
	abs_E_P = npa.abs(E_P)      # |Ez| at point P
	I_P = npa.square(abs_E_P)   # intensity at point P
	return I_P
```

Note: `rainbowfish` assumes you are trying to maximize `J`, so define your objective function accordingly.

Note: Please make sure you use `autograd.numpy` to define operations in your objective function.

Note: Objective functions are generally defined as a function of the linear fields `Ez` and nonlinear fields `Ez_nl`.  If you are working on a purely linear problem, you must still include `Ez_nl` but there is no need to use it within `J`.  We're currently working on a more flexible way to define objective functions.

With our problem defined, we may construct an optimization object simply as 

```python
optimization = Optimization(J, simulation, design_region, eps_m)
```

where `eps_m` is the upper bound on the relative permittivity.  (1 is assumed to be the lower bound).

`rainbowfish` allows users to perform filtering and projection techniques.  Filtering allows one to generate larger feature sizes by performing low-pass spatial filtering to the structures.  Projection is used to take less-than-physical intermediate permittivities and push them towards vacuum or material.

`R` is the feature size of the low pass filter (in pixels).  3-5 is usually a decent size depending on the grid size.  If `R is None` then no filtering will be applied.

`beta=1e-9` is the strength of the projection. Higher `beta` means more binarized strucures that are more difficult to optimize initially.  If `beta` is very close to zero, no projection will be applied.

`eta=0.5` is the bias of the projection towards vacuum (if > 0.5, will be more likely to give vacuum structures.)

Specific parameters of the nonlinear solver can be changed with the optimization initialization as well.  See `rainbowfish/optimization.py` for more details.

With the `optimization` defined, `rainbowfish` will automatically use autograd to compute the partial derivatives of `J` with respect to the fields and solve the corresponding adjoint problems.  The gradient of `J` with respect to the design parameters (representing the relative permittivity) may then be simply computed and stored during optimization.

To check these derivatives against direct numerical approximations:

```python
avm_grads, num_grads = optimization.check_deriv(Npts=5, d_rho=1e-3)
```
where `N_pts` is the number of points to perturb and `d_rho` is the amount to perturb the design parameters for the finite difference derivative approximation.

`avm_grads` and `num_grads` are lists of the derivatives for each of the `Npts` points and can be directly compared.  They should be close within a reasonable relative tolerance although this can depend on the problem, value of `d_rho`, and the projection and filtering parameters.

With the `opimitzation` defined, one may now perform inverse design using

```python
optimization.run(method='LBFGS', Nsteps=100)
```

which will try to find a permittivity to maximize `J` and print progress to the screen.

The `run` method accepts the following keyword arugments:

`method` defines the optimization method and can be one of `LBFGS`, `gd`, `adam`.

For `LBFGS` no other parameters are necesssary. For `gd` (gradient descent), a `step_size` must be supplied.  For `adam`, a `beta1` and `beta2` for the adam update may be supplied as well.

After running, the objective function can be plotted with 

```python
optimization.plt_objs()
```

The objective function vs. frequency can also be computed with 

```python
freqs, objs, FWHM = optimization.scan_frequency(Nf=50, df=1/20)
```
where `Nf` is the number of frequencies and `df` is the frequency range (relative to central frequency).

Power scanning and transmission plotting utilities are available but still a work in progress.  See `rainbowfish/optimization.py` for more details.

## Tests

To run all tests:

	python -im unittest discover tests

Or to run individually:
	
	python tests/your_test.py

