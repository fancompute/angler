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

#### Setting up a simulation

A `Simulation` object is initialized as

```python
S = Simulation(omega, eps_r, dl, NPML, pol)
```

`omega` is the angular frequency (in radians / second)
`eps_r` is a numpy array specifying the relative permittivity
`dl` is the grid size (in units of micron by default)
`NPML` is a list containing the number of PML grids in x and y directions
`pol` is the polarization (`'Ez'` or `'Hz'`)

Note:  a keyword argument `L0` may be supplied as the default length scale.  All length parameters (including `dl`) are be specified in units of L0.

Note: a reciprocal and non-magnetic system is assumed.  If you want to extend this, feel free to submit a pull request.

#### Sources are exciting!

Current sources may be specified by assigning a numpy array to `S.src`.  This is assumed to be a `Jz` source for `Ez` polarization or `Mz` source for `Hz` polarization.

Modal sources for waveguides may also be specified by using the `Simulation.add_mode()` method.  See the `Simple.ipynb` notebook for an example.

#### Adding nonlinearities

Adding nonlinearity to the system can be done simply as

```python
S.add_nl(chi3, nl_region, eps_scale=True, eps_max=5)
```

`chi3` is the (scalar) third order nonlinear susceptibility in units of m^2/V^2
`nl_region` is a boolean array indicating the spatial extent of the nonlinearity in the domain.
if `eps_scale` is `True`, the nonlinearity will be proportional to the density of material in the `nl_region` where `eps_max` is the maximum relative permittivity.

Note:  Right now only self-frequency, Kerr nonlinearity is currently supported, but feel free to add your own according to the template in the file `rainbowfish/nonlinearity.py`.

#### Solving for fields

## Tests

To run all tests:

	python -im unittest discover tests

Or to run individually:
	
	python tests/your_test.py

