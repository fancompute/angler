# angler

`angler` (named for '**a**djoint **n**onlinear **g**radients') is a package for simulating and optimizing optical structures.

It provides a finite-difference frequency-domain (FDFD) solver for simulating for linear and nonlinear devices in the frequency domain.

It also provides an easy to use package for inverse design and optimization of linear and nonlinear devices.

`angler` is released as part of a paper `Adjoint method and inverse design for nonlinear optical devices`, which can be downloaded [here](broken_link).  If you use this package, kindly cite us as:

	BibTeX Citation

## Installation

	python setup.py install
	
## Examples / Quickstart

There are several jupyter notebook examples in the `Notebooks/` directory.

For a good introduction, try:

	Notebooks/Splitter.ipynb

For more specific applications:

#### Electromagnetic simulations

For modeling linear devices with our FDFD solver (no optimization), see

	Notebooks/Simple.ipynb

For modeling nonlinear devices wuth FDFD (no optimization), see 

	Notebooks/Nonlinear_system.ipynb

#### Inverse design & optimization

For examples optimizing a linear device, see 

	Notebooks/Splitter.ipynb
	Notebooks/accelerator.ipynb

For examples of optimizing nonlinear devices, see

	Notebooks/2_port.ipynb
	Notebooks/3_port.ipynb
	Notebooks/T_port.ipynb

## Package Structure

`angler` provides two main classes, `Simulation` and `Optimization`, which perform most of the functionality.

Generally, `Simulation` objects are used to perform FDFD simulations, and `Optimization` classes run inverse design and optimization algorithms over `Simulation`s.  To learn more about how `angler` works and how to use it, please take a look at `angler/README.md` for a more detailed explanation.

## Contributing

`angler` is under development and we welcome suggestions, pull-requests, feature-requests, etc.  Please feel free to get in touch with us.

## Tests

To run all tests:

	python -im unittest discover tests

Or to run individually:
	
	python tests/your_test.py
