<link rel="icon" href="/img/favicon.png" type="image/x-icon" />
<img src="/img/anglerlogos/rainbow.png" title="Angler" alt="Angler">

# angler

`angler` (named for '**a**djoint **n**onlinear **g**radients') is a package for simulating and optimizing optical structures.

It provides a finite-difference frequency-domain (FDFD) solver for simulating for linear and nonlinear devices in the frequency domain.

It also provides an easy to use package for adjoint-based inverse design and optimization of linear and nonlinear devices.  For example, you can inverse design optical switches to transport power to different ports for different input powers:

<img src="/img/Tport.gif" title="Fields" alt="Fields">

`angler` is released as part of a paper `Adjoint method and inverse design for nonlinear optical devices`, which can be viewed [here](https://arxiv.org/abs/1811.01255).

## Installation

One can install the most stable version of `angler` and all of its dependencies (apart from MKL) using

	pip install angler
	
Alternatively, to use the most current version

	git clone https://github.com/fancompute/angler.git
	pip install -e angler

And then this directory can be added to path to import angler, i.e.

	import sys
	sys.path.append('path/to/angler')


## Make angler faster

The most computationally expensive operation in `angler` is the sparse linear system solve.  This is done with [`scipy.sparse.linalg.spsolve()`](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.spsolve.html) by default.  If MKL is installed, `angler` instead uses this with a python wrapper [`pyMKL`](https://github.com/dwfmarchant/pyMKL), which makes things significantly faster, depending on the problem.  The best way to install MKL, if using anaconda, is

	conda install MKL
	
(pyMKL does not work when MKL is pip installed.)

## Examples / Quickstart

There are several jupyter notebook examples in the `Notebooks/` directory.

For a good introduction, try:

	Notebooks/Splitter.ipynb

For more specific applications:

#### Electromagnetic simulations

For modeling linear devices with our FDFD solver (no optimization), see

	Notebooks/Linear_system.ipynb

For modeling nonlinear devices with FDFD (no optimization), see 

	Notebooks/Nonlinear_system.ipynb

#### Inverse design & optimization

For examples of optimizing linear devices, see 

	Notebooks/Splitter.ipynb
	Notebooks/Accelerator.ipynb

For examples of optimizing nonlinear devices, see

	Notebooks/2_port.ipynb
	Notebooks/3_port.ipynb
	Notebooks/T_port.ipynb

## Package Structure

`angler` provides two main classes, `Simulation` and `Optimization`, which perform most of the functionality.

Generally, `Simulation` objects are used to perform FDFD simulations, and `Optimization` classes run inverse design and optimization algorithms over `Simulation`s.  To learn more about how `angler` works and how to use it, please take a look at [angler/README.md](angler/README.md) for a more detailed explanation.

## Tests

To run all tests:

	python -m unittest discover tests

Or to run individually:
	
	python tests/individual_test.py

## Contributing

`angler` is under development and we welcome suggestions, pull-requests, feature-requests, etc.

If you contribute a new feature, please also write a few tests and document your changes in [angler/README.md](angler/README.md) or the wiki.

## Authors

`angler` was written by Tyler Hughes, Momchil Minkov, and Ian Williamson.

## Citing

If you use `angler`, please cite us using

	@article{hughes2018adjoint,
	  title={Adjoint method and inverse design for nonlinear nanophotonic devices},
	  author={Hughes, Tyler W and Minkov, Momchil and Williamson, Ian AD and Fan, Shanhui},
	  journal={ACS Photonics},
	  year={2018},
	  publisher={ACS Publications}
	}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. Copyright 2018 Tyler Hughes.

## Acknowledgments

* our logo was made by [Nadine Gilmer](http://nadinegilmer.com/) :)
* RIP Ian's contributions before the code merge
* We made use of a lot of code snippets (and advice) from [Jerry Shi](https://yujerryshi.github.io/index.html)
