# Rainbowfish

Rainbowfish is a package for performing inverse design of optical structures.

It supports linear and nonlinear devices.

## Examples

There are several jupyter notebook examples in the `Notebooks/` directory.  The most up to date and thorough example is 

	Notebooks/Three_Port_Al2S3.ipynb

which provives the code used to generate the results in the paper for an Al2S3 1 -> 2 port nonlinear optical switch.

## Creating devices

In `structures.py` one may define functions for creating permittivity distributions based on geometric parameters.

Right now, only the `three_port` system is included, which is a dielectric box with 1 input waveguide and 2 output wavguides.  The following geometric parameters may be specified:


	L     		# length of box (L0)
	H     		# height of box (L0)
	w     		# width of waveguides (L0)
	d 	  	    # distance between waveguides (L0)
	l     		# length of waveguide from PML to box (L0)
	spc   		# space between box and PML (L0)
	NPML  		# [num PML grids in x, num PML grids in y]
	eps_start       # starting relative permittivity of device

and then `structures.three_port` can be called to give a permittvity array for this device, which can be used in simulations.


	eps_r = three_port(L, H, w, d, dl, l, spc, NPML, eps_start=eps_m)

The total grid points in X and Y (`Nx` and `Ny`) are solved for and can be obtained by

	(Nx, Ny) = eps_r.shape

## Running optimization

We provide an `Optimization` class in `optimization.py` that may be used to run gradient-based optimization of the permittivity distribution within some design region.  This optimization may be a function of either the linear fields, nonlinear fields, or both.

To setup an optimization, one must define:

The objective function:

	J = {
		'linear':    lambda e_lin: pass,       	          # scalar function of the linear electric fields
		'nonlinear': lambda e_nl: pass,    		  # scalar function of the nonlinear electric fields
		'total':     lambda J_lin, J_nl: pass             # scalar function of J['linear'] and J['nonlinear']
	}

If only `J['linear']` or `J['nonlinear']` are defined, the simulation will simply solve for the linear or nonlinear parts, repsectively.  There is no need to define `J['total']` in this case.

The partial derivatives of the objective function with respect to e_lin, e_nl, :

	dJdE = {
		'linear':    lambda e_lin: pass,       		  # scalar function of the linear electric fields
		'nonlinear': lambda e_nl: pass,    		  # scalar function of the nonlinear electric fields
		'total':     lambda dJdE_lin, dJdE_nl: pass       # scalar function of dJdE['linear'] and dJdE['nonlinear']
	}

Again, if one of these terms was not included in `J`, there is no need to include here either.

The spatial regions defined for the optimization:

	regions = {
		'design':    np.array(),        # numpy array with 1 where permittivity is changed and 0 elsewhere
		'nonlin':    np.array()	        # numpy array with 1 where nonlinear effects are present and 0 elsewhere
	}

`regions['nonlin']` is only needed if the objective function has a nonlinear component.

If the objective function has a nonlinear component one must define the nonlinear function being applied and its derivative:

	nonlin_fns = {
		'eps_nl':   lambda e: pass,    # how relative permittivity changes with the electric field
		'dnl_de':    lambda e: pass     # partial derivative of deps_de with respect to e
	}

With these dictionaries defined, the a new optimization object may be initialilzed by calling:

	optimization = Optimization(Nsteps=Nsteps,
				    J=J,
				    dJdE=dJdE,
				    eps_max=eps_m,
				    step_size=step_size,
				    solver=solver,
				    opt_method=opt_method,
				    max_ind_shift=max_ind_shift)

Where the additional parameters are:

    Nsteps           # how many iterations to run. default = 100
    eps_max          # the maximum allowed relative permittivity (set to your material of interest). default = 0.5
    step_size        # step size for gradient updates. default = 0.01
    field_start      # what field to use to start nonlinear solver ('linear', 'previous').  default = 'linear'
    solver           # nonlinear equation solver (either 'born' or 'newton') default = 'born'
    opt_method       # optimization update method (either 'descent' or 'adam' for gradient descent or ADAM) default = 'adam'
    max_ind_shift    # maximum allowed index shift.  Default = None.  If specified, will adaptively decrease input power to make sure material avoids damage.

With the optimization object initialized, one may now run an optimization on the `Fdfd` `simulation` object that is defined previously defined.  For more details, see [fdfdpy](https://github.com/fancompute/fdfdpy).

	new_eps = optimization.run(simulation, regions=regions, nonlin_fns=nonlin_fns)

This will try to optimize the objective function `J['tot']` and will return a final, optimized permittivity distribution.

One may plot the objective function as a function of iterations by running the method:

	optimization.plt_objs()

If the objective function has both linear and nonlinear components, these will be displayed along with the total objective function.

Running the method
	
	dn = optimization.compute_index_shift(self, simulation, regions, nonlin_fns)

will return a numpy array representing the spatial distribution of the refractive index shift caused by the nonlinear effects.  

	np.max(dn)
	
Will return the max index shift, which is used to determine whether the device is operating above the damage threshold.

## Package Requirements
- fdfdpy
- numpy
- scipy
- matplotlib

## To Do
- [ ] Do Hz polarization sensitivity.
- [ ] Get a structure working with a lower index shift
- [ ] Frequency scanning for bandwidth.

