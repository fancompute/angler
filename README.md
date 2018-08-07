# nonlinear_avm
inverse design of a nonlinear optical device using AVM

# Modules

## nonlinear_solvers
For solving a nonlinear EM problem defined though a field-dependent permittivity using either the Born or the Newton method. See notebook Nonlinear_solvers_check

Syntax: 

convergence_array = born_solve(simulation, b, nonlinear_fn, nl_region, Estart, conv_threshold, max_num_iter)
convergence_array = newton_solve(simulation, b, nonlinear_fn, nl_region, nl_de, Estart, conv_threshold, max_num_iter)

Input:
- simulation: an object of the fdfdpy.Fdfd class (needs to be initialized with the linear permittivity but not necessarily solved)
- b: current density
- nonlinear_fn: an arbitrary function of Ez added to the relative permittivity such that eps_tot = eps_lin + nonlinear_fn(Ez)
- nl_region: region of the simulation space within which the nonlinear function is applied
- (only for newton_solve) nl_de: partial derivative of nonlinar_fn with respect to Ez
- (optional) Estart: starting field for the iteration; if not supplied, it will start from the solution of the linear problem
- (optional) conv_treshold: treshold on the convergence parameter for termination of the iteration; default is 1e-8
- (optional) max_num_iter: maximum number of iterations before termination; default is 10

Output:
- convergence_array: contains the convergence parameter at every step of the iteration
- simulation: the object is updated such that simulation.eps_r contains the nonlinear permittivity and simulation.fields contains the fields computed at the last step of the iteration

## adjoint
For computing the derivative of an objective function that depends on Ez with respect to the relative permittivity at every point within a specified region using the linear or nonlinear adjoint variable method. See notebooks "Linear_gradient" and "Nonlinear_gradient".

**dJdeps_linear()**

Syntax:

dJdeps = dJdeps_linear(simulation, design_region, J, dJdE)

Input:
- simulation: an object of the fdfdpy.Fdfd class that has has already been solved using simulation.solve_fields()
- design_region: the region of simulation space in which the gradient will be computed
- J: the objective function 
- dJdE: function defining the partial derivative of J with respect to Ez

Output:
- dJdeps: the total derivative of J with respect to the permittivity of every point in the design region

**dJdeps_nonlinear()**

Syntax:

dJdeps = dJdeps_nonlinear(simulation, design_region, J, dJdE, nonlinear_fn, nl_region, nl_de)

Input:
- simulation: an object of the fdfdpy.Fdfd class that has has already been solved using born_solve or newton_solve, or \_solve_nl() from the **optimization** module
- design_region: the region of simulation space in which the gradient will be computed
- J: the objective function 
- dJdE: function defining the partial derivative of J with respect to Ez
- nonlinear_fn, nl_region, nl_de: see input to the nonlinear_solvers functions

Output:
- dJdeps: the total derivative of J with respect to the *linear* permittivity of every point in the design region

## optimization
For running an optimization of either a linear or nonlinear problem 

Syntax:

obj_fns = run_optimization(simulation, b, J, dJdE, design_region, Nsteps, eps_max, 
					nonlinear_fn, nl_region, nl_de, field_start, solver, step_size)

Input:
- simulation, object of the fdfdpy.Fdfd class (not necessarily solved)
- b: current density
- J: dictionary of functions J['linear'], J['nonlinear'] and J['total'], defining objective functions that depend on the field solutions to the linear and the nonlinear problems, respectively, as well as how those two are combined
- dJdE: dictionary of functions corresponding to the derivatives of J with respect to Ez
- design_region: the region of simulation space in which the gradient will be computed
- Nsteps: total number of steps in the gradient descent
- eps_max: values above eps_max will be truncated back to eps_max
- (optional) nonlinear_fn, nl_region, nl_de: see input to the nonlinear_solvers functions; default is None as for optimizing a linear system
- (optional) field_start: 'linear' (default) or 'previous', defining whether, at each step of the gradient descent, the nonlinear solve starts from the solution to the linear system or from the nonlinear solution found at the previous step
- (optional) solver: 'born' (default) or 'newton', defines which nonlinear solver will be used
- (optional) step_size: defines the step size in the gradient descent; default is 0.1
          
Output:
obj_fns: the values of J['total'] at every step of the gradient descent
simulation: the object is updated such that simulation.eps_r contains the *linear* permittivity at the final step of the gradient descent; the simulation is *not* solved

## Requirements
- [Fdfdpy](https://github.com/fancompute/fdfdpy) package.
- numpy
- matplotlib
- [pypardiso](https://github.com/haasad/PyPardisoProject)
- scipy
- progressbar

## To Do
- [ ] Clean up code a bit
- [ ] Make Fdfd a submodule of this package
- [x] Write an adjoint gradient computation for a nonlinear system
- [ ] Write a gradient computation using the RNN-like approach
- [x] Test a nonlinear optimization
- [x] Handle cases where objective function is a function of the linear field and nonlinear field (for example, supply a `J` dictionary with keys `J_lin` and `J_nonlin` each containing functions for these parts.
