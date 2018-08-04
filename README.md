# nonlinear_avm
inverse design of a nonlinear optical device using AVM

## To Do
- [ ] Clean up code a bit
- [ ] Make Fdfd a submodule of this package
- [ ] Write an adjoint gradient computation for a nonlinear system
- [ ] Write a gradient computation using the RNN-like approach
- [ ] Test a nonlinear optimization
- [x] Handle cases where objective function is a function of the linear field and nonlinear field (for example, supply a `J` dictionary with keys `J_lin` and `J_nonlin` each containing functions for these parts.
