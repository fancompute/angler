import autograd.numpy as npa


class ObjFn():

    """ This class is meant to store the objective function, keep track of its
        arguments and their types, and compute gradients through the adjoint
        method.

        It is still a work in progress as I'm not sure we need it for this
        project but it might be good to have later for defining more custom
        objetive functions.  For example, J dependence on:

        - E_nl evaluated at several different powers.
        - E evaluated with a uniform index shift on material (E0 effect).
        - dependence on source or source amplitude.
    """

    def __init__(self):
        pass

    def define_arguments(self, arg_list):
        """ This would be where the user defines each of the arguments to the
            objective function.  Specifying which position in the objective
            function each comes and which corresponding adjoint solver to use.
            Some changes to adjoint.py should be made to make this simpler.

            Ex:
            arg_list is a list of named tuples, specifying order in obj_fn,
            argument name (for humans) and adjoint.py function to use.

                import collections
                import adjoint
                arg = collections.namedtuple('Argument',['name', 'partial', 'adjoint_fn'])
                arg_list = [('linear_field',    , None, adjoint.dJdE_linear),
                            ('nonlinear_field', , None, adjoint.dJdE_nonlinear),
                            ('permittivity',    , None, adjoint.dJdeps),
                            ('source_amp',      , None, adjoint.dJdb),
                            ('EO_field',        , None, adjoint.dJdE_shift)
                           ]

            Then, later, these args are assumed to be passed into J:

                def J(e, e_nl, eps, b, e_eo):
                    pass
        """
        pass

    def define_objfn(self, function):
        """ This is where the user adds the objective function defined by
            autograd functions.  This function will:

            - do some error checking.
            - link the objective function with the arguments defined earlier.
            - perform autograd to find the partials (below)
            - store this and wait for compute_gradients() to be called.
        """
        pass

    def compute_partials(self, simulation):
        """ Performs autograd for each of the arguments in the objective
            function.  Stores these in the arg_list.

            for index, arg in enumerate(arg_list):
                arg.partial = grad(self.J, index)
        """
        pass

    def compute_gradient(self):
        pass