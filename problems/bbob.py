import numpy as np
from pymoo.problems.many import *
from pymoo.core.problem import Problem
import cocoex

bbob_suite = cocoex.Suite('bbob', '', '')


class bbob_single_pymmo(Problem):
    def __init__(self, fun, dim, instance, **kwargs):
        super().__init__(n_var=dim, n_obj=1, xl=0, xu=1, vtype=float, **kwargs)
        self.fun = fun
        self.dim = dim
        self.instance = instance
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def _evaluate(self, x, out, *args, **kwargs):
        func = bbob_suite.get_problem_by_function_dimension_instance(self.fun, self.dim, self.instance)
        x = x * (self.ub - self.lb)[np.newaxis, :] + self.lb[np.newaxis, :]
        out["F"] = np.array([func(i) for i in x])

    def _evaluate_indi(self, x, *args, **kwargs):
        func = bbob_suite.get_problem_by_function_dimension_instance(self.fun, self.dim, self.instance)
        x = x * (self.ub - self.lb)[np.newaxis, :] + self.lb[np.newaxis, :]
        return func(x.flatten()),


class bbob_single_botorch:
    def __init__(self, fun, dim, instance, **kwargs):
        self.fun = fun
        self.dim = dim
        self.instance = instance
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)
        self.func = bbob_suite.get_problem_by_function_dimension_instance(self.fun, self.dim, self.instance)

    def __call__(self, x):
        x = x * (self.ub - self.lb)[np.newaxis, :] + self.lb[np.newaxis, :]

        return np.atleast_2d([-self.func(i) for i in x]).T


if __name__ == "__main__":
    p = bbob_single_botorch(1, 10, 1)
    p.__call__([[0 for i in range(10)]])
