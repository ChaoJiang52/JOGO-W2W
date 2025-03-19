import os
import numpy as np
import math
import warnings
from dataclasses import dataclass
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
import torch
from torch.quasirandom import SobolEngine
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.exceptions import BadInitialCandidatesWarning
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from problems.bbob import bbob_single_botorch, bbob_single_pymmo
import cma
import random
import array
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from scipy.optimize import minimize


# run GE, DE, and CMA-ES
def run_so(so_name, problem, instance, budget, save, popsize=100):
    path = f'results/{so_name}/{so_name}_bbob_{problem.fun}_d{problem.dim}_i{instance}_{budget}.npz'

    if os.path.isfile(path):
        print("already finished")
        return None

    if so_name == "GA":
        algorithm = GA(
            pop_size=popsize,
            crossover=SBX(prob=1, prob_var=0.5, eta=2),
            mutation=PM(prob=1 / problem.n_var, eta=20),
            sampling=FloatRandomSampling(),
            eliminate_duplicates=True)

        res = minimize(problem,
                       algorithm,
                       termination=('n_eval', budget),
                       save_history=True,
                       )
        X = []
        F = []
        for i, t in enumerate(res.history):
            X.extend([indi.X for indi in t.pop])
            F.extend([indi.F for indi in t.pop])

        final_X = np.array(X)
        final_F = np.array(F)

    elif so_name == "DE":
        # Problem dimension
        NDIM = problem.n_var
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # minimisation
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NDIM)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selRandom, k=3)
        toolbox.register("best", tools.selBest, k=1)
        toolbox.register("evaluate", problem._evaluate_indi)

        X = []
        F = []

        # Differential evolution parameters
        CR = 0.9
        _F = 0.5
        MU = popsize
        NGEN = int(budget / popsize)

        pop = toolbox.population(n=MU)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        X.extend([list(i) for i in pop])
        F.extend([i.fitness.values[0] for i in pop])

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(pop), **record)
        print(logbook.stream)

        for g in range(1, NGEN):
            # xbest, = toolbox.best(pop)
            for k, agent in enumerate(pop):
                a, b, c = toolbox.select(pop)
                y = toolbox.clone(agent)
                index = random.randrange(NDIM)
                for i, value in enumerate(agent):
                    if i == index or random.random() < CR:
                        # DE/rand/1/bin
                        y[i] = a[i] + _F * (b[i] - c[i])
                y.fitness.values = toolbox.evaluate(y)
                if y.fitness > agent.fitness:
                    pop[k] = y
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=g, evals=len(pop), **record)
            print(logbook.stream)
            X.extend([list(i) for i in pop])
            F.extend([i.fitness.values[0] for i in pop])

        print("Best individual is ", hof[0], hof[0].fitness.values[0])
        final_X = np.array(X)
        final_F = np.array(F)

    elif so_name == "CMAES":
        X = []
        F = []

        while len(F) < budget:
            x0 = np.random.random(problem.n_var)
            options = {'maxfevals': budget, 'seed': instance, 'bounds': [0, 1],
                       'popsize': int(4 + 3 * np.log(problem.dim))}
            es = cma.CMAEvolutionStrategy(x0, 0.2, options)

            while not es.stop():
                solutions = es.ask()
                fx = problem.evaluate(solutions).flatten()
                es.tell(solutions, fx)
                es.disp()
                # store solutions
                X.extend(solutions)
                F.extend(fx)

        final_X = np.array(X)
        final_F = np.array(F)

    if save:
        # np.savez(path, X=final_X, F=final_F)
        end_idx = np.min([final_F.shape[0], budget])
        np.savez(path, F=final_F[:end_idx])

    print(problem.fun, problem.dim, final_F.min())


# run vanilla BO
def run_bo(problem, instance, budget, save):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Problem function: {problem.fun}, Dimension: {problem.dim}")

    path = f'results/BO/BO_bbob_{problem.fun}_d{problem.dim}_i{instance}_{budget}.npz'

    if os.path.isfile(path):
        data = np.load(path)
        train_X = torch.from_numpy(data["X"]).to(device)
        train_Y = torch.from_numpy(data["F"]).to(device)
        if train_X.shape[0] >= budget:
            print("Optimisation already finished")
            return None
    else:
        num_initial = 2 * problem.dim
        sobol = SobolEngine(dimension=problem.dim, scramble=True)
        train_X = sobol.draw(n=num_initial, dtype=torch.double).to(device)
        train_Y = torch.from_numpy(problem.__call__(train_X.cpu())).to(device)

    for i in range(budget - train_X.shape[0]):
        outcome_transform = Standardize(m=1).to(device)
        model = SingleTaskGP(train_X, train_Y, covar_module=MaternKernel(nu=2.5),
                             outcome_transform=outcome_transform).to(device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device)
        fit_gpytorch_mll(mll)

        EI = ExpectedImprovement(model, best_f=train_Y.max()).to(device)

        bounds = torch.stack([torch.zeros(problem.dim), torch.ones(problem.dim)])
        candidate, acq_value = optimize_acqf(
            EI, bounds=bounds, q=1, num_restarts=10, raw_samples=512,
        )

        new_candidate = torch.from_numpy(np.array(candidate.cpu())).to(device)
        new_Y = torch.from_numpy(problem.__call__(candidate.cpu())).to(device)

        train_X = torch.cat((train_X, new_candidate))
        train_Y = torch.cat((train_Y, new_Y))

        best_idx = train_Y.argmax()

        print(f"New sample: X = {train_X[-1]}, Y = {train_Y[-1]}")
        print(f"Best sample: X = {train_X[best_idx]}, Y = {train_Y[best_idx]}")
        print(f"Total samples: {train_X.shape[0]}")
        if save:
            np.savez(path, X=train_X.cpu(), F=train_Y.cpu())


# run TuRBO
def run_TuRBO(f, dim, instance, budget, save):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    acf = "ts"
    path = f'results/TuRBO/TuRBO_bbob_{f}_d{dim}_i{instance}_{budget}.npz'
    try:
        if os.path.isfile(path):
            print(path)
            data = np.load(path)
            save_X = torch.from_numpy(data["X"]).to(device)
            save_Y = torch.from_numpy(data["F"]).to(device)
            if save_X.shape[0] >= budget:
                print("Optimisation already finished")
                return None
    except ValueError:
        print(path)
        return None

    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    problem = bbob_single_botorch(f, dim, 1)

    batch_size = 5
    n_init = 2 * dim
    max_cholesky_size = float("inf")  # Always use Cholesky

    def eval_objective(x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        # return fun(unnormalize(x, fun.bounds))
        return torch.from_numpy(problem.__call__(x.cpu())).to(device)

    @dataclass
    class TurboState:
        dim: int
        batch_size: int
        length: float = 0.8
        length_min: float = 0.5 ** 7
        length_max: float = 1.6
        failure_counter: int = 0
        failure_tolerance: int = float("nan")  # Note: Post-initialized
        success_counter: int = 0
        success_tolerance: int = 10  # Note: The original paper uses 3
        best_value: float = -float("inf")
        restart_triggered: bool = False

        def __post_init__(self):
            self.failure_tolerance = math.ceil(
                max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
            )

    def update_state(state, Y_next):
        if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1

        if state.success_counter == state.success_tolerance:  # Expand trust region
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.failure_counter = 0

        state.best_value = max(state.best_value, max(Y_next).item())
        if state.length < state.length_min:
            state.restart_triggered = True
        return state

        state = TurboState(dim=dim, batch_size=batch_size)
        print(state)

    def get_initial_points(dim, n_pts, seed=0):
        sobol = SobolEngine(dimension=dim, scramble=True)
        X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
        return X_init

    def generate_batch(
            state,
            model,  # GP model
            X,  # Evaluated points on the domain [0, 1]^d
            Y,  # Function values
            batch_size,
            n_candidates=None,  # Number of candidates for Thompson sampling
            num_restarts=10,
            raw_samples=512,
            acqf="ts",
    ):
        assert acqf in ("ts", "ei")
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        if acqf == "ts":
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=batch_size)

        return X_next

    if not os.path.isfile(path):
        X_turbo = get_initial_points(dim, n_init)
        Y_turbo = torch.tensor(
            [eval_objective(x) for x in X_turbo], dtype=dtype, device=device
        ).unsqueeze(-1)

        save_X = X_turbo.clone()
        save_Y = Y_turbo.clone()

    while (budget - save_X.shape[0]) > 0:
        if os.path.isfile(path):
            X_turbo = get_initial_points(dim, n_init)
            Y_turbo = torch.tensor(
                [eval_objective(x) for x in X_turbo], dtype=dtype, device=device
            ).unsqueeze(-1)

            save_X = torch.cat((save_X, X_turbo))
            save_Y = torch.cat((save_Y, Y_turbo))

        state = TurboState(dim, batch_size=batch_size, best_value=max(Y_turbo).item())

        NUM_RESTARTS = 10
        RAW_SAMPLES = 512
        N_CANDIDATES = min(5000, max(2000, 200 * dim))

        torch.manual_seed(0)

        while not state.restart_triggered:  # Run until TuRBO converges
            # Fit a GP model
            train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(
                    nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
                )
            )

            outcome_transform = Standardize(m=1).to(device)
            model = SingleTaskGP(
                X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood, outcome_transform=outcome_transform
            ).to(device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # Do the fitting and acquisition function optimization inside the Cholesky context
            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                # Fit the model
                fit_gpytorch_mll(mll)
                # Create a batch
                X_next = generate_batch(
                    state=state,
                    model=model,
                    X=X_turbo,
                    Y=train_Y,
                    batch_size=batch_size,
                    n_candidates=N_CANDIDATES,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,
                    acqf=acf,
                )

            Y_next = torch.tensor(
                [eval_objective(x) for x in X_next], dtype=dtype, device=device
            ).unsqueeze(-1)

            # Update state
            state = update_state(state=state, Y_next=Y_next)

            # Append data
            X_turbo = torch.cat((X_turbo, X_next), dim=0)
            Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

            # Print current status
            print(
                f"{len(X_turbo)}) Best value: {state.best_value:.4e}, TR length: {state.length:.2e}"
            )

            save_X = torch.cat((save_X, X_next))
            save_Y = torch.cat((save_Y, Y_next))

            best_idx = save_Y.argmax()

            print(f"Best sample: X = {save_X[best_idx]}, Y = {save_Y[best_idx]}")
            print(f"Total samples: {save_X.shape[0]}")
            if save:
                if save_X.shape[0] > budget:
                    np.savez(path, X=save_X[:budget].cpu(), F=save_Y[:budget].cpu())
                else:
                    np.savez(path, X=save_X.cpu(), F=save_Y.cpu())
            if budget - save_X.shape[0] <= 0:
                break

        torch.cuda.empty_cache()


# run Nelder-Mead
def nelder_mead(so_name, problem, instance, budget, save):
    path = f'results/{so_name}/{so_name}_bbob_{problem.fun}_d{problem.dim}_i{instance}_{budget}.npz'
    if os.path.isfile(path):
        print("already finished")
        return None

    def func_to_optimize(x, X, F):
        f = problem.evaluate(x)
        X.append(x)
        F.append(f)

        return f

    options = {
        'disp': True,
        'maxiter': budget,
        'return_all': True,
        'xatol': 1e-6,
        'fatol': 1e-6,
        'adaptive': False
    }
    X = []
    F = []

    while len(F) < budget:
        x0 = np.random.random(problem.n_var)
        result = minimize(func_to_optimize, x0, args=(X, F), method='Nelder-Mead', options=options)

    final_X = np.array(X)
    final_F = np.array(F)
    if save:
        # np.savez(path, X=final_X, F=final_F)
        end_idx = np.min([final_F.shape[0], budget])
        np.savez(path, F=final_F[:end_idx])

    print(problem.fun, problem.dim, final_F.min())


# run Random Search
def random_search(so_name, problem, instance, budget, save):
    print(f"Problem function: {problem.fun}, Dimension: {problem.dim}")

    path = f'results/{so_name}/{so_name}_bbob_{problem.fun}_d{problem.dim}_i{instance}_{budget}.npz'

    if os.path.isfile(path):
        print("already finished")
        return None

    final_X = np.random.random((budget, problem.dim))

    final_F = problem.__call__(final_X)
    if save:
        np.savez(path, F=final_F)


if __name__ == "__main__":
    # run GA, DE, CMA-ES, and NM
    budget = 10000
    for f in range(1, 25):
        for dim in [2, 10, 40]:
            for instance in range(1, 31):
                p = bbob_single_pymmo(f, dim, 1)
                for so_name in ["GA", "DE", "CMAES"]:
                    run_so(so_name, p, instance=instance, budget=budget, save=True)

                so_name = "NM"
                nelder_mead(so_name, p, instance=instance, budget=budget, save=True)

    # run vanilla BO
    budget = 1000
    for f in range(1, 25):
        for dim in [2, 10, 40]:
            for instance in range(1, 31):
                p = bbob_single_botorch(f, dim, 1)  # func dim bbob_instance
                run_bo(problem=p, instance=instance, budget=budget, save=True)

    # run TuRBO
    budget = 10000
    for f in range(1, 25):
        for dim in [2, 10, 40]:
            for instance in range(1, 31):
                print(f, dim, instance, budget)
                run_TuRBO(f=f, dim=dim, instance=instance, budget=budget, save=True)

    # run Random Search
    budget = 10000
    for f in range(1, 25):
        for dim in [2, 10, 40]:
            for instance in range(1, 31):
                so_name = "RS"
                p = bbob_single_botorch(f, dim, 1)  # func dim bbob_instance
                random_search(so_name, p, instance=instance, budget=budget, save=True)
