from pymoo.core.algorithm import Algorithm
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.callback import Callback
from pymoo.util.optimum import filter_optimum
import numpy as np

class RandomSearch(Algorithm):
    def __init__(self, pop_size=100, sampling=FloatRandomSampling(), **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.sampling = sampling
        self._best = None

    def _initialize_infill(self):
        return self.sampling(self.problem, self.pop_size)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills

    def _infill(self):
        return self.sampling(self.problem, self.pop_size)

    def _advance(self, infills=None, **kwargs):
        self.pop = infills

    def _set_optimum(self):
        current_best = filter_optimum(self.pop, least_infeasible=self.return_least_infeasible)

        if current_best is None:
            return

        if self._best is None or current_best.get("F")[0] < self._best.get("F")[0]:
            self._best = current_best  # Update absolute best found so far.

        self.opt = self._best


class TrackBest(Callback):
    """
    For debugging RandomSearch - track results and compare to final.
    """
    def __init__(self):
        super().__init__()
        self.best_F = None
        self.best_X = None
        self.best_gen = None
        self.best_n_evals = None
        self.all_X = []  # all X values across all generations
        self.all_F = []  # all F values across all generations

    def notify(self, algorithm):
        pop = algorithm.pop
        F = pop.get("F")
        G = pop.get("G")

        self.all_X.append(pop.get("X").copy())
        self.all_F.append(F.copy())

        if G is not None:
            cv = (G > 0).any(axis=1)
            feasible = pop[~cv]
            if len(feasible) == 0:
                return
            F = feasible.get("F")
            candidates = feasible
        else:
            candidates = pop

        best_idx = F.argmin()
        f = F[best_idx]
        if self.best_F is None or f < self.best_F:
            self.best_F = f
            self.best_X = candidates.get("X")[best_idx]
            self.best_gen = algorithm.n_gen
            self.best_n_evals = algorithm.evaluator.n_eval

    def print_summary(self, result):
        print(f"=== Callback best (tracked across all generations) ===")
        print(f"  Best X:      {self.best_X}")
        print(f"  Best F:      {self.best_F}")
        print(f"  Found gen:   {self.best_gen}")
        print(f"  Found evals: {self.best_n_evals}")
        print(f"  Total gens:  {len(self.all_X)}")
        print(f"  Total evals: {sum(len(x) for x in self.all_X)}")
        print(f"\n=== pymoo result (self.opt from final generation) ===")
        print(f"  result.X:    {result.X}")
        print(f"  result.F:    {result.F}")
        print(f"\n=== Match ===")
        print(f"  X match: {np.allclose(self.best_X, result.X)}")
        print(f"  F match: {np.allclose(self.best_F, result.F)}")
