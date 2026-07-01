"""
Constructive greedy heuristic and Ant Colony System (ACS) metaheuristic for the
ship-sequencing problem, following the formulation in the paper.

The decision variable optimised here is the SEQUENCE in which ships are served
at the berth (the paper's "by-pass" of the FIFO order). Each ant builds a
complete sequence; the simulation  evaluates it and returns the total ship stay time, which is the quantity ACS minimises.

ACS components, matching the paper's equations:
  pseudo-random-proportional transition rule (parameter q0)
  transition probability:     p_ij * tau_ij^alpha * eta_ij^beta
  local pheromone update:     tau_ij = (1-phi)*tau_ij + phi*tau0 (after each step)
  global pheromone update:    tau_ij = (1-rho)*tau_ij + rho/L_best   (best-so-far edges)
  heuristic attractiveness:   eta_ij = 1 / (expected completion time of task j)
"""

from __future__ import annotations
from dataclasses import dataclass
import random
import time

from instance import Instance
from simulation import ScenarioConfig, objective, simulate


# Greedy constructive heuristic  -> the paper's "initial solution"
def greedy_initial(inst: Instance, cfg: ScenarioConfig) -> list[int]:
    """Order ships by ETA which is Estimated Time of Arrival. It usually follows First-in-First-Out(FIFO) This function creates initial ids for each ship from instance"""
    return [s.sid for s in sorted(inst.ships, key=lambda s: s.eta_h)]


# Ant Colony System
@dataclass
class ACSParams:
    n_ants: int = 20
    iterations: int = 120
    alpha: float = 1.0      # pheromone exponent
    beta: float = 2.0       # heuristic exponent
    rho: float = 0.10       # global evaporation
    phi: float = 0.10       # local evaporation
    q0: float = 0.90        # exploitation probability
    seed: int = 1           #starting point of a random number generator


class ACS:
    def __init__(self, inst: Instance, cfg: ScenarioConfig, params: ACSParams):
        self.inst = inst
        self.cfg = cfg
        self.p = params
        self.rng = random.Random(params.seed)
        self.n = len(inst.ships)
        self.ships = sorted(inst.ships, key=lambda s: s.eta_h)
        self.id_order = [s.sid for s in self.ships]
        self.START = self.n  # virtual start node index

        base_rate = inst.eff_load_rate_tph * (cfg.sim_rate_multiplier if cfg.simultaneous_reclaim else 1.0)
        # static loading time per ship (hours) -> used by the heuristic only
        self.load_time = {s.sid: s.total_demand_t / base_rate for s in inst.ships}
        self.eta = {s.sid: s.eta_h for s in inst.ships}

        # initial pheromone tau0 = 1 / (n · L_greedy)
        greedy = greedy_initial(inst, cfg)
        self.L_greedy = objective(inst, greedy, cfg)
        self.tau0 = 1.0 / (self.n * self.L_greedy) if self.L_greedy > 0 else 1.0
        self.tau = [[self.tau0] * self.n for _ in range(self.n + 1)]
        self._start_row = [self.tau0] * self.n  # pheromone from virtual start

        self.best_seq = greedy[:]
        self.best_obj = self.L_greedy

    # heuristic attractiveness eta_ij
    def _eta_attr(self, current_finish: float, j: int) -> float:
        """Inverse of the expected finish time if ship j is served next. Basically if it's low cost and high eta the ants are attacted, but if it high cost and low eta then ands avoids that ship."""
        start = max(self.eta[j], current_finish)
        finish_contrib = (start - self.eta[j]) + self.load_time[j]  # sum of wait and load time
        return 1.0 / (finish_contrib + 1e-6)  #1e-6 added to break the error if the divisor is zero

    # build one ant's sequence 
    def _construct(self) -> list[int]:
        unscheduled = set(self.id_order)
        seq: list[int] = []
        i = self.START
        current_finish = 0.0
        while unscheduled:
            # candidate scores tau^alpha · eta^beta
            cand = list(unscheduled)
            scores = []
            for j in cand:
                eta_attr = self._eta_attr(current_finish, j)
                scores.append((self.tau_get(i, j) ** self.p.alpha) * (eta_attr ** self.p.beta))
            total = sum(scores)
            if total <= 0:
                choice = self.rng.choice(cand)
            elif self.rng.random() < self.p.q0:
                # exploitation: pick best
                choice = cand[max(range(len(cand)), key=lambda k: scores[k])]
            else:
                # biased exploration: roulette
                r = self.rng.random() * total
                acc = 0.0
                choice = cand[-1]
                for k, sc in enumerate(scores):
                    acc += sc
                    if acc >= r:
                        choice = cand[k]
                        break
            seq.append(choice)
            unscheduled.discard(choice)
            # local pheromone update on the traversed edge
            self.tau_local_update(i, choice)
            current_finish = max(self.eta[choice], current_finish) + self.load_time[choice]
            i = choice
        return seq

    # pheromone accessors handling the virtual-start row
    def tau_get(self, i: int, j: int) -> float:
        if i == self.START:
            return self._start_row[j]
        return self.tau[i][j]

    def tau_local_update(self, i: int, j: int) -> None:
        if i == self.START:
            self._start_row[j] = (1 - self.p.phi) * self._start_row[j] + self.p.phi * self.tau0
        else:
            self.tau[i][j] = (1 - self.p.phi) * self.tau[i][j] + self.p.phi * self.tau0

    def tau_global_update(self, seq: list[int], L_best: float) -> None:
        deposit = 1.0 / L_best
        prev = self.START
        for j in seq:
            if prev == self.START:
                self._start_row[j] = (1 - self.p.rho) * self._start_row[j] + self.p.rho * deposit
            else:
                self.tau[prev][j] = (1 - self.p.rho) * self.tau[prev][j] + self.p.rho * deposit
            prev = j

    # main loop
    def run(self) -> tuple[list[int], float, list[float]]:
        history = [self.best_obj]
        for _ in range(self.p.iterations):
            iter_best_seq, iter_best_obj = None, float("inf")
            for _ in range(self.p.n_ants):
                seq = self._construct()
                obj = objective(self.inst, seq, self.cfg)
                if obj < iter_best_obj:
                    iter_best_seq, iter_best_obj = seq, obj
            if iter_best_obj < self.best_obj:
                self.best_obj, self.best_seq = iter_best_obj, iter_best_seq[:]
            # global update uses the best-so-far solution
            self.tau_global_update(self.best_seq, self.best_obj)
            history.append(self.best_obj)
        return self.best_seq, self.best_obj, history


def optimize(inst: Instance, cfg: ScenarioConfig, params: ACSParams | None = None) -> dict:
    #Run greedy + ACS for one scenario and return a results dict
    params = params or ACSParams()
    t0 = time.perf_counter()
    greedy = greedy_initial(inst, cfg)
    init_obj = objective(inst, greedy, cfg)
    acs = ACS(inst, cfg, params)
    best_seq, best_obj, history = acs.run()
    elapsed = time.perf_counter() - t0
    sim_best = simulate(inst, best_seq, cfg)
    return {
        "scenario": cfg.name,
        "initial_seq": greedy,
        "initial_obj": init_obj,
        "best_seq": best_seq,
        "best_obj": best_obj,
        "history": history,
        "elapsed_s": elapsed,
        "sim_best": sim_best,
    }


if __name__ == "__main__":
    from instance import build_instance
    inst = build_instance(eta_window_days=5)
    cfg = ScenarioConfig("Base")
    res = optimize(inst, cfg, ACSParams(iterations=60))
    print(f"initial = {res['initial_obj']:.1f} d   best = {res['best_obj']:.1f} d"
          f"   ({100*(res['initial_obj']-res['best_obj'])/res['initial_obj']:.1f}% better)"
          f"   in {res['elapsed_s']:.1f}s")