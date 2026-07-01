"""
Two supporting studies, echoing the paper's statement that "various tests were
conducted ... to identify the best input parameters for the metaheuristic
(number of ants, attractiveness exponents, etc.)":
    1. ACS parameter sensitivity on the Base scenario  (beta, n_ants, q0)
    2. Multi-seed robustness of the scenario ordering   (is the pattern stable?)
"""

from __future__ import annotations
import os

from instance import build_instance
from simulation import ScenarioConfig
from acs import optimize, ACSParams

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def parameter_sensitivity():
    inst = build_instance(eta_window_days=5)
    cfg = ScenarioConfig("Base")
    print("(A) ACS parameter sensitivity (Base scenario)")
    print(f"{'param':<14}{'value':>8}{'best(d)':>12}{'impr%':>10}")

    rows = []
    for beta in [0.5, 1.0, 2.0, 4.0]:
        r = optimize(inst, cfg, ACSParams(beta=beta, iterations=80))
        impr = 100*(r['initial_obj']-r['best_obj'])/r['initial_obj']
        print(f"{'beta':<14}{beta:>8}{r['best_obj']:>12.2f}{impr:>10.2f}")
        rows.append(("beta", beta, r['best_obj']))

    for ants in [5, 10, 20, 40]:
        r = optimize(inst, cfg, ACSParams(n_ants=ants, iterations=80))
        impr = 100*(r['initial_obj']-r['best_obj'])/r['initial_obj']
        print(f"{'n_ants':<14}{ants:>8}{r['best_obj']:>12.2f}{impr:>10.2f}")
        rows.append(("n_ants", ants, r['best_obj']))

    for q0 in [0.0, 0.5, 0.9, 0.99]:
        r = optimize(inst, cfg, ACSParams(q0=q0, iterations=80))
        impr = 100*(r['initial_obj']-r['best_obj'])/r['initial_obj']
        print(f"{'q0':<14}{q0:>8}{r['best_obj']:>12.2f}{impr:>10.2f}")
        rows.append(("q0", q0, r['best_obj']))

    return rows


def multiseed_robustness():
    print("\n=== (B) Multi-seed robustness of scenario ordering ===")
    configs = [
        ("Base", ScenarioConfig("Base")),
        ("Direct", ScenarioConfig("Direct", direct_loading=True)),
        ("Direct+PM", ScenarioConfig("Direct+PM", direct_loading=True, pm_shiploader=True)),
        ("Direct+PM+Sim", ScenarioConfig("Direct+PM+Sim", direct_loading=True, pm_shiploader=True, simultaneous_reclaim=True)),
    ]
    print(f"{'seed':<6}" + "".join(f"{n:>16}" for n, _ in configs) + f"{'order ok?':>12}")
    all_ok = True
    for seed in [1, 7, 13, 42, 99]:
        inst = build_instance(seed=seed, eta_window_days=5)
        bests = []
        for _, cfg in configs:
            r = optimize(inst, cfg, ACSParams(iterations=60))
            bests.append(r['best_obj'])
        # expected: Sim < Direct < Base < PM
        ok = (bests[3] < bests[1] < bests[0]) and (bests[2] > bests[1])
        all_ok &= ok
        print(f"{seed:<6}" + "".join(f"{b:>16.1f}" for b in bests) +
              f"{('yes' if ok else 'NO'):>12}")
    print(f"Ordering (Sim<Direct<Base, PM>Direct) holds on all seeds: {all_ok}")

if __name__ == "__main__":
    rows = parameter_sensitivity()
    multiseed_robustness()