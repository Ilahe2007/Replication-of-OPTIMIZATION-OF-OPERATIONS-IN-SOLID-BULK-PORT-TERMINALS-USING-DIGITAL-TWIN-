"""
Runs the full replication study: the four cumulative scenarios from the paper  and produces a consolidated results table + figures.

Scenarios (cumulative, exactly as in the paper):
    ID 0  Base
    ID 1  + direct loading
    ID 2  + preventive maintenance of the ship loader (3 days)
    ID 3  + simultaneous reclaim route (yard -> ship via two reclaimers)
"""

from __future__ import annotations
import json
import os

import matplotlib
import matplotlib.pyplot as plt

from instance import build_instance, DEMURRAGE_USD_PER_DAY
from simulation import ScenarioConfig, objective
from acs import optimize, ACSParams, greedy_initial

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def scenario_configs() -> list[ScenarioConfig]:
    """Return the four cumulative scenarios in order (matching paper's Table 4)."""
    return [
        ScenarioConfig(name="0: Base"),
        ScenarioConfig(name="1: +Direct loading", direct_loading=True),
        ScenarioConfig(name="2: +Preventive maint.", direct_loading=True, pm_shiploader=True),
        ScenarioConfig(name="3: +Simultaneous reclaim", direct_loading=True, pm_shiploader=True, simultaneous_reclaim=True),
    ]


def calibrate_instance(seed: int = 42, target_greedy: float = 199.6, tolerance: float = 5.0) -> tuple[float, float]:
    """
    Find eta_window_days that reproduces the paper's baseline greedy result.
    
    Returns:
        (eta_window_days, greedy_obj)  # the calibrated value and resulting greedy objective
    """
    print("CALIBRATING INSTANCE TO MATCH PAPER'S BASELINE")
    print(f"Target greedy total stay: {target_greedy:.1f} h")
    print(f"Tolerance: {tolerance:.1f} h")
    
    best_window = 7.0  # default
    best_error = float("inf")
    
    for window in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        inst = build_instance(seed=seed, eta_window_days=window)
        greedy = greedy_initial(inst, ScenarioConfig("Base"))
        obj = objective(inst, greedy, ScenarioConfig("Base"))
        error = abs(obj - target_greedy)

        print(f"  eta_window_days={window:4.1f}  ->  greedy={obj:6.1f}h  error={error:5.1f}h ")
        
        if error < best_error:
            best_error = error
            best_window = window
        
        if error < tolerance:
            print(f"\nCalibration found: eta_window_days = {window}")
            print(f"   Greedy result: {obj:.1f}h (target: {target_greedy:.1f}h, error: {error:.1f}h)")
            return window, obj
    
    print(f"\n No value within tolerance. Closest: eta_window_days={best_window} (error={best_error:.1f}h)")
    return best_window, None


def run_all(seed: int = 42, eta_window_days: float = None, params: ACSParams | None = None) -> tuple[list[dict], float]:
    """
    Run all four scenarios with automatic calibration.
    
    If eta_window_days is None, auto-calibrate to match paper's baseline.
    """
    # Auto-calibrate if not specified
    if eta_window_days is None:
        calibrated_window, _ = calibrate_instance(seed=seed)
        eta_window_days = calibrated_window
        print(f"\nUsing calibrated eta_window_days = {eta_window_days}")
    
    inst = build_instance(seed=seed, eta_window_days=eta_window_days)
    params = params or ACSParams(iterations=120, n_ants=20)
    
    # Verify calibration
    greedy_check = greedy_initial(inst, ScenarioConfig("Base"))
    greedy_obj = objective(inst, greedy_check, ScenarioConfig("Base"))
    print(f"\nCalibration check: greedy = {greedy_obj:.1f}h (paper: 199.6h, diff: {greedy_obj - 199.6:.1f}h)")
    
    results = []
    for cfg in scenario_configs():
        res = optimize(inst, cfg, params)
        sb = res["sim_best"]
        demurrage = (res["initial_obj"] - res["best_obj"]) * DEMURRAGE_USD_PER_DAY
        results.append({
            "scenario": cfg.name,
            "initial_days": res["initial_obj"],
            "best_days": res["best_obj"],
            "improvement_pct": 100 * (res["initial_obj"] - res["best_obj"]) / res["initial_obj"],
            "demurrage_reduction_usd": demurrage,
            "total_queue_days": sb.total_queue_days,
            "ships_serviced": sb.ships_serviced,
            "avg_queue_days_per_ship": sb.avg_queue_days,
            "loaded_cargo_mt": sb.loaded_cargo_mt,
            "elapsed_s": res["elapsed_s"],
            "history": res["history"],
            "best_seq": res["best_seq"],
            "eta_window_days": eta_window_days,
            "greedy_baseline": greedy_obj,
            
        })
    return results, inst


def print_table(results: list[dict]) -> str:
    """Print a formatted results table."""
    cols = ["scenario", "initial_days", "best_days", "improvement_pct", "demurrage_reduction_usd", 
            "total_queue_days", "avg_queue_days_per_ship", "ships_serviced", "loaded_cargo_mt", "elapsed_s"]
    headers = {
        "scenario": "Scenario", "initial_days": "Initial(d)",
        "best_days": "Best(d)", "improvement_pct": "Impr%",
        "demurrage_reduction_usd": "Demurr.Red(USD)",
        "total_queue_days": "Queue(d)",
        "avg_queue_days_per_ship": "AvgQ(d/ship)",
        "ships_serviced": "Ships", "loaded_cargo_mt": "Loaded(Mt)",
        "elapsed_s": "Time(s)",
    }
    lines = []
    
    # Header
    head = (f"{headers['scenario']:<26}" + "".join(f"{headers[c]:>16}" for c in cols[1:]))
    lines.append(head)
    lines.append("-" * len(head))
    
    # Rows
    for r in results:
        row = f"{r['scenario']:<26}"
        for c in cols[1:]:
            v = r[c]
            if c == "demurrage_reduction_usd":
                row += f"{v:>16,.0f}"
            elif isinstance(v, float):
                row += f"{v:>16.2f}"
            else:
                row += f"{v:>16}"
        lines.append(row)
    
    # Calibration info footer
    lines.append("")
    lines.append(f"Calibration: eta_window_days = {results[0]['eta_window_days']:.1f}")
    lines.append(f"Greedy baseline: {results[0]['greedy_baseline']:.1f}h (paper: 199.6h)")
    
    out = "\n".join(lines)
    print(out)
    return out


def make_plots(results: list[dict]) -> None:
    """Generate bar chart and convergence plot."""
    names = [r["scenario"] for r in results]
    init = [r["initial_days"] for r in results]
    best = [r["best_days"] for r in results]

    # Bar chart: initial vs best per scenario
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(names))
    w = 0.38
    ax.bar([i - w/2 for i in x], init, w, label="Initial (greedy)", color="#9bb8d3")
    ax.bar([i + w/2 for i in x], best, w, label="Best (ACS)", color="#2c5f8a")
    ax.set_ylabel("Total ship stay time (ship-days)")
    ax.set_title("Replication: total stay time by scenario (initial vs ACS-optimized)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.split(": ")[-1] for n in names], rotation=15, ha="right")
    ax.legend()
    for i, (a, b) in enumerate(zip(init, best)):
        ax.text(i - w/2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(i + w/2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "scenarios_bar.png"), dpi=130)
    plt.show()
    plt.close(fig)

    # Convergence curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        ax.plot(r["history"], label=r["scenario"].split(": ")[-1], linewidth=1.8)
    ax.set_xlabel("ACS iteration")
    ax.set_ylabel("Best-so-far total stay time (ship-days)")
    ax.set_title("ACS convergence per scenario")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "convergence.png"), dpi=130)
    plt.close(fig)


def gantt_plot(inst, results: list[dict]) -> None:
    """Berth occupancy timeline for the Base scenario best solution."""
    from simulation import simulate
    base = results[0]
    cfg = scenario_configs()[0]
    sim = simulate(inst, base["best_seq"], cfg)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, sr in enumerate(sim.ship_results):
        eta_d, st_d, en_d = sr.eta_h/24, sr.start_h/24, sr.end_h/24
        ax.barh(k, en_d - st_d, left=st_d, color="#2c5f8a", height=0.6)
        ax.barh(k, st_d - eta_d, left=eta_d, color="#d98a3a", height=0.6, alpha=0.6)
        ax.plot(eta_d, k, "k|", markersize=8)
    ax.set_yticks(range(len(sim.ship_results)))
    ax.set_yticklabels([f"S{sr.sid}" for sr in sim.ship_results], fontsize=7)
    ax.set_xlabel("Time (days)")
    ax.set_title("Base scenario - berth timeline (orange = queue, blue = loading)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "gantt_base.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    print("REPLICATION STUDY: Solid Bulk Port Terminal Optimization")
    # Run with auto-calibration
    results, inst = run_all()
    
    # Print table
    print("RESULTS TABLE")
    table = print_table(results)
    
    # Generate plots
    print("\nGenerating plots...")
    make_plots(results)
    gantt_plot(inst, results)
    print("Plots saved to:", RESULTS_DIR)
    
    # Save machine-readable results
    slim = [{k: v for k, v in r.items() if k not in ("history", "best_seq")} for r in results]
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(slim, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "results_table.txt"), "w") as f:
        f.write(table + "\n")
