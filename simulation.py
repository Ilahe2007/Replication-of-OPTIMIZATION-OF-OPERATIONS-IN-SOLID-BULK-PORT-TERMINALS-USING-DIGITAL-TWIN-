"""
A simplified agent-flavoured simulation of the terminal that EVALUATES a given
ship schedule and returns the objective (total ship stay time) plus KPIs.

This plays the role of the paper's AnyLogic simulation: the optimizer (ACS)
proposes a sequence of ships; the simulation validates it against the physical
constraints and reports the resulting stay time. The two pieces together form
the Digital Twin loop (optimize -> simulate -> compare).

Constraints modelled

* Single berth + single ship loader  -> ships are loaded one at a time
  (this is the dominant queueing bottleneck, matching the paper's long queues).
* Cargo readiness: a ship can only start once enough of its product(s) has been
  unloaded from trains into the stockyard (the supply curve). Inventory is
  reserved in schedule order, so resequencing changes who waits for what.
* Reclaimer-limited loading rate; a single reclaim stream feeds the loader.

Scenario flags (cumulative, as in the paper: ID1=+direct, ID2=+PM, ID3=+sim.)

* direct_loading        : a ship may start as soon as a small starting buffer of
                          its product exists, then load while trains keep feeding
                          it ("train on board"), instead of waiting for the full
                          stockpile -> removes stockpiling wait.
* pm_shiploader         : the ship loader is unavailable for `pm_hours` starting
                          at `pm_start_h` (preventive maintenance) -> delays.
* simultaneous_reclaim  : two reclaimers feed the loader at once -> loading rate
                          increases (~1.85x) -> loading time roughly halves.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import bisect

from instance import Instance, DEMURRAGE_USD_PER_DAY, N_PRODUCTS


@dataclass
class ScenarioConfig:
    name: str = "Base"
    direct_loading: bool = False
    pm_shiploader: bool = False
    pm_start_h: float = 9 * 24       # PM window starts day 9
    pm_hours: float = 72.0           # 3 days (paper: "for 3 days")
    simultaneous_reclaim: bool = False
    sim_rate_multiplier: float = 1.85  # two reclaim streams (capped by loader)
    direct_start_fraction: float = 0.15  # buffer needed before direct start


@dataclass
class ShipResult:
    sid: int
    eta_h: float
    start_h: float
    end_h: float

    @property
    def queue_h(self) -> float:
        return self.start_h - self.eta_h

    @property
    def stay_h(self) -> float:
        return self.end_h - self.eta_h


@dataclass
class SimResult:
    total_stay_days: float
    total_queue_days: float
    avg_queue_days: float
    ships_serviced: int
    loaded_cargo_mt: float
    ship_results: list[ShipResult]
    makespan_days: float


def _cumulative_available(curve: list[tuple[float, float]], t: float) -> float:
    """Cumulative tonnage available at time t from a (time, cum) supply curve."""
    if not curve or t < curve[0][0]:
        return 0.0
    times = [e[0] for e in curve]
    idx = bisect.bisect_right(times, t) - 1
    return curve[idx][1]


def _ready_time(curve: list[tuple[float, float]], required: float, reserved: float) -> float:
    """Earliest time at which (cumulative_available - reserved) >= required."""
    for t, cum in curve:
        if cum - reserved >= required:
            return t
    return float("inf")  # supply never sufficient (infeasible)


def simulate(inst: Instance, sequence: list[int], cfg: ScenarioConfig) -> SimResult:
    """Evaluate a ship `sequence` (list of ship ids) under scenario `cfg`."""
    ships_by_id = {s.sid: s for s in inst.ships}
    curves = {p: inst.supply_curve(p) for p in range(N_PRODUCTS)}

    reserved: dict[int, float] = defaultdict(float)
    berth_free = 0.0
    results: list[ShipResult] = []
    loaded_total = 0.0

    base_rate = inst.eff_load_rate_tph
    rate = base_rate * (cfg.sim_rate_multiplier if cfg.simultaneous_reclaim else 1.0)

    pm_start = cfg.pm_start_h
    pm_end = cfg.pm_start_h + cfg.pm_hours

    for sid in sequence:
        ship = ships_by_id[sid]

        # 1. cargo readiness (per product, reserve in schedule order)
        # start_ready : when the ship can BEGIN loading
        # full_ready  : when ALL of the ship's ore has physically arrived
        # (loading can never out-run the trains feeding it)
        start_ready = ship.eta_h
        full_ready = ship.eta_h
        for p, d in zip(ship.products, ship.demands_t):
            full_rt = _ready_time(curves[p], d, reserved[p])
            if cfg.direct_loading:
                # need only a starting buffer; the rest streams in from trains.
                buf_rt = _ready_time(curves[p], d * cfg.direct_start_fraction, reserved[p])
                start_ready = max(start_ready, buf_rt)
            else:
                # must have the full stockpile recovered before loading begins.
                start_ready = max(start_ready, full_rt)
            full_ready = max(full_ready, full_rt)
            reserved[p] += d  # reserve full demand regardless

        # 2. berth + ship-loader availability
        start = max(ship.eta_h, berth_free, start_ready)

        # preventive maintenance: loader cannot operate during the PM window.
        if cfg.pm_shiploader and start < pm_end and (start + 0) < pm_end:
            # if loading would start during (or be blocked by) the outage,
            # push the start to the end of the PM window.
            if start < pm_end and start >= pm_start:
                start = pm_end
            elif start < pm_start:
                # could we finish before PM starts? if not, the loader stops and
                # resumes after PM, so effectively start after PM for simplicity
                dur_preview = ship.total_demand_t / rate
                if start + dur_preview > pm_start:
                    start = pm_end

        duration = ship.total_demand_t / rate
        end = start + duration
        # physical bound: the ship cannot finish loading before all of its ore
        # has actually arrived by train (relevant under direct loading, where it
        # starts on a partial buffer and loads while trains keep feeding it).
        if cfg.direct_loading:
            end = max(end, full_ready)
        berth_free = end
        loaded_total += ship.total_demand_t

        results.append(ShipResult(sid=sid, eta_h=ship.eta_h, start_h=start, end_h=end))

    total_stay_h = sum(r.stay_h for r in results)
    total_queue_h = sum(r.queue_h for r in results)
    n = len(results)
    makespan_h = max((r.end_h for r in results), default=0.0)

    return SimResult(
        total_stay_days=total_stay_h / 24.0,
        total_queue_days=total_queue_h / 24.0,
        avg_queue_days=(total_queue_h / 24.0 / n) if n else 0.0,
        ships_serviced=n,
        loaded_cargo_mt=loaded_total / 1e6,
        ship_results=results,
        makespan_days=makespan_h / 24.0,
    )


def objective(inst: Instance, sequence: list[int], cfg: ScenarioConfig) -> float:
    """Scalar objective minimised by the optimizer: total stay time (days)."""
    return simulate(inst, sequence, cfg).total_stay_days


if __name__ == "__main__":
    from instance import build_instance
    from acs import ACS, ACSParams, greedy_initial

    inst = build_instance()
    scenarios = [
        ScenarioConfig("Base"),
        ScenarioConfig("Direct", direct_loading=True),
        ScenarioConfig("Direct + PM", direct_loading=True, pm_shiploader=True),
        ScenarioConfig("Direct + PM + Sim", direct_loading=True, pm_shiploader=True, simultaneous_reclaim=True),
    ]

    print(f"{'Scenario':16s}  {'Method':8s}  {'stay':>8s}  {'queue':>8s}  {'avg_q':>9s}  {'loaded':>9s}")
    print("-" * 72)
    for cfg in scenarios:
        # greedy FIFO baseline
        greedy_seq = greedy_initial(inst, cfg)
        rg = simulate(inst, greedy_seq, cfg)
        print(f"{cfg.name:16s}  {'greedy':8s}  {rg.total_stay_days:7.1f} d  "
              f"{rg.total_queue_days:7.1f} d  "
              f"{rg.avg_queue_days:4.1f} d/ship  "
              f"{rg.loaded_cargo_mt:.2f} Mt")

        # ACS optimised sequence
        acs = ACS(inst, cfg, ACSParams())
        best_seq, best_obj, _ = acs.run()
        ra = simulate(inst, best_seq, cfg)
        imp = 100.0 * (rg.total_stay_days - ra.total_stay_days) / rg.total_stay_days
        print(f"{'':16s}  {'ACS':8s}  {ra.total_stay_days:7.1f} d  "
              f"{ra.total_queue_days:7.1f} d  "
              f"{ra.avg_queue_days:4.1f} d/ship  "
              f"{ra.loaded_cargo_mt:.2f} Mt  ({imp:+.1f}%)")
        print()