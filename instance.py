"""
Generates a synthetic-but-representative solid-bulk export terminal instance,
mirroring the scale of the Base Scenario is 19 ships, 231 trains, 15 products, 3 stockyards, 17 initial stacks

Rates are taken from the paper's validation table

All quantities are in tonnes (t) and hours (h). Time origin = 0 h.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import random

# Physical constants taken / derived from the paper
HORIZON_DAYS = 30
HORIZON_H = HORIZON_DAYS * 24

N_PRODUCTS = 15
N_SHIPS = 19
N_TRAINS = 231
N_YARDS = 3
N_INIT_STACKS = 17

TRAIN_CAPACITY_T = 13_974          # Table 2: "Train Capacity (t) 13,974"
EFF_LOAD_RATE_TPH = 6_400.0        # validation effective loading rate ~6,425.89
EFF_UNLOAD_RATE_TPH = 4_855.0      # validation effective unloading rate ~4,855.15
N_CAR_DUMPERS = 3                  # CD-1, CD-2, CD-3
N_RECLAIMERS = 3                   # SR-1, SR-2, R-4 (reclaim-capable units)
TOTAL_YARD_CAPACITY_T = 913_500    # Table 2

DEMURRAGE_USD_PER_DAY = 25_000.0   # paper saya that daily demurrage cost of 25,000$


@dataclass
class Ship:
    sid: int
    eta_h: float                       # arrival time at the port (hours)
    products: list[int]                # 1 or 2 product ids
    demands_t: list[float]             # tonnage required per product

    @property
    def total_demand_t(self) -> float:
        return sum(self.demands_t)


@dataclass
class Train:
    tid: int
    arrival_h: float
    product: int
    tonnage_t: float = TRAIN_CAPACITY_T


@dataclass
class Instance:
    ships: list[Ship]
    trains: list[Train]
    initial_inventory_t: dict[int, float]   # product -> tonnes available at t=0
    horizon_h: float = HORIZON_H
    eff_load_rate_tph: float = EFF_LOAD_RATE_TPH
    eff_unload_rate_tph: float = EFF_UNLOAD_RATE_TPH
    n_car_dumpers: int = N_CAR_DUMPERS
    n_reclaimer: int = N_RECLAIMERS

    # supply curve helper
    def supply_curve(self, product: int) -> list[tuple[float, float]]:
        """Cumulative available tonnage of `product` over time.

        Returns a sorted list of (time_h, cumulative_tonnes_available). A train
        contributes its tonnage once it has been unloaded (arrival + unload time).
        Initial inventory is available at t=0.
        """
        unload_time = TRAIN_CAPACITY_T / self.eff_unload_rate_tph  # ~2.88 h
        events: list[tuple[float, float]] = [(0.0, self.initial_inventory_t.get(product, 0.0))]
        for tr in self.trains:
            if tr.product == product:
                events.append((tr.arrival_h + unload_time, tr.tonnage_t))
        events.sort(key=lambda e: e[0])
        curve, cum = [], 0.0
        for t, amt in events:
            cum += amt
            curve.append((t, cum))
        return curve


def build_instance(seed: int = 42, eta_window_days: float = 7.0) -> Instance:
    """Construct a reproducible instance matching the Base Scenario scale.

    `eta_window_days` controls how tightly ship arrivals are clustered. Tighter
    clustering -> more congestion at the single berth -> longer queues (the
    paper reports ~8 days average queue per ship in the Base Scenario).
    """
    rng = random.Random(seed)

    # 1. Ships
    # There's two ship types, small and large.
    ships: list[Ship] = []
    dual_idx = set(rng.sample(range(N_SHIPS), 5))
    for sid in range(N_SHIPS):
        # cluster arrivals so a real queue forms at the single berth.
        eta = round(rng.uniform(0, eta_window_days * 24), 1)
        if sid in dual_idx:
            prods = rng.sample(range(N_PRODUCTS), 2)
            d1 = rng.uniform(70_000, 110_000)
            d2 = rng.uniform(60_000, 100_000)
            demands = [round(d1, -2), round(d2, -2)]
        else:
            prods = [rng.randrange(N_PRODUCTS)]
            # mix of small and large single-product cargoes
            base = rng.uniform(90_000, 220_000)
            demands = [round(base, -2)]
        ships.append(Ship(sid=sid, eta_h=eta, products=prods, demands_t=demands))
    ships.sort(key=lambda s: s.eta_h)
    for i, s in enumerate(ships):
        s.sid = i  # renumber by ETA

    # 2. Per-product demand, then trains to supply it
    demand_per_product: dict[int, float] = {p: 0.0 for p in range(N_PRODUCTS)}
    for s in ships:
        for p, d in zip(s.products, s.demands_t):
            demand_per_product[p] += d

    # For a subset of "late" products, bias train arrivals to later in the
    # horizon so that the stockpile genuinely binds in the base case (this is
    # exactly the situation that direct loading is meant to relieve). The bulk
    # of supply is front-loaded so that, once loading throughput is high cargo availability is no longer the bottleneck.
    late_products = set(rng.sample(range(N_PRODUCTS), 4))

    trains: list[Train] = []
    tid = 0
    for p in range(N_PRODUCTS):
        need = demand_per_product[p]
        if need <= 0:
            # still ship a couple of trains so the product exists
            need = TRAIN_CAPACITY_T
        # supply ~1.15x demand to keep the instance feasible with slack
        n_trains_p = max(1, int((need * 1.15) // TRAIN_CAPACITY_T) + 1)
        for _ in range(n_trains_p):
            if p in late_products:
                arr = rng.uniform(2 * 24, 11 * 24)       # arrive a bit later
            else:
                arr = rng.uniform(0, 8 * 24)             # mostly front-loaded
            trains.append(Train(tid=tid, arrival_h=round(arr, 1), product=p))
            tid += 1

    # Trim / pad to land near the reported 231 trains (cosmetic; supply already
    # feasible). We keep whatever we generated but report the count.
    trains.sort(key=lambda t: t.arrival_h)

    # 3. Initial inventory
    # Head-start: ~40% of each product's demand pre-stacked at t=0.
    initial_inventory = {
        p: round(0.40 * demand_per_product[p], -2) for p in range(N_PRODUCTS)
    }

    inst = Instance(
        ships=ships,
        trains=trains,
        initial_inventory_t=initial_inventory,
    )
    return inst


if __name__ == "__main__":
    inst = build_instance()
    print(f"Ships : {len(inst.ships)}")
    print(f"Trains: {len(inst.trains)}")
    tot_dem = sum(s.total_demand_t for s in inst.ships)
    tot_sup = sum(inst.initial_inventory_t.values()) + sum(t.tonnage_t for t in inst.trains)
    print(f"Total ship demand : {tot_dem/1e6:5.2f} Mt")
    print(f"Total supply       : {tot_sup/1e6:5.2f} Mt")
    print(f"Dual-product ships : {sum(1 for s in inst.ships if len(s.products) == 2)}")
    print("ETAs (days):", [round(s.eta_h/24, 1) for s in inst.ships])