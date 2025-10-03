import pandas as pd
import math
import random
from typing import List, Tuple, Dict

# ==============================
# Config
# ==============================
INSTANCE_FILE = "newspaper problem instance.xlsx"   # zelfde als jouw script
DEPOT_INDEX = 0                                     # depot is rij 0
NUM_DRIVERS = 4
STOPS_PER_DRIVER = 30
SEED = 42
random.seed(SEED)

# ==============================
# Data inlezen + afstandsmatrix
# ==============================
df = pd.read_excel(INSTANCE_FILE)

# lijst met coördinaten maken (zelfde naam als bij jou)
coordinates: List[Tuple[float, float]] = [
    (df.iloc[i]["xcoord"], df.iloc[i]["ycoord"]) for i in range(len(df))
]
NUM_LOCATIONS = len(coordinates)

def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx, dy = a[0]-b[0], a[1]-b[1]
    return math.hypot(dx, dy)

# afstandsmatrix bouwen (zelfde idee als jouw script, maar iets sneller in lijstvorm)
# distance_matrix[i][j] i->j
distance_matrix: List[List[float]] = [
    [euclid(coordinates[i], coordinates[j]) for j in range(NUM_LOCATIONS)]
    for i in range(NUM_LOCATIONS)
]

# ==============================
# Helpers
# ==============================
def tour_length_with_depot(tour: List[int]) -> float:
    """Lengte van tour inclusief start/finish bij depot."""
    if not tour:
        return 0.0
    full = [DEPOT_INDEX] + tour + [DEPOT_INDEX]
    return sum(distance_matrix[full[i]][full[i+1]] for i in range(len(full)-1))

def nearest_unassigned(current: int, unassigned: set) -> int:
    """Pak dichtstbijzijnde onverdeelde stop vanaf current."""
    return min(unassigned, key=lambda j: distance_matrix[current][j])

# ==============================
# Greedy assignment (jouw vibe)
# ==============================
def build_greedy_tours(num_drivers: int, stops_per_driver: int) -> List[List[int]]:
    """
    Verdeel alle stops greedy: per driver steeds de dichtstbijzijnde unassigned,
    startend vanaf het depot.
    """
    # alle niet-depot stops
    unassigned = set(range(NUM_LOCATIONS)) - {DEPOT_INDEX}

    tours: List[List[int]] = [[] for _ in range(num_drivers)]
    current: List[int] = [DEPOT_INDEX] * num_drivers

    # zolang er nog wat te verdelen is, loop drivers rond
    d = 0
    while unassigned:
        # sla driver over als ze al vol zitten
        if len(tours[d]) < stops_per_driver:
            j = nearest_unassigned(current[d], unassigned)
            tours[d].append(j)
            unassigned.remove(j)
            current[d] = j
        # volgende driver
        d = (d + 1) % num_drivers

        # safety: als iedereen vol zit maar er toch nog unassigned over zijn (te veel stops),
        # gooi ze dan bij de kortste tour (solid, low-effort fix).
        if all(len(t) >= stops_per_driver for t in tours) and unassigned:
            # kies de driver met kortste huidige tour lengte
            lengths = [tour_length_with_depot(t) for t in tours]
            d = min(range(num_drivers), key=lambda k: lengths[k])
            j = nearest_unassigned(current[d], unassigned)
            tours[d].append(j)
            unassigned.remove(j)
            current[d] = j

    return tours

# ==============================
# 2-opt (metaheuristic glow-up)
# ==============================
def two_opt_once(path: List[int]) -> Tuple[List[int], float, bool]:
    """
    Eén beste 2-opt verbetering op path (zonder depot in de lijst).
    Return: (nieuwe_path, delta, improved?)
    """
    n = len(path)
    if n < 4:
        return path, 0.0, False

    best_delta = 0.0
    best_i, best_k = -1, -1

    # werk met full tour incl. depot edges voor delta
    def segment_len(a, b):  # lengte edge a->b in full (met depot waar nodig)
        return distance_matrix[a][b]

    # indices over de 'full tour' [DEPOT] + path + [DEPOT]
    full = [DEPOT_INDEX] + path + [DEPOT_INDEX]

    # 2-opt swap breekt (i,i+1) en (k,k+1), 0<=i<k< n (in path-index)
    for i in range(n - 2):
        a, b = full[i], full[i + 1]               # edge a->b
        for k in range(i + 2, n):
            c, d = full[k], full[k + 1]           # edge c->d
            # skip triviale aansluitende edges die de tour openbreken
            if b == c or d == a:
                continue
            old = segment_len(a, b) + segment_len(c, d)
            new = segment_len(a, c) + segment_len(b, d)
            delta = old - new
            if delta > best_delta + 1e-12:
                best_delta = delta
                best_i, best_k = i, k

    if best_delta > 1e-12:
        # reverse segment path[i: k] (in path-index)
        new_path = path[:best_i] + path[best_i:best_k][::-1] + path[best_k:]
        return new_path, best_delta, True

    return path, 0.0, False

def two_opt_improve(path: List[int], max_iters: int = 10_000) -> Tuple[List[int], float]:
    """Itereer 2-opt tot geen verbetering. Return (best_path, gained_length)."""
    total_gain = 0.0
    iters = 0
    while iters < max_iters:
        path, gain, improved = two_opt_once(path)
        if not improved:
            break
        total_gain += gain
        iters += 1
    return path, total_gain

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    # 1) bouw greedy tours (zelfde outputstructuur/namen als jouw code)
    tours: List[List[int]] = build_greedy_tours(NUM_DRIVERS, STOPS_PER_DRIVER)

    # 2) print greedy resultaat
    print("=== Greedy resultaat ===")
    for d, tour in enumerate(tours, start=1):
        print(f"Tour {d}:", tour)

    tour_lengths_before = [tour_length_with_depot(t) for t in tours]
    for d, L in enumerate(tour_lengths_before, start=1):
        print(f"Tour {d} lengte (voor 2-opt): {L:.4f}")

    # 3) 2-opt per driver (metaheuristic sauce)
    improved_tours = []
    gains = []
    for t in tours:
        t2, gain = two_opt_improve(t)
        improved_tours.append(t2)
        gains.append(gain)

    print("\n=== Na 2-opt per tour ===")
    for d, (old, new, gain) in enumerate(zip(tours, improved_tours, gains), start=1):
        print(f"Tour {d} (gain {gain:.4f}):")
        print("  voor:", old)
        print("  na  :", new)

    tour_lengths_after = [tour_length_with_depot(t) for t in improved_tours]
    for d, (L0, L1) in enumerate(zip(tour_lengths_before, tour_lengths_after), start=1):
        print(f"Tour {d} lengte: {L0:.4f} -> {L1:.4f}  (Δ = {L0 - L1:.4f})")

    # Als je toch exact jouw oude variabelen wilt houden:
    # - 'tours' is nog steeds de list-of-lists met stops per driver
    # - 'tour_lengths' kun je zo bepalen:
    tour_lengths = tour_lengths_after
