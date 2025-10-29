
import pandas as pd
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# excel inlezen
df = pd.read_excel("newspaper problem instance.xlsx")
# lijst met coördinaten maken
coordinates = np.array([(df.iloc[i]["xcoord"], df.iloc[i]["ycoord"]) for i in range(len(df))])
NUM_LOCATIONS = len(coordinates) 

# afstandsmatrix bouwen zelfde als in voorbeeld van les 1
distance_matrix = {}
for i in range(NUM_LOCATIONS):
    for j in range(NUM_LOCATIONS):
        dx = coordinates[i][0] - coordinates[j][0]
        dy = coordinates[i][1] - coordinates[j][1]
        distance_matrix[(i, j)] = abs(dx) + abs(dy) #Manhattan Distance 
      
NUM_DRIVERS = 4
# we gaan ervan uit dat iedere driver max 30 stops krijgt,

# zodat dat eerlijk verdeeld is
STOPS_PER_DRIVER = 30
#dit kunnen we aanpassen als we willen, zodat de verdeling ander is en er een derde research question van gemaakt kan worden
unassigned = set(range(1, NUM_LOCATIONS)) 
tours = [[] for _ in range(NUM_DRIVERS)]
current = [0] * NUM_DRIVERS  
depot = 0

# alle drivers krijgen een stop toegewezen totdat alles is verdeeld
# dit wordt gedaan door steeds de dichtstbijzijnde stop te kiezen

for d in range(NUM_DRIVERS):
    while len(tours[d]) < STOPS_PER_DRIVER:
        best_j = min(unassigned, key=lambda j: distance_matrix[(current[d], j)])
        tours[d].append(best_j)
        unassigned.remove(best_j)
        current[d] = best_j

for d in range(NUM_DRIVERS):
    print(f"Tour {d+1}:", tours[d])


# lengte van iedere tour berekenen 
# inclusief terug naar depot (0)
tour_lengths = []
for tour in tours:
    tour_length = 0
    full_tour = [0] + tour 
    for i in range(len(full_tour)-1):
        tour_length += distance_matrix[(full_tour[i], full_tour[i+1])]
    tour_lengths.append(tour_length)

for d, L in enumerate(tour_lengths, start=1):
    print(f"Tour {d} lengte: {L} km")

# Visualisaltion
coords = np.array(coordinates)  # maak 'm indexeerbaar met [:, 0]
plt.figure(figsize=(8, 6))
colors = plt.cm.tab10.colors

for d, tour in enumerate(tours):
    if not tour:
        continue
    path = [depot] + tour   # sluit de route naar het depot
    xs = coords[path, 0]
    ys = coords[path, 1]

    # lijn + stops van deze driver
    plt.plot(xs, ys, linewidth=2.0, color=colors[d % len(colors)], label=f"Driver {d+1}")
    plt.scatter(coords[tour, 0], coords[tour, 1], s=60,
                color=colors[d % len(colors)], edgecolors="black")
total_distance = sum(tour_lengths)
print(f"\nTotale afstand (alle tours samen): {total_distance} km")
# depot apart
plt.scatter(coords[depot, 0], coords[depot, 1], s=120, color='hotpink', marker='s', label='Depot')

plt.title(f"Routes per chauffeur (Greedy) - Totale afstand: {total_distance:.1f} km")
plt.xlabel("X-coördinaat")
plt.ylabel("Y-coördinaat")
plt.legend()
plt.grid(True)
plt.show()

# vanaf hier begint de meta heuristiek code

# === METAHEURISTIEK DROP-IN (open routes, zelfde stijl plot, greedy wordt niet opnieuw geplot) ===
import math, random

# -----------------------------
# Kosten & helpers (open route)
# -----------------------------
def tour_length(tour):
    """Start bij depot -> eerste stop, geen retour naar depot."""
    if not tour:
        return 0
    L = distance_matrix[(depot, tour[0])]
    for i in range(len(tour)-1):
        L += distance_matrix[(tour[i], tour[i+1])]
    return L

def total_length(tours_):
    return sum(tour_length(t) for t in tours_)


# -----------------------------
# 2-opt (best improvement) voor OPEN routes
# -----------------------------
def two_opt_best_improvement(tour):
    n = len(tour)
    if n < 4:
        return tour, False

    best_delta, best_i, best_k = 0, None, None
    for i in range(n-1):
        for k in range(i+1, n):
            prev = depot if i == 0 else tour[i-1]
            nxt  = tour[k+1] if (k+1) < n else None

            old = distance_matrix[(prev, tour[i])] + (distance_matrix[(tour[k], nxt)] if nxt is not None else 0)
            new = distance_matrix[(prev, tour[k])] + (distance_matrix[(tour[i], nxt)] if nxt is not None else 0)

            delta = old - new
            if delta > best_delta:
                best_delta, best_i, best_k = delta, i, k

    if best_delta > 0:
        new_tour = tour[:best_i] + list(reversed(tour[best_i:best_k+1])) + tour[best_k+1:]
        return new_tour, True
    return tour, False

def local_opt_2opt(tour):
    cur = tour[:]
    improved = True
    while improved:
        cur, improved = two_opt_best_improvement(cur)
    return cur

# -----------------------------
# Inter-route moves (open route)
# -----------------------------
def try_relocate(tours_, from_d, to_d, idx_from, max_per_driver):
    if from_d == to_d or not tours_[from_d]:
        return None
    if len(tours_[to_d]) >= max_per_driver:
        return None

    node = tours_[from_d][idx_from]

    def removal_delta(route, idx):
        n = len(route)
        prev = depot if idx == 0 else route[idx-1]
        b    = route[idx]
        nxt  = route[idx+1] if (idx+1) < n else None
        old = distance_matrix[(prev, b)] + (distance_matrix[(b, nxt)] if nxt is not None else 0)
        new = distance_matrix[(prev, nxt)] if nxt is not None else 0
        return old - new

    rem_gain = removal_delta(tours_[from_d], idx_from)

    best_gain, best_pos = -math.inf, None
    to_route = tours_[to_d]
    for pos in range(len(to_route)+1):
        a = depot if pos == 0 else to_route[pos-1]
        c = to_route[pos] if pos < len(to_route) else None
        old = distance_matrix[(a, c)] if c is not None else 0
        new = distance_matrix[(a, node)] + (distance_matrix[(node, c)] if c is not None else 0)
        ins_gain = old - new
        if ins_gain > best_gain:
            best_gain, best_pos = ins_gain, pos

    total_gain = rem_gain + best_gain
    if total_gain > 0:
        new_tours = [r[:] for r in tours_]
        new_tours[from_d].pop(idx_from)
        new_tours[to_d].insert(best_pos, node)
        return new_tours, total_gain
    return None

def try_swap(tours_, d1, d2, i1, i2):
    if d1 == d2 or not tours_[d1] or not tours_[d2]:
        return None
    r1, r2 = tours_[d1], tours_[d2]
    a, b = r1[i1], r2[i2]

    def len_if_swap(route, idx, new_node):
        nr = route[:]
        nr[idx] = new_node
        return tour_length(nr)

    base = tour_length(r1) + tour_length(r2)
    new_len = len_if_swap(r1, i1, b) + len_if_swap(r2, i2, a)
    gain = base - new_len
    if gain > 0:
        new_tours = [r[:] for r in tours_]
        new_tours[d1][i1], new_tours[d2][i2] = new_tours[d2][i2], new_tours[d1][i1]
        return new_tours, gain
    return None

# -----------------------------
# Simulated Annealing (met 2-opt intensificatie)
# -----------------------------
def metaheuristic_sa(start_tours,
                     max_per_driver=30,
                     T_start=600.0,
                     T_end=1.0,
                     alpha=0.995,
                     iters_per_T=250,
                     seed=42):
    random.seed(seed)

    cur = [local_opt_2opt(r) for r in start_tours]
    cur_cost = total_length(cur)
    best = [r[:] for r in cur]
    best_cost = cur_cost

    T = T_start
    while T > T_end:
        for _ in range(iters_per_T):
            move_type = random.random()
            cand = None

            if move_type < 0.5:
                # relocate
                from_d = random.randrange(len(cur))
                if not cur[from_d]:
                    continue
                idx_from = random.randrange(len(cur[from_d]))
                to_choices = [d for d in range(len(cur)) if d != from_d and len(cur[d]) < max_per_driver]
                if not to_choices:
                    continue
                to_d = random.choice(to_choices)
                cand = try_relocate(cur, from_d, to_d, idx_from, max_per_driver)
            else:
                # swap
                d1, d2 = random.sample(range(len(cur)), 2)
                if not cur[d1] or not cur[d2]:
                    continue
                i1 = random.randrange(len(cur[d1]))
                i2 = random.randrange(len(cur[d2]))
                cand = try_swap(cur, d1, d2, i1, i2)

            if cand is None:
                continue

            new_tours, _gain = cand
            new_tours = [local_opt_2opt(r) for r in new_tours]
            new_cost = total_length(new_tours)
            delta = new_cost - cur_cost

            if delta < 0 or random.random() < math.exp(-delta / T):
                cur, cur_cost = new_tours, new_cost
                if cur_cost < best_cost:
                    best, best_cost = [r[:] for r in cur], cur_cost
        T *= alpha

    return best, best_cost

# -----------------------------
# Startpunt = jullie greedy
# -----------------------------
tours0 = [r[:] for r in tours]
cost0 = total_length(tours0)

best_tours, best_cost = metaheuristic_sa(
    tours0,
    max_per_driver=STOPS_PER_DRIVER
)

# -----------------------------
# Resultaten tonen zoals greedy
# -----------------------------
print("\n== METAHEURISTIEK RESULTATEN (open routes, geen retour depot) ==")
for d, t in enumerate(best_tours, start=1):
    print(f"Tour {d}:", t)

print()
for d, t in enumerate(best_tours, start=1):
    L = tour_length(t)
    print(f"Tour {d} lengte: {L} km")

print(f"\nTotale lengte alle tours: {best_cost} km")

# -----------------------------
# Visualisatie: zelfde stijl als greedy (maar meta-versie)
# -----------------------------
coords = np.array(coordinates)
plt.figure(figsize=(8, 6))
colors = plt.cm.tab10.colors

for d, tour in enumerate(best_tours):
    if not tour:
        continue
    path = [depot] + tour   # start bij depot, geen retour
    xs = coords[path, 0]
    ys = coords[path, 1]
    plt.plot(xs, ys, linewidth=2.0, color=colors[d % len(colors)], label=f"Driver {d+1}")
    plt.scatter(coords[tour, 0], coords[tour, 1], s=60,
                color=colors[d % len(colors)], edgecolors="black")

# depot
plt.scatter(coords[depot, 0], coords[depot, 1], s=120, color='hotpink', marker='s', label='Depot')

plt.title(f"Routes per chauffeur (Metaheuristiek) – Totale afstand: {best_cost:.1f} km")
plt.xlabel("X-coördinaat")
plt.ylabel("Y-coördinaat")
plt.legend()
plt.grid(True)
plt.show()
# === EINDE DROP-IN ===

