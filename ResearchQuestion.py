import pandas as pd
import math

# ===== Inlezen (zelfde als jouw script) =====
df = pd.read_excel("newspaper problem instance.xlsx")

coordinates = []
for i in range(len(df)):
    coordinates.append((df.iloc[i]["xcoord"], df.iloc[i]["ycoord"]))

NUM_LOCATIONS = len(coordinates)

# ===== Afstandsmatrix (zelfde idee) =====
distance_matrix = {}
for i in range(NUM_LOCATIONS):
    for j in range(NUM_LOCATIONS):
        dx = coordinates[i][0] - coordinates[j][0]
        dy = coordinates[i][1] - coordinates[j][1]
        distance_matrix[(i, j)] = math.hypot(dx, dy)

# ===== Config =====
NUM_DRIVERS = 4
STOPS_PER_DRIVER = None   # <-- geen limiet; zet op een int als je wél een cap wilt
DEPOT = 0

# ===== Helpers =====
def tour_length_with_depot(tour):
    """Lengte inclusief 0 -> ... -> 0"""
    if not tour:
        return 0.0
    full = [DEPOT] + tour 
    L = 0.0
    for i in range(len(full) - 1):
        L += distance_matrix[(full[i], full[i+1])]
    return L

def marginal_append_cost(current_end, tour_is_empty, j):
    """
    Kostenstijging als we stop j achteraan toevoegen.
    - Als tour leeg is: 0->j->0
    - Anders: vervang 'end->0' door 'end->j->0'
    """
    if tour_is_empty:
        return distance_matrix[(DEPOT, j)] + distance_matrix[(j, DEPOT)]
    old_return = distance_matrix[(current_end, DEPOT)]
    new_leg = distance_matrix[(current_end, j)] + distance_matrix[(j, DEPOT)]
    return new_leg - old_return

# ===== Fair greedy zonder vaste 30-per-driver =====
unassigned = set(range(1, NUM_LOCATIONS))  # alles behalve depot
tours = [[] for _ in range(NUM_DRIVERS)]
current_end = [DEPOT] * NUM_DRIVERS
tour_lengths = [0.0] * NUM_DRIVERS  # actuele lengte incl. terug naar depot

step = 1
assignment_log = []  # (step, driver, stop, delta, new_length)

while unassigned:
    # kies kandidaten
    if STOPS_PER_DRIVER is None:
        candidates = list(range(NUM_DRIVERS))
    else:
        candidates = [d for d in range(NUM_DRIVERS) if len(tours[d]) < STOPS_PER_DRIVER]
        if not candidates:
            candidates = list(range(NUM_DRIVERS))  # fallback

    # pak driver met de kortste huidige route
    d = min(candidates, key=lambda k: tour_lengths[k])

    # kies voor deze driver de stop met minimale marginale delta kosten 
    best_j = None
    best_delta = float("inf")
    for j in unassigned:
        delta = marginal_append_cost(current_end[d], len(tours[d]) == 0, j)
        if delta < best_delta:
            best_delta = delta
            best_j = j
            
    # append stop 
    tours[d].append(best_j)
    unassigned.remove(best_j)

    # update lengte
    if len(tours[d]) == 1:
        # 0->j->0
        tour_lengths[d] = distance_matrix[(DEPOT, best_j)] + distance_matrix[(best_j, DEPOT)]
    else:
        end = current_end[d]
        tour_lengths[d] = tour_lengths[d] - distance_matrix[(end, DEPOT)] \
                          + distance_matrix[(end, best_j)] + distance_matrix[(best_j, DEPOT)]
    current_end[d] = best_j


# Eindoverzicht tours
print("\n=== Tours ===")
for d in range(NUM_DRIVERS):
    print(f"Tour {d+1}:", tours[d])

# Aantal stops per driver
print("\n=== Aantal stops per driver ===")
for d, t in enumerate(tours, start=1):
    print(f"Driver {d}: {len(t)} stops")

# Definitieve lengtes opnieuw uitrekenen (zekerheid)
tour_lengths = [tour_length_with_depot(t) for t in tours]
print("\n=== Lengtes per tour ===")
for d, L in enumerate(tour_lengths, start=1):
    print(f"Tour {d} lengte: {L:.4f}")

# Bonus: fairness stats
total = sum(tour_lengths)
mx = max(tour_lengths)
mn = min(tour_lengths)
print(f"\nTotal: {total:.4f} | Max: {mx:.4f} | Min: {mn:.4f} | Spread: {mx - mn:.4f}")

# ===== (Optioneel) 2-opt per tour =====
USE_TWO_OPT = False

def two_opt_once(path):
    n = len(path)
    if n < 4:
        return path, 0.0, False
    best_gain = 0.0
    best_i, best_k = -1, -1
    full = [DEPOT] + path + [DEPOT]
    def w(a, b): return distance_matrix[(a, b)]
    for i in range(n - 2):
        a, b = full[i], full[i+1]
        for k in range(i + 2, n):
            c, d = full[k], full[k+1]
            if b == c or d == a:
                continue
            old = w(a, b) + w(c, d)
            new = w(a, c) + w(b, d)
            gain = old - new
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_i, best_k = i, k
    if best_gain > 1e-12:
        new_path = path[:best_i] + path[best_i:best_k][::-1] + path[best_k:]
        return new_path, best_gain, True
    return path, 0.0, False

def two_opt(path, max_iters=10_000):
    total = 0.0
    it = 0
    while it < max_iters:
        path, gain, ok = two_opt_once(path)
        if not ok:
            break
        total += gain
        it += 1
    return path, total

if USE_TWO_OPT:
    print("\n=== 2-opt ===")
    improved = []
    for d, t in enumerate(tours, start=1):
        t2, gain = two_opt(t)
        improved.append(t2)
        L0 = tour_length_with_depot(t)
        L1 = tour_length_with_depot(t2)
        print(f"Driver {d}: Δ-winst = {gain:.4f} | {L0:.4f} -> {L1:.4f}")
    tours = improved
    tour_lengths = [tour_length_with_depot(t) for t in tours]
    print("\nNa 2-opt lengtes:")
    for d, L in enumerate(tour_lengths, start=1):
        print(f"Tour {d} lengte: {L:.4f}")


# Plotting
import matplotlib.pyplot as plt

# ===== Plotten van alle tours =====
plt.figure(figsize=(8, 6))

# Plot alle routes per chauffeur
for d, t in enumerate(tours, start=1):
    if not t:
        continue
    # X en Y van deze tour
    x_coords = [coordinates[i][0] for i in ([DEPOT] + t )]
    y_coords = [coordinates[i][1] for i in ([DEPOT] + t )]

    plt.plot(x_coords, y_coords, marker='o', label=f"Driver {d}")

# Plot het depot apart (groen)
plt.scatter(coordinates[DEPOT][0], coordinates[DEPOT][1],
            s=120, color='hotpink', marker='s', label='Depot')

plt.title("Routes per chauffeur (Greedy Fair)")
plt.xlabel("X-coördinaat")
plt.ylabel("Y-coördinaat")
plt.legend()
plt.grid(True)
plt.show()