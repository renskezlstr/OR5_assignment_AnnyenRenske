
import math, random, copy
import pandas as pd

1. 
# excel inlezen
df = pd.read_excel("newspaper problem instance.xlsx")
# lijst met coördinaten maken
coordinates = []
for i in range(len(df)):
    coordinates.append((df.iloc[i]["xcoord"], df.iloc[i]["ycoord"]))

NUM_LOCATIONS = len(coordinates) 

# afstandsmatrix bouwen zelfde als in voorbeeld van les 1
distance_matrix = {}
for i in range(NUM_LOCATIONS):
    for j in range(NUM_LOCATIONS):
        dx = coordinates[i][0] - coordinates[j][0]
        dy = coordinates[i][1] - coordinates[j][1]
        distance_matrix[(i, j)] = (dx*dx + dy*dy) ** 0.5

      
NUM_DRIVERS = 4
# we gaan ervan uit dat iedere driver max 30 stops krijgt,
# zodat dat eerlijk verdeeld is
STOPS_PER_DRIVER = 30 
#dit kunnen we aanpassen als we willen, zodat de verdeling ander is en er een derde research question van gemaakt kan worden
unassigned = set(range(1, NUM_LOCATIONS)) 
tours = [[] for _ in range(NUM_DRIVERS)]
current = [0] * NUM_DRIVERS  

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


2.
# === Helpers (consistente lengte, incl. terug naar depot) ===
def tour_length_with_depot(tour):
    if not tour:
        return 0.0
    full = [depot] + tour  
    return sum(distance_matrix[(full[i], full[i+1])] for i in range(len(full)-1))

def total_length(all_tours):
    return sum(tour_length_with_depot(t) for t in all_tours)

def stops_ok(all_tours):
    """Respecteer cap als die is gezet (STOPS_PER_DRIVER kan ook None zijn)."""
    if STOPS_PER_DRIVER is None:
        return True
    return all(len(t) <= STOPS_PER_DRIVER for t in all_tours)

# === Buurmoves ===
def intra_two_opt(t):
    """2-opt op één route (retour naar depot zit in de evaluatie, maar niet in de slice)."""
    n = len(t)
    if n < 4:
        return t
    i, k = sorted(random.sample(range(n), 2))
    if k - i < 1:
        return t
    return t[:i] + t[i:k+1][::-1] + t[k+1:]

def intra_relocate(t):
    """Verplaats één stop naar een andere positie in dezelfde route."""
    n = len(t)
    if n < 2:
        return t
    i = random.randrange(n)
    j = random.randrange(n)
    if i == j:
        return t
    new = t[:]
    node = new.pop(i)
    new.insert(j, node)
    return new

def inter_swap(all_tours):
    """Wissel één stop tussen twee verschillende drivers."""
    d1, d2 = random.sample(range(NUM_DRIVERS), 2)
    t1, t2 = all_tours[d1], all_tours[d2]
    if not t1 or not t2:
        return None  # geen geldige move
    i = random.randrange(len(t1))
    j = random.randrange(len(t2))
    new = copy.deepcopy(all_tours)
    new[d1][i], new[d2][j] = new[d2][j], new[d1][i]
    return new if stops_ok(new) else None

def inter_relocate(all_tours):
    """Verplaats één stop van driver A naar driver B (respecteer cap)."""
    d_from, d_to = random.sample(range(NUM_DRIVERS), 2)
    if not all_tours[d_from]:
        return None
    new = copy.deepcopy(all_tours)
    i = random.randrange(len(new[d_from]))
    node = new[d_from].pop(i)
    insert_pos = random.randrange(len(new[d_to]) + 1)
    new[d_to].insert(insert_pos, node)
    return new if stops_ok(new) else None

def random_neighbor(all_tours):
    """Kies willekeurig een buurmove en pas ‘m toe."""
    move_type = random.choice(["intra2opt", "intrareloc", "interswap", "interreloc"])
    new = None
    if move_type == "intra2opt":
        d = random.randrange(NUM_DRIVERS)
        new = copy.deepcopy(all_tours)
        new[d] = intra_two_opt(new[d])
    elif move_type == "intrareloc":
        d = random.randrange(NUM_DRIVERS)
        new = copy.deepcopy(all_tours)
        new[d] = intra_relocate(new[d])
    elif move_type == "interswap":
        new = inter_swap(all_tours)
    else:  # interreloc
        new = inter_relocate(all_tours)

    # fallback: als move ongeldig was (None), probeer gewoon nog een simpele intra-move
    if new is None:
        d = random.randrange(NUM_DRIVERS)
        new = copy.deepcopy(all_tours)
        new[d] = intra_relocate(new[d])
    return new

# === Simulated Annealing ===
random.seed(42)

current = copy.deepcopy(tours)              # start vanaf jullie greedy
best = copy.deepcopy(current)
current_cost = total_length(current)
best_cost = current_cost

T = 1000.0          # starttemperatuur
alpha = 0.995       # afkoelratio per iteratie
T_min = 1e-3
max_iters = 20000

iters = 0
while T > T_min and iters < max_iters:
    candidate = random_neighbor(current)
    cand_cost = total_length(candidate)
    delta = cand_cost - current_cost

    # accepteer beter altijd; slechter met kans exp(-delta/T)
    if delta < 0 or random.random() < math.exp(-delta / T):
        current, current_cost = candidate, cand_cost
        if current_cost < best_cost:
            best, best_cost = copy.deepcopy(current), current_cost

    T *= alpha
    iters += 1

print("\n=== Beste oplossing na Simulated Annealing ===")
for d, t in enumerate(best, start=1):
    L = tour_length_with_depot(t)
    print(f"Driver {d}: {t} | stops: {len(t)} | lengte: {L:.2f}")
print(f"Totaal: {best_cost:.2f} (vanuit greedy start {total_length(tours):.2f})")
