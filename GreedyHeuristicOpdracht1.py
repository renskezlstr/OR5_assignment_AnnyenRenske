
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

# depot apart
plt.scatter(coords[depot, 0], coords[depot, 1], s=120, color='hotpink', marker='s', label='Depot')

plt.title("Routes per chauffeur (Greedy)")
plt.xlabel("X-coördinaat")
plt.ylabel("Y-coördinaat")
plt.legend()
plt.grid(True)
plt.show()

