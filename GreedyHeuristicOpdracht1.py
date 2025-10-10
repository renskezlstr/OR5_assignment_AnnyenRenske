
import pandas as pd

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
