
import pandas as pd
import math
import numpy as np

# stap 1: data inlezen
df = pd.read_excel('newspaper problem instance.xlsx')

#stap 2: coördinaten inlezen en deze in een tuple zetten
# hierdoor kunnen we later makkelijk de afstand tussen twee punten berekenen
coordinates = []
for i in range(len(df)):
    coordinates.append((df.iloc[i]['xcoord'], df.iloc[i]['ycoord']))

NUM_LOCATIONS = len(coordinates)

#stap 3: afstandsmatrix maken
distance_matrix = {}
for i in range(NUM_LOCATIONS):
    for j in range(NUM_LOCATIONS):
        dx = coordinates[i][0] - coordinates[j][0]
        dy = coordinates[i][1] - coordinates[j][1]
        distance_matrix[(i, j)] = math.hypot(dx, dy)
        
# Stap 4: Configuratie van het probleem (basisinstellingen voor de oplossing)
NUM_DRIVERS = 4
STOPS_PER_DRIVER = None  
DEPOT = 0


#Stap 5:De totale afstand van depot naar stops berekenen zonder terug naar het depot
def tour_length_with_depot(tour):
    """Totale afstand van depot -> stops """
    if not tour:
        return 0.0
    full = [DEPOT] + tour 
    return sum(distance_matrix[(full[i], full[i+1])] for i in range(len(full) - 1))

#stap 6: De marginale kosten berekenen als we een stop toevoegen aan het einde van een tour
def marginal_append_cost(current_end, tour_is_empty, j):
    return (distance_matrix[(DEPOT, j)] + distance_matrix[(j, DEPOT)]
            if tour_is_empty else
            distance_matrix[(current_end, j)] + distance_matrix[(j, DEPOT)] - distance_matrix[(current_end, DEPOT)])

#stap 7: Voorbereiding van de Greedy-toewijzing zonder vaste 30 stops per driver
unassigned = set(range(1, NUM_LOCATIONS))  # alles behalve depot
tours = [[] for _ in range(NUM_DRIVERS)]
current_end = [DEPOT] * NUM_DRIVERS
tour_lengths = [0.0] * NUM_DRIVERS  # actuele lengte incl. terug naar depot