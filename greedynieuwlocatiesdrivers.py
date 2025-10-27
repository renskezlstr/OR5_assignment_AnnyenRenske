import pandas as pd                  
import numpy as np                   
import matplotlib.pyplot as plt       
import math                           

# *******Data inlezen*******
df = pd.read_excel("newspaper problem instance.xlsx")   
coordinates = np.array([(df.iloc[i]["xcoord"], df.iloc[i]["ycoord"]) for i in range(len(df))])  # sla alle (x,y)-coördinaten op
NUM_LOCATIONS = len(coordinates)       # totaal aantal locaties plus het depot
depot = 0                              # depot is locatie 0
NUM_DRIVERS = 4                        # aantal bezorgers

# *******Manhattan distance-matrix (time units)*******
distance_matrix = {}                   # dictionary voor afstanden tussen alle punten
for i in range(NUM_LOCATIONS):         # voor elke locatie i
    for j in range(NUM_LOCATIONS):     # voor elke andere locatie j
        dx = abs(coordinates[i][0] - coordinates[j][0])   # verschil in x-richting
        dy = abs(coordinates[i][1] - coordinates[j][1])   # verschil in y-richting
        distance_matrix[(i, j)] = dx + dy                 # Manhattan afstand = |dx| + |dy|

# *******Customer-centric greedy (geen cap op stops)*******
tours = [[] for _ in range(NUM_DRIVERS)]     # lijst met routes per bezorger
current = [depot] * NUM_DRIVERS              # huidige positie van elke bezorger en elke bezorger start bij depot
unassigned = set(range(1, NUM_LOCATIONS))    # alle klanten die nog niet zijn toegewezen (behalve depot)

# *******Een while functie om alle klanten die nog niet bezorgd zijn*******
while unassigned:
    best_d, best_j, best_dist = None, None, math.inf   # variabelen om beste combinatie op te slaan
    for d in range(NUM_DRIVERS):                      # bekijk elke bezorger
        j_candidate = min(unassigned, key=lambda j: distance_matrix[(current[d], j)])  # dichtstbijzijnde klant voor deze bezorger
        dist = distance_matrix[(current[d], j_candidate)]                               # afstand naar die klant
        if dist < best_dist:                        # bekijkt of dit de kortste afstand tot nu toe is
            best_d, best_j, best_dist = d, j_candidate, dist   # zo ja dan update beste combinatie
    tours[best_d].append(best_j)                    # voeg de klant toe aan route van die bezorger
    current[best_d] = best_j                        # update de huidige positie van de bezorger
    unassigned.remove(best_j)                       # verwijder de klant uit de onbezette lijst

# *******Lengtes berekenen (in time units)*******
def path_time_units(path):
    if not path:                                    # als route leeg is, lengte = 0
        return 0
    total = distance_matrix[(depot, path[0])]       # begin: afstand van depot naar eerste klant
    for i in range(len(path) - 1):                  # ga door elke klant in de route
        total += distance_matrix[(path[i], path[i+1])]  # voeg afstand tussen opeenvolgende klanten toe
    return total                                    # geef totale reistijd terug

# bereken tijd per route
tour_times = []
for d, tour in enumerate(tours, start=1):
    T = path_time_units(tour)                       # bereken tijd voor deze chauffeur in time units
    tour_times.append(T)                            # sla op
    print(f"Tour {d}:", tour)                       # print de route

# print de lengte per tour in time units
for d, T in enumerate(tour_times, start=1):
    print(f"Tour {d} lengte: {T} time units")

# ===== Visualisatie (zelfde stijl) =====
plt.figure(figsize=(8, 6))                          # maak plot aan
colors = plt.cm.tab10.colors                        # gebruik standaard kleurenpalet
coords = coordinates                                # kortere naam

for d, tour in enumerate(tours):                    # plot elke chauffeur
    if not tour:                                    # sla lege routes over
        continue
    path = [depot] + tour                           # open route: start bij depot, geen retour
    xs, ys = coords[path, 0], coords[path, 1]       # haal x- en y-coördinaten op
    plt.plot(xs, ys, linewidth=2.0, color=colors[d % 10], label=f"Driver {d+1}")  # verbind de punten met een lijn
    plt.scatter(coords[tour, 0], coords[tour, 1], s=60, color=colors[d % 10], edgecolors="black")  # markeer stops

# markeer depot
plt.scatter(coords[depot, 0], coords[depot, 1], s=120, color='hotpink', marker='s', label='Depot')

# titel + labels
plt.title(f"Routes per chauffeur (Greedy: nearest-driver-to-customer)")
plt.xlabel("X-coördinaat")                          # x-as label
plt.ylabel("Y-coördinaat")                          # y-as label
plt.legend()                                        # toon legenda
plt.grid(True)                                      # rasterlijn
plt.tight_layout()                                  # netjes positioneren
plt.show()                                          # toon de grafiek
