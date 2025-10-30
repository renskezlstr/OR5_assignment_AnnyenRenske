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

# *******Visualisatie*******
plt.figure(figsize=(8, 6))                          
colors = ["#069AF3", "#F57BC7", "#8C000F", "#00FF00"]
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


# *******Metaheuristiek – 2-opt + swap + Simulated Annealing*******

import random, math             
import numpy as np              
import matplotlib.pyplot as plt 

# *******tour_len: reken hoeveel tijd één route kost*******
def tour_len(t):
    if not t:                   # als de route leeg is (niemand bezoeken)
        return 0                # kost het  0 tijd
    s = distance_matrix[(depot, t[0])]      # start: depot naar de eerste klant
    s += sum(                                  # + alles tussen klanten optellen
        distance_matrix[(t[i], t[i+1])]        # stukje van klant i naar klant i+1
        for i in range(len(t)-1)               # dat doen voor elke i
    )
    return s                   

#*******two_opt: maak de volgorde in één route slimmer (stukje omdraaien als dat korter is)*******
def two_opt(t):
    n = len(t)                 # hoeveel stops zitten er in de route?
    if n < 4:                  # minder dan 4? dan kan 2-opt niks verbeteren
        return t               # teruggeven zoals-ie is
    improved = True            # proberen te verbeteren totdat het niet meer kan
    while improved:            # blijf loopen zolang er winst te vinden is
        improved = False       # reset: er is nog geen nieuwe winst gevonden
        best_gain = 0          # beste winst tot nu toe = 0 (niks)
        best_i = best_k = None # waar we moeten stoppen? is nog niet bekend
        for i in range(n-1):   # eerste stop-plek proberen
            prev = depot if i == 0 else t[i-1]   # wat zit voor i? aan het begin is dat het depot
            for k in range(i+1, n):              # tweede stop-plek proberen
                nxt = t[k+1] if k+1 < n else None  # wat zit na k? niets als k het einde is
                # oud = wat het nu kost rondom het stukje (prev→i en k→nxt)
                old = distance_matrix[(prev, t[i])] + (distance_matrix[(t[k], nxt)] if nxt else 0)
                # nieuw = wat het kost als we het stukje i..k omdraaien (prev→k en i→nxt)
                new = distance_matrix[(prev, t[k])] + (distance_matrix[(t[i], nxt)] if nxt else 0)
                gain = old - new                 # positieve gain = korter 
                if gain > best_gain:            # is dit de beste tot nu toe?
                    best_gain, best_i, best_k = gain, i, k  # onthouden
        if best_gain > 0:                       # hebben we winst gevonden?
            # dan draaien we dat stukje om:
            t = t[:best_i] + list(reversed(t[best_i:best_k+1])) + t[best_k+1:]
            improved = True                     # we hebben verbeterd, dus nog een rondje proberen
    return t                                     # klaar, dit is de betere route

# *******try_swap: ruil één klant tussen twee routes als dat helpt*******
def try_swap(routes, d1, d2, i1, i2):
    if d1 == d2:                 # zelfde route ruilen is niet de bedoeling, dus doen we niet.
        return None              # None = geen verbetering
    if not routes[d1] or not routes[d2]:   # als één route leeg is, niks doen
        return None
    r1, r2 = routes[d1], routes[d2]        # pak de twee routes
    new1, new2 = r1[:], r2[:]              # maak kopieën (origineel niet kapot maken)
    new1[i1], new2[i2] = r2[i2], r1[i1]    # swap de twee gekozen klanten
    old = tour_len(r1) + tour_len(r2)      # tijd voor swap
    new = tour_len(new1) + tour_len(new2)  # tijd na swap
    if new < old:                          # beter? 
        out = [r[:] for r in routes]       # kopie van alle routes
        out[d1], out[d2] = new1, new2      # vervang de twee aangepaste routes
        return out                         # geef die nieuwe set terug
    return None                            # niet beter dus niks doen

# *******sa: Simulated Annealing = soms ook slechte routes toestaan om uit local trap te komen*******
def sa(open_tours, T=400.0, Tend=1.0, alpha=0.995, iters=200, seed=42):
    import random, math              # we gebruiken toeval (random) en (math) dus welke methode gaan we gebruiken
    random.seed(seed)                # zet de “dobbelsteen” vast (voor reproduceerbaarheid)
    # start: maak eerst elke route lokaal beter met 2-opt 
    cur = [two_opt(r[:]) for r in open_tours]   # kopieer elke route en 2-opt het
    best = [r[:] for r in cur]                  # “best tot nu toe” = wat we nu hebben
    best_max = max(tour_len(r) for r in cur)  # totale tijd van “best” uitrekenen

    while T > Tend:                   # zolang de temperatuur nog boven de eind-temp is (we zijn nog “los”)
        for _ in range(iters):        # doe een aantal pogingen (swaps) op deze temperatuur
            # kies 2 willekeurige verschillende routes
            d1, d2 = random.sample(range(len(cur)), 2)
            if not cur[d1] or not cur[d2]:   # als een van die routes leeg is: skip (niks te ruilen)
                continue
            # kies in elke route een willekeurige klantpositie
            i1 = random.randrange(len(cur[d1]))
            i2 = random.randrange(len(cur[d2]))

            # maak een buur-oplossing: kopieer de huidige routes en ruil precies die twee klanten
            new = [r[:] for r in cur]                 # diepe kopie (zodat we cur niet verpesten)
            new[d1][i1], new[d2][i2] = new[d2][i2], new[d1][i1]  # daadwerkelijke swap

            # reken de maximale tijd uit voor en na de swap (simpel > snel)
            cur_max = max(tour_len(r) for r in cur)  # bereken de tijd van de langste route (laatste bezorger klaar)
            new_max = max(tour_len(r) for r in new)  # nieuwe totale eindtijd (langste route) na een swap
            delta = new_max - cur_max              # positief = slechter, negatief = beter

            # Simulated Annealing-regel:
            # als beter (delta < 0): altijd accepteren
            # als slechter: soms toch accepteren met kans exp(-delta / T)
            if delta < 0 or random.random() < math.exp(-delta / T):
                cur = new                              # accepteer de buur-oplossing
                # verbeter alleen de twee aangepaste routes met 2-opt
                cur[d1] = two_opt(cur[d1])
                cur[d2] = two_opt(cur[d2])

                # update “best” als het geheel nu echt korter is dan wat we ooit hadden
                cur_max = max(tour_len(r) for r in cur)
                if cur_max < best_max:
                    best, best_max = [r[:] for r in cur], cur_max

        T *= alpha                         # afkoelfactor na elke ronde doen we T = T * alpha  langzaam afkoelen
        """
        Hoge temperatuur = veel experimenteren, kan ook slechtere dingen proberen
        Lage temperatuur = alleen nog kleine verbeteringen, focust op fine-tunen
        dus: hoe kouder → hoe strenger → het pakt minder snel slechtere routes,
        maar daardoor niet automatisch betere routes, want het durft minder te experimenteren."""
    return best                            

# *******run: start met greedy tours en optimaliseer*******
tours0 = [r[:] for r in tours]     # kopie van de input zodat de originele routes bewaard blijven
best_tours = sa(tours0)            # run de SA en krijg betere routes

# *******print per route de tijd*******
print("\n== RESULTATEN (open routes) ==")  
for d, t in enumerate(best_tours, 1):      # loop door alle routes
    print(f"Route {d}: {t} | tijd: {tour_len(t):.1f} time units")  # route + tijd

# *******vergelijk oud vs nieuw, per route*******
print("\n== Verschil oud vs nieuw per route ==")  
for d, (old_t, new_t) in enumerate(zip(tours, best_tours), 1):  # pak oude en nieuwe naast elkaar
    diff = tour_len(old_t) - tour_len(new_t)        # positief = nieuw is sneller (korter)
    tag = 'korter' if diff > 0 else ('langer' if diff < 0 else 'gelijk')  
    print(f"Het verschil van oud en nieuw in route {d} is {diff:.1f} time units ({tag}).")

# *******plot: teken alleen de nieuwe (open) routes*******
coords = np.array(coordinates)         # lijst met (x,y) van depot en klanten
palette = colors

plt.figure(figsize=(8,6))              # maak een plaatje van 8x6 
for d, t in enumerate(best_tours):     # voor elke route:
    if not t:                          # lege route? skip
        continue
    path = [depot] + t                 # open route:  starten bij depot
    xs, ys = coords[path,0], coords[path,1]  # pak de x-jes en y-tjes
    plt.plot(xs, ys, linewidth=2.0, color=palette[d % len(palette)], label=f"Driver {d+1}")  # lijn tekenen
    plt.scatter(coords[t,0], coords[t,1], s=60, color=palette[d % len(palette)], edgecolors="black")  # puntjes

plt.scatter(coords[depot,0], coords[depot,1], s=120, color='hotpink', marker='s', label='Depot')  # depot = roze vierkant
plt.title("Routes per chauffeur (open)")  
plt.xlabel("X"); plt.ylabel("Y")          
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show() 
