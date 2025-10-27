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


# === Metaheuristiek (open routes) – 2-opt + swap + Simulated Annealing ===

import random, math             # random = toevalspakketje, math = rekenknoppen
import numpy as np              # numpy = handige lijstjes met getallen
import matplotlib.pyplot as plt # plt = ding om plaatjes/plots te maken

# ---- tour_len: reken hoeveel tijd één route kost (we gaan NIET terug naar depot)
def tour_len(t):
    if not t:                   # als de route leeg is (niemand bezoeken)
        return 0                # kost het letterlijk 0 tijd, duh
    s = distance_matrix[(depot, t[0])]      # start: depot → eerste klant
    s += sum(                                  # + alles tussen klanten optellen
        distance_matrix[(t[i], t[i+1])]        # stukje van klant i → klant i+1
        for i in range(len(t)-1)               # dat doen we voor alle i's
    )
    return s                   # klaar, dit is de tijd van deze route

# ---- two_opt: maak de volgorde in één route slimmer (stukje omdraaien als dat korter is)
def two_opt(t):
    n = len(t)                 # hoeveel stops zitten er in de route?
    if n < 4:                  # minder dan 4? dan valt er niks zinnigs te flippen
        return t               # laat maar, teruggeven en door
    improved = True            # vlaggetje: we proberen te verbeteren totdat het niet meer kan
    while improved:            # blijf loopen zolang we winst vinden
        improved = False       # reset: we hebben nog geen nieuwe winst gevonden
        best_gain = 0          # beste winst tot nu toe = 0 (niks)
        best_i = best_k = None # waar we moeten knippen? nog niet bekend
        for i in range(n-1):   # eerste knip-plek proberen
            prev = depot if i == 0 else t[i-1]   # wat zit vóór i? aan het begin is dat het depot
            for k in range(i+1, n):              # tweede knip-plek proberen
                nxt = t[k+1] if k+1 < n else None  # wat zit na k? niets als k het einde is
                # oud = wat het nu kost rondom het stukje (prev→i en k→nxt)
                old = distance_matrix[(prev, t[i])] + (distance_matrix[(t[k], nxt)] if nxt else 0)
                # nieuw = wat het kost als we het stukje i..k omdraaien (prev→k en i→nxt)
                new = distance_matrix[(prev, t[k])] + (distance_matrix[(t[i], nxt)] if nxt else 0)
                gain = old - new                 # positieve gain = korter = yay
                if gain > best_gain:            # is dit de beste tot nu toe?
                    best_gain, best_i, best_k = gain, i, k  # onthouden
        if best_gain > 0:                       # hebben we winst gevonden?
            # dan draaien we dat stukje om. letterlijk list magic:
            t = t[:best_i] + list(reversed(t[best_i:best_k+1])) + t[best_k+1:]
            improved = True                     # we hebben verbeterd → nog een rondje proberen
    return t                                     # klaar, dit is de betere route

# ---- try_swap: ruil één klant tussen twee routes als dat helpt
def try_swap(routes, d1, d2, i1, i2):
    if d1 == d2:                 # zelfde route ruilen is… raar. doen we niet.
        return None              # None = laat maar zitten
    if not routes[d1] or not routes[d2]:   # als één route leeg is: ook niet handig
        return None
    r1, r2 = routes[d1], routes[d2]        # pak de twee routes
    new1, new2 = r1[:], r2[:]              # maak kopieën (origineel niet kapot maken)
    new1[i1], new2[i2] = r2[i2], r1[i1]    # swap de twee gekozen klanten
    old = tour_len(r1) + tour_len(r2)      # tijd vóór swap
    new = tour_len(new1) + tour_len(new2)  # tijd ná swap
    if new < old:                          # beter? love that for us
        out = [r[:] for r in routes]       # kopie van alle routes
        out[d1], out[d2] = new1, new2      # vervang de twee aangepaste routes
        return out                         # geef die nieuwe set terug
    return None                            # niet beter → doe maar niks

# ---- sa: Simulated Annealing = soms ook slechte moves toestaan om uit local trap te komen
def sa(open_tours, T=400.0, Tend=1.0, alpha=0.995, iters=200, seed=42):
    import random, math
    random.seed(seed)

    # start = maak elke route eerst wat slimmer
    cur = [two_opt(r[:]) for r in open_tours]
    best = [r[:] for r in cur]
    best_total = sum(tour_len(r) for r in cur)

    while T > Tend:                          # afkoelen
        for _ in range(iters):               # een paar pogingen per temperatuur
            # kies 2 routes en 2 posities
            d1, d2 = random.sample(range(len(cur)), 2)
            if not cur[d1] or not cur[d2]:   # lege route? skip
                continue
            i1 = random.randrange(len(cur[d1]))
            i2 = random.randrange(len(cur[d2]))

            # buur-oplossing = zelfde als nu maar met één swap
            new = [r[:] for r in cur]
            new[d1][i1], new[d2][i2] = new[d2][i2], new[d1][i1]

            # simpele delta: herbereken totale kost (lekker dom, maar duidelijk)
            cur_total = sum(tour_len(r) for r in cur)
            new_total = sum(tour_len(r) for r in new)
            delta = new_total - cur_total

            # SA-regel: accepteer als beter, of soms ook slechter met kans exp(-delta/T)
            if delta < 0 or random.random() < math.exp(-delta / T):
                cur = new
                # klein poetsrondje: 2-opt alleen op de twee aangeraakte routes
                cur[d1] = two_opt(cur[d1])
                cur[d2] = two_opt(cur[d2])

                cur_total = sum(tour_len(r) for r in cur)
                if cur_total < best_total:
                    best, best_total = [r[:] for r in cur], cur_total

        T *= alpha                           # koel verder af
    return best                              # beste set routes

# ---- run: start met jullie greedy tours en optimaliseer
tours0 = [r[:] for r in tours]     # kopie van de input (we slopen het origineel niet)
best_tours = sa(tours0)            # draai het SA-circus en krijg betere routes

# ---- print per route de tijd (we doen GEEN totaalprint, wilde je niet)
print("\n== RESULTATEN (open routes) ==")  # gewoon een kopje, voor de vibes
for d, t in enumerate(best_tours, 1):      # loop door alle routes
    print(f"Route {d}: {t} | tijd: {tour_len(t):.1f} time units")  # route + tijd

# ---- vergelijk oud vs nieuw, per route, in simpele mensentaal
print("\n== Verschil oud vs nieuw per route ==")  # nog een kopje (dramatisch effect)
for d, (old_t, new_t) in enumerate(zip(tours, best_tours), 1):  # pak oude en nieuwe naast elkaar
    diff = tour_len(old_t) - tour_len(new_t)        # positief = nieuw is sneller (korter)
    tag = 'korter' if diff > 0 else ('langer' if diff < 0 else 'gelijk')  # woorden zijn leuk
    print(f"Het verschil van oud en nieuw in route {d} is {diff:.1f} time units ({tag}).")

# ---- plot: teken alleen de nieuwe (open) routes, zodat je ook iets moois ziet
coords = np.array(coordinates)         # lijst met (x,y) van depot en klanten
try:
    palette = colors                   # als jullie al kleuren hadden: gebruik die
except NameError:
    palette = plt.cm.tab10.colors      # anders pak ik standaard kleurtjes

plt.figure(figsize=(8,6))              # maak een plaatje van 8x6 inch
for d, t in enumerate(best_tours):     # voor elke route:
    if not t:                          # lege route? skip, boring
        continue
    path = [depot] + t                 # open route: we starten wel bij depot
    xs, ys = coords[path,0], coords[path,1]  # pak de x-jes en y-tjes
    plt.plot(xs, ys, linewidth=2.0, color=palette[d % len(palette)], label=f"Driver {d+1}")  # lijn tekenen
    plt.scatter(coords[t,0], coords[t,1], s=60, color=palette[d % len(palette)], edgecolors="black")  # puntjes

plt.scatter(coords[depot,0], coords[depot,1], s=120, color='hotpink', marker='s', label='Depot')  # depot = roze vierkant
plt.title("Routes per chauffeur (open)")  # titel, want we zijn netjes opgevoed
plt.xlabel("X"); plt.ylabel("Y")          # as-jes met naam
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()  # legenda, raster, en laten zien
