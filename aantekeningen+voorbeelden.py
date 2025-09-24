# Alles importeren + excel bestand inlezen
import pandas as pd
from math import hypot

df = pd.read_excel('newspaper problem instance.xlsx')

# eerste rij als het depot definiëren
depot_row = df.iloc[0]

# depot coördinaten aflezen en een variabele van maken 
depot_x = depot_row["xcoord"]
depot_y = depot_row["ycoord"]
# de variabele samenvoegen in een tuple
depot = (depot_x, depot_y)


# ervoor zorgen dat ook de alle eerste rij wordt gebruikt
stops_df = df.iloc[1:]

# 2. kopie maken zodat we het origineel niet slopen
stops_df = stops_df.copy()

# 3. index resetten
stops_df = stops_df.reset_index(drop=True)

# 4. nieuwe kolom 'stop_id' toevoegen
stops_df["stop_id"] = stops_df["location"]






# Map: stop_id -> (x, y)
stops = {int(r["stop_id"]): (float(r["xcoord"]), float(r["ycoord"])) for _, r in stops_df.iterrows()}

# === 2) Vier lege lijsten: Tour 1 t/m 4 ===
tours = {1: [], 2: [], 3: [], 4: []}

# === 3) Helper: afstand ===
def dist(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])

# === 4) For-loop over drivers; per driver kies telkens de dichtstbijzijnde volgende stop ===
NUM_DRIVERS = 4
STOPS_PER_DRIVER = 30

unassigned = set(stops.keys())            # alle nog niet toegewezen stops
last_point = {d: depot for d in range(1, NUM_DRIVERS + 1)}  # start op depot

for d in range(1, NUM_DRIVERS + 1):
    while len(tours[d]) < STOPS_PER_DRIVER:
        lp = last_point[d]
        # vind dichtstbijzijnde stop uit unassigned
        best_stop = None
        best_dist = float("inf")
        for sid in unassigned:
            dval = dist(lp, stops[sid])
            if dval < best_dist or (dval == best_dist and (best_stop is None or sid < best_stop)):
                best_dist = dval
                best_stop = sid
        # voeg toe aan tour, verwijder uit unassigned, update last_point
        tours[d].append(best_stop)
        unassigned.remove(best_stop)
        last_point[d] = stops[best_stop]

# === 5) Output: print tours ===
for d in range(1, NUM_DRIVERS + 1):
    print(f"Tour {d} ({len(tours[d])} stops):", tours[d])
