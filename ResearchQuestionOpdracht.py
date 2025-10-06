
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