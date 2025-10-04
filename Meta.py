import math as math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the data from the excel file
df = pd.read_excel("newspaper problem instance.xlsx")
drivers = 4
coordinates = []
for i in range(len(df)):
    coordinates.append((df.iloc[i]["xcoord"], df.iloc[i]["ycoord"]))
coordinates = np.array(coordinates)
num_coordinates = len(coordinates)

# Define the distance matrix to define the distance between two points + locations
distance_matrix = {}
for i in range(num_coordinates):
    for j in range(num_coordinates):
        dx = coordinates[i][0] - coordinates[j][0]
        dy = coordinates[i][1] - coordinates[j][1]
        distance_matrix[(i, j)] = (dx*dx + dy*dy) ** 0.5

depot = coordinates[0]
# index of the depot in the coordinates array (used for distance_matrix keys)
depot_index = 0

# Available stops (excluding depot)
num_stops = num_coordinates - 1  # Minus 1 for the depot

# Check that the stops can be evenly divided among drivers
if num_stops % drivers != 0:
    raise ValueError(
        f"Your instance has {num_stops} stops (excl. depot), which is not evenly divisible by {drivers}. "
        "Fix your data or adjust drivers/stops_per_driver, thanks in advance."
    )

# integer number of stops each driver should handle
stops_per_driver = num_stops // drivers # To discard the decimal/fractional part
# Define the tour length, starting at the depot and the delivery boys delivering the last newspaper and not returning to depot

def tour_length_with_depot(tour):
    if not tour:
        return 0
    # tour should be a sequence of integer indices referring to rows in `coordinates`
    L = distance_matrix[(depot_index, tour[0])]  # from depot to first stop
    # add distances between subsequent stops
    for a, b in zip(tour[:-1], tour[1:]):
        L += distance_matrix[(a, b)]
    return L
    
print("disntace matricc FHSDKJFH = ", distance_matrix[(0, 1)])





















