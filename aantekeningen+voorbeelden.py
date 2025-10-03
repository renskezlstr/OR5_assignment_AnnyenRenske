"""
More generic implementation of the nearest neighbor heuristic for the Traveling Salesman Problem (TSP), 
on any instance defined in an Excel workbook.

The Excel file's name and sheet name are defined in constants in the code.
The distance metric can be set to either "manhattan" or "euclidean". It is defined in a constant in the code.

Important: 
- The Excel file should be stored in the same directory as this script.
- The sheet should contain three columns: 'name', 'x', and 'y'.
"""

import random
import pandas
import time
import matplotlib.pyplot as plt
random.seed(42)  # for reproducibility

# constants
#FILE_NAME = None  # Set to None to use random instance, or provide a filename like "tsp_instance.xlsx"
#FILE_NAME = "tsp_instance.xlsx"
FILE_NAME = "tsp_instance_or4exam.xlsx"
#FILE_NAME = "berlin52.xlsx"
#FILE_NAME = "rd100.xlsx"

SHEET_NAME = "Sheet1"
#DISTANCE = "manhattan"  
DISTANCE = "euclidean"

def generate_random_instance(num_locations=1000, x_max=100, y_max=100, seed=42):
    """
    Generate a random TSP instance with numbered location names.

    Parameters:
        num_locations (int): number of locations to generate
        x_max (int): maximum x-coordinate (default 100)
        y_max (int): maximum y-coordinate (default 100)
        seed (int or None): random seed for reproducibility

    Returns:
        tuple: (location_names, coordinates_location)
            - location_names: list of strings ["loc0", "loc1", ...]
            - coordinates_location: list of tuples [(x1, y1), (x2, y2), ...]
    """    
    if seed is not None:
        random.seed(seed)

    location_names = [f"loc{i}" for i in range(num_locations)]
    coordinates_location = [
        (random.uniform(0, x_max), random.uniform(0, y_max))
        for _ in range(num_locations)
    ]

    return location_names, coordinates_location

def read_locations(file_name, sheet_name):
    """
    Read location names and (x, y) coordinates from an Excel sheet.

    Parameters:
        file_name (str or path-like): Path to the Excel file.
        sheet_name (str or int): Name or index of the sheet to read. The sheet must contain 'name', 'x', and 'y' columns.

    Returns:
        tuple[list[str], list[tuple[float, float]]]: A pair with the list of location names and the list of coordinate tuples.
    """
    df_locations = pandas.read_excel(file_name, sheet_name=sheet_name)
    location_names = df_locations['name'].tolist()
    coordinates_location = list(zip(df_locations['x'], df_locations['y']))
    return location_names, coordinates_location

def compute_distances(coordinates_location, metric):
    """Compute pairwise distances between all coordinate pairs using the chosen metric.
    Args:
        coordinates_location: Iterable of (x, y) coordinate pairs.
        metric: Distance metric to use. Use "manhattan" for Manhattan distance; any other value selects Euclidean distance.
    Returns:
        A dictionary mapping (i, j) index pairs to the distance between coordinates i and j.
    Notes:
        Distances are computed for all ordered pairs, including i == j (which yields 0.0).
    """
    euclidean_distance_matrix = {
        (loc1, loc2): ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        for loc1, (x1, y1) in enumerate(coordinates_location)
        for loc2, (x2, y2) in enumerate(coordinates_location)
    }
    manhattan_distance_matrix = {
        (loc1, loc2): abs(x1 - x2) + abs(y1 - y2)
        for loc1, (x1, y1) in enumerate(coordinates_location)
        for loc2, (x2, y2) in enumerate(coordinates_location)
    }
    if metric == "manhattan":
        distance_matrix = manhattan_distance_matrix
    else:
        distance_matrix = euclidean_distance_matrix

    return distance_matrix

def nearest_neighbor(locations, distance, start_location):
    num_locations = len(locations)

    # apply nearest neighbor heuristic
    tour = [start_location]
    unvisited = set(locations) - {start_location}

    # repeat until all locations are visited
    while unvisited:
        # step 2: find the nearest unvisited location closest to the last visited location
        last_location = tour[-1]    # get the last location in the tour
        nearest_unvisted_location = min(unvisited, key=lambda j: distance[last_location,j]) # nearest unvisited location
        unvisited.remove(nearest_unvisted_location)  # remove it from unvisited set
        tour.append(nearest_unvisted_location)  

    # output: total distance of tour
    length = sum(distance[tour[i], tour[i+1]] for i in range(num_locations - 1)) \
        + distance[tour[num_locations - 1], tour[0]]

    return tour, length

def show_nearest_neighbor_tour(location_names, coordinates_location, metric, tour, tour_distance):
    num_locations = len(location_names)

    # output: # order of visiting of locations (tour visiting each location once)
    print("Tour:", [location_names[i] for i in tour])
    print(f"Tour distance based on {metric} metric:", tour_distance)

    # Plot all locations
    # Note: Copied the suggestion from GitHub Copilot
    x_coords = [coord[0] for coord in coordinates_location]
    y_coords = [coord[1] for coord in coordinates_location]
    plt.scatter(x_coords, y_coords, color='hotpink')

    # Annotate each location with its name and order
    for idx, (name, x, y) in enumerate(zip(location_names, x_coords, y_coords)):
        plt.text(x, y, f"{name}", fontsize=9, ha='right')
        
    # Plot the tour (solution)
    tour_coords = [coordinates_location[i] for i in tour] + [coordinates_location[tour[0]]]  # close the loop
    tour_x = [coord[0] for coord in tour_coords]
    tour_y = [coord[1] for coord in tour_coords]
    plt.plot(tour_x, tour_y, color='mediumorchid', linestyle='-', marker='o')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.suptitle("TSP Instance and Solution")
    plt.title(f"Number of locations: {num_locations}, Total distance ({metric}): {tour_distance:.2f}", fontsize=10)
    plt.show()

def main():
    if FILE_NAME is None:
        location_names, coordinates_location = generate_random_instance(num_locations=100, seed=84)
    else:
        location_names, coordinates_location = read_locations(FILE_NAME, SHEET_NAME)
    locations = list(range(len(location_names)))
    metric = DISTANCE
    distance_matrix = compute_distances(coordinates_location, metric)

    start_time = time.time()

    best_tour = None
    best_distance = float("inf")
    for start_location in locations:
        tour, tour_distance = nearest_neighbor(locations, distance_matrix, start_location)
        if tour_distance < best_distance:
            best_distance = tour_distance
            best_tour = tour
    start_location = random.choice(locations)
    tour, tour_distance = nearest_neighbor(locations, distance_matrix, start_location)
    
    elapsed_time = time.time() - start_time
    
    print(f"Elapsed time for nearest_neighbor: {elapsed_time:.4f} seconds")
    show_nearest_neighbor_tour(location_names, coordinates_location, metric, best_tour, best_distance)
    
    current_tour = [5, 2, 0, 3, 1, 4]
    for i in range(len(current_tour)-1):
        for j in range(len(current_tour)-1):
            if i >= j+2:
                print(f"distance {current_tour[i]} to {current_tour[j]} = {distance_matrix[current_tour[i], current_tour[j]]}")
                loci = current_tour[i]
                lociplus1 = current_tour[i+1]
                locj = current_tour[j]
                locjplus1 = current_tour[j+1]
                
                neighbor_tour = current_tour[0:i+1]+current_tour[i+1:j+1][::-1]+current_tour[j+1:]
                delta = (distance_matrix[loci, lociplus1] + distance_matrix[locj, locjplus1] - distance_matrix[loci, locj] - distance_matrix[lociplus1, locjplus1])
                print(f"delta for 2_opt swap: {delta}")
    # delta is korter omdat de nieuwe tour langer is dan de oude tour (die kruist namelijk)

if __name__ == "__main__":
    main()
