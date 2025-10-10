import matplotlib.pyplot as plt
 
def distance(city1, city2):
  # Replace this with your distance calculation function (e.g., Euclidean distance)
  x1, y1 = city1
  x2, y2 = city2
  return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
 
def tsp(cities):
  visited = [False] * len(cities)
  current_city = 0
 
  tour = []
  total_distance = 0
 
  for _ in range(len(cities)):
    visited[current_city] = True
    tour.append(current_city)
 
    next_city = None
    min_distance = float('inf')  # Represents positive infinity
 
    for i in range(len(cities)):
      if visited[i]:
        continue
 
      d = distance(cities[current_city], cities[i])
      if d < min_distance:
        min_distance = d
        next_city = i
 
    current_city = next_city
    total_distance += min_distance
 
  return tour, total_distance
 
# Example usage
cities = [(2, 4), (1, 8), (7, 1), (8, 5)]
 
tour, total_distance = tsp(cities)
 
print("Tour:", tour)
print("Total Distance:", total_distance)
 
# Plotting
x_coords = [cities[i][0] for i in tour]
y_coords = [cities[i][1] for i in tour]
x_coords.append(x_coords[0])  # Close the loop for plotting
y_coords.append(y_coords[0])
 
plt.plot(x_coords, y_coords)
plt.scatter(x_coords, y_coords, s=50, color='red')  # Plot cities as red dots
plt.title("Traveling Salesman Problem - Greedy Algorithm")
plt.show()