# Opdracht 1 voorbeeld van de les van week 1
# deze moeten we zo aanpassen dat het met 120 locaties is en 
# verdeeld is over 4 krantjochies
NUM_LOCATIONS = 4
locations = range(0,NUM_LOCATIONS)
coordinates = [(0,0), (0,1), (0.5,1), (0.5,0)]
distance_matrix = {}
for from_loc in locations:
    for to_loc in locations:
        distance_matrix[(from_loc,to_loc)] = ((coordinates[from_loc][0]-coordinates[to_loc][0])**2+(coordinates[from_loc][1]-coordinates[to_loc][1])**2)**0.5
print(distance_matrix)

tour = [0,1,2,3]
tour_length = 0
for i in range(len(tour)-1):
    tour_length += distance_matrix[tour[i],tour[i+1]]
tour_length += distance_matrix[tour[-1],tour[0]]

tour_length