import osmnx as ox
import pandas as pd
import numpy as np
from geopy.distance import geodesic

# Load the graphml file and station data
G = ox.load_graphml("states_graph/graphml/delhi_highways.graphml")
stations = pd.read_csv("station_visibility_summary.csv")
stations.columns = stations.columns.str.strip()  # Clean column names

total_nodes = set(G.nodes)
station_assigned_nodes = set()

# Helper function to calculate distance between two lat/lon points in km
def haversine(coord1, coord2):
    return geodesic(coord1, coord2).km

# Function to assign weather data to nodes for a given radius
def assign_weather_data(radius):
    current_radius_assigned = set()

    for _, station in stations.iterrows():
        station_coord = (station['latitute'], station['longitude'])

        for node, node_data in G.nodes(data=True):
            node_coord = (node_data['y'], node_data['x'])

            if haversine(station_coord, node_coord) <= radius:
                if 'visibility_sum' in node_data:
                    G.nodes[node]['visibility_sum'] += station['avg_visiblity']
                    G.nodes[node]['rainfall_sum'] += station['avg rainfall']
                    G.nodes[node]['cloud_cover_sum'] += station['avg_cloud_cover']
                    G.nodes[node]['station_count'] += 1
                else:
                    G.nodes[node]['visibility_sum'] = station['avg_visiblity']
                    G.nodes[node]['rainfall_sum'] = station['avg rainfall']
                    G.nodes[node]['cloud_cover_sum'] = station['avg_cloud_cover']
                    G.nodes[node]['station_count'] = 1

                current_radius_assigned.add(node)

    station_assigned_nodes.update(current_radius_assigned)

    return len(total_nodes - station_assigned_nodes) == 0  # Return if all nodes are assigned

# Phase 1: Coarse search with radii from 100 to 200 km (step = 10 km)
print("Phase 1: Coarse search")
for radius in range(100, 201, 10):
    print(f"Trying with radius: {radius} km")
    if assign_weather_data(radius):
        print(f"All nodes assigned data with radius: {radius} km")
        break
else:
    print("Failed to assign all nodes in Phase 1.")

# Phase 2: Refinement with radii from (x-10) to (x+10) km (step = 5 km)
refined_start = max(radius - 10, 0)
refined_end = radius + 10
print(f"\nPhase 2: Refining search around {radius} km")

for refined_radius in range(refined_start, refined_end + 1, 5):
    print(f"Trying with radius: {refined_radius} km")
    station_assigned_nodes.clear()  # Reset assigned nodes for refinement
    if assign_weather_data(refined_radius):
        print(f"All nodes assigned data with radius: {refined_radius} km")
        break

# Phase 3: Final refinement with radii from (x-5) to (x+5) km (step = 1 km)
final_start = max(refined_radius - 5, 0)
final_end = refined_radius + 5
print(f"\nPhase 3: Final refinement around {refined_radius} km")

for final_radius in range(final_start, final_end + 1, 1):
    print(f"Trying with radius: {final_radius} km")
    station_assigned_nodes.clear()  # Reset assigned nodes for final refinement
    if assign_weather_data(final_radius):
        print(f"All nodes assigned data with radius: {final_radius} km")
        break

# Calculate final averages and clean up intermediate fields
for node, node_data in G.nodes(data=True):
    if 'station_count' in node_data:
        G.nodes[node]['avg_visibility'] = node_data['visibility_sum'] / node_data['station_count']
        G.nodes[node]['avg_rainfall'] = node_data['rainfall_sum'] / node_data['station_count']
        G.nodes[node]['avg_cloud_cover'] = node_data['cloud_cover_sum'] / node_data['station_count']

        del G.nodes[node]['visibility_sum']
        del G.nodes[node]['rainfall_sum']
        del G.nodes[node]['cloud_cover_sum']
        del G.nodes[node]['station_count']

# Save the final graph with weather data
ox.save_graphml(G, "india_highways_with_weather.graphml")

print("Finished updating the graph with weather data.")
