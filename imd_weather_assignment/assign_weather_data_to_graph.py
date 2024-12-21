import osmnx as ox
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

# Load the graphml file and station data
G = ox.load_graphml("../data/india_highways.graphml")
stations = pd.read_csv("../data/monthly_station_metrics.csv")
stations.columns = stations.columns.str.strip()  # Clean column names

# Extract node coordinates into arrays (in radians for BallTree)
node_coords = np.array([(data['y'], data['x']) for node, data in G.nodes(data=True)])
node_coords_rad = np.radians(node_coords)

# Extract station coordinates into arrays (in radians for BallTree)
station_coords = stations[['latitude', 'longitude']].drop_duplicates().to_numpy()
station_coords_rad = np.radians(station_coords)

# Build a BallTree for fast spatial queries
node_tree = BallTree(node_coords_rad, metric='haversine')

# Constants
EARTH_RADIUS = 6371.0  # Radius of Earth in kilometers
total_nodes = set(G.nodes)
station_assigned_nodes = set()

# List of months for column names
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']


def initialize_weather_attributes(node_data):
    """Initialize monthly weather attributes for a node"""
    for month in MONTHS:
        for metric in ['visibility', 'rainfall', 'cloud_cover']:
            sum_key = f'{metric}_sum_{month}'
            node_data[sum_key] = 0
        node_data[f'station_count_{month}'] = 0


# Function to assign weather data to nodes within a radius
def assign_weather_data_for_radius(radius_km):
    current_radius_assigned = set()
    radius_rad = radius_km / EARTH_RADIUS  # Convert km to radians

    # Group stations by unique lat/lon pairs
    unique_locations = stations.drop_duplicates(subset=['latitude', 'longitude'])

    for idx, location in unique_locations.iterrows():
        # Query BallTree for nodes within the radius from the station
        station_coord = np.radians([location['latitude'], location['longitude']]).reshape(1, -1)
        node_indices = node_tree.query_radius(station_coord, r=radius_rad)[0]

        # Get all monthly data for this station location
        station_monthly_data = stations[
            (stations['latitude'] == location['latitude']) &
            (stations['longitude'] == location['longitude'])
            ]

        # Assign weather data to the found nodes
        for node_idx in node_indices:
            node = list(G.nodes)[node_idx]

            # Initialize node attributes if not already present
            if 'visibility_sum_January' not in G.nodes[node]:
                initialize_weather_attributes(G.nodes[node])

            # Process each month's data
            for _, monthly_data in station_monthly_data.iterrows():
                month = monthly_data['Month']
                # Update sums for this month
                G.nodes[node][f'visibility_sum_{month}'] += monthly_data['avg_visibility']
                G.nodes[node][f'rainfall_sum_{month}'] += monthly_data['avg_daily_rainfall']
                G.nodes[node][f'cloud_cover_sum_{month}'] += monthly_data['avg_cloud_cover']
                G.nodes[node][f'station_count_{month}'] += 1

            current_radius_assigned.add(node)

    # Update the global set of assigned nodes
    station_assigned_nodes.update(current_radius_assigned)
    return len(total_nodes - station_assigned_nodes) == 0  # Return if all nodes are assigned


# Phase 1: Coarse search with radii from 100 to 200 km (step = 10 km)
print("Phase 1: Coarse search")
for radius in range(100, 201, 10):
    print(f"Trying with radius: {radius} km")
    if assign_weather_data_for_radius(radius):
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
    if assign_weather_data_for_radius(refined_radius):
        print(f"All nodes assigned data with radius: {refined_radius} km")
        break

# Phase 3: Final refinement with radii from (x-5) to (x+5) km (step = 1 km)
final_start = max(refined_radius - 5, 0)
final_end = refined_radius + 5
print(f"\nPhase 3: Final refinement around {refined_radius} km")

for final_radius in range(final_start, final_end + 1, 1):
    print(f"Trying with radius: {final_radius} km")
    station_assigned_nodes.clear()  # Reset assigned nodes for final refinement
    if assign_weather_data_for_radius(final_radius):
        print(f"All nodes assigned data with radius: {final_radius} km")
        break

# Calculate final averages and clean up intermediate fields
for node, node_data in G.nodes(data=True):
    for month in MONTHS:
        station_count = node_data.get(f'station_count_{month}', 0)
        if station_count > 0:
            # Calculate averages for each month
            G.nodes[node][f'avg_visibility_{month}'] = node_data[f'visibility_sum_{month}'] / station_count
            G.nodes[node][f'avg_rainfall_{month}'] = node_data[f'rainfall_sum_{month}'] / station_count
            G.nodes[node][f'avg_cloud_cover_{month}'] = node_data[f'cloud_cover_sum_{month}'] / station_count

            # Clean up intermediate sum fields
            del G.nodes[node][f'visibility_sum_{month}']
            del G.nodes[node][f'rainfall_sum_{month}']
            del G.nodes[node][f'cloud_cover_sum_{month}']
            del G.nodes[node][f'station_count_{month}']

# Save the final graph with weather data
ox.save_graphml(G, "india_highways_with_weather_monthly_optimized.graphml")

print("Finished updating the graph with monthly weather data.")