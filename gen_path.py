import osmnx as ox
import random
from shapely.geometry import LineString
import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.ops import transform, linemerge


def custom_random_walk_with_distance(G, start_node, target_distance_km):
    """
    Custom random walk on the graph that stops once the target distance (in kilometers) is reached.
    Distance calculations are done in EPSG:32643, output is in EPSG:4326 for GeoJSON.
    """
    # Transformers: one for distance calculation (EPSG:32643), one for GeoJSON output (EPSG:4326)
    transformer_to_32643 = Transformer.from_crs('EPSG:4326', 'EPSG:32643', always_xy=True)
    transformer_to_4326 = Transformer.from_crs('EPSG:32643', 'EPSG:4326', always_xy=True)

    target_distance_m = target_distance_km * 1000  # Convert target distance to meters
    total_distance = 0
    current_node = start_node
    walk_path = [current_node]  # Store the nodes visited
    path_edges = []  # To store the geometries of the edges

    # Start walking until the target distance is reached
    while total_distance < target_distance_m:
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break  # Stop if no neighbors

        # Randomly choose a neighbor to move to
        next_node = random.choice(neighbors)

        # Get the edge data between the current and next node
        edge_data = G.get_edge_data(current_node, next_node)
        if 'length' in edge_data[0]:
            edge_length = edge_data[0]['length']  # Get edge length in meters

            # Accumulate the total distance
            total_distance += edge_length

            # Add the edge geometry (which is in EPSG:4326) to the path_edges
            if 'geometry' in edge_data[0]:
                path_edges.append(edge_data[0]['geometry'])
            else:
                # If no geometry is provided, create a straight line between nodes (backup option)
                path_edges.append(LineString([(G.nodes[current_node]['x'], G.nodes[current_node]['y']),
                                              (G.nodes[next_node]['x'], G.nodes[next_node]['y'])]))

            # Add the next node to the walk path
            walk_path.append(next_node)

            # Move to the next node
            current_node = next_node
        else:
            # If no length data, skip this edge
            continue

        # Break if we exceed the target distance
        if total_distance >= target_distance_m:
            break

    # Merge the individual edge geometries to form the full path geometry in EPSG:4326
    path_geom = linemerge(path_edges)

    # Transform the path geometry to EPSG:32643 for distance calculations
    path_32643 = transform(lambda x, y: transformer_to_32643.transform(x, y), path_geom)

    # Calculate the total path length in kilometers (using EPSG:32643)
    path_length_km = path_32643.length / 1000.0  # Convert meters to kilometers

    return walk_path, path_geom, path_length_km


# Function to generate random paths for test
def generate_custom_random_paths_for_test(G, nodes, path_lengths, num_paths=2):
    """
    Generate custom random walks with distance constraints for each node and save path details in CSV.
    Distance calculated in EPSG:32643, GeoJSON output in EPSG:4326.
    """
    # Data to store in the CSV
    path_data = []

    for node in nodes:
        for path_size, (min_len, max_len) in path_lengths.items():
            for i in range(num_paths):
                try:
                    # Define the target distance in kilometers
                    target_distance_km = random.uniform(min_len, max_len)

                    # Generate the random walk path
                    walk_path, path_geom, path_length_km = custom_random_walk_with_distance(G, node,
                                                                                                 target_distance_km)

                    # Prepare path data for CSV
                    path_data.append({
                        'start_node': node,
                        'end_node': walk_path[-1],
                        'path_size': path_size,
                        'path_length_km': path_length_km,
                        'path_nodes': walk_path,
                        'path_geometry': path_geom  # Save the path geometry as LineString (EPSG:4326)
                    })

                    # Save path as GeoJSON in EPSG:4326
                    geojson_file = f"geojson_output/custom_path_node_{node}_{path_size}_{i}.geojson"
                    gdf = gpd.GeoDataFrame(geometry=[path_geom], crs='EPSG:4326')
                    gdf.to_file(geojson_file, driver='GeoJSON')

                except Exception as e:
                    print(f"Failed to generate path for node {node} ({path_size} path {i}): {e}")

    # Convert path data to a DataFrame and save as CSV (including geometry as LineString)
    df = pd.DataFrame(path_data)

    # Convert 'path_geometry' to GeoDataFrame for proper handling of LineString
    gdf = gpd.GeoDataFrame(df, geometry='path_geometry', crs='EPSG:4326')

    # Save the paths with geometries to CSV (geometry as well-handled by GeoDataFrame)
    gdf.to_csv('custom_generated_paths_test.csv', index=False)

    return gdf

# Function to generate all paths
def generate_random_paths_all_nodes(G):
    """
    Generate 5 small, 5 medium, and 5 large random walk paths for each node in the graph.
    Paths are saved in CSV and GeoJSON.
    """
    # Path length categories in kilometers
    path_lengths = {
        'small': (10, 20),    # Small paths: 10-20 km
        'medium': (20, 80),   # Medium paths: 20-80 km
        'large': (80, 250)     # Large paths: 80+ km (80-250km for upper limit)
    }

    # Data to store in the CSV
    path_data = []

    # Loop through all nodes in the graph
    for node in G.nodes:
        for path_size, (min_len, max_len) in path_lengths.items():
            for i in range(5):  # Generate 5 paths per node per path size category
                try:
                    # Define the target distance in kilometers
                    target_distance_km = random.uniform(min_len, max_len)

                    # Generate the random walk path
                    walk_path, path_geom, path_length_km = custom_random_walk_with_distance(G, node, target_distance_km)

                    # Prepare path data for CSV
                    path_data.append({
                        'start_node': node,
                        'end_node': walk_path[-1],
                        'path_size': path_size,
                        'path_length_km': path_length_km,
                        'path_nodes': walk_path,
                        'path_geometry': path_geom  # Save the path geometry as WKT (EPSG:4326)
                    })

                    # Save path as GeoJSON in EPSG:4326
                    geojson_file = f'custom_path_node_{node}_{path_size}_{i}.geojson'
                    gdf = gpd.GeoDataFrame(geometry=[path_geom], crs='EPSG:4326')
                    gdf.to_file(geojson_file, driver='GeoJSON')

                except Exception as e:
                    print(f"Failed to generate path for node {node} ({path_size} path {i}): {e}")

    # Convert path data to a DataFrame and save as CSV
    df = pd.DataFrame(path_data)
    df.to_csv('custom_generated_paths_all_nodes.csv', index=False)

    return df

# Function to test path generation for 2 nodes
def test_custom_path_generation(G):
    """
    Test function for generating custom random walks between two random nodes.
    """
    # Randomly select two nodes for the test
    nodes = random.sample(list(G.nodes), 1)

    # Define the path lengths for testing
    path_lengths = {'small': (10, 15), 'medium': (20, 35)}

    # Generate paths between these two nodes
    df_paths = generate_custom_random_paths_for_test(G, nodes, path_lengths, num_paths=1)

    print("Custom paths generated and saved to 'custom_generated_paths_test.csv'")
    return df_paths

def create_undirected_graph_and_remove_nodes(G):
    G = ox.convert.to_undirected(G)
    degree_1_nodes = [node for node, degree in dict(G.degree()).items() if degree == 1]

    # Iterate over degree-1 nodes and remove them
    for node in degree_1_nodes:
        # Get the neighbors of this node
        neighbors = list(G.neighbors(node))

        if len(neighbors) == 1:
            # There should be only one neighbor since degree == 1
            neighbor = neighbors[0]

            # Get the edge geometry between the node and its neighbor
            edge_data = G.get_edge_data(node, neighbor)

            if len(edge_data) > 0:
                # Extract the geometry (LineString) of the edge
                linestrings = [data['geometry'] for key, data in edge_data.items() if 'geometry' in data]

                # Remove the node
                G.remove_node(node)

                # Combine the LineStrings if there are multiple geometries
                if len(linestrings) > 1:
                    merged_line = linemerge(linestrings)  # Merge multiple LineStrings
                else:
                    merged_line = linestrings[0]

                # Add the new merged edge between the neighbor and the node's neighbor
                G.add_edge(neighbor, neighbor, geometry=merged_line)

    # Step 3: Visualize the modified graph (optional)
    return G



G = ox.load_graphml(f"india_highways.graphml")
G_cleaned = create_undirected_graph_and_remove_nodes(G)

df_paths = generate_random_paths_all_nodes(G_cleaned)
