import osmnx as ox
import random
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, transform
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://sih24:sih24@localhost:27017/sih24?authSource=sih24")  # Adjust connection string as needed
db = client['map_matching']  # Replace 'path_database' with your database name
collection = db['paths']  # Replace 'paths_collection' with your collection name

print("Connected to MongoDB!")

def custom_random_walk_with_distance(G, start_node, target_distance_km):
    """
    Custom random walk on the graph that stops once the target distance (in kilometers) is reached.
    Distance calculations are done in EPSG:32643, output is in EPSG:4326 for MongoDB storage.
    """
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

        next_node = random.choice(neighbors)
        edge_data = G.get_edge_data(current_node, next_node)

        if 'length' in edge_data[0]:
            edge_length = edge_data[0]['length']  # Get edge length in meters
            total_distance += edge_length

            if 'geometry' in edge_data[0]:
                path_edges.append(edge_data[0]['geometry'])
            else:
                path_edges.append(LineString([(G.nodes[current_node]['x'], G.nodes[current_node]['y']),
                                              (G.nodes[next_node]['x'], G.nodes[next_node]['y'])]))

            walk_path.append(next_node)
            current_node = next_node
        else:
            continue

        if total_distance >= target_distance_m:
            break

    path_geom = linemerge(path_edges)  # Merge paths (may result in LineString or MultiLineString)

    # Transform the geometry to EPSG:32643 for distance calculation
    path_32643 = transform(lambda x, y: transformer_to_32643.transform(x, y), path_geom)
    path_length_km = path_32643.length / 1000.0  # Convert meters to kilometers

    return walk_path, path_geom, path_length_km

def save_path_to_mongodb(node, path_size, walk_path, path_geom, path_length_km):
    """
    Save path details to MongoDB, handling both LineString and MultiLineString geometries.
    """
    try:
        # Convert geometry to GeoJSON format
        if path_geom.geom_type == 'LineString':
            geojson_geometry = {
                "type": "LineString",
                "coordinates": list(path_geom.coords)
            }
        elif path_geom.geom_type == 'MultiLineString':
            geojson_geometry = {
                "type": "MultiLineString",
                "coordinates": [list(line.coords) for line in path_geom.geoms]
            }
        else:
            raise ValueError(f"Unsupported geometry type: {path_geom.geom_type}")

        # Prepare the document to be inserted
        path_document = {
            'start_node': node,
            'end_node': walk_path[-1],
            'path_size': path_size,
            'path_length_km': path_length_km,
            'path_nodes': walk_path,
            'path_geometry': geojson_geometry
        }

        # Insert the document into the MongoDB collection
        collection.insert_one(path_document)

    except Exception as e:
        print(f"Error saving path for node {node}: {e}")

def generate_random_paths_all_nodes(G):
    """
    Generate 5 small, 5 medium, and 5 large random walk paths for each node in the graph, save to MongoDB.
    """
    path_lengths = {
        'small': (10, 20),    # Small paths: 10-20 km
        'medium': (20, 80),   # Medium paths: 20-80 km
        'large': (80, 250)    # Large paths: 80-250 km
    }

    for node in G.nodes:
        print(node)
        for path_size, (min_len, max_len) in path_lengths.items():
            for i in range(5):
                try:
                    target_distance_km = random.uniform(min_len, max_len)
                    walk_path, path_geom, path_length_km = custom_random_walk_with_distance(G, node, target_distance_km)
                    save_path_to_mongodb(node, path_size, walk_path, path_geom, path_length_km)

                except Exception as e:
                    print(f"Failed to generate path for node {node} ({path_size} path {i}): {e}")

def create_undirected_graph_and_remove_nodes(G):
    G = ox.convert.to_undirected(G)
    degree_1_nodes = [node for node, degree in dict(G.degree()).items() if degree == 1]

    for node in degree_1_nodes:
        neighbors = list(G.neighbors(node))

        if len(neighbors) == 1:
            neighbor = neighbors[0]
            edge_data = G.get_edge_data(node, neighbor)

            if len(edge_data) > 0:
                linestrings = [data['geometry'] for key, data in edge_data.items() if 'geometry' in data]
                G.remove_node(node)

                if len(linestrings) > 1:
                    merged_line = linemerge(linestrings)
                else:
                    merged_line = linestrings[0]

                G.add_edge(neighbor, neighbor, geometry=merged_line)

    return G

def generate_custom_random_paths_for_test(G, nodes, path_lengths, num_paths=2):
    """
    Generate custom random walks with distance constraints for each node and save path details to MongoDB.
    """
    for node in nodes:
        for path_size, (min_len, max_len) in path_lengths.items():
            for i in range(num_paths):
                try:
                    target_distance_km = random.uniform(min_len, max_len)
                    walk_path, path_geom, path_length_km = custom_random_walk_with_distance(G, node, target_distance_km)
                    save_path_to_mongodb(node, path_size, walk_path, path_geom, path_length_km)

                except Exception as e:
                    print(f"Failed to generate path for node {node} ({path_size} path {i}): {e}")

def test_custom_path_generation(G):
    """
    Test function for generating custom random walks between two random nodes, save results to MongoDB.
    """
    nodes = random.sample(list(G.nodes), 1)
    path_lengths = {'small': (10, 15), 'medium': (20, 35)}

    generate_custom_random_paths_for_test(G, nodes, path_lengths, num_paths=1)
    print("Custom paths generated and saved to MongoDB")

# Example usage
G = ox.load_graphml(f"india_highways.graphml")
G_cleaned = create_undirected_graph_and_remove_nodes(G)
generate_random_paths_all_nodes(G_cleaned)