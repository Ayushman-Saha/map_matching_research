import networkx as nx
import osmnx as ox
import random

from geopy.distance import geodesic
from shapely.geometry import LineString
from shapely.geometry.point import Point
from shapely.ops import linemerge
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient(
    "mongodb://sih24:sih24@localhost:27018/sih24?authSource=sih24")  # Adjust connection string as needed
db = client['map_matching']  # Replace 'path_database' with your database name
collection = db['paths_new']  # Replace 'paths_collection' with your collection name

print("Connected to MongoDB!")


def custom_random_walk_with_distance(G, start_node, target_distance_km):
    target_distance_m = target_distance_km * 1000
    total_distance = 0
    current_node = start_node
    walk_path = [current_node]
    path_edges = []
    visited_nodes = {start_node}

    while total_distance < target_distance_m:
        neighbors = list(G.neighbors(current_node))
        unvisited_neighbors = [n for n in neighbors if n not in visited_nodes]

        if not unvisited_neighbors:
            unvisited_neighbors = [n for n in neighbors if n != walk_path[-2]] if len(walk_path) > 1 else neighbors

        if not unvisited_neighbors:
            break

        next_node = random.choice(unvisited_neighbors)
        edge_data = G.get_edge_data(current_node, next_node)[0]


        edge_length = edge_data.get('length', 0)
        to_node = edge_data.get('to')
        from_node = edge_data.get('from')
        remaining_distance = target_distance_m - total_distance

        if total_distance + edge_length > target_distance_m:
            # Decompose the edge into smaller segments
            if 'geometry' in edge_data:
                line_geom = edge_data['geometry']
            else:
                line_geom = LineString([(G.nodes[current_node]['x'], G.nodes[current_node]['y']),
                                        (G.nodes[next_node]['x'], G.nodes[next_node]['y'])])

            # Loop through the edge geometry
            accumulated_distance = 0
            segment_coords = list(line_geom.coords)
            partial_path = [Point(segment_coords[0])]

            for i in range(len(segment_coords) - 1):
                p1 = Point(segment_coords[i])
                p2 = Point(segment_coords[i + 1])
                segment_length = geodesic(segment_coords[i], segment_coords[i + 1]).meters

                if accumulated_distance + segment_length <= remaining_distance:
                    partial_path.append(p2)
                    accumulated_distance += segment_length
                else:
                    walk_path.append(to_node)
                    walk_path.append(from_node)
                    break

            # Add the traversed portion of the edge to path_edges
            extension = LineString(partial_path)
            path_edges.append(extension)
            total_distance += remaining_distance
            break
        else:
            total_distance += edge_length
            if 'geometry' in edge_data:
                path_edges.append(edge_data['geometry'])
            else:
                path_edges.append(LineString([(G.nodes[current_node]['x'], G.nodes[current_node]['y']),
                                              (G.nodes[next_node]['x'], G.nodes[next_node]['y'])]))

        walk_path.append(next_node)
        visited_nodes.add(next_node)
        current_node = next_node

    path_geom = linemerge(path_edges)  # Merge paths (may result in LineString or MultiLineString)

    return walk_path, path_geom, total_distance / 1000.0

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
        'small': (10, 20),  # Small paths: 10-20 km
        'medium': (20, 80),  # Medium paths: 20-80 km
        'large': (80, 250),
        'XL': (250, 1000)  # Large paths: 80-250 km
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
                    walk_path, path_geom, path_length_km = find_path_with_target_distance(G, node, target_distance_km)
                    print(path_length_km)
                    save_path_to_mongodb(node, path_size, walk_path, path_geom, path_length_km)

                except Exception as e:
                    print(f"Failed to generate path for node {node} ({path_size} path {i}): {e}")


def test_custom_path_generation(G):
    """
    Test function for generating custom random walks between two random nodes, save results to MongoDB.
    """
    # Filter out nodes that lie in a roundabout
    non_roundabout_nodes = [
        node for node in G.nodes
        if not any(
            edge_data.get('junction') == 'roundabout'
            for _, _, edge_data in G.edges(node, data=True)
        )
    ]

    # Sample nodes from the filtered list
    if len(non_roundabout_nodes) < 1:
        print("No valid nodes available outside roundabouts.")
        return

    nodes = random.sample(non_roundabout_nodes, 1)
    path_lengths = {'small': (10, 20),  # Small paths: 10-20 km
        'medium': (20, 80),  # Medium paths: 20-80 km
        'large': (80, 250),
        'XL': (250, 1000)}

    collection.delete_many({})
    generate_custom_random_paths_for_test(G, nodes, path_lengths, num_paths=5)
    print("Custom paths generated and saved to MongoDB")


def find_path_with_target_distance(G, start_node, max_distance_km, max_attempts=100):
    """
    Find a path from start_node to a random end node within a target distance range.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph with geographic information
    start_node : node
        Starting node for the path
    max_distance_km : float
        Maximum target distance in kilometers
    max_attempts : int, optional
        Maximum number of attempts to find a suitable path

    Returns:
    --------
    tuple: (path, path_geometry, actual_distance)
        - path: List of nodes in the path
        - path_geometry: Shapely LineString representing the path
        - actual_distance: Actual distance of the path in kilometers
    """
    # Get all nodes in the graph
    all_nodes = list(G.nodes())

    for attempt in range(max_attempts):
        # Choose a random end node
        end_node = random.choice(all_nodes)

        if end_node == start_node:
            continue

        try:
            # Find the shortest path
            path = nx.shortest_path(G, start_node, end_node, weight='length')

            # Calculate total path length
            path_length = sum(
                G[path[i]][path[i + 1]][0].get('length',
                                               geodesic(
                                                   (G.nodes[path[i]]['x'], G.nodes[path[i]]['y']),
                                                   (G.nodes[path[i + 1]]['x'], G.nodes[path[i + 1]]['y'])
                                               ).meters
                                               )
                for i in range(len(path) - 1)
            ) / 1000.0  # Convert to kilometers

            # Check if path length is within acceptable range
            if 0.5 * max_distance_km <= path_length <= 1.5 * max_distance_km:
                # Reconstruct path geometry
                path_edges = []
                for i in range(len(path) - 1):
                    edge_data = G.get_edge_data(path[i], path[i + 1])[0]

                    if 'geometry' in edge_data:
                        path_edges.append(edge_data['geometry'])
                    else:
                        # Create a simple line if no geometry exists
                        path_edges.append(LineString([
                            (G.nodes[path[i]]['x'], G.nodes[path[i]]['y']),
                            (G.nodes[path[i + 1]]['x'], G.nodes[path[i + 1]]['y'])
                        ]))

                # Merge path edges
                from shapely.ops import linemerge
                path_geometry = linemerge(path_edges)

                return path, path_geometry, path_length

        except nx.NetworkXNoPath:
            # No path exists between start and end nodes
            continue

    # If no suitable path found after max attempts
    raise ValueError(f"Could not find a path within {max_distance_km} km after {max_attempts} attempts")



# Example usage
G = ox.load_graphml(f"gujarat_highways.graphml")
G_cleaned = create_undirected_graph_and_remove_nodes(G)
test_custom_path_generation(G)
