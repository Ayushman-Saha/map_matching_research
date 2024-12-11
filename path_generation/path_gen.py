import osmnx as ox
import networkx as nx
import random
import numpy as np
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://sih24:sih24@localhost:27018/sih24?authSource=sih24")
db = client['map_matching']
collection = db['paths_tree']

print("Connected to MongoDB!")

#Load graph
G = ox.load_graphml("../data/india_highways.graphml")
graph = nx.DiGraph(G)

path_lengths = {
    'small': (10000, 20000),
    'medium': (20000, 80000),
    'large': (80000, 250000),
    'XL': (250000, 750000)
}

print("Graph loaded!")

def create_layers(start_node,graph):
    layers = {}
    visited = set()
    queue = [(start_node, 0)]

    while queue:
        node, layer = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        # Add the node to its corresponding layer
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)

        # Add neighbors to the queue for the next layer
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, layer + 1))
    return layers

def generate_walk(graph, start_node, layers, n=50000):
    all_walk = []  # To store all unique walks
    unique_walks = set()  # To track unique walks

    for _ in range(n):
        # Reset the walk for each run
        walk = [start_node]
        current_layer = 0

        while current_layer < len(layers):
            current_node = walk[-1]
            next_layer = current_layer + 1

            # Get valid neighbors in the next layer
            valid_next_nodes = [
                node for node in layers.get(next_layer, [])
                if node in G.neighbors(current_node) and node not in walk
            ]

            # If no valid next nodes, break
            if not valid_next_nodes:
                break

            # Randomly select the next node
            next_node = np.random.choice(valid_next_nodes)
            walk.append(next_node)
            current_layer += 1

        # Convert walk to tuple for hashing
        walk_tuple = tuple(walk)
        if walk_tuple in unique_walks:
            continue

        unique_walks.add(walk_tuple)

        # Calculate the length of the route only for unique walks
        route = ox.routing.route_to_gdf(graph, walk)
        route = route.to_crs(epsg=4326)
        route_length = sum(route['length'])

        # Save the walk and its length
        all_walk.append({
            "route": walk,
            "start_node": start_node,
            "route_length": route_length
        })
    return all_walk


def categorize_walks(all_walk, num_samples = 5):
    categorized_paths = {category: [] for category in path_lengths}

    # Categorize paths
    for walk in all_walk:
        for category, (min_length, max_length) in path_lengths.items():
            if min_length <= walk["route_length"] <= max_length:
                categorized_paths[category].append(walk)

    # Select random samples from each category
    selected_paths = {}
    for category, paths in categorized_paths.items():
        selected_paths[category] = random.sample(paths, min(len(paths), num_samples))
    return selected_paths

def save_to_mongo(category, route_details):
    try:
        path = [int(path_nodes) for path_nodes in route_details["route"]]
        path_doc = {
            "category": category,
            "start_node": route_details["start_node"],
            "route_length": route_details["route_length"],
            "route": path,
        }
        result = collection.insert_one(path_doc)

        print(result.inserted_id)
    except Exception as e:
        print(f"Failed to save {category} for {route_details['start_node']}: {e}")


node_list = list(graph.nodes)[1:3]

for start_node in node_list:
    print(f"Processing for node {start_node}")
    try:
        layers = create_layers(start_node, graph)
        walks = generate_walk(G, start_node, layers)
        categorized_walks = categorize_walks(walks, num_samples=5)

        for category in categorized_walks:
            print(f"Processing for category {category}")
            for route in categorized_walks[category]:
                # print(f"Processing for route {route}")
                save_to_mongo(category, route)
    except Exception as e:
        print(f"Failed to process for {start_node}: {e}")

print("Done!")





