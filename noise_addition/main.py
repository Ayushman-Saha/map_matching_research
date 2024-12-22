import osmnx as ox
from pymongo import MongoClient
import random
import geopandas as gpd
import matplotlib.pyplot as plt

from time_tracker import TimeTracker
from parameter import ParameterProcessor
from point_generator import PointGenerator

# Constants
INTERVAL = 1000
INITIAL_SAMPLING_RATE = {"car": 2 * (INTERVAL / 500), "truck": 3 * (INTERVAL / 500),
                         "motorcycle": 2.5 * (INTERVAL / 500)}
ANGLE_AND_RADIUS_LIMIT = {"car": (15, 75), "truck": (10, 45), "motorcycle": (20, 100)}
SEASONS = {
    'winter': ['November', 'December', 'January', 'February'],
    'spring': ['March', 'April', 'May'],
    'summer': ['June', 'July', 'August'],
    'autumn': ['September', 'October', 'November']
}

# TRAFFIC_VALUES = {
#     0: 380, 1: 250, 2: 180, 3: 150, 4: 150, 5: 180, 6: 250,
#     7: 500, 8: 750, 9: 1000, 10: 1200, 11: 1150, 12: 1100,
#     13: 1000, 14: 920, 15: 900, 16: 900, 17: 1000, 18: 1050,
#     19: 950, 20: 800, 21: 650, 22: 550, 23: 500
# }


class Simulation:
    def __init__(self, nodes, edges, chosen_vehicle_type, chosen_season, chosen_time):
        self.nodes = nodes
        self.edges = edges
        self.vehicle_type = chosen_vehicle_type
        self.season = chosen_season
        self.chosen_time = chosen_time

    def simulate(self):
        """
        Run the main simulation for all edges.
        """
        time_tracker = TimeTracker(initial_hour=self.chosen_time)

        # Add processors and process raw data
        rainfall_processor = ParameterProcessor(self.nodes, "rainfall", groups=SEASONS, type="grouped")
        rainfall_processor.process()

        visibility_processor = ParameterProcessor(self.nodes, "visibility", groups=SEASONS, type="grouped")
        visibility_processor.process()

        betweenness_centrality_processor = ParameterProcessor(self.edges, "betweenness_centrality", type="ungrouped")
        betweenness_centrality_processor.process()

        traffic_processor = ParameterProcessor(self.edges, "traffic", type="ungrouped", variants=[a for a in range(0, 24)])
        traffic_processor.process()

        params = {
            'grouped': [
                {
                    "name": 'rainfall',
                    "grouping_key": self.season,
                    "constant": (0.02, 0.10),
                    "average_effect": True
                },
                {
                    "name": 'visibility',
                    "grouping_key": self.season,
                    "constant": (0.04, 0.09),
                    "average_effect": True
                },

            ],
            'global': [
                {
                    "name": 'betweenness_centrality',
                    "constant": (0.5, 0.7),
                    "variant": None,
                    "average_effect": False
                },
                {
                    "name": 'traffic',
                    "constant": (0.5, 0.7),
                    "variant": self.chosen_time,
                    "average_effect": False
                },
            ],
        }

        all_points = []

        for index, edge in self.edges.iterrows():
            # Generate intermediate points
            generator = PointGenerator(edge, INTERVAL, INITIAL_SAMPLING_RATE, self.vehicle_type)
            gdf_4326_gen = generator.generate_intermediate_points()

            gdf_4326_gen = generator.assign_characteristics(gdf_4326_gen, self.nodes, edge, params)
            Y_values, speed_values = generator.generate_Y_values(gdf_4326_gen, params, time_tracker)
            expanded_points = generator.expand_points(Y_values, self.vehicle_type, ANGLE_AND_RADIUS_LIMIT)

            for point in expanded_points['geometry']:
                all_points.append(point)

        return all_points, time_tracker.current_hour, speed_values


# MongoDB connection
mongo_string = "mongodb://sih24:sih24@localhost:27018/sih24?authSource=sih24"
client = MongoClient(mongo_string)
collection = client['map_matching']['paths_tree']
print("Connected to MongoDB!")

# Load the road network graph
graph = ox.load_graphml("../data/merged_graph.graphml")
nodes, _ = ox.convert.graph_to_gdfs(graph)
print("Graph Loaded!")

# Process each document in the collection
for index, doc in enumerate(collection.find()):

    try:
        route = doc["route"]
        doc_id = doc["_id"]

        # Randomly choose vehicle type, season, and chosen time
        chosen_vehicle_type = random.choice(["car", "truck", "motorcycle"])
        chosen_season = random.choice(["winter", "spring", "summer", "autumn"])
        chosen_time = random.randint(0, 23)

        # Convert route to GeoDataFrame
        edges = ox.routing.route_to_gdf(graph, route)

        # Run the simulation
        simulation = Simulation(nodes, edges, chosen_vehicle_type, chosen_season, chosen_time)
        points, end_time, speed_values = simulation.simulate()

        # Prepare trajectory data
        trajectory = {
            "vehicle_type": chosen_vehicle_type,
            "season": chosen_season,
            "chosen_time": chosen_time,
            "end_time": end_time,
            "coordinates": [point.coords[0] for point in points],
            "speed": speed_values
        }

        # Update the document with the new trajectory field
        # collection.update_one({"_id": doc_id}, {"$set": {"trajectory": trajectory}})
        print(trajectory)
        print(f"Processed document {index} with _id: {doc_id}")

        # #Visualtion of the entire plot (Comment in production)
        # final_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
        #
        # fig, ax = plt.subplots(figsize=(10, 10))
        # edges.plot(ax=ax, color='blue')
        # final_gdf.plot(ax=ax, color='red', marker='x', label='Expanded Points')
        # plt.legend()
        # plt.show()
    except Exception as exception:
        print(f"Failed to process document {index} : {exception}")

print("All documents processed!")
