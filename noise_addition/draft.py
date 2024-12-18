import numpy as np
import geopandas as gpd
import osmnx as ox
from bson import ObjectId
from matplotlib import pyplot as plt
from pymongo import MongoClient

from time_tracker import TimeTracker
from parameter import ParameterProcessor, sigmoid_normalization
from point_generator import PointGenerator

# Constants
INTERVAL = 1000
INITIAL_SAMPLING_RATE = {"car": 2 * (INTERVAL / 500), "truck": 3 * (INTERVAL / 500), "motorcycle": 2.5 * (INTERVAL / 500)}
ANGLE_AND_RADIUS_LIMIT = {"car": (15, 75), "truck": (10, 45), "motorcycle": (20, 100)}
SEASONS = {
    'winter': ['November', 'December', 'January', 'February'],
    'spring': ['March', 'April', 'May'],
    'summer': ['June', 'July', 'August'],
    'autumn': ['September', 'October', 'November']
}
TRAFFIC_VALUES = {
    0: 380, 1: 250, 2: 180, 3: 150, 4: 150, 5: 180, 6: 250,
    7: 500, 8: 750, 9: 1000, 10: 1200, 11: 1150, 12: 1100,
    13: 1000, 14: 920, 15: 900, 16: 900, 17: 1000, 18: 1050,
    19: 950, 20: 800, 21: 650, 22: 550, 23: 500
}


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
        time_tracker = TimeTracker(initial_hour=0)

        #Add all the processor and process the raw data
        rainfall_processor = ParameterProcessor(self.nodes,"rainfall", groups=SEASONS, type="grouped")
        rainfall_processor.process()

        visibility_processor = ParameterProcessor(self.nodes,"visibility", groups=SEASONS, type="grouped")
        visibility_processor.process()

        betweenness_centrality_processor = ParameterProcessor(self.edges,"betweenness_centrality", type="ungrouped")
        betweenness_centrality_processor.process()

        traffic_values = TRAFFIC_VALUES
        traffic_mean = np.mean(list(traffic_values.values()))
        traffic_std = np.std(list(traffic_values.values()))

        normalized_traffic = {
            hour: sigmoid_normalization(value, traffic_mean, traffic_std)
            for hour, value in traffic_values.items()
        }

        all_points = []

        for index, edge in self.edges.iterrows():
            #Generate intermediate points
            generator = PointGenerator(edge, INTERVAL, INITIAL_SAMPLING_RATE, self.vehicle_type)
            gdf_4326_gen = generator.generate_intermediate_points()

            params = {
                'grouped': [
                    {
                        "name": 'rainfall',
                        "grouping_key": self.season,
                        "constant" : (0.02, 0.10),
                        "average_effect" : True
                    },
                    {
                        "name": 'visibility',
                        "grouping_key": self.season,
                        "constant" : (0.04, 0.09),
                        "average_effect" : True
                    }
                ],
                'global': [{"name":'traffic', "constant": normalized_traffic, "average_effect": False}],
            }

            gdf_4326_gen = generator.assign_characteristics(gdf_4326_gen, self.nodes, edge, params)
            Y_values = generator.generate_Y_values(gdf_4326_gen, params, time_tracker)
            expanded_points = generator.expand_points(Y_values, self.vehicle_type, ANGLE_AND_RADIUS_LIMIT)

            for point in expanded_points['geometry']:
                all_points.append(point)
            # print(f"Edge {index} expanded points:", expanded_points)
        return all_points

#Loading data

# Connect to MongoDB and retrieve path
mongo_string = "mongodb://sih24:sih24@localhost:27018/sih24?authSource=sih24"
client = MongoClient(mongo_string)
collection = client['map_matching']['paths_tree']
paths = collection.find_one({"_id": ObjectId("675ea5f7ebb710f20b8fd98d")})
route = paths["route"]
print("Connected to MongoDB!")

graph = ox.load_graphml("../data/merged_graph_gujarat.graphml")
nodes, _ = ox.convert.graph_to_gdfs(graph)
edges = ox.routing.route_to_gdf(graph, route)
print("Graph Loaded!")

simulation = Simulation(nodes, edges, chosen_vehicle_type="car", chosen_season="spring", chosen_time=0)
points = simulation.simulate()

gdf_point = gpd.GeoDataFrame(geometry=points)

fig1, ax = plt.subplots(figsize=(10, 10))
edges.plot(ax=ax, color='blue')
gdf_point.plot(ax=ax, color='red', marker='x', label='Expanded Points')
plt.legend()
plt.show()
