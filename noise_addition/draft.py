import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from bson import ObjectId
from pymongo import MongoClient
import random

from time_tracker import TimeTracker
from parameter import Parameter
from point_generator import PointGenerator

# Constants
INTERVAL = 1000
INITIAL_SAMPLING_RATE = {"car": 4 * (INTERVAL / 500), "truck": 6 * (INTERVAL / 500), "motorcycle": 5 * (INTERVAL / 500)}
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


def convert_to_numeric(df, columns):
    """
    Convert specified columns to numeric, handling errors.

    Args:
        df (pd.DataFrame): DataFrame to modify
        columns (list): Columns to convert to numeric
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


class Simulation:
    def __init__(self, nodes, edges, chosen_vehicle_type, chosen_season):
        self.nodes = nodes
        self.edges = edges
        self.vehicle_type = chosen_vehicle_type
        self.season = chosen_season

    def normalize_parameters(self, param_name):
        """
        Normalize a parameter for all nodes based on seasonal data.
        """
        season_months = SEASONS[self.season]
        param_columns = [f"avg_{param_name}_{month}" for month in season_months]
        values = self.nodes[param_columns].mean(axis=1).values
        parameter = Parameter(values)
        return parameter.normalize()

    def generate_Y_values(self, nodes, edge, time_tracker, params, betweenness_centrality_mean,
                          betweenness_centrality_std):
        """
        Generalized function to generate Y-values based on a list of parameters (e.g., rainfall, visibility, traffic).
        """
        gdf_4326_gen = PointGenerator(edge, INTERVAL).generate_intermediate_points()

        # Normalize and initialize parameter values
        normalized_params = {}
        for param_name, values in params.items():
            param = Parameter(values)
            normalized_params[param_name] = param.normalize()

        # Initialize previous values for parameters
        prev_values = {param_name: 0.5 for param_name in params.keys()}

        # Initial sampling rate for the chosen vehicle type
        Yo = INITIAL_SAMPLING_RATE[self.vehicle_type]
        Y_values = []

        for idx, point in gdf_4326_gen.iterrows():
            current_hour = time_tracker.current_hour

            # Get current values for each parameter
            current_values = {
                param_name: gdf_4326_gen.get(f"{param_name}_{self.season}", pd.Series([0.5] * len(gdf_4326_gen)))[idx]
                for param_name in params.keys()
            }

            # Get current traffic factor
            traffic_factor = normalized_params.get("traffic", [0.5] * 24)[current_hour]

            # Calculate impact factors
            Yo_adjusted = Yo
            if all(val is not None for val in prev_values.values()):
                adjustments = []

                for param_name, prev_value in prev_values.items():
                    delta = current_values[param_name] - prev_value

                    # Random coefficients for impact calculation
                    coefficient = np.random.uniform(0.02, 0.10)

                    # Calculate factor adjustments
                    factor = (1 + coefficient * abs(delta)) if delta >= 0 else (1 - coefficient * abs(delta))
                    adjustments.append(factor)

                # Traffic factor adjustment
                delta_T = traffic_factor - prev_values.get("traffic", 0.5)
                traffic_coefficient = 1 / (1 + np.exp(-(float(
                    edge["betweenness_centrality"]) - betweenness_centrality_mean) / betweenness_centrality_std))
                traffic_factor_adjustment = (1 + traffic_coefficient * abs(delta_T)) if delta_T >= 0 else (
                            1 - traffic_coefficient * abs(delta_T))

                # Combine all adjustments
                adjustments.append(traffic_factor_adjustment)

                for factor in adjustments:
                    Yo_adjusted *= factor

                # Calculate final Y value and segment time
                speed_kmph = 60 / Yo_adjusted  # Simplified speed calculation
                segment_time = ((INTERVAL / 1000) / speed_kmph) * 60
                time_tracker.update_time(segment_time)

                Y_values.append(Yo_adjusted)
                Yo = Yo_adjusted

            # Update previous values
            prev_values.update(current_values)

        return [round(value) for value in Y_values]

    def simulate(self):
        """
        Run the main simulation for all edges.
        """
        time_tracker = TimeTracker(initial_hour=0)
        convert_to_numeric(self.edges, ["betweenness_centrality"])
        betweenness_centrality_mean = self.edges["betweenness_centrality"].mean()
        betweenness_centrality_std = self.edges["betweenness_centrality"].std()

        all_points = []

        for index, edge in self.edges.iterrows():
            params = {
                "rainfall": [random.uniform(0, 1) for _ in range(len(self.edges))],
                "visibility": [random.uniform(0, 1) for _ in range(len(self.edges))],
                "traffic": [random.randint(0, 100) for _ in range(24)]  # Example traffic values
            }

            Y_values = self.generate_Y_values(
                nodes=self.nodes,
                edge=edge,
                time_tracker=time_tracker,
                params=params,
                betweenness_centrality_mean=betweenness_centrality_mean,
                betweenness_centrality_std=betweenness_centrality_std
            )

            expanded_points = PointGenerator(edge, INTERVAL).expand_points(Y_values, self.vehicle_type, ANGLE_AND_RADIUS_LIMIT)
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

simulation = Simulation(nodes, edges, chosen_vehicle_type="car", chosen_season="spring")
points = simulation.simulate()

gdf_point = gpd.GeoDataFrame(geometry=points)

# fig1, ax = plt.subplots(figsize=(10, 10))
# edges.plot(ax=ax, color='blue')
# gdf_point.plot(ax=ax, color='red', marker='x', label='Expanded Points')
# plt.legend()
# plt.show()
