from math import floor

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry import Point, LineString
from geopy.distance import geodesic
from pymongo import MongoClient
from bson import ObjectId

# Configuration and Constants
INTERVAL = 1000
SEASONS = {
    'winter': ['November', 'December', 'January', 'February'],
    'spring': ['March', 'April', 'May'],
    'summer': ['June', 'July', 'August'],
    'autumn': ['September', 'October', 'November']
}

INITIAL_SAMPLING_RATE = {
    "car": 2 * (INTERVAL / 500),
    "truck": 3 * (INTERVAL / 500),
    "motorcycle": 2.5 * (INTERVAL / 500)
}

ANGLE_LIMIT = {
    "car": 8,
    "truck": 5,
    "motorcycle": 10
}

TRAFFIC_VALUES = {
    0: 380, 1: 250, 2: 180, 3: 150, 4: 150, 5: 180, 6: 250,
    7: 500, 8: 750, 9: 1000, 10: 1200, 11: 1150, 12: 1100,
    13: 1000, 14: 920, 15: 900, 16: 900, 17: 1000, 18: 1050,
    19: 950, 20: 800, 21: 650, 22: 550, 23: 500
}


def sigmoid_normalization(value, mean, std):
    """
    Apply sigmoid normalization to a value.

    Args:
        value (float): Input value to normalize
        mean (float): Mean of the distribution
        std (float): Standard deviation of the distribution

    Returns:
        float: Normalized value between 0 and 1
    """
    return 1 / (1 + np.exp(-(value - mean) / std))


def convert_to_numeric(df, columns):
    """
    Convert specified columns to numeric, handling errors.

    Args:
        df (pd.DataFrame): DataFrame to modify
        columns (list): Columns to convert to numeric
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


def normalize_traffic_values():
    """
    Normalize traffic values using sigmoid function.

    Returns:
        dict: Normalized traffic values
    """
    traffic_values = TRAFFIC_VALUES
    traffic_mean = np.mean(list(traffic_values.values()))
    traffic_std = np.std(list(traffic_values.values()))

    return {
        hour: sigmoid_normalization(value, traffic_mean, traffic_std)
        for hour, value in traffic_values.items()
    }

def visualize_traffic_values(normalized_traffic):
    hours = list(normalized_traffic.keys())
    normalized_values = list(normalized_traffic.values())

    # Plotting the normalized traffic values
    plt.figure(figsize=(10, 6))
    plt.plot(hours, normalized_values, marker='o', color='blue', label='Normalized Traffic')

    # Adding labels and title
    plt.title('Normalized Traffic Values by Hour of the Day', fontsize=16)
    plt.xlabel('Hour of the Day', fontsize=14)
    plt.ylabel('Normalized Traffic (Sigmoid)', fontsize=14)
    plt.xticks(range(0, 24))  # Ensuring all hours are marked on x-axis
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    # Display the graph
    plt.show()


def calculate_speed_equation():
    """
    Calculate speed equation based on vehicle types.

    Returns:
        tuple: Slope and intercept of the speed equation
    """
    speed_car = 60
    speed_truck = 40
    points_car = INITIAL_SAMPLING_RATE['car']
    points_truck = INITIAL_SAMPLING_RATE['truck']

    m = (speed_truck - speed_car) / (points_truck - points_car)
    c = speed_car - m * points_car
    return m, c


def calculate_speed(Y):
    """
    Calculate speed based on sampling rate.

    Args:
        Y (float): Sampling rate

    Returns:
        float: Calculated speed in km/h
    """
    m, c = calculate_speed_equation()
    return m * Y + c

def normalize_weather(nodes):
    # Ensure all `avg_rainfall_<month>` and `avg_visibility_<month>` columns are numeric
    rainfall_columns = [f"avg_rainfall_{month}" for month in
                        SEASONS['winter'] + SEASONS['spring'] + SEASONS['summer'] + SEASONS['autumn']]
    visibility_columns = [f"avg_visibility_{month}" for month in
                          SEASONS['winter'] + SEASONS['spring'] + SEASONS['summer'] + SEASONS['autumn']]
    convert_to_numeric(nodes, rainfall_columns + visibility_columns)

    # Create normalized columns for each season
    for season_name, months in SEASONS.items():
        # Combine data for the given season
        rainfall_data = nodes[[f"avg_rainfall_{month}" for month in months]].mean(axis=1)
        visibility_data = nodes[[f"avg_visibility_{month}" for month in months]].mean(axis=1)

        # Compute seasonal statistics
        rainfall_mean = rainfall_data.mean()
        rainfall_std = rainfall_data.std()
        visibility_mean = visibility_data.mean()
        visibility_std = visibility_data.std()

        # Normalize rainfall and visibility using sigmoid function
        nodes[f"normalized_rainfall_{season_name}"] = sigmoid_normalization(rainfall_data, rainfall_mean, rainfall_std)
        nodes[f"normalized_visibility_{season_name}"] = sigmoid_normalization(visibility_data, visibility_mean,
                                                                              visibility_std)
        nodes[f"rainfall_mean_{season_name}"] = rainfall_mean
        nodes[f"rainfall_std_{season_name}"] = sigmoid_normalization(rainfall_std, rainfall_mean, rainfall_std)
        nodes[f"visibility_mean_{season_name}"] = visibility_mean
        nodes[f"visibility_std_{season_name}"] = sigmoid_normalization(visibility_std, visibility_mean, visibility_std)

    # List of columns to drop based on the pattern
    columns_to_drop = nodes.filter(regex='avg_visibility_|avg_rainfall_|avg_cloud_cover_').columns

    # Drop the columns
    nodes = nodes.drop(columns=columns_to_drop)

    return nodes


def interpolate_points(nodes, edge,
                       chosen_vehicle_type, chosen_season, current_hour,
                       interval, betweenness_centrality_mean,
                       betweenness_centrality_std):
    """
    Interpolate and expand points along a route with comprehensive environmental and traffic factors.

    Args:
        nodes (gpd.GeoDataFrame): Nodes dataframe with geographical and environmental information
        edge (pd.Series): Edge information
        chosen_vehicle_type (str): Type of vehicle
        chosen_season (str): Current season
        current_hour (int): Current hour of the day
        interval (int): Distance interval for point generation
        betweenness_centrality_mean (float): Mean betweenness centrality
        betweenness_centrality_std (float): Standard deviation of betweenness centrality

    Returns:
        tuple: List of Y values and modified GeoDataFrame with point characteristics
    """
    # Extract edge details
    to_node = edge['to']
    from_node = edge['from']
    total_length = edge.length

    #Generate the points
    gdf_4326_gen, gdf_edge = generate_intermediate_nodes(edge)

    #Traffic values
    normalized_traffic = normalize_traffic_values()

    # Rainfall and visibility calculations
    to_rainfall_mean = nodes[f"normalized_rainfall_{chosen_season}"].loc[to_node]
    from_rainfall_mean = nodes[f"normalized_rainfall_{chosen_season}"].loc[from_node]
    to_rainfall_std = nodes[f"rainfall_std_{chosen_season}"].loc[to_node]
    from_rainfall_std = nodes[f"rainfall_std_{chosen_season}"].loc[from_node]

    to_visibility_mean = nodes[f"normalized_visibility_{chosen_season}"].loc[to_node]
    from_visibility_mean = nodes[f"normalized_visibility_{chosen_season}"].loc[from_node]
    to_visibility_std = nodes[f"visibility_std_{chosen_season}"].loc[to_node]
    from_visibility_std = nodes[f"visibility_std_{chosen_season}"].loc[from_node]

    # Assign environmental characteristics to points
    for idx, point in gdf_4326_gen.iterrows():
        # Interpolate rainfall characteristics
        point_rainfall_mean = (
                ((interval * idx) / total_length) * to_rainfall_mean +
                ((interval * (len(gdf_4326_gen) - idx - 1)) / total_length) * from_rainfall_mean
        )
        point_rainfall_std = (to_rainfall_std + from_rainfall_std) / 2
        point_rainfall = np.random.normal(point_rainfall_mean, 0.5 * point_rainfall_std)

        # Interpolate visibility characteristics
        point_visibility_mean = (
                ((interval * idx) / total_length) * to_visibility_mean +
                ((interval * (len(gdf_4326_gen) - idx - 1)) / total_length) * from_visibility_mean
        )
        point_visibility_std = (to_visibility_std + from_visibility_std) / 2
        point_visibility = np.random.normal(point_visibility_mean, 0.5 * point_visibility_std)

        # Store characteristics in GeoDataFrame
        gdf_4326_gen.at[idx, f'rainfall_{chosen_season}'] = point_rainfall
        gdf_4326_gen.at[idx, f'visibility_{chosen_season}'] = point_visibility

    # Initialize previous values
    prev_R = 0.5
    prev_V = 0.5
    prev_traffic_factor = 0.5

    # Initial sampling rate for the chosen vehicle type
    Yo = INITIAL_SAMPLING_RATE[chosen_vehicle_type]
    Y_values = []

    # Iterate through points and calculate impact factors
    for idx, point in gdf_4326_gen.iterrows():
        # Get current rainfall and visibility
        R = gdf_4326_gen[f"rainfall_{chosen_season}"][idx]
        V = gdf_4326_gen[f"visibility_{chosen_season}"][idx]

        # Get current traffic factor
        traffic_factor = normalized_traffic[current_hour]

        # Initialize Yo_adjusted
        Yo_adjusted = Yo

        # Calculate impact factors if previous values are available
        if all(val is not None for val in [prev_R, prev_V, prev_traffic_factor]):
            # Compute deltas
            delta_R = R - prev_R
            delta_V = V - prev_V

            # Random coefficients for impact calculation
            a = np.random.uniform(0.02, 0.10)
            b = np.random.uniform(0.04, 0.09)
            c = 1 + sigmoid_normalization(
                float(edge["betweenness_centrality"]),
                betweenness_centrality_mean,
                betweenness_centrality_std
            )

            # Calculate factor adjustments
            factor_R = (1 + a * abs(delta_R)) if delta_R >= 0 else (1 - a * abs(delta_R))
            factor_V = (1 + b * abs(delta_V)) if delta_V >= 0 else (1 - b * abs(delta_V))

            # Compute traffic factor delta
            delta_T = traffic_factor - prev_traffic_factor
            factor_T = (1 + c * abs(delta_T)) if delta_T >= 0 else (1 - c * abs(delta_T))

            # Calculate final sampling rate
            Y = Yo_adjusted * factor_R * factor_V * factor_T

            # Calculate speed and segment time
            speed_kmph = calculate_speed(Y)
            segment_time = (interval / 1000) / speed_kmph

            # Store debug information if needed
            # You might want to add logging or print statements here

            Y_values.append(Y)

            # Update Yo for next iteration
            Yo = Y

        # Update previous values
        prev_R, prev_V, prev_traffic_factor = R, V, traffic_factor

    # Round Y values
    Y_values = [round(value) for value in Y_values]

    return Y_values, gdf_4326_gen, gdf_edge



def generate_intermediate_nodes(edge):
    gdf_edge = gpd.GeoDataFrame(geometry=edge[['geometry']], crs="EPSG:4326")
    # edge_linestring = LineString(edge.geometry)
    # points = [Point(coord) for coord in edge_linestring.coords]

    # gdf_points = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    gdf_line_proj = gdf_edge.to_crs(epsg=32618)
    edge_linestring_proj = gdf_line_proj.geometry.iloc[0]

    edge_length = edge.length

    #Generate point at interval metres
    points_projected = [edge_linestring_proj.interpolate(distance) for distance in range(0, int(edge_length) + 1, INTERVAL)]

    # Reproject points back to EPSG:4326
    transformer = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
    points_4326 = [Point(*transformer.transform(point.x, point.y)) for point in points_projected]

    gdf_4326_gen = gpd.GeoDataFrame(geometry=points_4326, crs="EPSG:4326")

    return gdf_4326_gen, gdf_edge

def calculate_new_point(origin, distance_m, bearing_deg):
    origin_lat, origin_lon = origin.y, origin.x
    new_point = geodesic(meters=distance_m).destination((origin_lat, origin_lon), bearing_deg)
    return Point(new_point[1], new_point[0])  # Return as Shapely Point (lon, lat)

def expand_points(Y_values, gdf_4326_gen, chosen_vehicle_type):
    # Initialize list to store all new points
    expanded_points = []

    # Initialize base distance and adjustment parameters
    base_distance = INTERVAL / (floor(np.mean(Y_values)))  # Initial average distance in meters
    delta_limit = 0.01 * base_distance  # Maximum random variation in meters

    print(base_distance)

    # Iterate over points and add intermediate points
    for idx in range(len(gdf_4326_gen) - 1):
        start_point = gdf_4326_gen.geometry.iloc[idx]
        end_point = gdf_4326_gen.geometry.iloc[idx + 1]
        current_point = start_point  # Start interpolation from the current point

        # Number of points to interpolate
        num_points = Y_values[idx] if idx < len(Y_values) else 0
        if num_points > 0:
            # Calculate bearing between start and end points
            bearing = np.degrees(np.arctan2(end_point.x - start_point.x, end_point.y - start_point.y)) % 360

            # Smoothly adjust distances for realism
            distances = []
            for _ in range(num_points):
                delta = np.random.uniform(0.5 * delta_limit, delta_limit)  # Small random adjustment
                base_distance = max(10, base_distance + delta)  # Update base distance, ensure minimum 10m
                distances.append(base_distance)

            # Generate intermediate points
            for distance_m in distances:
                random_bearing = bearing + np.random.uniform(-ANGLE_LIMIT[chosen_vehicle_type],
                                                             ANGLE_LIMIT[chosen_vehicle_type])
                new_point = calculate_new_point(current_point, distance_m, random_bearing)
                expanded_points.append(new_point)
                current_point = new_point  # Update the current point to the newly generated one

        # Add the original start point (or current point) to the list
        expanded_points.append(start_point)

    # Append the final point
    expanded_points.append(gdf_4326_gen.geometry.iloc[-1])

    # Create new GeoDataFrame with expanded points
    gdf_expanded = gpd.GeoDataFrame(geometry=expanded_points, crs=gdf_4326_gen.crs)
    return gdf_expanded


def main():
    """
    Main execution function for path simulation and expansion.
    """
    # Load graph and preprocess
    graph = ox.io.load_graphml("../data/merged_graph.graphml")
    graph = ox.convert.to_undirected(graph)
    print("Graph loaded!")

    #Get the edges and nodes
    nodes, edges = ox.graph_to_gdfs(graph)

    #Return nodes with the normalised weather
    nodes = normalize_weather(nodes)

    # Convert betweenness centrality to numeric
    convert_to_numeric(edges, ["betweenness_centrality"])

    #Calculate mean and std of betweenness centrality
    betweenness_centrality_mean = edges['betweenness_centrality'].mean()
    betweenness_centrality_std = edges['betweenness_centrality'].std()

    # Connect to MongoDB and retrieve path
    mongo_string = "mongodb://sih24:sih24@localhost:27018/sih24?authSource=sih24"
    client = MongoClient(mongo_string)
    collection = client['map_matching']['paths_new_algorithm']
    paths = collection.find_one({"_id": ObjectId("673e1f5e0f4ddc41d44150aa")})
    print("Connected to MongoDB!")

    # Select specific parameters
    chosen_vehicle_type = "motorcycle"
    chosen_season = "spring"
    current_hour = 9

    #Extracting the nodes and egdes
    path_nodes = set(paths['path_nodes'])
    s = graph.subgraph(path_nodes)
    sub_nodes, sub_edges = ox.graph_to_gdfs(s)

    # #Visualising the path
    # fig, ax = plt.subplots()
    # sub_edges.plot(ax=ax, color='blue')
    # sub_nodes.plot(ax=ax, color='red')
    # plt.title("Graph Visualization with Original Geometries", fontsize=20)
    # plt.show()

    # edge = sub_edges.iloc[9]
    #
    # #Generating the intermediate points and calculating the Y values
    # Y_values, gdf_4326_gen, gdf_edge = interpolate_points(nodes, edge, chosen_vehicle_type, chosen_season,current_hour,INTERVAL,betweenness_centrality_mean, betweenness_centrality_std)
    #
    # #Expanding the noisy points
    # gdf_expanded = expand_points(Y_values, gdf_4326_gen, chosen_vehicle_type)

    # Create an empty list to store the expanded GeoDataFrames
    expanded_gdfs = []

    # Loop through each edge in sub_edges
    for index, edge in sub_edges.iterrows():
        # Generating the intermediate points and calculating the Y values
        Y_values, gdf_4326_gen, gdf_edge = interpolate_points(
            nodes, edge, chosen_vehicle_type, chosen_season, current_hour, INTERVAL,
            betweenness_centrality_mean, betweenness_centrality_std
        )

        # Expanding the noisy points
        gdf_expanded = expand_points(Y_values, gdf_4326_gen, chosen_vehicle_type)

        # Append the expanded GeoDataFrame to the list
        expanded_gdfs.append(gdf_expanded)

    # Concatenate all the expanded GeoDataFrames into one final GeoDataFrame
    final_gdf = pd.concat(expanded_gdfs, ignore_index=True)

    print(final_gdf)
    final_gdf.to_file("final_expanded_points.geojson", driver="GeoJSON")

    # #Visualtion of the entire plot
    # fig, ax = plt.subplots(figsize=(10, 10))
    # gdf_4326_gen.plot(ax=ax, color='blue', marker='o', label='Original Points')
    # gdf_edge.plot(ax=ax, color='blue')
    # gdf_expanded.plot(ax=ax, color='red', marker='x', label='Expanded Points')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()