import random
from math import floor

import numpy as np
from geopy.distance import distance
from pyproj import Transformer, CRS
from shapely.geometry.point import Point
import geopandas as gpd



class PointGenerator:
    def __init__(self, edge, interval, initial_sampling_rate, vehicle_type):
        self.edge = edge
        self.interval = interval
        self.initial_sampling_rate = initial_sampling_rate
        self.vehicle_type = vehicle_type

    def calculate_speed_equation(self):
        """
        Calculate speed equation based on vehicle types.

        Returns:
            tuple: Slope and intercept of the speed equation
        """
        speed_car = 60
        speed_truck = 40
        points_car = self.initial_sampling_rate['car']
        points_truck = self.initial_sampling_rate['truck']

        m = (speed_truck - speed_car) / (points_truck - points_car)
        c = speed_car - m * points_car
        return m, c

    def calculate_speed(self, Y):
        """
        Calculate speed based on sampling rate.

        Args:
            Y (float): Sampling rate

        Returns:
            float: Calculated speed in km/h
        """
        m, c = self.calculate_speed_equation()
        return m * Y + c

    def generate_intermediate_points(self):
        """
        Generate intermediate points along the edge.
        """
        edge_proj = gpd.GeoDataFrame(geometry=self.edge[['geometry']], crs="EPSG:4326").to_crs(epsg=32618)
        edge_linestring_proj = edge_proj.geometry.iloc[0]

        # Generate points at interval distances
        points_projected = [edge_linestring_proj.interpolate(distance) for distance in
                            range(0, int(self.edge.length) + 1, self.interval)]

        # Reproject points back to original CRS
        transformer = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
        points = [Point(*transformer.transform(point.x, point.y)) for point in points_projected]

        return gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

    def assign_characteristics(self, gdf_4326_gen, nodes, edge, params):

        #Extract edge details
        to_node = edge.name[1]
        from_node = edge.name[0]
        total_length = edge.length

        for param in params['grouped']:
            if param['average_effect']:

                #Unpacking the params
                grouping_key = param['grouping_key']
                param_name = param['name']


                to_mean = nodes[f"normalized_{param_name}_{grouping_key}"].loc[to_node]
                from_mean = nodes[f"normalized_{param_name}_{grouping_key}"].loc[from_node]
                to_std = nodes[f"{param_name}_std_{grouping_key}"].loc[to_node]
                from_std = nodes[f"{param_name}_std_{grouping_key}"].loc[from_node]

                for idx in gdf_4326_gen.index:
                    param_mean = ((self.interval * idx) / total_length) * to_mean + \
                                 ((self.interval * (len(gdf_4326_gen) - idx - 1)) / total_length) * from_mean
                    param_std = (to_std + from_std) / 2
                    param_value = np.random.normal(param_mean, 0.5 * param_std)
                    gdf_4326_gen.at[idx, f'{param_name}_{grouping_key}'] = param_value
        return gdf_4326_gen

    def generate_Y_values(self, gdf_4326_gen, params, time_tracker):

        # Initialize previous values for grouped and global parameters
        prev_values = {param['name']: 0.5 for param in params['grouped']}
        prev_global_values = {param['name']: 0.5 for param in params['global']}
        prev_traffic_factor = 0.5
        Yo = self.initial_sampling_rate[self.vehicle_type]
        Y_values = []
        speed_values = []

        for idx, point in gdf_4326_gen.iterrows():

            current_hour = time_tracker.current_hour
            traffic_factor = params['global'][0]['constant'][current_hour]
            Yo_adjusted = Yo

            # Retrieve grouped parameter values
            current_values = {
                param['name']: gdf_4326_gen.get(f"{param['name']}_{param['grouping_key']}")[idx]
                for param in params['grouped']
            }

            # Retrieve global parameter values (handle both time-dependent and static values)
            current_global_values = {}
            for param in params['global']:
                if isinstance(param['constant'], dict):  # Time-dependent global parameter
                    current_global_values[param['name']] = param['constant'].get(current_hour, 0.5)
                elif isinstance(param['constant'], tuple):  # Static value range
                    current_global_values[param['name']] = np.random.uniform(*param['constant'])
                else:  # Static single value
                    current_global_values[param['name']] = param['constant']

            # # Adjust Y values using grouped parameters
            if all(prev_values.values()):
                for param, prev_value in prev_values.items():
                    delta = current_values[param] - prev_value
                    coef = np.random.uniform(
                        *[const for const in [p['constant'] for p in params['grouped'] if p['name'] == param][0]])
                    factor = (1 + coef * abs(delta)) if delta >= 0 else (1 - coef * abs(delta))
                    Yo_adjusted *= factor

            # # Adjust Y values using global parameters
            if all(prev_global_values.values()):
                for param, prev_global_value in prev_global_values.items():
                    delta = current_global_values[param] - prev_global_value
                    if param == 'traffic':  # traffic-specific logic (if needed)
                        coef = self.edge['normalized_betweenness_centrality']
                    else:  # General logic for other global parameters
                        coef = np.random.uniform(0.02, 0.10)
                    factor = (1 + coef * abs(delta)) if delta >= 0 else (1 - coef * abs(delta))
                    Yo_adjusted *= factor

            # # Traffic adjustments (specific adjustment logic remains)
            delta_T = traffic_factor - prev_traffic_factor
            traffic_coef = self.edge['normalized_betweenness_centrality']
            traffic_factor_adjustment = (1 + traffic_coef * abs(delta_T)) if delta_T >= 0 else (
                        1 - traffic_coef * abs(delta_T))
            Yo_adjusted *= traffic_factor_adjustment

            # Calculate speed and time
            speed_kmph = self.calculate_speed(Yo_adjusted)
            speed_values.append(speed_kmph)
            segment_time = ((self.interval / 1000) / speed_kmph) * 60
            time_tracker.update_time(segment_time)
            Y_values.append(Yo_adjusted)
            Yo = Yo_adjusted

            # Update previous values for grouped and global parameters
            prev_values.update(current_values)
            prev_global_values.update(current_global_values)
            prev_traffic_factor = traffic_factor

        return [round(value) for value in Y_values], speed_values

    def expand_points(self, y_values, vehicle_type, angle_and_radius_limit):
        """
        Expand points based on y-values and vehicle-specific parameters.
        """
        gdf_edge = gpd.GeoDataFrame(geometry=[self.edge.geometry], crs="EPSG:4326")

        expanded_points = []
        utm_crs = CRS.from_user_input(gdf_edge.estimate_utm_crs())
        edge_projected = gdf_edge.to_crs(utm_crs)

        base_distance = self.interval / (floor(np.mean(y_values)))
        delta_limit = 0.1 * base_distance
        point_distance = -base_distance

        for idx in range(len(y_values) - 1):
            # start_point = self.edge.geometry.iloc[idx]
            num_points = y_values[idx] if idx < len(y_values) else 0

            if num_points > 0:
                distances = []
                for _ in range(num_points):
                    delta = random.uniform(0.5 * delta_limit, 1.5 * delta_limit)
                    point_distance += max(base_distance + 5, base_distance + delta)
                    distances.append(point_distance)

                for distance_m in distances:
                    random_bearing = random.uniform(0,360)
                    new_point = edge_projected.geometry.iloc[0].interpolate(distance_m)

                    #Reproject the generated points back to lat and lng
                    new_point_gdf = gpd.GeoDataFrame(geometry=[new_point], crs=utm_crs).to_crs("EPSG:4326")
                    new_point = new_point_gdf.geometry.iloc[0]

                    #Displace the point
                    new_location = distance(meters=np.random.uniform(30, angle_and_radius_limit[vehicle_type][1])).destination((new_point.y, new_point.x), random_bearing)
                    displaced_point = Point(new_location[1], new_location[0])
                    expanded_points.append(displaced_point)

        return gpd.GeoDataFrame(geometry=expanded_points, crs="EPSG:4326")