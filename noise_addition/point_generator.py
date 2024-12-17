import random
from math import floor

import numpy as np
from geopy.distance import distance
from pyproj import Transformer
from shapely.geometry.point import Point
import geopandas as gpd

class PointGenerator:
    def __init__(self, edge, interval):
        self.edge = edge
        self.interval = interval

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

    def expand_points(self, y_values, vehicle_type, angle_and_radius_limit):
        """
        Expand points based on y-values and vehicle-specific parameters.
        """
        expanded_points = []
        utm_crs = "EPSG:32618"
        edge_projected = gpd.GeoDataFrame(geometry=self.edge[['geometry']], crs=utm_crs)

        base_distance = self.interval / (floor(np.mean(y_values)))
        delta_limit = 0.1 * base_distance
        point_distance = -base_distance

        for idx in range(len(self.edge) - 1):
            # start_point = self.edge.geometry.iloc[idx]
            num_points = y_values[idx] if idx < len(y_values) else 0

            if num_points > 0:
                distances = []
                for _ in range(num_points):
                    delta = random.uniform(0.5 * delta_limit, 1.5 * delta_limit)
                    point_distance += max(base_distance + 5, base_distance + delta)
                    distances.append(point_distance)

                for distance_m in distances:
                    random_bearing = random.uniform(-angle_and_radius_limit[vehicle_type][0],
                                                    angle_and_radius_limit[vehicle_type][0])
                    new_point = edge_projected.geometry.iloc[0].interpolate(distance_m)

                    new_point_gdf = gpd.GeoDataFrame(geometry=[new_point], crs=utm_crs).to_crs("EPSG:4326")
                    displaced_location = distance(
                        meters=random.uniform(30, angle_and_radius_limit[vehicle_type][1])).destination(
                        (new_point.y, new_point.x), random_bearing)

                    displaced_point = Point(displaced_location[1], displaced_location[0])
                    expanded_points.append(displaced_point)

        return gpd.GeoDataFrame(geometry=expanded_points, crs="EPSG:4326")