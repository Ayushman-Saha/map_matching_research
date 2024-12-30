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

        turning_index_processor = ParameterProcessor(self.edges, "turn_severity_index", type="ungrouped")
        turning_index_processor.process()

        params = {
            'grouped': [
                {
                    "name": 'rainfall',
                    "grouping_key": self.season,
                    "constant": (0.02, 0.10),
                    "average_effect": True,
                    "proportionality" : "direct" #Proportionality with Y values
                },
                {
                    "name": 'visibility',
                    "grouping_key": self.season,
                    "constant": (0.04, 0.09),
                    "average_effect": True,
                    "proportionality" : "inverse"
                },

            ],
            'global': [
                {
                    "name": 'betweenness_centrality',
                    "constant": (0.05, 0.07),
                    "variant": None,
                    "average_effect": False,
                    "proportionality": "direct"
                },
                {
                    "name": 'traffic',
                    "constant": (0.05, 0.07),
                    "variant": self.chosen_time,
                    "average_effect": False,
                    "proportionality": "direct"
                },
                {
                    "name": 'turn_severity_index',
                    "constant": (0.5, 0.9),
                    "average_effect": False,
                    "variant": None,
                    "proportionality": "direct"
                }
            ],
        }

        all_points = []
        all_speed_values = []
        all_factor_values = {}
        error_distances = []

        for index, edge in self.edges.iterrows():
            # Generate intermediate points
            generator = PointGenerator(edge, INTERVAL, INITIAL_SAMPLING_RATE, self.vehicle_type)
            gdf_4326_gen = generator.generate_intermediate_points()

            gdf_4326_gen = generator.assign_characteristics(gdf_4326_gen, self.nodes, edge, params)
            Y_values, speed_values, factor_values = generator.generate_Y_values(gdf_4326_gen, params, time_tracker)
            expanded_points, error_distance = generator.expand_points(Y_values, self.vehicle_type, ANGLE_AND_RADIUS_LIMIT)

            all_speed_values.extend(speed_values)
            for key, value in factor_values.items():
                if key in all_factor_values:
                    all_factor_values[key].extend(value)
                else:
                    all_factor_values[key] = value[:]

            for point in expanded_points['geometry']:
                all_points.append(point)

            error_distances.extend(error_distance)

        return all_points, time_tracker.current_hour, all_speed_values, all_factor_values, error_distances

