import numpy as np
import cupy as cp
import torch
from shapely.geometry import Point
import geopandas as gpd
import logging
from time_tracker import TimeTracker

logger = logging.getLogger(__name__)

class GPUSimulator:
    def __init__(self, nodes, edges, gpu_manager):
        self.nodes = nodes
        self.edges = edges
        self.gpu = gpu_manager
        self._setup_parameters()

    def _setup_parameters(self):
        """Prepare simulation parameters"""
        try:
            self.node_coords = self.gpu.to_gpu(
                np.column_stack((self.nodes.geometry.x, self.nodes.geometry.y))
            )
        except Exception as e:
            logger.error(f"Error setting up parameters: {str(e)}")
            raise

    def simulate_trajectory(self, route, vehicle_type, season, time):
        """Main simulation method"""
        try:
            logger.debug(f"Starting simulation for route with {len(route)} points")
            
            # Initialize time tracker
            time_tracker = TimeTracker(initial_hour=time)
            
            # Process route segments
            all_points = []
            all_speeds = []
            all_errors = []
            
            for i in range(len(route) - 1):
                segment = route[i:i+2]
                points, speeds, errors = self._process_segment(
                    segment, vehicle_type, season, time_tracker
                )
                
                all_points.extend(points)
                all_speeds.extend(speeds)
                all_errors.extend(errors)
            
            return {
                'points': all_points,
                'speeds': all_speeds,
                'errors': all_errors,
                'end_time': time_tracker.current_hour
            }
            
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            raise

    def _process_segment(self, segment, vehicle_type, season, time_tracker):
        """Process individual route segment"""
        try:
            # Generate intermediate points
            points = self._generate_points(segment)
            
            # Calculate characteristics
            characteristics = self._calculate_characteristics(points, season)
            
            # Generate trajectory
            trajectory = self._generate_trajectory(
                points, characteristics, vehicle_type, time_tracker
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Error processing segment: {str(e)}")
            raise
