import numpy as np
import osmnx as ox
from pymongo import MongoClient, UpdateOne
import random
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import logging
import time
from typing import List, Dict, Any, Tuple
from functools import partial
import os
import psutil
from datetime import datetime, timedelta
from simulation import Simulation

def setup_logging():
    """Configure basic console logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger()

class BatchProgressLogger:
    def __init__(self, total_batches: int, batch_size: int):
        self.total_batches = total_batches
        self.batch_size = batch_size
        self.processed_batches = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.update_interval = 5  # Update log every 5 seconds
        self.total_errors = 0
        self.total_docs = total_batches * batch_size
        
    def update(self, batch_index: int, successful_docs: int, error_count: int = 0):
        """Update progress for a completed batch"""
        self.processed_batches += 1
        self.total_errors += error_count
        current_time = time.time()
        
        # Only update if enough time has passed or we're done
        if (current_time - self.last_update_time) >= self.update_interval or self.processed_batches >= self.total_batches:
            progress = (self.processed_batches / self.total_batches) * 100
            elapsed_time = current_time - self.start_time
            docs_processed = self.processed_batches * self.batch_size
            
            status = (
                f"Batch {batch_index + 1}/{self.total_batches} completed | "
                f"Documents: {docs_processed:,}/{self.total_docs:,} "
                f"({progress:.1f}%) | "
                f"Success in batch: {successful_docs}/{self.batch_size} | "
                f"Total Errors: {self.total_errors:,} | "
                f"Time: {timedelta(seconds=int(elapsed_time))}"
            )
            
            logging.info(status)
            self.last_update_time = current_time

    def finalize(self):
        """Log final statistics"""
        elapsed_time = time.time() - self.start_time
        docs_processed = self.processed_batches * self.batch_size
        
        final_stats = (
            f"\nProcessing completed:\n"
            f"Batches processed: {self.processed_batches:,}/{self.total_batches:,}\n"
            f"Documents processed: {docs_processed:,}/{self.total_docs:,}\n"
            f"Total errors: {self.total_errors:,}\n"
            f"Total time: {timedelta(seconds=int(elapsed_time))}"
        )
        
        logging.info(final_stats)

class ParallelMapMatcher:
    def __init__(
        self,
        mongo_string: str,
        graph_path: str,
        batch_size: int = 100,
        max_workers: int = None
    ):
        self.logger = logging.getLogger('ParallelMapMatcher')
        self.mongo_string = mongo_string
        self.graph_path = graph_path
        self.batch_size = batch_size
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        # Constants remain the same as in your original code
        self.INTERVAL = 1000
        self.INITIAL_SAMPLING_RATE = {
            "car": 2 * (self.INTERVAL / 500),
            "truck": 3 * (self.INTERVAL / 500),
            "motorcycle": 2.5 * (self.INTERVAL / 500)
        }
        self.ANGLE_AND_RADIUS_LIMIT = {
            "car": (15, 75),
            "truck": (10, 45),
            "motorcycle": (20, 100)
        }
        self.SEASONS = {
            'winter': ['November', 'December', 'January', 'February'],
            'spring': ['March', 'April', 'May'],
            'summer': ['June', 'July', 'August'],
            'autumn': ['September', 'October', 'November']
        }
        
        self.load_graph()

    def load_graph(self):
        """Load the graph and convert to nodes/edges"""
        self.logger.info("Loading graph from %s...", self.graph_path)
        start_time = time.time()
        
        try:
            self.graph = ox.load_graphml(self.graph_path)
            self.nodes, _ = ox.convert.graph_to_gdfs(self.graph)
            
            load_time = time.time() - start_time
            self.logger.info(f"Graph loaded successfully in {load_time:.2f} seconds")
            self.logger.info(f"Graph contains {len(self.nodes):,} nodes")
            
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Initial memory usage: {memory_usage:.1f} MB")
            
        except Exception as e:
            self.logger.error(f"Failed to load graph: {str(e)}")
            raise

    def process_batch(self, batch_data: Tuple[int, List[Dict[str, Any]]]) -> Tuple[int, List[UpdateOne], int, int]:
        """Process a batch of documents
        Returns: (batch_index, bulk_operations, successful_docs, error_count)
        """
        batch_index, batch = batch_data
        bulk_operations = []
        error_count = 0
        
        for doc in batch:
            try:
                route = doc["route"]
                doc_id = doc["_id"]

                chosen_vehicle_type = random.choice(["car", "truck", "motorcycle"])
                chosen_season = random.choice(["winter", "spring", "summer", "autumn"])
                chosen_time = random.randint(0, 23)

                edges = ox.routing.route_to_gdf(self.graph, route)

                simulation = Simulation(
                    self.nodes, edges, chosen_vehicle_type,
                    chosen_season, chosen_time
                )
                points, end_time, speed_values, factor_values, error_distances = simulation.simulate()

                trajectory = {
                    "vehicle_type": chosen_vehicle_type,
                    "season": chosen_season,
                    "chosen_time": chosen_time,
                    "end_time": end_time,
                    "coordinates": [point.coords[0] for point in points],
                    "speed": speed_values,
                    "avg_error_distance": float(np.average(error_distances))
                }

                bulk_operations.append(
                    UpdateOne(
                        {"_id": doc_id},
                        {"$set": {"trajectory": trajectory}}
                    )
                )

            except Exception as e:
                error_count += 1
                self.logger.error(f"Error processing document {doc.get('_id', 'unknown')}: {str(e)}")

        successful_docs = len(bulk_operations)
        return batch_index, bulk_operations, successful_docs, error_count

    def process_all_documents(self):
        """Process all documents in parallel with improved batch distribution"""
        client = MongoClient(self.mongo_string)
        collection = client['map_matching']['paths_tree']
        
        # Get total document count and create batches
        total_docs = collection.count_documents({})
        total_batches = (total_docs + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"Starting processing of {total_docs:,} documents in {total_batches:,} batches")
        progress_logger = BatchProgressLogger(total_batches, self.batch_size)
        
        # Create batch chunks efficiently
        cursor = collection.find({}, batch_size=self.batch_size)
        batches = []
        current_batch = []
        
        for doc in cursor:
            current_batch.append(doc)
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        # Process batches in parallel
        try:
            with ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=self._init_worker,
                initargs=(self.graph, self.nodes)
            ) as executor:
                # Create tasks with batch indices
                future_to_batch = {
                    executor.submit(self.process_batch, (i, batch)): i 
                    for i, batch in enumerate(batches)
                }
                
                for future in as_completed(future_to_batch):
                    try:
                        batch_index, bulk_operations, successful_docs, error_count = future.result()
                        
                        # Perform bulk write if we have operations
                        if bulk_operations:
                            collection.bulk_write(bulk_operations, ordered=False)
                        
                        # Update progress
                        progress_logger.update(batch_index, successful_docs, error_count)
                    
                    except Exception as e:
                        self.logger.error(f"Batch processing error: {str(e)}")
        
        finally:
            progress_logger.finalize()
            client.close()

    @staticmethod
    def _init_worker(graph, nodes):
        """Initialize worker process with shared graph and nodes"""
        global _shared_graph
        global _shared_nodes
        _shared_graph = graph
        _shared_nodes = nodes

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting Map Matching Process")
    
    try:
        processor = ParallelMapMatcher(
            mongo_string="mongodb://sih24:sih24@localhost:27017/sih24?authSource=sih24",
            graph_path="../data/merged_graph.graphml",
            batch_size=1000,
            max_workers=10
        )
        
        processor.process_all_documents()
        
    except Exception as e:
        logger.error(f"Fatal error in main process: {str(e)}", exc_info=True)
        raise
