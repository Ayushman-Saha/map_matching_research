import numpy as np
import osmnx as ox
from pymongo import MongoClient
import random
import geopandas as gpd
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import logging
import time
from datetime import datetime, timedelta
import sys
from tqdm import tqdm

from simulation import Simulation

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


class ProgressTracker:
    def __init__(self, total_documents):
        self.total = total_documents
        self.processed = 0
        self.start_time = time.time()
        self.success_count = 0
        self.error_count = 0
        self.pbar = tqdm(total=total_documents, desc="Processing documents")

    def update(self, success=True):
        self.processed += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.pbar.update(1)
        
        # Calculate progress statistics
        elapsed_time = time.time() - self.start_time
        docs_per_second = self.processed / elapsed_time if elapsed_time > 0 else 0
        remaining_docs = self.total - self.processed
        estimated_remaining_time = remaining_docs / docs_per_second if docs_per_second > 0 else 0
        
        completion_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
        
        # Log progress
        if self.processed % 10 == 0 or self.processed == self.total:  # Log every 10 documents
            logging.info(
                f"\nProgress Update:\n"
                f"Processed: {self.processed}/{self.total} ({(self.processed/self.total*100):.2f}%)\n"
                f"Successful: {self.success_count} | Errors: {self.error_count}\n"
                f"Processing Speed: {docs_per_second:.2f} docs/second\n"
                f"Estimated Completion Time: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Remaining Time: {timedelta(seconds=int(estimated_remaining_time))}"
            )

    def finalize(self):
        self.pbar.close()
        total_time = time.time() - self.start_time
        logging.info(
            f"\nFinal Statistics:\n"
            f"Total Documents Processed: {self.processed}\n"
            f"Successful: {self.success_count} | Errors: {self.error_count}\n"
            f"Total Processing Time: {timedelta(seconds=int(total_time))}\n"
            f"Average Processing Speed: {self.processed/total_time:.2f} docs/second"
        )

def process_single_document(doc, graph, nodes, progress_tracker):
    """Process a single document with route data"""
    try:
        route = doc["route"]
        doc_id = doc["_id"]

        # Random selections
        chosen_vehicle_type = random.choice(["car", "truck", "motorcycle"])
        chosen_season = random.choice(["winter", "spring", "summer", "autumn"])
        chosen_time = random.randint(0, 23)

        # Convert route to GeoDataFrame
        edges = ox.routing.route_to_gdf(graph, route)

        # Run simulation
        simulation = Simulation(nodes, edges, chosen_vehicle_type, chosen_season, chosen_time)
        points, end_time, speed_values, factor_values, error_distances = simulation.simulate()

        # Prepare trajectory data
        trajectory = {
            "vehicle_type": chosen_vehicle_type,
            "season": chosen_season,
            "chosen_time": chosen_time,
            "end_time": end_time,
            "coordinates": [point.coords[0] for point in points],
            "speed": speed_values,
            "avg_error_distance": np.average(error_distances),
        }

        progress_tracker.update(success=True)
        return doc_id, trajectory

    except Exception as e:
        logging.error(f"Error processing document {doc.get('_id', 'Unknown ID')}: {str(e)}")
        progress_tracker.update(success=False)
        return None, None

def main():
    try:
        # MongoDB connection
        logging.info("Connecting to MongoDB...")
        mongo_string = "mongodb://sih24:sih24@localhost:27018/sih24?authSource=sih24"
        client = MongoClient(mongo_string)
        collection = client['map_matching']['paths_tree']
        logging.info("Successfully connected to MongoDB")

        # Load graph
        logging.info("Loading graph data...")
        graph = ox.load_graphml("../data/merged_graph.graphml")
        nodes, _ = ox.convert.graph_to_gdfs(graph)
        logging.info("Graph successfully loaded")

        # Get all documents
        documents = list(collection.find())
        total_documents = len(documents)
        logging.info(f"Found {total_documents} documents to process")

        # Calculate optimal number of processes
        num_processes = min(cpu_count(), total_documents)
        logging.info(f"Using {num_processes} processes")

        # Initialize progress tracker
        progress_tracker = ProgressTracker(total_documents)

        # Create process pool
        with Pool(num_processes) as pool:
            # Create partial function with fixed arguments
            process_func = partial(process_single_document, 
                                 graph=graph, 
                                 nodes=nodes, 
                                 progress_tracker=progress_tracker)
            
            # Process documents in parallel
            results = pool.map(process_func, documents)

            # Update MongoDB with results
            successful_updates = 0
            for doc_id, trajectory in results:
                if doc_id and trajectory:
                    try:
                        collection.update_one(
                            {"_id": doc_id},
                            {"$set": {"trajectory": trajectory}}
                        )
                        successful_updates += 1
                    except Exception as e:
                        logging.error(f"Error updating document {doc_id}: {str(e)}")

        # Log final statistics
        progress_tracker.finalize()
        logging.info(f"Successfully updated {successful_updates} documents in MongoDB")

    except Exception as e:
        logging.error(f"Critical error in main process: {str(e)}")
        raise

    finally:
        logging.info("Processing completed")
        if 'client' in locals():
            client.close()
            logging.info("MongoDB connection closed")

if __name__ == "__main__":
    main()
