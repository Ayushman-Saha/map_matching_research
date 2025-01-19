import numpy as np
import osmnx as ox
from pymongo import MongoClient
import random
from multiprocessing import Pool, cpu_count, Manager
import logging
from tqdm import tqdm
from functools import partial
from simulation import Simulation


def configure_logging():
    """Configure logging for each worker process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )


def process_single_document(doc, graph, nodes):
    """Process a single document with route data."""
    configure_logging()
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

        return doc_id, trajectory, True  # Success
    except Exception as e:
        logging.error(f"Error processing document {doc.get('_id', 'Unknown ID')}: {str(e)}")
        return doc.get("_id", None), None, False  # Failure


def track_progress(progress, lock, tqdm_bar):
    """Update progress in the main process."""
    with lock:
        progress.value += 1
        tqdm_bar.update(1)


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

        # Shared progress tracker
        with Manager() as manager:
            progress = manager.Value('i', 0)  # Shared counter
            lock = manager.Lock()
            tqdm_bar = tqdm(total=total_documents, desc="Processing documents")

            # Create process pool
            with Pool(num_processes, initializer=configure_logging) as pool:
                process_func = partial(process_single_document, graph=graph, nodes=nodes)

                # Process documents and update progress
                results = []
                for res in pool.imap_unordered(process_func, documents):
                    track_progress(progress, lock, tqdm_bar)
                    results.append(res)

            tqdm_bar.close()

            # Update MongoDB with results
            successful_updates = 0
            for doc_id, trajectory, success in results:
                if success and doc_id and trajectory:
                    try:
                        collection.update_one(
                            {"_id": doc_id},
                            {"$set": {"trajectory": trajectory}}
                        )
                        successful_updates += 1
                    except Exception as e:
                        logging.error(f"Error updating document {doc_id}: {str(e)}", exc_info=True)

        # Log final statistics
        logging.info(
            f"\nFinal Statistics:\n"
            f"Total Documents Processed: {total_documents}\n"
            f"Successful Updates: {successful_updates}\n"
        )

    except Exception as e:
        logging.error(f"Critical error in main process: {str(e)}")
        raise

    finally:
        logging.info("Processing completed")
        if 'client' in locals():
            client.close()
            logging.info("MongoDB connection closed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler()
        ]
    )
    main()
