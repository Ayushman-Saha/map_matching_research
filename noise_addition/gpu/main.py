import logging
from pymongo import MongoClient
import osmnx as ox
import time
import random
from config import setup_logging, MONGO_URI, DB_NAME, COLLECTION_NAME , SEASONS
import numpy as np
from gpu_utils import GPUManager
from progress_tracker import ProgressTracker
from simulator import GPUSimulator

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize GPU
        gpu_manager = GPUManager()
        
        # Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        # Load graph
        logger.info("Loading graph data...")
        graph = ox.load_graphml("../data/merged_graph.graphml")
        nodes, _ = ox.convert.graph_to_gdfs(graph)
        
        # Initialize simulator
        simulator = GPUSimulator(nodes, graph, gpu_manager)
        
        # Get documents
        documents = list(collection.find())
        progress = ProgressTracker(len(documents))
        
        # Process documents
        for doc in documents:
            try:
                start_time = time.time()
                
                # Generate random parameters
                vehicle_type = random.choice(["car", "truck", "motorcycle"])
                season = random.choice(list(SEASONS.keys()))
                chosen_time = random.randint(0, 23)
                
                # Run simulation
                results = simulator.simulate_trajectory(
                    doc['route'],
                    vehicle_type,
                    season,
                    chosen_time
                )
                
                # Prepare trajectory data
                trajectory = {
                    "vehicle_type": vehicle_type,
                    "season": season,
                    "chosen_time": chosen_time,
                    "end_time": results['end_time'],
                    "coordinates": [(p.x, p.y) for p in results['points']],
                    "speed": results['speeds'],
                    "avg_error_distance": np.mean(results['errors'])
                }
                
                # Update MongoDB
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"trajectory": trajectory}}
                )
                
                # Update progress
                time_taken = time.time() - start_time
                progress.update(doc["_id"], success=True, time_taken=time_taken)
                
            except Exception as e:
                logger.error(f"Error processing document {doc.get('_id')}: {str(e)}")
                progress.update(doc["_id"], success=False)
                
        progress.finalize()
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise
        
    finally:
        if 'client' in locals():
            client.close()
            logger.info("MongoDB connection closed")

if __name__ == "__main__":
    main()