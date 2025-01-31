import pymongo
import pandas as pd
import numpy as np
import multiprocessing as mp

# MongoDB Connection
MONGO_URI = "mongodb://sih24:sih24@localhost:27017/sih24?authSource=sih24"
DB_NAME = "map_matching"
COLLECTION_NAME = "paths_tree"
NUM_CORES = 10  # Adjust based on CPU

def fetch_data(skip, limit, process_id):
    """Fetch data in chunks to prevent repetition and process it."""
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    cursor = collection.find({"trajectory": {"$exists": True}}, {
        "trajectory.vehicle_type": 1,
        "trajectory.season": 1,
        "trajectory.chosen_time": 1,
        "trajectory.avg_error_distance": 1,
        "trajectory.speed": 1
    }).skip(skip).limit(limit)

    data = []
    for index, doc in enumerate(cursor):
        if index % 5000 == 0:
            print(f"Process {process_id} - Processed {index + skip} documents")

        traj = doc.get("trajectory", {})
        speed_array = traj.get("speed", [])
        avg_speed = np.mean(speed_array) if speed_array else None

        data.append({
            "vehicle_type": traj.get("vehicle_type"),
            "season": traj.get("season"),
            "chosen_time": traj.get("chosen_time"),
            "avg_error_distance": traj.get("avg_error_distance"),
            "avg_speed": avg_speed
        })

    df = pd.DataFrame(data)
    csv_filename = f"processed_data_{process_id}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Process {process_id} - Saved {len(df)} records to {csv_filename}")
    client.close()
    return csv_filename

def main():
    """Main function to divide workload across multiple processes."""
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    total_docs = collection.count_documents({"trajectory": {"$exists": True}})
    client.close()

    print(f"Total Documents: {total_docs}")
    batch_size = total_docs // NUM_CORES

    # Create a process pool
    pool = mp.Pool(NUM_CORES)
    tasks = [(i * batch_size, batch_size, i) for i in range(NUM_CORES)]

    # Run processes in parallel
    results = pool.starmap(fetch_data, tasks)

    # Merge all CSVs into one
    print("Merging all CSVs...")
    all_dfs = [pd.read_csv(csv) for csv in results]
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv("final_output_post_process.csv", index=False)
    print("Final CSV saved as final_output_post_process.csv")

if __name__ == "__main__":
    main()
