from pymongo import MongoClient, UpdateOne
import numpy as np
import multiprocessing

client = MongoClient("mongodb://sih24:sih24@localhost:27017/sih24?authSource=sih24")
db = client["map_matching"]
collection = db["paths_tree"]

traffic_weights = np.array([
    0.2951, 0.2243, 0.1916, 0.1787, 0.1787, 0.1916, 0.2243, 0.3708,
    0.5455, 0.7098, 0.8121, 0.7894, 0.7648, 0.7098, 0.6607, 0.6479,
    0.6479, 0.7098, 0.7382, 0.6796, 0.5805, 0.4745, 0.4045, 0.3708
])
traffic_weights /= traffic_weights.sum()  # Normalize

NUM_WORKERS = 10

def process_chunk(skip, limit):
    """
    Worker function to process a subset of MongoDB documents.
    """
    client = MongoClient("mongodb://sih24:sih24@localhost:27017/sih24?authSource=sih24")
    db = client["map_matching"]
    collection = db["paths_tree"]

    cursor = collection.find({}, skip=skip, limit=limit)
    bulk_updates = []

    for index, doc in enumerate(cursor):
        chosen_time = np.random.choice(range(24), p=traffic_weights)

        start_time = doc.get("trajectory", {}).get("chosen_time", 0)
        end_time = doc.get("trajectory", {}).get("end_time", (start_time + 1) % 24)

        delta = (end_time - start_time) % 24
        new_end_time = (chosen_time + delta) % 24  # Ensure 24-hour wraparound

        bulk_updates.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {"trajectory.chosen_time": chosen_time.item(), "trajectory.end_time": new_end_time.item()}}
            )
        )

    if bulk_updates:
        collection.bulk_write(bulk_updates)

    print(f"Processed {limit} docs from skip {skip}")

def main():
    total_docs = collection.count_documents({})
    chunk_size = total_docs // NUM_WORKERS  # Divide dataset evenly

    processes = []
    for i in range(NUM_WORKERS):
        skip = i * chunk_size
        limit = chunk_size if i < NUM_WORKERS - 1 else total_docs - skip  # Ensure last process gets remaining docs
        p = multiprocessing.Process(target=process_chunk, args=(skip, limit))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("MongoDB documents updated successfully!")

if __name__ == "__main__":
    main()
