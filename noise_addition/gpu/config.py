import logging
import sys
from pathlib import Path

# Constants
INTERVAL = 1000
INITIAL_SAMPLING_RATE = {
    "car": 2 * (INTERVAL / 500),
    "truck": 3 * (INTERVAL / 500),
    "motorcycle": 2.5 * (INTERVAL / 500)
}
ANGLE_AND_RADIUS_LIMIT = {
    "car": (15, 75),
    "truck": (10, 45),
    "motorcycle": (20, 100)
}
SEASONS = {
    'winter': ['November', 'December', 'January', 'February'],
    'spring': ['March', 'April', 'May'],
    'summer': ['June', 'July', 'August'],
    'autumn': ['September', 'October', 'November']
}

DATA_DIR = "../../data"
LOG_DIR = "./logs"

# MongoDB Configuration
MONGO_URI = "mongodb://sih24:sih24@localhost:27017/sih24?authSource=sih24"
DB_NAME = "map_matching"
COLLECTION_NAME = "paths_tree"

# GPU Configuration
BATCH_SIZE = 512  # Number of trajectories to simulate in parallel
GPU_MEMORY_FRACTION = 0.8  # Fraction of GPU memory to use

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = "./logs/simulation.log"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )



