import os

# ADAPT File paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTAMINATION_RATIO = 0.001
PII_SAMPLE_SIZE = 2000
PROFILING_SAMPLE_ROWS = 100_000
FORMATTED_ADDRESS_SAMPLE_SIZE = 100

# Miscellaneous constants
MAX_RETRIES = 5
CHUNK_SIZE = 1024
BATCH_SIZE = 128

# Application settings
APP_NAME = "ADAPT"
VERSION = "0.1.0"

# Database settings
HOST = "localhost"
PORT = 8058


# Logging levels
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_ERROR = "ERROR"

# API settings
TIMEOUT = 30  # seconds


