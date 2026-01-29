import logging
import os
from datetime import datetime

# Define log file name with current date and time
Log_file = f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Define log file path
Log_path = os.path.join(os.getcwd(),"Logs", Log_file)

# Create directory for log file if it doesn't exist
os.makedirs(os.path.dirname(Log_path), exist_ok=True)


logging.basicConfig(
    filename=Log_path,
    level=logging.INFO,
    format='%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)


if __name__ == "__main__":
    logging.info("Logger initialized successfully.")