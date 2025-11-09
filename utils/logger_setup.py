import logging
import sys
import os
import datetime

def setup_logging(log_dir="logs", log_file="training.log"):
    """
    Configures the root logger.

    Moves the previous log file to an 'archive' sub-directory 
    within the 'log_dir' with a timestamp.

    Parameters:
    - log_dir (str): Directory to store log files.
    - log_file (str): Name of the current log file.
    """
    
    # Define paths
    log_path = os.path.join(log_dir, log_file)
    archive_dir = os.path.join(log_dir, "archive") # <-- Old logs go here

    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)

    # --- Archive previous log if it exists ---
    if os.path.exists(log_path):
        try:
            # Create a timestamped name for the old log
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_name, ext = os.path.splitext(log_file)
            archive_name = f"{base_name}_{now}{ext}"
            
            # Full path for the archived log
            archive_file_path = os.path.join(archive_dir, archive_name)

            # Move the old log file
            os.rename(log_path, archive_file_path)
        except Exception as e:
            print(f"Warning: Could not archive old log file: {e}")
            # Continue anyway, will overwrite

    # set up logging for the new run
    logger = logging.getLogger()
    
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # --- File Handler (for current log) ---
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # --- Console (Stream) Handler ---
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        logging.info("Logging configured. Outputting to console and %s", log_path)
        if 'archive_file_path' in locals():
            logging.info("Archived previous log to %s", archive_file_path)
    else:
        logging.info("Logger already configured.")