import logging
import sys
import os
import datetime
import copy

# --- ANSI Color Codes ---
class Colors:
    RESET = "\033[0m"
    GREY = "\033[90m"
    CYAN = "\033[96m"   # Initiator
    GREEN = "\033[92m"  # Responder
    YELLOW = "\033[93m" # System/Main
    RED = "\033[91m"    # Error

class ColoredFormatter(logging.Formatter):
    """Custom formatter to color the levelname and message based on the source."""
    
    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)
        self.base_fmt = fmt

    def format(self, record):
        # Create a copy so we don't alter the record for other handlers (like files)
        # which shouldn't have color codes
        record = copy.copy(record)
        
        # Colorize based on Logger Name
        if "initiator" in record.name:
            color = Colors.CYAN
        elif "responder" in record.name:
            color = Colors.GREEN
        else:
            color = Colors.YELLOW

        # Apply color to the level name and the message
        levelname = record.levelname
        if levelname == "ERROR":
            color = Colors.RED
            
        record.levelname = f"{color}{levelname}{Colors.RESET}"
        record.msg = f"{color}{record.msg}{Colors.RESET}"
        
        return super().format(record)

def setup_logging(log_dir="logs", agent_id="agent"):
    """
    Sets up logging and archives ALL previous logs (main, initiator, responder)
    with a synchronized timestamp.
    """
    
    os.makedirs(log_dir, exist_ok=True)
    archive_dir = os.path.join(log_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    # --- 1. Synchronized Archiving ---
    # We archive ALL potential log files now, so they get the same timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_files_to_archive = [
        f"{agent_id}_main.log",
        f"{agent_id}_initiator.log",
        f"{agent_id}_responder.log"
    ]
    
    for filename in log_files_to_archive:
        file_path = os.path.join(log_dir, filename)
        _archive_if_exists(file_path, archive_dir, timestamp)

    # --- 2. Define Formatters ---
    file_fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    console_fmt = ColoredFormatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%H:%M:%S")

    # --- 3. Configure Root Logger (System) ---
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(console_fmt)
    root_logger.addHandler(ch)

    # Main Log File
    main_log_path = os.path.join(log_dir, f"{agent_id}_main.log")
    fh_main = logging.FileHandler(main_log_path, mode='w')
    fh_main.setFormatter(file_fmt)
    root_logger.addHandler(fh_main)
    
    logging.info(f"--- Logging Setup Complete. Archived old logs to {archive_dir} ---")

def get_specialized_logger(name, log_dir="logs", agent_id="agent"):
    """
    Returns a logger that writes to the root handlers AND a specific file.
    Name should be 'initiator' or 'responder'.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = True # Let it bubble up to Root (for Console + Main Log)

    # Unique File Handler for this logger
    log_file = os.path.join(log_dir, f"{agent_id}_{name}.log")
    
    # Check if handler already exists to avoid duplicates
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in logger.handlers):
        fh = logging.FileHandler(log_file, mode='a') # Append mode so we don't wipe it every round
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)

    return logger

def _archive_if_exists(file_path, archive_dir, timestamp):
    if os.path.exists(file_path):
        try:
            base_name = os.path.basename(file_path)
            new_name = f"{timestamp}_{base_name}"
            os.rename(file_path, os.path.join(archive_dir, new_name))
        except Exception as e:
            print(f"Could not archive log {file_path}: {e}")