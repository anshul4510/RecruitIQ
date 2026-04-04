import os
import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logging(name=None, level=logging.INFO):
    """
    Sets up a logger with a console and daily rotating file handler.
    If 'name' is None, it returns the root logger.
    """
    # Define log directory and file
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, "app.log")
    
    # Configure format
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    
    # Root logger configuration
    root_logger = logging.getLogger()
    
    # Only add handlers if they haven't been added yet (prevents duplicate logs)
    if not root_logger.handlers:
        root_logger.setLevel(level)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        root_logger.addHandler(console_handler)
        
        # File Handler (Timed Rotating)
        file_handler = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=7
        )
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)
    
    # Return specific logger if name is provided
    if name:
        return logging.getLogger(name)
    return root_logger

def get_logger(name):
    """Helper to get a named logger."""
    return logging.getLogger(name)
