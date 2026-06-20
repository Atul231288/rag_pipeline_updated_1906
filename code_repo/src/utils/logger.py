# Centralized logger
import logging
import sys
import os
from config.settings import LOG_DIR, LOG_FILE, LOG_LEVEL

# def get_logger(name: str, level=logging.INFO) -> logging.Logger:
#     """
#     Get a logger with the specified name and level.
    
#     Args:
#         name (str): Name of the logger.
#         level: Logging level (default: logging.INFO).
        
#     Returns:
#         logging.Logger: Configured logger instance.
#     """
#     os.mkdir(LOG_DIR, exist_ok=True)  # Create log directory if it doesn't exist
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
    
#     # Create console handler and set level
#     ch = logging.StreamHandler(sys.stdout)
#     ch.setLevel(level)
    
#     # Create formatter and add it to the handlers
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     ch.setFormatter(formatter)
    
#     # Add the handlers to the logger
#     if not logger.hasHandlers():
#         logger.addHandler(ch)
    
#     return logger
import logging
import os
import sys

from config.settings import (
    LOG_DIR,
    LOG_FILE,
    LOG_LEVEL
)

def get_logger(name):

    os.makedirs(
        LOG_DIR,
        exist_ok=True
    )

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console Logs
    console_handler = logging.StreamHandler(
        sys.stdout
    )
    console_handler.setFormatter(
        formatter
    )

    # File Logs
    file_handler = logging.FileHandler(
        os.path.join(
            LOG_DIR,
            LOG_FILE
        ),
        encoding="utf-8"
    )
    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    return logger