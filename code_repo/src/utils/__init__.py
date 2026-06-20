"""
Utility modules for RAG pipeline
"""

import logging
import yaml
from pathlib import Path


def get_logger(name, log_file=None):
    """Get configured logger instance"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f)


class Metrics:
    """Metrics calculation utilities"""
    
    @staticmethod
    def calculate_metrics(predictions, ground_truth):
        """Calculate evaluation metrics"""
        pass
