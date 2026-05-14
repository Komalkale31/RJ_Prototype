import logging
import sys

def setup_logger(name="RJ_Prototype", level=logging.INFO):
    """Sets up a standardized logger for the project."""
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
    return logger

logger = setup_logger()
