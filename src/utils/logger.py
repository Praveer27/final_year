"""
Logger Configuration for Gesture2Speech Project
Provides centralized logging functionality
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "30 days"
):
    """
    Setup logger with file and console handlers
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log files
        retention: How long to keep old log files
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with custom format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler if log directory is provided
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # General log file
        logger.add(
            log_dir / "gesture2speech_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
        
        # Error log file
        logger.add(
            log_dir / "errors_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    logger.info(f"Logger initialized with level: {log_level}")
    return logger


def get_logger():
    """Get the configured logger instance"""
    return logger


if __name__ == "__main__":
    # Test logger
    test_logger = setup_logger(log_dir=Path("logs"), log_level="DEBUG")
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")

# Made with Bob
