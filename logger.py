"""
logger.py

A centralized logging module for the Compound Trading Bot.
Provides consistent logging functionality across all modules.
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Add reference to src directory if needed
try:
    from config import Config, LOGS_DIR
except ImportError:
    # If running as standalone or config not imported
    LOGS_DIR = Path("logs")

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log file retention
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# Color codes for console output
COLORS = {
    'RESET': '\033[0m',
    'DEBUG': '\033[94m',    # Blue
    'INFO': '\033[92m',     # Green
    'WARNING': '\033[93m',  # Yellow
    'ERROR': '\033[91m',    # Red
    'CRITICAL': '\033[1;91m'  # Bold Red
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to console logs."""
    
    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
        return super().format(record)

def get_logger(
    name: str,
    log_level: int = DEFAULT_LOG_LEVEL,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_LOG_DATE_FORMAT,
    colored: bool = True,
    propagate: bool = False
) -> logging.Logger:
    """
    Get or create a logger with the specified configuration.
    
    Args:
        name: Logger name (usually the module name)
        log_level: Logging level (default: INFO)
        log_file: Log file path (default: None, will use name to generate path)
        console: Whether to log to console (default: True)
        log_format: Log message format (default: DEFAULT_LOG_FORMAT)
        date_format: Log date format (default: DEFAULT_LOG_DATE_FORMAT)
        colored: Whether to use colored output in console (default: True)
        propagate: Whether the logger should propagate to parent (default: False)
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Return logger if already configured
    if logger.handlers:
        return logger
    
    # Set log level
    logger.setLevel(log_level)
    
    # Set propagation
    logger.propagate = propagate
    
    # Create file handler if log file is specified or create default
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"{name}_{timestamp}.log"
    
    log_file = Path(log_file)
    
    # Create logs directory if it doesn't exist
    log_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Use rotating file handler to manage log size
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    
    # Set file handler formatter
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    # Add console handler if console is True
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if colored:
            console_formatter = ColoredFormatter(log_format, date_format)
        else:
            console_formatter = logging.Formatter(log_format, date_format)
            
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

def setup_logging(
    default_level: int = DEFAULT_LOG_LEVEL,
    config_dict: Optional[Dict[str, Any]] = None
) -> None:
    """
    Setup basic logging configuration for the application.
    
    Args:
        default_level: Default log level for the root logger
        config_dict: Optional configuration dictionary for logging
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(default_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    
    console_formatter = ColoredFormatter(DEFAULT_LOG_FORMAT, DEFAULT_LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(console_handler)
    
    # Setup file handler for all logs
    all_log_file = LOGS_DIR / f"all_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(all_log_file),
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    
    file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_LOG_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(default_level)
    
    root_logger.addHandler(file_handler)
    
    # Apply any additional configuration
    if config_dict:
        for logger_name, logger_config in config_dict.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(logger_config.get('level', default_level))
            
            # Add handlers as specified in config
            if logger_config.get('file'):
                log_file = LOGS_DIR / logger_config['file']
                handler = logging.handlers.RotatingFileHandler(
                    filename=str(log_file),
                    maxBytes=MAX_LOG_SIZE,
                    backupCount=BACKUP_COUNT,
                    encoding='utf-8'
                )
                handler.setFormatter(file_formatter)
                logger.addHandler(handler)

def log_exception(logger: logging.Logger, exc_info=True, stack_info=False) -> None:
    """
    Log an exception with detailed information.
    
    Args:
        logger: Logger instance
        exc_info: Whether to include exception info (default: True)
        stack_info: Whether to include stack info (default: False)
    """
    logger.exception(
        "An exception occurred", 
        exc_info=exc_info, 
        stack_info=stack_info
    )

# Example usage
if __name__ == "__main__":
    # Setup basic logging
    setup_logging()
    
    # Get a logger for this module
    logger = get_logger("logger_test")
    
    # Log some messages
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    try:
        # Cause an exception
        1 / 0
    except Exception as e:
        # Log the exception
        log_exception(logger)