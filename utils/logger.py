# utils/logger.py
import logging
import sys
from pathlib import Path

class Logger:
    """Centralized logging utility"""
    
    _loggers = {}
    
    @staticmethod
    def get_logger(name: str, log_file: str = './logs/app.log', level=logging.INFO):
        """Get or create logger"""
        if name in Logger._loggers:
            return Logger._loggers[name]
        
        # Create logs directory
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        Logger._loggers[name] = logger
        return logger
