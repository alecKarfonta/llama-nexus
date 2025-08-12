"""
Enhanced Logging Utility for LlamaCPP Management API
Provides colored, structured logging with class.function prefixes and variable highlighting
"""

import logging
import os
import sys
import inspect
from typing import Any, Dict, Optional
from datetime import datetime
import json


class ColorCodes:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


class EnhancedFormatter(logging.Formatter):
    """Custom formatter with colors and enhanced structure"""
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        # Get timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Determine log level color
        level_colors = {
            'DEBUG': ColorCodes.BRIGHT_BLACK,
            'INFO': ColorCodes.BRIGHT_BLUE,
            'WARNING': ColorCodes.BRIGHT_YELLOW,
            'ERROR': ColorCodes.BRIGHT_RED,
            'CRITICAL': ColorCodes.RED + ColorCodes.BG_WHITE + ColorCodes.BOLD,
        }
        
        level_color = level_colors.get(record.levelname, ColorCodes.WHITE)
        reset = ColorCodes.RESET if self.use_colors else ''
        
        # Format log level with padding
        level_str = f"[{record.levelname:8}]"
        if self.use_colors:
            level_str = f"{level_color}{level_str}{reset}"
        
        # Get class.function prefix from record
        class_func = getattr(record, 'class_func', 'Unknown.unknown')
        if self.use_colors:
            # Highlight class name and function name differently
            if '.' in class_func:
                class_name, func_name = class_func.split('.', 1)
                class_func_colored = f"{ColorCodes.BRIGHT_CYAN}{class_name}{ColorCodes.WHITE}.{ColorCodes.BRIGHT_GREEN}{func_name}{reset}"
            else:
                class_func_colored = f"{ColorCodes.BRIGHT_CYAN}{class_func}{reset}"
        else:
            class_func_colored = class_func
        
        # Format the message with variable highlighting
        message = self._highlight_variables(record.getMessage()) if self.use_colors else record.getMessage()
        
        # Build the final log line
        parts = [
            f"{ColorCodes.DIM if self.use_colors else ''}{timestamp}{reset}",
            level_str,
            f"[{class_func_colored}]",
            message
        ]
        
        formatted = " ".join(parts)
        
        # Add exception info if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        
        return formatted
    
    def _highlight_variables(self, message: str) -> str:
        """Highlight variables and values in log messages"""
        import re
        
        # Highlight quoted strings
        message = re.sub(
            r'"([^"]*)"',
            f'{ColorCodes.BRIGHT_YELLOW}"\\1"{ColorCodes.RESET}',
            message
        )
        
        # Highlight single quoted strings
        message = re.sub(
            r"'([^']*)'",
            f"{ColorCodes.BRIGHT_YELLOW}'\\1'{ColorCodes.RESET}",
            message
        )
        
        # Highlight numbers
        message = re.sub(
            r'\b(\d+(?:\.\d+)?)\b',
            f'{ColorCodes.BRIGHT_MAGENTA}\\1{ColorCodes.RESET}',
            message
        )
        
        # Highlight key=value pairs
        message = re.sub(
            r'\b(\w+)=([^\s,\]}\)]+)',
            f'{ColorCodes.CYAN}\\1{ColorCodes.WHITE}={ColorCodes.BRIGHT_YELLOW}\\2{ColorCodes.RESET}',
            message
        )
        
        # Highlight file paths
        message = re.sub(
            r'(/[^\s:,\]}\)]+)',
            f'{ColorCodes.BRIGHT_GREEN}\\1{ColorCodes.RESET}',
            message
        )
        
        # Highlight URLs and ports
        message = re.sub(
            r'\b(https?://[^\s]+)',
            f'{ColorCodes.BLUE}\\1{ColorCodes.RESET}',
            message
        )
        
        message = re.sub(
            r':(\d{4,5})\b',
            f':{ColorCodes.BRIGHT_MAGENTA}\\1{ColorCodes.RESET}',
            message
        )
        
        # Highlight container/docker IDs
        message = re.sub(
            r'\b([a-f0-9]{12,})\b',
            f'{ColorCodes.BRIGHT_CYAN}\\1{ColorCodes.RESET}',
            message
        )
        
        # Highlight success/failure keywords
        success_keywords = ['success', 'successful', 'completed', 'started', 'running', 'healthy', 'available']
        error_keywords = ['error', 'failed', 'failure', 'exception', 'critical', 'stopped', 'crashed', 'timeout']
        
        for keyword in success_keywords:
            message = re.sub(
                rf'\b({keyword})\b',
                f'{ColorCodes.BRIGHT_GREEN}\\1{ColorCodes.RESET}',
                message,
                flags=re.IGNORECASE
            )
        
        for keyword in error_keywords:
            message = re.sub(
                rf'\b({keyword})\b',
                f'{ColorCodes.BRIGHT_RED}\\1{ColorCodes.RESET}',
                message,
                flags=re.IGNORECASE
            )
        
        return message


class EnhancedLogger:
    """Enhanced logger with automatic class.function detection and structured logging"""
    
    def __init__(self, name: str = "llamacpp-api"):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the logger with enhanced formatting"""
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Determine if we should use colors
        # Force enable colors via environment variable, or auto-detect
        force_colors = os.getenv('FORCE_COLOR', '').lower() in ('1', 'true', 'yes')
        auto_colors = (
            sys.stderr.isatty() and 
            os.getenv('TERM') != 'dumb' and 
            os.getenv('NO_COLOR') is None
        )
        
        # Enable colors if forced or auto-detected, but allow NO_COLOR to override
        use_colors = (force_colors or auto_colors) and os.getenv('NO_COLOR') is None
        
        # Remove debug print
        # print(f"Color Detection Debug: force_colors={force_colors}, auto_colors={auto_colors}, use_colors={use_colors}, TERM={os.getenv('TERM')}, isatty={sys.stderr.isatty()}", file=sys.stderr)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stderr)
        formatter = EnhancedFormatter(use_colors=use_colors)
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        
        # Set log level from environment
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def _get_caller_info(self) -> str:
        """Get the class.function of the calling method"""
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual caller (skip this method and the log method)
            caller_frame = frame.f_back.f_back
            
            # Get function name
            func_name = caller_frame.f_code.co_name
            
            # Try to get class name from 'self' in locals
            if 'self' in caller_frame.f_locals:
                class_name = caller_frame.f_locals['self'].__class__.__name__
                return f"{class_name}.{func_name}"
            
            # Try to get class name from 'cls' in locals (class methods)
            elif 'cls' in caller_frame.f_locals:
                class_name = caller_frame.f_locals['cls'].__name__
                return f"{class_name}.{func_name}"
            
            # If no class context, just return function name
            else:
                return f"Module.{func_name}"
                
        except Exception:
            return "Unknown.unknown"
        finally:
            del frame
    
    def _log(self, level: int, message: str, *args, **kwargs):
        """Internal log method that adds class.function info"""
        if self.logger.isEnabledFor(level):
            # Create log record
            record = self.logger.makeRecord(
                self.logger.name, level, "", 0, message, args, None
            )
            
            # Add class.function information
            record.class_func = self._get_caller_info()
            
            # Handle the record
            self.logger.handle(record)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message with class.function prefix"""
        self._log(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message with class.function prefix"""
        self._log(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message with class.function prefix"""
        self._log(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, exc_info=None, **kwargs):
        """Log error message with class.function prefix"""
        # Create log record with exception info
        if self.logger.isEnabledFor(logging.ERROR):
            record = self.logger.makeRecord(
                self.logger.name, logging.ERROR, "", 0, message, args, exc_info
            )
            record.class_func = self._get_caller_info()
            self.logger.handle(record)
    
    def critical(self, message: str, *args, exc_info=None, **kwargs):
        """Log critical message with class.function prefix"""
        if self.logger.isEnabledFor(logging.CRITICAL):
            record = self.logger.makeRecord(
                self.logger.name, logging.CRITICAL, "", 0, message, args, exc_info
            )
            record.class_func = self._get_caller_info()
            self.logger.handle(record)
    
    def log_startup(self, component: str, **kwargs):
        """Log startup information with key-value pairs"""
        kv_pairs = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        self.info(f"Starting component '{component}' with config: {kv_pairs}")
    
    def log_config_change(self, component: str, old_value: Any, new_value: Any, field: str):
        """Log configuration changes"""
        self.info(f"Config change in '{component}': {field} changed from '{old_value}' to '{new_value}'")
    
    def log_operation_start(self, operation: str, **kwargs):
        """Log the start of an operation with context"""
        kv_pairs = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        self.info(f"=== Starting operation '{operation}' with params: {kv_pairs} ===")
    
    def log_operation_success(self, operation: str, duration: Optional[float] = None, **kwargs):
        """Log successful completion of an operation"""
        extra_info = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        duration_str = f" in {duration:.2f}s" if duration else ""
        self.info(f"Operation '{operation}' completed successfully{duration_str}. {extra_info}")
    
    def log_operation_failure(self, operation: str, error: str, **kwargs):
        """Log failed operation"""
        extra_info = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        self.error(f"Operation '{operation}' failed: {error}. Context: {extra_info}")
    
    def log_resource_usage(self, component: str, **metrics):
        """Log resource usage metrics"""
        metric_str = ', '.join([f"{k}={v}" for k, v in metrics.items()])
        self.info(f"Resource usage for '{component}': {metric_str}")
    
    def log_api_call(self, method: str, endpoint: str, status_code: int, duration: float, **kwargs):
        """Log API calls with timing and status"""
        extra_info = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        self.info(f"API {method} {endpoint} -> {status_code} in {duration:.3f}s. {extra_info}")


# Create the global enhanced logger instance
enhanced_logger = EnhancedLogger()

# Suppress other loggers (like uvicorn)
def configure_logging(suppress_uvicorn=True, suppress_other_loggers=True):
    """Configure logging to suppress other loggers and only show our enhanced logger"""
    if suppress_uvicorn:
        # Configure uvicorn's loggers to a higher level
        for logger_name in ['uvicorn', 'uvicorn.error', 'uvicorn.access']:
            uvicorn_logger = logging.getLogger(logger_name)
            uvicorn_logger.setLevel(logging.WARNING)  # Only show warnings and above
            
            # Remove existing handlers
            for handler in uvicorn_logger.handlers[:]:
                uvicorn_logger.removeHandler(handler)
    
    if suppress_other_loggers:
        # Configure root logger to a higher level
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)  # Only show warnings and above

# Convenience functions for backward compatibility
def info(message: str, *args, **kwargs):
    enhanced_logger.info(message, *args, **kwargs)

def debug(message: str, *args, **kwargs):
    enhanced_logger.debug(message, *args, **kwargs)

def warning(message: str, *args, **kwargs):
    enhanced_logger.warning(message, *args, **kwargs)

def error(message: str, *args, exc_info=None, **kwargs):
    enhanced_logger.error(message, *args, exc_info=exc_info, **kwargs)

def critical(message: str, *args, exc_info=None, **kwargs):
    enhanced_logger.critical(message, *args, exc_info=exc_info, **kwargs)

# Configure logging by default
configure_logging(suppress_uvicorn=True, suppress_other_loggers=True)
