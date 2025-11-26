"""
Advanced Logging Configuration for GPU Trading Platform
Provides structured logging with multiple outputs and monitoring integration
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

import structlog
from pythonjsonlogger import jsonlogger


class GPUMetricsFilter(logging.Filter):
    """Custom filter to add GPU metrics to log records"""
    
    def filter(self, record):
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                record.gpu_utilization = gpu.load * 100
                record.gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                record.gpu_temperature = gpu.temperature
            else:
                record.gpu_utilization = 0
                record.gpu_memory_percent = 0
                record.gpu_temperature = 0
        except Exception:
            record.gpu_utilization = 0
            record.gpu_memory_percent = 0
            record.gpu_temperature = 0
        
        return True


class TradingContextFilter(logging.Filter):
    """Add trading-specific context to log records"""
    
    def filter(self, record):
        record.trading_mode = os.getenv('TRADING_MODE', 'unknown')
        record.app_version = os.getenv('APP_VERSION', 'unknown')
        record.instance_id = os.getenv('FLY_MACHINE_ID', 'local')
        record.region = os.getenv('FLY_REGION', 'local')
        record.app_name = os.getenv('FLY_APP_NAME', 'ruvtrade')
        
        return True


class JSONFormatter(jsonlogger.JsonFormatter):
    """Enhanced JSON formatter with additional fields"""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add level
        log_record['level'] = record.levelname.lower()
        
        # Add source information
        log_record['source'] = {
            'file': record.filename,
            'line': record.lineno,
            'function': record.funcName,
            'module': record.module
        }
        
        # Add GPU metrics if available
        if hasattr(record, 'gpu_utilization'):
            log_record['gpu_metrics'] = {
                'utilization_percent': getattr(record, 'gpu_utilization', 0),
                'memory_percent': getattr(record, 'gpu_memory_percent', 0),
                'temperature_celsius': getattr(record, 'gpu_temperature', 0)
            }
        
        # Add trading context
        if hasattr(record, 'trading_mode'):
            log_record['context'] = {
                'trading_mode': getattr(record, 'trading_mode', 'unknown'),
                'app_version': getattr(record, 'app_version', 'unknown'),
                'instance_id': getattr(record, 'instance_id', 'unknown'),
                'region': getattr(record, 'region', 'unknown'),
                'app_name': getattr(record, 'app_name', 'unknown')
            }


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "/app/logs",
    enable_json: bool = True,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_syslog: bool = False,
    max_file_size: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_json: Enable JSON structured logging
        enable_console: Enable console output
        enable_file: Enable file logging
        enable_syslog: Enable syslog output
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    if enable_json:
        json_formatter = JSONFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
        
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        detailed_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)8s] %(name)s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        
        # Add filters
        console_handler.addFilter(TradingContextFilter())
        console_handler.addFilter(GPUMetricsFilter())
        
        root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file:
        # Main application log
        app_log_file = os.path.join(log_dir, 'trading_platform.log')
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        app_handler.setLevel(getattr(logging, log_level.upper()))
        
        if enable_json:
            app_handler.setFormatter(json_formatter)
        else:
            app_handler.setFormatter(detailed_formatter)
        
        app_handler.addFilter(TradingContextFilter())
        app_handler.addFilter(GPUMetricsFilter())
        root_logger.addHandler(app_handler)
        
        # Error log (errors and above only)
        error_log_file = os.path.join(log_dir, 'errors.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        
        if enable_json:
            error_handler.setFormatter(json_formatter)
        else:
            error_handler.setFormatter(detailed_formatter)
        
        error_handler.addFilter(TradingContextFilter())
        error_handler.addFilter(GPUMetricsFilter())
        root_logger.addHandler(error_handler)
        
        # GPU metrics log
        gpu_log_file = os.path.join(log_dir, 'gpu_metrics.log')
        gpu_handler = logging.handlers.RotatingFileHandler(
            gpu_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=3
        )
        gpu_handler.setLevel(logging.INFO)
        
        # Create a separate logger for GPU metrics
        gpu_logger = logging.getLogger('gpu_metrics')
        gpu_logger.setLevel(logging.INFO)
        gpu_logger.addHandler(gpu_handler)
        gpu_handler.setFormatter(json_formatter if enable_json else detailed_formatter)
        gpu_handler.addFilter(GPUMetricsFilter())
    
    # Syslog handler for centralized logging
    if enable_syslog:
        try:
            syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
            syslog_handler.setLevel(logging.INFO)
            
            syslog_formatter = logging.Formatter(
                'ruvtrade[%(process)d]: %(levelname)s %(name)s %(message)s'
            )
            syslog_handler.setFormatter(syslog_formatter)
            syslog_handler.addFilter(TradingContextFilter())
            
            root_logger.addHandler(syslog_handler)
        except Exception as e:
            print(f"Failed to setup syslog handler: {e}")
    
    # Configure specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Create application logger
    app_logger = logging.getLogger('trading_platform')
    
    return app_logger


def setup_structlog():
    """Setup structured logging with structlog"""
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class MetricsLogger:
    """Specialized logger for metrics and monitoring data"""
    
    def __init__(self, logger_name: str = "metrics"):
        self.logger = logging.getLogger(logger_name)
        self.struct_logger = structlog.get_logger(logger_name)
    
    def log_gpu_metrics(self, gpu_data: Dict[str, Any]):
        """Log GPU performance metrics"""
        self.struct_logger.info(
            "GPU metrics recorded",
            metric_type="gpu_performance",
            **gpu_data
        )
    
    def log_trading_metrics(self, trading_data: Dict[str, Any]):
        """Log trading performance metrics"""
        self.struct_logger.info(
            "Trading metrics recorded",
            metric_type="trading_performance",
            **trading_data
        )
    
    def log_system_metrics(self, system_data: Dict[str, Any]):
        """Log system performance metrics"""
        self.struct_logger.info(
            "System metrics recorded",
            metric_type="system_performance",
            **system_data
        )
    
    def log_health_check(self, health_data: Dict[str, Any]):
        """Log health check results"""
        self.struct_logger.info(
            "Health check completed",
            metric_type="health_check",
            **health_data
        )
    
    def log_error_metrics(self, error_data: Dict[str, Any]):
        """Log error and exception metrics"""
        self.struct_logger.error(
            "Error metrics recorded",
            metric_type="error_tracking",
            **error_data
        )


def get_log_config_for_environment() -> Dict[str, Any]:
    """Get logging configuration based on environment"""
    
    env = os.getenv('ENVIRONMENT', 'development')
    
    if env == 'production':
        return {
            'log_level': 'INFO',
            'enable_json': True,
            'enable_console': True,
            'enable_file': True,
            'enable_syslog': True,
            'log_dir': '/app/logs'
        }
    elif env == 'staging':
        return {
            'log_level': 'INFO',
            'enable_json': True,
            'enable_console': True,
            'enable_file': True,
            'enable_syslog': False,
            'log_dir': '/app/logs'
        }
    else:  # development
        return {
            'log_level': 'DEBUG',
            'enable_json': False,
            'enable_console': True,
            'enable_file': True,
            'enable_syslog': False,
            'log_dir': './logs'
        }


if __name__ == "__main__":
    # Test logging configuration
    config = get_log_config_for_environment()
    logger = setup_logging(**config)
    setup_structlog()
    
    # Test logging
    logger.info("Testing logging configuration")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test structured logging
    struct_logger = structlog.get_logger("test")
    struct_logger.info("Structured log test", user_id=123, action="test")
    
    # Test metrics logging
    metrics = MetricsLogger()
    metrics.log_gpu_metrics({
        "utilization": 75.5,
        "memory_used": 16.2,
        "temperature": 68
    })
    
    print("Logging configuration test completed")