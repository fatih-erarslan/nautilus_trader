"""
Environment Configuration Manager

Handles environment-specific configurations for development, staging, and production
environments with proper isolation and security considerations.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class EnvironmentConfig:
    """Configuration for a specific environment."""
    name: str
    debug: bool
    log_level: str
    database_url: Optional[str] = None
    api_url: Optional[str] = None
    cache_ttl: int = 3600
    max_workers: int = 4
    enable_metrics: bool = True
    security_mode: str = "standard"
    resource_limits: Optional[Dict[str, Any]] = None
    feature_flags: Optional[Dict[str, bool]] = None


class EnvironmentManager:
    """
    Manages environment-specific configurations with secure handling
    and proper isolation between environments.
    """
    
    def __init__(self, config_dir: Optional[str] = None, default_env: str = "development"):
        """
        Initialize environment manager.
        
        Args:
            config_dir: Directory containing environment configuration files
            default_env: Default environment to use
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "environments"
        self.default_env = Environment(default_env)
        self.current_env = None
        self.configs = {}
        self.overrides = {}
        
        # Environment variable for overriding environment
        env_override = os.getenv('NEURAL_FORECAST_ENV', '').lower()
        if env_override and env_override in [e.value for e in Environment]:
            self.current_env = Environment(env_override)
        else:
            self.current_env = self.default_env
        
        self._ensure_config_directory()
        self._load_all_configs()
        
        logger.info(f"Environment manager initialized for: {self.current_env.value}")
    
    def _ensure_config_directory(self):
        """Ensure configuration directory and default files exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default configuration files if they don't exist
        for env in Environment:
            config_file = self.config_dir / f"{env.value}.yaml"
            if not config_file.exists():
                self._create_default_config(env, config_file)
    
    def _create_default_config(self, env: Environment, config_file: Path):
        """Create default configuration file for environment."""
        default_configs = {
            Environment.DEVELOPMENT: {
                'environment': env.value,
                'debug': True,
                'log_level': 'DEBUG',
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'name': 'neural_forecast_dev',
                    'pool_size': 5,
                    'echo': True
                },
                'api': {
                    'host': '0.0.0.0',
                    'port': 8000,
                    'workers': 1,
                    'reload': True,
                    'access_log': True
                },
                'cache': {
                    'ttl': 300,  # 5 minutes
                    'max_size': 1000,
                    'backend': 'memory'
                },
                'gpu': {
                    'enabled': True,
                    'memory_fraction': 0.5,  # Conservative for development
                    'device_ids': [0]
                },
                'model': {
                    'cache_size': 10,
                    'max_steps': 100,  # Faster training for development
                    'batch_size': 16,
                    'early_stopping_patience': 10
                },
                'monitoring': {
                    'enabled': True,
                    'metrics_port': 9090,
                    'collect_gpu_metrics': True
                },
                'security': {
                    'mode': 'development',
                    'cors_origins': ['*'],
                    'api_key_required': False,
                    'rate_limiting': False
                },
                'features': {
                    'auto_optimization': True,
                    'experimental_models': True,
                    'detailed_logging': True
                }
            },
            Environment.STAGING: {
                'environment': env.value,
                'debug': False,
                'log_level': 'INFO',
                'database': {
                    'host': 'staging-db.internal',
                    'port': 5432,
                    'name': 'neural_forecast_staging',
                    'pool_size': 10,
                    'echo': False
                },
                'api': {
                    'host': '0.0.0.0',
                    'port': 8000,
                    'workers': 2,
                    'reload': False,
                    'access_log': True
                },
                'cache': {
                    'ttl': 1800,  # 30 minutes
                    'max_size': 5000,
                    'backend': 'redis'
                },
                'gpu': {
                    'enabled': True,
                    'memory_fraction': 0.7,
                    'device_ids': [0]
                },
                'model': {
                    'cache_size': 50,
                    'max_steps': 500,
                    'batch_size': 32,
                    'early_stopping_patience': 20
                },
                'monitoring': {
                    'enabled': True,
                    'metrics_port': 9090,
                    'collect_gpu_metrics': True,
                    'alert_webhooks': ['${SLACK_WEBHOOK_URL}']
                },
                'security': {
                    'mode': 'staging',
                    'cors_origins': ['https://staging-app.company.com'],
                    'api_key_required': True,
                    'rate_limiting': True,
                    'rate_limit': 100  # requests per minute
                },
                'features': {
                    'auto_optimization': True,
                    'experimental_models': False,
                    'detailed_logging': False
                }
            },
            Environment.PRODUCTION: {
                'environment': env.value,
                'debug': False,
                'log_level': 'WARNING',
                'database': {
                    'host': 'prod-db.internal',
                    'port': 5432,
                    'name': 'neural_forecast_prod',
                    'pool_size': 20,
                    'echo': False,
                    'ssl_mode': 'require'
                },
                'api': {
                    'host': '0.0.0.0',
                    'port': 8000,
                    'workers': 8,
                    'reload': False,
                    'access_log': False
                },
                'cache': {
                    'ttl': 3600,  # 1 hour
                    'max_size': 10000,
                    'backend': 'redis',
                    'cluster_mode': True
                },
                'gpu': {
                    'enabled': True,
                    'memory_fraction': 0.9,
                    'device_ids': [0, 1, 2, 3],  # Multi-GPU
                    'mixed_precision': True
                },
                'model': {
                    'cache_size': 100,
                    'max_steps': 1000,
                    'batch_size': 64,
                    'early_stopping_patience': 30,
                    'checkpoint_interval': 100
                },
                'monitoring': {
                    'enabled': True,
                    'metrics_port': 9090,
                    'collect_gpu_metrics': True,
                    'alert_webhooks': ['${PROD_ALERT_WEBHOOK}'],
                    'log_sampling_rate': 0.1  # Sample 10% of logs
                },
                'security': {
                    'mode': 'production',
                    'cors_origins': ['https://app.company.com'],
                    'api_key_required': True,
                    'rate_limiting': True,
                    'rate_limit': 1000,  # requests per minute
                    'ssl_required': True,
                    'encryption_at_rest': True
                },
                'features': {
                    'auto_optimization': False,  # Manual optimization in prod
                    'experimental_models': False,
                    'detailed_logging': False
                },
                'backup': {
                    'enabled': True,
                    'interval_hours': 6,
                    'retention_days': 30,
                    's3_bucket': '${BACKUP_S3_BUCKET}'
                }
            },
            Environment.TESTING: {
                'environment': env.value,
                'debug': True,
                'log_level': 'ERROR',  # Minimize test output
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'name': 'neural_forecast_test',
                    'pool_size': 1,
                    'echo': False
                },
                'api': {
                    'host': '127.0.0.1',
                    'port': 0,  # Random port for testing
                    'workers': 1,
                    'reload': False,
                    'access_log': False
                },
                'cache': {
                    'ttl': 60,  # Short TTL for testing
                    'max_size': 100,
                    'backend': 'memory'
                },
                'gpu': {
                    'enabled': False,  # Use CPU for faster test setup
                    'memory_fraction': 0.3,
                    'device_ids': [0]
                },
                'model': {
                    'cache_size': 5,
                    'max_steps': 10,  # Very fast for testing
                    'batch_size': 8,
                    'early_stopping_patience': 3
                },
                'monitoring': {
                    'enabled': False,  # Disable monitoring in tests
                    'metrics_port': 0,
                    'collect_gpu_metrics': False
                },
                'security': {
                    'mode': 'testing',
                    'cors_origins': ['*'],
                    'api_key_required': False,
                    'rate_limiting': False
                },
                'features': {
                    'auto_optimization': False,
                    'experimental_models': True,
                    'detailed_logging': False
                }
            }
        }
        
        config = default_configs.get(env, default_configs[Environment.DEVELOPMENT])
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created default configuration for {env.value}")
    
    def _load_all_configs(self):
        """Load all environment configurations."""
        for env in Environment:
            config_file = self.config_dir / f"{env.value}.yaml"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    self.configs[env] = config
                    logger.debug(f"Loaded configuration for {env.value}")
                except Exception as e:
                    logger.error(f"Error loading config for {env.value}: {e}")
                    self.configs[env] = {}
    
    def get_config(self, environment: Optional[Union[str, Environment]] = None) -> Dict[str, Any]:
        """
        Get configuration for specified environment.
        
        Args:
            environment: Environment to get config for (defaults to current)
            
        Returns:
            Configuration dictionary
        """
        if environment is None:
            environment = self.current_env
        elif isinstance(environment, str):
            environment = Environment(environment)
        
        base_config = self.configs.get(environment, {}).copy()
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(base_config)
        
        # Apply programmatic overrides
        if environment in self.overrides:
            config = self._deep_merge(config, self.overrides[environment])
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        config = copy.deepcopy(config)
        
        # Expand environment variables in string values
        config = self._expand_env_vars(config)
        
        # Specific environment variable overrides
        env_overrides = {
            'DATABASE_URL': ['database', 'url'],
            'API_PORT': ['api', 'port'],
            'LOG_LEVEL': ['log_level'],
            'DEBUG': ['debug'],
            'GPU_ENABLED': ['gpu', 'enabled'],
            'GPU_MEMORY_FRACTION': ['gpu', 'memory_fraction'],
            'CACHE_TTL': ['cache', 'ttl'],
            'API_WORKERS': ['api', 'workers']
        }
        
        for env_var, config_path in env_overrides.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config, config_path, self._convert_type(value))
        
        return config
    
    def _expand_env_vars(self, obj: Any) -> Any:
        """Recursively expand environment variables in configuration."""
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Expand ${VAR} syntax
            import re
            def replace_var(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))
            
            return re.sub(r'\$\{([^}]+)\}', replace_var, obj)
        else:
            return obj
    
    def _convert_type(self, value: str) -> Any:
        """Convert string environment variable to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: List[str], value: Any):
        """Set nested value in configuration dictionary."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def set_environment(self, environment: Union[str, Environment]):
        """
        Set current environment.
        
        Args:
            environment: Environment to switch to
        """
        if isinstance(environment, str):
            environment = Environment(environment)
        
        self.current_env = environment
        logger.info(f"Switched to environment: {environment.value}")
    
    def override_config(self, environment: Union[str, Environment], 
                       config_overrides: Dict[str, Any]):
        """
        Set configuration overrides for an environment.
        
        Args:
            environment: Environment to override
            config_overrides: Configuration overrides
        """
        if isinstance(environment, str):
            environment = Environment(environment)
        
        self.overrides[environment] = config_overrides
        logger.info(f"Set configuration overrides for {environment.value}")
    
    def get_current_environment(self) -> Environment:
        """Get current environment."""
        return self.current_env
    
    def get_available_environments(self) -> List[str]:
        """Get list of available environments."""
        return [env.value for env in Environment]
    
    def validate_environment_config(self, environment: Optional[Union[str, Environment]] = None) -> Dict[str, Any]:
        """
        Validate configuration for specified environment.
        
        Args:
            environment: Environment to validate (defaults to current)
            
        Returns:
            Validation results
        """
        if environment is None:
            environment = self.current_env
        elif isinstance(environment, str):
            environment = Environment(environment)
        
        config = self.get_config(environment)
        errors = []
        warnings = []
        
        # Required fields validation
        required_fields = ['environment', 'log_level']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Environment-specific validation
        if environment == Environment.PRODUCTION:
            prod_requirements = [
                ('security.api_key_required', True),
                ('security.ssl_required', True),
                ('debug', False),
                ('monitoring.enabled', True)
            ]
            
            for field_path, expected_value in prod_requirements:
                actual_value = self._get_nested_value(config, field_path.split('.'))
                if actual_value != expected_value:
                    errors.append(f"Production requirement not met: {field_path} should be {expected_value}")
        
        # Security validation
        if config.get('debug', False) and environment == Environment.PRODUCTION:
            errors.append("Debug mode should not be enabled in production")
        
        # GPU configuration validation
        if config.get('gpu', {}).get('enabled', False):
            gpu_config = config['gpu']
            if gpu_config.get('memory_fraction', 0) > 0.95:
                warnings.append("GPU memory fraction > 95% may cause out-of-memory errors")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'environment': environment.value,
            'config_file': str(self.config_dir / f"{environment.value}.yaml")
        }
    
    def _get_nested_value(self, config: Dict[str, Any], path: List[str]) -> Any:
        """Get nested value from configuration."""
        current = config
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def export_config(self, environment: Optional[Union[str, Environment]] = None, 
                     output_file: Optional[str] = None) -> str:
        """
        Export configuration to file.
        
        Args:
            environment: Environment to export (defaults to current)
            output_file: Output file path
            
        Returns:
            Path to exported file
        """
        if environment is None:
            environment = self.current_env
        elif isinstance(environment, str):
            environment = Environment(environment)
        
        config = self.get_config(environment)
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"config_export_{environment.value}_{timestamp}.yaml"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Exported configuration for {environment.value} to {output_path}")
        return str(output_path)
    
    def reload_configs(self):
        """Reload all configuration files."""
        self.configs.clear()
        self._load_all_configs()
        logger.info("Reloaded all configuration files")