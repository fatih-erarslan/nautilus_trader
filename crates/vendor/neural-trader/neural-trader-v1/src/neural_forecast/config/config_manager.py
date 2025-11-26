"""
Central Configuration Manager

Provides unified configuration management for the neural forecasting system,
integrating all configuration components including models, GPU, secrets, and environments.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from datetime import datetime
from dataclasses import asdict

from .schema_validator import ConfigSchemaValidator
from .environment_manager import EnvironmentManager, Environment
from .secrets_manager import SecretsManager
from .model_config import ModelConfigManager, ModelType, OptimizationPreset
from .gpu_config import GPUConfigManager

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Central configuration manager that coordinates all configuration subsystems
    and provides a unified interface for the neural forecasting system.
    """
    
    def __init__(self, 
                 config_dir: Optional[str] = None,
                 environment: str = "development",
                 auto_detect_gpu: bool = True,
                 enable_secrets: bool = True):
        """
        Initialize the central configuration manager.
        
        Args:
            config_dir: Root configuration directory
            environment: Current environment (development, staging, production)
            auto_detect_gpu: Automatically detect and configure GPU
            enable_secrets: Enable secrets management
        """
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".neural_forecast"
        self.current_environment = environment
        
        # Initialize subsystem managers
        self.schema_validator = ConfigSchemaValidator(
            schema_dir=str(self.config_dir / "schemas")
        )
        
        self.environment_manager = EnvironmentManager(
            config_dir=str(self.config_dir / "environments"),
            default_env=environment
        )
        
        self.model_config_manager = ModelConfigManager(
            templates_dir=str(self.config_dir / "model_templates")
        )
        
        if auto_detect_gpu:
            self.gpu_config_manager = GPUConfigManager(
                config_file=str(self.config_dir / "gpu_config.json")
            )
        else:
            self.gpu_config_manager = None
        
        if enable_secrets:
            self.secrets_manager = SecretsManager(
                secrets_dir=str(self.config_dir / "secrets")
            )
        else:
            self.secrets_manager = None
        
        # Configuration cache
        self._config_cache = {}
        self._cache_timestamp = None
        
        # Setup configuration directory structure
        self._setup_config_directory()
        
        # Load and validate configurations
        self._load_configurations()
        
        logger.info(f"Configuration manager initialized for environment: {environment}")
    
    def _setup_config_directory(self):
        """Setup the configuration directory structure."""
        directories = [
            self.config_dir,
            self.config_dir / "schemas",
            self.config_dir / "environments", 
            self.config_dir / "model_templates",
            self.config_dir / "secrets",
            self.config_dir / "deployments",
            self.config_dir / "monitoring",
            self.config_dir / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create default schema files if they don't exist
        self._create_default_schemas()
    
    def _create_default_schemas(self):
        """Create default schema files."""
        # Copy schema files from package to config directory
        schemas_dir = self.config_dir / "schemas"
        
        # Check if we need to copy from package schemas
        package_schemas_dir = Path(__file__).parent / "schemas"
        if package_schemas_dir.exists():
            for schema_file in package_schemas_dir.glob("*.json"):
                target_file = schemas_dir / schema_file.name
                if not target_file.exists():
                    import shutil
                    shutil.copy2(schema_file, target_file)
                    logger.debug(f"Copied schema: {schema_file.name}")
    
    def _load_configurations(self):
        """Load and cache all configurations."""
        try:
            # Get environment configuration
            env_config = self.environment_manager.get_config()
            
            # Get GPU configuration if available
            gpu_config = {}
            if self.gpu_config_manager:
                gpu_config = asdict(self.gpu_config_manager.get_current_config())
            
            # Combine configurations
            self._config_cache = {
                'environment': env_config,
                'gpu': gpu_config,
                'timestamp': datetime.now().isoformat()
            }
            
            self._cache_timestamp = datetime.now()
            
            logger.debug("Loaded and cached configurations")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
    
    def get_config(self, section: Optional[str] = None, refresh: bool = False) -> Dict[str, Any]:
        """
        Get configuration for specified section or all configurations.
        
        Args:
            section: Configuration section (environment, gpu, models, etc.)
            refresh: Force refresh from source
            
        Returns:
            Configuration dictionary
        """
        if refresh or not self._config_cache:
            self._load_configurations()
        
        if section is None:
            return self._config_cache
        
        return self._config_cache.get(section, {})
    
    def create_model_config(self, 
                          model_type: Union[str, ModelType],
                          preset: Union[str, OptimizationPreset] = OptimizationPreset.BALANCED,
                          **kwargs) -> Dict[str, Any]:
        """
        Create model configuration with automatic optimization.
        
        Args:
            model_type: Type of model
            preset: Optimization preset
            **kwargs: Additional configuration overrides
            
        Returns:
            Complete model configuration
        """
        # Get base model configuration
        model_config = self.model_config_manager.create_model_config(
            model_type=model_type,
            preset=preset,
            overrides=kwargs
        )
        
        # Apply GPU optimizations if available
        if self.gpu_config_manager:
            gpu_suggestions = self.gpu_config_manager.optimize_for_model(model_config)
            
            # Apply GPU-specific model optimizations
            if gpu_suggestions.get('batch_size_scaling'):
                # Adjust batch size based on available GPU memory
                gpu_info = self.gpu_config_manager.get_gpu_info(0)
                if gpu_info and gpu_info.memory_total > 8000:  # > 8GB
                    model_config['batch_size'] = min(model_config['batch_size'] * 2, 128)
                elif gpu_info and gpu_info.memory_total < 4000:  # < 4GB
                    model_config['batch_size'] = max(model_config['batch_size'] // 2, 8)
        
        # Apply environment-specific adjustments
        env_config = self.environment_manager.get_config()
        environment = env_config.get('environment', 'development')
        
        if environment == 'development':
            # Faster training for development
            model_config['max_steps'] = min(model_config.get('max_steps', 500), 200)
            model_config['early_stopping_patience'] = 10
        elif environment == 'production':
            # More robust training for production
            model_config['save_checkpoints'] = True
            model_config['checkpoint_interval'] = 50
            model_config['early_stopping_patience'] = max(model_config.get('early_stopping_patience', 20), 30)
        
        return model_config
    
    def create_ensemble_config(self, 
                             models: List[str],
                             horizon: int = 24,
                             **kwargs) -> Dict[str, Any]:
        """
        Create ensemble configuration.
        
        Args:
            models: List of model types
            horizon: Forecast horizon
            **kwargs: Additional configuration
            
        Returns:
            Ensemble configuration
        """
        # Convert model names to (model_type, preset) tuples
        model_presets = []
        for model in models:
            if isinstance(model, str):
                # Use balanced preset by default
                model_presets.append((model, OptimizationPreset.BALANCED))
            else:
                model_presets.append(model)
        
        ensemble_config = self.model_config_manager.create_ensemble_config(
            models=model_presets,
            horizon=horizon,
            **kwargs
        )
        
        # Add environment and GPU configurations
        ensemble_config['environment'] = self.environment_manager.get_config()
        
        if self.gpu_config_manager:
            ensemble_config['gpu'] = asdict(self.gpu_config_manager.get_current_config())
        
        return ensemble_config
    
    def validate_configuration(self, config: Dict[str, Any], 
                             config_type: str = "model_config") -> Dict[str, Any]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            config_type: Type of configuration (model_config, gpu_config, etc.)
            
        Returns:
            Validation results
        """
        return self.schema_validator.validate_config(config, config_type)
    
    def store_secret(self, name: str, value: Any, **kwargs) -> bool:
        """
        Store a secret securely.
        
        Args:
            name: Secret name
            value: Secret value
            **kwargs: Additional options
            
        Returns:
            True if stored successfully
        """
        if not self.secrets_manager:
            logger.error("Secrets manager not initialized")
            return False
        
        environment = self.environment_manager.get_current_environment().value
        return self.secrets_manager.store_secret(
            name=name,
            value=value,
            environment=environment,
            **kwargs
        )
    
    def get_secret(self, name: str, default: Any = None) -> Any:
        """
        Retrieve a secret.
        
        Args:
            name: Secret name
            default: Default value if not found
            
        Returns:
            Secret value or default
        """
        if not self.secrets_manager:
            logger.warning("Secrets manager not initialized")
            return default
        
        return self.secrets_manager.get_secret(name, default)
    
    def switch_environment(self, environment: str) -> bool:
        """
        Switch to a different environment.
        
        Args:
            environment: Target environment
            
        Returns:
            True if switched successfully
        """
        try:
            self.environment_manager.set_environment(environment)
            self.current_environment = environment
            
            # Clear cache to force reload
            self._config_cache.clear()
            self._load_configurations()
            
            logger.info(f"Switched to environment: {environment}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching environment: {e}")
            return False
    
    def get_recommended_model_config(self, 
                                   use_case: str,
                                   data_characteristics: Optional[Dict[str, Any]] = None,
                                   constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get recommended model configuration for use case.
        
        Args:
            use_case: Description of use case
            data_characteristics: Data statistics and characteristics
            constraints: Resource/performance constraints
            
        Returns:
            Recommended configuration
        """
        # Get base recommendation
        recommendation = self.model_config_manager.get_recommended_config(
            use_case=use_case,
            constraints=constraints
        )
        
        # Apply data-specific optimizations if provided
        if data_characteristics:
            optimized_config = self.model_config_manager.optimize_config_for_data(
                model_type=recommendation['recommended_model'],
                data_characteristics=data_characteristics,
                performance_target=constraints.get('performance_target', 'balanced') if constraints else 'balanced'
            )
            recommendation['config'] = optimized_config
        
        # Add environment and GPU context
        recommendation['environment'] = self.current_environment
        recommendation['gpu_available'] = len(self.gpu_config_manager.detected_gpus) > 0 if self.gpu_config_manager else False
        
        return recommendation
    
    def export_configuration(self, 
                           output_file: Optional[str] = None,
                           include_secrets: bool = False) -> str:
        """
        Export complete configuration to file.
        
        Args:
            output_file: Output file path
            include_secrets: Include secrets in export (DANGEROUS)
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"neural_forecast_config_export_{timestamp}.yaml"
        
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'environment': self.current_environment,
                'include_secrets': include_secrets
            },
            'environment_config': self.environment_manager.get_config(),
            'model_templates': self.model_config_manager.get_available_templates(),
        }
        
        # Add GPU configuration if available
        if self.gpu_config_manager:
            export_data['gpu_config'] = asdict(self.gpu_config_manager.get_current_config())
            export_data['gpu_info'] = [asdict(gpu) for gpu in self.gpu_config_manager.detected_gpus]
        
        # Add secrets if requested
        if include_secrets and self.secrets_manager:
            export_data['secrets'] = self.secrets_manager.list_secrets()
            logger.warning("Exporting configuration with secrets - ensure file security!")
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Exported configuration to {output_path}")
        return str(output_path)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of configuration system.
        
        Returns:
            Health check results
        """
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Check environment manager
        try:
            env_validation = self.environment_manager.validate_environment_config()
            health_status['components']['environment_manager'] = {
                'status': 'healthy' if env_validation['valid'] else 'unhealthy',
                'details': env_validation
            }
        except Exception as e:
            health_status['components']['environment_manager'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Check GPU manager
        if self.gpu_config_manager:
            try:
                gpu_info = self.gpu_config_manager.get_gpu_info()
                health_status['components']['gpu_manager'] = {
                    'status': 'healthy',
                    'gpus_detected': len(gpu_info),
                    'gpu_info': [asdict(gpu) for gpu in gpu_info]
                }
            except Exception as e:
                health_status['components']['gpu_manager'] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Check secrets manager
        if self.secrets_manager:
            try:
                secrets_integrity = self.secrets_manager.validate_secrets_integrity()
                health_status['components']['secrets_manager'] = {
                    'status': 'healthy' if secrets_integrity['integrity_score'] > 0.9 else 'degraded',
                    'details': secrets_integrity
                }
            except Exception as e:
                health_status['components']['secrets_manager'] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Check model config manager
        try:
            available_templates = self.model_config_manager.get_available_templates()
            health_status['components']['model_config_manager'] = {
                'status': 'healthy',
                'templates_available': len(available_templates)
            }
        except Exception as e:
            health_status['components']['model_config_manager'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Determine overall status
        component_statuses = [comp['status'] for comp in health_status['components'].values()]
        if 'error' in component_statuses:
            health_status['overall_status'] = 'unhealthy'
        elif 'degraded' in component_statuses:
            health_status['overall_status'] = 'degraded'
        
        return health_status
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        system_info = {
            'config_manager': {
                'version': '1.0.0',
                'environment': self.current_environment,
                'config_directory': str(self.config_dir)
            }
        }
        
        # Add GPU system info if available
        if self.gpu_config_manager:
            system_info['gpu'] = self.gpu_config_manager.get_system_info()
        
        # Add environment info
        system_info['environment'] = {
            'current': self.current_environment,
            'available': self.environment_manager.get_available_environments()
        }
        
        return system_info