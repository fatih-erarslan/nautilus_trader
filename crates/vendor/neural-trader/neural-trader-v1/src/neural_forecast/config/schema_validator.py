"""
Configuration Schema Validator

Provides comprehensive validation for neural forecasting configuration using JSON Schema
with custom validation rules for financial modeling parameters.
"""

import json
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator
from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path
import yaml
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class ConfigSchemaValidator:
    """
    Validates neural forecasting configurations against predefined schemas.
    Supports validation for model parameters, GPU settings, security configs, etc.
    """
    
    def __init__(self, schema_dir: Optional[str] = None):
        """
        Initialize schema validator.
        
        Args:
            schema_dir: Directory containing schema files
        """
        self.schema_dir = Path(schema_dir) if schema_dir else Path(__file__).parent / "schemas"
        self.schemas = {}
        self.custom_validators = {}
        self._load_schemas()
        self._register_custom_validators()
    
    def _load_schemas(self):
        """Load all schema files from the schema directory."""
        if not self.schema_dir.exists():
            logger.warning(f"Schema directory not found: {self.schema_dir}")
            return
        
        for schema_file in self.schema_dir.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema_name = schema_file.stem
                    self.schemas[schema_name] = json.load(f)
                    logger.debug(f"Loaded schema: {schema_name}")
            except Exception as e:
                logger.error(f"Error loading schema {schema_file}: {e}")
    
    def _register_custom_validators(self):
        """Register custom validation functions for financial parameters."""
        
        def validate_learning_rate(validator, value, instance, schema):
            """Validate learning rate is in reasonable range for neural networks."""
            if not isinstance(instance, (int, float)):
                yield ValidationError("Learning rate must be numeric")
            elif instance <= 0 or instance >= 1:
                yield ValidationError("Learning rate must be between 0 and 1")
            elif instance < 1e-6:
                yield ValidationError("Learning rate too small (< 1e-6)")
            elif instance > 0.1:
                yield ValidationError("Learning rate unusually high (> 0.1)")
        
        def validate_forecast_horizon(validator, value, instance, schema):
            """Validate forecast horizon is reasonable for financial data."""
            if not isinstance(instance, int):
                yield ValidationError("Forecast horizon must be integer")
            elif instance < 1:
                yield ValidationError("Forecast horizon must be positive")
            elif instance > 168:  # 1 week in hours
                yield ValidationError("Forecast horizon too long (> 168 hours)")
        
        def validate_symbol_format(validator, value, instance, schema):
            """Validate trading symbol format."""
            if not isinstance(instance, str):
                yield ValidationError("Symbol must be string")
            elif not re.match(r'^[A-Z]{1,5}$', instance):
                yield ValidationError("Symbol must be 1-5 uppercase letters")
        
        def validate_gpu_memory(validator, value, instance, schema):
            """Validate GPU memory allocation."""
            if not isinstance(instance, (int, float)):
                yield ValidationError("GPU memory must be numeric")
            elif instance < 0 or instance > 1:
                yield ValidationError("GPU memory fraction must be between 0 and 1")
        
        def validate_batch_size(validator, value, instance, schema):
            """Validate batch size for neural networks."""
            if not isinstance(instance, int):
                yield ValidationError("Batch size must be integer")
            elif instance < 1:
                yield ValidationError("Batch size must be positive")
            elif instance > 1024:
                yield ValidationError("Batch size too large (> 1024)")
            elif instance & (instance - 1) != 0:  # Check if power of 2
                yield ValidationError("Batch size should be power of 2 for optimal GPU performance")
        
        # Register custom validators
        self.custom_validators = {
            'learning_rate': validate_learning_rate,
            'forecast_horizon': validate_forecast_horizon,
            'symbol_format': validate_symbol_format,
            'gpu_memory': validate_gpu_memory,
            'batch_size': validate_batch_size
        }
    
    def validate_config(self, config: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """
        Validate configuration against specified schema.
        
        Args:
            config: Configuration dictionary to validate
            schema_name: Name of schema to validate against
            
        Returns:
            Dictionary with validation results
        """
        if schema_name not in self.schemas:
            return {
                'valid': False,
                'errors': [f"Schema '{schema_name}' not found"],
                'warnings': []
            }
        
        schema = self.schemas[schema_name]
        errors = []
        warnings = []
        
        try:
            # Create validator with custom validators
            validator_class = jsonschema.validators.validator_for(schema)
            validator_class.check_schema(schema)
            
            # Add custom validators
            all_validators = dict(validator_class.META_SCHEMA.get('validators', {}))
            for name, func in self.custom_validators.items():
                all_validators[name] = func
            
            CustomValidator = jsonschema.validators.create(
                meta_schema=validator_class.META_SCHEMA,
                validators=all_validators
            )
            
            validator = CustomValidator(schema)
            
            # Validate configuration
            validation_errors = list(validator.iter_errors(config))
            
            if validation_errors:
                for error in validation_errors:
                    error_path = " -> ".join(str(p) for p in error.absolute_path)
                    error_msg = f"{error_path}: {error.message}" if error_path else error.message
                    errors.append(error_msg)
            
            # Additional semantic validation
            semantic_warnings = self._semantic_validation(config, schema_name)
            warnings.extend(semantic_warnings)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'schema_used': schema_name,
            'timestamp': datetime.now().isoformat()
        }
    
    def _semantic_validation(self, config: Dict[str, Any], schema_name: str) -> List[str]:
        """
        Perform semantic validation beyond schema structure.
        
        Args:
            config: Configuration to validate
            schema_name: Schema being used
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        if schema_name == 'model_config':
            # Model-specific semantic validation
            if 'models' in config:
                for model_name, model_config in config['models'].items():
                    # Check for reasonable parameter combinations
                    if 'input_size' in model_config and 'batch_size' in model_config:
                        if model_config['input_size'] < model_config['batch_size']:
                            warnings.append(f"Model {model_name}: input_size should typically be >= batch_size")
                    
                    # Check learning rate vs max_steps
                    if 'learning_rate' in model_config and 'max_steps' in model_config:
                        if model_config['learning_rate'] > 0.01 and model_config['max_steps'] > 1000:
                            warnings.append(f"Model {model_name}: high learning rate with many steps may cause instability")
        
        elif schema_name == 'gpu_config':
            # GPU-specific validation
            if 'memory_fraction' in config and 'enable_growth' in config:
                if config['memory_fraction'] > 0.8 and not config['enable_growth']:
                    warnings.append("High memory fraction without growth enabled may cause OOM errors")
        
        elif schema_name == 'deployment_config':
            # Deployment-specific validation
            if 'environment' in config and 'debug' in config:
                if config['environment'] == 'production' and config['debug']:
                    warnings.append("Debug mode enabled in production environment")
        
        return warnings
    
    def validate_environment_consistency(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate consistency across multiple configuration files.
        
        Args:
            configs: Dictionary of config_name -> config_dict
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Check for environment consistency
        environments = set()
        for config_name, config in configs.items():
            if 'environment' in config:
                environments.add(config['environment'])
        
        if len(environments) > 1:
            errors.append(f"Inconsistent environments found: {environments}")
        
        # Check for GPU configuration consistency
        gpu_enabled_configs = []
        for config_name, config in configs.items():
            if 'gpu' in config and config['gpu'].get('enabled', False):
                gpu_enabled_configs.append(config_name)
        
        if gpu_enabled_configs and 'gpu_config' not in configs:
            warnings.append("GPU enabled in configs but no GPU configuration found")
        
        # Check for model references
        model_references = set()
        defined_models = set()
        
        for config_name, config in configs.items():
            if 'models' in config:
                defined_models.update(config['models'].keys())
            
            # Look for model references in other configs
            if 'default_model' in config:
                model_references.add(config['default_model'])
            if 'ensemble_models' in config:
                model_references.update(config['ensemble_models'])
        
        undefined_models = model_references - defined_models
        if undefined_models:
            errors.append(f"Referenced models not defined: {undefined_models}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'environments_found': list(environments),
            'models_defined': list(defined_models),
            'models_referenced': list(model_references)
        }
    
    def validate_file(self, file_path: str, schema_name: str) -> Dict[str, Any]:
        """
        Validate configuration file.
        
        Args:
            file_path: Path to configuration file
            schema_name: Schema to validate against
            
        Returns:
            Validation results
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {
                    'valid': False,
                    'errors': [f"Configuration file not found: {file_path}"],
                    'warnings': []
                }
            
            # Load configuration based on file extension
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    config = json.load(f)
            elif file_path.suffix.lower() in ['.yml', '.yaml']:
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                return {
                    'valid': False,
                    'errors': [f"Unsupported file format: {file_path.suffix}"],
                    'warnings': []
                }
            
            result = self.validate_config(config, schema_name)
            result['file_path'] = str(file_path)
            
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Error loading configuration file: {str(e)}"],
                'warnings': [],
                'file_path': str(file_path) if 'file_path' in locals() else None
            }
    
    def get_available_schemas(self) -> List[str]:
        """Get list of available schema names."""
        return list(self.schemas.keys())
    
    def get_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get schema by name."""
        return self.schemas.get(schema_name)
    
    def add_custom_validator(self, name: str, validator_func):
        """
        Add custom validator function.
        
        Args:
            name: Name of the validator
            validator_func: Validation function
        """
        self.custom_validators[name] = validator_func
        logger.info(f"Added custom validator: {name}")
    
    def generate_sample_config(self, schema_name: str) -> Dict[str, Any]:
        """
        Generate sample configuration from schema.
        
        Args:
            schema_name: Schema to generate sample for
            
        Returns:
            Sample configuration dictionary
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found")
        
        schema = self.schemas[schema_name]
        return self._generate_from_schema(schema)
    
    def _generate_from_schema(self, schema: Dict[str, Any]) -> Any:
        """Recursively generate sample values from schema."""
        if 'type' not in schema:
            return None
        
        schema_type = schema['type']
        
        if schema_type == 'object':
            result = {}
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            for prop_name, prop_schema in properties.items():
                if prop_name in required or 'default' in prop_schema:
                    if 'default' in prop_schema:
                        result[prop_name] = prop_schema['default']
                    else:
                        result[prop_name] = self._generate_from_schema(prop_schema)
            
            return result
        
        elif schema_type == 'array':
            items_schema = schema.get('items', {})
            min_items = schema.get('minItems', 1)
            return [self._generate_from_schema(items_schema) for _ in range(min_items)]
        
        elif schema_type == 'string':
            if 'default' in schema:
                return schema['default']
            elif 'enum' in schema:
                return schema['enum'][0]
            else:
                return "sample_string"
        
        elif schema_type == 'number':
            if 'default' in schema:
                return schema['default']
            elif 'minimum' in schema and 'maximum' in schema:
                return (schema['minimum'] + schema['maximum']) / 2
            else:
                return 1.0
        
        elif schema_type == 'integer':
            if 'default' in schema:
                return schema['default']
            elif 'minimum' in schema and 'maximum' in schema:
                return int((schema['minimum'] + schema['maximum']) / 2)
            else:
                return 1
        
        elif schema_type == 'boolean':
            return schema.get('default', True)
        
        else:
            return None