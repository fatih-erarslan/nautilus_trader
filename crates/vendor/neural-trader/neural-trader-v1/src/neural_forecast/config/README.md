# Neural Forecast Configuration Management System

## Overview

The Neural Forecast Configuration Management System provides comprehensive, secure, and flexible configuration management for neural forecasting applications. It integrates model configurations, GPU optimization, secrets management, and environment-specific settings into a unified, easy-to-use system.

## Features

### 🔧 **Core Components**
- **Unified Configuration Manager**: Central hub for all configuration needs
- **Schema Validation**: JSON Schema-based validation with custom validators
- **Environment Management**: Dev/staging/production environment isolation
- **Model Templates**: Pre-built configurations for NHITS, NBEATS, TFT, PatchTST
- **GPU Configuration**: Automatic hardware detection and optimization
- **Secrets Management**: Encrypted storage for API keys and credentials
- **Health Monitoring**: System health checks and validation

### 🚀 **Key Benefits**
- **Type Safety**: Comprehensive validation prevents configuration errors
- **Security**: Encrypted secrets management with audit logging
- **Performance**: GPU-optimized configurations for maximum throughput
- **Flexibility**: Environment-specific configurations with easy switching
- **Scalability**: Support for single models to complex ensembles
- **Maintainability**: Clear separation of concerns and modular design

## Quick Start

### Installation

```python
# The configuration system is part of the neural forecast package
from neural_forecast.config import ConfigManager
```

### Basic Usage

```python
# Initialize configuration manager
config_manager = ConfigManager(
    environment="development",
    auto_detect_gpu=True,
    enable_secrets=True
)

# Create model configuration
model_config = config_manager.create_model_config(
    model_type="NHITS",
    preset="balanced"
)

# Store API credentials securely
config_manager.store_secret("api_key", "your-secret-key")

# Get optimized ensemble configuration
ensemble_config = config_manager.create_ensemble_config(
    models=["NBEATS", "NHITS"],
    horizon=24
)
```

## Architecture

```
ConfigManager (Central Hub)
├── SchemaValidator     - JSON Schema validation
├── EnvironmentManager  - Environment configurations
├── ModelConfigManager  - Model templates and optimization
├── GPUConfigManager    - Hardware detection and optimization
├── SecretsManager      - Encrypted secrets storage
└── Examples           - Usage patterns and best practices
```

## Detailed Components

### 1. Configuration Manager (`config_manager.py`)

The central hub that coordinates all configuration subsystems.

```python
from neural_forecast.config import ConfigManager

# Initialize with custom settings
config_manager = ConfigManager(
    config_dir="/path/to/config",
    environment="production",
    auto_detect_gpu=True,
    enable_secrets=True
)

# Get system information
system_info = config_manager.get_system_info()

# Health check
health_status = config_manager.health_check()
```

**Key Methods:**
- `create_model_config()` - Create optimized model configurations
- `create_ensemble_config()` - Configure model ensembles
- `switch_environment()` - Change deployment environment
- `export_configuration()` - Export complete configuration
- `health_check()` - System health validation

### 2. Schema Validator (`schema_validator.py`)

Provides comprehensive JSON Schema validation with custom validators for financial parameters.

```python
from neural_forecast.config import ConfigSchemaValidator

validator = ConfigSchemaValidator()

# Validate configuration
result = validator.validate_config(config, "model_config")

# Generate sample configuration
sample = validator.generate_sample_config("model_config")
```

**Custom Validators:**
- `learning_rate` - Validates neural network learning rates
- `forecast_horizon` - Validates forecast time horizons
- `gpu_memory` - Validates GPU memory allocation
- `batch_size` - Validates and suggests optimal batch sizes

### 3. Environment Manager (`environment_manager.py`)

Manages environment-specific configurations with secure isolation.

```python
from neural_forecast.config import EnvironmentManager

env_manager = EnvironmentManager()

# Switch environments
env_manager.set_environment("production")

# Get environment-specific config
config = env_manager.get_config("staging")

# Validate environment configuration
validation = env_manager.validate_environment_config()
```

**Supported Environments:**
- **Development**: Fast iteration, debug mode, reduced resource usage
- **Staging**: Production-like testing environment
- **Production**: Optimized for performance, security, and reliability
- **Testing**: Minimal resource usage for automated testing

### 4. Model Configuration Manager (`model_config.py`)

Provides model templates and automatic parameter optimization.

```python
from neural_forecast.config import ModelConfigManager, ModelType, OptimizationPreset

model_manager = ModelConfigManager()

# Get recommended configuration
recommendation = model_manager.get_recommended_config(
    use_case="real_time_trading",
    constraints={"max_memory_gb": 4}
)

# Create data-optimized configuration
optimized_config = model_manager.optimize_config_for_data(
    model_type=ModelType.NHITS,
    data_characteristics={
        "data_points": 10000,
        "seasonality_strength": 0.7,
        "frequency": "H"
    }
)
```

**Available Models:**
- **NBEATS**: Neural basis expansion analysis for forecasting
- **NHITS**: Neural hierarchical interpolation for time series
- **TFT**: Temporal Fusion Transformer
- **PatchTST**: Patch-based Time Series Transformer

**Optimization Presets:**
- **Fast**: Quick training, good for development and testing
- **Balanced**: Good balance of speed and accuracy
- **Accurate**: Maximum accuracy, longer training time
- **Production**: Optimized for deployment with checkpointing

### 5. GPU Configuration Manager (`gpu_config.py`)

Automatic hardware detection and GPU optimization.

```python
from neural_forecast.config import GPUConfigManager

gpu_manager = GPUConfigManager()

# Get detected hardware
gpu_info = gpu_manager.get_gpu_info()

# Optimize for specific model
optimizations = gpu_manager.optimize_for_model(model_config)

# Run performance benchmark
benchmark = gpu_manager.benchmark_gpu(duration_seconds=30)
```

**GPU Features:**
- **Hardware Detection**: Automatic NVIDIA/AMD/Intel GPU detection
- **Memory Optimization**: Dynamic memory allocation and growth
- **Mixed Precision**: FP16/FP32 optimization for supported hardware
- **Multi-GPU Support**: Data parallel and model parallel strategies
- **Performance Profiling**: Built-in benchmarking and monitoring

### 6. Secrets Manager (`secrets_manager.py`)

Secure storage for API keys, credentials, and sensitive configuration data.

```python
from neural_forecast.config import SecretsManager

secrets_manager = SecretsManager(encryption_enabled=True)

# Store secrets securely
secrets_manager.store_secret(
    name="openai_api_key",
    value="sk-...",
    expires_in_days=90,
    environment="production"
)

# Retrieve secrets
api_key = secrets_manager.get_secret("openai_api_key")

# List available secrets
secrets_list = secrets_manager.list_secrets(environment="production")
```

**Security Features:**
- **Encryption**: AES-256 encryption for stored secrets
- **Access Control**: Environment-based access restrictions
- **Audit Logging**: Complete audit trail of secret access
- **Expiration**: Time-based secret expiration
- **Key Rotation**: Support for secret rotation and versioning

## Configuration Schemas

### Model Configuration Schema

```yaml
# Example model configuration
version: "1.0.0"
environment: "production"
models:
  NHITS:
    input_size: 168        # 1 week of hourly data
    h: 24                  # 24-hour forecast
    max_steps: 1000
    learning_rate: 0.001
    batch_size: 32
    n_freq_downsample: [24, 12, 1]
    dropout: 0.1
ensemble:
  enabled: true
  models: ["NBEATS", "NHITS"]
  weighting_strategy: "performance"
training:
  frequency: "H"
  validation_split: 0.2
  early_stopping: true
```

### GPU Configuration Schema

```yaml
# Example GPU configuration
version: "1.0.0"
gpu:
  enabled: true
  device_ids: [0]
  memory_fraction: 0.8
  mixed_precision: true
  benchmark_mode: true
optimization:
  tensor_cores: true
  kernel_fusion: true
  batch_size_scaling: true
monitoring:
  enabled: true
  metrics: ["utilization", "memory", "temperature"]
  sampling_interval: 10
```

## Environment Configuration

### Development Environment

```yaml
environment: development
debug: true
log_level: DEBUG
gpu:
  memory_fraction: 0.5
  mixed_precision: false
model:
  max_steps: 200
  early_stopping_patience: 10
monitoring:
  enabled: true
  detailed_logging: true
```

### Production Environment

```yaml
environment: production
debug: false
log_level: WARNING
gpu:
  memory_fraction: 0.9
  mixed_precision: true
model:
  max_steps: 1000
  save_checkpoints: true
  checkpoint_interval: 100
security:
  api_key_required: true
  ssl_required: true
  encryption_at_rest: true
monitoring:
  enabled: true
  alert_webhooks: ["${PROD_ALERT_WEBHOOK}"]
```

## Usage Patterns

### 1. Real-Time Trading

```python
# High-frequency trading configuration
config_manager = ConfigManager(environment="production")

model_config = config_manager.create_model_config(
    model_type="NHITS",
    preset="fast",
    horizon=6,  # 6-hour forecast
    overrides={
        "batch_size": 128,
        "max_steps": 150
    }
)
```

### 2. Research and Analysis

```python
# Research configuration with maximum accuracy
config_manager = ConfigManager(environment="development")

ensemble_config = config_manager.create_ensemble_config(
    models=["NBEATS", "NHITS", "TFT"],
    horizon=24,
    weighting_strategy="performance"
)
```

### 3. Production Deployment

```python
# Production deployment with monitoring
config_manager = ConfigManager(environment="production")

# Store deployment secrets
config_manager.store_secret("database_url", "postgresql://...")
config_manager.store_secret("monitoring_api_key", "key-...")

# Create production-optimized configuration
model_config = config_manager.create_model_config(
    model_type="NHITS",
    preset="production"
)
```

## Best Practices

### 🔒 **Security**
- Always use secrets manager for sensitive data
- Never commit secrets to version control
- Rotate secrets regularly
- Use environment-specific access controls
- Enable audit logging in production

### ⚡ **Performance**
- Use GPU acceleration when available
- Enable mixed precision for compatible hardware
- Optimize batch sizes for your hardware
- Use ensemble models for critical applications
- Monitor GPU utilization and memory usage

### 🏗️ **Architecture**
- Separate configurations by environment
- Use schema validation to prevent errors
- Implement proper error handling
- Monitor system health regularly
- Plan for configuration migration

### 🧪 **Testing**
- Validate configurations before deployment
- Test with different hardware configurations
- Use development environment for experimentation
- Implement configuration regression tests
- Document configuration changes

## Troubleshooting

### Common Issues

**GPU Not Detected**
```python
# Check GPU availability
system_info = config_manager.get_system_info()
print(system_info['gpu'])

# Force CPU fallback
config_manager.gpu_config_manager.update_config(enabled=False)
```

**Configuration Validation Errors**
```python
# Validate configuration
validation = config_manager.validate_configuration(config, "model_config")
if not validation['valid']:
    print("Errors:", validation['errors'])
    print("Warnings:", validation['warnings'])
```

**Environment Issues**
```python
# Check environment status
env_validation = config_manager.environment_manager.validate_environment_config()
print("Environment valid:", env_validation['valid'])
```

**Secrets Access Issues**
```python
# Check secrets integrity
if config_manager.secrets_manager:
    integrity = config_manager.secrets_manager.validate_secrets_integrity()
    print("Secrets integrity:", integrity['integrity_score'])
```

### Health Monitoring

```python
# Comprehensive health check
health_status = config_manager.health_check()
print(f"Overall Status: {health_status['overall_status']}")

for component, status in health_status['components'].items():
    print(f"{component}: {status['status']}")
    if status['status'] != 'healthy':
        print(f"  Issues: {status.get('details', {})}")
```

## API Reference

### ConfigManager

| Method | Description | Parameters |
|--------|-------------|------------|
| `create_model_config()` | Create optimized model configuration | `model_type`, `preset`, `**kwargs` |
| `create_ensemble_config()` | Create ensemble configuration | `models`, `horizon`, `**kwargs` |
| `switch_environment()` | Change deployment environment | `environment` |
| `store_secret()` | Store encrypted secret | `name`, `value`, `**kwargs` |
| `get_secret()` | Retrieve secret | `name`, `default` |
| `health_check()` | System health validation | None |
| `export_configuration()` | Export complete configuration | `output_file`, `include_secrets` |

### ModelConfigManager

| Method | Description | Parameters |
|--------|-------------|------------|
| `get_template()` | Get model template | `model_type`, `preset` |
| `get_recommended_config()` | Get recommended configuration | `use_case`, `constraints` |
| `optimize_config_for_data()` | Optimize for data characteristics | `model_type`, `data_characteristics` |
| `save_custom_template()` | Save custom template | `name`, `config`, `description` |

### GPUConfigManager

| Method | Description | Parameters |
|--------|-------------|------------|
| `get_gpu_info()` | Get GPU hardware information | `device_id` |
| `optimize_for_model()` | Optimize GPU config for model | `model_config` |
| `benchmark_gpu()` | Run GPU performance benchmark | `duration_seconds` |
| `apply_profile()` | Apply performance profile | `profile_name` |

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Follow PEP 8 and use type hints
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update documentation for API changes
4. **Security**: Never commit secrets or sensitive data
5. **Validation**: Ensure all configurations validate against schemas

## License

This configuration management system is part of the Neural Forecast project and follows the same licensing terms.

---

For more examples and advanced usage patterns, see `examples_and_usage.py` in this directory.