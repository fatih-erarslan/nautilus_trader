"""
Neural Forecast Configuration Management - Examples and Usage Guide

This file demonstrates comprehensive usage of the neural forecasting configuration
management system with practical examples for different scenarios.
"""

import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Import the configuration management system
from .config_manager import ConfigManager
from .model_config import ModelType, OptimizationPreset
from .environment_manager import Environment

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_setup():
    """Example 1: Basic configuration setup and usage."""
    print("=== Example 1: Basic Configuration Setup ===")
    
    # Initialize configuration manager
    config_manager = ConfigManager(
        environment="development",
        auto_detect_gpu=True,
        enable_secrets=True
    )
    
    # Get system information
    system_info = config_manager.get_system_info()
    print(f"System Info: {json.dumps(system_info, indent=2)}")
    
    # Create a basic model configuration
    model_config = config_manager.create_model_config(
        model_type=ModelType.NHITS,
        preset=OptimizationPreset.BALANCED
    )
    
    print(f"Model Config: {json.dumps(model_config, indent=2)}")
    
    # Validate the configuration
    validation_result = config_manager.validate_configuration(
        config=model_config,
        config_type="model_config"
    )
    
    print(f"Validation Result: {json.dumps(validation_result, indent=2)}")
    
    return config_manager


def example_environment_management():
    """Example 2: Environment-specific configuration management."""
    print("\n=== Example 2: Environment Management ===")
    
    config_manager = ConfigManager()
    
    # Get development configuration
    dev_config = config_manager.get_config()
    print(f"Development Config Keys: {list(dev_config.keys())}")
    
    # Switch to production environment
    config_manager.switch_environment("production")
    prod_config = config_manager.get_config()
    print(f"Production Config Keys: {list(prod_config.keys())}")
    
    # Compare environment differences
    print("\nEnvironment Differences:")
    dev_env = config_manager.environment_manager.get_config("development")
    prod_env = config_manager.environment_manager.get_config("production")
    
    print(f"Dev Debug Mode: {dev_env.get('debug', False)}")
    print(f"Prod Debug Mode: {prod_env.get('debug', False)}")
    print(f"Dev GPU Memory Fraction: {dev_env.get('gpu', {}).get('memory_fraction', 'N/A')}")
    print(f"Prod GPU Memory Fraction: {prod_env.get('gpu', {}).get('memory_fraction', 'N/A')}")


def example_secrets_management():
    """Example 3: Secure secrets management."""
    print("\n=== Example 3: Secrets Management ===")
    
    config_manager = ConfigManager(enable_secrets=True)
    
    # Store API keys and credentials
    secrets_to_store = {
        "openai_api_key": "sk-example-key-12345",
        "database_password": "super-secure-password",
        "s3_credentials": {
            "access_key": "AKIA12345",
            "secret_key": "secret-key-67890",
            "bucket": "neural-forecast-data"
        }
    }
    
    for name, value in secrets_to_store.items():
        success = config_manager.store_secret(
            name=name,
            value=value,
            description=f"API credentials for {name}",
            expires_in_days=90
        )
        print(f"Stored secret '{name}': {success}")
    
    # Retrieve secrets
    api_key = config_manager.get_secret("openai_api_key")
    db_password = config_manager.get_secret("database_password")
    s3_creds = config_manager.get_secret("s3_credentials")
    
    print(f"Retrieved API Key: {api_key[:10]}..." if api_key else "Not found")
    print(f"Retrieved DB Password: {'*' * len(db_password)}" if db_password else "Not found")
    print(f"Retrieved S3 Credentials: {type(s3_creds)}")
    
    # List all secrets
    if config_manager.secrets_manager:
        secrets_list = config_manager.secrets_manager.list_secrets()
        print(f"\nAll secrets: {[s['name'] for s in secrets_list]}")


def example_model_configuration_templates():
    """Example 4: Model configuration templates and optimization."""
    print("\n=== Example 4: Model Configuration Templates ===")
    
    config_manager = ConfigManager()
    
    # Get available model templates
    templates = config_manager.model_config_manager.get_available_templates()
    print(f"Available templates: {len(templates)}")
    
    for template in templates[:3]:  # Show first 3
        print(f"  - {template['name']}: {template['description']}")
    
    # Create configurations for different use cases
    use_cases = [
        ("real_time", {"max_memory_gb": 4, "max_training_time_minutes": 5}),
        ("production_forecasting", {"min_accuracy": 0.95}),
        ("research", {"max_memory_gb": 16})
    ]
    
    for use_case, constraints in use_cases:
        recommendation = config_manager.get_recommended_model_config(
            use_case=use_case,
            constraints=constraints
        )
        print(f"\n{use_case.title()} Recommendation:")
        print(f"  Model: {recommendation['recommended_model']}")
        print(f"  Preset: {recommendation['recommended_preset']}")
        print(f"  Batch Size: {recommendation['config']['batch_size']}")
        print(f"  Max Steps: {recommendation['config']['max_steps']}")


def example_ensemble_configuration():
    """Example 5: Ensemble model configuration."""
    print("\n=== Example 5: Ensemble Configuration ===")
    
    config_manager = ConfigManager()
    
    # Create ensemble with multiple models
    ensemble_config = config_manager.create_ensemble_config(
        models=["NBEATS", "NHITS", "TFT"],
        horizon=24,
        weighting_strategy="performance"
    )
    
    print("Ensemble Configuration:")
    print(f"  Models: {list(ensemble_config['ensemble']['models'].keys())}")
    print(f"  Weighting: {ensemble_config['ensemble']['weighting_strategy']}")
    
    # Show model-specific configurations
    for model_name, model_config in ensemble_config['ensemble']['models'].items():
        print(f"\n  {model_name} Config:")
        print(f"    Input Size: {model_config['input_size']}")
        print(f"    Batch Size: {model_config['batch_size']}")
        print(f"    Learning Rate: {model_config['learning_rate']}")


def example_gpu_optimization():
    """Example 6: GPU configuration and optimization."""
    print("\n=== Example 6: GPU Configuration ===")
    
    config_manager = ConfigManager(auto_detect_gpu=True)
    
    if config_manager.gpu_config_manager:
        # Get GPU information
        gpu_info = config_manager.gpu_config_manager.get_gpu_info()
        print(f"Detected GPUs: {len(gpu_info)}")
        
        for gpu in gpu_info:
            print(f"  GPU {gpu.device_id}: {gpu.name}")
            print(f"    Memory: {gpu.memory_total}MB total, {gpu.memory_free}MB free")
            print(f"    Architecture: {gpu.architecture.value}")
            print(f"    Tensor Cores: {gpu.tensor_cores}")
        
        # Get current GPU configuration
        gpu_config = config_manager.gpu_config_manager.get_current_config()
        print(f"\nCurrent GPU Config:")
        print(f"  Enabled: {gpu_config.enabled}")
        print(f"  Memory Fraction: {gpu_config.memory_fraction}")
        print(f"  Mixed Precision: {gpu_config.mixed_precision}")
        
        # Create optimized model configuration for available hardware
        model_config = config_manager.create_model_config(
            model_type=ModelType.NHITS,
            preset=OptimizationPreset.BALANCED
        )
        
        # Get GPU optimization suggestions
        gpu_suggestions = config_manager.gpu_config_manager.optimize_for_model(model_config)
        print(f"\nGPU Optimization Suggestions: {gpu_suggestions}")
        
        # Run GPU benchmark if possible
        benchmark_results = config_manager.gpu_config_manager.benchmark_gpu(duration_seconds=5)
        if 'error' not in benchmark_results:
            print(f"\nGPU Benchmark Results:")
            print(f"  Operations/sec: {benchmark_results['compute_performance']['operations_per_second']:.2f}")
            print(f"  Memory Bandwidth: {benchmark_results['memory_bandwidth']['bandwidth_gb_s']:.2f} GB/s")
    else:
        print("No GPU configuration manager available")


def example_data_driven_optimization():
    """Example 7: Data-driven configuration optimization."""
    print("\n=== Example 7: Data-Driven Optimization ===")
    
    config_manager = ConfigManager()
    
    # Simulate different data characteristics
    data_scenarios = [
        {
            "name": "High Frequency Trading Data",
            "characteristics": {
                "data_points": 100000,
                "seasonality_strength": 0.3,
                "trend_strength": 0.2,
                "noise_level": 0.15,
                "frequency": "S"  # Seconds
            }
        },
        {
            "name": "Daily Stock Prices",
            "characteristics": {
                "data_points": 2000,
                "seasonality_strength": 0.6,
                "trend_strength": 0.8,
                "noise_level": 0.1,
                "frequency": "D"  # Daily
            }
        },
        {
            "name": "Monthly Economic Indicators",
            "characteristics": {
                "data_points": 120,
                "seasonality_strength": 0.9,
                "trend_strength": 0.5,
                "noise_level": 0.05,
                "frequency": "M"  # Monthly
            }
        }
    ]
    
    for scenario in data_scenarios:
        print(f"\n{scenario['name']}:")
        
        # Get optimized configuration
        optimized_config = config_manager.model_config_manager.optimize_config_for_data(
            model_type=ModelType.NHITS,
            data_characteristics=scenario['characteristics'],
            performance_target="balanced"
        )
        
        print(f"  Optimized Input Size: {optimized_config['input_size']}")
        print(f"  Frequency Downsampling: {optimized_config.get('n_freq_downsample', 'N/A')}")
        print(f"  Dropout Rate: {optimized_config.get('dropout', 'N/A')}")
        print(f"  Training Steps: {optimized_config['max_steps']}")


def example_configuration_export_import():
    """Example 8: Configuration export and import."""
    print("\n=== Example 8: Configuration Export/Import ===")
    
    config_manager = ConfigManager()
    
    # Export configuration
    export_file = config_manager.export_configuration(
        output_file="example_config_export.yaml",
        include_secrets=False  # Never include secrets in real exports
    )
    
    print(f"Configuration exported to: {export_file}")
    
    # Show export contents (first few lines)
    try:
        with open(export_file, 'r') as f:
            lines = f.readlines()[:20]  # First 20 lines
        
        print("\nExport file contents (first 20 lines):")
        for i, line in enumerate(lines, 1):
            print(f"  {i:2d}: {line.rstrip()}")
        
        if len(lines) >= 20:
            print("  ... (truncated)")
    
    except Exception as e:
        print(f"Error reading export file: {e}")


def example_health_monitoring():
    """Example 9: System health monitoring and validation."""
    print("\n=== Example 9: Health Monitoring ===")
    
    config_manager = ConfigManager()
    
    # Perform health check
    health_status = config_manager.health_check()
    
    print(f"Overall System Health: {health_status['overall_status'].upper()}")
    print(f"Components Checked: {len(health_status['components'])}")
    
    for component, status in health_status['components'].items():
        print(f"\n{component.replace('_', ' ').title()}:")
        print(f"  Status: {status['status'].upper()}")
        
        if 'details' in status:
            details = status['details']
            if isinstance(details, dict):
                for key, value in details.items():
                    if key not in ['errors', 'warnings']:
                        print(f"  {key.replace('_', ' ').title()}: {value}")
        
        if 'error' in status:
            print(f"  Error: {status['error']}")


def example_custom_model_template():
    """Example 10: Creating and using custom model templates."""
    print("\n=== Example 10: Custom Model Templates ===")
    
    config_manager = ConfigManager()
    
    # Create a custom model template for high-frequency trading
    custom_config = {
        "input_size": 48,  # 2 days of hourly data
        "h": 12,           # 12-hour forecast
        "max_steps": 150,  # Fast training
        "learning_rate": 0.005,
        "batch_size": 128,  # Large batch for speed
        "windows_batch_size": 512,
        "n_freq_downsample": [12, 6, 1],
        "dropout": 0.05,  # Low dropout for speed
        "early_stopping_patience": 8
    }
    
    success = config_manager.model_config_manager.save_custom_template(
        name="hft_optimized",
        model_type=ModelType.NHITS,
        config=custom_config,
        description="High-frequency trading optimized NHITS configuration",
        use_cases=["High-frequency trading", "Real-time forecasting", "Low-latency applications"]
    )
    
    print(f"Custom template saved: {success}")
    
    # Load and use the custom template
    if success:
        # Note: In a real implementation, you'd reload the templates
        print("Custom template 'hft_optimized' created with:")
        print(f"  Input Size: {custom_config['input_size']}")
        print(f"  Forecast Horizon: {custom_config['h']}")
        print(f"  Training Steps: {custom_config['max_steps']}")
        print(f"  Batch Size: {custom_config['batch_size']}")


def run_all_examples():
    """Run all configuration management examples."""
    print("Neural Forecast Configuration Management - Examples")
    print("=" * 60)
    
    try:
        # Basic setup
        config_manager = example_basic_setup()
        
        # Environment management
        example_environment_management()
        
        # Secrets management
        example_secrets_management()
        
        # Model templates
        example_model_configuration_templates()
        
        # Ensemble configuration
        example_ensemble_configuration()
        
        # GPU optimization
        example_gpu_optimization()
        
        # Data-driven optimization
        example_data_driven_optimization()
        
        # Export/Import
        example_configuration_export_import()
        
        # Health monitoring
        example_health_monitoring()
        
        # Custom templates
        example_custom_model_template()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


# Configuration usage patterns and best practices
USAGE_PATTERNS = {
    "development": {
        "description": "Fast iteration and testing",
        "recommended_models": ["NBEATS_fast", "NHITS_fast"],
        "gpu_settings": {"memory_fraction": 0.5, "mixed_precision": False},
        "training_settings": {"max_steps": 200, "early_stopping_patience": 10}
    },
    "production": {
        "description": "Stable, high-performance deployment",
        "recommended_models": ["NHITS_production", "TFT_balanced"],
        "gpu_settings": {"memory_fraction": 0.8, "mixed_precision": True},
        "training_settings": {"max_steps": 1000, "early_stopping_patience": 50}
    },
    "research": {
        "description": "Maximum accuracy for research",
        "recommended_models": ["TFT_accurate", "NBEATS_accurate"],
        "gpu_settings": {"memory_fraction": 0.9, "mixed_precision": True},
        "training_settings": {"max_steps": 2000, "early_stopping_patience": 100}
    }
}


def print_usage_guide():
    """Print comprehensive usage guide."""
    print("\n" + "=" * 60)
    print("NEURAL FORECAST CONFIGURATION MANAGEMENT GUIDE")
    print("=" * 60)
    
    print("\n1. QUICK START:")
    print("""
    from neural_forecast.config import ConfigManager
    
    # Initialize configuration manager
    config_manager = ConfigManager(environment="development")
    
    # Create model configuration
    model_config = config_manager.create_model_config(
        model_type="NHITS",
        preset="balanced"
    )
    
    # Get secrets
    api_key = config_manager.get_secret("api_key")
    """)
    
    print("\n2. ENVIRONMENT PATTERNS:")
    for pattern_name, pattern_info in USAGE_PATTERNS.items():
        print(f"\n   {pattern_name.upper()}:")
        print(f"   - {pattern_info['description']}")
        print(f"   - Models: {', '.join(pattern_info['recommended_models'])}")
        print(f"   - GPU Memory: {pattern_info['gpu_settings']['memory_fraction']*100}%")
        print(f"   - Training Steps: {pattern_info['training_settings']['max_steps']}")
    
    print("\n3. BEST PRACTICES:")
    print("""
   - Always validate configurations before use
   - Use environment-specific settings for dev/staging/prod
   - Store sensitive data in secrets manager
   - Monitor GPU usage and optimize accordingly
   - Use ensemble models for critical applications
   - Regularly backup configurations
   - Run health checks in production
   """)
    
    print("\n4. COMMON CONFIGURATION PATTERNS:")
    print("""
   Real-time Trading:
   - Model: NHITS fast preset
   - GPU: High memory fraction, mixed precision
   - Horizon: 1-6 hours
   
   Daily Forecasting:
   - Model: NBEATS balanced preset
   - Ensemble: NBEATS + NHITS
   - Horizon: 24 hours
   
   Research/Analysis:
   - Model: TFT accurate preset
   - GPU: Maximum memory allocation
   - Ensemble: All available models
   """)


if __name__ == "__main__":
    # Run examples if script is executed directly
    run_all_examples()
    print_usage_guide()