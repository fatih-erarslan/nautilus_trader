"""Configuration management for benchmark CLI."""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class MergeStrategy(Enum):
    """Strategy for merging configurations."""
    
    DEEP = "deep"  # Deep merge nested dictionaries
    SHALLOW = "shallow"  # Replace entire sections
    OVERRIDE = "override"  # Override individual values only


@dataclass
class GlobalSettings:
    """Global configuration settings."""
    
    output_dir: str = "./benchmark_results"
    log_level: str = "INFO"
    parallel_workers: int = 8
    verbose: bool = False
    quiet: bool = False
    format: str = "terminal"
    profile: bool = False
    debug: bool = False


@dataclass
class BenchmarkSettings:
    """Benchmark-specific settings."""
    
    default_suite: str = "standard"
    warmup_duration: int = 60
    measurement_duration: int = 300
    parallel: bool = True
    save_baseline: bool = False


@dataclass
class SimulationSettings:
    """Simulation-specific settings."""
    
    data_source: str = "historical"
    tick_resolution: str = "1s"
    market_hours_only: bool = True
    default_capital: float = 100000.0
    speed_factor: float = 1.0


@dataclass
class OptimizationSettings:
    """Optimization-specific settings."""
    
    default_algorithm: str = "bayesian"
    max_trials: int = 1000
    early_stopping: bool = True
    parallel: bool = True
    timeout_minutes: int = 60


@dataclass
class MetricsSettings:
    """Metrics configuration."""
    
    latency: Dict[str, Any] = field(default_factory=lambda: {
        "percentiles": [50, 90, 95, 99, 99.9],
        "window_size": 1000,
    })
    throughput: Dict[str, Any] = field(default_factory=lambda: {
        "interval": "1s",
    })
    memory: Dict[str, Any] = field(default_factory=lambda: {
        "sample_rate": "10Hz",
    })


@dataclass
class BenchmarkConfig:
    """Main configuration container."""
    
    global_settings: GlobalSettings = field(default_factory=GlobalSettings)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    metrics: MetricsSettings = field(default_factory=MetricsSettings)
    
    def __init__(self, **kwargs):
        """Initialize configuration from keyword arguments."""
        # Handle legacy format with plain dictionaries
        if "global_settings" in kwargs and isinstance(kwargs["global_settings"], dict):
            self.global_settings = GlobalSettings(**kwargs["global_settings"])
        else:
            self.global_settings = kwargs.get("global_settings", GlobalSettings())
            
        if "benchmark" in kwargs and isinstance(kwargs["benchmark"], dict):
            self.benchmark = BenchmarkSettings(**kwargs["benchmark"])
        else:
            self.benchmark = kwargs.get("benchmark", BenchmarkSettings())
            
        if "simulation" in kwargs and isinstance(kwargs["simulation"], dict):
            self.simulation = SimulationSettings(**kwargs["simulation"])
        else:
            self.simulation = kwargs.get("simulation", SimulationSettings())
            
        if "optimization" in kwargs and isinstance(kwargs["optimization"], dict):
            self.optimization = OptimizationSettings(**kwargs["optimization"])
        else:
            self.optimization = kwargs.get("optimization", OptimizationSettings())
            
        if "metrics" in kwargs and isinstance(kwargs["metrics"], dict):
            self.metrics = MetricsSettings(**kwargs["metrics"])
        else:
            self.metrics = kwargs.get("metrics", MetricsSettings())
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BenchmarkConfig":
        """Create BenchmarkConfig from dictionary."""
        # Map legacy names to new structure
        mapped_dict = {}
        
        if "global" in config_dict:
            mapped_dict["global_settings"] = config_dict["global"]
        if "benchmark" in config_dict:
            mapped_dict["benchmark"] = config_dict["benchmark"]
        if "simulation" in config_dict:
            mapped_dict["simulation"] = config_dict["simulation"]
        if "optimization" in config_dict:
            mapped_dict["optimization"] = config_dict["optimization"]
        if "metrics" in config_dict:
            mapped_dict["metrics"] = config_dict["metrics"]
            
        return cls(**mapped_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "global": {
                "output_dir": self.global_settings.output_dir,
                "log_level": self.global_settings.log_level,
                "parallel_workers": self.global_settings.parallel_workers,
                "verbose": self.global_settings.verbose,
                "quiet": self.global_settings.quiet,
                "format": self.global_settings.format,
                "profile": self.global_settings.profile,
                "debug": self.global_settings.debug,
            },
            "benchmark": {
                "default_suite": self.benchmark.default_suite,
                "warmup_duration": self.benchmark.warmup_duration,
                "measurement_duration": self.benchmark.measurement_duration,
                "parallel": self.benchmark.parallel,
                "save_baseline": self.benchmark.save_baseline,
            },
            "simulation": {
                "data_source": self.simulation.data_source,
                "tick_resolution": self.simulation.tick_resolution,
                "market_hours_only": self.simulation.market_hours_only,
                "default_capital": self.simulation.default_capital,
                "speed_factor": self.simulation.speed_factor,
            },
            "optimization": {
                "default_algorithm": self.optimization.default_algorithm,
                "max_trials": self.optimization.max_trials,
                "early_stopping": self.optimization.early_stopping,
                "parallel": self.optimization.parallel,
                "timeout_minutes": self.optimization.timeout_minutes,
            },
            "metrics": {
                "latency": self.metrics.latency,
                "throughput": self.metrics.throughput,
                "memory": self.metrics.memory,
            },
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation."""
        parts = key.split(".")
        value = self.to_dict()
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value


class ConfigValidator:
    """Validates configuration against schema."""
    
    def __init__(self):
        """Initialize validator."""
        self.required_sections = ["benchmark"]
        self.valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.valid_algorithms = ["grid", "random", "bayesian", "genetic", "ml"]
        self.valid_data_sources = ["historical", "synthetic", "live"]
    
    def validate(self, config: Union[Dict[str, Any], BenchmarkConfig]) -> bool:
        """Validate configuration."""
        if isinstance(config, BenchmarkConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config
            
        # Check required sections
        for section in self.required_sections:
            if section not in config_dict:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate global settings
        if "global" in config_dict:
            global_config = config_dict["global"]
            
            # Validate parallel_workers
            if "parallel_workers" in global_config:
                workers = global_config["parallel_workers"]
                if not isinstance(workers, int) or workers < 1:
                    raise ValueError("parallel_workers must be positive integer")
            
            # Validate log_level
            if "log_level" in global_config:
                log_level = global_config["log_level"]
                if log_level not in self.valid_log_levels:
                    raise ValueError(f"Invalid log_level: {log_level}")
        
        # Validate benchmark settings
        if "benchmark" in config_dict:
            benchmark_config = config_dict["benchmark"]
            
            # Validate durations
            for duration_key in ["warmup_duration", "measurement_duration"]:
                if duration_key in benchmark_config:
                    duration = benchmark_config[duration_key]
                    if not isinstance(duration, (int, float)) or duration <= 0:
                        raise ValueError(f"{duration_key} must be positive number")
        
        # Validate optimization settings
        if "optimization" in config_dict:
            opt_config = config_dict["optimization"]
            
            # Validate algorithm
            if "default_algorithm" in opt_config:
                algorithm = opt_config["default_algorithm"]
                if algorithm not in self.valid_algorithms:
                    raise ValueError(f"Invalid optimization algorithm: {algorithm}")
        
        # Validate simulation settings
        if "simulation" in config_dict:
            sim_config = config_dict["simulation"]
            
            # Validate data source
            if "data_source" in sim_config:
                source = sim_config["data_source"]
                if source not in self.valid_data_sources:
                    raise ValueError(f"Invalid data_source: {source}")
        
        return True


class ConfigManager:
    """Manages configuration loading, merging, and saving."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager."""
        self.config_dir = config_dir or Path.cwd()
        self.validator = ConfigValidator()
    
    def get_default_config(self) -> BenchmarkConfig:
        """Get default configuration."""
        return BenchmarkConfig()
    
    def load_config(
        self,
        config_file: Union[str, Path],
        use_env: bool = True
    ) -> BenchmarkConfig:
        """Load configuration from file."""
        config_path = Path(config_file)
        
        # Handle relative paths
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
        
        # Load based on file extension
        if config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Validate configuration
        self.validator.validate(config_dict)
        
        # Create config object
        config = BenchmarkConfig.from_dict(config_dict)
        
        # Apply environment overrides if enabled
        if use_env:
            config = self._apply_env_overrides(config)
        
        return config
    
    def save_config(self, config: BenchmarkConfig, config_file: Union[str, Path]):
        """Save configuration to file."""
        config_path = Path(config_file)
        config_dict = config.to_dict()
        
        if config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif config_path.suffix == ".json":
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    def merge_configs(
        self,
        base_config: BenchmarkConfig,
        override_config: Dict[str, Any],
        strategy: MergeStrategy = MergeStrategy.DEEP
    ) -> BenchmarkConfig:
        """Merge configurations based on strategy."""
        base_dict = base_config.to_dict()
        
        if strategy == MergeStrategy.DEEP:
            merged_dict = self._deep_merge(base_dict, override_config)
        elif strategy == MergeStrategy.SHALLOW:
            merged_dict = {**base_dict, **override_config}
        else:  # OVERRIDE
            merged_dict = base_dict.copy()
            for key, value in override_config.items():
                if key in merged_dict:
                    if isinstance(merged_dict[key], dict) and isinstance(value, dict):
                        merged_dict[key].update(value)
                    else:
                        merged_dict[key] = value
        
        return BenchmarkConfig.from_dict(merged_dict)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep merge of dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: BenchmarkConfig) -> BenchmarkConfig:
        """Apply environment variable overrides."""
        # Check for environment variables with BENCHMARK_ prefix
        env_prefix = "BENCHMARK_"
        
        # Log level override
        if f"{env_prefix}LOG_LEVEL" in os.environ:
            config.global_settings.log_level = os.environ[f"{env_prefix}LOG_LEVEL"]
        
        # Parallel workers override
        if f"{env_prefix}PARALLEL_WORKERS" in os.environ:
            config.global_settings.parallel_workers = int(
                os.environ[f"{env_prefix}PARALLEL_WORKERS"]
            )
        
        # Output directory override
        if f"{env_prefix}OUTPUT_DIR" in os.environ:
            config.global_settings.output_dir = os.environ[f"{env_prefix}OUTPUT_DIR"]
        
        # Algorithm override
        if f"{env_prefix}ALGORITHM" in os.environ:
            config.optimization.default_algorithm = os.environ[f"{env_prefix}ALGORITHM"]
        
        return config