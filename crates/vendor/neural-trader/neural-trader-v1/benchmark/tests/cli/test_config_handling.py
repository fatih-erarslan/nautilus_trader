"""Tests for configuration management module."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import pytest
import yaml

from benchmark.src.cli.config import (
    BenchmarkConfig,
    ConfigManager,
    ConfigValidator,
    MergeStrategy,
)


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        
        # Create a sample config file
        self.sample_config = {
            "global": {
                "output_dir": "./benchmark_results",
                "log_level": "INFO",
                "parallel_workers": 8,
            },
            "benchmark": {
                "default_suite": "standard",
                "warmup_duration": 60,
                "measurement_duration": 300,
            },
            "simulation": {
                "data_source": "historical",
                "tick_resolution": "1s",
                "market_hours_only": True,
            },
        }
        
        with open(self.config_file, "w") as f:
            yaml.dump(self.sample_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_yaml_config(self):
        """Test loading YAML configuration file."""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_file)
        
        assert config.global_settings.output_dir == "./benchmark_results"
        assert config.global_settings.log_level == "INFO"
        assert config.global_settings.parallel_workers == 8
        assert config.benchmark.default_suite == "standard"

    def test_load_json_config(self):
        """Test loading JSON configuration file."""
        json_file = Path(self.temp_dir) / "test_config.json"
        with open(json_file, "w") as f:
            json.dump(self.sample_config, f)
        
        config_manager = ConfigManager()
        config = config_manager.load_config(json_file)
        
        assert config.global_settings.output_dir == "./benchmark_results"
        assert config.benchmark.warmup_duration == 60

    def test_merge_configs(self):
        """Test merging multiple configuration sources."""
        config_manager = ConfigManager()
        
        base_config = config_manager.load_config(self.config_file)
        
        override_config = {
            "global": {"log_level": "DEBUG"},
            "benchmark": {"warmup_duration": 120},
        }
        
        merged = config_manager.merge_configs(
            base_config, override_config, strategy=MergeStrategy.DEEP
        )
        
        assert merged.global_settings.log_level == "DEBUG"
        assert merged.global_settings.parallel_workers == 8  # Unchanged
        assert merged.benchmark.warmup_duration == 120

    def test_validate_config(self):
        """Test configuration validation."""
        validator = ConfigValidator()
        
        # Valid config
        assert validator.validate(self.sample_config) is True
        
        # Invalid config - negative workers
        invalid_config = self.sample_config.copy()
        invalid_config["global"]["parallel_workers"] = -1
        
        with pytest.raises(ValueError, match="parallel_workers must be positive"):
            validator.validate(invalid_config)

    def test_config_defaults(self):
        """Test default configuration values."""
        config_manager = ConfigManager()
        config = config_manager.get_default_config()
        
        assert config.global_settings.output_dir == "./benchmark_results"
        assert config.global_settings.log_level == "INFO"
        assert config.benchmark.default_suite == "standard"
        assert config.simulation.market_hours_only is True

    def test_config_environment_override(self):
        """Test environment variable overrides."""
        os.environ["BENCHMARK_LOG_LEVEL"] = "DEBUG"
        os.environ["BENCHMARK_PARALLEL_WORKERS"] = "16"
        
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_file, use_env=True)
        
        assert config.global_settings.log_level == "DEBUG"
        assert config.global_settings.parallel_workers == 16
        
        # Clean up
        del os.environ["BENCHMARK_LOG_LEVEL"]
        del os.environ["BENCHMARK_PARALLEL_WORKERS"]

    def test_save_config(self):
        """Test saving configuration to file."""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_file)
        
        # Modify config
        config.global_settings.log_level = "DEBUG"
        
        # Save to new file
        new_config_file = Path(self.temp_dir) / "saved_config.yaml"
        config_manager.save_config(config, new_config_file)
        
        # Load and verify
        loaded_config = config_manager.load_config(new_config_file)
        assert loaded_config.global_settings.log_level == "DEBUG"

    def test_config_schema_validation(self):
        """Test configuration schema validation."""
        validator = ConfigValidator()
        
        # Missing required field
        incomplete_config = {
            "global": {"output_dir": "./results"},
            # Missing benchmark section
        }
        
        with pytest.raises(ValueError, match="Missing required section: benchmark"):
            validator.validate(incomplete_config)

    def test_custom_config_path(self):
        """Test loading config from custom path."""
        custom_dir = Path(self.temp_dir) / "custom"
        custom_dir.mkdir()
        custom_file = custom_dir / "benchmark.yaml"
        
        with open(custom_file, "w") as f:
            yaml.dump(self.sample_config, f)
        
        config_manager = ConfigManager(config_dir=custom_dir)
        config = config_manager.load_config("benchmark.yaml")
        
        assert config.global_settings.output_dir == "./benchmark_results"


class TestBenchmarkConfig(unittest.TestCase):
    """Test cases for BenchmarkConfig dataclass."""

    def test_config_dataclass_creation(self):
        """Test creating BenchmarkConfig instance."""
        config = BenchmarkConfig(
            global_settings={
                "output_dir": "./results",
                "log_level": "INFO",
                "parallel_workers": 4,
            },
            benchmark={
                "default_suite": "quick",
                "warmup_duration": 30,
                "measurement_duration": 180,
            },
            simulation={
                "data_source": "synthetic",
                "tick_resolution": "100ms",
                "market_hours_only": False,
            },
        )
        
        assert config.global_settings.output_dir == "./results"
        assert config.benchmark.default_suite == "quick"
        assert config.simulation.market_hours_only is False

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = BenchmarkConfig.from_dict({
            "global": {
                "output_dir": "./results",
                "log_level": "INFO",
                "parallel_workers": 4,
            },
            "benchmark": {
                "default_suite": "quick",
                "warmup_duration": 30,
                "measurement_duration": 180,
            },
        })
        
        config_dict = config.to_dict()
        assert config_dict["global"]["output_dir"] == "./results"
        assert config_dict["benchmark"]["warmup_duration"] == 30

    def test_config_get_nested_value(self):
        """Test getting nested configuration values."""
        config = BenchmarkConfig.from_dict({
            "global": {"output_dir": "./results"},
            "metrics": {
                "latency": {
                    "percentiles": [50, 90, 95, 99],
                    "window_size": 1000,
                }
            },
        })
        
        # Test nested access
        percentiles = config.get("metrics.latency.percentiles")
        assert percentiles == [50, 90, 95, 99]
        
        # Test default value
        missing = config.get("metrics.throughput.interval", default="1s")
        assert missing == "1s"


if __name__ == "__main__":
    unittest.main()