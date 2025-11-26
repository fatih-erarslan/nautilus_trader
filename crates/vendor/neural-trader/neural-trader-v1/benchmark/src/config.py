"""Configuration management for benchmark CLI."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigManager:
    """Manage configuration from files and environment variables."""
    
    def __init__(self):
        self.config = {}
        self._load_defaults()
        
    def _load_defaults(self):
        """Load default configuration."""
        self.config = {
            'benchmark': {
                'default_strategy': 'all',
                'default_duration': '1h',
                'default_assets': ['stocks'],
                'cache_enabled': True,
                'cache_dir': '.benchmark_cache'
            },
            'simulation': {
                'initial_capital': 10000,
                'default_start_date': '2024-01-01',
                'default_end_date': '2024-12-31'
            },
            'optimization': {
                'default_metric': 'sharpe',
                'default_iterations': 100,
                'parallel_enabled': True
            },
            'report': {
                'default_format': 'html',
                'include_charts': True
            }
        }
    
    def load_from_file(self, file_path: str):
        """Load configuration from YAML or JSON file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                file_config = yaml.safe_load(f)
            elif file_path.endswith('.json'):
                file_config = json.load(f)
            else:
                raise ValueError("Configuration file must be YAML or JSON")
        
        # Merge with existing config
        self._merge_config(file_config)
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        load_dotenv()
        
        # Map environment variables to config
        env_mappings = {
            'BENCHMARK_STRATEGY': ('benchmark', 'default_strategy'),
            'BENCHMARK_DURATION': ('benchmark', 'default_duration'),
            'BENCHMARK_ASSETS': ('benchmark', 'default_assets'),
            'BENCHMARK_CACHE_DIR': ('benchmark', 'cache_dir'),
            'SIMULATION_CAPITAL': ('simulation', 'initial_capital'),
            'OPTIMIZATION_METRIC': ('optimization', 'default_metric'),
            'OPTIMIZATION_ITERATIONS': ('optimization', 'default_iterations'),
            'REPORT_FORMAT': ('report', 'default_format')
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                # Convert value types as needed
                if key == 'default_assets' and isinstance(value, str):
                    value = value.split(',')
                elif key in ['initial_capital', 'default_iterations']:
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                
                # Set in config
                if section not in self.config:
                    self.config[section] = {}
                self.config[section][key] = value
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing."""
        def deep_merge(base: dict, update: dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(self.config, new_config)
    
    def save_to_file(self, file_path: str, format: str = 'yaml'):
        """Save current configuration to file."""
        with open(file_path, 'w') as f:
            if format == 'yaml':
                yaml.dump(self.config, f, default_flow_style=False)
            else:
                json.dump(self.config, f, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration."""
        # Add validation logic here
        required_sections = ['benchmark', 'simulation', 'optimization', 'report']
        for section in required_sections:
            if section not in self.config:
                return False
        return True