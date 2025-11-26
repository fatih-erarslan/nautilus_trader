"""
Configuration Loader for Trading APIs

Loads and validates trading API configurations with support for
environment variables, secrets management, and dynamic reloading.
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import re
from datetime import datetime
import logging
import hashlib
from cryptography.fernet import Fernet
import asyncio
import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent


@dataclass
class APIConfig:
    """Configuration for a single trading API"""
    name: str
    provider: str
    enabled: bool = True
    credentials: Dict[str, Any] = field(default_factory=dict)
    endpoints: Dict[str, str] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    rate_limits: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.name:
            raise ValueError("API name is required")
        if not self.provider:
            raise ValueError("API provider is required")


@dataclass
class TradingConfig:
    """Complete trading configuration"""
    apis: Dict[str, APIConfig] = field(default_factory=dict)
    global_settings: Dict[str, Any] = field(default_factory=dict)
    connection_pool: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=datetime.now)
    config_hash: str = ""


class ConfigFileWatcher(FileSystemEventHandler):
    """Watch configuration files for changes"""
    
    def __init__(self, config_loader, callback):
        self.config_loader = config_loader
        self.callback = callback
        
    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent):
            if event.src_path.endswith('.yaml') or event.src_path.endswith('.yml'):
                self.callback(event.src_path)


class ConfigLoader:
    """
    Advanced configuration loader for trading APIs.
    
    Features:
    - YAML/JSON configuration support
    - Environment variable interpolation
    - Encrypted secrets management
    - Configuration validation
    - Hot reloading
    - Multiple configuration sources
    """
    
    def __init__(self,
                 config_path: Optional[Union[str, Path]] = None,
                 enable_hot_reload: bool = True,
                 encryption_key: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to main configuration file
            enable_hot_reload: Enable automatic config reloading
            encryption_key: Key for encrypted values (base64 encoded)
        """
        self.config_path = Path(config_path) if config_path else None
        self.enable_hot_reload = enable_hot_reload
        
        # Setup encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        else:
            # Generate a new key if none provided
            self.cipher = Fernet(Fernet.generate_key())
        
        # Current configuration
        self.config: Optional[TradingConfig] = None
        
        # Configuration sources priority (highest to lowest)
        self.config_sources = [
            self._load_env_config,
            self._load_file_config,
            self._load_default_config
        ]
        
        # File watcher
        self.observer: Optional[Observer] = None
        self._reload_callbacks: List[callable] = []
        
        self.logger = logging.getLogger(__name__)
    
    async def load_config(self) -> TradingConfig:
        """
        Load configuration from all sources.
        
        Returns:
            TradingConfig object with merged configuration
        """
        # Start with empty config
        merged_config = {}
        
        # Load from each source
        for source in reversed(self.config_sources):  # Start with lowest priority
            try:
                source_config = await source()
                if source_config:
                    merged_config = self._deep_merge(merged_config, source_config)
            except Exception as e:
                self.logger.error(f"Error loading config from {source.__name__}: {e}")
        
        # Parse into structured config
        self.config = self._parse_config(merged_config)
        
        # Start file watcher if enabled
        if self.enable_hot_reload and self.config_path:
            self._start_file_watcher()
        
        return self.config
    
    async def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {
            'apis': {},
            'global_settings': {},
            'connection_pool': {},
            'monitoring': {}
        }
        
        # Pattern for API-specific env vars: TRADING_API_<NAME>_<SETTING>
        api_pattern = re.compile(r'^TRADING_API_([A-Z0-9_]+)_(.+)$')
        
        for key, value in os.environ.items():
            match = api_pattern.match(key)
            if match:
                api_name = match.group(1).lower()
                setting_path = match.group(2).lower().split('_')
                
                if api_name not in config['apis']:
                    config['apis'][api_name] = {}
                
                # Navigate to the correct nested location
                current = config['apis'][api_name]
                for part in setting_path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value
                current[setting_path[-1]] = self._parse_env_value(value)
            
            # Global settings pattern: TRADING_<CATEGORY>_<SETTING>
            elif key.startswith('TRADING_'):
                parts = key[8:].lower().split('_')
                if parts[0] in ['GLOBAL', 'CONNECTION', 'MONITORING']:
                    category_map = {
                        'GLOBAL': 'global_settings',
                        'CONNECTION': 'connection_pool',
                        'MONITORING': 'monitoring'
                    }
                    category = category_map[parts[0]]
                    
                    current = config[category]
                    for part in parts[1:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    current[parts[-1]] = self._parse_env_value(value)
        
        return config if any(config['apis']) else None
    
    async def _load_file_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path or not self.config_path.exists():
            return None
        
        try:
            async with aiofiles.open(self.config_path, 'r') as f:
                content = await f.read()
            
            # Parse based on extension
            if self.config_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(content)
            elif self.config_path.suffix == '.json':
                config = json.loads(content)
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
            
            # Process includes
            if 'include' in config:
                for include_path in config['include']:
                    include_config = await self._load_include_file(include_path)
                    config = self._deep_merge(config, include_config)
            
            # Decrypt encrypted values
            config = self._decrypt_config(config)
            
            # Interpolate variables
            config = self._interpolate_config(config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading config file: {e}")
            return None
    
    async def _load_include_file(self, include_path: str) -> Dict[str, Any]:
        """Load an included configuration file."""
        # Resolve relative to main config
        if self.config_path:
            full_path = self.config_path.parent / include_path
        else:
            full_path = Path(include_path)
        
        if not full_path.exists():
            self.logger.warning(f"Include file not found: {full_path}")
            return {}
        
        try:
            async with aiofiles.open(full_path, 'r') as f:
                content = await f.read()
            
            if full_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(content)
            elif full_path.suffix == '.json':
                return json.loads(content)
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error loading include file {full_path}: {e}")
            return {}
    
    async def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'global_settings': {
                'timezone': 'UTC',
                'log_level': 'INFO',
                'enable_paper_trading': True
            },
            'connection_pool': {
                'min_connections': 2,
                'max_connections': 10,
                'health_check_interval': 30,
                'connection_timeout': 10
            },
            'monitoring': {
                'enable_latency_monitoring': True,
                'latency_alert_threshold_ms': 100,
                'metrics_export_interval': 300
            }
        }
    
    def _parse_config(self, raw_config: Dict[str, Any]) -> TradingConfig:
        """Parse raw configuration into structured format."""
        config = TradingConfig()
        
        # Parse APIs
        if 'apis' in raw_config:
            for api_name, api_config in raw_config['apis'].items():
                config.apis[api_name] = APIConfig(
                    name=api_name,
                    provider=api_config.get('provider', api_name),
                    enabled=api_config.get('enabled', True),
                    credentials=api_config.get('credentials', {}),
                    endpoints=api_config.get('endpoints', {}),
                    settings=api_config.get('settings', {}),
                    rate_limits=api_config.get('rate_limits', {})
                )
        
        # Set other sections
        config.global_settings = raw_config.get('global_settings', {})
        config.connection_pool = raw_config.get('connection_pool', {})
        config.monitoring = raw_config.get('monitoring', {})
        
        # Calculate config hash
        config_str = json.dumps(raw_config, sort_keys=True)
        config.config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        return config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # JSON
        if value.startswith('[') or value.startswith('{'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # String
        return value
    
    def _decrypt_config(self, config: Dict) -> Dict:
        """Decrypt encrypted values in config."""
        def decrypt_value(value):
            if isinstance(value, str) and value.startswith('ENC[') and value.endswith(']'):
                encrypted = value[4:-1]
                try:
                    decrypted = self.cipher.decrypt(encrypted.encode()).decode()
                    return decrypted
                except Exception as e:
                    self.logger.error(f"Failed to decrypt value: {e}")
                    return value
            elif isinstance(value, dict):
                return {k: decrypt_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [decrypt_value(v) for v in value]
            else:
                return value
        
        return decrypt_value(config)
    
    def _interpolate_config(self, config: Dict) -> Dict:
        """Interpolate variables in config."""
        def interpolate_value(value, context):
            if isinstance(value, str):
                # Replace ${VAR} with environment variable
                pattern = re.compile(r'\$\{([^}]+)\}')
                
                def replacer(match):
                    var_name = match.group(1)
                    # Check environment first
                    if var_name in os.environ:
                        return os.environ[var_name]
                    # Then check context
                    parts = var_name.split('.')
                    current = context
                    for part in parts:
                        if isinstance(current, dict) and part in current:
                            current = current[part]
                        else:
                            return match.group(0)  # Return unchanged
                    return str(current)
                
                return pattern.sub(replacer, value)
            elif isinstance(value, dict):
                return {k: interpolate_value(v, context) for k, v in value.items()}
            elif isinstance(value, list):
                return [interpolate_value(v, context) for v in value]
            else:
                return value
        
        # Do multiple passes for nested references
        result = config
        for _ in range(3):
            result = interpolate_value(result, result)
        
        return result
    
    def _start_file_watcher(self) -> None:
        """Start watching configuration files for changes."""
        if self.observer:
            self.observer.stop()
        
        self.observer = Observer()
        handler = ConfigFileWatcher(self, self._on_config_changed)
        
        # Watch the config file and its directory
        watch_dir = self.config_path.parent
        self.observer.schedule(handler, str(watch_dir), recursive=False)
        self.observer.start()
        
        self.logger.info(f"Started config file watcher for {watch_dir}")
    
    def _on_config_changed(self, filepath: str) -> None:
        """Handle configuration file changes."""
        self.logger.info(f"Configuration file changed: {filepath}")
        
        # Reload configuration
        asyncio.create_task(self._reload_config())
    
    async def _reload_config(self) -> None:
        """Reload configuration and notify callbacks."""
        try:
            old_hash = self.config.config_hash if self.config else ""
            new_config = await self.load_config()
            
            if new_config.config_hash != old_hash:
                self.logger.info("Configuration reloaded successfully")
                
                # Notify callbacks
                for callback in self._reload_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(new_config)
                        else:
                            callback(new_config)
                    except Exception as e:
                        self.logger.error(f"Error in reload callback: {e}")
            else:
                self.logger.info("Configuration unchanged")
                
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")
    
    def add_reload_callback(self, callback: callable) -> None:
        """Add a callback to be called when config is reloaded."""
        self._reload_callbacks.append(callback)
    
    def get_api_config(self, api_name: str) -> Optional[APIConfig]:
        """Get configuration for a specific API."""
        if self.config and api_name in self.config.apis:
            return self.config.apis[api_name]
        return None
    
    def get_enabled_apis(self) -> List[APIConfig]:
        """Get list of enabled API configurations."""
        if not self.config:
            return []
        
        return [
            api for api in self.config.apis.values()
            if api.enabled
        ]
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a value for storage in config."""
        encrypted = self.cipher.encrypt(value.encode()).decode()
        return f"ENC[{encrypted}]"
    
    async def save_config(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file."""
        save_path = Path(filepath) if filepath else self.config_path
        
        if not save_path:
            raise ValueError("No save path specified")
        
        # Convert config to dict
        config_dict = {
            'apis': {
                name: {
                    'provider': api.provider,
                    'enabled': api.enabled,
                    'credentials': api.credentials,
                    'endpoints': api.endpoints,
                    'settings': api.settings,
                    'rate_limits': api.rate_limits
                }
                for name, api in self.config.apis.items()
            },
            'global_settings': self.config.global_settings,
            'connection_pool': self.config.connection_pool,
            'monitoring': self.config.monitoring
        }
        
        # Save based on extension
        if save_path.suffix in ['.yaml', '.yml']:
            content = yaml.dump(config_dict, default_flow_style=False)
        else:
            content = json.dumps(config_dict, indent=2)
        
        async with aiofiles.open(save_path, 'w') as f:
            await f.write(content)
        
        self.logger.info(f"Configuration saved to {save_path}")
    
    def validate_config(self) -> List[str]:
        """Validate current configuration."""
        errors = []
        
        if not self.config:
            errors.append("No configuration loaded")
            return errors
        
        # Check for at least one enabled API
        if not any(api.enabled for api in self.config.apis.values()):
            errors.append("No APIs are enabled")
        
        # Validate each API
        for name, api in self.config.apis.items():
            if api.enabled:
                # Check required credentials
                if not api.credentials:
                    errors.append(f"API '{name}' has no credentials configured")
                
                # Check endpoints
                if not api.endpoints:
                    errors.append(f"API '{name}' has no endpoints configured")
        
        # Validate connection pool settings
        pool_config = self.config.connection_pool
        if pool_config.get('min_connections', 0) > pool_config.get('max_connections', 10):
            errors.append("min_connections cannot be greater than max_connections")
        
        return errors
    
    def __enter__(self):
        """Context manager entry."""
        asyncio.run(self.load_config())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.observer:
            self.observer.stop()
            self.observer.join()