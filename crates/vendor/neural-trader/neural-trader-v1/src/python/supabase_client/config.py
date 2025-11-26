"""
Supabase Configuration Management
================================

Configuration classes and utilities for Supabase client initialization.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class SupabaseConfig:
    """Configuration for Supabase client."""
    
    # Required settings
    url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    anon_key: str = field(default_factory=lambda: os.getenv("SUPABASE_ANON_KEY", ""))
    service_key: Optional[str] = field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
    
    # Connection settings
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Real-time settings
    realtime_enabled: bool = True
    realtime_timeout: int = 10
    events_per_second: int = 100
    
    # Authentication settings
    auto_refresh_token: bool = True
    persist_session: bool = True
    detect_session_in_url: bool = True
    
    # Database settings
    schema: str = "public"
    
    # Additional headers
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Performance settings
    connection_pool_size: int = 10
    connection_pool_timeout: int = 30
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.url:
            raise ValueError("SUPABASE_URL is required")
        if not self.anon_key:
            raise ValueError("SUPABASE_ANON_KEY is required")
        
        # Ensure URL format is correct
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("SUPABASE_URL must be a valid HTTP/HTTPS URL")
        
        # Set default headers
        default_headers = {
            "x-application-name": "neural-trader-python",
            "x-client-info": "neural-trader-python/1.0.0"
        }
        self.headers = {**default_headers, **self.headers}
    
    @classmethod
    def from_env(cls, **overrides) -> "SupabaseConfig":
        """Create configuration from environment variables."""
        return cls(**overrides)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SupabaseConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "url": self.url,
            "key": self.anon_key,
            "options": {
                "auth": {
                    "autoRefreshToken": self.auto_refresh_token,
                    "persistSession": self.persist_session,
                    "detectSessionInUrl": self.detect_session_in_url
                },
                "realtime": {
                    "timeout": self.realtime_timeout,
                    "params": {
                        "eventsPerSecond": self.events_per_second
                    }
                } if self.realtime_enabled else None,
                "global": {
                    "headers": self.headers
                },
                "db": {
                    "schema": self.schema
                }
            }
        }

@dataclass
class DatabaseConfig:
    """Database-specific configuration."""
    
    # Connection settings
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", ""))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    database: str = field(default_factory=lambda: os.getenv("DB_NAME", "postgres"))
    username: str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    
    # Connection pool settings
    min_connections: int = 1
    max_connections: int = 20
    max_inactive_connection_lifetime: float = 300.0
    
    # Query settings
    command_timeout: float = 60.0
    query_timeout: float = 30.0
    
    # SSL settings
    ssl_mode: str = "prefer"
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None
    
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )

@dataclass  
class RealtimeConfig:
    """Real-time specific configuration."""
    
    # Connection settings
    enabled: bool = True
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    reconnect_backoff_multiplier: float = 2.0
    max_reconnect_delay: float = 30.0
    
    # Message settings
    heartbeat_interval: float = 30.0
    timeout: float = 10.0
    
    # Performance settings
    buffer_size: int = 1000
    max_events_per_second: int = 100
    batch_updates: bool = True
    batch_interval: float = 0.1
    
    # Channels
    channels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default channel configurations."""
        if not self.channels:
            self.channels = {
                "market_data": {
                    "enabled": True,
                    "filter": None,
                    "batch_updates": True
                },
                "trading_signals": {
                    "enabled": True,
                    "filter": None,
                    "batch_updates": False
                },
                "bot_status": {
                    "enabled": True,
                    "filter": None,
                    "batch_updates": True
                },
                "alerts": {
                    "enabled": True,
                    "filter": None,
                    "batch_updates": False
                },
                "performance": {
                    "enabled": True,
                    "filter": None,
                    "batch_updates": True
                },
                "neural_training": {
                    "enabled": True,
                    "filter": None,
                    "batch_updates": True
                }
            }

@dataclass
class E2BConfig:
    """E2B sandbox integration configuration."""
    
    # API settings
    api_key: str = field(default_factory=lambda: os.getenv("E2B_API_KEY", ""))
    api_url: str = "https://api.e2b.dev"
    timeout: int = 30
    
    # Default sandbox settings
    default_template: str = "neural-trader-base"
    default_cpu_count: int = 1
    default_memory_mb: int = 512
    default_timeout_seconds: int = 3600
    
    # Deployment settings
    max_concurrent_deployments: int = 10
    deployment_timeout: int = 300
    cleanup_inactive_after_hours: int = 24
    
    # Environment variables for sandboxes
    sandbox_env_vars: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate E2B configuration."""
        if not self.api_key:
            raise ValueError("E2B_API_KEY is required for sandbox integration")

# Configuration factory
class ConfigFactory:
    """Factory for creating various configuration objects."""
    
    @staticmethod
    def create_supabase_config(**overrides) -> SupabaseConfig:
        """Create Supabase configuration with overrides."""
        return SupabaseConfig.from_env(**overrides)
    
    @staticmethod
    def create_database_config(**overrides) -> DatabaseConfig:
        """Create database configuration with overrides."""
        return DatabaseConfig(**overrides)
    
    @staticmethod
    def create_realtime_config(**overrides) -> RealtimeConfig:
        """Create real-time configuration with overrides."""
        return RealtimeConfig(**overrides)
    
    @staticmethod
    def create_e2b_config(**overrides) -> E2BConfig:
        """Create E2B configuration with overrides."""
        return E2BConfig(**overrides)
    
    @staticmethod
    def create_complete_config(**overrides) -> Dict[str, Any]:
        """Create complete configuration with all components."""
        return {
            "supabase": ConfigFactory.create_supabase_config(**overrides.get("supabase", {})),
            "database": ConfigFactory.create_database_config(**overrides.get("database", {})),
            "realtime": ConfigFactory.create_realtime_config(**overrides.get("realtime", {})),
            "e2b": ConfigFactory.create_e2b_config(**overrides.get("e2b", {}))
        }

# Environment-specific configurations
def get_development_config() -> Dict[str, Any]:
    """Get configuration for development environment."""
    return {
        "supabase": {
            "timeout": 10,
            "max_retries": 1,
            "realtime_enabled": True
        },
        "database": {
            "min_connections": 1,
            "max_connections": 5,
            "command_timeout": 30.0
        },
        "realtime": {
            "max_reconnect_attempts": 3,
            "batch_updates": False
        }
    }

def get_production_config() -> Dict[str, Any]:
    """Get configuration for production environment."""
    return {
        "supabase": {
            "timeout": 30,
            "max_retries": 5,
            "realtime_enabled": True,
            "connection_pool_size": 20
        },
        "database": {
            "min_connections": 5,
            "max_connections": 50,
            "command_timeout": 60.0,
            "ssl_mode": "require"
        },
        "realtime": {
            "max_reconnect_attempts": 10,
            "batch_updates": True,
            "buffer_size": 5000
        }
    }

def get_testing_config() -> Dict[str, Any]:
    """Get configuration for testing environment."""
    return {
        "supabase": {
            "timeout": 5,
            "max_retries": 1,
            "realtime_enabled": False
        },
        "database": {
            "min_connections": 1,
            "max_connections": 2,
            "command_timeout": 10.0
        },
        "realtime": {
            "enabled": False
        }
    }