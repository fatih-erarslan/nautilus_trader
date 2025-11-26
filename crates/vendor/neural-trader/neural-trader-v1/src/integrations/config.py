"""
Configuration for News API Integrations
Centralized configuration management for all news providers and integration settings
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a single news provider"""
    enabled: bool = True
    api_key: Optional[str] = None
    plan: str = "free"
    timeout: int = 30
    max_retries: int = 3
    priority: int = 1
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: str = "Exception"


@dataclass
class CacheConfig:
    """Cache configuration"""
    redis_url: str = "redis://localhost:6379"
    default_ttl: int = 3600
    compression_enabled: bool = True
    max_items_per_key: int = 500


@dataclass
class ScoringConfig:
    """News relevance scoring configuration"""
    symbol_weight: float = 0.35
    sentiment_weight: float = 0.20
    source_weight: float = 0.20
    recency_weight: float = 0.15
    impact_weight: float = 0.10
    similarity_threshold: float = 0.85


class NewsIntegrationConfig:
    """Main configuration class for news integrations"""
    
    def __init__(self):
        self.providers = self._load_provider_configs()
        self.cache = self._load_cache_config()
        self.scoring = self._load_scoring_config()
        self.circuit_breaker = self._load_circuit_breaker_config()
        self.general = self._load_general_config()
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Loaded news integration configuration")
    
    def _load_provider_configs(self) -> Dict[str, ProviderConfig]:
        """Load provider-specific configurations"""
        configs = {}
        
        # Alpha Vantage configuration
        configs['alpha_vantage'] = ProviderConfig(
            enabled=self._get_bool_env('ALPHA_VANTAGE_ENABLED', True),
            api_key=os.environ.get('ALPHA_VANTAGE_API_KEY'),
            plan=os.environ.get('ALPHA_VANTAGE_PLAN', 'free'),
            timeout=self._get_int_env('ALPHA_VANTAGE_TIMEOUT', 30),
            max_retries=self._get_int_env('ALPHA_VANTAGE_RETRIES', 3),
            priority=self._get_int_env('ALPHA_VANTAGE_PRIORITY', 1),
            custom_params={
                'rate_limit_calls_per_minute': self._get_int_env('ALPHA_VANTAGE_RATE_LIMIT', 5),
                'include_sentiment': self._get_bool_env('ALPHA_VANTAGE_SENTIMENT', True),
                'min_relevance_score': self._get_float_env('ALPHA_VANTAGE_MIN_RELEVANCE', 0.3)
            }
        )
        
        # NewsAPI configuration
        configs['newsapi'] = ProviderConfig(
            enabled=self._get_bool_env('NEWSAPI_ENABLED', True),
            api_key=os.environ.get('NEWSAPI_API_KEY'),
            plan=os.environ.get('NEWSAPI_PLAN', 'developer'),
            timeout=self._get_int_env('NEWSAPI_TIMEOUT', 30),
            max_retries=self._get_int_env('NEWSAPI_RETRIES', 3),
            priority=self._get_int_env('NEWSAPI_PRIORITY', 2),
            custom_params={
                'languages': os.environ.get('NEWSAPI_LANGUAGES', 'en').split(','),
                'countries': os.environ.get('NEWSAPI_COUNTRIES', 'us').split(','),
                'enable_content_filtering': self._get_bool_env('NEWSAPI_CONTENT_FILTER', True),
                'min_article_length': self._get_int_env('NEWSAPI_MIN_LENGTH', 100),
                'financial_sources_only': self._get_bool_env('NEWSAPI_FINANCIAL_ONLY', True)
            }
        )
        
        # Finnhub configuration
        configs['finnhub'] = ProviderConfig(
            enabled=self._get_bool_env('FINNHUB_ENABLED', True),
            api_key=os.environ.get('FINNHUB_API_KEY'),
            plan=os.environ.get('FINNHUB_PLAN', 'free'),
            timeout=self._get_int_env('FINNHUB_TIMEOUT', 30),
            max_retries=self._get_int_env('FINNHUB_RETRIES', 3),
            priority=self._get_int_env('FINNHUB_PRIORITY', 3),
            custom_params={
                'enable_websocket': self._get_bool_env('FINNHUB_WEBSOCKET', False),
                'websocket_symbols': os.environ.get('FINNHUB_WS_SYMBOLS', '').split(','),
                'include_market_data': self._get_bool_env('FINNHUB_MARKET_DATA', True),
                'news_categories': os.environ.get('FINNHUB_CATEGORIES', 'general,earnings').split(',')
            }
        )
        
        return configs
    
    def _load_cache_config(self) -> CacheConfig:
        """Load cache configuration"""
        return CacheConfig(
            redis_url=os.environ.get('REDIS_URL', 'redis://localhost:6379'),
            default_ttl=self._get_int_env('CACHE_DEFAULT_TTL', 3600),
            compression_enabled=self._get_bool_env('CACHE_COMPRESSION', True),
            max_items_per_key=self._get_int_env('CACHE_MAX_ITEMS', 500)
        )
    
    def _load_scoring_config(self) -> ScoringConfig:
        """Load scoring configuration"""
        return ScoringConfig(
            symbol_weight=self._get_float_env('SCORING_SYMBOL_WEIGHT', 0.35),
            sentiment_weight=self._get_float_env('SCORING_SENTIMENT_WEIGHT', 0.20),
            source_weight=self._get_float_env('SCORING_SOURCE_WEIGHT', 0.20),
            recency_weight=self._get_float_env('SCORING_RECENCY_WEIGHT', 0.15),
            impact_weight=self._get_float_env('SCORING_IMPACT_WEIGHT', 0.10),
            similarity_threshold=self._get_float_env('SCORING_SIMILARITY_THRESHOLD', 0.85)
        )
    
    def _load_circuit_breaker_config(self) -> CircuitBreakerConfig:
        """Load circuit breaker configuration"""
        return CircuitBreakerConfig(
            failure_threshold=self._get_int_env('CIRCUIT_BREAKER_THRESHOLD', 5),
            recovery_timeout=self._get_int_env('CIRCUIT_BREAKER_TIMEOUT', 60),
            expected_exception=os.environ.get('CIRCUIT_BREAKER_EXCEPTION', 'Exception')
        )
    
    def _load_general_config(self) -> Dict[str, Any]:
        """Load general configuration settings"""
        return {
            'max_concurrent_requests': self._get_int_env('MAX_CONCURRENT_REQUESTS', 10),
            'request_timeout': self._get_int_env('REQUEST_TIMEOUT', 30),
            'min_source_reliability': self._get_float_env('MIN_SOURCE_RELIABILITY', 0.3),
            'enable_gpu_acceleration': self._get_bool_env('ENABLE_GPU_ACCELERATION', False),
            'log_level': os.environ.get('LOG_LEVEL', 'INFO'),
            'metrics_enabled': self._get_bool_env('METRICS_ENABLED', True),
            'health_check_interval': self._get_int_env('HEALTH_CHECK_INTERVAL', 300),
            
            # Rate limiting
            'global_rate_limit_enabled': self._get_bool_env('GLOBAL_RATE_LIMIT', True),
            'global_requests_per_minute': self._get_int_env('GLOBAL_RATE_LIMIT_RPM', 100),
            
            # Failover settings
            'failover_enabled': self._get_bool_env('FAILOVER_ENABLED', True),
            'failover_retry_attempts': self._get_int_env('FAILOVER_RETRIES', 3),
            'failover_retry_delay': self._get_float_env('FAILOVER_RETRY_DELAY', 1.0),
            
            # Data processing
            'enable_sentiment_analysis': self._get_bool_env('ENABLE_SENTIMENT_ANALYSIS', True),
            'enable_entity_extraction': self._get_bool_env('ENABLE_ENTITY_EXTRACTION', True),
            'enable_impact_scoring': self._get_bool_env('ENABLE_IMPACT_SCORING', True),
            
            # Storage settings
            'store_full_content': self._get_bool_env('STORE_FULL_CONTENT', False),
            'data_retention_days': self._get_int_env('DATA_RETENTION_DAYS', 30),
            
            # Performance tuning
            'batch_processing_enabled': self._get_bool_env('BATCH_PROCESSING', True),
            'batch_size': self._get_int_env('BATCH_SIZE', 32),
            'parallel_processing_workers': self._get_int_env('PARALLEL_WORKERS', 4)
        }
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable"""
        value = os.environ.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer environment variable"""
        try:
            return int(os.environ.get(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid integer value for {key}, using default: {default}")
            return default
    
    def _get_float_env(self, key: str, default: float) -> float:
        """Get float environment variable"""
        try:
            return float(os.environ.get(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid float value for {key}, using default: {default}")
            return default
    
    def _validate_config(self):
        """Validate configuration settings"""
        errors = []
        
        # Check that at least one provider is enabled
        enabled_providers = [name for name, config in self.providers.items() if config.enabled]
        if not enabled_providers:
            errors.append("At least one news provider must be enabled")
        
        # Check API keys for enabled providers
        for name, config in self.providers.items():
            if config.enabled and not config.api_key:
                errors.append(f"API key required for enabled provider: {name}")
        
        # Validate scoring weights
        scoring_weights = [
            self.scoring.symbol_weight,
            self.scoring.sentiment_weight,
            self.scoring.source_weight,
            self.scoring.recency_weight,
            self.scoring.impact_weight
        ]
        
        total_weight = sum(scoring_weights)
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Scoring weights must sum to 1.0, got {total_weight}")
        
        # Validate cache configuration
        if not self.cache.redis_url:
            errors.append("Redis URL is required for caching")
        
        if self.cache.default_ttl <= 0:
            errors.append("Cache TTL must be positive")
        
        # Log validation results
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            logger.info("Configuration validation passed")
    
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider"""
        return self.providers.get(provider_name)
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled provider names"""
        return [name for name, config in self.providers.items() if config.enabled]
    
    def update_provider_config(self, provider_name: str, **kwargs):
        """Update provider configuration"""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        config = self.providers[provider_name]
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration key for {provider_name}: {key}")
        
        logger.info(f"Updated configuration for {provider_name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'providers': {
                name: {
                    'enabled': config.enabled,
                    'plan': config.plan,
                    'timeout': config.timeout,
                    'max_retries': config.max_retries,
                    'priority': config.priority,
                    'has_api_key': bool(config.api_key),
                    'custom_params': config.custom_params
                }
                for name, config in self.providers.items()
            },
            'cache': {
                'redis_url': self.cache.redis_url,
                'default_ttl': self.cache.default_ttl,
                'compression_enabled': self.cache.compression_enabled,
                'max_items_per_key': self.cache.max_items_per_key
            },
            'scoring': {
                'symbol_weight': self.scoring.symbol_weight,
                'sentiment_weight': self.scoring.sentiment_weight,
                'source_weight': self.scoring.source_weight,
                'recency_weight': self.scoring.recency_weight,
                'impact_weight': self.scoring.impact_weight,
                'similarity_threshold': self.scoring.similarity_threshold
            },
            'circuit_breaker': {
                'failure_threshold': self.circuit_breaker.failure_threshold,
                'recovery_timeout': self.circuit_breaker.recovery_timeout,
                'expected_exception': self.circuit_breaker.expected_exception
            },
            'general': self.general
        }
    
    @classmethod
    def create_example_env_file(cls, filepath: str = ".env.example"):
        """Create example environment file with all configuration options"""
        env_content = """# News API Integration Configuration

# Alpha Vantage
ALPHA_VANTAGE_ENABLED=true
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
ALPHA_VANTAGE_PLAN=free
ALPHA_VANTAGE_TIMEOUT=30
ALPHA_VANTAGE_RETRIES=3
ALPHA_VANTAGE_PRIORITY=1
ALPHA_VANTAGE_RATE_LIMIT=5
ALPHA_VANTAGE_SENTIMENT=true
ALPHA_VANTAGE_MIN_RELEVANCE=0.3

# NewsAPI
NEWSAPI_ENABLED=true
NEWSAPI_API_KEY=your_newsapi_key
NEWSAPI_PLAN=developer
NEWSAPI_TIMEOUT=30
NEWSAPI_RETRIES=3
NEWSAPI_PRIORITY=2
NEWSAPI_LANGUAGES=en
NEWSAPI_COUNTRIES=us
NEWSAPI_CONTENT_FILTER=true
NEWSAPI_MIN_LENGTH=100
NEWSAPI_FINANCIAL_ONLY=true

# Finnhub
FINNHUB_ENABLED=true
FINNHUB_API_KEY=your_finnhub_api_key
FINNHUB_PLAN=free
FINNHUB_TIMEOUT=30
FINNHUB_RETRIES=3
FINNHUB_PRIORITY=3
FINNHUB_WEBSOCKET=false
FINNHUB_WS_SYMBOLS=
FINNHUB_MARKET_DATA=true
FINNHUB_CATEGORIES=general,earnings

# Cache Configuration
REDIS_URL=redis://localhost:6379
CACHE_DEFAULT_TTL=3600
CACHE_COMPRESSION=true
CACHE_MAX_ITEMS=500

# Scoring Configuration
SCORING_SYMBOL_WEIGHT=0.35
SCORING_SENTIMENT_WEIGHT=0.20
SCORING_SOURCE_WEIGHT=0.20
SCORING_RECENCY_WEIGHT=0.15
SCORING_IMPACT_WEIGHT=0.10
SCORING_SIMILARITY_THRESHOLD=0.85

# Circuit Breaker
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60
CIRCUIT_BREAKER_EXCEPTION=Exception

# General Settings
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30
MIN_SOURCE_RELIABILITY=0.3
ENABLE_GPU_ACCELERATION=false
LOG_LEVEL=INFO
METRICS_ENABLED=true
HEALTH_CHECK_INTERVAL=300

# Rate Limiting
GLOBAL_RATE_LIMIT=true
GLOBAL_RATE_LIMIT_RPM=100

# Failover
FAILOVER_ENABLED=true
FAILOVER_RETRIES=3
FAILOVER_RETRY_DELAY=1.0

# Data Processing
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_ENTITY_EXTRACTION=true
ENABLE_IMPACT_SCORING=true

# Storage
STORE_FULL_CONTENT=false
DATA_RETENTION_DAYS=30

# Performance
BATCH_PROCESSING=true
BATCH_SIZE=32
PARALLEL_WORKERS=4
"""
        
        with open(filepath, 'w') as f:
            f.write(env_content)
        
        logger.info(f"Created example environment file: {filepath}")


# Global configuration instance
_config_instance = None


def get_config() -> NewsIntegrationConfig:
    """Get global configuration instance (singleton)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = NewsIntegrationConfig()
    return _config_instance


def reload_config():
    """Reload configuration from environment"""
    global _config_instance
    _config_instance = NewsIntegrationConfig()
    logger.info("Configuration reloaded")


# Development/testing configurations
def get_test_config() -> NewsIntegrationConfig:
    """Get configuration optimized for testing"""
    config = NewsIntegrationConfig()
    
    # Disable actual API calls in tests
    for provider_config in config.providers.values():
        provider_config.timeout = 5
        provider_config.max_retries = 1
    
    # Use faster cache TTL
    config.cache.default_ttl = 60
    
    # More aggressive circuit breaker for testing
    config.circuit_breaker.failure_threshold = 2
    config.circuit_breaker.recovery_timeout = 10
    
    return config


def get_development_config() -> NewsIntegrationConfig:
    """Get configuration optimized for development"""
    config = NewsIntegrationConfig()
    
    # More verbose logging
    config.general['log_level'] = 'DEBUG'
    
    # Faster health checks
    config.general['health_check_interval'] = 60
    
    # Lower rate limits to avoid hitting API limits during development
    for provider_config in config.providers.values():
        if 'rate_limit_calls_per_minute' in provider_config.custom_params:
            provider_config.custom_params['rate_limit_calls_per_minute'] = 2
    
    return config