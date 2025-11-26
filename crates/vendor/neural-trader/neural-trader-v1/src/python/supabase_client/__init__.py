"""
Supabase Python Client for Neural Trading Platform
==================================================

Comprehensive Python wrappers for Supabase integration with neural trading systems.

This package provides:
- Database operations with async support
- Real-time data streaming
- Neural model management
- Trading bot orchestration
- Performance monitoring
- E2B sandbox integration

Usage:
    >>> from supabase_client import NeuralTradingClient
    >>> client = NeuralTradingClient(url="your-url", key="your-key")
    >>> await client.connect()

Author: Neural Trading Platform Team
Version: 1.0.0
"""

from .config import SupabaseConfig
from .client import SupabaseClient, AsyncSupabaseClient
from .clients.neural_models import NeuralModelsClient
from .clients.trading_bots import TradingBotsClient
from .clients.sandbox_integration import SandboxIntegrationClient
from .real_time.channel_manager import RealtimeChannelManager
from .monitoring.performance_monitor import PerformanceMonitor
from .models.database_models import *

__version__ = "1.0.0"
__author__ = "Neural Trading Platform Team"

# Main client class for easy access
class NeuralTradingClient:
    """
    Main client class that provides access to all Supabase functionality
    for the neural trading platform.
    """
    
    def __init__(self, url: str, key: str, service_key: str = None, **kwargs):
        """
        Initialize the Neural Trading Client.
        
        Args:
            url: Supabase project URL
            key: Supabase anon key
            service_key: Supabase service role key (for admin operations)
            **kwargs: Additional configuration options
        """
        self.config = SupabaseConfig(
            url=url,
            anon_key=key,
            service_key=service_key,
            **kwargs
        )
        
        # Initialize core client
        self.supabase = AsyncSupabaseClient(self.config)
        
        # Initialize specialized clients
        self.neural_models = NeuralModelsClient(self.supabase)
        self.trading_bots = TradingBotsClient(self.supabase)
        self.sandbox = SandboxIntegrationClient(self.supabase)
        self.realtime = RealtimeChannelManager(self.supabase)
        self.performance = PerformanceMonitor(self.supabase)
    
    async def connect(self):
        """Establish connection to Supabase."""
        await self.supabase.connect()
        return self
    
    async def disconnect(self):
        """Close connection to Supabase."""
        await self.supabase.disconnect()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return await self.connect()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

# Convenience exports
__all__ = [
    "NeuralTradingClient",
    "SupabaseConfig", 
    "SupabaseClient",
    "AsyncSupabaseClient",
    "NeuralModelsClient",
    "TradingBotsClient", 
    "SandboxIntegrationClient",
    "RealtimeChannelManager",
    "PerformanceMonitor",
    # Database models
    "Profile",
    "Symbol", 
    "MarketData",
    "NewsData",
    "TradingAccount",
    "Position",
    "Order",
    "NeuralModel",
    "TrainingRun",
    "ModelPrediction",
    "TradingBot",
    "BotExecution",
    "SandboxDeployment",
    "PerformanceMetric",
    "Alert",
    "AuditLog"
]