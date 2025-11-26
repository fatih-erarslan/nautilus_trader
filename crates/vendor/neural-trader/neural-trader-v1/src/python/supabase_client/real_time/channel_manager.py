"""
Real-time Channel Manager
========================

Python client for managing real-time Supabase channels and subscriptions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime
from uuid import UUID
import json

from ..client import AsyncSupabaseClient
from ..models.database_models import (
    MarketData, BotExecution, TradingBot, Alert, 
    PerformanceMetric, TrainingRun
)

logger = logging.getLogger(__name__)

class RealtimeChannelManager:
    """
    Manager for real-time Supabase channels and event subscriptions.
    """
    
    def __init__(self, supabase_client: AsyncSupabaseClient, user_id: Optional[UUID] = None):
        """
        Initialize real-time channel manager.
        
        Args:
            supabase_client: Async Supabase client instance
            user_id: Optional user ID for user-specific subscriptions
        """
        self.client = supabase_client
        self.user_id = user_id
        self.channels: Dict[str, Any] = {}
        self.handlers: Dict[str, Callable] = {}
        self.is_connected = False
        
        # Event handlers
        self.on_market_update: Optional[Callable[[MarketData], None]] = None
        self.on_signal_generated: Optional[Callable[[BotExecution], None]] = None
        self.on_bot_status_change: Optional[Callable[[TradingBot], None]] = None
        self.on_alert_triggered: Optional[Callable[[Alert], None]] = None
        self.on_performance_update: Optional[Callable[[PerformanceMetric], None]] = None
        self.on_training_progress: Optional[Callable[[TrainingRun], None]] = None
        self.on_order_executed: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_position_update: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_subscribed: Optional[Callable[[str], None]] = None
    
    def set_handlers(self, **handlers):
        """
        Set event handlers.
        
        Args:
            **handlers: Event handler functions
        """
        for event_name, handler in handlers.items():
            if hasattr(self, event_name):
                setattr(self, event_name, handler)
            else:
                logger.warning(f"Unknown event handler: {event_name}")
    
    async def connect(self) -> bool:
        """
        Connect to real-time services.
        
        Returns:
            True if connected successfully
        """
        try:
            if not self.client._connected:
                await self.client.connect()
            
            self.is_connected = True
            logger.info("Connected to real-time services")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to real-time services: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def disconnect(self):
        """Disconnect from all real-time channels."""
        try:
            # Unsubscribe from all channels
            for channel_name in list(self.channels.keys()):
                await self.unsubscribe(channel_name)
            
            self.is_connected = False
            logger.info("Disconnected from real-time services")
            
        except Exception as e:
            logger.error(f"Error disconnecting from real-time services: {e}")
    
    async def subscribe_to_market_data(
        self, 
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None
    ) -> bool:
        """
        Subscribe to market data updates.
        
        Args:
            symbols: Optional list of symbols to filter
            timeframes: Optional list of timeframes to filter
            
        Returns:
            True if subscription successful
        """
        try:
            channel_name = "market_data_channel"
            
            # In a real implementation, you'd set up the Supabase realtime subscription here
            # For now, we'll simulate the subscription
            
            logger.info(f"Subscribed to market data for symbols: {symbols}")
            self.channels[channel_name] = {
                "type": "market_data",
                "symbols": symbols,
                "timeframes": timeframes,
                "subscribed_at": datetime.utcnow()
            }
            
            if self.on_subscribed:
                self.on_subscribed(f"Market data: SUBSCRIBED")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to market data: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def subscribe_to_trading_signals(
        self, 
        bot_ids: Optional[List[UUID]] = None
    ) -> bool:
        """
        Subscribe to trading signal updates.
        
        Args:
            bot_ids: Optional list of bot IDs to filter
            
        Returns:
            True if subscription successful
        """
        try:
            channel_name = "trading_signals_channel"
            
            logger.info(f"Subscribed to trading signals for bots: {bot_ids}")
            self.channels[channel_name] = {
                "type": "trading_signals",
                "bot_ids": bot_ids,
                "subscribed_at": datetime.utcnow()
            }
            
            if self.on_subscribed:
                self.on_subscribed(f"Trading signals: SUBSCRIBED")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to trading signals: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def subscribe_to_bot_status(
        self, 
        bot_ids: Optional[List[UUID]] = None
    ) -> bool:
        """
        Subscribe to bot status changes.
        
        Args:
            bot_ids: Optional list of bot IDs to filter
            
        Returns:
            True if subscription successful
        """
        try:
            channel_name = "bot_status_channel"
            
            logger.info(f"Subscribed to bot status for bots: {bot_ids}")
            self.channels[channel_name] = {
                "type": "bot_status",
                "bot_ids": bot_ids,
                "subscribed_at": datetime.utcnow()
            }
            
            if self.on_subscribed:
                self.on_subscribed(f"Bot status: SUBSCRIBED")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to bot status: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def subscribe_to_alerts(self) -> bool:
        """
        Subscribe to user alerts.
        
        Returns:
            True if subscription successful
        """
        try:
            if not self.user_id:
                raise ValueError("User ID required for alert subscription")
            
            channel_name = "alerts_channel"
            
            logger.info(f"Subscribed to alerts for user: {self.user_id}")
            self.channels[channel_name] = {
                "type": "alerts",
                "user_id": self.user_id,
                "subscribed_at": datetime.utcnow()
            }
            
            if self.on_subscribed:
                self.on_subscribed(f"Alerts: SUBSCRIBED")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to alerts: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def subscribe_to_performance_metrics(
        self, 
        entity_ids: Optional[List[UUID]] = None
    ) -> bool:
        """
        Subscribe to performance metric updates.
        
        Args:
            entity_ids: Optional list of entity IDs to filter
            
        Returns:
            True if subscription successful
        """
        try:
            channel_name = "performance_channel"
            
            logger.info(f"Subscribed to performance metrics for entities: {entity_ids}")
            self.channels[channel_name] = {
                "type": "performance",
                "entity_ids": entity_ids,
                "subscribed_at": datetime.utcnow()
            }
            
            if self.on_subscribed:
                self.on_subscribed(f"Performance: SUBSCRIBED")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to performance metrics: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def subscribe_to_neural_training(
        self, 
        model_ids: Optional[List[UUID]] = None
    ) -> bool:
        """
        Subscribe to neural network training progress.
        
        Args:
            model_ids: Optional list of model IDs to filter
            
        Returns:
            True if subscription successful
        """
        try:
            channel_name = "neural_training_channel"
            
            logger.info(f"Subscribed to neural training for models: {model_ids}")
            self.channels[channel_name] = {
                "type": "neural_training",
                "model_ids": model_ids,
                "subscribed_at": datetime.utcnow()
            }
            
            if self.on_subscribed:
                self.on_subscribed(f"Neural training: SUBSCRIBED")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to neural training: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def subscribe_to_order_executions(
        self, 
        account_ids: Optional[List[UUID]] = None
    ) -> bool:
        """
        Subscribe to order execution updates.
        
        Args:
            account_ids: Optional list of account IDs to filter
            
        Returns:
            True if subscription successful
        """
        try:
            channel_name = "orders_channel"
            
            logger.info(f"Subscribed to order executions for accounts: {account_ids}")
            self.channels[channel_name] = {
                "type": "orders",
                "account_ids": account_ids,
                "subscribed_at": datetime.utcnow()
            }
            
            if self.on_subscribed:
                self.on_subscribed(f"Orders: SUBSCRIBED")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to order executions: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def subscribe_to_position_updates(
        self, 
        account_ids: Optional[List[UUID]] = None
    ) -> bool:
        """
        Subscribe to position updates.
        
        Args:
            account_ids: Optional list of account IDs to filter
            
        Returns:
            True if subscription successful
        """
        try:
            channel_name = "positions_channel"
            
            logger.info(f"Subscribed to position updates for accounts: {account_ids}")
            self.channels[channel_name] = {
                "type": "positions",
                "account_ids": account_ids,
                "subscribed_at": datetime.utcnow()
            }
            
            if self.on_subscribed:
                self.on_subscribed(f"Positions: SUBSCRIBED")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to position updates: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def subscribe_to_all(self) -> bool:
        """
        Subscribe to all user-relevant channels.
        
        Returns:
            True if all subscriptions successful
        """
        try:
            if not self.user_id:
                raise ValueError("User ID required for full subscription")
            
            results = await asyncio.gather(
                self.subscribe_to_alerts(),
                self.subscribe_to_bot_status(),
                self.subscribe_to_neural_training(),
                self.subscribe_to_performance_metrics(),
                return_exceptions=True
            )
            
            # Check if any subscriptions failed
            failed_subscriptions = [r for r in results if isinstance(r, Exception)]
            
            if failed_subscriptions:
                logger.warning(f"Some subscriptions failed: {failed_subscriptions}")
                return False
            
            logger.info("Successfully subscribed to all channels")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to all channels: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def unsubscribe(self, channel_name: str) -> bool:
        """
        Unsubscribe from a specific channel.
        
        Args:
            channel_name: Name of the channel to unsubscribe from
            
        Returns:
            True if unsubscription successful
        """
        try:
            if channel_name in self.channels:
                # In a real implementation, you'd unsubscribe from the Supabase channel here
                del self.channels[channel_name]
                logger.info(f"Unsubscribed from channel: {channel_name}")
                return True
            else:
                logger.warning(f"Channel not found: {channel_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {channel_name}: {e}")
            return False
    
    async def unsubscribe_all(self):
        """Unsubscribe from all channels."""
        try:
            for channel_name in list(self.channels.keys()):
                await self.unsubscribe(channel_name)
            
            logger.info("Unsubscribed from all channels")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from all channels: {e}")
    
    def get_active_channels(self) -> List[str]:
        """
        Get list of active channel names.
        
        Returns:
            List of active channel names
        """
        return list(self.channels.keys())
    
    def is_subscribed(self, channel_name: str) -> bool:
        """
        Check if subscribed to a specific channel.
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            True if subscribed to the channel
        """
        return channel_name in self.channels
    
    def get_channel_info(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific channel.
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            Channel information dictionary or None if not found
        """
        return self.channels.get(channel_name)
    
    async def simulate_market_update(self, symbol: str, price: float, volume: float):
        """
        Simulate a market data update (for testing).
        
        Args:
            symbol: Symbol name
            price: Current price
            volume: Volume
        """
        if self.on_market_update and "market_data_channel" in self.channels:
            # Create a mock MarketData object
            market_data = MarketData(
                symbol_id=UUID('00000000-0000-0000-0000-000000000000'),
                timestamp=datetime.utcnow(),
                open=price * 0.99,
                high=price * 1.01,
                low=price * 0.98,
                close=price,
                volume=volume,
                timeframe="1m"
            )
            
            try:
                await asyncio.create_task(
                    asyncio.coroutine(lambda: self.on_market_update(market_data))()
                )
            except Exception as e:
                logger.error(f"Error in market update handler: {e}")
    
    async def simulate_trading_signal(
        self, 
        bot_id: UUID, 
        symbol_id: UUID, 
        action: str, 
        strength: float
    ):
        """
        Simulate a trading signal (for testing).
        
        Args:
            bot_id: Bot ID
            symbol_id: Symbol ID
            action: Trading action
            strength: Signal strength
        """
        if self.on_signal_generated and "trading_signals_channel" in self.channels:
            # Create a mock BotExecution object
            signal = BotExecution(
                bot_id=bot_id,
                symbol_id=symbol_id,
                action=action,
                signal_strength=strength,
                executed_at=datetime.utcnow()
            )
            
            try:
                await asyncio.create_task(
                    asyncio.coroutine(lambda: self.on_signal_generated(signal))()
                )
            except Exception as e:
                logger.error(f"Error in signal handler: {e}")
    
    async def start_heartbeat(self, interval: float = 30.0):
        """
        Start heartbeat to maintain connection.
        
        Args:
            interval: Heartbeat interval in seconds
        """
        while self.is_connected:
            try:
                # In a real implementation, you'd send a heartbeat to Supabase
                logger.debug("Heartbeat sent")
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                if self.on_error:
                    self.on_error(e)
                break
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

# Utility classes for batching and filtering

class BatchedUpdateManager:
    """Manager for batching real-time updates to reduce UI thrashing."""
    
    def __init__(self, batch_interval: float = 0.1):
        """
        Initialize batched update manager.
        
        Args:
            batch_interval: Time to wait before processing batch (seconds)
        """
        self.batch_interval = batch_interval
        self.update_queue: List[Any] = []
        self.batch_timer: Optional[asyncio.Task] = None
        self.processors: Dict[str, Callable] = {}
    
    def add_update(self, update_type: str, data: Any):
        """
        Add an update to the batch queue.
        
        Args:
            update_type: Type of update
            data: Update data
        """
        self.update_queue.append({"type": update_type, "data": data})
        
        if self.batch_timer:
            self.batch_timer.cancel()
        
        self.batch_timer = asyncio.create_task(self._process_batch())
    
    async def _process_batch(self):
        """Process the batch after delay."""
        await asyncio.sleep(self.batch_interval)
        
        if self.update_queue:
            # Group updates by type and symbol
            grouped_updates = {}
            
            for update in self.update_queue:
                update_type = update["type"]
                data = update["data"]
                
                if update_type not in grouped_updates:
                    grouped_updates[update_type] = []
                
                grouped_updates[update_type].append(data)
            
            # Process each group
            for update_type, updates in grouped_updates.items():
                if update_type in self.processors:
                    try:
                        await self.processors[update_type](updates)
                    except Exception as e:
                        logger.error(f"Error processing batch for {update_type}: {e}")
            
            self.update_queue.clear()
    
    def set_processor(self, update_type: str, processor: Callable):
        """
        Set a processor for a specific update type.
        
        Args:
            update_type: Type of update
            processor: Async function to process batched updates
        """
        self.processors[update_type] = processor

class RateLimitedChannelManager(RealtimeChannelManager):
    """Rate-limited version of channel manager to prevent overwhelming."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_counts: Dict[str, int] = {}
        self.rate_limits: Dict[str, int] = {
            "market_data": 1000,  # per minute
            "trading_signals": 100,
            "bot_status": 50,
            "alerts": 20
        }
        self.reset_interval = 60  # seconds
        self._start_rate_limit_reset()
    
    def _start_rate_limit_reset(self):
        """Start the rate limit reset timer."""
        async def reset_counts():
            while True:
                await asyncio.sleep(self.reset_interval)
                self.update_counts.clear()
        
        asyncio.create_task(reset_counts())
    
    def _check_rate_limit(self, channel_type: str) -> bool:
        """
        Check if rate limit allows processing.
        
        Args:
            channel_type: Type of channel
            
        Returns:
            True if within rate limit
        """
        count = self.update_counts.get(channel_type, 0)
        limit = self.rate_limits.get(channel_type, 10)
        
        if count >= limit:
            logger.warning(f"Rate limit exceeded for {channel_type}")
            return False
        
        self.update_counts[channel_type] = count + 1
        return True