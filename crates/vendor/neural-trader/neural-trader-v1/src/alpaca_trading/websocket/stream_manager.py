"""Stream Manager for Alpaca WebSocket subscriptions."""

import asyncio
import logging
from typing import Dict, List, Set, Optional, Callable, Any
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class StreamManager:
    """Manages WebSocket stream subscriptions and routing.
    
    Features:
    - Batch subscription management
    - Symbol filtering and routing
    - Subscription tracking and optimization
    - Rate limiting for subscription changes
    """
    
    def __init__(self, client: 'AlpacaWebSocketClient', batch_size: int = 100):
        """Initialize stream manager.
        
        Args:
            client: AlpacaWebSocketClient instance
            batch_size: Maximum symbols per subscription batch
        """
        self.client = client
        self.batch_size = batch_size
        
        # Active subscriptions by type and symbol
        self.active_subscriptions: Dict[str, Set[str]] = {
            "trades": set(),
            "quotes": set(),
            "bars": set(),
            "dailyBars": set(),
            "statuses": set(),
            "lulds": set()
        }
        
        # Pending subscription changes
        self.pending_subscribe: Dict[str, Set[str]] = defaultdict(set)
        self.pending_unsubscribe: Dict[str, Set[str]] = defaultdict(set)
        
        # Symbol routing
        self.symbol_filters: Dict[str, List[Callable]] = defaultdict(list)
        self.symbol_handlers: Dict[str, Dict[str, List[Callable]]] = defaultdict(lambda: defaultdict(list))
        
        # Rate limiting
        self.last_subscription_time = 0
        self.min_subscription_interval = 0.5  # seconds
        
        # Subscription task
        self.subscription_task: Optional[asyncio.Task] = None
        self.running = False
        
    def start(self) -> None:
        """Start the stream manager."""
        if not self.running:
            self.running = True
            self.subscription_task = asyncio.create_task(self._process_subscription_queue())
            logger.info("Stream manager started")
    
    def stop(self) -> None:
        """Stop the stream manager."""
        self.running = False
        if self.subscription_task:
            self.subscription_task.cancel()
        logger.info("Stream manager stopped")
    
    async def subscribe(
        self,
        symbols: List[str],
        data_types: List[str],
        handler: Optional[Callable] = None,
        filter_func: Optional[Callable] = None
    ) -> None:
        """Subscribe to market data for symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            data_types: List of data types ("trades", "quotes", "bars", etc.)
            handler: Optional handler for these specific symbols
            filter_func: Optional filter function for messages
        """
        # Validate data types
        valid_types = {"trades", "quotes", "bars", "dailyBars", "statuses", "lulds"}
        for data_type in data_types:
            if data_type not in valid_types:
                raise ValueError(f"Invalid data type: {data_type}")
        
        # Add to pending subscriptions
        for data_type in data_types:
            for symbol in symbols:
                if symbol not in self.active_subscriptions[data_type]:
                    self.pending_subscribe[data_type].add(symbol)
                    self.pending_unsubscribe[data_type].discard(symbol)
                
                # Register handler if provided
                if handler:
                    self.symbol_handlers[symbol][data_type].append(handler)
                
                # Register filter if provided
                if filter_func:
                    self.symbol_filters[symbol].append(filter_func)
        
        logger.info(f"Queued subscription for {len(symbols)} symbols, types: {data_types}")
    
    async def unsubscribe(
        self,
        symbols: List[str],
        data_types: Optional[List[str]] = None
    ) -> None:
        """Unsubscribe from market data for symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
            data_types: Optional list of data types to unsubscribe from (all if None)
        """
        if data_types is None:
            data_types = list(self.active_subscriptions.keys())
        
        # Add to pending unsubscriptions
        for data_type in data_types:
            for symbol in symbols:
                if symbol in self.active_subscriptions[data_type]:
                    self.pending_unsubscribe[data_type].add(symbol)
                    self.pending_subscribe[data_type].discard(symbol)
                
                # Remove handlers
                if symbol in self.symbol_handlers:
                    self.symbol_handlers[symbol].pop(data_type, None)
                    if not self.symbol_handlers[symbol]:
                        del self.symbol_handlers[symbol]
                
                # Remove filters if no handlers left
                if symbol not in self.symbol_handlers:
                    self.symbol_filters.pop(symbol, None)
        
        logger.info(f"Queued unsubscription for {len(symbols)} symbols, types: {data_types}")
    
    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all active subscriptions."""
        for data_type, symbols in self.active_subscriptions.items():
            if symbols:
                self.pending_unsubscribe[data_type].update(symbols)
                self.pending_subscribe[data_type].clear()
        
        logger.info("Queued unsubscription from all symbols")
    
    def register_global_handler(self, data_type: str, handler: Callable) -> None:
        """Register a global handler for all symbols of a data type.
        
        Args:
            data_type: Type of data ("trades", "quotes", etc.)
            handler: Callback function
        """
        self.client.register_handler(self._get_message_type(data_type), handler)
    
    async def _process_subscription_queue(self) -> None:
        """Process pending subscription changes."""
        while self.running:
            try:
                # Check if there are pending changes
                has_changes = any(self.pending_subscribe.values()) or any(self.pending_unsubscribe.values())
                
                if has_changes and self.client.connected and self.client.authenticated:
                    # Rate limit subscription changes
                    time_since_last = time.time() - self.last_subscription_time
                    if time_since_last < self.min_subscription_interval:
                        await asyncio.sleep(self.min_subscription_interval - time_since_last)
                    
                    # Process unsubscriptions
                    if any(self.pending_unsubscribe.values()):
                        await self._process_unsubscriptions()
                    
                    # Process subscriptions
                    if any(self.pending_subscribe.values()):
                        await self._process_subscriptions()
                    
                    self.last_subscription_time = time.time()
                
                # Sleep briefly to avoid busy loop
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing subscription queue: {e}")
                await asyncio.sleep(1)
    
    async def _process_subscriptions(self) -> None:
        """Process pending subscriptions in batches."""
        subscription_batch = {}
        
        for data_type, symbols in self.pending_subscribe.items():
            if symbols:
                # Convert to list and batch
                symbol_list = list(symbols)
                
                # Process in batches
                for i in range(0, len(symbol_list), self.batch_size):
                    batch = symbol_list[i:i + self.batch_size]
                    
                    if data_type not in subscription_batch:
                        subscription_batch[data_type] = []
                    subscription_batch[data_type].extend(batch)
                    
                    # Update active subscriptions
                    self.active_subscriptions[data_type].update(batch)
                
                # Clear pending
                self.pending_subscribe[data_type].clear()
        
        # Send subscription request
        if subscription_batch:
            await self.client.subscribe(**subscription_batch)
            logger.info(f"Subscribed to {sum(len(v) for v in subscription_batch.values())} symbols")
    
    async def _process_unsubscriptions(self) -> None:
        """Process pending unsubscriptions in batches."""
        unsubscription_batch = {}
        
        for data_type, symbols in self.pending_unsubscribe.items():
            if symbols:
                # Convert to list and batch
                symbol_list = list(symbols)
                
                # Process in batches
                for i in range(0, len(symbol_list), self.batch_size):
                    batch = symbol_list[i:i + self.batch_size]
                    
                    if data_type not in unsubscription_batch:
                        unsubscription_batch[data_type] = []
                    unsubscription_batch[data_type].extend(batch)
                    
                    # Update active subscriptions
                    self.active_subscriptions[data_type].difference_update(batch)
                
                # Clear pending
                self.pending_unsubscribe[data_type].clear()
        
        # Send unsubscription request
        if unsubscription_batch:
            await self.client.unsubscribe(**unsubscription_batch)
            logger.info(f"Unsubscribed from {sum(len(v) for v in unsubscription_batch.values())} symbols")
    
    def _get_message_type(self, data_type: str) -> str:
        """Get message type identifier for data type."""
        message_types = {
            "trades": "t",
            "quotes": "q",
            "bars": "b",
            "dailyBars": "d",
            "statuses": "s",
            "lulds": "l"
        }
        return message_types.get(data_type, data_type)
    
    async def route_message(self, message: Dict[str, Any]) -> None:
        """Route message to appropriate handlers.
        
        Args:
            message: Incoming message from WebSocket
        """
        symbol = message.get("S", message.get("symbol"))
        msg_type = message.get("T", message.get("msg"))
        
        if not symbol:
            return
        
        # Apply filters
        if symbol in self.symbol_filters:
            for filter_func in self.symbol_filters[symbol]:
                if not filter_func(message):
                    return
        
        # Route to symbol-specific handlers
        if symbol in self.symbol_handlers:
            # Determine data type from message type
            data_type = self._get_data_type_from_message(msg_type)
            
            if data_type and data_type in self.symbol_handlers[symbol]:
                for handler in self.symbol_handlers[symbol][data_type]:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Handler error for {symbol}/{data_type}: {e}")
    
    def _get_data_type_from_message(self, msg_type: str) -> Optional[str]:
        """Get data type from message type."""
        type_mapping = {
            "t": "trades",
            "q": "quotes",
            "b": "bars",
            "d": "dailyBars",
            "s": "statuses",
            "l": "lulds"
        }
        return type_mapping.get(msg_type)
    
    def get_subscription_stats(self) -> Dict[str, Any]:
        """Get subscription statistics."""
        return {
            "active_subscriptions": {k: len(v) for k, v in self.active_subscriptions.items()},
            "total_active": sum(len(v) for v in self.active_subscriptions.values()),
            "pending_subscribe": {k: len(v) for k, v in self.pending_subscribe.items()},
            "pending_unsubscribe": {k: len(v) for k, v in self.pending_unsubscribe.items()},
            "symbol_handlers": len(self.symbol_handlers),
            "symbol_filters": len(self.symbol_filters)
        }
    
    def get_active_symbols(self, data_type: Optional[str] = None) -> List[str]:
        """Get list of actively subscribed symbols.
        
        Args:
            data_type: Optional data type filter
        
        Returns:
            List of symbols
        """
        if data_type:
            return list(self.active_subscriptions.get(data_type, set()))
        
        # Return all unique symbols
        all_symbols = set()
        for symbols in self.active_subscriptions.values():
            all_symbols.update(symbols)
        return list(all_symbols)