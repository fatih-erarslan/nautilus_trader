"""
CCXT Client Manager

Manages multiple exchange clients with connection pooling, failover,
and load balancing capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
from enum import Enum

from ..interfaces.ccxt_interface import CCXTInterface, ExchangeConfig

logger = logging.getLogger(__name__)


class ExchangeStatus(Enum):
    """Exchange connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


@dataclass
class ExchangeHealth:
    """Health status of an exchange"""
    exchange_name: str
    status: ExchangeStatus
    last_check: datetime
    response_time_ms: float
    error_count: int = 0
    success_count: int = 0
    last_error: Optional[str] = None


class ClientManager:
    """
    Manages multiple CCXT exchange clients with health monitoring,
    load balancing, and failover capabilities.
    """
    
    def __init__(self, max_clients_per_exchange: int = 3):
        """
        Initialize the client manager.
        
        Args:
            max_clients_per_exchange: Maximum concurrent clients per exchange
        """
        self.max_clients_per_exchange = max_clients_per_exchange
        self.clients: Dict[str, List[CCXTInterface]] = {}
        self.health_status: Dict[str, ExchangeHealth] = {}
        self.active_exchanges: Set[str] = set()
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
    async def add_exchange(self, config: ExchangeConfig) -> None:
        """
        Add a new exchange to the manager.
        
        Args:
            config: Exchange configuration
        """
        async with self._lock:
            exchange_name = config.name
            
            if exchange_name not in self.clients:
                self.clients[exchange_name] = []
                self.health_status[exchange_name] = ExchangeHealth(
                    exchange_name=exchange_name,
                    status=ExchangeStatus.DISCONNECTED,
                    last_check=datetime.now(),
                    response_time_ms=0
                )
                
            # Create and initialize client
            client = CCXTInterface(config)
            await client.initialize()
            
            self.clients[exchange_name].append(client)
            self.active_exchanges.add(exchange_name)
            
            # Update health status
            self.health_status[exchange_name].status = ExchangeStatus.CONNECTED
            self.health_status[exchange_name].last_check = datetime.now()
            
            logger.info(f"Added {exchange_name} exchange to manager")
            
    async def remove_exchange(self, exchange_name: str) -> None:
        """
        Remove an exchange from the manager.
        
        Args:
            exchange_name: Name of the exchange to remove
        """
        async with self._lock:
            if exchange_name in self.clients:
                # Close all clients for this exchange
                for client in self.clients[exchange_name]:
                    await client.close()
                    
                del self.clients[exchange_name]
                del self.health_status[exchange_name]
                self.active_exchanges.discard(exchange_name)
                
                logger.info(f"Removed {exchange_name} exchange from manager")
                
    async def get_client(self, exchange_name: str, prefer_healthy: bool = True) -> Optional[CCXTInterface]:
        """
        Get a client for the specified exchange.
        
        Args:
            exchange_name: Name of the exchange
            prefer_healthy: Prefer clients with good health status
            
        Returns:
            CCXTInterface instance or None if unavailable
        """
        if exchange_name not in self.clients:
            logger.warning(f"Exchange {exchange_name} not found in manager")
            return None
            
        clients = self.clients[exchange_name]
        if not clients:
            logger.warning(f"No clients available for {exchange_name}")
            return None
            
        if prefer_healthy and exchange_name in self.health_status:
            health = self.health_status[exchange_name]
            if health.status == ExchangeStatus.ERROR or health.status == ExchangeStatus.RATE_LIMITED:
                logger.warning(f"Exchange {exchange_name} is unhealthy: {health.status}")
                return None
                
        # Return random client for load balancing
        return random.choice(clients)
        
    async def get_all_clients(self, exchange_name: str) -> List[CCXTInterface]:
        """
        Get all clients for a specific exchange.
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            List of CCXTInterface instances
        """
        return self.clients.get(exchange_name, [])
        
    async def get_best_exchange(self, symbol: str) -> Optional[str]:
        """
        Get the best exchange for trading a specific symbol based on health and availability.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Exchange name or None if no suitable exchange found
        """
        best_exchange = None
        best_score = float('inf')
        
        for exchange_name in self.active_exchanges:
            client = await self.get_client(exchange_name)
            if not client:
                continue
                
            try:
                # Check if symbol is available
                markets = await client.get_markets()
                if symbol not in markets:
                    continue
                    
                # Calculate score based on health
                health = self.health_status.get(exchange_name)
                if health and health.status == ExchangeStatus.CONNECTED:
                    score = health.response_time_ms + (health.error_count * 100)
                    if score < best_score:
                        best_score = score
                        best_exchange = exchange_name
                        
            except Exception as e:
                logger.error(f"Error checking {exchange_name} for {symbol}: {str(e)}")
                continue
                
        return best_exchange
        
    async def execute_with_failover(
        self,
        method_name: str,
        *args,
        exchanges: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """
        Execute a method with automatic failover to backup exchanges.
        
        Args:
            method_name: Name of the method to execute
            exchanges: List of exchanges to try (uses all if None)
            *args, **kwargs: Arguments for the method
            
        Returns:
            Result from the first successful execution
            
        Raises:
            Exception if all exchanges fail
        """
        if exchanges is None:
            exchanges = list(self.active_exchanges)
            
        errors = []
        
        for exchange_name in exchanges:
            client = await self.get_client(exchange_name)
            if not client:
                continue
                
            try:
                method = getattr(client, method_name)
                result = await method(*args, **kwargs)
                
                # Update success count
                if exchange_name in self.health_status:
                    self.health_status[exchange_name].success_count += 1
                    
                return result
                
            except Exception as e:
                errors.append(f"{exchange_name}: {str(e)}")
                
                # Update error count
                if exchange_name in self.health_status:
                    self.health_status[exchange_name].error_count += 1
                    self.health_status[exchange_name].last_error = str(e)
                    
                continue
                
        # All exchanges failed
        error_msg = f"All exchanges failed for {method_name}: " + "; ".join(errors)
        logger.error(error_msg)
        raise Exception(error_msg)
        
    async def start_health_monitoring(self, interval_seconds: int = 60) -> None:
        """
        Start periodic health monitoring of all exchanges.
        
        Args:
            interval_seconds: Health check interval in seconds
        """
        if self._health_check_task and not self._health_check_task.done():
            logger.warning("Health monitoring already running")
            return
            
        async def health_check_loop():
            while True:
                await self._check_all_health()
                await asyncio.sleep(interval_seconds)
                
        self._health_check_task = asyncio.create_task(health_check_loop())
        logger.info(f"Started health monitoring with {interval_seconds}s interval")
        
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Stopped health monitoring")
            
    async def _check_all_health(self) -> None:
        """Check health of all exchanges."""
        for exchange_name in list(self.active_exchanges):
            await self._check_exchange_health(exchange_name)
            
    async def _check_exchange_health(self, exchange_name: str) -> None:
        """
        Check health of a specific exchange.
        
        Args:
            exchange_name: Name of the exchange to check
        """
        client = await self.get_client(exchange_name, prefer_healthy=False)
        if not client:
            return
            
        start_time = datetime.now()
        
        try:
            # Simple health check - fetch ticker for a major pair
            test_symbols = ['BTC/USDT', 'ETH/USDT', 'BTC/USD']
            
            for symbol in test_symbols:
                try:
                    await asyncio.wait_for(
                        client.get_ticker(symbol),
                        timeout=5.0
                    )
                    break
                except:
                    continue
                    
            # Calculate response time
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update health status
            self.health_status[exchange_name].status = ExchangeStatus.CONNECTED
            self.health_status[exchange_name].response_time_ms = response_time_ms
            self.health_status[exchange_name].last_check = datetime.now()
            
        except asyncio.TimeoutError:
            self.health_status[exchange_name].status = ExchangeStatus.ERROR
            self.health_status[exchange_name].last_error = "Health check timeout"
            logger.warning(f"Health check timeout for {exchange_name}")
            
        except Exception as e:
            self.health_status[exchange_name].status = ExchangeStatus.ERROR
            self.health_status[exchange_name].last_error = str(e)
            logger.error(f"Health check failed for {exchange_name}: {str(e)}")
            
    async def get_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive health report for all exchanges.
        
        Returns:
            Dictionary with health information
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_exchanges': len(self.active_exchanges),
            'healthy_exchanges': 0,
            'unhealthy_exchanges': 0,
            'exchanges': {}
        }
        
        for exchange_name, health in self.health_status.items():
            is_healthy = health.status == ExchangeStatus.CONNECTED
            
            if is_healthy:
                report['healthy_exchanges'] += 1
            else:
                report['unhealthy_exchanges'] += 1
                
            report['exchanges'][exchange_name] = {
                'status': health.status.value,
                'last_check': health.last_check.isoformat(),
                'response_time_ms': health.response_time_ms,
                'error_count': health.error_count,
                'success_count': health.success_count,
                'last_error': health.last_error,
                'uptime_percentage': (
                    health.success_count / (health.success_count + health.error_count) * 100
                    if (health.success_count + health.error_count) > 0 else 0
                )
            }
            
        return report
        
    async def close_all(self) -> None:
        """Close all exchange connections."""
        await self.stop_health_monitoring()
        
        for exchange_name in list(self.active_exchanges):
            await self.remove_exchange(exchange_name)
            
        logger.info("Closed all exchange connections")