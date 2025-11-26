"""
WebSocket stream health monitoring for trading systems.
Tracks connection status, message rates, and data quality.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import time
import logging
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ConnectionEvent:
    """Represents a connection state change event."""
    timestamp: datetime
    event_type: str  # 'connected', 'disconnected', 'error', 'reconnected'
    duration_ms: Optional[float] = None  # For disconnection duration
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageStats:
    """Statistics for a specific message type."""
    message_type: str
    count: int = 0
    rate_per_second: float = 0.0
    avg_size_bytes: float = 0.0
    last_received: Optional[datetime] = None
    errors: int = 0
    latency_ms: float = 0.0  # From timestamp to processing


class StreamHealthMonitor:
    """
    Monitor WebSocket stream health and performance.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        rate_window_seconds: int = 60,
        health_check_interval: float = 5.0
    ):
        """
        Initialize stream health monitor.
        
        Args:
            window_size: Number of events to keep in memory
            rate_window_seconds: Window for rate calculations
            health_check_interval: Interval for health checks
        """
        self.window_size = window_size
        self.rate_window_seconds = rate_window_seconds
        self.health_check_interval = health_check_interval
        
        # Connection tracking
        self.connection_events = deque(maxlen=window_size)
        self.current_status = 'disconnected'
        self.connection_start_time: Optional[datetime] = None
        self.last_disconnect_time: Optional[datetime] = None
        self.total_disconnections = 0
        self.total_reconnections = 0
        
        # Message tracking
        self.message_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.message_stats: Dict[str, MessageStats] = {}
        self.total_messages = 0
        self.message_errors = 0
        
        # Data quality tracking
        self.sequence_gaps: List[Dict[str, Any]] = []
        self.duplicate_messages = 0
        self.out_of_order_messages = 0
        self.last_sequence_numbers: Dict[str, int] = {}
        
        # Health metrics
        self.health_checks: deque = deque(maxlen=100)
        self._health_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
    
    async def start(self):
        """Start health monitoring."""
        if not self._health_task:
            self._health_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self):
        """Stop health monitoring."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None
    
    async def on_connected(self, metadata: Optional[Dict[str, Any]] = None):
        """Handle connection established."""
        async with self._lock:
            now = datetime.now()
            
            # Calculate disconnection duration if applicable
            duration_ms = None
            if self.last_disconnect_time:
                duration_ms = (now - self.last_disconnect_time).total_seconds() * 1000
                self.total_reconnections += 1
            
            event = ConnectionEvent(
                timestamp=now,
                event_type='reconnected' if duration_ms else 'connected',
                duration_ms=duration_ms,
                metadata=metadata or {}
            )
            
            self.connection_events.append(event)
            self.current_status = 'connected'
            self.connection_start_time = now
            self.last_disconnect_time = None
            
            logger.info(f"WebSocket connected. Duration since last disconnect: {duration_ms:.1f}ms" 
                       if duration_ms else "WebSocket connected")
    
    async def on_disconnected(self, error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Handle connection lost."""
        async with self._lock:
            now = datetime.now()
            
            event = ConnectionEvent(
                timestamp=now,
                event_type='disconnected',
                error_message=error,
                metadata=metadata or {}
            )
            
            self.connection_events.append(event)
            self.current_status = 'disconnected'
            self.last_disconnect_time = now
            self.total_disconnections += 1
            
            # Trigger alerts
            await self._trigger_alert('disconnection', {
                'error': error,
                'total_disconnections': self.total_disconnections
            })
            
            logger.warning(f"WebSocket disconnected. Error: {error}")
    
    async def on_error(self, error: str, metadata: Optional[Dict[str, Any]] = None):
        """Handle connection error."""
        async with self._lock:
            event = ConnectionEvent(
                timestamp=datetime.now(),
                event_type='error',
                error_message=error,
                metadata=metadata or {}
            )
            
            self.connection_events.append(event)
            self.message_errors += 1
            
            logger.error(f"WebSocket error: {error}")
    
    async def on_message(
        self,
        message_type: str,
        size_bytes: int,
        timestamp: Optional[datetime] = None,
        sequence_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track received message.
        
        Args:
            message_type: Type of message (e.g., 'trade', 'quote')
            size_bytes: Message size in bytes
            timestamp: Message timestamp (for latency calculation)
            sequence_number: Optional sequence number for gap detection
            metadata: Additional message metadata
        """
        async with self._lock:
            now = datetime.now()
            self.total_messages += 1
            
            # Initialize stats if needed
            if message_type not in self.message_stats:
                self.message_stats[message_type] = MessageStats(message_type=message_type)
            
            stats = self.message_stats[message_type]
            stats.count += 1
            stats.last_received = now
            
            # Update average size
            if stats.avg_size_bytes == 0:
                stats.avg_size_bytes = size_bytes
            else:
                stats.avg_size_bytes = (stats.avg_size_bytes * 0.9) + (size_bytes * 0.1)
            
            # Calculate latency if timestamp provided
            if timestamp:
                latency_ms = (now - timestamp).total_seconds() * 1000
                stats.latency_ms = (stats.latency_ms * 0.9) + (latency_ms * 0.1)
            
            # Track for rate calculation
            self.message_timestamps[message_type].append(now)
            
            # Check sequence if provided
            if sequence_number is not None:
                await self._check_sequence(message_type, sequence_number)
    
    async def _check_sequence(self, message_type: str, sequence_number: int):
        """Check for sequence gaps or out-of-order messages."""
        last_seq = self.last_sequence_numbers.get(message_type)
        
        if last_seq is not None:
            expected = last_seq + 1
            
            if sequence_number < expected:
                self.out_of_order_messages += 1
                logger.warning(f"Out-of-order message: {message_type} seq {sequence_number}, expected >= {expected}")
            elif sequence_number > expected:
                gap_size = sequence_number - expected
                self.sequence_gaps.append({
                    'timestamp': datetime.now(),
                    'message_type': message_type,
                    'gap_start': expected,
                    'gap_end': sequence_number - 1,
                    'gap_size': gap_size
                })
                logger.warning(f"Sequence gap detected: {message_type} missing {gap_size} messages")
            elif sequence_number == last_seq:
                self.duplicate_messages += 1
                logger.warning(f"Duplicate message: {message_type} seq {sequence_number}")
        
        self.last_sequence_numbers[message_type] = sequence_number
    
    def calculate_message_rates(self) -> Dict[str, float]:
        """Calculate current message rates per second."""
        rates = {}
        now = datetime.now()
        window_start = now - timedelta(seconds=self.rate_window_seconds)
        
        for msg_type, timestamps in self.message_timestamps.items():
            # Count messages in window
            recent_messages = sum(1 for ts in timestamps if ts > window_start)
            rate = recent_messages / self.rate_window_seconds
            rates[msg_type] = rate
            
            # Update stats
            if msg_type in self.message_stats:
                self.message_stats[msg_type].rate_per_second = rate
        
        return rates
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        uptime_seconds = 0
        if self.current_status == 'connected' and self.connection_start_time:
            uptime_seconds = (datetime.now() - self.connection_start_time).total_seconds()
        
        # Calculate average reconnection time
        reconnection_times = []
        for event in self.connection_events:
            if event.event_type == 'reconnected' and event.duration_ms:
                reconnection_times.append(event.duration_ms)
        
        avg_reconnect_time = statistics.mean(reconnection_times) if reconnection_times else 0
        
        return {
            'current_status': self.current_status,
            'uptime_seconds': uptime_seconds,
            'total_disconnections': self.total_disconnections,
            'total_reconnections': self.total_reconnections,
            'avg_reconnect_time_ms': avg_reconnect_time,
            'last_disconnect': self.last_disconnect_time.isoformat() if self.last_disconnect_time else None
        }
    
    def get_message_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get message statistics."""
        rates = self.calculate_message_rates()
        
        stats = {}
        for msg_type, msg_stats in self.message_stats.items():
            stats[msg_type] = {
                'count': msg_stats.count,
                'rate_per_second': rates.get(msg_type, 0.0),
                'avg_size_bytes': msg_stats.avg_size_bytes,
                'avg_latency_ms': msg_stats.latency_ms,
                'last_received': msg_stats.last_received.isoformat() if msg_stats.last_received else None,
                'errors': msg_stats.errors
            }
        
        return stats
    
    def get_data_quality_stats(self) -> Dict[str, Any]:
        """Get data quality statistics."""
        return {
            'total_messages': self.total_messages,
            'message_errors': self.message_errors,
            'error_rate': (self.message_errors / self.total_messages * 100) if self.total_messages > 0 else 0,
            'sequence_gaps': len(self.sequence_gaps),
            'duplicate_messages': self.duplicate_messages,
            'out_of_order_messages': self.out_of_order_messages,
            'recent_gaps': self.sequence_gaps[-10:]  # Last 10 gaps
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        async with self._lock:
            connection_stats = self.get_connection_stats()
            message_stats = self.get_message_stats()
            quality_stats = self.get_data_quality_stats()
            
            # Determine health score (0-100)
            health_score = 100
            
            # Deduct for disconnections
            if connection_stats['current_status'] != 'connected':
                health_score -= 50
            
            # Deduct for high error rate
            error_rate = quality_stats['error_rate']
            if error_rate > 5:
                health_score -= min(30, error_rate * 2)
            
            # Deduct for low message rate
            total_rate = sum(stats['rate_per_second'] for stats in message_stats.values())
            if total_rate < 1 and connection_stats['current_status'] == 'connected':
                health_score -= 20
            
            # Deduct for sequence gaps
            if quality_stats['sequence_gaps'] > 10:
                health_score -= 10
            
            health_status = 'healthy' if health_score >= 80 else 'degraded' if health_score >= 50 else 'unhealthy'
            
            return {
                'timestamp': datetime.now().isoformat(),
                'status': health_status,
                'health_score': health_score,
                'connection': connection_stats,
                'messages': message_stats,
                'data_quality': quality_stats
            }
    
    async def _health_check_loop(self):
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                health_status = await self.get_health_status()
                self.health_checks.append(health_status)
                
                # Check for alerts
                if health_status['status'] == 'unhealthy':
                    await self._trigger_alert('unhealthy', health_status)
                
                # Check for stale data
                for msg_type, stats in self.message_stats.items():
                    if stats.last_received:
                        time_since_last = (datetime.now() - stats.last_received).total_seconds()
                        if time_since_last > 30:  # 30 seconds stale
                            await self._trigger_alert('stale_data', {
                                'message_type': msg_type,
                                'seconds_stale': time_since_last
                            })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_type, data)
                else:
                    callback(alert_type, data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics."""
        return {
            'connection_events': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'event_type': e.event_type,
                    'duration_ms': e.duration_ms,
                    'error': e.error_message
                } for e in self.connection_events
            ],
            'health_checks': list(self.health_checks),
            'current_health': asyncio.run(self.get_health_status())
        }