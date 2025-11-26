"""Message Handler for Alpaca WebSocket messages."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MessageMetrics:
    """Metrics for message processing."""
    total_messages: int = 0
    messages_per_second: float = 0.0
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    errors: int = 0
    last_update: float = field(default_factory=time.time)


@dataclass
class TradeMessage:
    """Parsed trade message."""
    symbol: str
    price: float
    size: int
    timestamp: datetime
    conditions: List[str]
    exchange: str
    tape: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeMessage':
        """Create from Alpaca message dict."""
        return cls(
            symbol=data['S'],
            price=float(data['p']),
            size=int(data['s']),
            timestamp=datetime.fromisoformat(data['t'].replace('Z', '+00:00')),
            conditions=data.get('c', []),
            exchange=data.get('x', ''),
            tape=data.get('z', '')
        )


@dataclass
class QuoteMessage:
    """Parsed quote message."""
    symbol: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    bid_exchange: str
    ask_exchange: str
    timestamp: datetime
    conditions: List[str]
    tape: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuoteMessage':
        """Create from Alpaca message dict."""
        return cls(
            symbol=data['S'],
            bid_price=float(data['bp']),
            bid_size=int(data['bs']),
            ask_price=float(data['ap']),
            ask_size=int(data['as']),
            bid_exchange=data.get('bx', ''),
            ask_exchange=data.get('ax', ''),
            timestamp=datetime.fromisoformat(data['t'].replace('Z', '+00:00')),
            conditions=data.get('c', []),
            tape=data.get('z', '')
        )


@dataclass
class BarMessage:
    """Parsed bar message."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    trade_count: int
    vwap: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BarMessage':
        """Create from Alpaca message dict."""
        return cls(
            symbol=data['S'],
            open=float(data['o']),
            high=float(data['h']),
            low=float(data['l']),
            close=float(data['c']),
            volume=int(data['v']),
            timestamp=datetime.fromisoformat(data['t'].replace('Z', '+00:00')),
            trade_count=int(data.get('n', 0)),
            vwap=float(data.get('vw', 0))
        )


class MessageHandler:
    """Async message processing pipeline for Alpaca WebSocket.
    
    Features:
    - Async message processing pipeline
    - Message type routing
    - Latency measurement
    - Error handling and recovery
    - Message buffering and batching
    - Performance metrics
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        batch_size: int = 100,
        batch_timeout: float = 0.1,
        worker_count: int = 4
    ):
        """Initialize message handler.
        
        Args:
            buffer_size: Size of message buffer
            batch_size: Number of messages to process in a batch
            batch_timeout: Timeout for batch accumulation
            worker_count: Number of worker tasks for processing
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.worker_count = worker_count
        
        # Message queues by type
        self.message_queues: Dict[str, asyncio.Queue] = {
            "trades": asyncio.Queue(maxsize=buffer_size),
            "quotes": asyncio.Queue(maxsize=buffer_size),
            "bars": asyncio.Queue(maxsize=buffer_size),
            "statuses": asyncio.Queue(maxsize=buffer_size),
            "lulds": asyncio.Queue(maxsize=buffer_size),
            "other": asyncio.Queue(maxsize=buffer_size)
        }
        
        # Message processors
        self.processors: Dict[str, List[Callable]] = defaultdict(list)
        
        # Metrics
        self.metrics: Dict[str, MessageMetrics] = {
            msg_type: MessageMetrics() for msg_type in self.message_queues
        }
        self.latency_buffer = deque(maxlen=1000)
        
        # Worker tasks
        self.workers: List[asyncio.Task] = []
        self.running = False
        
        # Error handling
        self.error_handler: Optional[Callable] = None
        self.max_retries = 3
        
    def start(self) -> None:
        """Start message processing workers."""
        if not self.running:
            self.running = True
            
            # Start workers for each message type
            for msg_type in self.message_queues:
                for i in range(self.worker_count):
                    worker = asyncio.create_task(
                        self._process_messages(msg_type),
                        name=f"worker-{msg_type}-{i}"
                    )
                    self.workers.append(worker)
            
            # Start metrics updater
            self.workers.append(
                asyncio.create_task(self._update_metrics(), name="metrics-updater")
            )
            
            logger.info(f"Started {len(self.workers)} message processing workers")
    
    def stop(self) -> None:
        """Stop message processing workers."""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        self.workers.clear()
        logger.info("Stopped message processing workers")
    
    async def handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming message from WebSocket.
        
        Args:
            message: Raw message from Alpaca
        """
        try:
            # Determine message type
            msg_type = message.get('T', message.get('msg', '')).lower()
            
            # Route to appropriate queue
            if msg_type in ['t', 'trade']:
                queue = self.message_queues['trades']
                queue_type = 'trades'
            elif msg_type in ['q', 'quote']:
                queue = self.message_queues['quotes']
                queue_type = 'quotes'
            elif msg_type in ['b', 'bar', 'd', 'dailybar']:
                queue = self.message_queues['bars']
                queue_type = 'bars'
            elif msg_type in ['s', 'status']:
                queue = self.message_queues['statuses']
                queue_type = 'statuses'
            elif msg_type in ['l', 'luld']:
                queue = self.message_queues['lulds']
                queue_type = 'lulds'
            else:
                queue = self.message_queues['other']
                queue_type = 'other'
            
            # Add timestamp for latency measurement
            message['_received_at'] = time.time()
            
            # Try to add to queue (non-blocking)
            try:
                queue.put_nowait(message)
                self.metrics[queue_type].total_messages += 1
            except asyncio.QueueFull:
                logger.warning(f"Queue full for {queue_type}, dropping message")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self.metrics['other'].errors += 1
    
    def register_processor(
        self,
        message_type: str,
        processor: Callable,
        priority: int = 0
    ) -> None:
        """Register a message processor.
        
        Args:
            message_type: Type of message to process
            processor: Async function to process messages
            priority: Processing priority (higher = earlier)
        """
        if message_type not in self.processors:
            self.processors[message_type] = []
        
        # Insert by priority
        self.processors[message_type].append((priority, processor))
        self.processors[message_type].sort(key=lambda x: x[0], reverse=True)
        
        logger.debug(f"Registered processor for {message_type} with priority {priority}")
    
    def set_error_handler(self, handler: Callable) -> None:
        """Set error handler for processing errors."""
        self.error_handler = handler
    
    async def _process_messages(self, queue_type: str) -> None:
        """Process messages from a specific queue."""
        queue = self.message_queues[queue_type]
        batch: List[Dict[str, Any]] = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                # Try to get a message with timeout
                try:
                    message = await asyncio.wait_for(
                        queue.get(),
                        timeout=self.batch_timeout
                    )
                    batch.append(message)
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if full or timeout reached
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and time.time() - last_batch_time >= self.batch_timeout)
                )
                
                if should_process and batch:
                    await self._process_batch(queue_type, batch)
                    batch = []
                    last_batch_time = time.time()
                    
            except Exception as e:
                logger.error(f"Error in message processor for {queue_type}: {e}")
                if self.error_handler:
                    await self.error_handler(e)
                
                # Clear batch on error
                batch = []
    
    async def _process_batch(
        self,
        queue_type: str,
        messages: List[Dict[str, Any]]
    ) -> None:
        """Process a batch of messages."""
        start_time = time.time()
        
        try:
            # Parse messages based on type
            parsed_messages = []
            for msg in messages:
                try:
                    parsed = self._parse_message(queue_type, msg)
                    if parsed:
                        parsed_messages.append(parsed)
                        
                        # Calculate latency
                        if '_received_at' in msg:
                            latency = (msg['_received_at'] - 
                                     parsed.timestamp.timestamp()) * 1000
                            self.latency_buffer.append(latency)
                            
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    self.metrics[queue_type].errors += 1
            
            # Process with registered processors
            if parsed_messages and queue_type in self.processors:
                for priority, processor in self.processors[queue_type]:
                    try:
                        await processor(parsed_messages)
                    except Exception as e:
                        logger.error(f"Processor error for {queue_type}: {e}")
                        self.metrics[queue_type].errors += 1
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            metrics = self.metrics[queue_type]
            
            metrics.avg_processing_time_ms = (
                (metrics.avg_processing_time_ms * 0.9) + (processing_time * 0.1)
            )
            metrics.max_processing_time_ms = max(
                metrics.max_processing_time_ms,
                processing_time
            )
            metrics.min_processing_time_ms = min(
                metrics.min_processing_time_ms,
                processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing batch for {queue_type}: {e}")
            self.metrics[queue_type].errors += 1
    
    def _parse_message(
        self,
        queue_type: str,
        message: Dict[str, Any]
    ) -> Optional[Any]:
        """Parse message based on type."""
        try:
            if queue_type == 'trades':
                return TradeMessage.from_dict(message)
            elif queue_type == 'quotes':
                return QuoteMessage.from_dict(message)
            elif queue_type == 'bars':
                return BarMessage.from_dict(message)
            else:
                # Return raw message for other types
                return message
                
        except Exception as e:
            logger.error(f"Error parsing {queue_type} message: {e}")
            return None
    
    async def _update_metrics(self) -> None:
        """Update message processing metrics."""
        window_size = 5  # seconds
        message_counts = defaultdict(lambda: deque(maxlen=window_size))
        
        while self.running:
            try:
                await asyncio.sleep(1)
                
                current_time = time.time()
                
                # Update messages per second
                for queue_type, metrics in self.metrics.items():
                    # Add current count
                    message_counts[queue_type].append(metrics.total_messages)
                    
                    # Calculate rate if we have enough samples
                    if len(message_counts[queue_type]) >= 2:
                        oldest = message_counts[queue_type][0]
                        newest = message_counts[queue_type][-1]
                        time_diff = len(message_counts[queue_type]) - 1
                        
                        if time_diff > 0:
                            metrics.messages_per_second = (newest - oldest) / time_diff
                    
                    metrics.last_update = current_time
                    
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        total_messages = sum(m.total_messages for m in self.metrics.values())
        total_errors = sum(m.errors for m in self.metrics.values())
        
        # Calculate latency stats
        latency_stats = {}
        if self.latency_buffer:
            latencies = list(self.latency_buffer)
            latency_stats = {
                "avg_ms": statistics.mean(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "p50_ms": statistics.median(latencies),
                "p95_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                "p99_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
            }
        
        return {
            "total_messages": total_messages,
            "total_errors": total_errors,
            "error_rate": total_errors / total_messages if total_messages > 0 else 0,
            "latency": latency_stats,
            "by_type": {
                queue_type: {
                    "total": metrics.total_messages,
                    "rate": metrics.messages_per_second,
                    "avg_processing_ms": round(metrics.avg_processing_time_ms, 2),
                    "max_processing_ms": round(metrics.max_processing_time_ms, 2),
                    "errors": metrics.errors,
                    "queue_size": self.message_queues[queue_type].qsize()
                }
                for queue_type, metrics in self.metrics.items()
            }
        }
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes."""
        return {
            queue_type: queue.qsize()
            for queue_type, queue in self.message_queues.items()
        }