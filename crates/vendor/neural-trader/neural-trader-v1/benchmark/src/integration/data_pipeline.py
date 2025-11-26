"""
Data Pipeline - End-to-end data flow orchestration for the benchmark system.

This module orchestrates data flow from real-time sources through processing,
simulation, optimization, and reporting stages.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
from pathlib import Path

from ..config import Config
from ..data.realtime_manager import RealtimeManager
from ..data.data_aggregator import DataAggregator
from ..data.data_validator import DataValidator
from ..simulation.simulator import Simulator
from ..optimization.optimizer import Optimizer


class PipelineStage(Enum):
    """Pipeline processing stages."""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    AGGREGATION = "aggregation"
    PROCESSING = "processing"
    SIMULATION = "simulation"
    OPTIMIZATION = "optimization"
    REPORTING = "reporting"


class DataType(Enum):
    """Types of data flowing through the pipeline."""
    MARKET_DATA = "market_data"
    NEWS_DATA = "news_data"
    SENTIMENT_DATA = "sentiment_data"
    TRADE_SIGNAL = "trade_signal"
    PERFORMANCE_METRIC = "performance_metric"
    OPTIMIZATION_RESULT = "optimization_result"


@dataclass
class DataPacket:
    """Data packet flowing through the pipeline."""
    data_type: DataType
    timestamp: float
    source: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    pipeline_stage: PipelineStage = PipelineStage.INGESTION
    processing_time: float = 0.0
    error_count: int = 0
    retry_count: int = 0


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    packets_processed: int = 0
    packets_failed: int = 0
    total_processing_time: float = 0.0
    stage_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    throughput: float = 0.0
    latency_avg: float = 0.0
    latency_p95: float = 0.0
    error_rate: float = 0.0


class DataPipeline:
    """
    End-to-end data flow orchestrator for the benchmark system.
    
    Features:
    - Multi-stage data processing pipeline
    - Real-time data ingestion and validation
    - Automatic error handling and retry logic
    - Performance monitoring and metrics
    - Backpressure management
    - Data persistence and recovery
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logging()
        
        # Pipeline components
        self.realtime_manager: Optional[RealtimeManager] = None
        self.data_aggregator: Optional[DataAggregator] = None
        self.data_validator: Optional[DataValidator] = None
        self.simulator: Optional[Simulator] = None
        self.optimizer: Optional[Optimizer] = None
        
        # Pipeline state
        self.is_running = False
        self.processing_queue = asyncio.Queue(maxsize=1000)
        self.error_queue = asyncio.Queue(maxsize=100)
        self.metrics = PipelineMetrics()
        
        # Stage processors
        self.stage_processors: Dict[PipelineStage, List[Callable]] = {
            stage: [] for stage in PipelineStage
        }
        
        # Processing tasks
        self.processing_tasks: List[asyncio.Task] = []
        self.worker_count = 4
        
        # Backpressure management
        self.max_queue_size = 1000
        self.processing_rate_limit = 100  # packets per second
        self.rate_limiter = asyncio.Semaphore(self.processing_rate_limit)
        
        # Persistence
        self.persistence_enabled = True
        self.persistence_path = Path("pipeline_data")
        self.persistence_path.mkdir(exist_ok=True)
        
        # Latency tracking
        self.latency_samples = deque(maxlen=1000)
        
        self.logger.info("Data Pipeline initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup pipeline logging."""
        logger = logging.getLogger('data_pipeline')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def start(self) -> bool:
        """
        Start the data pipeline.
        
        Returns:
            bool: True if pipeline started successfully
        """
        if self.is_running:
            self.logger.warning("Pipeline already running")
            return True
        
        try:
            self.logger.info("Starting data pipeline...")
            
            # Initialize components
            await self._initialize_components()
            
            # Register stage processors
            self._register_stage_processors()
            
            # Start processing workers
            self.processing_tasks = [
                asyncio.create_task(self._processing_worker(i))
                for i in range(self.worker_count)
            ]
            
            # Start metrics collection
            asyncio.create_task(self._metrics_collector())
            
            # Start error handler
            asyncio.create_task(self._error_handler())
            
            self.is_running = True
            self.logger.info("Data pipeline started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> bool:
        """
        Stop the data pipeline.
        
        Returns:
            bool: True if pipeline stopped successfully
        """
        if not self.is_running:
            return True
        
        self.logger.info("Stopping data pipeline...")
        
        try:
            self.is_running = False
            
            # Cancel processing tasks
            for task in self.processing_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.processing_tasks:
                await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            
            # Process remaining items in queue
            await self._drain_queue()
            
            # Save metrics
            await self._save_metrics()
            
            self.logger.info("Data pipeline stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping pipeline: {e}")
            return False
    
    async def _initialize_components(self):
        """Initialize pipeline components."""
        self.realtime_manager = RealtimeManager(self.config)
        self.data_aggregator = DataAggregator(self.config)
        self.data_validator = DataValidator(self.config)
        
        # Initialize component connections
        if hasattr(self.realtime_manager, 'set_data_handler'):
            self.realtime_manager.set_data_handler(self._handle_realtime_data)
    
    def _register_stage_processors(self):
        """Register processors for each pipeline stage."""
        # Ingestion stage
        self.stage_processors[PipelineStage.INGESTION] = [
            self._process_ingestion
        ]
        
        # Validation stage
        self.stage_processors[PipelineStage.VALIDATION] = [
            self._process_validation
        ]
        
        # Aggregation stage
        self.stage_processors[PipelineStage.AGGREGATION] = [
            self._process_aggregation
        ]
        
        # Processing stage
        self.stage_processors[PipelineStage.PROCESSING] = [
            self._process_data_processing
        ]
        
        # Simulation stage
        self.stage_processors[PipelineStage.SIMULATION] = [
            self._process_simulation
        ]
        
        # Optimization stage
        self.stage_processors[PipelineStage.OPTIMIZATION] = [
            self._process_optimization
        ]
        
        # Reporting stage
        self.stage_processors[PipelineStage.REPORTING] = [
            self._process_reporting
        ]
    
    async def _handle_realtime_data(self, data: Dict[str, Any]):
        """Handle incoming real-time data."""
        packet = DataPacket(
            data_type=self._determine_data_type(data),
            timestamp=time.time(),
            source="realtime_manager",
            data=data
        )
        
        await self.enqueue_packet(packet)
    
    def _determine_data_type(self, data: Dict[str, Any]) -> DataType:
        """Determine the type of incoming data."""
        if "price" in data or "volume" in data:
            return DataType.MARKET_DATA
        elif "news" in data or "headline" in data:
            return DataType.NEWS_DATA
        elif "sentiment" in data:
            return DataType.SENTIMENT_DATA
        elif "signal" in data:
            return DataType.TRADE_SIGNAL
        else:
            return DataType.MARKET_DATA  # Default
    
    async def enqueue_packet(self, packet: DataPacket) -> bool:
        """
        Enqueue a data packet for processing.
        
        Args:
            packet: Data packet to process
        
        Returns:
            bool: True if packet was enqueued successfully
        """
        try:
            # Check backpressure
            if self.processing_queue.qsize() >= self.max_queue_size:
                self.logger.warning("Pipeline queue full, dropping packet")
                self.metrics.packets_failed += 1
                return False
            
            # Apply rate limiting
            async with self.rate_limiter:
                await self.processing_queue.put(packet)
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to enqueue packet: {e}")
            return False
    
    async def _processing_worker(self, worker_id: int):
        """Processing worker task."""
        self.logger.info(f"Processing worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get packet from queue with timeout
                packet = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=1.0
                )
                
                # Process packet through pipeline stages
                await self._process_packet(packet)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                continue  # No packet available, continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                # Add to error queue for retry
                if packet:
                    await self.error_queue.put(packet)
        
        self.logger.info(f"Processing worker {worker_id} stopped")
    
    async def _process_packet(self, packet: DataPacket):
        """Process a data packet through all pipeline stages."""
        start_time = time.time()
        
        try:
            # Process through each stage
            for stage in PipelineStage:
                if stage.value in self.stage_processors:
                    packet.pipeline_stage = stage
                    
                    # Process with all registered processors for this stage
                    for processor in self.stage_processors[stage]:
                        try:
                            await processor(packet)
                        except Exception as e:
                            self.logger.error(f"Processor error in {stage.value}: {e}")
                            packet.error_count += 1
                            
                            if packet.error_count >= 3:
                                raise  # Fail after 3 errors
            
            # Calculate processing time
            processing_time = time.time() - start_time
            packet.processing_time = processing_time
            
            # Update metrics
            self.metrics.packets_processed += 1
            self.metrics.total_processing_time += processing_time
            self.latency_samples.append(processing_time)
            
            # Persist if enabled
            if self.persistence_enabled:
                await self._persist_packet(packet)
            
        except Exception as e:
            self.logger.error(f"Packet processing failed: {e}")
            self.metrics.packets_failed += 1
            packet.error_count += 1
            
            # Retry logic
            if packet.retry_count < 3:
                packet.retry_count += 1
                await self.error_queue.put(packet)
    
    async def _process_ingestion(self, packet: DataPacket):
        """Process ingestion stage."""
        # Add ingestion metadata
        packet.metadata['ingestion_time'] = time.time()
        packet.metadata['source_validated'] = True
    
    async def _process_validation(self, packet: DataPacket):
        """Process validation stage."""
        if self.data_validator:
            is_valid = await self.data_validator.validate(packet.data)
            packet.metadata['validation_passed'] = is_valid
            
            if not is_valid:
                raise ValueError("Data validation failed")
    
    async def _process_aggregation(self, packet: DataPacket):
        """Process aggregation stage."""
        if self.data_aggregator:
            aggregated_data = await self.data_aggregator.aggregate(packet.data)
            packet.data.update(aggregated_data)
            packet.metadata['aggregated'] = True
    
    async def _process_data_processing(self, packet: DataPacket):
        """Process data processing stage."""
        # Apply data transformations
        if packet.data_type == DataType.MARKET_DATA:
            await self._process_market_data(packet)
        elif packet.data_type == DataType.NEWS_DATA:
            await self._process_news_data(packet)
        elif packet.data_type == DataType.SENTIMENT_DATA:
            await self._process_sentiment_data(packet)
    
    async def _process_market_data(self, packet: DataPacket):
        """Process market data specifically."""
        # Add technical indicators
        if 'price' in packet.data:
            packet.data['price_change'] = packet.data.get('price_change', 0)
            packet.data['volatility'] = packet.data.get('volatility', 0)
    
    async def _process_news_data(self, packet: DataPacket):
        """Process news data specifically."""
        # Add sentiment analysis
        if 'headline' in packet.data:
            # Simulate sentiment analysis
            packet.data['sentiment_score'] = 0.5  # Placeholder
    
    async def _process_sentiment_data(self, packet: DataPacket):
        """Process sentiment data specifically."""
        # Normalize sentiment scores
        if 'sentiment' in packet.data:
            packet.data['normalized_sentiment'] = max(-1, min(1, packet.data['sentiment']))
    
    async def _process_simulation(self, packet: DataPacket):
        """Process simulation stage."""
        if self.simulator and packet.data_type == DataType.TRADE_SIGNAL:
            # Run simulation with the signal
            simulation_result = await self.simulator.simulate_signal(packet.data)
            packet.data['simulation_result'] = simulation_result
    
    async def _process_optimization(self, packet: DataPacket):
        """Process optimization stage."""
        if self.optimizer and packet.data_type == DataType.PERFORMANCE_METRIC:
            # Apply optimization
            optimization_result = await self.optimizer.optimize(packet.data)
            packet.data['optimization_result'] = optimization_result
    
    async def _process_reporting(self, packet: DataPacket):
        """Process reporting stage."""
        # Generate reports for processed data
        report_data = {
            'timestamp': packet.timestamp,
            'data_type': packet.data_type.value,
            'processing_time': packet.processing_time,
            'metadata': packet.metadata,
            'data_summary': self._generate_data_summary(packet.data)
        }
        
        # Save report
        await self._save_report(report_data)
    
    def _generate_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the data for reporting."""
        return {
            'keys': list(data.keys()),
            'size': len(str(data)),
            'has_price': 'price' in data,
            'has_volume': 'volume' in data,
            'has_sentiment': 'sentiment' in data or 'sentiment_score' in data
        }
    
    async def _save_report(self, report_data: Dict[str, Any]):
        """Save report data to persistent storage."""
        if self.persistence_enabled:
            report_file = self.persistence_path / f"report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
    
    async def _persist_packet(self, packet: DataPacket):
        """Persist processed packet data."""
        if self.persistence_enabled:
            packet_file = self.persistence_path / f"packet_{int(packet.timestamp)}.json"
            packet_data = {
                'data_type': packet.data_type.value,
                'timestamp': packet.timestamp,
                'source': packet.source,
                'data': packet.data,
                'metadata': packet.metadata,
                'processing_time': packet.processing_time
            }
            
            with open(packet_file, 'w') as f:
                json.dump(packet_data, f, indent=2)
    
    async def _error_handler(self):
        """Handle errors and retry failed packets."""
        while self.is_running:
            try:
                # Get error packet with timeout
                packet = await asyncio.wait_for(
                    self.error_queue.get(),
                    timeout=5.0
                )
                
                # Retry processing after delay
                await asyncio.sleep(1.0)
                await self.enqueue_packet(packet)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error handler error: {e}")
    
    async def _metrics_collector(self):
        """Collect and update pipeline metrics."""
        while self.is_running:
            try:
                # Calculate throughput
                if self.metrics.total_processing_time > 0:
                    self.metrics.throughput = (
                        self.metrics.packets_processed / self.metrics.total_processing_time
                    )
                
                # Calculate latency metrics
                if self.latency_samples:
                    sorted_latencies = sorted(self.latency_samples)
                    self.metrics.latency_avg = sum(sorted_latencies) / len(sorted_latencies)
                    p95_index = int(len(sorted_latencies) * 0.95)
                    self.metrics.latency_p95 = sorted_latencies[p95_index]
                
                # Calculate error rate
                total_packets = self.metrics.packets_processed + self.metrics.packets_failed
                if total_packets > 0:
                    self.metrics.error_rate = self.metrics.packets_failed / total_packets
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
    
    async def _drain_queue(self):
        """Drain remaining items from processing queue."""
        self.logger.info("Draining processing queue...")
        
        while not self.processing_queue.empty():
            try:
                packet = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=1.0
                )
                await self._process_packet(packet)
                self.processing_queue.task_done()
            except asyncio.TimeoutError:
                break
            except Exception as e:
                self.logger.error(f"Error draining queue: {e}")
    
    async def _save_metrics(self):
        """Save pipeline metrics to file."""
        if self.persistence_enabled:
            metrics_file = self.persistence_path / "pipeline_metrics.json"
            metrics_data = {
                'packets_processed': self.metrics.packets_processed,
                'packets_failed': self.metrics.packets_failed,
                'total_processing_time': self.metrics.total_processing_time,
                'throughput': self.metrics.throughput,
                'latency_avg': self.metrics.latency_avg,
                'latency_p95': self.metrics.latency_p95,
                'error_rate': self.metrics.error_rate,
                'timestamp': time.time()
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
    
    async def reduce_processing_rate(self, factor: float):
        """Reduce processing rate by a factor (for backpressure management)."""
        new_rate = max(1, int(self.processing_rate_limit * factor))
        self.processing_rate_limit = new_rate
        self.rate_limiter = asyncio.Semaphore(new_rate)
        self.logger.info(f"Reduced processing rate to {new_rate} packets/sec")
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status information."""
        return {
            'running': self.is_running,
            'queue_size': self.processing_queue.qsize(),
            'error_queue_size': self.error_queue.qsize(),
            'worker_count': len(self.processing_tasks),
            'processing_rate_limit': self.processing_rate_limit,
            'metrics': {
                'packets_processed': self.metrics.packets_processed,
                'packets_failed': self.metrics.packets_failed,
                'throughput': self.metrics.throughput,
                'latency_avg': self.metrics.latency_avg,
                'latency_p95': self.metrics.latency_p95,
                'error_rate': self.metrics.error_rate
            }
        }
    
    def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        return self.metrics