"""Test suite for real-time data feed integration.

Tests cover WebSocket/REST integration, latency requirements,
throughput, failover, and circuit breaker functionality.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict
import aiohttp
import websockets

# Module to be implemented
from benchmark.src.data.realtime_feed import (
    RealtimeFeed,
    DataSource,
    FeedConfig,
    ConnectionState,
    DataUpdate,
    BackpressureHandler,
    CircuitBreaker,
)


class TestRealtimeFeed:
    """Test real-time data feed functionality."""
    
    @pytest.fixture
    async def feed_config(self):
        """Create test feed configuration."""
        return FeedConfig(
            websocket_url="wss://test-feed.example.com/stream",
            rest_url="https://test-feed.example.com/api",
            max_latency_ms=10,
            max_updates_per_second=10000,
            enable_failover=True,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=30,
            backpressure_buffer_size=50000,
        )
    
    @pytest.fixture
    async def realtime_feed(self, feed_config):
        """Create realtime feed instance."""
        feed = RealtimeFeed(feed_config)
        yield feed
        await feed.close()
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, realtime_feed):
        """Test WebSocket connection establishment and management."""
        with patch('websockets.connect') as mock_connect:
            mock_ws = AsyncMock()
            mock_ws.recv = AsyncMock(return_value='{"symbol": "AAPL", "price": 150.25}')
            mock_connect.return_value.__aenter__.return_value = mock_ws
            
            await realtime_feed.connect()
            
            assert realtime_feed.connection_state == ConnectionState.CONNECTED
            assert realtime_feed.is_websocket_active
            mock_connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rest_fallback(self, realtime_feed):
        """Test automatic fallback to REST API when WebSocket fails."""
        # Simulate WebSocket failure
        with patch('websockets.connect', side_effect=Exception("Connection failed")):
            with patch.object(realtime_feed, '_connect_rest') as mock_rest:
                await realtime_feed.connect()
                
                assert realtime_feed.connection_state == ConnectionState.CONNECTED
                assert not realtime_feed.is_websocket_active
                mock_rest.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_latency_requirement(self, realtime_feed):
        """Test that data updates meet <10ms latency requirement."""
        test_data = {"symbol": "AAPL", "price": 150.25, "timestamp": time.time()}
        
        with patch.object(realtime_feed, '_process_update') as mock_process:
            start_time = time.perf_counter()
            await realtime_feed._handle_websocket_message(test_data)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            assert latency_ms < 10, f"Latency {latency_ms}ms exceeds 10ms requirement"
    
    @pytest.mark.asyncio
    async def test_high_throughput(self, realtime_feed):
        """Test handling 10,000+ updates per second."""
        num_updates = 10000
        updates = []
        
        async def generate_updates():
            for i in range(num_updates):
                update = DataUpdate(
                    symbol=f"SYMBOL{i % 100}",
                    price=100 + (i % 50),
                    timestamp=time.time(),
                    source=DataSource.WEBSOCKET
                )
                updates.append(update)
                await realtime_feed.handle_update(update)
        
        start_time = time.perf_counter()
        await generate_updates()
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        updates_per_second = num_updates / duration
        
        assert updates_per_second > 10000, f"Only processed {updates_per_second:.0f} updates/sec"
        assert len(realtime_feed.get_buffered_updates()) > 0
    
    @pytest.mark.asyncio
    async def test_automatic_failover(self, realtime_feed):
        """Test automatic failover between data sources."""
        # Start with WebSocket
        await realtime_feed.connect()
        assert realtime_feed.is_websocket_active
        
        # Simulate WebSocket failure
        with patch.object(realtime_feed, '_websocket_healthy', return_value=False):
            await realtime_feed._check_connection_health()
            
            # Should failover to REST
            assert not realtime_feed.is_websocket_active
            assert realtime_feed.connection_state == ConnectionState.CONNECTED
    
    @pytest.mark.asyncio
    async def test_data_deduplication(self, realtime_feed):
        """Test that duplicate updates are filtered out."""
        update1 = DataUpdate("AAPL", 150.25, time.time(), DataSource.WEBSOCKET)
        update2 = DataUpdate("AAPL", 150.25, time.time(), DataSource.WEBSOCKET)
        
        await realtime_feed.handle_update(update1)
        await realtime_feed.handle_update(update2)
        
        processed_updates = realtime_feed.get_processed_updates()
        assert len(processed_updates) == 1
    
    @pytest.mark.asyncio
    async def test_backpressure_handling(self, realtime_feed):
        """Test backpressure handling when buffer fills up."""
        # Fill buffer beyond capacity
        for i in range(60000):  # More than buffer size
            update = DataUpdate(f"SYM{i}", 100.0, time.time(), DataSource.WEBSOCKET)
            await realtime_feed.handle_update(update)
        
        # Check backpressure was applied
        assert realtime_feed.backpressure_handler.is_active
        assert realtime_feed.get_dropped_count() > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, realtime_feed):
        """Test circuit breaker functionality."""
        circuit_breaker = realtime_feed.circuit_breaker
        
        # Simulate failures
        for _ in range(6):  # More than threshold
            await circuit_breaker.record_failure()
        
        assert circuit_breaker.is_open
        assert not await circuit_breaker.can_execute()
        
        # Test half-open state after timeout
        await asyncio.sleep(0.1)  # Simulate timeout
        with patch.object(circuit_breaker, '_time_since_open', return_value=31):
            assert circuit_breaker.is_half_open
            assert await circuit_breaker.can_execute()


class TestDataSource:
    """Test data source abstraction."""
    
    @pytest.mark.asyncio
    async def test_websocket_source(self):
        """Test WebSocket data source."""
        source = DataSource.WEBSOCKET
        assert source.is_realtime
        assert source.priority == 1
    
    @pytest.mark.asyncio
    async def test_rest_source(self):
        """Test REST API data source."""
        source = DataSource.REST
        assert not source.is_realtime
        assert source.priority == 2


class TestBackpressureHandler:
    """Test backpressure handling mechanisms."""
    
    @pytest.fixture
    def handler(self):
        """Create backpressure handler."""
        return BackpressureHandler(
            buffer_size=1000,
            high_watermark=0.8,
            low_watermark=0.6
        )
    
    def test_activation_threshold(self, handler):
        """Test backpressure activation at high watermark."""
        # Fill to 79% - should not activate
        for i in range(790):
            handler.add_item(f"item_{i}")
        assert not handler.is_active
        
        # Fill to 81% - should activate
        for i in range(20):
            handler.add_item(f"item_{i}")
        assert handler.is_active
    
    def test_deactivation_threshold(self, handler):
        """Test backpressure deactivation at low watermark."""
        # Fill and activate
        for i in range(810):
            handler.add_item(f"item_{i}")
        assert handler.is_active
        
        # Drain to 61% - should still be active
        for _ in range(200):
            handler.remove_item()
        assert handler.is_active
        
        # Drain to 59% - should deactivate
        for _ in range(20):
            handler.remove_item()
        assert not handler.is_active
    
    def test_drop_policy(self, handler):
        """Test item dropping when buffer is full."""
        # Fill buffer completely
        for i in range(1100):
            handler.add_item(f"item_{i}")
        
        assert handler.buffer_size == 1000
        assert handler.dropped_count == 100


class TestCircuitBreaker:
    """Test circuit breaker pattern implementation."""
    
    @pytest.fixture
    def breaker(self):
        """Create circuit breaker."""
        return CircuitBreaker(
            failure_threshold=3,
            timeout_seconds=10,
            half_open_max_calls=1
        )
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, breaker):
        """Test circuit breaker state transitions."""
        # Initial state: CLOSED
        assert breaker.state == "CLOSED"
        assert await breaker.can_execute()
        
        # Record failures to trigger OPEN state
        for _ in range(3):
            await breaker.record_failure()
        
        assert breaker.state == "OPEN"
        assert not await breaker.can_execute()
        
        # Simulate timeout for HALF_OPEN state
        with patch.object(breaker, '_time_since_open', return_value=11):
            assert breaker.state == "HALF_OPEN"
            assert await breaker.can_execute()
            
            # Success should close circuit
            await breaker.record_success()
            assert breaker.state == "CLOSED"
    
    @pytest.mark.asyncio
    async def test_half_open_failure(self, breaker):
        """Test circuit reopening on failure in half-open state."""
        # Get to OPEN state
        for _ in range(3):
            await breaker.record_failure()
        
        # Move to HALF_OPEN
        with patch.object(breaker, '_time_since_open', return_value=11):
            assert breaker.state == "HALF_OPEN"
            
            # Failure should reopen circuit
            await breaker.record_failure()
            assert breaker.state == "OPEN"


class TestIntegration:
    """Integration tests for complete data feed system."""
    
    @pytest.mark.asyncio
    async def test_multi_source_aggregation(self):
        """Test aggregating data from multiple sources."""
        config = FeedConfig(
            sources=[
                {"type": "websocket", "url": "wss://source1.com"},
                {"type": "rest", "url": "https://source2.com"},
                {"type": "websocket", "url": "wss://source3.com"}
            ],
            aggregation_strategy="merge_dedupe"
        )
        
        feed = RealtimeFeed(config)
        await feed.connect_all_sources()
        
        # Simulate updates from different sources
        updates = [
            DataUpdate("AAPL", 150.25, time.time(), "source1"),
            DataUpdate("AAPL", 150.30, time.time() + 0.001, "source2"),
            DataUpdate("GOOGL", 2800.50, time.time(), "source3"),
        ]
        
        for update in updates:
            await feed.handle_update(update)
        
        aggregated = await feed.get_aggregated_data()
        assert len(aggregated) == 2  # AAPL and GOOGL
        assert aggregated["AAPL"].price == 150.30  # Latest price
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance metrics collection."""
        feed = RealtimeFeed(FeedConfig())
        
        # Process some updates
        for i in range(1000):
            update = DataUpdate(f"SYM{i}", 100.0, time.time(), DataSource.WEBSOCKET)
            await feed.handle_update(update)
        
        metrics = feed.get_performance_metrics()
        assert "avg_latency_ms" in metrics
        assert "updates_per_second" in metrics
        assert "buffer_utilization" in metrics
        assert metrics["avg_latency_ms"] < 10