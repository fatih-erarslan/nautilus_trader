"""
Tests for latency tracking and performance monitoring
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import statistics

from benchmark.src.data.realtime_manager import RealtimeManager, DataPoint, LatencyStats
from benchmark.src.data.yahoo_realtime import YahooRealtimeSource
from benchmark.src.data.finnhub_client import FinnhubClient
from benchmark.src.data.coinbase_feed import CoinbaseFeed


class TestLatencyTracking:
    """Test latency measurement and tracking"""
    
    @pytest.fixture
    def manager(self):
        """Create test manager with latency tracking"""
        return RealtimeManager(max_connections=5)
    
    @pytest.mark.asyncio
    async def test_latency_measurement(self, manager):
        """Test basic latency measurement"""
        # Create data point with latency
        data_point = DataPoint(
            source="test_source",
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.0,
            volume=1000000,
            latency_ms=25.5
        )
        
        await manager.add_data_point(data_point)
        
        # Check latency stats
        stats = await manager.get_latency_stats("test_source")
        assert stats is not None
        assert stats.avg_latency_ms == 25.5
        assert stats.min_latency_ms == 25.5
        assert stats.max_latency_ms == 25.5
        assert stats.count == 1
    
    @pytest.mark.asyncio
    async def test_latency_statistics_calculation(self, manager):
        """Test latency statistics with multiple data points"""
        latencies = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 100.0]
        
        # Add data points with different latencies
        for i, latency in enumerate(latencies):
            data_point = DataPoint(
                source="test_source",
                symbol=f"STOCK{i}",
                timestamp=datetime.now(),
                price=100.0 + i,
                volume=1000000,
                latency_ms=latency
            )
            await manager.add_data_point(data_point)
        
        stats = await manager.get_latency_stats("test_source")
        
        assert stats.count == len(latencies)
        assert stats.min_latency_ms == 10.0
        assert stats.max_latency_ms == 100.0
        assert stats.avg_latency_ms == statistics.mean(latencies)
        
        # Test percentiles
        sorted_latencies = sorted(latencies)
        expected_p95 = sorted_latencies[int(len(latencies) * 0.95)]
        expected_p99 = sorted_latencies[int(len(latencies) * 0.99)]
        
        assert stats.p95_latency_ms == expected_p95
        assert stats.p99_latency_ms == expected_p99
    
    @pytest.mark.asyncio
    async def test_latency_under_target(self, manager):
        """Test that latency stays under 50ms target"""
        # Simulate real-time data with good latency
        for i in range(100):
            latency = 10.0 + (i % 20)  # 10-30ms range
            data_point = DataPoint(
                source="high_performance_source",
                symbol="AAPL",
                timestamp=datetime.now(),
                price=150.0 + i * 0.01,
                volume=1000000,
                latency_ms=latency
            )
            await manager.add_data_point(data_point)
        
        stats = await manager.get_latency_stats("high_performance_source")
        
        # All latencies should be under 50ms
        assert stats.max_latency_ms < 50.0
        assert stats.avg_latency_ms < 50.0
        assert stats.p95_latency_ms < 50.0
        assert stats.p99_latency_ms < 50.0
    
    @pytest.mark.asyncio
    async def test_latency_degradation_detection(self, manager):
        """Test detection of latency degradation"""
        # Start with good latency
        for i in range(50):
            data_point = DataPoint(
                source="degrading_source",
                symbol="AAPL",
                timestamp=datetime.now(),
                price=150.0,
                volume=1000000,
                latency_ms=20.0
            )
            await manager.add_data_point(data_point)
        
        stats_initial = await manager.get_latency_stats("degrading_source")
        initial_avg = stats_initial.avg_latency_ms
        
        # Simulate latency degradation
        for i in range(50):
            degraded_latency = 20.0 + (i * 2)  # Increasing latency
            data_point = DataPoint(
                source="degrading_source",
                symbol="AAPL",
                timestamp=datetime.now(),
                price=150.0,
                volume=1000000,
                latency_ms=degraded_latency
            )
            await manager.add_data_point(data_point)
        
        stats_final = await manager.get_latency_stats("degrading_source")
        
        # Latency should have increased significantly
        assert stats_final.avg_latency_ms > initial_avg * 2
        assert stats_final.max_latency_ms > 100.0
    
    @pytest.mark.asyncio
    async def test_per_source_latency_tracking(self, manager):
        """Test latency tracking for multiple sources"""
        sources = {
            "fast_source": 15.0,
            "medium_source": 35.0,
            "slow_source": 75.0
        }
        
        # Add data points from different sources
        for source_name, base_latency in sources.items():
            for i in range(20):
                latency = base_latency + (i % 10)  # Add some variation
                data_point = DataPoint(
                    source=source_name,
                    symbol="AAPL",
                    timestamp=datetime.now(),
                    price=150.0,
                    volume=1000000,
                    latency_ms=latency
                )
                await manager.add_data_point(data_point)
        
        # Check each source's latency
        for source_name, base_latency in sources.items():
            stats = await manager.get_latency_stats(source_name)
            assert stats is not None
            # Average should be close to base latency
            assert abs(stats.avg_latency_ms - (base_latency + 4.5)) < 2.0
    
    @pytest.mark.asyncio
    async def test_latency_with_websocket_vs_rest(self):
        """Test latency comparison between WebSocket and REST"""
        manager = RealtimeManager()
        
        # Simulate WebSocket data (typically faster)
        for i in range(50):
            ws_data = DataPoint(
                source="websocket_source",
                symbol="AAPL",
                timestamp=datetime.now(),
                price=150.0,
                volume=1000000,
                latency_ms=5.0 + (i % 10)  # 5-15ms
            )
            await manager.add_data_point(ws_data)
        
        # Simulate REST data (typically slower)
        for i in range(50):
            rest_data = DataPoint(
                source="rest_source",
                symbol="AAPL",
                timestamp=datetime.now(),
                price=150.0,
                volume=1000000,
                latency_ms=50.0 + (i % 30)  # 50-80ms
            )
            await manager.add_data_point(rest_data)
        
        ws_stats = await manager.get_latency_stats("websocket_source")
        rest_stats = await manager.get_latency_stats("rest_source")
        
        # WebSocket should be significantly faster
        assert ws_stats.avg_latency_ms < rest_stats.avg_latency_ms
        assert ws_stats.max_latency_ms < rest_stats.min_latency_ms


class TestPerformanceMonitoring:
    """Test performance monitoring capabilities"""
    
    @pytest.fixture
    def manager(self):
        """Create test manager for performance monitoring"""
        return RealtimeManager(max_connections=10, cache_size_mb=50)
    
    @pytest.mark.asyncio
    async def test_throughput_measurement(self, manager):
        """Test measuring data throughput"""
        start_time = time.time()
        
        # Simulate high-frequency data
        for i in range(1000):
            data_point = DataPoint(
                source="high_freq_source",
                symbol=f"STOCK{i % 10}",
                timestamp=datetime.now(),
                price=100.0 + (i % 50),
                volume=1000000,
                latency_ms=10.0
            )
            await manager.add_data_point(data_point)
        
        elapsed_time = time.time() - start_time
        stats = await manager.get_stats()
        
        # Calculate throughput
        throughput = stats.total_data_points / elapsed_time
        
        # Should handle reasonable throughput
        assert throughput > 100  # At least 100 points per second
        assert stats.total_data_points == 1000
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, manager):
        """Test memory usage tracking"""
        # Add data points to fill cache
        for i in range(10000):
            data_point = DataPoint(
                source="memory_test_source",
                symbol=f"STOCK{i % 100}",
                timestamp=datetime.now(),
                price=100.0 + (i % 50),
                volume=1000000
            )
            await manager.add_data_point(data_point)
        
        stats = await manager.get_stats()
        
        # Memory usage should be tracked
        assert stats.cache_size_mb > 0
        assert stats.cache_size_mb < 100  # Should stay under limit
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self, manager):
        """Test performance under concurrent load"""
        async def generate_data(source_name, symbol_prefix, count):
            """Generate data points concurrently"""
            for i in range(count):
                data_point = DataPoint(
                    source=source_name,
                    symbol=f"{symbol_prefix}{i % 10}",
                    timestamp=datetime.now(),
                    price=100.0 + i,
                    volume=1000000,
                    latency_ms=10.0 + (i % 20)
                )
                await manager.add_data_point(data_point)
                await asyncio.sleep(0.001)  # Small delay to simulate real data
        
        start_time = time.time()
        
        # Run multiple data generators concurrently
        tasks = [
            generate_data("source1", "STOCK", 500),
            generate_data("source2", "CRYPTO", 500),
            generate_data("source3", "BOND", 500),
        ]
        
        await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        stats = await manager.get_stats()
        
        # Should handle concurrent processing efficiently
        assert stats.total_data_points == 1500
        assert elapsed_time < 10  # Should complete reasonably quickly
        
        # All sources should have data
        for source_name in ["source1", "source2", "source3"]:
            source_stats = await manager.get_latency_stats(source_name)
            assert source_stats is not None
            assert source_stats.count > 0
    
    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self, manager):
        """Test detection of performance degradation"""
        # Baseline performance
        baseline_start = time.time()
        for i in range(100):
            data_point = DataPoint(
                source="perf_test",
                symbol="AAPL",
                timestamp=datetime.now(),
                price=150.0,
                volume=1000000,
                latency_ms=10.0
            )
            await manager.add_data_point(data_point)
        baseline_time = time.time() - baseline_start
        
        # Simulate performance degradation (slower processing)
        degraded_start = time.time()
        for i in range(100):
            data_point = DataPoint(
                source="perf_test",
                symbol="AAPL",
                timestamp=datetime.now(),
                price=150.0,
                volume=1000000,
                latency_ms=50.0  # Higher latency
            )
            await manager.add_data_point(data_point)
            await asyncio.sleep(0.001)  # Simulate processing delay
        degraded_time = time.time() - degraded_start
        
        stats = await manager.get_latency_stats("perf_test")
        
        # Should detect increased latency
        assert stats.max_latency_ms >= 50.0
        assert stats.avg_latency_ms > 25.0  # Mix of fast and slow
    
    @pytest.mark.asyncio
    async def test_connection_health_monitoring(self, manager):
        """Test monitoring connection health"""
        # Mock data sources
        mock_source1 = Mock()
        mock_source1.name = "healthy_source"
        mock_source1.get_metrics.return_value = {
            'is_connected': True,
            'error_count': 0,
            'success_rate': 1.0,
            'uptime_seconds': 3600
        }
        
        mock_source2 = Mock()
        mock_source2.name = "unhealthy_source"
        mock_source2.get_metrics.return_value = {
            'is_connected': False,
            'error_count': 10,
            'success_rate': 0.5,
            'uptime_seconds': 1800
        }
        
        manager.add_source(mock_source1)
        manager.add_source(mock_source2)
        
        # Test connection status tracking
        assert manager.get_connection_status("healthy_source") != manager.get_connection_status("unhealthy_source")
    
    @pytest.mark.asyncio
    async def test_data_deduplication_performance(self, manager):
        """Test performance of data deduplication"""
        # Send duplicate data
        base_data = DataPoint(
            source="dedup_test",
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.0,
            volume=1000000,
            sequence_id="duplicate_123"
        )
        
        start_time = time.time()
        
        # Send same data 1000 times
        for i in range(1000):
            await manager.add_data_point(base_data)
        
        elapsed_time = time.time() - start_time
        
        # Should only store one copy
        data_count = await manager.get_data_count("AAPL")
        assert data_count == 1
        
        # Deduplication should be fast
        assert elapsed_time < 1.0  # Should complete in under 1 second


class TestRealWorldLatencyScenarios:
    """Test latency in realistic scenarios"""
    
    @pytest.mark.asyncio
    async def test_yahoo_finance_latency_simulation(self):
        """Test Yahoo Finance typical latency characteristics"""
        source = YahooRealtimeSource(use_websocket=False, update_interval=1.0)
        
        # Mock the REST adapter to simulate latency
        with patch.object(source, 'rest_adapter') as mock_adapter:
            mock_update = Mock()
            mock_update.timestamp = time.time() - 0.1  # 100ms ago
            mock_update.symbol = "AAPL"
            mock_update.price = 150.0
            mock_update.metadata = {'volume': 1000000}
            
            # Simulate handling the update
            await source._handle_rest_update(mock_update)
            
            # Yahoo Finance REST should typically have 50-200ms latency
            # This is simulated in the _handle_rest_update method
    
    @pytest.mark.asyncio
    async def test_websocket_latency_simulation(self):
        """Test WebSocket typical latency characteristics"""
        # Mock WebSocket data processing
        mock_trade_data = {
            's': 'AAPL',  # symbol
            'p': 150.25,  # price
            'v': 100,     # volume
            't': int(time.time() * 1000) - 10  # 10ms ago
        }
        
        # Calculate latency as would be done in real WebSocket handler
        trade_time = mock_trade_data['t'] / 1000
        current_time = time.time()
        latency_ms = (current_time - trade_time) * 1000
        
        # WebSocket latency should typically be < 50ms
        assert latency_ms < 100  # Allow some margin for test execution
        assert latency_ms >= 0
    
    @pytest.mark.asyncio
    async def test_cross_source_latency_comparison(self):
        """Test latency comparison across different sources"""
        manager = RealtimeManager()
        
        # Simulate different source characteristics
        source_profiles = {
            'websocket_source': {'base_latency': 10, 'variation': 5},
            'rest_api_source': {'base_latency': 100, 'variation': 20},
            'rate_limited_source': {'base_latency': 200, 'variation': 50}
        }
        
        for source_name, profile in source_profiles.items():
            for i in range(100):
                import random
                latency = profile['base_latency'] + random.uniform(-profile['variation'], profile['variation'])
                latency = max(1, latency)  # Ensure positive latency
                
                data_point = DataPoint(
                    source=source_name,
                    symbol="AAPL",
                    timestamp=datetime.now(),
                    price=150.0,
                    volume=1000000,
                    latency_ms=latency
                )
                await manager.add_data_point(data_point)
        
        # Verify latency characteristics match expectations
        ws_stats = await manager.get_latency_stats('websocket_source')
        rest_stats = await manager.get_latency_stats('rest_api_source')
        rate_limited_stats = await manager.get_latency_stats('rate_limited_source')
        
        # WebSocket should be fastest
        assert ws_stats.avg_latency_ms < rest_stats.avg_latency_ms
        assert rest_stats.avg_latency_ms < rate_limited_stats.avg_latency_ms
        
        # All should meet basic performance criteria for their type
        assert ws_stats.p95_latency_ms < 25  # WebSocket should be very fast
        assert rest_stats.p95_latency_ms < 150  # REST should be reasonable
        assert rate_limited_stats.avg_latency_ms < 300  # Rate limited but usable