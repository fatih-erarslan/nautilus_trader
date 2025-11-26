"""
Test suite for real-time data manager
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

from benchmark.src.data.realtime_manager import (
    RealtimeManager, 
    DataSource, 
    DataFeed,
    ConnectionStatus,
    DataPoint
)


class TestRealtimeManager:
    """Test real-time data manager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create a test manager instance"""
        return RealtimeManager(max_connections=10)
    
    @pytest.fixture
    def mock_yahoo_source(self):
        """Create a mock Yahoo Finance data source"""
        source = Mock(spec=DataSource)
        source.name = "yahoo_finance"
        source.is_connected = False
        source.connect = AsyncMock(return_value=True)
        source.disconnect = AsyncMock()
        source.subscribe = AsyncMock()
        source.unsubscribe = AsyncMock()
        return source
    
    @pytest.fixture
    def mock_finnhub_source(self):
        """Create a mock Finnhub data source"""
        source = Mock(spec=DataSource)
        source.name = "finnhub"
        source.is_connected = False
        source.connect = AsyncMock(return_value=True)
        source.disconnect = AsyncMock()
        source.subscribe = AsyncMock()
        source.unsubscribe = AsyncMock()
        return source
    
    @pytest.mark.asyncio
    async def test_add_data_source(self, manager, mock_yahoo_source):
        """Test adding a data source"""
        manager.add_source(mock_yahoo_source)
        assert "yahoo_finance" in manager.sources
        assert manager.sources["yahoo_finance"] == mock_yahoo_source
    
    @pytest.mark.asyncio
    async def test_connect_to_source(self, manager, mock_yahoo_source):
        """Test connecting to a data source"""
        manager.add_source(mock_yahoo_source)
        
        result = await manager.connect_source("yahoo_finance")
        assert result is True
        mock_yahoo_source.connect.assert_called_once()
        assert manager.get_connection_status("yahoo_finance") == ConnectionStatus.CONNECTED
    
    @pytest.mark.asyncio
    async def test_connect_all_sources(self, manager, mock_yahoo_source, mock_finnhub_source):
        """Test connecting to all sources"""
        manager.add_source(mock_yahoo_source)
        manager.add_source(mock_finnhub_source)
        
        results = await manager.connect_all()
        assert results == {"yahoo_finance": True, "finnhub": True}
        mock_yahoo_source.connect.assert_called_once()
        mock_finnhub_source.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_subscribe_to_symbols(self, manager, mock_yahoo_source):
        """Test subscribing to symbols"""
        manager.add_source(mock_yahoo_source)
        await manager.connect_source("yahoo_finance")
        
        symbols = ["AAPL", "GOOGL", "MSFT"]
        await manager.subscribe("yahoo_finance", symbols)
        
        mock_yahoo_source.subscribe.assert_called_once_with(symbols)
    
    @pytest.mark.asyncio
    async def test_data_aggregation(self, manager):
        """Test aggregating data from multiple sources"""
        # Create mock data points
        yahoo_data = DataPoint(
            source="yahoo_finance",
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            price=150.25,
            volume=1000000,
            bid=150.20,
            ask=150.30
        )
        
        finnhub_data = DataPoint(
            source="finnhub",
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            price=150.28,
            volume=1000100,
            bid=150.23,
            ask=150.33
        )
        
        # Add data points
        await manager.add_data_point(yahoo_data)
        await manager.add_data_point(finnhub_data)
        
        # Get aggregated data
        aggregated = await manager.get_aggregated_data("AAPL")
        
        assert aggregated is not None
        assert aggregated.symbol == "AAPL"
        assert aggregated.price == pytest.approx(150.265, 0.001)  # Average
        assert aggregated.volume == 2000100  # Sum
        assert aggregated.sources == ["yahoo_finance", "finnhub"]
    
    @pytest.mark.asyncio
    async def test_data_latency_tracking(self, manager):
        """Test tracking data latency"""
        data_point = DataPoint(
            source="yahoo_finance",
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            price=150.25,
            volume=1000000,
            latency_ms=25.5
        )
        
        await manager.add_data_point(data_point)
        
        latency_stats = await manager.get_latency_stats("yahoo_finance")
        assert latency_stats.avg_latency_ms <= 50  # Target <50ms
        assert latency_stats.min_latency_ms == 25.5
        assert latency_stats.max_latency_ms == 25.5
        assert latency_stats.count == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_symbol_handling(self, manager, mock_yahoo_source):
        """Test handling 100+ symbols concurrently"""
        manager.add_source(mock_yahoo_source)
        await manager.connect_source("yahoo_finance")
        
        # Generate 150 symbols
        symbols = [f"STOCK{i:03d}" for i in range(150)]
        
        # Subscribe to all symbols
        await manager.subscribe("yahoo_finance", symbols)
        
        # Simulate concurrent data updates
        tasks = []
        for symbol in symbols[:100]:  # First 100 symbols
            data_point = DataPoint(
                source="yahoo_finance",
                symbol=symbol,
                timestamp=datetime.utcnow(),
                price=100.0 + (hash(symbol) % 50),
                volume=1000000
            )
            tasks.append(manager.add_data_point(data_point))
        
        await asyncio.gather(*tasks)
        
        # Verify all data points were processed
        stats = await manager.get_stats()
        assert stats.total_symbols >= 100
        assert stats.total_data_points >= 100
    
    @pytest.mark.asyncio
    async def test_automatic_reconnection(self, manager, mock_yahoo_source):
        """Test automatic reconnection on disconnect"""
        # Configure reconnection
        mock_yahoo_source.reconnect_attempts = 0
        mock_yahoo_source.max_reconnect_attempts = 3
        
        manager.add_source(mock_yahoo_source)
        await manager.connect_source("yahoo_finance")
        
        # Simulate disconnect
        mock_yahoo_source.is_connected = False
        await manager.handle_disconnect("yahoo_finance")
        
        # Verify reconnection attempt
        assert mock_yahoo_source.connect.call_count >= 2  # Initial + reconnect
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, manager):
        """Test rate limit handling for API sources"""
        from benchmark.src.data.alpha_vantage import AlphaVantageSource
        
        # Create source with rate limiting
        source = AlphaVantageSource(api_key="test_key", rate_limit=5)  # 5 req/min
        manager.add_source(source)
        
        # Make rapid requests
        requests_made = 0
        start_time = asyncio.get_event_loop().time()
        
        for i in range(10):
            try:
                await source.fetch_quote("AAPL")
                requests_made += 1
            except Exception:
                pass
        
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        # Should respect rate limit
        assert requests_made <= 6  # Allow for slight variance
        assert elapsed_time >= 1.0  # Should take at least 1 second for throttling
    
    @pytest.mark.asyncio
    async def test_data_deduplication(self, manager):
        """Test deduplication of identical data points"""
        # Add duplicate data points
        for i in range(5):
            data_point = DataPoint(
                source="yahoo_finance",
                symbol="AAPL",
                timestamp=datetime.utcnow(),
                price=150.25,
                volume=1000000,
                sequence_id="duplicate_123"
            )
            await manager.add_data_point(data_point)
        
        # Should only store one copy
        data_count = await manager.get_data_count("AAPL")
        assert data_count == 1
    
    @pytest.mark.asyncio
    async def test_websocket_fallback_to_rest(self, manager):
        """Test fallback from WebSocket to REST API"""
        from benchmark.src.data.yahoo_realtime import YahooRealtimeSource
        
        source = YahooRealtimeSource(use_websocket=True)
        manager.add_source(source)
        
        # Simulate WebSocket failure
        with patch.object(source, 'connect_websocket', side_effect=Exception("WS Failed")):
            result = await manager.connect_source("yahoo_realtime")
            
            # Should fallback to REST
            assert result is True
            assert source.connection_type == "REST"
    
    @pytest.mark.asyncio 
    async def test_memory_efficient_caching(self, manager):
        """Test memory-efficient caching for large datasets"""
        # Add many data points
        for i in range(10000):
            data_point = DataPoint(
                source="yahoo_finance",
                symbol=f"STOCK{i % 100:03d}",
                timestamp=datetime.utcnow(),
                price=100.0 + (i % 50),
                volume=1000000
            )
            await manager.add_data_point(data_point)
        
        # Check memory usage
        stats = await manager.get_stats()
        assert stats.cache_size_mb < 100  # Should stay under 100MB
        assert stats.total_data_points == 10000
    
    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, manager, mock_yahoo_source, mock_finnhub_source):
        """Test proper cleanup on shutdown"""
        manager.add_source(mock_yahoo_source)
        manager.add_source(mock_finnhub_source)
        
        await manager.connect_all()
        await manager.shutdown()
        
        # Verify all sources disconnected
        mock_yahoo_source.disconnect.assert_called_once()
        mock_finnhub_source.disconnect.assert_called_once()
        
        # Verify manager state
        assert manager.is_running is False
        assert len(manager.active_connections) == 0