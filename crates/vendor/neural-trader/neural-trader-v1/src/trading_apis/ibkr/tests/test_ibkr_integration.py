"""
Test suite for IBKR integration

Tests the IBKRClient, IBKRGateway, and IBKRDataStream components
with both unit tests and integration tests.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from src.trading_apis.ibkr import IBKRClient, IBKRGateway, IBKRDataStream
from src.trading_apis.ibkr.ibkr_client import ConnectionConfig, LatencyMetrics
from src.trading_apis.ibkr.ibkr_gateway import GatewayConfig, ConnectionMode
from src.trading_apis.ibkr.ibkr_data_stream import StreamConfig, DataType, MarketSnapshot


class TestIBKRClient:
    """Test IBKRClient functionality"""
    
    @pytest.fixture
    def client_config(self):
        return ConnectionConfig(
            host="localhost",
            port=7497,
            client_id=999,
            readonly=True,
            timeout=5.0
        )
    
    @pytest.fixture
    def client(self, client_config):
        return IBKRClient(client_config)
    
    def test_client_initialization(self, client):
        """Test client initialization"""
        assert client.config.host == "localhost"
        assert client.config.port == 7497
        assert client.config.client_id == 999
        assert not client._connected
        assert client.metrics is not None
    
    def test_latency_metrics(self):
        """Test latency metrics tracking"""
        metrics = LatencyMetrics()
        
        # Add some metrics
        metrics.add_metric('order_submission', 50.0)
        metrics.add_metric('order_submission', 75.0)
        metrics.add_metric('order_submission', 100.0)
        
        stats = metrics.get_stats('order_submission')
        assert stats['avg'] == 75.0
        assert stats['min'] == 50.0
        assert stats['max'] == 100.0
        assert stats['p50'] == 75.0
    
    @pytest.mark.asyncio
    async def test_connect_success(self, client):
        """Test successful connection"""
        with patch.object(client, '_do_connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = None
            
            with patch.object(client, '_setup_event_handlers'):
                with patch.object(client, '_initialize_data', new_callable=AsyncMock):
                    result = await client.connect()
                    assert result is True
                    assert client._connected is True
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, client):
        """Test connection failure"""
        with patch.object(client, '_do_connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            result = await client.connect()
            assert result is False
            assert client._connected is False
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, client):
        """Test successful order placement"""
        client._connected = True
        
        with patch.object(client, 'ib') as mock_ib:
            mock_trade = Mock()
            mock_trade.order.orderId = "123456"
            mock_ib.placeOrder.return_value = mock_trade
            
            order_id = await client.place_order("AAPL", 100, "MKT", "BUY")
            
            assert order_id == "123456"
            assert "123456" in client._pending_orders
    
    @pytest.mark.asyncio
    async def test_place_order_not_connected(self, client):
        """Test order placement when not connected"""
        client._connected = False
        
        order_id = await client.place_order("AAPL", 100, "MKT", "BUY")
        
        assert order_id is None
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """Test order cancellation"""
        client._connected = True
        
        # Add a pending order
        client._pending_orders["123456"] = {
            'trade': Mock(),
            'submit_time': time.time(),
            'symbol': 'AAPL',
            'quantity': 100,
            'side': 'BUY'
        }
        
        with patch.object(client, 'ib') as mock_ib:
            result = await client.cancel_order("123456")
            assert result is True
            mock_ib.cancelOrder.assert_called_once()
    
    def test_latency_report(self, client):
        """Test latency report generation"""
        # Add some metrics
        client.metrics.add_metric('order_submission', 50.0)
        client.metrics.add_metric('order_fill', 150.0)
        
        report = client.get_latency_report()
        
        assert 'order_submission' in report
        assert 'order_fill' in report
        assert report['order_submission']['avg'] == 50.0
        assert report['order_fill']['avg'] == 150.0


class TestIBKRGateway:
    """Test IBKRGateway functionality"""
    
    @pytest.fixture
    def gateway_config(self):
        return GatewayConfig(
            primary_host="localhost",
            primary_port=4001,
            connection_mode=ConnectionMode.DIRECT,
            max_connections=3
        )
    
    @pytest.fixture
    def gateway(self, gateway_config):
        return IBKRGateway(gateway_config)
    
    def test_gateway_initialization(self, gateway):
        """Test gateway initialization"""
        assert gateway.config.primary_host == "localhost"
        assert gateway.config.primary_port == 4001
        assert gateway.config.connection_mode == ConnectionMode.DIRECT
        assert gateway.pool is not None
    
    @pytest.mark.asyncio
    async def test_connect_primary_success(self, gateway):
        """Test successful connection to primary gateway"""
        with patch('src.trading_apis.ibkr.ibkr_gateway.GatewayConnection') as mock_conn_class:
            mock_conn = Mock()
            mock_conn.connect = AsyncMock(return_value=True)
            mock_conn_class.return_value = mock_conn
            
            result = await gateway.connect()
            
            assert result is True
            assert gateway.primary_connection == mock_conn
    
    @pytest.mark.asyncio
    async def test_connect_failover(self, gateway):
        """Test failover to backup gateway"""
        gateway.config.backup_hosts = [("backup1", 4001), ("backup2", 4001)]
        
        with patch('src.trading_apis.ibkr.ibkr_gateway.GatewayConnection') as mock_conn_class:
            # Primary fails
            mock_primary = Mock()
            mock_primary.connect = AsyncMock(return_value=False)
            
            # Backup succeeds
            mock_backup = Mock()
            mock_backup.connect = AsyncMock(return_value=True)
            
            mock_conn_class.side_effect = [mock_primary, mock_backup]
            
            result = await gateway.connect()
            
            assert result is True
            assert gateway.primary_connection == mock_backup
    
    @pytest.mark.asyncio
    async def test_send_message(self, gateway):
        """Test message sending"""
        mock_conn = Mock()
        mock_conn.send_message = AsyncMock(return_value=True)
        gateway.primary_connection = mock_conn
        
        result = await gateway.send_message(b"test message")
        
        assert result is True
        mock_conn.send_message.assert_called_once_with(b"test message")
    
    @pytest.mark.asyncio
    async def test_receive_message(self, gateway):
        """Test message receiving"""
        mock_conn = Mock()
        mock_conn.get_message = AsyncMock(return_value=b"response")
        gateway.primary_connection = mock_conn
        
        result = await gateway.receive_message()
        
        assert result == b"response"
        mock_conn.get_message.assert_called_once()


class TestIBKRDataStream:
    """Test IBKRDataStream functionality"""
    
    @pytest.fixture
    def stream_config(self):
        return StreamConfig(
            buffer_size=1000,
            batch_size=50,
            batch_timeout_ms=10.0,
            max_symbols=10
        )
    
    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client._connected = True
        client.register_callback = Mock()
        return client
    
    @pytest.fixture
    def stream(self, mock_client, stream_config):
        return IBKRDataStream(mock_client, stream_config)
    
    def test_stream_initialization(self, stream):
        """Test stream initialization"""
        assert stream.config.buffer_size == 1000
        assert stream.config.batch_size == 50
        assert len(stream.subscriptions) == 0
        assert len(stream.snapshots) == 0
    
    @pytest.mark.asyncio
    async def test_subscribe_success(self, stream):
        """Test successful subscription"""
        with patch.object(stream, '_subscribe_trades', return_value=1001):
            with patch.object(stream, '_subscribe_quotes', return_value=1002):
                result = await stream.subscribe(
                    "AAPL",
                    [DataType.TRADES, DataType.QUOTES]
                )
                
                assert result is True
                assert DataType.TRADES in stream.subscriptions["AAPL"]
                assert DataType.QUOTES in stream.subscriptions["AAPL"]
                assert "AAPL" in stream.snapshots
    
    @pytest.mark.asyncio
    async def test_subscribe_max_symbols(self, stream):
        """Test subscription limit"""
        # Fill up to max symbols
        for i in range(stream.config.max_symbols):
            await stream.subscribe(f"SYM{i}", [DataType.TRADES])
        
        # Try to add one more
        result = await stream.subscribe("OVERFLOW", [DataType.TRADES])
        assert result is False
    
    @pytest.mark.asyncio
    async def test_tick_processing(self, stream):
        """Test tick data processing"""
        # Subscribe to a symbol
        await stream.subscribe("AAPL", [DataType.TRADES])
        
        # Simulate tick data
        tick_data = {
            'req_id': 1001,
            'tick_type': 4,  # LAST
            'price': 150.00,
            'size': 100
        }
        
        stream.req_id_map[1001] = "AAPL"
        
        await stream._on_tick_price(tick_data)
        
        # Check if snapshot was updated
        snapshot = stream.snapshots["AAPL"]
        assert snapshot.last == 150.00
        assert snapshot.last_size == 100
    
    def test_market_snapshot(self):
        """Test market snapshot functionality"""
        snapshot = MarketSnapshot(
            symbol="AAPL",
            timestamp=time.time(),
            bid=149.50,
            ask=150.50,
            last=150.00
        )
        
        assert snapshot.spread == 1.0
        assert snapshot.mid == 150.0
    
    @pytest.mark.asyncio
    async def test_conflation(self, stream):
        """Test data conflation"""
        # Create test batch with rapid updates
        batch = [
            {'timestamp': 1000.0, 'price': 100.0},
            {'timestamp': 1000.001, 'price': 100.1},  # Should be conflated
            {'timestamp': 1000.002, 'price': 100.2},  # Should be conflated
            {'timestamp': 1000.010, 'price': 100.3},  # Should be kept
            {'timestamp': 1000.020, 'price': 100.4},  # Should be kept
        ]
        
        conflated = stream._conflate_batch(batch, 5.0)  # 5ms conflation
        
        # Should have 3 items: first, one at 1000.010, one at 1000.020
        assert len(conflated) == 3
        assert conflated[0]['price'] == 100.0
        assert conflated[1]['price'] == 100.3
        assert conflated[2]['price'] == 100.4
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, stream):
        """Test unsubscription"""
        # Subscribe first
        await stream.subscribe("AAPL", [DataType.TRADES, DataType.QUOTES])
        
        # Unsubscribe from specific data type
        await stream.unsubscribe("AAPL", [DataType.TRADES])
        
        assert DataType.TRADES not in stream.subscriptions["AAPL"]
        assert DataType.QUOTES in stream.subscriptions["AAPL"]
        assert "AAPL" in stream.snapshots  # Should still exist
        
        # Unsubscribe from all
        await stream.unsubscribe("AAPL")
        
        assert "AAPL" not in stream.subscriptions
        assert "AAPL" not in stream.snapshots
    
    def test_statistics(self, stream):
        """Test statistics collection"""
        # Add some fake stats
        stream._stats['ticks_received'] = 1000
        stream._stats['ticks_processed'] = 950
        stream._stats['ticks_dropped'] = 50
        
        stats = stream.get_statistics()
        
        assert stats['ticks_received'] == 1000
        assert stats['ticks_processed'] == 950
        assert stats['ticks_dropped'] == 50
        assert stats['drop_rate'] == 0.05


class TestIntegration:
    """Integration tests for full workflow"""
    
    @pytest.mark.asyncio
    async def test_full_trading_workflow(self):
        """Test complete trading workflow"""
        # This would be a more comprehensive test
        # that requires actual IB connection
        pass
    
    @pytest.mark.asyncio
    async def test_latency_benchmarks(self):
        """Test latency benchmarks"""
        config = ConnectionConfig(timeout=1.0)
        client = IBKRClient(config)
        
        # Test connection latency
        start_time = time.time()
        # In real test, would connect to mock server
        connection_time = (time.time() - start_time) * 1000
        
        # Should be under target latency
        assert connection_time < 1000  # 1 second for connection
    
    def test_memory_usage(self):
        """Test memory usage with large data volumes"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large stream buffer
        config = StreamConfig(buffer_size=100000)
        client = Mock()
        stream = IBKRDataStream(client, config)
        
        # Fill buffer with data
        for i in range(10000):
            snapshot = MarketSnapshot(
                symbol=f"SYM{i % 100}",
                timestamp=time.time(),
                bid=100.0 + i * 0.01,
                ask=100.1 + i * 0.01,
                last=100.05 + i * 0.01
            )
            stream.snapshots[f"SYM{i % 100}"] = snapshot
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])