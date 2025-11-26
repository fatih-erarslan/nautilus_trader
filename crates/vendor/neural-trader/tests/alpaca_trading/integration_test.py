"""
End-to-end integration tests for the Alpaca trading system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
import websockets

from src.alpaca_trading.websocket_client import AlpacaWebSocketClient
from src.alpaca_trading.strategies import StrategyManager, MomentumStrategy, VWAPStrategy
from src.alpaca_trading.execution import ExecutionEngine, Order, OrderSide, OrderType
from src.alpaca_trading.risk_management import RiskManager, RiskLimits
from src.alpaca_trading.monitoring.performance_metrics import PerformanceMetrics, Trade
from src.alpaca_trading.monitoring.latency_tracker import LatencyTracker
from src.alpaca_trading.monitoring.stream_health import StreamHealthMonitor
from src.alpaca_trading.monitoring.alert_system import AlertSystem, AlertType, AlertSeverity


@pytest.fixture
def mock_alpaca_client():
    """Mock Alpaca REST client."""
    client = Mock()
    client.submit_order = AsyncMock()
    client.cancel_order = AsyncMock()
    client.get_account = AsyncMock(return_value={
        'buying_power': '100000',
        'cash': '100000',
        'equity': '100000'
    })
    client.get_positions = AsyncMock(return_value=[])
    return client


@pytest.fixture
async def trading_system(mock_alpaca_client):
    """Create complete trading system."""
    # Initialize components
    ws_client = AlpacaWebSocketClient('test_key', 'test_secret')
    strategy_manager = StrategyManager()
    execution_engine = ExecutionEngine(mock_alpaca_client)
    risk_manager = RiskManager(RiskLimits())
    performance_metrics = PerformanceMetrics()
    latency_tracker = LatencyTracker()
    stream_monitor = StreamHealthMonitor()
    alert_system = AlertSystem()
    
    # Set up strategies
    strategy_manager.add_strategy('momentum', MomentumStrategy(['AAPL', 'GOOGL']))
    strategy_manager.add_strategy('vwap', VWAPStrategy(['AAPL', 'GOOGL']))
    
    # Initialize
    await execution_engine.initialize()
    await stream_monitor.start()
    
    system = {
        'ws_client': ws_client,
        'strategies': strategy_manager,
        'execution': execution_engine,
        'risk': risk_manager,
        'performance': performance_metrics,
        'latency': latency_tracker,
        'stream_monitor': stream_monitor,
        'alerts': alert_system
    }
    
    yield system
    
    # Cleanup
    await stream_monitor.stop()
    await execution_engine.shutdown()


class TestTradingSystemIntegration:
    """Test complete trading system integration."""
    
    @pytest.mark.asyncio
    async def test_full_trading_cycle(self, trading_system, mock_alpaca_client):
        """Test complete trading cycle from signal to execution."""
        # Mock WebSocket messages
        mock_messages = [
            json.dumps({'T': 'success', 'msg': 'authenticated'}),
            # Price surge triggers momentum signal
            json.dumps({'T': 't', 'S': 'AAPL', 'p': 150.0, 's': 100, 't': '2024-01-10T10:00:00Z'}),
            json.dumps({'T': 't', 'S': 'AAPL', 'p': 151.0, 's': 200, 't': '2024-01-10T10:00:01Z'}),
            json.dumps({'T': 't', 'S': 'AAPL', 'p': 152.0, 's': 150, 't': '2024-01-10T10:00:02Z'}),
            json.dumps({'T': 't', 'S': 'AAPL', 'p': 153.0, 's': 100, 't': '2024-01-10T10:00:03Z'}),
        ]
        
        message_index = 0
        
        async def mock_recv():
            nonlocal message_index
            if message_index < len(mock_messages):
                msg = mock_messages[message_index]
                message_index += 1
                return msg
            await asyncio.sleep(0.1)
            return json.dumps({'T': 'keepalive'})
        
        mock_ws = AsyncMock()
        mock_ws.recv = mock_recv
        
        # Track order submissions
        submitted_orders = []
        
        async def mock_submit_order(**kwargs):
            submitted_orders.append(kwargs)
            return {
                'id': f'order_{len(submitted_orders)}',
                'status': 'accepted',
                'symbol': kwargs['symbol'],
                'qty': kwargs['qty'],
                'side': kwargs['side']
            }
        
        mock_alpaca_client.submit_order = mock_submit_order
        
        # Set up data flow
        async def on_trade(data):
            # Feed to strategies
            await trading_system['strategies'].on_trade(data)
            
            # Check for signals
            signal = await trading_system['strategies'].get_signal(data['symbol'])
            
            if signal and signal.strength > 0.6:
                # Risk check
                risk_check = await trading_system['risk'].check_trade_risk(
                    symbol=signal.symbol,
                    side='buy' if signal.type.value == 'BUY' else 'sell',
                    quantity=signal.quantity,
                    price=signal.price,
                    portfolio={'cash': 100000, 'equity': 100000}
                )
                
                if risk_check.approved:
                    # Submit order
                    order = Order(
                        symbol=signal.symbol,
                        side=OrderSide.BUY if signal.type.value == 'BUY' else OrderSide.SELL,
                        quantity=signal.quantity,
                        order_type=OrderType.MARKET
                    )
                    
                    await trading_system['execution'].submit_order(order)
        
        trading_system['ws_client'].on_trade(on_trade)
        
        with patch('websockets.connect', AsyncMock(return_value=mock_ws)):
            # Subscribe and start
            await trading_system['ws_client'].subscribe_trades(['AAPL'])
            
            # Run for a short time
            run_task = asyncio.create_task(trading_system['ws_client'].run())
            await asyncio.sleep(0.5)
            
            # Should have submitted at least one order
            assert len(submitted_orders) > 0
            assert submitted_orders[0]['symbol'] == 'AAPL'
            assert submitted_orders[0]['side'] == 'buy'
            
            await trading_system['ws_client'].stop()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_latency_monitoring(self, trading_system):
        """Test latency tracking across the system."""
        latency = trading_system['latency']
        
        # Simulate order lifecycle with latency tracking
        with latency.measure('signal_generation'):
            await asyncio.sleep(0.001)  # 1ms signal generation
        
        with latency.measure('risk_check'):
            await asyncio.sleep(0.002)  # 2ms risk check
        
        with latency.measure('order_submission'):
            await asyncio.sleep(0.005)  # 5ms order submission
        
        # Check latency stats
        signal_stats = latency.get_statistics('signal_generation')
        risk_stats = latency.get_statistics('risk_check')
        order_stats = latency.get_statistics('order_submission')
        
        assert signal_stats['mean'] >= 1000  # >= 1ms in microseconds
        assert risk_stats['mean'] >= 2000    # >= 2ms
        assert order_stats['mean'] >= 5000   # >= 5ms
        
        # Check total latency
        total_latency = sum([
            signal_stats['mean'],
            risk_stats['mean'],
            order_stats['mean']
        ])
        
        assert total_latency < 100000  # < 100ms total
    
    @pytest.mark.asyncio
    async def test_risk_circuit_breaker(self, trading_system):
        """Test risk management circuit breaker."""
        risk = trading_system['risk']
        execution = trading_system['execution']
        
        # Record significant losses
        await risk.record_trade_result('AAPL', -300)
        await risk.record_trade_result('GOOGL', -400)
        await risk.record_trade_result('MSFT', -350)
        
        # Try to submit new order - should be blocked
        order = Order('TSLA', OrderSide.BUY, 100, OrderType.MARKET)
        
        # Check risk before submission
        risk_check = await risk.check_trade_risk(
            symbol='TSLA',
            side='buy',
            quantity=100,
            price=700,
            portfolio={'cash': 50000, 'equity': 90000}
        )
        
        assert not risk_check.approved
        assert 'Daily loss limit' in risk_check.reason
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, trading_system):
        """Test performance metrics tracking."""
        perf = trading_system['performance']
        
        # Simulate completed trades
        trades = [
            Trade(
                symbol='AAPL',
                entry_time=datetime.now() - timedelta(minutes=10),
                exit_time=datetime.now() - timedelta(minutes=5),
                entry_price=150.0,
                exit_price=152.0,
                quantity=100,
                side='long',
                pnl=200,
                fees=2,
                slippage=5
            ),
            Trade(
                symbol='GOOGL',
                entry_time=datetime.now() - timedelta(minutes=8),
                exit_time=datetime.now() - timedelta(minutes=3),
                entry_price=2800.0,
                exit_price=2790.0,
                quantity=10,
                side='long',
                pnl=-100,
                fees=2,
                slippage=3
            )
        ]
        
        for trade in trades:
            await perf.add_trade(trade)
        
        # Check metrics
        metrics = await perf.get_real_time_metrics()
        
        assert metrics['trade_stats']['total_trades'] == 2
        assert metrics['trade_stats']['win_rate'] == 50.0
        assert metrics['capital']['total_pnl'] == 93  # 200 - 100 - 7 (fees+slippage)
    
    @pytest.mark.asyncio
    async def test_alert_system_integration(self, trading_system):
        """Test alert system integration."""
        alerts = trading_system['alerts']
        
        # Set up alert handler
        received_alerts = []
        
        def alert_handler(alert, resolved=False):
            received_alerts.append((alert, resolved))
        
        alerts.add_handler(alert_handler)
        
        # Trigger various alerts
        await alerts.check_latency('order_submission', 150)  # High latency
        await alerts.check_risk({
            'drawdown_pct': 12,
            'max_drawdown_pct': 10
        })
        await alerts.check_connection('disconnected', 'Network error')
        
        # Check alerts were triggered
        assert len(received_alerts) >= 3
        
        alert_types = [alert[0].type for alert in received_alerts]
        assert AlertType.LATENCY in alert_types
        assert AlertType.RISK_BREACH in alert_types
        assert AlertType.CONNECTION_ISSUE in alert_types
    
    @pytest.mark.asyncio
    async def test_stream_health_monitoring(self, trading_system):
        """Test stream health monitoring."""
        monitor = trading_system['stream_monitor']
        
        # Simulate connection lifecycle
        await monitor.on_connected()
        
        # Simulate message flow
        for i in range(100):
            await monitor.on_message(
                message_type='trade',
                size_bytes=150,
                timestamp=datetime.now() - timedelta(microseconds=500)
            )
            await asyncio.sleep(0.01)  # 100 messages/second
        
        # Check health
        health = await monitor.get_health_status()
        
        assert health['status'] == 'healthy'
        assert health['connection']['current_status'] == 'connected'
        assert health['messages']['trade']['rate_per_second'] > 50
        assert health['health_score'] >= 80
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, trading_system):
        """Test graceful system shutdown."""
        # Start some background tasks
        monitor_task = asyncio.create_task(
            trading_system['risk'].start_monitoring()
        )
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Shutdown sequence
        trading_system['risk'].stop_monitoring()
        await trading_system['stream_monitor'].stop()
        await trading_system['execution'].shutdown()
        
        # Wait for tasks to complete
        await asyncio.gather(monitor_task, return_exceptions=True)
        
        # Verify clean shutdown
        assert not trading_system['ws_client'].is_connected
        assert not trading_system['risk']._monitoring
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, trading_system):
        """Test system error recovery."""
        ws_client = trading_system['ws_client']
        
        # Simulate connection error
        error_count = 0
        
        def on_error(error):
            nonlocal error_count
            error_count += 1
        
        ws_client.on_error(on_error)
        
        # Mock connection that fails then succeeds
        attempt = 0
        
        async def mock_connect(*args, **kwargs):
            nonlocal attempt
            attempt += 1
            
            if attempt < 3:
                raise websockets.exceptions.ConnectionClosed(None, None)
            
            mock_ws = AsyncMock()
            mock_ws.recv = AsyncMock(return_value=json.dumps({
                'T': 'success',
                'msg': 'authenticated'
            }))
            return mock_ws
        
        with patch('websockets.connect', mock_connect):
            # Should eventually connect after retries
            run_task = asyncio.create_task(ws_client.run())
            await asyncio.sleep(1.0)
            
            assert ws_client.is_connected
            assert error_count >= 2  # At least 2 connection errors
            
            await ws_client.stop()
            try:
                await run_task
            except asyncio.CancelledError:
                pass