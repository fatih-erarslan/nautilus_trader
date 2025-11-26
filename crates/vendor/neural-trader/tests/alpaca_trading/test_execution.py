"""
Comprehensive tests for order execution engine.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from decimal import Decimal

from src.alpaca_trading.execution import (
    OrderType,
    OrderSide,
    OrderStatus,
    Order,
    ExecutionEngine,
    OrderValidator,
    PositionTracker,
    ExecutionError
)


@pytest.fixture
def mock_alpaca_client():
    """Mock Alpaca REST client."""
    client = Mock()
    client.submit_order = AsyncMock()
    client.cancel_order = AsyncMock()
    client.get_order = AsyncMock()
    client.get_position = AsyncMock()
    client.get_positions = AsyncMock(return_value=[])
    client.get_account = AsyncMock(return_value={
        'buying_power': '100000',
        'cash': '100000',
        'equity': '100000'
    })
    return client


@pytest.fixture
async def execution_engine(mock_alpaca_client):
    """Create execution engine with mock client."""
    engine = ExecutionEngine(mock_alpaca_client)
    await engine.initialize()
    yield engine
    await engine.shutdown()


class TestOrder:
    """Test Order class."""
    
    def test_order_creation(self):
        """Test order creation."""
        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0
        )
        
        assert order.symbol == 'AAPL'
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 150.0
        assert order.status == OrderStatus.PENDING
    
    def test_order_validation(self):
        """Test order validation."""
        # Valid limit order
        order = Order('AAPL', OrderSide.BUY, 100, OrderType.LIMIT, limit_price=150.0)
        assert order.is_valid()
        
        # Invalid limit order (no price)
        order = Order('AAPL', OrderSide.BUY, 100, OrderType.LIMIT)
        assert not order.is_valid()
        
        # Valid market order
        order = Order('AAPL', OrderSide.BUY, 100, OrderType.MARKET)
        assert order.is_valid()
        
        # Invalid quantity
        order = Order('AAPL', OrderSide.BUY, -100, OrderType.MARKET)
        assert not order.is_valid()


class TestOrderValidator:
    """Test order validation."""
    
    @pytest.mark.asyncio
    async def test_buying_power_validation(self):
        """Test buying power validation."""
        validator = OrderValidator()
        
        account = {
            'buying_power': '10000',
            'cash': '10000'
        }
        
        # Valid order within buying power
        order = Order('AAPL', OrderSide.BUY, 50, OrderType.LIMIT, limit_price=150.0)
        result = await validator.validate_order(order, account)
        assert result.is_valid
        
        # Invalid order exceeding buying power
        order = Order('AAPL', OrderSide.BUY, 100, OrderType.LIMIT, limit_price=150.0)
        result = await validator.validate_order(order, account)
        assert not result.is_valid
        assert 'Insufficient buying power' in result.reason
    
    @pytest.mark.asyncio
    async def test_position_size_validation(self):
        """Test position size limits."""
        validator = OrderValidator(max_position_size=1000)
        
        positions = {
            'AAPL': {'qty': '800', 'side': 'long'}
        }
        
        # Valid order within limit
        order = Order('AAPL', OrderSide.BUY, 100, OrderType.MARKET)
        result = await validator.validate_order(order, {}, positions)
        assert result.is_valid
        
        # Invalid order exceeding limit
        order = Order('AAPL', OrderSide.BUY, 300, OrderType.MARKET)
        result = await validator.validate_order(order, {}, positions)
        assert not result.is_valid
        assert 'Position size limit' in result.reason
    
    @pytest.mark.asyncio
    async def test_short_selling_validation(self):
        """Test short selling validation."""
        validator = OrderValidator(allow_short_selling=False)
        
        # No existing position
        positions = {}
        
        # Should not allow short selling
        order = Order('AAPL', OrderSide.SELL, 100, OrderType.MARKET)
        result = await validator.validate_order(order, {}, positions)
        assert not result.is_valid
        assert 'Short selling not allowed' in result.reason
        
        # With existing long position
        positions = {'AAPL': {'qty': '200', 'side': 'long'}}
        
        # Should allow selling existing position
        order = Order('AAPL', OrderSide.SELL, 100, OrderType.MARKET)
        result = await validator.validate_order(order, {}, positions)
        assert result.is_valid


class TestPositionTracker:
    """Test position tracking."""
    
    @pytest.mark.asyncio
    async def test_position_update(self):
        """Test position updates."""
        tracker = PositionTracker()
        
        # Add long position
        await tracker.update_position('AAPL', 100, 150.0, 'long')
        
        position = tracker.get_position('AAPL')
        assert position['quantity'] == 100
        assert position['avg_price'] == 150.0
        assert position['side'] == 'long'
        
        # Add to position
        await tracker.update_position('AAPL', 50, 152.0, 'long')
        
        position = tracker.get_position('AAPL')
        assert position['quantity'] == 150
        assert position['avg_price'] == 150.67  # Weighted average
    
    @pytest.mark.asyncio
    async def test_position_closure(self):
        """Test closing positions."""
        tracker = PositionTracker()
        
        # Open position
        await tracker.update_position('AAPL', 100, 150.0, 'long')
        
        # Partial close
        await tracker.update_position('AAPL', -50, 155.0, 'long')
        
        position = tracker.get_position('AAPL')
        assert position['quantity'] == 50
        
        # Full close
        await tracker.update_position('AAPL', -50, 155.0, 'long')
        
        position = tracker.get_position('AAPL')
        assert position is None
    
    @pytest.mark.asyncio
    async def test_pnl_calculation(self):
        """Test P&L calculation."""
        tracker = PositionTracker()
        
        # Open position
        await tracker.update_position('AAPL', 100, 150.0, 'long')
        
        # Update market price
        await tracker.update_market_price('AAPL', 155.0)
        
        pnl = tracker.get_unrealized_pnl('AAPL')
        assert pnl == 500.0  # 100 * (155 - 150)
        
        # Close with profit
        await tracker.update_position('AAPL', -100, 155.0, 'long')
        
        realized_pnl = tracker.get_realized_pnl('AAPL')
        assert realized_pnl == 500.0


class TestExecutionEngine:
    """Test execution engine."""
    
    @pytest.mark.asyncio
    async def test_order_submission(self, execution_engine, mock_alpaca_client):
        """Test order submission."""
        # Mock successful order submission
        mock_alpaca_client.submit_order.return_value = {
            'id': 'order123',
            'status': 'accepted',
            'filled_qty': '0',
            'filled_avg_price': None
        }
        
        order = Order('AAPL', OrderSide.BUY, 100, OrderType.LIMIT, limit_price=150.0)
        order_id = await execution_engine.submit_order(order)
        
        assert order_id == 'order123'
        assert order.status == OrderStatus.SUBMITTED
        mock_alpaca_client.submit_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_order_cancellation(self, execution_engine, mock_alpaca_client):
        """Test order cancellation."""
        # Submit order first
        mock_alpaca_client.submit_order.return_value = {
            'id': 'order123',
            'status': 'accepted'
        }
        
        order = Order('AAPL', OrderSide.BUY, 100, OrderType.LIMIT, limit_price=150.0)
        order_id = await execution_engine.submit_order(order)
        
        # Cancel order
        mock_alpaca_client.cancel_order.return_value = {
            'id': 'order123',
            'status': 'canceled'
        }
        
        success = await execution_engine.cancel_order(order_id)
        assert success
        assert execution_engine.orders[order_id].status == OrderStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_order_fill_handling(self, execution_engine, mock_alpaca_client):
        """Test order fill handling."""
        # Submit order
        mock_alpaca_client.submit_order.return_value = {
            'id': 'order123',
            'status': 'accepted'
        }
        
        order = Order('AAPL', OrderSide.BUY, 100, OrderType.MARKET)
        order_id = await execution_engine.submit_order(order)
        
        # Simulate fill
        fill_data = {
            'order_id': order_id,
            'symbol': 'AAPL',
            'filled_qty': 100,
            'filled_avg_price': 150.5,
            'side': 'buy'
        }
        
        await execution_engine._handle_fill(fill_data)
        
        # Check order status
        assert execution_engine.orders[order_id].status == OrderStatus.FILLED
        assert execution_engine.orders[order_id].filled_quantity == 100
        assert execution_engine.orders[order_id].avg_fill_price == 150.5
        
        # Check position update
        position = execution_engine.position_tracker.get_position('AAPL')
        assert position['quantity'] == 100
        assert position['avg_price'] == 150.5
    
    @pytest.mark.asyncio
    async def test_partial_fill_handling(self, execution_engine, mock_alpaca_client):
        """Test partial fill handling."""
        # Submit order
        mock_alpaca_client.submit_order.return_value = {
            'id': 'order123',
            'status': 'accepted'
        }
        
        order = Order('AAPL', OrderSide.BUY, 100, OrderType.LIMIT, limit_price=150.0)
        order_id = await execution_engine.submit_order(order)
        
        # First partial fill
        await execution_engine._handle_fill({
            'order_id': order_id,
            'symbol': 'AAPL',
            'filled_qty': 30,
            'filled_avg_price': 150.0,
            'side': 'buy'
        })
        
        assert execution_engine.orders[order_id].status == OrderStatus.PARTIALLY_FILLED
        assert execution_engine.orders[order_id].filled_quantity == 30
        
        # Second partial fill
        await execution_engine._handle_fill({
            'order_id': order_id,
            'symbol': 'AAPL',
            'filled_qty': 70,
            'filled_avg_price': 150.0,
            'side': 'buy'
        })
        
        assert execution_engine.orders[order_id].status == OrderStatus.FILLED
        assert execution_engine.orders[order_id].filled_quantity == 100
    
    @pytest.mark.asyncio
    async def test_slippage_calculation(self, execution_engine, mock_alpaca_client):
        """Test slippage calculation."""
        # Submit limit order
        mock_alpaca_client.submit_order.return_value = {
            'id': 'order123',
            'status': 'accepted'
        }
        
        order = Order('AAPL', OrderSide.BUY, 100, OrderType.LIMIT, limit_price=150.0)
        order_id = await execution_engine.submit_order(order)
        
        # Fill at worse price
        await execution_engine._handle_fill({
            'order_id': order_id,
            'symbol': 'AAPL',
            'filled_qty': 100,
            'filled_avg_price': 150.5,
            'side': 'buy'
        })
        
        slippage = execution_engine.calculate_slippage(order_id)
        assert slippage == -50.0  # 100 * (150.5 - 150.0) = -$50 slippage
    
    @pytest.mark.asyncio
    async def test_concurrent_order_handling(self, execution_engine, mock_alpaca_client):
        """Test handling multiple concurrent orders."""
        # Mock order submissions
        mock_alpaca_client.submit_order.side_effect = [
            {'id': 'order1', 'status': 'accepted'},
            {'id': 'order2', 'status': 'accepted'},
            {'id': 'order3', 'status': 'accepted'}
        ]
        
        # Submit multiple orders concurrently
        orders = [
            Order('AAPL', OrderSide.BUY, 100, OrderType.MARKET),
            Order('GOOGL', OrderSide.BUY, 50, OrderType.MARKET),
            Order('MSFT', OrderSide.SELL, 75, OrderType.MARKET)
        ]
        
        order_ids = await asyncio.gather(*[
            execution_engine.submit_order(order) for order in orders
        ])
        
        assert len(order_ids) == 3
        assert all(oid in execution_engine.orders for oid in order_ids)
    
    @pytest.mark.asyncio
    async def test_order_timeout(self, execution_engine, mock_alpaca_client):
        """Test order timeout handling."""
        # Submit order with timeout
        mock_alpaca_client.submit_order.return_value = {
            'id': 'order123',
            'status': 'accepted'
        }
        
        order = Order(
            'AAPL', 
            OrderSide.BUY, 
            100, 
            OrderType.LIMIT, 
            limit_price=150.0,
            time_in_force='IOC'  # Immediate or cancel
        )
        
        order_id = await execution_engine.submit_order(order)
        
        # Simulate timeout
        await execution_engine._handle_order_timeout(order_id)
        
        assert execution_engine.orders[order_id].status == OrderStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_risk_check_integration(self, execution_engine, mock_alpaca_client):
        """Test risk management integration."""
        # Configure risk limits
        execution_engine.order_validator.max_order_value = 10000
        
        # Try to submit order exceeding limit
        order = Order('AAPL', OrderSide.BUY, 100, OrderType.LIMIT, limit_price=150.0)
        
        with pytest.raises(ExecutionError) as exc_info:
            await execution_engine.submit_order(order)
        
        assert 'exceeds maximum order value' in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execution_metrics(self, execution_engine, mock_alpaca_client):
        """Test execution metrics tracking."""
        # Submit and fill multiple orders
        for i in range(3):
            mock_alpaca_client.submit_order.return_value = {
                'id': f'order{i}',
                'status': 'accepted'
            }
            
            order = Order('AAPL', OrderSide.BUY, 100, OrderType.MARKET)
            order_id = await execution_engine.submit_order(order)
            
            # Simulate fill
            await execution_engine._handle_fill({
                'order_id': order_id,
                'symbol': 'AAPL',
                'filled_qty': 100,
                'filled_avg_price': 150.0 + i,
                'side': 'buy'
            })
        
        metrics = execution_engine.get_execution_metrics()
        
        assert metrics['total_orders'] == 3
        assert metrics['filled_orders'] == 3
        assert metrics['fill_rate'] == 1.0
        assert metrics['avg_fill_time'] >= 0