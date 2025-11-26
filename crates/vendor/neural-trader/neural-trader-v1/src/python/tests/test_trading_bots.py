"""
Trading Bots Client Tests
=======================

Test suite for the trading bots client functionality.
"""

import pytest
import asyncio
from uuid import uuid4, UUID
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from supabase_client.clients.trading_bots import (
    TradingBotsClient,
    CreateBotRequest,
    PlaceOrderRequest,
    BotStatus,
    OrderSide,
    OrderType
)
from supabase_client.client import AsyncSupabaseClient

class TestTradingBotsClient:
    """Test suite for TradingBotsClient."""
    
    @pytest.fixture
    def mock_supabase(self):
        """Create mock Supabase client."""
        return AsyncMock(spec=AsyncSupabaseClient)
    
    @pytest.fixture
    def trading_client(self, mock_supabase):
        """Create trading bots client with mock."""
        return TradingBotsClient(mock_supabase)
    
    @pytest.fixture
    def sample_user_id(self):
        """Sample user ID for testing."""
        return uuid4()
    
    @pytest.fixture
    def sample_account_id(self):
        """Sample account ID for testing."""
        return uuid4()
    
    @pytest.fixture
    def sample_bot_data(self, sample_user_id, sample_account_id):
        """Sample bot data for testing."""
        return {
            "id": str(uuid4()),
            "user_id": str(sample_user_id),
            "account_id": str(sample_account_id),
            "name": "Test Bot",
            "strategy": "momentum",
            "status": BotStatus.STOPPED.value,
            "symbols": ["AAPL", "GOOGL"],
            "risk_params": {
                "max_position_size": 0.1,
                "stop_loss": 0.05,
                "take_profit": 0.15
            },
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
    
    @pytest.fixture
    def sample_account_data(self, sample_user_id, sample_account_id):
        """Sample account data for testing."""
        return {
            "id": str(sample_account_id),
            "user_id": str(sample_user_id),
            "account_name": "Test Account",
            "account_type": "demo",
            "broker": "test_broker",
            "balance": 10000.0,
            "is_active": True
        }
    
    @pytest.mark.asyncio
    async def test_create_bot_success(self, trading_client, mock_supabase, sample_user_id, sample_account_data):
        """Test successful bot creation."""
        # Arrange
        request = CreateBotRequest(
            name="Test Trading Bot",
            strategy="momentum",
            account_id=UUID(sample_account_data["id"]),
            symbols=["AAPL", "GOOGL"],
            risk_params={
                "max_position_size": 0.1,
                "stop_loss": 0.05
            }
        )
        
        mock_supabase.select.return_value = [sample_account_data]  # Account exists
        mock_supabase.count.return_value = 2  # Under limit
        mock_supabase.insert.return_value = [{"id": "bot-123", "name": "Test Trading Bot"}]
        
        # Act
        result, error = await trading_client.create_bot(sample_user_id, request)
        
        # Assert
        assert error is None
        assert result is not None
        assert result["name"] == "Test Trading Bot"
        mock_supabase.insert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_bot_account_not_found(self, trading_client, mock_supabase, sample_user_id, sample_account_id):
        """Test bot creation with non-existent account."""
        # Arrange
        request = CreateBotRequest(
            name="Test Bot",
            strategy="momentum",
            account_id=sample_account_id,
            symbols=["AAPL"]
        )
        
        mock_supabase.select.return_value = []  # Account not found
        
        # Act
        result, error = await trading_client.create_bot(sample_user_id, request)
        
        # Assert
        assert result is None
        assert error == "Account not found or not owned by user"
        mock_supabase.insert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_start_bot_success(self, trading_client, mock_supabase, sample_bot_data):
        """Test successful bot start."""
        # Arrange
        bot_id = sample_bot_data["id"]
        mock_supabase.select.return_value = [sample_bot_data]  # Bot exists
        updated_bot = {**sample_bot_data, "status": BotStatus.RUNNING.value}
        mock_supabase.update.return_value = [updated_bot]
        mock_supabase.insert.return_value = [{}]  # Log entry
        
        # Act
        success, error = await trading_client.start_bot(bot_id)
        
        # Assert
        assert error is None
        assert success is True
        mock_supabase.update.assert_called_once()
        mock_supabase.insert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_bot_not_found(self, trading_client, mock_supabase):
        """Test bot start with non-existent bot."""
        # Arrange
        bot_id = str(uuid4())
        mock_supabase.select.return_value = []  # Bot not found
        
        # Act
        success, error = await trading_client.start_bot(bot_id)
        
        # Assert
        assert success is False
        assert error == "Bot not found"
        mock_supabase.update.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_stop_bot_success(self, trading_client, mock_supabase, sample_bot_data):
        """Test successful bot stop."""
        # Arrange
        bot_id = sample_bot_data["id"]
        running_bot = {**sample_bot_data, "status": BotStatus.RUNNING.value}
        mock_supabase.select.return_value = [running_bot]
        
        stopped_bot = {**running_bot, "status": BotStatus.STOPPED.value}
        mock_supabase.update.return_value = [stopped_bot]
        mock_supabase.insert.return_value = [{}]  # Log entry
        
        # Act
        success, error = await trading_client.stop_bot(bot_id)
        
        # Assert
        assert error is None
        assert success is True
        mock_supabase.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, trading_client, mock_supabase, sample_bot_data, sample_account_data):
        """Test successful order placement."""
        # Arrange
        request = PlaceOrderRequest(
            bot_id=sample_bot_data["id"],
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10
        )
        
        # Mock bot and account exist
        mock_supabase.select.side_effect = [
            [sample_bot_data],  # Bot exists
            [sample_account_data]  # Account exists
        ]
        
        mock_supabase.insert.return_value = [{"id": "order-123", "status": "pending"}]
        
        # Act
        result, error = await trading_client.place_order(request)
        
        # Assert
        assert error is None
        assert result is not None
        mock_supabase.insert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_place_order_bot_not_running(self, trading_client, mock_supabase, sample_bot_data):
        """Test order placement with stopped bot."""
        # Arrange
        request = PlaceOrderRequest(
            bot_id=sample_bot_data["id"],
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10
        )
        
        # Bot exists but is stopped
        mock_supabase.select.return_value = [sample_bot_data]
        
        # Act
        result, error = await trading_client.place_order(request)
        
        # Assert
        assert result is None
        assert "Bot is not running" in error
        mock_supabase.insert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_bot_status(self, trading_client, mock_supabase, sample_bot_data):
        """Test bot status retrieval."""
        # Arrange
        bot_id = sample_bot_data["id"]
        mock_supabase.select.side_effect = [
            [sample_bot_data],  # Bot data
            [],  # Recent orders
            []   # Recent executions
        ]
        
        # Act
        result, error = await trading_client.get_bot_status(bot_id)
        
        # Assert
        assert error is None
        assert result is not None
        assert result["bot"]["id"] == bot_id
        assert "recent_orders" in result
        assert "recent_executions" in result
    
    @pytest.mark.asyncio
    async def test_list_user_bots(self, trading_client, mock_supabase, sample_user_id, sample_bot_data):
        """Test listing user bots."""
        # Arrange
        bots_data = [sample_bot_data, {**sample_bot_data, "id": str(uuid4()), "name": "Bot 2"}]
        mock_supabase.select.return_value = bots_data
        
        # Act
        result, error = await trading_client.list_user_bots(sample_user_id)
        
        # Assert
        assert error is None
        assert result is not None
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_calculate_bot_performance(self, trading_client, mock_supabase, sample_bot_data):
        """Test bot performance calculation."""
        # Arrange
        bot_id = sample_bot_data["id"]
        
        # Mock executions data
        executions_data = [
            {
                "id": "exec-1",
                "side": "buy",
                "quantity": 10,
                "price": 150.0,
                "executed_at": (datetime.utcnow() - timedelta(hours=1)).isoformat()
            },
            {
                "id": "exec-2", 
                "side": "sell",
                "quantity": 10,
                "price": 155.0,
                "executed_at": datetime.utcnow().isoformat()
            }
        ]
        
        mock_supabase.select.side_effect = [
            [sample_bot_data],  # Bot exists
            executions_data     # Executions
        ]
        
        # Act
        result, error = await trading_client.calculate_bot_performance(bot_id)
        
        # Assert
        assert error is None
        assert result is not None
        assert "total_pnl" in result
        assert "win_rate" in result
        assert "total_trades" in result
    
    @pytest.mark.asyncio
    async def test_update_bot_risk_params(self, trading_client, mock_supabase, sample_bot_data):
        """Test updating bot risk parameters."""
        # Arrange
        bot_id = sample_bot_data["id"]
        new_risk_params = {
            "max_position_size": 0.2,
            "stop_loss": 0.03,
            "take_profit": 0.1
        }
        
        mock_supabase.select.return_value = [sample_bot_data]  # Bot exists
        updated_bot = {**sample_bot_data, "risk_params": new_risk_params}
        mock_supabase.update.return_value = [updated_bot]
        
        # Act
        result, error = await trading_client.update_bot_risk_params(bot_id, new_risk_params)
        
        # Assert
        assert error is None
        assert result is not None
        assert result["risk_params"] == new_risk_params
        mock_supabase.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_bot_orders(self, trading_client, mock_supabase, sample_bot_data):
        """Test getting bot orders."""
        # Arrange
        bot_id = sample_bot_data["id"]
        orders_data = [
            {
                "id": "order-1",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10,
                "status": "filled"
            },
            {
                "id": "order-2",
                "symbol": "GOOGL", 
                "side": "sell",
                "quantity": 5,
                "status": "pending"
            }
        ]
        
        mock_supabase.select.side_effect = [
            [sample_bot_data],  # Bot exists
            orders_data         # Orders
        ]
        
        # Act
        result, error = await trading_client.get_bot_orders(bot_id)
        
        # Assert
        assert error is None
        assert result is not None
        assert len(result) == 2
        assert all(order["bot_id"] == bot_id for order in result if "bot_id" in order)
    
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, trading_client, mock_supabase):
        """Test successful order cancellation."""
        # Arrange
        order_id = str(uuid4())
        order_data = {
            "id": order_id,
            "status": "pending",
            "bot_id": str(uuid4())
        }
        
        mock_supabase.select.return_value = [order_data]  # Order exists and is pending
        mock_supabase.update.return_value = [{"id": order_id, "status": "cancelled"}]
        
        # Act
        success, error = await trading_client.cancel_order(order_id)
        
        # Assert
        assert error is None
        assert success is True
        mock_supabase.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cancel_order_not_pending(self, trading_client, mock_supabase):
        """Test cancelling non-pending order."""
        # Arrange
        order_id = str(uuid4())
        order_data = {
            "id": order_id,
            "status": "filled",
            "bot_id": str(uuid4())
        }
        
        mock_supabase.select.return_value = [order_data]  # Order exists but filled
        
        # Act
        success, error = await trading_client.cancel_order(order_id)
        
        # Assert
        assert success is False
        assert "Cannot cancel order" in error
        mock_supabase.update.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_delete_bot_success(self, trading_client, mock_supabase, sample_bot_data):
        """Test successful bot deletion."""
        # Arrange
        bot_id = sample_bot_data["id"]
        mock_supabase.select.return_value = [sample_bot_data]  # Bot exists
        mock_supabase.update.return_value = [{}]  # Soft delete
        
        # Act
        success, error = await trading_client.delete_bot(bot_id)
        
        # Assert
        assert error is None
        assert success is True
        mock_supabase.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, trading_client, mock_supabase, sample_user_id, sample_account_id):
        """Test exception handling in client methods."""
        # Arrange
        request = CreateBotRequest(
            name="Test Bot",
            strategy="momentum",
            account_id=sample_account_id,
            symbols=["AAPL"]
        )
        
        mock_supabase.select.side_effect = Exception("Database error")
        
        # Act
        result, error = await trading_client.create_bot(sample_user_id, request)
        
        # Assert
        assert result is None
        assert "Bot creation failed" in error
        assert "Database error" in error

@pytest.mark.asyncio
async def test_trading_bots_integration():
    """Integration test for trading bots workflow."""
    # This would be a higher-level test that exercises multiple methods
    # in a realistic trading workflow scenario
    pass

if __name__ == "__main__":
    pytest.main([__file__])