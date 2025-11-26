"""
Polymarket MCP Integration Tests

Following TDD principles - these tests validate the Model Context Protocol (MCP)
integration for Polymarket trading tools in Claude Code.
All tests should fail initially until MCP tools are implemented.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock


class TestMCPToolRegistration:
    """Test MCP tool registration and discovery."""
    
    @pytest.mark.unit
    @pytest.mark.mcp
    def test_mcp_server_initialization(self):
        """Test MCP server initialization with Polymarket tools."""
        from src.polymarket.mcp import PolymarketMCPServer
        
        server = PolymarketMCPServer()
        
        assert server.name == "polymarket-trader"
        assert server.version == "1.0.0"
        assert len(server.tools) > 0
        
        # Verify required tools are registered
        tool_names = [tool.name for tool in server.tools]
        required_tools = [
            "mcp__polymarket__get_markets",
            "mcp__polymarket__place_order",
            "mcp__polymarket__cancel_order",
            "mcp__polymarket__get_positions",
            "mcp__polymarket__analyze_market",
            "mcp__polymarket__get_sentiment",
            "mcp__polymarket__run_strategy",
            "mcp__polymarket__backtest_strategy"
        ]
        
        for tool_name in required_tools:
            assert tool_name in tool_names
    
    @pytest.mark.unit
    @pytest.mark.mcp
    def test_tool_schema_validation(self):
        """Test MCP tool schema validation."""
        from src.polymarket.mcp import PolymarketMCPServer
        
        server = PolymarketMCPServer()
        
        for tool in server.tools:
            # Each tool should have required MCP fields
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'parameters')
            
            # Parameters should be valid JSON schema
            params = tool.parameters
            assert 'type' in params
            assert params['type'] == 'object'
            
            if 'properties' in params:
                for prop_name, prop_schema in params['properties'].items():
                    assert 'type' in prop_schema
                    assert 'description' in prop_schema
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_tool_discovery(self):
        """Test MCP tool discovery endpoint."""
        from src.polymarket.mcp import PolymarketMCPServer
        
        server = PolymarketMCPServer()
        
        # Test list_tools request
        request = {
            "method": "tools/list",
            "params": {}
        }
        
        response = await server.handle_request(request)
        
        assert response["result"]["tools"] is not None
        assert len(response["result"]["tools"]) > 0
        
        # Each tool should have complete metadata
        for tool in response["result"]["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool


class TestMarketDataTools:
    """Test market data MCP tools."""
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_get_markets_tool(self, mock_api_responses):
        """Test get_markets MCP tool."""
        from src.polymarket.mcp.tools import get_markets
        
        # Mock API client
        mock_client = AsyncMock()
        mock_client.get_markets.return_value = mock_api_responses["markets"]
        
        # Test basic call
        result = await get_markets(api_client=mock_client)
        
        assert "markets" in result
        assert len(result["markets"]) == 2
        assert result["status"] == "success"
        
        # Test with filters
        result_filtered = await get_markets(
            api_client=mock_client,
            status="active",
            tag="crypto",
            limit=10
        )
        
        mock_client.get_markets.assert_called_with(
            status="active",
            tag="crypto",
            limit=10
        )
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_get_market_tool(self, mock_market_data):
        """Test get_market MCP tool for specific market."""
        from src.polymarket.mcp.tools import get_market
        
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market_data
        
        market_id = "0x" + "a" * 40
        result = await get_market(api_client=mock_client, market_id=market_id)
        
        assert result["market_id"] == market_id
        assert "question" in result
        assert "outcomes" in result
        assert "order_book" in result
        assert result["status"] == "success"
        
        # Test invalid market ID
        mock_client.get_market.side_effect = Exception("Market not found")
        
        result_error = await get_market(api_client=mock_client, market_id="invalid_id")
        
        assert result_error["status"] == "error"
        assert "error" in result_error
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_get_order_book_tool(self, mock_market_data):
        """Test get_order_book MCP tool."""
        from src.polymarket.mcp.tools import get_order_book
        
        mock_client = AsyncMock()
        mock_client.get_order_book.return_value = mock_market_data["order_book"]
        
        market_id = "0x" + "a" * 40
        result = await get_order_book(
            api_client=mock_client,
            market_id=market_id,
            outcome="Yes"
        )
        
        assert "bids" in result
        assert "asks" in result
        assert len(result["bids"]) > 0
        assert len(result["asks"]) > 0
        assert result["status"] == "success"
        
        # Verify bid/ask ordering
        bids = result["bids"]
        asks = result["asks"]
        
        # Bids should be sorted descending
        for i in range(len(bids) - 1):
            assert bids[i]["price"] >= bids[i + 1]["price"]
        
        # Asks should be sorted ascending
        for i in range(len(asks) - 1):
            assert asks[i]["price"] <= asks[i + 1]["price"]


class TestTradingTools:
    """Test trading MCP tools."""
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_place_order_tool(self, mock_api_responses):
        """Test place_order MCP tool."""
        from src.polymarket.mcp.tools import place_order
        
        mock_client = AsyncMock()
        mock_client.place_order.return_value = mock_api_responses["order_created"]
        
        order_params = {
            "market_id": "0x" + "a" * 40,
            "side": "buy",
            "outcome": "Yes",
            "size": 100,
            "price": 0.65,
            "order_type": "limit"
        }
        
        result = await place_order(api_client=mock_client, **order_params)
        
        assert result["status"] == "success"
        assert "order_id" in result
        assert result["order_status"] == "pending"
        
        # Verify API call parameters
        mock_client.place_order.assert_called_once()
        call_args = mock_client.place_order.call_args
        assert call_args[1]["market_id"] == order_params["market_id"]
        assert call_args[1]["side"] == order_params["side"]
        assert float(call_args[1]["size"]) == order_params["size"]
        assert float(call_args[1]["price"]) == order_params["price"]
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_place_order_validation(self):
        """Test place_order input validation."""
        from src.polymarket.mcp.tools import place_order
        
        mock_client = AsyncMock()
        
        # Test invalid price (> 1.0)
        invalid_order = {
            "market_id": "0x123",
            "side": "buy",
            "outcome": "Yes",
            "size": 100,
            "price": 1.5,  # Invalid
            "order_type": "limit"
        }
        
        result = await place_order(api_client=mock_client, **invalid_order)
        
        assert result["status"] == "error"
        assert "price must be between 0 and 1" in result["error"].lower()
        
        # Test invalid size (negative)
        invalid_order["price"] = 0.65
        invalid_order["size"] = -100
        
        result = await place_order(api_client=mock_client, **invalid_order)
        
        assert result["status"] == "error"
        assert "size must be positive" in result["error"].lower()
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_cancel_order_tool(self, mock_api_responses):
        """Test cancel_order MCP tool."""
        from src.polymarket.mcp.tools import cancel_order
        
        mock_client = AsyncMock()
        mock_client.cancel_order.return_value = mock_api_responses["order_canceled"]
        
        order_id = "order_123"
        result = await cancel_order(api_client=mock_client, order_id=order_id)
        
        assert result["status"] == "success"
        assert result["order_id"] == order_id
        assert result["order_status"] == "canceled"
        
        mock_client.cancel_order.assert_called_once_with(order_id)
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_get_positions_tool(self, mock_position_data):
        """Test get_positions MCP tool."""
        from src.polymarket.mcp.tools import get_positions
        
        mock_client = AsyncMock()
        mock_client.get_positions.return_value = [mock_position_data]
        
        result = await get_positions(api_client=mock_client)
        
        assert result["status"] == "success"
        assert "positions" in result
        assert len(result["positions"]) == 1
        
        position = result["positions"][0]
        assert "market_id" in position
        assert "unrealized_pnl" in position
        assert "realized_pnl" in position
        
        # Test with market filter
        await get_positions(api_client=mock_client, market_id="0x123")
        mock_client.get_positions.assert_called_with(market_id="0x123")
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_get_open_orders_tool(self):
        """Test get_open_orders MCP tool."""
        from src.polymarket.mcp.tools import get_open_orders
        
        mock_orders = [
            {"order_id": "order_1", "status": "open", "market_id": "0x123"},
            {"order_id": "order_2", "status": "open", "market_id": "0x456"}
        ]
        
        mock_client = AsyncMock()
        mock_client.get_open_orders.return_value = mock_orders
        
        result = await get_open_orders(api_client=mock_client)
        
        assert result["status"] == "success"
        assert "orders" in result
        assert len(result["orders"]) == 2
        assert all(order["status"] == "open" for order in result["orders"])


class TestAnalysisTools:
    """Test market analysis MCP tools."""
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_analyze_market_tool(self, mock_market_data):
        """Test analyze_market MCP tool."""
        from src.polymarket.mcp.tools import analyze_market
        
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market_data
        
        # Mock analysis results
        mock_analysis = {
            "sentiment_score": 0.75,
            "momentum": 0.12,
            "volatility": 0.18,
            "recommendation": "buy",
            "confidence": 0.82,
            "reasoning": "Strong bullish sentiment with positive momentum"
        }
        
        market_id = "0x" + "a" * 40
        
        with patch('src.polymarket.mcp.tools.MarketAnalyzer') as mock_analyzer:
            mock_analyzer.return_value.analyze.return_value = mock_analysis
            
            result = await analyze_market(api_client=mock_client, market_id=market_id)
        
        assert result["status"] == "success"
        assert result["market_id"] == market_id
        assert "sentiment_score" in result
        assert "recommendation" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_get_sentiment_tool(self):
        """Test get_sentiment MCP tool for news analysis."""
        from src.polymarket.mcp.tools import get_sentiment
        
        mock_client = AsyncMock()
        
        # Mock news and sentiment data
        mock_news = [
            {"title": "Bitcoin surges past key resistance", "sentiment": 0.8},
            {"title": "Crypto market shows strength", "sentiment": 0.6},
            {"title": "Regulatory concerns persist", "sentiment": -0.2}
        ]
        
        mock_client.get_market_news.return_value = mock_news
        
        market_id = "0x" + "a" * 40
        result = await get_sentiment(api_client=mock_client, market_id=market_id)
        
        assert result["status"] == "success"
        assert "overall_sentiment" in result
        assert "sentiment_breakdown" in result
        assert "news_count" in result
        assert -1 <= result["overall_sentiment"] <= 1
        assert result["news_count"] == 3
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_get_market_stats_tool(self, mock_market_data):
        """Test get_market_stats MCP tool."""
        from src.polymarket.mcp.tools import get_market_stats
        
        mock_client = AsyncMock()
        mock_client.get_market.return_value = mock_market_data
        mock_client.get_market_history.return_value = [
            {"timestamp": datetime.now() - timedelta(hours=i), "price": 0.60 + 0.01 * i}
            for i in range(24)
        ]
        
        market_id = "0x" + "a" * 40
        result = await get_market_stats(
            api_client=mock_client,
            market_id=market_id,
            timeframe="24h"
        )
        
        assert result["status"] == "success"
        assert "price_change_24h" in result
        assert "volume_24h" in result
        assert "high_24h" in result
        assert "low_24h" in result
        assert "volatility" in result


class TestStrategyTools:
    """Test strategy-related MCP tools."""
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_run_strategy_tool(self, mock_strategy_signals):
        """Test run_strategy MCP tool."""
        from src.polymarket.mcp.tools import run_strategy
        
        mock_client = AsyncMock()
        
        with patch('src.polymarket.strategies.SentimentStrategy') as mock_strategy:
            mock_strategy_instance = AsyncMock()
            mock_strategy_instance.generate_signals.return_value = mock_strategy_signals
            mock_strategy.return_value = mock_strategy_instance
            
            result = await run_strategy(
                api_client=mock_client,
                strategy_name="sentiment",
                market_ids=["0x123", "0x456"],
                config={"sentiment_threshold": 0.6}
            )
        
        assert result["status"] == "success"
        assert "signals" in result
        assert len(result["signals"]) == 2
        
        for signal in result["signals"]:
            assert "market_id" in signal
            assert "action" in signal
            assert "confidence" in signal
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_backtest_strategy_tool(self):
        """Test backtest_strategy MCP tool."""
        from src.polymarket.mcp.tools import backtest_strategy
        
        mock_client = AsyncMock()
        
        # Mock backtest results
        mock_backtest_results = {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08,
            "number_of_trades": 25,
            "win_rate": 0.64,
            "profit_factor": 1.8
        }
        
        with patch('src.polymarket.backtesting.Backtester') as mock_backtester:
            mock_backtester_instance = AsyncMock()
            mock_backtester_instance.run_backtest.return_value = mock_backtest_results
            mock_backtester.return_value = mock_backtester_instance
            
            result = await backtest_strategy(
                api_client=mock_client,
                strategy_name="momentum",
                start_date="2024-01-01",
                end_date="2024-03-01",
                initial_capital=10000,
                config={"momentum_threshold": 0.05}
            )
        
        assert result["status"] == "success"
        assert "performance" in result
        assert result["performance"]["total_return"] == 0.15
        assert result["performance"]["sharpe_ratio"] == 1.2
        assert result["performance"]["win_rate"] == 0.64
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_optimize_strategy_tool(self):
        """Test optimize_strategy MCP tool."""
        from src.polymarket.mcp.tools import optimize_strategy
        
        mock_client = AsyncMock()
        
        # Mock optimization results
        mock_optimization = {
            "best_parameters": {
                "sentiment_threshold": 0.65,
                "position_size": 0.12,
                "stop_loss": 0.08
            },
            "best_score": 1.45,
            "optimization_metric": "sharpe_ratio",
            "trials": 100
        }
        
        with patch('src.polymarket.optimization.StrategyOptimizer') as mock_optimizer:
            mock_optimizer_instance = AsyncMock()
            mock_optimizer_instance.optimize.return_value = mock_optimization
            mock_optimizer.return_value = mock_optimizer_instance
            
            result = await optimize_strategy(
                api_client=mock_client,
                strategy_name="sentiment",
                parameter_ranges={
                    "sentiment_threshold": [0.5, 0.8],
                    "position_size": [0.05, 0.20]
                },
                optimization_metric="sharpe_ratio",
                trials=50
            )
        
        assert result["status"] == "success"
        assert "best_parameters" in result
        assert "best_score" in result
        assert result["best_score"] == 1.45


class TestPortfolioTools:
    """Test portfolio management MCP tools."""
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_get_portfolio_summary_tool(self, mock_position_data):
        """Test get_portfolio_summary MCP tool."""
        from src.polymarket.mcp.tools import get_portfolio_summary
        
        mock_client = AsyncMock()
        mock_client.get_positions.return_value = [mock_position_data]
        mock_client.get_account_balance.return_value = {"balance": "5000.00"}
        
        result = await get_portfolio_summary(api_client=mock_client)
        
        assert result["status"] == "success"
        assert "total_value" in result
        assert "cash_balance" in result
        assert "positions_value" in result
        assert "total_pnl" in result
        assert "number_of_positions" in result
        assert result["number_of_positions"] == 1
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_calculate_portfolio_risk_tool(self, mock_position_data):
        """Test calculate_portfolio_risk MCP tool."""
        from src.polymarket.mcp.tools import calculate_portfolio_risk
        
        mock_client = AsyncMock()
        mock_client.get_positions.return_value = [mock_position_data]
        
        # Mock risk calculation
        mock_risk_metrics = {
            "value_at_risk": 250.0,
            "expected_shortfall": 380.0,
            "portfolio_volatility": 0.15,
            "beta": 0.8,
            "max_correlation": 0.65,
            "concentration_risk": 0.25
        }
        
        with patch('src.polymarket.risk.RiskCalculator') as mock_risk_calc:
            mock_risk_calc.return_value.calculate_portfolio_risk.return_value = mock_risk_metrics
            
            result = await calculate_portfolio_risk(
                api_client=mock_client,
                confidence_level=0.95,
                time_horizon=1
            )
        
        assert result["status"] == "success"
        assert "value_at_risk" in result
        assert "portfolio_volatility" in result
        assert "risk_score" in result
        assert 0 <= result["risk_score"] <= 1
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_rebalance_portfolio_tool(self, mock_position_data):
        """Test rebalance_portfolio MCP tool."""
        from src.polymarket.mcp.tools import rebalance_portfolio
        
        mock_client = AsyncMock()
        mock_client.get_positions.return_value = [mock_position_data]
        
        target_allocation = {
            "0x" + "a" * 40: 0.6,
            "0x" + "b" * 40: 0.4
        }
        
        # Mock rebalancing actions
        mock_actions = [
            {"action": "sell", "market_id": "0x" + "a" * 40, "size": 50, "reason": "overweight"},
            {"action": "buy", "market_id": "0x" + "b" * 40, "size": 30, "reason": "underweight"}
        ]
        
        with patch('src.polymarket.strategies.PortfolioOptimizer') as mock_optimizer:
            mock_optimizer.return_value.calculate_rebalancing.return_value = mock_actions
            
            result = await rebalance_portfolio(
                api_client=mock_client,
                target_allocation=target_allocation,
                threshold=0.05
            )
        
        assert result["status"] == "success"
        assert "rebalance_actions" in result
        assert len(result["rebalance_actions"]) == 2
        assert result["rebalance_actions"][0]["action"] == "sell"
        assert result["rebalance_actions"][1]["action"] == "buy"


class TestMCPErrorHandling:
    """Test MCP error handling and edge cases."""
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_api_error_handling(self):
        """Test MCP tool error handling for API failures."""
        from src.polymarket.mcp.tools import get_markets
        
        # Mock API client that raises exception
        mock_client = AsyncMock()
        mock_client.get_markets.side_effect = Exception("API connection failed")
        
        result = await get_markets(api_client=mock_client)
        
        assert result["status"] == "error"
        assert "error" in result
        assert "API connection failed" in result["error"]
        assert "timestamp" in result
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_rate_limit_handling(self):
        """Test MCP tool handling of rate limits."""
        from src.polymarket.mcp.tools import place_order
        from src.polymarket.exceptions import RateLimitError
        
        mock_client = AsyncMock()
        mock_client.place_order.side_effect = RateLimitError("Rate limit exceeded")
        
        result = await place_order(
            api_client=mock_client,
            market_id="0x123",
            side="buy",
            outcome="Yes",
            size=100,
            price=0.65
        )
        
        assert result["status"] == "error"
        assert result["error_type"] == "rate_limit"
        assert "rate limit" in result["error"].lower()
        assert "retry_after" in result
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_validation_error_handling(self):
        """Test MCP tool input validation error handling."""
        from src.polymarket.mcp.tools import place_order
        
        mock_client = AsyncMock()
        
        # Test missing required parameter
        result = await place_order(
            api_client=mock_client,
            market_id="0x123",
            # Missing required parameters
        )
        
        assert result["status"] == "error"
        assert result["error_type"] == "validation"
        assert "required parameter" in result["error"].lower()
    
    @pytest.mark.unit
    @pytest.mark.mcp
    async def test_timeout_handling(self):
        """Test MCP tool timeout handling."""
        from src.polymarket.mcp.tools import analyze_market
        
        mock_client = AsyncMock()
        
        # Mock timeout exception
        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate slow operation
        
        mock_client.get_market.side_effect = slow_operation
        
        # Tool should timeout and return error
        result = await analyze_market(
            api_client=mock_client,
            market_id="0x123",
            timeout=1  # 1 second timeout
        )
        
        assert result["status"] == "error"
        assert result["error_type"] == "timeout"
        assert "timeout" in result["error"].lower()


class TestMCPIntegrationEnd2End:
    """Test end-to-end MCP integration scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.mcp
    async def test_complete_trading_workflow(self, mock_api_responses, mock_market_data):
        """Test complete trading workflow through MCP tools."""
        from src.polymarket.mcp.tools import (
            get_markets, analyze_market, place_order, get_positions
        )
        
        mock_client = AsyncMock()
        mock_client.get_markets.return_value = mock_api_responses["markets"]
        mock_client.get_market.return_value = mock_market_data
        mock_client.place_order.return_value = mock_api_responses["order_created"]
        mock_client.get_positions.return_value = []
        
        # Step 1: Get available markets
        markets_result = await get_markets(api_client=mock_client, status="active")
        assert markets_result["status"] == "success"
        
        # Step 2: Analyze a specific market
        market_id = markets_result["markets"][0]["id"]
        
        with patch('src.polymarket.mcp.tools.MarketAnalyzer') as mock_analyzer:
            mock_analyzer.return_value.analyze.return_value = {
                "recommendation": "buy",
                "confidence": 0.85,
                "reasoning": "Strong bullish signals"
            }
            
            analysis_result = await analyze_market(api_client=mock_client, market_id=market_id)
        
        assert analysis_result["status"] == "success"
        assert analysis_result["recommendation"] == "buy"
        
        # Step 3: Place order based on analysis
        if analysis_result["recommendation"] == "buy" and analysis_result["confidence"] > 0.8:
            order_result = await place_order(
                api_client=mock_client,
                market_id=market_id,
                side="buy",
                outcome="Yes",
                size=100,
                price=0.65
            )
            
            assert order_result["status"] == "success"
            assert "order_id" in order_result
        
        # Step 4: Check positions
        positions_result = await get_positions(api_client=mock_client)
        assert positions_result["status"] == "success"
    
    @pytest.mark.integration
    @pytest.mark.mcp
    async def test_strategy_backtesting_workflow(self):
        """Test strategy backtesting workflow through MCP tools."""
        from src.polymarket.mcp.tools import (
            backtest_strategy, optimize_strategy, run_strategy
        )
        
        mock_client = AsyncMock()
        
        # Mock backtest results
        with patch('src.polymarket.backtesting.Backtester') as mock_backtester:
            mock_backtester.return_value.run_backtest.return_value = {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08
            }
            
            # Step 1: Backtest strategy
            backtest_result = await backtest_strategy(
                api_client=mock_client,
                strategy_name="sentiment",
                start_date="2024-01-01",
                end_date="2024-03-01",
                initial_capital=10000
            )
        
        assert backtest_result["status"] == "success"
        assert backtest_result["performance"]["sharpe_ratio"] == 1.2
        
        # Step 2: Optimize strategy if performance is good
        if backtest_result["performance"]["sharpe_ratio"] > 1.0:
            with patch('src.polymarket.optimization.StrategyOptimizer') as mock_optimizer:
                mock_optimizer.return_value.optimize.return_value = {
                    "best_parameters": {"sentiment_threshold": 0.65},
                    "best_score": 1.45
                }
                
                optimize_result = await optimize_strategy(
                    api_client=mock_client,
                    strategy_name="sentiment",
                    parameter_ranges={"sentiment_threshold": [0.5, 0.8]},
                    trials=50
                )
            
            assert optimize_result["status"] == "success"
            assert optimize_result["best_score"] > backtest_result["performance"]["sharpe_ratio"]
        
        # Step 3: Run optimized strategy
        with patch('src.polymarket.strategies.SentimentStrategy') as mock_strategy:
            mock_strategy.return_value.generate_signals.return_value = [
                {"market_id": "0x123", "action": "buy", "confidence": 0.85}
            ]
            
            signals_result = await run_strategy(
                api_client=mock_client,
                strategy_name="sentiment",
                market_ids=["0x123"],
                config=optimize_result["best_parameters"]
            )
        
        assert signals_result["status"] == "success"
        assert len(signals_result["signals"]) > 0
    
    @pytest.mark.slow
    @pytest.mark.mcp
    async def test_concurrent_mcp_calls(self, mock_api_responses):
        """Test concurrent MCP tool calls."""
        from src.polymarket.mcp.tools import get_markets, get_positions, analyze_market
        
        mock_client = AsyncMock()
        mock_client.get_markets.return_value = mock_api_responses["markets"]
        mock_client.get_positions.return_value = []
        
        with patch('src.polymarket.mcp.tools.MarketAnalyzer') as mock_analyzer:
            mock_analyzer.return_value.analyze.return_value = {
                "recommendation": "hold",
                "confidence": 0.5
            }
            
            # Launch multiple concurrent MCP calls
            tasks = [
                get_markets(api_client=mock_client),
                get_positions(api_client=mock_client),
                analyze_market(api_client=mock_client, market_id="0x123"),
                analyze_market(api_client=mock_client, market_id="0x456")
            ]
            
            results = await asyncio.gather(*tasks)
        
        # All calls should succeed
        assert len(results) == 4
        assert all(result["status"] == "success" for result in results)
        
        # Should respect API rate limits
        total_api_calls = (
            mock_client.get_markets.call_count +
            mock_client.get_positions.call_count +
            mock_client.get_market.call_count
        )
        assert total_api_calls >= 4  # At least one call per task