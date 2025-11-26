"""
MCP Server Tool Integration Tests

This module tests the complete integration of Polymarket functionality
with the MCP (Model Context Protocol) server, ensuring all 6 Polymarket
tools work correctly with proper error handling and GPU acceleration.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
import numpy as np

# Import MCP server components
import sys
sys.path.append('/workspaces/ai-news-trader/src')

from mcp.mcp_server_enhanced import (
    PolymarketTools, get_prediction_markets_tool,
    analyze_market_sentiment_tool, get_market_orderbook_tool,
    place_prediction_order_tool, get_prediction_positions_tool,
    calculate_expected_value_tool
)
from polymarket.api import PolymarketClient
from polymarket.models import Market, Order, Position, OrderSide, OrderType
from polymarket.strategies import SentimentCorrelationStrategy
from polymarket.utils.config import PolymarketConfig


class TestMCPIntegration:
    """Test MCP server integration with Polymarket tools."""

    @pytest.fixture
    def polymarket_tools(self):
        """Create PolymarketTools instance with mocked client."""
        tools = PolymarketTools()
        tools.client = AsyncMock(spec=PolymarketClient)
        tools.sentiment_strategy = AsyncMock(spec=SentimentCorrelationStrategy)
        return tools

    @pytest.fixture
    def sample_markets(self):
        """Generate sample market data."""
        return [
            Market(
                id="crypto_btc_100k",
                question="Will Bitcoin reach $100k in 2024?",
                category="Crypto",
                outcomes=["Yes", "No"],
                outcome_prices=[Decimal("0.65"), Decimal("0.35")],
                volume_24h=Decimal("250000"),
                liquidity=Decimal("100000"),
                num_traders=1500
            ),
            Market(
                id="politics_election_2024",
                question="Will Biden win the 2024 election?",
                category="Politics",
                outcomes=["Yes", "No"],
                outcome_prices=[Decimal("0.45"), Decimal("0.55")],
                volume_24h=Decimal("500000"),
                liquidity=Decimal("200000"),
                num_traders=3000
            ),
            Market(
                id="sports_superbowl",
                question="Will the Chiefs win the Super Bowl?",
                category="Sports",
                outcomes=["Yes", "No"],
                outcome_prices=[Decimal("0.30"), Decimal("0.70")],
                volume_24h=Decimal("150000"),
                liquidity=Decimal("75000"),
                num_traders=800
            )
        ]

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_get_prediction_markets_tool(self, polymarket_tools, sample_markets):
        """Test market discovery tool with filtering and sorting."""
        polymarket_tools.client.get_markets.return_value = sample_markets
        
        # Test basic market listing
        result = await get_prediction_markets_tool(
            category=None,
            sort_by="volume",
            limit=10
        )
        
        assert result["status"] == "success"
        assert len(result["markets"]) == 3
        assert result["markets"][0]["volume_24h"] == "500000.00"  # Politics highest volume
        
        # Test category filtering
        polymarket_tools.client.get_markets.return_value = [sample_markets[0]]
        result = await get_prediction_markets_tool(
            category="Crypto",
            sort_by="liquidity",
            limit=5
        )
        
        assert len(result["markets"]) == 1
        assert result["markets"][0]["category"] == "Crypto"
        
        # Test different sort options
        for sort_by in ["volume", "liquidity", "participants", "newest"]:
            result = await get_prediction_markets_tool(sort_by=sort_by)
            assert result["status"] == "success"

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_analyze_market_sentiment_tool(self, polymarket_tools):
        """Test market sentiment analysis with GPU acceleration."""
        # Mock market data
        market = Market(
            id="test_market",
            question="Test market?",
            outcomes=["Yes", "No"],
            outcome_prices=[Decimal("0.75"), Decimal("0.25")],
            volume_24h=Decimal("100000")
        )
        polymarket_tools.client.get_market.return_value = market
        
        # Mock orderbook
        polymarket_tools.client.get_orderbook.return_value = {
            "bids": [{"price": "0.74", "size": "1000"}],
            "asks": [{"price": "0.76", "size": "800"}]
        }
        
        # Mock sentiment strategy
        polymarket_tools.sentiment_strategy.analyze_market.return_value = MagicMock(
            confidence=0.85,
            metadata={"sentiment_score": 0.8}
        )
        
        # Test standard analysis
        result = await analyze_market_sentiment_tool(
            market_id="test_market",
            analysis_depth="standard",
            include_correlations=True,
            use_gpu=False
        )
        
        assert result["status"] == "success"
        assert result["analysis"]["current_probabilities"]["Yes"] == 0.75
        assert result["analysis"]["sentiment_indicators"]["momentum"] > 0
        assert "market_efficiency" in result["analysis"]
        
        # Test GPU-enhanced analysis
        with patch('torch.cuda.is_available', return_value=True):
            result = await analyze_market_sentiment_tool(
                market_id="test_market",
                analysis_depth="gpu_enhanced",
                use_gpu=True
            )
            
            assert result["processing"]["gpu_acceleration"] is True
            assert "monte_carlo_confidence" in result["analysis"]["advanced_metrics"]
            assert "kelly_criterion" in result["analysis"]["advanced_metrics"]

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_get_market_orderbook_tool(self, polymarket_tools):
        """Test orderbook data retrieval and analysis."""
        # Mock detailed orderbook
        mock_orderbook = {
            "market_id": "test_market",
            "bids": [
                {"price": "0.50", "size": "1000"},
                {"price": "0.49", "size": "1500"},
                {"price": "0.48", "size": "2000"}
            ],
            "asks": [
                {"price": "0.51", "size": "800"},
                {"price": "0.52", "size": "1200"},
                {"price": "0.53", "size": "1800"}
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        polymarket_tools.client.get_orderbook.return_value = mock_orderbook
        polymarket_tools.client.get_trades.return_value = [
            {"price": "0.505", "size": "100", "timestamp": datetime.now().isoformat()}
        ]
        
        result = await get_market_orderbook_tool(
            market_id="test_market",
            depth=10
        )
        
        assert result["status"] == "success"
        assert len(result["orderbook"]["Yes"]["bids"]) == 3
        assert len(result["orderbook"]["Yes"]["asks"]) == 3
        assert result["orderbook"]["Yes"]["spread"] == 0.01
        assert result["liquidity"]["total_bid_size"] == 4500
        assert result["liquidity"]["total_ask_size"] == 3800

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_place_prediction_order_tool(self, polymarket_tools):
        """Test order placement with validation and risk checks."""
        # Mock successful order placement
        mock_order = Order(
            id="order_123",
            market_id="test_market",
            outcome="Yes",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal("100"),
            price=Decimal("0.65"),
            status="OPEN"
        )
        
        polymarket_tools.client.place_order.return_value = mock_order
        
        # Mock current positions
        polymarket_tools.positions = {}
        
        # Test market order
        result = await place_prediction_order_tool(
            market_id="test_market",
            outcome="Yes",
            side="buy",
            quantity=100,
            order_type="market",
            limit_price=None
        )
        
        assert result["status"] == "success"
        assert result["order"]["id"] == "order_123"
        assert result["demo_mode"] is True
        assert "test_market" in polymarket_tools.positions
        
        # Test limit order
        result = await place_prediction_order_tool(
            market_id="test_market",
            outcome="No",
            side="sell",
            quantity=50,
            order_type="limit",
            limit_price=0.70
        )
        
        assert result["status"] == "success"
        polymarket_tools.client.place_order.assert_called_with(
            market_id="test_market",
            outcome="No",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            size=Decimal("50"),
            price=Decimal("0.70")
        )

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_get_prediction_positions_tool(self, polymarket_tools):
        """Test position tracking and portfolio analysis."""
        # Set up test positions
        polymarket_tools.positions = {
            "market_1": {
                "market_id": "market_1",
                "outcome": "Yes",
                "shares": 100,
                "average_price": 0.60,
                "current_price": 0.70,
                "invested": 60.0,
                "current_value": 70.0,
                "pnl": 10.0,
                "pnl_percentage": 16.67,
                "timestamp": datetime.now().isoformat()
            },
            "market_2": {
                "market_id": "market_2",
                "outcome": "No",
                "shares": 200,
                "average_price": 0.40,
                "current_price": 0.35,
                "invested": 80.0,
                "current_value": 70.0,
                "pnl": -10.0,
                "pnl_percentage": -12.5,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        result = await get_prediction_positions_tool()
        
        assert result["status"] == "success"
        assert len(result["positions"]) == 2
        assert result["summary"]["total_invested"] == 140.0
        assert result["summary"]["total_value"] == 140.0
        assert result["summary"]["total_pnl"] == 0.0
        assert result["summary"]["total_return_percentage"] == 0.0
        
        # Verify risk metrics
        assert "risk_metrics" in result
        assert result["risk_metrics"]["largest_position_percentage"] == 80/140  # market_2
        assert result["risk_metrics"]["winning_positions"] == 1
        assert result["risk_metrics"]["losing_positions"] == 1

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_calculate_expected_value_tool(self, polymarket_tools):
        """Test EV calculation with Kelly Criterion and Monte Carlo."""
        # Mock market data
        market = Market(
            id="test_market",
            outcomes=["Yes", "No"],
            outcome_prices=[Decimal("0.30"), Decimal("0.70")]
        )
        polymarket_tools.client.get_market.return_value = market
        
        # Test basic EV calculation
        result = await calculate_expected_value_tool(
            market_id="test_market",
            investment_amount=1000.0,
            confidence_adjustment=1.2,  # User thinks Yes is underpriced
            include_fees=True,
            use_gpu=False
        )
        
        assert result["status"] == "success"
        
        # Check EV calculations
        yes_ev = result["expected_values"]["Yes"]
        assert yes_ev["probability"] == 0.30
        assert yes_ev["adjusted_probability"] == 0.36  # 0.30 * 1.2
        assert yes_ev["expected_value"] > 0  # Positive EV due to adjustment
        
        # Verify Kelly Criterion
        assert "kelly_criterion" in result
        assert result["kelly_criterion"]["recommended_percentage"] > 0
        assert result["kelly_criterion"]["recommended_amount"] <= 1000.0
        
        # Test GPU-accelerated Monte Carlo
        with patch('torch.cuda.is_available', return_value=True):
            result = await calculate_expected_value_tool(
                market_id="test_market",
                investment_amount=1000.0,
                confidence_adjustment=1.0,
                use_gpu=True
            )
            
            assert result["processing"]["gpu_acceleration"] is True
            assert "confidence_interval" in result["monte_carlo"]
            assert result["monte_carlo"]["simulations"] == 100000

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_mcp_error_handling(self, polymarket_tools):
        """Test error handling across all MCP tools."""
        # Test network error handling
        polymarket_tools.client.get_markets.side_effect = Exception("Network error")
        
        result = await get_prediction_markets_tool()
        assert result["status"] == "error"
        assert "Network error" in result["error"]
        
        # Test invalid market ID
        polymarket_tools.client.get_market.side_effect = Exception("Market not found")
        
        result = await analyze_market_sentiment_tool(
            market_id="invalid_id",
            analysis_depth="standard"
        )
        assert result["status"] == "error"
        assert "Market not found" in result["error"]
        
        # Test order placement failure
        polymarket_tools.client.place_order.side_effect = Exception("Insufficient balance")
        
        result = await place_prediction_order_tool(
            market_id="test",
            outcome="Yes",
            side="buy",
            quantity=1000
        )
        assert result["status"] == "error"
        assert "Insufficient balance" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_mcp_data_flow_integration(self, polymarket_tools, sample_markets):
        """Test complete data flow from external APIs to MCP responses."""
        # Set up mock data flow
        polymarket_tools.client.get_markets.return_value = sample_markets
        
        # Step 1: Discover markets
        markets_result = await get_prediction_markets_tool(
            category="Crypto",
            sort_by="volume"
        )
        
        market_id = markets_result["markets"][0]["id"]
        
        # Step 2: Analyze sentiment
        polymarket_tools.client.get_market.return_value = sample_markets[0]
        polymarket_tools.client.get_orderbook.return_value = {
            "bids": [{"price": "0.64", "size": "500"}],
            "asks": [{"price": "0.66", "size": "400"}]
        }
        
        sentiment_result = await analyze_market_sentiment_tool(
            market_id=market_id,
            analysis_depth="standard"
        )
        
        # Step 3: Calculate expected value
        ev_result = await calculate_expected_value_tool(
            market_id=market_id,
            investment_amount=500.0,
            confidence_adjustment=1.1
        )
        
        # Step 4: Place order based on analysis
        if ev_result["recommendation"]["action"] == "bet":
            order_result = await place_prediction_order_tool(
                market_id=market_id,
                outcome=ev_result["recommendation"]["outcome"],
                side="buy",
                quantity=int(ev_result["kelly_criterion"]["recommended_amount"])
            )
            
            # Step 5: Check position
            position_result = await get_prediction_positions_tool()
            
            # Verify complete flow
            assert markets_result["status"] == "success"
            assert sentiment_result["status"] == "success"
            assert ev_result["status"] == "success"
            assert order_result["status"] == "success"
            assert position_result["status"] == "success"
            assert len(position_result["positions"]) > 0

    @pytest.mark.asyncio
    @pytest.mark.mcp
    @pytest.mark.benchmark
    async def test_mcp_performance_benchmarks(self, polymarket_tools, sample_markets, benchmark):
        """Benchmark MCP tool performance."""
        polymarket_tools.client.get_markets.return_value = sample_markets
        
        # Benchmark market listing
        async def list_markets():
            return await get_prediction_markets_tool(limit=50)
        
        result = await benchmark.pedantic(
            list_markets,
            rounds=10,
            iterations=5,
            warmup_rounds=2
        )
        
        assert result["status"] == "success"
        assert benchmark.stats["mean"] < 0.1  # Should complete in < 100ms
        
        # Benchmark sentiment analysis
        polymarket_tools.client.get_market.return_value = sample_markets[0]
        polymarket_tools.client.get_orderbook.return_value = {"bids": [], "asks": []}
        
        async def analyze_sentiment():
            return await analyze_market_sentiment_tool(
                market_id="test",
                analysis_depth="standard"
            )
        
        result = await benchmark.pedantic(
            analyze_sentiment,
            rounds=10,
            iterations=5,
            warmup_rounds=2
        )
        
        assert benchmark.stats["mean"] < 0.2  # Should complete in < 200ms

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_gpu_acceleration_validation(self, polymarket_tools):
        """Validate GPU acceleration functionality."""
        # Mock GPU availability
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=2):
                # Test GPU-accelerated sentiment analysis
                market = Market(
                    id="gpu_test",
                    outcomes=["Yes", "No"],
                    outcome_prices=[Decimal("0.5"), Decimal("0.5")]
                )
                polymarket_tools.client.get_market.return_value = market
                polymarket_tools.client.get_orderbook.return_value = {
                    "bids": [], "asks": []
                }
                
                result = await analyze_market_sentiment_tool(
                    market_id="gpu_test",
                    analysis_depth="gpu_enhanced",
                    use_gpu=True
                )
                
                assert result["processing"]["gpu_acceleration"] is True
                assert result["processing"]["gpu_device"] == "cuda:0"
                assert "monte_carlo_paths" in result["analysis"]["advanced_metrics"]
                
                # Test GPU-accelerated EV calculation
                result = await calculate_expected_value_tool(
                    market_id="gpu_test",
                    investment_amount=1000.0,
                    use_gpu=True
                )
                
                assert result["processing"]["gpu_acceleration"] is True
                assert result["monte_carlo"]["simulations"] == 100000
                assert result["processing"]["execution_time"] < 1.0  # GPU should be fast

    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_concurrent_mcp_requests(self, polymarket_tools, sample_markets):
        """Test handling of concurrent MCP tool requests."""
        polymarket_tools.client.get_markets.return_value = sample_markets
        polymarket_tools.client.get_market.return_value = sample_markets[0]
        polymarket_tools.client.get_orderbook.return_value = {
            "bids": [], "asks": []
        }
        
        # Create concurrent requests
        tasks = []
        
        # Mix of different tool calls
        for i in range(20):
            if i % 4 == 0:
                task = get_prediction_markets_tool(limit=10)
            elif i % 4 == 1:
                task = analyze_market_sentiment_tool(
                    market_id=f"market_{i}",
                    analysis_depth="standard"
                )
            elif i % 4 == 2:
                task = calculate_expected_value_tool(
                    market_id=f"market_{i}",
                    investment_amount=100.0
                )
            else:
                task = get_prediction_positions_tool()
            
            tasks.append(task)
        
        # Execute concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        
        # Verify results
        successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
        errors = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful) >= 18  # At least 90% success rate
        assert elapsed < 2.0  # Should complete within 2 seconds
        assert len(errors) < 3  # Minimal errors


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])