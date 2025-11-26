"""
Integration tests for the fully integrated MCP server
Tests all 40 tools and their interactions
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mcp.mcp_server_integrated import (
    mcp, NEWS_AGGREGATION_AVAILABLE, STRATEGY_MANAGER_AVAILABLE,
    GPU_AVAILABLE, POLYMARKET_TOOLS_AVAILABLE
)

# Test fixtures
@pytest.fixture
def sample_symbols():
    return ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

@pytest.fixture
def sample_market_conditions():
    return {
        'volatility': 'moderate',
        'trend': 'bullish',
        'sentiment': 0.35,
        'rsi': 65,
        'volume': 'high'
    }

@pytest.fixture
def sample_trades():
    return [
        {"symbol": "AAPL", "action": "buy", "quantity": 100, "price": 189.50},
        {"symbol": "GOOGL", "action": "buy", "quantity": 50, "price": 141.25},
        {"symbol": "MSFT", "action": "sell", "quantity": 25, "price": 377.90}
    ]

# Test Original Tools (27)
class TestOriginalTools:
    """Test the 27 original MCP tools"""
    
    @pytest.mark.asyncio
    async def test_core_tools(self):
        """Test core tools (6)"""
        # Test ping
        result = await mcp.call_tool("ping", {})
        assert result == "pong"
        
        # Test list_strategies
        result = await mcp.call_tool("list_strategies", {})
        assert "strategies" in result
        assert "count" in result
        assert result["status"] == "success"
        
        # Test get_strategy_info
        result = await mcp.call_tool("get_strategy_info", {"strategy": "momentum_trading"})
        assert result["status"] == "operational"
        assert "performance_metrics" in result
        
        # Test quick_analysis
        result = await mcp.call_tool("quick_analysis", {"symbol": "AAPL", "use_gpu": False})
        assert result["status"] == "success"
        assert "analysis" in result
        
        # Test simulate_trade
        result = await mcp.call_tool("simulate_trade", {
            "strategy": "momentum_trading",
            "symbol": "AAPL",
            "action": "buy",
            "use_gpu": False
        })
        assert result["status"] == "executed"
        
        # Test get_portfolio_status
        result = await mcp.call_tool("get_portfolio_status", {"include_analytics": True})
        assert result["status"] == "success"
        assert "positions" in result
    
    @pytest.mark.asyncio
    async def test_news_analysis_tools(self):
        """Test news analysis tools (2)"""
        # Test analyze_news
        result = await mcp.call_tool("analyze_news", {
            "symbol": "AAPL",
            "lookback_hours": 24,
            "sentiment_model": "enhanced",
            "use_gpu": False
        })
        assert result["status"] == "success"
        assert "overall_sentiment" in result
        
        # Test get_news_sentiment
        result = await mcp.call_tool("get_news_sentiment", {
            "symbol": "AAPL",
            "sources": ["Reuters", "Bloomberg"]
        })
        assert result["status"] == "success"
        assert "real_time_sentiment" in result
    
    @pytest.mark.asyncio
    async def test_neural_forecasting_tools(self):
        """Test neural forecasting tools (6)"""
        # Test neural_forecast
        result = await mcp.call_tool("neural_forecast", {
            "symbol": "AAPL",
            "horizon": 7,
            "confidence_level": 0.95,
            "use_gpu": False
        })
        assert result["status"] == "success"
        assert "forecast" in result
        
        # Test neural_model_status
        result = await mcp.call_tool("neural_model_status", {})
        assert result["status"] == "success"
        assert "total_models" in result

# Test New Integration Tools (13)
class TestNewsCollectionTools:
    """Test news collection control tools (4)"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not NEWS_AGGREGATION_AVAILABLE, reason="News aggregation not available")
    async def test_control_news_collection(self, sample_symbols):
        """Test news collection control"""
        # Start collection
        result = await mcp.call_tool("control_news_collection", {
            "action": "start",
            "symbols": sample_symbols,
            "update_frequency": 300,
            "lookback_hours": 24
        })
        assert result["status"] == "active"
        assert result["symbols"] == sample_symbols
        
        # Configure collection
        result = await mcp.call_tool("control_news_collection", {
            "action": "configure",
            "sources": ["newsapi", "finnhub"],
            "update_frequency": 600
        })
        assert result["status"] == "configured"
        
        # Stop collection
        result = await mcp.call_tool("control_news_collection", {
            "action": "stop"
        })
        assert result["status"] == "stopped"
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not NEWS_AGGREGATION_AVAILABLE, reason="News aggregation not available")
    async def test_get_news_provider_status(self):
        """Test news provider status"""
        result = await mcp.call_tool("get_news_provider_status", {})
        assert result["status"] == "success"
        assert "providers" in result
        assert "metrics" in result
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not NEWS_AGGREGATION_AVAILABLE, reason="News aggregation not available")
    async def test_fetch_filtered_news(self, sample_symbols):
        """Test filtered news fetching"""
        result = await mcp.call_tool("fetch_filtered_news", {
            "symbols": sample_symbols[:2],
            "sentiment_filter": "positive",
            "relevance_threshold": 0.5,
            "limit": 10
        })
        assert result["status"] == "success"
        assert "filtered_items" in result
        assert result["filters_applied"]["sentiment"] == "positive"
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not NEWS_AGGREGATION_AVAILABLE, reason="News aggregation not available")
    async def test_get_news_trends(self, sample_symbols):
        """Test news trend analysis"""
        result = await mcp.call_tool("get_news_trends", {
            "symbols": sample_symbols[:2],
            "time_intervals": [1, 6, 24]
        })
        assert result["status"] == "success"
        assert "trends" in result
        assert "trend_analysis" in result

class TestStrategySelectionTools:
    """Test strategy selection tools (4)"""
    
    @pytest.mark.asyncio
    async def test_recommend_strategy(self, sample_market_conditions):
        """Test strategy recommendation"""
        result = await mcp.call_tool("recommend_strategy", {
            "market_conditions": sample_market_conditions,
            "risk_tolerance": "moderate",
            "objectives": ["profit", "stability"]
        })
        assert result["status"] == "success"
        assert "recommendation" in result
        assert "strategy_rankings" in result
        assert result["confidence"] >= 0 and result["confidence"] <= 1
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not STRATEGY_MANAGER_AVAILABLE, reason="Strategy manager not available")
    async def test_switch_active_strategy(self):
        """Test strategy switching"""
        result = await mcp.call_tool("switch_active_strategy", {
            "from_strategy": "momentum_trading",
            "to_strategy": "swing_trading",
            "close_positions": False
        })
        assert result["status"] == "success"
        assert result["switch_status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_get_strategy_comparison(self):
        """Test strategy comparison"""
        result = await mcp.call_tool("get_strategy_comparison", {
            "strategies": ["momentum_trading", "swing_trading", "mean_reversion"],
            "metrics": ["sharpe_ratio", "total_return", "max_drawdown"]
        })
        assert result["status"] == "success"
        assert "comparison" in result
        assert "best_by_metric" in result
    
    @pytest.mark.asyncio
    async def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection"""
        result = await mcp.call_tool("adaptive_strategy_selection", {
            "symbol": "AAPL",
            "auto_switch": False
        })
        assert result["status"] == "success"
        assert "selected_strategy" in result
        assert "market_conditions" in result

class TestPerformanceMonitoringTools:
    """Test performance monitoring tools (3)"""
    
    @pytest.mark.asyncio
    async def test_get_system_metrics(self):
        """Test system metrics"""
        result = await mcp.call_tool("get_system_metrics", {
            "metrics": ["cpu", "memory", "latency", "throughput"],
            "include_history": False,
            "time_range_minutes": 60
        })
        assert result["status"] == "success"
        assert "current_metrics" in result
        assert "system_health" in result
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not STRATEGY_MANAGER_AVAILABLE, reason="Strategy manager not available")
    async def test_monitor_strategy_health(self):
        """Test strategy health monitoring"""
        result = await mcp.call_tool("monitor_strategy_health", {
            "strategy": "momentum_trading"
        })
        assert result["status"] == "success"
        assert "health_score" in result
        assert "health_status" in result
        assert result["health_score"] >= 0 and result["health_score"] <= 100
    
    @pytest.mark.asyncio
    async def test_get_execution_analytics(self):
        """Test execution analytics"""
        result = await mcp.call_tool("get_execution_analytics", {
            "time_period": "1h"
        })
        assert result["status"] == "success"
        assert "execution_metrics" in result
        assert "slippage_analysis" in result

class TestMultiAssetTradingTools:
    """Test multi-asset trading tools (3)"""
    
    @pytest.mark.asyncio
    async def test_execute_multi_asset_trade(self, sample_trades):
        """Test multi-asset trade execution"""
        result = await mcp.call_tool("execute_multi_asset_trade", {
            "trades": sample_trades,
            "strategy": "momentum_trading",
            "risk_limit": 100000,
            "execute_parallel": True
        })
        assert result["status"] == "success"
        assert result["trades_executed"] == len(sample_trades)
        assert "execution_results" in result
    
    @pytest.mark.asyncio
    async def test_portfolio_rebalance(self):
        """Test portfolio rebalancing"""
        target_allocations = {
            "AAPL": 0.3,
            "GOOGL": 0.25,
            "MSFT": 0.25,
            "CASH": 0.2
        }
        
        result = await mcp.call_tool("portfolio_rebalance", {
            "target_allocations": target_allocations,
            "rebalance_threshold": 0.05
        })
        assert result["status"] == "success"
        assert "rebalance_trades" in result
        assert "tracking_error" in result
    
    @pytest.mark.asyncio
    async def test_cross_asset_correlation_matrix(self, sample_symbols):
        """Test cross-asset correlation analysis"""
        result = await mcp.call_tool("cross_asset_correlation_matrix", {
            "assets": sample_symbols[:4],
            "lookback_days": 90,
            "include_prediction_confidence": True
        })
        assert result["status"] == "success"
        assert "correlation_matrix" in result
        assert "diversification_metrics" in result

# Integration Flow Tests
class TestIntegrationFlows:
    """Test complete integration workflows"""
    
    @pytest.mark.asyncio
    async def test_news_to_trading_flow(self):
        """Test complete flow from news analysis to trading"""
        # Step 1: Analyze news
        news_result = await mcp.call_tool("analyze_news", {
            "symbol": "AAPL",
            "lookback_hours": 24,
            "sentiment_model": "enhanced",
            "use_gpu": False
        })
        assert news_result["status"] == "success"
        
        # Step 2: Get market analysis
        market_result = await mcp.call_tool("quick_analysis", {
            "symbol": "AAPL",
            "use_gpu": False
        })
        assert market_result["status"] == "success"
        
        # Step 3: Get strategy recommendation
        market_conditions = {
            'volatility': market_result['analysis']['volatility'],
            'trend': market_result['analysis']['trend'],
            'sentiment': news_result['overall_sentiment']
        }
        
        strategy_result = await mcp.call_tool("recommend_strategy", {
            "market_conditions": market_conditions,
            "risk_tolerance": "moderate",
            "objectives": ["profit", "stability"]
        })
        assert strategy_result["status"] == "success"
        
        # Step 4: Simulate trade
        trade_result = await mcp.call_tool("simulate_trade", {
            "strategy": strategy_result["recommendation"],
            "symbol": "AAPL",
            "action": "buy",
            "use_gpu": False
        })
        assert trade_result["status"] == "executed"
    
    @pytest.mark.asyncio
    async def test_multi_strategy_optimization_flow(self):
        """Test multi-strategy optimization workflow"""
        # Step 1: Compare strategies
        comparison_result = await mcp.call_tool("get_strategy_comparison", {
            "strategies": ["momentum_trading", "swing_trading", "mean_reversion"],
            "metrics": ["sharpe_ratio", "total_return", "max_drawdown"]
        })
        assert comparison_result["status"] == "success"
        
        # Step 2: Run backtest on best strategy
        best_strategy = comparison_result["best_by_metric"]["sharpe_ratio"]
        backtest_result = await mcp.call_tool("run_backtest", {
            "strategy": best_strategy,
            "symbol": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "use_gpu": False
        })
        assert backtest_result["status"] == "completed"
        
        # Step 3: Monitor strategy health
        if STRATEGY_MANAGER_AVAILABLE:
            health_result = await mcp.call_tool("monitor_strategy_health", {
                "strategy": best_strategy
            })
            assert health_result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_portfolio_management_flow(self, sample_symbols):
        """Test complete portfolio management workflow"""
        # Step 1: Analyze correlations
        correlation_result = await mcp.call_tool("cross_asset_correlation_matrix", {
            "assets": sample_symbols[:4],
            "lookback_days": 90,
            "include_prediction_confidence": True
        })
        assert correlation_result["status"] == "success"
        
        # Step 2: Get portfolio status
        portfolio_result = await mcp.call_tool("get_portfolio_status", {
            "include_analytics": True
        })
        assert portfolio_result["status"] == "success"
        
        # Step 3: Calculate rebalancing
        target_allocations = {
            "AAPL": 0.25,
            "GOOGL": 0.25,
            "MSFT": 0.25,
            "CASH": 0.25
        }
        
        rebalance_result = await mcp.call_tool("portfolio_rebalance", {
            "target_allocations": target_allocations,
            "rebalance_threshold": 0.05
        })
        assert rebalance_result["status"] == "success"
        
        # Step 4: Execute multi-asset trades if needed
        if rebalance_result["trades_required"] > 0:
            trades = [
                {
                    "symbol": trade["symbol"],
                    "action": trade["action"],
                    "quantity": trade["quantity"],
                    "price": 100  # Mock price
                }
                for trade in rebalance_result["rebalance_trades"][:3]
            ]
            
            if trades:
                execution_result = await mcp.call_tool("execute_multi_asset_trade", {
                    "trades": trades,
                    "strategy": "mean_reversion",
                    "execute_parallel": True
                })
                assert execution_result["status"] == "success"

# Performance and Load Tests
class TestPerformanceAndLoad:
    """Test system performance under load"""
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self):
        """Test concurrent execution of multiple tools"""
        tasks = [
            mcp.call_tool("ping", {}),
            mcp.call_tool("list_strategies", {}),
            mcp.call_tool("quick_analysis", {"symbol": "AAPL", "use_gpu": False}),
            mcp.call_tool("get_system_metrics", {"metrics": ["cpu", "memory"]}),
            mcp.call_tool("neural_model_status", {})
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all calls succeeded
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Concurrent calls failed: {errors}"
    
    @pytest.mark.asyncio
    async def test_high_frequency_trading_simulation(self):
        """Test system under high-frequency trading conditions"""
        start_time = datetime.now()
        trade_count = 0
        
        # Simulate 10 seconds of trading
        while (datetime.now() - start_time).seconds < 10:
            # Quick analysis
            analysis = await mcp.call_tool("quick_analysis", {
                "symbol": "AAPL",
                "use_gpu": GPU_AVAILABLE
            })
            
            # Simulate trade based on analysis
            if analysis["analysis"]["recommendation"] in ["buy", "sell"]:
                trade = await mcp.call_tool("simulate_trade", {
                    "strategy": "momentum_trading",
                    "symbol": "AAPL",
                    "action": analysis["analysis"]["recommendation"],
                    "use_gpu": GPU_AVAILABLE
                })
                if trade["status"] == "executed":
                    trade_count += 1
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        # Verify reasonable trade throughput
        assert trade_count > 0, "No trades executed in simulation"
        
        # Get execution analytics
        analytics = await mcp.call_tool("get_execution_analytics", {
            "time_period": "1h"
        })
        assert analytics["status"] == "success"

# Resource Tests
class TestResources:
    """Test MCP resources"""
    
    @pytest.mark.asyncio
    async def test_strategies_resource(self):
        """Test strategies resource"""
        result = await mcp.get_resource("strategies://available")
        data = json.loads(result)
        assert "strategies" in data
        assert "count" in data
        assert "gpu_available" in data
    
    @pytest.mark.asyncio
    async def test_integration_status_resource(self):
        """Test integration status resource"""
        result = await mcp.get_resource("integration://status")
        data = json.loads(result)
        assert "integration_components" in data
        assert "total_tools" in data
        assert data["total_tools"] >= 40
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not NEWS_AGGREGATION_AVAILABLE, reason="News aggregation not available")
    async def test_news_resource(self):
        """Test news aggregation resource"""
        result = await mcp.get_resource("news://aggregated/AAPL")
        data = json.loads(result)
        assert "symbol" in data
        assert data["symbol"] == "AAPL"

# Error Handling Tests
class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_invalid_strategy_name(self):
        """Test handling of invalid strategy names"""
        result = await mcp.call_tool("get_strategy_info", {
            "strategy": "non_existent_strategy"
        })
        assert result["status"] == "failed"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_invalid_tool_parameters(self):
        """Test handling of invalid parameters"""
        # Missing required parameter
        with pytest.raises(Exception):
            await mcp.call_tool("quick_analysis", {})
        
        # Invalid parameter type
        result = await mcp.call_tool("neural_forecast", {
            "symbol": "AAPL",
            "horizon": "invalid",  # Should be int
            "use_gpu": False
        })
        assert "error" in result or result["status"] == "failed"
    
    @pytest.mark.asyncio
    async def test_risk_limit_enforcement(self):
        """Test risk limit enforcement in trades"""
        large_trades = [
            {"symbol": "AAPL", "action": "buy", "quantity": 10000, "price": 189.50}
        ]
        
        result = await mcp.call_tool("execute_multi_asset_trade", {
            "trades": large_trades,
            "strategy": "momentum_trading",
            "risk_limit": 100000,  # Limit that will be exceeded
            "execute_parallel": True
        })
        assert result["status"] == "failed"
        assert "risk limit" in result["error"]

# Cleanup Tests
class TestCleanup:
    """Test cleanup and resource management"""
    
    @pytest.mark.asyncio
    async def test_news_aggregator_cleanup(self):
        """Test news aggregator cleanup"""
        if NEWS_AGGREGATION_AVAILABLE:
            # Start collection
            await mcp.call_tool("control_news_collection", {
                "action": "start",
                "symbols": ["AAPL"],
                "update_frequency": 300
            })
            
            # Stop collection
            result = await mcp.call_tool("control_news_collection", {
                "action": "stop"
            })
            assert result["status"] == "stopped"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])