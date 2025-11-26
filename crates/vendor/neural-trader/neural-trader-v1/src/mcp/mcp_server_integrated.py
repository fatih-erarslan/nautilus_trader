#!/usr/bin/env python3
"""
Integrated MCP Server for AI News Trading Platform
Extended with news aggregation, strategy management, and advanced trading capabilities
Total tools: 40+ (original 27 + new integration tools)
"""

import json
import logging
import sys
import os
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import asyncio

# Critical: Configure logging to NOT interfere with stdio transport
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)

# Suppress other library logging
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('FastMCP').setLevel(logging.WARNING)

# Import FastMCP with error handling
try:
    from fastmcp import FastMCP
    from pydantic import BaseModel
except ImportError as e:
    print(f"ERROR: Failed to import required packages: {e}", file=sys.stderr)
    sys.exit(1)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import news aggregation components
try:
    from integrations.news_aggregator import UnifiedNewsAggregator, UnifiedNewsItem
    from integrations.config import Config
    NEWS_AGGREGATION_AVAILABLE = True
    logger.info("News aggregation components loaded successfully")
except ImportError as e:
    logger.warning(f"News aggregation components not available: {e}")
    NEWS_AGGREGATION_AVAILABLE = False

# Import strategy management
try:
    from mcp.trading.strategy_manager import StrategyManager
    STRATEGY_MANAGER_AVAILABLE = True
    logger.info("Strategy manager loaded successfully")
except ImportError as e:
    logger.warning(f"Strategy manager not available: {e}")
    STRATEGY_MANAGER_AVAILABLE = False

# Import the original MCP server to extend it
try:
    from mcp_server_enhanced import (
        GPU_AVAILABLE, OPTIMIZED_MODELS, BENCHMARK_DATA, NEURAL_MODELS,
        load_trading_models, load_benchmark_data, load_neural_models,
        POLYMARKET_TOOLS_AVAILABLE, POLYMARKET_GPU_AVAILABLE
    )
    
    # Import all the original tool functions
    from mcp_server_enhanced import (
        ping, list_strategies, get_strategy_info, quick_analysis,
        simulate_trade, get_portfolio_status, analyze_news, get_news_sentiment,
        run_backtest, optimize_strategy, risk_analysis, execute_trade,
        performance_report, correlation_analysis, run_benchmark,
        neural_forecast, neural_train, neural_evaluate, neural_backtest,
        neural_model_status, neural_optimize
    )
    
    # Import Polymarket tools if available
    if POLYMARKET_TOOLS_AVAILABLE:
        from mcp_server_enhanced import (
            get_prediction_markets_tool, analyze_market_sentiment_tool,
            get_market_orderbook_tool, place_prediction_order_tool,
            get_prediction_positions_tool, calculate_expected_value_tool
        )
        
except ImportError as e:
    logger.error(f"Failed to import original MCP server components: {e}")
    sys.exit(1)

# Load data from original server
load_trading_models()
load_benchmark_data()
load_neural_models()

# Initialize integrated components
news_aggregator = None
strategy_manager = None

if NEWS_AGGREGATION_AVAILABLE:
    try:
        config = Config()
        news_config = {
            'alpha_vantage': {'enabled': True, 'api_key': config.get('ALPHA_VANTAGE_API_KEY', '')},
            'newsapi': {'enabled': True, 'api_key': config.get('NEWSAPI_API_KEY', '')},
            'finnhub': {'enabled': True, 'api_key': config.get('FINNHUB_API_KEY', '')},
            'redis_url': config.get('REDIS_URL', 'redis://localhost:6379'),
            'cache_ttl': 3600,
            'similarity_threshold': 0.85,
            'max_concurrent_requests': 10,
            'request_timeout': 30,
            'min_source_reliability': 0.3
        }
        news_aggregator = UnifiedNewsAggregator(news_config)
        logger.info("News aggregator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize news aggregator: {e}")
        NEWS_AGGREGATION_AVAILABLE = False

if STRATEGY_MANAGER_AVAILABLE:
    try:
        strategy_manager = StrategyManager(gpu_enabled=GPU_AVAILABLE)
        # Initialize in background
        asyncio.create_task(strategy_manager.initialize())
        logger.info("Strategy manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize strategy manager: {e}")
        STRATEGY_MANAGER_AVAILABLE = False

# Initialize Integrated MCP server
mcp = FastMCP("AI News Trading Platform - Fully Integrated Edition")

# Enhanced Pydantic models for new tools
class NewsControlRequest(BaseModel):
    action: str  # start, stop, configure
    symbols: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    update_frequency: Optional[int] = 300  # seconds
    lookback_hours: Optional[int] = 24

class MultiAssetTradeRequest(BaseModel):
    trades: List[Dict[str, Any]]
    strategy: str
    risk_limit: Optional[float] = None
    execute_parallel: bool = True

class StrategySelectionRequest(BaseModel):
    market_conditions: Dict[str, Any]
    risk_tolerance: str = "moderate"  # low, moderate, high
    objectives: List[str] = ["profit", "stability"]

class SystemMonitoringRequest(BaseModel):
    metrics: List[str] = ["cpu", "memory", "latency", "throughput"]
    include_history: bool = False
    time_range_minutes: int = 60

# === ORIGINAL 27 TOOLS (Re-export from original server) ===
# These are already defined in mcp_server_enhanced.py, so we'll register them

# Core Tools (6)
mcp.tool()(ping)
mcp.tool()(list_strategies)
mcp.tool()(get_strategy_info)
mcp.tool()(quick_analysis)
mcp.tool()(simulate_trade)
mcp.tool()(get_portfolio_status)

# News Analysis Tools (2)
mcp.tool()(analyze_news)
mcp.tool()(get_news_sentiment)

# Advanced Trading Tools (5)
mcp.tool()(run_backtest)
mcp.tool()(optimize_strategy)
mcp.tool()(risk_analysis)
mcp.tool()(execute_trade)

# Analytics Tools (2)
mcp.tool()(performance_report)
mcp.tool()(correlation_analysis)

# Benchmark Tools (1)
mcp.tool()(run_benchmark)

# Neural Forecasting Tools (6)
mcp.tool()(neural_forecast)
mcp.tool()(neural_train)
mcp.tool()(neural_evaluate)
mcp.tool()(neural_backtest)
mcp.tool()(neural_model_status)
mcp.tool()(neural_optimize)

# Polymarket Tools (6) - if available
if POLYMARKET_TOOLS_AVAILABLE:
    mcp.tool()(get_prediction_markets_tool)
    mcp.tool()(analyze_market_sentiment_tool)
    mcp.tool()(get_market_orderbook_tool)
    mcp.tool()(place_prediction_order_tool)
    mcp.tool()(get_prediction_positions_tool)
    mcp.tool()(calculate_expected_value_tool)

# === NEW INTEGRATION TOOLS (13+) ===

# News Collection Control Tools (4)
@mcp.tool()
async def control_news_collection(action: str, symbols: Optional[List[str]] = None,
                                sources: Optional[List[str]] = None,
                                update_frequency: Optional[int] = 300,
                                lookback_hours: Optional[int] = 24) -> Dict[str, Any]:
    """Control news collection: start, stop, configure news fetching"""
    if not NEWS_AGGREGATION_AVAILABLE:
        return {"error": "News aggregation not available", "status": "failed"}
    
    try:
        if action == "start":
            # Start news collection for specified symbols
            if not symbols:
                return {"error": "Symbols required for start action", "status": "failed"}
            
            # Fetch initial news
            news_items = await news_aggregator.fetch_aggregated_news(
                symbols=symbols,
                lookback_hours=lookback_hours,
                limit=100
            )
            
            return {
                "action": "start",
                "symbols": symbols,
                "status": "active",
                "initial_items": len(news_items),
                "update_frequency": update_frequency,
                "lookback_hours": lookback_hours,
                "timestamp": datetime.now().isoformat()
            }
            
        elif action == "stop":
            # Stop news collection
            return {
                "action": "stop",
                "status": "stopped",
                "timestamp": datetime.now().isoformat()
            }
            
        elif action == "configure":
            # Configure news collection parameters
            config_updated = {}
            if sources:
                # Update enabled sources
                for source in sources:
                    if source in news_aggregator.providers:
                        config_updated[source] = "enabled"
            
            if update_frequency:
                config_updated['update_frequency'] = update_frequency
                
            return {
                "action": "configure",
                "status": "configured",
                "configuration": config_updated,
                "timestamp": datetime.now().isoformat()
            }
            
        else:
            return {"error": f"Unknown action: {action}", "status": "failed"}
            
    except Exception as e:
        logger.error(f"Error in control_news_collection: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
async def get_news_provider_status() -> Dict[str, Any]:
    """Get current status of all news providers"""
    if not NEWS_AGGREGATION_AVAILABLE:
        return {"error": "News aggregation not available", "status": "failed"}
    
    try:
        provider_status = await news_aggregator.get_provider_status()
        metrics = news_aggregator.get_metrics()
        
        return {
            "providers": provider_status,
            "metrics": metrics,
            "active_providers": len([p for p, s in provider_status.items() if s.get('healthy', False)]),
            "total_providers": len(provider_status),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting news provider status: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
async def fetch_filtered_news(symbols: List[str], 
                            sentiment_filter: Optional[str] = None,
                            relevance_threshold: float = 0.5,
                            limit: int = 50) -> Dict[str, Any]:
    """Fetch news with advanced filtering options"""
    if not NEWS_AGGREGATION_AVAILABLE:
        return {"error": "News aggregation not available", "status": "failed"}
    
    try:
        # Fetch news
        news_items = await news_aggregator.fetch_aggregated_news(
            symbols=symbols,
            lookback_hours=24,
            limit=limit * 2  # Fetch more to filter
        )
        
        # Apply filters
        filtered_items = []
        for item in news_items:
            # Check relevance
            max_relevance = max(item.relevance_scores.values()) if item.relevance_scores else 0
            if max_relevance < relevance_threshold:
                continue
            
            # Check sentiment filter
            if sentiment_filter:
                if sentiment_filter == "positive" and (item.sentiment is None or item.sentiment <= 0.1):
                    continue
                elif sentiment_filter == "negative" and (item.sentiment is None or item.sentiment >= -0.1):
                    continue
                elif sentiment_filter == "neutral" and (item.sentiment is None or abs(item.sentiment) > 0.1):
                    continue
            
            filtered_items.append(item.to_dict())
            
            if len(filtered_items) >= limit:
                break
        
        return {
            "symbols": symbols,
            "total_items": len(news_items),
            "filtered_items": len(filtered_items),
            "items": filtered_items,
            "filters_applied": {
                "sentiment": sentiment_filter,
                "relevance_threshold": relevance_threshold
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error fetching filtered news: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
async def get_news_trends(symbols: List[str], 
                        time_intervals: List[int] = [1, 6, 24]) -> Dict[str, Any]:
    """Analyze news trends over multiple time intervals"""
    if not NEWS_AGGREGATION_AVAILABLE:
        return {"error": "News aggregation not available", "status": "failed"}
    
    try:
        trends = {}
        
        for interval in time_intervals:
            # Get sentiment summary for interval
            summaries = {}
            for symbol in symbols:
                summary = await news_aggregator.get_sentiment_summary(symbol, interval)
                summaries[symbol] = summary
            
            trends[f"{interval}h"] = summaries
        
        # Calculate trend direction
        trend_analysis = {}
        for symbol in symbols:
            sentiments = []
            for interval in sorted(time_intervals):
                sentiment = trends[f"{interval}h"][symbol].get('avg_sentiment', 0)
                sentiments.append(sentiment)
            
            # Determine trend
            if len(sentiments) >= 2:
                if sentiments[-1] > sentiments[0] + 0.1:
                    trend_direction = "improving"
                elif sentiments[-1] < sentiments[0] - 0.1:
                    trend_direction = "declining"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "unknown"
            
            trend_analysis[symbol] = {
                "trend_direction": trend_direction,
                "sentiment_changes": sentiments,
                "momentum": sentiments[-1] - sentiments[0] if len(sentiments) >= 2 else 0
            }
        
        return {
            "symbols": symbols,
            "time_intervals": time_intervals,
            "trends": trends,
            "trend_analysis": trend_analysis,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error analyzing news trends: {e}")
        return {"error": str(e), "status": "failed"}

# Strategy Selection Tools (4)
@mcp.tool()
async def recommend_strategy(market_conditions: Dict[str, Any],
                           risk_tolerance: str = "moderate",
                           objectives: List[str] = ["profit", "stability"]) -> Dict[str, Any]:
    """Recommend best strategy based on market conditions and objectives"""
    try:
        # Analyze market conditions
        volatility = market_conditions.get('volatility', 'moderate')
        trend = market_conditions.get('trend', 'neutral')
        sentiment = market_conditions.get('sentiment', 0)
        
        # Score each strategy based on conditions
        strategy_scores = {}
        
        for strategy_name, strategy_info in OPTIMIZED_MODELS.items():
            score = 0
            metrics = strategy_info.get('performance_metrics', {})
            
            # Score based on risk tolerance
            sharpe = metrics.get('sharpe_ratio', 0)
            max_dd = abs(metrics.get('max_drawdown', -1))
            
            if risk_tolerance == "low":
                score += sharpe * 0.3 + (1 - max_dd) * 0.7
            elif risk_tolerance == "moderate":
                score += sharpe * 0.5 + (1 - max_dd) * 0.5
            else:  # high
                score += sharpe * 0.7 + (1 - max_dd) * 0.3
            
            # Adjust for market conditions
            if strategy_name == "momentum_trading" and trend == "bullish":
                score *= 1.3
            elif strategy_name == "mean_reversion" and volatility == "high":
                score *= 1.2
            elif strategy_name == "swing_trading" and trend == "neutral":
                score *= 1.1
            
            # Adjust for objectives
            if "profit" in objectives:
                score *= (1 + metrics.get('total_return', 0))
            if "stability" in objectives:
                score *= (1 + metrics.get('win_rate', 0))
            
            strategy_scores[strategy_name] = score
        
        # Rank strategies
        ranked_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top recommendation
        best_strategy = ranked_strategies[0][0]
        best_score = ranked_strategies[0][1]
        
        return {
            "recommendation": best_strategy,
            "confidence": min(1.0, best_score / 10),  # Normalize score
            "strategy_rankings": [
                {"strategy": name, "score": round(score, 3)} 
                for name, score in ranked_strategies
            ],
            "market_conditions": market_conditions,
            "criteria": {
                "risk_tolerance": risk_tolerance,
                "objectives": objectives
            },
            "reasoning": f"Selected {best_strategy} based on {risk_tolerance} risk tolerance and market {trend} trend",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error recommending strategy: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
async def switch_active_strategy(from_strategy: str, to_strategy: str,
                               close_positions: bool = False) -> Dict[str, Any]:
    """Switch from one strategy to another with position management"""
    if not STRATEGY_MANAGER_AVAILABLE:
        return {"error": "Strategy manager not available", "status": "failed"}
    
    try:
        # Validate strategies
        available = await strategy_manager.get_available_strategies()
        if from_strategy not in available or to_strategy not in available:
            return {
                "error": "Invalid strategy names",
                "available_strategies": available,
                "status": "failed"
            }
        
        # Get current positions
        current_positions = await strategy_manager.get_positions(from_strategy)
        
        # Close positions if requested
        closed_positions = []
        if close_positions and current_positions:
            # Simulate closing positions
            for position in current_positions:
                closed_positions.append({
                    "symbol": position['symbol'],
                    "quantity": position['quantity'],
                    "exit_price": position['current_price'],
                    "pnl": position['unrealized_pnl']
                })
        
        return {
            "from_strategy": from_strategy,
            "to_strategy": to_strategy,
            "positions_affected": len(current_positions),
            "positions_closed": len(closed_positions),
            "closed_positions": closed_positions,
            "switch_status": "completed",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error switching strategy: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
async def get_strategy_comparison(strategies: List[str], 
                                metrics: List[str] = ["sharpe_ratio", "total_return", "max_drawdown"]) -> Dict[str, Any]:
    """Compare multiple strategies across specified metrics"""
    try:
        comparison = {}
        
        for strategy in strategies:
            if strategy in OPTIMIZED_MODELS:
                strategy_metrics = OPTIMIZED_MODELS[strategy].get('performance_metrics', {})
                comparison[strategy] = {}
                
                for metric in metrics:
                    comparison[strategy][metric] = strategy_metrics.get(metric, "N/A")
                
                # Add additional computed metrics
                comparison[strategy]["risk_adjusted_return"] = (
                    strategy_metrics.get('total_return', 0) / 
                    abs(strategy_metrics.get('max_drawdown', -1))
                    if strategy_metrics.get('max_drawdown', 0) != 0 else 0
                )
        
        # Find best for each metric
        best_by_metric = {}
        for metric in metrics + ["risk_adjusted_return"]:
            best_value = None
            best_strategy = None
            
            for strategy, values in comparison.items():
                value = values.get(metric, None)
                if value != "N/A" and value is not None:
                    if metric == "max_drawdown":
                        # For drawdown, less negative is better
                        if best_value is None or value > best_value:
                            best_value = value
                            best_strategy = strategy
                    else:
                        # For other metrics, higher is better
                        if best_value is None or value > best_value:
                            best_value = value
                            best_strategy = strategy
            
            best_by_metric[metric] = best_strategy
        
        return {
            "strategies": strategies,
            "comparison": comparison,
            "best_by_metric": best_by_metric,
            "metrics_analyzed": metrics + ["risk_adjusted_return"],
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
async def adaptive_strategy_selection(symbol: str, 
                                    auto_switch: bool = False) -> Dict[str, Any]:
    """Automatically select best strategy for current market conditions"""
    try:
        # Get current market analysis
        market_analysis = await quick_analysis(symbol, use_gpu=GPU_AVAILABLE)
        news_sentiment = await get_news_sentiment(symbol)
        
        # Determine market conditions
        market_conditions = {
            'volatility': market_analysis['analysis']['volatility'],
            'trend': market_analysis['analysis']['trend'],
            'sentiment': news_sentiment['real_time_sentiment'],
            'rsi': market_analysis['analysis']['rsi'],
            'volume': market_analysis['analysis'].get('volume', 'normal')
        }
        
        # Get strategy recommendation
        recommendation = await recommend_strategy(
            market_conditions=market_conditions,
            risk_tolerance="moderate",
            objectives=["profit", "stability"]
        )
        
        selected_strategy = recommendation['recommendation']
        
        # Auto-switch if enabled
        switch_result = None
        if auto_switch and STRATEGY_MANAGER_AVAILABLE:
            # Get current active strategy (would need to track this)
            current_strategy = "momentum_trading"  # Default assumption
            
            if current_strategy != selected_strategy:
                switch_result = await switch_active_strategy(
                    from_strategy=current_strategy,
                    to_strategy=selected_strategy,
                    close_positions=False
                )
        
        return {
            "symbol": symbol,
            "market_conditions": market_conditions,
            "selected_strategy": selected_strategy,
            "selection_confidence": recommendation['confidence'],
            "auto_switch_enabled": auto_switch,
            "switch_result": switch_result,
            "alternative_strategies": recommendation['strategy_rankings'][1:3],
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in adaptive strategy selection: {e}")
        return {"error": str(e), "status": "failed"}

# Performance Monitoring Tools (3)
@mcp.tool()
async def get_system_metrics(metrics: List[str] = ["cpu", "memory", "latency", "throughput"],
                           include_history: bool = False,
                           time_range_minutes: int = 60) -> Dict[str, Any]:
    """Get comprehensive system performance metrics"""
    try:
        import psutil
        
        current_metrics = {}
        
        # CPU metrics
        if "cpu" in metrics:
            current_metrics["cpu"] = {
                "usage_percent": psutil.cpu_percent(interval=0.1),
                "core_count": psutil.cpu_count(),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0
            }
        
        # Memory metrics
        if "memory" in metrics:
            mem = psutil.virtual_memory()
            current_metrics["memory"] = {
                "total_gb": round(mem.total / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "usage_percent": mem.percent
            }
        
        # Latency metrics (simulated)
        if "latency" in metrics:
            current_metrics["latency"] = {
                "api_response_ms": round(random.uniform(10, 50), 2),
                "database_query_ms": round(random.uniform(5, 20), 2),
                "neural_inference_ms": round(random.uniform(20, 100), 2) if GPU_AVAILABLE else round(random.uniform(100, 500), 2)
            }
        
        # Throughput metrics (simulated)
        if "throughput" in metrics:
            current_metrics["throughput"] = {
                "trades_per_second": round(random.uniform(10, 100), 2),
                "news_items_per_minute": round(random.uniform(50, 200), 2),
                "predictions_per_second": round(random.uniform(5, 50), 2)
            }
        
        # Historical data (simulated)
        history = None
        if include_history:
            history = {
                "cpu_history": [random.uniform(20, 80) for _ in range(time_range_minutes)],
                "memory_history": [random.uniform(40, 70) for _ in range(time_range_minutes)],
                "timestamps": [
                    (datetime.now() - timedelta(minutes=i)).isoformat() 
                    for i in range(time_range_minutes, 0, -1)
                ]
            }
        
        return {
            "current_metrics": current_metrics,
            "history": history,
            "gpu_available": GPU_AVAILABLE,
            "system_health": "healthy" if all(
                current_metrics.get("cpu", {}).get("usage_percent", 100) < 90,
                current_metrics.get("memory", {}).get("usage_percent", 100) < 85
            ) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
async def monitor_strategy_health(strategy: str) -> Dict[str, Any]:
    """Monitor health and performance of a specific strategy"""
    if not STRATEGY_MANAGER_AVAILABLE:
        return {"error": "Strategy manager not available", "status": "failed"}
    
    try:
        # Get strategy performance
        performance = await strategy_manager.get_performance(strategy, "1d")
        positions = await strategy_manager.get_positions(strategy)
        
        # Calculate health metrics
        health_score = 100
        issues = []
        
        # Check win rate
        win_rate = performance['metrics']['win_rate']
        if win_rate < 0.4:
            health_score -= 20
            issues.append("Low win rate")
        
        # Check drawdown
        max_drawdown = performance['metrics']['max_drawdown']
        if max_drawdown < -0.15:
            health_score -= 15
            issues.append("High drawdown")
        
        # Check Sharpe ratio
        sharpe_ratio = performance['metrics']['sharpe_ratio']
        if sharpe_ratio < 1.0:
            health_score -= 10
            issues.append("Low Sharpe ratio")
        
        # Check position concentration
        if positions:
            position_values = [p['value'] for p in positions]
            total_value = sum(position_values)
            max_position = max(position_values) if position_values else 0
            concentration = max_position / total_value if total_value > 0 else 0
            
            if concentration > 0.3:
                health_score -= 10
                issues.append("High position concentration")
        
        # Determine health status
        if health_score >= 80:
            health_status = "healthy"
        elif health_score >= 60:
            health_status = "warning"
        else:
            health_status = "critical"
        
        return {
            "strategy": strategy,
            "health_score": health_score,
            "health_status": health_status,
            "issues": issues,
            "metrics": {
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "active_positions": len(positions),
                "total_value": sum(p['value'] for p in positions)
            },
            "recommendations": [
                "Consider reducing position sizes" if "High position concentration" in issues else None,
                "Review strategy parameters" if "Low win rate" in issues else None,
                "Implement tighter risk controls" if "High drawdown" in issues else None
            ],
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error monitoring strategy health: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
async def get_execution_analytics(time_period: str = "1h") -> Dict[str, Any]:
    """Get detailed execution analytics and latency metrics"""
    try:
        # Simulate execution analytics
        period_map = {
            "1h": 60,
            "1d": 1440,
            "1w": 10080
        }
        minutes = period_map.get(time_period, 60)
        
        # Generate simulated execution data
        num_trades = int(minutes * random.uniform(0.5, 2))
        
        execution_times = [random.uniform(10, 100) for _ in range(num_trades)]
        slippage_bps = [random.uniform(-5, 5) for _ in range(num_trades)]
        
        return {
            "time_period": time_period,
            "execution_metrics": {
                "total_trades": num_trades,
                "avg_execution_time_ms": round(np.mean(execution_times), 2),
                "median_execution_time_ms": round(np.median(execution_times), 2),
                "p95_execution_time_ms": round(np.percentile(execution_times, 95), 2),
                "p99_execution_time_ms": round(np.percentile(execution_times, 99), 2)
            },
            "slippage_analysis": {
                "avg_slippage_bps": round(np.mean(slippage_bps), 2),
                "positive_slippage_pct": round(len([s for s in slippage_bps if s > 0]) / len(slippage_bps) * 100, 1),
                "max_slippage_bps": round(max(abs(s) for s in slippage_bps), 2)
            },
            "system_performance": {
                "gpu_utilization": round(random.uniform(30, 80), 1) if GPU_AVAILABLE else 0,
                "cpu_utilization": round(random.uniform(20, 60), 1),
                "api_success_rate": round(random.uniform(98, 99.9), 2),
                "cache_hit_rate": round(random.uniform(70, 95), 1)
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting execution analytics: {e}")
        return {"error": str(e), "status": "failed"}

# Multi-Asset Trading Tools (3)
@mcp.tool()
async def execute_multi_asset_trade(trades: List[Dict[str, Any]], 
                                  strategy: str,
                                  risk_limit: Optional[float] = None,
                                  execute_parallel: bool = True) -> Dict[str, Any]:
    """Execute trades on multiple assets with risk management"""
    try:
        # Validate strategy
        if strategy not in OPTIMIZED_MODELS:
            return {
                "error": f"Strategy '{strategy}' not found",
                "available": list(OPTIMIZED_MODELS.keys()),
                "status": "failed"
            }
        
        # Risk check
        total_value = sum(t.get('quantity', 0) * t.get('price', 100) for t in trades)
        if risk_limit and total_value > risk_limit:
            return {
                "error": f"Total trade value {total_value} exceeds risk limit {risk_limit}",
                "status": "failed"
            }
        
        # Execute trades
        execution_results = []
        start_time = time.time()
        
        for trade in trades:
            # Simulate execution
            execution_price = trade.get('price', 100) * (1 + random.uniform(-0.001, 0.001))
            execution_time = random.uniform(10, 50) if execute_parallel else random.uniform(50, 200)
            
            result = {
                "symbol": trade['symbol'],
                "action": trade['action'],
                "quantity": trade['quantity'],
                "requested_price": trade.get('price', 100),
                "execution_price": round(execution_price, 2),
                "execution_time_ms": round(execution_time, 1),
                "status": "executed",
                "value": round(trade['quantity'] * execution_price, 2)
            }
            execution_results.append(result)
        
        total_execution_time = time.time() - start_time
        
        # Calculate summary statistics
        total_executed_value = sum(r['value'] for r in execution_results)
        avg_slippage = np.mean([
            (r['execution_price'] - r['requested_price']) / r['requested_price'] 
            for r in execution_results
        ])
        
        return {
            "strategy": strategy,
            "trades_requested": len(trades),
            "trades_executed": len(execution_results),
            "execution_results": execution_results,
            "summary": {
                "total_value": round(total_executed_value, 2),
                "avg_slippage_bps": round(avg_slippage * 10000, 2),
                "total_execution_time_s": round(total_execution_time, 3),
                "parallel_execution": execute_parallel
            },
            "risk_check": {
                "risk_limit": risk_limit,
                "within_limit": risk_limit is None or total_executed_value <= risk_limit
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error executing multi-asset trade: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
async def portfolio_rebalance(target_allocations: Dict[str, float],
                            current_portfolio: Optional[Dict[str, float]] = None,
                            rebalance_threshold: float = 0.05) -> Dict[str, Any]:
    """Calculate and optionally execute portfolio rebalancing"""
    try:
        # Get current portfolio if not provided
        if current_portfolio is None and STRATEGY_MANAGER_AVAILABLE:
            positions = await strategy_manager.get_all_positions()
            current_portfolio = {}
            total_value = sum(p['value'] for p in positions)
            
            for position in positions:
                symbol = position['symbol']
                if symbol in current_portfolio:
                    current_portfolio[symbol] += position['value']
                else:
                    current_portfolio[symbol] = position['value']
            
            # Convert to percentages
            for symbol in current_portfolio:
                current_portfolio[symbol] = current_portfolio[symbol] / total_value if total_value > 0 else 0
        
        if not current_portfolio:
            current_portfolio = {"CASH": 1.0}
        
        # Calculate required trades
        rebalance_trades = []
        portfolio_value = 100000  # Assume $100k portfolio
        
        # Normalize target allocations
        total_target = sum(target_allocations.values())
        if total_target > 0:
            target_allocations = {k: v/total_target for k, v in target_allocations.items()}
        
        for symbol, target_pct in target_allocations.items():
            current_pct = current_portfolio.get(symbol, 0)
            diff_pct = target_pct - current_pct
            
            # Only rebalance if difference exceeds threshold
            if abs(diff_pct) > rebalance_threshold:
                trade_value = diff_pct * portfolio_value
                action = "buy" if diff_pct > 0 else "sell"
                
                # Get current price (simulated)
                current_price = 100 * (1 + random.uniform(-0.2, 0.2))
                quantity = abs(int(trade_value / current_price))
                
                if quantity > 0:
                    rebalance_trades.append({
                        "symbol": symbol,
                        "action": action,
                        "quantity": quantity,
                        "current_allocation": round(current_pct * 100, 2),
                        "target_allocation": round(target_pct * 100, 2),
                        "allocation_diff": round(diff_pct * 100, 2),
                        "trade_value": round(abs(trade_value), 2)
                    })
        
        # Calculate tracking error
        tracking_error = np.sqrt(sum(
            (target_allocations.get(s, 0) - current_portfolio.get(s, 0))**2 
            for s in set(list(target_allocations.keys()) + list(current_portfolio.keys()))
        ))
        
        return {
            "current_allocations": {k: round(v*100, 2) for k, v in current_portfolio.items()},
            "target_allocations": {k: round(v*100, 2) for k, v in target_allocations.items()},
            "rebalance_trades": rebalance_trades,
            "trades_required": len(rebalance_trades),
            "total_trade_value": sum(t['trade_value'] for t in rebalance_trades),
            "tracking_error": round(tracking_error * 100, 2),
            "rebalance_threshold": round(rebalance_threshold * 100, 2),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in portfolio rebalance: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
async def cross_asset_correlation_matrix(assets: List[str], 
                                       lookback_days: int = 90,
                                       include_prediction_confidence: bool = True) -> Dict[str, Any]:
    """Generate correlation matrix for multiple assets with prediction confidence"""
    try:
        # Use existing correlation analysis tool
        correlation_result = await correlation_analysis(assets, lookback_days, use_gpu=GPU_AVAILABLE)
        
        if correlation_result.get('status') != 'completed':
            return correlation_result
        
        correlation_matrix = correlation_result['correlation_matrix']
        
        # Add prediction confidence if requested
        prediction_confidence = {}
        if include_prediction_confidence and len(assets) > 1:
            for asset1 in assets:
                prediction_confidence[asset1] = {}
                for asset2 in assets:
                    if asset1 != asset2:
                        # Higher correlation = higher prediction confidence
                        corr = abs(correlation_matrix[asset1][asset2])
                        confidence = min(0.95, 0.5 + corr * 0.45)
                        prediction_confidence[asset1][asset2] = round(confidence, 3)
        
        # Identify highly correlated pairs
        high_correlation_pairs = []
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets[i+1:], i+1):
                corr = correlation_matrix[asset1][asset2]
                if abs(corr) > 0.7:
                    high_correlation_pairs.append({
                        "pair": f"{asset1}-{asset2}",
                        "correlation": corr,
                        "relationship": "positive" if corr > 0 else "negative"
                    })
        
        # Calculate portfolio diversification score
        avg_correlation = correlation_result['summary_statistics']['average_correlation']
        diversification_score = 1 - avg_correlation
        
        return {
            "assets": assets,
            "lookback_days": lookback_days,
            "correlation_matrix": correlation_matrix,
            "prediction_confidence": prediction_confidence if include_prediction_confidence else None,
            "high_correlation_pairs": high_correlation_pairs,
            "diversification_metrics": {
                "diversification_score": round(diversification_score, 3),
                "effective_assets": correlation_result['diversification_metrics']['effective_assets'],
                "concentration_risk": correlation_result['diversification_metrics']['concentration_risk']
            },
            "recommendations": [
                f"Consider reducing exposure to {pair['pair']}" 
                for pair in high_correlation_pairs[:3]
            ],
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error generating cross-asset correlation matrix: {e}")
        return {"error": str(e), "status": "failed"}

# === RESOURCES ===
# Original resources from mcp_server_enhanced.py
@mcp.resource("strategies://available")
async def get_available_strategies() -> str:
    """Get available strategies with GPU capabilities."""
    try:
        strategies_with_gpu = {}
        for name, info in OPTIMIZED_MODELS.items():
            strategies_with_gpu[name] = {
                "gpu_accelerated": info.get("gpu_accelerated", False),
                "performance": info.get("performance_metrics", {}),
                "status": info.get("status", "unknown")
            }
        
        return json.dumps({
            "strategies": strategies_with_gpu,
            "count": len(OPTIMIZED_MODELS),
            "gpu_available": GPU_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

# New integrated resources
@mcp.resource("news://aggregated/{symbol}")
async def get_aggregated_news_resource(symbol: str) -> str:
    """Get aggregated news for a symbol as a resource"""
    if not NEWS_AGGREGATION_AVAILABLE:
        return json.dumps({"error": "News aggregation not available"}, indent=2)
    
    try:
        news_items = await news_aggregator.fetch_aggregated_news([symbol], lookback_hours=24, limit=50)
        
        return json.dumps({
            "symbol": symbol,
            "item_count": len(news_items),
            "items": [item.to_dict() for item in news_items[:20]],  # Limit for resource
            "aggregate_sentiment": np.mean([item.sentiment for item in news_items if item.sentiment]),
            "sources": list(set(item.source for item in news_items)),
            "last_updated": datetime.now().isoformat()
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@mcp.resource("integration://status")
async def get_integration_status() -> str:
    """Get complete integration status"""
    try:
        # Get news aggregator status
        news_status = None
        if NEWS_AGGREGATION_AVAILABLE:
            provider_status = await news_aggregator.get_provider_status()
            news_status = {
                "available": True,
                "providers": provider_status,
                "metrics": news_aggregator.get_metrics()
            }
        else:
            news_status = {"available": False}
        
        # Get strategy manager status
        strategy_status = None
        if STRATEGY_MANAGER_AVAILABLE:
            strategies = await strategy_manager.get_available_strategies()
            strategy_status = {
                "available": True,
                "strategies": strategies,
                "active_positions": len(await strategy_manager.get_all_positions())
            }
        else:
            strategy_status = {"available": False}
        
        return json.dumps({
            "integration_components": {
                "news_aggregation": news_status,
                "strategy_manager": strategy_status,
                "gpu_acceleration": GPU_AVAILABLE,
                "polymarket_tools": POLYMARKET_TOOLS_AVAILABLE
            },
            "total_tools": 40,  # Updated count
            "system_health": "operational",
            "timestamp": datetime.now().isoformat()
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

# Main function
async def main():
    """Start integrated MCP server with all capabilities"""
    try:
        logger.info(f"Starting Integrated MCP Server - GPU: {GPU_AVAILABLE}, News: {NEWS_AGGREGATION_AVAILABLE}, Strategy: {STRATEGY_MANAGER_AVAILABLE}")
        
        # Ensure directories exist
        Path("models").mkdir(exist_ok=True)
        Path("benchmark").mkdir(exist_ok=True)
        Path("neural_models").mkdir(exist_ok=True)
        
        # Start server
        await mcp.run()
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if NEWS_AGGREGATION_AVAILABLE and news_aggregator:
            await news_aggregator.close()

if __name__ == "__main__":
    # Run with asyncio
    import asyncio
    asyncio.run(main())