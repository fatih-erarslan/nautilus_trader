#!/usr/bin/env python3
"""
Python API Examples for AI News Trading Platform with Neural Forecasting

This file contains comprehensive examples of using the Python API for:
- Neural forecasting integration
- Trading strategy development
- MCP server interaction
- Portfolio optimization
- Risk management
- Real-time trading

Usage:
    python python_api.py --example basic_forecast
    python python_api.py --example trading_integration
    python python_api.py --example portfolio_optimization
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Neural forecasting imports
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, NBEATSx

# MCP client imports
import aiohttp
import requests

# Example 1: Basic Neural Forecasting
def example_basic_neural_forecasting():
    """Basic neural forecasting example with NHITS model"""
    
    print("=" * 60)
    print("EXAMPLE 1: Basic Neural Forecasting with NHITS")
    print("=" * 60)
    
    # Step 1: Create sample financial data
    print("Step 1: Creating sample financial data...")
    
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-06-01', freq='D')
    
    # Simulate realistic stock price movement
    initial_price = 150.0
    returns = np.random.normal(0.0008, 0.02, len(dates))  # 0.08% daily return, 2% volatility
    prices = [initial_price]
    
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    # Add trend and seasonality
    trend = np.linspace(0, 30, len(dates))
    seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    prices = np.array(prices) + trend + seasonal
    
    # Create neural forecasting compatible dataframe
    data = pd.DataFrame({
        'ds': dates,
        'unique_id': 'SAMPLE_STOCK',
        'y': prices
    })
    
    print(f"✓ Created {len(data)} days of sample data")
    print(f"Price range: ${data['y'].min():.2f} - ${data['y'].max():.2f}")
    
    # Step 2: Initialize neural forecasting model
    print("\nStep 2: Initializing NHITS neural forecasting model...")
    
    model = NHITS(
        input_size=56,        # Look back 8 weeks
        h=30,                 # Forecast 30 days ahead
        max_epochs=50,        # Quick training for example
        batch_size=32,        
        learning_rate=1e-3,
        
        # NHITS architecture
        n_freq_downsample=[168, 24, 1],
        stack_types=['trend', 'seasonality'],
        n_blocks=[1, 1],
        mlp_units=[[256, 256], [256, 256]],
        
        # Performance settings
        accelerator='auto',    # Use GPU if available
        enable_progress_bar=True,
        alias='NHITS_Example'
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    print("✓ Neural forecasting model initialized")
    
    # Step 3: Train the model
    print("\nStep 3: Training the neural forecasting model...")
    
    # Split data for training and testing
    train_size = int(len(data) * 0.9)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"Training data: {len(train_data)} days")
    print(f"Test data: {len(test_data)} days")
    
    # Train the model
    start_time = datetime.now()
    nf.fit(train_data)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✓ Model trained in {training_time:.1f} seconds")
    
    # Step 4: Generate forecasts
    print("\nStep 4: Generating 30-day forecasts...")
    
    forecasts = nf.predict(h=30, level=[80, 95])
    
    print(f"✓ Generated forecasts with shape: {forecasts.shape}")
    print(f"Forecast columns: {list(forecasts.columns)}")
    
    # Step 5: Evaluate forecast accuracy
    print("\nStep 5: Evaluating forecast accuracy...")
    
    # Compare with actual test data
    comparison = forecasts.merge(test_data[['ds', 'y']], on='ds', how='inner')
    
    if len(comparison) > 0:
        actual = comparison['y']
        predicted = comparison['NHITS_Example']
        
        # Calculate accuracy metrics
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Root Mean Square Error (RMSE): ${rmse:.2f}")
        
        # Directional accuracy
        actual_direction = (actual.diff() > 0).dropna()
        predicted_direction = (predicted.diff() > 0).dropna()
        
        if len(actual_direction) > 0:
            directional_accuracy = (actual_direction == predicted_direction).mean() * 100
            print(f"Directional Accuracy: {directional_accuracy:.1f}%")
    else:
        print("No overlapping dates for accuracy evaluation")
    
    # Step 6: Display forecast summary
    print("\nStep 6: Forecast Summary")
    print("-" * 40)
    
    forecast_start = forecasts['NHITS_Example'].iloc[0]
    forecast_end = forecasts['NHITS_Example'].iloc[-1]
    forecast_change = forecast_end - forecast_start
    forecast_percent = (forecast_change / forecast_start) * 100
    
    print(f"Starting forecast price: ${forecast_start:.2f}")
    print(f"Ending forecast price: ${forecast_end:.2f}")
    print(f"Predicted change: ${forecast_change:.2f} ({forecast_percent:.1f}%)")
    
    # Confidence intervals
    if 'NHITS_Example-lo-95' in forecasts.columns:
        ci_lower = forecasts['NHITS_Example-lo-95'].iloc[-1]
        ci_upper = forecasts['NHITS_Example-hi-95'].iloc[-1]
        print(f"95% Confidence Interval: ${ci_lower:.2f} - ${ci_upper:.2f}")
    
    return {
        'model': nf,
        'forecasts': forecasts,
        'training_time': training_time,
        'accuracy_metrics': {
            'mae': mae if 'mae' in locals() else None,
            'mape': mape if 'mape' in locals() else None,
            'rmse': rmse if 'rmse' in locals() else None
        }
    }


# Example 2: Trading Strategy Integration
def example_trading_strategy_integration():
    """Integration of neural forecasting with trading strategies"""
    
    print("=" * 60)
    print("EXAMPLE 2: Neural Forecasting + Trading Strategy Integration")
    print("=" * 60)
    
    # Import trading strategy components
    try:
        from trading.strategies.momentum_trader import MomentumEngine
        from trading.strategies.enhanced_momentum_trader import EnhancedMomentumTrader
    except ImportError:
        print("⚠️  Trading strategy modules not found, using mock implementations")
        
        # Mock trading strategy for example
        class MockMomentumEngine:
            def __init__(self):
                self.lookback_periods = [3, 11, 33]
                self.momentum_thresholds = {'strong': 0.40, 'moderate': 0.35, 'weak': 0.12}
            
            def generate_signals(self, symbol, data):
                # Mock signal generation
                price_change = data['y'].iloc[-1] - data['y'].iloc[-20]
                momentum_score = price_change / data['y'].iloc[-20]
                
                if momentum_score > 0.05:
                    return {'action': 'BUY', 'confidence': 0.7, 'momentum': momentum_score}
                elif momentum_score < -0.05:
                    return {'action': 'SELL', 'confidence': 0.6, 'momentum': momentum_score}
                else:
                    return {'action': 'HOLD', 'confidence': 0.4, 'momentum': momentum_score}
        
        MomentumEngine = MockMomentumEngine
    
    # Step 1: Create enhanced trading strategy with neural forecasting
    print("Step 1: Creating neural-enhanced trading strategy...")
    
    class NeuralEnhancedMomentumStrategy:
        """Momentum strategy enhanced with neural forecasting"""
        
        def __init__(self, forecast_weight=0.3):
            self.momentum_engine = MomentumEngine()
            self.forecast_engine = None
            self.forecast_weight = forecast_weight
            
        def set_forecast_engine(self, forecast_engine):
            """Set the neural forecasting engine"""
            self.forecast_engine = forecast_engine
            
        def generate_enhanced_signals(self, symbol, data, forecast_horizon=5):
            """Generate trading signals using both momentum and neural forecasts"""
            
            # Get traditional momentum signals
            momentum_signals = self.momentum_engine.generate_signals(symbol, data)
            
            # Get neural forecast if available
            neural_signals = {'confidence': 0, 'expected_return': 0}
            
            if self.forecast_engine:
                try:
                    # Generate short-term forecast
                    forecast = self.forecast_engine.predict(h=forecast_horizon)
                    
                    if len(forecast) > 0:
                        current_price = data['y'].iloc[-1]
                        future_price = forecast['NHITS_Example'].iloc[-1]
                        expected_return = (future_price - current_price) / current_price
                        
                        # Neural signal confidence based on expected return magnitude
                        neural_confidence = min(abs(expected_return) * 10, 1.0)
                        
                        neural_signals = {
                            'expected_return': expected_return,
                            'confidence': neural_confidence,
                            'future_price': future_price
                        }
                        
                except Exception as e:
                    print(f"Neural forecasting failed: {e}")
            
            # Combine signals
            momentum_weight = 1 - self.forecast_weight
            
            # Enhanced confidence combines both signals
            enhanced_confidence = (
                momentum_signals['confidence'] * momentum_weight +
                neural_signals['confidence'] * self.forecast_weight
            )
            
            # Action based on combined signals
            momentum_score = momentum_signals.get('momentum', 0)
            neural_score = neural_signals.get('expected_return', 0)
            
            combined_score = (momentum_score * momentum_weight + 
                            neural_score * self.forecast_weight)
            
            if combined_score > 0.03 and enhanced_confidence > 0.6:
                action = 'BUY'
            elif combined_score < -0.03 and enhanced_confidence > 0.6:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            return {
                'action': action,
                'confidence': enhanced_confidence,
                'momentum_signals': momentum_signals,
                'neural_signals': neural_signals,
                'combined_score': combined_score,
                'reasoning': f"Momentum: {momentum_score:.3f}, Neural: {neural_score:.3f}"
            }
    
    # Step 2: Set up sample data and neural model
    print("\nStep 2: Setting up neural forecasting model...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='D')
    prices = 200 + np.cumsum(np.random.normal(0.1, 2, len(dates)))
    
    symbol_data = pd.DataFrame({
        'ds': dates,
        'unique_id': 'ENHANCED_STOCK',
        'y': prices
    })
    
    # Quick neural model
    quick_model = NHITS(
        input_size=28,
        h=5,
        max_epochs=25,
        batch_size=16,
        enable_progress_bar=False,
        alias='NHITS_Example'
    )
    
    nf = NeuralForecast(models=[quick_model], freq='D')
    nf.fit(symbol_data)
    
    print("✓ Neural forecasting model trained")
    
    # Step 3: Create enhanced strategy and test
    print("\nStep 3: Testing neural-enhanced trading strategy...")
    
    enhanced_strategy = NeuralEnhancedMomentumStrategy(forecast_weight=0.4)
    enhanced_strategy.set_forecast_engine(nf)
    
    # Generate signals for last 30 days
    test_dates = symbol_data.tail(30)
    signals_history = []
    
    for i in range(10, len(test_dates)):
        current_data = symbol_data.iloc[:len(symbol_data) - len(test_dates) + i]
        
        signals = enhanced_strategy.generate_enhanced_signals(
            'ENHANCED_STOCK', current_data, forecast_horizon=5
        )
        
        signals['date'] = test_dates.iloc[i]['ds']
        signals['price'] = test_dates.iloc[i]['y']
        signals_history.append(signals)
    
    # Step 4: Display results
    print("\nStep 4: Enhanced Trading Signals Results")
    print("-" * 50)
    
    buy_signals = sum(1 for s in signals_history if s['action'] == 'BUY')
    sell_signals = sum(1 for s in signals_history if s['action'] == 'SELL')
    hold_signals = sum(1 for s in signals_history if s['action'] == 'HOLD')
    
    print(f"Total signals generated: {len(signals_history)}")
    print(f"BUY signals: {buy_signals}")
    print(f"SELL signals: {sell_signals}")
    print(f"HOLD signals: {hold_signals}")
    
    # Show recent signals
    print(f"\nRecent trading signals:")
    for signal in signals_history[-5:]:
        print(f"{signal['date'].strftime('%Y-%m-%d')}: {signal['action']} "
              f"(Confidence: {signal['confidence']:.2f}, "
              f"Price: ${signal['price']:.2f})")
        print(f"  Reasoning: {signal['reasoning']}")
    
    # Calculate performance metrics
    avg_confidence = np.mean([s['confidence'] for s in signals_history])
    neural_contribution = np.mean([
        abs(s['neural_signals'].get('expected_return', 0)) 
        for s in signals_history
    ])
    
    print(f"\nStrategy Performance:")
    print(f"Average signal confidence: {avg_confidence:.2f}")
    print(f"Neural forecast contribution: {neural_contribution:.3f}")
    
    return {
        'strategy': enhanced_strategy,
        'signals_history': signals_history,
        'performance_metrics': {
            'avg_confidence': avg_confidence,
            'neural_contribution': neural_contribution,
            'signal_distribution': {
                'buy': buy_signals,
                'sell': sell_signals,
                'hold': hold_signals
            }
        }
    }


# Example 3: MCP Server Integration
async def example_mcp_server_integration():
    """Example of integrating with MCP server for trading operations"""
    
    print("=" * 60)
    print("EXAMPLE 3: MCP Server Integration")
    print("=" * 60)
    
    # MCP server configuration
    MCP_SERVER_URL = "http://localhost:3000/mcp"
    
    # Step 1: Test MCP server connectivity
    print("Step 1: Testing MCP server connectivity...")
    
    async def call_mcp_tool(method, params=None, timeout=30):
        """Call MCP server tool"""
        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    MCP_SERVER_URL, 
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {
                            "error": f"HTTP {response.status}",
                            "message": await response.text()
                        }
        except aiohttp.ClientConnectorError:
            return {
                "error": "connection_failed",
                "message": "Could not connect to MCP server. Is it running?"
            }
        except asyncio.TimeoutError:
            return {
                "error": "timeout",
                "message": f"Request timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "error": "unknown",
                "message": str(e)
            }
    
    # Test server ping
    ping_result = await call_mcp_tool("ping")
    
    if "error" in ping_result:
        print(f"❌ MCP server not available: {ping_result['message']}")
        print("Please start the MCP server with: python mcp_server_enhanced.py")
        return None
    else:
        print(f"✓ MCP server is running")
        if "result" in ping_result:
            print(f"Server info: {ping_result['result']}")
    
    # Step 2: List available trading strategies
    print("\nStep 2: Listing available trading strategies...")
    
    strategies_result = await call_mcp_tool("list_strategies")
    
    if "error" in strategies_result:
        print(f"❌ Failed to list strategies: {strategies_result['message']}")
    else:
        strategies = strategies_result.get("result", {}).get("strategies", [])
        print(f"✓ Found {len(strategies)} available strategies:")
        
        for strategy in strategies:
            print(f"  - {strategy['name']}: {strategy['description']}")
            if strategy.get('gpu_accelerated'):
                print(f"    GPU Speedup: {strategy.get('speedup', 'N/A')}")
    
    # Step 3: Get detailed strategy information
    print("\nStep 3: Getting detailed strategy information...")
    
    if strategies:
        strategy_name = strategies[0]['name']
        strategy_info_result = await call_mcp_tool("get_strategy_info", {
            "strategy": strategy_name
        })
        
        if "error" in strategy_info_result:
            print(f"❌ Failed to get strategy info: {strategy_info_result['message']}")
        else:
            info = strategy_info_result.get("result", {})
            print(f"✓ Strategy: {strategy_name}")
            
            if "parameters" in info:
                print("  Parameters:")
                for key, value in info["parameters"].items():
                    print(f"    {key}: {value}")
            
            if "performance" in info:
                print("  Performance Metrics:")
                for key, value in info["performance"].items():
                    print(f"    {key}: {value}")
    
    # Step 4: Perform quick market analysis
    print("\nStep 4: Performing quick market analysis...")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    analysis_results = []
    
    for symbol in symbols:
        analysis_result = await call_mcp_tool("quick_analysis", {
            "symbol": symbol,
            "use_gpu": True
        })
        
        if "error" in analysis_result:
            print(f"❌ Analysis failed for {symbol}: {analysis_result['message']}")
        else:
            result = analysis_result.get("result", {})
            analysis_results.append(result)
            
            print(f"✓ {symbol} Analysis:")
            analysis = result.get("analysis", {})
            print(f"  Trend: {analysis.get('trend', 'N/A')}")
            print(f"  Momentum: {analysis.get('momentum', 'N/A')}")
            print(f"  Volatility: {analysis.get('volatility', 'N/A')}")
            
            neural_forecast = result.get("neural_forecast", {})
            if neural_forecast:
                print(f"  Neural Forecast:")
                print(f"    Next Day: ${neural_forecast.get('next_day', 'N/A')}")
                print(f"    Confidence: {neural_forecast.get('confidence', 'N/A')}")
    
    # Step 5: Simulate trading operations
    print("\nStep 5: Simulating trading operations...")
    
    if strategies and analysis_results:
        strategy_name = strategies[0]['name']
        
        for symbol in symbols[:2]:  # Test with first 2 symbols
            simulation_result = await call_mcp_tool("simulate_trade", {
                "strategy": strategy_name,
                "symbol": symbol,
                "action": "buy",
                "use_gpu": True
            })
            
            if "error" in simulation_result:
                print(f"❌ Simulation failed for {symbol}: {simulation_result['message']}")
            else:
                result = simulation_result.get("result", {})
                print(f"✓ {symbol} Trade Simulation:")
                print(f"  Action: {result.get('action', 'N/A')}")
                print(f"  Entry Price: ${result.get('entry_price', 'N/A')}")
                print(f"  Position Size: {result.get('position_size', 'N/A')}")
                print(f"  Expected Return: {result.get('expected_return', 'N/A')}")
                
                neural_support = result.get("neural_forecast_support", {})
                if neural_support:
                    print(f"  Neural Forecast Alignment: {neural_support.get('forecast_alignment', 'N/A')}")
    
    # Step 6: Get portfolio status
    print("\nStep 6: Getting portfolio status...")
    
    portfolio_result = await call_mcp_tool("get_portfolio_status", {
        "include_analytics": True
    })
    
    if "error" in portfolio_result:
        print(f"❌ Failed to get portfolio status: {portfolio_result['message']}")
    else:
        portfolio = portfolio_result.get("result", {})
        portfolio_info = portfolio.get("portfolio", {})
        
        print(f"✓ Portfolio Status:")
        print(f"  Total Value: ${portfolio_info.get('total_value', 'N/A')}")
        print(f"  Cash: ${portfolio_info.get('cash', 'N/A')}")
        
        positions = portfolio_info.get("positions", [])
        if positions:
            print(f"  Positions ({len(positions)}):")
            for position in positions[:3]:  # Show first 3 positions
                print(f"    {position.get('symbol', 'N/A')}: {position.get('quantity', 'N/A')} shares")
                print(f"      Value: ${position.get('market_value', 'N/A')}")
                print(f"      P&L: ${position.get('unrealized_pnl', 'N/A')}")
        
        analytics = portfolio.get("analytics", {})
        if analytics:
            print(f"  Analytics:")
            print(f"    Total Return: {analytics.get('total_return', 'N/A')}")
            print(f"    Sharpe Ratio: {analytics.get('sharpe_ratio', 'N/A')}")
            print(f"    Max Drawdown: {analytics.get('max_drawdown', 'N/A')}")
    
    return {
        'server_status': 'connected',
        'available_strategies': strategies if 'strategies' in locals() else [],
        'analysis_results': analysis_results,
        'portfolio_status': portfolio if 'portfolio' in locals() else {}
    }


# Example 4: Portfolio Optimization with Neural Forecasts
def example_portfolio_optimization():
    """Portfolio optimization using neural forecasting"""
    
    print("=" * 60)
    print("EXAMPLE 4: Portfolio Optimization with Neural Forecasts")
    print("=" * 60)
    
    # Import optimization libraries
    from scipy.optimize import minimize
    
    # Step 1: Create multi-asset portfolio data
    print("Step 1: Creating multi-asset portfolio data...")
    
    symbols = ['TECH_STOCK', 'FINANCE_STOCK', 'HEALTHCARE_STOCK', 'ENERGY_STOCK']
    portfolio_data = []
    
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-06-01', freq='D')
    
    # Create correlated asset prices
    market_factor = np.random.normal(0.0005, 0.015, len(dates))  # Market risk
    
    for i, symbol in enumerate(symbols):
        # Sector-specific parameters
        if 'TECH' in symbol:
            beta, alpha, volatility = 1.2, 0.0010, 0.025
        elif 'FINANCE' in symbol:
            beta, alpha, volatility = 1.0, 0.0008, 0.020
        elif 'HEALTHCARE' in symbol:
            beta, alpha, volatility = 0.8, 0.0012, 0.018
        else:  # ENERGY
            beta, alpha, volatility = 1.1, 0.0006, 0.030
        
        # Generate correlated returns
        idiosyncratic = np.random.normal(alpha, volatility, len(dates))
        returns = beta * market_factor + idiosyncratic
        
        # Convert to prices
        initial_price = 100 + i * 50  # Different starting prices
        prices = [initial_price]
        
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        
        # Create DataFrame
        symbol_data = pd.DataFrame({
            'ds': dates,
            'unique_id': symbol,
            'y': prices
        })
        
        portfolio_data.append(symbol_data)
    
    # Combine all assets
    full_portfolio_data = pd.concat(portfolio_data, ignore_index=True)
    
    print(f"✓ Created portfolio with {len(symbols)} assets")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # Step 2: Train neural forecasting models for each asset
    print("\nStep 2: Training neural forecasting models...")
    
    trained_models = {}
    forecast_results = {}
    
    for symbol in symbols:
        print(f"  Training model for {symbol}...")
        
        symbol_data = full_portfolio_data[
            full_portfolio_data['unique_id'] == symbol
        ].copy()
        
        # Create model
        model = NHITS(
            input_size=42,  # 6 weeks lookback
            h=21,           # 3 weeks forecast
            max_epochs=30,  # Quick training for example
            batch_size=16,
            enable_progress_bar=False,
            alias=f'NHITS_{symbol}'
        )
        
        nf = NeuralForecast(models=[model], freq='D')
        
        # Train model
        train_data = symbol_data.iloc[:-21]  # Hold out last 3 weeks
        nf.fit(train_data)
        
        # Generate forecasts
        forecasts = nf.predict(h=21)
        
        trained_models[symbol] = nf
        forecast_results[symbol] = forecasts
        
        print(f"    ✓ {symbol} model trained and forecasted")
    
    # Step 3: Calculate expected returns and covariance
    print("\nStep 3: Calculating expected returns and risk metrics...")
    
    expected_returns = {}
    current_prices = {}
    
    for symbol in symbols:
        symbol_data = full_portfolio_data[
            full_portfolio_data['unique_id'] == symbol
        ].tail(1)
        
        current_price = symbol_data['y'].iloc[0]
        current_prices[symbol] = current_price
        
        # Get forecast
        forecasts = forecast_results[symbol]
        forecast_price = forecasts[f'NHITS_{symbol}'].iloc[-1]
        
        # Calculate expected return
        expected_return = (forecast_price - current_price) / current_price
        expected_returns[symbol] = expected_return
        
        print(f"  {symbol}:")
        print(f"    Current: ${current_price:.2f}")
        print(f"    Forecast: ${forecast_price:.2f}")
        print(f"    Expected Return: {expected_return:.2%}")
    
    # Calculate historical covariance matrix
    returns_data = []
    
    for symbol in symbols:
        symbol_data = full_portfolio_data[
            full_portfolio_data['unique_id'] == symbol
        ].copy()
        symbol_returns = symbol_data['y'].pct_change().dropna()
        returns_data.append(symbol_returns)
    
    returns_matrix = pd.concat(returns_data, axis=1)
    returns_matrix.columns = symbols
    covariance_matrix = returns_matrix.cov()
    
    print(f"\n  Correlation matrix:")
    correlation_matrix = returns_matrix.corr()
    for i, symbol1 in enumerate(symbols):
        for j, symbol2 in enumerate(symbols):
            if i <= j:
                corr = correlation_matrix.loc[symbol1, symbol2]
                print(f"    {symbol1} - {symbol2}: {corr:.3f}")
    
    # Step 4: Portfolio optimization
    print("\nStep 4: Optimizing portfolio weights...")
    
    # Convert to arrays for optimization
    expected_returns_array = np.array([expected_returns[symbol] for symbol in symbols])
    cov_matrix = covariance_matrix.values
    
    def portfolio_objective(weights, risk_aversion=1.0):
        """Portfolio optimization objective (negative utility)"""
        portfolio_return = np.sum(weights * expected_returns_array)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
        return -utility  # Minimize negative utility
    
    # Constraints and bounds
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
    ]
    bounds = tuple((0, 1) for _ in range(len(symbols)))  # No short selling
    
    # Initial guess (equal weights)
    initial_weights = np.array([1.0/len(symbols)] * len(symbols))
    
    # Optimize for different risk aversion levels
    risk_aversions = [0.5, 1.0, 2.0]
    optimization_results = {}
    
    for risk_aversion in risk_aversions:
        result = minimize(
            portfolio_objective,
            initial_weights,
            args=(risk_aversion,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(optimal_weights * expected_returns_array)
            portfolio_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            optimization_results[risk_aversion] = {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            }
            
            print(f"\n  Risk Aversion: {risk_aversion}")
            print(f"    Expected Return: {portfolio_return:.2%}")
            print(f"    Volatility: {portfolio_volatility:.2%}")
            print(f"    Sharpe Ratio: {sharpe_ratio:.3f}")
            print(f"    Optimal Weights:")
            
            for symbol, weight in zip(symbols, optimal_weights):
                print(f"      {symbol}: {weight:.1%}")
    
    # Step 5: Backtesting optimized portfolio
    print("\nStep 5: Backtesting optimized portfolio...")
    
    # Use moderate risk aversion portfolio
    if 1.0 in optimization_results:
        optimal_portfolio = optimization_results[1.0]
        optimal_weights = optimal_portfolio['weights']
        
        # Calculate portfolio performance over forecast period
        portfolio_values = []
        
        for symbol, weight in zip(symbols, optimal_weights):
            forecasts = forecast_results[symbol]
            forecast_returns = forecasts[f'NHITS_{symbol}'].pct_change().fillna(0)
            weighted_returns = forecast_returns * weight
            portfolio_values.append(weighted_returns)
        
        # Sum weighted returns
        portfolio_returns = sum(portfolio_values)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        final_return = cumulative_returns.iloc[-1] - 1
        volatility = portfolio_returns.std()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        print(f"  Backtest Results (Forecast Period):")
        print(f"    Total Return: {final_return:.2%}")
        print(f"    Annualized Volatility: {volatility * np.sqrt(252):.2%}")
        print(f"    Max Drawdown: {max_drawdown:.2%}")
        
        # Compare with equal-weight portfolio
        equal_weight = 1.0 / len(symbols)
        equal_weight_returns = sum([
            forecast_results[symbol][f'NHITS_{symbol}'].pct_change().fillna(0) * equal_weight
            for symbol in symbols
        ])
        equal_weight_final = (1 + equal_weight_returns).prod() - 1
        
        print(f"  Equal-Weight Benchmark:")
        print(f"    Total Return: {equal_weight_final:.2%}")
        print(f"    Outperformance: {(final_return - equal_weight_final):.2%}")
    
    return {
        'symbols': symbols,
        'trained_models': trained_models,
        'expected_returns': expected_returns,
        'optimization_results': optimization_results,
        'backtest_results': {
            'portfolio_return': final_return if 'final_return' in locals() else None,
            'benchmark_return': equal_weight_final if 'equal_weight_final' in locals() else None
        }
    }


# Example 5: Real-time Risk Management
def example_realtime_risk_management():
    """Real-time risk management with neural forecasting"""
    
    print("=" * 60)
    print("EXAMPLE 5: Real-time Risk Management")
    print("=" * 60)
    
    # Step 1: Create risk management framework
    print("Step 1: Setting up risk management framework...")
    
    class NeuralRiskManager:
        """Risk manager with neural forecasting capabilities"""
        
        def __init__(self, max_portfolio_risk=0.02, confidence_threshold=0.7):
            self.max_portfolio_risk = max_portfolio_risk  # 2% daily VaR
            self.confidence_threshold = confidence_threshold
            self.risk_metrics = {}
            
        def calculate_position_risk(self, symbol, position_size, price, forecast_data):
            """Calculate risk for individual position"""
            
            # Get forecast distribution
            if 'confidence_intervals' in forecast_data:
                # Use confidence intervals for risk calculation
                lower_bound = forecast_data['confidence_intervals']['5%']
                upper_bound = forecast_data['confidence_intervals']['95%']
                
                # Calculate potential losses
                worst_case_loss = (lower_bound - price) / price
                best_case_gain = (upper_bound - price) / price
                
                # Position risk metrics
                position_value = position_size * price
                var_95 = position_value * abs(worst_case_loss)
                
            else:
                # Fallback to volatility-based calculation
                volatility = forecast_data.get('volatility', 0.02)
                var_95 = position_value * 1.65 * volatility  # 95% VaR
                worst_case_loss = -1.65 * volatility
                best_case_gain = 1.65 * volatility
            
            return {
                'symbol': symbol,
                'position_value': position_value,
                'var_95': var_95,
                'worst_case_loss': worst_case_loss,
                'best_case_gain': best_case_gain,
                'risk_contribution': var_95
            }
        
        def calculate_portfolio_risk(self, positions, correlations=None):
            """Calculate portfolio-level risk"""
            
            total_value = sum(pos['position_value'] for pos in positions)
            total_var = sum(pos['var_95'] for pos in positions)
            
            # Adjust for correlations if available
            if correlations is not None and len(positions) > 1:
                # Simple correlation adjustment (full implementation would use covariance matrix)
                avg_correlation = np.mean([correlations.get(f"{p1['symbol']}_{p2['symbol']}", 0.5) 
                                         for p1 in positions for p2 in positions if p1 != p2])
                diversification_factor = 1 - (1 - avg_correlation) * 0.3
                total_var *= diversification_factor
            
            portfolio_var_pct = total_var / total_value if total_value > 0 else 0
            
            return {
                'total_portfolio_value': total_value,
                'portfolio_var_dollar': total_var,
                'portfolio_var_percent': portfolio_var_pct,
                'risk_limit_utilization': portfolio_var_pct / self.max_portfolio_risk,
                'within_limits': portfolio_var_pct <= self.max_portfolio_risk
            }
        
        def generate_risk_alerts(self, portfolio_risk, position_risks):
            """Generate risk alerts and recommendations"""
            
            alerts = []
            
            # Portfolio level alerts
            if not portfolio_risk['within_limits']:
                alerts.append({
                    'level': 'CRITICAL',
                    'type': 'PORTFOLIO_RISK_LIMIT',
                    'message': f"Portfolio VaR ({portfolio_risk['portfolio_var_percent']:.2%}) exceeds limit ({self.max_portfolio_risk:.2%})",
                    'action': 'REDUCE_POSITIONS'
                })
            
            # Position level alerts
            for pos in position_risks:
                risk_contrib_pct = pos['risk_contribution'] / portfolio_risk['total_portfolio_value']
                
                if risk_contrib_pct > 0.005:  # 0.5% of portfolio
                    alerts.append({
                        'level': 'WARNING',
                        'type': 'HIGH_POSITION_RISK',
                        'message': f"{pos['symbol']} contributes {risk_contrib_pct:.2%} to portfolio risk",
                        'symbol': pos['symbol'],
                        'action': 'MONITOR_CLOSELY'
                    })
                
                if abs(pos['worst_case_loss']) > 0.05:  # 5% potential loss
                    alerts.append({
                        'level': 'HIGH',
                        'type': 'HIGH_POSITION_LOSS_POTENTIAL',
                        'message': f"{pos['symbol']} has {pos['worst_case_loss']:.2%} worst-case loss potential",
                        'symbol': pos['symbol'],
                        'action': 'CONSIDER_HEDGING'
                    })
            
            return alerts
    
    # Step 2: Set up sample portfolio with neural forecasts
    print("\nStep 2: Creating sample portfolio with neural forecasts...")
    
    # Sample portfolio positions
    portfolio_positions = [
        {'symbol': 'TECH_A', 'shares': 100, 'price': 150.0},
        {'symbol': 'FINANCE_B', 'shares': 200, 'price': 75.0},
        {'symbol': 'HEALTHCARE_C', 'shares': 150, 'price': 120.0},
        {'symbol': 'ENERGY_D', 'shares': 300, 'price': 45.0},
    ]
    
    # Mock neural forecast data with confidence intervals
    neural_forecasts = {
        'TECH_A': {
            'next_day_price': 148.5,
            'confidence': 0.82,
            'volatility': 0.025,
            'confidence_intervals': {
                '5%': 142.0,
                '25%': 146.0,
                '75%': 152.0,
                '95%': 157.0
            }
        },
        'FINANCE_B': {
            'next_day_price': 76.2,
            'confidence': 0.75,
            'volatility': 0.018,
            'confidence_intervals': {
                '5%': 72.0,
                '25%': 74.5,
                '75%': 77.5,
                '95%': 80.0
            }
        },
        'HEALTHCARE_C': {
            'next_day_price': 118.8,
            'confidence': 0.88,
            'volatility': 0.015,
            'confidence_intervals': {
                '5%': 115.0,
                '25%': 117.0,
                '75%': 121.0,
                '95%': 124.0
            }
        },
        'ENERGY_D': {
            'next_day_price': 47.1,
            'confidence': 0.65,
            'volatility': 0.035,
            'confidence_intervals': {
                '5%': 41.0,
                '25%': 44.0,
                '75%': 49.0,
                '95%': 52.0
            }
        }
    }
    
    # Mock correlations
    correlations = {
        'TECH_A_FINANCE_B': 0.4,
        'TECH_A_HEALTHCARE_C': 0.3,
        'TECH_A_ENERGY_D': 0.2,
        'FINANCE_B_HEALTHCARE_C': 0.5,
        'FINANCE_B_ENERGY_D': 0.3,
        'HEALTHCARE_C_ENERGY_D': 0.1
    }
    
    print(f"✓ Portfolio with {len(portfolio_positions)} positions created")
    
    # Step 3: Initialize risk manager and calculate risks
    print("\nStep 3: Calculating portfolio risk metrics...")
    
    risk_manager = NeuralRiskManager(max_portfolio_risk=0.02, confidence_threshold=0.7)
    
    # Calculate individual position risks
    position_risks = []
    for position in portfolio_positions:
        symbol = position['symbol']
        position_size = position['shares']
        price = position['price']
        forecast_data = neural_forecasts[symbol]
        
        position_risk = risk_manager.calculate_position_risk(
            symbol, position_size, price, forecast_data
        )
        position_risks.append(position_risk)
    
    # Calculate portfolio risk
    portfolio_risk = risk_manager.calculate_portfolio_risk(position_risks, correlations)
    
    # Generate alerts
    risk_alerts = risk_manager.generate_risk_alerts(portfolio_risk, position_risks)
    
    # Step 4: Display risk analysis results
    print("\nStep 4: Risk Analysis Results")
    print("-" * 40)
    
    print(f"Portfolio Overview:")
    print(f"  Total Value: ${portfolio_risk['total_portfolio_value']:,.2f}")
    print(f"  Portfolio VaR (95%): ${portfolio_risk['portfolio_var_dollar']:,.2f}")
    print(f"  Portfolio VaR (%): {portfolio_risk['portfolio_var_percent']:.2%}")
    print(f"  Risk Limit Utilization: {portfolio_risk['risk_limit_utilization']:.1%}")
    print(f"  Within Limits: {'✓' if portfolio_risk['within_limits'] else '❌'}")
    
    print(f"\nPosition Risk Breakdown:")
    for risk in position_risks:
        print(f"  {risk['symbol']}:")
        print(f"    Position Value: ${risk['position_value']:,.2f}")
        print(f"    VaR (95%): ${risk['var_95']:,.2f}")
        print(f"    Worst Case Loss: {risk['worst_case_loss']:.2%}")
        print(f"    Best Case Gain: {risk['best_case_gain']:.2%}")
    
    # Display alerts
    if risk_alerts:
        print(f"\n🚨 Risk Alerts ({len(risk_alerts)}):")
        for alert in risk_alerts:
            level_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'WARNING': '🟡'}.get(alert['level'], '⚪')
            print(f"  {level_emoji} {alert['level']}: {alert['message']}")
            print(f"    Recommended Action: {alert['action']}")
    else:
        print(f"\n✅ No risk alerts - portfolio within acceptable limits")
    
    # Step 5: Risk-adjusted position sizing recommendations
    print("\nStep 5: Risk-adjusted position sizing recommendations...")
    
    def calculate_optimal_position_size(symbol, forecast_data, max_position_risk=0.005):
        """Calculate optimal position size based on risk budget"""
        
        current_price = next(p['price'] for p in portfolio_positions if p['symbol'] == symbol)
        worst_case_loss_pct = abs(
            (forecast_data['confidence_intervals']['5%'] - current_price) / current_price
        )
        
        if worst_case_loss_pct > 0:
            # Position size that limits risk to max_position_risk of portfolio
            max_position_value = (max_position_risk * portfolio_risk['total_portfolio_value']) / worst_case_loss_pct
            max_shares = int(max_position_value / current_price)
            
            return {
                'symbol': symbol,
                'current_shares': next(p['shares'] for p in portfolio_positions if p['symbol'] == symbol),
                'recommended_shares': max_shares,
                'adjustment_needed': max_shares - next(p['shares'] for p in portfolio_positions if p['symbol'] == symbol),
                'risk_budget_used': worst_case_loss_pct * max_shares * current_price / portfolio_risk['total_portfolio_value']
            }
        
        return None
    
    print(f"Risk-adjusted position recommendations:")
    for symbol in neural_forecasts.keys():
        recommendation = calculate_optimal_position_size(symbol, neural_forecasts[symbol])
        
        if recommendation:
            print(f"  {symbol}:")
            print(f"    Current: {recommendation['current_shares']} shares")
            print(f"    Recommended: {recommendation['recommended_shares']} shares")
            
            if recommendation['adjustment_needed'] != 0:
                action = "REDUCE" if recommendation['adjustment_needed'] < 0 else "INCREASE"
                print(f"    Action: {action} by {abs(recommendation['adjustment_needed'])} shares")
            else:
                print(f"    Action: HOLD (optimal size)")
            
            print(f"    Risk Budget Usage: {recommendation['risk_budget_used']:.2%}")
    
    return {
        'portfolio_risk': portfolio_risk,
        'position_risks': position_risks,
        'risk_alerts': risk_alerts,
        'neural_forecasts': neural_forecasts,
        'risk_manager': risk_manager
    }


def main():
    """Main function to run examples"""
    
    parser = argparse.ArgumentParser(description='AI News Trading Platform Python API Examples')
    parser.add_argument('--example', 
                       choices=['basic_forecast', 'trading_integration', 'mcp_server', 
                               'portfolio_optimization', 'risk_management', 'all'],
                       default='all',
                       help='Example to run')
    
    args = parser.parse_args()
    
    print("🚀 AI News Trading Platform - Python API Examples")
    print("=" * 70)
    
    results = {}
    
    try:
        if args.example in ['basic_forecast', 'all']:
            results['basic_forecast'] = example_basic_neural_forecasting()
            
        if args.example in ['trading_integration', 'all']:
            results['trading_integration'] = example_trading_strategy_integration()
            
        if args.example in ['mcp_server', 'all']:
            results['mcp_server'] = asyncio.run(example_mcp_server_integration())
            
        if args.example in ['portfolio_optimization', 'all']:
            results['portfolio_optimization'] = example_portfolio_optimization()
            
        if args.example in ['risk_management', 'all']:
            results['risk_management'] = example_realtime_risk_management()
        
        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)
        
        # Summary of results
        print("\n📊 Results Summary:")
        for example_name, result in results.items():
            if result:
                print(f"  ✓ {example_name.replace('_', ' ').title()}: Completed")
            else:
                print(f"  ❌ {example_name.replace('_', ' ').title()}: Failed or skipped")
        
        print(f"\n📖 For more information, see:")
        print(f"  - API Documentation: docs/api/")
        print(f"  - User Guides: docs/guides/")
        print(f"  - Tutorials: docs/tutorials/")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()