#!/usr/bin/env python3
"""
Neural Momentum Trading Strategy - Example Usage
Comprehensive example showing how to use the momentum trading system
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from strategies.momentum.strategy_orchestrator import StrategyOrchestrator
from strategies.momentum.neural_momentum_strategy import NeuralMomentumStrategy
from strategies.momentum.backtesting_engine import BacktestingEngine, BacktestConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def example_basic_usage():
    """Basic usage example"""
    logger.info("=== Basic Neural Momentum Strategy Example ===")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'neural_momentum_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create strategy instance
    strategy = NeuralMomentumStrategy(config['strategy'])
    
    # Analyze market conditions
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    logger.info(f"Analyzing market conditions for {symbols}")
    
    regime = await strategy.analyze_market_conditions(symbols)
    logger.info(f"Market regime: {regime}")
    
    # Generate trading signals
    logger.info("Generating trading signals...")
    signals = await strategy.generate_signals(symbols)
    
    logger.info(f"Generated {len(signals)} signals")
    for signal in signals:
        logger.info(f"Signal: {signal.direction} {signal.symbol} @ {signal.entry_price:.2f} "
                   f"(confidence: {signal.confidence:.2f}, strength: {signal.strength:.2f})")

async def example_backtesting():
    """Backtesting example"""
    logger.info("\n=== Backtesting Example ===")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'neural_momentum_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create orchestrator
    orchestrator = StrategyOrchestrator(config)
    
    # Run backtest
    backtest_config = {
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'symbols': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'],
        'benchmark_symbol': 'SPY',
        'transaction_costs': 0.001,
        'slippage': 0.0005
    }
    
    logger.info("Running backtest...")
    results = await orchestrator.run_backtest(backtest_config)
    
    if results['status'] == 'completed':
        summary = results['summary']
        logger.info("Backtest Results:")
        logger.info(f"  Total Return: {summary['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {summary['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {summary['win_rate']:.2%}")
        logger.info(f"  Total Trades: {summary['total_trades']}")
    else:
        logger.error(f"Backtest failed: {results.get('message')}")

async def example_paper_trading():
    """Paper trading example"""
    logger.info("\n=== Paper Trading Example ===")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'neural_momentum_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Modify config for paper trading
    config['trading']['execution']['paper_trading'] = True
    config['trading']['symbols'] = ['SPY', 'QQQ', 'AAPL']  # Smaller universe for demo
    
    # Create orchestrator
    orchestrator = StrategyOrchestrator(config)
    
    logger.info("Starting paper trading session (will run for 60 seconds)...")
    
    # Start trading in a separate task
    trading_task = asyncio.create_task(orchestrator.start_trading())
    
    # Let it run for a minute
    await asyncio.sleep(60)
    
    # Stop trading
    await orchestrator.stop_trading()
    
    # Get final status
    status = orchestrator.get_status()
    logger.info(f"Final status: {status}")

async def example_risk_analysis():
    """Risk analysis example"""
    logger.info("\n=== Risk Analysis Example ===")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'neural_momentum_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    from risk_management.adaptive_risk_manager import AdaptiveRiskManager
    
    # Create risk manager
    risk_manager = AdaptiveRiskManager(config['risk_management'])
    
    # Example portfolio positions
    positions = {
        'AAPL': {
            'symbol': 'AAPL',
            'position_size': 0.04,
            'direction': 'long',
            'entry_price': 150.0,
            'value': 6000
        },
        'MSFT': {
            'symbol': 'MSFT',
            'position_size': 0.03,
            'direction': 'long',
            'entry_price': 300.0,
            'value': 4500
        }
    }
    
    # Assess portfolio risk
    logger.info("Assessing portfolio risk...")
    risk_metrics = await risk_manager.assess_portfolio_risk(positions)
    
    logger.info("Risk Metrics:")
    logger.info(f"  VaR (95%): {risk_metrics.var_95:.4f}")
    logger.info(f"  Expected Shortfall: {risk_metrics.expected_shortfall:.4f}")
    logger.info(f"  Volatility: {risk_metrics.volatility:.4f}")
    logger.info(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown: {risk_metrics.max_drawdown:.4f}")
    
    # Test new position
    new_position = {
        'symbol': 'GOOGL',
        'position_size': 0.05,
        'direction': 'long',
        'value': 7500
    }
    
    risk_checks = await risk_manager.check_risk_limits(new_position)
    logger.info(f"Risk checks for new position: {risk_checks}")

async def example_neural_training():
    """Neural model training example"""
    logger.info("\n=== Neural Training Example ===")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'neural_momentum_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    from models.neural.momentum_predictor import MomentumPredictor
    
    # Create predictor
    predictor = MomentumPredictor(config['strategy']['neural_config'])
    
    # Generate sample training data
    logger.info("Generating sample training data...")
    training_data = []
    
    for i in range(200):  # Generate 200 samples
        sample = {
            'symbol': 'AAPL',
            'market_data': {
                'price': 150 + (i % 20),
                'volume': 1000000 + (i * 10000),
                'rsi': 50 + (i % 40),
                'macd': 0.5 * (i % 5),
                'volatility': 0.2 + 0.1 * (i % 3)
            },
            'momentum_direction': 0.1 * (i % 20 - 10),  # -1 to 1
            'momentum_strength': 0.5 + 0.5 * (i % 2),   # 0.5 to 1
            'prediction_confidence': 0.6 + 0.4 * (i % 2)  # 0.6 to 1
        }
        training_data.append(sample)
    
    # Train the model
    logger.info(f"Training neural model with {len(training_data)} samples...")
    training_result = await predictor.train(training_data, epochs=50)
    
    if training_result['status'] == 'success':
        logger.info(f"Training completed. Final loss: {training_result['final_loss']:.6f}")
        
        # Test prediction
        test_data = {
            'price': 155.0,
            'volume': 1200000,
            'rsi': 60.0,
            'macd': 0.8,
            'volatility': 0.25
        }
        
        prediction = await predictor.predict('AAPL', test_data)
        logger.info(f"Test prediction: {prediction}")
    else:
        logger.error(f"Training failed: {training_result.get('message')}")

async def example_performance_monitoring():
    """Performance monitoring example"""
    logger.info("\n=== Performance Monitoring Example ===")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'neural_momentum_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    from monitoring.performance_tracker import PerformanceTracker
    
    # Create performance tracker
    tracker = PerformanceTracker(config['monitoring'])
    
    # Simulate some trades
    logger.info("Simulating trade history...")
    
    # Record some example trades
    for i in range(10):
        trade_data = {
            'trade_id': f'DEMO_{i}',
            'symbol': ['AAPL', 'MSFT', 'GOOGL'][i % 3],
            'direction': 'long',
            'quantity': 100 + i * 10,
            'price': 150 + i * 5,
            'timestamp': datetime.now() - timedelta(days=10-i)
        }
        
        trade_id = await tracker.record_trade(trade_data)
        
        # Simulate trade closure
        if i < 7:  # Close first 7 trades
            exit_price = trade_data['price'] + (-5 + i * 2)  # Some wins, some losses
            await tracker.close_trade(trade_id, exit_price)
    
    # Generate performance report
    logger.info("Generating performance report...")
    report = await tracker.generate_report()
    
    if 'summary' in report:
        summary = report['summary']
        logger.info("Performance Summary:")
        logger.info(f"  Total Trades: {summary.get('total_trades', 0)}")
        logger.info(f"  Win Rate: {summary.get('win_rate', '0%')}")
        logger.info(f"  Profit Factor: {summary.get('profit_factor', '0.00')}")
        logger.info(f"  Sharpe Ratio: {summary.get('sharpe_ratio', '0.00')}")

async def example_market_integration():
    """Example with real market data integration"""
    logger.info("\n=== Market Integration Example ===")
    
    # This would integrate with the existing AI News Trader MCP tools
    try:
        # Use the AI News Trader tools for real market data
        from examples.momentum_strategy_example import mcp_integration_example
        await mcp_integration_example()
    except ImportError:
        logger.info("MCP integration example requires AI News Trader tools")
        logger.info("Running with simulated market data instead...")
        
        # Simulate market analysis
        symbols = ['SPY', 'QQQ', 'AAPL']
        logger.info(f"Analyzing market conditions for {symbols}")
        
        # Mock market analysis results
        market_analysis = {
            'SPY': {'price': 450.0, 'volatility': 'medium', 'trend': 'bullish'},
            'QQQ': {'price': 380.0, 'volatility': 'high', 'trend': 'neutral'},
            'AAPL': {'price': 175.0, 'volatility': 'low', 'trend': 'bullish'}
        }
        
        for symbol, data in market_analysis.items():
            logger.info(f"  {symbol}: {data}")

async def run_all_examples():
    """Run all examples"""
    logger.info("Running all Neural Momentum Strategy examples...\n")
    
    try:
        await example_basic_usage()
        await example_risk_analysis()
        await example_neural_training()
        await example_performance_monitoring()
        await example_backtesting()
        await example_market_integration()
        # await example_paper_trading()  # Uncomment to test paper trading
        
        logger.info("\n=== All examples completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Run all examples
    asyncio.run(run_all_examples())