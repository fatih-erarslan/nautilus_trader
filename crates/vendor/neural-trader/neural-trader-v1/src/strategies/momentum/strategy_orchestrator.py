"""
Neural Momentum Strategy Orchestrator
Main coordination system that integrates all components for live trading
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from src.strategies.momentum.neural_momentum_strategy import NeuralMomentumStrategy
from src.strategies.momentum.backtesting_engine import BacktestingEngine, BacktestConfig
from src.models.neural.momentum_predictor import MomentumPredictor
from src.risk_management.adaptive_risk_manager import AdaptiveRiskManager
from src.monitoring.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

class StrategyOrchestrator:
    """
    Main orchestrator for the Neural Momentum Trading Strategy
    Coordinates all components and provides unified interface
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize core components
        self.strategy = NeuralMomentumStrategy(config.get('strategy', {}))
        self.risk_manager = AdaptiveRiskManager(config.get('risk_management', {}))
        self.performance_tracker = PerformanceTracker(config.get('monitoring', {}))
        
        # Trading state
        self.is_trading = False
        self.current_positions = {}
        self.trading_symbols = config.get('symbols', ['SPY', 'QQQ', 'IWM'])
        
        # Scheduling
        self.trading_schedule = config.get('trading_schedule', {
            'market_open': '09:30',
            'market_close': '16:00',
            'signal_generation_frequency': 300,  # 5 minutes
            'position_management_frequency': 60   # 1 minute
        })
        
        # Integration with external systems
        self.news_integration = config.get('news_integration', True)
        self.neural_training = config.get('neural_training', True)
        
        logger.info("Neural Momentum Strategy Orchestrator initialized")
    
    async def start_trading(self):
        """Start live trading operations"""
        try:
            logger.info("Starting Neural Momentum Trading System...")
            
            # Pre-trading checks
            await self._pre_trading_checks()
            
            # Initialize neural models
            if self.neural_training:
                await self._initialize_neural_models()
            
            # Start main trading loop
            self.is_trading = True
            
            # Run concurrent tasks
            tasks = [
                self._signal_generation_loop(),
                self._position_management_loop(),
                self._risk_monitoring_loop(),
                self._performance_monitoring_loop(),
                self._neural_training_loop()
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            await self.stop_trading()
    
    async def stop_trading(self):
        """Stop trading operations and cleanup"""
        try:
            logger.info("Stopping Neural Momentum Trading System...")
            
            self.is_trading = False
            
            # Close all positions if configured
            if self.config.get('close_positions_on_stop', True):
                await self._close_all_positions()
            
            # Generate final performance report
            await self._generate_final_report()
            
            logger.info("Trading system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
    
    async def run_backtest(self, backtest_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive backtest"""
        try:
            logger.info("Running comprehensive backtest...")
            
            # Create backtest configuration
            config = BacktestConfig(
                start_date=datetime.fromisoformat(backtest_config['start_date']),
                end_date=datetime.fromisoformat(backtest_config['end_date']),
                initial_capital=backtest_config.get('initial_capital', 100000),
                symbols=backtest_config.get('symbols', self.trading_symbols),
                benchmark_symbol=backtest_config.get('benchmark_symbol', 'SPY'),
                transaction_costs=backtest_config.get('transaction_costs', 0.001),
                slippage=backtest_config.get('slippage', 0.0005),
                max_positions=backtest_config.get('max_positions', 10),
                rebalance_frequency=backtest_config.get('rebalance_frequency', 'daily'),
                walk_forward_periods=backtest_config.get('walk_forward_periods', 12),
                optimization_metric=backtest_config.get('optimization_metric', 'sharpe_ratio'),
                regime_analysis=backtest_config.get('regime_analysis', True),
                monte_carlo_runs=backtest_config.get('monte_carlo_runs', 1000)
            )
            
            # Run backtest
            engine = BacktestingEngine(config)
            results = await engine.run_backtest()
            
            # Save results
            results_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            engine.save_results(f"tests/results/{results_file}")
            
            logger.info(f"Backtest completed. Results saved to {results_file}")
            
            return {
                'status': 'completed',
                'results_file': results_file,
                'summary': {
                    'total_return': results.performance_metrics.get('total_return', 0),
                    'sharpe_ratio': results.performance_metrics.get('sharpe_ratio', 0),
                    'max_drawdown': results.performance_metrics.get('max_drawdown', 0),
                    'win_rate': results.trade_analysis.get('win_rate', 0),
                    'total_trades': results.trade_analysis.get('total_trades', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _pre_trading_checks(self):
        """Perform pre-trading system checks"""
        try:
            logger.info("Performing pre-trading checks...")
            
            # Check market data connectivity
            market_data_status = await self._check_market_data()
            if not market_data_status:
                raise RuntimeError("Market data connectivity failed")
            
            # Check news data connectivity
            if self.news_integration:
                news_status = await self._check_news_data()
                if not news_status:
                    logger.warning("News data connectivity issues - continuing without news")
                    self.news_integration = False
            
            # Check neural models
            if self.neural_training:
                model_status = await self._check_neural_models()
                if not model_status:
                    logger.warning("Neural models not ready - using fallback parameters")
            
            # Check risk management system
            risk_status = await self._check_risk_management()
            if not risk_status:
                raise RuntimeError("Risk management system not ready")
            
            logger.info("All pre-trading checks passed")
            
        except Exception as e:
            logger.error(f"Pre-trading checks failed: {e}")
            raise
    
    async def _signal_generation_loop(self):
        """Main signal generation loop"""
        try:
            while self.is_trading:
                try:
                    # Generate trading signals
                    signals = await self.strategy.generate_signals(self.trading_symbols)
                    
                    if signals:
                        logger.info(f"Generated {len(signals)} trading signals")
                        
                        # Execute trades based on signals
                        executed_trades = await self._execute_signals(signals)
                        
                        if executed_trades:
                            logger.info(f"Executed {len(executed_trades)} trades")
                    
                    # Wait for next signal generation cycle
                    await asyncio.sleep(self.trading_schedule['signal_generation_frequency'])
                    
                except Exception as e:
                    logger.error(f"Error in signal generation loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Signal generation loop failed: {e}")
    
    async def _position_management_loop(self):
        """Position management and monitoring loop"""
        try:
            while self.is_trading:
                try:
                    # Manage existing positions
                    management_actions = await self.strategy.manage_positions()
                    
                    if management_actions:
                        logger.info(f"Processing {len(management_actions)} management actions")
                        await self._process_management_actions(management_actions)
                    
                    # Update position tracking
                    await self._update_position_tracking()
                    
                    # Wait for next management cycle
                    await asyncio.sleep(self.trading_schedule['position_management_frequency'])
                    
                except Exception as e:
                    logger.error(f"Error in position management loop: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"Position management loop failed: {e}")
    
    async def _risk_monitoring_loop(self):
        """Continuous risk monitoring loop"""
        try:
            while self.is_trading:
                try:
                    # Assess current portfolio risk
                    risk_metrics = await self.risk_manager.assess_portfolio_risk(self.current_positions)
                    
                    # Check for risk limit breaches
                    if risk_metrics.var_95 and abs(risk_metrics.var_95) > self.risk_manager.max_portfolio_risk:
                        logger.warning(f"Portfolio VaR breach detected: {risk_metrics.var_95}")
                        await self._handle_risk_breach('var_breach', risk_metrics)
                    
                    # Check drawdown limits
                    if risk_metrics.max_drawdown and abs(risk_metrics.max_drawdown) > 0.15:  # 15% max drawdown
                        logger.warning(f"Maximum drawdown exceeded: {risk_metrics.max_drawdown}")
                        await self._handle_risk_breach('drawdown_breach', risk_metrics)
                    
                    # Update volatility regime
                    await self.risk_manager.update_volatility_regime({
                        'returns': await self._get_recent_market_returns()
                    })
                    
                    # Wait for next risk check
                    await asyncio.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    logger.error(f"Error in risk monitoring loop: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"Risk monitoring loop failed: {e}")
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring and reporting loop"""
        try:
            while self.is_trading:
                try:
                    # Generate performance report
                    if datetime.now().minute == 0:  # Hourly reports
                        report = await self.performance_tracker.generate_report()
                        logger.info(f"Performance Update: "
                                  f"Total Trades: {report.get('summary', {}).get('total_trades', 0)}, "
                                  f"Win Rate: {report.get('summary', {}).get('win_rate', '0%')}, "
                                  f"Sharpe: {report.get('summary', {}).get('sharpe_ratio', '0.00')}")
                    
                    # Wait for next monitoring cycle
                    await asyncio.sleep(3600)  # 1 hour
                    
                except Exception as e:
                    logger.error(f"Error in performance monitoring loop: {e}")
                    await asyncio.sleep(300)
                    
        except Exception as e:
            logger.error(f"Performance monitoring loop failed: {e}")
    
    async def _neural_training_loop(self):
        """Continuous neural model training loop"""
        try:
            while self.is_trading and self.neural_training:
                try:
                    # Collect recent trading data for training
                    training_data = await self._collect_training_data()
                    
                    if len(training_data) >= 100:  # Minimum samples for training
                        logger.info(f"Starting neural model training with {len(training_data)} samples")
                        
                        # Train momentum predictor
                        training_result = await self.strategy.neural_predictor.train(training_data, epochs=50)
                        
                        if training_result['status'] == 'success':
                            logger.info(f"Neural training completed. Final loss: {training_result['final_loss']:.6f}")
                        
                    # Wait for next training cycle (daily)
                    await asyncio.sleep(86400)  # 24 hours
                    
                except Exception as e:
                    logger.error(f"Error in neural training loop: {e}")
                    await asyncio.sleep(3600)  # Wait 1 hour before retry
                    
        except Exception as e:
            logger.error(f"Neural training loop failed: {e}")
    
    async def _execute_signals(self, signals: List[Any]) -> List[Dict[str, Any]]:
        """Execute trading signals with risk checks"""
        executed_trades = []
        
        try:
            for signal in signals:
                # Check risk limits before execution
                risk_check = await self.risk_manager.check_risk_limits({
                    'symbol': signal.symbol,
                    'position_size': signal.position_size,
                    'direction': signal.direction,
                    'value': signal.entry_price * signal.position_size
                })
                
                if not all(risk_check.values()):
                    logger.warning(f"Risk check failed for {signal.symbol}: {risk_check}")
                    continue
                
                # Execute the trade (mock implementation)
                trade_result = await self._execute_trade(signal)
                
                if trade_result:
                    executed_trades.append(trade_result)
                    self.current_positions[signal.symbol] = signal
                    
                    # Record trade for performance tracking
                    await self.performance_tracker.record_trade(trade_result)
            
            return executed_trades
            
        except Exception as e:
            logger.error(f"Error executing signals: {e}")
            return []
    
    async def _execute_trade(self, signal: Any) -> Optional[Dict[str, Any]]:
        """Execute individual trade (mock implementation)"""
        try:
            # Mock trade execution - in production, integrate with broker API
            trade_result = {
                'trade_id': f"LIVE_{signal.symbol}_{int(datetime.now().timestamp())}",
                'symbol': signal.symbol,
                'direction': signal.direction,
                'quantity': signal.position_size,
                'price': signal.entry_price,
                'timestamp': datetime.now(),
                'status': 'filled'
            }
            
            logger.info(f"Executed trade: {signal.direction} {signal.position_size} {signal.symbol} @ {signal.entry_price}")
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing trade for {signal.symbol}: {e}")
            return None
    
    # Helper methods (mock implementations)
    async def _check_market_data(self) -> bool:
        """Check market data connectivity"""
        return True
    
    async def _check_news_data(self) -> bool:
        """Check news data connectivity"""
        return True
    
    async def _check_neural_models(self) -> bool:
        """Check neural model status"""
        return True
    
    async def _check_risk_management(self) -> bool:
        """Check risk management system"""
        return True
    
    async def _initialize_neural_models(self):
        """Initialize and load neural models"""
        logger.info("Initializing neural models...")
    
    async def _process_management_actions(self, actions: List[Dict[str, Any]]):
        """Process position management actions"""
        for action in actions:
            logger.info(f"Processing management action: {action}")
    
    async def _update_position_tracking(self):
        """Update position tracking and valuations"""
        pass
    
    async def _handle_risk_breach(self, breach_type: str, risk_metrics: Any):
        """Handle risk limit breaches"""
        logger.warning(f"Handling risk breach: {breach_type}")
        
        if breach_type == 'var_breach':
            # Reduce position sizes
            pass
        elif breach_type == 'drawdown_breach':
            # Consider stopping trading temporarily
            pass
    
    async def _get_recent_market_returns(self) -> List[float]:
        """Get recent market returns for volatility analysis"""
        return [0.01, -0.005, 0.02, 0.008, -0.01]  # Mock data
    
    async def _collect_training_data(self) -> List[Dict[str, Any]]:
        """Collect recent trading data for neural training"""
        return []  # Mock implementation
    
    async def _close_all_positions(self):
        """Close all open positions"""
        logger.info("Closing all positions...")
    
    async def _generate_final_report(self):
        """Generate final performance report"""
        report = await self.performance_tracker.generate_report()
        
        # Save final report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"final_performance_report_{timestamp}.json"
        
        with open(f"docs/reports/{report_file}", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final performance report saved to {report_file}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_trading': self.is_trading,
            'active_positions': len(self.current_positions),
            'trading_symbols': self.trading_symbols,
            'neural_training_enabled': self.neural_training,
            'news_integration_enabled': self.news_integration,
            'last_update': datetime.now().isoformat()
        }