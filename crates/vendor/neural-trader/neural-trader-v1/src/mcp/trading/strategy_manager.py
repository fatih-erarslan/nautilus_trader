"""
Strategy Manager for MCP Trading Integration

Manages all trading strategies and provides unified interface for MCP operations
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Trading position structure"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: str
    strategy: str
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    value: float = 0.0


@dataclass
class Trade:
    """Completed trade structure"""
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: str
    exit_time: str
    pnl: float
    return_pct: float
    strategy: str


class StrategyManager:
    """Manages trading strategies for MCP server"""
    
    def __init__(self, gpu_enabled: bool = False):
        self.gpu_enabled = gpu_enabled
        self.strategies: Dict[str, Any] = {}
        self.positions: Dict[str, List[Position]] = {}
        self.trades: Dict[str, List[Trade]] = {}
        self.market_data: Dict[str, Dict] = {}
        self.model_loader = None
        
        # Initialize strategy tracking
        self.strategy_names = [
            'mirror_trader',
            'momentum_trader',
            'swing_trader',
            'mean_reversion_trader'
        ]
        
        for strategy in self.strategy_names:
            self.positions[strategy] = []
            self.trades[strategy] = []
    
    async def initialize(self):
        """Initialize strategy manager and load models"""
        logger.info("Initializing Strategy Manager...")
        
        # Load model loader
        from .model_loader import ModelLoader
        self.model_loader = ModelLoader(gpu_enabled=self.gpu_enabled)
        
        # Load all strategies
        for strategy_name in self.strategy_names:
            try:
                strategy = await self.model_loader.load_strategy(strategy_name)
                self.strategies[strategy_name] = strategy
                logger.info(f"Loaded strategy: {strategy_name}")
            except Exception as e:
                logger.error(f"Failed to load strategy {strategy_name}: {str(e)}")
        
        # Start market data updates
        asyncio.create_task(self._update_market_data())
        
        logger.info("Strategy Manager initialized successfully")
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies"""
        return list(self.strategies.keys())
    
    async def execute_trade(self, order) -> Dict:
        """Execute a trade order"""
        strategy_name = order.strategy
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        
        # Simulate order execution
        execution_price = await self._get_execution_price(
            order.symbol,
            order.side,
            order.order_type,
            order.price
        )
        
        execution_time = datetime.now().isoformat()
        
        if order.side == 'buy':
            # Create new position
            position = Position(
                symbol=order.symbol,
                quantity=order.quantity,
                entry_price=execution_price,
                current_price=execution_price,
                entry_time=execution_time,
                strategy=strategy_name,
                value=order.quantity * execution_price
            )
            self.positions[strategy_name].append(position)
        else:
            # Close existing position
            await self._close_position(strategy_name, order.symbol, order.quantity, execution_price)
        
        return {
            'executed_price': execution_price,
            'executed_quantity': order.quantity,
            'execution_time': execution_time,
            'commission': order.quantity * execution_price * 0.001  # 0.1% commission
        }
    
    async def backtest(self, strategy: str, start_date: str, end_date: str,
                      symbols: List[str], initial_capital: float,
                      parameters: Optional[Dict] = None) -> Dict:
        """Run backtesting for a strategy"""
        if strategy not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy}")
        
        strategy_instance = self.strategies[strategy]
        
        # This would implement full backtesting logic
        # For now, return simulated results
        num_days = 252  # Approximate trading days
        
        # Generate simulated equity curve
        daily_returns = np.random.normal(0.0005, 0.02, num_days)
        equity_curve = initial_capital * np.cumprod(1 + daily_returns)
        
        # Generate simulated trades
        trades = []
        for i in range(20):  # Simulate 20 trades
            entry_day = np.random.randint(0, num_days - 10)
            exit_day = entry_day + np.random.randint(1, 10)
            
            trade = {
                'symbol': np.random.choice(symbols),
                'entry_price': 100 * (1 + np.random.normal(0, 0.1)),
                'exit_price': 100 * (1 + np.random.normal(0.001, 0.1)),
                'quantity': np.random.randint(10, 100),
                'entry_time': f"2023-01-{entry_day+1:02d}",
                'exit_time': f"2023-01-{exit_day+1:02d}",
            }
            trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['quantity']
            trade['return_pct'] = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
            trades.append(trade)
        
        return {
            'equity_curve': equity_curve.tolist(),
            'trades': trades,
            'final_capital': float(equity_curve[-1]),
            'total_return': float((equity_curve[-1] - initial_capital) / initial_capital),
            'strategy_parameters': parameters or {}
        }
    
    async def optimize(self, strategy: str, objective: str,
                      constraints: Optional[Dict], iterations: int,
                      population_size: int) -> Dict:
        """Optimize strategy parameters"""
        if strategy not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy}")
        
        logger.info(f"Running optimization for {strategy} with objective: {objective}")
        
        # This would implement real optimization
        # For now, return the loaded optimized parameters
        optimized_params = await self.model_loader.get_optimized_parameters(strategy)
        
        # Simulate optimization progress
        convergence_history = []
        for i in range(min(iterations, 20)):
            convergence_history.append({
                'iteration': i,
                'best_fitness': 1.5 + i * 0.1 + np.random.normal(0, 0.05),
                'avg_fitness': 1.0 + i * 0.05 + np.random.normal(0, 0.1)
            })
        
        return {
            'best_parameters': optimized_params.get('best_parameters', {}),
            'performance': optimized_params.get('performance_metrics', {}),
            'convergence_history': convergence_history,
            'optimization_time': 154.69  # From the optimization results
        }
    
    async def get_positions(self, strategy: str) -> List[Dict]:
        """Get positions for a strategy"""
        if strategy not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy}")
        
        positions = self.positions.get(strategy, [])
        
        # Update current prices and P&L
        for position in positions:
            current_price = await self._get_current_price(position.symbol)
            position.current_price = current_price
            position.value = position.quantity * current_price
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
        
        return [asdict(p) for p in positions]
    
    async def get_all_positions(self) -> List[Dict]:
        """Get all positions across strategies"""
        all_positions = []
        
        for strategy in self.strategies:
            positions = await self.get_positions(strategy)
            all_positions.extend(positions)
        
        return all_positions
    
    async def get_performance(self, strategy: str, period: str) -> Dict:
        """Get strategy performance metrics"""
        if strategy not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy}")
        
        # Calculate period start date
        end_date = datetime.now()
        period_map = {
            '1d': timedelta(days=1),
            '1w': timedelta(weeks=1),
            '1m': timedelta(days=30),
            '3m': timedelta(days=90),
            '1y': timedelta(days=365),
            'all': timedelta(days=3650)  # 10 years
        }
        
        delta = period_map.get(period, timedelta(days=30))
        start_date = end_date - delta
        
        # Get trades in period
        strategy_trades = self.trades.get(strategy, [])
        period_trades = [
            t for t in strategy_trades
            if start_date.isoformat() <= t.exit_time <= end_date.isoformat()
        ]
        
        # Calculate metrics
        if period_trades:
            returns = [t.return_pct for t in period_trades]
            total_pnl = sum(t.pnl for t in period_trades)
            win_rate = len([t for t in period_trades if t.pnl > 0]) / len(period_trades)
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            sharpe_ratio = (avg_return / volatility * np.sqrt(252)) if volatility > 0 else 0
        else:
            total_pnl = 0
            win_rate = 0
            avg_return = 0
            volatility = 0
            sharpe_ratio = 0
        
        # Generate equity curve (simulated)
        num_points = 100
        equity_curve = self._generate_equity_curve(num_points, avg_return, volatility)
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        
        return {
            'metrics': {
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'trades_count': len(period_trades),
                'max_drawdown': float(np.min(drawdown_series)) if drawdown_series else 0
            },
            'equity_curve': equity_curve,
            'drawdown_series': drawdown_series
        }
    
    async def get_recent_performance_summary(self) -> Dict:
        """Get recent performance summary for all strategies"""
        summary = {}
        
        for strategy in self.strategies:
            try:
                performance = await self.get_performance(strategy, '1m')
                summary[strategy] = {
                    'sharpe_ratio': performance['metrics']['sharpe_ratio'],
                    'win_rate': performance['metrics']['win_rate'],
                    'total_pnl': performance['metrics']['total_pnl']
                }
            except Exception as e:
                logger.error(f"Error getting performance for {strategy}: {str(e)}")
                summary[strategy] = {
                    'sharpe_ratio': 'N/A',
                    'win_rate': 'N/A',
                    'total_pnl': 'N/A'
                }
        
        return summary
    
    async def get_recent_signals(self, symbol: str, strategy: str) -> List[Dict]:
        """Get recent trading signals for a symbol/strategy"""
        # This would get real signals from the strategy
        # For now, return simulated signals
        
        current_price = await self._get_current_price(symbol)
        
        signals = [
            {
                'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                'signal': 'buy',
                'strength': 0.75,
                'price': current_price * 0.98,
                'reason': 'Momentum breakout detected'
            },
            {
                'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                'signal': 'hold',
                'strength': 0.6,
                'price': current_price * 0.99,
                'reason': 'Waiting for confirmation'
            },
            {
                'timestamp': datetime.now().isoformat(),
                'signal': 'buy',
                'strength': 0.85,
                'price': current_price,
                'reason': 'Strong momentum confirmed'
            }
        ]
        
        return signals
    
    async def get_all_model_states(self) -> Dict:
        """Get current state of all models"""
        states = {}
        
        for strategy_name, strategy in self.strategies.items():
            states[strategy_name] = {
                'loaded': True,
                'last_update': datetime.now().isoformat(),
                'positions_count': len(self.positions.get(strategy_name, [])),
                'trades_today': len([
                    t for t in self.trades.get(strategy_name, [])
                    if t.exit_time.startswith(datetime.now().strftime('%Y-%m-%d'))
                ]),
                'status': 'active'
            }
        
        return states
    
    async def _get_execution_price(self, symbol: str, side: str,
                                  order_type: str, limit_price: Optional[float]) -> float:
        """Get execution price for an order"""
        current_price = await self._get_current_price(symbol)
        
        if order_type == 'market':
            # Add slight slippage
            slippage = 0.001 if side == 'buy' else -0.001
            return current_price * (1 + slippage)
        elif order_type == 'limit':
            return limit_price or current_price
        else:
            return current_price
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        # In production, this would fetch real market data
        # For now, return simulated price
        base_prices = {
            'AAPL': 189.46,
            'GOOGL': 141.24,
            'MSFT': 377.90,
            'TSLA': 245.00,
            'AMZN': 127.50
        }
        
        base_price = base_prices.get(symbol, 100.0)
        # Add some random variation
        return base_price * (1 + np.random.normal(0, 0.001))
    
    async def _close_position(self, strategy: str, symbol: str,
                             quantity: float, exit_price: float):
        """Close a position and record trade"""
        positions = self.positions.get(strategy, [])
        
        # Find matching position
        for i, position in enumerate(positions):
            if position.symbol == symbol and position.quantity >= quantity:
                # Create trade record
                trade = Trade(
                    symbol=symbol,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    entry_time=position.entry_time,
                    exit_time=datetime.now().isoformat(),
                    pnl=(exit_price - position.entry_price) * quantity,
                    return_pct=(exit_price - position.entry_price) / position.entry_price,
                    strategy=strategy
                )
                
                self.trades[strategy].append(trade)
                
                # Update or remove position
                if position.quantity == quantity:
                    positions.pop(i)
                else:
                    position.quantity -= quantity
                    position.value = position.quantity * position.current_price
                
                break
    
    async def _update_market_data(self):
        """Continuously update market data"""
        while True:
            try:
                # In production, this would fetch real market data
                # For now, just log
                logger.debug("Updating market data...")
                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                logger.error(f"Market data update error: {str(e)}")
                await asyncio.sleep(5)  # Wait longer on error
    
    def _generate_equity_curve(self, num_points: int, avg_return: float,
                              volatility: float) -> List[float]:
        """Generate simulated equity curve"""
        returns = np.random.normal(avg_return / 252, volatility / np.sqrt(252), num_points)
        equity = 100000 * np.cumprod(1 + returns)
        return equity.tolist()
    
    def _calculate_drawdown_series(self, equity_curve: List[float]) -> List[float]:
        """Calculate drawdown series from equity curve"""
        if not equity_curve:
            return []
        
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return drawdown.tolist()