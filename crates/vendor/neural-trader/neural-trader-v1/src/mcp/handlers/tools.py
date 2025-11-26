"""
MCP Tools Handler

Handles trading operations, backtesting, and optimization through MCP tools
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeOrder:
    """Trade order structure"""
    strategy: str
    symbol: str
    quantity: float
    order_type: str  # 'market', 'limit', 'stop'
    side: str  # 'buy', 'sell'
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'day'  # 'day', 'gtc', 'ioc', 'fok'


@dataclass
class BacktestRequest:
    """Backtest request structure"""
    strategy: str
    start_date: str
    end_date: str
    symbols: List[str]
    initial_capital: float = 100000.0
    parameters: Optional[Dict] = None


@dataclass
class OptimizationRequest:
    """Optimization request structure"""
    strategy: str
    objective: str  # 'sharpe', 'returns', 'drawdown'
    constraints: Optional[Dict] = None
    iterations: int = 100
    population_size: int = 50


class ToolsHandler:
    """Handles MCP tool operations for trading"""
    
    def __init__(self, server):
        self.server = server
        self.active_orders: Dict[str, TradeOrder] = {}
        self.backtest_cache: Dict[str, Any] = {}
        self.strategy_manager = None  # Will be initialized lazily
        self.syndicate_tools = {}  # Will be populated by register_syndicate_tools
        
        # Register syndicate tools on initialization
        self.register_syndicate_tools()
        
    async def _get_strategy_manager(self):
        """Lazy load strategy manager"""
        if self.strategy_manager is None:
            from ..trading.strategy_manager import StrategyManager
            self.strategy_manager = StrategyManager(gpu_enabled=self.server.gpu_available)
            await self.strategy_manager.initialize()
        return self.strategy_manager
    
    async def handle_list_tools(self, params: Dict) -> Dict:
        """List available trading tools"""
        tools_list = [
                {
                    'name': 'execute_trade',
                    'description': 'Execute a trading order',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'strategy': {'type': 'string'},
                            'symbol': {'type': 'string'},
                            'quantity': {'type': 'number'},
                            'order_type': {'type': 'string', 'enum': ['market', 'limit', 'stop']},
                            'side': {'type': 'string', 'enum': ['buy', 'sell']},
                            'price': {'type': 'number'},
                            'stop_price': {'type': 'number'}
                        },
                        'required': ['strategy', 'symbol', 'quantity', 'order_type', 'side']
                    }
                },
                {
                    'name': 'backtest',
                    'description': 'Run historical backtesting on a strategy',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'strategy': {'type': 'string'},
                            'start_date': {'type': 'string', 'format': 'date'},
                            'end_date': {'type': 'string', 'format': 'date'},
                            'symbols': {'type': 'array', 'items': {'type': 'string'}},
                            'initial_capital': {'type': 'number'},
                            'parameters': {'type': 'object'}
                        },
                        'required': ['strategy', 'start_date', 'end_date', 'symbols']
                    }
                },
                {
                    'name': 'optimize',
                    'description': 'Optimize strategy parameters',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'strategy': {'type': 'string'},
                            'objective': {'type': 'string', 'enum': ['sharpe', 'returns', 'drawdown']},
                            'constraints': {'type': 'object'},
                            'iterations': {'type': 'integer'},
                            'population_size': {'type': 'integer'}
                        },
                        'required': ['strategy', 'objective']
                    }
                },
                {
                    'name': 'get_positions',
                    'description': 'Get current portfolio positions',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'strategy': {'type': 'string'}
                        }
                    }
                },
                {
                    'name': 'get_performance',
                    'description': 'Get strategy performance metrics',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'strategy': {'type': 'string'},
                            'period': {'type': 'string', 'enum': ['1d', '1w', '1m', '3m', '1y', 'all']}
                        },
                        'required': ['strategy']
                    }
                }
            ]
        
        # Add syndicate tools to the list
        for tool_name, tool_config in self.syndicate_tools.items():
            tools_list.append({
                'name': tool_name,
                'description': tool_config['description'],
                'inputSchema': tool_config['inputSchema']
            })
        
        return {
            'tools': tools_list
        }
    
    async def handle_call_tool(self, params: Dict) -> Dict:
        """Execute a tool with given arguments"""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        if not tool_name:
            raise ValueError("Tool name is required")
        
        # Route to appropriate handler
        tool_handlers = {
            'execute_trade': self._execute_trade,
            'backtest': self._run_backtest,
            'optimize': self._run_optimization,
            'get_positions': self._get_positions,
            'get_performance': self._get_performance
        }
        
        # Check if it's a syndicate tool
        if tool_name in self.syndicate_tools:
            handler = self.syndicate_tools[tool_name]['handler']
            result = await handler(arguments)
        elif tool_name in tool_handlers:
            handler = tool_handlers[tool_name]
            result = await handler(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return {
            'tool': tool_name,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_trade(self, args: Dict) -> Dict:
        """Execute a trading order"""
        try:
            order = TradeOrder(**args)
            strategy_manager = await self._get_strategy_manager()
            
            # Validate strategy exists
            if order.strategy not in strategy_manager.get_available_strategies():
                raise ValueError(f"Unknown strategy: {order.strategy}")
            
            # Generate order ID
            order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.active_orders)}"
            
            # Store order
            self.active_orders[order_id] = order
            
            # Execute through strategy
            execution_result = await strategy_manager.execute_trade(order)
            
            # Broadcast update
            await self.server.broadcast_update('trade_executed', {
                'order_id': order_id,
                'order': asdict(order),
                'execution': execution_result
            })
            
            logger.info(f"Trade executed: {order_id} - {order.symbol} {order.side} {order.quantity}")
            
            return {
                'order_id': order_id,
                'status': 'executed',
                'execution': execution_result
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _run_backtest(self, args: Dict) -> Dict:
        """Run historical backtesting"""
        try:
            backtest_req = BacktestRequest(**args)
            strategy_manager = await self._get_strategy_manager()
            
            # Check cache
            cache_key = f"{backtest_req.strategy}_{backtest_req.start_date}_{backtest_req.end_date}_{'_'.join(backtest_req.symbols)}"
            if cache_key in self.backtest_cache:
                logger.info(f"Returning cached backtest results for {cache_key}")
                return self.backtest_cache[cache_key]
            
            # Run backtest
            logger.info(f"Running backtest for {backtest_req.strategy} from {backtest_req.start_date} to {backtest_req.end_date}")
            
            results = await strategy_manager.backtest(
                strategy=backtest_req.strategy,
                start_date=backtest_req.start_date,
                end_date=backtest_req.end_date,
                symbols=backtest_req.symbols,
                initial_capital=backtest_req.initial_capital,
                parameters=backtest_req.parameters
            )
            
            # Calculate metrics
            metrics = self._calculate_backtest_metrics(results)
            
            backtest_result = {
                'strategy': backtest_req.strategy,
                'period': {
                    'start': backtest_req.start_date,
                    'end': backtest_req.end_date
                },
                'symbols': backtest_req.symbols,
                'metrics': metrics,
                'equity_curve': results.get('equity_curve', []),
                'trades': results.get('trades', [])
            }
            
            # Cache results
            self.backtest_cache[cache_key] = backtest_result
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Backtest error: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _run_optimization(self, args: Dict) -> Dict:
        """Run strategy parameter optimization"""
        try:
            opt_req = OptimizationRequest(**args)
            strategy_manager = await self._get_strategy_manager()
            
            logger.info(f"Running optimization for {opt_req.strategy} with objective: {opt_req.objective}")
            
            # Run optimization (this could be GPU-accelerated)
            optimization_result = await strategy_manager.optimize(
                strategy=opt_req.strategy,
                objective=opt_req.objective,
                constraints=opt_req.constraints,
                iterations=opt_req.iterations,
                population_size=opt_req.population_size
            )
            
            return {
                'strategy': opt_req.strategy,
                'objective': opt_req.objective,
                'best_parameters': optimization_result.get('best_parameters', {}),
                'performance': optimization_result.get('performance', {}),
                'convergence_history': optimization_result.get('convergence_history', []),
                'optimization_time': optimization_result.get('optimization_time', 0)
            }
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _get_positions(self, args: Dict) -> Dict:
        """Get current portfolio positions"""
        try:
            strategy = args.get('strategy')
            strategy_manager = await self._get_strategy_manager()
            
            if strategy:
                positions = await strategy_manager.get_positions(strategy)
            else:
                positions = await strategy_manager.get_all_positions()
            
            return {
                'positions': positions,
                'total_value': sum(p.get('value', 0) for p in positions),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Get positions error: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _get_performance(self, args: Dict) -> Dict:
        """Get strategy performance metrics"""
        try:
            strategy = args.get('strategy')
            period = args.get('period', '1m')
            strategy_manager = await self._get_strategy_manager()
            
            performance = await strategy_manager.get_performance(strategy, period)
            
            return {
                'strategy': strategy,
                'period': period,
                'metrics': performance.get('metrics', {}),
                'equity_curve': performance.get('equity_curve', []),
                'drawdown_series': performance.get('drawdown_series', []),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Get performance error: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _calculate_backtest_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics from backtest results"""
        equity_curve = np.array(results.get('equity_curve', []))
        
        if len(equity_curve) < 2:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'trades_count': 0
            }
        
        # Calculate returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Calculate metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Trade statistics
        trades = results.get('trades', [])
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        return {
            'total_return': float(total_return),
            'annualized_return': float((1 + total_return) ** (252 / len(equity_curve)) - 1),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'trades_count': len(trades),
            'avg_trade_return': float(np.mean([t.get('pnl', 0) for t in trades])) if trades else 0,
            'volatility': float(np.std(returns) * np.sqrt(252))
        }
    
    def register_syndicate_tools(self):
        """Register syndicate management tools"""
        try:
            from ..tools.syndicate_tools import SYNDICATE_TOOLS
            
            # Store syndicate tools for access
            self.syndicate_tools = SYNDICATE_TOOLS
            
            logger.info(f"Registered {len(SYNDICATE_TOOLS)} syndicate tools")
            
        except ImportError as e:
            logger.warning(f"Failed to import syndicate tools: {e}")
            self.syndicate_tools = {}