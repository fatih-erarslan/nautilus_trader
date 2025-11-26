"""
Performance Tracking and Monitoring System
Comprehensive performance analytics with real-time monitoring and alerting
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TradeMetrics:
    """Individual trade metrics"""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: Optional[float]
    pnl_pct: Optional[float]
    holding_period: Optional[timedelta]
    max_adverse_excursion: Optional[float]
    max_favorable_excursion: Optional[float]
    trade_status: str  # 'open', 'closed', 'stopped'

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    total_return: float
    annualized_return: float
    volatility: float
    var_95: float
    skewness: float
    kurtosis: float

@dataclass
class RiskMetrics:
    """Risk-specific metrics"""
    current_drawdown: float
    max_consecutive_losses: int
    tail_ratio: float
    downside_deviation: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float

class PerformanceTracker:
    """
    Comprehensive performance tracking system with real-time monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trades = {}
        self.daily_pnl = defaultdict(float)
        self.portfolio_values = {}
        self.benchmarks = {}
        
        # Alerting configuration
        self.alert_thresholds = {
            'max_drawdown': config.get('max_drawdown_alert', 0.1),
            'consecutive_losses': config.get('max_consecutive_losses', 5),
            'daily_loss_limit': config.get('daily_loss_limit', 0.05),
            'var_breach': config.get('var_breach_threshold', 0.02)
        }
        
        # Performance calculation parameters
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.benchmark_symbol = config.get('benchmark_symbol', 'SPY')
        
        logger.info("Performance Tracker initialized")
    
    async def record_trade(self, trade_data: Dict[str, Any]) -> str:
        """Record a new trade"""
        try:
            trade_id = trade_data.get('trade_id', f"TRADE_{int(datetime.now().timestamp())}")
            
            trade_metrics = TradeMetrics(
                trade_id=trade_id,
                symbol=trade_data['symbol'],
                direction=trade_data['direction'],
                entry_price=trade_data['price'],
                exit_price=None,
                quantity=trade_data['quantity'],
                entry_time=trade_data.get('timestamp', datetime.now()),
                exit_time=None,
                pnl=None,
                pnl_pct=None,
                holding_period=None,
                max_adverse_excursion=0.0,
                max_favorable_excursion=0.0,
                trade_status='open'
            )
            
            self.trades[trade_id] = trade_metrics
            logger.info(f"Recorded new trade: {trade_id} - {trade_data['symbol']}")
            
            return trade_id
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return ""
    
    async def update_trade(self, trade_id: str, current_price: float, timestamp: Optional[datetime] = None):
        """Update trade with current price information"""
        try:
            if trade_id not in self.trades:
                return
            
            trade = self.trades[trade_id]
            if trade.trade_status != 'open':
                return
            
            timestamp = timestamp or datetime.now()
            
            # Calculate unrealized PnL
            if trade.direction == 'long':
                unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
            else:
                unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
            
            unrealized_pnl_pct = unrealized_pnl / (trade.entry_price * trade.quantity) * 100
            
            # Update max adverse/favorable excursion
            if unrealized_pnl > 0:
                trade.max_favorable_excursion = max(trade.max_favorable_excursion or 0, unrealized_pnl_pct)
            else:
                trade.max_adverse_excursion = min(trade.max_adverse_excursion or 0, unrealized_pnl_pct)
            
            # Update daily PnL
            today = timestamp.date()
            self.daily_pnl[today] += unrealized_pnl - (trade.pnl or 0)  # Add change in PnL
            trade.pnl = unrealized_pnl
            trade.pnl_pct = unrealized_pnl_pct
            
        except Exception as e:
            logger.error(f"Error updating trade {trade_id}: {e}")
    
    async def close_trade(self, trade_id: str, exit_price: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Close a trade and finalize metrics"""
        try:
            if trade_id not in self.trades:
                return {}
            
            trade = self.trades[trade_id]
            timestamp = timestamp or datetime.now()
            
            # Calculate final PnL
            if trade.direction == 'long':
                realized_pnl = (exit_price - trade.entry_price) * trade.quantity
            else:
                realized_pnl = (trade.entry_price - exit_price) * trade.quantity
            
            realized_pnl_pct = realized_pnl / (trade.entry_price * trade.quantity) * 100
            
            # Update trade metrics
            trade.exit_price = exit_price
            trade.exit_time = timestamp
            trade.pnl = realized_pnl
            trade.pnl_pct = realized_pnl_pct
            trade.holding_period = timestamp - trade.entry_time
            trade.trade_status = 'closed'
            
            # Update daily PnL
            today = timestamp.date()
            self.daily_pnl[today] = realized_pnl
            
            # Check for alerts
            await self._check_performance_alerts()
            
            logger.info(f"Closed trade {trade_id}: PnL = {realized_pnl:.2f} ({realized_pnl_pct:.2f}%)")
            
            return {
                'trade_id': trade_id,
                'pnl': realized_pnl,
                'pnl_pct': realized_pnl_pct,
                'holding_period': trade.holding_period
            }
            
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            return {}
    
    async def calculate_performance_metrics(self, start_date: Optional[datetime] = None, 
                                          end_date: Optional[datetime] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Filter trades by date range
            filtered_trades = self._filter_trades_by_date(start_date, end_date)
            
            if not filtered_trades:
                return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            # Basic trade statistics
            total_trades = len(filtered_trades)
            closed_trades = [t for t in filtered_trades if t.trade_status == 'closed' and t.pnl is not None]
            
            if not closed_trades:
                return PerformanceMetrics(total_trades, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            # Win/Loss analysis
            winning_trades = len([t for t in closed_trades if t.pnl > 0])
            losing_trades = len([t for t in closed_trades if t.pnl <= 0])
            win_rate = winning_trades / len(closed_trades) if closed_trades else 0
            
            # Average win/loss
            wins = [t.pnl for t in closed_trades if t.pnl > 0]
            losses = [t.pnl for t in closed_trades if t.pnl <= 0]
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            
            # Profit factor
            total_wins = sum(wins)
            total_losses = abs(sum(losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Returns analysis
            returns = [t.pnl_pct / 100 for t in closed_trades if t.pnl_pct is not None]
            total_return = sum(returns) if returns else 0
            
            # Time-based metrics
            trading_days = self._calculate_trading_days(start_date, end_date, closed_trades)
            annualized_return = total_return * 252 / trading_days if trading_days > 0 else 0
            
            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252) if returns else 0
            
            # Sharpe ratio
            excess_returns = [r - self.risk_free_rate / 252 for r in returns] if returns else []
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if excess_returns and np.std(excess_returns) > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = np.cumsum(returns) if returns else []
            max_drawdown = self._calculate_max_drawdown(cumulative_returns)
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # VaR calculation
            var_95 = np.percentile(returns, 5) if returns else 0
            
            # Higher moments
            skewness = self._calculate_skewness(returns)
            kurtosis = self._calculate_kurtosis(returns)
            
            metrics = PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                var_95=var_95,
                skewness=skewness,
                kurtosis=kurtosis
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    async def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Calculate metrics for different periods
            all_time_metrics = await self.calculate_performance_metrics()
            ytd_metrics = await self.calculate_performance_metrics(
                start_date=datetime(datetime.now().year, 1, 1)
            )
            monthly_metrics = await self.calculate_performance_metrics(
                start_date=datetime.now() - timedelta(days=30)
            )
            weekly_metrics = await self.calculate_performance_metrics(
                start_date=datetime.now() - timedelta(days=7)
            )
            
            # Trade analysis
            trade_analysis = await self._analyze_trades()
            
            # Risk analysis
            risk_metrics = await self._calculate_risk_metrics()
            
            # Performance attribution
            attribution = await self._calculate_performance_attribution()
            
            # Create comprehensive report
            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_trades': all_time_metrics.total_trades,
                    'win_rate': f"{all_time_metrics.win_rate:.2%}",
                    'profit_factor': f"{all_time_metrics.profit_factor:.2f}",
                    'total_return': f"{all_time_metrics.total_return:.2%}",
                    'max_drawdown': f"{all_time_metrics.max_drawdown:.2%}",
                    'sharpe_ratio': f"{all_time_metrics.sharpe_ratio:.2f}"
                },
                'performance_metrics': {
                    'all_time': asdict(all_time_metrics),
                    'year_to_date': asdict(ytd_metrics),
                    'last_30_days': asdict(monthly_metrics),
                    'last_7_days': asdict(weekly_metrics)
                },
                'trade_analysis': trade_analysis,
                'risk_metrics': asdict(risk_metrics),
                'performance_attribution': attribution,
                'recent_trades': self._get_recent_trades(10),
                'daily_pnl': dict(self.daily_pnl)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    async def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trading patterns and statistics"""
        try:
            closed_trades = [t for t in self.trades.values() if t.trade_status == 'closed']
            
            if not closed_trades:
                return {}
            
            # Winning vs losing streaks
            streak_analysis = self._analyze_win_loss_streaks(closed_trades)
            
            # Holding period analysis
            holding_periods = [t.holding_period.total_seconds() / 3600 for t in closed_trades if t.holding_period]
            
            # Trade timing analysis
            hour_analysis = self._analyze_trade_timing(closed_trades)
            
            # Symbol performance
            symbol_performance = self._analyze_symbol_performance(closed_trades)
            
            return {
                'streaks': streak_analysis,
                'holding_period': {
                    'avg_hours': np.mean(holding_periods) if holding_periods else 0,
                    'median_hours': np.median(holding_periods) if holding_periods else 0,
                    'min_hours': np.min(holding_periods) if holding_periods else 0,
                    'max_hours': np.max(holding_periods) if holding_periods else 0
                },
                'timing_analysis': hour_analysis,
                'symbol_performance': symbol_performance
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return {}
    
    async def _calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            closed_trades = [t for t in self.trades.values() if t.trade_status == 'closed' and t.pnl is not None]
            
            if not closed_trades:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
            
            returns = [t.pnl_pct / 100 for t in closed_trades if t.pnl_pct is not None]
            
            # Current drawdown
            cumulative_returns = np.cumsum(returns) if returns else []
            current_drawdown = self._calculate_current_drawdown(cumulative_returns)
            
            # Max consecutive losses
            max_consecutive_losses = self._calculate_max_consecutive_losses(closed_trades)
            
            # Tail ratio
            tail_ratio = self._calculate_tail_ratio(returns)
            
            # Downside deviation
            downside_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(downside_returns) if downside_returns else 0
            
            # Beta and alpha (vs benchmark)
            beta, alpha = await self._calculate_beta_alpha(returns)
            
            # Information ratio
            information_ratio = await self._calculate_information_ratio(returns)
            
            # Tracking error
            tracking_error = await self._calculate_tracking_error(returns)
            
            return RiskMetrics(
                current_drawdown=current_drawdown,
                max_consecutive_losses=max_consecutive_losses,
                tail_ratio=tail_ratio,
                downside_deviation=downside_deviation,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                tracking_error=tracking_error
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    async def _check_performance_alerts(self):
        """Check for performance alerts and send notifications"""
        try:
            metrics = await self.calculate_performance_metrics()
            
            alerts = []
            
            # Drawdown alert
            if abs(metrics.max_drawdown) > self.alert_thresholds['max_drawdown']:
                alerts.append(f"Max drawdown exceeded: {metrics.max_drawdown:.2%}")
            
            # Daily loss limit
            today_pnl = self.daily_pnl.get(datetime.now().date(), 0)
            if today_pnl < -self.alert_thresholds['daily_loss_limit']:
                alerts.append(f"Daily loss limit exceeded: {today_pnl:.2%}")
            
            # VaR breach
            if abs(metrics.var_95) > self.alert_thresholds['var_breach']:
                alerts.append(f"VaR breached: {metrics.var_95:.2%}")
            
            # Consecutive losses
            consecutive_losses = self._get_current_consecutive_losses()
            if consecutive_losses > self.alert_thresholds['consecutive_losses']:
                alerts.append(f"Consecutive losses: {consecutive_losses}")
            
            if alerts:
                await self._send_alerts(alerts)
                
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    # Helper methods
    def _filter_trades_by_date(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> List[TradeMetrics]:
        """Filter trades by date range"""
        trades = list(self.trades.values())
        
        if start_date:
            trades = [t for t in trades if t.entry_time >= start_date]
        
        if end_date:
            trades = [t for t in trades if t.entry_time <= end_date]
        
        return trades
    
    def _calculate_trading_days(self, start_date: Optional[datetime], end_date: Optional[datetime], trades: List[TradeMetrics]) -> int:
        """Calculate number of trading days"""
        if not trades:
            return 1
        
        first_trade = min(trades, key=lambda t: t.entry_time)
        last_trade = max(trades, key=lambda t: t.entry_time)
        
        start = start_date or first_trade.entry_time
        end = end_date or last_trade.entry_time
        
        return max((end - start).days, 1)
    
    def _calculate_max_drawdown(self, cumulative_returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not cumulative_returns:
            return 0
        
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        return np.min(drawdowns)
    
    def _calculate_current_drawdown(self, cumulative_returns: List[float]) -> float:
        """Calculate current drawdown"""
        if not cumulative_returns:
            return 0
        
        peak = np.max(cumulative_returns)
        current = cumulative_returns[-1]
        return current - peak
    
    def _calculate_skewness(self, returns: List[float]) -> float:
        """Calculate skewness"""
        if len(returns) < 3:
            return 0
        return float(pd.Series(returns).skew())
    
    def _calculate_kurtosis(self, returns: List[float]) -> float:
        """Calculate kurtosis"""
        if len(returns) < 4:
            return 0
        return float(pd.Series(returns).kurtosis())
    
    def _get_recent_trades(self, count: int) -> List[Dict[str, Any]]:
        """Get recent trades"""
        recent = sorted(self.trades.values(), key=lambda t: t.entry_time, reverse=True)[:count]
        return [asdict(trade) for trade in recent]
    
    def _get_current_consecutive_losses(self) -> int:
        """Get current consecutive losses"""
        closed_trades = sorted([t for t in self.trades.values() if t.trade_status == 'closed'], 
                             key=lambda t: t.entry_time, reverse=True)
        
        consecutive = 0
        for trade in closed_trades:
            if trade.pnl and trade.pnl <= 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    # Mock implementations for complex calculations
    def _analyze_win_loss_streaks(self, trades: List[TradeMetrics]) -> Dict[str, Any]:
        """Analyze win/loss streaks"""
        return {'max_win_streak': 5, 'max_loss_streak': 3, 'current_streak': 2}
    
    def _analyze_trade_timing(self, trades: List[TradeMetrics]) -> Dict[str, Any]:
        """Analyze trade timing patterns"""
        return {'best_hour': 10, 'worst_hour': 15, 'best_day': 'Tuesday'}
    
    def _analyze_symbol_performance(self, trades: List[TradeMetrics]) -> Dict[str, Any]:
        """Analyze performance by symbol"""
        return {'AAPL': {'trades': 10, 'win_rate': 0.7}, 'TSLA': {'trades': 8, 'win_rate': 0.6}}
    
    async def _calculate_performance_attribution(self) -> Dict[str, Any]:
        """Calculate performance attribution"""
        return {'sector_allocation': 0.02, 'security_selection': 0.03, 'timing': 0.01}
    
    def _calculate_max_consecutive_losses(self, trades: List[TradeMetrics]) -> int:
        """Calculate maximum consecutive losses"""
        return 3  # Mock implementation
    
    def _calculate_tail_ratio(self, returns: List[float]) -> float:
        """Calculate tail ratio"""
        if len(returns) < 10:
            return 1.0
        return abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5))
    
    async def _calculate_beta_alpha(self, returns: List[float]) -> Tuple[float, float]:
        """Calculate beta and alpha vs benchmark"""
        return 1.0, 0.02  # Mock implementation
    
    async def _calculate_information_ratio(self, returns: List[float]) -> float:
        """Calculate information ratio"""
        return 0.5  # Mock implementation
    
    async def _calculate_tracking_error(self, returns: List[float]) -> float:
        """Calculate tracking error"""
        return 0.03  # Mock implementation
    
    async def _send_alerts(self, alerts: List[str]):
        """Send performance alerts"""
        for alert in alerts:
            logger.warning(f"PERFORMANCE ALERT: {alert}")