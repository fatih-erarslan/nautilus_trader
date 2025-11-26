# Alpaca API Portfolio Management Research

## Overview
Alpaca's portfolio management capabilities provide comprehensive tools for tracking positions, analyzing performance, managing risk, and implementing sophisticated trading strategies. The API supports real-time portfolio monitoring, automatic rebalancing, and advanced analytics.

## Portfolio Management Components

### 1. Account Information and Portfolio Summary
```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest
import pandas as pd

class PortfolioManager:
    def __init__(self, api_key, secret_key, paper=True):
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper
        )

    def get_account_summary(self):
        """Get comprehensive account information"""
        account = self.trading_client.get_account()

        return {
            'account_id': account.id,
            'status': account.status,
            'currency': account.currency,
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'long_market_value': float(account.long_market_value),
            'short_market_value': float(account.short_market_value),
            'equity': float(account.equity),
            'last_equity': float(account.last_equity),
            'multiplier': int(account.multiplier),
            'initial_margin': float(account.initial_margin),
            'maintenance_margin': float(account.maintenance_margin),
            'sma': float(account.sma),
            'daytrade_count': account.daytrade_count,
            'daytrading_buying_power': float(account.daytrading_buying_power),
            'regt_buying_power': float(account.regt_buying_power),
            'pattern_day_trader': account.pattern_day_trader,
            'trading_blocked': account.trading_blocked,
            'transfers_blocked': account.transfers_blocked,
            'account_blocked': account.account_blocked
        }

    def get_portfolio_history(self, period='1M', timeframe='1Day'):
        """Get historical portfolio performance"""
        request = GetPortfolioHistoryRequest(
            period=period,
            timeframe=timeframe,
            extended_hours=True
        )

        portfolio_history = self.trading_client.get_portfolio_history(request)

        # Convert to DataFrame for analysis
        df = pd.DataFrame({
            'timestamp': portfolio_history.timestamp,
            'equity': portfolio_history.equity,
            'profit_loss': portfolio_history.profit_loss,
            'profit_loss_pct': portfolio_history.profit_loss_pct,
            'base_value': portfolio_history.base_value
        })

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)

        return df
```

### 2. Position Tracking and Management
```python
from alpaca.trading.requests import ClosePositionRequest
from alpaca.trading.enums import OrderSide

class PositionTracker:
    def __init__(self, trading_client):
        self.trading_client = trading_client

    def get_all_positions(self):
        """Get all current positions with detailed information"""
        positions = self.trading_client.get_all_positions()

        position_data = []
        for position in positions:
            position_info = {
                'symbol': position.symbol,
                'qty': float(position.qty),
                'side': position.side,
                'market_value': float(position.market_value),
                'cost_basis': float(position.cost_basis),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'unrealized_intraday_pl': float(position.unrealized_intraday_pl),
                'unrealized_intraday_plpc': float(position.unrealized_intraday_plpc),
                'current_price': float(position.current_price),
                'lastday_price': float(position.lastday_price),
                'change_today': float(position.change_today),
                'avg_entry_price': float(position.avg_entry_price)
            }
            position_data.append(position_info)

        return pd.DataFrame(position_data)

    def get_position(self, symbol):
        """Get specific position details"""
        try:
            position = self.trading_client.get_open_position(symbol)
            return {
                'symbol': position.symbol,
                'qty': float(position.qty),
                'market_value': float(position.market_value),
                'cost_basis': float(position.cost_basis),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'current_price': float(position.current_price),
                'avg_entry_price': float(position.avg_entry_price)
            }
        except Exception as e:
            return None

    def close_position(self, symbol, qty=None, percentage=None):
        """Close position (full or partial)"""
        close_request = ClosePositionRequest(
            qty=qty,
            percentage=percentage
        )

        try:
            order = self.trading_client.close_position(symbol, close_request)
            return order
        except Exception as e:
            print(f"Error closing position for {symbol}: {e}")
            return None

    def close_all_positions(self):
        """Close all open positions"""
        try:
            orders = self.trading_client.close_all_positions(cancel_orders=True)
            return orders
        except Exception as e:
            print(f"Error closing all positions: {e}")
            return None
```

### 3. Portfolio Analytics and Performance Metrics
```python
import numpy as np
from datetime import datetime, timedelta

class PortfolioAnalytics:
    def __init__(self, portfolio_manager):
        self.portfolio_manager = portfolio_manager

    def calculate_performance_metrics(self, period='1Y'):
        """Calculate comprehensive performance metrics"""
        portfolio_history = self.portfolio_manager.get_portfolio_history(period)

        if portfolio_history.empty:
            return None

        # Calculate returns
        portfolio_history['daily_return'] = portfolio_history['equity'].pct_change()
        portfolio_history['cumulative_return'] = (portfolio_history['equity'] / portfolio_history['equity'].iloc[0]) - 1

        # Performance metrics
        total_return = portfolio_history['cumulative_return'].iloc[-1]
        daily_returns = portfolio_history['daily_return'].dropna()

        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = self.calculate_sharpe_ratio(daily_returns)
        max_drawdown = self.calculate_max_drawdown(portfolio_history['equity'])

        # Additional metrics
        win_rate = self.calculate_win_rate(daily_returns)
        profit_factor = self.calculate_profit_factor(daily_returns)
        sortino_ratio = self.calculate_sortino_ratio(daily_returns)

        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(portfolio_history)) - 1,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(daily_returns),
            'current_equity': portfolio_history['equity'].iloc[-1],
            'starting_equity': portfolio_history['equity'].iloc[0]
        }

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_sortino_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

    def calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()

    def calculate_win_rate(self, returns):
        """Calculate percentage of winning trades"""
        winning_trades = returns[returns > 0]
        return len(winning_trades) / len(returns) if len(returns) > 0 else 0

    def calculate_profit_factor(self, returns):
        """Calculate profit factor (gross profit / gross loss)"""
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return profits / losses if losses != 0 else float('inf')

    def generate_performance_report(self, period='1Y'):
        """Generate comprehensive performance report"""
        metrics = self.calculate_performance_metrics(period)
        account_summary = self.portfolio_manager.get_account_summary()

        if not metrics:
            return "No performance data available"

        report = f"""
Portfolio Performance Report ({period})
{'='*50}
Account Information:
  Portfolio Value: ${account_summary['portfolio_value']:,.2f}
  Cash Available: ${account_summary['cash']:,.2f}
  Buying Power: ${account_summary['buying_power']:,.2f}

Performance Metrics:
  Total Return: {metrics['total_return']*100:.2f}%
  Annualized Return: {metrics['annualized_return']*100:.2f}%
  Volatility: {metrics['volatility']*100:.2f}%
  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
  Sortino Ratio: {metrics['sortino_ratio']:.3f}
  Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%

Trading Statistics:
  Win Rate: {metrics['win_rate']*100:.2f}%
  Profit Factor: {metrics['profit_factor']:.2f}
  Total Trading Days: {metrics['total_trades']}

Risk Metrics:
  Current Equity: ${metrics['current_equity']:,.2f}
  Starting Equity: ${metrics['starting_equity']:,.2f}
  Equity Growth: ${metrics['current_equity'] - metrics['starting_equity']:,.2f}
        """

        return report
```

### 4. Risk Management
```python
class RiskManager:
    def __init__(self, trading_client, max_position_size=0.1, max_portfolio_risk=0.02):
        self.trading_client = trading_client
        self.max_position_size = max_position_size  # 10% max per position
        self.max_portfolio_risk = max_portfolio_risk  # 2% max portfolio risk

    def calculate_position_size(self, symbol, entry_price, stop_loss_price, portfolio_value):
        """Calculate appropriate position size based on risk management"""
        if stop_loss_price >= entry_price:
            return 0  # Invalid stop loss

        # Calculate risk per share
        risk_per_share = entry_price - stop_loss_price

        # Calculate maximum risk amount
        max_risk_amount = portfolio_value * self.max_portfolio_risk

        # Calculate position size based on risk
        risk_based_shares = max_risk_amount / risk_per_share

        # Calculate position size based on max position percentage
        max_position_value = portfolio_value * self.max_position_size
        max_position_shares = max_position_value / entry_price

        # Use the smaller of the two
        recommended_shares = min(risk_based_shares, max_position_shares)

        return int(recommended_shares)

    def check_portfolio_risk(self):
        """Check current portfolio risk exposure"""
        account = self.trading_client.get_account()
        positions = self.trading_client.get_all_positions()

        portfolio_value = float(account.portfolio_value)
        total_exposure = 0
        position_risks = []

        for position in positions:
            position_value = abs(float(position.market_value))
            position_exposure = position_value / portfolio_value
            total_exposure += position_exposure

            position_risks.append({
                'symbol': position.symbol,
                'value': position_value,
                'exposure': position_exposure,
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc)
            })

        return {
            'total_exposure': total_exposure,
            'portfolio_value': portfolio_value,
            'position_risks': position_risks,
            'risk_within_limits': total_exposure <= 1.0,  # Not over-leveraged
            'max_position_exposure': max(pos['exposure'] for pos in position_risks) if position_risks else 0
        }

    def get_risk_alerts(self):
        """Generate risk alerts for portfolio"""
        risk_analysis = self.check_portfolio_risk()
        alerts = []

        # Check for over-concentration
        if risk_analysis['max_position_exposure'] > self.max_position_size:
            alerts.append(f"Position concentration exceeds {self.max_position_size*100}% limit")

        # Check for over-leverage
        if risk_analysis['total_exposure'] > 1.0:
            alerts.append(f"Portfolio is over-leveraged at {risk_analysis['total_exposure']*100:.1f}%")

        # Check for large unrealized losses
        for position in risk_analysis['position_risks']:
            if position['unrealized_plpc'] < -0.05:  # More than 5% loss
                alerts.append(f"{position['symbol']}: Large unrealized loss of {position['unrealized_plpc']*100:.1f}%")

        return alerts
```

### 5. Portfolio Rebalancing
```python
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class PortfolioRebalancer:
    def __init__(self, trading_client):
        self.trading_client = trading_client

    def rebalance_to_target_allocation(self, target_allocations, rebalance_threshold=0.05):
        """
        Rebalance portfolio to target allocations
        target_allocations: dict with symbol: target_percentage pairs
        """
        account = self.trading_client.get_account()
        portfolio_value = float(account.cash) + float(account.long_market_value)

        current_positions = self.get_current_allocations()
        rebalance_orders = []

        for symbol, target_pct in target_allocations.items():
            current_pct = current_positions.get(symbol, 0)
            allocation_diff = target_pct - current_pct

            # Only rebalance if difference exceeds threshold
            if abs(allocation_diff) > rebalance_threshold:
                target_value = portfolio_value * target_pct
                current_value = portfolio_value * current_pct
                value_diff = target_value - current_value

                # Get current price to calculate shares needed
                current_price = self.get_current_price(symbol)
                if current_price:
                    shares_needed = int(value_diff / current_price)

                    if shares_needed != 0:
                        order = self.create_rebalance_order(symbol, shares_needed)
                        if order:
                            rebalance_orders.append(order)

        return rebalance_orders

    def get_current_allocations(self):
        """Get current portfolio allocations as percentages"""
        account = self.trading_client.get_account()
        positions = self.trading_client.get_all_positions()

        portfolio_value = float(account.cash) + float(account.long_market_value)
        allocations = {}

        for position in positions:
            symbol = position.symbol
            position_value = float(position.market_value)
            allocation_pct = position_value / portfolio_value
            allocations[symbol] = allocation_pct

        return allocations

    def create_rebalance_order(self, symbol, shares):
        """Create buy or sell order for rebalancing"""
        try:
            if shares > 0:
                # Buy order
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
            else:
                # Sell order
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=abs(shares),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )

            order = self.trading_client.submit_order(order_request)
            return order

        except Exception as e:
            print(f"Error creating rebalance order for {symbol}: {e}")
            return None

    def get_current_price(self, symbol):
        """Get current market price for symbol"""
        try:
            # This would typically use the market data API
            # For now, we'll get it from the position if it exists
            position = self.trading_client.get_open_position(symbol)
            return float(position.current_price)
        except:
            return None

    def schedule_periodic_rebalancing(self, target_allocations, frequency='monthly'):
        """Schedule automatic rebalancing"""
        # This would integrate with a scheduler like APScheduler
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler()

        if frequency == 'daily':
            scheduler.add_job(
                func=self.rebalance_to_target_allocation,
                trigger="cron",
                hour=9, minute=45,  # 15 minutes after market open
                args=[target_allocations]
            )
        elif frequency == 'weekly':
            scheduler.add_job(
                func=self.rebalance_to_target_allocation,
                trigger="cron",
                day_of_week='mon', hour=9, minute=45,
                args=[target_allocations]
            )
        elif frequency == 'monthly':
            scheduler.add_job(
                func=self.rebalance_to_target_allocation,
                trigger="cron",
                day=1, hour=9, minute=45,
                args=[target_allocations]
            )

        scheduler.start()
        return scheduler
```

### 6. Portfolio Optimization
```python
import scipy.optimize as sco
from scipy import stats

class PortfolioOptimizer:
    def __init__(self, returns_data):
        self.returns_data = returns_data  # DataFrame with symbol returns
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        self.num_assets = len(self.mean_returns)

    def portfolio_performance(self, weights):
        """Calculate portfolio performance metrics"""
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return portfolio_return, portfolio_std

    def negative_sharpe_ratio(self, weights, risk_free_rate=0.02):
        """Calculate negative Sharpe ratio for optimization"""
        p_return, p_std = self.portfolio_performance(weights)
        return -(p_return - risk_free_rate) / p_std

    def optimize_sharpe_ratio(self):
        """Find portfolio weights that maximize Sharpe ratio"""
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)

        result = sco.minimize(
            self.negative_sharpe_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return dict(zip(self.returns_data.columns, result.x))

    def optimize_minimum_variance(self):
        """Find minimum variance portfolio"""
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)

        result = sco.minimize(
            portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return dict(zip(self.returns_data.columns, result.x))

    def efficient_frontier(self, num_portfolios=100):
        """Generate efficient frontier"""
        results = np.zeros((3, num_portfolios))
        weights_array = np.zeros((num_portfolios, self.num_assets))

        # Define target returns
        target_returns = np.linspace(self.mean_returns.min(), self.mean_returns.max(), num_portfolios)

        for i, target in enumerate(target_returns):
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, target=target: np.sum(self.mean_returns * x) - target}
            ]
            bounds = tuple((0, 1) for _ in range(self.num_assets))
            initial_guess = np.array([1/self.num_assets] * self.num_assets)

            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(self.cov_matrix, weights))

            result = sco.minimize(
                portfolio_variance,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                weights_array[i] = result.x
                p_return, p_std = self.portfolio_performance(result.x)
                results[0, i] = p_return
                results[1, i] = p_std
                results[2, i] = (p_return - 0.02) / p_std  # Sharpe ratio

        return results, weights_array
```

### 7. Complete Portfolio Management System
```python
class CompletePortfolioManager:
    def __init__(self, api_key, secret_key, paper=True):
        self.trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
        self.portfolio_manager = PortfolioManager(api_key, secret_key, paper)
        self.position_tracker = PositionTracker(self.trading_client)
        self.analytics = PortfolioAnalytics(self.portfolio_manager)
        self.risk_manager = RiskManager(self.trading_client)
        self.rebalancer = PortfolioRebalancer(self.trading_client)

    def daily_portfolio_check(self):
        """Perform daily portfolio health check"""
        print("Daily Portfolio Check")
        print("=" * 30)

        # Account summary
        account_summary = self.portfolio_manager.get_account_summary()
        print(f"Portfolio Value: ${account_summary['portfolio_value']:,.2f}")
        print(f"Cash Available: ${account_summary['cash']:,.2f}")
        print(f"Buying Power: ${account_summary['buying_power']:,.2f}")

        # Risk analysis
        risk_alerts = self.risk_manager.get_risk_alerts()
        if risk_alerts:
            print("\n⚠️ Risk Alerts:")
            for alert in risk_alerts:
                print(f"  - {alert}")
        else:
            print("\n✅ No risk alerts")

        # Performance metrics
        metrics = self.analytics.calculate_performance_metrics('1M')
        if metrics:
            print(f"\nMonthly Performance:")
            print(f"  Return: {metrics['total_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")

        # Current positions
        positions = self.position_tracker.get_all_positions()
        if not positions.empty:
            print(f"\nCurrent Positions ({len(positions)}):")
            for _, pos in positions.iterrows():
                print(f"  {pos['symbol']}: {pos['qty']} shares, "
                      f"P&L: {pos['unrealized_plpc']*100:+.2f}%")

    def execute_strategy(self, strategy_config):
        """Execute a complete trading strategy"""
        # This would integrate with your trading strategy logic
        pass

    def generate_daily_report(self):
        """Generate comprehensive daily report"""
        return self.analytics.generate_performance_report('1D')

    def emergency_stop(self):
        """Emergency stop - close all positions"""
        print("🚨 EMERGENCY STOP - Closing all positions")
        closed_orders = self.position_tracker.close_all_positions()
        return closed_orders
```

This comprehensive portfolio management system provides all the tools needed for sophisticated portfolio tracking, risk management, performance analysis, and automated rebalancing using the Alpaca API.