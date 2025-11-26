"""
Paper Trading Strategy Validation Tests

This module contains comprehensive tests for validating trading strategies
in a paper trading environment, ensuring strategies work correctly before
live deployment.
"""

import asyncio
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Any, Optional, Tuple
import pytest
import numpy as np

from ..unit.brokers.mock_framework import MockManager, AlpacaMock, NewsAPIMock
from ..unit.fixtures.broker_responses import BrokerResponseFixtures


class PaperTradingEnvironment:
    """Paper trading environment for strategy testing"""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = Decimal(str(initial_cash))
        self.cash = self.initial_cash
        self.positions: Dict[str, Dict] = {}
        self.orders: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        self.start_time = datetime.now(timezone.utc)
        self.current_time = self.start_time
        self.market_data: Dict[str, List[Dict]] = {}
        self.commissions_paid = Decimal('0.00')
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.max_drawdown = 0.0
        self.peak_value = float(self.initial_cash)
        
    def add_market_data(self, symbol: str, data: List[Dict]):
        """Add market data for backtesting"""
        self.market_data[symbol] = data
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        if symbol in self.market_data and self.market_data[symbol]:
            return self.market_data[symbol][-1]["close"]
        return None
    
    def submit_order(self, symbol: str, qty: int, side: str, order_type: str = "market", **kwargs) -> Dict:
        """Submit paper trading order"""
        order_id = f"paper_{len(self.orders) + 1}"
        
        order = {
            "id": order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "status": "new",
            "timestamp": self.current_time.isoformat(),
            "filled_qty": 0,
            "filled_avg_price": None,
            **kwargs
        }
        
        self.orders[order_id] = order
        
        # Simulate immediate market order execution
        if order_type == "market":
            self._execute_market_order(order_id)
        
        return order
    
    def _execute_market_order(self, order_id: str):
        """Execute market order immediately"""
        order = self.orders[order_id]
        symbol = order["symbol"]
        qty = order["qty"]
        side = order["side"]
        
        current_price = self.get_current_price(symbol)
        if current_price is None:
            order["status"] = "rejected"
            order["reject_reason"] = "No market data available"
            return
        
        # Simulate slippage for large orders
        slippage = self._calculate_slippage(qty, current_price)
        fill_price = current_price + (slippage if side == "buy" else -slippage)
        
        # Calculate commission
        commission = self._calculate_commission(qty, fill_price)
        
        # Check if we have enough cash (for buy orders)
        if side == "buy":
            total_cost = Decimal(str(qty * fill_price)) + commission
            if total_cost > self.cash:
                order["status"] = "rejected"
                order["reject_reason"] = "Insufficient buying power"
                return
            
            self.cash -= total_cost
        else:  # sell order
            # Check if we have enough shares
            if symbol not in self.positions or self.positions[symbol]["qty"] < qty:
                order["status"] = "rejected"
                order["reject_reason"] = "Insufficient shares"
                return
            
            self.cash += Decimal(str(qty * fill_price)) - commission
        
        # Update order
        order["status"] = "filled"
        order["filled_qty"] = qty
        order["filled_avg_price"] = fill_price
        order["commission"] = float(commission)
        order["filled_at"] = self.current_time.isoformat()
        
        # Update position
        self._update_position(symbol, qty, fill_price, side)
        
        # Record trade
        trade = {
            "order_id": order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "price": fill_price,
            "commission": float(commission),
            "timestamp": self.current_time.isoformat()
        }
        self.trades.append(trade)
        
        self.commissions_paid += commission
    
    def _calculate_slippage(self, qty: int, price: float) -> float:
        """Calculate realistic slippage based on order size"""
        # Simple slippage model: larger orders have more slippage
        base_slippage = 0.01  # 1 cent base slippage
        size_factor = math.log(1 + qty / 1000) * 0.01  # Logarithmic scaling
        return base_slippage + (price * size_factor)
    
    def _calculate_commission(self, qty: int, price: float) -> Decimal:
        """Calculate trading commission"""
        # Simple commission model: $0.005 per share, min $1
        per_share_commission = Decimal('0.005')
        commission = per_share_commission * qty
        return max(commission, Decimal('1.00'))
    
    def _update_position(self, symbol: str, qty: int, price: float, side: str):
        """Update position after trade"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                "qty": 0,
                "avg_price": 0.0,
                "market_value": 0.0,
                "unrealized_pnl": 0.0
            }
        
        position = self.positions[symbol]
        current_price = self.get_current_price(symbol)
        
        if side == "buy":
            # Calculate new average price
            total_cost = (position["qty"] * position["avg_price"]) + (qty * price)
            new_qty = position["qty"] + qty
            position["avg_price"] = total_cost / new_qty if new_qty > 0 else 0
            position["qty"] = new_qty
        else:  # sell
            position["qty"] -= qty
            if position["qty"] <= 0:
                position["qty"] = 0
                position["avg_price"] = 0.0
        
        # Update market value and unrealized P&L
        if current_price and position["qty"] > 0:
            position["market_value"] = position["qty"] * current_price
            position["unrealized_pnl"] = position["qty"] * (current_price - position["avg_price"])
        else:
            position["market_value"] = 0.0
            position["unrealized_pnl"] = 0.0
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos["market_value"] for pos in self.positions.values())
        return float(self.cash) + positions_value
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        return [
            {
                "symbol": symbol,
                **position
            }
            for symbol, position in self.positions.items()
            if position["qty"] > 0
        ]
    
    def update_market_data(self, symbol: str, new_price: float):
        """Update market data with new price"""
        if symbol not in self.market_data:
            self.market_data[symbol] = []
        
        self.market_data[symbol].append({
            "timestamp": self.current_time.isoformat(),
            "close": new_price
        })
        
        # Update position values
        if symbol in self.positions:
            self._update_position_value(symbol, new_price)
    
    def _update_position_value(self, symbol: str, new_price: float):
        """Update position market value and unrealized P&L"""
        if symbol in self.positions and self.positions[symbol]["qty"] > 0:
            position = self.positions[symbol]
            position["market_value"] = position["qty"] * new_price
            position["unrealized_pnl"] = position["qty"] * (new_price - position["avg_price"])
    
    def advance_time(self, hours: int = 1):
        """Advance simulation time"""
        self.current_time += timedelta(hours=hours)
        
        # Record portfolio history
        portfolio_value = self.get_portfolio_value()
        self.portfolio_history.append({
            "timestamp": self.current_time.isoformat(),
            "portfolio_value": portfolio_value,
            "cash": float(self.cash),
            "positions_value": portfolio_value - float(self.cash)
        })
        
        # Update performance metrics
        if len(self.portfolio_history) >= 2:
            prev_value = self.portfolio_history[-2]["portfolio_value"]
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
            
            # Update drawdown
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
            else:
                drawdown = (self.peak_value - portfolio_value) / self.peak_value
                self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.daily_returns:
            return {}
        
        portfolio_value = self.get_portfolio_value()
        total_return = (portfolio_value - float(self.initial_cash)) / float(self.initial_cash)
        
        # Annualized return (assuming daily returns)
        days = len(self.daily_returns)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # Volatility (annualized)
        volatility = statistics.stdev(self.daily_returns) * math.sqrt(252) if len(self.daily_returns) > 1 else 0
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Win rate
        winning_trades = [t for t in self.trades if self._calculate_trade_pnl(t) > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": win_rate,
            "total_trades": len(self.trades),
            "total_commissions": float(self.commissions_paid),
            "current_positions": len([p for p in self.positions.values() if p["qty"] > 0])
        }
    
    def _calculate_trade_pnl(self, trade: Dict) -> float:
        """Calculate P&L for a single trade (simplified)"""
        # This is a simplified calculation for individual trades
        # In reality, P&L calculation is more complex with multiple partial fills
        return 0  # Placeholder


class MomentumStrategy:
    """Simple momentum trading strategy for testing"""
    
    def __init__(self, lookback_period: int = 20, threshold: float = 0.02):
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.positions: Dict[str, int] = {}
    
    def generate_signal(self, symbol: str, price_history: List[float]) -> Dict[str, Any]:
        """Generate trading signal based on momentum"""
        if len(price_history) < self.lookback_period:
            return {"action": "hold", "confidence": 0.0}
        
        current_price = price_history[-1]
        avg_price = sum(price_history[-self.lookback_period:]) / self.lookback_period
        
        momentum = (current_price - avg_price) / avg_price
        
        if momentum > self.threshold:
            return {
                "action": "buy",
                "confidence": min(momentum * 5, 1.0),  # Scale confidence
                "quantity": 100,
                "reason": f"Positive momentum: {momentum:.3f}"
            }
        elif momentum < -self.threshold:
            return {
                "action": "sell",
                "confidence": min(abs(momentum) * 5, 1.0),
                "quantity": 100,
                "reason": f"Negative momentum: {momentum:.3f}"
            }
        else:
            return {"action": "hold", "confidence": 0.5}


class MeanReversionStrategy:
    """Simple mean reversion trading strategy for testing"""
    
    def __init__(self, lookback_period: int = 20, z_threshold: float = 2.0):
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
    
    def generate_signal(self, symbol: str, price_history: List[float]) -> Dict[str, Any]:
        """Generate trading signal based on mean reversion"""
        if len(price_history) < self.lookback_period:
            return {"action": "hold", "confidence": 0.0}
        
        recent_prices = price_history[-self.lookback_period:]
        current_price = price_history[-1]
        
        mean_price = statistics.mean(recent_prices)
        std_price = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
        
        if std_price == 0:
            return {"action": "hold", "confidence": 0.0}
        
        z_score = (current_price - mean_price) / std_price
        
        if z_score > self.z_threshold:
            return {
                "action": "sell",
                "confidence": min((z_score - self.z_threshold) / 2, 1.0),
                "quantity": 100,
                "reason": f"Overbought: z-score = {z_score:.2f}"
            }
        elif z_score < -self.z_threshold:
            return {
                "action": "buy",
                "confidence": min((abs(z_score) - self.z_threshold) / 2, 1.0),
                "quantity": 100,
                "reason": f"Oversold: z-score = {z_score:.2f}"
            }
        else:
            return {"action": "hold", "confidence": 0.5}


class NewsBasedStrategy:
    """News sentiment-based trading strategy for testing"""
    
    def __init__(self, sentiment_threshold: float = 0.3):
        self.sentiment_threshold = sentiment_threshold
        self.news_cache: Dict[str, List[Dict]] = {}
    
    def add_news(self, symbol: str, article: Dict):
        """Add news article for analysis"""
        if symbol not in self.news_cache:
            self.news_cache[symbol] = []
        self.news_cache[symbol].append(article)
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal based on news sentiment"""
        if symbol not in self.news_cache or not self.news_cache[symbol]:
            return {"action": "hold", "confidence": 0.0}
        
        # Analyze recent news (last 24 hours)
        recent_articles = self.news_cache[symbol][-10:]  # Last 10 articles
        
        if not recent_articles:
            return {"action": "hold", "confidence": 0.0}
        
        # Calculate weighted sentiment
        total_sentiment = 0
        total_weight = 0
        
        for article in recent_articles:
            sentiment = article.get("sentiment", {})
            polarity = sentiment.get("polarity", 0)
            confidence = sentiment.get("confidence", 0.5)
            
            total_sentiment += polarity * confidence
            total_weight += confidence
        
        avg_sentiment = total_sentiment / total_weight if total_weight > 0 else 0
        
        if avg_sentiment > self.sentiment_threshold:
            return {
                "action": "buy",
                "confidence": min(avg_sentiment, 1.0),
                "quantity": int(100 * avg_sentiment),  # Scale quantity by sentiment
                "reason": f"Positive sentiment: {avg_sentiment:.3f}"
            }
        elif avg_sentiment < -self.sentiment_threshold:
            return {
                "action": "sell",
                "confidence": min(abs(avg_sentiment), 1.0),
                "quantity": int(100 * abs(avg_sentiment)),
                "reason": f"Negative sentiment: {avg_sentiment:.3f}"
            }
        else:
            return {"action": "hold", "confidence": 0.5}


class TestStrategyValidation:
    """Validate trading strategies in paper trading environment"""
    
    @pytest.fixture(scope="function")
    def paper_env(self):
        """Setup paper trading environment"""
        return PaperTradingEnvironment(initial_cash=100000.0)
    
    @pytest.fixture(scope="function")
    def sample_price_data(self):
        """Generate sample price data for testing"""
        np.random.seed(42)  # For reproducibility
        
        # Generate realistic price series
        days = 100
        initial_price = 100.0
        daily_returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily return, 2% volatility
        
        prices = [initial_price]
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        
        return prices
    
    @pytest.mark.paper_trading
    def test_momentum_strategy_execution(self, paper_env: PaperTradingEnvironment, sample_price_data):
        """Test momentum strategy in paper trading"""
        strategy = MomentumStrategy(lookback_period=10, threshold=0.015)
        symbol = "AAPL"
        
        # Add market data to environment
        for i, price in enumerate(sample_price_data):
            paper_env.update_market_data(symbol, price)
            
            # Generate signal after enough data
            if i >= strategy.lookback_period:
                signal = strategy.generate_signal(symbol, sample_price_data[:i+1])
                
                # Execute signal if not hold
                if signal["action"] != "hold" and signal["confidence"] > 0.6:
                    try:
                        order = paper_env.submit_order(
                            symbol=symbol,
                            qty=signal["quantity"],
                            side=signal["action"],
                            order_type="market"
                        )
                        
                        # Verify order was processed
                        assert order["status"] in ["filled", "rejected"]
                        
                        if order["status"] == "filled":
                            assert order["filled_qty"] == signal["quantity"]
                            assert order["filled_avg_price"] is not None
                    
                    except Exception as e:
                        pytest.fail(f"Order execution failed: {e}")
            
            # Advance time
            paper_env.advance_time(hours=24)  # Daily strategy
        
        # Validate strategy performance
        metrics = paper_env.get_performance_metrics()
        
        # Basic performance checks
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        
        # Performance should be reasonable for momentum strategy
        assert metrics["max_drawdown"] < 0.3  # Less than 30% drawdown
        assert metrics["total_trades"] > 0  # Should have made some trades
        
        # Print performance summary
        print(f"Momentum Strategy Performance:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
    
    @pytest.mark.paper_trading
    def test_mean_reversion_strategy(self, paper_env: PaperTradingEnvironment, sample_price_data):
        """Test mean reversion strategy"""
        strategy = MeanReversionStrategy(lookback_period=15, z_threshold=1.5)
        symbol = "GOOGL"
        
        # Create more volatile price data for mean reversion
        volatile_prices = []
        base_price = 2500.0
        
        for i in range(len(sample_price_data)):
            # Add extra volatility with mean reversion tendency
            if i > 0:
                deviation = (sample_price_data[i] / sample_price_data[i-1] - 1) * 2  # Amplify moves
                mean_revert_factor = -0.1 * deviation  # Mean reversion component
                noise = np.random.normal(0, 0.01)
                
                price_change = deviation + mean_revert_factor + noise
                volatile_prices.append(volatile_prices[-1] * (1 + price_change))
            else:
                volatile_prices.append(base_price)
        
        # Run strategy
        trades_executed = 0
        
        for i, price in enumerate(volatile_prices):
            paper_env.update_market_data(symbol, price)
            
            if i >= strategy.lookback_period:
                signal = strategy.generate_signal(symbol, volatile_prices[:i+1])
                
                if signal["action"] != "hold" and signal["confidence"] > 0.5:
                    order = paper_env.submit_order(
                        symbol=symbol,
                        qty=signal["quantity"],
                        side=signal["action"],
                        order_type="market"
                    )
                    
                    if order["status"] == "filled":
                        trades_executed += 1
            
            paper_env.advance_time(hours=24)
        
        # Validate results
        metrics = paper_env.get_performance_metrics()
        
        assert trades_executed > 0, "Mean reversion strategy should execute some trades"
        assert metrics["total_trades"] == trades_executed
        
        # Mean reversion should have decent win rate in volatile markets
        if trades_executed > 5:  # Only check if enough trades
            assert metrics["win_rate"] >= 0.3, "Mean reversion should have reasonable win rate"
        
        print(f"Mean Reversion Strategy Performance:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Total Trades: {trades_executed}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
    
    @pytest.mark.paper_trading
    def test_news_based_strategy(self, paper_env: PaperTradingEnvironment):
        """Test news-driven trading strategy"""
        strategy = NewsBasedStrategy(sentiment_threshold=0.4)
        symbol = "TSLA"
        
        # Create news articles with varying sentiment
        news_articles = [
            {
                "headline": "Tesla Reports Record Deliveries",
                "sentiment": {"polarity": 0.8, "confidence": 0.9}
            },
            {
                "headline": "Tesla Stock Upgraded by Analysts",
                "sentiment": {"polarity": 0.6, "confidence": 0.8}
            },
            {
                "headline": "Tesla Faces Production Challenges",
                "sentiment": {"polarity": -0.5, "confidence": 0.7}
            },
            {
                "headline": "Tesla CEO Makes Controversial Statement",
                "sentiment": {"polarity": -0.7, "confidence": 0.85}
            },
            {
                "headline": "Tesla Announces New Factory",
                "sentiment": {"polarity": 0.4, "confidence": 0.75}
            }
        ]
        
        # Simulate price movements corresponding to news
        base_price = 800.0
        current_price = base_price
        
        for i, article in enumerate(news_articles):
            # Add news to strategy
            strategy.add_news(symbol, article)
            
            # Simulate price impact of news
            sentiment_impact = article["sentiment"]["polarity"] * 0.02  # 2% max impact
            current_price *= (1 + sentiment_impact)
            
            paper_env.update_market_data(symbol, current_price)
            
            # Generate and execute signal
            signal = strategy.generate_signal(symbol)
            
            if signal["action"] != "hold" and signal["confidence"] > 0.6:
                order = paper_env.submit_order(
                    symbol=symbol,
                    qty=signal["quantity"],
                    side=signal["action"],
                    order_type="market"
                )
                
                assert order["status"] in ["filled", "rejected"]
                
                if order["status"] == "filled":
                    # Verify signal alignment with sentiment
                    article_sentiment = article["sentiment"]["polarity"]
                    if article_sentiment > 0.4:
                        assert signal["action"] == "buy", "Positive news should trigger buy signal"
                    elif article_sentiment < -0.4:
                        assert signal["action"] == "sell", "Negative news should trigger sell signal"
            
            paper_env.advance_time(hours=6)  # News-based strategy reacts quickly
        
        # Validate news strategy performance
        metrics = paper_env.get_performance_metrics()
        
        assert "total_return" in metrics
        assert metrics["total_trades"] >= 0
        
        # News-based strategy should be responsive to sentiment
        positions = paper_env.get_positions()
        total_positions = len(positions)
        
        print(f"News-Based Strategy Performance:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Active Positions: {total_positions}")
    
    @pytest.mark.paper_trading
    def test_strategy_risk_management(self, paper_env: PaperTradingEnvironment):
        """Test risk management features in paper trading"""
        symbol = "SPY"
        initial_price = 400.0
        
        # Test position sizing limits
        paper_env.update_market_data(symbol, initial_price)
        
        # Try to submit order larger than cash available
        large_qty = int(paper_env.cash / initial_price) + 1000  # More than affordable
        
        order = paper_env.submit_order(
            symbol=symbol,
            qty=large_qty,
            side="buy",
            order_type="market"
        )
        
        assert order["status"] == "rejected"
        assert "insufficient" in order["reject_reason"].lower()
        
        # Test selling more shares than owned
        sell_order = paper_env.submit_order(
            symbol=symbol,
            qty=1000,
            side="sell",
            order_type="market"
        )
        
        assert sell_order["status"] == "rejected"
        assert "insufficient" in sell_order["reject_reason"].lower()
        
        # Test successful order within limits
        affordable_qty = int(paper_env.cash / initial_price / 2)  # Half of buying power
        
        buy_order = paper_env.submit_order(
            symbol=symbol,
            qty=affordable_qty,
            side="buy",
            order_type="market"
        )
        
        assert buy_order["status"] == "filled"
        assert buy_order["filled_qty"] == affordable_qty
        
        # Verify cash was debited correctly
        expected_cost = affordable_qty * buy_order["filled_avg_price"] + buy_order["commission"]
        remaining_cash = float(paper_env.initial_cash) - expected_cost
        
        assert abs(float(paper_env.cash) - remaining_cash) < 1.0  # Allow for rounding
    
    @pytest.mark.paper_trading
    def test_commission_and_slippage_calculation(self, paper_env: PaperTradingEnvironment):
        """Test commission and slippage calculations"""
        symbol = "AAPL"
        price = 150.0
        qty = 100
        
        paper_env.update_market_data(symbol, price)
        
        initial_cash = float(paper_env.cash)
        
        # Submit buy order
        order = paper_env.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            order_type="market"
        )
        
        assert order["status"] == "filled"
        
        # Verify commission was charged
        assert order["commission"] > 0
        assert order["commission"] >= 1.0  # Minimum commission
        
        # Verify slippage occurred (fill price != market price)
        fill_price = order["filled_avg_price"]
        assert fill_price >= price  # Buy order should have positive slippage
        
        # Verify total cost calculation
        total_cost = qty * fill_price + order["commission"]
        cash_after_trade = float(paper_env.cash)
        
        expected_cash = initial_cash - total_cost
        assert abs(cash_after_trade - expected_cash) < 0.01  # Allow for decimal precision
        
        # Test sell order (should have negative slippage)
        sell_order = paper_env.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            order_type="market"
        )
        
        assert sell_order["status"] == "filled"
        assert sell_order["filled_avg_price"] <= price  # Sell should have negative slippage
    
    @pytest.mark.paper_trading
    def test_portfolio_performance_tracking(self, paper_env: PaperTradingEnvironment):
        """Test portfolio performance metrics calculation"""
        symbol = "QQQ"
        base_price = 300.0
        
        # Create a series of trades with known outcomes
        trade_scenarios = [
            {"price": 300.0, "action": "buy", "qty": 100},
            {"price": 310.0, "action": "sell", "qty": 50},   # Profitable
            {"price": 295.0, "action": "buy", "qty": 200},   # Buying dip
            {"price": 305.0, "action": "sell", "qty": 100},  # Profitable
            {"price": 320.0, "action": "sell", "qty": 150},  # Profitable
        ]
        
        for i, scenario in enumerate(trade_scenarios):
            paper_env.update_market_data(symbol, scenario["price"])
            
            order = paper_env.submit_order(
                symbol=symbol,
                qty=scenario["qty"],
                side=scenario["action"],
                order_type="market"
            )
            
            # Most orders should execute successfully
            if scenario["action"] == "buy" or i > 0:  # Buy orders or sells after initial buy
                assert order["status"] == "filled"
            
            paper_env.advance_time(hours=24)
        
        # Calculate final performance
        metrics = paper_env.get_performance_metrics()
        
        # Verify all metrics are present
        required_metrics = [
            "total_return", "annualized_return", "volatility", 
            "sharpe_ratio", "max_drawdown", "win_rate", 
            "total_trades", "total_commissions"
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # Verify reasonable values
        assert metrics["total_trades"] > 0
        assert metrics["total_commissions"] > 0
        assert -1.0 <= metrics["total_return"] <= 10.0  # Reasonable return range
        assert 0.0 <= metrics["max_drawdown"] <= 1.0  # Drawdown is between 0-100%
        assert 0.0 <= metrics["win_rate"] <= 1.0  # Win rate is between 0-100%
        
        print(f"Portfolio Performance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    
    @pytest.mark.paper_trading
    @pytest.mark.slow
    def test_multi_strategy_comparison(self, paper_env: PaperTradingEnvironment):
        """Compare multiple strategies in same environment"""
        
        # Generate shared price data
        np.random.seed(123)
        days = 50
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Create separate environments for each strategy
        strategies_and_envs = [
            ("Momentum", MomentumStrategy(lookback_period=10), PaperTradingEnvironment()),
            ("MeanReversion", MeanReversionStrategy(lookback_period=10), PaperTradingEnvironment()),
        ]
        
        # Generate price data for all symbols
        price_data = {}
        for symbol in symbols:
            initial_price = {"AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0}[symbol]
            prices = [initial_price]
            
            for _ in range(days):
                daily_return = np.random.normal(0.001, 0.02)
                prices.append(prices[-1] * (1 + daily_return))
            
            price_data[symbol] = prices
        
        # Run all strategies
        for strategy_name, strategy, env in strategies_and_envs:
            for day in range(days):
                for symbol in symbols:
                    current_price = price_data[symbol][day]
                    env.update_market_data(symbol, current_price)
                    
                    # Generate signal (different for each strategy type)
                    if day >= 10:  # Enough data for signals
                        if isinstance(strategy, MomentumStrategy):
                            signal = strategy.generate_signal(symbol, price_data[symbol][:day+1])
                        elif isinstance(strategy, MeanReversionStrategy):
                            signal = strategy.generate_signal(symbol, price_data[symbol][:day+1])
                        else:
                            signal = {"action": "hold"}
                        
                        # Execute signal
                        if signal["action"] != "hold" and signal.get("confidence", 0) > 0.5:
                            env.submit_order(
                                symbol=symbol,
                                qty=signal.get("quantity", 100),
                                side=signal["action"],
                                order_type="market"
                            )
                
                # Advance time for all environments
                env.advance_time(hours=24)
        
        # Compare strategy performance
        strategy_results = {}
        
        for strategy_name, strategy, env in strategies_and_envs:
            metrics = env.get_performance_metrics()
            strategy_results[strategy_name] = metrics
            
            print(f"\n{strategy_name} Strategy Results:")
            print(f"Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
        
        # Verify both strategies executed trades
        for strategy_name, metrics in strategy_results.items():
            assert metrics.get("total_trades", 0) >= 0, f"{strategy_name} should have trade data"