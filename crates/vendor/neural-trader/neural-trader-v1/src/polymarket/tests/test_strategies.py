"""
Polymarket Trading Strategies Tests

Following TDD principles - these tests validate trading strategy implementations,
signal generation, risk management, and portfolio optimization.
All tests should fail initially until strategies are implemented.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
import pandas as pd


class TestBaseStrategy:
    """Test base strategy functionality."""
    
    @pytest.mark.unit
    def test_strategy_initialization(self, polymarket_config):
        """Test strategy initialization with configuration."""
        from src.polymarket.strategies import BaseStrategy
        
        strategy_config = {
            "name": "test_strategy",
            "max_position_size": Decimal("1000"),
            "risk_tolerance": 0.05,
            "rebalance_frequency": "daily",
            "stop_loss_threshold": 0.10,
            "take_profit_threshold": 0.25
        }
        
        strategy = BaseStrategy(config=strategy_config, api_client=Mock())
        
        assert strategy.name == "test_strategy"
        assert strategy.max_position_size == Decimal("1000")
        assert strategy.risk_tolerance == 0.05
        assert strategy.positions == {}
        assert strategy.is_active == False
    
    @pytest.mark.unit
    def test_strategy_risk_management(self):
        """Test base risk management functionality."""
        from src.polymarket.strategies import BaseStrategy
        
        strategy = BaseStrategy(
            config={"risk_tolerance": 0.05, "max_position_size": Decimal("1000")},
            api_client=Mock()
        )
        
        # Test position size calculation
        market_data = {
            "market_id": "0x123",
            "outcome_prices": {"Yes": 0.60, "No": 0.40},
            "liquidity": 10000,
            "volume": 50000
        }
        
        confidence = 0.8
        size = strategy.calculate_position_size(market_data, confidence)
        
        assert size <= strategy.max_position_size
        assert size > 0
        
        # Higher confidence should result in larger position
        higher_confidence_size = strategy.calculate_position_size(market_data, 0.9)
        assert higher_confidence_size >= size
    
    @pytest.mark.unit
    def test_strategy_signal_validation(self):
        """Test trading signal validation."""
        from src.polymarket.strategies import BaseStrategy, TradingSignal
        
        strategy = BaseStrategy(config={}, api_client=Mock())
        
        # Valid signal
        valid_signal = TradingSignal(
            market_id="0x123",
            action="buy",
            outcome="Yes",
            confidence=0.75,
            size=Decimal("100"),
            reasoning="Strong bullish sentiment"
        )
        
        assert strategy.validate_signal(valid_signal) == True
        
        # Invalid signal - confidence too low
        invalid_signal = TradingSignal(
            market_id="0x123",
            action="buy",
            outcome="Yes",
            confidence=0.30,  # Below minimum threshold
            size=Decimal("100"),
            reasoning="Weak signal"
        )
        
        assert strategy.validate_signal(invalid_signal) == False
    
    @pytest.mark.unit
    async def test_strategy_lifecycle(self):
        """Test strategy start/stop lifecycle."""
        from src.polymarket.strategies import BaseStrategy
        
        strategy = BaseStrategy(config={}, api_client=AsyncMock())
        
        # Test start
        await strategy.start()
        assert strategy.is_active == True
        
        # Test stop
        await strategy.stop()
        assert strategy.is_active == False
        
        # Test restart
        await strategy.start()
        assert strategy.is_active == True


class TestSentimentStrategy:
    """Test sentiment-based trading strategy."""
    
    @pytest.mark.unit
    async def test_sentiment_analysis(self, mock_api_responses):
        """Test sentiment analysis for market prediction."""
        from src.polymarket.strategies import SentimentStrategy
        
        api_client = AsyncMock()
        strategy = SentimentStrategy(
            config={"sentiment_threshold": 0.6},
            api_client=api_client
        )
        
        # Mock news data
        news_data = [
            {"title": "Bitcoin surges to new highs", "sentiment": 0.8, "source": "CoinDesk"},
            {"title": "Crypto market shows strong momentum", "sentiment": 0.7, "source": "Reuters"},
            {"title": "Regulatory concerns dampen enthusiasm", "sentiment": -0.3, "source": "Bloomberg"}
        ]
        
        sentiment_score = await strategy.analyze_sentiment(news_data)
        
        # Should be positive overall
        assert sentiment_score > 0
        assert -1 <= sentiment_score <= 1
        
        # Test signal generation
        market_data = {
            "market_id": "0x123",
            "question": "Will BTC reach $100k by end of 2024?",
            "outcomes": ["Yes", "No"],
            "outcome_prices": {"Yes": 0.55, "No": 0.45}
        }
        
        signal = await strategy.generate_signal(market_data, news_data)
        
        if sentiment_score > strategy.config["sentiment_threshold"]:
            assert signal.action == "buy"
            assert signal.outcome == "Yes"
        else:
            assert signal is None or signal.action == "hold"
    
    @pytest.mark.unit
    async def test_sentiment_confidence_calculation(self):
        """Test confidence calculation based on sentiment strength."""
        from src.polymarket.strategies import SentimentStrategy
        
        strategy = SentimentStrategy(config={}, api_client=AsyncMock())
        
        # Test high confidence (strong, consistent sentiment)
        high_confidence_news = [
            {"sentiment": 0.9}, {"sentiment": 0.8}, {"sentiment": 0.85}
        ]
        confidence = strategy.calculate_confidence(high_confidence_news)
        assert confidence > 0.8
        
        # Test low confidence (mixed sentiment)
        mixed_sentiment_news = [
            {"sentiment": 0.7}, {"sentiment": -0.2}, {"sentiment": 0.1}
        ]
        confidence = strategy.calculate_confidence(mixed_sentiment_news)
        assert confidence < 0.5
        
        # Test medium confidence (moderate sentiment)
        medium_sentiment_news = [
            {"sentiment": 0.6}, {"sentiment": 0.5}, {"sentiment": 0.7}
        ]
        confidence = strategy.calculate_confidence(medium_sentiment_news)
        assert 0.5 <= confidence <= 0.8
    
    @pytest.mark.integration
    async def test_sentiment_real_time_updates(self):
        """Test real-time sentiment updates via WebSocket."""
        from src.polymarket.strategies import SentimentStrategy
        
        api_client = AsyncMock()
        strategy = SentimentStrategy(config={}, api_client=api_client)
        
        # Mock WebSocket connection
        ws_mock = AsyncMock()
        api_client.connect_websocket.return_value = ws_mock
        
        # Start strategy
        await strategy.start()
        
        # Simulate incoming sentiment updates
        sentiment_updates = [
            {"type": "sentiment_update", "sentiment": 0.8, "timestamp": datetime.now()},
            {"type": "sentiment_update", "sentiment": 0.6, "timestamp": datetime.now()},
        ]
        
        for update in sentiment_updates:
            await strategy.handle_sentiment_update(update)
        
        # Strategy should have processed updates
        assert len(strategy.sentiment_history) == 2
        assert strategy.current_sentiment == 0.6  # Latest update


class TestArbitrageStrategy:
    """Test arbitrage trading strategy."""
    
    @pytest.mark.unit
    async def test_arbitrage_detection(self):
        """Test detection of arbitrage opportunities."""
        from src.polymarket.strategies import ArbitrageStrategy
        
        strategy = ArbitrageStrategy(
            config={"min_profit_threshold": 0.02},
            api_client=AsyncMock()
        )
        
        # Test same market arbitrage (prices don't sum to 1)
        market_data = {
            "market_id": "0x123",
            "outcomes": ["Yes", "No"],
            "outcome_prices": {"Yes": 0.60, "No": 0.35}  # Sum = 0.95, opportunity exists
        }
        
        opportunity = await strategy.detect_same_market_arbitrage(market_data)
        
        assert opportunity is not None
        assert opportunity["profit_margin"] > 0
        assert opportunity["total_investment"] > 0
        assert opportunity["expected_profit"] > 0
        
        # Test cross-market arbitrage
        market1 = {
            "market_id": "0x123",
            "question": "Will BTC reach $100k?",
            "outcome_prices": {"Yes": 0.65, "No": 0.35}
        }
        
        market2 = {
            "market_id": "0x456",
            "question": "Will BTC reach $100k by end of year?",  # Similar question
            "outcome_prices": {"Yes": 0.55, "No": 0.45}
        }
        
        cross_opportunity = await strategy.detect_cross_market_arbitrage(market1, market2)
        
        if cross_opportunity:
            assert cross_opportunity["profit_margin"] > strategy.config["min_profit_threshold"]
    
    @pytest.mark.unit
    async def test_arbitrage_execution(self):
        """Test arbitrage execution strategy."""
        from src.polymarket.strategies import ArbitrageStrategy
        
        api_client = AsyncMock()
        strategy = ArbitrageStrategy(config={}, api_client=api_client)
        
        # Mock successful order placement
        api_client.place_order.return_value = {"order_id": "order_123", "status": "pending"}
        
        opportunity = {
            "market_id": "0x123",
            "trades": [
                {"outcome": "Yes", "action": "buy", "size": 100, "price": 0.60},
                {"outcome": "No", "action": "buy", "size": 100, "price": 0.35}
            ],
            "expected_profit": 5.0,
            "total_investment": 95.0
        }
        
        result = await strategy.execute_arbitrage(opportunity)
        
        assert result["success"] == True
        assert len(result["orders"]) == 2
        assert api_client.place_order.call_count == 2
        
        # Verify order parameters
        call_args = api_client.place_order.call_args_list
        assert call_args[0][1]["outcome"] == "Yes"
        assert call_args[1][1]["outcome"] == "No"
    
    @pytest.mark.unit
    async def test_arbitrage_risk_management(self):
        """Test arbitrage risk management."""
        from src.polymarket.strategies import ArbitrageStrategy
        
        strategy = ArbitrageStrategy(
            config={"max_arbitrage_size": 1000, "min_profit_margin": 0.02},
            api_client=AsyncMock()
        )
        
        # Test position sizing
        large_opportunity = {
            "total_investment": 5000,  # Exceeds max size
            "profit_margin": 0.05
        }
        
        adjusted_size = strategy.calculate_arbitrage_size(large_opportunity)
        assert adjusted_size <= strategy.config["max_arbitrage_size"]
        
        # Test minimum profit check
        low_profit_opportunity = {
            "profit_margin": 0.01,  # Below minimum
            "total_investment": 100
        }
        
        should_execute = strategy.should_execute_arbitrage(low_profit_opportunity)
        assert should_execute == False


class TestMomentumStrategy:
    """Test momentum-based trading strategy."""
    
    @pytest.mark.unit
    async def test_price_momentum_calculation(self):
        """Test price momentum calculation."""
        from src.polymarket.strategies import MomentumStrategy
        
        strategy = MomentumStrategy(
            config={"lookback_period": 24, "momentum_threshold": 0.05},
            api_client=AsyncMock()
        )
        
        # Create price history
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
        prices = [0.50 + 0.01 * i for i in range(24)]  # Upward trend
        
        price_history = [
            {"timestamp": ts, "price": price}
            for ts, price in zip(timestamps, prices)
        ]
        
        momentum = strategy.calculate_momentum(price_history)
        
        assert momentum > 0  # Positive momentum for upward trend
        assert isinstance(momentum, float)
        
        # Test downward trend
        downward_prices = [0.50 - 0.01 * i for i in range(24)]
        downward_history = [
            {"timestamp": ts, "price": price}
            for ts, price in zip(timestamps, downward_prices)
        ]
        
        downward_momentum = strategy.calculate_momentum(downward_history)
        assert downward_momentum < 0
    
    @pytest.mark.unit
    async def test_volume_momentum(self):
        """Test volume-based momentum analysis."""
        from src.polymarket.strategies import MomentumStrategy
        
        strategy = MomentumStrategy(config={}, api_client=AsyncMock())
        
        # High volume momentum
        volume_data = [
            {"timestamp": datetime.now() - timedelta(hours=i), "volume": 1000 + 100 * i}
            for i in range(12, 0, -1)
        ]
        
        volume_momentum = strategy.calculate_volume_momentum(volume_data)
        assert volume_momentum > 0
        
        # Test volume + price momentum combination
        market_data = {
            "price_history": [{"price": 0.60, "timestamp": datetime.now()}],
            "volume_history": volume_data
        }
        
        combined_signal = await strategy.generate_signal(market_data)
        
        if combined_signal:
            assert combined_signal.confidence > 0
            assert combined_signal.reasoning.find("momentum") != -1
    
    @pytest.mark.unit
    async def test_momentum_reversal_detection(self):
        """Test momentum reversal detection."""
        from src.polymarket.strategies import MomentumStrategy
        
        strategy = MomentumStrategy(
            config={"reversal_threshold": 0.1},
            api_client=AsyncMock()
        )
        
        # Create reversal pattern (up then down)
        reversal_prices = (
            [0.40 + 0.02 * i for i in range(10)] +  # Upward momentum
            [0.60 - 0.03 * i for i in range(10)]    # Reversal downward
        )
        
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(20, 0, -1)]
        price_history = [
            {"timestamp": ts, "price": price}
            for ts, price in zip(timestamps, reversal_prices)
        ]
        
        reversal_detected = strategy.detect_momentum_reversal(price_history)
        assert reversal_detected == True
        
        # Test no reversal (consistent trend)
        consistent_prices = [0.40 + 0.01 * i for i in range(20)]
        consistent_history = [
            {"timestamp": ts, "price": price}
            for ts, price in zip(timestamps, consistent_prices)
        ]
        
        no_reversal = strategy.detect_momentum_reversal(consistent_history)
        assert no_reversal == False


class TestMeanReversionStrategy:
    """Test mean reversion trading strategy."""
    
    @pytest.mark.unit
    async def test_mean_calculation(self):
        """Test rolling mean calculation."""
        from src.polymarket.strategies import MeanReversionStrategy
        
        strategy = MeanReversionStrategy(
            config={"lookback_period": 20, "std_dev_threshold": 2.0},
            api_client=AsyncMock()
        )
        
        # Create price data
        prices = [0.50 + 0.05 * np.sin(i / 5) for i in range(50)]  # Oscillating around 0.50
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(50, 0, -1)]
        
        price_history = [
            {"timestamp": ts, "price": price}
            for ts, price in zip(timestamps, prices)
        ]
        
        mean_price = strategy.calculate_rolling_mean(price_history)
        std_dev = strategy.calculate_rolling_std(price_history)
        
        assert abs(mean_price - 0.50) < 0.1  # Should be close to center
        assert std_dev > 0
        
        # Test deviation from mean
        current_price = 0.40  # Below mean
        deviation = strategy.calculate_deviation(current_price, mean_price, std_dev)
        
        assert deviation < -1  # Negative deviation (below mean)
    
    @pytest.mark.unit
    async def test_reversion_signal_generation(self):
        """Test mean reversion signal generation."""
        from src.polymarket.strategies import MeanReversionStrategy
        
        strategy = MeanReversionStrategy(
            config={"std_dev_threshold": 1.5, "confidence_scaling": 0.1},
            api_client=AsyncMock()
        )
        
        market_data = {
            "market_id": "0x123",
            "current_price": 0.35,  # Significantly below mean
            "price_history": [
                {"price": 0.50, "timestamp": datetime.now() - timedelta(hours=i)}
                for i in range(20)
            ]
        }
        
        signal = await strategy.generate_signal(market_data)
        
        # Should generate buy signal (price below mean)
        assert signal.action == "buy"
        assert signal.confidence > 0.5
        assert "reversion" in signal.reasoning.lower()
        
        # Test opposite case (price above mean)
        market_data["current_price"] = 0.65
        signal_high = await strategy.generate_signal(market_data)
        
        assert signal_high.action == "sell"
    
    @pytest.mark.unit
    async def test_bollinger_bands(self):
        """Test Bollinger Bands implementation."""
        from src.polymarket.strategies import MeanReversionStrategy
        
        strategy = MeanReversionStrategy(config={"bollinger_period": 20}, api_client=AsyncMock())
        
        # Create price series
        prices = [0.50 + 0.02 * np.random.randn() for _ in range(50)]
        
        upper_band, middle_band, lower_band = strategy.calculate_bollinger_bands(prices)
        
        assert upper_band > middle_band > lower_band
        assert abs(middle_band - np.mean(prices[-20:])) < 0.01  # Middle should be close to SMA
        
        # Test signal generation with Bollinger Bands
        current_price = lower_band - 0.01  # Below lower band
        signal_strength = strategy.calculate_bollinger_signal(current_price, upper_band, lower_band)
        
        assert signal_strength > 0  # Buy signal when below lower band


class TestPortfolioOptimization:
    """Test portfolio optimization strategies."""
    
    @pytest.mark.unit
    async def test_kelly_criterion(self):
        """Test Kelly Criterion position sizing."""
        from src.polymarket.strategies import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer(config={"max_kelly_fraction": 0.25})
        
        # Test Kelly calculation
        win_probability = 0.6
        win_amount = 1.0  # 100% gain
        loss_amount = -0.5  # 50% loss
        
        kelly_fraction = optimizer.calculate_kelly_fraction(
            win_probability, win_amount, abs(loss_amount)
        )
        
        expected_kelly = (win_probability * win_amount - (1 - win_probability) * abs(loss_amount)) / win_amount
        assert abs(kelly_fraction - expected_kelly) < 0.001
        
        # Should be capped at maximum
        assert kelly_fraction <= optimizer.config["max_kelly_fraction"]
    
    @pytest.mark.unit
    async def test_portfolio_rebalancing(self):
        """Test portfolio rebalancing logic."""
        from src.polymarket.strategies import PortfolioOptimizer
        from src.polymarket.models import Position
        
        optimizer = PortfolioOptimizer(config={"rebalance_threshold": 0.05})
        
        # Create current portfolio
        current_positions = [
            Position(
                market_id="0x123",
                outcome="Yes",
                size=Decimal("1000"),
                average_price=Decimal("0.60"),
                current_price=Decimal("0.70")
            ),
            Position(
                market_id="0x456",
                outcome="No",
                size=Decimal("500"),
                average_price=Decimal("0.40"),
                current_price=Decimal("0.35")
            )
        ]
        
        # Define target allocation
        target_allocation = {
            "0x123": 0.6,  # 60% of portfolio
            "0x456": 0.4   # 40% of portfolio
        }
        
        rebalance_actions = await optimizer.calculate_rebalancing(
            current_positions, target_allocation
        )
        
        assert len(rebalance_actions) > 0
        assert all("action" in action for action in rebalance_actions)
        assert all("size" in action for action in rebalance_actions)
    
    @pytest.mark.unit
    async def test_risk_parity(self):
        """Test risk parity portfolio construction."""
        from src.polymarket.strategies import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer(config={})
        
        # Mock market data with different volatilities
        markets = [
            {"market_id": "0x123", "volatility": 0.2, "expected_return": 0.08},
            {"market_id": "0x456", "volatility": 0.3, "expected_return": 0.12},
            {"market_id": "0x789", "volatility": 0.15, "expected_return": 0.06}
        ]
        
        risk_parity_weights = optimizer.calculate_risk_parity_weights(markets)
        
        # Weights should sum to 1
        assert abs(sum(risk_parity_weights.values()) - 1.0) < 0.001
        
        # Lower volatility assets should have higher weights
        assert risk_parity_weights["0x789"] > risk_parity_weights["0x456"]
    
    @pytest.mark.unit
    async def test_correlation_analysis(self):
        """Test correlation analysis for portfolio diversification."""
        from src.polymarket.strategies import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer(config={})
        
        # Create correlated price data
        n_periods = 100
        base_returns = [0.01 * np.random.randn() for _ in range(n_periods)]
        
        market_returns = {
            "0x123": base_returns,  # Perfectly correlated
            "0x456": [r + 0.005 * np.random.randn() for r in base_returns],  # Highly correlated
            "0x789": [0.01 * np.random.randn() for _ in range(n_periods)]  # Uncorrelated
        }
        
        correlation_matrix = optimizer.calculate_correlation_matrix(market_returns)
        
        # Check correlation properties
        assert correlation_matrix["0x123"]["0x123"] == 1.0  # Self-correlation
        assert correlation_matrix["0x123"]["0x456"] > 0.7  # High correlation
        assert abs(correlation_matrix["0x123"]["0x789"]) < 0.3  # Low correlation
        
        # Test diversification score
        div_score = optimizer.calculate_diversification_score(correlation_matrix)
        assert 0 <= div_score <= 1


class TestStrategyBacktesting:
    """Test strategy backtesting functionality."""
    
    @pytest.mark.integration
    async def test_backtest_execution(self):
        """Test complete backtest execution."""
        from src.polymarket.strategies import SentimentStrategy
        from src.polymarket.backtesting import Backtester
        
        strategy = SentimentStrategy(config={"sentiment_threshold": 0.6}, api_client=AsyncMock())
        
        # Create historical data
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        historical_data = {
            "markets": [
                {
                    "market_id": "0x123",
                    "price_history": [
                        {"timestamp": start_date + timedelta(days=i), "price": 0.50 + 0.01 * i}
                        for i in range(30)
                    ],
                    "news_history": [
                        {"timestamp": start_date + timedelta(days=i), "sentiment": 0.7}
                        for i in range(30)
                    ]
                }
            ]
        }
        
        backtester = Backtester(strategy=strategy, initial_capital=Decimal("10000"))
        results = await backtester.run_backtest(historical_data, start_date, end_date)
        
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "number_of_trades" in results
        assert "win_rate" in results
        
        # Verify results are reasonable
        assert results["number_of_trades"] >= 0
        assert 0 <= results["win_rate"] <= 1
    
    @pytest.mark.unit
    async def test_performance_metrics(self):
        """Test performance metrics calculation."""
        from src.polymarket.backtesting import PerformanceCalculator
        
        calculator = PerformanceCalculator()
        
        # Create sample returns
        returns = [0.02, -0.01, 0.03, 0.01, -0.02, 0.025, 0.01, -0.015, 0.02, 0.005]
        
        # Test Sharpe ratio
        sharpe = calculator.calculate_sharpe_ratio(returns, risk_free_rate=0.001)
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Positive returns should yield positive Sharpe
        
        # Test maximum drawdown
        cumulative_returns = [1.0]
        for r in returns:
            cumulative_returns.append(cumulative_returns[-1] * (1 + r))
        
        max_dd = calculator.calculate_max_drawdown(cumulative_returns)
        assert 0 <= max_dd <= 1  # Drawdown is a percentage
        
        # Test Sortino ratio
        sortino = calculator.calculate_sortino_ratio(returns, risk_free_rate=0.001)
        assert sortino >= sharpe  # Sortino should be >= Sharpe (only downside risk)
    
    @pytest.mark.unit
    async def test_trade_analysis(self):
        """Test individual trade analysis."""
        from src.polymarket.backtesting import TradeAnalyzer
        
        analyzer = TradeAnalyzer()
        
        # Create sample trades
        trades = [
            {"entry_price": 0.60, "exit_price": 0.65, "size": 100, "side": "buy"},
            {"entry_price": 0.70, "exit_price": 0.68, "size": 50, "side": "buy"},
            {"entry_price": 0.45, "exit_price": 0.50, "size": 200, "side": "buy"},
            {"entry_price": 0.55, "exit_price": 0.52, "size": 75, "side": "buy"}
        ]
        
        analysis = analyzer.analyze_trades(trades)
        
        assert analysis["total_trades"] == 4
        assert analysis["winning_trades"] == 2
        assert analysis["losing_trades"] == 2
        assert analysis["win_rate"] == 0.5
        assert analysis["average_win"] > 0
        assert analysis["average_loss"] < 0
        assert "profit_factor" in analysis
        assert "max_consecutive_wins" in analysis
        assert "max_consecutive_losses" in analysis


class TestStrategyIntegration:
    """Test strategy integration with API and real-time data."""
    
    @pytest.mark.integration
    async def test_multi_strategy_coordination(self):
        """Test coordination between multiple strategies."""
        from src.polymarket.strategies import (
            SentimentStrategy, ArbitrageStrategy, StrategyCoordinator
        )
        
        api_client = AsyncMock()
        
        # Create multiple strategies
        sentiment_strategy = SentimentStrategy(config={"weight": 0.6}, api_client=api_client)
        arbitrage_strategy = ArbitrageStrategy(config={"weight": 0.4}, api_client=api_client)
        
        # Create coordinator
        coordinator = StrategyCoordinator(
            strategies=[sentiment_strategy, arbitrage_strategy],
            config={"consensus_threshold": 0.7}
        )
        
        # Mock signals from different strategies
        sentiment_signal = Mock(action="buy", confidence=0.8, size=100)
        arbitrage_signal = Mock(action="buy", confidence=0.6, size=50)
        
        # Test signal aggregation
        combined_signal = await coordinator.aggregate_signals([sentiment_signal, arbitrage_signal])
        
        assert combined_signal.action == "buy"
        assert combined_signal.confidence > 0.6
        assert combined_signal.size > 0
    
    @pytest.mark.integration
    async def test_real_time_strategy_updates(self):
        """Test strategy updates with real-time market data."""
        from src.polymarket.strategies import MomentumStrategy
        
        api_client = AsyncMock()
        strategy = MomentumStrategy(config={}, api_client=api_client)
        
        # Mock WebSocket updates
        market_updates = [
            {"type": "price_update", "market_id": "0x123", "price": 0.65, "timestamp": datetime.now()},
            {"type": "volume_update", "market_id": "0x123", "volume": 1000, "timestamp": datetime.now()},
            {"type": "trade_update", "market_id": "0x123", "size": 100, "price": 0.66, "timestamp": datetime.now()}
        ]
        
        # Process updates
        for update in market_updates:
            await strategy.process_market_update(update)
        
        # Strategy should have updated internal state
        assert len(strategy.market_data.get("0x123", {}).get("price_history", [])) > 0
        
        # Should generate signals if conditions are met
        current_signals = await strategy.get_current_signals()
        assert isinstance(current_signals, list)
    
    @pytest.mark.slow
    async def test_strategy_performance_monitoring(self):
        """Test continuous performance monitoring."""
        from src.polymarket.strategies import BaseStrategy
        from src.polymarket.monitoring import PerformanceMonitor
        
        strategy = BaseStrategy(config={}, api_client=AsyncMock())
        monitor = PerformanceMonitor(strategy=strategy)
        
        # Simulate strategy running for some time
        await strategy.start()
        
        # Mock some performance data
        mock_trades = [
            {"pnl": 5.0, "timestamp": datetime.now()},
            {"pnl": -2.0, "timestamp": datetime.now()},
            {"pnl": 3.0, "timestamp": datetime.now()}
        ]
        
        for trade in mock_trades:
            monitor.record_trade(trade)
        
        # Check performance metrics
        metrics = monitor.get_current_metrics()
        
        assert "total_pnl" in metrics
        assert "trade_count" in metrics
        assert "win_rate" in metrics
        assert "current_drawdown" in metrics
        
        # Test alert generation
        alerts = monitor.check_alerts()
        assert isinstance(alerts, list)
        
        await strategy.stop()