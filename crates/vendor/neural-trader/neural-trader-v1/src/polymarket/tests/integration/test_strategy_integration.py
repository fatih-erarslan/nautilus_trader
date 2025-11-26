"""
End-to-End Strategy Integration Tests

This module tests complete strategy execution flows from signal generation
through order placement and position management, including multi-strategy
ensemble coordination and risk management.
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from polymarket.strategies import (
    TradingStrategy, MomentumStrategy, MeanReversionStrategy,
    ArbitrageStrategy, MarketMakerStrategy, EnsembleStrategy,
    SentimentCorrelationStrategy
)
from polymarket.strategies.base import (
    TradingSignal, SignalStrength, SignalDirection,
    StrategyConfig, StrategyMetrics, StrategyError
)
from polymarket.models import Market, MarketStatus, Order, OrderSide, Position
from polymarket.api import PolymarketClient
from polymarket.utils.monitoring import PerformanceTracker, StrategyMonitor


class TestStrategyIntegration:
    """End-to-end strategy execution tests."""

    @pytest.fixture
    async def api_client(self):
        """Create mock API client with realistic responses."""
        client = AsyncMock(spec=PolymarketClient)
        
        # Configure realistic market data
        client.get_markets.return_value = self._generate_test_markets(20)
        client.get_orderbook.return_value = self._generate_orderbook()
        client.get_positions.return_value = []
        client.place_order.side_effect = self._mock_place_order
        
        return client

    @pytest.fixture
    def strategy_config(self):
        """Standard strategy configuration."""
        return StrategyConfig(
            max_position_size=Decimal("1000.0"),
            min_confidence=0.7,
            min_signal_strength=SignalStrength.MODERATE,
            max_markets_monitored=10,
            stop_loss_percentage=0.05,
            take_profit_percentage=0.15,
            max_drawdown=0.10
        )

    @pytest.fixture
    async def momentum_strategy(self, api_client, strategy_config):
        """Create momentum strategy instance."""
        return MomentumStrategy(api_client, strategy_config)

    @pytest.fixture
    async def ensemble_strategy(self, api_client, strategy_config):
        """Create ensemble strategy with multiple sub-strategies."""
        strategies = [
            MomentumStrategy(api_client, strategy_config),
            MeanReversionStrategy(api_client, strategy_config),
            ArbitrageStrategy(api_client, strategy_config)
        ]
        
        weights = {
            "momentum": 0.4,
            "mean_reversion": 0.3,
            "arbitrage": 0.3
        }
        
        return EnsembleStrategy(
            client=api_client,
            strategies=strategies,
            weights=weights,
            config=strategy_config
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_trading_cycle(self, momentum_strategy, api_client):
        """Test complete trading cycle: scan → analyze → signal → execute."""
        # Start strategy
        await momentum_strategy.start()
        
        # Run market scan
        markets = await momentum_strategy.scan_markets()
        assert len(markets) > 0
        
        # Analyze top market
        market = markets[0]
        signal = await momentum_strategy.analyze_market(market)
        
        assert isinstance(signal, TradingSignal)
        assert signal.market_id == market.id
        assert signal.confidence >= 0.7
        
        # Execute trade if signal is strong
        if signal.strength >= SignalStrength.MODERATE:
            order = await momentum_strategy.execute_signal(signal)
            assert isinstance(order, Order)
            assert order.market_id == market.id
            
            # Verify order placement
            api_client.place_order.assert_called_once()
            
        # Stop strategy
        await momentum_strategy.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_strategy_coordination(self, ensemble_strategy):
        """Test ensemble strategy coordination and signal aggregation."""
        await ensemble_strategy.start()
        
        # Get signals from all strategies
        markets = await ensemble_strategy.scan_markets()
        market = markets[0] if markets else None
        
        if market:
            # Get ensemble signal
            ensemble_signal = await ensemble_strategy.analyze_market(market)
            
            # Get individual signals
            individual_signals = await ensemble_strategy.get_individual_signals(market)
            
            # Verify signal aggregation
            assert len(individual_signals) == 3
            assert ensemble_signal.confidence == ensemble_strategy._calculate_weighted_confidence(
                individual_signals
            )
            
            # Test conflict resolution
            if any(s.direction != ensemble_signal.direction for s in individual_signals.values()):
                # Ensemble should handle conflicting signals gracefully
                assert ensemble_signal.strength <= SignalStrength.MODERATE
        
        await ensemble_strategy.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_risk_management_integration(self, momentum_strategy, api_client):
        """Test integrated risk management across strategy execution."""
        # Configure positions to test risk limits
        test_positions = [
            Position(
                market_id="market_1",
                outcome="Yes",
                size=Decimal("500.0"),
                average_price=Decimal("0.60"),
                current_price=Decimal("0.55"),  # Losing position
                pnl=Decimal("-25.0"),
                pnl_percentage=-0.083
            ),
            Position(
                market_id="market_2",
                outcome="No",
                size=Decimal("300.0"),
                average_price=Decimal("0.40"),
                current_price=Decimal("0.45"),  # Losing position
                pnl=Decimal("-15.0"),
                pnl_percentage=-0.125
            )
        ]
        api_client.get_positions.return_value = test_positions
        
        await momentum_strategy.start()
        
        # Check position limits
        can_trade = await momentum_strategy.check_risk_limits()
        assert not can_trade  # Should block due to drawdown
        
        # Test stop loss execution
        await momentum_strategy.check_stop_losses()
        
        # Verify stop loss orders placed
        assert api_client.place_order.call_count >= 1
        stop_loss_call = api_client.place_order.call_args_list[0]
        assert stop_loss_call[1]["side"] == OrderSide.SELL
        
        await momentum_strategy.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_performance_tracking_integration(self, momentum_strategy):
        """Test performance tracking throughout strategy execution."""
        tracker = PerformanceTracker()
        momentum_strategy.set_performance_tracker(tracker)
        
        await momentum_strategy.start()
        
        # Execute some trades
        for i in range(5):
            markets = await momentum_strategy.scan_markets()
            if markets:
                signal = await momentum_strategy.analyze_market(markets[0])
                if signal.strength >= SignalStrength.MODERATE:
                    await momentum_strategy.execute_signal(signal)
        
        # Get performance metrics
        metrics = tracker.get_metrics()
        
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert metrics["total_trades"] >= 0
        
        await momentum_strategy.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_market_maker_integration(self, api_client, strategy_config):
        """Test market maker strategy with order book management."""
        strategy = MarketMakerStrategy(api_client, strategy_config)
        
        # Configure market maker parameters
        strategy.set_parameters({
            "spread_percentage": 0.02,
            "order_levels": 3,
            "level_spacing": 0.01,
            "rebalance_threshold": 0.05
        })
        
        await strategy.start()
        
        # Test quote placement
        markets = await strategy.scan_markets()
        liquid_market = next((m for m in markets if m.volume_24h > 10000), None)
        
        if liquid_market:
            # Place quotes
            orders = await strategy.place_quotes(liquid_market)
            
            # Verify symmetric quotes
            buy_orders = [o for o in orders if o.side == OrderSide.BUY]
            sell_orders = [o for o in orders if o.side == OrderSide.SELL]
            
            assert len(buy_orders) == strategy.parameters["order_levels"]
            assert len(sell_orders) == strategy.parameters["order_levels"]
            
            # Test quote updates on market movement
            await asyncio.sleep(1)
            await strategy.update_quotes(liquid_market)
            
            # Verify old orders cancelled and new ones placed
            assert api_client.cancel_order.called
        
        await strategy.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sentiment_strategy_with_news(self, api_client, strategy_config):
        """Test sentiment correlation strategy with news integration."""
        strategy = SentimentCorrelationStrategy(api_client, strategy_config)
        
        # Mock news sentiment data
        with patch.object(strategy, 'get_news_sentiment') as mock_sentiment:
            mock_sentiment.return_value = {
                "overall_sentiment": 0.8,
                "confidence": 0.9,
                "article_count": 25,
                "trending_topics": ["election", "polls", "debate"],
                "sentiment_momentum": 0.15
            }
            
            await strategy.start()
            
            # Find correlated markets
            markets = await strategy.scan_markets()
            political_markets = [m for m in markets if "election" in m.question.lower()]
            
            if political_markets:
                market = political_markets[0]
                signal = await strategy.analyze_market(market)
                
                # Verify sentiment influence
                assert signal.metadata["sentiment_score"] == 0.8
                assert signal.confidence >= 0.7  # High confidence due to strong sentiment
                
                # Test sentiment-based position sizing
                position_size = strategy.calculate_position_size(signal)
                assert position_size > strategy.config.max_position_size * Decimal("0.5")
            
            await strategy.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_arbitrage_detection_and_execution(self, api_client, strategy_config):
        """Test arbitrage strategy detecting and exploiting price differences."""
        strategy = ArbitrageStrategy(api_client, strategy_config)
        
        # Create arbitrage opportunity
        correlated_markets = [
            Market(
                id="market_1",
                question="Will BTC reach $100k in 2024?",
                outcomes=["Yes", "No"],
                outcome_prices=[Decimal("0.60"), Decimal("0.40")]
            ),
            Market(
                id="market_2",
                question="Will Bitcoin hit $100,000 by end of 2024?",
                outcomes=["Yes", "No"],
                outcome_prices=[Decimal("0.65"), Decimal("0.35")]  # 5% arbitrage
            )
        ]
        
        api_client.get_markets.return_value = correlated_markets
        
        await strategy.start()
        
        # Detect arbitrage
        opportunities = await strategy.find_arbitrage_opportunities()
        assert len(opportunities) > 0
        
        opportunity = opportunities[0]
        assert opportunity["profit_percentage"] >= 0.03  # After fees
        
        # Execute arbitrage
        orders = await strategy.execute_arbitrage(opportunity)
        assert len(orders) == 2  # Buy low, sell high
        assert orders[0].side != orders[1].side
        
        await strategy.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_strategy_failure_recovery(self, momentum_strategy, api_client):
        """Test strategy recovery from failures and errors."""
        # Simulate API failures
        api_client.get_markets.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            self._generate_test_markets(10)  # Success on third try
        ]
        
        await momentum_strategy.start()
        
        # Strategy should retry and recover
        markets = await momentum_strategy.scan_markets()
        assert len(markets) > 0
        assert api_client.get_markets.call_count == 3
        
        # Test order failure handling
        api_client.place_order.side_effect = Exception("Insufficient balance")
        
        signal = TradingSignal(
            market_id="test_market",
            direction=SignalDirection.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.9
        )
        
        # Should handle error gracefully
        order = await momentum_strategy.execute_signal(signal)
        assert order is None
        assert momentum_strategy.get_error_count() > 0
        
        await momentum_strategy.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_continuous_monitoring_integration(self, ensemble_strategy):
        """Test continuous market monitoring and signal generation."""
        monitor = StrategyMonitor(ensemble_strategy)
        await monitor.start()
        
        # Run for multiple cycles
        signals_generated = []
        orders_placed = []
        
        async def collect_signals():
            for _ in range(5):
                await asyncio.sleep(1)
                signals = monitor.get_recent_signals()
                signals_generated.extend(signals)
                
                orders = monitor.get_recent_orders()
                orders_placed.extend(orders)
        
        await collect_signals()
        
        # Verify continuous operation
        assert len(signals_generated) > 0
        assert monitor.get_uptime() > 0
        assert monitor.get_markets_monitored() > 0
        
        # Check performance over time
        performance = monitor.get_performance_summary()
        assert "signals_per_hour" in performance
        assert "average_confidence" in performance
        assert "execution_success_rate" in performance
        
        await monitor.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.benchmark
    async def test_strategy_performance_benchmark(self, momentum_strategy, benchmark):
        """Benchmark strategy analysis performance."""
        markets = self._generate_test_markets(50)
        
        async def analyze_all_markets():
            signals = []
            for market in markets:
                signal = await momentum_strategy.analyze_market(market)
                signals.append(signal)
            return signals
        
        # Benchmark analysis speed
        signals = await benchmark.pedantic(
            analyze_all_markets,
            rounds=5,
            iterations=2,
            warmup_rounds=1
        )
        
        assert len(signals) == 50
        # Should analyze 50 markets in under 2 seconds
        assert benchmark.stats["mean"] < 2.0
        
        # Calculate signals per second
        signals_per_second = 50 / benchmark.stats["mean"]
        assert signals_per_second > 20  # At least 20 markets/second

    def _generate_test_markets(self, count: int) -> List[Market]:
        """Generate realistic test market data."""
        markets = []
        categories = ["Crypto", "Politics", "Sports", "Entertainment", "Economics"]
        
        for i in range(count):
            yes_price = Decimal(str(np.random.uniform(0.1, 0.9)))
            no_price = Decimal("1.0") - yes_price
            
            market = Market(
                id=f"market_{i}",
                question=f"Test market {i}?",
                slug=f"test-market-{i}",
                category=np.random.choice(categories),
                outcomes=["Yes", "No"],
                outcome_prices=[yes_price, no_price],
                volume_24h=Decimal(str(np.random.uniform(100, 100000))),
                liquidity=Decimal(str(np.random.uniform(1000, 50000))),
                status=MarketStatus.ACTIVE,
                created_at=datetime.now() - timedelta(days=np.random.randint(1, 30)),
                close_time=datetime.now() + timedelta(days=np.random.randint(1, 90))
            )
            markets.append(market)
        
        return markets

    def _generate_orderbook(self) -> Dict[str, Any]:
        """Generate realistic orderbook data."""
        bids = []
        asks = []
        
        mid_price = 0.5
        
        # Generate bids
        for i in range(10):
            price = mid_price - (i + 1) * 0.01
            size = np.random.uniform(10, 1000)
            bids.append({
                "price": str(price),
                "size": str(size)
            })
        
        # Generate asks
        for i in range(10):
            price = mid_price + (i + 1) * 0.01
            size = np.random.uniform(10, 1000)
            asks.append({
                "price": str(price),
                "size": str(size)
            })
        
        return {
            "market_id": "test_market",
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.now().isoformat()
        }

    async def _mock_place_order(self, **kwargs) -> Order:
        """Mock order placement with realistic response."""
        return Order(
            id=f"order_{np.random.randint(1000, 9999)}",
            market_id=kwargs["market_id"],
            outcome=kwargs["outcome"],
            side=kwargs["side"],
            order_type=kwargs.get("order_type", OrderType.LIMIT),
            size=kwargs["size"],
            price=kwargs.get("price", Decimal("0.5")),
            status=OrderStatus.OPEN,
            created_at=datetime.now()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])