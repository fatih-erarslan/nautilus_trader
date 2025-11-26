"""
End-to-End Trading Workflow Integration Tests

This module contains comprehensive end-to-end tests for complete trading workflows,
ensuring all components work together correctly from signal generation to execution.
"""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
import pytest
import logging

from ..unit.brokers.mock_framework import MockManager, AlpacaMock
from ..unit.fixtures.broker_responses import BrokerResponseFixtures


class TestE2EWorkflows:
    """End-to-end trading workflow tests"""
    
    @pytest.fixture(scope="function")
    def mock_manager(self):
        """Setup mock manager for tests"""
        manager = MockManager()
        manager.setup_broker_mocks(["alpaca", "ibkr"])
        manager.setup_news_mock()
        yield manager
        manager.reset_all_mocks()
    
    @pytest.fixture(scope="function")
    def sample_portfolio(self):
        """Sample portfolio for testing"""
        return {
            "cash": Decimal("100000.00"),
            "positions": {
                "AAPL": {"qty": 100, "avg_price": Decimal("150.00")},
                "GOOGL": {"qty": 50, "avg_price": Decimal("2500.00")},
            },
            "buying_power": Decimal("200000.00")
        }
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_trading_cycle(self, mock_manager: MockManager):
        """Test complete order lifecycle from signal to execution"""
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        news_mock = mock_manager.get_news_mock()
        
        # Step 1: Setup market environment
        news_mock.add_article(BrokerResponseFixtures.NEWS_ARTICLE_FIXTURE)
        
        # Step 2: Authenticate with broker
        account_info = alpaca_mock.get_account()
        assert account_info["status"] == "ACTIVE"
        assert float(account_info["cash"]) > 0
        
        # Step 3: Get market data and analyze sentiment
        articles = news_mock.get_articles("AAPL", limit=5)
        assert len(articles) > 0
        
        # Calculate aggregated sentiment
        total_sentiment = sum(article["sentiment"]["polarity"] for article in articles)
        avg_sentiment = total_sentiment / len(articles)
        
        # Step 4: Generate trading signal based on sentiment
        signal = self._generate_trading_signal(avg_sentiment, account_info)
        assert signal["action"] in ["buy", "sell", "hold"]
        
        if signal["action"] != "hold":
            # Step 5: Submit order
            order = alpaca_mock.submit_order(
                symbol=signal["symbol"],
                qty=signal["quantity"],
                side=signal["action"],
                order_type="market"
            )
            
            assert order["status"] in ["new", "accepted"]
            assert order["symbol"] == signal["symbol"]
            assert int(order["qty"]) == signal["quantity"]
            
            # Step 6: Monitor execution
            order_id = order["id"]
            
            # Simulate order progression
            await self._simulate_order_execution(alpaca_mock, order_id, signal)
            
            # Step 7: Verify final order state
            final_order = alpaca_mock.orders[order_id]
            assert final_order["status"] == "filled"
            assert float(final_order["filled_avg_price"]) > 0
            
            # Step 8: Update portfolio and verify positions
            positions = alpaca_mock.get_positions()
            updated_account = alpaca_mock.get_account()
            
            # Verify position was created/updated
            symbol_position = next(
                (pos for pos in positions if pos["symbol"] == signal["symbol"]), 
                None
            )
            if signal["action"] == "buy":
                assert symbol_position is not None
                assert int(symbol_position["qty"]) > 0
            
            # Step 9: Generate execution report
            execution_report = self._generate_execution_report(
                signal, final_order, updated_account
            )
            
            assert execution_report["success"] is True
            assert execution_report["execution_time"] < 1.0  # Less than 1 second
            assert execution_report["slippage"] >= 0
        
        # Verify all API calls were logged
        assert alpaca_mock.get_call_count() >= 3  # get_account, submit_order, get_positions
    
    @pytest.mark.integration
    def test_multi_asset_portfolio_rebalancing(self, mock_manager: MockManager):
        """Test portfolio rebalancing workflow"""
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        
        # Setup initial positions
        initial_positions = {
            "AAPL": {"current_weight": 0.40, "target_weight": 0.30},
            "GOOGL": {"current_weight": 0.35, "target_weight": 0.35},
            "MSFT": {"current_weight": 0.15, "target_weight": 0.25},
            "AMZN": {"current_weight": 0.10, "target_weight": 0.10}
        }
        
        portfolio_value = Decimal("500000.00")
        
        # Calculate rebalancing orders
        rebalancing_orders = self._calculate_rebalancing_orders(
            initial_positions, portfolio_value
        )
        
        # Execute rebalancing orders
        executed_orders = []
        for order_spec in rebalancing_orders:
            if order_spec["action"] != "hold":
                order = alpaca_mock.submit_order(
                    symbol=order_spec["symbol"],
                    qty=abs(order_spec["quantity"]),
                    side=order_spec["action"],
                    order_type="market"
                )
                executed_orders.append(order)
                
                # Simulate fill
                alpaca_mock.simulate_order_fill(
                    order["id"], 
                    order_spec["estimated_price"],
                    abs(order_spec["quantity"])
                )
        
        # Verify rebalancing was successful
        assert len(executed_orders) > 0
        
        # Check that orders were executed for symbols that needed rebalancing
        symbols_traded = {order["symbol"] for order in executed_orders}
        expected_symbols = {
            symbol for symbol, data in initial_positions.items()
            if abs(data["current_weight"] - data["target_weight"]) > 0.05
        }
        
        assert symbols_traded.intersection(expected_symbols)
    
    @pytest.mark.integration
    @pytest.mark.timeout(30)
    def test_stop_loss_trigger_workflow(self, mock_manager: MockManager):
        """Test stop loss order execution workflow"""
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        
        # Setup initial position
        symbol = "AAPL"
        entry_price = 150.00
        stop_loss_price = 140.00  # 10 below entry
        position_qty = 100
        
        # Simulate existing position
        alpaca_mock.positions[symbol] = {
            "symbol": symbol,
            "qty": str(position_qty),
            "avg_entry_price": str(entry_price),
            "side": "long",
            "market_value": str(position_qty * entry_price),
            "unrealized_pl": "0.00",
            "current_price": str(entry_price)
        }
        
        # Submit stop loss order
        stop_order = alpaca_mock.submit_order(
            symbol=symbol,
            qty=position_qty,
            side="sell",
            order_type="stop",
            stop_price=stop_loss_price
        )
        
        assert stop_order["type"] == "stop"
        assert float(stop_order["stop_price"]) == stop_loss_price
        
        # Simulate price decline triggering stop loss
        trigger_price = 139.50  # Below stop loss
        
        # Update position with new price
        alpaca_mock.positions[symbol]["current_price"] = str(trigger_price)
        alpaca_mock.positions[symbol]["unrealized_pl"] = str(
            position_qty * (trigger_price - entry_price)
        )
        
        # Simulate stop loss trigger and execution
        alpaca_mock.simulate_order_fill(
            stop_order["id"], 
            trigger_price, 
            position_qty
        )
        
        # Verify stop loss execution
        filled_order = alpaca_mock.orders[stop_order["id"]]
        assert filled_order["status"] == "filled"
        assert float(filled_order["filled_avg_price"]) <= stop_loss_price
        
        # Verify position was closed
        updated_position = alpaca_mock.positions[symbol]
        assert int(updated_position["qty"]) == 0
        
        # Calculate realized P&L
        realized_pnl = position_qty * (float(filled_order["filled_avg_price"]) - entry_price)
        assert realized_pnl < 0  # Should be a loss
        assert abs(realized_pnl) <= position_qty * (entry_price - stop_loss_price) * 1.1  # Allow for slippage
    
    @pytest.mark.integration
    def test_news_driven_trading_workflow(self, mock_manager: MockManager):
        """Test complete news-driven trading workflow"""
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        news_mock = mock_manager.get_news_mock()
        
        # Setup multiple news articles with different sentiment
        news_articles = [
            {**BrokerResponseFixtures.NEWS_ARTICLE_FIXTURE, "id": "pos_1"},
            {**BrokerResponseFixtures.NEWS_ARTICLE_FIXTURE, "id": "pos_2"},
            {**BrokerResponseFixtures.NEGATIVE_NEWS_FIXTURE, "id": "neg_1"},
            {**BrokerResponseFixtures.NEUTRAL_NEWS_FIXTURE, "id": "neu_1"}
        ]
        
        for article in news_articles:
            news_mock.add_article(article)
        
        # Analyze news sentiment over time
        symbol = "AAPL"
        sentiment_timeline = self._analyze_sentiment_timeline(news_mock, symbol, hours=24)
        
        # Generate trading signals based on sentiment analysis
        trading_signals = self._generate_news_based_signals(sentiment_timeline)
        
        executed_trades = []
        for signal in trading_signals:
            if signal["confidence"] > 0.7 and signal["action"] != "hold":
                order = alpaca_mock.submit_order(
                    symbol=signal["symbol"],
                    qty=signal["quantity"],
                    side=signal["action"],
                    order_type="market"
                )
                
                # Simulate execution
                alpaca_mock.simulate_order_fill(
                    order["id"],
                    signal["expected_price"],
                    signal["quantity"]
                )
                
                executed_trades.append({
                    "order": order,
                    "signal": signal,
                    "sentiment_score": signal["sentiment_score"]
                })
        
        # Verify trading decisions align with sentiment
        for trade in executed_trades:
            sentiment = trade["sentiment_score"]
            action = trade["signal"]["action"]
            
            if sentiment > 0.5:
                assert action == "buy", "Positive sentiment should trigger buy"
            elif sentiment < -0.5:
                assert action == "sell", "Negative sentiment should trigger sell"
        
        # Verify news analysis was comprehensive
        assert news_mock.get_call_count() >= len(news_articles)
    
    @pytest.mark.integration
    def test_multi_timeframe_analysis_workflow(self, mock_manager: MockManager):
        """Test trading workflow with multiple timeframe analysis"""
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        news_mock = mock_manager.get_news_mock()
        
        symbol = "AAPL"
        
        # Setup news data for different timeframes
        timeframes = ["1h", "4h", "1d", "1w"]
        analysis_results = {}
        
        for timeframe in timeframes:
            # Simulate different sentiment patterns for each timeframe
            sentiment_data = self._generate_timeframe_sentiment(symbol, timeframe)
            analysis_results[timeframe] = sentiment_data
        
        # Aggregate multi-timeframe analysis
        combined_signal = self._combine_timeframe_signals(analysis_results)
        
        # Execute trade based on combined signal
        if combined_signal["action"] != "hold" and combined_signal["strength"] > 0.6:
            order = alpaca_mock.submit_order(
                symbol=combined_signal["symbol"],
                qty=combined_signal["quantity"],
                side=combined_signal["action"],
                order_type="limit",
                limit_price=combined_signal["target_price"]
            )
            
            # Simulate partial fill scenario
            partial_qty = combined_signal["quantity"] // 2
            alpaca_mock.simulate_order_fill(
                order["id"],
                combined_signal["target_price"],
                partial_qty
            )
            
            # Verify partial fill handling
            partial_order = alpaca_mock.orders[order["id"]]
            assert partial_order["status"] == "partially_filled"
            assert int(partial_order["filled_qty"]) == partial_qty
            
            # Complete the fill
            remaining_qty = combined_signal["quantity"] - partial_qty
            alpaca_mock.simulate_order_fill(
                order["id"],
                combined_signal["target_price"],
                remaining_qty
            )
            
            # Verify complete fill
            final_order = alpaca_mock.orders[order["id"]]
            assert final_order["status"] == "filled"
            assert int(final_order["filled_qty"]) == combined_signal["quantity"]
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_high_frequency_trading_simulation(self, mock_manager: MockManager):
        """Test high-frequency trading scenario"""
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        
        # Enable low latency mode
        alpaca_mock.set_latency(1)  # 1ms latency
        
        symbol = "SPY"
        num_trades = 100
        start_time = time.time()
        
        executed_orders = []
        
        for i in range(num_trades):
            # Alternate between buy and sell
            side = "buy" if i % 2 == 0 else "sell"
            qty = 100
            
            order = alpaca_mock.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type="market"
            )
            
            # Immediate fill simulation for HFT
            fill_price = 400.0 + (i * 0.01)  # Simulate small price movements
            alpaca_mock.simulate_order_fill(order["id"], fill_price, qty)
            
            executed_orders.append(order)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify HFT performance criteria
        assert len(executed_orders) == num_trades
        assert total_time < 5.0  # All trades completed in under 5 seconds
        
        avg_time_per_trade = total_time / num_trades
        assert avg_time_per_trade < 0.05  # Less than 50ms per trade
        
        # Verify all orders were filled
        for order in executed_orders:
            final_order = alpaca_mock.orders[order["id"]]
            assert final_order["status"] == "filled"
    
    @pytest.mark.integration
    def test_error_recovery_workflow(self, mock_manager: MockManager):
        """Test error recovery and resilience"""
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        
        # Test connection error recovery
        alpaca_mock.enable_error_mode("connection_error")
        
        with pytest.raises(ConnectionError):
            alpaca_mock.get_account()
        
        # Simulate connection recovery
        alpaca_mock.disable_error_mode()
        account = alpaca_mock.get_account()
        assert account["status"] == "ACTIVE"
        
        # Test order submission with rate limiting
        alpaca_mock.enable_rate_limiting(max_calls=5, window=60)
        
        orders_submitted = 0
        for i in range(10):
            try:
                order = alpaca_mock.submit_order(
                    symbol="AAPL",
                    qty=100,
                    side="buy",
                    order_type="market"
                )
                orders_submitted += 1
            except Exception as e:
                if "rate limit" in str(e).lower():
                    break
        
        # Should hit rate limit before submitting all orders
        assert orders_submitted <= 5
        
        # Test graceful degradation
        alpaca_mock.enable_error_mode("random")
        
        successful_calls = 0
        total_calls = 20
        
        for i in range(total_calls):
            try:
                alpaca_mock.get_account()
                successful_calls += 1
            except Exception:
                pass  # Expected random failures
        
        # Should have some successful calls despite random errors
        assert successful_calls > 0
        assert successful_calls < total_calls  # But not all should succeed
    
    # Helper methods
    
    def _generate_trading_signal(self, sentiment: float, account_info: Dict) -> Dict[str, Any]:
        """Generate trading signal based on sentiment and account info"""
        cash_available = float(account_info["cash"])
        
        if sentiment > 0.6 and cash_available > 10000:
            return {
                "action": "buy",
                "symbol": "AAPL",
                "quantity": 100,
                "confidence": sentiment,
                "expected_price": 150.25
            }
        elif sentiment < -0.6:
            return {
                "action": "sell",
                "symbol": "AAPL",
                "quantity": 50,
                "confidence": abs(sentiment),
                "expected_price": 150.25
            }
        else:
            return {
                "action": "hold",
                "symbol": "AAPL",
                "quantity": 0,
                "confidence": 0.5,
                "expected_price": 150.25
            }
    
    async def _simulate_order_execution(self, broker_mock: AlpacaMock, order_id: str, signal: Dict) -> None:
        """Simulate realistic order execution timing"""
        # Wait for order processing (simulate market conditions)
        await asyncio.sleep(0.1)
        
        # Simulate order fill
        broker_mock.simulate_order_fill(
            order_id,
            signal["expected_price"],
            signal["quantity"]
        )
    
    def _generate_execution_report(self, signal: Dict, order: Dict, account: Dict) -> Dict[str, Any]:
        """Generate execution report"""
        expected_price = signal["expected_price"]
        actual_price = float(order["filled_avg_price"])
        
        return {
            "success": True,
            "signal": signal,
            "order_id": order["id"],
            "execution_time": 0.5,  # Simulated
            "slippage": abs(actual_price - expected_price),
            "account_impact": {
                "cash_change": -float(order["qty"]) * actual_price,
                "position_change": int(order["qty"])
            }
        }
    
    def _calculate_rebalancing_orders(self, positions: Dict, portfolio_value: Decimal) -> List[Dict]:
        """Calculate orders needed for portfolio rebalancing"""
        orders = []
        
        for symbol, weights in positions.items():
            current_value = portfolio_value * Decimal(str(weights["current_weight"]))
            target_value = portfolio_value * Decimal(str(weights["target_weight"]))
            difference = target_value - current_value
            
            if abs(difference) > 1000:  # Only rebalance if difference > $1000
                estimated_price = 150.0  # Simplified for testing
                quantity = int(abs(difference) / Decimal(str(estimated_price)))
                
                orders.append({
                    "symbol": symbol,
                    "action": "buy" if difference > 0 else "sell",
                    "quantity": quantity,
                    "estimated_price": estimated_price
                })
            else:
                orders.append({
                    "symbol": symbol,
                    "action": "hold",
                    "quantity": 0,
                    "estimated_price": 150.0
                })
        
        return orders
    
    def _analyze_sentiment_timeline(self, news_mock, symbol: str, hours: int) -> List[Dict]:
        """Analyze sentiment over time"""
        timeline = []
        
        for hour in range(hours):
            articles = news_mock.get_articles(symbol, limit=5)
            if articles:
                avg_sentiment = sum(a["sentiment"]["polarity"] for a in articles) / len(articles)
                timeline.append({
                    "hour": hour,
                    "sentiment": avg_sentiment,
                    "article_count": len(articles),
                    "confidence": sum(a["sentiment"]["confidence"] for a in articles) / len(articles)
                })
        
        return timeline
    
    def _generate_news_based_signals(self, sentiment_timeline: List[Dict]) -> List[Dict]:
        """Generate trading signals based on sentiment timeline"""
        signals = []
        
        for i, data in enumerate(sentiment_timeline):
            sentiment = data["sentiment"]
            confidence = data["confidence"]
            
            if sentiment > 0.5 and confidence > 0.8:
                signals.append({
                    "symbol": "AAPL",
                    "action": "buy",
                    "quantity": 100,
                    "confidence": confidence,
                    "sentiment_score": sentiment,
                    "expected_price": 150.0 + sentiment * 5  # Price based on sentiment
                })
            elif sentiment < -0.5 and confidence > 0.8:
                signals.append({
                    "symbol": "AAPL",
                    "action": "sell",
                    "quantity": 50,
                    "confidence": confidence,
                    "sentiment_score": sentiment,
                    "expected_price": 150.0 + sentiment * 5
                })
        
        return signals
    
    def _generate_timeframe_sentiment(self, symbol: str, timeframe: str) -> Dict:
        """Generate sentiment data for specific timeframe"""
        # Simulate different sentiment patterns by timeframe
        timeframe_multipliers = {"1h": 0.2, "4h": 0.5, "1d": 0.8, "1w": 1.0}
        base_sentiment = 0.6
        
        return {
            "timeframe": timeframe,
            "sentiment": base_sentiment * timeframe_multipliers.get(timeframe, 1.0),
            "confidence": 0.8,
            "strength": timeframe_multipliers.get(timeframe, 1.0)
        }
    
    def _combine_timeframe_signals(self, analysis_results: Dict) -> Dict:
        """Combine signals from multiple timeframes"""
        # Weight longer timeframes more heavily
        weights = {"1h": 0.1, "4h": 0.2, "1d": 0.3, "1w": 0.4}
        
        weighted_sentiment = sum(
            analysis_results[tf]["sentiment"] * weights[tf] 
            for tf in weights.keys()
        )
        
        combined_strength = sum(
            analysis_results[tf]["strength"] * weights[tf] 
            for tf in weights.keys()
        )
        
        action = "buy" if weighted_sentiment > 0.3 else "sell" if weighted_sentiment < -0.3 else "hold"
        
        return {
            "symbol": "AAPL",
            "action": action,
            "quantity": 100,
            "strength": combined_strength,
            "target_price": 150.0 + weighted_sentiment * 10,
            "sentiment": weighted_sentiment
        }