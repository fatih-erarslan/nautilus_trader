"""
Mock factories for external API services.

This module provides factory classes for creating consistent mock objects
for external services used in the AI News Trading Platform.
"""

import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock

from faker import Faker

fake = Faker()


class MockCryptoAPIFactory:
    """Factory for creating mock cryptocurrency API responses."""
    
    @staticmethod
    def create_mock(async_mode: bool = True) -> Mock:
        """Create a mock crypto API client."""
        MockClass = AsyncMock if async_mode else Mock
        mock = MockClass()
        
        # Configure common methods
        mock.get_ticker = MockClass(side_effect=MockCryptoAPIFactory.get_ticker_response)
        mock.get_orderbook = MockClass(side_effect=MockCryptoAPIFactory.get_orderbook_response)
        mock.get_candles = MockClass(side_effect=MockCryptoAPIFactory.get_candles_response)
        mock.place_order = MockClass(side_effect=MockCryptoAPIFactory.place_order_response)
        mock.get_balance = MockClass(side_effect=MockCryptoAPIFactory.get_balance_response)
        mock.get_order_status = MockClass(side_effect=MockCryptoAPIFactory.get_order_status_response)
        mock.cancel_order = MockClass(side_effect=MockCryptoAPIFactory.cancel_order_response)
        mock.get_trade_history = MockClass(side_effect=MockCryptoAPIFactory.get_trade_history_response)
        
        return mock
    
    @staticmethod
    def get_ticker_response(symbol: str = "BTC_USDT") -> Dict[str, Any]:
        """Generate mock ticker data."""
        base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
        return {
            "symbol": symbol,
            "price": round(base_price * random.uniform(0.95, 1.05), 2),
            "bid": round(base_price * random.uniform(0.949, 0.951), 2),
            "ask": round(base_price * random.uniform(1.049, 1.051), 2),
            "volume_24h": random.uniform(1000000, 5000000),
            "price_change_24h": random.uniform(-5, 5),
            "high_24h": round(base_price * 1.08, 2),
            "low_24h": round(base_price * 0.92, 2),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def get_orderbook_response(symbol: str = "BTC_USDT", depth: int = 10) -> Dict[str, Any]:
        """Generate mock orderbook data."""
        base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
        
        bids = []
        asks = []
        
        for i in range(depth):
            bid_price = round(base_price * (0.999 - i * 0.0001), 2)
            ask_price = round(base_price * (1.001 + i * 0.0001), 2)
            
            bids.append({
                "price": bid_price,
                "quantity": round(random.uniform(0.1, 2.0), 4),
                "total": round(bid_price * random.uniform(0.1, 2.0), 2)
            })
            
            asks.append({
                "price": ask_price,
                "quantity": round(random.uniform(0.1, 2.0), 4),
                "total": round(ask_price * random.uniform(0.1, 2.0), 2)
            })
        
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def get_candles_response(
        symbol: str = "BTC_USDT",
        interval: str = "1h",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate mock candle/OHLCV data."""
        base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
        candles = []
        
        current_time = datetime.now(timezone.utc)
        interval_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440
        }.get(interval, 60)
        
        for i in range(limit):
            timestamp = current_time - timedelta(minutes=interval_minutes * i)
            open_price = base_price * random.uniform(0.98, 1.02)
            close_price = open_price * random.uniform(0.99, 1.01)
            high_price = max(open_price, close_price) * random.uniform(1.0, 1.02)
            low_price = min(open_price, close_price) * random.uniform(0.98, 1.0)
            
            candles.append({
                "timestamp": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(random.uniform(100, 1000), 2)
            })
        
        return candles
    
    @staticmethod
    def place_order_response(
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "limit"
    ) -> Dict[str, Any]:
        """Generate mock order placement response."""
        order_id = fake.uuid4()
        return {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price or MockCryptoAPIFactory.get_ticker_response(symbol)["price"],
            "order_type": order_type,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fills": []
        }
    
    @staticmethod
    def get_balance_response() -> Dict[str, Any]:
        """Generate mock balance data."""
        return {
            "BTC": {
                "free": round(random.uniform(0.1, 2.0), 8),
                "locked": round(random.uniform(0, 0.5), 8),
                "total": round(random.uniform(0.1, 2.5), 8)
            },
            "ETH": {
                "free": round(random.uniform(1, 10), 8),
                "locked": round(random.uniform(0, 2), 8),
                "total": round(random.uniform(1, 12), 8)
            },
            "USDT": {
                "free": round(random.uniform(1000, 50000), 2),
                "locked": round(random.uniform(0, 5000), 2),
                "total": round(random.uniform(1000, 55000), 2)
            }
        }
    
    @staticmethod
    def get_order_status_response(order_id: str) -> Dict[str, Any]:
        """Generate mock order status."""
        statuses = ["pending", "partial", "filled", "cancelled"]
        status = random.choice(statuses)
        
        return {
            "order_id": order_id,
            "status": status,
            "filled_quantity": random.uniform(0, 1) if status != "cancelled" else 0,
            "remaining_quantity": random.uniform(0, 1) if status == "partial" else 0,
            "average_price": round(random.uniform(49000, 51000), 2),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def cancel_order_response(order_id: str) -> Dict[str, Any]:
        """Generate mock order cancellation response."""
        return {
            "order_id": order_id,
            "status": "cancelled",
            "cancelled_at": datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def get_trade_history_response(symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Generate mock trade history."""
        trades = []
        symbols = [symbol] if symbol else ["BTC_USDT", "ETH_USDT", "XRP_USDT"]
        
        for i in range(limit):
            trade_symbol = random.choice(symbols)
            base_price = 50000 if "BTC" in trade_symbol else 3000 if "ETH" in trade_symbol else 1
            
            trades.append({
                "trade_id": fake.uuid4(),
                "order_id": fake.uuid4(),
                "symbol": trade_symbol,
                "side": random.choice(["buy", "sell"]),
                "price": round(base_price * random.uniform(0.98, 1.02), 2),
                "quantity": round(random.uniform(0.01, 1.0), 4),
                "fee": round(random.uniform(0.1, 10), 4),
                "fee_currency": "USDT",
                "timestamp": (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat()
            })
        
        return trades


class MockLLMClientFactory:
    """Factory for creating mock LLM client responses."""
    
    @staticmethod
    def create_mock(async_mode: bool = True) -> Mock:
        """Create a mock LLM client."""
        MockClass = AsyncMock if async_mode else Mock
        mock = MockClass()
        
        # Configure common methods
        mock.analyze_sentiment = MockClass(side_effect=MockLLMClientFactory.analyze_sentiment_response)
        mock.generate_trading_signal = MockClass(side_effect=MockLLMClientFactory.generate_trading_signal_response)
        mock.summarize_news = MockClass(side_effect=MockLLMClientFactory.summarize_news_response)
        mock.analyze_market_conditions = MockClass(side_effect=MockLLMClientFactory.analyze_market_conditions_response)
        mock.generate_report = MockClass(side_effect=MockLLMClientFactory.generate_report_response)
        
        return mock
    
    @staticmethod
    def analyze_sentiment_response(text: str) -> Dict[str, Any]:
        """Generate mock sentiment analysis response."""
        sentiments = ["bullish", "bearish", "neutral"]
        sentiment = random.choice(sentiments)
        
        return {
            "sentiment": sentiment,
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "scores": {
                "positive": round(random.uniform(0, 1), 2),
                "negative": round(random.uniform(0, 1), 2),
                "neutral": round(random.uniform(0, 1), 2)
            },
            "keywords": fake.words(nb=5),
            "entities": [
                {"text": "Bitcoin", "type": "CRYPTO", "relevance": 0.9},
                {"text": "Federal Reserve", "type": "ORG", "relevance": 0.7}
            ]
        }
    
    @staticmethod
    def generate_trading_signal_response(market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock trading signal."""
        strategies = ["swing", "momentum", "mirror", "mean_reversion"]
        actions = ["buy", "sell", "hold"]
        
        return {
            "strategy": random.choice(strategies),
            "action": random.choice(actions),
            "confidence": round(random.uniform(0.6, 0.9), 2),
            "entry_price": market_data.get("price", 50000) * random.uniform(0.99, 1.01),
            "stop_loss": market_data.get("price", 50000) * random.uniform(0.95, 0.98),
            "take_profit": market_data.get("price", 50000) * random.uniform(1.02, 1.05),
            "position_size": round(random.uniform(0.01, 0.1), 4),
            "reasoning": fake.paragraph(nb_sentences=3),
            "risk_score": round(random.uniform(1, 5), 1)
        }
    
    @staticmethod
    def summarize_news_response(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate mock news summary."""
        return {
            "summary": fake.paragraph(nb_sentences=5),
            "key_points": fake.sentences(nb=3),
            "market_impact": random.choice(["high", "medium", "low"]),
            "affected_assets": ["BTC", "ETH", "stocks"],
            "sentiment_shift": round(random.uniform(-0.5, 0.5), 2),
            "recommendation": random.choice(["monitor", "action_required", "no_action"])
        }
    
    @staticmethod
    def analyze_market_conditions_response() -> Dict[str, Any]:
        """Generate mock market analysis."""
        return {
            "overall_trend": random.choice(["bullish", "bearish", "ranging"]),
            "volatility": random.choice(["high", "medium", "low"]),
            "market_phase": random.choice(["accumulation", "markup", "distribution", "markdown"]),
            "key_levels": {
                "support": [48000, 47000, 45000],
                "resistance": [52000, 53000, 55000]
            },
            "indicators": {
                "rsi": round(random.uniform(30, 70), 1),
                "macd": random.choice(["bullish", "bearish"]),
                "volume_trend": random.choice(["increasing", "decreasing", "stable"])
            },
            "recommendation": fake.paragraph(nb_sentences=2)
        }
    
    @staticmethod
    def generate_report_response(report_type: str) -> Dict[str, Any]:
        """Generate mock report."""
        return {
            "report_type": report_type,
            "title": f"{report_type.title()} Report - {fake.date()}",
            "executive_summary": fake.paragraph(nb_sentences=4),
            "sections": [
                {
                    "title": "Market Overview",
                    "content": fake.paragraph(nb_sentences=5)
                },
                {
                    "title": "Performance Analysis",
                    "content": fake.paragraph(nb_sentences=5)
                },
                {
                    "title": "Recommendations",
                    "content": fake.paragraph(nb_sentences=3)
                }
            ],
            "metrics": {
                "total_trades": random.randint(10, 100),
                "win_rate": round(random.uniform(0.4, 0.7), 2),
                "profit_factor": round(random.uniform(1.2, 2.5), 2),
                "sharpe_ratio": round(random.uniform(0.5, 2.0), 2)
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }


class MockNewsAPIFactory:
    """Factory for creating mock news API responses."""
    
    @staticmethod
    def create_mock(async_mode: bool = True) -> Mock:
        """Create a mock news API client."""
        MockClass = AsyncMock if async_mode else Mock
        mock = MockClass()
        
        # Configure common methods
        mock.get_latest_news = MockClass(side_effect=MockNewsAPIFactory.get_latest_news_response)
        mock.search_news = MockClass(side_effect=MockNewsAPIFactory.search_news_response)
        mock.get_news_by_symbol = MockClass(side_effect=MockNewsAPIFactory.get_news_by_symbol_response)
        mock.get_economic_calendar = MockClass(side_effect=MockNewsAPIFactory.get_economic_calendar_response)
        
        return mock
    
    @staticmethod
    def get_latest_news_response(limit: int = 10) -> List[Dict[str, Any]]:
        """Generate mock latest news."""
        news = []
        categories = ["crypto", "stocks", "forex", "commodities", "economy"]
        
        for i in range(limit):
            news.append({
                "id": fake.uuid4(),
                "title": fake.sentence(nb_words=10),
                "summary": fake.paragraph(nb_sentences=3),
                "content": fake.text(max_nb_chars=500),
                "source": random.choice(["Reuters", "Bloomberg", "CoinDesk", "WSJ", "FT"]),
                "category": random.choice(categories),
                "tags": fake.words(nb=3),
                "published_at": (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat(),
                "url": fake.url(),
                "image_url": fake.image_url(),
                "relevance_score": round(random.uniform(0.5, 1.0), 2)
            })
        
        return news
    
    @staticmethod
    def search_news_response(query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Generate mock news search results."""
        results = MockNewsAPIFactory.get_latest_news_response(limit)
        # Add query relevance
        for result in results:
            result["query_relevance"] = round(random.uniform(0.7, 1.0), 2)
            result["highlights"] = [fake.sentence() for _ in range(2)]
        return results
    
    @staticmethod
    def get_news_by_symbol_response(symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Generate mock news for specific symbol."""
        news = MockNewsAPIFactory.get_latest_news_response(limit)
        for item in news:
            item["symbols"] = [symbol]
            item["symbol_relevance"] = round(random.uniform(0.8, 1.0), 2)
        return news
    
    @staticmethod
    def get_economic_calendar_response(days: int = 7) -> List[Dict[str, Any]]:
        """Generate mock economic calendar events."""
        events = []
        event_types = ["Interest Rate Decision", "GDP Report", "Inflation Data", "Employment Report", "Earnings Release"]
        impact_levels = ["high", "medium", "low"]
        
        for i in range(days * 3):  # 3 events per day average
            events.append({
                "id": fake.uuid4(),
                "event": random.choice(event_types),
                "country": fake.country_code(),
                "currency": random.choice(["USD", "EUR", "GBP", "JPY", "CNY"]),
                "impact": random.choice(impact_levels),
                "forecast": f"{random.uniform(-2, 5):.1f}%",
                "previous": f"{random.uniform(-2, 5):.1f}%",
                "actual": f"{random.uniform(-2, 5):.1f}%" if random.random() > 0.5 else None,
                "scheduled_time": (datetime.now(timezone.utc) + timedelta(days=i//3, hours=i%24)).isoformat(),
                "description": fake.sentence()
            })
        
        return events


class MockMarketDataAPIFactory:
    """Factory for creating mock market data API responses."""
    
    @staticmethod
    def create_mock(async_mode: bool = True) -> Mock:
        """Create a mock market data API client."""
        MockClass = AsyncMock if async_mode else Mock
        mock = MockClass()
        
        # Configure common methods
        mock.get_market_overview = MockClass(side_effect=MockMarketDataAPIFactory.get_market_overview_response)
        mock.get_sector_performance = MockClass(side_effect=MockMarketDataAPIFactory.get_sector_performance_response)
        mock.get_market_movers = MockClass(side_effect=MockMarketDataAPIFactory.get_market_movers_response)
        mock.get_technical_indicators = MockClass(side_effect=MockMarketDataAPIFactory.get_technical_indicators_response)
        
        return mock
    
    @staticmethod
    def get_market_overview_response() -> Dict[str, Any]:
        """Generate mock market overview."""
        return {
            "indices": {
                "SP500": {
                    "value": round(random.uniform(4200, 4400), 2),
                    "change": round(random.uniform(-2, 2), 2),
                    "change_percent": round(random.uniform(-0.5, 0.5), 2)
                },
                "NASDAQ": {
                    "value": round(random.uniform(13000, 14000), 2),
                    "change": round(random.uniform(-3, 3), 2),
                    "change_percent": round(random.uniform(-0.5, 0.5), 2)
                },
                "DJI": {
                    "value": round(random.uniform(34000, 35000), 2),
                    "change": round(random.uniform(-2, 2), 2),
                    "change_percent": round(random.uniform(-0.5, 0.5), 2)
                }
            },
            "currencies": {
                "EUR_USD": round(random.uniform(1.05, 1.10), 4),
                "GBP_USD": round(random.uniform(1.20, 1.30), 4),
                "USD_JPY": round(random.uniform(140, 150), 2)
            },
            "commodities": {
                "gold": round(random.uniform(1900, 2100), 2),
                "oil": round(random.uniform(70, 90), 2),
                "silver": round(random.uniform(22, 26), 2)
            },
            "crypto": {
                "bitcoin": round(random.uniform(45000, 55000), 2),
                "ethereum": round(random.uniform(2800, 3500), 2)
            },
            "market_status": random.choice(["open", "closed", "pre-market", "after-hours"]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def get_sector_performance_response() -> Dict[str, Any]:
        """Generate mock sector performance data."""
        sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer", "Industrial", "Materials", "Utilities"]
        performance = {}
        
        for sector in sectors:
            performance[sector] = {
                "change_1d": round(random.uniform(-3, 3), 2),
                "change_1w": round(random.uniform(-5, 5), 2),
                "change_1m": round(random.uniform(-10, 10), 2),
                "change_ytd": round(random.uniform(-20, 30), 2),
                "volume": random.randint(1000000, 10000000),
                "leaders": [fake.company() for _ in range(3)]
            }
        
        return {
            "sectors": performance,
            "best_performing": max(sectors, key=lambda s: performance[s]["change_1d"]),
            "worst_performing": min(sectors, key=lambda s: performance[s]["change_1d"]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def get_market_movers_response() -> Dict[str, Any]:
        """Generate mock market movers data."""
        def generate_mover():
            return {
                "symbol": fake.lexify(text='????').upper(),
                "name": fake.company(),
                "price": round(random.uniform(10, 500), 2),
                "change": round(random.uniform(-20, 20), 2),
                "change_percent": round(random.uniform(-15, 15), 2),
                "volume": random.randint(1000000, 50000000),
                "market_cap": random.randint(1000000000, 100000000000)
            }
        
        return {
            "gainers": [generate_mover() for _ in range(10)],
            "losers": [generate_mover() for _ in range(10)],
            "most_active": [generate_mover() for _ in range(10)],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def get_technical_indicators_response(symbol: str) -> Dict[str, Any]:
        """Generate mock technical indicators."""
        return {
            "symbol": symbol,
            "indicators": {
                "rsi": {
                    "value": round(random.uniform(20, 80), 2),
                    "signal": random.choice(["oversold", "neutral", "overbought"])
                },
                "macd": {
                    "macd": round(random.uniform(-5, 5), 2),
                    "signal": round(random.uniform(-5, 5), 2),
                    "histogram": round(random.uniform(-2, 2), 2),
                    "trend": random.choice(["bullish", "bearish"])
                },
                "moving_averages": {
                    "sma_20": round(random.uniform(45000, 55000), 2),
                    "sma_50": round(random.uniform(44000, 54000), 2),
                    "sma_200": round(random.uniform(43000, 53000), 2),
                    "ema_12": round(random.uniform(45500, 55500), 2),
                    "ema_26": round(random.uniform(45000, 55000), 2)
                },
                "bollinger_bands": {
                    "upper": round(random.uniform(52000, 58000), 2),
                    "middle": round(random.uniform(48000, 52000), 2),
                    "lower": round(random.uniform(44000, 48000), 2)
                },
                "stochastic": {
                    "k": round(random.uniform(0, 100), 2),
                    "d": round(random.uniform(0, 100), 2),
                    "signal": random.choice(["buy", "sell", "neutral"])
                },
                "atr": round(random.uniform(500, 2000), 2),
                "volume": {
                    "current": random.randint(1000000, 10000000),
                    "average": random.randint(800000, 8000000),
                    "trend": random.choice(["increasing", "decreasing", "stable"])
                }
            },
            "support_resistance": {
                "support_levels": sorted([round(random.uniform(44000, 48000), 2) for _ in range(3)]),
                "resistance_levels": sorted([round(random.uniform(52000, 56000), 2) for _ in range(3)])
            },
            "trend": {
                "short_term": random.choice(["up", "down", "sideways"]),
                "medium_term": random.choice(["up", "down", "sideways"]),
                "long_term": random.choice(["up", "down", "sideways"])
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class MockBrokerAPIFactory:
    """Factory for creating mock broker API responses."""
    
    @staticmethod
    def create_mock(async_mode: bool = True) -> Mock:
        """Create a mock broker API client."""
        MockClass = AsyncMock if async_mode else Mock
        mock = MockClass()
        
        # Configure common methods
        mock.get_account_info = MockClass(side_effect=MockBrokerAPIFactory.get_account_info_response)
        mock.get_positions = MockClass(side_effect=MockBrokerAPIFactory.get_positions_response)
        mock.get_orders = MockClass(side_effect=MockBrokerAPIFactory.get_orders_response)
        mock.place_order = MockClass(side_effect=MockBrokerAPIFactory.place_order_response)
        mock.cancel_order = MockClass(side_effect=MockBrokerAPIFactory.cancel_order_response)
        mock.get_margin_info = MockClass(side_effect=MockBrokerAPIFactory.get_margin_info_response)
        
        return mock
    
    @staticmethod
    def get_account_info_response() -> Dict[str, Any]:
        """Generate mock account information."""
        return {
            "account_id": fake.uuid4(),
            "account_type": random.choice(["cash", "margin", "portfolio"]),
            "currency": "USD",
            "balance": {
                "cash": round(random.uniform(10000, 100000), 2),
                "securities": round(random.uniform(50000, 500000), 2),
                "total": round(random.uniform(60000, 600000), 2)
            },
            "buying_power": round(random.uniform(20000, 200000), 2),
            "day_trading_buying_power": round(random.uniform(40000, 400000), 2),
            "maintenance_margin": round(random.uniform(5000, 50000), 2),
            "status": "active",
            "created_at": fake.date_time_this_year().isoformat(),
            "restrictions": []
        }
    
    @staticmethod
    def get_positions_response() -> List[Dict[str, Any]]:
        """Generate mock positions data."""
        positions = []
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC", "ETH"]
        
        for _ in range(random.randint(3, 10)):
            symbol = random.choice(symbols)
            quantity = random.randint(10, 1000) if symbol not in ["BTC", "ETH"] else round(random.uniform(0.1, 5), 4)
            avg_price = random.uniform(100, 1000) if symbol not in ["BTC", "ETH"] else random.uniform(20000, 60000)
            current_price = avg_price * random.uniform(0.9, 1.1)
            
            positions.append({
                "symbol": symbol,
                "quantity": quantity,
                "average_price": round(avg_price, 2),
                "current_price": round(current_price, 2),
                "market_value": round(quantity * current_price, 2),
                "cost_basis": round(quantity * avg_price, 2),
                "unrealized_pnl": round(quantity * (current_price - avg_price), 2),
                "unrealized_pnl_percent": round((current_price - avg_price) / avg_price * 100, 2),
                "side": "long",
                "asset_class": "crypto" if symbol in ["BTC", "ETH"] else "equity"
            })
        
        return positions
    
    @staticmethod
    def get_orders_response(status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate mock orders data."""
        orders = []
        statuses = ["pending", "partial", "filled", "cancelled", "rejected"]
        if status:
            statuses = [status]
        
        for _ in range(random.randint(2, 8)):
            order_status = random.choice(statuses)
            orders.append({
                "order_id": fake.uuid4(),
                "symbol": random.choice(["AAPL", "MSFT", "GOOGL", "BTC", "ETH"]),
                "side": random.choice(["buy", "sell"]),
                "order_type": random.choice(["market", "limit", "stop", "stop_limit"]),
                "quantity": round(random.uniform(1, 100), 2),
                "price": round(random.uniform(100, 1000), 2) if order_status != "market" else None,
                "status": order_status,
                "filled_quantity": round(random.uniform(0, 100), 2) if order_status in ["partial", "filled"] else 0,
                "average_fill_price": round(random.uniform(100, 1000), 2) if order_status in ["partial", "filled"] else None,
                "created_at": fake.date_time_this_month().isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
        
        return orders
    
    @staticmethod
    def place_order_response(**kwargs) -> Dict[str, Any]:
        """Generate mock order placement response."""
        return {
            "order_id": fake.uuid4(),
            "status": "pending",
            "symbol": kwargs.get("symbol", "AAPL"),
            "side": kwargs.get("side", "buy"),
            "order_type": kwargs.get("order_type", "limit"),
            "quantity": kwargs.get("quantity", 100),
            "price": kwargs.get("price", 150.00),
            "time_in_force": kwargs.get("time_in_force", "day"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "commission": round(random.uniform(0.5, 5), 2)
        }
    
    @staticmethod
    def cancel_order_response(order_id: str) -> Dict[str, Any]:
        """Generate mock order cancellation response."""
        return {
            "order_id": order_id,
            "status": "cancelled",
            "cancelled_at": datetime.now(timezone.utc).isoformat(),
            "reason": "User requested cancellation"
        }
    
    @staticmethod
    def get_margin_info_response() -> Dict[str, Any]:
        """Generate mock margin information."""
        return {
            "margin_balance": round(random.uniform(50000, 500000), 2),
            "margin_used": round(random.uniform(10000, 100000), 2),
            "margin_available": round(random.uniform(40000, 400000), 2),
            "margin_call": False,
            "maintenance_margin_requirement": round(random.uniform(20000, 200000), 2),
            "initial_margin_requirement": round(random.uniform(25000, 250000), 2),
            "leverage": round(random.uniform(1, 4), 1),
            "equity": round(random.uniform(60000, 600000), 2),
            "last_margin_call": None
        }