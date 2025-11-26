"""
Broker API Response Fixtures for Unit Testing

This module contains comprehensive test fixtures for all supported broker APIs,
providing standardized mock responses for testing various scenarios.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Any


class BrokerResponseFixtures:
    """Centralized broker API response fixtures"""
    
    # Alpaca API Fixtures
    ALPACA_ACCOUNT_RESPONSE = {
        "id": "test_account_123",
        "account_number": "TEST123456",
        "status": "ACTIVE",
        "currency": "USD",
        "buying_power": "200000.00",
        "cash": "100000.00",
        "portfolio_value": "150000.00",
        "pattern_day_trader": False,
        "trade_suspended_by_user": False,
        "trading_blocked": False,
        "transfers_blocked": False,
        "account_blocked": False,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-15T10:00:00Z",
        "equity": "150000.00",
        "last_equity": "148500.00",
        "multiplier": "2",
        "shorting_enabled": True,
        "long_market_value": "50000.00",
        "short_market_value": "0.00",
        "position_market_value": "50000.00",
        "initial_margin": "25000.00",
        "maintenance_margin": "12500.00",
        "sma": "0.00",
        "daytrade_count": 0
    }
    
    ALPACA_ORDER_RESPONSE = {
        "id": "order_123",
        "client_order_id": "client_123",
        "created_at": "2024-01-15T10:00:00Z",
        "updated_at": "2024-01-15T10:00:01Z",
        "submitted_at": "2024-01-15T10:00:00Z",
        "filled_at": None,
        "expired_at": None,
        "canceled_at": None,
        "failed_at": None,
        "replaced_at": None,
        "replaced_by": None,
        "replaces": None,
        "asset_id": "asset_123",
        "symbol": "AAPL",
        "asset_class": "us_equity",
        "qty": "100",
        "filled_qty": "0",
        "filled_avg_price": None,
        "order_class": "simple",
        "order_type": "market",
        "type": "market",
        "side": "buy",
        "time_in_force": "day",
        "limit_price": None,
        "stop_price": None,
        "status": "accepted",
        "extended_hours": False,
        "legs": None,
        "trail_percent": None,
        "trail_price": None,
        "hwm": None
    }
    
    ALPACA_FILLED_ORDER_RESPONSE = {
        **ALPACA_ORDER_RESPONSE,
        "status": "filled",
        "filled_at": "2024-01-15T10:00:02Z",
        "filled_qty": "100",
        "filled_avg_price": "150.25"
    }
    
    ALPACA_POSITION_RESPONSE = {
        "asset_id": "asset_123",
        "symbol": "AAPL",
        "exchange": "NASDAQ",
        "asset_class": "us_equity",
        "avg_entry_price": "150.00",
        "qty": "100",
        "side": "long",
        "market_value": "15025.00",
        "cost_basis": "15000.00",
        "unrealized_pl": "25.00",
        "unrealized_plpc": "0.0017",
        "unrealized_intraday_pl": "25.00",
        "unrealized_intraday_plpc": "0.0017",
        "current_price": "150.25",
        "lastday_price": "149.80",
        "change_today": "0.0030"
    }
    
    ALPACA_PORTFOLIO_HISTORY = {
        "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "equity": ["148000.00", "149500.00", "150000.00"],
        "profit_loss": ["-2000.00", "-500.00", "0.00"],
        "profit_loss_pct": ["-0.0133", "-0.0033", "0.0000"],
        "base_value": "150000.00",
        "timeframe": "1D"
    }
    
    # Interactive Brokers API Fixtures
    IBKR_ACCOUNT_RESPONSE = {
        "accountId": "U123456",
        "accountType": "INDIVIDUAL",
        "tradingPermissions": ["STOCKS", "OPTIONS", "FUTURES"],
        "accountTitle": "Test Account",
        "currency": "USD",
        "state": "OPEN",
        "covestor": False,
        "parent": {},
        "desc": "U123456",
        "displayName": "U123456",
        "faclient": False,
        "clearing": "IB",
        "capabilities": ["BRACKET", "STOP", "TRAIL"],
        "noTradingByAlgoOrders": False
    }
    
    IBKR_PORTFOLIO_RESPONSE = {
        "accounts": [{
            "id": "U123456",
            "accountcode": "U123456",
            "accountready": True,
            "accounttype": "INDIVIDUAL",
            "accruedcash": "0.00",
            "accruedcash-c": "USD",
            "availablefunds": "100000.00",
            "availablefunds-c": "USD",
            "balance": "150000.00",
            "balance-c": "USD",
            "buyingpower": "200000.00",
            "cashbalance": "100000.00",
            "currency": "USD",
            "cushion": "0.15",
            "daytradesremaining": 3,
            "equity": "150000.00",
            "excessliquidity": "125000.00",
            "grosspositionvalue": "50000.00",
            "guarantee": "0.00",
            "initmarginreq": "25000.00",
            "maintmarginreq": "12500.00",
            "netliquidation": "150000.00",
            "portfoliovalue": "150000.00",
            "realizedpnl": "0.00",
            "totalcashvalue": "100000.00",
            "unrealizedpnl": "25.00"
        }]
    }
    
    IBKR_ORDER_RESPONSE = {
        "orderId": 12345,
        "conid": 265598,
        "symbol": "AAPL",
        "side": "BUY",
        "orderType": "MKT",
        "totalSize": 100,
        "filledQuantity": 0,
        "avgPrice": 0,
        "status": "Submitted",
        "orderStatus": "Submitted",
        "remainingQuantity": 100,
        "timestamp": 1642176000000,
        "warning_message": None
    }
    
    # TD Ameritrade API Fixtures
    TD_ACCOUNT_RESPONSE = {
        "securitiesAccount": {
            "type": "CASH",
            "accountId": "123456789",
            "roundTrips": 0,
            "isDayTrader": False,
            "isClosingOnlyRestricted": False,
            "currentBalances": {
                "accruedInterest": 0,
                "cashBalance": 100000.00,
                "cashReceipts": 0,
                "longOptionMarketValue": 0,
                "liquidationValue": 150000.00,
                "longMarketValue": 50000.00,
                "moneyMarketFund": 0,
                "savings": 0,
                "shortMarketValue": 0,
                "pendingDeposits": 0,
                "shortOptionMarketValue": 0,
                "mutualFundValue": 0,
                "bondValue": 0,
                "cashAvailableForTrading": 100000.00,
                "cashAvailableForWithdrawal": 100000.00,
                "cashCall": 0,
                "longNonMarginableMarketValue": 0,
                "totalCash": 100000.00,
                "cashDebitCallValue": 0,
                "unsettledCash": 0
            },
            "initialBalances": {
                "accruedInterest": 0,
                "availableFundsNonMarginableTrade": 100000.00,
                "bondValue": 0,
                "buyingPower": 100000.00,
                "cashBalance": 100000.00,
                "cashAvailableForTrading": 100000.00,
                "cashReceipts": 0,
                "dayTradingBuyingPower": 0,
                "dayTradingBuyingPowerCall": 0,
                "dayTradingEquityCall": 0,
                "equity": 150000.00,
                "equityPercentage": 100,
                "liquidationValue": 150000.00,
                "longMarketValue": 50000.00,
                "longOptionMarketValue": 0,
                "longStockValue": 50000.00,
                "maintenanceCall": 0,
                "maintenanceRequirement": 0,
                "margin": 0,
                "marginEquity": 150000.00,
                "moneyMarketFund": 0,
                "mutualFundValue": 0,
                "regTCall": 0,
                "shortMarketValue": 0,
                "shortOptionMarketValue": 0,
                "shortStockValue": 0,
                "totalCash": 100000.00,
                "isInCall": False,
                "unsettledCash": 0,
                "pendingDeposits": 0,
                "marginBalance": 0,
                "shortBalance": 0,
                "accountValue": 150000.00
            }
        }
    }
    
    TD_ORDER_RESPONSE = {
        "session": "NORMAL",
        "duration": "DAY",
        "orderType": "MARKET",
        "complexOrderStrategyType": "NONE",
        "quantity": 100,
        "filledQuantity": 0,
        "remainingQuantity": 100,
        "requestedDestination": "AUTO",
        "destinationLinkName": "AutoRoute",
        "price": 0,
        "orderLegCollection": [{
            "orderLegType": "EQUITY",
            "legId": 1,
            "instrument": {
                "assetType": "EQUITY",
                "cusip": "037833100",
                "symbol": "AAPL"
            },
            "instruction": "BUY",
            "positionEffect": "OPENING",
            "quantity": 100
        }],
        "orderStrategyType": "SINGLE",
        "orderId": 987654321,
        "cancelable": True,
        "editable": True,
        "status": "QUEUED",
        "enteredTime": "2024-01-15T10:00:00+0000",
        "closeTime": None,
        "tag": "API_TRD",
        "accountId": 123456789
    }
    
    # News API Fixtures
    NEWS_ARTICLE_FIXTURE = {
        "id": "article_123",
        "headline": "Apple Reports Record Q4 Earnings",
        "summary": "Apple Inc. reported record-breaking Q4 earnings, beating analyst expectations across all key metrics including revenue, earnings per share, and forward guidance.",
        "content": "Apple Inc. (NASDAQ: AAPL) today announced financial results for its fiscal 2024 fourth quarter ended September 30, 2024. The Company posted quarterly revenue of $94.9 billion, up 6 percent year over year, and quarterly earnings per share of $1.64, up 12 percent year over year...",
        "author": "John Doe",
        "source": "reuters",
        "created_at": "2024-01-15T09:00:00Z",
        "updated_at": "2024-01-15T09:00:00Z",
        "published_at": "2024-01-15T09:00:00Z",
        "url": "https://example.com/article/123",
        "symbols": ["AAPL"],
        "categories": ["earnings", "technology"],
        "sentiment": {
            "polarity": 0.8,
            "magnitude": 0.9,
            "confidence": 0.95,
            "label": "positive"
        },
        "relevance_score": 0.95,
        "language": "en",
        "word_count": 1250,
        "reading_time": 5
    }
    
    NEGATIVE_NEWS_FIXTURE = {
        **NEWS_ARTICLE_FIXTURE,
        "id": "article_456",
        "headline": "Apple Faces Supply Chain Disruptions",
        "summary": "Apple warns of potential iPhone production delays due to ongoing supply chain disruptions in Asia.",
        "sentiment": {
            "polarity": -0.6,
            "magnitude": 0.8,
            "confidence": 0.90,
            "label": "negative"
        }
    }
    
    NEUTRAL_NEWS_FIXTURE = {
        **NEWS_ARTICLE_FIXTURE,
        "id": "article_789",
        "headline": "Apple Announces New Product Event Date",
        "summary": "Apple has announced that its next product event will be held on March 8th, 2024.",
        "sentiment": {
            "polarity": 0.1,
            "magnitude": 0.3,
            "confidence": 0.85,
            "label": "neutral"
        }
    }
    
    # Market Data Fixtures
    MARKET_DATA_BAR_FIXTURE = {
        "symbol": "AAPL",
        "timestamp": "2024-01-15T10:00:00Z",
        "open": 150.00,
        "high": 150.50,
        "low": 149.75,
        "close": 150.25,
        "volume": 1000000,
        "trade_count": 5000,
        "vwap": 150.12
    }
    
    MARKET_DATA_QUOTE_FIXTURE = {
        "symbol": "AAPL",
        "timestamp": "2024-01-15T10:00:00Z",
        "bid_price": 150.20,
        "bid_size": 100,
        "ask_price": 150.25,
        "ask_size": 200,
        "last_price": 150.25,
        "last_size": 50
    }
    
    # Error Response Fixtures
    ERROR_RESPONSES = {
        "unauthorized": {
            "status_code": 401,
            "response": {
                "code": 40110000,
                "message": "request is not authorized"
            }
        },
        "bad_request": {
            "status_code": 400,
            "response": {
                "code": 40010001,
                "message": "request body cannot be parsed"
            }
        },
        "rate_limited": {
            "status_code": 429,
            "response": {
                "code": 42910000,
                "message": "rate limit exceeded"
            }
        },
        "server_error": {
            "status_code": 500,
            "response": {
                "code": 50010001,
                "message": "internal server error"
            }
        },
        "service_unavailable": {
            "status_code": 503,
            "response": {
                "code": 50310000,
                "message": "service temporarily unavailable"
            }
        }
    }
    
    # Scenario-based Fixtures
    BULL_MARKET_SCENARIO = {
        "market_conditions": "bull",
        "volatility": "low",
        "trend": "upward",
        "sentiment": "positive",
        "volume": "normal"
    }
    
    BEAR_MARKET_SCENARIO = {
        "market_conditions": "bear",
        "volatility": "high",
        "trend": "downward",
        "sentiment": "negative",
        "volume": "high"
    }
    
    HIGH_VOLATILITY_SCENARIO = {
        "market_conditions": "volatile",
        "volatility": "extreme",
        "trend": "sideways",
        "sentiment": "mixed",
        "volume": "very_high"
    }
    
    @classmethod
    def get_order_sequence(cls, symbol: str, quantity: int, side: str) -> List[Dict[str, Any]]:
        """Get sequence of order status updates for testing"""
        base_order = cls.ALPACA_ORDER_RESPONSE.copy()
        base_order.update({
            "symbol": symbol,
            "qty": str(quantity),
            "side": side
        })
        
        sequence = [
            {**base_order, "status": "new"},
            {**base_order, "status": "accepted"},
            {**base_order, "status": "partially_filled", "filled_qty": str(quantity // 2)},
            {**base_order, "status": "filled", "filled_qty": str(quantity), "filled_avg_price": "150.25"}
        ]
        
        return sequence
    
    @classmethod
    def get_portfolio_timeline(cls, days: int) -> List[Dict[str, Any]]:
        """Generate portfolio value timeline for testing"""
        timeline = []
        base_value = 150000.00
        
        for i in range(days):
            date = datetime.now(timezone.utc).replace(day=i+1).isoformat()
            value = base_value + (i * 500)  # Simulate growth
            timeline.append({
                "date": date,
                "portfolio_value": value,
                "cash": 100000.00 - (i * 100),
                "positions_value": value - (100000.00 - (i * 100))
            })
        
        return timeline
    
    @classmethod
    def get_news_sentiment_timeline(cls, symbol: str, hours: int) -> List[Dict[str, Any]]:
        """Generate news sentiment timeline for testing"""
        timeline = []
        
        for i in range(hours):
            timestamp = datetime.now(timezone.utc).replace(hour=i).isoformat()
            sentiment_score = 0.5 + (i % 3 - 1) * 0.3  # Oscillating sentiment
            
            timeline.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "sentiment_score": sentiment_score,
                "article_count": 5 + (i % 10),
                "confidence": 0.8 + (i % 5) * 0.04
            })
        
        return timeline