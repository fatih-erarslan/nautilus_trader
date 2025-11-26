"""
HTTP response fixtures for Polymarket API testing.

Contains mock responses for all API endpoints to ensure comprehensive testing
without requiring actual API connections.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal


class HTTPResponseFixtures:
    """Collection of HTTP response fixtures for Polymarket API testing."""
    
    @staticmethod
    def get_markets_response(
        count: int = 5,
        status: str = "active",
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate mock response for GET /markets endpoint."""
        markets = []
        base_time = datetime.now()
        
        for i in range(count):
            market = {
                "id": f"0x{'a' * 40}{i:02d}",
                "question": f"Will crypto market reach new ATH in Q{i+1} 2024?",
                "description": f"Market resolves to Yes if crypto market reaches new all-time high in Q{i+1} 2024",
                "outcomes": ["Yes", "No"],
                "outcomePrices": [str(0.6 + i * 0.05), str(0.4 - i * 0.05)],
                "volume": str(10000 + i * 5000),
                "liquidity": str(50000 + i * 10000),
                "endDate": (base_time + timedelta(days=30 + i * 10)).isoformat() + "Z",
                "status": status,
                "category": category or "Crypto",
                "tags": ["crypto", "market", f"q{i+1}"],
                "createdAt": (base_time - timedelta(days=i)).isoformat() + "Z",
                "updatedAt": base_time.isoformat() + "Z",
                "feeRate": "0.02",
                "minOrderSize": "1.00",
                "maxOrderSize": "10000.00",
                "active": True,
                "closed": False,
                "marketMakerAddress": f"0x{'b' * 40}",
                "conditionId": f"0x{'c' * 64}",
                "questionId": f"0x{'d' * 64}",
                "resolutionSource": "UMA",
                "resolved": False,
                "resolutionTime": None,
                "winningOutcome": None,
                "participants": 150 + i * 25,
                "volume24h": str(1000 + i * 500),
                "priceChange24h": f"{(i - 2) * 0.05:.3f}",
                "metadata": {
                    "rules": "Market resolves based on official market data",
                    "sources": ["CoinGecko", "CoinMarketCap"],
                    "criteria": "Objective price data verification"
                }
            }
            markets.append(market)
        
        return {
            "status": "success",
            "data": markets,
            "pagination": {
                "page": 1,
                "limit": count,
                "total": count,
                "hasMore": False
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }
    
    @staticmethod
    def get_market_response(market_id: str) -> Dict[str, Any]:
        """Generate mock response for GET /markets/{id} endpoint."""
        base_time = datetime.now()
        
        return {
            "status": "success",
            "data": {
                "id": market_id,
                "question": "Will Bitcoin reach $100,000 by end of 2024?",
                "description": "Market resolves to Yes if Bitcoin reaches $100,000 by December 31, 2024",
                "outcomes": ["Yes", "No"],
                "outcomePrices": ["0.65", "0.35"],
                "volume": "75000.50",
                "liquidity": "150000.75",
                "endDate": (base_time + timedelta(days=45)).isoformat() + "Z",
                "status": "active",
                "category": "Crypto",
                "subcategory": "Bitcoin",
                "tags": ["bitcoin", "price", "2024"],
                "createdAt": (base_time - timedelta(days=10)).isoformat() + "Z",
                "updatedAt": base_time.isoformat() + "Z",
                "feeRate": "0.02",
                "minOrderSize": "1.00",
                "maxOrderSize": "10000.00",
                "active": True,
                "closed": False,
                "marketMakerAddress": "0x" + "b" * 40,
                "conditionId": "0x" + "c" * 64,
                "questionId": "0x" + "d" * 64,
                "resolutionSource": "UMA",
                "resolved": False,
                "resolutionTime": None,
                "winningOutcome": None,
                "participants": 243,
                "volume24h": "12500.25",
                "priceChange24h": "0.045",
                "orderBooks": {
                    "Yes": {
                        "bids": [
                            {"price": "0.64", "size": "1000.00"},
                            {"price": "0.63", "size": "1500.00"},
                            {"price": "0.62", "size": "2000.00"}
                        ],
                        "asks": [
                            {"price": "0.66", "size": "1200.00"},
                            {"price": "0.67", "size": "1800.00"},
                            {"price": "0.68", "size": "2500.00"}
                        ]
                    },
                    "No": {
                        "bids": [
                            {"price": "0.34", "size": "1100.00"},
                            {"price": "0.33", "size": "1600.00"},
                            {"price": "0.32", "size": "2100.00"}
                        ],
                        "asks": [
                            {"price": "0.36", "size": "1300.00"},
                            {"price": "0.37", "size": "1900.00"},
                            {"price": "0.38", "size": "2600.00"}
                        ]
                    }
                },
                "metadata": {
                    "rules": "Market resolves based on official Bitcoin price data",
                    "sources": ["CoinGecko", "CoinMarketCap", "Binance"],
                    "criteria": "USD price must reach $100,000 on at least one major exchange"
                }
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }
    
    @staticmethod
    def get_order_book_response(market_id: str, outcome: str) -> Dict[str, Any]:
        """Generate mock response for GET /markets/{id}/orderbook endpoint."""
        return {
            "status": "success",
            "data": {
                "marketId": market_id,
                "outcome": outcome,
                "bids": [
                    {"price": "0.64", "size": "1000.00", "timestamp": datetime.now().isoformat() + "Z"},
                    {"price": "0.63", "size": "1500.00", "timestamp": datetime.now().isoformat() + "Z"},
                    {"price": "0.62", "size": "2000.00", "timestamp": datetime.now().isoformat() + "Z"},
                    {"price": "0.61", "size": "2500.00", "timestamp": datetime.now().isoformat() + "Z"},
                    {"price": "0.60", "size": "3000.00", "timestamp": datetime.now().isoformat() + "Z"},
                ],
                "asks": [
                    {"price": "0.66", "size": "1200.00", "timestamp": datetime.now().isoformat() + "Z"},
                    {"price": "0.67", "size": "1800.00", "timestamp": datetime.now().isoformat() + "Z"},
                    {"price": "0.68", "size": "2500.00", "timestamp": datetime.now().isoformat() + "Z"},
                    {"price": "0.69", "size": "3000.00", "timestamp": datetime.now().isoformat() + "Z"},
                    {"price": "0.70", "size": "3500.00", "timestamp": datetime.now().isoformat() + "Z"},
                ],
                "spread": "0.02",
                "midPrice": "0.65",
                "totalBidSize": "10000.00",
                "totalAskSize": "12000.00",
                "timestamp": datetime.now().isoformat() + "Z"
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }
    
    @staticmethod
    def place_order_response(
        order_id: str,
        market_id: str,
        outcome: str,
        side: str,
        order_type: str,
        size: str,
        price: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate mock response for POST /orders endpoint."""
        now = datetime.now()
        
        return {
            "status": "success",
            "data": {
                "id": order_id,
                "marketId": market_id,
                "outcomeId": outcome,
                "side": side,
                "type": order_type,
                "size": size,
                "price": price,
                "filled": "0.00",
                "remaining": size,
                "status": "open",
                "timeInForce": "gtc",
                "createdAt": now.isoformat() + "Z",
                "updatedAt": now.isoformat() + "Z",
                "expiresAt": None,
                "fills": [],
                "feeRate": "0.02",
                "clientOrderId": None,
                "notionalValue": str(float(size) * float(price or "0.5")),
                "remainingValue": str(float(size) * float(price or "0.5"))
            },
            "timestamp": now.isoformat() + "Z"
        }
    
    @staticmethod
    def get_orders_response(
        market_id: Optional[str] = None,
        status: Optional[str] = None,
        count: int = 3
    ) -> Dict[str, Any]:
        """Generate mock response for GET /orders endpoint."""
        orders = []
        base_time = datetime.now()
        
        for i in range(count):
            order_time = base_time - timedelta(hours=i)
            order = {
                "id": f"order_{i}_{int(order_time.timestamp())}",
                "marketId": market_id or f"0x{'a' * 40}{i:02d}",
                "outcomeId": "Yes" if i % 2 == 0 else "No",
                "side": "buy" if i % 2 == 0 else "sell",
                "type": "limit",
                "size": str(100 + i * 50),
                "price": str(0.6 + i * 0.05),
                "filled": str(i * 25),
                "remaining": str(100 + i * 50 - i * 25),
                "status": status or ("open" if i < 2 else "filled"),
                "timeInForce": "gtc",
                "createdAt": order_time.isoformat() + "Z",
                "updatedAt": (order_time + timedelta(minutes=i * 15)).isoformat() + "Z",
                "expiresAt": None,
                "fills": [
                    {
                        "id": f"fill_{i}_{j}",
                        "orderId": f"order_{i}_{int(order_time.timestamp())}",
                        "price": str(0.6 + i * 0.05 + j * 0.01),
                        "size": str(12.5),
                        "side": "buy" if i % 2 == 0 else "sell",
                        "timestamp": (order_time + timedelta(minutes=j * 5)).isoformat() + "Z",
                        "fee": str(0.25),
                        "feeCurrency": "USDC"
                    }
                    for j in range(i)
                ] if i > 0 else [],
                "feeRate": "0.02",
                "clientOrderId": f"client_order_{i}"
            }
            orders.append(order)
        
        return {
            "status": "success",
            "data": orders,
            "pagination": {
                "page": 1,
                "limit": count,
                "total": count,
                "hasMore": False
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }
    
    @staticmethod
    def cancel_order_response(order_id: str) -> Dict[str, Any]:
        """Generate mock response for DELETE /orders/{id} endpoint."""
        return {
            "status": "success",
            "data": {
                "id": order_id,
                "status": "cancelled",
                "cancelledAt": datetime.now().isoformat() + "Z",
                "reason": "user_cancelled"
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }
    
    @staticmethod
    def get_positions_response(count: int = 3) -> Dict[str, Any]:
        """Generate mock response for GET /positions endpoint."""
        positions = []
        base_time = datetime.now()
        
        for i in range(count):
            position = {
                "marketId": f"0x{'a' * 40}{i:02d}",
                "outcome": "Yes" if i % 2 == 0 else "No",
                "size": str(100 + i * 50),
                "averagePrice": str(0.55 + i * 0.05),
                "currentPrice": str(0.6 + i * 0.05),
                "unrealizedPnl": str((0.6 + i * 0.05 - 0.55 - i * 0.05) * (100 + i * 50)),
                "realizedPnl": str(i * 25),
                "totalPnl": str((0.6 + i * 0.05 - 0.55 - i * 0.05) * (100 + i * 50) + i * 25),
                "currentValue": str((0.6 + i * 0.05) * (100 + i * 50)),
                "costBasis": str((0.55 + i * 0.05) * (100 + i * 50)),
                "trades": [
                    {
                        "id": f"trade_{i}_{j}",
                        "marketId": f"0x{'a' * 40}{i:02d}",
                        "outcome": "Yes" if i % 2 == 0 else "No",
                        "side": "buy",
                        "size": str(25),
                        "price": str(0.55 + i * 0.05 + j * 0.01),
                        "fee": str(0.5),
                        "timestamp": (base_time - timedelta(hours=j * 2)).isoformat() + "Z"
                    }
                    for j in range(i + 1)
                ],
                "createdAt": (base_time - timedelta(days=i)).isoformat() + "Z",
                "updatedAt": base_time.isoformat() + "Z"
            }
            positions.append(position)
        
        return {
            "status": "success",
            "data": positions,
            "summary": {
                "totalPositions": count,
                "totalValue": str(sum(float(p["currentValue"]) for p in positions)),
                "totalPnl": str(sum(float(p["totalPnl"]) for p in positions)),
                "totalUnrealizedPnl": str(sum(float(p["unrealizedPnl"]) for p in positions)),
                "totalRealizedPnl": str(sum(float(p["realizedPnl"]) for p in positions))
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }
    
    @staticmethod
    def get_portfolio_response() -> Dict[str, Any]:
        """Generate mock response for GET /portfolio endpoint."""
        return {
            "status": "success",
            "data": {
                "cashBalance": "5000.00",
                "totalValue": "7500.00",
                "totalPnl": "250.00",
                "totalUnrealizedPnl": "150.00",
                "totalRealizedPnl": "100.00",
                "positionsValue": "2500.00",
                "positionsCount": 5,
                "marketsCount": 3,
                "dayChange": "125.50",
                "dayChangePercent": "1.67",
                "weekChange": "340.25",
                "weekChangePercent": "4.53",
                "monthChange": "750.75",
                "monthChangePercent": "10.01",
                "performance": {
                    "totalReturn": "3.33",
                    "annualizedReturn": "15.25",
                    "sharpeRatio": "1.85",
                    "maxDrawdown": "8.75",
                    "winRate": "62.5",
                    "avgWin": "45.25",
                    "avgLoss": "28.50",
                    "profitFactor": "1.58"
                },
                "riskMetrics": {
                    "var95": "125.50",
                    "cvar95": "187.75",
                    "volatility": "12.35",
                    "beta": "0.85",
                    "correlation": "0.72"
                }
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }
    
    @staticmethod
    def error_response(
        status_code: int,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate mock error response."""
        return {
            "status": "error",
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {},
                "statusCode": status_code
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }
    
    @staticmethod
    def rate_limit_response() -> Dict[str, Any]:
        """Generate mock rate limit error response."""
        return HTTPResponseFixtures.error_response(
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            message="Rate limit exceeded, try again later",
            details={
                "retryAfter": 60,
                "limit": 1000,
                "remaining": 0,
                "resetTime": (datetime.now() + timedelta(seconds=60)).isoformat() + "Z"
            }
        )
    
    @staticmethod
    def authentication_error_response() -> Dict[str, Any]:
        """Generate mock authentication error response."""
        return HTTPResponseFixtures.error_response(
            status_code=401,
            error_code="AUTHENTICATION_FAILED",
            message="Invalid API key or signature",
            details={
                "reason": "Invalid signature",
                "timestamp": datetime.now().isoformat() + "Z"
            }
        )
    
    @staticmethod
    def insufficient_balance_response() -> Dict[str, Any]:
        """Generate mock insufficient balance error response."""
        return HTTPResponseFixtures.error_response(
            status_code=400,
            error_code="INSUFFICIENT_BALANCE",
            message="Insufficient balance for this operation",
            details={
                "required": "1000.00",
                "available": "500.00",
                "currency": "USDC"
            }
        )