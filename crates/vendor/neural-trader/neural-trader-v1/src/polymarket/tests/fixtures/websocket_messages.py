"""
WebSocket message fixtures for Polymarket streaming data testing.

Contains mock WebSocket messages for order updates, market data streams,
and trade notifications to test real-time functionality.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


class WebSocketMessageFixtures:
    """Collection of WebSocket message fixtures for real-time data testing."""
    
    @staticmethod
    def connection_established() -> str:
        """WebSocket connection established message."""
        return json.dumps({
            "type": "connection",
            "status": "connected",
            "connectionId": "conn_123456789",
            "timestamp": datetime.now().isoformat() + "Z",
            "supportedChannels": [
                "markets",
                "orderbook",
                "trades",
                "orders",
                "positions"
            ]
        })
    
    @staticmethod
    def subscription_confirmed(channel: str, symbol: Optional[str] = None) -> str:
        """Subscription confirmation message."""
        return json.dumps({
            "type": "subscription",
            "status": "subscribed",
            "channel": channel,
            "symbol": symbol,
            "subscriptionId": f"sub_{channel}_{symbol or 'all'}",
            "timestamp": datetime.now().isoformat() + "Z"
        })
    
    @staticmethod
    def market_update(market_id: str, price_changes: Dict[str, float]) -> str:
        """Market price update message."""
        return json.dumps({
            "type": "market_update",
            "channel": "markets",
            "data": {
                "marketId": market_id,
                "priceUpdates": [
                    {
                        "outcome": outcome,
                        "price": str(price),
                        "priceChange": str(price - 0.5),  # Assuming base price of 0.5
                        "priceChangePercent": str((price - 0.5) / 0.5 * 100),
                        "volume24h": str(1000 + abs(price - 0.5) * 10000),
                        "timestamp": datetime.now().isoformat() + "Z"
                    }
                    for outcome, price in price_changes.items()
                ],
                "lastUpdate": datetime.now().isoformat() + "Z"
            },
            "timestamp": datetime.now().isoformat() + "Z"
        })
    
    @staticmethod
    def orderbook_update(
        market_id: str,
        outcome: str,
        bids: List[Dict[str, float]],
        asks: List[Dict[str, float]]
    ) -> str:
        """Order book update message."""
        return json.dumps({
            "type": "orderbook_update",
            "channel": "orderbook",
            "data": {
                "marketId": market_id,
                "outcome": outcome,
                "bids": [
                    {
                        "price": str(bid["price"]),
                        "size": str(bid["size"]),
                        "timestamp": datetime.now().isoformat() + "Z"
                    }
                    for bid in bids
                ],
                "asks": [
                    {
                        "price": str(ask["price"]),
                        "size": str(ask["size"]),
                        "timestamp": datetime.now().isoformat() + "Z"
                    }
                    for ask in asks
                ],
                "spread": str(min(ask["price"] for ask in asks) - max(bid["price"] for bid in bids)) if bids and asks else "0",
                "midPrice": str((min(ask["price"] for ask in asks) + max(bid["price"] for bid in bids)) / 2) if bids and asks else "0",
                "timestamp": datetime.now().isoformat() + "Z"
            },
            "timestamp": datetime.now().isoformat() + "Z"
        })
    
    @staticmethod
    def trade_executed(
        trade_id: str,
        market_id: str,
        outcome: str,
        price: float,
        size: float,
        side: str
    ) -> str:
        """Trade execution message."""
        return json.dumps({
            "type": "trade",
            "channel": "trades",
            "data": {
                "id": trade_id,
                "marketId": market_id,
                "outcome": outcome,
                "price": str(price),
                "size": str(size),
                "side": side,
                "value": str(price * size),
                "fee": str(price * size * 0.02),  # 2% fee
                "timestamp": datetime.now().isoformat() + "Z",
                "blockNumber": 12345678,
                "transactionHash": "0x" + "a" * 64
            },
            "timestamp": datetime.now().isoformat() + "Z"
        })
    
    @staticmethod
    def order_update(
        order_id: str,
        status: str,
        filled: Optional[float] = None,
        price: Optional[float] = None
    ) -> str:
        """Order status update message."""
        data = {
            "type": "order_update",
            "channel": "orders",
            "data": {
                "id": order_id,
                "status": status,
                "updatedAt": datetime.now().isoformat() + "Z"
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }
        
        if filled is not None:
            data["data"]["filled"] = str(filled)
            data["data"]["fillPercentage"] = str(filled * 100) if filled <= 1.0 else str(100)
        
        if price is not None:
            data["data"]["averageFillPrice"] = str(price)
        
        return json.dumps(data)
    
    @staticmethod
    def position_update(
        market_id: str,
        outcome: str,
        size: float,
        pnl: float,
        current_price: float
    ) -> str:
        """Position update message."""
        return json.dumps({
            "type": "position_update",
            "channel": "positions",
            "data": {
                "marketId": market_id,
                "outcome": outcome,
                "size": str(size),
                "currentPrice": str(current_price),
                "currentValue": str(size * current_price),
                "unrealizedPnl": str(pnl),
                "pnlPercentage": str(pnl / (size * current_price) * 100) if size * current_price != 0 else "0",
                "updatedAt": datetime.now().isoformat() + "Z"
            },
            "timestamp": datetime.now().isoformat() + "Z"
        })
    
    @staticmethod
    def market_closed(market_id: str, winning_outcome: str) -> str:
        """Market resolution message."""
        return json.dumps({
            "type": "market_closed",
            "channel": "markets",
            "data": {
                "marketId": market_id,
                "status": "resolved",
                "winningOutcome": winning_outcome,
                "resolutionTime": datetime.now().isoformat() + "Z",
                "finalPrices": {
                    winning_outcome: "1.00",
                    # Assume binary market with opposite outcome
                    "Yes" if winning_outcome == "No" else "No": "0.00"
                },
                "payouts": {
                    "totalPayout": "100000.00",
                    "winnersCount": 125,
                    "avgPayout": "800.00"
                }
            },
            "timestamp": datetime.now().isoformat() + "Z"
        })
    
    @staticmethod
    def heartbeat() -> str:
        """WebSocket heartbeat/ping message."""
        return json.dumps({
            "type": "heartbeat",
            "timestamp": datetime.now().isoformat() + "Z"
        })
    
    @staticmethod
    def error_message(error_code: str, message: str) -> str:
        """WebSocket error message."""
        return json.dumps({
            "type": "error",
            "error": {
                "code": error_code,
                "message": message,
                "timestamp": datetime.now().isoformat() + "Z"
            }
        })
    
    @staticmethod
    def rate_limit_warning() -> str:
        """Rate limit warning message."""
        return json.dumps({
            "type": "warning",
            "warning": {
                "code": "RATE_LIMIT_WARNING",
                "message": "Approaching rate limit, reduce request frequency",
                "details": {
                    "currentRate": 95,
                    "limit": 100,
                    "resetTime": (datetime.now() + timedelta(seconds=60)).isoformat() + "Z"
                }
            },
            "timestamp": datetime.now().isoformat() + "Z"
        })
    
    @staticmethod
    def connection_lost() -> str:
        """Connection lost message."""
        return json.dumps({
            "type": "connection",
            "status": "disconnected",
            "reason": "connection_lost",
            "reconnectIn": 5,
            "timestamp": datetime.now().isoformat() + "Z"
        })
    
    @staticmethod
    def bulk_market_updates(market_updates: List[Dict[str, Any]]) -> str:
        """Bulk market updates message."""
        return json.dumps({
            "type": "bulk_update",
            "channel": "markets",
            "data": {
                "updates": [
                    {
                        "marketId": update["market_id"],
                        "priceUpdates": [
                            {
                                "outcome": outcome,
                                "price": str(price),
                                "volume": str(update.get("volume", 1000)),
                                "timestamp": datetime.now().isoformat() + "Z"
                            }
                            for outcome, price in update["prices"].items()
                        ]
                    }
                    for update in market_updates
                ],
                "updateCount": len(market_updates)
            },
            "timestamp": datetime.now().isoformat() + "Z"
        })
    
    @staticmethod
    def order_fill_notification(
        order_id: str,
        fill_size: float,
        fill_price: float,
        remaining: float
    ) -> str:
        """Order fill notification message."""
        return json.dumps({
            "type": "order_fill",
            "channel": "orders",
            "data": {
                "orderId": order_id,
                "fillId": f"fill_{int(datetime.now().timestamp())}",
                "fillSize": str(fill_size),
                "fillPrice": str(fill_price),
                "fillValue": str(fill_size * fill_price),
                "remainingSize": str(remaining),
                "status": "filled" if remaining == 0 else "partial",
                "fee": str(fill_size * fill_price * 0.02),
                "timestamp": datetime.now().isoformat() + "Z"
            },
            "timestamp": datetime.now().isoformat() + "Z"
        })
    
    @staticmethod
    def market_volatility_alert(market_id: str, volatility: float) -> str:
        """High volatility alert message."""
        return json.dumps({
            "type": "alert",
            "alertType": "volatility",
            "data": {
                "marketId": market_id,
                "volatility": str(volatility),
                "threshold": "0.10",
                "severity": "high" if volatility > 0.15 else "medium",
                "message": f"High volatility detected: {volatility:.2%}",
                "recommendations": [
                    "Consider adjusting position sizes",
                    "Monitor market closely",
                    "Review stop-loss orders"
                ]
            },
            "timestamp": datetime.now().isoformat() + "Z"
        })
    
    @staticmethod
    def generate_message_sequence(
        message_types: List[str],
        market_id: str = "0x" + "a" * 40,
        interval_seconds: int = 1
    ) -> List[str]:
        """Generate a sequence of WebSocket messages for testing."""
        messages = []
        base_time = datetime.now()
        
        for i, msg_type in enumerate(message_types):
            # Adjust timestamp for sequence
            current_time = base_time + timedelta(seconds=i * interval_seconds)
            
            if msg_type == "market_update":
                messages.append(WebSocketMessageFixtures.market_update(
                    market_id, 
                    {"Yes": 0.6 + i * 0.01, "No": 0.4 - i * 0.01}
                ))
            elif msg_type == "trade":
                messages.append(WebSocketMessageFixtures.trade_executed(
                    f"trade_{i}",
                    market_id,
                    "Yes",
                    0.65 + i * 0.005,
                    100 + i * 10,
                    "buy"
                ))
            elif msg_type == "order_update":
                messages.append(WebSocketMessageFixtures.order_update(
                    f"order_{i}",
                    "partial",
                    filled=0.5 + i * 0.1,
                    price=0.64 + i * 0.01
                ))
            elif msg_type == "heartbeat":
                messages.append(WebSocketMessageFixtures.heartbeat())
        
        return messages