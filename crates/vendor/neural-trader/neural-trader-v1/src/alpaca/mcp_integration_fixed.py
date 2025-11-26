#!/usr/bin/env python3
"""
Fixed MCP Integration for Alpaca Trading
Forces real API connection with provided credentials
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force real credentials
ALPACA_API_KEY = "PKAJQDPYIZ1S8BHWU7GD"
ALPACA_SECRET_KEY = "zJvREGAi3qQi6zdjhMuemKeUlWhDid78mPIGLkTw"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"


class AlpacaMCPBridge:
    """Bridge between MCP tools and Alpaca API for real trading"""

    def __init__(self):
        """Initialize with real Alpaca credentials"""
        self.demo_mode = False
        self.client = None
        self._initialize_real_client()

    def _initialize_real_client(self):
        """Initialize real Alpaca client"""
        try:
            from alpaca.trading.client import TradingClient

            self.client = TradingClient(
                ALPACA_API_KEY,
                ALPACA_SECRET_KEY,
                paper=True
            )

            # Verify connection
            account = self.client.get_account()
            logger.info(f"✅ Connected to Alpaca Paper Trading")
            logger.info(f"   Account: {account.account_number}")
            logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}")
            self.demo_mode = False

        except Exception as e:
            logger.error(f"Error initializing Alpaca: {e}")
            self.demo_mode = True

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get real portfolio status"""
        try:
            account = self.client.get_account()
            positions = self.client.get_all_positions()

            position_list = []
            total_pnl = 0

            for pos in positions:
                pl = float(pos.unrealized_pl)
                total_pnl += pl
                position_list.append({
                    "symbol": pos.symbol,
                    "quantity": float(pos.qty),
                    "value": float(pos.market_value),
                    "pnl": pl,
                    "pnl_pct": float(pos.unrealized_plpc) * 100
                })

            return {
                "portfolio_value": float(account.portfolio_value),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "positions": position_list,
                "total_pnl": total_pnl,
                "status": "success",
                "demo_mode": False,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return {"status": "error", "error": str(e)}

    def execute_trade(self, symbol: str, action: str, quantity: int,
                     order_type: str = "market", **kwargs) -> Dict[str, Any]:
        """Execute real trade through Alpaca"""
        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            side = OrderSide.BUY if action.lower() == "buy" else OrderSide.SELL

            if order_type == "market":
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            else:
                limit_price = kwargs.get("limit_price")
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )

            order = self.client.submit_order(order_request)

            return {
                "status": "success",
                "demo_mode": False,
                "order_id": order.id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "order_status": order.status,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data"""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest

            data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = data_client.get_stock_latest_quote(request)

            quote_data = quote[symbol]
            return {
                "symbol": symbol,
                "bid": float(quote_data.bid_price),
                "ask": float(quote_data.ask_price),
                "bid_size": int(quote_data.bid_size),
                "ask_size": int(quote_data.ask_size),
                "timestamp": quote_data.timestamp.isoformat(),
                "demo_mode": False
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton instance
_bridge_instance = None

def get_mcp_bridge() -> AlpacaMCPBridge:
    """Get or create the MCP bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = AlpacaMCPBridge()
    return _bridge_instance