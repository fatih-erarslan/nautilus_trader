#!/usr/bin/env python3
"""
MCP Integration for Alpaca Trading
Provides seamless integration between MCP tools and Alpaca API
Automatically detects and uses real credentials when available
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from root .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path, override=True)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca.alpaca_client import AlpacaClient
from alpaca.trading_strategies import TradingBot, MomentumStrategy, MeanReversionStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlpacaMCPBridge:
    """Bridge between MCP tools and Alpaca API for real trading"""

    def __init__(self):
        """Initialize the bridge with Alpaca credentials from environment"""
        self.client = None
        self.trading_bot = None

        # Check for real credentials
        self.has_credentials = self._check_credentials()

        if self.has_credentials:
            self.demo_mode = False
            self._initialize_real_client()
        else:
            self.demo_mode = True
            logger.warning("Alpaca credentials not found - running in DEMO mode")

    def _check_credentials(self) -> bool:
        """Check if valid Alpaca credentials are present"""
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')

        # Check if both keys are present and valid
        if api_key and secret_key:
            # Skip placeholder values
            if 'your-' in api_key.lower() or 'your-' in secret_key.lower():
                logger.debug("Found placeholder credentials, skipping")
                return False

            # Check for minimum length (real keys are longer)
            if len(api_key) >= 15 and len(secret_key) >= 15:
                logger.info(f"Found valid Alpaca credentials (API key: {api_key[:10]}...)")
                return True

        logger.debug(f"Invalid credentials - API key: {api_key[:10] if api_key else 'None'}...")
        return False

    def _initialize_real_client(self):
        """Initialize real Alpaca client with credentials"""
        try:
            # Get credentials from environment
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')

            logger.info(f"Initializing Alpaca client with base URL: {base_url}")

            self.client = AlpacaClient(
                api_key=api_key,
                secret_key=secret_key,
                base_url=base_url
            )

            # Verify connection
            account = self.client.get_account()
            if account:
                logger.info(f"✅ Connected to Alpaca account: {account.get('account_number')}")
                logger.info(f"   Buying Power: ${account.get('buying_power')}")
                self.demo_mode = False

                # Initialize trading bot
                self.trading_bot = TradingBot(self.client)
            else:
                logger.error("Failed to connect to Alpaca")
                self.demo_mode = True
        except Exception as e:
            logger.error(f"Error initializing Alpaca client: {e}")
            self.demo_mode = True

    def execute_trade(self, symbol: str, action: str, quantity: int,
                     strategy: str = "market", **kwargs) -> Dict[str, Any]:
        """
        Execute a trade through MCP
        Automatically uses real trading when credentials are available
        """
        if self.demo_mode:
            return self._execute_demo_trade(symbol, action, quantity, strategy, **kwargs)
        else:
            return self._execute_real_trade(symbol, action, quantity, strategy, **kwargs)

    def _execute_real_trade(self, symbol: str, action: str, quantity: int,
                           strategy: str, **kwargs) -> Dict[str, Any]:
        """Execute real trade through Alpaca API"""
        try:
            # Place order
            order = self.client.place_order(
                symbol=symbol,
                qty=quantity,
                side=action,  # 'buy' or 'sell'
                order_type=kwargs.get('order_type', 'market'),
                time_in_force=kwargs.get('time_in_force', 'day'),
                limit_price=kwargs.get('limit_price'),
                stop_price=kwargs.get('stop_price')
            )

            return {
                "status": "success",
                "demo_mode": False,
                "order_id": order.get('id'),
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "order_status": order.get('status'),
                "submitted_at": order.get('submitted_at'),
                "message": f"Real order placed: {order.get('id')}"
            }
        except Exception as e:
            logger.error(f"Error executing real trade: {e}")
            return {
                "status": "error",
                "demo_mode": False,
                "error": str(e)
            }

    def _execute_demo_trade(self, symbol: str, action: str, quantity: int,
                           strategy: str, **kwargs) -> Dict[str, Any]:
        """Execute demo trade for testing"""
        import random
        demo_price = round(random.uniform(100, 500), 2)

        return {
            "status": "success",
            "demo_mode": True,
            "order_id": f"DEMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "demo_price": demo_price,
            "message": "Demo trade executed (no real order placed)"
        }

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get portfolio status from real account or demo"""
        if self.demo_mode:
            return self._get_demo_portfolio()

        try:
            # Use official Alpaca SDK
            import alpaca_trade_api as tradeapi

            trading_client = tradeapi.REST(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                base_url='https://paper-api.alpaca.markets'
            )

            account = trading_client.get_account()
            positions = trading_client.list_positions()

            position_list = []
            for pos in positions:
                position_list.append({
                    "symbol": pos.symbol,
                    "quantity": float(pos.qty),
                    "value": float(pos.market_value),
                    "pnl": float(pos.unrealized_pl),
                    "pnl_pct": float(pos.unrealized_plpc) * 100
                })

            return {
                "status": "success",
                "demo_mode": False,
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "positions": position_list,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {"status": "error", "error": str(e), "demo_mode": False}

    def _get_demo_portfolio(self) -> Dict[str, Any]:
        """Get demo portfolio for testing"""
        return {
            "status": "success",
            "demo_mode": True,
            "account_number": "DEMO_ACCOUNT",
            "buying_power": 100000.0,
            "cash": 50000.0,
            "portfolio_value": 100000.0,
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "avg_price": 150.0, "current_price": 155.0, "unrealized_pl": 500.0},
                {"symbol": "TSLA", "quantity": 50, "avg_price": 240.0, "current_price": 245.0, "unrealized_pl": 250.0}
            ]
        }

    def run_strategy(self, strategy_name: str, symbol: str, **params) -> Dict[str, Any]:
        """Run a specific trading strategy"""
        if self.demo_mode:
            return {
                "status": "success",
                "demo_mode": True,
                "strategy": strategy_name,
                "symbol": symbol,
                "message": "Demo strategy execution"
            }

        try:
            # Map strategy names to strategy classes
            strategies = {
                "momentum": MomentumStrategy,
                "mean_reversion": MeanReversionStrategy,
            }

            if strategy_name.lower() in strategies:
                strategy_class = strategies[strategy_name.lower()]
                strategy = strategy_class(self.client)

                # Generate signals
                signal = strategy.generate_signal(symbol)

                # Execute based on signal
                if signal == 'BUY':
                    order = self.client.place_order(
                        symbol=symbol,
                        qty=params.get('quantity', 1),
                        side='buy',
                        order_type='market',
                        time_in_force='day'
                    )
                    return {
                        "status": "success",
                        "demo_mode": False,
                        "strategy": strategy_name,
                        "signal": signal,
                        "order_id": order.get('id'),
                        "message": f"Strategy {strategy_name} executed BUY order"
                    }
                elif signal == 'SELL':
                    # Check if we have position to sell
                    positions = self.client.get_positions()
                    has_position = any(p.get('symbol') == symbol for p in positions)

                    if has_position:
                        order = self.client.place_order(
                            symbol=symbol,
                            qty=params.get('quantity', 1),
                            side='sell',
                            order_type='market',
                            time_in_force='day'
                        )
                        return {
                            "status": "success",
                            "demo_mode": False,
                            "strategy": strategy_name,
                            "signal": signal,
                            "order_id": order.get('id'),
                            "message": f"Strategy {strategy_name} executed SELL order"
                        }
                    else:
                        return {
                            "status": "success",
                            "demo_mode": False,
                            "strategy": strategy_name,
                            "signal": signal,
                            "message": f"Strategy generated SELL signal but no position to sell"
                        }
                else:
                    return {
                        "status": "success",
                        "demo_mode": False,
                        "strategy": strategy_name,
                        "signal": signal,
                        "message": f"Strategy {strategy_name} signal: {signal} (no action taken)"
                    }
            else:
                return {
                    "status": "error",
                    "demo_mode": False,
                    "error": f"Unknown strategy: {strategy_name}"
                }
        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            return {
                "status": "error",
                "demo_mode": False,
                "error": str(e)
            }

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data"""
        if self.demo_mode:
            import random
            return {
                "status": "success",
                "demo_mode": True,
                "symbol": symbol,
                "price": round(random.uniform(100, 500), 2),
                "volume": random.randint(1000000, 10000000),
                "change": round(random.uniform(-5, 5), 2)
            }

        try:
            quote = self.client.get_latest_quote(symbol)
            bars = self.client.get_bars(symbol, limit=1)

            current_price = float(quote.get('ap', 0)) if quote else 0

            return {
                "status": "success",
                "demo_mode": False,
                "symbol": symbol,
                "price": current_price,
                "bid": float(quote.get('bp', 0)) if quote else 0,
                "ask": float(quote.get('ap', 0)) if quote else 0,
                "volume": int(bars.iloc[-1]['volume']) if not bars.empty else 0,
                "timestamp": quote.get('t') if quote else None
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {
                "status": "error",
                "demo_mode": False,
                "error": str(e)
            }


# Global bridge instance for MCP tools
_bridge_instance = None


def get_mcp_bridge() -> AlpacaMCPBridge:
    """Get or create the MCP bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = AlpacaMCPBridge()
    return _bridge_instance


# MCP Tool Functions
def mcp_execute_trade(symbol: str, action: str, quantity: int, strategy: str = "market", **kwargs):
    """MCP tool wrapper for executing trades"""
    bridge = get_mcp_bridge()
    return bridge.execute_trade(symbol, action, quantity, strategy, **kwargs)


def mcp_get_portfolio_status():
    """MCP tool wrapper for getting portfolio status"""
    bridge = get_mcp_bridge()
    return bridge.get_portfolio_status()


def mcp_run_strategy(strategy_name: str, symbol: str, **params):
    """MCP tool wrapper for running strategies"""
    bridge = get_mcp_bridge()
    return bridge.run_strategy(strategy_name, symbol, **params)


def mcp_get_market_data(symbol: str):
    """MCP tool wrapper for getting market data"""
    bridge = get_mcp_bridge()
    return bridge.get_market_data(symbol)


if __name__ == "__main__":
    # Test the integration
    print("🔍 Testing Alpaca MCP Integration")
    print("=" * 60)

    bridge = get_mcp_bridge()

    if bridge.demo_mode:
        print("⚠️  Running in DEMO mode (no credentials found)")
    else:
        print("✅ Running in REAL mode with Alpaca credentials")

    # Test portfolio status
    portfolio = bridge.get_portfolio_status()
    print(f"\n📊 Portfolio Status:")
    print(f"   Demo Mode: {portfolio.get('demo_mode')}")
    print(f"   Account: {portfolio.get('account_number')}")
    print(f"   Buying Power: ${portfolio.get('buying_power')}")
    print(f"   Portfolio Value: ${portfolio.get('portfolio_value')}")

    # Test market data
    market_data = bridge.get_market_data("AAPL")
    print(f"\n📈 Market Data for AAPL:")
    print(f"   Price: ${market_data.get('price')}")
    print(f"   Demo Mode: {market_data.get('demo_mode')}")