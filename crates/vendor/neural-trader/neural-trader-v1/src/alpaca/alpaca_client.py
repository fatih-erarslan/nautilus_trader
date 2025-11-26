"""
Alpaca Trading API Client
A comprehensive client for interacting with Alpaca's trading API
Supports both paper and live trading environments
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import pandas as pd
import requests
import websocket
import json
import time
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types supported by Alpaca"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class TimeInForce(Enum):
    """Time in force options"""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pl: float
    unrealized_plpc: float
    side: str
    # Optional fields from API
    cost_basis: Optional[float] = None
    current_price: Optional[float] = None
    asset_id: Optional[str] = None
    asset_class: Optional[str] = None
    exchange: Optional[str] = None
    qty_available: Optional[float] = None
    lastday_price: Optional[float] = None
    change_today: Optional[float] = None

@dataclass
class Order:
    """Represents a trading order"""
    id: str
    symbol: str
    qty: float
    side: str
    order_type: str
    time_in_force: str
    status: str
    filled_qty: float = 0
    filled_avg_price: float = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class AlpacaClient:
    """
    Alpaca Trading API Client

    This client provides a comprehensive interface to Alpaca's trading API,
    including market data, order management, and portfolio tracking.
    """

    def __init__(self, api_key: str = None, secret_key: str = None, base_url: str = None):
        """
        Initialize Alpaca client

        Args:
            api_key: Alpaca API key (defaults to env var ALPACA_API_KEY)
            secret_key: Alpaca secret key (defaults to env var ALPACA_SECRET_KEY)
            base_url: Alpaca base URL (defaults to env var ALPACA_BASE_URL)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.base_url = base_url or os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY")

        # Remove /v2 suffix if present and ensure it's added
        if self.base_url.endswith('/v2'):
            self.base_url = self.base_url[:-3]

        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }

        # WebSocket connection for real-time data
        self.ws = None
        self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        if "paper" in self.base_url:
            self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"

        logger.info(f"Alpaca client initialized with base URL: {self.base_url}")

    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """
        Make HTTP request to Alpaca API

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint
            params: URL parameters
            data: Request body data

        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}/v2/{endpoint}"

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data,
                timeout=30
            )

            if response.status_code == 429:
                # Rate limit hit, wait and retry
                logger.warning("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(method, endpoint, params, data)

            response.raise_for_status()
            return response.json() if response.content else {}

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_account(self) -> Dict:
        """Get account information"""
        return self._make_request('GET', 'account')

    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        response = self._make_request('GET', 'positions')
        return [Position(**pos) for pos in response]

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        try:
            response = self._make_request('GET', f'positions/{symbol}')
            return Position(**response)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str] = OrderType.MARKET,
        time_in_force: Union[TimeInForce, str] = TimeInForce.DAY,
        limit_price: float = None,
        stop_price: float = None,
        trail_price: float = None,
        trail_percent: float = None,
        extended_hours: bool = False
    ) -> Dict:
        """
        Place a trading order

        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: Order side (buy/sell)
            order_type: Order type
            time_in_force: Time in force
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            trail_price: Trail price for trailing stop orders
            trail_percent: Trail percent for trailing stop orders
            extended_hours: Allow extended hours trading

        Returns:
            Order object
        """
        # Handle string or enum inputs
        if isinstance(side, str):
            side_value = side
        else:
            side_value = side.value

        if isinstance(order_type, str):
            type_value = order_type
        else:
            type_value = order_type.value

        if isinstance(time_in_force, str):
            tif_value = time_in_force
        else:
            tif_value = time_in_force.value

        order_data = {
            'symbol': symbol,
            'qty': qty,
            'side': side_value,
            'type': type_value,
            'time_in_force': tif_value,
            'extended_hours': extended_hours
        }

        # Add price parameters based on order type
        order_type_check = order_type if isinstance(order_type, OrderType) else type_value
        if order_type_check in [OrderType.LIMIT, OrderType.STOP_LIMIT] or type_value in ['limit', 'stop_limit']:
            if limit_price is None:
                raise ValueError(f"limit_price required for {type_value} orders")
            order_data['limit_price'] = limit_price

        if order_type_check in [OrderType.STOP, OrderType.STOP_LIMIT] or type_value in ['stop', 'stop_limit']:
            if stop_price is None:
                raise ValueError(f"stop_price required for {type_value} orders")
            order_data['stop_price'] = stop_price

        if order_type_check == OrderType.TRAILING_STOP or type_value == 'trailing_stop':
            if trail_price is not None:
                order_data['trail_price'] = trail_price
            elif trail_percent is not None:
                order_data['trail_percent'] = trail_percent
            else:
                raise ValueError("Either trail_price or trail_percent required for trailing stop orders")

        response = self._make_request('POST', 'orders', data=order_data)
        return response  # Return raw response instead of Order object for now

    def get_orders(self, status: str = 'open', limit: int = 50) -> List[Dict]:
        """Get orders"""
        params = {'status': status, 'limit': limit}
        response = self._make_request('GET', 'orders', params=params)
        return response  # Return raw response for now

    def get_order(self, order_id: str) -> Order:
        """Get specific order"""
        response = self._make_request('GET', f'orders/{order_id}')
        return Order(**response)

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order"""
        self._make_request('DELETE', f'orders/{order_id}')

    def cancel_all_orders(self) -> None:
        """Cancel all open orders"""
        self._make_request('DELETE', 'orders')

    def get_bars(
        self,
        symbol: str,
        timeframe: str = '1Day',
        start: datetime = None,
        end: datetime = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical price bars

        Args:
            symbol: Stock symbol
            timeframe: Timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            start: Start date
            end: End date
            limit: Maximum number of bars

        Returns:
            DataFrame with OHLCV data
        """
        # Use data API endpoint
        data_url = "https://data.alpaca.markets/v2/stocks/bars"

        params = {
            'symbols': symbol,
            'timeframe': timeframe,
            'limit': limit
        }

        if start:
            params['start'] = start.isoformat()
        if end:
            params['end'] = end.isoformat()

        response = requests.get(data_url, headers=self.headers, params=params)
        response.raise_for_status()

        data = response.json()
        if symbol in data.get('bars', {}):
            bars = data['bars'][symbol]
            df = pd.DataFrame(bars)
            df['timestamp'] = pd.to_datetime(df['t'])
            df = df.set_index('timestamp')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            return df[['open', 'high', 'low', 'close', 'volume']]
        else:
            return pd.DataFrame()

    def get_latest_quote(self, symbol: str) -> Dict:
        """Get latest quote for symbol"""
        data_url = f"https://data.alpaca.markets/v2/stocks/quotes/latest"
        params = {'symbols': symbol}

        response = requests.get(data_url, headers=self.headers, params=params)
        response.raise_for_status()

        data = response.json()
        return data.get('quotes', {}).get(symbol, {})

    def get_portfolio_history(self, timeframe: str = '1D', extended_hours: bool = False) -> Dict:
        """Get portfolio history"""
        params = {
            'timeframe': timeframe,
            'extended_hours': extended_hours
        }
        return self._make_request('GET', 'account/portfolio/history', params=params)

    def start_streaming(self, symbols: List[str], on_message_callback) -> None:
        """
        Start WebSocket streaming for real-time data

        Args:
            symbols: List of symbols to stream
            on_message_callback: Function to handle incoming messages
        """
        def on_open(ws):
            logger.info("WebSocket connection opened")
            # Authenticate
            auth_message = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }
            ws.send(json.dumps(auth_message))

            # Subscribe to symbols
            subscribe_message = {
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols,
                "bars": symbols
            }
            ws.send(json.dumps(subscribe_message))

        def on_message(ws, message):
            try:
                data = json.loads(message)
                on_message_callback(data)
            except Exception as e:
                logger.error(f"Error processing message: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")

        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Run in a separate thread
        import threading
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

    def stop_streaming(self) -> None:
        """Stop WebSocket streaming"""
        if self.ws:
            self.ws.close()

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        clock = self._make_request('GET', 'clock')
        return clock.get('is_open', False)

    def get_market_calendar(self, start: datetime = None, end: datetime = None) -> List[Dict]:
        """Get market calendar"""
        params = {}
        if start:
            params['start'] = start.strftime('%Y-%m-%d')
        if end:
            params['end'] = end.strftime('%Y-%m-%d')

        return self._make_request('GET', 'calendar', params=params)

# Example usage and testing
if __name__ == "__main__":
    # Initialize client
    client = AlpacaClient()

    # Test connection
    try:
        account = client.get_account()
        print(f"Account Status: {account.get('status')}")
        print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")

        # Test market data
        bars = client.get_bars('AAPL', timeframe='1Day', limit=10)
        print(f"\nLatest AAPL price: ${bars['close'].iloc[-1]:.2f}")

        # Check positions
        positions = client.get_positions()
        print(f"\nOpen positions: {len(positions)}")

    except Exception as e:
        print(f"Error: {e}")