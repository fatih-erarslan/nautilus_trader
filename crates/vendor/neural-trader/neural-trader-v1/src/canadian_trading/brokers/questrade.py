"""
Questrade API Integration Module

Production-ready implementation with OAuth2 authentication, REST API client,
market data streaming, order management, and comprehensive error handling.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlencode
import aiohttp
from aiohttp import ClientSession, ClientTimeout
import backoff

logger = logging.getLogger(__name__)


class QuestradeAPIError(Exception):
    """Custom exception for Questrade API errors"""
    def __init__(self, message: str, code: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class RateLimiter:
    """Rate limiter for API requests"""
    def __init__(self, calls_per_second: int = 30):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        async with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call
            
            if time_since_last_call < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last_call)
            
            self.last_call = time.time()


class QuestradeAPI:
    """
    Questrade API client with OAuth2 authentication and comprehensive features
    """
    
    # API endpoints
    AUTH_URL = "https://login.questrade.com/oauth2/token"
    API_URL_PATTERN = "https://api{server}.iq.questrade.com/v1/"
    
    # Rate limits (per Questrade documentation)
    MAX_REQUESTS_PER_SECOND = 30
    
    def __init__(self, 
                 refresh_token: Optional[str] = None,
                 access_token: Optional[str] = None,
                 api_server: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3):
        """
        Initialize Questrade API client
        
        Args:
            refresh_token: OAuth2 refresh token
            access_token: Current access token (optional)
            api_server: API server URL (optional)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.refresh_token = refresh_token
        self.access_token = access_token
        self.api_server = api_server
        self.timeout = ClientTimeout(total=timeout)
        self.max_retries = max_retries
        
        # Connection pooling
        self.session: Optional[ClientSession] = None
        self.connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Per-host connection limit
            ttl_dns_cache=300  # DNS cache timeout
        )
        
        # Rate limiting
        self.rate_limiter = RateLimiter(calls_per_second=self.MAX_REQUESTS_PER_SECOND)
        
        # Token management
        self.token_expiry: Optional[datetime] = None
        self._auth_lock = asyncio.Lock()
        
        # Streaming
        self.streaming_session: Optional[ClientSession] = None
        self.streaming_callbacks = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize the API client session"""
        if not self.session:
            self.session = ClientSession(
                connector=self.connector,
                timeout=self.timeout
            )
        
        # Authenticate if we have a refresh token but no access token
        if self.refresh_token and not self.access_token:
            await self.authenticate()
    
    async def close(self):
        """Close all sessions and cleanup resources"""
        if self.session:
            await self.session.close()
        if self.streaming_session:
            await self.streaming_session.close()
        await self.connector.close()
    
    async def authenticate(self):
        """
        Authenticate using OAuth2 refresh token
        """
        async with self._auth_lock:
            if not self.refresh_token:
                raise QuestradeAPIError("No refresh token provided")
            
            params = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token
            }
            
            try:
                async with ClientSession() as session:
                    async with session.post(
                        self.AUTH_URL,
                        params=params,
                        timeout=self.timeout
                    ) as response:
                        if response.status != 200:
                            error_data = await response.json()
                            raise QuestradeAPIError(
                                f"Authentication failed: {error_data}",
                                code=response.status,
                                details=error_data
                            )
                        
                        auth_data = await response.json()
                        
                        # Update tokens and server info
                        self.access_token = auth_data["access_token"]
                        self.refresh_token = auth_data["refresh_token"]
                        self.api_server = auth_data["api_server"]
                        
                        # Calculate token expiry
                        expires_in = auth_data.get("expires_in", 1800)  # Default 30 min
                        self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                        
                        logger.info("Successfully authenticated with Questrade API")
                        
                        return auth_data
                        
            except aiohttp.ClientError as e:
                raise QuestradeAPIError(f"Network error during authentication: {str(e)}")
    
    async def _ensure_authenticated(self):
        """Ensure we have a valid access token"""
        if not self.access_token or (self.token_expiry and datetime.now() >= self.token_expiry):
            await self.authenticate()
    
    def _get_api_url(self, endpoint: str) -> str:
        """Get full API URL for an endpoint"""
        if not self.api_server:
            raise QuestradeAPIError("API server not set. Please authenticate first.")
        
        base_url = self.API_URL_PATTERN.format(server=self.api_server)
        return f"{base_url}{endpoint.lstrip('/')}"
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, QuestradeAPIError),
        max_tries=3,
        max_time=60
    )
    async def _make_request(self,
                          method: str,
                          endpoint: str,
                          params: Optional[Dict] = None,
                          data: Optional[Dict] = None,
                          headers: Optional[Dict] = None) -> Dict:
        """
        Make an authenticated API request with retry logic
        """
        await self._ensure_authenticated()
        await self.rate_limiter.acquire()
        
        if not self.session:
            await self.initialize()
        
        url = self._get_api_url(endpoint)
        
        request_headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        if headers:
            request_headers.update(headers)
        
        try:
            async with self.session.request(
                method,
                url,
                params=params,
                json=data,
                headers=request_headers
            ) as response:
                response_data = await response.json()
                
                if response.status >= 400:
                    error_msg = response_data.get("message", "Unknown error")
                    raise QuestradeAPIError(
                        f"API request failed: {error_msg}",
                        code=response.status,
                        details=response_data
                    )
                
                return response_data
                
        except aiohttp.ClientError as e:
            raise QuestradeAPIError(f"Network error: {str(e)}")
    
    # Account Management Methods
    
    async def get_accounts(self) -> List[Dict]:
        """Get all accounts associated with the user"""
        response = await self._make_request("GET", "/accounts")
        return response.get("accounts", [])
    
    async def get_account_positions(self, account_id: str) -> List[Dict]:
        """Get positions for a specific account"""
        response = await self._make_request("GET", f"/accounts/{account_id}/positions")
        return response.get("positions", [])
    
    async def get_account_balances(self, account_id: str) -> Dict:
        """Get account balances"""
        response = await self._make_request("GET", f"/accounts/{account_id}/balances")
        return response.get("perCurrencyBalances", [])
    
    async def get_account_activities(self, 
                                   account_id: str,
                                   start_time: datetime,
                                   end_time: datetime) -> List[Dict]:
        """Get account activities within a time range"""
        params = {
            "startTime": start_time.isoformat() + "-05:00",  # EST timezone
            "endTime": end_time.isoformat() + "-05:00"
        }
        response = await self._make_request(
            "GET", 
            f"/accounts/{account_id}/activities",
            params=params
        )
        return response.get("activities", [])
    
    async def get_account_orders(self,
                               account_id: str,
                               state_filter: Optional[str] = None,
                               ids: Optional[List[int]] = None) -> List[Dict]:
        """
        Get orders for an account
        
        Args:
            account_id: Account identifier
            state_filter: Filter by state (All, Open, Closed)
            ids: List of specific order IDs to retrieve
        """
        params = {}
        if state_filter:
            params["stateFilter"] = state_filter
        if ids:
            params["ids"] = ",".join(map(str, ids))
        
        response = await self._make_request(
            "GET",
            f"/accounts/{account_id}/orders",
            params=params
        )
        return response.get("orders", [])
    
    # Market Data Methods
    
    async def search_symbols(self, prefix: str, offset: int = 0) -> List[Dict]:
        """Search for symbols by prefix"""
        params = {
            "prefix": prefix,
            "offset": offset
        }
        response = await self._make_request("GET", "/symbols/search", params=params)
        return response.get("symbols", [])
    
    async def get_symbol(self, symbol_id: Union[str, int]) -> Dict:
        """Get detailed information about a symbol"""
        response = await self._make_request("GET", f"/symbols/{symbol_id}")
        return response.get("symbols", [{}])[0]
    
    async def get_quotes(self, ids: List[Union[str, int]]) -> List[Dict]:
        """Get real-time quotes for multiple symbols"""
        params = {"ids": ",".join(map(str, ids))}
        response = await self._make_request("GET", "/markets/quotes", params=params)
        return response.get("quotes", [])
    
    async def get_candles(self,
                         symbol_id: Union[str, int],
                         start_time: datetime,
                         end_time: datetime,
                         interval: str = "OneDay") -> List[Dict]:
        """
        Get historical candles for a symbol
        
        Args:
            symbol_id: Symbol identifier
            start_time: Start of time range
            end_time: End of time range
            interval: Candle interval (OneMinute, FiveMinutes, etc.)
        """
        params = {
            "startTime": start_time.isoformat() + "-05:00",
            "endTime": end_time.isoformat() + "-05:00",
            "interval": interval
        }
        response = await self._make_request(
            "GET",
            f"/markets/candles/{symbol_id}",
            params=params
        )
        return response.get("candles", [])
    
    async def get_option_chain(self, symbol_id: Union[str, int]) -> List[Dict]:
        """Get option chain for a symbol"""
        response = await self._make_request("GET", f"/symbols/{symbol_id}/options")
        return response.get("optionChain", [])
    
    # Order Management Methods
    
    async def place_order(self,
                         account_id: str,
                         symbol_id: Union[str, int],
                         order_type: str,
                         action: str,
                         quantity: int,
                         limit_price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: str = "Day",
                         primary_route: str = "AUTO",
                         secondary_route: str = "AUTO") -> Dict:
        """
        Place a new order
        
        Args:
            account_id: Account identifier
            symbol_id: Symbol identifier
            order_type: Order type (Market, Limit, Stop, StopLimit)
            action: Buy or Sell
            quantity: Number of shares
            limit_price: Limit price (required for Limit orders)
            stop_price: Stop price (required for Stop orders)
            time_in_force: Order duration (Day, GTC, etc.)
            primary_route: Primary order route
            secondary_route: Secondary order route
        """
        order_data = {
            "symbolId": symbol_id,
            "orderType": order_type,
            "action": action,
            "quantity": quantity,
            "timeInForce": time_in_force,
            "primaryRoute": primary_route,
            "secondaryRoute": secondary_route
        }
        
        if limit_price is not None:
            order_data["limitPrice"] = limit_price
        if stop_price is not None:
            order_data["stopPrice"] = stop_price
        
        response = await self._make_request(
            "POST",
            f"/accounts/{account_id}/orders",
            data=order_data
        )
        return response
    
    async def modify_order(self,
                          account_id: str,
                          order_id: int,
                          quantity: Optional[int] = None,
                          limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None) -> Dict:
        """
        Modify an existing order
        
        Args:
            account_id: Account identifier
            order_id: Order identifier
            quantity: New quantity (optional)
            limit_price: New limit price (optional)
            stop_price: New stop price (optional)
        """
        order_data = {}
        if quantity is not None:
            order_data["quantity"] = quantity
        if limit_price is not None:
            order_data["limitPrice"] = limit_price
        if stop_price is not None:
            order_data["stopPrice"] = stop_price
        
        response = await self._make_request(
            "PUT",
            f"/accounts/{account_id}/orders/{order_id}",
            data=order_data
        )
        return response
    
    async def cancel_order(self, account_id: str, order_id: int) -> Dict:
        """Cancel an existing order"""
        response = await self._make_request(
            "DELETE",
            f"/accounts/{account_id}/orders/{order_id}"
        )
        return response
    
    # Market Data Streaming
    
    async def stream_quotes(self,
                           symbol_ids: List[Union[str, int]],
                           callback: callable):
        """
        Stream real-time quotes for symbols
        
        Args:
            symbol_ids: List of symbol IDs to stream
            callback: Async function to call with quote updates
        """
        if not self.streaming_session:
            self.streaming_session = ClientSession()
        
        # Register callback
        for symbol_id in symbol_ids:
            self.streaming_callbacks[symbol_id] = callback
        
        # Get streaming port
        response = await self._make_request("GET", "/markets/quotes/options")
        stream_port = response.get("streamPort")
        
        if not stream_port:
            raise QuestradeAPIError("Unable to get streaming port")
        
        # Build streaming URL
        stream_url = f"https://stream{self.api_server}.questrade.com:{stream_port}/v1/markets/quotes"
        params = {"ids": ",".join(map(str, symbol_ids)), "mode": "streaming"}
        
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        
        try:
            async with self.streaming_session.get(
                stream_url,
                params=params,
                headers=headers
            ) as response:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            quote = data.get("quote", {})
                            symbol_id = quote.get("symbolId")
                            
                            if symbol_id in self.streaming_callbacks:
                                await self.streaming_callbacks[symbol_id](quote)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse streaming data: {line}")
                        except Exception as e:
                            logger.error(f"Error in streaming callback: {e}")
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise QuestradeAPIError(f"Streaming failed: {str(e)}")
    
    async def stop_streaming(self, symbol_ids: Optional[List[Union[str, int]]] = None):
        """Stop streaming for specific symbols or all symbols"""
        if symbol_ids:
            for symbol_id in symbol_ids:
                self.streaming_callbacks.pop(symbol_id, None)
        else:
            self.streaming_callbacks.clear()
        
        if not self.streaming_callbacks and self.streaming_session:
            await self.streaming_session.close()
            self.streaming_session = None
    
    # Utility Methods
    
    async def get_server_time(self) -> datetime:
        """Get current server time"""
        response = await self._make_request("GET", "/time")
        time_str = response.get("time")
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    
    async def get_markets(self) -> List[Dict]:
        """Get market operating hours"""
        response = await self._make_request("GET", "/markets")
        return response.get("markets", [])
    
    async def validate_order(self,
                           account_id: str,
                           order_data: Dict,
                           impact: bool = False) -> Dict:
        """
        Validate an order before placing it
        
        Args:
            account_id: Account identifier
            order_data: Order details
            impact: Whether to include impact estimates
        """
        endpoint = f"/accounts/{account_id}/orders"
        if impact:
            endpoint += "/impact"
        
        # Add validate flag
        params = {"validate": "true"}
        
        response = await self._make_request(
            "POST",
            endpoint,
            params=params,
            data=order_data
        )
        return response


class QuestradeDataFeed:
    """
    High-level interface for Questrade market data
    """
    
    def __init__(self, api: QuestradeAPI):
        self.api = api
        self._symbol_cache = {}
        self._cache_expiry = {}
        self.cache_ttl = 3600  # 1 hour cache
    
    async def get_symbol_id(self, symbol: str) -> Optional[int]:
        """Get symbol ID from symbol name with caching"""
        # Check cache
        if symbol in self._symbol_cache:
            if time.time() < self._cache_expiry.get(symbol, 0):
                return self._symbol_cache[symbol]
        
        # Search for symbol
        results = await self.api.search_symbols(symbol)
        
        for result in results:
            if result.get("symbol") == symbol:
                symbol_id = result.get("symbolId")
                # Cache result
                self._symbol_cache[symbol] = symbol_id
                self._cache_expiry[symbol] = time.time() + self.cache_ttl
                return symbol_id
        
        return None
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote for a symbol"""
        symbol_id = await self.get_symbol_id(symbol)
        if not symbol_id:
            return None
        
        quotes = await self.api.get_quotes([symbol_id])
        return quotes[0] if quotes else None
    
    async def get_quotes_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols"""
        symbol_ids = []
        symbol_map = {}
        
        for symbol in symbols:
            symbol_id = await self.get_symbol_id(symbol)
            if symbol_id:
                symbol_ids.append(symbol_id)
                symbol_map[symbol_id] = symbol
        
        if not symbol_ids:
            return {}
        
        quotes = await self.api.get_quotes(symbol_ids)
        
        result = {}
        for quote in quotes:
            symbol_id = quote.get("symbolId")
            if symbol_id in symbol_map:
                result[symbol_map[symbol_id]] = quote
        
        return result
    
    async def get_historical_data(self,
                                symbol: str,
                                start_date: datetime,
                                end_date: datetime,
                                interval: str = "OneDay") -> List[Dict]:
        """Get historical candle data for a symbol"""
        symbol_id = await self.get_symbol_id(symbol)
        if not symbol_id:
            return []
        
        return await self.api.get_candles(symbol_id, start_date, end_date, interval)
    
    async def stream_quotes(self, symbols: List[str], callback: callable):
        """Stream real-time quotes for multiple symbols"""
        symbol_ids = []
        
        for symbol in symbols:
            symbol_id = await self.get_symbol_id(symbol)
            if symbol_id:
                symbol_ids.append(symbol_id)
        
        if symbol_ids:
            await self.api.stream_quotes(symbol_ids, callback)


class QuestradeOrderManager:
    """
    High-level interface for Questrade order management
    """
    
    def __init__(self, api: QuestradeAPI):
        self.api = api
        self.data_feed = QuestradeDataFeed(api)
    
    async def place_market_order(self,
                               account_id: str,
                               symbol: str,
                               quantity: int,
                               action: str = "Buy") -> Dict:
        """Place a market order"""
        symbol_id = await self.data_feed.get_symbol_id(symbol)
        if not symbol_id:
            raise QuestradeAPIError(f"Symbol not found: {symbol}")
        
        return await self.api.place_order(
            account_id=account_id,
            symbol_id=symbol_id,
            order_type="Market",
            action=action,
            quantity=quantity
        )
    
    async def place_limit_order(self,
                              account_id: str,
                              symbol: str,
                              quantity: int,
                              limit_price: float,
                              action: str = "Buy",
                              time_in_force: str = "Day") -> Dict:
        """Place a limit order"""
        symbol_id = await self.data_feed.get_symbol_id(symbol)
        if not symbol_id:
            raise QuestradeAPIError(f"Symbol not found: {symbol}")
        
        return await self.api.place_order(
            account_id=account_id,
            symbol_id=symbol_id,
            order_type="Limit",
            action=action,
            quantity=quantity,
            limit_price=limit_price,
            time_in_force=time_in_force
        )
    
    async def place_stop_loss_order(self,
                                  account_id: str,
                                  symbol: str,
                                  quantity: int,
                                  stop_price: float,
                                  action: str = "Sell") -> Dict:
        """Place a stop loss order"""
        symbol_id = await self.data_feed.get_symbol_id(symbol)
        if not symbol_id:
            raise QuestradeAPIError(f"Symbol not found: {symbol}")
        
        return await self.api.place_order(
            account_id=account_id,
            symbol_id=symbol_id,
            order_type="Stop",
            action=action,
            quantity=quantity,
            stop_price=stop_price
        )
    
    async def place_bracket_order(self,
                                account_id: str,
                                symbol: str,
                                quantity: int,
                                limit_price: float,
                                stop_loss_price: float,
                                take_profit_price: float) -> Dict:
        """
        Place a bracket order (entry with stop loss and take profit)
        Note: This requires multiple API calls as Questrade doesn't support
        native bracket orders
        """
        results = {
            "entry_order": None,
            "stop_loss_order": None,
            "take_profit_order": None,
            "errors": []
        }
        
        try:
            # Place entry order
            results["entry_order"] = await self.place_limit_order(
                account_id=account_id,
                symbol=symbol,
                quantity=quantity,
                limit_price=limit_price,
                action="Buy"
            )
            
            # Wait for fill (in production, use order status monitoring)
            await asyncio.sleep(1)
            
            # Place stop loss
            try:
                results["stop_loss_order"] = await self.place_stop_loss_order(
                    account_id=account_id,
                    symbol=symbol,
                    quantity=quantity,
                    stop_price=stop_loss_price,
                    action="Sell"
                )
            except Exception as e:
                results["errors"].append(f"Stop loss order failed: {str(e)}")
            
            # Place take profit
            try:
                results["take_profit_order"] = await self.place_limit_order(
                    account_id=account_id,
                    symbol=symbol,
                    quantity=quantity,
                    limit_price=take_profit_price,
                    action="Sell",
                    time_in_force="GTC"
                )
            except Exception as e:
                results["errors"].append(f"Take profit order failed: {str(e)}")
            
        except Exception as e:
            results["errors"].append(f"Entry order failed: {str(e)}")
        
        return results
    
    async def get_order_status(self, account_id: str, order_id: int) -> Optional[Dict]:
        """Get status of a specific order"""
        orders = await self.api.get_account_orders(account_id, ids=[order_id])
        return orders[0] if orders else None
    
    async def cancel_all_orders(self, account_id: str, symbol: Optional[str] = None) -> List[Dict]:
        """Cancel all open orders for an account, optionally filtered by symbol"""
        orders = await self.api.get_account_orders(account_id, state_filter="Open")
        
        results = []
        for order in orders:
            # Filter by symbol if specified
            if symbol:
                order_symbol = order.get("symbol")
                if order_symbol != symbol:
                    continue
            
            order_id = order.get("id")
            try:
                result = await self.api.cancel_order(account_id, order_id)
                results.append({
                    "order_id": order_id,
                    "status": "cancelled",
                    "result": result
                })
            except Exception as e:
                results.append({
                    "order_id": order_id,
                    "status": "error",
                    "error": str(e)
                })
        
        return results