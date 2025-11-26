"""
Alpaca REST API client for trading operations
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import aiohttp
import json
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AlpacaRESTClient:
    """Alpaca REST API client for trading operations"""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None,
                 base_url: Optional[str] = None,
                 api_version: str = "v2"):
        """
        Initialize Alpaca REST client
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret  
            base_url: Base API URL
            api_version: API version
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_API_SECRET')
        self.base_url = base_url or os.getenv('ALPACA_API_ENDPOINT', 'https://paper-api.alpaca.markets')
        self.api_version = api_version
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key and secret are required")
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._base_headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._base_headers
            )
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to Alpaca API"""
        await self._ensure_session()
        
        url = f"{self.base_url}/{self.api_version}/{endpoint}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                content = await response.text()
                
                if response.status >= 400:
                    logger.error(f"API Error {response.status}: {content}")
                    raise Exception(f"Alpaca API error: {response.status} - {content}")
                
                if content:
                    return json.loads(content)
                return {}
                
        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Request failed: {e}")
    
    # Account endpoints
    async def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        return await self._request('GET', 'account')
    
    async def get_portfolio_history(self, 
                                   timeframe: str = "1D",
                                   extended_hours: bool = False) -> Dict[str, Any]:
        """Get portfolio history"""
        params = {
            'timeframe': timeframe,
            'extended_hours': extended_hours
        }
        return await self._request('GET', 'account/portfolio/history', params=params)
    
    # Positions endpoints
    async def list_positions(self) -> List[Dict[str, Any]]:
        """List all positions"""
        result = await self._request('GET', 'positions')
        return result if isinstance(result, list) else []
    
    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get specific position"""
        return await self._request('GET', f'positions/{symbol}')
    
    async def close_position(self, symbol: str, 
                           qty: Optional[Union[int, str]] = None,
                           percentage: Optional[float] = None) -> Dict[str, Any]:
        """Close position"""
        data = {}
        if qty:
            data['qty'] = str(qty)
        if percentage:
            data['percentage'] = str(percentage)
            
        return await self._request('DELETE', f'positions/{symbol}', json=data)
    
    async def close_all_positions(self, cancel_orders: bool = True) -> List[Dict[str, Any]]:
        """Close all positions"""
        params = {'cancel_orders': cancel_orders}
        result = await self._request('DELETE', 'positions', params=params)
        return result if isinstance(result, list) else []
    
    # Orders endpoints
    async def list_orders(self, 
                         status: Optional[str] = None,
                         limit: int = 50,
                         after: Optional[str] = None,
                         until: Optional[str] = None,
                         direction: str = "desc") -> List[Dict[str, Any]]:
        """List orders"""
        params = {
            'limit': limit,
            'direction': direction
        }
        if status:
            params['status'] = status
        if after:
            params['after'] = after
        if until:
            params['until'] = until
            
        result = await self._request('GET', 'orders', params=params)
        return result if isinstance(result, list) else []
    
    async def create_order(self,
                          symbol: str,
                          qty: Union[int, str],
                          side: str,
                          type: str = "market",
                          time_in_force: str = "day",
                          limit_price: Optional[Union[float, str]] = None,
                          stop_price: Optional[Union[float, str]] = None,
                          order_class: Optional[str] = None,
                          take_profit: Optional[Dict[str, str]] = None,
                          stop_loss: Optional[Dict[str, str]] = None,
                          client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Create new order"""
        data = {
            'symbol': symbol,
            'qty': str(qty),
            'side': side,
            'type': type,
            'time_in_force': time_in_force
        }
        
        if limit_price:
            data['limit_price'] = str(limit_price)
        if stop_price:
            data['stop_price'] = str(stop_price)
        if order_class:
            data['order_class'] = order_class
        if take_profit:
            data['take_profit'] = take_profit
        if stop_loss:
            data['stop_loss'] = stop_loss
        if client_order_id:
            data['client_order_id'] = client_order_id
            
        return await self._request('POST', 'orders', json=data)
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order by ID"""
        return await self._request('GET', f'orders/{order_id}')
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order"""
        return await self._request('DELETE', f'orders/{order_id}')
    
    async def cancel_all_orders(self) -> List[Dict[str, Any]]:
        """Cancel all orders"""
        result = await self._request('DELETE', 'orders')
        return result if isinstance(result, list) else []
    
    async def replace_order(self, order_id: str, **kwargs) -> Dict[str, Any]:
        """Replace order"""
        return await self._request('PATCH', f'orders/{order_id}', json=kwargs)
    
    # Market data endpoints
    async def get_bars(self, 
                      symbols: Union[str, List[str]],
                      timeframe: str = "1Day",
                      start: Optional[str] = None,
                      end: Optional[str] = None,
                      limit: Optional[int] = None,
                      adjustment: str = "raw") -> Dict[str, Any]:
        """Get historical bars"""
        if isinstance(symbols, list):
            symbols = ','.join(symbols)
            
        params = {
            'symbols': symbols,
            'timeframe': timeframe,
            'adjustment': adjustment
        }
        
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        if limit:
            params['limit'] = limit
            
        # Use data API endpoint
        url = f"https://data.alpaca.markets/{self.api_version}/stocks/bars"
        
        try:
            async with self.session.get(url, params=params) as response:
                content = await response.text()
                
                if response.status >= 400:
                    logger.error(f"Market data error {response.status}: {content}")
                    raise Exception(f"Market data error: {response.status} - {content}")
                
                return json.loads(content) if content else {}
                
        except aiohttp.ClientError as e:
            logger.error(f"Market data request failed: {e}")
            raise Exception(f"Market data request failed: {e}")
    
    async def get_latest_trade(self, symbol: str) -> Dict[str, Any]:
        """Get latest trade for symbol"""
        url = f"https://data.alpaca.markets/{self.api_version}/stocks/{symbol}/trades/latest"
        
        try:
            async with self.session.get(url) as response:
                content = await response.text()
                
                if response.status >= 400:
                    logger.error(f"Trade data error {response.status}: {content}")
                    raise Exception(f"Trade data error: {response.status} - {content}")
                
                return json.loads(content) if content else {}
                
        except aiohttp.ClientError as e:
            logger.error(f"Trade data request failed: {e}")
            raise Exception(f"Trade data request failed: {e}")
    
    async def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Get latest quote for symbol"""
        url = f"https://data.alpaca.markets/{self.api_version}/stocks/{symbol}/quotes/latest"
        
        try:
            async with self.session.get(url) as response:
                content = await response.text()
                
                if response.status >= 400:
                    logger.error(f"Quote data error {response.status}: {content}")
                    raise Exception(f"Quote data error: {response.status} - {content}")
                
                return json.loads(content) if content else {}
                
        except aiohttp.ClientError as e:
            logger.error(f"Quote data request failed: {e}")
            raise Exception(f"Quote data request failed: {e}")
    
    # Trading status endpoints  
    async def get_clock(self) -> Dict[str, Any]:
        """Get market clock"""
        return await self._request('GET', 'clock')
    
    async def get_calendar(self, 
                          start: Optional[str] = None,
                          end: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trading calendar"""
        params = {}
        if start:
            params['start'] = start
        if end:
            params['end'] = end
            
        result = await self._request('GET', 'calendar', params=params)
        return result if isinstance(result, list) else []
    
    # Assets endpoints
    async def list_assets(self, 
                         status: str = "active",
                         asset_class: Optional[str] = None) -> List[Dict[str, Any]]:
        """List assets"""
        params = {'status': status}
        if asset_class:
            params['class'] = asset_class
            
        result = await self._request('GET', 'assets', params=params)
        return result if isinstance(result, list) else []
    
    async def get_asset(self, symbol: str) -> Dict[str, Any]:
        """Get asset info"""
        return await self._request('GET', f'assets/{symbol}')


# Convenience alias for backwards compatibility
AlpacaClient = AlpacaRESTClient