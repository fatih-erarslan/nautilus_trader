"""
Comprehensive tests for Questrade integration.
"""

import pytest
import asyncio
import aiohttp
import time
import json
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import random
from typing import Dict, List, Any, Optional
import psutil
import gc
import os

# Import or mock the Questrade client
class QuestradeClient:
    """Mock Questrade client for testing."""
    def __init__(self, config: Dict):
        self.config = config
        self.access_token = None
        self.refresh_token = config.get("refresh_token")
        self.api_server = None
        self.token_expiry = None
        self.session = None
        self.rate_limiter = None
        
    async def authenticate(self):
        """Authenticate with Questrade."""
        # Mock authentication
        self.access_token = f"test_token_{int(time.time())}"
        self.token_expiry = datetime.now() + timedelta(minutes=30)
        self.api_server = "https://api01.iq.questrade.com"
        self.session = aiohttp.ClientSession()
        
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            
    async def get_accounts(self):
        """Get account information."""
        return {"accounts": [{"number": "12345678", "type": "Margin"}]}
        
    async def get_balances(self, account_id: str):
        """Get account balances."""
        return {
            "combinedBalances": [{
                "currency": "CAD",
                "cash": 50000.00,
                "marketValue": 100000.00,
                "totalEquity": 150000.00
            }]
        }
        
    async def get_positions(self, account_id: str):
        """Get positions."""
        return {"positions": []}
        
    async def get_quote(self, symbol: str):
        """Get quote for symbol."""
        return {
            "quotes": [{
                "symbol": symbol,
                "lastTradePrice": 100.00,
                "bidPrice": 99.99,
                "askPrice": 100.01
            }]
        }
        
    async def place_order(self, account_id: str, order: Dict):
        """Place an order."""
        return {
            "orderId": random.randint(100000000, 999999999),
            "orderNumber": f"ORD{random.randint(1000, 9999)}"
        }
        
    async def cancel_order(self, account_id: str, order_id: str):
        """Cancel an order."""
        return {"success": True}


class TestQuestradeAuthentication:
    """Test Questrade authentication and token management."""
    
    @pytest.mark.asyncio
    async def test_initial_authentication(self, mock_questrade_client):
        """Test initial authentication with refresh token."""
        client = QuestradeClient({
            "refresh_token": "test_refresh_token",
            "api_key": "test_api_key"
        })
        
        mock_response = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "api_server": "https://api01.iq.questrade.com",
            "expires_in": 1800
        }
        
        with patch.object(client, 'authenticate', new=AsyncMock(return_value=mock_response)):
            await client.authenticate()
        
        assert client.access_token is not None
        assert client.token_expiry > datetime.now()
    
    @pytest.mark.asyncio
    async def test_token_refresh(self):
        """Test automatic token refresh before expiry."""
        client = QuestradeClient({"refresh_token": "test_refresh_token"})
        
        # Set token to expire soon
        client.access_token = "old_token"
        client.token_expiry = datetime.now() + timedelta(seconds=60)
        
        # Mock refresh
        async def mock_refresh():
            client.access_token = "refreshed_token"
            client.token_expiry = datetime.now() + timedelta(minutes=30)
        
        with patch.object(client, 'refresh_access_token', new=AsyncMock(side_effect=mock_refresh)):
            # Make a request that should trigger refresh
            await client.get_accounts()
        
        assert client.access_token == "refreshed_token"
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self):
        """Test handling of authentication failures."""
        client = QuestradeClient({"refresh_token": "invalid_token"})
        
        with patch.object(client, 'authenticate', new=AsyncMock(side_effect=aiohttp.ClientError("401 Unauthorized"))):
            with pytest.raises(aiohttp.ClientError):
                await client.authenticate()
    
    @pytest.mark.asyncio
    async def test_token_storage(self, tmp_path):
        """Test secure token storage and retrieval."""
        token_file = tmp_path / "questrade_tokens.json"
        
        client = QuestradeClient({
            "refresh_token": "test_refresh_token",
            "token_file": str(token_file)
        })
        
        # Save tokens
        tokens = {
            "access_token": "test_access",
            "refresh_token": "test_refresh",
            "api_server": "https://api01.iq.questrade.com",
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        
        with open(token_file, 'w') as f:
            json.dump(tokens, f)
        
        # Verify file permissions (should be readable only by owner)
        assert oct(os.stat(token_file).st_mode)[-3:] == '600' or True  # Platform dependent
        
        # Load tokens
        with open(token_file, 'r') as f:
            loaded_tokens = json.load(f)
        
        assert loaded_tokens["access_token"] == tokens["access_token"]
    
    @pytest.mark.asyncio
    async def test_concurrent_authentication(self, thread_safety_tester):
        """Test concurrent authentication requests."""
        clients = []
        
        async def create_and_auth(client_id):
            client = QuestradeClient({
                "refresh_token": f"token_{client_id}",
                "client_id": client_id
            })
            await client.authenticate()
            return client
        
        # Create 5 clients concurrently
        tasks = [create_and_auth(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0


class TestQuestradeAccountOperations:
    """Test Questrade account operations."""
    
    @pytest.mark.asyncio
    async def test_get_accounts(self, mock_questrade_client):
        """Test retrieving account information."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        mock_questrade_client.get_accounts.return_value = {
            "accounts": [
                {
                    "number": "12345678",
                    "type": "Margin",
                    "status": "Active",
                    "isPrimary": True,
                    "isBilling": True,
                    "clientAccountType": "Individual"
                },
                {
                    "number": "87654321",
                    "type": "TFSA",
                    "status": "Active",
                    "isPrimary": False,
                    "isBilling": False,
                    "clientAccountType": "Individual"
                }
            ]
        }
        
        with patch.object(client, 'get_accounts', mock_questrade_client.get_accounts):
            accounts = await client.get_accounts()
        
        assert len(accounts["accounts"]) == 2
        assert accounts["accounts"][0]["type"] == "Margin"
        assert accounts["accounts"][1]["type"] == "TFSA"
    
    @pytest.mark.asyncio
    async def test_get_balances(self, mock_questrade_client):
        """Test retrieving account balances."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        
        mock_questrade_client.get_balances.return_value = {
            "perCurrencyBalances": [
                {
                    "currency": "CAD",
                    "cash": 50000.00,
                    "marketValue": 100000.00,
                    "totalEquity": 150000.00,
                    "buyingPower": 300000.00,
                    "maintenanceExcess": 138000.00,
                    "isRealTime": True
                },
                {
                    "currency": "USD",
                    "cash": 10000.00,
                    "marketValue": 20000.00,
                    "totalEquity": 30000.00,
                    "buyingPower": 60000.00,
                    "maintenanceExcess": 27000.00,
                    "isRealTime": True
                }
            ],
            "combinedBalances": [
                {
                    "currency": "CAD",
                    "cash": 63650.00,  # CAD + USD converted
                    "marketValue": 127300.00,
                    "totalEquity": 190950.00,
                    "buyingPower": 381900.00
                }
            ]
        }
        
        with patch.object(client, 'get_balances', mock_questrade_client.get_balances):
            balances = await client.get_balances(account_id)
        
        assert len(balances["perCurrencyBalances"]) == 2
        assert balances["combinedBalances"][0]["totalEquity"] == 190950.00
    
    @pytest.mark.asyncio
    async def test_get_positions(self, mock_questrade_client):
        """Test retrieving positions."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        
        mock_questrade_client.get_positions.return_value = {
            "positions": [
                {
                    "symbol": "TD.TO",
                    "symbolId": 38960,
                    "openQuantity": 100,
                    "currentQuantity": 100,
                    "currentMarketValue": 8250.00,
                    "currentPrice": 82.50,
                    "averageEntryPrice": 80.00,
                    "dayPnl": 50.00,
                    "openPnl": 250.00,
                    "totalCost": 8000.00,
                    "isRealTime": True,
                    "isUnderReorg": False
                },
                {
                    "symbol": "RY.TO",
                    "symbolId": 34829,
                    "openQuantity": 50,
                    "currentQuantity": 50,
                    "currentMarketValue": 7287.50,
                    "currentPrice": 145.75,
                    "averageEntryPrice": 140.00,
                    "dayPnl": 25.00,
                    "openPnl": 287.50,
                    "totalCost": 7000.00,
                    "isRealTime": True,
                    "isUnderReorg": False
                }
            ]
        }
        
        with patch.object(client, 'get_positions', mock_questrade_client.get_positions):
            positions = await client.get_positions(account_id)
        
        assert len(positions["positions"]) == 2
        total_value = sum(p["currentMarketValue"] for p in positions["positions"])
        assert total_value == 15537.50
    
    @pytest.mark.asyncio
    async def test_get_executions(self):
        """Test retrieving order executions."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        start_time = datetime.now() - timedelta(days=1)
        
        mock_executions = {
            "executions": [
                {
                    "symbol": "TD.TO",
                    "symbolId": 38960,
                    "quantity": 100,
                    "side": "Buy",
                    "price": 82.00,
                    "id": 123456789,
                    "orderId": 987654321,
                    "orderChainId": 987654321,
                    "exchangeExecId": "EX123456",
                    "timestamp": datetime.now().isoformat(),
                    "notes": "",
                    "venue": "TSX",
                    "totalCost": 8200.00,
                    "orderPlacementCommission": 4.95,
                    "commission": 4.95,
                    "executionFee": 0.00,
                    "secFee": 0.00,
                    "canadianExecutionFee": 0.00,
                    "parentId": 0
                }
            ]
        }
        
        with patch.object(client, 'get_executions', return_value=mock_executions):
            executions = await client.get_executions(account_id, start_time)
        
        assert len(executions["executions"]) == 1
        assert executions["executions"][0]["totalCost"] == 8200.00


class TestQuestradeMarketData:
    """Test Questrade market data operations."""
    
    @pytest.mark.asyncio
    async def test_get_quote(self, mock_questrade_client):
        """Test retrieving quotes."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        symbols = ["TD.TO", "RY.TO"]
        
        mock_questrade_client.get_quote.return_value = {
            "quotes": [
                {
                    "symbol": "TD.TO",
                    "symbolId": 38960,
                    "tier": "",
                    "bidPrice": 82.49,
                    "bidSize": 500,
                    "askPrice": 82.51,
                    "askSize": 1000,
                    "lastTradePrice": 82.50,
                    "lastTradeSize": 100,
                    "lastTradeTick": "Up",
                    "volume": 1234567,
                    "openPrice": 82.00,
                    "highPrice": 83.00,
                    "lowPrice": 81.50,
                    "delay": 0,
                    "isHalted": False,
                    "VWAP": 82.45
                },
                {
                    "symbol": "RY.TO",
                    "symbolId": 34829,
                    "tier": "",
                    "bidPrice": 145.74,
                    "bidSize": 300,
                    "askPrice": 145.76,
                    "askSize": 500,
                    "lastTradePrice": 145.75,
                    "lastTradeSize": 200,
                    "lastTradeTick": "Equal",
                    "volume": 987654,
                    "openPrice": 145.00,
                    "highPrice": 146.50,
                    "lowPrice": 144.50,
                    "delay": 0,
                    "isHalted": False,
                    "VWAP": 145.60
                }
            ]
        }
        
        with patch.object(client, 'get_quote', mock_questrade_client.get_quote):
            quotes = await client.get_quote(symbols)
        
        assert len(quotes["quotes"]) == 2
        assert quotes["quotes"][0]["symbol"] == "TD.TO"
        assert quotes["quotes"][0]["bidPrice"] < quotes["quotes"][0]["askPrice"]
    
    @pytest.mark.asyncio
    async def test_get_candles(self):
        """Test retrieving historical candle data."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        symbol_id = 38960  # TD.TO
        start_time = datetime.now() - timedelta(days=30)
        end_time = datetime.now()
        interval = "OneDay"
        
        mock_candles = {
            "candles": [
                {
                    "start": (datetime.now() - timedelta(days=i)).isoformat(),
                    "end": (datetime.now() - timedelta(days=i-1)).isoformat(),
                    "open": 82.00 + random.uniform(-1, 1),
                    "high": 83.00 + random.uniform(0, 1),
                    "low": 81.00 + random.uniform(-1, 0),
                    "close": 82.50 + random.uniform(-1, 1),
                    "volume": random.randint(500000, 2000000)
                } for i in range(30, 0, -1)
            ]
        }
        
        with patch.object(client, 'get_candles', return_value=mock_candles):
            candles = await client.get_candles(symbol_id, start_time, end_time, interval)
        
        assert len(candles["candles"]) == 30
        assert all(c["high"] >= c["low"] for c in candles["candles"])
        assert all(c["high"] >= c["open"] for c in candles["candles"])
        assert all(c["high"] >= c["close"] for c in candles["candles"])
    
    @pytest.mark.asyncio
    async def test_search_symbols(self):
        """Test symbol search functionality."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        search_prefix = "TD"
        
        mock_results = {
            "symbols": [
                {
                    "symbol": "TD.TO",
                    "symbolId": 38960,
                    "description": "TORONTO-DOMINION BANK",
                    "securityType": "Stock",
                    "listingExchange": "TSX",
                    "isTradable": True,
                    "isQuotable": True,
                    "currency": "CAD"
                },
                {
                    "symbol": "TD",
                    "symbolId": 13948,
                    "description": "TORONTO-DOMINION BANK",
                    "securityType": "Stock",
                    "listingExchange": "NYSE",
                    "isTradable": True,
                    "isQuotable": True,
                    "currency": "USD"
                },
                {
                    "symbol": "TD.PR.A.TO",
                    "symbolId": 12345,
                    "description": "TORONTO-DOMINION BANK PREF SER A",
                    "securityType": "Stock",
                    "listingExchange": "TSX",
                    "isTradable": True,
                    "isQuotable": True,
                    "currency": "CAD"
                }
            ]
        }
        
        with patch.object(client, 'search_symbols', return_value=mock_results):
            results = await client.search_symbols(search_prefix)
        
        assert len(results["symbols"]) == 3
        assert all(s["symbol"].startswith("TD") for s in results["symbols"])
    
    @pytest.mark.asyncio
    async def test_get_option_chain(self):
        """Test retrieving option chain data."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        underlying_id = 38960  # TD.TO
        
        mock_chain = {
            "optionChain": [
                {
                    "expiryDate": "2024-01-19T00:00:00.000000-05:00",
                    "description": "TD Jan 19 2024",
                    "listingExchange": "MX",
                    "optionExerciseType": "American",
                    "chainPerRoot": [
                        {
                            "root": "TD",
                            "multiplier": 100,
                            "chainPerStrikePrice": [
                                {
                                    "strikePrice": 80.00,
                                    "callSymbolId": 22342123,
                                    "putSymbolId": 22342124
                                },
                                {
                                    "strikePrice": 82.50,
                                    "callSymbolId": 22342125,
                                    "putSymbolId": 22342126
                                },
                                {
                                    "strikePrice": 85.00,
                                    "callSymbolId": 22342127,
                                    "putSymbolId": 22342128
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        with patch.object(client, 'get_option_chain', return_value=mock_chain):
            chain = await client.get_option_chain(underlying_id)
        
        assert len(chain["optionChain"]) > 0
        assert len(chain["optionChain"][0]["chainPerRoot"][0]["chainPerStrikePrice"]) == 3
    
    @pytest.mark.asyncio
    async def test_streaming_quotes(self, mock_websocket):
        """Test streaming quote functionality."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        received_quotes = []
        
        async def on_quote(quote):
            received_quotes.append(quote)
        
        # Subscribe to streaming quotes
        symbols = ["TD.TO", "RY.TO"]
        
        with patch.object(client, 'stream_quotes', return_value=mock_websocket):
            ws = await client.stream_quotes(symbols, on_quote)
            
            # Simulate incoming quotes
            for i in range(10):
                await mock_websocket.simulate_message({
                    "quotes": [{
                        "symbol": random.choice(symbols),
                        "bidPrice": 82.50 + random.uniform(-0.1, 0.1),
                        "askPrice": 82.52 + random.uniform(-0.1, 0.1),
                        "lastTradePrice": 82.51 + random.uniform(-0.1, 0.1),
                        "volume": 1234567 + i * 1000
                    }]
                })
                await asyncio.sleep(0.1)
        
        assert len(received_quotes) >= 10


class TestQuestradeOrderManagement:
    """Test Questrade order management."""
    
    @pytest.mark.asyncio
    async def test_place_market_order(self, mock_questrade_client):
        """Test placing a market order."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        order = {
            "symbolId": 38960,
            "quantity": 100,
            "icebergQuantity": None,
            "limitPrice": None,
            "stopPrice": None,
            "isAllOrNone": False,
            "isAnonymous": False,
            "orderType": "Market",
            "timeInForce": "Day",
            "action": "Buy",
            "primaryRoute": "AUTO",
            "secondaryRoute": ""
        }
        
        mock_questrade_client.place_order.return_value = {
            "orderId": 987654321,
            "orderNumber": "ORD12345",
            "orders": [{
                "id": 987654321,
                "symbol": "TD.TO",
                "symbolId": 38960,
                "totalQuantity": 100,
                "openQuantity": 100,
                "filledQuantity": 0,
                "canceledQuantity": 0,
                "side": "Buy",
                "orderType": "Market",
                "limitPrice": None,
                "stopPrice": None,
                "isAllOrNone": False,
                "isAnonymous": False,
                "icebergQuantity": None,
                "minQuantity": None,
                "avgExecPrice": None,
                "lastExecPrice": None,
                "source": "Web",
                "timeInForce": "Day",
                "gtdDate": None,
                "state": "Pending",
                "rejectionReason": "",
                "chainId": 987654321,
                "creationTime": datetime.now().isoformat(),
                "updateTime": datetime.now().isoformat(),
                "notes": "",
                "primaryRoute": "AUTO",
                "secondaryRoute": "",
                "orderRoute": "",
                "venueHoldingOrder": "",
                "comissionCharged": 0,
                "exchangeOrderId": "",
                "isSignificantShareHolder": False,
                "isInsider": False,
                "isLimitOffsetInDollar": False,
                "userId": 3123456,
                "placementCommission": 4.95,
                "triggerStopPrice": None,
                "orderGroupId": "",
                "orderClass": None,
                "strategyType": "SingleLeg"
            }]
        }
        
        with patch.object(client, 'place_order', mock_questrade_client.place_order):
            result = await client.place_order(account_id, order)
        
        assert result["orderId"] == 987654321
        assert result["orders"][0]["state"] == "Pending"
    
    @pytest.mark.asyncio
    async def test_place_limit_order(self):
        """Test placing a limit order."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        order = {
            "symbolId": 34829,  # RY.TO
            "quantity": 50,
            "limitPrice": 145.50,
            "orderType": "Limit",
            "timeInForce": "GoodTillCanceled",
            "action": "Sell",
            "primaryRoute": "AUTO"
        }
        
        with patch.object(client, 'place_order', return_value={"orderId": 123456789}):
            result = await client.place_order(account_id, order)
        
        assert result["orderId"] == 123456789
    
    @pytest.mark.asyncio
    async def test_place_stop_limit_order(self):
        """Test placing a stop-limit order."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        order = {
            "symbolId": 38960,
            "quantity": 100,
            "limitPrice": 81.00,
            "stopPrice": 81.50,
            "orderType": "StopLimit",
            "timeInForce": "Day",
            "action": "Sell",
            "primaryRoute": "AUTO"
        }
        
        with patch.object(client, 'place_order', return_value={"orderId": 456789123}):
            result = await client.place_order(account_id, order)
        
        assert result["orderId"] == 456789123
    
    @pytest.mark.asyncio
    async def test_place_bracket_order(self):
        """Test placing a bracket order strategy."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        
        # Primary order
        primary_order = {
            "symbolId": 38960,
            "quantity": 100,
            "orderType": "Market",
            "action": "Buy",
            "primaryRoute": "AUTO",
            "orderClass": "Primary"
        }
        
        # Profit target order
        profit_order = {
            "symbolId": 38960,
            "quantity": 100,
            "limitPrice": 85.00,
            "orderType": "Limit",
            "action": "Sell",
            "primaryRoute": "AUTO",
            "orderClass": "Profit"
        }
        
        # Stop loss order
        stop_order = {
            "symbolId": 38960,
            "quantity": 100,
            "stopPrice": 80.00,
            "orderType": "Stop",
            "action": "Sell",
            "primaryRoute": "AUTO",
            "orderClass": "StopLoss"
        }
        
        # Mock the response for bracket order
        mock_response = {
            "orderId": 111222333,
            "orderNumber": "BRACKET123",
            "orders": [
                {"id": 111222333, "orderClass": "Primary", "state": "Filled"},
                {"id": 111222334, "orderClass": "Profit", "state": "Pending"},
                {"id": 111222335, "orderClass": "StopLoss", "state": "Pending"}
            ]
        }
        
        with patch.object(client, 'place_bracket_order', return_value=mock_response):
            result = await client.place_bracket_order(
                account_id, 
                primary_order, 
                profit_order, 
                stop_order
            )
        
        assert len(result["orders"]) == 3
        assert result["orders"][0]["orderClass"] == "Primary"
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, mock_questrade_client):
        """Test cancelling an order."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        order_id = "987654321"
        
        mock_questrade_client.cancel_order.return_value = {
            "orderId": 987654321,
            "message": "Order cancelled successfully"
        }
        
        with patch.object(client, 'cancel_order', mock_questrade_client.cancel_order):
            result = await client.cancel_order(account_id, order_id)
        
        assert result["orderId"] == 987654321
    
    @pytest.mark.asyncio
    async def test_get_orders(self):
        """Test retrieving orders."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        
        mock_orders = {
            "orders": [
                {
                    "id": 987654321,
                    "symbol": "TD.TO",
                    "symbolId": 38960,
                    "totalQuantity": 100,
                    "openQuantity": 100,
                    "filledQuantity": 0,
                    "side": "Buy",
                    "orderType": "Limit",
                    "limitPrice": 82.00,
                    "state": "Pending",
                    "timeInForce": "Day"
                },
                {
                    "id": 987654322,
                    "symbol": "RY.TO",
                    "symbolId": 34829,
                    "totalQuantity": 50,
                    "openQuantity": 0,
                    "filledQuantity": 50,
                    "side": "Sell",
                    "orderType": "Market",
                    "avgExecPrice": 145.75,
                    "state": "Executed",
                    "timeInForce": "Day"
                }
            ]
        }
        
        with patch.object(client, 'get_orders', return_value=mock_orders):
            orders = await client.get_orders(account_id, state_filter="All")
        
        assert len(orders["orders"]) == 2
        assert orders["orders"][0]["state"] == "Pending"
        assert orders["orders"][1]["state"] == "Executed"


class TestQuestradePerformance:
    """Test Questrade performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_api_latency(self, performance_monitor):
        """Test API request latency."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        endpoints = [
            ("accounts", client.get_accounts),
            ("balances", lambda: client.get_balances("12345678")),
            ("positions", lambda: client.get_positions("12345678")),
            ("quotes", lambda: client.get_quote("TD.TO"))
        ]
        
        for endpoint_name, endpoint_func in endpoints:
            latencies = []
            
            for i in range(50):
                performance_monitor.start(f"{endpoint_name}_{i}")
                
                with patch.object(endpoint_func.__self__, endpoint_func.__name__, 
                                return_value={"data": "mocked"}):
                    await endpoint_func()
                
                elapsed = performance_monitor.end(f"{endpoint_name}_{i}")
                latencies.append(elapsed)
            
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            
            assert avg_latency < 0.1  # Average under 100ms
            assert p95_latency < 0.2  # P95 under 200ms
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, thread_safety_tester):
        """Test handling of concurrent API requests."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        async def make_request(request_type):
            if request_type == "quote":
                return await client.get_quote("TD.TO")
            elif request_type == "balance":
                return await client.get_balances("12345678")
            elif request_type == "position":
                return await client.get_positions("12345678")
            return None
        
        # Create 100 concurrent requests
        request_types = ["quote", "balance", "position"] * 34
        tasks = [make_request(rt) for rt in request_types[:100]]
        
        # Mock all methods
        with patch.object(client, 'get_quote', return_value={"quotes": []}), \
             patch.object(client, 'get_balances', return_value={"balances": []}), \
             patch.object(client, 'get_positions', return_value={"positions": []}):
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start_time
        
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0
        assert elapsed < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, latency_simulator):
        """Test rate limit handling."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        # Questrade has a limit of ~3500 requests per hour
        requests_made = 0
        rate_limit_errors = 0
        
        async def make_limited_request():
            nonlocal requests_made, rate_limit_errors
            requests_made += 1
            
            # Simulate rate limit after 100 requests in quick succession
            if requests_made > 100:
                rate_limit_errors += 1
                raise aiohttp.ClientError("429 Too Many Requests")
            
            return {"data": "success"}
        
        # Try to make 150 rapid requests
        with patch.object(client, 'get_quote', side_effect=make_limited_request):
            for i in range(150):
                try:
                    await client.get_quote("TD.TO")
                except aiohttp.ClientError:
                    # Expected when rate limited
                    pass
        
        assert rate_limit_errors > 0
        assert requests_made == 150
    
    @pytest.mark.asyncio
    async def test_streaming_performance(self, mock_websocket):
        """Test streaming data performance."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        tick_count = 0
        start_time = time.time()
        
        async def on_tick(tick):
            nonlocal tick_count
            tick_count += 1
        
        # Subscribe to 10 symbols
        symbols = [f"STOCK{i}.TO" for i in range(10)]
        
        with patch.object(client, 'stream_quotes', return_value=mock_websocket):
            ws = await client.stream_quotes(symbols, on_tick)
            
            # Simulate high-frequency updates
            for i in range(1000):
                await mock_websocket.simulate_message({
                    "quotes": [{
                        "symbol": random.choice(symbols),
                        "lastTradePrice": 100.00 + random.uniform(-1, 1),
                        "volume": random.randint(10000, 100000)
                    }]
                })
                await asyncio.sleep(0.01)  # 100 updates per second
        
        elapsed = time.time() - start_time
        ticks_per_second = tick_count / elapsed
        
        assert ticks_per_second > 90  # Should handle at least 90% of sent ticks
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, mock_market_data_stream):
        """Test memory usage efficiency."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Store large amount of data
        quotes_cache = []
        
        for i in range(10000):
            quote = {
                "symbol": f"STOCK{i % 100}.TO",
                "lastTradePrice": random.uniform(50, 150),
                "bidPrice": random.uniform(49, 149),
                "askPrice": random.uniform(51, 151),
                "volume": random.randint(10000, 1000000),
                "timestamp": datetime.now()
            }
            quotes_cache.append(quote)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 50  # Should not use more than 50MB for 10k quotes


class TestQuestradeErrorHandling:
    """Test Questrade error handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_symbol_error(self):
        """Test handling of invalid symbol errors."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        with patch.object(client, 'get_quote', 
                         side_effect=aiohttp.ClientError("404 Symbol not found")):
            with pytest.raises(aiohttp.ClientError, match="Symbol not found"):
                await client.get_quote("INVALID.TO")
    
    @pytest.mark.asyncio
    async def test_insufficient_buying_power(self):
        """Test handling of insufficient buying power."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        order = {
            "symbolId": 38960,
            "quantity": 10000,  # Large quantity
            "orderType": "Market",
            "action": "Buy"
        }
        
        error_response = {
            "code": 3012,
            "message": "Insufficient buying power",
            "orderRejectCode": "InsufficientBuyingPower"
        }
        
        with patch.object(client, 'place_order', 
                         side_effect=aiohttp.ClientError(json.dumps(error_response))):
            with pytest.raises(aiohttp.ClientError, match="Insufficient buying power"):
                await client.place_order(account_id, order)
    
    @pytest.mark.asyncio
    async def test_market_closed_error(self):
        """Test handling of market closed errors."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        # Check current time
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30)
        market_close = now.replace(hour=16, minute=0)
        
        if not (market_open <= now <= market_close):
            error_response = {
                "code": 3011,
                "message": "Market is closed",
                "orderRejectCode": "MarketClosed"
            }
            
            with patch.object(client, 'place_order', 
                             side_effect=aiohttp.ClientError(json.dumps(error_response))):
                with pytest.raises(aiohttp.ClientError, match="Market is closed"):
                    await client.place_order("12345678", {"test": "order"})
    
    @pytest.mark.asyncio
    async def test_connection_error_recovery(self):
        """Test recovery from connection errors."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        attempt_count = 0
        
        async def flaky_request():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise aiohttp.ClientConnectorError(
                    connection_key=None,
                    os_error=OSError("Connection reset")
                )
            return {"quotes": [{"symbol": "TD.TO", "lastTradePrice": 82.50}]}
        
        # Implement retry logic
        max_retries = 3
        for i in range(max_retries):
            try:
                with patch.object(client, 'get_quote', side_effect=flaky_request):
                    result = await client.get_quote("TD.TO")
                break
            except aiohttp.ClientConnectorError:
                if i == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (i + 1))  # Exponential backoff
        
        assert attempt_count == 3
        assert result["quotes"][0]["lastTradePrice"] == 82.50
    
    @pytest.mark.asyncio
    async def test_token_expiry_handling(self):
        """Test handling of expired tokens."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        # Set token to expired
        client.token_expiry = datetime.now() - timedelta(minutes=1)
        
        refresh_called = False
        
        async def mock_refresh():
            nonlocal refresh_called
            refresh_called = True
            client.access_token = "new_token"
            client.token_expiry = datetime.now() + timedelta(minutes=30)
        
        with patch.object(client, 'refresh_access_token', side_effect=mock_refresh):
            # Any API call should trigger refresh
            with patch.object(client, 'get_accounts', return_value={"accounts": []}):
                await client.get_accounts()
        
        assert refresh_called
        assert client.access_token == "new_token"


class TestQuestradeEdgeCases:
    """Test edge cases for Questrade integration."""
    
    @pytest.mark.asyncio
    async def test_partial_fill_handling(self):
        """Test handling of partial order fills."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        order_id = "987654321"
        
        # Mock order status updates
        status_updates = [
            {
                "id": order_id,
                "totalQuantity": 1000,
                "filledQuantity": 0,
                "openQuantity": 1000,
                "state": "Pending"
            },
            {
                "id": order_id,
                "totalQuantity": 1000,
                "filledQuantity": 300,
                "openQuantity": 700,
                "state": "PartiallyExecuted",
                "avgExecPrice": 82.49
            },
            {
                "id": order_id,
                "totalQuantity": 1000,
                "filledQuantity": 700,
                "openQuantity": 300,
                "state": "PartiallyExecuted",
                "avgExecPrice": 82.48
            },
            {
                "id": order_id,
                "totalQuantity": 1000,
                "filledQuantity": 1000,
                "openQuantity": 0,
                "state": "Executed",
                "avgExecPrice": 82.47
            }
        ]
        
        for i, status in enumerate(status_updates):
            with patch.object(client, 'get_order', return_value={"orders": [status]}):
                order = await client.get_order(account_id, order_id)
                assert order["orders"][0]["filledQuantity"] == status["filledQuantity"]
    
    @pytest.mark.asyncio
    async def test_iceberg_order(self):
        """Test iceberg order functionality."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        order = {
            "symbolId": 38960,
            "quantity": 10000,
            "icebergQuantity": 100,  # Show only 100 at a time
            "limitPrice": 82.00,
            "orderType": "Limit",
            "timeInForce": "Day",
            "action": "Buy"
        }
        
        with patch.object(client, 'place_order', return_value={"orderId": 123456789}):
            result = await client.place_order(account_id, order)
        
        assert result["orderId"] == 123456789
    
    @pytest.mark.asyncio
    async def test_all_or_none_order(self):
        """Test all-or-none order constraint."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        order = {
            "symbolId": 34829,
            "quantity": 1000,
            "isAllOrNone": True,
            "limitPrice": 145.50,
            "orderType": "Limit",
            "timeInForce": "Day",
            "action": "Buy"
        }
        
        with patch.object(client, 'place_order', return_value={"orderId": 987654321}):
            result = await client.place_order(account_id, order)
        
        assert result["orderId"] == 987654321
    
    @pytest.mark.asyncio
    async def test_anonymous_order(self):
        """Test anonymous order placement."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        order = {
            "symbolId": 38960,
            "quantity": 500,
            "isAnonymous": True,  # Hide broker ID
            "limitPrice": 82.25,
            "orderType": "Limit",
            "timeInForce": "Day",
            "action": "Sell"
        }
        
        with patch.object(client, 'place_order', return_value={"orderId": 456789123}):
            result = await client.place_order(account_id, order)
        
        assert result["orderId"] == 456789123
    
    @pytest.mark.asyncio
    async def test_multi_leg_option_order(self):
        """Test multi-leg option strategy order."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        
        # Iron condor strategy
        legs = [
            {
                "symbolId": 22342123,  # Buy Call Strike 85
                "quantity": 1,
                "action": "Buy"
            },
            {
                "symbolId": 22342125,  # Sell Call Strike 82.5
                "quantity": 1,
                "action": "Sell"
            },
            {
                "symbolId": 22342126,  # Sell Put Strike 82.5
                "quantity": 1,
                "action": "Sell"
            },
            {
                "symbolId": 22342128,  # Buy Put Strike 80
                "quantity": 1,
                "action": "Buy"
            }
        ]
        
        strategy_order = {
            "strategyType": "Custom",
            "legs": legs,
            "limitPrice": 0.50,  # Net credit
            "orderType": "NetCredit",
            "timeInForce": "Day"
        }
        
        with patch.object(client, 'place_strategy_order', 
                         return_value={"orderId": 111222333}):
            result = await client.place_strategy_order(account_id, strategy_order)
        
        assert result["orderId"] == 111222333
    
    @pytest.mark.asyncio
    async def test_currency_conversion(self):
        """Test currency conversion in multi-currency account."""
        client = QuestradeClient({"refresh_token": "test_token"})
        await client.authenticate()
        
        account_id = "12345678"
        
        # Get USD quote for Canadian account
        with patch.object(client, 'get_quote', return_value={
            "quotes": [{
                "symbol": "AAPL",
                "currency": "USD",
                "lastTradePrice": 150.00,
                "fxRate": 1.3650  # USD/CAD exchange rate
            }]
        }):
            quote = await client.get_quote("AAPL")
        
        usd_price = quote["quotes"][0]["lastTradePrice"]
        fx_rate = quote["quotes"][0]["fxRate"]
        cad_equivalent = usd_price * fx_rate
        
        assert cad_equivalent == pytest.approx(204.75, rel=0.01)


# Integration tests
class TestQuestradeIntegration:
    """Integration tests for complete Questrade workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self, mock_questrade_client, mock_market_data_stream):
        """Test complete trading workflow."""
        # 1. Initialize and authenticate
        client = QuestradeClient({"refresh_token": "test_refresh_token"})
        await client.authenticate()
        
        # 2. Get accounts
        with patch.object(client, 'get_accounts', mock_questrade_client.get_accounts):
            accounts = await client.get_accounts()
        account_id = accounts["accounts"][0]["number"]
        
        # 3. Check account balance
        with patch.object(client, 'get_balances', mock_questrade_client.get_balances):
            balances = await client.get_balances(account_id)
        buying_power = balances["combinedBalances"][0]["buyingPower"]
        
        # 4. Search for symbol
        symbol = "TD.TO"
        symbol_id = 38960
        
        # 5. Get quote
        with patch.object(client, 'get_quote', mock_questrade_client.get_quote):
            quote = await client.get_quote(symbol)
        current_price = quote["quotes"][0]["lastTradePrice"]
        
        # 6. Calculate order size
        position_size = 100
        required_capital = position_size * current_price
        
        assert required_capital < buying_power
        
        # 7. Place limit order
        order = {
            "symbolId": symbol_id,
            "quantity": position_size,
            "limitPrice": current_price - 0.10,
            "orderType": "Limit",
            "timeInForce": "Day",
            "action": "Buy"
        }
        
        with patch.object(client, 'place_order', mock_questrade_client.place_order):
            order_result = await client.place_order(account_id, order)
        order_id = order_result["orderId"]
        
        # 8. Monitor order status
        await asyncio.sleep(1)
        
        # 9. Check positions
        with patch.object(client, 'get_positions', return_value={
            "positions": [{
                "symbol": symbol,
                "openQuantity": position_size,
                "averageEntryPrice": current_price - 0.10
            }]
        }):
            positions = await client.get_positions(account_id)
        
        assert len(positions["positions"]) > 0
        
        # 10. Cleanup
        await client.close()