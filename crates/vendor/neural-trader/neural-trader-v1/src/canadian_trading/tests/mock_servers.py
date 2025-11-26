"""
Mock servers for API testing.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import socket
from unittest.mock import MagicMock
import websocket_server
import ssl
import logging

logger = logging.getLogger(__name__)

class MockIBGateway:
    """Mock Interactive Brokers Gateway for testing."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.clients = {}
        self.market_data = {}
        self.orders = {}
        self.positions = {}
        self.executions = []
        self.is_running = False
        
        # Simulated account data
        self.account_data = {
            "DU123456": {
                "TotalCashValue": 100000.00,
                "NetLiquidation": 150000.00,
                "BuyingPower": 400000.00,
                "MaintMarginReq": 12000.00,
                "AvailableFunds": 88000.00,
                "Currency": "CAD"
            }
        }
        
        # Simulated market data
        self.market_prices = {
            "TD": {"bid": 82.49, "ask": 82.51, "last": 82.50, "volume": 1234567},
            "RY": {"bid": 145.74, "ask": 145.76, "last": 145.75, "volume": 987654},
            "BNS": {"bid": 65.99, "ask": 66.01, "last": 66.00, "volume": 765432}
        }
    
    def start(self):
        """Start mock IB Gateway server."""
        self.is_running = True
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.5)  # Allow server to start
    
    def stop(self):
        """Stop mock IB Gateway server."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _run_server(self):
        """Run the mock server."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        server_socket.settimeout(0.5)
        
        while self.is_running:
            try:
                client_socket, address = server_socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Server error: {e}")
                break
        
        server_socket.close()
    
    def _handle_client(self, client_socket, address):
        """Handle client connections."""
        client_id = f"{address[0]}:{address[1]}"
        self.clients[client_id] = {
            "socket": client_socket,
            "subscriptions": set(),
            "orders": {}
        }
        
        try:
            while self.is_running:
                data = client_socket.recv(1024)
                if not data:
                    break
                
                # Parse and handle commands
                command = data.decode('utf-8').strip()
                response = self._process_command(client_id, command)
                
                if response:
                    client_socket.send(response.encode('utf-8'))
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            del self.clients[client_id]
            client_socket.close()
    
    def _process_command(self, client_id: str, command: str) -> str:
        """Process client commands."""
        parts = command.split(',')
        cmd_type = parts[0] if parts else ""
        
        if cmd_type == "CONNECT":
            return "CONNECTED,DU123456\n"
        
        elif cmd_type == "REQ_ACCOUNT_SUMMARY":
            account_id = parts[1] if len(parts) > 1 else "DU123456"
            return self._get_account_summary(account_id)
        
        elif cmd_type == "REQ_POSITIONS":
            return self._get_positions()
        
        elif cmd_type == "REQ_MKT_DATA":
            symbol = parts[1] if len(parts) > 1 else "TD"
            return self._get_market_data(symbol)
        
        elif cmd_type == "PLACE_ORDER":
            return self._place_order(client_id, parts[1:])
        
        elif cmd_type == "REQ_OPEN_ORDERS":
            return self._get_open_orders()
        
        elif cmd_type == "CANCEL_ORDER":
            order_id = parts[1] if len(parts) > 1 else ""
            return self._cancel_order(order_id)
        
        return "ERROR,Unknown command\n"
    
    def _get_account_summary(self, account_id: str) -> str:
        """Get account summary."""
        if account_id in self.account_data:
            data = self.account_data[account_id]
            lines = []
            for key, value in data.items():
                lines.append(f"ACCOUNT_VALUE,{key},{value},{data['Currency']}")
            return "\n".join(lines) + "\n"
        return "ERROR,Account not found\n"
    
    def _get_positions(self) -> str:
        """Get positions."""
        positions = [
            "POSITION,TD,TSE,100,80.00",
            "POSITION,RY,TSE,50,140.00",
            "POSITION,BNS,TSE,75,64.50"
        ]
        return "\n".join(positions) + "\n"
    
    def _get_market_data(self, symbol: str) -> str:
        """Get market data."""
        if symbol in self.market_prices:
            data = self.market_prices[symbol]
            # Add some randomness
            data["last"] += random.uniform(-0.10, 0.10)
            data["bid"] = data["last"] - 0.01
            data["ask"] = data["last"] + 0.01
            
            return f"TICK_PRICE,{symbol},{data['bid']},{data['ask']},{data['last']},{data['volume']}\n"
        return "ERROR,Symbol not found\n"
    
    def _place_order(self, client_id: str, order_parts: List[str]) -> str:
        """Place an order."""
        order_id = str(random.randint(1000, 9999))
        
        # Simulate order placement
        time.sleep(0.05)  # 50ms latency
        
        # Random fill or partial fill
        if random.random() > 0.1:  # 90% fill rate
            status = "FILLED"
            filled_qty = order_parts[2] if len(order_parts) > 2 else "100"
        else:
            status = "SUBMITTED"
            filled_qty = "0"
        
        return f"ORDER_STATUS,{order_id},{status},{filled_qty}\n"
    
    def _get_open_orders(self) -> str:
        """Get open orders."""
        orders = [
            "OPEN_ORDER,1001,TD,BUY,100,LIMIT,82.00,SUBMITTED",
            "OPEN_ORDER,1002,RY,SELL,25,MARKET,0,PENDING"
        ]
        return "\n".join(orders) + "\n"
    
    def _cancel_order(self, order_id: str) -> str:
        """Cancel an order."""
        if random.random() > 0.1:  # 90% success rate
            return f"ORDER_CANCELLED,{order_id}\n"
        return f"ERROR,Cannot cancel order {order_id}\n"


class MockQuestradeAPI:
    """Mock Questrade REST API server."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.server = None
        self.thread = None
        self.access_tokens = {"test_token": datetime.now() + timedelta(minutes=30)}
        self.refresh_tokens = {"test_refresh_token": "new_refresh_token"}
        
        # Mock data
        self.accounts = {
            "12345678": {
                "number": "12345678",
                "type": "Margin",
                "status": "Active",
                "clientAccountType": "Individual"
            }
        }
        
        self.balances = {
            "12345678": {
                "combinedBalances": [{
                    "currency": "CAD",
                    "cash": 50000.00,
                    "marketValue": 100000.00,
                    "totalEquity": 150000.00,
                    "buyingPower": 300000.00
                }]
            }
        }
        
        self.positions = {
            "12345678": [{
                "symbol": "TD.TO",
                "symbolId": 38960,
                "openQuantity": 100,
                "currentQuantity": 100,
                "averageEntryPrice": 80.00,
                "currentMarketValue": 8250.00,
                "currentPrice": 82.50,
                "openPnl": 250.00
            }]
        }
    
    def start(self):
        """Start mock Questrade server."""
        handler = self._create_handler()
        self.server = HTTPServer(('localhost', self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.5)
    
    def stop(self):
        """Stop mock Questrade server."""
        if self.server:
            self.server.shutdown()
        if self.thread:
            self.thread.join(timeout=2)
    
    def _create_handler(self):
        """Create HTTP request handler."""
        parent = self
        
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Add latency simulation
                time.sleep(random.uniform(0.01, 0.05))
                
                if self.path == "/accounts":
                    self._send_json({"accounts": list(parent.accounts.values())})
                
                elif self.path.startswith("/accounts/") and self.path.endswith("/balances"):
                    account_id = self.path.split('/')[2]
                    if account_id in parent.balances:
                        self._send_json(parent.balances[account_id])
                    else:
                        self._send_error(404, "Account not found")
                
                elif self.path.startswith("/accounts/") and self.path.endswith("/positions"):
                    account_id = self.path.split('/')[2]
                    if account_id in parent.positions:
                        self._send_json({"positions": parent.positions[account_id]})
                    else:
                        self._send_error(404, "Account not found")
                
                elif self.path.startswith("/markets/quotes/"):
                    symbol = self.path.split('/')[-1]
                    self._send_json({
                        "quotes": [{
                            "symbol": symbol,
                            "lastTradePrice": 82.50 + random.uniform(-0.5, 0.5),
                            "bidPrice": 82.49,
                            "askPrice": 82.51,
                            "volume": random.randint(100000, 2000000)
                        }]
                    })
                
                else:
                    self._send_error(404, "Not found")
            
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Add latency simulation
                time.sleep(random.uniform(0.02, 0.08))
                
                if self.path == "/auth/token":
                    # Token refresh
                    self._send_json({
                        "access_token": f"new_token_{int(time.time())}",
                        "refresh_token": f"refresh_token_{int(time.time())}",
                        "expires_in": 1800
                    })
                
                elif self.path.startswith("/accounts/") and "/orders" in self.path:
                    # Place order
                    order_id = random.randint(100000000, 999999999)
                    self._send_json({
                        "orderId": order_id,
                        "orderNumber": f"ORD{order_id}"
                    })
                
                else:
                    self._send_error(404, "Not found")
            
            def do_DELETE(self):
                if self.path.startswith("/accounts/") and "/orders/" in self.path:
                    # Cancel order
                    self._send_json({"success": True})
                else:
                    self._send_error(404, "Not found")
            
            def _send_json(self, data):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            
            def _send_error(self, code, message):
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": message}).encode())
            
            def log_message(self, format, *args):
                pass  # Suppress logs
        
        return Handler


class MockOandaAPI:
    """Mock OANDA REST API and streaming server."""
    
    def __init__(self, api_port: int = 8081, stream_port: int = 8082):
        self.api_port = api_port
        self.stream_port = stream_port
        self.api_server = None
        self.stream_server = None
        self.api_thread = None
        self.stream_thread = None
        
        # Mock data
        self.account_data = {
            "101-001-1234567-001": {
                "id": "101-001-1234567-001",
                "currency": "CAD",
                "balance": 100000.00,
                "unrealizedPL": 1250.50,
                "realizedPL": 5420.75,
                "marginUsed": 5000.00,
                "marginAvailable": 95000.00,
                "openTradeCount": 3,
                "NAV": 101250.50
            }
        }
        
        self.positions = {
            "101-001-1234567-001": [{
                "instrument": "USD_CAD",
                "long": {
                    "units": "10000",
                    "averagePrice": "1.3600",
                    "pl": "50.00",
                    "resettablePL": "50.00",
                    "unrealizedPL": "50.00"
                },
                "short": {"units": "0"}
            }]
        }
        
        self.instruments = {
            "USD_CAD": {
                "name": "USD_CAD",
                "type": "CURRENCY",
                "displayName": "USD/CAD",
                "pipLocation": -4,
                "displayPrecision": 5,
                "minimumTradeSize": "1"
            }
        }
    
    def start(self):
        """Start mock OANDA servers."""
        # Start REST API server
        handler = self._create_api_handler()
        self.api_server = HTTPServer(('localhost', self.api_port), handler)
        self.api_thread = threading.Thread(target=self.api_server.serve_forever)
        self.api_thread.daemon = True
        self.api_thread.start()
        
        # Start streaming server
        self.stream_thread = threading.Thread(target=self._run_stream_server)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
        time.sleep(0.5)
    
    def stop(self):
        """Stop mock OANDA servers."""
        if self.api_server:
            self.api_server.shutdown()
        if self.api_thread:
            self.api_thread.join(timeout=2)
        if self.stream_thread:
            # Stream server will stop when thread ends
            self.stream_thread.join(timeout=2)
    
    def _create_api_handler(self):
        """Create HTTP request handler for REST API."""
        parent = self
        
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Add latency simulation
                time.sleep(random.uniform(0.01, 0.05))
                
                if self.path.startswith("/v3/accounts/") and self.path.endswith("/summary"):
                    account_id = self.path.split('/')[3]
                    if account_id in parent.account_data:
                        self._send_json({"account": parent.account_data[account_id]})
                    else:
                        self._send_error(404, "Account not found")
                
                elif self.path.startswith("/v3/accounts/") and "/positions" in self.path:
                    account_id = self.path.split('/')[3]
                    if account_id in parent.positions:
                        self._send_json({"positions": parent.positions[account_id]})
                    else:
                        self._send_error(404, "Account not found")
                
                elif self.path.startswith("/v3/accounts/") and "/pricing" in self.path:
                    # Extract instruments from query
                    instruments = ["USD_CAD", "EUR_USD", "GBP_USD"]
                    prices = []
                    for inst in instruments:
                        base_price = 1.3650 if inst == "USD_CAD" else 1.1000
                        prices.append({
                            "instrument": inst,
                            "time": datetime.now().isoformat(),
                            "bids": [{"price": str(base_price - 0.0002), "liquidity": 10000000}],
                            "asks": [{"price": str(base_price + 0.0002), "liquidity": 10000000}]
                        })
                    self._send_json({"prices": prices})
                
                elif self.path.startswith("/v3/instruments/"):
                    instrument = self.path.split('/')[-2]
                    if instrument in parent.instruments:
                        self._send_json(parent.instruments[instrument])
                    else:
                        self._send_error(404, "Instrument not found")
                
                else:
                    self._send_error(404, "Not found")
            
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Add latency simulation
                time.sleep(random.uniform(0.02, 0.08))
                
                if self.path.startswith("/v3/accounts/") and "/orders" in self.path:
                    # Create order
                    order_data = json.loads(post_data)
                    order_id = str(random.randint(1000, 9999))
                    
                    response = {
                        "orderCreateTransaction": {
                            "id": order_id,
                            "accountID": self.path.split('/')[3],
                            "type": "MARKET_ORDER",
                            "instrument": order_data.get("order", {}).get("instrument", "USD_CAD"),
                            "units": order_data.get("order", {}).get("units", "1000"),
                            "time": datetime.now().isoformat()
                        },
                        "orderFillTransaction": {
                            "id": str(int(order_id) + 1),
                            "orderID": order_id,
                            "instrument": order_data.get("order", {}).get("instrument", "USD_CAD"),
                            "units": order_data.get("order", {}).get("units", "1000"),
                            "price": "1.3650",
                            "pl": "0.00"
                        }
                    }
                    self._send_json(response)
                
                else:
                    self._send_error(404, "Not found")
            
            def do_PUT(self):
                if self.path.startswith("/v3/accounts/") and "/orders/" in self.path:
                    # Cancel order
                    self._send_json({
                        "orderCancelTransaction": {
                            "id": str(random.randint(10000, 99999)),
                            "type": "ORDER_CANCEL",
                            "orderID": self.path.split('/')[-2]
                        }
                    })
                else:
                    self._send_error(404, "Not found")
            
            def _send_json(self, data):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            
            def _send_error(self, code, message):
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"errorMessage": message}).encode())
            
            def log_message(self, format, *args):
                pass  # Suppress logs
        
        return Handler
    
    def _run_stream_server(self):
        """Run streaming price server."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', self.stream_port))
        server_socket.listen(5)
        server_socket.settimeout(0.5)
        
        while True:
            try:
                client_socket, address = server_socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_stream_client,
                    args=(client_socket,)
                )
                client_thread.daemon = True
                client_thread.start()
            except socket.timeout:
                continue
            except Exception:
                break
        
        server_socket.close()
    
    def _handle_stream_client(self, client_socket):
        """Handle streaming client."""
        try:
            while True:
                # Send price updates
                prices = {
                    "USD_CAD": 1.3650 + random.uniform(-0.001, 0.001),
                    "EUR_USD": 1.1000 + random.uniform(-0.001, 0.001)
                }
                
                for instrument, price in prices.items():
                    tick = {
                        "type": "PRICE",
                        "time": datetime.now().isoformat(),
                        "instrument": instrument,
                        "bids": [{"price": price - 0.0002, "liquidity": 10000000}],
                        "asks": [{"price": price + 0.0002, "liquidity": 10000000}]
                    }
                    
                    message = json.dumps(tick) + "\n"
                    client_socket.send(message.encode())
                
                time.sleep(0.1)  # 100ms updates
        except Exception:
            pass
        finally:
            client_socket.close()


class MockMarketDataFeed:
    """Mock real-time market data feed for testing."""
    
    def __init__(self):
        self.subscribers = {}
        self.is_running = False
        self.thread = None
        self.base_prices = {
            "TD.TO": 82.50,
            "RY.TO": 145.75,
            "BNS.TO": 66.00,
            "BMO.TO": 135.25,
            "CM.TO": 78.50,
            "USD/CAD": 1.3650,
            "EUR/CAD": 1.4950,
            "GBP/CAD": 1.7250
        }
    
    def start(self):
        """Start market data feed."""
        self.is_running = True
        self.thread = threading.Thread(target=self._generate_data)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop market data feed."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to market data."""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
    
    def unsubscribe(self, symbol: str, callback: Callable):
        """Unsubscribe from market data."""
        if symbol in self.subscribers:
            self.subscribers[symbol].remove(callback)
    
    def _generate_data(self):
        """Generate market data."""
        while self.is_running:
            for symbol, callbacks in self.subscribers.items():
                if symbol in self.base_prices:
                    # Generate realistic price movements
                    base_price = self.base_prices[symbol]
                    change_pct = random.gauss(0, 0.001)  # 0.1% standard deviation
                    new_price = base_price * (1 + change_pct)
                    
                    # Update base price with momentum
                    self.base_prices[symbol] = new_price
                    
                    # Create tick data
                    tick = {
                        "symbol": symbol,
                        "timestamp": datetime.now(),
                        "last": new_price,
                        "bid": new_price - 0.01,
                        "ask": new_price + 0.01,
                        "volume": random.randint(1000, 100000),
                        "bid_size": random.randint(100, 10000),
                        "ask_size": random.randint(100, 10000)
                    }
                    
                    # Send to subscribers
                    for callback in callbacks:
                        try:
                            callback(tick)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
            
            time.sleep(0.05)  # 50ms updates


# Helper functions for test scenarios
def simulate_network_issues():
    """Simulate various network issues."""
    scenarios = {
        "latency_spike": lambda: time.sleep(random.uniform(1, 3)),
        "packet_loss": lambda: random.random() < 0.1,  # 10% packet loss
        "connection_timeout": lambda: time.sleep(10),
        "intermittent_failure": lambda: random.random() < 0.3  # 30% failure rate
    }
    return scenarios


def generate_load_test_data(num_orders: int = 1000) -> List[Dict]:
    """Generate data for load testing."""
    symbols = ["TD.TO", "RY.TO", "BNS.TO", "BMO.TO", "CM.TO"]
    order_types = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
    sides = ["BUY", "SELL"]
    
    orders = []
    for i in range(num_orders):
        order = {
            "id": f"LOAD_TEST_{i}",
            "symbol": random.choice(symbols),
            "side": random.choice(sides),
            "quantity": random.randint(10, 1000),
            "order_type": random.choice(order_types),
            "price": round(random.uniform(50, 150), 2),
            "timestamp": datetime.now() + timedelta(seconds=i * 0.1)
        }
        orders.append(order)
    
    return orders


class ErrorInjector:
    """Inject errors for testing error handling."""
    
    def __init__(self):
        self.error_rate = 0.0
        self.error_types = [
            ConnectionError("Connection lost"),
            TimeoutError("Request timed out"),
            ValueError("Invalid parameter"),
            RuntimeError("Internal server error"),
            PermissionError("Insufficient permissions")
        ]
    
    def set_error_rate(self, rate: float):
        """Set error injection rate (0.0 to 1.0)."""
        self.error_rate = max(0.0, min(1.0, rate))
    
    def maybe_raise_error(self):
        """Randomly raise an error based on error rate."""
        if random.random() < self.error_rate:
            raise random.choice(self.error_types)
    
    def inject_response_error(self, response: Dict) -> Dict:
        """Inject error into API response."""
        if random.random() < self.error_rate:
            return {
                "error": random.choice([
                    "Rate limit exceeded",
                    "Invalid API key",
                    "Market closed",
                    "Insufficient funds",
                    "Order rejected"
                ]),
                "code": random.choice([400, 401, 403, 429, 500, 503])
            }
        return response