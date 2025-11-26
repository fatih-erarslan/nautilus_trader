"""
Locust performance testing for AI News Trading Platform.

This file defines load testing scenarios for the trading platform API
and core functionality using Locust.
"""

import json
import random
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser


class TradingPlatformUser(FastHttpUser):
    """Base user class for trading platform testing."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    weight = 3  # Relative weight for user selection
    
    def on_start(self):
        """Initialize user session."""
        self.auth_token = self.login()
        self.portfolio_id = None
        
    def login(self):
        """Simulate user login."""
        response = self.client.post("/api/v1/auth/login", json={
            "username": f"test_user_{random.randint(1, 1000)}",
            "password": "test_password"
        })
        
        if response.status_code == 200:
            return response.json().get("token", "fake_token")
        return "fake_token"
    
    @task(5)
    def get_market_data(self):
        """Fetch market data for various symbols."""
        symbols = ["BTC_USDT", "ETH_USDT", "XRP_USDT", "ADA_USDT"]
        symbol = random.choice(symbols)
        
        with self.client.get(
            f"/api/v1/market/{symbol}",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            catch_response=True,
            name="/api/v1/market/[symbol]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "price" in data and "volume" in data:
                    response.success()
                else:
                    response.failure("Invalid market data response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(3)
    def get_news_feed(self):
        """Fetch latest news."""
        with self.client.get(
            "/api/v1/news/latest",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                news = response.json()
                if isinstance(news, list) and len(news) > 0:
                    response.success()
                else:
                    response.failure("Empty news response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def analyze_sentiment(self):
        """Request sentiment analysis."""
        sample_text = random.choice([
            "Bitcoin price is surging to new heights",
            "Market volatility creates uncertainty",
            "Institutional adoption drives growth",
            "Regulatory concerns weigh on crypto markets"
        ])
        
        with self.client.post(
            "/api/v1/analysis/sentiment",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            json={"text": sample_text},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "sentiment" in result and "confidence" in result:
                    response.success()
                else:
                    response.failure("Invalid sentiment analysis response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def get_portfolio(self):
        """Get user portfolio."""
        with self.client.get(
            "/api/v1/portfolio",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                portfolio = response.json()
                self.portfolio_id = portfolio.get("id")
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def place_order(self):
        """Place a trading order."""
        symbols = ["BTC_USDT", "ETH_USDT", "XRP_USDT"]
        order_data = {
            "symbol": random.choice(symbols),
            "side": random.choice(["buy", "sell"]),
            "quantity": round(random.uniform(0.01, 1.0), 4),
            "order_type": "limit",
            "price": round(random.uniform(40000, 60000), 2)
        }
        
        with self.client.post(
            "/api/v1/orders",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            json=order_data,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                order = response.json()
                if "order_id" in order:
                    response.success()
                else:
                    response.failure("Invalid order response")
            else:
                response.failure(f"HTTP {response.status_code}")


class HighFrequencyTrader(FastHttpUser):
    """High-frequency trading user for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Very fast requests
    weight = 1  # Lower weight for stress testing
    
    def on_start(self):
        """Initialize HFT session."""
        self.auth_token = "hft_token"
        self.active_orders = []
    
    @task(10)
    def rapid_market_data_requests(self):
        """Make rapid market data requests."""
        symbol = "BTC_USDT"  # Focus on one symbol for HFT
        
        with self.client.get(
            f"/api/v1/market/{symbol}/orderbook",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            catch_response=True,
            name="/api/v1/market/[symbol]/orderbook"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(3)
    def rapid_order_placement(self):
        """Place orders rapidly."""
        order_data = {
            "symbol": "BTC_USDT",
            "side": random.choice(["buy", "sell"]),
            "quantity": 0.01,
            "order_type": "limit",
            "price": round(random.uniform(49900, 50100), 2)
        }
        
        with self.client.post(
            "/api/v1/orders",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            json=order_data,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                order = response.json()
                self.active_orders.append(order.get("order_id"))
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def cancel_orders(self):
        """Cancel existing orders."""
        if self.active_orders:
            order_id = self.active_orders.pop()
            with self.client.delete(
                f"/api/v1/orders/{order_id}",
                headers={"Authorization": f"Bearer {self.auth_token}"},
                catch_response=True,
                name="/api/v1/orders/[order_id]"
            ) as response:
                if response.status_code in [200, 204]:
                    response.success()
                else:
                    response.failure(f"HTTP {response.status_code}")


class NewsAnalysisUser(HttpUser):
    """User focused on news and analysis endpoints."""
    
    wait_time = between(2, 5)  # Slower, more thoughtful requests
    weight = 2
    
    def on_start(self):
        """Initialize news analysis session."""
        self.auth_token = "news_token"
    
    @task(5)
    def search_news(self):
        """Search for specific news."""
        queries = ["bitcoin", "ethereum", "regulation", "adoption", "market"]
        query = random.choice(queries)
        
        with self.client.get(
            f"/api/v1/news/search?q={query}&limit=20",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            catch_response=True,
            name="/api/v1/news/search"
        ) as response:
            if response.status_code == 200:
                results = response.json()
                if isinstance(results, list):
                    response.success()
                else:
                    response.failure("Invalid search results")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(3)
    def get_market_analysis(self):
        """Get comprehensive market analysis."""
        symbols = ["BTC_USDT", "ETH_USDT"]
        symbol = random.choice(symbols)
        
        with self.client.get(
            f"/api/v1/analysis/market/{symbol}",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            catch_response=True,
            name="/api/v1/analysis/market/[symbol]"
        ) as response:
            if response.status_code == 200:
                analysis = response.json()
                if "sentiment" in analysis and "indicators" in analysis:
                    response.success()
                else:
                    response.failure("Invalid analysis response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def generate_report(self):
        """Generate trading report."""
        report_types = ["daily", "weekly", "portfolio", "market"]
        report_type = random.choice(report_types)
        
        with self.client.post(
            "/api/v1/reports/generate",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            json={"type": report_type, "format": "json"},
            catch_response=True
        ) as response:
            if response.status_code in [200, 202]:  # 202 for async generation
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


# Custom event handlers for detailed metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Track custom metrics for requests."""
    if exception:
        print(f"Request failed: {name} - {exception}")
    elif response_time > 2000:  # Log slow requests
        print(f"Slow request: {name} took {response_time}ms")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test metrics."""
    print("Starting performance test...")
    print(f"Target host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Report final test metrics."""
    print("Performance test completed")
    
    # Calculate and report custom metrics
    stats = environment.stats
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    
    print(f"Total requests: {total_requests}")
    print(f"Total failures: {total_failures}")
    print(f"Failure rate: {(total_failures/total_requests)*100:.2f}%")
    
    # Report slowest endpoints
    sorted_stats = sorted(stats.entries.items(), key=lambda x: x[1].avg_response_time, reverse=True)
    print("\nSlowest endpoints:")
    for name, stat in sorted_stats[:5]:
        print(f"  {name[1]}: {stat.avg_response_time:.2f}ms avg")


# Task sets for different testing scenarios
class SmokeTestTasks:
    """Minimal task set for smoke testing."""
    
    @task(1)
    def health_check(self):
        """Basic health check."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class LoadTestTasks:
    """Comprehensive task set for load testing."""
    
    @task(10)
    def market_data_load(self):
        """High-volume market data requests."""
        # Implementation would be similar to TradingPlatformUser.get_market_data
        pass
    
    @task(5)
    def trading_load(self):
        """Trading operation load."""
        # Implementation would be similar to TradingPlatformUser.place_order
        pass


# Configuration for different test scenarios
if __name__ == "__main__":
    print("Locust performance test configuration loaded")
    print("Available user classes:")
    print("- TradingPlatformUser: General platform usage")
    print("- HighFrequencyTrader: High-frequency trading simulation")
    print("- NewsAnalysisUser: News and analysis focused testing")
    print("\nRun with: locust -f locustfile.py --host=http://localhost:8000")