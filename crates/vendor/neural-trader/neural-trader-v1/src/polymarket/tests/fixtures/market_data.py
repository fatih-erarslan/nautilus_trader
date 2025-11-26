"""
Market data fixtures for comprehensive testing of Polymarket models.

Contains sample market data, order samples, position samples, and other
test data structures for thorough testing coverage.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
import uuid


class MarketDataFixtures:
    """Collection of market data fixtures for testing."""
    
    @staticmethod
    def sample_market_basic() -> Dict[str, Any]:
        """Basic market data for simple testing."""
        return {
            "id": "0x" + "a" * 40,
            "question": "Will Bitcoin reach $100,000 by end of 2024?",
            "outcomes": ["Yes", "No"],
            "end_date": datetime.now() + timedelta(days=45),
            "status": "active",
            "current_prices": {"Yes": Decimal("0.65"), "No": Decimal("0.35")},
            "created_at": datetime.now() - timedelta(days=10),
            "updated_at": datetime.now()
        }
    
    @staticmethod
    def sample_market_complex() -> Dict[str, Any]:
        """Complex market data with full metadata."""
        base_time = datetime.now()
        
        return {
            "id": "0x" + "b" * 40,
            "question": "Which cryptocurrency will have the highest market cap on January 1, 2025?",
            "outcomes": ["Bitcoin", "Ethereum", "Binance Coin", "Other"],
            "end_date": base_time + timedelta(days=60),
            "status": "active",
            "current_prices": {
                "Bitcoin": Decimal("0.45"),
                "Ethereum": Decimal("0.30"),
                "Binance Coin": Decimal("0.15"),
                "Other": Decimal("0.10")
            },
            "metadata": {
                "category": "Crypto",
                "subcategory": "Market Cap",
                "tags": ["crypto", "market-cap", "2025"],
                "description": "Market resolves based on CoinGecko market cap data at 00:00 UTC on January 1, 2025",
                "rules": "Winner is determined by highest market capitalization",
                "resolution_source": "CoinGecko",
                "fee_rate": Decimal("0.02"),
                "minimum_order_size": Decimal("1.00"),
                "maximum_order_size": Decimal("10000.00")
            },
            "liquidity": {
                "total_volume": Decimal("150000.75"),
                "volume_24h": Decimal("12500.25"),
                "total_liquidity": Decimal("75000.50"),
                "available_liquidity": Decimal("60000.25"),
                "bid_liquidity": Decimal("30000.15"),
                "ask_liquidity": Decimal("30000.10"),
                "turnover_rate": Decimal("0.08"),
                "last_updated": base_time
            },
            "created_at": base_time - timedelta(days=15),
            "updated_at": base_time
        }
    
    @staticmethod
    def sample_markets_list(count: int = 5) -> List[Dict[str, Any]]:
        """Generate a list of sample markets."""
        markets = []
        base_time = datetime.now()
        
        categories = ["Crypto", "Politics", "Sports", "Entertainment", "Technology"]
        questions = [
            "Will {} reach new all-time high this year?",
            "Will {} be adopted by major institutions?",
            "Will {} outperform the market this quarter?",
            "Will {} face regulatory challenges?",
            "Will {} expand to new markets?"
        ]
        
        for i in range(count):
            category = categories[i % len(categories)]
            subject = f"Asset_{i+1}"
            
            market = {
                "id": f"0x{'a' * 38}{i:02d}",
                "question": questions[i % len(questions)].format(subject),
                "outcomes": ["Yes", "No"],
                "end_date": base_time + timedelta(days=30 + i * 10),
                "status": "active",
                "current_prices": {
                    "Yes": Decimal(f"{0.5 + i * 0.05:.2f}"),
                    "No": Decimal(f"{0.5 - i * 0.05:.2f}")
                },
                "metadata": {
                    "category": category,
                    "tags": [category.lower(), f"asset-{i+1}"],
                    "fee_rate": Decimal("0.02")
                },
                "created_at": base_time - timedelta(days=i),
                "updated_at": base_time
            }
            markets.append(market)
        
        return markets
    
    @staticmethod
    def sample_order_book(market_id: str, outcome: str) -> Dict[str, Any]:
        """Generate sample order book data."""
        base_time = datetime.now()
        
        # Generate realistic bid/ask levels
        bids = []
        asks = []
        
        # Create 5 bid levels (descending prices)
        for i in range(5):
            bid_price = Decimal(f"{0.60 - i * 0.02:.2f}")
            bid_size = Decimal(f"{1000 + i * 200:.2f}")
            bids.append({
                "price": bid_price,
                "size": bid_size,
                "timestamp": base_time - timedelta(seconds=i * 10)
            })
        
        # Create 5 ask levels (ascending prices)
        for i in range(5):
            ask_price = Decimal(f"{0.62 + i * 0.02:.2f}")
            ask_size = Decimal(f"{1100 + i * 250:.2f}")
            asks.append({
                "price": ask_price,
                "size": ask_size,
                "timestamp": base_time - timedelta(seconds=i * 10)
            })
        
        return {
            "market_id": market_id,
            "outcome_id": outcome,
            "bids": bids,
            "asks": asks,
            "timestamp": base_time
        }
    
    @staticmethod
    def sample_order_data(
        order_id: Optional[str] = None,
        market_id: Optional[str] = None,
        outcome: str = "Yes",
        side: str = "buy",
        price: float = 0.65,
        size: float = 100.0
    ) -> Dict[str, Any]:
        """Generate sample order data."""
        return {
            "id": order_id or f"order_{uuid.uuid4().hex[:8]}",
            "market_id": market_id or "0x" + "a" * 40,
            "outcome_id": outcome,
            "side": side,
            "type": "limit",
            "size": size,
            "price": price,
            "filled": 0.0,
            "remaining": size,
            "status": "open",
            "time_in_force": "gtc",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "fee_rate": 0.02
        }
    
    @staticmethod
    def sample_position_data(
        market_id: Optional[str] = None,
        outcome: str = "Yes",
        size: float = 100.0,
        average_price: float = 0.60,
        current_price: float = 0.65
    ) -> Dict[str, Any]:
        """Generate sample position data."""
        unrealized_pnl = (current_price - average_price) * size
        
        return {
            "market_id": market_id or "0x" + "a" * 40,
            "outcome": outcome,
            "size": size,
            "average_price": average_price,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": 0.0,
            "trades": [],
            "created_at": datetime.now() - timedelta(hours=1),
            "updated_at": datetime.now()
        }
    
    @staticmethod
    def sample_trade_data(
        trade_id: Optional[str] = None,
        market_id: Optional[str] = None,
        outcome: str = "Yes",
        side: str = "buy",
        size: float = 50.0,
        price: float = 0.64
    ) -> Dict[str, Any]:
        """Generate sample trade data."""
        return {
            "id": trade_id or f"trade_{uuid.uuid4().hex[:8]}",
            "market_id": market_id or "0x" + "a" * 40,
            "outcome": outcome,
            "side": side,
            "size": size,
            "price": price,
            "fee": size * price * 0.02,  # 2% fee
            "timestamp": datetime.now()
        }
    
    @staticmethod
    def sample_portfolio_data() -> Dict[str, Any]:
        """Generate sample portfolio data."""
        return {
            "cash_balance": Decimal("5000.00"),
            "positions": [
                MarketDataFixtures.sample_position_data(
                    market_id=f"0x{'a' * 38}{i:02d}",
                    outcome="Yes" if i % 2 == 0 else "No",
                    size=100.0 + i * 25,
                    average_price=0.55 + i * 0.05,
                    current_price=0.60 + i * 0.05
                )
                for i in range(3)
            ],
            "timestamp": datetime.now()
        }
    
    @staticmethod
    def edge_case_market_data() -> List[Dict[str, Any]]:
        """Generate edge case market data for testing boundary conditions."""
        base_time = datetime.now()
        
        return [
            # Market ending very soon
            {
                "id": "0x" + "e1" + "a" * 38,
                "question": "Market ending in 1 minute?",
                "outcomes": ["Yes", "No"],
                "end_date": base_time + timedelta(minutes=1),
                "status": "active",
                "current_prices": {"Yes": Decimal("0.99"), "No": Decimal("0.01")},
                "created_at": base_time - timedelta(hours=1),
                "updated_at": base_time
            },
            # Market with extreme prices
            {
                "id": "0x" + "e2" + "a" * 38,
                "question": "Market with extreme prices?",
                "outcomes": ["Yes", "No"],
                "end_date": base_time + timedelta(days=30),
                "status": "active",
                "current_prices": {"Yes": Decimal("0.01"), "No": Decimal("0.99")},
                "created_at": base_time - timedelta(days=5),
                "updated_at": base_time
            },
            # Multi-outcome market with many options
            {
                "id": "0x" + "e3" + "a" * 38,
                "question": "Which outcome will win? (10 options)",
                "outcomes": [f"Option_{i}" for i in range(1, 11)],
                "end_date": base_time + timedelta(days=45),
                "status": "active",
                "current_prices": {f"Option_{i}": Decimal(f"0.{i:02d}") for i in range(1, 11)},
                "created_at": base_time - timedelta(days=7),
                "updated_at": base_time
            },
            # Paused market
            {
                "id": "0x" + "e4" + "a" * 38,
                "question": "Paused market for testing?",
                "outcomes": ["Yes", "No"],
                "end_date": base_time + timedelta(days=30),
                "status": "paused",
                "current_prices": {"Yes": Decimal("0.50"), "No": Decimal("0.50")},
                "created_at": base_time - timedelta(days=2),
                "updated_at": base_time
            },
            # Resolved market
            {
                "id": "0x" + "e5" + "a" * 38,
                "question": "Resolved market for testing?",
                "outcomes": ["Yes", "No"],
                "end_date": base_time - timedelta(days=1),
                "status": "resolved",
                "current_prices": {"Yes": Decimal("1.00"), "No": Decimal("0.00")},
                "created_at": base_time - timedelta(days=10),
                "updated_at": base_time
            }
        ]
    
    @staticmethod
    def invalid_market_data_cases() -> List[Dict[str, Any]]:
        """Generate invalid market data for testing validation."""
        base_time = datetime.now()
        
        return [
            # Missing required fields
            {
                "question": "Market without ID?",
                "outcomes": ["Yes", "No"],
                "end_date": base_time + timedelta(days=30),
                "status": "active"
            },
            # Invalid prices (sum > 1)
            {
                "id": "0x" + "i1" + "a" * 38,
                "question": "Invalid price sum?",
                "outcomes": ["Yes", "No"],
                "end_date": base_time + timedelta(days=30),
                "status": "active",
                "current_prices": {"Yes": Decimal("0.75"), "No": Decimal("0.75")}
            },
            # Past end date for active market
            {
                "id": "0x" + "i2" + "a" * 38,
                "question": "Past end date active market?",
                "outcomes": ["Yes", "No"],
                "end_date": base_time - timedelta(days=1),
                "status": "active",
                "current_prices": {"Yes": Decimal("0.50"), "No": Decimal("0.50")}
            },
            # Single outcome
            {
                "id": "0x" + "i3" + "a" * 38,
                "question": "Single outcome market?",
                "outcomes": ["Only"],
                "end_date": base_time + timedelta(days=30),
                "status": "active",
                "current_prices": {"Only": Decimal("1.00")}
            },
            # Negative prices
            {
                "id": "0x" + "i4" + "a" * 38,
                "question": "Negative prices market?",
                "outcomes": ["Yes", "No"],
                "end_date": base_time + timedelta(days=30),
                "status": "active",
                "current_prices": {"Yes": Decimal("-0.25"), "No": Decimal("1.25")}
            }
        ]