"""
Test data generators for Polymarket integration testing.

Provides utilities to generate test data dynamically with realistic
variations and edge cases for comprehensive testing coverage.
"""

import random
import secrets
import string
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
import uuid
from faker import Faker

fake = Faker()


class TestDataGenerator:
    """Dynamic test data generator for Polymarket entities."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            fake.seed_instance(seed)
        
        self.market_categories = [
            "Crypto", "Politics", "Sports", "Entertainment", "Technology",
            "Finance", "Climate", "Economics", "Healthcare", "Science"
        ]
        
        self.crypto_assets = [
            "Bitcoin", "Ethereum", "Binance Coin", "Cardano", "Solana",
            "Polygon", "Chainlink", "Polkadot", "Avalanche", "Cosmos"
        ]
        
        self.question_templates = [
            "Will {} reach ${:,} by {}?",
            "Will {} be adopted by {} major institutions by {}?",
            "Will {} outperform {} in {}?",
            "Will {} face regulatory approval in {} by {}?",
            "Will {} launch {} feature before {}?"
        ]
    
    def generate_market_id(self) -> str:
        """Generate a realistic market ID."""
        return "0x" + secrets.token_hex(20)
    
    def generate_order_id(self) -> str:
        """Generate a realistic order ID."""
        return f"order_{uuid.uuid4().hex[:12]}"
    
    def generate_trade_id(self) -> str:
        """Generate a realistic trade ID."""
        return f"trade_{uuid.uuid4().hex[:12]}"
    
    def generate_question(self, category: Optional[str] = None) -> str:
        """Generate a realistic market question."""
        if not category:
            category = random.choice(self.market_categories)
        
        if category == "Crypto":
            asset = random.choice(self.crypto_assets)
            price = random.randint(1000, 200000)
            date = fake.date_between(start_date='today', end_date='+1y')
            return random.choice(self.question_templates).format(
                asset, price, date.strftime("%B %Y")
            )
        elif category == "Politics":
            return f"Will {fake.name()} win the {fake.date_between(start_date='today', end_date='+2y').year} election?"
        elif category == "Sports":
            team = f"{fake.city()} {fake.word().title()}"
            return f"Will {team} win the championship this season?"
        else:
            return f"Will {fake.company()} achieve {fake.word()} by {fake.date_between(start_date='today', end_date='+1y').strftime('%B %Y')}?"
    
    def generate_market(
        self,
        market_id: Optional[str] = None,
        status: str = "active",
        outcomes: Optional[List[str]] = None,
        category: Optional[str] = None,
        include_metadata: bool = True,
        include_liquidity: bool = True
    ) -> Dict[str, Any]:
        """Generate a realistic market."""
        if not market_id:
            market_id = self.generate_market_id()
        
        if not category:
            category = random.choice(self.market_categories)
        
        if not outcomes:
            if random.random() < 0.8:  # 80% binary markets
                outcomes = ["Yes", "No"]
            else:  # Multi-outcome markets
                num_outcomes = random.randint(3, 6)
                outcomes = [f"Option_{i}" for i in range(1, num_outcomes + 1)]
        
        question = self.generate_question(category)
        
        # Generate realistic prices that sum close to 1
        if len(outcomes) == 2:
            price1 = random.uniform(0.1, 0.9)
            prices = {outcomes[0]: Decimal(f"{price1:.3f}"), outcomes[1]: Decimal(f"{1-price1:.3f}")}
        else:
            # Generate prices that sum to ~1 for multi-outcome
            raw_prices = [random.uniform(0.1, 0.8) for _ in outcomes]
            total = sum(raw_prices)
            normalized_prices = [p / total for p in raw_prices]
            prices = {outcome: Decimal(f"{price:.3f}") for outcome, price in zip(outcomes, normalized_prices)}
        
        base_time = datetime.now()
        market_data = {
            "id": market_id,
            "question": question,
            "outcomes": outcomes,
            "end_date": base_time + timedelta(days=random.randint(1, 365)),
            "status": status,
            "current_prices": prices,
            "created_at": base_time - timedelta(days=random.randint(1, 30)),
            "updated_at": base_time - timedelta(hours=random.randint(0, 24))
        }
        
        if include_metadata:
            market_data["metadata"] = {
                "category": category,
                "subcategory": fake.word().title(),
                "tags": [category.lower(), fake.word(), fake.word()],
                "description": fake.text(max_nb_chars=200),
                "rules": fake.text(max_nb_chars=100),
                "resolution_source": random.choice(["UMA", "Reality.eth", "Manual", "Oracle"]),
                "fee_rate": Decimal(str(random.choice([0.01, 0.02, 0.025, 0.03]))),
                "minimum_order_size": Decimal(str(random.choice([0.01, 0.1, 1.0]))),
                "maximum_order_size": Decimal(str(random.choice([1000, 5000, 10000])))
            }
        
        if include_liquidity:
            volume = random.uniform(1000, 500000)
            market_data["liquidity"] = {
                "total_volume": Decimal(f"{volume:.2f}"),
                "volume_24h": Decimal(f"{volume * 0.1:.2f}"),
                "total_liquidity": Decimal(f"{volume * 0.5:.2f}"),
                "available_liquidity": Decimal(f"{volume * 0.4:.2f}"),
                "bid_liquidity": Decimal(f"{volume * 0.2:.2f}"),
                "ask_liquidity": Decimal(f"{volume * 0.2:.2f}"),
                "turnover_rate": Decimal(f"{random.uniform(0.01, 0.2):.3f}"),
                "last_updated": base_time
            }
        
        return market_data
    
    def generate_order(
        self,
        market_id: Optional[str] = None,
        order_id: Optional[str] = None,
        status: str = "open",
        side: Optional[str] = None,
        order_type: str = "limit"
    ) -> Dict[str, Any]:
        """Generate a realistic order."""
        if not order_id:
            order_id = self.generate_order_id()
        
        if not market_id:
            market_id = self.generate_market_id()
        
        if not side:
            side = random.choice(["buy", "sell"])
        
        outcome = random.choice(["Yes", "No"])
        size = random.uniform(1, 1000)
        price = random.uniform(0.01, 0.99) if order_type != "market" else None
        
        base_time = datetime.now()
        
        order_data = {
            "id": order_id,
            "market_id": market_id,
            "outcome_id": outcome,
            "side": side,
            "type": order_type,
            "size": size,
            "price": price,
            "filled": 0.0 if status == "open" else random.uniform(0, size),
            "remaining": size,
            "status": status,
            "time_in_force": random.choice(["gtc", "ioc", "fok", "day"]),
            "created_at": base_time - timedelta(minutes=random.randint(1, 1440)),
            "updated_at": base_time - timedelta(minutes=random.randint(0, 60)),
            "expires_at": None if random.random() < 0.8 else base_time + timedelta(hours=random.randint(1, 24)),
            "fills": [],
            "fee_rate": 0.02,
            "client_order_id": f"client_{uuid.uuid4().hex[:8]}" if random.random() < 0.5 else None
        }
        
        # Adjust remaining based on filled
        if order_data["filled"] > 0:
            order_data["remaining"] = max(0, size - order_data["filled"])
        
        return order_data
    
    def generate_position(
        self,
        market_id: Optional[str] = None,
        outcome: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a realistic position."""
        if not market_id:
            market_id = self.generate_market_id()
        
        if not outcome:
            outcome = random.choice(["Yes", "No"])
        
        size = random.uniform(10, 1000)
        average_price = random.uniform(0.1, 0.9)
        current_price = average_price + random.uniform(-0.2, 0.2)
        current_price = max(0.01, min(0.99, current_price))  # Clamp to valid range
        
        unrealized_pnl = (current_price - average_price) * size
        realized_pnl = random.uniform(-50, 100)
        
        base_time = datetime.now()
        
        return {
            "market_id": market_id,
            "outcome": outcome,
            "size": size,
            "average_price": average_price,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "trades": self.generate_trades_for_position(market_id, outcome, size, average_price),
            "created_at": base_time - timedelta(days=random.randint(1, 30)),
            "updated_at": base_time - timedelta(hours=random.randint(0, 12))
        }
    
    def generate_trades_for_position(
        self,
        market_id: str,
        outcome: str,
        total_size: float,
        average_price: float,
        num_trades: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate trades that build up to a position."""
        if num_trades is None:
            num_trades = random.randint(1, 5)
        
        trades = []
        remaining_size = total_size
        
        for i in range(num_trades):
            # Last trade gets remaining size
            if i == num_trades - 1:
                trade_size = remaining_size
            else:
                trade_size = min(remaining_size, random.uniform(10, remaining_size * 0.6))
            
            # Price should vary around average
            price_variance = random.uniform(-0.1, 0.1)
            trade_price = max(0.01, min(0.99, average_price + price_variance))
            
            base_time = datetime.now()
            trade = {
                "id": self.generate_trade_id(),
                "market_id": market_id,
                "outcome": outcome,
                "side": "buy",  # Assuming all trades built the position
                "size": trade_size,
                "price": trade_price,
                "fee": trade_size * trade_price * 0.02,
                "timestamp": base_time - timedelta(days=random.randint(1, 30))
            }
            
            trades.append(trade)
            remaining_size -= trade_size
            
            if remaining_size <= 0:
                break
        
        return trades
    
    def generate_order_book(
        self,
        market_id: str,
        outcome: str,
        depth: int = 5,
        spread_bps: int = 200  # 2% spread in basis points
    ) -> Dict[str, Any]:
        """Generate a realistic order book."""
        mid_price = random.uniform(0.3, 0.7)
        spread = spread_bps / 10000  # Convert to decimal
        
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        bids = []
        asks = []
        
        # Generate bid levels (descending prices)
        for i in range(depth):
            price = best_bid - i * 0.01
            if price <= 0.01:
                break
            size = random.uniform(100, 2000) * (1 + i * 0.5)  # More size at worse prices
            bids.append({
                "price": Decimal(f"{price:.3f}"),
                "size": Decimal(f"{size:.2f}"),
                "timestamp": datetime.now() - timedelta(seconds=random.randint(1, 300))
            })
        
        # Generate ask levels (ascending prices)
        for i in range(depth):
            price = best_ask + i * 0.01
            if price >= 0.99:
                break
            size = random.uniform(100, 2000) * (1 + i * 0.5)
            asks.append({
                "price": Decimal(f"{price:.3f}"),
                "size": Decimal(f"{size:.2f}"),
                "timestamp": datetime.now() - timedelta(seconds=random.randint(1, 300))
            })
        
        return {
            "market_id": market_id,
            "outcome_id": outcome,
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.now()
        }
    
    def generate_portfolio(self, num_positions: int = 5) -> Dict[str, Any]:
        """Generate a realistic portfolio."""
        positions = []
        total_value = 0
        
        for _ in range(num_positions):
            position = self.generate_position()
            positions.append(position)
            total_value += position["size"] * position["current_price"]
        
        cash_balance = random.uniform(1000, 10000)
        
        return {
            "cash_balance": Decimal(f"{cash_balance:.2f}"),
            "positions": positions,
            "timestamp": datetime.now()
        }
    
    def generate_edge_case_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate edge case test data."""
        return {
            "extreme_prices": [
                self.generate_market(outcomes=["Yes", "No"]) for _ in range(3)
            ],
            "multi_outcome_markets": [
                self.generate_market(outcomes=[f"Option_{i}" for i in range(1, 8)]) for _ in range(2)
            ],
            "expiring_soon": [
                {**self.generate_market(), "end_date": datetime.now() + timedelta(minutes=random.randint(1, 60))}
                for _ in range(3)
            ],
            "large_orders": [
                {**self.generate_order(), "size": random.uniform(10000, 100000)} for _ in range(3)
            ],
            "small_orders": [
                {**self.generate_order(), "size": random.uniform(0.01, 1.0)} for _ in range(3)
            ]
        }
    
    def generate_test_scenario(
        self,
        scenario_name: str,
        num_markets: int = 3,
        num_orders: int = 5,
        num_positions: int = 2
    ) -> Dict[str, Any]:
        """Generate a complete test scenario with related data."""
        markets = [self.generate_market() for _ in range(num_markets)]
        market_ids = [market["id"] for market in markets]
        
        orders = []
        for _ in range(num_orders):
            market_id = random.choice(market_ids)
            orders.append(self.generate_order(market_id=market_id))
        
        positions = []
        for _ in range(num_positions):
            market_id = random.choice(market_ids)
            positions.append(self.generate_position(market_id=market_id))
        
        return {
            "scenario_name": scenario_name,
            "markets": markets,
            "orders": orders,
            "positions": positions,
            "portfolio": self.generate_portfolio(num_positions),
            "order_books": {
                market["id"]: self.generate_order_book(market["id"], "Yes")
                for market in markets[:2]  # Generate order books for first 2 markets
            },
            "generated_at": datetime.now().isoformat()
        }