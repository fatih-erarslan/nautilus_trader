"""
Comprehensive unit tests for Market model.

Tests all methods, properties, validation rules, edge cases, and error conditions
to achieve 100% coverage for the Market model class.
"""

import pytest
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional

from polymarket.models.market import (
    Market, MarketStatus, PricePoint, OrderBook, LiquidityMetrics, MarketMetadata
)


class TestPricePoint:
    """Test PricePoint model."""
    
    @pytest.mark.unit
    def test_price_point_creation_valid(self):
        """Test creating valid PricePoint instance."""
        price_point = PricePoint(
            price=Decimal("0.65"),
            size=Decimal("100.0"),
            timestamp=datetime.now()
        )
        
        assert price_point.price == Decimal("0.65")
        assert price_point.size == Decimal("100.0")
        assert isinstance(price_point.timestamp, datetime)
    
    @pytest.mark.unit
    def test_price_point_decimal_conversion(self):
        """Test automatic conversion to Decimal types."""
        price_point = PricePoint(
            price=0.65,  # float
            size="100.0",  # string
            timestamp=datetime.now()
        )
        
        assert isinstance(price_point.price, Decimal)
        assert isinstance(price_point.size, Decimal)
        assert price_point.price == Decimal("0.65")
        assert price_point.size == Decimal("100.0")
    
    @pytest.mark.unit
    def test_price_point_validation_invalid_price_range(self):
        """Test price validation for invalid ranges."""
        timestamp = datetime.now()
        
        # Price too high
        with pytest.raises(ValueError, match="Price must be between 0 and 1"):
            PricePoint(price=Decimal("1.5"), size=Decimal("100"), timestamp=timestamp)
        
        # Price negative
        with pytest.raises(ValueError, match="Price must be between 0 and 1"):
            PricePoint(price=Decimal("-0.1"), size=Decimal("100"), timestamp=timestamp)
    
    @pytest.mark.unit
    def test_price_point_validation_invalid_size(self):
        """Test size validation for invalid values."""
        timestamp = datetime.now()
        
        # Negative size
        with pytest.raises(ValueError, match="Size must be non-negative"):
            PricePoint(price=Decimal("0.5"), size=Decimal("-10"), timestamp=timestamp)
    
    @pytest.mark.unit
    def test_price_point_edge_cases(self):
        """Test edge case values."""
        timestamp = datetime.now()
        
        # Minimum valid price
        price_point_min = PricePoint(price=Decimal("0.0"), size=Decimal("0"), timestamp=timestamp)
        assert price_point_min.price == Decimal("0.0")
        assert price_point_min.size == Decimal("0")
        
        # Maximum valid price
        price_point_max = PricePoint(price=Decimal("1.0"), size=Decimal("0"), timestamp=timestamp)
        assert price_point_max.price == Decimal("1.0")


class TestOrderBook:
    """Test OrderBook model."""
    
    @pytest.mark.unit
    def test_order_book_creation(self):
        """Test creating OrderBook instance."""
        bids = [
            PricePoint(Decimal("0.60"), Decimal("100"), datetime.now()),
            PricePoint(Decimal("0.59"), Decimal("150"), datetime.now())
        ]
        asks = [
            PricePoint(Decimal("0.61"), Decimal("120"), datetime.now()),
            PricePoint(Decimal("0.62"), Decimal("180"), datetime.now())
        ]
        
        order_book = OrderBook(
            market_id="0x123",
            outcome_id="Yes",
            bids=bids,
            asks=asks
        )
        
        assert order_book.market_id == "0x123"
        assert order_book.outcome_id == "Yes"
        assert len(order_book.bids) == 2
        assert len(order_book.asks) == 2
    
    @pytest.mark.unit
    def test_order_book_sorting(self):
        """Test that order book levels are sorted correctly."""
        # Unsorted bids and asks
        bids = [
            PricePoint(Decimal("0.58"), Decimal("100"), datetime.now()),
            PricePoint(Decimal("0.60"), Decimal("150"), datetime.now()),  # Should be first
            PricePoint(Decimal("0.59"), Decimal("120"), datetime.now())
        ]
        asks = [
            PricePoint(Decimal("0.63"), Decimal("180"), datetime.now()),
            PricePoint(Decimal("0.61"), Decimal("120"), datetime.now()),  # Should be first
            PricePoint(Decimal("0.62"), Decimal("150"), datetime.now())
        ]
        
        order_book = OrderBook(
            market_id="0x123",
            outcome_id="Yes",
            bids=bids,
            asks=asks
        )
        
        # Bids should be sorted descending (highest first)
        assert order_book.bids[0].price == Decimal("0.60")
        assert order_book.bids[1].price == Decimal("0.59")
        assert order_book.bids[2].price == Decimal("0.58")
        
        # Asks should be sorted ascending (lowest first)
        assert order_book.asks[0].price == Decimal("0.61")
        assert order_book.asks[1].price == Decimal("0.62")
        assert order_book.asks[2].price == Decimal("0.63")
    
    @pytest.mark.unit
    def test_order_book_properties(self):
        """Test OrderBook computed properties."""
        bids = [
            PricePoint(Decimal("0.60"), Decimal("100"), datetime.now()),
            PricePoint(Decimal("0.59"), Decimal("150"), datetime.now())
        ]
        asks = [
            PricePoint(Decimal("0.61"), Decimal("120"), datetime.now()),
            PricePoint(Decimal("0.62"), Decimal("180"), datetime.now())
        ]
        
        order_book = OrderBook(
            market_id="0x123",
            outcome_id="Yes",
            bids=bids,
            asks=asks
        )
        
        # Test best bid/ask
        assert order_book.best_bid == Decimal("0.60")
        assert order_book.best_ask == Decimal("0.61")
        
        # Test spread
        assert order_book.spread == Decimal("0.01")
        
        # Test midpoint
        assert order_book.midpoint == Decimal("0.605")
    
    @pytest.mark.unit
    def test_order_book_empty_sides(self):
        """Test OrderBook with empty bids or asks."""
        # Empty bids
        order_book_no_bids = OrderBook(
            market_id="0x123",
            outcome_id="Yes",
            bids=[],
            asks=[PricePoint(Decimal("0.61"), Decimal("120"), datetime.now())]
        )
        
        assert order_book_no_bids.best_bid is None
        assert order_book_no_bids.best_ask == Decimal("0.61")
        assert order_book_no_bids.spread is None
        assert order_book_no_bids.midpoint is None
        
        # Empty asks
        order_book_no_asks = OrderBook(
            market_id="0x123",
            outcome_id="Yes",
            bids=[PricePoint(Decimal("0.60"), Decimal("100"), datetime.now())],
            asks=[]
        )
        
        assert order_book_no_asks.best_bid == Decimal("0.60")
        assert order_book_no_asks.best_ask is None
        assert order_book_no_asks.spread is None
        assert order_book_no_asks.midpoint is None


class TestLiquidityMetrics:
    """Test LiquidityMetrics model."""
    
    @pytest.mark.unit
    def test_liquidity_metrics_creation(self):
        """Test creating LiquidityMetrics instance."""
        metrics = LiquidityMetrics(
            total_volume=Decimal("50000"),
            volume_24h=Decimal("5000"),
            total_liquidity=Decimal("25000"),
            available_liquidity=Decimal("20000"),
            bid_liquidity=Decimal("10000"),
            ask_liquidity=Decimal("10000"),
            turnover_rate=Decimal("0.1")
        )
        
        assert metrics.total_volume == Decimal("50000")
        assert metrics.volume_24h == Decimal("5000")
        assert metrics.total_liquidity == Decimal("25000")
        assert metrics.available_liquidity == Decimal("20000")
        assert metrics.bid_liquidity == Decimal("10000")
        assert metrics.ask_liquidity == Decimal("10000")
        assert metrics.turnover_rate == Decimal("0.1")
        assert isinstance(metrics.last_updated, datetime)
    
    @pytest.mark.unit
    def test_liquidity_metrics_decimal_conversion(self):
        """Test automatic conversion to Decimal types."""
        metrics = LiquidityMetrics(
            total_volume="50000.50",  # string
            volume_24h=5000.25,       # float
            total_liquidity=25000,    # int
            available_liquidity="20000.75",
            bid_liquidity=10000.0,
            ask_liquidity="10000.25",
            turnover_rate=0.1
        )
        
        # All should be converted to Decimal
        assert isinstance(metrics.total_volume, Decimal)
        assert isinstance(metrics.volume_24h, Decimal)
        assert isinstance(metrics.total_liquidity, Decimal)
        assert isinstance(metrics.available_liquidity, Decimal)
        assert isinstance(metrics.bid_liquidity, Decimal)
        assert isinstance(metrics.ask_liquidity, Decimal)
        assert isinstance(metrics.turnover_rate, Decimal)
        
        assert metrics.total_volume == Decimal("50000.50")
        assert metrics.volume_24h == Decimal("5000.25")


class TestMarketMetadata:
    """Test MarketMetadata model."""
    
    @pytest.mark.unit
    def test_market_metadata_creation(self):
        """Test creating MarketMetadata instance."""
        metadata = MarketMetadata(
            category="Crypto",
            subcategory="Bitcoin",
            tags=["btc", "price", "prediction"],
            description="Bitcoin price prediction market",
            rules="Market resolves based on CoinGecko price data",
            resolution_source="CoinGecko",
            created_by="market_maker_1",
            fee_rate=Decimal("0.02"),
            minimum_order_size=Decimal("1.0"),
            maximum_order_size=Decimal("10000.0")
        )
        
        assert metadata.category == "Crypto"
        assert metadata.subcategory == "Bitcoin"
        assert metadata.tags == ["btc", "price", "prediction"]
        assert metadata.description == "Bitcoin price prediction market"
        assert metadata.rules == "Market resolves based on CoinGecko price data"
        assert metadata.resolution_source == "CoinGecko"
        assert metadata.created_by == "market_maker_1"
        assert metadata.fee_rate == Decimal("0.02")
        assert metadata.minimum_order_size == Decimal("1.0")
        assert metadata.maximum_order_size == Decimal("10000.0")
    
    @pytest.mark.unit
    def test_market_metadata_defaults(self):
        """Test MarketMetadata default values."""
        metadata = MarketMetadata(category="Test")
        
        assert metadata.category == "Test"
        assert metadata.subcategory is None
        assert metadata.tags == []
        assert metadata.description == ""
        assert metadata.rules == ""
        assert metadata.resolution_source == ""
        assert metadata.created_by is None
        assert metadata.fee_rate == Decimal('0.02')  # 2% default
        assert metadata.minimum_order_size == Decimal('0.01')
        assert metadata.maximum_order_size is None
        assert metadata.additional_info == {}
    
    @pytest.mark.unit
    def test_market_metadata_decimal_conversion(self):
        """Test automatic conversion to Decimal types."""
        metadata = MarketMetadata(
            category="Test",
            fee_rate="0.025",  # string
            minimum_order_size=0.5,  # float
            maximum_order_size="5000"  # string
        )
        
        assert isinstance(metadata.fee_rate, Decimal)
        assert isinstance(metadata.minimum_order_size, Decimal)
        assert isinstance(metadata.maximum_order_size, Decimal)
        
        assert metadata.fee_rate == Decimal("0.025")
        assert metadata.minimum_order_size == Decimal("0.5")
        assert metadata.maximum_order_size == Decimal("5000")


class TestMarket:
    """Test Market model."""
    
    @pytest.mark.unit
    def test_market_creation_basic(self):
        """Test creating basic Market instance."""
        market = Market(
            id="0x123456789",
            question="Will Bitcoin reach $100,000 by end of 2024?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.65"), "No": Decimal("0.35")}
        )
        
        assert market.id == "0x123456789"
        assert market.question == "Will Bitcoin reach $100,000 by end of 2024?"
        assert market.outcomes == ["Yes", "No"]
        assert market.status == MarketStatus.ACTIVE
        assert market.current_prices["Yes"] == Decimal("0.65")
        assert market.current_prices["No"] == Decimal("0.35")
        assert isinstance(market.created_at, datetime)
        assert isinstance(market.updated_at, datetime)
    
    @pytest.mark.unit
    def test_market_validation_required_fields(self):
        """Test validation of required fields."""
        base_time = datetime.now()
        
        # Missing ID
        with pytest.raises(ValueError, match="Market ID is required"):
            Market(
                id="",
                question="Test question?",
                outcomes=["Yes", "No"],
                end_date=base_time + timedelta(days=30),
                status=MarketStatus.ACTIVE
            )
        
        # Missing question
        with pytest.raises(ValueError, match="Market question is required"):
            Market(
                id="0x123",
                question="",
                outcomes=["Yes", "No"],
                end_date=base_time + timedelta(days=30),
                status=MarketStatus.ACTIVE
            )
        
        # Insufficient outcomes
        with pytest.raises(ValueError, match="Market must have at least 2 outcomes"):
            Market(
                id="0x123",
                question="Test question?",
                outcomes=["Only"],
                end_date=base_time + timedelta(days=30),
                status=MarketStatus.ACTIVE
            )
    
    @pytest.mark.unit
    def test_market_validation_end_date(self):
        """Test end date validation for active markets."""
        past_date = datetime.now() - timedelta(days=1)
        
        # Active market with past end date should fail
        with pytest.raises(ValueError, match="Active market end date must be in the future"):
            Market(
                id="0x123",
                question="Test question?",
                outcomes=["Yes", "No"],
                end_date=past_date,
                status=MarketStatus.ACTIVE
            )
        
        # Non-active market with past end date should be OK
        market = Market(
            id="0x123",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=past_date,
            status=MarketStatus.RESOLVED
        )
        assert market.status == MarketStatus.RESOLVED
    
    @pytest.mark.unit
    def test_market_price_conversion(self):
        """Test automatic price conversion to Decimal."""
        market = Market(
            id="0x123",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": 0.65, "No": "0.35"}  # mixed types
        )
        
        assert isinstance(market.current_prices["Yes"], Decimal)
        assert isinstance(market.current_prices["No"], Decimal)
        assert market.current_prices["Yes"] == Decimal("0.65")
        assert market.current_prices["No"] == Decimal("0.35")
    
    @pytest.mark.unit
    def test_market_price_validation(self):
        """Test price validation for outcomes."""
        # Price for non-existent outcome
        with pytest.raises(ValueError, match="Price outcome 'Maybe' not in market outcomes"):
            Market(
                id="0x123",
                question="Test question?",
                outcomes=["Yes", "No"],
                end_date=datetime.now() + timedelta(days=30),
                status=MarketStatus.ACTIVE,
                current_prices={"Yes": Decimal("0.65"), "Maybe": Decimal("0.35")}
            )
    
    @pytest.mark.unit
    def test_market_properties(self):
        """Test Market computed properties."""
        future_date = datetime.now() + timedelta(days=30)
        
        market = Market(
            id="0x123",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=future_date,
            status=MarketStatus.ACTIVE
        )
        
        # Test is_active
        assert market.is_active is True
        
        # Test time_to_close
        time_to_close = market.time_to_close
        assert time_to_close is not None
        assert time_to_close > 0
        
        # Test with past end date
        past_market = Market(
            id="0x456",
            question="Past question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() - timedelta(days=1),
            status=MarketStatus.RESOLVED
        )
        
        assert past_market.is_active is False
        assert past_market.time_to_close is None
    
    @pytest.mark.unit
    def test_market_liquidity_properties(self):
        """Test liquidity-related properties."""
        liquidity = LiquidityMetrics(
            total_volume=Decimal("50000"),
            volume_24h=Decimal("5000"),
            total_liquidity=Decimal("25000"),
            available_liquidity=Decimal("20000"),
            bid_liquidity=Decimal("10000"),
            ask_liquidity=Decimal("10000"),
            turnover_rate=Decimal("0.1")
        )
        
        market = Market(
            id="0x123",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE,
            liquidity=liquidity
        )
        
        assert market.total_liquidity == Decimal("25000")
        assert market.volume_24h == Decimal("5000")
        
        # Test without liquidity
        market_no_liquidity = Market(
            id="0x456",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE
        )
        
        assert market_no_liquidity.total_liquidity == Decimal("0")
        assert market_no_liquidity.volume_24h == Decimal("0")
    
    @pytest.mark.unit
    def test_market_outcome_price_methods(self):
        """Test outcome price getter methods."""
        market = Market(
            id="0x123",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.65"), "No": Decimal("0.35")}
        )
        
        # Test existing outcome
        assert market.get_outcome_price("Yes") == Decimal("0.65")
        assert market.get_outcome_price("No") == Decimal("0.35")
        
        # Test non-existent outcome
        assert market.get_outcome_price("Maybe") is None
    
    @pytest.mark.unit
    def test_market_update_price_method(self):
        """Test update_price method."""
        market = Market(
            id="0x123",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.65"), "No": Decimal("0.35")}
        )
        
        original_updated_at = market.updated_at
        
        # Valid price update
        market.update_price("Yes", Decimal("0.70"))
        assert market.current_prices["Yes"] == Decimal("0.70")
        assert market.updated_at > original_updated_at
        
        # Update with float (should convert to Decimal)
        market.update_price("No", 0.30)
        assert market.current_prices["No"] == Decimal("0.30")
        assert isinstance(market.current_prices["No"], Decimal)
        
        # Invalid outcome
        with pytest.raises(ValueError, match="Outcome 'Maybe' not found in market"):
            market.update_price("Maybe", Decimal("0.50"))
        
        # Invalid price range
        with pytest.raises(ValueError, match="Price must be between 0 and 1"):
            market.update_price("Yes", Decimal("1.50"))
        
        with pytest.raises(ValueError, match="Price must be between 0 and 1"):
            market.update_price("Yes", Decimal("-0.10"))
    
    @pytest.mark.unit
    def test_market_order_book_methods(self):
        """Test order book methods."""
        market = Market(
            id="0x123",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE
        )
        
        # Initially no order books
        assert market.get_order_book("Yes") is None
        
        # Add order book
        order_book = OrderBook(
            market_id="0x123",
            outcome_id="Yes",
            bids=[PricePoint(Decimal("0.60"), Decimal("100"), datetime.now())],
            asks=[PricePoint(Decimal("0.61"), Decimal("120"), datetime.now())]
        )
        
        original_updated_at = market.updated_at
        market.add_order_book("Yes", order_book)
        
        assert market.get_order_book("Yes") == order_book
        assert market.updated_at > original_updated_at
        
        # Invalid outcome
        with pytest.raises(ValueError, match="Outcome 'Maybe' not found in market"):
            market.add_order_book("Maybe", order_book)
    
    @pytest.mark.unit
    def test_market_serialization(self):
        """Test Market to_dict method."""
        market = Market(
            id="0x123",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.65"), "No": Decimal("0.35")}
        )
        
        market_dict = market.to_dict()
        
        assert isinstance(market_dict, dict)
        assert market_dict["id"] == "0x123"
        assert market_dict["question"] == "Test question?"
        assert market_dict["outcomes"] == ["Yes", "No"]
        assert market_dict["status"] == "active"
        assert market_dict["current_prices"]["Yes"] == 0.65  # converted to float
        assert market_dict["current_prices"]["No"] == 0.35
        assert "end_date" in market_dict
        assert "created_at" in market_dict
        assert "updated_at" in market_dict
        assert "is_active" in market_dict
        assert "time_to_close" in market_dict
        assert "total_liquidity" in market_dict
        assert "volume_24h" in market_dict
    
    @pytest.mark.unit
    def test_market_deserialization(self):
        """Test Market from_dict method."""
        market_data = {
            "id": "0x123",
            "question": "Test question?",
            "outcomes": ["Yes", "No"],
            "end_date": "2024-12-31T23:59:59Z",
            "status": "active",
            "current_prices": {"Yes": 0.65, "No": 0.35},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T12:00:00Z"
        }
        
        market = Market.from_dict(market_data)
        
        assert market.id == "0x123"
        assert market.question == "Test question?"
        assert market.outcomes == ["Yes", "No"]
        assert market.status == MarketStatus.ACTIVE
        assert market.current_prices["Yes"] == Decimal("0.65")
        assert market.current_prices["No"] == Decimal("0.35")
        assert isinstance(market.end_date, datetime)
        assert isinstance(market.created_at, datetime)
        assert isinstance(market.updated_at, datetime)
    
    @pytest.mark.unit
    def test_market_serialization_roundtrip(self):
        """Test serialization roundtrip (to_dict -> from_dict)."""
        original_market = Market(
            id="0x123",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.65"), "No": Decimal("0.35")}
        )
        
        # Serialize and deserialize
        market_dict = original_market.to_dict()
        restored_market = Market.from_dict(market_dict)
        
        # Key fields should match
        assert restored_market.id == original_market.id
        assert restored_market.question == original_market.question
        assert restored_market.outcomes == original_market.outcomes
        assert restored_market.status == original_market.status
        assert restored_market.current_prices == original_market.current_prices
    
    @pytest.mark.unit
    def test_market_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Market ending very soon
        near_future = datetime.now() + timedelta(seconds=1)
        market_ending_soon = Market(
            id="0x123",
            question="Ending soon?",
            outcomes=["Yes", "No"],
            end_date=near_future,
            status=MarketStatus.ACTIVE
        )
        
        assert market_ending_soon.is_active is True
        assert market_ending_soon.time_to_close is not None
        assert market_ending_soon.time_to_close <= 1
        
        # Multi-outcome market
        multi_market = Market(
            id="0x456",
            question="Which option?",
            outcomes=["A", "B", "C", "D", "E"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE,
            current_prices={
                "A": Decimal("0.30"),
                "B": Decimal("0.25"),
                "C": Decimal("0.20"),
                "D": Decimal("0.15"),
                "E": Decimal("0.10")
            }
        )
        
        assert len(multi_market.outcomes) == 5
        assert len(multi_market.current_prices) == 5
        assert multi_market.get_outcome_price("C") == Decimal("0.20")
    
    @pytest.mark.unit
    def test_market_different_statuses(self):
        """Test market behavior with different statuses."""
        base_data = {
            "id": "0x123",
            "question": "Test question?",
            "outcomes": ["Yes", "No"],
            "end_date": datetime.now() + timedelta(days=30)
        }
        
        # Test each status
        for status in MarketStatus:
            market = Market(**base_data, status=status)
            assert market.status == status
            
            # Only ACTIVE markets with future end dates should be active
            if status == MarketStatus.ACTIVE:
                assert market.is_active is True
            else:
                assert market.is_active is False