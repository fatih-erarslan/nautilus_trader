"""
Comprehensive tests for data feed handlers
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json

from benchmark.src.data.feeds import StockFeed, CryptoFeed, BondFeed, NewsFeed
from benchmark.src.data.realtime_manager import DataPoint
from benchmark.src.data.feeds.news_feed import NewsItem


class TestStockFeed:
    """Test stock feed functionality"""
    
    @pytest.fixture
    def stock_feed(self):
        """Create test stock feed"""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        config = {
            'enable_yahoo': True,
            'enable_finnhub': False,  # Skip API key requirement
            'enable_alpha_vantage': False,
            'yahoo_update_interval': 0.1
        }
        return StockFeed(symbols, config)
    
    @pytest.mark.asyncio
    async def test_stock_feed_initialization(self, stock_feed):
        """Test stock feed initialization"""
        assert len(stock_feed.symbols) == 3
        assert "AAPL" in stock_feed.symbols
        assert not stock_feed.is_running
    
    @pytest.mark.asyncio
    async def test_stock_feed_start_stop(self, stock_feed):
        """Test starting and stopping stock feed"""
        with patch.object(stock_feed, '_initialize_sources', new_callable=AsyncMock):
            with patch.object(stock_feed, '_connect_sources', new_callable=AsyncMock):
                with patch.object(stock_feed, '_subscribe_to_symbols', new_callable=AsyncMock):
                    await stock_feed.start()
                    assert stock_feed.is_running
                    
                    await stock_feed.stop()
                    assert not stock_feed.is_running
    
    @pytest.mark.asyncio
    async def test_stock_feed_subscribe_unsubscribe(self, stock_feed):
        """Test subscribing and unsubscribing to symbols"""
        initial_count = len(stock_feed.symbols)
        
        # Subscribe to new symbols
        new_symbols = ["TSLA", "NVDA"]
        await stock_feed.subscribe(new_symbols)
        
        assert len(stock_feed.symbols) == initial_count + 2
        assert "TSLA" in stock_feed.symbols
        
        # Unsubscribe
        await stock_feed.unsubscribe(["TSLA"])
        assert "TSLA" not in stock_feed.symbols
        assert len(stock_feed.symbols) == initial_count + 1
    
    @pytest.mark.asyncio
    async def test_stock_feed_data_handling(self, stock_feed):
        """Test handling of data points"""
        # Mock data point
        data_point = DataPoint(
            source="test_source",
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.25,
            volume=1000000,
            latency_ms=25.0
        )
        
        # Test callback
        callback_called = False
        received_data = None
        
        async def test_callback(dp):
            nonlocal callback_called, received_data
            callback_called = True
            received_data = dp
        
        stock_feed.add_callback(test_callback)
        
        # Handle data point
        await stock_feed._handle_data_point(data_point)
        
        assert callback_called
        assert received_data.symbol == "AAPL"
        assert stock_feed.get_latest_price("AAPL") == 150.25
    
    @pytest.mark.asyncio
    async def test_stock_feed_source_priority(self, stock_feed):
        """Test source priority handling"""
        # Lower priority data
        low_priority_data = DataPoint(
            source="yahoo_realtime",
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.00,
            volume=1000000
        )
        
        # Higher priority data
        high_priority_data = DataPoint(
            source="finnhub",
            symbol="AAPL",
            timestamp=datetime.now(),
            price=151.00,
            volume=1100000
        )
        
        # Add lower priority first
        await stock_feed._handle_data_point(low_priority_data)
        assert stock_feed.get_latest_price("AAPL") == 150.00
        
        # Add higher priority - should override
        await stock_feed._handle_data_point(high_priority_data)
        assert stock_feed.get_latest_price("AAPL") == 151.00
    
    @pytest.mark.asyncio
    async def test_stock_feed_symbol_validation(self, stock_feed):
        """Test symbol validation"""
        with patch.object(stock_feed, 'get_quote', new_callable=AsyncMock) as mock_quote:
            # Mock successful quote
            mock_quote.return_value = DataPoint(
                source="test",
                symbol="AAPL",
                timestamp=datetime.now(),
                price=150.00,
                volume=1000000
            )
            
            results = await stock_feed.validate_symbols(["AAPL", "INVALID"])
            
            assert results["AAPL"] is True
            # INVALID symbol should also be True since we mocked get_quote


class TestCryptoFeed:
    """Test crypto feed functionality"""
    
    @pytest.fixture
    def crypto_feed(self):
        """Create test crypto feed"""
        symbols = ["BTC-USD", "ETH-USD"]
        config = {
            'enable_coinbase': True,
            'coinbase_sandbox': True,  # Use sandbox
            'enable_order_books': True
        }
        return CryptoFeed(symbols, config)
    
    @pytest.mark.asyncio
    async def test_crypto_feed_initialization(self, crypto_feed):
        """Test crypto feed initialization"""
        assert len(crypto_feed.symbols) == 2
        assert "BTC-USD" in crypto_feed.symbols
        assert crypto_feed.enable_order_books
    
    @pytest.mark.asyncio
    async def test_crypto_symbol_normalization(self, crypto_feed):
        """Test symbol normalization"""
        assert crypto_feed._normalize_symbol("BTC/USD") == "BTC-USD"
        assert crypto_feed._normalize_symbol("btc") == "BTC-USD"
        assert crypto_feed._normalize_symbol("ETH-USD") == "ETH-USD"
    
    @pytest.mark.asyncio
    async def test_crypto_feed_order_book_update(self, crypto_feed):
        """Test order book updates"""
        data_point = DataPoint(
            source="coinbase",
            symbol="BTC-USD",
            timestamp=datetime.now(),
            price=50000.00,
            volume=1.5,
            bid=49999.50,
            ask=50000.50,
            metadata={
                'best_bid_size': 0.1,
                'best_ask_size': 0.2
            }
        )
        
        crypto_feed._update_order_book("BTC-USD", data_point)
        
        order_book = crypto_feed.get_order_book("BTC-USD")
        assert order_book is not None
        assert 49999.50 in order_book['bids']
        assert 50000.50 in order_book['asks']
    
    @pytest.mark.asyncio
    async def test_crypto_feed_spread_calculation(self, crypto_feed):
        """Test spread calculation"""
        data_point = DataPoint(
            source="coinbase",
            symbol="BTC-USD",
            timestamp=datetime.now(),
            price=50000.00,
            volume=1.5,
            bid=49999.00,
            ask=50001.00
        )
        
        await crypto_feed._handle_data_point(data_point)
        
        spread_info = crypto_feed.get_spread("BTC-USD")
        assert spread_info is not None
        assert spread_info['spread'] == 2.0
        assert spread_info['mid'] == 50000.0
    
    @pytest.mark.asyncio
    async def test_crypto_feed_24_7_market(self, crypto_feed):
        """Test crypto market is always open"""
        assert crypto_feed.is_market_open() is True


class TestBondFeed:
    """Test bond feed functionality"""
    
    @pytest.fixture
    def bond_feed(self):
        """Create test bond feed"""
        symbols = ["^TNX", "^FVX", "TLT"]
        config = {
            'enable_yahoo': True,
            'enable_alpha_vantage': False,
            'track_yield_curve': True,
            'update_interval': 1.0
        }
        return BondFeed(symbols, config)
    
    @pytest.mark.asyncio
    async def test_bond_feed_initialization(self, bond_feed):
        """Test bond feed initialization"""
        assert len(bond_feed.symbols) == 3
        assert "^TNX" in bond_feed.symbols
        assert bond_feed.track_yield_curve
    
    @pytest.mark.asyncio
    async def test_bond_feed_yield_curve_tracking(self, bond_feed):
        """Test yield curve data tracking"""
        # Mock Treasury yield data
        tnx_data = DataPoint(
            source="test",
            symbol="^TNX",
            timestamp=datetime.now(),
            price=4.5,  # 10-year yield
            volume=0
        )
        
        fvx_data = DataPoint(
            source="test",
            symbol="^FVX",
            timestamp=datetime.now(),
            price=4.2,  # 5-year yield
            volume=0
        )
        
        await bond_feed._handle_data_point(tnx_data)
        await bond_feed._handle_data_point(fvx_data)
        
        assert bond_feed.yield_curve_data["^TNX"] == 4.5
        assert bond_feed.yield_curve_data["^FVX"] == 4.2
    
    @pytest.mark.asyncio
    async def test_bond_feed_yield_curve_metrics(self, bond_feed):
        """Test yield curve metrics calculation"""
        # Mock complete yield curve
        yields = {
            "^IRX": 4.0,   # 3-month
            "^TNS": 4.1,   # 2-year
            "^FVX": 4.2,   # 5-year
            "^TNX": 4.5,   # 10-year
            "^TYX": 4.8,   # 30-year
        }
        
        for symbol, yield_value in yields.items():
            bond_feed.yield_curve_data[symbol] = yield_value
        
        metrics = bond_feed._calculate_yield_curve_metrics()
        
        assert metrics is not None
        assert metrics['slope'] == 0.8  # 30Y - 3M
        assert metrics['steepness'] == 0.4  # 10Y - 2Y
        assert metrics['curve_shape'] == 'normal'
    
    @pytest.mark.asyncio
    async def test_bond_feed_data_enhancement(self, bond_feed):
        """Test bond data enhancement"""
        # Treasury yield data
        treasury_data = DataPoint(
            source="test",
            symbol="^TNX",
            timestamp=datetime.now(),
            price=4.5,
            volume=0
        )
        
        enhanced = await bond_feed._enhance_bond_data(treasury_data)
        
        assert enhanced.metadata['instrument_type'] == 'treasury_yield'
        assert enhanced.metadata['yield_value'] == 4.5
        
        # Bond ETF data
        etf_data = DataPoint(
            source="test",
            symbol="TLT",
            timestamp=datetime.now(),
            price=95.50,
            volume=1000000
        )
        
        enhanced_etf = await bond_feed._enhance_bond_data(etf_data)
        
        assert enhanced_etf.metadata['instrument_type'] == 'bond_etf'
        assert enhanced_etf.metadata['nav'] == 95.50


class TestNewsFeed:
    """Test news feed functionality"""
    
    @pytest.fixture
    def news_feed(self):
        """Create test news feed"""
        symbols = ["AAPL", "GOOGL"]
        config = {
            'newsapi_key': 'test_key',
            'update_interval': 60,
            'enable_sentiment': True,
            'filter_financial_news': True,
            'max_cache_size': 100
        }
        return NewsFeed(symbols, config)
    
    @pytest.mark.asyncio
    async def test_news_feed_initialization(self, news_feed):
        """Test news feed initialization"""
        assert len(news_feed.symbols) == 2
        assert news_feed.enable_sentiment_analysis
        assert news_feed.filter_financial_news
    
    def test_news_relevance_filtering(self, news_feed):
        """Test news relevance filtering"""
        # Financial news
        financial_article = {
            'title': 'Apple reports strong quarterly earnings',
            'description': 'AAPL stock rises on revenue beat',
            'content': 'Apple Inc. reported earnings...'
        }
        
        assert news_feed._is_relevant_news(financial_article) is True
        
        # Non-financial news
        sports_article = {
            'title': 'Local team wins championship',
            'description': 'Sports news update',
            'content': 'The team won the game...'
        }
        
        assert news_feed._is_relevant_news(sports_article) is False
    
    def test_symbol_extraction(self, news_feed):
        """Test symbol extraction from news"""
        article = {
            'title': 'Apple (AAPL) and Google $GOOGL report earnings',
            'description': 'Tech stocks surge',
            'content': 'Both companies exceeded expectations...'
        }
        
        symbols = news_feed._extract_symbols(article)
        
        assert 'AAPL' in symbols
        assert 'GOOGL' in symbols
    
    def test_sentiment_analysis(self, news_feed):
        """Test sentiment analysis"""
        # Positive sentiment
        positive_sentiment = news_feed._analyze_sentiment(
            "Apple stock surges on strong earnings beat",
            "Company reports record profit and growth"
        )
        assert positive_sentiment > 0
        
        # Negative sentiment
        negative_sentiment = news_feed._analyze_sentiment(
            "Apple stock crashes on weak guidance",
            "Company reports significant loss and decline"
        )
        assert negative_sentiment < 0
        
        # Neutral sentiment
        neutral_sentiment = news_feed._analyze_sentiment(
            "Apple announces new product",
            "Company makes announcement"
        )
        assert abs(neutral_sentiment) < 0.5
    
    def test_news_categorization(self, news_feed):
        """Test news categorization"""
        # Earnings news
        earnings_item = NewsItem(
            title="Apple reports Q3 earnings",
            summary="Company beats revenue expectations",
            url="http://example.com",
            source="test",
            published_at=datetime.now(),
            symbols=["AAPL"]
        )
        
        category = news_feed._categorize_news(earnings_item)
        assert category == 'earnings'
        
        # Merger news
        merger_item = NewsItem(
            title="Microsoft acquires gaming company",
            summary="Major acquisition in gaming sector",
            url="http://example.com",
            source="test",
            published_at=datetime.now(),
            symbols=["MSFT"]
        )
        
        category = news_feed._categorize_news(merger_item)
        assert category == 'merger'
    
    def test_impact_score_calculation(self, news_feed):
        """Test impact score calculation"""
        # High impact news
        high_impact_item = NewsItem(
            title="Apple beats earnings by 20%",
            summary="Record quarterly results",
            url="http://example.com",
            source="test",
            published_at=datetime.now(),
            symbols=["AAPL", "GOOGL"],
            sentiment=0.8
        )
        
        impact_score = news_feed._calculate_impact_score(high_impact_item)
        assert impact_score > 0.7
        
        # Low impact news
        low_impact_item = NewsItem(
            title="Minor company update",
            summary="Routine announcement",
            url="http://example.com",
            source="test",
            published_at=datetime.now() - timedelta(days=1),
            symbols=[],
            sentiment=0.1
        )
        
        impact_score = news_feed._calculate_impact_score(low_impact_item)
        assert impact_score < 0.5
    
    @pytest.mark.asyncio
    async def test_news_feed_cache_management(self, news_feed):
        """Test news cache management"""
        # Add news items
        for i in range(5):
            item = NewsItem(
                title=f"News item {i}",
                summary=f"Summary {i}",
                url=f"http://example.com/{i}",
                source="test",
                published_at=datetime.now() - timedelta(hours=i),
                symbols=["AAPL"]
            )
            news_feed._add_to_cache(item)
        
        assert len(news_feed.news_cache) == 5
        
        # Test duplicate detection
        duplicate_item = NewsItem(
            title="News item 0",  # Same title
            summary="Different summary",
            url="http://example.com/0",
            source="test",
            published_at=datetime.now(),
            symbols=["AAPL"]
        )
        
        initial_size = len(news_feed.news_cache)
        news_feed._add_to_cache(duplicate_item)
        assert len(news_feed.news_cache) == initial_size  # No increase
    
    def test_sentiment_summary(self, news_feed):
        """Test sentiment summary calculation"""
        # Add test news items
        news_items = [
            NewsItem("Good news", "Positive", "http://1", "test", datetime.now(), ["AAPL"], sentiment=0.8),
            NewsItem("Bad news", "Negative", "http://2", "test", datetime.now(), ["AAPL"], sentiment=-0.6),
            NewsItem("Neutral news", "Neutral", "http://3", "test", datetime.now(), ["AAPL"], sentiment=0.0),
        ]
        
        for item in news_items:
            news_feed._add_to_cache(item)
        
        summary = news_feed.get_sentiment_summary("AAPL", hours=1)
        
        assert summary['count'] == 3
        assert summary['positive'] == 1
        assert summary['negative'] == 1
        assert summary['neutral'] == 1
        assert -1 <= summary['avg_sentiment'] <= 1


class TestDataIntegration:
    """Test integration between different data feeds"""
    
    @pytest.mark.asyncio
    async def test_multi_feed_coordination(self):
        """Test coordination between multiple feeds"""
        # Create feeds
        stock_feed = StockFeed(["AAPL"], {'enable_yahoo': True, 'enable_finnhub': False, 'enable_alpha_vantage': False})
        crypto_feed = CryptoFeed(["BTC-USD"], {'enable_coinbase': True, 'coinbase_sandbox': True})
        
        # Mock data coordination
        all_data = {}
        
        async def collect_data(data_point):
            all_data[f"{data_point.source}_{data_point.symbol}"] = data_point
        
        stock_feed.add_callback(collect_data)
        crypto_feed.add_callback(collect_data)
        
        # Simulate data points
        stock_data = DataPoint(
            source="yahoo_realtime",
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.0,
            volume=1000000
        )
        
        crypto_data = DataPoint(
            source="coinbase",
            symbol="BTC-USD",
            timestamp=datetime.now(),
            price=50000.0,
            volume=1.5
        )
        
        await stock_feed._handle_data_point(stock_data)
        await crypto_feed._handle_data_point(crypto_data)
        
        assert len(all_data) == 2
        assert "yahoo_realtime_AAPL" in all_data
        assert "coinbase_BTC-USD" in all_data
    
    @pytest.mark.asyncio
    async def test_cross_asset_correlation_data(self):
        """Test data for cross-asset correlation analysis"""
        # This would test getting synchronized data across asset classes
        # for correlation analysis in trading strategies
        
        stock_feed = StockFeed(["SPY"], {'enable_yahoo': True, 'enable_finnhub': False, 'enable_alpha_vantage': False})
        bond_feed = BondFeed(["^TNX"], {'enable_yahoo': True, 'enable_alpha_vantage': False})
        
        correlation_data = []
        
        async def collect_correlation_data(data_point):
            correlation_data.append({
                'timestamp': data_point.timestamp,
                'asset_class': 'stock' if 'SPY' in data_point.symbol else 'bond',
                'symbol': data_point.symbol,
                'price': data_point.price
            })
        
        stock_feed.add_callback(collect_correlation_data)
        bond_feed.add_callback(collect_correlation_data)
        
        # Simulate synchronized data
        timestamp = datetime.now()
        
        spy_data = DataPoint(
            source="yahoo_realtime",
            symbol="SPY",
            timestamp=timestamp,
            price=400.0,
            volume=1000000
        )
        
        bond_data = DataPoint(
            source="yahoo_realtime",
            symbol="^TNX",
            timestamp=timestamp,
            price=4.5,
            volume=0
        )
        
        await stock_feed._handle_data_point(spy_data)
        await bond_feed._handle_data_point(bond_data)
        
        assert len(correlation_data) == 2
        # Both data points should have same timestamp for correlation analysis
        assert correlation_data[0]['timestamp'] == correlation_data[1]['timestamp']