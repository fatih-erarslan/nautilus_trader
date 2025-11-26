"""
Comprehensive tests for Alpha Vantage German Stock API integration
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json

from src.trading_apis.alpha_vantage.alpha_vantage_client import AlphaVantageClient, AlphaVantageConfig
from src.trading_apis.alpha_vantage.german_stock_processor import GermanStockProcessor, GermanStockData
from src.trading_apis.alpha_vantage.alpha_vantage_trading_api import AlphaVantageTradingAPI


class TestAlphaVantageClient:
    """Test Alpha Vantage client functionality"""
    
    @pytest.fixture
    def config(self):
        return AlphaVantageConfig(
            api_key="test_api_key",
            tier="free",
            timeout=30
        )
    
    @pytest.fixture
    def client(self, config):
        return AlphaVantageClient(config)
    
    @pytest.fixture
    def mock_response_data(self):
        return {
            "Global Quote": {
                "01. Symbol": "SAP.DE",
                "02. Open": "100.00",
                "03. High": "105.00",
                "04. Low": "99.00",
                "05. Price": "102.50",
                "06. Volume": "1234567",
                "07. Latest Trading Day": "2024-01-15",
                "08. Previous Close": "101.00",
                "09. Change": "1.50",
                "10. Change Percent": "1.49%"
            }
        }
    
    def test_client_initialization(self, client):
        """Test client initialization"""
        assert client.config.api_key == "test_api_key"
        assert client.config.tier == "free"
        assert client.current_limit['calls_per_minute'] == 5
        assert client.current_limit['daily_limit'] == 500
        assert 'SAP.DE' in client.dax_symbols
    
    def test_normalize_german_symbol(self, client):
        """Test German symbol normalization"""
        assert client.normalize_german_symbol('SAP') == 'SAP.DE'
        assert client.normalize_german_symbol('SAP.DE') == 'SAP.DE'
        assert client.normalize_german_symbol('BMW', 'STUTTGART') == 'BMW.STU'
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client):
        """Test rate limiting functionality"""
        # Mock time to control rate limiting
        with patch('time.time', return_value=0):
            # Should not wait on first request
            await client._rate_limit()
            assert len(client.request_times) == 0
            
            # Add requests to simulate rate limiting
            client.request_times = [0, 10, 20, 30, 40]  # 5 requests
            
            # Should wait on 6th request
            with patch('asyncio.sleep') as mock_sleep:
                await client._rate_limit()
                mock_sleep.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_quote_success(self, client, mock_response_data):
        """Test successful quote retrieval"""
        with patch.object(client, '_make_request', return_value=mock_response_data):
            quote = await client.get_quote('SAP.DE')
            assert quote is not None
            assert quote['01. Symbol'] == 'SAP.DE'
            assert quote['05. Price'] == '102.50'
    
    @pytest.mark.asyncio
    async def test_get_quote_failure(self, client):
        """Test quote retrieval failure"""
        with patch.object(client, '_make_request', return_value=None):
            quote = await client.get_quote('INVALID.DE')
            assert quote is None
    
    @pytest.mark.asyncio
    async def test_batch_quotes(self, client, mock_response_data):
        """Test batch quote retrieval"""
        symbols = ['SAP.DE', 'BMW.DE', 'SIE.DE']
        
        with patch.object(client, 'get_quote', return_value=mock_response_data['Global Quote']):
            with patch('asyncio.sleep'):  # Mock the delay
                results = await client.get_german_stocks_batch(symbols)
                assert len(results) == 3
                assert all(symbol in results for symbol in symbols)
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check functionality"""
        with patch.object(client, 'get_quote', return_value={'test': 'data'}):
            health = await client.health_check()
            assert health is True
        
        with patch.object(client, 'get_quote', return_value=None):
            health = await client.health_check()
            assert health is False
    
    def test_rate_limit_info(self, client):
        """Test rate limit information"""
        info = client.get_rate_limit_info()
        assert 'tier' in info
        assert 'calls_per_minute' in info
        assert 'daily_limit' in info
        assert info['tier'] == 'free'


class TestGermanStockProcessor:
    """Test German stock data processor"""
    
    @pytest.fixture
    def processor(self):
        config = {
            'validation_config': {
                'min_price': 0.01,
                'max_price': 10000.0,
                'max_change_percent': 50.0
            }
        }
        return GermanStockProcessor(config)
    
    @pytest.fixture
    def sample_quote_data(self):
        return {
            "01. Symbol": "SAP.DE",
            "02. Open": "100.00",
            "03. High": "105.00",
            "04. Low": "99.00",
            "05. Price": "102.50",
            "06. Volume": "1234567",
            "07. Latest Trading Day": datetime.now().strftime("%Y-%m-%d"),
            "08. Previous Close": "101.00",
            "09. Change": "1.50",
            "10. Change Percent": "1.49%"
        }
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert 'XETRA' in processor.german_exchanges
        assert 'SAP.DE' in processor.dax_components
        assert processor.dax_components['SAP.DE']['name'] == 'SAP SE'
    
    def test_process_alpha_vantage_quote(self, processor, sample_quote_data):
        """Test quote processing"""
        result = processor.process_alpha_vantage_quote(sample_quote_data, 'SAP.DE')
        
        assert result is not None
        assert isinstance(result, GermanStockData)
        assert result.symbol == 'SAP.DE'
        assert result.price == 102.50
        assert result.change == 1.50
        assert result.change_percent == 1.49
        assert result.volume == 1234567
        assert result.currency == 'EUR'
        assert result.exchange == 'XETRA'
    
    def test_validate_stock_data(self, processor):
        """Test stock data validation"""
        # Valid data
        valid_data = GermanStockData(
            symbol='SAP.DE',
            name='SAP SE',
            exchange='XETRA',
            currency='EUR',
            price=100.0,
            change=1.0,
            change_percent=1.0,
            volume=1000000,
            market_cap=None,
            timestamp=datetime.now(),
            trading_session='market',
            raw_data={}
        )
        
        assert processor.validate_stock_data(valid_data) is True
        
        # Invalid data - price too low
        invalid_data = valid_data
        invalid_data.price = 0.001
        assert processor.validate_stock_data(invalid_data) is False
        
        # Invalid data - change too large
        invalid_data.price = 100.0
        invalid_data.change_percent = 60.0
        assert processor.validate_stock_data(invalid_data) is False
    
    def test_get_exchange_from_symbol(self, processor):
        """Test exchange detection from symbol"""
        assert processor._get_exchange_from_symbol('SAP.DE') == 'XETRA'
        assert processor._get_exchange_from_symbol('BMW.STU') == 'STUTTGART'
        assert processor._get_exchange_from_symbol('UNKNOWN') == 'XETRA'  # Default
    
    def test_trading_session_detection(self, processor):
        """Test trading session detection"""
        # Market hours (9:00-17:30 CET)
        market_time = datetime(2024, 1, 15, 10, 0)  # 10:00 AM
        assert processor._get_trading_session(market_time, 'XETRA') == 'market'
        
        # Pre-market (8:00-9:00 CET)
        pre_market_time = datetime(2024, 1, 15, 8, 30)  # 8:30 AM
        assert processor._get_trading_session(pre_market_time, 'XETRA') == 'pre_market'
        
        # Closed
        closed_time = datetime(2024, 1, 15, 6, 0)  # 6:00 AM
        assert processor._get_trading_session(closed_time, 'XETRA') == 'closed'
    
    def test_dax_components(self, processor):
        """Test DAX component functionality"""
        dax_symbols = processor.get_dax_symbols()
        assert 'SAP.DE' in dax_symbols
        assert 'BMW.DE' in dax_symbols
        assert len(dax_symbols) > 10
        
        assert processor.is_dax_component('SAP.DE') is True
        assert processor.is_dax_component('UNKNOWN.DE') is False
        
        company_info = processor.get_company_info('SAP.DE')
        assert company_info['name'] == 'SAP SE'
        assert company_info['sector'] == 'Technology'
    
    @pytest.mark.asyncio
    async def test_process_batch_quotes(self, processor, sample_quote_data):
        """Test batch quote processing"""
        raw_quotes = {
            'SAP.DE': sample_quote_data,
            'BMW.DE': {**sample_quote_data, "01. Symbol": "BMW.DE"},
            'ERROR.DE': {'error': 'No data available'}
        }
        
        results = await processor.process_batch_quotes(raw_quotes)
        assert len(results) == 2  # Two successful, one error
        assert all(isinstance(result, GermanStockData) for result in results)
    
    def test_performance_metrics(self, processor):
        """Test performance metrics calculation"""
        sample_data = [
            GermanStockData(
                symbol='SAP.DE', name='SAP SE', exchange='XETRA', currency='EUR',
                price=100.0, change=1.0, change_percent=1.0, volume=1000000,
                market_cap=None, timestamp=datetime.now(), trading_session='market', raw_data={}
            ),
            GermanStockData(
                symbol='BMW.DE', name='BMW AG', exchange='XETRA', currency='EUR',
                price=80.0, change=-2.0, change_percent=-2.5, volume=800000,
                market_cap=None, timestamp=datetime.now(), trading_session='market', raw_data={}
            )
        ]
        
        metrics = processor.calculate_performance_metrics(sample_data)
        assert metrics['total_symbols'] == 2
        assert metrics['avg_price'] == 90.0
        assert metrics['positive_movers'] == 1
        assert metrics['negative_movers'] == 1
        assert metrics['top_gainer'].symbol == 'SAP.DE'
        assert metrics['top_loser'].symbol == 'BMW.DE'


class TestAlphaVantageTradingAPI:
    """Test Alpha Vantage trading API"""
    
    @pytest.fixture
    def api_config(self):
        return {
            'credentials': {'api_key': 'test_key'},
            'settings': {'tier': 'free', 'timezone': 'Europe/Berlin'},
            'german_exchanges': [
                {'exchange': 'XETRA', 'suffix': '.DE', 'currency': 'EUR'}
            ]
        }
    
    @pytest.fixture
    def trading_api(self, api_config):
        return AlphaVantageTradingAPI(api_config)
    
    def test_api_initialization(self, trading_api):
        """Test API initialization"""
        assert trading_api.base_currency == 'EUR'
        assert trading_api.timezone == 'Europe/Berlin'
        assert isinstance(trading_api.client, AlphaVantageClient)
        assert isinstance(trading_api.processor, GermanStockProcessor)
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, trading_api):
        """Test connection and disconnection"""
        with patch.object(trading_api.client, 'connect', return_value=True):
            result = await trading_api.connect()
            assert result is True
            assert trading_api.is_connected is True
        
        with patch.object(trading_api.client, 'disconnect', return_value=True):
            result = await trading_api.disconnect()
            assert result is True
            assert trading_api.is_connected is False
    
    def test_trading_methods_not_implemented(self, trading_api):
        """Test that trading methods raise NotImplementedError"""
        from src.trading_apis.base.api_interface import OrderRequest
        
        order = OrderRequest(
            symbol='SAP.DE',
            quantity=100,
            side='buy',
            order_type='market'
        )
        
        with pytest.raises(NotImplementedError):
            asyncio.run(trading_api.place_order(order))
        
        with pytest.raises(NotImplementedError):
            asyncio.run(trading_api.cancel_order('test_id'))
        
        with pytest.raises(NotImplementedError):
            asyncio.run(trading_api.get_order_status('test_id'))
    
    @pytest.mark.asyncio
    async def test_get_market_data(self, trading_api):
        """Test market data retrieval"""
        mock_quotes = {
            'SAP.DE': {
                "01. Symbol": "SAP.DE",
                "05. Price": "102.50",
                "06. Volume": "1234567",
                "07. Latest Trading Day": "2024-01-15",
                "09. Change": "1.50",
                "10. Change Percent": "1.49%"
            }
        }
        
        with patch.object(trading_api.client, 'get_german_stocks_batch', return_value=mock_quotes):
            market_data = await trading_api.get_market_data(['SAP.DE'])
            
            assert len(market_data) == 1
            assert market_data[0].symbol == 'SAP.DE'
            assert market_data[0].last == 102.50
            assert market_data[0].volume == 1234567
            assert market_data[0].bid < market_data[0].last < market_data[0].ask
    
    @pytest.mark.asyncio
    async def test_market_data_subscription(self, trading_api):
        """Test market data subscription"""
        callback = Mock()
        
        # Test subscription
        result = await trading_api.subscribe_market_data(['SAP.DE'], callback)
        assert result is True
        assert 'SAP.DE' in trading_api.subscriptions
        
        # Test unsubscription
        result = await trading_api.unsubscribe_market_data(['SAP.DE'])
        assert result is True
        assert 'SAP.DE' not in trading_api.subscriptions
    
    @pytest.mark.asyncio
    async def test_german_stock_methods(self, trading_api):
        """Test German stock specific methods"""
        # Test fundamentals
        with patch.object(trading_api.client, 'get_company_overview', return_value={'test': 'data'}):
            fundamentals = await trading_api.get_german_stock_fundamentals('SAP.DE')
            assert fundamentals == {'test': 'data'}
        
        # Test news
        with patch.object(trading_api.client, 'get_news_sentiment', return_value={'feed': []}):
            news = await trading_api.get_german_stock_news(['SAP.DE'])
            assert news == {'feed': []}
        
        # Test EUR/USD rate
        with patch.object(trading_api.client, 'get_exchange_rate', return_value={'5. Exchange Rate': '1.0850'}):
            rate = await trading_api.get_eur_usd_rate()
            assert rate == 1.0850
    
    def test_dax_components(self, trading_api):
        """Test DAX components access"""
        dax_symbols = trading_api.get_dax_components()
        assert isinstance(dax_symbols, list)
        assert 'SAP.DE' in dax_symbols
    
    def test_rate_limit_info(self, trading_api):
        """Test rate limit information"""
        with patch.object(trading_api.client, 'get_rate_limit_info', return_value={'tier': 'free'}):
            info = trading_api.get_rate_limit_info()
            assert info['tier'] == 'free'
    
    @pytest.mark.asyncio
    async def test_health_check(self, trading_api):
        """Test health check"""
        with patch.object(trading_api.client, 'get_quote', return_value={'test': 'data'}):
            health = await trading_api.health_check()
            assert health['status'] == 'healthy'
            assert 'latency_ms' in health
            assert 'test_symbol' in health
        
        with patch.object(trading_api.client, 'get_quote', return_value=None):
            health = await trading_api.health_check()
            assert health['status'] == 'unhealthy'
    
    def test_trading_session_info(self, trading_api):
        """Test trading session information"""
        info = trading_api.get_trading_session_info()
        assert info['timezone'] == 'Europe/Berlin'
        assert info['base_currency'] == 'EUR'
        assert 'exchanges' in info
        assert 'trading_calendar' in info


# Integration tests
class TestIntegration:
    """Integration tests for the full Alpha Vantage system"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_workflow(self):
        """Test complete workflow with real API (if API key available)"""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            pytest.skip("No Alpha Vantage API key available")
        
        config = {
            'credentials': {'api_key': api_key},
            'settings': {'tier': 'free', 'timezone': 'Europe/Berlin'},
            'german_exchanges': [
                {'exchange': 'XETRA', 'suffix': '.DE', 'currency': 'EUR'}
            ]
        }
        
        api = AlphaVantageTradingAPI(config)
        
        try:
            # Test connection
            connected = await api.connect()
            assert connected is True
            
            # Test health check
            health = await api.health_check()
            assert health['status'] == 'healthy'
            
            # Test market data
            market_data = await api.get_market_data(['SAP.DE'])
            assert len(market_data) > 0
            assert market_data[0].symbol == 'SAP.DE'
            
            # Test rate limit info
            rate_info = api.get_rate_limit_info()
            assert rate_info['tier'] == 'free'
            
        finally:
            await api.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])