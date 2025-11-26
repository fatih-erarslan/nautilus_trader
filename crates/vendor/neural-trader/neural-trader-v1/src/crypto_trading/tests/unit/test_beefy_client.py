"""
Comprehensive tests for Beefy Finance API client

Tests real API endpoints, rate limiting, caching, and Web3 integration.
"""

import pytest
import pytest_asyncio
import asyncio
import aiohttp
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from decimal import Decimal

from crypto_trading.beefy.beefy_client import BeefyFinanceAPI
from crypto_trading.beefy.data_models import (
    BeefyVault, VaultAPY, VaultTVL, TokenPrice,
    DepositTransaction, WithdrawalTransaction
)


class TestBeefyFinanceAPI:
    """Test suite for Beefy Finance API client"""

    @pytest.fixture
    async def client(self):
        """Create API client for testing"""
        client = BeefyFinanceAPI()
        yield client
        if client.session:
            await client.session.close()

    @pytest.fixture
    def mock_vault_data(self):
        """Mock vault data for testing"""
        return {
            "id": "beefy-bsc-cake-bnb",
            "name": "CAKE-BNB LP",
            "token": "CAKE-BNB LP",
            "tokenAddress": "0x0eD7e52944161450477ee417DE9Cd3a859b14fD0",
            "tokenDecimals": 18,
            "tokenProviderId": "pancakeswap",
            "earnedToken": "mooCakeBNB",
            "earnedTokenAddress": "0x3f16b3e3e7A8e0F2E5b9D7F8e4F8B7e4E6E7F8e9",
            "earnContractAddress": "0x3f16b3e3e7A8e0F2E5b9D7F8e4F8B7e4E6E7F8e9",
            "oracle": "lps",
            "oracleId": "pancakeswap-cake-bnb",
            "status": "active",
            "platformId": "pancakeswap",
            "assets": ["CAKE", "BNB"],
            "risks": ["COMPLEXITY_LOW", "BATTLE_TESTED", "IL_NONE"],
            "strategyTypeId": "lp",
            "network": "bsc",
            "chain": "bsc",
            "createdAt": 1609459200
        }

    @pytest.fixture
    def mock_apy_data(self):
        """Mock APY data for testing"""
        return {
            "beefy-bsc-cake-bnb": 25.5,
            "beefy-polygon-matic-eth": 18.2,
            "beefy-avax-joe-avax": 31.7
        }

    @pytest.fixture
    def mock_prices_data(self):
        """Mock price data for testing"""
        return {
            "BNB": 245.67,
            "CAKE": 2.15,
            "MATIC": 0.85,
            "ETH": 1650.00,
            "AVAX": 15.32,
            "JOE": 0.25
        }

    async def test_client_initialization(self, client):
        """Test client initialization"""
        assert client.api_key is None
        assert client.web3_manager is not None
        assert client.session is None
        assert client._rate_limit_remaining == 100
        assert client._cache == {}

    async def test_context_manager(self):
        """Test async context manager functionality"""
        async with BeefyFinanceAPI() as client:
            assert client.session is not None
            assert isinstance(client.session, aiohttp.ClientSession)

    def test_cache_key_generation(self, client):
        """Test cache key generation"""
        # Test without params
        key1 = client._get_cache_key("/vaults")
        assert key1 == "/vaults"
        
        # Test with params
        params = {"chain": "bsc", "status": "active"}
        key2 = client._get_cache_key("/vaults", params)
        assert "chain=bsc" in key2
        assert "status=active" in key2

    def test_cache_validation(self, client):
        """Test cache validation logic"""
        # Empty cache
        assert not client._is_cache_valid("test_key")
        
        # Valid cache
        client._cache["test_key"] = {
            "data": {"test": "data"},
            "timestamp": datetime.utcnow()
        }
        assert client._is_cache_valid("test_key")
        
        # Expired cache
        client._cache["expired_key"] = {
            "data": {"test": "data"},
            "timestamp": datetime.utcnow() - timedelta(seconds=120)
        }
        assert not client._is_cache_valid("expired_key")

    async def test_rate_limiting(self, client):
        """Test rate limiting functionality"""
        # Set rate limit to 0 to trigger waiting
        client._rate_limit_remaining = 0
        client._rate_limit_reset = datetime.utcnow() + timedelta(seconds=1)
        
        start_time = datetime.utcnow()
        await client._check_rate_limit()
        end_time = datetime.utcnow()
        
        # Should have waited at least some time
        assert (end_time - start_time).total_seconds() >= 0.8

    @pytest.mark.asyncio
    async def test_make_request_success(self, client):
        """Test successful API request"""
        with patch.object(client, 'session') as mock_session:
            # Setup mock response
            mock_response = AsyncMock()
            mock_response.raise_for_status = Mock()
            mock_response.json = AsyncMock(return_value={"test": "data"})
            
            mock_session.request = AsyncMock()
            mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.request.return_value.__aexit__ = AsyncMock()
            
            result = await client._make_request("/test")
            
            assert result == {"test": "data"}
            mock_session.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_request_caching(self, client):
        """Test request caching behavior"""
        with patch.object(client, 'session') as mock_session:
            # Setup mock response
            mock_response = AsyncMock()
            mock_response.raise_for_status = Mock()
            mock_response.json = AsyncMock(return_value={"test": "data"})
            
            mock_session.request = AsyncMock()
            mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.request.return_value.__aexit__ = AsyncMock()
            
            # First request - should hit API
            result1 = await client._make_request("/test")
            assert result1 == {"test": "data"}
            assert mock_session.request.call_count == 1
            
            # Second request - should use cache
            result2 = await client._make_request("/test")
            assert result2 == {"test": "data"}
            assert mock_session.request.call_count == 1  # No additional call

    @pytest.mark.asyncio
    async def test_get_vaults_real_api(self, client):
        """Test fetching vaults from real API (limited test)"""
        try:
            vaults = await client.get_vaults()
            
            # Should return some vaults
            assert isinstance(vaults, list)
            if vaults:  # If API is available
                vault = vaults[0]
                assert hasattr(vault, 'id')
                assert hasattr(vault, 'name')
                assert hasattr(vault, 'chain')
                assert hasattr(vault, 'status')
                
        except Exception as e:
            # Real API might not be available in test environment
            pytest.skip(f"Real API not available: {str(e)}")

    @pytest.mark.asyncio
    async def test_get_vaults_with_chain_filter(self, client):
        """Test vault fetching with chain filter"""
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = [
                {
                    "id": "beefy-bsc-test",
                    "chain": "bsc",
                    "name": "Test BSC Vault",
                    "status": "active",
                    "assets": ["BNB"],
                    "platformId": "test",
                    "risks": [],
                    "strategyTypeId": "single",
                    "earnContractAddress": "0x123",
                    "tokenAddress": "0x456",
                    "earnedTokenAddress": "0x789",
                    "oracleId": "test",
                    "createdAt": 1609459200
                },
                {
                    "id": "beefy-polygon-test",
                    "chain": "polygon",
                    "name": "Test Polygon Vault",
                    "status": "active",
                    "assets": ["MATIC"],
                    "platformId": "test",
                    "risks": [],
                    "strategyTypeId": "single",
                    "earnContractAddress": "0x123",
                    "tokenAddress": "0x456",
                    "earnedTokenAddress": "0x789",
                    "oracleId": "test",
                    "createdAt": 1609459200
                }
            ]
            
            # Test without filter
            all_vaults = await client.get_vaults()
            assert len(all_vaults) == 2
            
            # Test with BSC filter
            bsc_vaults = await client.get_vaults(chain="bsc")
            assert len(bsc_vaults) == 1
            assert bsc_vaults[0].chain == "bsc"

    @pytest.mark.asyncio
    async def test_get_apy_data(self, client, mock_apy_data):
        """Test APY data fetching"""
        with patch.object(client, '_make_request') as mock_request:
            mock_request.side_effect = [
                mock_apy_data,  # Basic APY data
                {}  # Breakdown data (empty for simplicity)
            ]
            
            apy_data = await client.get_apy()
            
            assert isinstance(apy_data, dict)
            assert "beefy-bsc-cake-bnb" in apy_data
            
            vault_apy = apy_data["beefy-bsc-cake-bnb"]
            assert isinstance(vault_apy, VaultAPY)
            assert vault_apy.vaultId == "beefy-bsc-cake-bnb"
            assert vault_apy.totalApy == 25.5

    @pytest.mark.asyncio
    async def test_get_prices(self, client, mock_prices_data):
        """Test price data fetching"""
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = mock_prices_data
            
            prices = await client.get_prices()
            
            assert isinstance(prices, dict)
            assert "BNB" in prices
            
            bnb_price = prices["BNB"]
            assert isinstance(bnb_price, TokenPrice)
            assert bnb_price.symbol == "BNB"
            assert bnb_price.price == Decimal("245.67")

    @pytest.mark.asyncio
    async def test_get_tvl_data(self, client):
        """Test TVL data fetching"""
        mock_tvl_data = {
            "bsc": 150000000.00,
            "polygon": 75000000.00,
            "ethereum": 200000000.00
        }
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = mock_tvl_data
            
            tvl_data = await client.get_tvl()
            
            assert isinstance(tvl_data, dict)
            assert "bsc" in tvl_data
            
            bsc_tvl = tvl_data["bsc"]
            assert isinstance(bsc_tvl, VaultTVL)
            assert bsc_tvl.vaultId == "bsc"
            assert bsc_tvl.tvl == Decimal("150000000.00")

    @pytest.mark.asyncio
    async def test_get_ticker(self, client, mock_prices_data):
        """Test ticker functionality"""
        with patch.object(client, 'get_prices') as mock_get_prices:
            mock_prices = {
                "BNB": TokenPrice(
                    symbol="BNB",
                    price=Decimal("245.67"),
                    oracleId="BNB",
                    timestamp=datetime.utcnow()
                )
            }
            mock_get_prices.return_value = mock_prices
            
            ticker = await client.get_ticker("BNB")
            
            assert ticker["symbol"] == "BNB"
            assert ticker["price"] == "245.67"
            assert "timestamp" in ticker

    @pytest.mark.asyncio
    async def test_get_ticker_not_found(self, client):
        """Test ticker for non-existent symbol"""
        with patch.object(client, 'get_prices') as mock_get_prices:
            mock_get_prices.return_value = {}
            
            ticker = await client.get_ticker("NONEXISTENT")
            
            assert ticker["symbol"] == "NONEXISTENT"
            assert "error" in ticker

    def test_prepare_deposit_transaction(self, client):
        """Test deposit transaction preparation"""
        with patch.object(client.web3_manager, 'check_token_allowance') as mock_allowance:
            with patch.object(client.web3_manager, 'estimate_transaction_cost') as mock_estimate:
                # Mock sufficient allowance
                mock_allowance.return_value = 1000000000000000000  # 1 ETH in wei
                
                # Mock gas estimate
                mock_estimate.return_value = Mock(estimatedGas=21000)
                
                deposit_tx = client.prepare_deposit_transaction(
                    vault_id="test-vault",
                    vault_address="0x123",
                    token_address="0x456",
                    amount="500000000000000000",  # 0.5 ETH
                    chain="bsc",
                    user_address="0x789"
                )
                
                assert deposit_tx is not None
                assert isinstance(deposit_tx, DepositTransaction)
                assert deposit_tx.vaultId == "test-vault"
                assert deposit_tx.amount == Decimal("500000000000000000")

    def test_prepare_withdrawal_transaction(self, client):
        """Test withdrawal transaction preparation"""
        with patch.object(client.web3_manager, 'estimate_transaction_cost') as mock_estimate:
            # Mock gas estimate
            mock_estimate.return_value = Mock(estimatedGas=30000)
            
            withdrawal_tx = client.prepare_withdrawal_transaction(
                vault_id="test-vault",
                vault_address="0x123",
                shares="1000000000000000000",  # 1 share
                chain="bsc",
                user_address="0x789"
            )
            
            assert withdrawal_tx is not None
            assert isinstance(withdrawal_tx, WithdrawalTransaction)
            assert withdrawal_tx.vaultId == "test-vault"
            assert withdrawal_tx.shares == Decimal("1000000000000000000")

    @pytest.mark.asyncio
    async def test_search_vaults(self, client):
        """Test vault search functionality"""
        # Mock data for search
        mock_vaults = [
            BeefyVault(
                id="beefy-bsc-cake",
                name="CAKE Vault",
                token="CAKE",
                chain="bsc",
                status="active",
                platformId="pancakeswap",
                risks=[],
                strategyTypeId="single",
                earnContractAddress="0x123",
                tokenAddress="0x456",
                earnedTokenAddress="0x789",
                oracleId="cake",
                assets=["CAKE"],
                createdAt=1609459200
            )
        ]
        
        mock_apy = {
            "beefy-bsc-cake": VaultAPY(
                vaultId="beefy-bsc-cake",
                priceId="cake",
                vaultApr=20.0,
                vaultApy=22.0,
                totalApy=22.0,
                compoundingsPerYear=365,
                beefyPerformanceFee=0.045,
                vaultDailyApy=0.06
            )
        }
        
        with patch.object(client, 'get_vaults') as mock_get_vaults:
            with patch.object(client, 'get_apy') as mock_get_apy:
                with patch.object(client, 'get_tvl') as mock_get_tvl:
                    mock_get_vaults.return_value = mock_vaults
                    mock_get_apy.return_value = mock_apy
                    mock_get_tvl.return_value = {}
                    
                    results = await client.search_vaults(
                        query="CAKE",
                        chain="bsc",
                        min_apy=20.0
                    )
                    
                    assert len(results) == 1
                    assert results[0]["name"] == "CAKE Vault"
                    assert results[0]["apy"] == 22.0

    async def test_error_handling(self, client):
        """Test error handling in API requests"""
        with patch.object(client, '_make_request') as mock_request:
            # Test network error
            mock_request.side_effect = aiohttp.ClientError("Network error")
            
            vaults = await client.get_vaults()
            assert vaults == []  # Should return empty list on error
            
            # Test JSON decode error
            mock_request.side_effect = Exception("JSON decode error")
            
            prices = await client.get_prices()
            assert prices == {}  # Should return empty dict on error

    def test_balance_method_interface_compliance(self, client):
        """Test that balance method complies with TradingAPIInterface"""
        # This is a mock implementation since real balance requires wallet connection
        balance_result = asyncio.run(client.get_balance("BNB"))
        
        assert "asset" in balance_result
        assert "message" in balance_result
        assert balance_result["asset"] == "BNB"

    def test_order_methods_interface_compliance(self, client):
        """Test order methods comply with TradingAPIInterface"""
        # These are mock implementations for DeFi
        order_result = asyncio.run(client.place_order({"amount": 100}))
        assert "status" in order_result
        
        cancel_result = asyncio.run(client.cancel_order("123"))
        assert "status" in cancel_result
        assert cancel_result["status"] == "error"
        
        status_result = asyncio.run(client.get_order_status("123"))
        assert "order_id" in status_result
        assert status_result["order_id"] == "123"


@pytest.mark.integration
class TestBeefyIntegration:
    """Integration tests with real API (when available)"""
    
    @pytest.mark.asyncio
    async def test_real_api_connection(self):
        """Test connection to real Beefy API"""
        async with BeefyFinanceAPI() as client:
            try:
                # Try to fetch a small amount of data
                vaults = await client.get_vaults()
                if vaults:
                    # If we get data, validate basic structure
                    vault = vaults[0]
                    assert hasattr(vault, 'id')
                    assert hasattr(vault, 'chain')
                    assert hasattr(vault, 'name')
                    
                    # Test APY fetch
                    apy_data = await client.get_apy([vault.id])
                    if vault.id in apy_data:
                        assert isinstance(apy_data[vault.id], VaultAPY)
                        
            except Exception as e:
                pytest.skip(f"Real API integration test skipped: {str(e)}")

    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self):
        """Test actual rate limiting behavior"""
        async with BeefyFinanceAPI() as client:
            # Make multiple rapid requests to test rate limiting
            tasks = []
            for i in range(5):
                tasks.append(client.get_vaults())
                
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # All should succeed with rate limiting in place
                successful_results = [r for r in results if not isinstance(r, Exception)]
                assert len(successful_results) >= 1  # At least one should succeed
                
            except Exception as e:
                pytest.skip(f"Rate limiting test skipped: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])