"""
Beefy Finance API Client

This module implements the Beefy Finance API client with full Web3 integration
for interacting with yield farming vaults across multiple chains.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from web3 import Web3

from ..interfaces import TradingAPIInterface
from .data_models import (
    BeefyVault, VaultAPY, VaultTVL, TokenPrice,
    DepositTransaction, WithdrawalTransaction,
    TransactionEstimate, BeefyAPIResponse
)
from .web3_manager import Web3Manager

logger = logging.getLogger(__name__)


class BeefyFinanceAPI(TradingAPIInterface):
    """
    Beefy Finance API client implementation.
    
    Provides access to Beefy's yield farming vaults, APY data, and Web3 interactions.
    """
    
    BASE_URL = "https://api.beefy.finance"
    
    # API endpoints
    ENDPOINTS = {
        "vaults": "/vaults",
        "apy": "/apy",
        "apy_breakdown": "/apy/breakdown",
        "tvl": "/tvl",
        "prices": "/prices",
        "lps": "/lps",
        "lps_breakdown": "/lps/breakdown",
        "earnings": "/earnings",
        "holders": "/holders"
    }
    
    # Rate limiting configuration
    RATE_LIMIT_CALLS = 100
    RATE_LIMIT_PERIOD = 60  # seconds
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Beefy Finance API client.
        
        Note: Beefy Finance API is public and doesn't require authentication.
        """
        self.api_key = api_key  # Not used for Beefy, but kept for interface compatibility
        self.web3_manager = Web3Manager()
        self.session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = self.RATE_LIMIT_CALLS
        self._rate_limit_reset = datetime.utcnow()
        
        # Cache for API responses
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 60  # seconds
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for API responses."""
        if params:
            param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            return f"{endpoint}?{param_str}"
        return endpoint
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
            
        cached_data = self._cache[cache_key]
        cache_time = cached_data.get("timestamp", datetime.min)
        return (datetime.utcnow() - cache_time).total_seconds() < self._cache_ttl
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        now = datetime.utcnow()
        
        if now >= self._rate_limit_reset:
            self._rate_limit_remaining = self.RATE_LIMIT_CALLS
            self._rate_limit_reset = now + timedelta(seconds=self.RATE_LIMIT_PERIOD)
        
        if self._rate_limit_remaining <= 0:
            wait_time = (self._rate_limit_reset - now).total_seconds()
            logger.warning(f"Rate limit exceeded. Waiting {wait_time:.2f} seconds.")
            await asyncio.sleep(wait_time)
            self._rate_limit_remaining = self.RATE_LIMIT_CALLS
            
        self._rate_limit_remaining -= 1
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Any:
        """Make HTTP request to Beefy API with retry logic."""
        await self._check_rate_limit()
        
        # Check cache for GET requests
        if method == "GET":
            cache_key = self._get_cache_key(endpoint, params)
            if self._is_cache_valid(cache_key):
                logger.debug(f"Returning cached data for {endpoint}")
                return self._cache[cache_key]["data"]
        
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with self.session.request(
                method,
                url,
                params=params,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Cache successful GET responses
                if method == "GET":
                    self._cache[cache_key] = {
                        "data": data,
                        "timestamp": datetime.utcnow()
                    }
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    # TradingAPIInterface implementation
    
    async def get_balance(self, asset: str) -> Dict[str, Any]:
        """Get balance for a specific asset across all vaults."""
        # This would require wallet connection, returning mock for now
        return {
            "asset": asset,
            "free": "0",
            "used": "0",
            "total": "0",
            "message": "Wallet connection required for balance queries"
        }
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get price ticker for a symbol."""
        prices = await self.get_prices()
        
        if symbol in prices:
            return {
                "symbol": symbol,
                "price": str(prices[symbol].price),
                "timestamp": prices[symbol].timestamp.isoformat()
            }
        
        return {
            "symbol": symbol,
            "error": "Symbol not found"
        }
    
    async def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Place order (deposit to vault)."""
        # This would execute a Web3 transaction
        return {
            "status": "pending",
            "message": "Web3 wallet connection required for deposits",
            "params": order_params
        }
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Cancel order - not applicable for DeFi."""
        return {
            "status": "error",
            "message": "Order cancellation not applicable for DeFi protocols"
        }
    
    async def get_order_status(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get order status - would check transaction status."""
        return {
            "order_id": order_id,
            "status": "unknown",
            "message": "Transaction tracking requires Web3 connection"
        }
    
    # Beefy-specific methods
    
    async def get_vaults(self, chain: Optional[str] = None) -> List[BeefyVault]:
        """
        Fetch all available vaults from Beefy Finance.
        
        Args:
            chain: Optional chain filter (e.g., 'bsc', 'polygon')
            
        Returns:
            List of BeefyVault objects
        """
        try:
            data = await self._make_request(self.ENDPOINTS["vaults"])
            
            vaults = []
            for vault_data in data:
                try:
                    vault = BeefyVault(**vault_data)
                    if chain is None or vault.chain == chain:
                        vaults.append(vault)
                except Exception as e:
                    logger.warning(f"Failed to parse vault data: {str(e)}")
                    
            logger.info(f"Fetched {len(vaults)} vaults")
            return vaults
            
        except Exception as e:
            logger.error(f"Failed to fetch vaults: {str(e)}")
            return []
    
    async def get_apy(self, vault_ids: Optional[List[str]] = None) -> Dict[str, VaultAPY]:
        """
        Fetch APY data for vaults.
        
        Args:
            vault_ids: Optional list of vault IDs to filter
            
        Returns:
            Dictionary mapping vault IDs to VaultAPY objects
        """
        try:
            # Get base APY data
            apy_data = await self._make_request(self.ENDPOINTS["apy"])
            
            # Get detailed breakdown if available
            try:
                breakdown_data = await self._make_request(self.ENDPOINTS["apy_breakdown"])
            except:
                breakdown_data = {}
            
            result = {}
            
            for vault_id, apy_value in apy_data.items():
                if vault_ids and vault_id not in vault_ids:
                    continue
                    
                # Get breakdown details if available
                breakdown = breakdown_data.get(vault_id, {})
                
                vault_apy = VaultAPY(
                    vaultId=vault_id,
                    priceId=breakdown.get("priceId", vault_id),
                    vaultApr=breakdown.get("vaultApr", 0),
                    vaultApy=float(apy_value) if isinstance(apy_value, (int, float, str)) else 0,
                    compoundingsPerYear=breakdown.get("compoundingsPerYear", 365),
                    beefyPerformanceFee=breakdown.get("beefyPerformanceFee", 0.045),
                    vaultDailyApy=breakdown.get("vaultDailyApy", float(apy_value) / 365 if isinstance(apy_value, (int, float)) else 0),
                    totalApy=float(apy_value) if isinstance(apy_value, (int, float, str)) else 0,
                    tradingApr=breakdown.get("tradingApr"),
                    liquidStakingApr=breakdown.get("liquidStakingApr"),
                    composablePoolApr=breakdown.get("composablePoolApr"),
                    merklApr=breakdown.get("merklApr")
                )
                
                result[vault_id] = vault_apy
                
            logger.info(f"Fetched APY data for {len(result)} vaults")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch APY data: {str(e)}")
            return {}
    
    async def get_tvl(self, chain: Optional[str] = None) -> Dict[str, VaultTVL]:
        """
        Fetch Total Value Locked (TVL) data.
        
        Args:
            chain: Optional chain filter
            
        Returns:
            Dictionary mapping chain/protocol to TVL data
        """
        try:
            data = await self._make_request(self.ENDPOINTS["tvl"])
            
            result = {}
            
            for key, value in data.items():
                if chain and key != chain:
                    continue
                    
                tvl = VaultTVL(
                    vaultId=key,
                    tvl=Decimal(str(value)),
                    pricePerFullShare="1",  # Default value
                    timestamp=datetime.utcnow()
                )
                result[key] = tvl
                
            logger.info(f"Fetched TVL data for {len(result)} items")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch TVL data: {str(e)}")
            return {}
    
    async def get_prices(self) -> Dict[str, TokenPrice]:
        """
        Fetch current token prices.
        
        Returns:
            Dictionary mapping token symbols to TokenPrice objects
        """
        try:
            data = await self._make_request(self.ENDPOINTS["prices"])
            
            result = {}
            
            for symbol, price in data.items():
                token_price = TokenPrice(
                    symbol=symbol,
                    price=Decimal(str(price)),
                    oracleId=symbol,
                    timestamp=datetime.utcnow()
                )
                result[symbol] = token_price
                
            logger.info(f"Fetched prices for {len(result)} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch prices: {str(e)}")
            return {}
    
    def prepare_deposit_transaction(
        self,
        vault_id: str,
        vault_address: str,
        token_address: str,
        amount: Union[int, str],
        chain: str,
        user_address: str
    ) -> Optional[DepositTransaction]:
        """
        Prepare a deposit transaction for a vault.
        
        Args:
            vault_id: Vault identifier
            vault_address: Vault contract address
            token_address: Deposit token address
            amount: Amount to deposit (in wei)
            chain: Blockchain network
            user_address: User's wallet address
            
        Returns:
            DepositTransaction object or None
        """
        try:
            # Check token allowance
            allowance = self.web3_manager.check_token_allowance(
                chain=chain,
                token_address=token_address,
                owner_address=user_address,
                spender_address=vault_address
            )
            
            if allowance is None:
                logger.error("Failed to check token allowance")
                return None
                
            amount_int = int(amount)
            
            # Prepare approval if needed
            if allowance < amount_int:
                logger.info(f"Token approval needed. Current allowance: {allowance}")
                # Would return approval transaction details
                
            # Estimate gas for deposit
            estimate = self.web3_manager.estimate_transaction_cost(
                chain=chain,
                vault_address=vault_address,
                function_name="deposit",
                from_address=user_address,
                params=[amount_int]
            )
            
            return DepositTransaction(
                vaultId=vault_id,
                amount=Decimal(str(amount)),
                tokenAddress=token_address,
                vaultAddress=vault_address,
                functionName="deposit",
                functionParams=[amount_int],
                estimatedGas=estimate.estimatedGas if estimate else None
            )
            
        except Exception as e:
            logger.error(f"Failed to prepare deposit transaction: {str(e)}")
            return None
    
    def prepare_withdrawal_transaction(
        self,
        vault_id: str,
        vault_address: str,
        shares: Union[int, str],
        chain: str,
        user_address: str
    ) -> Optional[WithdrawalTransaction]:
        """
        Prepare a withdrawal transaction from a vault.
        
        Args:
            vault_id: Vault identifier
            vault_address: Vault contract address
            shares: Amount of shares to withdraw
            chain: Blockchain network
            user_address: User's wallet address
            
        Returns:
            WithdrawalTransaction object or None
        """
        try:
            shares_int = int(shares)
            
            # Estimate gas for withdrawal
            estimate = self.web3_manager.estimate_transaction_cost(
                chain=chain,
                vault_address=vault_address,
                function_name="withdraw",
                from_address=user_address,
                params=[shares_int]
            )
            
            return WithdrawalTransaction(
                vaultId=vault_id,
                shares=Decimal(str(shares)),
                vaultAddress=vault_address,
                functionName="withdraw",
                functionParams=[shares_int],
                estimatedGas=estimate.estimatedGas if estimate else None
            )
            
        except Exception as e:
            logger.error(f"Failed to prepare withdrawal transaction: {str(e)}")
            return None
    
    def estimate_gas_costs(
        self,
        chain: str,
        vault_address: str,
        action: str,
        user_address: str,
        amount: Optional[int] = None
    ) -> Optional[TransactionEstimate]:
        """
        Estimate gas costs for a vault interaction.
        
        Args:
            chain: Blockchain network
            vault_address: Vault contract address
            action: Action to perform ('deposit' or 'withdraw')
            user_address: User's wallet address
            amount: Amount for the action (if applicable)
            
        Returns:
            TransactionEstimate object or None
        """
        params = [amount] if amount else []
        
        return self.web3_manager.estimate_transaction_cost(
            chain=chain,
            vault_address=vault_address,
            function_name=action,
            from_address=user_address,
            params=params
        )
    
    async def get_user_vault_positions(
        self,
        user_address: str,
        chain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get user's positions across vaults.
        
        Args:
            user_address: User's wallet address
            chain: Optional chain filter
            
        Returns:
            List of position data
        """
        positions = []
        
        # Get all vaults
        vaults = await self.get_vaults(chain)
        
        # Get APY data
        apy_data = await self.get_apy()
        
        # Get prices
        prices = await self.get_prices()
        
        # Check each vault for user balance
        chains_to_check = [chain] if chain else self.web3_manager.get_active_chains()
        
        for vault in vaults:
            if vault.chain not in chains_to_check:
                continue
                
            try:
                balance = self.web3_manager.get_vault_balance(
                    chain=vault.chain,
                    vault_address=vault.earnContractAddress,
                    user_address=user_address
                )
                
                if balance and balance > 0:
                    price_per_share = self.web3_manager.get_price_per_share(
                        chain=vault.chain,
                        vault_address=vault.earnContractAddress
                    )
                    
                    # Calculate position value
                    # This is simplified - would need proper decimal handling
                    position_value = 0
                    if price_per_share and vault.oracleId in prices:
                        token_price = float(prices[vault.oracleId].price)
                        shares_value = (balance * price_per_share) / (10 ** 18)
                        position_value = shares_value * token_price
                    
                    position = {
                        "vault_id": vault.id,
                        "vault_name": vault.name,
                        "chain": vault.chain,
                        "balance": str(balance),
                        "value_usd": str(position_value),
                        "apy": apy_data.get(vault.id, {}).totalApy if isinstance(apy_data.get(vault.id), VaultAPY) else 0,
                        "token": vault.token,
                        "platform": vault.platformId
                    }
                    
                    positions.append(position)
                    
            except Exception as e:
                logger.warning(f"Failed to check balance for vault {vault.id}: {str(e)}")
                
        return positions
    
    async def search_vaults(
        self,
        query: str,
        chain: Optional[str] = None,
        min_apy: Optional[float] = None,
        max_apy: Optional[float] = None,
        min_tvl: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search vaults with filters.
        
        Args:
            query: Search query for vault name/token
            chain: Chain filter
            min_apy: Minimum APY filter
            max_apy: Maximum APY filter
            min_tvl: Minimum TVL filter
            
        Returns:
            List of matching vaults with details
        """
        # Get all data
        vaults = await self.get_vaults(chain)
        apy_data = await self.get_apy()
        tvl_data = await self.get_tvl()
        
        results = []
        
        for vault in vaults:
            # Apply search filter
            if query.lower() not in vault.name.lower() and query.lower() not in vault.token.lower():
                continue
                
            # Get vault metrics
            vault_apy = apy_data.get(vault.id)
            vault_tvl = tvl_data.get(vault.chain, {}).tvl if vault.chain in tvl_data else 0
            
            # Apply APY filters
            if vault_apy:
                apy_value = vault_apy.totalApy
                if min_apy and apy_value < min_apy:
                    continue
                if max_apy and apy_value > max_apy:
                    continue
            else:
                apy_value = 0
                
            # Apply TVL filter
            if min_tvl and float(vault_tvl) < min_tvl:
                continue
                
            result = {
                "id": vault.id,
                "name": vault.name,
                "token": vault.token,
                "chain": vault.chain,
                "platform": vault.platformId,
                "apy": apy_value,
                "tvl": str(vault_tvl),
                "status": vault.status,
                "risks": vault.risks,
                "strategy": vault.strategyTypeId,
                "contract_address": vault.earnContractAddress
            }
            
            results.append(result)
            
        # Sort by APY descending
        results.sort(key=lambda x: x["apy"], reverse=True)
        
        return results