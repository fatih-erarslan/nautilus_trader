"""
Web3 Manager for Multi-Chain Support

This module manages Web3 connections for multiple blockchain networks
to interact with Beefy Finance vaults.
"""

import logging
from typing import Dict, Optional, Any, List
from decimal import Decimal
from web3 import Web3
from web3.middleware import geth_poa_middleware
from web3.types import TxParams, Wei
from eth_account import Account
from eth_typing import Address, ChecksumAddress
from tenacity import retry, stop_after_attempt, wait_exponential

from .data_models import ChainConfig, TransactionEstimate

logger = logging.getLogger(__name__)


class Web3Manager:
    """
    Manages Web3 connections for multiple blockchain networks.
    
    Supports: Ethereum, BSC, Polygon, Arbitrum, Optimism, Fantom, Avalanche
    """
    
    # Real RPC endpoints (free tier)
    CHAIN_CONFIGS = {
        "ethereum": ChainConfig(
            name="Ethereum",
            chainId=1,
            rpcUrl="https://eth.llamarpc.com",
            nativeToken="ETH",
            nativeTokenSymbol="ETH",
            blockExplorerUrl="https://etherscan.io"
        ),
        "bsc": ChainConfig(
            name="BSC",
            chainId=56,
            rpcUrl="https://bsc-dataseed1.binance.org",
            nativeToken="BNB",
            nativeTokenSymbol="BNB",
            blockExplorerUrl="https://bscscan.com"
        ),
        "polygon": ChainConfig(
            name="Polygon",
            chainId=137,
            rpcUrl="https://polygon-rpc.com",
            nativeToken="MATIC",
            nativeTokenSymbol="MATIC",
            blockExplorerUrl="https://polygonscan.com"
        ),
        "arbitrum": ChainConfig(
            name="Arbitrum",
            chainId=42161,
            rpcUrl="https://arb1.arbitrum.io/rpc",
            nativeToken="ETH",
            nativeTokenSymbol="ETH",
            blockExplorerUrl="https://arbiscan.io"
        ),
        "optimism": ChainConfig(
            name="Optimism",
            chainId=10,
            rpcUrl="https://mainnet.optimism.io",
            nativeToken="ETH",
            nativeTokenSymbol="ETH",
            blockExplorerUrl="https://optimistic.etherscan.io"
        ),
        "fantom": ChainConfig(
            name="Fantom",
            chainId=250,
            rpcUrl="https://rpc.ftm.tools",
            nativeToken="FTM",
            nativeTokenSymbol="FTM",
            blockExplorerUrl="https://ftmscan.com"
        ),
        "avalanche": ChainConfig(
            name="Avalanche",
            chainId=43114,
            rpcUrl="https://api.avax.network/ext/bc/C/rpc",
            nativeToken="AVAX",
            nativeTokenSymbol="AVAX",
            blockExplorerUrl="https://snowtrace.io"
        )
    }
    
    # Standard Beefy Vault ABI (minimal required functions)
    VAULT_ABI = [
        {
            "inputs": [{"name": "_amount", "type": "uint256"}],
            "name": "deposit",
            "outputs": [],
            "type": "function"
        },
        {
            "inputs": [],
            "name": "depositAll",
            "outputs": [],
            "type": "function"
        },
        {
            "inputs": [{"name": "_shares", "type": "uint256"}],
            "name": "withdraw",
            "outputs": [],
            "type": "function"
        },
        {
            "inputs": [],
            "name": "withdrawAll",
            "outputs": [],
            "type": "function"
        },
        {
            "inputs": [],
            "name": "balance",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function",
            "constant": True
        },
        {
            "inputs": [],
            "name": "getPricePerFullShare",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function",
            "constant": True
        },
        {
            "inputs": [],
            "name": "want",
            "outputs": [{"name": "", "type": "address"}],
            "type": "function",
            "constant": True
        },
        {
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function",
            "constant": True
        }
    ]
    
    # ERC20 ABI for token approvals
    ERC20_ABI = [
        {
            "inputs": [
                {"name": "spender", "type": "address"},
                {"name": "amount", "type": "uint256"}
            ],
            "name": "approve",
            "outputs": [{"name": "", "type": "bool"}],
            "type": "function"
        },
        {
            "inputs": [
                {"name": "owner", "type": "address"},
                {"name": "spender", "type": "address"}
            ],
            "name": "allowance",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function",
            "constant": True
        },
        {
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function",
            "constant": True
        },
        {
            "inputs": [],
            "name": "decimals",
            "outputs": [{"name": "", "type": "uint8"}],
            "type": "function",
            "constant": True
        }
    ]
    
    def __init__(self):
        """Initialize Web3Manager with connections to all supported chains."""
        self.connections: Dict[str, Web3] = {}
        self.chain_configs = self.CHAIN_CONFIGS.copy()
        self._initialize_connections()
        
    def _initialize_connections(self):
        """Initialize Web3 connections for all configured chains."""
        for chain_id, config in self.chain_configs.items():
            try:
                w3 = Web3(Web3.HTTPProvider(config.rpcUrl))
                
                # Add POA middleware for chains that need it
                if chain_id in ["bsc", "polygon", "avalanche"]:
                    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                
                if w3.is_connected():
                    self.connections[chain_id] = w3
                    logger.info(f"Connected to {config.name} (Chain ID: {config.chainId})")
                else:
                    logger.warning(f"Failed to connect to {config.name}")
                    
            except Exception as e:
                logger.error(f"Error connecting to {config.name}: {str(e)}")
    
    def get_connection(self, chain: str) -> Optional[Web3]:
        """Get Web3 connection for specified chain."""
        return self.connections.get(chain)
    
    def get_chain_config(self, chain: str) -> Optional[ChainConfig]:
        """Get configuration for specified chain."""
        return self.chain_configs.get(chain)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_gas_price(self, chain: str) -> Optional[Wei]:
        """Get current gas price for specified chain."""
        w3 = self.get_connection(chain)
        if not w3:
            return None
            
        try:
            return w3.eth.gas_price
        except Exception as e:
            logger.error(f"Error getting gas price for {chain}: {str(e)}")
            return None
    
    def estimate_transaction_cost(
        self,
        chain: str,
        vault_address: str,
        function_name: str,
        from_address: str,
        value: int = 0,
        params: Optional[List[Any]] = None
    ) -> Optional[TransactionEstimate]:
        """Estimate gas cost for a transaction."""
        w3 = self.get_connection(chain)
        if not w3:
            return None
            
        try:
            vault_contract = w3.eth.contract(
                address=Web3.to_checksum_address(vault_address),
                abi=self.VAULT_ABI
            )
            
            # Build transaction
            if params is None:
                params = []
                
            function = getattr(vault_contract.functions, function_name)
            tx = function(*params).build_transaction({
                'from': from_address,
                'value': value,
                'gas': 0,  # Will be estimated
                'gasPrice': 0  # Will be set
            })
            
            # Estimate gas
            estimated_gas = w3.eth.estimate_gas(tx)
            gas_price = self.get_gas_price(chain)
            
            if not gas_price:
                return None
                
            total_cost_wei = estimated_gas * gas_price
            total_cost_eth = w3.from_wei(total_cost_wei, 'ether')
            
            # Get native token price (would need price oracle)
            native_token_price = Decimal('2000')  # Placeholder
            
            return TransactionEstimate(
                chain=chain,
                action=function_name,
                vaultId=vault_address,
                estimatedGas=estimated_gas,
                gasPrice=Decimal(str(gas_price)),
                totalCostWei=Decimal(str(total_cost_wei)),
                totalCostETH=Decimal(str(total_cost_eth)),
                totalCostUSD=Decimal(str(total_cost_eth)) * native_token_price,
                nativeTokenPrice=native_token_price
            )
            
        except Exception as e:
            logger.error(f"Error estimating transaction cost: {str(e)}")
            return None
    
    def get_vault_contract(self, chain: str, vault_address: str):
        """Get vault contract instance."""
        w3 = self.get_connection(chain)
        if not w3:
            return None
            
        return w3.eth.contract(
            address=Web3.to_checksum_address(vault_address),
            abi=self.VAULT_ABI
        )
    
    def get_token_contract(self, chain: str, token_address: str):
        """Get ERC20 token contract instance."""
        w3 = self.get_connection(chain)
        if not w3:
            return None
            
        return w3.eth.contract(
            address=Web3.to_checksum_address(token_address),
            abi=self.ERC20_ABI
        )
    
    def check_token_allowance(
        self,
        chain: str,
        token_address: str,
        owner_address: str,
        spender_address: str
    ) -> Optional[int]:
        """Check token allowance for vault deposits."""
        token_contract = self.get_token_contract(chain, token_address)
        if not token_contract:
            return None
            
        try:
            allowance = token_contract.functions.allowance(
                Web3.to_checksum_address(owner_address),
                Web3.to_checksum_address(spender_address)
            ).call()
            return allowance
        except Exception as e:
            logger.error(f"Error checking allowance: {str(e)}")
            return None
    
    def prepare_approve_transaction(
        self,
        chain: str,
        token_address: str,
        spender_address: str,
        amount: int,
        from_address: str
    ) -> Optional[TxParams]:
        """Prepare token approval transaction."""
        w3 = self.get_connection(chain)
        token_contract = self.get_token_contract(chain, token_address)
        
        if not w3 or not token_contract:
            return None
            
        try:
            tx = token_contract.functions.approve(
                Web3.to_checksum_address(spender_address),
                amount
            ).build_transaction({
                'from': Web3.to_checksum_address(from_address),
                'gas': 100000,  # Standard gas limit for approval
                'gasPrice': self.get_gas_price(chain),
                'nonce': w3.eth.get_transaction_count(from_address)
            })
            return tx
        except Exception as e:
            logger.error(f"Error preparing approval transaction: {str(e)}")
            return None
    
    def get_vault_balance(
        self,
        chain: str,
        vault_address: str,
        user_address: str
    ) -> Optional[int]:
        """Get user's balance in a vault."""
        vault_contract = self.get_vault_contract(chain, vault_address)
        if not vault_contract:
            return None
            
        try:
            balance = vault_contract.functions.balanceOf(
                Web3.to_checksum_address(user_address)
            ).call()
            return balance
        except Exception as e:
            logger.error(f"Error getting vault balance: {str(e)}")
            return None
    
    def get_price_per_share(self, chain: str, vault_address: str) -> Optional[int]:
        """Get vault's price per full share."""
        vault_contract = self.get_vault_contract(chain, vault_address)
        if not vault_contract:
            return None
            
        try:
            price = vault_contract.functions.getPricePerFullShare().call()
            return price
        except Exception as e:
            logger.error(f"Error getting price per share: {str(e)}")
            return None
    
    def get_active_chains(self) -> List[str]:
        """Get list of active chain connections."""
        return list(self.connections.keys())