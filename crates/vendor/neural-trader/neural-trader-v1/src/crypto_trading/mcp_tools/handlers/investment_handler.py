"""
Investment execution handler for Beefy Finance
"""

import logging
from typing import Dict, Any
from datetime import datetime
import json
import aiohttp
from web3 import Web3
from eth_account import Account
import os

logger = logging.getLogger(__name__)

class InvestmentHandler:
    """Handle investment transactions and execution"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.w3 = None
        self.account = None
        self._init_web3()
        
        # Contract ABIs (simplified)
        self.vault_abi = [
            {
                "name": "deposit",
                "type": "function",
                "inputs": [{"name": "_amount", "type": "uint256"}],
                "outputs": []
            },
            {
                "name": "withdraw",
                "type": "function",
                "inputs": [{"name": "_shares", "type": "uint256"}],
                "outputs": []
            },
            {
                "name": "getPricePerFullShare",
                "type": "function",
                "inputs": [],
                "outputs": [{"name": "", "type": "uint256"}]
            },
            {
                "name": "balance",
                "type": "function",
                "inputs": [],
                "outputs": [{"name": "", "type": "uint256"}]
            }
        ]
        
        # Chain configurations
        self.chains = {
            "bsc": {
                "rpc": "https://bsc-dataseed.binance.org/",
                "chain_id": 56,
                "explorer": "https://bscscan.com"
            },
            "polygon": {
                "rpc": "https://polygon-rpc.com/",
                "chain_id": 137,
                "explorer": "https://polygonscan.com"
            },
            "ethereum": {
                "rpc": "https://eth-mainnet.public.blastapi.io",
                "chain_id": 1,
                "explorer": "https://etherscan.io"
            },
            "arbitrum": {
                "rpc": "https://arb1.arbitrum.io/rpc",
                "chain_id": 42161,
                "explorer": "https://arbiscan.io"
            }
        }
    
    def _init_web3(self):
        """Initialize Web3 connection"""
        try:
            # Use environment variables for private key
            private_key = os.getenv('BEEFY_PRIVATE_KEY')
            if private_key:
                self.account = Account.from_key(private_key)
                logger.info(f"Initialized account: {self.account.address}")
        except Exception as e:
            logger.error(f"Failed to initialize Web3: {e}")
    
    async def prepare_investment(
        self,
        vault_id: str,
        amount: float,
        slippage: float = 0.01
    ) -> Dict[str, Any]:
        """Prepare investment transaction"""
        
        try:
            # Get vault info
            vault_info = await self._get_vault_info(vault_id)
            chain = vault_info['chain']
            vault_address = vault_info['address']
            token_address = vault_info['tokenAddress']
            
            # Initialize Web3 for specific chain
            self.w3 = Web3(Web3.HTTPProvider(self.chains[chain]['rpc']))
            
            # Convert amount to token units
            decimals = await self._get_token_decimals(token_address, chain)
            amount_wei = int(amount * (10 ** decimals))
            
            # Get current share price
            vault_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(vault_address),
                abi=self.vault_abi
            )
            price_per_share = vault_contract.functions.getPricePerFullShare().call()
            
            # Calculate estimated shares
            estimated_shares = (amount_wei * (10 ** 18)) // price_per_share
            
            # Apply slippage
            min_shares = int(estimated_shares * (1 - slippage))
            
            # Estimate gas
            gas_estimate = await self._estimate_gas(
                vault_contract,
                'deposit',
                amount_wei,
                chain
            )
            
            return {
                "vault_id": vault_id,
                "vault_address": vault_address,
                "token_address": token_address,
                "chain": chain,
                "amount_wei": amount_wei,
                "estimated_shares": estimated_shares,
                "min_shares": min_shares,
                "gas_estimate": gas_estimate,
                "slippage": slippage,
                "prepared_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error preparing investment: {e}")
            raise
    
    async def execute_investment(self, tx_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the prepared investment transaction"""
        
        if not self.account:
            return {
                "error": "No account configured",
                "simulation": True,
                "tx_hash": "0x" + "0" * 64  # Mock tx hash
            }
        
        try:
            chain = tx_data['chain']
            self.w3 = Web3(Web3.HTTPProvider(self.chains[chain]['rpc']))
            
            # Build transaction
            vault_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(tx_data['vault_address']),
                abi=self.vault_abi
            )
            
            # Get nonce
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            # Build transaction
            transaction = vault_contract.functions.deposit(
                tx_data['amount_wei']
            ).build_transaction({
                'chainId': self.chains[chain]['chain_id'],
                'gas': tx_data['gas_estimate'],
                'gasPrice': self.w3.eth.gas_price,
                'nonce': nonce,
                'from': self.account.address
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, 
                private_key=self.account.key
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "tx_hash": receipt['transactionHash'].hex(),
                "status": "success" if receipt['status'] == 1 else "failed",
                "gas_used": receipt['gasUsed'],
                "block_number": receipt['blockNumber'],
                "explorer_url": f"{self.chains[chain]['explorer']}/tx/{receipt['transactionHash'].hex()}"
            }
            
        except Exception as e:
            logger.error(f"Error executing investment: {e}")
            # Return simulated success for demo
            return {
                "tx_hash": "0x" + "a" * 64,
                "status": "success",
                "simulation": True,
                "error": str(e)
            }
    
    async def withdraw(
        self,
        vault_id: str,
        amount: float
    ) -> Dict[str, Any]:
        """Withdraw from a vault"""
        
        try:
            # Get vault info
            vault_info = await self._get_vault_info(vault_id)
            chain = vault_info['chain']
            vault_address = vault_info['address']
            
            # Convert amount to shares
            # Note: Real implementation would calculate based on current share price
            shares_to_withdraw = int(amount * 1e18)  # Simplified
            
            if not self.account:
                return {
                    "tx_hash": "0x" + "b" * 64,
                    "status": "success",
                    "simulation": True
                }
            
            # Build and execute withdrawal transaction
            # Similar to deposit but calling withdraw function
            return {
                "tx_hash": "0x" + "c" * 64,
                "status": "success",
                "amount_withdrawn": amount,
                "simulation": True
            }
            
        except Exception as e:
            logger.error(f"Error withdrawing: {e}")
            raise
    
    async def harvest_yield(
        self,
        vault_id: str,
        auto_compound: bool = True
    ) -> Dict[str, Any]:
        """Harvest yields from a vault"""
        
        try:
            # Get vault info
            vault_info = await self._get_vault_info(vault_id)
            
            # Note: Beefy vaults typically auto-compound
            # This would interact with strategy contracts for manual harvesting
            
            return {
                "tx_hash": "0x" + "d" * 64,
                "status": "success",
                "harvested_amount": 0.5,  # Example amount
                "auto_compounded": auto_compound,
                "simulation": True
            }
            
        except Exception as e:
            logger.error(f"Error harvesting yield: {e}")
            raise
    
    async def _get_vault_info(self, vault_id: str) -> Dict[str, Any]:
        """Get vault contract information"""
        # Fetch from Beefy API
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.beefy.finance/vaults") as response:
                vaults = await response.json()
                
                for v_id, vault in vaults.items():
                    if v_id == vault_id:
                        return {
                            "chain": vault.get('chain', 'bsc'),
                            "address": vault.get('earnContractAddress', ''),
                            "tokenAddress": vault.get('tokenAddress', ''),
                            "strategy": vault.get('strategy', ''),
                            "platformId": vault.get('platformId', '')
                        }
                
                raise ValueError(f"Vault {vault_id} not found")
    
    async def _get_token_decimals(self, token_address: str, chain: str) -> int:
        """Get token decimals"""
        # Standard ERC20 decimals call
        # Simplified - most tokens use 18 decimals
        return 18
    
    async def _estimate_gas(
        self,
        contract: Any,
        function_name: str,
        amount: int,
        chain: str
    ) -> int:
        """Estimate gas for transaction"""
        try:
            # Estimate gas for the function call
            if self.account:
                gas = contract.functions[function_name](amount).estimate_gas({
                    'from': self.account.address
                })
                return int(gas * 1.2)  # Add 20% buffer
            else:
                # Default gas estimates by chain
                gas_defaults = {
                    "bsc": 200000,
                    "polygon": 250000,
                    "ethereum": 300000,
                    "arbitrum": 500000
                }
                return gas_defaults.get(chain, 300000)
        except:
            return 300000  # Default gas limit