"""
Beefy Finance API Integration

This package provides a complete integration with Beefy Finance,
a multi-chain yield optimizer that maximizes returns from yield farming.

Key Features:
- Multi-chain support (Ethereum, BSC, Polygon, Arbitrum, Optimism, Fantom, Avalanche)
- Real-time vault data and APY information
- Web3 integration for deposits and withdrawals
- Gas estimation and transaction preparation
- Comprehensive error handling and retry logic
"""

from .beefy_client import BeefyFinanceAPI
from .web3_manager import Web3Manager
from .data_models import (
    BeefyVault,
    VaultAPY,
    VaultTVL,
    TokenPrice,
    DepositTransaction,
    WithdrawalTransaction,
    TransactionEstimate,
    ChainConfig
)

__all__ = [
    'BeefyFinanceAPI',
    'Web3Manager',
    'BeefyVault',
    'VaultAPY',
    'VaultTVL',
    'TokenPrice',
    'DepositTransaction',
    'WithdrawalTransaction',
    'TransactionEstimate',
    'ChainConfig'
]

__version__ = '1.0.0'