"""
Beefy Finance MCP Tools for Crypto Trading
"""

from .beefy_tools import BeefyToolHandlers
from .schemas import (
    GetVaultsInput, GetVaultsOutput,
    AnalyzeVaultInput, AnalyzeVaultOutput,
    InvestInput, InvestOutput,
    HarvestInput, HarvestOutput,
    RebalanceInput, RebalanceOutput
)

__all__ = [
    'BeefyToolHandlers',
    'GetVaultsInput', 'GetVaultsOutput',
    'AnalyzeVaultInput', 'AnalyzeVaultOutput',
    'InvestInput', 'InvestOutput',
    'HarvestInput', 'HarvestOutput',
    'RebalanceInput', 'RebalanceOutput'
]