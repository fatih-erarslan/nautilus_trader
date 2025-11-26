"""
Beefy Finance API Data Models

This module defines Pydantic models for Beefy Finance API responses.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from decimal import Decimal
from datetime import datetime


class VaultAsset(BaseModel):
    """Model for vault underlying assets"""
    symbol: str
    address: str
    decimals: int
    oracle: Optional[str] = None
    oracleId: Optional[str] = None


class VaultEarnedToken(BaseModel):
    """Model for earned tokens in a vault"""
    address: str
    symbol: str
    decimals: int
    oracle: Optional[str] = None
    oracleId: Optional[str] = None


class VaultStrategy(BaseModel):
    """Model for vault strategy information"""
    address: str
    name: Optional[str] = None
    withdrawalFee: Optional[float] = Field(None, description="Withdrawal fee as percentage")
    paused: bool = False
    retireReason: Optional[str] = None


class BeefyVault(BaseModel):
    """Model for Beefy Finance vault data"""
    id: str = Field(..., description="Unique vault identifier")
    name: str = Field(..., description="Human-readable vault name")
    token: str = Field(..., description="Vault token symbol")
    tokenAddress: Optional[str] = Field(None, description="Vault token contract address")
    earnedToken: str = Field(..., description="Token earned by the vault")
    earnedTokenAddress: str = Field(..., description="Earned token contract address")
    earnContractAddress: str = Field(..., description="Earn contract address")
    oracle: str = Field(..., description="Price oracle type")
    oracleId: str = Field(..., description="Oracle identifier")
    status: str = Field(..., description="Vault status (active, eol, paused)")
    platformId: str = Field(..., description="Platform identifier")
    assets: List[str] = Field(..., description="List of asset symbols")
    risks: List[str] = Field(default_factory=list, description="Risk categories")
    strategyTypeId: str = Field(..., description="Strategy type identifier")
    network: str = Field(..., description="Blockchain network")
    chain: str = Field(..., description="Chain identifier")
    retiredAt: Optional[datetime] = None
    pausedAt: Optional[datetime] = None
    createdAt: datetime
    pricePerFullShare: Optional[str] = None
    strategy: Optional[VaultStrategy] = None
    addLiquidityUrl: Optional[str] = None
    removeLiquidityUrl: Optional[str] = None
    buyTokenUrl: Optional[str] = None
    assetDetails: Optional[List[VaultAsset]] = None
    earnedTokenDetails: Optional[VaultEarnedToken] = None


class VaultAPY(BaseModel):
    """Model for vault APY data"""
    vaultId: str = Field(..., description="Vault identifier")
    priceId: str = Field(..., description="Price oracle ID")
    vaultApr: float = Field(..., description="Vault base APR")
    vaultApy: float = Field(..., description="Vault APY with compounding")
    compoundingsPerYear: int = Field(..., description="Number of compoundings per year")
    beefyPerformanceFee: float = Field(..., description="Beefy performance fee")
    vaultDailyApy: float = Field(..., description="Daily APY")
    totalApy: float = Field(..., description="Total APY including all rewards")
    tradingApr: Optional[float] = Field(None, description="Trading fees APR")
    liquidStakingApr: Optional[float] = Field(None, description="Liquid staking APR")
    composablePoolApr: Optional[float] = Field(None, description="Composable pool APR")
    merklApr: Optional[float] = Field(None, description="Merkl rewards APR")


class VaultTVL(BaseModel):
    """Model for vault TVL data"""
    vaultId: str = Field(..., description="Vault identifier")
    tvl: Decimal = Field(..., description="Total value locked in USD")
    pricePerFullShare: str = Field(..., description="Price per full share")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TokenPrice(BaseModel):
    """Model for token price data"""
    symbol: str = Field(..., description="Token symbol")
    price: Decimal = Field(..., description="Token price in USD")
    oracleId: str = Field(..., description="Oracle identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TransactionEstimate(BaseModel):
    """Model for transaction gas estimates"""
    chain: str
    action: str  # 'deposit' or 'withdrawal'
    vaultId: str
    estimatedGas: int
    gasPrice: Decimal
    totalCostWei: Decimal
    totalCostETH: Decimal
    totalCostUSD: Decimal
    nativeTokenPrice: Decimal


class DepositTransaction(BaseModel):
    """Model for deposit transaction data"""
    vaultId: str
    amount: Decimal
    tokenAddress: str
    vaultAddress: str
    functionName: str = "deposit"
    functionParams: List[Any]
    estimatedGas: Optional[int] = None
    
    
class WithdrawalTransaction(BaseModel):
    """Model for withdrawal transaction data"""
    vaultId: str
    shares: Decimal
    vaultAddress: str
    functionName: str = "withdrawAll"
    functionParams: List[Any]
    estimatedGas: Optional[int] = None


class BeefyAPIResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool = True
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChainConfig(BaseModel):
    """Configuration for blockchain connections"""
    name: str
    chainId: int
    rpcUrl: str
    nativeToken: str
    nativeTokenSymbol: str
    blockExplorerUrl: str
    beefyContractAddress: Optional[str] = None
    isActive: bool = True