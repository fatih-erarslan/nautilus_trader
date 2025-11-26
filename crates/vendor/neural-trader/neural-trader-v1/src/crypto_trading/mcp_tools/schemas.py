"""
Pydantic schemas for Beefy Finance MCP tool inputs and outputs
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class ChainEnum(str, Enum):
    """Supported blockchain networks"""
    BSC = "bsc"
    POLYGON = "polygon"
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    FANTOM = "fantom"

class SortByEnum(str, Enum):
    """Vault sorting options"""
    APY = "apy"
    TVL = "tvl"
    CREATED = "created"
    NAME = "name"

class RebalanceStrategyEnum(str, Enum):
    """Portfolio rebalancing strategies"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MAX_APY = "max_apy"
    CUSTOM = "custom"

# Input Schemas
class GetVaultsInput(BaseModel):
    """Input for getting vaults"""
    chain: Optional[ChainEnum] = Field(None, description="Filter by blockchain")
    min_apy: Optional[float] = Field(None, ge=0, description="Minimum APY filter")
    max_tvl: Optional[float] = Field(None, ge=0, description="Maximum TVL filter")
    sort_by: SortByEnum = Field(SortByEnum.APY, description="Sort criteria")
    limit: int = Field(50, ge=1, le=200, description="Number of results")

class AnalyzeVaultInput(BaseModel):
    """Input for analyzing a vault"""
    vault_id: str = Field(..., description="Vault ID to analyze")
    include_history: bool = Field(True, description="Include historical data")

class InvestInput(BaseModel):
    """Input for investing in a vault"""
    vault_id: str = Field(..., description="Vault ID to invest in")
    amount: float = Field(..., gt=0, description="Amount to invest in USD")
    slippage: float = Field(0.01, ge=0, le=0.1, description="Max slippage tolerance")
    simulate: bool = Field(False, description="Simulate without executing")
    
    @validator('slippage')
    def validate_slippage(cls, v):
        if v < 0 or v > 0.1:
            raise ValueError("Slippage must be between 0 and 10%")
        return v

class HarvestInput(BaseModel):
    """Input for harvesting yields"""
    vault_ids: Optional[List[str]] = Field(None, description="Vault IDs to harvest from")
    auto_compound: bool = Field(True, description="Auto-compound harvested yields")
    simulate: bool = Field(False, description="Simulate without executing")

class RebalanceInput(BaseModel):
    """Input for portfolio rebalancing"""
    strategy: RebalanceStrategyEnum = Field(..., description="Rebalancing strategy")
    target_allocations: Optional[Dict[str, float]] = Field(None, description="Target allocations for custom strategy")
    simulate: bool = Field(False, description="Simulate without executing")
    
    @validator('target_allocations')
    def validate_allocations(cls, v, values):
        if values.get('strategy') == RebalanceStrategyEnum.CUSTOM and not v:
            raise ValueError("Target allocations required for custom strategy")
        if v:
            total = sum(v.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError("Target allocations must sum to 1.0")
        return v

# Output Schemas
class VaultInfo(BaseModel):
    """Vault information"""
    vault_id: str
    name: str
    chain: str
    asset: str
    apy: float
    tvl: float
    risk_score: float
    strategy_type: str
    platform_fees: float
    created_at: datetime

class GetVaultsOutput(BaseModel):
    """Output for getting vaults"""
    vaults: List[VaultInfo]
    total_count: int
    timestamp: datetime

class VaultAnalysis(BaseModel):
    """Detailed vault analysis"""
    current_apy: float
    average_apy_30d: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    risk_adjusted_apy: float
    correlation_to_market: float
    impermanent_loss_risk: float
    smart_contract_audit: bool
    days_active: int

class RiskMetrics(BaseModel):
    """Risk assessment metrics"""
    overall_risk_score: float
    liquidity_risk: float
    smart_contract_risk: float
    protocol_risk: float
    market_risk: float
    recommendations: List[str]

class AnalyzeVaultOutput(BaseModel):
    """Output for vault analysis"""
    vault_id: str
    analysis: VaultAnalysis
    risk_metrics: RiskMetrics
    timestamp: datetime

class InvestmentResult(BaseModel):
    """Investment execution result"""
    tx_hash: Optional[str] = None
    simulation: Optional[bool] = None
    estimated_shares: Optional[float] = None
    estimated_gas: Optional[float] = None
    status: str = "success"

class InvestOutput(BaseModel):
    """Output for investment"""
    vault_id: str
    amount_invested: float
    result: Dict[str, Any]
    timestamp: datetime

class HarvestResult(BaseModel):
    """Harvest execution result"""
    vault_id: str
    harvested: Optional[float] = None
    harvestable: Optional[float] = None
    tx_hash: Optional[str] = None
    simulation: Optional[bool] = None

class HarvestOutput(BaseModel):
    """Output for yield harvesting"""
    yields_harvested: List[Dict[str, Any]]
    total_harvested: float
    timestamp: datetime

class RebalanceAction(BaseModel):
    """Rebalancing action"""
    type: str  # 'withdraw' or 'deposit'
    vault_id: str
    amount: float
    reason: str

class RebalanceOutput(BaseModel):
    """Output for portfolio rebalancing"""
    rebalance_actions: List[Dict[str, Any]]
    new_allocations: Dict[str, float]
    timestamp: datetime

# WebSocket schemas
class APYUpdate(BaseModel):
    """Real-time APY update"""
    vault_id: str
    apy: float
    tvl: float
    timestamp: datetime

class WSMessage(BaseModel):
    """WebSocket message format"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime