"""
Base Strategy for all crypto trading strategies

Provides the abstract base class and common functionality for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from enum import Enum


class ChainType(Enum):
    """Supported blockchain networks"""
    ETHEREUM = "ethereum"
    BSC = "bsc"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    FANTOM = "fantom"


class RiskLevel(Enum):
    """Risk levels for strategies and positions"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4


@dataclass
class Position:
    """Represents a position in a yield farming opportunity"""
    vault_id: str
    chain: ChainType
    protocol: str
    token_pair: Tuple[str, str]
    apy: float
    tvl: float
    amount: float
    entry_time: datetime
    risk_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def risk_adjusted_apy(self) -> float:
        """Calculate risk-adjusted APY"""
        risk_factor = 1 - (self.risk_score / 100)
        return self.apy * risk_factor


@dataclass
class PortfolioState:
    """Current state of the portfolio"""
    positions: List[Position]
    total_value: float
    available_capital: float
    timestamp: datetime
    chain_allocations: Dict[ChainType, float] = field(default_factory=dict)
    protocol_allocations: Dict[str, float] = field(default_factory=dict)

    @property
    def total_allocated(self) -> float:
        """Total amount allocated across all positions"""
        return sum(pos.amount for pos in self.positions)

    @property
    def utilization_rate(self) -> float:
        """Percentage of capital utilized"""
        total_capital = self.total_value + self.available_capital
        return (self.total_allocated / total_capital * 100) if total_capital > 0 else 0

    def get_chain_exposure(self, chain: ChainType) -> float:
        """Get exposure to a specific chain"""
        return self.chain_allocations.get(chain, 0.0)

    def get_protocol_exposure(self, protocol: str) -> float:
        """Get exposure to a specific protocol"""
        return self.protocol_allocations.get(protocol, 0.0)


@dataclass
class VaultOpportunity:
    """Represents a yield farming opportunity from Beefy Finance"""
    vault_id: str
    chain: ChainType
    protocol: str
    token_pair: Tuple[str, str]
    apy: float
    daily_apy: float
    tvl: float
    platform_fee: float
    withdraw_fee: float
    is_paused: bool
    has_boost: bool
    boost_apy: Optional[float]
    risk_factors: Dict[str, float]
    created_at: datetime
    last_harvest: datetime
    
    @property
    def total_apy(self) -> float:
        """Total APY including boost if available"""
        base_apy = self.apy
        if self.has_boost and self.boost_apy:
            return base_apy + self.boost_apy
        return base_apy

    @property
    def net_apy(self) -> float:
        """APY after fees"""
        gross_apy = self.total_apy
        # Account for platform fees
        return gross_apy * (1 - self.platform_fee)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, 
                 name: str,
                 risk_level: RiskLevel,
                 min_apy_threshold: float = 5.0,
                 max_position_size: float = 0.25,
                 rebalance_threshold: float = 0.1):
        """
        Initialize base strategy
        
        Args:
            name: Strategy name
            risk_level: Risk tolerance level
            min_apy_threshold: Minimum APY to consider a position
            max_position_size: Maximum position size as percentage of portfolio
            rebalance_threshold: Threshold for triggering rebalancing
        """
        self.name = name
        self.risk_level = risk_level
        self.min_apy_threshold = min_apy_threshold
        self.max_position_size = max_position_size
        self.rebalance_threshold = rebalance_threshold
        self.execution_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def evaluate_opportunities(self, 
                             opportunities: List[VaultOpportunity],
                             portfolio: PortfolioState) -> List[Tuple[VaultOpportunity, float]]:
        """
        Evaluate yield farming opportunities and return ranked list with allocation amounts
        
        Args:
            opportunities: List of available vault opportunities
            portfolio: Current portfolio state
            
        Returns:
            List of tuples (opportunity, suggested_allocation)
        """
        pass
    
    @abstractmethod
    def calculate_risk_score(self, opportunity: VaultOpportunity) -> float:
        """
        Calculate risk score for a specific opportunity
        
        Args:
            opportunity: Vault opportunity to evaluate
            
        Returns:
            Risk score between 0-100 (0=lowest risk, 100=highest risk)
        """
        pass
    
    @abstractmethod
    def should_rebalance(self, portfolio: PortfolioState) -> bool:
        """
        Determine if portfolio should be rebalanced
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            True if rebalancing is needed
        """
        pass
    
    @abstractmethod
    def generate_rebalance_trades(self, 
                                portfolio: PortfolioState,
                                opportunities: List[VaultOpportunity]) -> List[Dict[str, Any]]:
        """
        Generate trades needed to rebalance portfolio
        
        Args:
            portfolio: Current portfolio state
            opportunities: Available opportunities
            
        Returns:
            List of trade instructions
        """
        pass
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio for a series of returns
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (default 2% annually)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - risk_free_rate/365  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0
            
        return np.sqrt(365) * np.mean(excess_returns) / np.std(excess_returns)
    
    def calculate_max_drawdown(self, values: np.ndarray) -> float:
        """
        Calculate maximum drawdown from a series of portfolio values
        
        Args:
            values: Array of portfolio values
            
        Returns:
            Maximum drawdown as a percentage
        """
        if len(values) < 2:
            return 0.0
            
        cumulative = np.maximum.accumulate(values)
        drawdowns = (values - cumulative) / cumulative
        return abs(np.min(drawdowns)) * 100
    
    def diversification_score(self, portfolio: PortfolioState) -> float:
        """
        Calculate portfolio diversification score (0-100)
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            Diversification score
        """
        if not portfolio.positions:
            return 0.0
            
        # Chain diversification
        chain_values = list(portfolio.chain_allocations.values())
        chain_hhi = sum([(v/100)**2 for v in chain_values]) if chain_values else 1
        
        # Protocol diversification  
        protocol_values = list(portfolio.protocol_allocations.values())
        protocol_hhi = sum([(v/100)**2 for v in protocol_values]) if protocol_values else 1
        
        # Calculate diversification score (inverse of HHI)
        chain_div = (1 - chain_hhi) * 50
        protocol_div = (1 - protocol_hhi) * 50
        
        return chain_div + protocol_div
    
    def validate_position_size(self, 
                             amount: float,
                             portfolio: PortfolioState,
                             opportunity: VaultOpportunity) -> float:
        """
        Validate and adjust position size based on constraints
        
        Args:
            amount: Proposed position size
            portfolio: Current portfolio state
            opportunity: Target opportunity
            
        Returns:
            Adjusted position size
        """
        total_capital = portfolio.total_value + portfolio.available_capital
        max_amount = total_capital * self.max_position_size
        
        # Check available capital
        if amount > portfolio.available_capital:
            amount = portfolio.available_capital
            
        # Check max position size
        if amount > max_amount:
            amount = max_amount
            
        # Check minimum position size (e.g., $100)
        if amount < 100:
            return 0.0
            
        return amount
    
    def log_execution(self, action: str, details: Dict[str, Any]):
        """Log strategy execution for analysis"""
        self.execution_history.append({
            'timestamp': datetime.now(),
            'action': action,
            'strategy': self.name,
            'details': details
        })