"""
Sports Betting Platform with Comprehensive Syndicate Management

Advanced sports betting platform featuring:
- Risk management and portfolio optimization
- Syndicate collaboration and governance
- Capital management and profit distribution
- Smart contract integration and automation
- Member management with role-based permissions
- Real-time communication and collaboration tools
"""

# Risk Management Framework
from .risk_management import (
    PortfolioRiskManager,
    BettingLimitsController,
    MarketRiskAnalyzer,
    SyndicateRiskController,
    PerformanceMonitor,
    RiskFramework
)

# Syndicate Management System
from .syndicate import (
    CapitalManager,
    VotingSystem,
    MemberManager,
    CollaborationManager,
    SmartContractManager
)

__all__ = [
    # Risk Management
    'PortfolioRiskManager',
    'BettingLimitsController',
    'MarketRiskAnalyzer',
    'SyndicateRiskController',
    'PerformanceMonitor',
    'RiskFramework',
    
    # Syndicate Management
    'CapitalManager',
    'VotingSystem',
    'MemberManager',
    'CollaborationManager',
    'SmartContractManager'
]

# Version and metadata
__version__ = "2.0.0"
__description__ = "Advanced sports betting platform with syndicate management"
__author__ = "AI Trading Platform Team"

# Feature flags
FEATURES = {
    "risk_management": True,
    "syndicate_management": True,
    "capital_pooling": True,
    "voting_governance": True,
    "collaboration_tools": True,
    "smart_contracts": True,
    "member_management": True,
    "performance_tracking": True
}