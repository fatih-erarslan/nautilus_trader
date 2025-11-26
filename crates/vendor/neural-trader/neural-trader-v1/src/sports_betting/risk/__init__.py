"""
Sports Betting Risk Management Framework

A comprehensive risk management system for sports betting operations including:
- Kelly Criterion optimization
- Portfolio risk management  
- Circuit breaker systems
- Performance monitoring
- Compliance framework

Components:
    kelly_criterion: Advanced Kelly Criterion optimization with portfolio-level risk management
    portfolio_manager: Multi-sport correlation analysis and diversification optimization
    circuit_breakers: Automated stop-loss mechanisms and emergency shutdown procedures
    performance_monitor: Real-time P&L tracking and risk-adjusted performance metrics
    compliance: KYC/AML integration and regulatory reporting
"""

from .kelly_criterion import (
    KellyCriterionOptimizer,
    BettingOpportunity,
    KellyResult,
    KellyMethod,
    calculate_optimal_kelly_fraction
)

from .portfolio_manager import (
    PortfolioRiskManager,
    Portfolio,
    Position,
    RiskMetrics,
    RiskLevel,
    DiversificationMetric
)

from .circuit_breakers import (
    CircuitBreakerSystem,
    CircuitBreaker,
    DrawdownBreaker,
    ConsecutiveLossBreaker,
    DailyLossBreaker,
    VelocityBreaker,
    ConcentrationBreaker,
    BankrollThresholdBreaker,
    CircuitBreakerType,
    BreakerStatus,
    ActionType,
    BreakerTrigger,
    BreakerConfig
)

from .performance_monitor import (
    PerformanceMonitor,
    BettingTransaction,
    PerformanceMetrics,
    PerformanceAlert,
    AlertType,
    TimeFrame
)

from .compliance import (
    ComplianceFramework,
    Customer,
    KYCDocument,
    JurisdictionRule,
    ResponsibleGamblingMonitor,
    ComplianceStatus,
    DocumentType,
    RiskLevel as AMLRiskLevel,
    JurisdictionType,
    ReportType,
    ComplianceEvent,
    AuditEntry
)

__version__ = "1.0.0"

__all__ = [
    # Kelly Criterion
    "KellyCriterionOptimizer",
    "BettingOpportunity", 
    "KellyResult",
    "KellyMethod",
    "calculate_optimal_kelly_fraction",
    
    # Portfolio Management
    "PortfolioRiskManager",
    "Portfolio",
    "Position", 
    "RiskMetrics",
    "RiskLevel",
    "DiversificationMetric",
    
    # Circuit Breakers
    "CircuitBreakerSystem",
    "CircuitBreaker",
    "DrawdownBreaker",
    "ConsecutiveLossBreaker", 
    "DailyLossBreaker",
    "VelocityBreaker",
    "ConcentrationBreaker",
    "BankrollThresholdBreaker",
    "CircuitBreakerType",
    "BreakerStatus",
    "ActionType",
    "BreakerTrigger",
    "BreakerConfig",
    
    # Performance Monitoring
    "PerformanceMonitor",
    "BettingTransaction",
    "PerformanceMetrics", 
    "PerformanceAlert",
    "AlertType",
    "TimeFrame",
    
    # Compliance
    "ComplianceFramework",
    "Customer",
    "KYCDocument",
    "JurisdictionRule",
    "ResponsibleGamblingMonitor", 
    "ComplianceStatus",
    "DocumentType",
    "AMLRiskLevel",
    "JurisdictionType",
    "ReportType", 
    "ComplianceEvent",
    "AuditEntry"
]