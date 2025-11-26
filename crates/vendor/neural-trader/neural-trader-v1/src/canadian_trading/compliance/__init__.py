"""
Canadian Trading Compliance Module
Provides comprehensive regulatory compliance for Canadian securities trading.
"""

from .ciro_compliance import (
    CIROCompliance,
    ClientIdentification,
    TradeReport,
    OrderType,
    SecurityType
)

from .tax_reporting import (
    TaxReporting,
    T5008Slip,
    ACBTracker,
    TaxYear,
    SecurityDisposition
)

from .audit_trail import (
    AuditTrail,
    AuditRecord,
    AuditDatabase,
    AuditEventType,
    AuditSeverity
)

from .monitoring import (
    ComplianceMonitor,
    PositionMonitor,
    TradingPatternDetector,
    MonitoringAlert,
    AlertSeverity,
    Alert,
    MonitoringRule
)

__all__ = [
    # CIRO Compliance
    'CIROCompliance',
    'ClientIdentification',
    'TradeReport',
    'OrderType',
    'SecurityType',
    
    # Tax Reporting
    'TaxReporting',
    'T5008Slip',
    'ACBTracker',
    'TaxYear',
    'SecurityDisposition',
    
    # Audit Trail
    'AuditTrail',
    'AuditRecord',
    'AuditDatabase',
    'AuditEventType',
    'AuditSeverity',
    
    # Monitoring
    'ComplianceMonitor',
    'PositionMonitor',
    'TradingPatternDetector',
    'MonitoringAlert',
    'AlertSeverity',
    'Alert',
    'MonitoringRule'
]

__version__ = '1.0.0'