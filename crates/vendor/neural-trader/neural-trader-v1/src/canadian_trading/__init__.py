"""
Canadian Trading APIs Integration Module
=======================================

This module provides comprehensive integration with Canadian trading platforms:
- Interactive Brokers Canada
- Questrade
- OANDA Canada (Forex)

With full regulatory compliance for CIRO and CRA requirements.
"""

__version__ = "1.0.0"

# Import main components for easy access
from .brokers.ib_canada import IBCanadaClient, ConnectionConfig
from .brokers.questrade import QuestradeAPI, QuestradeDataFeed
from .brokers.oanda_canada import OANDACanada
from .compliance import CIROCompliance, TaxReporting, AuditTrail, ComplianceMonitor
from .utils.auth import OAuth2Manager

__all__ = [
    # IB Canada
    "IBCanadaClient",
    "ConnectionConfig",
    
    # Questrade
    "QuestradeAPI",
    "QuestradeDataFeed",
    
    # OANDA
    "OANDACanada",
    
    # Compliance
    "CIROCompliance",
    "TaxReporting",
    "AuditTrail",
    "ComplianceMonitor",
    
    # Utils
    "OAuth2Manager",
]