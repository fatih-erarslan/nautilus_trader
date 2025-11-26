"""
Syndicate Management System for AI Sports Betting Platform

This module provides comprehensive syndicate collaboration features including:
- Capital management and pooled funds
- Voting and consensus mechanisms  
- Member management with role-based permissions
- Collaboration tools for shared research
- Smart contract integration for governance

Components:
- CapitalManager: Handles pooled funds, distributions, and member contributions
- VotingSystem: Manages proposals, voting, and consensus mechanisms
- MemberManager: Handles user roles, permissions, and performance tracking
- CollaborationManager: Provides shared research and communication tools
- SmartContractManager: Integrates governance automation and escrow
"""

from .capital_manager import CapitalManager
from .voting_system import VotingSystem
from .member_manager import MemberManager
from .collaboration import CollaborationManager
from .smart_contracts import SmartContractManager

__all__ = [
    'CapitalManager',
    'VotingSystem', 
    'MemberManager',
    'CollaborationManager',
    'SmartContractManager'
]

__version__ = "1.0.0"