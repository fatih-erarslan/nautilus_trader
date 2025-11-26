"""Crypto Trading Database Package

Provides SQLite database functionality for Beefy Finance crypto trading system.
"""

from .connection import DatabaseConnection, get_db_session, init_database
from .models import (
    VaultPosition,
    YieldHistory,
    CryptoTransaction,
    PortfolioSummary,
    Base
)
from .utils import (
    get_active_positions,
    calculate_total_portfolio_value,
    get_yield_history_by_vault,
    batch_insert_yield_history,
    create_portfolio_snapshot,
    get_chain_allocation
)

__all__ = [
    'DatabaseConnection',
    'get_db_session',
    'init_database',
    'VaultPosition',
    'YieldHistory',
    'CryptoTransaction',
    'PortfolioSummary',
    'Base',
    'get_active_positions',
    'calculate_total_portfolio_value',
    'get_yield_history_by_vault',
    'batch_insert_yield_history',
    'create_portfolio_snapshot',
    'get_chain_allocation'
]