"""
Fantasy Collective Database Package

This package contains all database-related components for the Fantasy Collective system.
"""

__version__ = "1.0.0"
__author__ = "Fantasy Collective Development Team"

from .connection import DatabaseConnection
from .models import *
from .migrations import MigrationManager

__all__ = [
    'DatabaseConnection',
    'MigrationManager',
]