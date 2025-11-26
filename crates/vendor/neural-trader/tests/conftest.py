"""Pytest configuration and fixtures."""

import pytest
from datetime import datetime, timedelta


class Helpers:
    """Test helper methods."""
    
    @staticmethod
    def now():
        """Get current datetime."""
        return datetime.now()
    
    @staticmethod
    def days_ago(days):
        """Get datetime N days ago."""
        return datetime.now() - timedelta(days=days)


# Make helpers available to all tests
pytest.helpers = Helpers()