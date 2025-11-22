"""
Simplified pytest configuration for demonstration
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_data():
    """Simple test data"""
    return {
        'prices': [100.0, 101.5, 99.8, 102.3, 98.7, 103.1],
        'volumes': [1000, 1200, 800, 1500, 900, 1300],
        'weights': [0.4, 0.3, 0.2, 0.1]
    }

@pytest.fixture
def performance_metrics():
    """Sample performance metrics"""
    return {
        'success_rate': 0.85,
        'execution_time': 2.5,
        'coverage_percent': 95.0
    }