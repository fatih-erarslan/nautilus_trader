"""
Global pytest configuration and fixtures
Comprehensive test setup for TDD methodology
"""

import pytest
import asyncio
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from unittest.mock import Mock, patch
import tempfile
import shutil
from decimal import Decimal

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'freqtrade'))

# Import test configurations
from test_configuration import TEST_CONFIG, CAS_CONFIG, DYNAMIC_MANAGER

# Global test configuration
pytest_plugins = [
    'pytest_asyncio',
    'pytest_cov',
    'pytest_xdist'
]

def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Create necessary directories
    directories = [
        'coverage',
        'tests/screenshots',
        'tests/videos',
        'tests/har',
        'tests/reports',
        'tests/fixtures',
        'tests/temp'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Set test markers
    config.addinivalue_line(
        "markers", 
        "cas: Complex Adaptive Systems tests"
    )
    config.addinivalue_line(
        "markers", 
        "mathematical: Mathematical rigor validation"
    )
    config.addinivalue_line(
        "markers", 
        "financial: Financial calculation tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add markers based on test names/paths
        if 'financial' in item.nodeid:
            item.add_marker(pytest.mark.financial)
        if 'cas' in item.nodeid or 'complex_adaptive' in item.nodeid:
            item.add_marker(pytest.mark.cas)
        if 'mathematical' in item.nodeid or 'rigor' in item.nodeid:
            item.add_marker(pytest.mark.mathematical)
        if 'playwright' in item.nodeid or 'visual' in item.nodeid:
            item.add_marker(pytest.mark.playwright)
        if 'integration' in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif 'e2e' in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        else:
            item.add_marker(pytest.mark.unit)

@pytest.fixture(scope='session', autouse=True)
def test_environment_setup():
    """Setup test environment at session start"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Set environment variables
    os.environ['TESTING'] = '1'
    os.environ['PYTHONPATH'] = str(project_root)
    
    # Configure pandas for testing
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    
    yield
    
    # Cleanup after tests
    if 'TESTING' in os.environ:
        del os.environ['TESTING']

@pytest.fixture
def temp_directory():
    """Provide temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
    
    # Generate realistic price data with trends
    initial_price = Decimal('50000.00')
    prices = []
    current_price = float(initial_price)
    
    for i in range(1000):
        # Add trend and noise
        trend = 0.0001 * (i % 100 - 50) / 50  # Cyclical trend
        noise = np.random.normal(0, 0.002)
        change = trend + noise
        current_price *= (1 + change)
        prices.append(max(current_price, 1.0))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'close': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'volume': np.random.lognormal(10, 1, 1000)
    })
    
    # Ensure OHLC consistency
    for i in range(len(df)):
        df.loc[i, 'high'] = max(df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'high'])
        df.loc[i, 'low'] = min(df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'low'])
    
    return df.set_index('timestamp')

@pytest.fixture
def mock_freqtrade_config():
    """Mock FreqTrade configuration"""
    return {
        'strategy': 'CWTSProfitableStrategy',
        'timeframe': '5m',
        'stake_currency': 'USDT',
        'stake_amount': 'unlimited',
        'dry_run': True,
        'datadir': str(project_root / 'tests' / 'fixtures'),
        'exchange': {
            'name': 'binance',
            'sandbox': True,
            'ccxt_config': {},
            'ccxt_async_config': {},
        },
        'pair_whitelist': ['BTC/USDT', 'ETH/USDT'],
        'trading_mode': 'spot',
        'margin_mode': '',
    }

@pytest.fixture
def performance_metrics():
    """Generate sample performance metrics"""
    return {
        'total_return': 0.15,
        'sharpe_ratio': 1.8,
        'max_drawdown': -0.08,
        'volatility': 0.12,
        'win_rate': 0.65,
        'avg_win': 0.025,
        'avg_loss': -0.015,
        'profit_factor': 1.6,
        'calmar_ratio': 1.9,
        'sortino_ratio': 2.1
    }

@pytest.fixture
def mock_exchange_data():
    """Mock exchange data for testing"""
    mock_exchange = Mock()
    mock_exchange.fetch_ohlcv.return_value = [
        [1640995200000, 50000.0, 50500.0, 49800.0, 50200.0, 1.5],
        [1640995500000, 50200.0, 50400.0, 49900.0, 50100.0, 1.8],
        [1640995800000, 50100.0, 50600.0, 50000.0, 50300.0, 2.1],
    ]
    mock_exchange.fetch_ticker.return_value = {
        'symbol': 'BTC/USDT',
        'last': 50200.0,
        'bid': 50180.0,
        'ask': 50220.0,
        'volume': 1500.0
    }
    return mock_exchange

@pytest.fixture
def complex_adaptive_system():
    """Create a Complex Adaptive System for testing"""
    from test_configuration import ComplexAdaptiveSystemsConfig
    
    cas = ComplexAdaptiveSystemsConfig()
    
    # Initialize system state
    system_state = {
        'agents': cas.agents,
        'network': cas.connectivity_matrix,
        'memory': [],
        'adaptation_history': [],
        'emergence_patterns': cas.generate_emergent_behavior()
    }
    
    return system_state

@pytest.fixture
def mathematical_validator():
    """Mathematical validation utilities"""
    class MathValidator:
        @staticmethod
        def validate_precision(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
            return abs(actual - expected) <= tolerance
        
        @staticmethod
        def validate_range(value: float, min_val: float, max_val: float) -> bool:
            return min_val <= value <= max_val
        
        @staticmethod
        def validate_statistical_significance(p_value: float, alpha: float = 0.05) -> bool:
            return p_value < alpha
        
        @staticmethod
        def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
            n = len(data)
            mean = np.mean(data)
            std_err = np.std(data, ddof=1) / np.sqrt(n)
            
            from scipy import stats
            t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin_of_error = t_value * std_err
            
            return (mean - margin_of_error, mean + margin_of_error)
    
    return MathValidator()

@pytest.fixture
def dynamic_config_manager():
    """Dynamic configuration manager"""
    return DYNAMIC_MANAGER

@pytest.fixture
def test_database():
    """In-memory test database"""
    import sqlite3
    
    # Create in-memory database
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create test tables
    cursor.execute('''
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            amount REAL,
            price REAL,
            timestamp INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE performance_metrics (
            id INTEGER PRIMARY KEY,
            strategy TEXT,
            metric_name TEXT,
            metric_value REAL,
            timestamp INTEGER
        )
    ''')
    
    conn.commit()
    
    yield conn
    
    conn.close()

@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test"""
    np.random.seed(42)
    import random
    random.seed(42)

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Performance monitoring fixture
@pytest.fixture(autouse=True)
def monitor_test_performance(request):
    """Monitor test performance and adapt configuration"""
    import time
    
    start_time = time.time()
    
    yield
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Update dynamic configuration based on performance
    if hasattr(request.node, 'rep_call') and request.node.rep_call.passed:
        success_rate = 1.0
    else:
        success_rate = 0.0
    
    performance_metrics = {
        'execution_time': execution_time,
        'success_rate': success_rate,
        'test_name': request.node.name
    }
    
    # Adapt configuration for future tests
    DYNAMIC_MANAGER.adapt_configuration(performance_metrics)

# Pytest hooks for reporting
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Make test results available to fixtures"""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)