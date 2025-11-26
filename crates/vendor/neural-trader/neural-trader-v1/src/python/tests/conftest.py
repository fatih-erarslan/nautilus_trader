"""
Pytest Configuration and Fixtures
================================

Shared test configuration and fixtures for the test suite.
"""

import pytest
import asyncio
from uuid import uuid4
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from supabase_client.config import SupabaseConfig
from supabase_client.client import AsyncSupabaseClient

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_config():
    """Create test Supabase configuration."""
    return SupabaseConfig(
        url="https://test.supabase.co",
        anon_key="test-anon-key",
        service_key="test-service-key"
    )

@pytest.fixture
def mock_supabase_client():
    """Create mock async Supabase client."""
    mock = AsyncMock(spec=AsyncSupabaseClient)
    
    # Configure default return values
    mock.select.return_value = []
    mock.insert.return_value = []
    mock.update.return_value = []
    mock.delete.return_value = []
    mock.count.return_value = 0
    mock.upsert.return_value = []
    
    return mock

@pytest.fixture
def sample_user_id():
    """Generate sample user ID."""
    return uuid4()

@pytest.fixture
def sample_account_id():
    """Generate sample account ID."""
    return uuid4()

@pytest.fixture
def sample_model_id():
    """Generate sample model ID."""
    return uuid4()

@pytest.fixture
def sample_bot_id():
    """Generate sample bot ID."""
    return uuid4()

@pytest.fixture
def sample_user_profile(sample_user_id):
    """Create sample user profile data."""
    return {
        "id": str(sample_user_id),
        "email": "test@example.com",
        "full_name": "Test User",
        "is_active": True,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

@pytest.fixture
def sample_trading_account(sample_user_id, sample_account_id):
    """Create sample trading account data."""
    return {
        "id": str(sample_account_id),
        "user_id": str(sample_user_id),
        "account_name": "Test Account",
        "account_type": "demo",
        "broker": "test_broker",
        "balance": 10000.0,
        "currency": "USD",
        "is_active": True,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

@pytest.fixture
def sample_neural_model(sample_user_id, sample_model_id):
    """Create sample neural model data."""
    return {
        "id": str(sample_model_id),
        "user_id": str(sample_user_id),
        "name": "Test LSTM Model",
        "model_type": "lstm",
        "status": "training",
        "symbols": ["AAPL", "GOOGL"],
        "configuration": {
            "layers": 3,
            "units": 128,
            "dropout": 0.2
        },
        "metadata": {},
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

@pytest.fixture
def sample_trading_bot(sample_user_id, sample_account_id, sample_bot_id):
    """Create sample trading bot data."""
    return {
        "id": str(sample_bot_id),
        "user_id": str(sample_user_id),
        "account_id": str(sample_account_id),
        "name": "Test Trading Bot",
        "strategy": "momentum",
        "status": "stopped",
        "symbols": ["AAPL", "GOOGL"],
        "risk_params": {
            "max_position_size": 0.1,
            "stop_loss": 0.05,
            "take_profit": 0.15
        },
        "strategy_params": {
            "momentum_window": 20,
            "signal_threshold": 0.02
        },
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

@pytest.fixture
def sample_order_data(sample_user_id, sample_account_id, sample_bot_id):
    """Create sample order data."""
    return {
        "id": str(uuid4()),
        "user_id": str(sample_user_id),
        "account_id": str(sample_account_id),
        "bot_id": str(sample_bot_id),
        "symbol": "AAPL",
        "order_type": "market",
        "side": "buy",
        "quantity": 10,
        "price": None,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    return {
        "id": str(uuid4()),
        "symbol": "AAPL",
        "price": 150.00,
        "volume": 1000000,
        "timestamp": datetime.utcnow().isoformat(),
        "high": 152.00,
        "low": 148.00,
        "open": 149.00,
        "close": 150.00
    }

@pytest.fixture
def sample_performance_metric():
    """Create sample performance metric."""
    return {
        "id": str(uuid4()),
        "name": "response_time",
        "value": 150.5,
        "metric_type": "gauge",
        "tags": {
            "component": "api",
            "endpoint": "/orders"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self):
        self.tables = {}
        self.reset()
    
    def reset(self):
        """Reset all tables."""
        self.tables = {
            "profiles": [],
            "trading_accounts": [],
            "neural_models": [],
            "trading_bots": [],
            "orders": [],
            "market_data": [],
            "performance_metrics": [],
            "performance_alerts": []
        }
    
    def insert(self, table: str, data: dict):
        """Insert data into table."""
        if table not in self.tables:
            self.tables[table] = []
        
        # Add ID if not present
        if "id" not in data:
            data["id"] = str(uuid4())
        
        # Add timestamps if not present
        now = datetime.utcnow().isoformat()
        if "created_at" not in data:
            data["created_at"] = now
        if "updated_at" not in data:
            data["updated_at"] = now
        
        self.tables[table].append(data.copy())
        return data
    
    def select(self, table: str, filter_dict: dict = None):
        """Select data from table."""
        if table not in self.tables:
            return []
        
        results = self.tables[table].copy()
        
        if filter_dict:
            filtered_results = []
            for item in results:
                match = True
                for key, value in filter_dict.items():
                    if key not in item or item[key] != value:
                        match = False
                        break
                if match:
                    filtered_results.append(item)
            results = filtered_results
        
        return results
    
    def update(self, table: str, data: dict, filter_dict: dict):
        """Update data in table."""
        if table not in self.tables:
            return []
        
        updated_items = []
        for item in self.tables[table]:
            match = True
            for key, value in filter_dict.items():
                if key not in item or item[key] != value:
                    match = False
                    break
            
            if match:
                item.update(data)
                item["updated_at"] = datetime.utcnow().isoformat()
                updated_items.append(item.copy())
        
        return updated_items
    
    def delete(self, table: str, filter_dict: dict):
        """Delete data from table."""
        if table not in self.tables:
            return []
        
        deleted_items = []
        remaining_items = []
        
        for item in self.tables[table]:
            match = True
            for key, value in filter_dict.items():
                if key not in item or item[key] != value:
                    match = False
                    break
            
            if match:
                deleted_items.append(item.copy())
            else:
                remaining_items.append(item)
        
        self.tables[table] = remaining_items
        return deleted_items
    
    def count(self, table: str, filter_dict: dict = None):
        """Count records in table."""
        return len(self.select(table, filter_dict))

@pytest.fixture
def mock_database():
    """Create mock database instance."""
    return MockDatabase()

@pytest.fixture
def configured_mock_client(mock_supabase_client, mock_database):
    """Create configured mock client with database."""
    
    # Configure mock methods to use mock database
    async def mock_select(table, **kwargs):
        filter_dict = kwargs.get('filter_dict', {})
        return mock_database.select(table, filter_dict)
    
    async def mock_insert(table, data):
        if isinstance(data, list):
            return [mock_database.insert(table, item) for item in data]
        else:
            return [mock_database.insert(table, data)]
    
    async def mock_update(table, data, filter_dict):
        return mock_database.update(table, data, filter_dict)
    
    async def mock_delete(table, filter_dict):
        return mock_database.delete(table, filter_dict)
    
    async def mock_count(table, filter_dict=None):
        return mock_database.count(table, filter_dict)
    
    mock_supabase_client.select.side_effect = mock_select
    mock_supabase_client.insert.side_effect = mock_insert
    mock_supabase_client.update.side_effect = mock_update
    mock_supabase_client.delete.side_effect = mock_delete
    mock_supabase_client.count.side_effect = mock_count
    
    return mock_supabase_client

# Test utilities
def assert_uuid_format(value):
    """Assert that a value is a valid UUID format."""
    try:
        uuid4(value)
        return True
    except (ValueError, TypeError):
        return False

def assert_iso_timestamp(value):
    """Assert that a value is a valid ISO timestamp."""
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return True
    except (ValueError, TypeError):
        return False