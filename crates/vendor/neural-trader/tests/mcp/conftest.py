"""
Pytest configuration for MCP syndicate tests

This module provides shared fixtures and configuration for all MCP tests,
with special focus on syndicate tool testing.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from typing import Dict, Any, List
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Disable verbose logging during tests
logging.getLogger("src.sports_betting.syndicate").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server with syndicate tools"""
    server = AsyncMock()
    
    # Configure server attributes
    server.syndicate_id = "TEST_SYNDICATE_001"
    server.gpu_available = True
    server.tools_registry = {}
    
    # Mock tool registration
    async def register_tool(name: str, handler: Any, description: str = ""):
        server.tools_registry[name] = {
            "handler": handler,
            "description": description
        }
    
    server.register_tool = register_tool
    
    # Mock tool calling
    async def call_tool(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name in server.tools_registry:
            handler = server.tools_registry[tool_name]["handler"]
            return await handler(params)
        return {"error": f"Tool {tool_name} not found"}
    
    server.call_tool = AsyncMock(side_effect=call_tool)
    
    # Mock tool listing
    async def list_tools() -> List[Dict[str, str]]:
        return [
            {"name": name, "description": info["description"]}
            for name, info in server.tools_registry.items()
        ]
    
    server.list_tools = AsyncMock(side_effect=list_tools)
    
    return server


@pytest.fixture
def mock_database():
    """Mock database for syndicate operations"""
    db = Mock()
    
    # In-memory storage
    db.members = {}
    db.proposals = {}
    db.bets = {}
    db.transactions = []
    db.audit_log = []
    
    # Mock methods
    db.execute = AsyncMock()
    db.fetchone = AsyncMock()
    db.fetchall = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    
    return db


@pytest.fixture
def mock_redis():
    """Mock Redis for caching"""
    redis = AsyncMock()
    
    # In-memory cache
    cache = {}
    
    async def get(key: str) -> Any:
        return cache.get(key)
    
    async def set(key: str, value: Any, expire: int = None) -> None:
        cache[key] = value
    
    async def delete(key: str) -> None:
        cache.pop(key, None)
    
    async def exists(key: str) -> bool:
        return key in cache
    
    redis.get = AsyncMock(side_effect=get)
    redis.set = AsyncMock(side_effect=set)
    redis.delete = AsyncMock(side_effect=delete)
    redis.exists = AsyncMock(side_effect=exists)
    
    return redis


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for real-time updates"""
    ws = AsyncMock()
    
    # Message queue
    ws.messages = []
    
    async def send_json(data: Dict[str, Any]) -> None:
        ws.messages.append(data)
    
    async def receive_json() -> Dict[str, Any]:
        if ws.messages:
            return ws.messages.pop(0)
        return {"type": "ping"}
    
    ws.send_json = AsyncMock(side_effect=send_json)
    ws.receive_json = AsyncMock(side_effect=receive_json)
    ws.close = AsyncMock()
    
    return ws


@pytest.fixture
def syndicate_test_config():
    """Test configuration for syndicate operations"""
    return {
        "syndicate_id": "TEST_SYNDICATE_001",
        "min_members": 3,
        "max_members": 50,
        "min_contribution": 1000,
        "max_contribution": 100000,
        "voting_quorum": 0.5,
        "approval_threshold": 0.6,
        "profit_distribution_methods": ["proportional", "performance_weighted", "hybrid"],
        "allowed_bet_types": ["moneyline", "spread", "over_under", "prop"],
        "max_bet_percentage": 0.2,  # Max 20% of pool per bet
        "withdrawal_cooldown_days": 7,
        "performance_window_days": 90
    }


@pytest.fixture
def performance_monitor():
    """Monitor test performance metrics"""
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = []
        
        def start_operation(self, operation_name: str):
            return {
                "name": operation_name,
                "start_time": time.time(),
                "end_time": None,
                "duration": None
            }
        
        def end_operation(self, operation: Dict[str, Any]):
            operation["end_time"] = time.time()
            operation["duration"] = operation["end_time"] - operation["start_time"]
            self.metrics.append(operation)
            return operation["duration"]
        
        def get_summary(self):
            if not self.metrics:
                return {}
            
            durations = [m["duration"] for m in self.metrics]
            return {
                "total_operations": len(self.metrics),
                "total_duration": sum(durations),
                "average_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "operations_per_second": len(self.metrics) / sum(durations) if sum(durations) > 0 else 0
            }
    
    return PerformanceMonitor()


# Pytest markers for categorizing tests
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests with MCP server")
    config.addinivalue_line("markers", "security: Security and permission tests")
    config.addinivalue_line("markers", "performance: Performance and concurrency tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to execute")
    config.addinivalue_line("markers", "edge_case: Edge case and error handling tests")


# Test data generators
def generate_test_member(role="contributor", **kwargs):
    """Generate test member data"""
    import uuid
    from decimal import Decimal
    
    member_id = kwargs.get("member_id", f"MEMBER_{uuid.uuid4().hex[:8]}")
    
    return {
        "member_id": member_id,
        "username": kwargs.get("username", f"test_user_{uuid.uuid4().hex[:6]}"),
        "email": kwargs.get("email", f"test_{uuid.uuid4().hex[:6]}@example.com"),
        "role": role,
        "status": kwargs.get("status", "active"),
        "capital_contribution": kwargs.get("capital_contribution", Decimal("10000")),
        "performance_score": kwargs.get("performance_score", 0.75),
        "joined_at": kwargs.get("joined_at", "2024-01-01T00:00:00Z")
    }


def generate_test_proposal(proposal_type="large_bet", **kwargs):
    """Generate test proposal data"""
    import uuid
    
    proposal_id = kwargs.get("proposal_id", f"PROP_{uuid.uuid4().hex[:8]}")
    
    return {
        "proposal_id": proposal_id,
        "type": proposal_type,
        "title": kwargs.get("title", f"Test Proposal {proposal_id}"),
        "description": kwargs.get("description", "Test proposal description"),
        "proposer_id": kwargs.get("proposer_id", "MEMBER_001"),
        "status": kwargs.get("status", "active"),
        "created_at": kwargs.get("created_at", "2024-01-01T00:00:00Z"),
        "voting_deadline": kwargs.get("voting_deadline", "2024-01-03T00:00:00Z"),
        "details": kwargs.get("details", {})
    }


def generate_test_bet(sport="NFL", **kwargs):
    """Generate test bet data"""
    import uuid
    from decimal import Decimal
    
    bet_id = kwargs.get("bet_id", f"BET_{uuid.uuid4().hex[:8]}")
    
    return {
        "bet_id": bet_id,
        "sport": sport,
        "event": kwargs.get("event", f"Team A vs Team B"),
        "bet_type": kwargs.get("bet_type", "moneyline"),
        "stake": kwargs.get("stake", Decimal("5000")),
        "odds": kwargs.get("odds", 2.5),
        "status": kwargs.get("status", "pending"),
        "created_at": kwargs.get("created_at", "2024-01-01T00:00:00Z")
    }


# Export test data generators
pytest.generate_test_member = generate_test_member
pytest.generate_test_proposal = generate_test_proposal
pytest.generate_test_bet = generate_test_bet