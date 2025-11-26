"""
Base test classes for AI News Trading Platform.

This module provides base classes for different types of tests following TDD principles
and the AAA (Arrange, Act, Assert) pattern.
"""

import asyncio
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
from faker import Faker
from freezegun import freeze_time
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize faker for test data generation
fake = Faker()


class BaseTest(ABC):
    """
    Abstract base class for all tests.
    
    Provides common functionality and enforces the AAA pattern.
    """
    
    # Class-level test configuration
    TEST_TIMEOUT = 30  # seconds
    ASYNC_TIMEOUT = 10  # seconds
    
    @classmethod
    def setup_class(cls):
        """Set up test class - runs once per test class."""
        cls.logger = logging.getLogger(cls.__name__)
        cls.logger.setLevel(logging.DEBUG)
        cls.test_start_time = datetime.now(timezone.utc)
    
    @classmethod
    def teardown_class(cls):
        """Tear down test class - runs once after all tests in class."""
        duration = datetime.now(timezone.utc) - cls.test_start_time
        cls.logger.info(f"Test class {cls.__name__} completed in {duration.total_seconds():.2f}s")
    
    def setup_method(self, method):
        """Set up test method - runs before each test."""
        self.method_name = method.__name__
        self.logger.debug(f"Starting test: {self.method_name}")
        self.test_id = fake.uuid4()
        self._setup_test_data()
    
    def teardown_method(self, method):
        """Tear down test method - runs after each test."""
        self.logger.debug(f"Completed test: {self.method_name}")
        self._cleanup_test_data()
    
    @abstractmethod
    def _setup_test_data(self):
        """Set up test-specific data. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _cleanup_test_data(self):
        """Clean up test-specific data. Must be implemented by subclasses."""
        pass
    
    # Utility methods for all tests
    @contextmanager
    def assert_performance(self, max_duration: float):
        """Context manager to assert performance requirements."""
        start_time = datetime.now(timezone.utc)
        yield
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        assert duration < max_duration, f"Operation took {duration:.2f}s, expected < {max_duration}s"
    
    def assert_json_equal(self, actual: Dict[str, Any], expected: Dict[str, Any]):
        """Assert two JSON objects are equal with helpful diff output."""
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), \
            f"JSON mismatch:\nActual: {json.dumps(actual, indent=2)}\nExpected: {json.dumps(expected, indent=2)}"


class BaseUnitTest(BaseTest):
    """
    Base class for unit tests.
    
    Unit tests should:
    - Test individual functions/methods in isolation
    - Use mocks for all external dependencies
    - Execute quickly (< 100ms per test)
    - Follow the AAA pattern strictly
    """
    
    TEST_TIMEOUT = 5  # Unit tests should be fast
    
    def _setup_test_data(self):
        """Set up unit test data."""
        self.mock_dependencies = {}
        self.test_data = self._create_test_data()
    
    def _cleanup_test_data(self):
        """Clean up unit test data."""
        self.mock_dependencies.clear()
        self.test_data = None
    
    def _create_test_data(self) -> Dict[str, Any]:
        """Create test data for unit tests. Override in subclasses."""
        return {
            'id': fake.uuid4(),
            'timestamp': datetime.now(timezone.utc),
            'value': fake.random_number()
        }
    
    @contextmanager
    def mock_external_service(self, service_name: str, **kwargs):
        """Context manager for mocking external services."""
        mock_service = Mock(**kwargs)
        self.mock_dependencies[service_name] = mock_service
        with patch(service_name, mock_service):
            yield mock_service


class BaseIntegrationTest(BaseTest):
    """
    Base class for integration tests.
    
    Integration tests should:
    - Test interaction between components
    - Use real implementations where possible
    - Mock only external services (APIs, databases)
    - May take longer than unit tests (< 5s per test)
    """
    
    TEST_TIMEOUT = 30
    
    def _setup_test_data(self):
        """Set up integration test data."""
        self.test_db = self._setup_test_database()
        self.test_cache = self._setup_test_cache()
        self.external_mocks = self._setup_external_mocks()
    
    def _cleanup_test_data(self):
        """Clean up integration test data."""
        self._teardown_test_database()
        self._teardown_test_cache()
        self._teardown_external_mocks()
    
    def _setup_test_database(self) -> Session:
        """Set up test database with transaction rollback."""
        engine = create_engine("sqlite:///:memory:")
        SessionLocal = sessionmaker(bind=engine)
        return SessionLocal()
    
    def _teardown_test_database(self):
        """Tear down test database."""
        if hasattr(self, 'test_db'):
            self.test_db.close()
    
    def _setup_test_cache(self) -> Dict[str, Any]:
        """Set up test cache."""
        return {}
    
    def _teardown_test_cache(self):
        """Clear test cache."""
        if hasattr(self, 'test_cache'):
            self.test_cache.clear()
    
    def _setup_external_mocks(self) -> Dict[str, Mock]:
        """Set up mocks for external services."""
        return {
            'crypto_api': Mock(),
            'llm_client': Mock(),
            'market_data_api': Mock()
        }
    
    def _teardown_external_mocks(self):
        """Clean up external mocks."""
        if hasattr(self, 'external_mocks'):
            self.external_mocks.clear()


class BaseAsyncTest(BaseTest):
    """
    Base class for async tests.
    
    Provides utilities for testing async code.
    """
    
    def _setup_test_data(self):
        """Set up async test data."""
        self.event_loop = asyncio.get_event_loop()
        self.async_mocks = {}
    
    def _cleanup_test_data(self):
        """Clean up async test data."""
        # Cancel any pending tasks
        pending = asyncio.all_tasks(self.event_loop)
        for task in pending:
            task.cancel()
        self.async_mocks.clear()
    
    @asynccontextmanager
    async def async_performance(self, max_duration: float):
        """Async context manager to assert performance requirements."""
        start_time = datetime.now(timezone.utc)
        yield
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        assert duration < max_duration, f"Async operation took {duration:.2f}s, expected < {max_duration}s"
    
    def create_async_mock(self, **kwargs) -> AsyncMock:
        """Create an async mock with default configuration."""
        return AsyncMock(**kwargs)
    
    async def wait_for_condition(self, condition_func, timeout: float = 5.0, interval: float = 0.1):
        """Wait for a condition to become true."""
        start_time = datetime.now(timezone.utc)
        while not condition_func():
            if (datetime.now(timezone.utc) - start_time).total_seconds() > timeout:
                raise TimeoutError(f"Condition not met within {timeout}s")
            await asyncio.sleep(interval)


class BaseE2ETest(BaseTest):
    """
    Base class for end-to-end tests.
    
    E2E tests should:
    - Test complete user workflows
    - Use real implementations of all components
    - May use test doubles for external services
    - May take significant time (< 60s per test)
    """
    
    TEST_TIMEOUT = 60
    
    def _setup_test_data(self):
        """Set up E2E test environment."""
        self.test_config = self._create_test_config()
        self.test_server = self._start_test_server()
        self.test_client = self._create_test_client()
        self.test_data_dir = self._create_test_data_directory()
    
    def _cleanup_test_data(self):
        """Clean up E2E test environment."""
        self._stop_test_server()
        self._cleanup_test_data_directory()
    
    def _create_test_config(self) -> Dict[str, Any]:
        """Create test configuration."""
        return {
            'api_base_url': 'http://localhost:8000',
            'database_url': 'sqlite:///:memory:',
            'cache_enabled': False,
            'rate_limiting_enabled': False,
            'test_mode': True
        }
    
    def _start_test_server(self):
        """Start test server. Override in subclasses."""
        return Mock()
    
    def _stop_test_server(self):
        """Stop test server."""
        if hasattr(self, 'test_server') and self.test_server:
            self.test_server.stop()
    
    def _create_test_client(self):
        """Create test client for API calls."""
        return Mock()
    
    def _create_test_data_directory(self) -> Path:
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp(prefix='e2e_test_')
        return Path(temp_dir)
    
    def _cleanup_test_data_directory(self):
        """Clean up test data directory."""
        if hasattr(self, 'test_data_dir') and self.test_data_dir.exists():
            import shutil
            shutil.rmtree(self.test_data_dir)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def make_api_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make API request with retry logic."""
        response = self.test_client.request(method, endpoint, **kwargs)
        response.raise_for_status()
        return response.json()


class BasePerformanceTest(BaseTest):
    """
    Base class for performance tests.
    
    Performance tests should:
    - Measure response times, throughput, and resource usage
    - Compare against defined benchmarks
    - Generate performance reports
    """
    
    def _setup_test_data(self):
        """Set up performance test data."""
        self.metrics = {
            'response_times': [],
            'throughput': [],
            'cpu_usage': [],
            'memory_usage': []
        }
        self.benchmarks = self._load_benchmarks()
    
    def _cleanup_test_data(self):
        """Generate performance report after test."""
        self._generate_performance_report()
        self.metrics.clear()
    
    def _load_benchmarks(self) -> Dict[str, float]:
        """Load performance benchmarks."""
        return {
            'max_response_time_ms': 100,
            'min_throughput_rps': 100,
            'max_cpu_percent': 80,
            'max_memory_mb': 512
        }
    
    def _generate_performance_report(self):
        """Generate performance test report."""
        if not self.metrics['response_times']:
            return
            
        import statistics
        
        report = {
            'response_time': {
                'mean': statistics.mean(self.metrics['response_times']),
                'median': statistics.median(self.metrics['response_times']),
                'p95': statistics.quantiles(self.metrics['response_times'], n=20)[18],
                'p99': statistics.quantiles(self.metrics['response_times'], n=100)[98]
            }
        }
        
        self.logger.info(f"Performance Report: {json.dumps(report, indent=2)}")
    
    def record_metric(self, metric_type: str, value: float):
        """Record a performance metric."""
        if metric_type in self.metrics:
            self.metrics[metric_type].append(value)
    
    def assert_performance_benchmark(self, metric_type: str, value: float):
        """Assert that a metric meets its benchmark."""
        benchmark_key = f"max_{metric_type}"
        if benchmark_key in self.benchmarks:
            assert value <= self.benchmarks[benchmark_key], \
                f"{metric_type} ({value}) exceeds benchmark ({self.benchmarks[benchmark_key]})"


class BaseSecurityTest(BaseTest):
    """
    Base class for security tests.
    
    Security tests should:
    - Test authentication and authorization
    - Check for common vulnerabilities
    - Validate input sanitization
    - Test rate limiting and abuse prevention
    """
    
    def _setup_test_data(self):
        """Set up security test data."""
        self.attack_vectors = self._load_attack_vectors()
        self.test_users = self._create_test_users()
    
    def _cleanup_test_data(self):
        """Clean up security test data."""
        self.attack_vectors.clear()
        self.test_users.clear()
    
    def _load_attack_vectors(self) -> Dict[str, list]:
        """Load common attack vectors for testing."""
        return {
            'sql_injection': [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "admin'--",
                "1' UNION SELECT * FROM users--"
            ],
            'xss': [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "javascript:alert('XSS')"
            ],
            'path_traversal': [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "%2e%2e%2f%2e%2e%2f"
            ]
        }
    
    def _create_test_users(self) -> Dict[str, Dict[str, Any]]:
        """Create test users with different permission levels."""
        return {
            'admin': {'role': 'admin', 'permissions': ['*']},
            'user': {'role': 'user', 'permissions': ['read', 'write']},
            'guest': {'role': 'guest', 'permissions': ['read']}
        }
    
    def test_attack_vector(self, vector_type: str, test_func):
        """Test a function against attack vectors."""
        vectors = self.attack_vectors.get(vector_type, [])
        for vector in vectors:
            with pytest.raises(Exception):
                test_func(vector)