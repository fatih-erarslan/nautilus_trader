"""
Test utilities for AI News Trading Platform.

This module provides utility functions and fixtures for testing,
including async helpers, time mocking, and response fixtures.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aioresponses import aioresponses
from faker import Faker
from freezegun import freeze_time
from responses import RequestsMock

fake = Faker()


class AsyncTestHelper:
    """Helper class for async testing utilities."""
    
    @staticmethod
    def run_sync(coro):
        """Run an async coroutine synchronously."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    @staticmethod
    @asynccontextmanager
    async def timeout_after(seconds: float):
        """Context manager to timeout after specified seconds."""
        try:
            async with asyncio.timeout(seconds):
                yield
        except asyncio.TimeoutError:
            pytest.fail(f"Operation timed out after {seconds} seconds")
    
    @staticmethod
    async def wait_for_condition(
        condition_func,
        timeout: float = 5.0,
        interval: float = 0.1,
        error_message: str = "Condition not met"
    ):
        """Wait for a condition to become true with timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return
            await asyncio.sleep(interval)
        
        pytest.fail(f"{error_message} within {timeout} seconds")
    
    @staticmethod
    async def collect_async_generator(async_gen, max_items: int = 100):
        """Collect items from an async generator."""
        items = []
        async for item in async_gen:
            items.append(item)
            if len(items) >= max_items:
                break
        return items
    
    @staticmethod
    def create_async_mock(**kwargs) -> AsyncMock:
        """Create an AsyncMock with common configuration."""
        return AsyncMock(**kwargs)


class TimeHelper:
    """Helper class for time-related testing utilities."""
    
    @staticmethod
    @contextmanager
    def freeze_at(timestamp: Union[str, datetime]):
        """Freeze time at specific timestamp."""
        with freeze_time(timestamp):
            yield
    
    @staticmethod
    @contextmanager
    def travel_to(timestamp: Union[str, datetime]):
        """Travel to specific time and allow time to move."""
        with freeze_time(timestamp, auto_tick_seconds=1):
            yield
    
    @staticmethod
    def create_timestamp_sequence(
        start: datetime,
        count: int,
        interval_seconds: int = 60
    ) -> List[datetime]:
        """Create a sequence of timestamps."""
        timestamps = []
        current = start
        for i in range(count):
            timestamps.append(current)
            current = current.replace(
                timestamp=current.timestamp() + interval_seconds
            )
        return timestamps
    
    @staticmethod
    def assert_recent(timestamp: datetime, tolerance_seconds: int = 5):
        """Assert that timestamp is recent (within tolerance)."""
        now = datetime.now(timezone.utc)
        diff = abs((now - timestamp).total_seconds())
        assert diff <= tolerance_seconds, \
            f"Timestamp {timestamp} is not recent (diff: {diff}s)"


class APIResponseHelper:
    """Helper class for mocking API responses."""
    
    @staticmethod
    @contextmanager
    def mock_requests():
        """Context manager for mocking requests."""
        with RequestsMock() as rsps:
            yield rsps
    
    @staticmethod
    @contextmanager
    def mock_aiohttp():
        """Context manager for mocking aiohttp requests."""
        with aioresponses() as m:
            yield m
    
    @staticmethod
    def create_successful_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a successful API response."""
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": fake.uuid4()
        }
    
    @staticmethod
    def create_error_response(
        error_code: str,
        error_message: str,
        status_code: int = 400
    ) -> Dict[str, Any]:
        """Create an error API response."""
        return {
            "status": "error",
            "error": {
                "code": error_code,
                "message": error_message
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": fake.uuid4()
        }
    
    @staticmethod
    def create_rate_limit_response() -> Dict[str, Any]:
        """Create a rate limit error response."""
        return APIResponseHelper.create_error_response(
            "RATE_LIMIT_EXCEEDED",
            "API rate limit exceeded. Please retry after 60 seconds.",
            429
        )
    
    @staticmethod
    def create_auth_error_response() -> Dict[str, Any]:
        """Create an authentication error response."""
        return APIResponseHelper.create_error_response(
            "UNAUTHORIZED",
            "Invalid API key or insufficient permissions.",
            401
        )


class DatabaseHelper:
    """Helper class for database testing utilities."""
    
    @staticmethod
    @contextmanager
    def temporary_db_session(engine):
        """Create a temporary database session with rollback."""
        connection = engine.connect()
        transaction = connection.begin()
        
        try:
            from sqlalchemy.orm import sessionmaker
            Session = sessionmaker(bind=connection)
            session = Session()
            yield session
        finally:
            session.close()
            transaction.rollback()
            connection.close()
    
    @staticmethod
    def create_test_data(session, factory_class, count: int = 1, **kwargs):
        """Create test data using a factory."""
        instances = []
        for _ in range(count):
            instance = factory_class(**kwargs)
            session.add(instance)
            instances.append(instance)
        session.commit()
        return instances if count > 1 else instances[0]
    
    @staticmethod
    def assert_db_count(session, model_class, expected_count: int):
        """Assert database record count."""
        actual_count = session.query(model_class).count()
        assert actual_count == expected_count, \
            f"Expected {expected_count} {model_class.__name__} records, got {actual_count}"


class FixtureHelper:
    """Helper class for test fixtures and data."""
    
    @staticmethod
    def load_fixture(fixture_name: str) -> Dict[str, Any]:
        """Load test fixture from JSON file."""
        fixture_path = Path(__file__).parent / "fixtures" / f"{fixture_name}.json"
        with open(fixture_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_fixture(fixture_name: str, data: Dict[str, Any]):
        """Save test data as fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / f"{fixture_name}.json"
        fixture_path.parent.mkdir(exist_ok=True)
        with open(fixture_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def create_market_data_fixture() -> Dict[str, Any]:
        """Create market data fixture."""
        return {
            "symbol": "BTC_USDT",
            "price": 50000.0,
            "volume_24h": 1000000.0,
            "price_change_24h": 2.5,
            "high_24h": 51000.0,
            "low_24h": 49000.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def create_news_fixture() -> Dict[str, Any]:
        """Create news article fixture."""
        return {
            "id": fake.uuid4(),
            "title": "Bitcoin Reaches New All-Time High",
            "summary": "Bitcoin price surged to a new record high.",
            "content": fake.text(max_nb_chars=1000),
            "source": "CoinDesk",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "sentiment": "bullish",
            "impact_score": 0.8
        }


class PerformanceHelper:
    """Helper class for performance testing utilities."""
    
    @staticmethod
    @contextmanager
    def measure_time():
        """Context manager to measure execution time."""
        start_time = time.perf_counter()
        yield lambda: time.perf_counter() - start_time
    
    @staticmethod
    @contextmanager
    def assert_max_time(max_seconds: float, operation_name: str = "Operation"):
        """Assert that operation completes within time limit."""
        start_time = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start_time
        assert elapsed <= max_seconds, \
            f"{operation_name} took {elapsed:.3f}s, expected <= {max_seconds}s"
    
    @staticmethod
    def benchmark_function(func, *args, iterations: int = 100, **kwargs):
        """Benchmark a function over multiple iterations."""
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
        
        return {
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": sum(times) / len(times),
            "total_time": sum(times),
            "iterations": iterations
        }


class MockHelper:
    """Helper class for creating mocks and stubs."""
    
    @staticmethod
    def create_crypto_api_mock() -> Mock:
        """Create a mock crypto API client."""
        mock = Mock()
        mock.get_ticker.return_value = {
            "symbol": "BTC_USDT",
            "price": 50000.0,
            "volume": 1000.0
        }
        mock.get_orderbook.return_value = {
            "bids": [[49990, 1.0], [49980, 2.0]],
            "asks": [[50010, 1.0], [50020, 2.0]]
        }
        return mock
    
    @staticmethod
    def create_llm_mock() -> AsyncMock:
        """Create a mock LLM client."""
        mock = AsyncMock()
        mock.analyze_sentiment.return_value = {
            "sentiment": "bullish",
            "confidence": 0.85,
            "reasoning": "Positive market indicators"
        }
        mock.generate_signal.return_value = {
            "action": "buy",
            "confidence": 0.8,
            "price_target": 52000
        }
        return mock
    
    @staticmethod
    def create_news_api_mock() -> Mock:
        """Create a mock news API client."""
        mock = Mock()
        mock.get_latest_news.return_value = [
            {
                "title": "Bitcoin price surges",
                "content": "Bitcoin reached new highs...",
                "source": "CoinDesk",
                "published_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        return mock


class AssertionHelper:
    """Helper class for custom assertions."""
    
    @staticmethod
    def assert_dict_contains(actual: Dict, expected_subset: Dict):
        """Assert that actual dict contains all key-value pairs from expected."""
        for key, value in expected_subset.items():
            assert key in actual, f"Key '{key}' not found in actual dict"
            assert actual[key] == value, \
                f"Value mismatch for key '{key}': expected {value}, got {actual[key]}"
    
    @staticmethod
    def assert_list_contains_item(items: List, predicate):
        """Assert that list contains at least one item matching predicate."""
        matching_items = [item for item in items if predicate(item)]
        assert len(matching_items) > 0, "No items in list match the predicate"
    
    @staticmethod
    def assert_timestamp_between(
        timestamp: datetime,
        start: datetime,
        end: datetime
    ):
        """Assert that timestamp is between start and end."""
        assert start <= timestamp <= end, \
            f"Timestamp {timestamp} not between {start} and {end}"
    
    @staticmethod
    def assert_numeric_close(actual: float, expected: float, tolerance: float = 0.01):
        """Assert that two numbers are close within tolerance."""
        diff = abs(actual - expected)
        assert diff <= tolerance, \
            f"Numbers not close enough: {actual} vs {expected} (diff: {diff}, tolerance: {tolerance})"


# Pytest fixtures that can be imported by test modules
@pytest.fixture
def async_helper():
    """Provide AsyncTestHelper instance."""
    return AsyncTestHelper()


@pytest.fixture
def time_helper():
    """Provide TimeHelper instance."""
    return TimeHelper()


@pytest.fixture
def api_helper():
    """Provide APIResponseHelper instance."""
    return APIResponseHelper()


@pytest.fixture
def db_helper():
    """Provide DatabaseHelper instance."""
    return DatabaseHelper()


@pytest.fixture
def fixture_helper():
    """Provide FixtureHelper instance."""
    return FixtureHelper()


@pytest.fixture
def performance_helper():
    """Provide PerformanceHelper instance."""
    return PerformanceHelper()


@pytest.fixture
def mock_helper():
    """Provide MockHelper instance."""
    return MockHelper()


@pytest.fixture
def assert_helper():
    """Provide AssertionHelper instance."""
    return AssertionHelper()


@pytest.fixture
def mock_crypto_api(mock_helper):
    """Provide a mocked crypto API."""
    return mock_helper.create_crypto_api_mock()


@pytest.fixture
def mock_llm_client(mock_helper):
    """Provide a mocked LLM client."""
    return mock_helper.create_llm_mock()


@pytest.fixture
def mock_news_api(mock_helper):
    """Provide a mocked news API."""
    return mock_helper.create_news_api_mock()


@pytest.fixture
def sample_market_data(fixture_helper):
    """Provide sample market data."""
    return fixture_helper.create_market_data_fixture()


@pytest.fixture
def sample_news_article(fixture_helper):
    """Provide sample news article."""
    return fixture_helper.create_news_fixture()


# Context managers that can be used in tests
@contextmanager
def temporary_environment_variable(key: str, value: str):
    """Temporarily set an environment variable."""
    import os
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@contextmanager
def capture_logs(logger_name: str, level: str = "INFO"):
    """Capture logs from a specific logger."""
    import logging
    from io import StringIO
    
    logger = logging.getLogger(logger_name)
    old_level = logger.level
    logger.setLevel(getattr(logging, level))
    
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


# Decorators for common test scenarios
def requires_external_service(service_name: str):
    """Skip test if external service is not available."""
    def decorator(test_func):
        return pytest.mark.skipif(
            condition=True,  # Would check if service is available
            reason=f"External service {service_name} not available"
        )(test_func)
    return decorator


def slow_test(test_func):
    """Mark test as slow."""
    return pytest.mark.slow(test_func)


def integration_test(test_func):
    """Mark test as integration test."""
    return pytest.mark.integration(test_func)


def e2e_test(test_func):
    """Mark test as end-to-end test."""
    return pytest.mark.e2e(test_func)