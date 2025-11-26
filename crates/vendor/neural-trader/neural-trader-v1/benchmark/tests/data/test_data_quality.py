"""Test suite for data quality and validation.

Tests data integrity, validation rules, anomaly detection,
and quality metrics collection.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock, patch

from benchmark.src.data.data_validator import (
    DataValidator,
    ValidationRule,
    ValidationResult,
    DataQualityMetrics,
    AnomalyDetector,
    ValidationConfig,
    QualityThreshold,
)
from benchmark.src.data.realtime_feed import DataUpdate, DataSource


class TestDataValidator:
    """Test data validation functionality."""
    
    @pytest.fixture
    def validation_config(self):
        """Create validation configuration."""
        return ValidationConfig(
            price_min=0.01,
            price_max=100000.0,
            volume_min=0,
            volume_max=1e12,
            max_price_change_percent=20.0,
            max_spread_percent=5.0,
            required_fields=["symbol", "price", "timestamp"],
            timestamp_tolerance_seconds=60,
            enable_anomaly_detection=True,
            anomaly_sensitivity=2.5,
        )
    
    @pytest.fixture
    def validator(self, validation_config):
        """Create data validator instance."""
        return DataValidator(validation_config)
    
    def test_basic_validation(self, validator):
        """Test basic data validation rules."""
        # Valid update
        update = DataUpdate("AAPL", 150.25, time.time(), DataSource.WEBSOCKET)
        result = validator.validate(update)
        assert result.is_valid
        assert len(result.errors) == 0
        
        # Invalid price (negative)
        update = DataUpdate("AAPL", -10.0, time.time(), DataSource.WEBSOCKET)
        result = validator.validate(update)
        assert not result.is_valid
        assert "price" in str(result.errors[0])
        
        # Invalid price (too high)
        update = DataUpdate("AAPL", 1000000.0, time.time(), DataSource.WEBSOCKET)
        result = validator.validate(update)
        assert not result.is_valid
        assert "price" in str(result.errors[0])
    
    def test_timestamp_validation(self, validator):
        """Test timestamp validation."""
        # Current timestamp - valid
        update = DataUpdate("AAPL", 150.25, time.time(), DataSource.WEBSOCKET)
        result = validator.validate(update)
        assert result.is_valid
        
        # Future timestamp - invalid
        future_time = time.time() + 120  # 2 minutes in future
        update = DataUpdate("AAPL", 150.25, future_time, DataSource.WEBSOCKET)
        result = validator.validate(update)
        assert not result.is_valid
        assert "timestamp" in str(result.errors[0])
        
        # Old timestamp - invalid
        old_time = time.time() - 120  # 2 minutes old
        update = DataUpdate("AAPL", 150.25, old_time, DataSource.WEBSOCKET)
        result = validator.validate(update)
        assert not result.is_valid
        assert "stale" in str(result.errors[0])
    
    def test_price_change_validation(self, validator):
        """Test price change validation."""
        # Set previous price
        validator.update_price_history("AAPL", 150.0)
        
        # Normal price change - valid
        update = DataUpdate("AAPL", 155.0, time.time(), DataSource.WEBSOCKET)
        result = validator.validate(update)
        assert result.is_valid
        
        # Excessive price change - invalid
        update = DataUpdate("AAPL", 200.0, time.time(), DataSource.WEBSOCKET)
        result = validator.validate(update)
        assert not result.is_valid
        assert "price change" in str(result.errors[0])
    
    def test_custom_validation_rules(self, validator):
        """Test custom validation rules."""
        # Add custom rule
        def crypto_rule(update: DataUpdate) -> ValidationResult:
            if update.symbol.startswith("BTC") and update.price < 10000:
                return ValidationResult(
                    is_valid=False,
                    errors=["Bitcoin price suspiciously low"]
                )
            return ValidationResult(is_valid=True)
        
        validator.add_custom_rule("crypto_check", crypto_rule)
        
        # Test crypto rule
        update = DataUpdate("BTC-USD", 5000.0, time.time(), DataSource.WEBSOCKET)
        result = validator.validate(update)
        assert not result.is_valid
        assert "suspiciously low" in str(result.errors[0])
    
    def test_batch_validation(self, validator):
        """Test batch validation performance."""
        updates = []
        for i in range(1000):
            updates.append(
                DataUpdate(
                    f"SYM{i % 100}",
                    100 + (i % 50),
                    time.time(),
                    DataSource.WEBSOCKET
                )
            )
        
        start_time = time.perf_counter()
        results = validator.validate_batch(updates)
        end_time = time.perf_counter()
        
        assert len(results) == 1000
        assert all(r.is_valid for r in results)
        
        # Should process 1000 updates in under 100ms
        processing_time = (end_time - start_time) * 1000
        assert processing_time < 100


class TestAnomalyDetector:
    """Test anomaly detection functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create anomaly detector."""
        return AnomalyDetector(
            window_size=100,
            sensitivity=2.0,
            min_samples=20
        )
    
    def test_normal_data_stream(self, detector):
        """Test anomaly detection on normal data."""
        # Feed normal price data
        for i in range(50):
            price = 100 + (i % 5) * 0.1  # Small variations
            detector.add_observation("AAPL", price)
        
        # Check new normal value
        is_anomaly = detector.is_anomaly("AAPL", 100.3)
        assert not is_anomaly
    
    def test_anomaly_detection(self, detector):
        """Test detection of anomalous values."""
        # Feed consistent data
        for i in range(30):
            detector.add_observation("AAPL", 100.0 + (i % 3) * 0.1)
        
        # Test anomalous value
        is_anomaly = detector.is_anomaly("AAPL", 150.0)  # 50% jump
        assert is_anomaly
        
        # Test another anomaly
        is_anomaly = detector.is_anomaly("AAPL", 50.0)  # 50% drop
        assert is_anomaly
    
    def test_insufficient_data(self, detector):
        """Test behavior with insufficient historical data."""
        # Add only a few observations
        for i in range(5):
            detector.add_observation("AAPL", 100.0 + i)
        
        # Should not detect anomalies without enough data
        is_anomaly = detector.is_anomaly("AAPL", 200.0)
        assert not is_anomaly  # Not enough data to determine
    
    def test_multiple_symbols(self, detector):
        """Test anomaly detection for multiple symbols."""
        # Feed data for multiple symbols
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        for symbol in symbols:
            base_price = {"AAPL": 150, "GOOGL": 2800, "MSFT": 300}[symbol]
            for i in range(30):
                price = base_price + (i % 5) * 0.1
                detector.add_observation(symbol, price)
        
        # Test each symbol independently
        assert not detector.is_anomaly("AAPL", 150.5)
        assert detector.is_anomaly("AAPL", 200.0)
        
        assert not detector.is_anomaly("GOOGL", 2800.5)
        assert detector.is_anomaly("GOOGL", 3500.0)


class TestDataQualityMetrics:
    """Test data quality metrics collection."""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics collector."""
        return DataQualityMetrics()
    
    def test_metrics_collection(self, metrics):
        """Test basic metrics collection."""
        # Record some validations
        for i in range(100):
            if i % 10 == 0:
                metrics.record_validation(False, ["Test error"])
            else:
                metrics.record_validation(True, [])
        
        stats = metrics.get_statistics()
        
        assert stats["total_validations"] == 100
        assert stats["valid_count"] == 90
        assert stats["invalid_count"] == 10
        assert stats["validity_rate"] == 0.9
    
    def test_error_frequency(self, metrics):
        """Test error frequency tracking."""
        # Record various errors
        errors = [
            "Price out of range",
            "Invalid timestamp",
            "Price out of range",
            "Missing symbol",
            "Price out of range",
        ]
        
        for error in errors:
            metrics.record_validation(False, [error])
        
        error_freq = metrics.get_error_frequency()
        assert error_freq["Price out of range"] == 3
        assert error_freq["Invalid timestamp"] == 1
        assert error_freq["Missing symbol"] == 1
    
    def test_time_based_metrics(self, metrics):
        """Test time-based quality metrics."""
        # Simulate data over time
        start_time = time.time()
        
        # Good quality period
        for _ in range(50):
            metrics.record_validation(True, [])
        
        # Poor quality period
        for _ in range(20):
            metrics.record_validation(False, ["Quality degradation"])
        
        # Recovery period
        for _ in range(30):
            metrics.record_validation(True, [])
        
        stats = metrics.get_statistics()
        assert stats["validity_rate"] == 0.8
        
        # Get time series
        time_series = metrics.get_quality_time_series(interval_seconds=1)
        assert len(time_series) > 0


class TestQualityThreshold:
    """Test quality threshold monitoring."""
    
    @pytest.fixture
    def threshold_monitor(self):
        """Create threshold monitor."""
        return QualityThreshold(
            min_validity_rate=0.95,
            max_error_rate=0.05,
            evaluation_window=100
        )
    
    def test_threshold_monitoring(self, threshold_monitor):
        """Test quality threshold monitoring."""
        # Feed good quality data
        for _ in range(90):
            threshold_monitor.record_result(True)
        
        # Feed some errors
        for _ in range(10):
            threshold_monitor.record_result(False)
        
        # Check if threshold is breached
        assert not threshold_monitor.is_quality_acceptable()
        
        alerts = threshold_monitor.get_alerts()
        assert len(alerts) > 0
        assert "validity rate" in alerts[0].lower()
    
    def test_sliding_window(self, threshold_monitor):
        """Test sliding window evaluation."""
        # Fill window with good data
        for _ in range(100):
            threshold_monitor.record_result(True)
        
        assert threshold_monitor.is_quality_acceptable()
        
        # Add errors to shift window
        for _ in range(10):
            threshold_monitor.record_result(False)
        
        # Should still be acceptable as old good data is in window
        assert threshold_monitor.is_quality_acceptable()
        
        # Add more errors
        for _ in range(20):
            threshold_monitor.record_result(False)
        
        # Now should be unacceptable
        assert not threshold_monitor.is_quality_acceptable()


class TestIntegrationScenarios:
    """Test integrated validation scenarios."""
    
    @pytest.fixture
    def full_validator(self):
        """Create fully configured validator."""
        config = ValidationConfig(
            enable_anomaly_detection=True,
            collect_metrics=True,
            quality_thresholds=QualityThreshold(0.95, 0.05, 100)
        )
        return DataValidator(config)
    
    @pytest.mark.asyncio
    async def test_market_open_surge(self, full_validator):
        """Test validation during market open surge."""
        # Simulate pre-market calm
        for i in range(50):
            update = DataUpdate("AAPL", 150.0 + i * 0.01, time.time(), DataSource.WEBSOCKET)
            result = full_validator.validate(update)
            assert result.is_valid
        
        # Simulate market open surge
        surge_updates = []
        for i in range(1000):
            # Higher volatility, more updates
            price = 150.0 + (i % 20) * 0.5
            update = DataUpdate("AAPL", price, time.time(), DataSource.WEBSOCKET)
            surge_updates.append(update)
        
        # Validate surge
        results = full_validator.validate_batch(surge_updates)
        valid_count = sum(1 for r in results if r.is_valid)
        
        # Most should still be valid despite volatility
        assert valid_count / len(results) > 0.8
    
    @pytest.mark.asyncio
    async def test_flash_crash_detection(self, full_validator):
        """Test detection of flash crash events."""
        # Establish normal trading
        for i in range(100):
            update = DataUpdate("SPY", 400.0 + i * 0.1, time.time(), DataSource.WEBSOCKET)
            full_validator.validate(update)
        
        # Simulate flash crash
        crash_updates = [
            DataUpdate("SPY", 380.0, time.time(), DataSource.WEBSOCKET),  # -5%
            DataUpdate("SPY", 360.0, time.time() + 0.1, DataSource.WEBSOCKET),  # -10%
            DataUpdate("SPY", 340.0, time.time() + 0.2, DataSource.WEBSOCKET),  # -15%
        ]
        
        crash_results = []
        for update in crash_updates:
            result = full_validator.validate(update)
            crash_results.append(result)
        
        # Should detect anomalies
        assert not all(r.is_valid for r in crash_results)
        
        # Check if alerts were generated
        alerts = full_validator.get_quality_alerts()
        assert any("anomaly" in alert.lower() for alert in alerts)