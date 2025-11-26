"""
Comprehensive tests for risk management system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from decimal import Decimal

from src.alpaca_trading.risk_management import (
    RiskManager,
    PositionSizer,
    StopLossManager,
    DrawdownMonitor,
    RiskLimits,
    RiskMetrics,
    RiskAlert,
    RiskAlertType
)


@pytest.fixture
def risk_limits():
    """Default risk limits."""
    return RiskLimits(
        max_position_size=10000,
        max_portfolio_risk=0.02,  # 2%
        max_daily_loss=1000,
        max_drawdown=0.10,  # 10%
        max_correlation=0.70,
        position_limit_per_symbol=5000
    )


@pytest.fixture
def mock_portfolio():
    """Mock portfolio data."""
    return {
        'cash': 50000,
        'equity': 100000,
        'positions': {
            'AAPL': {'quantity': 100, 'avg_price': 150, 'current_price': 155},
            'GOOGL': {'quantity': 50, 'avg_price': 2800, 'current_price': 2850}
        }
    }


@pytest.fixture
async def risk_manager(risk_limits):
    """Create risk manager."""
    manager = RiskManager(risk_limits)
    yield manager
    await manager.shutdown()


class TestRiskLimits:
    """Test risk limit validation."""
    
    def test_risk_limits_validation(self):
        """Test risk limits are properly validated."""
        # Valid limits
        limits = RiskLimits(
            max_position_size=10000,
            max_portfolio_risk=0.02,
            max_daily_loss=1000
        )
        assert limits.max_position_size == 10000
        assert limits.max_portfolio_risk == 0.02
        
        # Invalid limits should raise error
        with pytest.raises(ValueError):
            RiskLimits(max_position_size=-1000)
        
        with pytest.raises(ValueError):
            RiskLimits(max_portfolio_risk=1.5)  # > 100%


class TestPositionSizer:
    """Test position sizing algorithms."""
    
    def test_fixed_position_sizing(self):
        """Test fixed position sizing."""
        sizer = PositionSizer(strategy='fixed', fixed_size=1000)
        
        size = sizer.calculate_position_size(
            symbol='AAPL',
            price=150,
            portfolio_value=100000,
            volatility=0.02
        )
        
        assert size == 1000
    
    def test_kelly_criterion_sizing(self):
        """Test Kelly criterion position sizing."""
        sizer = PositionSizer(strategy='kelly', max_kelly_fraction=0.25)
        
        size = sizer.calculate_position_size(
            symbol='AAPL',
            price=150,
            portfolio_value=100000,
            volatility=0.02,
            expected_return=0.10,
            win_rate=0.60
        )
        
        # Kelly fraction = (p*b - q) / b
        # where p = win_rate, q = 1-p, b = expected_return / volatility
        assert size > 0
        assert size <= 25000  # Max 25% of portfolio
    
    def test_volatility_based_sizing(self):
        """Test volatility-based position sizing."""
        sizer = PositionSizer(strategy='volatility', risk_per_trade=0.01)
        
        # High volatility should result in smaller position
        size_high_vol = sizer.calculate_position_size(
            symbol='AAPL',
            price=150,
            portfolio_value=100000,
            volatility=0.05  # 5% volatility
        )
        
        # Low volatility should result in larger position
        size_low_vol = sizer.calculate_position_size(
            symbol='AAPL',
            price=150,
            portfolio_value=100000,
            volatility=0.01  # 1% volatility
        )
        
        assert size_low_vol > size_high_vol
        
        # Check risk limit
        assert size_high_vol * 150 * 0.05 <= 1000  # 1% of portfolio
    
    def test_position_size_limits(self):
        """Test position size limits are enforced."""
        sizer = PositionSizer(
            strategy='fixed',
            fixed_size=10000,
            max_position_value=5000
        )
        
        size = sizer.calculate_position_size(
            symbol='AAPL',
            price=150,
            portfolio_value=100000
        )
        
        # Should be limited to max_position_value / price
        assert size == 33  # floor(5000 / 150)


class TestStopLossManager:
    """Test stop loss management."""
    
    @pytest.mark.asyncio
    async def test_fixed_stop_loss(self):
        """Test fixed percentage stop loss."""
        manager = StopLossManager(default_stop_percentage=0.02)
        
        # Set stop loss
        stop_price = await manager.calculate_stop_loss(
            symbol='AAPL',
            entry_price=150,
            position_side='long',
            strategy='fixed'
        )
        
        assert stop_price == 147  # 2% below entry
    
    @pytest.mark.asyncio
    async def test_atr_based_stop_loss(self):
        """Test ATR-based stop loss."""
        manager = StopLossManager(atr_multiplier=2.0)
        
        stop_price = await manager.calculate_stop_loss(
            symbol='AAPL',
            entry_price=150,
            position_side='long',
            strategy='atr',
            atr_value=2.5
        )
        
        assert stop_price == 145  # 150 - (2.0 * 2.5)
    
    @pytest.mark.asyncio
    async def test_trailing_stop_update(self):
        """Test trailing stop loss updates."""
        manager = StopLossManager(default_stop_percentage=0.02)
        
        # Initial stop
        await manager.set_stop_loss('AAPL', 150, 147, trailing=True)
        
        # Price moves up - stop should trail
        new_stop = await manager.update_trailing_stop('AAPL', 155)
        assert new_stop == 151.9  # 2% below new high
        
        # Price moves down - stop should not change
        new_stop = await manager.update_trailing_stop('AAPL', 153)
        assert new_stop == 151.9  # Unchanged
    
    @pytest.mark.asyncio
    async def test_stop_loss_triggered(self):
        """Test stop loss trigger detection."""
        manager = StopLossManager()
        
        # Set stop loss
        await manager.set_stop_loss('AAPL', 150, 147)
        
        # Check if triggered
        triggered = await manager.check_stop_triggered('AAPL', 146)
        assert triggered
        
        triggered = await manager.check_stop_triggered('AAPL', 148)
        assert not triggered


class TestDrawdownMonitor:
    """Test drawdown monitoring."""
    
    @pytest.mark.asyncio
    async def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        monitor = DrawdownMonitor(max_drawdown_threshold=0.10)
        
        # Simulate equity curve
        equity_values = [100000, 105000, 110000, 108000, 105000, 103000]
        
        for value in equity_values:
            await monitor.update_equity(value)
        
        metrics = monitor.get_drawdown_metrics()
        
        # Max drawdown from 110000 to 103000
        assert metrics['current_drawdown'] == pytest.approx(0.0636, rel=0.01)
        assert metrics['max_drawdown'] == pytest.approx(0.0636, rel=0.01)
        assert metrics['peak_equity'] == 110000
    
    @pytest.mark.asyncio
    async def test_drawdown_alert(self):
        """Test drawdown alerts."""
        monitor = DrawdownMonitor(max_drawdown_threshold=0.05)
        
        alert_triggered = False
        
        def alert_callback(alert):
            nonlocal alert_triggered
            alert_triggered = True
        
        monitor.add_alert_callback(alert_callback)
        
        # Simulate drawdown exceeding threshold
        await monitor.update_equity(100000)
        await monitor.update_equity(105000)
        await monitor.update_equity(99000)  # > 5% drawdown
        
        assert alert_triggered
    
    @pytest.mark.asyncio
    async def test_recovery_tracking(self):
        """Test drawdown recovery tracking."""
        monitor = DrawdownMonitor()
        
        # Simulate drawdown and recovery
        await monitor.update_equity(100000)
        await monitor.update_equity(110000)  # Peak
        await monitor.update_equity(99000)   # Drawdown
        drawdown_start = datetime.now()
        
        await asyncio.sleep(0.1)
        
        await monitor.update_equity(110000)  # Recovery
        
        metrics = monitor.get_drawdown_metrics()
        assert metrics['current_drawdown'] == 0
        assert metrics['recovery_time'] > timedelta(0)


class TestRiskManager:
    """Test main risk manager."""
    
    @pytest.mark.asyncio
    async def test_pre_trade_risk_check(self, risk_manager, mock_portfolio):
        """Test pre-trade risk validation."""
        # Valid trade
        result = await risk_manager.check_trade_risk(
            symbol='AAPL',
            side='buy',
            quantity=10,
            price=150,
            portfolio=mock_portfolio
        )
        
        assert result.approved
        assert result.risk_score < 1.0
    
    @pytest.mark.asyncio
    async def test_position_limit_check(self, risk_manager, mock_portfolio):
        """Test position limit enforcement."""
        # Try to exceed position limit
        result = await risk_manager.check_trade_risk(
            symbol='AAPL',
            side='buy',
            quantity=100,  # Would make total position too large
            price=150,
            portfolio=mock_portfolio
        )
        
        assert not result.approved
        assert 'Position limit exceeded' in result.reason
    
    @pytest.mark.asyncio
    async def test_daily_loss_limit(self, risk_manager):
        """Test daily loss limit enforcement."""
        # Record losses
        await risk_manager.record_trade_result('AAPL', -500)
        await risk_manager.record_trade_result('GOOGL', -400)
        
        # Next trade should be blocked (would exceed $1000 daily loss limit)
        result = await risk_manager.check_trade_risk(
            symbol='MSFT',
            side='buy',
            quantity=100,
            price=300,
            portfolio={'cash': 50000, 'equity': 100000}
        )
        
        assert not result.approved
        assert 'Daily loss limit' in result.reason
    
    @pytest.mark.asyncio
    async def test_correlation_check(self, risk_manager):
        """Test portfolio correlation limits."""
        # Set correlation data
        await risk_manager.update_correlation_matrix({
            ('AAPL', 'MSFT'): 0.85,
            ('AAPL', 'GOOGL'): 0.75,
            ('MSFT', 'GOOGL'): 0.80
        })
        
        portfolio = {
            'positions': {
                'AAPL': {'quantity': 100, 'value': 15000},
                'MSFT': {'quantity': 50, 'value': 15000}
            }
        }
        
        # Try to add highly correlated position
        result = await risk_manager.check_trade_risk(
            symbol='GOOGL',
            side='buy',
            quantity=10,
            price=2800,
            portfolio=portfolio
        )
        
        assert not result.approved
        assert 'correlation' in result.reason.lower()
    
    @pytest.mark.asyncio
    async def test_risk_metrics_calculation(self, risk_manager, mock_portfolio):
        """Test risk metrics calculation."""
        metrics = await risk_manager.calculate_portfolio_risk(mock_portfolio)
        
        assert 'total_exposure' in metrics
        assert 'var_95' in metrics
        assert 'expected_shortfall' in metrics
        assert 'sharpe_ratio' in metrics
        assert metrics['total_exposure'] > 0
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, risk_manager):
        """Test real-time risk monitoring."""
        alerts = []
        
        async def alert_handler(alert):
            alerts.append(alert)
        
        risk_manager.add_alert_handler(alert_handler)
        
        # Start monitoring
        monitor_task = asyncio.create_task(risk_manager.start_monitoring())
        
        # Simulate risky conditions
        await risk_manager.update_portfolio_value(100000)
        await risk_manager.update_portfolio_value(88000)  # 12% drop
        
        # Give monitor time to detect
        await asyncio.sleep(0.1)
        
        # Should have triggered drawdown alert
        assert len(alerts) > 0
        assert any(a.type == RiskAlertType.DRAWDOWN_WARNING for a in alerts)
        
        # Stop monitoring
        risk_manager.stop_monitoring()
        await monitor_task
    
    @pytest.mark.asyncio
    async def test_emergency_liquidation(self, risk_manager):
        """Test emergency liquidation trigger."""
        liquidation_triggered = False
        
        async def liquidation_handler():
            nonlocal liquidation_triggered
            liquidation_triggered = True
        
        risk_manager.set_emergency_handler(liquidation_handler)
        
        # Trigger emergency conditions
        await risk_manager.update_portfolio_value(100000)
        await risk_manager.update_portfolio_value(85000)  # 15% drop
        
        assert liquidation_triggered
    
    @pytest.mark.asyncio
    async def test_risk_report_generation(self, risk_manager, mock_portfolio):
        """Test risk report generation."""
        # Add some trading history
        await risk_manager.record_trade_result('AAPL', 500)
        await risk_manager.record_trade_result('GOOGL', -200)
        await risk_manager.record_trade_result('MSFT', 300)
        
        report = await risk_manager.generate_risk_report(mock_portfolio)
        
        assert 'summary' in report
        assert 'positions' in report
        assert 'metrics' in report
        assert 'alerts' in report
        assert report['summary']['total_trades'] == 3
        assert report['summary']['profit_loss'] == 600