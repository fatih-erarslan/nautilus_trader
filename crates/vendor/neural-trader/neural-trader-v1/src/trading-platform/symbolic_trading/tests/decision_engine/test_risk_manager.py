"""
Tests for Risk Manager
Following TDD - Red-Green-Refactor approach
"""
import pytest
from datetime import datetime
from src.trading.decision_engine.models import (
    TradingSignal, SignalType, RiskLevel, 
    TradingStrategy, AssetType, PortfolioPosition
)


def test_risk_manager_initialization():
    """Test RiskManager initialization"""
    from src.trading.decision_engine.risk_manager import RiskManager
    
    # Default initialization
    risk_manager = RiskManager()
    assert risk_manager.max_position_size == 0.1  # 10% default
    assert risk_manager.max_portfolio_risk == 0.2  # 20% default
    assert risk_manager.max_correlation == 0.7
    
    # Custom initialization
    risk_manager = RiskManager(
        max_position_size=0.05,  # 5% max per position
        max_portfolio_risk=0.15,  # 15% max portfolio risk
        max_correlation=0.6      # Max correlation between positions
    )
    
    assert risk_manager.max_position_size == 0.05
    assert risk_manager.max_portfolio_risk == 0.15
    assert risk_manager.max_correlation == 0.6


@pytest.mark.asyncio
async def test_position_size_validation():
    """Test position size validation against risk limits"""
    from src.trading.decision_engine.risk_manager import RiskManager
    
    risk_manager = RiskManager(max_position_size=0.1)
    
    # Signal exceeding max position size
    signal = TradingSignal(
        id="test-signal",
        timestamp=datetime.now(),
        asset="BTC",
        asset_type=AssetType.CRYPTO,
        signal_type=SignalType.BUY,
        strategy=TradingStrategy.SWING,
        strength=0.9,
        confidence=0.8,
        risk_level=RiskLevel.HIGH,
        position_size=0.15,  # Exceeds max of 0.1
        entry_price=50000,
        stop_loss=48000,
        take_profit=52000,
        holding_period="3-7 days",
        source_events=["test"],
        reasoning="Test signal"
    )
    
    validated_signal = await risk_manager.validate_signal(signal)
    assert validated_signal.position_size <= 0.1
    # HIGH risk level has 0.05 limit, so position is capped at that instead
    assert validated_signal.position_size == 0.05


@pytest.mark.asyncio
async def test_portfolio_risk_calculation():
    """Test portfolio risk calculation"""
    from src.trading.decision_engine.risk_manager import RiskManager
    
    risk_manager = RiskManager()
    
    positions = {
        "BTC": {"size": 0.3, "risk": 0.05},  # 30% position, 5% risk per unit
        "ETH": {"size": 0.2, "risk": 0.04},  # 20% position, 4% risk per unit
        "ADA": {"size": 0.1, "risk": 0.03}   # 10% position, 3% risk per unit
    }
    
    total_risk = risk_manager.calculate_portfolio_risk(positions)
    # Expected: 0.3*0.05 + 0.2*0.04 + 0.1*0.03 = 0.015 + 0.008 + 0.003 = 0.026
    assert total_risk == pytest.approx(0.026, 0.001)
    assert total_risk > 0
    assert total_risk < 1


@pytest.mark.asyncio
async def test_portfolio_risk_adjustment():
    """Test position size adjustment based on current portfolio risk"""
    from src.trading.decision_engine.risk_manager import RiskManager
    
    risk_manager = RiskManager(max_portfolio_risk=0.2)
    
    # Current positions with high risk
    current_positions = {
        "BTC": {"size": 0.5, "risk": 0.06, "entry_price": 45000, "current_price": 48000},
        "ETH": {"size": 0.3, "risk": 0.05, "entry_price": 3000, "current_price": 3200}
    }
    
    # New signal that would exceed portfolio risk
    signal = TradingSignal(
        id="new-signal",
        timestamp=datetime.now(),
        asset="SOL",
        asset_type=AssetType.CRYPTO,
        signal_type=SignalType.BUY,
        strategy=TradingStrategy.MOMENTUM,
        strength=0.85,
        confidence=0.9,
        risk_level=RiskLevel.HIGH,
        position_size=0.1,  # Would add 10% position
        entry_price=100,
        stop_loss=90,
        take_profit=120,
        holding_period="1-3 days",
        source_events=["test"],
        reasoning="Test signal"
    )
    
    validated_signal = await risk_manager.validate_signal(signal, current_positions)
    
    # Position size should be reduced to keep total portfolio risk under limit
    assert validated_signal.position_size < signal.position_size
    
    # Calculate new total risk
    new_positions = current_positions.copy()
    new_positions[signal.asset] = {
        "size": validated_signal.position_size,
        "risk": 0.1  # 10% risk for this position
    }
    new_total_risk = risk_manager.calculate_portfolio_risk(new_positions)
    assert new_total_risk <= risk_manager.max_portfolio_risk


def test_stop_loss_validation():
    """Test stop loss validation and adjustment"""
    from src.trading.decision_engine.risk_manager import RiskManager
    
    risk_manager = RiskManager()
    
    # Test buy signal with invalid stop loss
    signal = TradingSignal(
        id="test-signal",
        timestamp=datetime.now(),
        asset="BTC",
        asset_type=AssetType.CRYPTO,
        signal_type=SignalType.BUY,
        strategy=TradingStrategy.SWING,
        strength=0.8,
        confidence=0.8,
        risk_level=RiskLevel.MEDIUM,
        position_size=0.05,
        entry_price=50000,
        stop_loss=52000,  # Invalid: above entry for buy
        take_profit=55000,
        holding_period="3-7 days",
        source_events=["test"],
        reasoning="Test"
    )
    
    validated = risk_manager._validate_stop_loss(signal)
    assert validated.stop_loss < validated.entry_price  # Should be corrected
    
    # Test sell signal with invalid stop loss
    signal_sell = TradingSignal(
        id="test-signal-2",
        timestamp=datetime.now(),
        asset="BTC",
        asset_type=AssetType.CRYPTO,
        signal_type=SignalType.SELL,
        strategy=TradingStrategy.SWING,
        strength=0.8,
        confidence=0.8,
        risk_level=RiskLevel.MEDIUM,
        position_size=0.05,
        entry_price=50000,
        stop_loss=48000,  # Invalid: below entry for sell
        take_profit=45000,
        holding_period="3-7 days",
        source_events=["test"],
        reasoning="Test"
    )
    
    validated_sell = risk_manager._validate_stop_loss(signal_sell)
    assert validated_sell.stop_loss > validated_sell.entry_price  # Should be corrected


def test_max_loss_per_trade():
    """Test maximum loss per trade calculation"""
    from src.trading.decision_engine.risk_manager import RiskManager
    
    risk_manager = RiskManager(max_loss_per_trade=0.02)  # 2% max loss
    
    signal = TradingSignal(
        id="test-signal",
        timestamp=datetime.now(),
        asset="BTC",
        asset_type=AssetType.CRYPTO,
        signal_type=SignalType.BUY,
        strategy=TradingStrategy.SWING,
        strength=0.8,
        confidence=0.8,
        risk_level=RiskLevel.MEDIUM,
        position_size=0.1,  # 10% position
        entry_price=50000,
        stop_loss=47500,    # 5% stop loss
        take_profit=55000,
        holding_period="3-7 days",
        source_events=["test"],
        reasoning="Test"
    )
    
    # Calculate actual loss if stop hit
    loss_pct = (signal.entry_price - signal.stop_loss) / signal.entry_price
    potential_loss = signal.position_size * loss_pct
    
    # Should adjust position size to limit loss
    adjusted = risk_manager.adjust_position_for_max_loss(signal)
    
    # Recalculate potential loss with adjusted position
    new_potential_loss = adjusted.position_size * loss_pct
    assert new_potential_loss <= risk_manager.max_loss_per_trade


def test_portfolio_position_tracking():
    """Test portfolio position tracking and conversion"""
    from src.trading.decision_engine.risk_manager import RiskManager
    
    risk_manager = RiskManager()
    
    # Add position from signal
    signal = TradingSignal(
        id="test-signal",
        timestamp=datetime.now(),
        asset="ETH",
        asset_type=AssetType.CRYPTO,
        signal_type=SignalType.BUY,
        strategy=TradingStrategy.SWING,
        strength=0.8,
        confidence=0.8,
        risk_level=RiskLevel.MEDIUM,
        position_size=0.05,
        entry_price=3000,
        stop_loss=2850,
        take_profit=3300,
        holding_period="3-7 days",
        source_events=["test"],
        reasoning="Test"
    )
    
    position = risk_manager.signal_to_position(signal, current_price=3100)
    
    assert isinstance(position, PortfolioPosition)
    assert position.asset == "ETH"
    assert position.size == 0.05
    assert position.entry_price == 3000
    assert position.current_price == 3100
    assert position.unrealized_pnl == pytest.approx(0.0333, 0.01)  # 3.33% gain


@pytest.mark.asyncio
async def test_risk_level_limits():
    """Test risk level based position limits"""
    from src.trading.decision_engine.risk_manager import RiskManager
    
    risk_manager = RiskManager()
    
    # Configure risk level limits
    risk_manager.set_risk_level_limits({
        RiskLevel.LOW: 0.10,      # 10% max for low risk
        RiskLevel.MEDIUM: 0.07,   # 7% max for medium risk
        RiskLevel.HIGH: 0.05,     # 5% max for high risk
        RiskLevel.EXTREME: 0.02   # 2% max for extreme risk
    })
    
    # Test high risk signal
    signal = TradingSignal(
        id="high-risk",
        timestamp=datetime.now(),
        asset="DOGE",
        asset_type=AssetType.CRYPTO,
        signal_type=SignalType.BUY,
        strategy=TradingStrategy.MOMENTUM,
        strength=0.9,
        confidence=0.7,
        risk_level=RiskLevel.HIGH,
        position_size=0.08,  # 8% position, exceeds 5% limit for high risk
        entry_price=0.10,
        stop_loss=0.08,
        take_profit=0.15,
        holding_period="1-3 days",
        source_events=["test"],
        reasoning="Test"
    )
    
    validated = await risk_manager.validate_signal(signal)
    assert validated.position_size <= 0.05  # Should be capped at risk level limit


def test_diversification_rules():
    """Test diversification rules enforcement"""
    from src.trading.decision_engine.risk_manager import RiskManager
    
    risk_manager = RiskManager()
    
    # Set diversification rules
    risk_manager.set_diversification_rules({
        "max_per_asset_type": {
            AssetType.CRYPTO: 0.4,   # Max 40% in crypto
            AssetType.EQUITY: 0.6,   # Max 60% in equities
            AssetType.BOND: 0.3      # Max 30% in bonds
        },
        "min_positions": 3,          # Minimum 3 positions
        "max_positions": 20          # Maximum 20 positions
    })
    
    # Current positions heavily weighted in crypto
    current_positions = {
        "BTC": {"size": 0.2, "risk": 0.05, "asset_type": AssetType.CRYPTO},
        "ETH": {"size": 0.15, "risk": 0.04, "asset_type": AssetType.CRYPTO},
        "AAPL": {"size": 0.1, "risk": 0.02, "asset_type": AssetType.EQUITY}
    }
    
    # Check if new crypto position is allowed (35% + 10% = 45% > 40% limit)
    can_add_crypto = risk_manager.check_diversification(
        AssetType.CRYPTO, 0.1, current_positions
    )
    assert can_add_crypto is False  # Should be rejected as it exceeds 40% limit
    
    # Smaller position should be allowed
    can_add_small_crypto = risk_manager.check_diversification(
        AssetType.CRYPTO, 0.04, current_positions  # 35% + 4% = 39% < 40%
    )
    assert can_add_small_crypto is True  # Should be allowed