//! # Test Data Generation for Integration Tests
//!
//! Provides realistic market data scenarios for testing the aggressive
//! Talebian risk management system.

use talebian_risk_rs::MarketData;

/// Create standard market data for baseline testing
pub fn create_standard_market_data() -> MarketData {
    MarketData {
        timestamp: 1640995200, // 2022-01-01
        price: 50000.0,
        volume: 1000.0,
        bid: 49995.0,
        ask: 50005.0,
        bid_volume: 500.0,
        ask_volume: 500.0,
        volatility: 0.02,
        returns: vec![0.01, -0.005, 0.015, -0.008, 0.02],
        volume_history: vec![900.0, 1100.0, 950.0, 1050.0, 1000.0],
    }
}

/// Create bull market scenario with strong upward momentum
pub fn create_bull_market_data() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 55000.0,
        volume: 1200.0,
        bid: 54990.0,
        ask: 55010.0,
        bid_volume: 700.0, // Strong bid pressure
        ask_volume: 300.0,
        volatility: 0.025,
        returns: vec![0.02, 0.015, 0.03, 0.01, 0.025, 0.018, 0.022],
        volume_history: vec![1000.0, 1100.0, 1300.0, 1150.0, 1200.0],
    }
}

/// Create bear market scenario with downward pressure
pub fn create_bear_market_data() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 45000.0,
        volume: 1500.0, // High volume in sell-off
        bid: 44980.0,
        ask: 45020.0,
        bid_volume: 300.0, // Weak bid support
        ask_volume: 900.0, // Strong sell pressure
        volatility: 0.035,
        returns: vec![-0.02, -0.015, -0.025, -0.01, -0.03, -0.012, -0.018],
        volume_history: vec![1200.0, 1400.0, 1600.0, 1350.0, 1500.0],
    }
}

/// Create whale activity scenario with large volume anomalies
pub fn create_whale_activity_data() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 52000.0,
        volume: 5000.0, // 5x normal volume
        bid: 51950.0,
        ask: 52050.0,
        bid_volume: 2000.0, // Large whale bid
        ask_volume: 800.0,
        volatility: 0.04,
        returns: vec![0.005, 0.03, 0.008, 0.025, 0.012], // Gradual price movement despite large volume
        volume_history: vec![1000.0, 1100.0, 1050.0, 900.0, 1200.0], // Normal history
    }
}

/// Create high volatility scenario for antifragility testing
pub fn create_high_volatility_data() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 48000.0,
        volume: 2000.0,
        bid: 47900.0,
        ask: 48100.0,
        bid_volume: 600.0,
        ask_volume: 600.0,
        volatility: 0.08, // Very high volatility
        returns: vec![0.05, -0.04, 0.06, -0.03, 0.04, -0.05, 0.03],
        volume_history: vec![1800.0, 2200.0, 1900.0, 2100.0, 2000.0],
    }
}

/// Create high opportunity scenario with multiple positive factors
pub fn create_high_opportunity_data() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 53000.0,
        volume: 3000.0, // High volume
        bid: 52980.0,
        ask: 53020.0,
        bid_volume: 1200.0, // Strong bid pressure
        ask_volume: 400.0,
        volatility: 0.045, // High but manageable volatility
        returns: vec![0.025, 0.03, 0.02, 0.035, 0.015, 0.028], // Strong momentum
        volume_history: vec![1500.0, 2000.0, 2500.0, 2200.0, 3000.0],
    }
}

/// Create scenario with high expected returns
pub fn create_high_return_scenario() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 60000.0,
        volume: 1500.0,
        bid: 59990.0,
        ask: 60010.0,
        bid_volume: 800.0,
        ask_volume: 400.0,
        volatility: 0.03,
        returns: vec![0.04, 0.05, 0.06, 0.035, 0.045, 0.04, 0.055], // Very high returns
        volume_history: vec![1400.0, 1600.0, 1450.0, 1550.0, 1500.0],
    }
}

/// Create scenario with low expected returns
pub fn create_low_return_scenario() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 49500.0,
        volume: 800.0,
        bid: 49499.0,
        ask: 49501.0,
        bid_volume: 400.0,
        ask_volume: 400.0,
        volatility: 0.005, // Very low volatility
        returns: vec![0.001, 0.002, -0.001, 0.0005, 0.001, 0.002], // Minimal returns
        volume_history: vec![750.0, 850.0, 780.0, 820.0, 800.0],
    }
}

/// Create whale momentum scenario
pub fn create_whale_momentum_scenario() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 54000.0,
        volume: 4000.0, // Large whale volume
        bid: 53980.0,
        ask: 54020.0,
        bid_volume: 1500.0, // Whale accumulation
        ask_volume: 500.0,
        volatility: 0.035,
        returns: vec![0.01, 0.02, 0.025, 0.03, 0.015, 0.035], // Building momentum
        volume_history: vec![1000.0, 1200.0, 1800.0, 2500.0, 4000.0], // Volume crescendo
    }
}

/// Create beneficial black swan scenario (positive volatility shock)
pub fn create_beneficial_black_swan() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 65000.0, // Major price jump
        volume: 8000.0, // Extreme volume
        bid: 64900.0,
        ask: 65100.0,
        bid_volume: 3000.0,
        ask_volume: 1000.0,
        volatility: 0.12,                            // Extreme volatility
        returns: vec![0.15, 0.08, 0.12, 0.05, 0.10], // Major positive returns
        volume_history: vec![1000.0, 1200.0, 1100.0, 1300.0, 1150.0], // Normal history before shock
    }
}

/// Create destructive black swan scenario (negative volatility shock)
pub fn create_destructive_black_swan() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 35000.0,  // Major price crash
        volume: 10000.0, // Panic selling volume
        bid: 34800.0,
        ask: 35200.0,
        bid_volume: 1000.0,                                           // Weak bids
        ask_volume: 4000.0,                                           // Heavy selling
        volatility: 0.15,                                             // Extreme volatility
        returns: vec![-0.12, -0.08, -0.15, -0.06, -0.10],             // Major negative returns
        volume_history: vec![1000.0, 1200.0, 1100.0, 1300.0, 1150.0], // Normal history before shock
    }
}

/// Create extreme opportunity scenario for testing limits
pub fn create_extreme_opportunity_data() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 58000.0,
        volume: 6000.0, // Very high volume
        bid: 57950.0,
        ask: 58050.0,
        bid_volume: 2500.0, // Massive bid support
        ask_volume: 500.0,
        volatility: 0.06, // High volatility
        returns: vec![0.04, 0.05, 0.045, 0.06, 0.035, 0.055, 0.04], // Excellent returns
        volume_history: vec![2000.0, 3000.0, 4000.0, 5000.0, 6000.0], // Volume growth
    }
}

/// Create crypto crash pattern (flash crash scenario)
pub fn create_crypto_crash_pattern() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 42000.0,
        volume: 15000.0, // Extreme panic volume
        bid: 41500.0,    // Wide spread due to panic
        ask: 42500.0,
        bid_volume: 500.0,                                       // No bid support
        ask_volume: 5000.0,                                      // Heavy selling
        volatility: 0.18,                                        // Extreme volatility
        returns: vec![-0.08, -0.12, -0.15, -0.05, -0.20, -0.08], // Flash crash pattern
        volume_history: vec![1000.0, 1500.0, 3000.0, 8000.0, 15000.0], // Volume explosion
    }
}

/// Create crypto bull run pattern (parabolic rise)
pub fn create_crypto_bull_run_pattern() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 75000.0,
        volume: 5000.0, // High but sustainable volume
        bid: 74950.0,
        ask: 75050.0,
        bid_volume: 2000.0, // Strong institutional buying
        ask_volume: 600.0,
        volatility: 0.05,
        returns: vec![0.08, 0.06, 0.10, 0.04, 0.12, 0.07, 0.09], // Parabolic pattern
        volume_history: vec![2000.0, 3000.0, 3500.0, 4200.0, 5000.0], // Steady volume growth
    }
}

/// Create whale manipulation pattern (pump and dump setup)
pub fn create_whale_manipulation_pattern() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 51000.0,
        volume: 3500.0,
        bid: 50990.0,
        ask: 51010.0,
        bid_volume: 1000.0,
        ask_volume: 1000.0, // Balanced but large
        volatility: 0.03,
        returns: vec![0.002, 0.001, 0.025, 0.001, 0.002], // Sudden spike amid calm
        volume_history: vec![800.0, 900.0, 850.0, 3500.0, 900.0], // Volume spike pattern
    }
}

/// Create low liquidity scenario
pub fn create_low_liquidity_data() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 49000.0,
        volume: 200.0, // Very low volume
        bid: 48900.0,  // Wide spread
        ask: 49100.0,
        bid_volume: 50.0, // Thin order book
        ask_volume: 50.0,
        volatility: 0.04, // High volatility due to thin liquidity
        returns: vec![0.02, -0.03, 0.04, -0.02, 0.01], // Erratic moves
        volume_history: vec![180.0, 220.0, 190.0, 210.0, 200.0],
    }
}

/// Create consolidation pattern (sideways market)
pub fn create_consolidation_pattern() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 50000.0,
        volume: 1000.0,
        bid: 49998.0,
        ask: 50002.0,
        bid_volume: 500.0,
        ask_volume: 500.0,
        volatility: 0.008, // Low volatility
        returns: vec![0.001, -0.002, 0.0015, -0.001, 0.002, -0.0015], // Range-bound
        volume_history: vec![950.0, 1050.0, 980.0, 1020.0, 1000.0],
    }
}

/// Create breakout pattern (volume surge with direction)
pub fn create_breakout_pattern() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 52500.0,
        volume: 4000.0, // Volume surge
        bid: 52480.0,
        ask: 52520.0,
        bid_volume: 1500.0, // Strong buying
        ask_volume: 600.0,
        volatility: 0.035,
        returns: vec![0.001, 0.002, 0.001, 0.035, 0.015], // Breakout move
        volume_history: vec![800.0, 900.0, 850.0, 4000.0, 3500.0], // Volume confirmation
    }
}

/// Create institutional flow pattern (smart money accumulation)
pub fn create_institutional_flow_pattern() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 51500.0,
        volume: 2500.0,
        bid: 51495.0, // Tight spread (good liquidity)
        ask: 51505.0,
        bid_volume: 1200.0, // Steady accumulation
        ask_volume: 800.0,
        volatility: 0.02,
        returns: vec![0.005, 0.008, 0.004, 0.006, 0.007, 0.005], // Steady upward drift
        volume_history: vec![2000.0, 2200.0, 2400.0, 2300.0, 2500.0], // Consistent volume
    }
}

/// Create mean reversion setup (oversold bounce potential)
pub fn create_mean_reversion_setup() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 46000.0, // Oversold level
        volume: 1800.0,
        bid: 45990.0,
        ask: 46010.0,
        bid_volume: 800.0, // Building support
        ask_volume: 400.0,
        volatility: 0.03,
        returns: vec![-0.02, -0.015, -0.01, -0.005, 0.001], // Bottoming pattern
        volume_history: vec![1200.0, 1500.0, 1600.0, 1700.0, 1800.0], // Volume building
    }
}

/// Create trending market with pullback (buy the dip scenario)
pub fn create_trending_pullback_pattern() -> MarketData {
    MarketData {
        timestamp: 1640995200,
        price: 54000.0,
        volume: 1200.0,
        bid: 53990.0,
        ask: 54010.0,
        bid_volume: 700.0,
        ask_volume: 300.0,
        volatility: 0.025,
        returns: vec![0.02, 0.03, 0.025, -0.01, -0.005], // Strong trend with pullback
        volume_history: vec![1500.0, 1800.0, 1600.0, 1000.0, 1200.0], // Volume decline on pullback
    }
}
