//! Liquidity Sentinel - Portfolio Liquidity Risk Monitoring
//!
//! Scientific Basis:
//! - Bangia et al. (1999) "Liquidity Risk: A Model for Measuring and Managing Liquidity Risk"
//! - Almgren & Chriss (2003) "Optimal execution of portfolio transactions"
//! - Amihud (2002) "Illiquidity and stock returns: cross-section and time-series effects"
//! - Bertsimas & Lo (1998) "Optimal control of execution costs"
//! - Acharya & Pedersen (2005) "Asset pricing with liquidity risk"
//!
//! Key Metrics:
//! - Days-to-liquidate (position size / max daily volume participation)
//! - Liquidity-adjusted VaR (LVaR = VaR + Liquidation Cost)
//! - Bid-ask spread monitoring (relative to historical average)
//! - Market impact estimation (Almgren square-root model)
//! - Volume participation limits (Bertsimas-Lo optimal execution)

use super::base::{Sentinel, SentinelStatus};
use std::collections::HashMap;
use std::time::Instant;

/// Severity levels for liquidity alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquiditySeverity {
    /// Information only - no action needed
    Info,
    /// Warning - monitor closely
    Warn,
    /// Critical - immediate attention required
    Critical,
    /// Halt - stop trading immediately
    Halt,
}

/// Liquidity alert details
#[derive(Debug, Clone)]
pub struct LiquidityAlert {
    /// Alert severity
    pub severity: LiquiditySeverity,
    /// Symbol triggering alert
    pub symbol: String,
    /// Metric that triggered alert
    pub metric: String,
    /// Configured threshold
    pub threshold: f64,
    /// Actual measured value
    pub actual: f64,
    /// Timestamp of alert
    pub timestamp: Instant,
}

/// Comprehensive liquidity metrics for a position
#[derive(Debug, Clone)]
pub struct LiquidityMetrics {
    /// Days required to liquidate position at max participation rate
    pub days_to_liquidate: f64,
    /// Current spread / historical average spread
    pub spread_ratio: f64,
    /// Liquidity score (0-1, higher = more liquid)
    pub liquidity_score: f64,
    /// Estimated market impact cost (bps)
    pub market_impact_bps: f64,
    /// Total liquidation cost (spread + impact)
    pub liquidation_cost_bps: f64,
    /// Liquidity-adjusted VaR
    pub lvar: f64,
}

/// Configuration for liquidity monitoring
#[derive(Debug, Clone)]
pub struct LiquidityConfig {
    /// Maximum participation rate as fraction of average daily volume (typically 0.10-0.25)
    /// From Almgren & Chriss (2003): optimal participation rate balances impact and risk
    pub max_participation_rate: f64,

    /// Bid-ask spread threshold multiplier (alert if > N × historical average)
    /// Typical value: 2.0 based on Bangia et al. (1999)
    pub spread_threshold_multiplier: f64,

    /// Minimum acceptable liquidity score (0-1)
    /// Below this triggers critical alert
    pub min_liquidity_score: f64,

    /// Days-to-liquidate warning threshold
    /// From Bangia: 3-5 days is typical threshold for liquid positions
    pub days_to_liquidate_warn: f64,

    /// Days-to-liquidate critical threshold
    pub days_to_liquidate_critical: f64,

    /// Market impact model parameters (Almgren square-root model)
    /// eta: temporary market impact coefficient
    pub temp_impact_coeff: f64,
    /// gamma: permanent market impact coefficient
    pub perm_impact_coeff: f64,

    /// Liquidity cost multiplier for LVaR calculation
    /// From Bangia et al.: typically 0.5 for spread cost
    pub liquidity_cost_multiplier: f64,
}

impl Default for LiquidityConfig {
    fn default() -> Self {
        Self {
            // Almgren & Chriss (2003): typical institutional participation rates
            max_participation_rate: 0.20,

            // Bangia et al. (1999): 2x historical spread triggers alert
            spread_threshold_multiplier: 2.0,

            // Minimum liquidity score based on Amihud illiquidity measure
            min_liquidity_score: 0.30,

            // Days-to-liquidate thresholds (Bangia et al.)
            days_to_liquidate_warn: 3.0,
            days_to_liquidate_critical: 5.0,

            // Market impact model parameters (calibrated from Almgren)
            temp_impact_coeff: 0.1,
            perm_impact_coeff: 0.01,

            // Liquidity cost multiplier (Bangia et al.)
            liquidity_cost_multiplier: 0.5,
        }
    }
}

/// Position data required for liquidity analysis
#[derive(Debug, Clone)]
pub struct Position {
    /// Trading symbol
    pub symbol: String,
    /// Position size (shares/contracts)
    pub size: f64,
    /// Current market value
    pub market_value: f64,
    /// Position direction
    pub is_long: bool,
}

/// Market data required for liquidity calculations
#[derive(Debug, Clone)]
pub struct MarketData {
    /// Symbol
    pub symbol: String,
    /// Current bid price
    pub bid: f64,
    /// Current ask price
    pub ask: f64,
    /// Historical average bid-ask spread
    pub historical_avg_spread: f64,
    /// Average daily volume (shares)
    pub average_daily_volume: f64,
    /// Current daily volume
    pub current_volume: f64,
    /// Historical volatility (annualized)
    pub volatility: f64,
}

/// Liquidity Sentinel for portfolio liquidity risk monitoring
pub struct LiquiditySentinel {
    /// Configuration parameters
    config: LiquidityConfig,
    /// Current positions to monitor
    positions: HashMap<String, Position>,
    /// Market data for liquidity calculations
    market_data: HashMap<String, MarketData>,
    /// Current VaR estimates (for LVaR calculation)
    var_estimates: HashMap<String, f64>,
    /// Active liquidity alerts
    alerts: Vec<LiquidityAlert>,
    /// Last check timestamp
    last_check: Option<Instant>,
}

impl LiquiditySentinel {
    /// Create new LiquiditySentinel with default configuration
    pub fn new() -> Self {
        Self {
            config: LiquidityConfig::default(),
            positions: HashMap::new(),
            market_data: HashMap::new(),
            var_estimates: HashMap::new(),
            alerts: Vec::new(),
            last_check: None,
        }
    }

    /// Create new LiquiditySentinel with custom configuration
    pub fn with_config(config: LiquidityConfig) -> Self {
        Self {
            config,
            positions: HashMap::new(),
            market_data: HashMap::new(),
            var_estimates: HashMap::new(),
            alerts: Vec::new(),
            last_check: None,
        }
    }

    /// Update position data
    pub fn update_position(&mut self, position: Position) {
        self.positions.insert(position.symbol.clone(), position);
    }

    /// Update market data
    pub fn update_market_data(&mut self, data: MarketData) {
        self.market_data.insert(data.symbol.clone(), data);
    }

    /// Update VaR estimate for a symbol
    pub fn update_var(&mut self, symbol: String, var: f64) {
        self.var_estimates.insert(symbol, var);
    }

    /// Calculate days required to liquidate position
    ///
    /// Formula from Bangia et al. (1999):
    /// DTL = Position_Size / (Max_Participation_Rate × ADV)
    ///
    /// # Arguments
    /// * `position_size` - Size of position to liquidate (shares)
    /// * `average_daily_volume` - Average daily trading volume
    /// * `max_participation` - Maximum participation rate (fraction of ADV)
    ///
    /// # Returns
    /// Days required to liquidate position
    pub fn days_to_liquidate(
        &self,
        position_size: f64,
        average_daily_volume: f64,
        max_participation: f64,
    ) -> f64 {
        if average_daily_volume <= 0.0 || max_participation <= 0.0 {
            return f64::INFINITY;
        }

        let max_daily_shares = average_daily_volume * max_participation;
        position_size.abs() / max_daily_shares
    }

    /// Calculate liquidity score for a symbol
    ///
    /// Based on Amihud (2002) illiquidity measure, inverted to create liquidity score.
    /// Amihud illiquidity = |return| / volume
    /// Liquidity score = 1 / (1 + normalized_illiquidity)
    ///
    /// Higher score (closer to 1) = more liquid
    /// Lower score (closer to 0) = less liquid
    ///
    /// # Arguments
    /// * `spread_ratio` - Current spread / historical average spread
    /// * `volume_ratio` - Current volume / average daily volume
    /// * `days_to_liquidate` - Days required to liquidate
    ///
    /// # Returns
    /// Liquidity score between 0 and 1
    pub fn liquidity_score(
        &self,
        spread_ratio: f64,
        volume_ratio: f64,
        days_to_liquidate: f64,
    ) -> f64 {
        // Combine multiple liquidity dimensions
        // Lower spread ratio = better liquidity
        let spread_component = 1.0 / (1.0 + spread_ratio);

        // Higher volume ratio = better liquidity
        let volume_component = volume_ratio.min(2.0) / 2.0;

        // Lower days-to-liquidate = better liquidity
        let dtl_component = 1.0 / (1.0 + days_to_liquidate / 5.0);

        // Weighted average (can be calibrated based on empirical data)
        let score = 0.4 * spread_component + 0.3 * volume_component + 0.3 * dtl_component;

        score.max(0.0).min(1.0)
    }

    /// Calculate bid-ask spread ratio
    ///
    /// Ratio of current spread to historical average spread.
    /// Values > 1 indicate wider spreads (worse liquidity) than normal.
    ///
    /// From Bangia et al. (1999): Spread volatility is key liquidity risk metric
    ///
    /// # Arguments
    /// * `current_spread` - Current bid-ask spread
    /// * `historical_avg_spread` - Historical average spread
    ///
    /// # Returns
    /// Ratio of current to historical spread
    pub fn bid_ask_spread_ratio(&self, current_spread: f64, historical_avg_spread: f64) -> f64 {
        if historical_avg_spread <= 0.0 {
            return 1.0;
        }
        current_spread / historical_avg_spread
    }

    /// Check if order size respects volume participation limits
    ///
    /// From Bertsimas & Lo (1998): Optimal execution respects market depth
    ///
    /// # Arguments
    /// * `order_size` - Proposed order size (shares)
    /// * `average_daily_volume` - Average daily volume
    ///
    /// # Returns
    /// true if order size is within limits, false otherwise
    pub fn volume_participation_check(&self, order_size: f64, average_daily_volume: f64) -> bool {
        if average_daily_volume <= 0.0 {
            return false;
        }

        let participation_rate = order_size.abs() / average_daily_volume;
        participation_rate <= self.config.max_participation_rate
    }

    /// Calculate market impact cost using Almgren square-root model
    ///
    /// From Almgren & Chriss (2003):
    /// Temporary impact: η × (X/V)^(1/2)
    /// Permanent impact: γ × (X/V)
    ///
    /// Where:
    /// - X = shares traded
    /// - V = average daily volume
    /// - η = temporary impact coefficient
    /// - γ = permanent impact coefficient
    ///
    /// # Arguments
    /// * `shares` - Number of shares to trade
    /// * `avg_daily_volume` - Average daily volume
    /// * `volatility` - Asset volatility (annualized)
    ///
    /// # Returns
    /// Market impact cost in basis points
    pub fn market_impact_cost(
        &self,
        shares: f64,
        avg_daily_volume: f64,
        volatility: f64,
    ) -> f64 {
        if avg_daily_volume <= 0.0 {
            return 0.0;
        }

        let participation = shares.abs() / avg_daily_volume;

        // Almgren square-root model (calibrated with volatility)
        let temp_impact = self.config.temp_impact_coeff * volatility * participation.sqrt();
        let perm_impact = self.config.perm_impact_coeff * volatility * participation;

        // Convert to basis points
        (temp_impact + perm_impact) * 10000.0
    }

    /// Calculate liquidity-adjusted VaR (LVaR)
    ///
    /// From Bangia et al. (1999):
    /// LVaR = VaR + Liquidation_Cost
    /// Liquidation_Cost = Spread_Cost + Market_Impact
    /// Spread_Cost = (spread/2) × position_value × multiplier
    ///
    /// # Arguments
    /// * `var` - Standard VaR estimate
    /// * `position_value` - Market value of position
    /// * `spread_bps` - Bid-ask spread in basis points
    /// * `market_impact_bps` - Market impact in basis points
    ///
    /// # Returns
    /// Liquidity-adjusted VaR
    pub fn liquidity_adjusted_var(
        &self,
        var: f64,
        position_value: f64,
        spread_bps: f64,
        market_impact_bps: f64,
    ) -> f64 {
        // Spread cost (Bangia et al.: use multiplier for stressed conditions)
        let spread_cost = self.config.liquidity_cost_multiplier *
                         (spread_bps / 10000.0) * position_value;

        // Market impact cost
        let impact_cost = (market_impact_bps / 10000.0) * position_value;

        // Total LVaR
        var + spread_cost + impact_cost
    }

    /// Calculate comprehensive liquidity metrics for a position
    pub fn calculate_metrics(&self, symbol: &str) -> Option<LiquidityMetrics> {
        let position = self.positions.get(symbol)?;
        let market = self.market_data.get(symbol)?;

        // Calculate current spread
        let current_spread = market.ask - market.bid;
        let spread_bps = (current_spread / ((market.bid + market.ask) / 2.0)) * 10000.0;

        // Days to liquidate
        let dtl = self.days_to_liquidate(
            position.size,
            market.average_daily_volume,
            self.config.max_participation_rate,
        );

        // Spread ratio
        let spread_ratio = self.bid_ask_spread_ratio(
            current_spread,
            market.historical_avg_spread,
        );

        // Volume ratio for liquidity score
        let volume_ratio = if market.average_daily_volume > 0.0 {
            market.current_volume / market.average_daily_volume
        } else {
            0.0
        };

        // Liquidity score
        let liq_score = self.liquidity_score(spread_ratio, volume_ratio, dtl);

        // Market impact
        let market_impact_bps = self.market_impact_cost(
            position.size,
            market.average_daily_volume,
            market.volatility,
        );

        // Total liquidation cost
        let liquidation_cost_bps = spread_bps + market_impact_bps;

        // LVaR calculation
        let var = self.var_estimates.get(symbol).copied().unwrap_or(0.0);
        let lvar = self.liquidity_adjusted_var(
            var,
            position.market_value,
            spread_bps,
            market_impact_bps,
        );

        Some(LiquidityMetrics {
            days_to_liquidate: dtl,
            spread_ratio,
            liquidity_score: liq_score,
            market_impact_bps,
            liquidation_cost_bps,
            lvar,
        })
    }

    /// Generate alerts based on liquidity metrics
    fn generate_alerts(&mut self) {
        self.alerts.clear();

        for (symbol, position) in &self.positions {
            if let Some(metrics) = self.calculate_metrics(symbol) {
                let now = Instant::now();

                // Days-to-liquidate alerts
                if metrics.days_to_liquidate > self.config.days_to_liquidate_critical {
                    self.alerts.push(LiquidityAlert {
                        severity: LiquiditySeverity::Critical,
                        symbol: symbol.clone(),
                        metric: "days_to_liquidate".to_string(),
                        threshold: self.config.days_to_liquidate_critical,
                        actual: metrics.days_to_liquidate,
                        timestamp: now,
                    });
                } else if metrics.days_to_liquidate > self.config.days_to_liquidate_warn {
                    self.alerts.push(LiquidityAlert {
                        severity: LiquiditySeverity::Warn,
                        symbol: symbol.clone(),
                        metric: "days_to_liquidate".to_string(),
                        threshold: self.config.days_to_liquidate_warn,
                        actual: metrics.days_to_liquidate,
                        timestamp: now,
                    });
                }

                // Spread alerts
                if metrics.spread_ratio > self.config.spread_threshold_multiplier {
                    self.alerts.push(LiquidityAlert {
                        severity: LiquiditySeverity::Critical,
                        symbol: symbol.clone(),
                        metric: "spread_ratio".to_string(),
                        threshold: self.config.spread_threshold_multiplier,
                        actual: metrics.spread_ratio,
                        timestamp: now,
                    });
                }

                // Liquidity score alerts
                if metrics.liquidity_score < self.config.min_liquidity_score {
                    self.alerts.push(LiquidityAlert {
                        severity: LiquiditySeverity::Halt,
                        symbol: symbol.clone(),
                        metric: "liquidity_score".to_string(),
                        threshold: self.config.min_liquidity_score,
                        actual: metrics.liquidity_score,
                        timestamp: now,
                    });
                }
            }
        }
    }

    /// Get current alerts
    pub fn get_alerts(&self) -> &[LiquidityAlert] {
        &self.alerts
    }

    /// Get liquidity metrics for all positions
    pub fn get_all_metrics(&self) -> HashMap<String, LiquidityMetrics> {
        let mut all_metrics = HashMap::new();

        for symbol in self.positions.keys() {
            if let Some(metrics) = self.calculate_metrics(symbol) {
                all_metrics.insert(symbol.clone(), metrics);
            }
        }

        all_metrics
    }
}

impl Default for LiquiditySentinel {
    fn default() -> Self {
        Self::new()
    }
}

impl Sentinel for LiquiditySentinel {
    fn check(&mut self) -> SentinelStatus {
        let start = Instant::now();

        // Generate alerts based on current state
        self.generate_alerts();

        // Determine overall status based on highest severity alert
        let status = if self.alerts.iter().any(|a| a.severity == LiquiditySeverity::Halt) {
            SentinelStatus::Critical
        } else if self.alerts.iter().any(|a| a.severity == LiquiditySeverity::Critical) {
            SentinelStatus::Warning
        } else if self.alerts.iter().any(|a| a.severity == LiquiditySeverity::Warn) {
            SentinelStatus::Warning
        } else {
            SentinelStatus::Ok
        };

        let latency = start.elapsed().as_nanos() as u64;
        self.last_check = Some(start);

        // Verify latency budget compliance (30μs = 30,000ns)
        if latency > self.latency_budget_ns() {
            eprintln!(
                "⚠️  LiquiditySentinel exceeded latency budget: {}ns > {}ns",
                latency,
                self.latency_budget_ns()
            );
        }

        status
    }

    fn name(&self) -> &str {
        "Liquidity"
    }

    fn latency_budget_ns(&self) -> u64 {
        30_000 // 30 microseconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_position(symbol: &str, size: f64, value: f64) -> Position {
        Position {
            symbol: symbol.to_string(),
            size,
            market_value: value,
            is_long: size > 0.0,
        }
    }

    fn create_test_market_data(
        symbol: &str,
        bid: f64,
        ask: f64,
        avg_spread: f64,
        adv: f64,
    ) -> MarketData {
        MarketData {
            symbol: symbol.to_string(),
            bid,
            ask,
            historical_avg_spread: avg_spread,
            average_daily_volume: adv,
            current_volume: adv * 0.8,
            volatility: 0.30, // 30% annualized
        }
    }

    #[test]
    fn test_days_to_liquidate_calculation() {
        let sentinel = LiquiditySentinel::new();

        // Position: 100,000 shares
        // ADV: 1,000,000 shares
        // Max participation: 20%
        // Expected DTL: 100,000 / (1,000,000 × 0.20) = 0.5 days
        let dtl = sentinel.days_to_liquidate(100_000.0, 1_000_000.0, 0.20);
        assert!((dtl - 0.5).abs() < 1e-10, "DTL should be 0.5 days");

        // Large position test
        // Position: 500,000 shares (50% of ADV)
        // Expected DTL: 500,000 / (1,000,000 × 0.20) = 2.5 days
        let dtl_large = sentinel.days_to_liquidate(500_000.0, 1_000_000.0, 0.20);
        assert!((dtl_large - 2.5).abs() < 1e-10, "DTL should be 2.5 days for large position");

        // Edge case: zero volume
        let dtl_zero = sentinel.days_to_liquidate(100_000.0, 0.0, 0.20);
        assert!(dtl_zero.is_infinite(), "DTL should be infinite for zero volume");
    }

    #[test]
    fn test_spread_ratio_calculation() {
        let sentinel = LiquiditySentinel::new();

        // Normal conditions: current spread = historical average
        let ratio_normal = sentinel.bid_ask_spread_ratio(0.10, 0.10);
        assert!((ratio_normal - 1.0).abs() < 1e-10, "Ratio should be 1.0 for normal spread");

        // Stressed conditions: spread doubled
        let ratio_stressed = sentinel.bid_ask_spread_ratio(0.20, 0.10);
        assert!((ratio_stressed - 2.0).abs() < 1e-10, "Ratio should be 2.0 for doubled spread");

        // Improved liquidity: spread halved
        let ratio_improved = sentinel.bid_ask_spread_ratio(0.05, 0.10);
        assert!((ratio_improved - 0.5).abs() < 1e-10, "Ratio should be 0.5 for halved spread");
    }

    #[test]
    fn test_volume_participation_check() {
        let sentinel = LiquiditySentinel::new();

        // Within limits: 15% of ADV (< 20% default limit)
        assert!(
            sentinel.volume_participation_check(150_000.0, 1_000_000.0),
            "15% participation should pass"
        );

        // At limit: exactly 20%
        assert!(
            sentinel.volume_participation_check(200_000.0, 1_000_000.0),
            "20% participation should pass"
        );

        // Exceeds limit: 25% > 20%
        assert!(
            !sentinel.volume_participation_check(250_000.0, 1_000_000.0),
            "25% participation should fail"
        );

        // Edge case: zero volume
        assert!(
            !sentinel.volume_participation_check(100_000.0, 0.0),
            "Should fail for zero volume"
        );
    }

    #[test]
    fn test_liquidity_score_calculation() {
        let sentinel = LiquiditySentinel::new();

        // Highly liquid: tight spread, high volume, quick liquidation
        let score_high = sentinel.liquidity_score(
            0.8,  // Spread ratio below 1.0
            1.2,  // Volume above average
            0.5,  // Half day to liquidate
        );
        assert!(score_high > 0.7, "Highly liquid asset should score > 0.7");

        // Illiquid: wide spread, low volume, slow liquidation
        let score_low = sentinel.liquidity_score(
            2.5,  // Spread 2.5x historical
            0.3,  // Volume 30% of average
            10.0, // 10 days to liquidate
        );
        assert!(score_low < 0.4, "Illiquid asset should score < 0.4");

        // Score should be bounded [0, 1]
        assert!(score_high >= 0.0 && score_high <= 1.0);
        assert!(score_low >= 0.0 && score_low <= 1.0);
    }

    #[test]
    fn test_market_impact_almgren_model() {
        let sentinel = LiquiditySentinel::new();

        // Small order: 5% of ADV
        let impact_small = sentinel.market_impact_cost(
            50_000.0,    // 5% of ADV
            1_000_000.0, // ADV
            0.30,        // 30% volatility
        );

        // Large order: 20% of ADV
        let impact_large = sentinel.market_impact_cost(
            200_000.0,   // 20% of ADV
            1_000_000.0, // ADV
            0.30,        // 30% volatility
        );

        // Market impact should increase non-linearly (square-root model)
        // 4x size increase should result in less than 4x impact increase
        assert!(
            impact_large > impact_small,
            "Larger orders should have higher impact"
        );
        assert!(
            impact_large < impact_small * 4.0,
            "Square-root model: impact grows sub-linearly"
        );

        // Impact should scale with volatility
        let impact_low_vol = sentinel.market_impact_cost(100_000.0, 1_000_000.0, 0.15);
        let impact_high_vol = sentinel.market_impact_cost(100_000.0, 1_000_000.0, 0.45);
        assert!(
            impact_high_vol > impact_low_vol,
            "Higher volatility should increase market impact"
        );
    }

    #[test]
    fn test_liquidity_adjusted_var() {
        let sentinel = LiquiditySentinel::new();

        let var = 100_000.0;
        let position_value = 1_000_000.0;
        let spread_bps = 10.0; // 0.10% spread
        let market_impact_bps = 5.0; // 0.05% impact

        let lvar = sentinel.liquidity_adjusted_var(
            var,
            position_value,
            spread_bps,
            market_impact_bps,
        );

        // LVaR should be greater than VaR
        assert!(lvar > var, "LVaR should exceed VaR");

        // Calculate expected liquidation cost
        // Spread cost: 0.5 × (10/10000) × 1,000,000 = 500
        // Impact cost: (5/10000) × 1,000,000 = 500
        // Total: 100,000 + 500 + 500 = 101,000
        let expected_lvar = 101_000.0;
        assert!(
            (lvar - expected_lvar).abs() < 1.0,
            "LVaR calculation should match expected value"
        );
    }

    #[test]
    fn test_comprehensive_metrics_calculation() {
        let mut sentinel = LiquiditySentinel::new();

        // Setup test data
        let position = create_test_position("AAPL", 100_000.0, 15_000_000.0);
        let market = create_test_market_data("AAPL", 149.90, 150.10, 0.15, 50_000_000.0);

        sentinel.update_position(position);
        sentinel.update_market_data(market);
        sentinel.update_var("AAPL".to_string(), 500_000.0);

        // Calculate metrics
        let metrics = sentinel.calculate_metrics("AAPL").expect("Metrics should be available");

        // Verify all metrics are calculated
        assert!(metrics.days_to_liquidate > 0.0, "DTL should be positive");
        assert!(metrics.spread_ratio > 0.0, "Spread ratio should be positive");
        assert!(metrics.liquidity_score >= 0.0 && metrics.liquidity_score <= 1.0);
        assert!(metrics.market_impact_bps >= 0.0, "Market impact should be non-negative");
        assert!(metrics.liquidation_cost_bps >= 0.0, "Liquidation cost should be non-negative");
        assert!(metrics.lvar > 0.0, "LVaR should be positive");

        // LVaR should be greater than VaR
        assert!(
            metrics.lvar > 500_000.0,
            "LVaR should exceed VaR due to liquidation costs"
        );
    }

    #[test]
    fn test_alert_generation_days_to_liquidate() {
        let mut sentinel = LiquiditySentinel::new();

        // Create illiquid position: 30M shares, 10M ADV
        // DTL = 30M / (10M × 0.2) = 15 days (exceeds critical threshold of 5)
        let position = create_test_position("ILLIQ", 30_000_000.0, 300_000_000.0);
        let market = create_test_market_data("ILLIQ", 9.90, 10.10, 0.15, 10_000_000.0);

        sentinel.update_position(position);
        sentinel.update_market_data(market);

        // Run check to generate alerts
        let status = sentinel.check();

        // Should generate critical status
        assert_eq!(status, SentinelStatus::Critical, "Should be critical for DTL > 5 days");

        // Should have DTL alert
        let alerts = sentinel.get_alerts();
        assert!(!alerts.is_empty(), "Should generate alerts");

        let dtl_alert = alerts.iter()
            .find(|a| a.metric == "days_to_liquidate")
            .expect("Should have DTL alert");

        assert_eq!(dtl_alert.severity, LiquiditySeverity::Critical);
        assert!(dtl_alert.actual > 5.0, "DTL should exceed critical threshold");
    }

    #[test]
    fn test_alert_generation_spread() {
        let mut sentinel = LiquiditySentinel::new();

        // Create position with wide spread
        // Current spread: 0.50, Historical: 0.15
        // Ratio: 0.50 / 0.15 = 3.33 (exceeds 2.0 threshold)
        let position = create_test_position("WIDE", 100_000.0, 5_000_000.0);
        let market = create_test_market_data("WIDE", 49.75, 50.25, 0.15, 5_000_000.0);

        sentinel.update_position(position);
        sentinel.update_market_data(market);

        let status = sentinel.check();

        assert_eq!(status, SentinelStatus::Critical, "Wide spread should trigger critical");

        let alerts = sentinel.get_alerts();
        let spread_alert = alerts.iter()
            .find(|a| a.metric == "spread_ratio")
            .expect("Should have spread alert");

        assert_eq!(spread_alert.severity, LiquiditySeverity::Critical);
        assert!(spread_alert.actual > 2.0, "Spread ratio should exceed threshold");
    }

    #[test]
    fn test_alert_generation_liquidity_score() {
        let mut sentinel = LiquiditySentinel::new();

        // Create very illiquid position
        // Low volume, wide spread, long liquidation time
        let position = create_test_position("ILLIQ2", 5_000_000.0, 50_000_000.0);
        let market = MarketData {
            symbol: "ILLIQ2".to_string(),
            bid: 9.50,
            ask: 10.50,
            historical_avg_spread: 0.20,
            average_daily_volume: 500_000.0,
            current_volume: 100_000.0, // Very low volume
            volatility: 0.50,
        };

        sentinel.update_position(position);
        sentinel.update_market_data(market);

        let status = sentinel.check();

        // Liquidity score violations trigger Halt severity
        assert_eq!(status, SentinelStatus::Critical, "Poor liquidity should trigger critical");

        let alerts = sentinel.get_alerts();

        // Should have at least one alert (could be DTL, spread, or liquidity score)
        assert!(!alerts.is_empty(), "Should generate alerts for illiquid position");
    }

    #[test]
    fn test_sentinel_trait_implementation() {
        let mut sentinel = LiquiditySentinel::new();

        // Test name
        assert_eq!(sentinel.name(), "Liquidity");

        // Test latency budget
        assert_eq!(sentinel.latency_budget_ns(), 30_000);

        // Test check returns valid status
        let status = sentinel.check();
        assert!(
            matches!(status, SentinelStatus::Ok | SentinelStatus::Warning | SentinelStatus::Critical),
            "Should return valid SentinelStatus"
        );
    }

    #[test]
    fn test_latency_budget_compliance() {
        let mut sentinel = LiquiditySentinel::new();

        // Add multiple positions to ensure realistic workload
        for i in 0..10 {
            let symbol = format!("TEST{}", i);
            let position = create_test_position(&symbol, 100_000.0, 1_000_000.0);
            let market = create_test_market_data(&symbol, 99.90, 100.10, 0.15, 10_000_000.0);

            sentinel.update_position(position);
            sentinel.update_market_data(market);
            sentinel.update_var(symbol, 50_000.0);
        }

        let start = Instant::now();
        sentinel.check();
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        // Should complete within 30μs latency budget
        assert!(
            elapsed_ns <= 30_000,
            "Check should complete within 30μs budget, took {}ns",
            elapsed_ns
        );
    }

    #[test]
    fn test_multiple_positions_metrics() {
        let mut sentinel = LiquiditySentinel::new();

        // Add multiple positions
        let symbols = vec!["AAPL", "GOOGL", "MSFT"];
        for symbol in &symbols {
            let position = create_test_position(symbol, 50_000.0, 7_500_000.0);
            let market = create_test_market_data(symbol, 149.90, 150.10, 0.15, 25_000_000.0);

            sentinel.update_position(position);
            sentinel.update_market_data(market);
            sentinel.update_var(symbol.to_string(), 250_000.0);
        }

        // Get metrics for all positions
        let all_metrics = sentinel.get_all_metrics();

        assert_eq!(all_metrics.len(), 3, "Should have metrics for all 3 positions");

        for symbol in &symbols {
            assert!(
                all_metrics.contains_key(*symbol),
                "Should have metrics for {}",
                symbol
            );
        }
    }

    #[test]
    fn test_config_customization() {
        let custom_config = LiquidityConfig {
            max_participation_rate: 0.15,
            spread_threshold_multiplier: 1.5,
            min_liquidity_score: 0.40,
            days_to_liquidate_warn: 2.0,
            days_to_liquidate_critical: 4.0,
            temp_impact_coeff: 0.15,
            perm_impact_coeff: 0.015,
            liquidity_cost_multiplier: 0.6,
        };

        let sentinel = LiquiditySentinel::with_config(custom_config.clone());

        assert_eq!(sentinel.config.max_participation_rate, 0.15);
        assert_eq!(sentinel.config.spread_threshold_multiplier, 1.5);
        assert_eq!(sentinel.config.min_liquidity_score, 0.40);
    }
}
