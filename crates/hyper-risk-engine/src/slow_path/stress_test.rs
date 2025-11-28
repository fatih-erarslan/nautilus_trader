//! Stress testing framework.
//!
//! Evaluates portfolio performance under historical and hypothetical
//! stress scenarios.

use crate::core::types::Portfolio;

/// Stress scenario definition.
#[derive(Debug, Clone)]
pub struct StressScenario {
    /// Scenario name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Asset shocks (symbol -> percentage change).
    pub shocks: Vec<(String, f64)>,
    /// Correlation override (if any).
    pub correlation_shock: Option<f64>,
    /// Liquidity haircut.
    pub liquidity_haircut: f64,
}

impl StressScenario {
    /// Create 2008 Financial Crisis scenario.
    pub fn financial_crisis_2008() -> Self {
        Self {
            name: "2008 Financial Crisis".to_string(),
            description: "Lehman Brothers collapse, credit freeze".to_string(),
            shocks: vec![
                ("SPY".to_string(), -0.50),    // S&P 500 -50%
                ("QQQ".to_string(), -0.45),    // Nasdaq -45%
                ("IWM".to_string(), -0.55),    // Small caps -55%
                ("XLF".to_string(), -0.80),    // Financials -80%
                ("TLT".to_string(), 0.25),     // Treasuries +25%
                ("GLD".to_string(), 0.05),     // Gold +5%
                ("VIX".to_string(), 3.00),     // VIX +300%
            ],
            correlation_shock: Some(0.9), // Correlations spike
            liquidity_haircut: 0.20,
        }
    }

    /// Create COVID crash scenario.
    pub fn covid_crash_2020() -> Self {
        Self {
            name: "COVID-19 Crash".to_string(),
            description: "March 2020 pandemic market crash".to_string(),
            shocks: vec![
                ("SPY".to_string(), -0.34),
                ("QQQ".to_string(), -0.28),
                ("XLE".to_string(), -0.60),    // Energy -60%
                ("XLF".to_string(), -0.40),
                ("TLT".to_string(), 0.15),
                ("GLD".to_string(), -0.05),    // Brief gold selloff
            ],
            correlation_shock: Some(0.85),
            liquidity_haircut: 0.15,
        }
    }

    /// Create interest rate shock scenario.
    pub fn rate_shock_up() -> Self {
        Self {
            name: "Interest Rate Shock +300bp".to_string(),
            description: "Rapid Fed tightening".to_string(),
            shocks: vec![
                ("SPY".to_string(), -0.15),
                ("QQQ".to_string(), -0.25),    // Growth hit harder
                ("TLT".to_string(), -0.25),    // Bonds down
                ("XLF".to_string(), 0.10),     // Financials up
                ("XLU".to_string(), -0.20),    // Utilities down
                ("XLRE".to_string(), -0.30),   // Real estate down
            ],
            correlation_shock: None,
            liquidity_haircut: 0.05,
        }
    }

    /// Create custom scenario.
    pub fn custom(name: String, shocks: Vec<(String, f64)>) -> Self {
        Self {
            name,
            description: "Custom scenario".to_string(),
            shocks,
            correlation_shock: None,
            liquidity_haircut: 0.0,
        }
    }
}

/// Stress test result.
#[derive(Debug, Clone)]
pub struct StressResult {
    /// Scenario name.
    pub scenario_name: String,
    /// Portfolio P&L under scenario.
    pub pnl: f64,
    /// Portfolio value after scenario.
    pub final_value: f64,
    /// Percentage change.
    pub pct_change: f64,
    /// Individual position impacts.
    pub position_impacts: Vec<PositionImpact>,
    /// Worst hit position.
    pub worst_position: Option<String>,
    /// Best performing position.
    pub best_position: Option<String>,
}

/// Impact on individual position.
#[derive(Debug, Clone)]
pub struct PositionImpact {
    /// Symbol.
    pub symbol: String,
    /// Original value.
    pub original_value: f64,
    /// Shocked value.
    pub shocked_value: f64,
    /// P&L.
    pub pnl: f64,
    /// Percentage change.
    pub pct_change: f64,
}

/// Stress testing engine.
#[derive(Debug)]
pub struct StressTest {
    /// Available scenarios.
    scenarios: Vec<StressScenario>,
    /// Default shock for unmapped symbols.
    default_shock: f64,
}

impl StressTest {
    /// Create new stress test engine.
    pub fn new() -> Self {
        Self {
            scenarios: vec![
                StressScenario::financial_crisis_2008(),
                StressScenario::covid_crash_2020(),
                StressScenario::rate_shock_up(),
            ],
            default_shock: -0.20, // Default 20% down
        }
    }

    /// Add custom scenario.
    pub fn add_scenario(&mut self, scenario: StressScenario) {
        self.scenarios.push(scenario);
    }

    /// Run single scenario.
    pub fn run_scenario(&self, portfolio: &Portfolio, scenario: &StressScenario) -> StressResult {
        let mut total_pnl = 0.0;
        let mut position_impacts = Vec::new();
        let mut worst_pnl = f64::MAX;
        let mut worst_symbol = None;
        let mut best_pnl = f64::MIN;
        let mut best_symbol = None;

        // Build shock lookup
        let shock_map: std::collections::HashMap<&str, f64> = scenario
            .shocks
            .iter()
            .map(|(s, v)| (s.as_str(), *v))
            .collect();

        for pos in &portfolio.positions {
            let symbol = pos.symbol.as_str();
            let shock = shock_map.get(symbol).cloned().unwrap_or(self.default_shock);

            // Apply liquidity haircut
            let effective_shock = shock - scenario.liquidity_haircut;

            let original_value = pos.market_value();
            let pnl = original_value * effective_shock;
            let shocked_value = original_value + pnl;

            let impact = PositionImpact {
                symbol: symbol.to_string(),
                original_value,
                shocked_value,
                pnl,
                pct_change: effective_shock * 100.0,
            };

            if pnl < worst_pnl {
                worst_pnl = pnl;
                worst_symbol = Some(symbol.to_string());
            }
            if pnl > best_pnl {
                best_pnl = pnl;
                best_symbol = Some(symbol.to_string());
            }

            total_pnl += pnl;
            position_impacts.push(impact);
        }

        let final_value = portfolio.total_value + total_pnl;
        let pct_change = if portfolio.total_value > 0.0 {
            total_pnl / portfolio.total_value * 100.0
        } else {
            0.0
        };

        StressResult {
            scenario_name: scenario.name.clone(),
            pnl: total_pnl,
            final_value,
            pct_change,
            position_impacts,
            worst_position: worst_symbol,
            best_position: best_symbol,
        }
    }

    /// Run all scenarios.
    pub fn run_all(&self, portfolio: &Portfolio) -> Vec<StressResult> {
        self.scenarios
            .iter()
            .map(|s| self.run_scenario(portfolio, s))
            .collect()
    }

    /// Get worst case scenario result.
    pub fn worst_case(&self, portfolio: &Portfolio) -> Option<StressResult> {
        let results = self.run_all(portfolio);
        results.into_iter().min_by(|a, b| {
            a.pnl.partial_cmp(&b.pnl).unwrap()
        })
    }

    /// Get scenario by name.
    pub fn get_scenario(&self, name: &str) -> Option<&StressScenario> {
        self.scenarios.iter().find(|s| s.name == name)
    }
}

impl Default for StressTest {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{Symbol, Position, PositionId, Quantity, Price, Timestamp};

    fn create_test_portfolio() -> Portfolio {
        let mut portfolio = Portfolio::new(100_000.0);
        portfolio.positions = vec![
            Position {
                id: PositionId::new(),
                symbol: Symbol::new("SPY"),
                quantity: Quantity::from_f64(100.0),
                avg_entry_price: Price::from_f64(400.0),
                current_price: Price::from_f64(400.0),
                unrealized_pnl: 0.0,
                realized_pnl: 0.0,
                opened_at: Timestamp::now(),
                updated_at: Timestamp::now(),
            },
            Position {
                id: PositionId::new(),
                symbol: Symbol::new("TLT"),
                quantity: Quantity::from_f64(200.0),
                avg_entry_price: Price::from_f64(100.0),
                current_price: Price::from_f64(100.0),
                unrealized_pnl: 0.0,
                realized_pnl: 0.0,
                opened_at: Timestamp::now(),
                updated_at: Timestamp::now(),
            },
        ];
        portfolio.recalculate();
        portfolio
    }

    #[test]
    fn test_stress_test_creation() {
        let st = StressTest::new();
        assert!(!st.scenarios.is_empty());
    }

    #[test]
    fn test_2008_scenario() {
        let st = StressTest::new();
        let portfolio = create_test_portfolio();

        let scenario = StressScenario::financial_crisis_2008();
        let result = st.run_scenario(&portfolio, &scenario);

        // Should have significant negative P&L
        assert!(result.pnl < 0.0);
        // SPY should be worst
        // TLT might be best (bonds rally)
    }

    #[test]
    fn test_all_scenarios() {
        let st = StressTest::new();
        let portfolio = create_test_portfolio();

        let results = st.run_all(&portfolio);

        // Should run all scenarios
        assert_eq!(results.len(), 3);

        // All should have results
        for result in &results {
            assert!(result.pnl.is_finite());
        }
    }

    #[test]
    fn test_worst_case() {
        let st = StressTest::new();
        let portfolio = create_test_portfolio();

        let worst = st.worst_case(&portfolio);

        assert!(worst.is_some());
        let worst = worst.unwrap();
        assert!(worst.pnl < 0.0);
    }

    #[test]
    fn test_custom_scenario() {
        let scenario = StressScenario::custom(
            "Tech Crash".to_string(),
            vec![("SPY".to_string(), -0.25)],
        );

        assert_eq!(scenario.name, "Tech Crash");
        assert_eq!(scenario.shocks.len(), 1);
    }
}
