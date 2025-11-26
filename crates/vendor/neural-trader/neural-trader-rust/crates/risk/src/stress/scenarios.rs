//! Stress test scenarios for portfolio risk analysis
//!
//! Implements historical and custom stress scenarios including:
//! - 2008 Financial Crisis
//! - 2020 COVID-19 Crash
//! - Custom multi-factor scenarios

use crate::{Result, RiskError};
use crate::types::{Portfolio, Position, Symbol, StressTestResult};
use crate::var::{MonteCarloVaR, VaRConfig};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Predefined stress test scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressScenario {
    /// 2008 Financial Crisis (-55% S&P 500, high correlation)
    FinancialCrisis2008,
    /// 2020 COVID-19 Crash (-34% S&P 500 in 23 days)
    CovidCrash2020,
    /// Market crash with severe volatility
    SevereMarketCrash { magnitude: f64 },
    /// Rapid interest rate hike scenario
    InterestRateShock { rate_change_bps: f64 },
    /// Liquidity crisis scenario
    LiquidityCrisis,
    /// Custom multi-factor scenario
    Custom(CustomScenario),
}

impl StressScenario {
    /// Get scenario name
    pub fn name(&self) -> String {
        match self {
            Self::FinancialCrisis2008 => "2008 Financial Crisis".to_string(),
            Self::CovidCrash2020 => "2020 COVID-19 Crash".to_string(),
            Self::SevereMarketCrash { magnitude } => {
                format!("Severe Market Crash (-{:.1}%)", magnitude * 100.0)
            }
            Self::InterestRateShock { rate_change_bps } => {
                format!("Interest Rate Shock (+{:.0}bps)", rate_change_bps)
            }
            Self::LiquidityCrisis => "Liquidity Crisis".to_string(),
            Self::Custom(scenario) => scenario.name.clone(),
        }
    }

    /// Get scenario parameters (shocks for different asset classes)
    pub fn get_shocks(&self) -> StressShocks {
        match self {
            Self::FinancialCrisis2008 => StressShocks {
                equity_shock: -0.55,
                fixed_income_shock: -0.15,
                commodity_shock: -0.40,
                currency_shock: 0.10,
                volatility_multiplier: 3.0,
                correlation_increase: 0.30,
            },
            Self::CovidCrash2020 => StressShocks {
                equity_shock: -0.34,
                fixed_income_shock: 0.05,
                commodity_shock: -0.60, // Oil crashed
                currency_shock: 0.15,
                volatility_multiplier: 2.5,
                correlation_increase: 0.25,
            },
            Self::SevereMarketCrash { magnitude } => StressShocks {
                equity_shock: -*magnitude,
                fixed_income_shock: -magnitude * 0.3,
                commodity_shock: -magnitude * 0.5,
                currency_shock: magnitude * 0.2,
                volatility_multiplier: 4.0,
                correlation_increase: 0.40,
            },
            Self::InterestRateShock { rate_change_bps } => {
                let rate_impact = rate_change_bps / 100.0; // Convert bps to decimal
                StressShocks {
                    equity_shock: -rate_impact * 0.05,
                    fixed_income_shock: -rate_impact * 0.10,
                    commodity_shock: rate_impact * 0.02,
                    currency_shock: rate_impact * 0.03,
                    volatility_multiplier: 1.5,
                    correlation_increase: 0.10,
                }
            }
            Self::LiquidityCrisis => StressShocks {
                equity_shock: -0.25,
                fixed_income_shock: -0.20,
                commodity_shock: -0.30,
                currency_shock: 0.25,
                volatility_multiplier: 3.5,
                correlation_increase: 0.50, // Everything correlated in liquidity crisis
            },
            Self::Custom(scenario) => scenario.shocks.clone(),
        }
    }
}

/// Stress shocks for different asset classes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressShocks {
    /// Equity shock (e.g., -0.30 for -30%)
    pub equity_shock: f64,
    /// Fixed income shock
    pub fixed_income_shock: f64,
    /// Commodity shock
    pub commodity_shock: f64,
    /// Currency shock
    pub currency_shock: f64,
    /// Volatility multiplier (e.g., 3.0 for 3x normal vol)
    pub volatility_multiplier: f64,
    /// Correlation increase (e.g., 0.30 for +30% correlation)
    pub correlation_increase: f64,
}

/// Custom stress scenario definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomScenario {
    pub name: String,
    pub shocks: StressShocks,
    pub duration_days: usize,
}

/// Stress test engine
pub struct StressTester {
    /// VaR calculator for post-shock analysis
    var_calculator: MonteCarloVaR,
    /// Number of scenarios to simulate
    num_scenarios: usize,
}

impl StressTester {
    /// Create new stress tester
    pub fn new(num_scenarios: usize) -> Self {
        let var_config = VaRConfig {
            confidence_level: 0.95,
            time_horizon_days: 1,
            num_simulations: 10_000,
            use_gpu: false,
        };

        Self {
            var_calculator: MonteCarloVaR::new(var_config),
            num_scenarios,
        }
    }

    /// Run stress test on portfolio
    pub async fn run_stress_test(
        &self,
        portfolio: &Portfolio,
        scenario: StressScenario,
    ) -> Result<StressTestResult> {
        info!("Running stress test: {}", scenario.name());

        let shocks = scenario.get_shocks();
        let positions: Vec<Position> = portfolio.positions.values().cloned().collect();

        // Apply immediate shock
        let immediate_impact = self.calculate_immediate_impact(&positions, &shocks)?;

        // Simulate post-shock returns distribution
        let shocked_positions = self.apply_shocks(&positions, &shocks)?;
        let post_shock_returns = self.simulate_post_shock_returns(&shocked_positions, &shocks)?;

        // Calculate tail risk metrics
        let var_result = self.var_calculator.calculate(&shocked_positions).await?;

        // Calculate survival probability (probability portfolio value stays positive)
        let survival_prob = self.calculate_survival_probability(&post_shock_returns);

        let result = StressTestResult {
            scenario_name: scenario.name(),
            immediate_impact,
            final_returns_distribution: post_shock_returns.clone(),
            survival_probability: survival_prob,
            var_95: var_result.var_95,
            cvar_95: var_result.cvar_95,
            expected_return: post_shock_returns.iter().sum::<f64>() / post_shock_returns.len() as f64,
            volatility: Self::calculate_volatility(&post_shock_returns),
            worst_case: post_shock_returns.iter().cloned().fold(f64::INFINITY, f64::min),
            best_case: post_shock_returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        };

        info!(
            "Stress test complete: immediate_impact={:.2}%, survival_prob={:.1}%",
            immediate_impact * 100.0,
            survival_prob * 100.0
        );

        Ok(result)
    }

    /// Calculate immediate portfolio impact from shocks
    fn calculate_immediate_impact(
        &self,
        positions: &[Position],
        shocks: &StressShocks,
    ) -> Result<f64> {
        let total_value: f64 = positions.iter().map(|p| p.exposure()).sum();
        if total_value == 0.0 {
            return Ok(0.0);
        }

        let mut shocked_value = 0.0;
        for position in positions {
            let asset_class = Self::classify_asset(&position.symbol);
            let shock = match asset_class {
                AssetClass::Equity => shocks.equity_shock,
                AssetClass::FixedIncome => shocks.fixed_income_shock,
                AssetClass::Commodity => shocks.commodity_shock,
                AssetClass::Currency => shocks.currency_shock,
            };

            let position_value = position.exposure();
            let shocked_position_value = position_value * (1.0 + shock);
            shocked_value += shocked_position_value;
        }

        let impact = (shocked_value - total_value) / total_value;
        debug!("Immediate impact: {:.2}%", impact * 100.0);
        Ok(impact)
    }

    /// Apply shocks to positions
    fn apply_shocks(&self, positions: &[Position], shocks: &StressShocks) -> Result<Vec<Position>> {
        let mut shocked_positions = Vec::new();

        for position in positions {
            let asset_class = Self::classify_asset(&position.symbol);
            let shock = match asset_class {
                AssetClass::Equity => shocks.equity_shock,
                AssetClass::FixedIncome => shocks.fixed_income_shock,
                AssetClass::Commodity => shocks.commodity_shock,
                AssetClass::Currency => shocks.currency_shock,
            };

            let mut shocked_position = position.clone();
            let shocked_price = position.current_price.to_f64().unwrap_or(0.0) * (1.0 + shock);
            shocked_position.current_price = rust_decimal::Decimal::from_f64_retain(shocked_price)
                .unwrap_or(position.current_price);

            shocked_positions.push(shocked_position);
        }

        Ok(shocked_positions)
    }

    /// Simulate post-shock returns distribution
    fn simulate_post_shock_returns(
        &self,
        positions: &[Position],
        shocks: &StressShocks,
    ) -> Result<Vec<f64>> {
        let mut rng = thread_rng();
        let mut returns = Vec::with_capacity(self.num_scenarios);

        let total_value: f64 = positions.iter().map(|p| p.exposure()).sum();
        if total_value == 0.0 {
            return Ok(vec![0.0; self.num_scenarios]);
        }

        // Use stressed volatility for simulation
        let base_vol = 0.15; // 15% annual volatility
        let stressed_vol = base_vol * shocks.volatility_multiplier;
        let daily_vol = stressed_vol / 252.0_f64.sqrt();

        for _ in 0..self.num_scenarios {
            let mut scenario_value = 0.0;

            for position in positions {
                let weight = position.exposure() / total_value;

                // Sample from stressed distribution
                let normal = Normal::new(0.0, daily_vol).map_err(|e| {
                    RiskError::StressTestError(format!("Failed to create distribution: {}", e))
                })?;
                let return_sample = normal.sample(&mut rng);

                scenario_value += weight * return_sample;
            }

            returns.push(scenario_value);
        }

        Ok(returns)
    }

    /// Calculate survival probability
    fn calculate_survival_probability(&self, returns: &[f64]) -> f64 {
        let surviving = returns.iter().filter(|&&r| r > -1.0).count();
        surviving as f64 / returns.len() as f64
    }

    /// Calculate volatility from returns
    fn calculate_volatility(returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64;

        variance.sqrt()
    }

    /// Classify asset for shock application
    fn classify_asset(symbol: &Symbol) -> AssetClass {
        let s = symbol.as_str().to_uppercase();

        // Simple heuristic classification
        if s.ends_with("USD") || s.ends_with("EUR") || s.ends_with("JPY") || s.ends_with("GBP") {
            AssetClass::Currency
        } else if s.starts_with("GC") || s.starts_with("CL") || s == "GOLD" || s == "OIL" {
            AssetClass::Commodity
        } else if s.contains("BOND") || s.contains("TLT") || s.contains("AGG") {
            AssetClass::FixedIncome
        } else {
            AssetClass::Equity
        }
    }

    /// Run multiple scenarios and aggregate results
    pub async fn run_multiple_scenarios(
        &self,
        portfolio: &Portfolio,
        scenarios: Vec<StressScenario>,
    ) -> Result<Vec<StressTestResult>> {
        let mut results = Vec::new();

        for scenario in scenarios {
            let result = self.run_stress_test(portfolio, scenario).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get worst-case scenario result
    pub async fn worst_case_analysis(
        &self,
        portfolio: &Portfolio,
    ) -> Result<StressTestResult> {
        let scenarios = vec![
            StressScenario::FinancialCrisis2008,
            StressScenario::CovidCrash2020,
            StressScenario::SevereMarketCrash { magnitude: 0.60 },
            StressScenario::LiquidityCrisis,
        ];

        let results = self.run_multiple_scenarios(portfolio, scenarios).await?;

        // Find worst result by immediate impact
        results
            .into_iter()
            .min_by(|a, b| {
                a.immediate_impact
                    .partial_cmp(&b.immediate_impact)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| RiskError::StressTestError("No worst case found".to_string()))
    }
}

/// Asset class for shock application
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AssetClass {
    Equity,
    FixedIncome,
    Commodity,
    Currency,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PositionSide};
    use chrono::Utc;
    use rust_decimal_macros::dec;

    fn create_test_portfolio() -> Portfolio {
        let mut portfolio = Portfolio::new(dec!(100000));

        portfolio.update_position(Position {
            symbol: Symbol::new("AAPL"),
            quantity: dec!(100),
            avg_entry_price: dec!(150.0),
            current_price: dec!(150.0),
            market_value: dec!(15000),
            unrealized_pnl: dec!(0),
            unrealized_pnl_percent: dec!(0),
            side: PositionSide::Long,
            opened_at: Utc::now(),
        });

        portfolio
    }

    #[tokio::test]
    async fn test_stress_test_2008() {
        let portfolio = create_test_portfolio();
        let tester = StressTester::new(1000);

        let result = tester
            .run_stress_test(&portfolio, StressScenario::FinancialCrisis2008)
            .await
            .unwrap();

        assert!(result.immediate_impact < 0.0); // Should show loss
        assert!(result.survival_probability > 0.0);
        assert!(result.survival_probability <= 1.0);
    }

    #[tokio::test]
    async fn test_stress_test_covid() {
        let portfolio = create_test_portfolio();
        let tester = StressTester::new(1000);

        let result = tester
            .run_stress_test(&portfolio, StressScenario::CovidCrash2020)
            .await
            .unwrap();

        assert!(result.immediate_impact < 0.0);
        assert_eq!(result.scenario_name, "2020 COVID-19 Crash");
    }

    #[tokio::test]
    async fn test_multiple_scenarios() {
        let portfolio = create_test_portfolio();
        let tester = StressTester::new(500);

        let scenarios = vec![
            StressScenario::FinancialCrisis2008,
            StressScenario::CovidCrash2020,
        ];

        let results = tester
            .run_multiple_scenarios(&portfolio, scenarios)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_worst_case_analysis() {
        let portfolio = create_test_portfolio();
        let tester = StressTester::new(500);

        let result = tester.worst_case_analysis(&portfolio).await.unwrap();

        // Worst case should have significant negative impact
        assert!(result.immediate_impact < -0.30);
    }

    #[test]
    fn test_custom_scenario() {
        let custom = CustomScenario {
            name: "Test Crisis".to_string(),
            shocks: StressShocks {
                equity_shock: -0.40,
                fixed_income_shock: -0.10,
                commodity_shock: -0.30,
                currency_shock: 0.05,
                volatility_multiplier: 2.5,
                correlation_increase: 0.25,
            },
            duration_days: 30,
        };

        let scenario = StressScenario::Custom(custom);
        assert_eq!(scenario.name(), "Test Crisis");

        let shocks = scenario.get_shocks();
        assert_eq!(shocks.equity_shock, -0.40);
    }
}
