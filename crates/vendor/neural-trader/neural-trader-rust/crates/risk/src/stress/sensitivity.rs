//! Sensitivity analysis for multi-factor risk assessment
//!
//! Analyzes portfolio sensitivity to changes in market factors:
//! - Interest rates
//! - Volatility
//! - Correlations
//! - Spreads

use crate::{Result};
use crate::types::{Portfolio, Position};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Sensitivity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityResult {
    pub factor_name: String,
    pub base_value: f64,
    pub scenarios: Vec<SensitivityScenario>,
    pub total_range: f64,
}

/// Individual sensitivity scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityScenario {
    pub factor_change: f64,
    pub portfolio_impact: f64,
    pub duration: f64, // Sensitivity measure
}

/// Market factors for sensitivity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketFactor {
    /// Interest rate changes (in basis points)
    InterestRate,
    /// Volatility changes (multiplicative, e.g., 1.5 = 50% increase)
    Volatility,
    /// Correlation changes (additive, e.g., 0.2 = +20%)
    Correlation,
    /// Credit spread changes (in basis points)
    CreditSpread,
    /// Exchange rate changes (%)
    ExchangeRate,
    /// Custom factor
    Custom(String),
}

impl MarketFactor {
    pub fn name(&self) -> String {
        match self {
            Self::InterestRate => "Interest Rate".to_string(),
            Self::Volatility => "Volatility".to_string(),
            Self::Correlation => "Correlation".to_string(),
            Self::CreditSpread => "Credit Spread".to_string(),
            Self::ExchangeRate => "Exchange Rate".to_string(),
            Self::Custom(name) => name.clone(),
        }
    }
}

/// Sensitivity analyzer
pub struct SensitivityAnalyzer {
    /// Number of scenarios per factor
    num_scenarios: usize,
    /// Range of factor changes (symmetric around current value)
    factor_range: f64,
}

impl SensitivityAnalyzer {
    /// Create new sensitivity analyzer
    ///
    /// # Arguments
    /// * `num_scenarios` - Number of scenarios to test per factor
    /// * `factor_range` - Range of factor changes (e.g., 2.0 = +/-200 bps for rates)
    pub fn new(num_scenarios: usize, factor_range: f64) -> Self {
        Self {
            num_scenarios,
            factor_range,
        }
    }

    /// Run sensitivity analysis for a single factor
    pub fn analyze_factor(
        &self,
        portfolio: &Portfolio,
        factor: MarketFactor,
    ) -> Result<SensitivityResult> {
        info!("Running sensitivity analysis for {}", factor.name());

        let base_value = self.get_base_factor_value(&factor)?;
        let mut scenarios = Vec::with_capacity(self.num_scenarios);

        // Generate scenarios from -range to +range
        for i in 0..self.num_scenarios {
            let pct = -1.0 + (2.0 * i as f64 / (self.num_scenarios - 1) as f64);
            let factor_change = pct * self.factor_range;

            let impact = self.calculate_factor_impact(portfolio, &factor, factor_change)?;
            let duration = self.calculate_duration(&factor, factor_change);

            scenarios.push(SensitivityScenario {
                factor_change,
                portfolio_impact: impact,
                duration,
            });
        }

        // Calculate total range
        let impacts: Vec<f64> = scenarios.iter().map(|s| s.portfolio_impact).collect();
        let total_range = impacts.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - impacts.iter().cloned().fold(f64::INFINITY, f64::min);

        let result = SensitivityResult {
            factor_name: factor.name(),
            base_value,
            scenarios,
            total_range,
        };

        debug!(
            "Sensitivity analysis complete: {} scenarios, range={:.2}%",
            self.num_scenarios,
            total_range * 100.0
        );

        Ok(result)
    }

    /// Run multi-factor sensitivity analysis
    pub fn analyze_multiple_factors(
        &self,
        portfolio: &Portfolio,
        factors: Vec<MarketFactor>,
    ) -> Result<Vec<SensitivityResult>> {
        let mut results = Vec::new();

        for factor in factors {
            let result = self.analyze_factor(portfolio, factor)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Calculate cross-factor sensitivity (interaction effects)
    pub fn analyze_cross_sensitivity(
        &self,
        portfolio: &Portfolio,
        factor1: MarketFactor,
        factor2: MarketFactor,
    ) -> Result<CrossSensitivityResult> {
        info!(
            "Analyzing cross-sensitivity: {} vs {}",
            factor1.name(),
            factor2.name()
        );

        let n_points = (self.num_scenarios as f64).sqrt() as usize;
        let mut grid = Vec::new();

        for i in 0..n_points {
            for j in 0..n_points {
                let pct1 = -1.0 + (2.0 * i as f64 / (n_points - 1) as f64);
                let pct2 = -1.0 + (2.0 * j as f64 / (n_points - 1) as f64);

                let change1 = pct1 * self.factor_range;
                let change2 = pct2 * self.factor_range;

                let impact1 = self.calculate_factor_impact(portfolio, &factor1, change1)?;
                let impact2 = self.calculate_factor_impact(portfolio, &factor2, change2)?;
                let combined_impact = impact1 + impact2;

                grid.push(CrossSensitivityPoint {
                    factor1_change: change1,
                    factor2_change: change2,
                    portfolio_impact: combined_impact,
                });
            }
        }

        Ok(CrossSensitivityResult {
            factor1_name: factor1.name(),
            factor2_name: factor2.name(),
            grid,
        })
    }

    /// Get base value for a market factor
    fn get_base_factor_value(&self, factor: &MarketFactor) -> Result<f64> {
        // In production, these would come from market data
        Ok(match factor {
            MarketFactor::InterestRate => 5.0,        // 5% base rate
            MarketFactor::Volatility => 1.0,          // Base volatility multiplier
            MarketFactor::Correlation => 0.5,         // 50% base correlation
            MarketFactor::CreditSpread => 100.0,      // 100 bps base spread
            MarketFactor::ExchangeRate => 1.0,        // Par exchange rate
            MarketFactor::Custom(_) => 0.0,
        })
    }

    /// Calculate portfolio impact from factor change
    fn calculate_factor_impact(
        &self,
        portfolio: &Portfolio,
        factor: &MarketFactor,
        change: f64,
    ) -> Result<f64> {
        let total_value = portfolio.total_value();
        if total_value == 0.0 {
            return Ok(0.0);
        }

        let mut impact = 0.0;

        for position in portfolio.positions.values() {
            let position_impact = match factor {
                MarketFactor::InterestRate => {
                    // Duration-based impact: ΔP/P ≈ -Duration × Δr
                    let duration = self.estimate_duration(position);
                    -duration * change / 100.0 // change is in bps
                }
                MarketFactor::Volatility => {
                    // Volatility impact (gamma and vega effects)
                    let gamma = self.estimate_gamma(position);
                    gamma * (change - 1.0)
                }
                MarketFactor::Correlation => {
                    // Correlation impact on diversification benefits
                    let diversification_impact = 0.05 * change; // Simplified
                    diversification_impact
                }
                MarketFactor::CreditSpread => {
                    // Credit spread impact
                    let credit_duration = self.estimate_duration(position) * 0.7;
                    -credit_duration * change / 100.0
                }
                MarketFactor::ExchangeRate => {
                    // FX exposure impact
                    change * 0.01 // Simplified
                }
                MarketFactor::Custom(_) => 0.0,
            };

            let weight = position.exposure() / total_value;
            impact += weight * position_impact;
        }

        Ok(impact)
    }

    /// Estimate position duration (interest rate sensitivity)
    fn estimate_duration(&self, position: &Position) -> f64 {
        let symbol = position.symbol.as_str().to_uppercase();

        // Simplified duration estimates by asset type
        if symbol.contains("BOND") || symbol.contains("TLT") {
            10.0 // Long-term bonds
        } else if symbol.contains("AGG") {
            5.0 // Aggregate bonds
        } else {
            2.0 // Equity duration (from dividend discount model)
        }
    }

    /// Estimate position gamma (convexity)
    fn estimate_gamma(&self, _position: &Position) -> f64 {
        // Simplified gamma estimate
        0.01 // Small gamma for most positions
    }

    /// Calculate duration metric for factor sensitivity
    fn calculate_duration(&self, factor: &MarketFactor, change: f64) -> f64 {
        match factor {
            MarketFactor::InterestRate => change.abs() / 100.0,
            MarketFactor::Volatility => change.abs() - 1.0,
            _ => change.abs(),
        }
    }

    /// Identify key risk factors (highest sensitivity)
    pub fn identify_key_risks(
        &self,
        portfolio: &Portfolio,
    ) -> Result<Vec<(MarketFactor, f64)>> {
        let factors = vec![
            MarketFactor::InterestRate,
            MarketFactor::Volatility,
            MarketFactor::Correlation,
            MarketFactor::CreditSpread,
        ];

        let mut risk_rankings = Vec::new();

        for factor in factors {
            let result = self.analyze_factor(portfolio, factor.clone())?;
            risk_rankings.push((factor, result.total_range));
        }

        // Sort by range (descending)
        risk_rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        info!("Key risks identified: {} factors analyzed", risk_rankings.len());

        Ok(risk_rankings)
    }
}

/// Cross-sensitivity analysis result (2D grid)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSensitivityResult {
    pub factor1_name: String,
    pub factor2_name: String,
    pub grid: Vec<CrossSensitivityPoint>,
}

/// Point in cross-sensitivity grid
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSensitivityPoint {
    pub factor1_change: f64,
    pub factor2_change: f64,
    pub portfolio_impact: f64,
}

impl Default for SensitivityAnalyzer {
    fn default() -> Self {
        Self::new(11, 2.0) // 11 scenarios, +/-200 bps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PositionSide, Symbol};
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

        portfolio.update_position(Position {
            symbol: Symbol::new("TLT"),
            quantity: dec!(50),
            avg_entry_price: dec!(100.0),
            current_price: dec!(100.0),
            market_value: dec!(5000),
            unrealized_pnl: dec!(0),
            unrealized_pnl_percent: dec!(0),
            side: PositionSide::Long,
            opened_at: Utc::now(),
        });

        portfolio
    }

    #[test]
    fn test_interest_rate_sensitivity() {
        let portfolio = create_test_portfolio();
        let analyzer = SensitivityAnalyzer::new(11, 2.0);

        let result = analyzer
            .analyze_factor(&portfolio, MarketFactor::InterestRate)
            .unwrap();

        assert_eq!(result.scenarios.len(), 11);
        assert!(result.total_range > 0.0);
    }

    #[test]
    fn test_volatility_sensitivity() {
        let portfolio = create_test_portfolio();
        let analyzer = SensitivityAnalyzer::new(11, 1.5);

        let result = analyzer
            .analyze_factor(&portfolio, MarketFactor::Volatility)
            .unwrap();

        assert_eq!(result.scenarios.len(), 11);
        assert_eq!(result.factor_name, "Volatility");
    }

    #[test]
    fn test_multiple_factors() {
        let portfolio = create_test_portfolio();
        let analyzer = SensitivityAnalyzer::default();

        let factors = vec![
            MarketFactor::InterestRate,
            MarketFactor::Volatility,
        ];

        let results = analyzer
            .analyze_multiple_factors(&portfolio, factors)
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_cross_sensitivity() {
        let portfolio = create_test_portfolio();
        let analyzer = SensitivityAnalyzer::new(9, 1.0); // 3x3 grid

        let result = analyzer
            .analyze_cross_sensitivity(
                &portfolio,
                MarketFactor::InterestRate,
                MarketFactor::Volatility,
            )
            .unwrap();

        assert_eq!(result.grid.len(), 9); // 3x3 grid
    }

    #[test]
    fn test_key_risks_identification() {
        let portfolio = create_test_portfolio();
        let analyzer = SensitivityAnalyzer::default();

        let key_risks = analyzer.identify_key_risks(&portfolio).unwrap();

        assert!(!key_risks.is_empty());
        // Should be sorted by risk magnitude
        if key_risks.len() >= 2 {
            assert!(key_risks[0].1 >= key_risks[1].1);
        }
    }
}
