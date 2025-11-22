/// Risk calculation engine integrating all risk models
use ndarray::Array1;
use crate::types::{FinanceError, L2Snapshot};
use super::{RiskMetrics, OptionParams, Greeks, calculate_black_scholes, VarModel, calculate_var};

/// Configuration for risk calculations
#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// VaR confidence level (default: 0.95)
    pub var_confidence: f64,

    /// VaR model to use
    pub var_model: VarModel,

    /// Periods per year for annualization (252 for daily, 12 for monthly)
    pub periods_per_year: f64,

    /// Minimum data points required
    pub min_data_points: usize,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            var_confidence: 0.95,
            var_model: VarModel::EWMA,
            periods_per_year: 252.0,
            min_data_points: 30,
        }
    }
}

/// Main risk calculation engine
pub struct RiskEngine {
    config: RiskConfig,
    price_history: Vec<f64>,
}

impl RiskEngine {
    /// Create new risk engine with configuration
    pub fn new(config: RiskConfig) -> Self {
        Self {
            config,
            price_history: Vec::new(),
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(RiskConfig::default())
    }

    /// Update price history from order book snapshot
    pub fn update_from_snapshot(&mut self, snapshot: &L2Snapshot) -> Result<(), FinanceError> {
        snapshot.validate()?;

        if let Some(mid_price) = snapshot.mid_price() {
            self.price_history.push(mid_price.value());

            // Keep only last 1000 prices to avoid unbounded memory
            if self.price_history.len() > 1000 {
                self.price_history.drain(0..self.price_history.len() - 1000);
            }
        }

        Ok(())
    }

    /// Calculate returns from price history
    fn calculate_returns(&self) -> Result<Array1<f64>, FinanceError> {
        if self.price_history.len() < 2 {
            return Err(FinanceError::InsufficientData);
        }

        let returns: Vec<f64> = self.price_history
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        Ok(Array1::from(returns))
    }

    /// Calculate comprehensive risk metrics
    pub fn calculate_metrics(&self) -> Result<RiskMetrics, FinanceError> {
        if self.price_history.len() < self.config.min_data_points {
            return Err(FinanceError::InsufficientData);
        }

        let returns = self.calculate_returns()?;
        RiskMetrics::from_returns(returns.view(), self.config.periods_per_year)
    }

    /// Calculate VaR using configured model
    pub fn calculate_var(&self) -> Result<f64, FinanceError> {
        if self.price_history.len() < self.config.min_data_points {
            return Err(FinanceError::InsufficientData);
        }

        let returns = self.calculate_returns()?;
        calculate_var(returns.view(), self.config.var_model, self.config.var_confidence)
    }

    /// Calculate Black-Scholes Greeks for current price
    pub fn calculate_greeks(&self, params: &OptionParams) -> Result<Greeks, FinanceError> {
        let (_, greeks) = calculate_black_scholes(params)?;
        Ok(greeks)
    }

    /// Get current price (latest mid-price)
    pub fn current_price(&self) -> Option<f64> {
        self.price_history.last().copied()
    }

    /// Get realized volatility from price history
    pub fn realized_volatility(&self) -> Result<f64, FinanceError> {
        if self.price_history.len() < self.config.min_data_points {
            return Err(FinanceError::InsufficientData);
        }

        let returns = self.calculate_returns()?;
        let mean = returns.mean().unwrap_or(0.0);
        let variance = returns
            .iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;

        Ok(variance.sqrt() * self.config.periods_per_year.sqrt())
    }

    /// Clear price history
    pub fn clear_history(&mut self) {
        self.price_history.clear();
    }

    /// Get number of prices in history
    pub fn history_size(&self) -> usize {
        self.price_history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Price, Quantity};

    fn create_test_snapshot(mid_price: f64) -> L2Snapshot {
        let bid_price = mid_price - 0.5;
        let ask_price = mid_price + 0.5;

        L2Snapshot {
            symbol: "BTC-USD".to_string(),
            timestamp_us: 1000000,
            bids: vec![
                (Price::new(bid_price).unwrap(), Quantity::new(1.0).unwrap()),
            ],
            asks: vec![
                (Price::new(ask_price).unwrap(), Quantity::new(1.0).unwrap()),
            ],
        }
    }

    #[test]
    fn test_risk_engine_initialization() {
        let engine = RiskEngine::with_defaults();
        assert_eq!(engine.history_size(), 0);
        assert!(engine.current_price().is_none());
    }

    #[test]
    fn test_update_from_snapshot() {
        let mut engine = RiskEngine::with_defaults();
        let snapshot = create_test_snapshot(100.0);

        engine.update_from_snapshot(&snapshot).unwrap();
        assert_eq!(engine.history_size(), 1);
        assert_eq!(engine.current_price().unwrap(), 100.0);
    }

    #[test]
    fn test_insufficient_data() {
        let engine = RiskEngine::with_defaults();

        // Should fail with no data
        assert!(engine.calculate_metrics().is_err());
        assert!(engine.calculate_var().is_err());
    }

    #[test]
    fn test_calculate_metrics_with_data() {
        let mut engine = RiskEngine::with_defaults();

        // Add 50 prices with some volatility
        for i in 0..50 {
            let price = 100.0 + (i as f64 * 0.5) + ((i % 3) as f64 - 1.0) * 2.0;
            let snapshot = create_test_snapshot(price);
            engine.update_from_snapshot(&snapshot).unwrap();
        }

        let metrics = engine.calculate_metrics().unwrap();
        assert!(metrics.volatility > 0.0);
        assert!(metrics.var_95 > 0.0);
        assert!(metrics.var_99 > metrics.var_95);
    }

    #[test]
    fn test_calculate_greeks() {
        let engine = RiskEngine::with_defaults();

        let params = OptionParams {
            spot: 100.0,
            strike: 100.0,
            rate: 0.05,
            volatility: 0.20,
            time_to_maturity: 1.0,
        };

        let greeks = engine.calculate_greeks(&params).unwrap();
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0);
        assert!(greeks.gamma > 0.0);
        assert!(greeks.vega > 0.0);
    }

    #[test]
    fn test_realized_volatility() {
        let mut engine = RiskEngine::with_defaults();

        // Add prices with known volatility pattern
        for i in 0..100 {
            let price = 100.0 + (i as f64 / 10.0).sin() * 5.0;
            let snapshot = create_test_snapshot(price);
            engine.update_from_snapshot(&snapshot).unwrap();
        }

        let vol = engine.realized_volatility().unwrap();
        assert!(vol > 0.0);
        assert!(vol < 2.0);  // Should be reasonable for this pattern
    }

    #[test]
    fn test_history_size_limit() {
        let mut engine = RiskEngine::with_defaults();

        // Add more than 1000 prices
        for i in 0..1200 {
            let snapshot = create_test_snapshot(100.0 + i as f64);
            engine.update_from_snapshot(&snapshot).unwrap();
        }

        // Should cap at 1000
        assert_eq!(engine.history_size(), 1000);
    }

    #[test]
    fn test_clear_history() {
        let mut engine = RiskEngine::with_defaults();

        let snapshot = create_test_snapshot(100.0);
        engine.update_from_snapshot(&snapshot).unwrap();
        assert_eq!(engine.history_size(), 1);

        engine.clear_history();
        assert_eq!(engine.history_size(), 0);
    }
}
