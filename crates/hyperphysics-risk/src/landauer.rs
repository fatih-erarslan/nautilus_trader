use hyperphysics_thermo::landauer::LandauerEnforcer;
use crate::error::{Result, RiskError};

/// Transaction cost model based on Landauer's principle
///
/// Models minimum energy cost per trade as thermodynamic bit erasure
/// E_min = k_B T ln(2)
pub struct TransactionCostModel {
    enforcer: LandauerEnforcer,
}

impl TransactionCostModel {
    pub fn new(temperature: f64) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(RiskError::InvalidTemperature(
                "Temperature must be positive".to_string()
            ));
        }

        Ok(Self {
            enforcer: LandauerEnforcer::new(temperature),
        })
    }

    /// Minimum energy cost per trade (bit erasure)
    /// E_min = k_B T ln(2)
    ///
    /// In financial context, represents fundamental transaction cost
    /// arising from information processing
    pub fn min_cost_per_trade(&self) -> f64 {
        self.enforcer.temperature() * std::f64::consts::LN_2
    }

    /// Total cost for n trades
    ///
    /// Linear scaling with number of trades represents
    /// cumulative information erasure
    pub fn total_transaction_cost(&self, n_trades: usize) -> f64 {
        self.min_cost_per_trade() * n_trades as f64
    }

    /// Calculate amortized cost per unit of value traded
    ///
    /// cost_density = E_min / trade_value
    pub fn cost_density(&self, trade_value: f64) -> Result<f64> {
        if trade_value <= 0.0 {
            return Err(RiskError::CalculationError(
                "Trade value must be positive".to_string()
            ));
        }

        Ok(self.min_cost_per_trade() / trade_value)
    }

    /// Calculate break-even trade size given expected return
    ///
    /// For a trade to be thermodynamically favorable:
    /// expected_return * trade_value >= E_min
    ///
    /// Therefore minimum trade size:
    /// trade_value_min = E_min / expected_return
    pub fn break_even_size(&self, expected_return: f64) -> Result<f64> {
        if expected_return <= 0.0 {
            return Err(RiskError::CalculationError(
                "Expected return must be positive".to_string()
            ));
        }

        Ok(self.min_cost_per_trade() / expected_return)
    }

    /// Get temperature parameter
    pub fn temperature(&self) -> f64 {
        self.enforcer.temperature()
    }

    /// Get underlying Landauer enforcer
    pub fn enforcer(&self) -> &LandauerEnforcer {
        &self.enforcer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_min_cost_per_trade() {
        let model = TransactionCostModel::new(1.0).unwrap();
        let cost = model.min_cost_per_trade();

        // Should be T * ln(2) = 1.0 * 0.693...
        assert_relative_eq!(cost, std::f64::consts::LN_2, epsilon = 1e-10);
    }

    #[test]
    fn test_total_transaction_cost() {
        let model = TransactionCostModel::new(2.0).unwrap();

        let n_trades = 10;
        let total_cost = model.total_transaction_cost(n_trades);

        // Should be n * T * ln(2)
        let expected = 10.0 * 2.0 * std::f64::consts::LN_2;
        assert_relative_eq!(total_cost, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_cost_density() {
        let model = TransactionCostModel::new(1.0).unwrap();

        let trade_value = 1000.0;
        let density = model.cost_density(trade_value).unwrap();

        // Should be E_min / trade_value
        let expected = std::f64::consts::LN_2 / 1000.0;
        assert_relative_eq!(density, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_break_even_size() {
        let model = TransactionCostModel::new(1.0).unwrap();

        let expected_return = 0.05; // 5% expected return
        let min_size = model.break_even_size(expected_return).unwrap();

        // Should be E_min / expected_return
        let expected = std::f64::consts::LN_2 / 0.05;
        assert_relative_eq!(min_size, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_temperature_scaling() {
        let model_low = TransactionCostModel::new(0.5).unwrap();
        let model_high = TransactionCostModel::new(2.0).unwrap();

        // Cost should scale linearly with temperature
        let ratio = model_high.min_cost_per_trade() / model_low.min_cost_per_trade();
        assert_relative_eq!(ratio, 4.0, epsilon = 1e-10);
    }
}
