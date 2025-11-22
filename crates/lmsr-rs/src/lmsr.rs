//! Core LMSR (Logarithmic Market Scoring Rule) implementation
//! 
//! This module provides the mathematical foundation for LMSR market making,
//! including price calculation, cost functions, and market state updates.

use crate::errors::{LMSRError, Result};
use crate::utils::{log_sum_exp, validate_finite_vector, validate_positive_finite};
use serde::{Deserialize, Serialize};
use std::f64;

/// Core LMSR calculator providing mathematical operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSRCalculator {
    /// Number of outcomes in the market
    pub num_outcomes: usize,
    /// Liquidity parameter (b) controlling market sensitivity
    pub liquidity_parameter: f64,
}

impl LMSRCalculator {
    /// Create a new LMSR calculator
    pub fn new(num_outcomes: usize, liquidity_parameter: f64) -> Result<Self> {
        if num_outcomes < 2 {
            return Err(LMSRError::invalid_market(
                format!("Number of outcomes must be at least 2, got {}", num_outcomes)
            ));
        }
        
        validate_positive_finite(liquidity_parameter, "liquidity_parameter")?;
        
        Ok(Self {
            num_outcomes,
            liquidity_parameter,
        })
    }
    
    /// Calculate the cost function C(q) = b * log(sum(exp(q_i / b)))
    /// 
    /// This is the core LMSR cost function that determines the total cost
    /// of holding a portfolio of shares.
    pub fn cost_function(&self, quantities: &[f64]) -> Result<f64> {
        if quantities.len() != self.num_outcomes {
            return Err(LMSRError::invalid_quantity(
                format!("Expected {} quantities, got {}", self.num_outcomes, quantities.len())
            ));
        }
        
        validate_finite_vector(quantities, "quantities")?;
        
        // Normalize quantities by liquidity parameter for numerical stability
        let normalized: Vec<f64> = quantities.iter()
            .map(|&q| q / self.liquidity_parameter)
            .collect();
        
        let log_sum = log_sum_exp(&normalized)?;
        let cost = self.liquidity_parameter * log_sum;
        
        if !cost.is_finite() {
            return Err(LMSRError::numerical_error("Cost function result is not finite"));
        }
        
        Ok(cost)
    }
    
    /// Calculate the marginal cost (price) for outcome i
    /// 
    /// Price_i = ∂C/∂q_i = exp(q_i / b) / sum(exp(q_j / b))
    /// 
    /// This gives the instantaneous price for buying an infinitesimal
    /// amount of shares in outcome i.
    pub fn marginal_price(&self, quantities: &[f64], outcome: usize) -> Result<f64> {
        if outcome >= self.num_outcomes {
            return Err(LMSRError::invalid_outcome(outcome, self.num_outcomes));
        }
        
        if quantities.len() != self.num_outcomes {
            return Err(LMSRError::invalid_quantity(
                format!("Expected {} quantities, got {}", self.num_outcomes, quantities.len())
            ));
        }
        
        validate_finite_vector(quantities, "quantities")?;
        
        // Normalize quantities by liquidity parameter
        let normalized: Vec<f64> = quantities.iter()
            .map(|&q| q / self.liquidity_parameter)
            .collect();
        
        let log_sum = log_sum_exp(&normalized)?;
        let price = (normalized[outcome] - log_sum).exp();
        
        if !price.is_finite() {
            return Err(LMSRError::numerical_error("Price calculation resulted in non-finite value"));
        }
        
        // Ensure price is in valid range [0, 1]
        let clamped_price = price.max(f64::EPSILON).min(1.0 - f64::EPSILON);
        
        Ok(clamped_price)
    }
    
    /// Calculate all marginal prices simultaneously for efficiency
    pub fn all_marginal_prices(&self, quantities: &[f64]) -> Result<Vec<f64>> {
        if quantities.len() != self.num_outcomes {
            return Err(LMSRError::invalid_quantity(
                format!("Expected {} quantities, got {}", self.num_outcomes, quantities.len())
            ));
        }
        
        validate_finite_vector(quantities, "quantities")?;
        
        // Use softmax for numerical stability
        let normalized: Vec<f64> = quantities.iter()
            .map(|&q| q / self.liquidity_parameter)
            .collect();
        
        crate::utils::softmax(&normalized)
    }
    
    /// Calculate the cost of trading from current quantities to new quantities
    pub fn trade_cost(&self, current_quantities: &[f64], new_quantities: &[f64]) -> Result<f64> {
        let current_cost = self.cost_function(current_quantities)?;
        let new_cost = self.cost_function(new_quantities)?;
        
        Ok(new_cost - current_cost)
    }
    
    /// Calculate the quantities after buying specified amounts
    pub fn buy_shares(&self, current_quantities: &[f64], buy_amounts: &[f64]) -> Result<Vec<f64>> {
        if current_quantities.len() != self.num_outcomes || buy_amounts.len() != self.num_outcomes {
            return Err(LMSRError::invalid_quantity("Mismatched vector lengths"));
        }
        
        validate_finite_vector(current_quantities, "current_quantities")?;
        validate_finite_vector(buy_amounts, "buy_amounts")?;
        
        let new_quantities: Vec<f64> = current_quantities.iter()
            .zip(buy_amounts.iter())
            .map(|(&current, &buy)| current + buy)
            .collect();
        
        validate_finite_vector(&new_quantities, "new_quantities")?;
        
        Ok(new_quantities)
    }
    
    /// Calculate the exact cost of buying specific amounts of shares
    pub fn calculate_buy_cost(&self, current_quantities: &[f64], buy_amounts: &[f64]) -> Result<f64> {
        let new_quantities = self.buy_shares(current_quantities, buy_amounts)?;
        self.trade_cost(current_quantities, &new_quantities)
    }
    
    /// Calculate the implied probabilities from current quantities
    /// These represent the market's assessment of outcome probabilities
    pub fn implied_probabilities(&self, quantities: &[f64]) -> Result<Vec<f64>> {
        self.all_marginal_prices(quantities)
    }
    
    /// Calculate the maximum profitable trade given price constraints
    /// Returns optimal quantities to trade for arbitrage opportunities
    pub fn optimal_arbitrage_trade(
        &self, 
        current_quantities: &[f64], 
        external_prices: &[f64]
    ) -> Result<Vec<f64>> {
        if external_prices.len() != self.num_outcomes {
            return Err(LMSRError::invalid_quantity("External prices length mismatch"));
        }
        
        validate_finite_vector(external_prices, "external_prices")?;
        
        // Check that external prices sum to approximately 1.0
        let price_sum: f64 = external_prices.iter().sum();
        if (price_sum - 1.0).abs() > 1e-6 {
            return Err(LMSRError::invalid_quantity(
                format!("External prices sum to {}, should be 1.0", price_sum)
            ));
        }
        
        let current_prices = self.all_marginal_prices(current_quantities)?;
        
        // Calculate arbitrage opportunities
        let mut trade_amounts = vec![0.0; self.num_outcomes];
        
        for i in 0..self.num_outcomes {
            let price_diff = external_prices[i] - current_prices[i];
            
            // Simple linear approximation for trade size
            // In a real implementation, this would use more sophisticated optimization
            trade_amounts[i] = self.liquidity_parameter * price_diff;
        }
        
        Ok(trade_amounts)
    }
}

/// Market maker implementing LMSR with additional features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMSRMarketMaker {
    pub calculator: LMSRCalculator,
    /// Current share quantities held by the market maker
    pub quantities: Vec<f64>,
    /// Total volume traded
    pub total_volume: f64,
    /// Number of trades executed
    pub trade_count: u64,
    /// Market creation timestamp
    pub created_at: std::time::SystemTime,
}

impl LMSRMarketMaker {
    /// Create a new LMSR market maker with initial liquidity
    pub fn new(num_outcomes: usize, liquidity_parameter: f64) -> Result<Self> {
        let calculator = LMSRCalculator::new(num_outcomes, liquidity_parameter)?;
        
        Ok(Self {
            calculator,
            quantities: vec![0.0; num_outcomes],
            total_volume: 0.0,
            trade_count: 0,
            created_at: std::time::SystemTime::now(),
        })
    }
    
    /// Execute a trade and update market state
    pub fn execute_trade(&mut self, buy_amounts: &[f64]) -> Result<f64> {
        let cost = self.calculator.calculate_buy_cost(&self.quantities, buy_amounts)?;
        
        // Update quantities
        for i in 0..self.quantities.len() {
            self.quantities[i] += buy_amounts[i];
        }
        
        // Update statistics
        let trade_volume: f64 = buy_amounts.iter().map(|x| x.abs()).sum();
        self.total_volume += trade_volume;
        self.trade_count += 1;
        
        Ok(cost)
    }
    
    /// Get current market prices
    pub fn get_prices(&self) -> Result<Vec<f64>> {
        self.calculator.all_marginal_prices(&self.quantities)
    }
    
    /// Get price for specific outcome
    pub fn get_price(&self, outcome: usize) -> Result<f64> {
        self.calculator.marginal_price(&self.quantities, outcome)
    }
    
    /// Get market statistics
    pub fn get_statistics(&self) -> Result<MarketStatistics> {
        let prices = self.get_prices()?;
        let cost = self.calculator.cost_function(&self.quantities)?;
        
        Ok(MarketStatistics {
            current_prices: prices,
            total_cost: cost,
            total_volume: self.total_volume,
            trade_count: self.trade_count,
            liquidity_parameter: self.calculator.liquidity_parameter,
            quantities: self.quantities.clone(),
        })
    }
    
    /// Reset market to initial state
    pub fn reset(&mut self) {
        self.quantities.fill(0.0);
        self.total_volume = 0.0;
        self.trade_count = 0;
        self.created_at = std::time::SystemTime::now();
    }
}

/// Market statistics for monitoring and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStatistics {
    pub current_prices: Vec<f64>,
    pub total_cost: f64,
    pub total_volume: f64,
    pub trade_count: u64,
    pub liquidity_parameter: f64,
    pub quantities: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lmsr_calculator_creation() {
        let calc = LMSRCalculator::new(2, 100.0).unwrap();
        assert_eq!(calc.num_outcomes, 2);
        assert_eq!(calc.liquidity_parameter, 100.0);
        
        // Test invalid parameters
        assert!(LMSRCalculator::new(1, 100.0).is_err());
        assert!(LMSRCalculator::new(2, 0.0).is_err());
        assert!(LMSRCalculator::new(2, -1.0).is_err());
    }

    #[test]
    fn test_cost_function() {
        let calc = LMSRCalculator::new(2, 100.0).unwrap();
        
        // Initial state (no shares)
        let cost = calc.cost_function(&[0.0, 0.0]).unwrap();
        assert_relative_eq!(cost, 100.0 * (2.0_f64).ln(), epsilon = 1e-10);
        
        // Test with some shares
        let cost = calc.cost_function(&[50.0, 50.0]).unwrap();
        assert!(cost > 0.0);
    }

    #[test]
    fn test_marginal_prices() {
        let calc = LMSRCalculator::new(2, 100.0).unwrap();
        
        // Equal quantities should give equal prices around 0.5
        let price0 = calc.marginal_price(&[0.0, 0.0], 0).unwrap();
        let price1 = calc.marginal_price(&[0.0, 0.0], 1).unwrap();
        
        assert_relative_eq!(price0, 0.5, epsilon = 1e-10);
        assert_relative_eq!(price1, 0.5, epsilon = 1e-10);
        
        // Higher quantity should lead to higher price
        let price0 = calc.marginal_price(&[100.0, 0.0], 0).unwrap();
        let price1 = calc.marginal_price(&[100.0, 0.0], 1).unwrap();
        
        assert!(price0 > price1);
        assert!(price0 > 0.5);
        assert!(price1 < 0.5);
    }

    #[test]
    fn test_all_marginal_prices_sum() {
        let calc = LMSRCalculator::new(3, 100.0).unwrap();
        let prices = calc.all_marginal_prices(&[10.0, 20.0, 30.0]).unwrap();
        
        let sum: f64 = prices.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // Prices should be ordered
        assert!(prices[0] < prices[1]);
        assert!(prices[1] < prices[2]);
    }

    #[test]
    fn test_market_maker_trades() {
        let mut market = LMSRMarketMaker::new(2, 100.0).unwrap();
        
        // Buy some shares
        let cost = market.execute_trade(&[10.0, 0.0]).unwrap();
        assert!(cost > 0.0);
        
        // Check that quantities were updated
        assert_eq!(market.quantities[0], 10.0);
        assert_eq!(market.quantities[1], 0.0);
        
        // Check that statistics were updated
        assert!(market.total_volume > 0.0);
        assert_eq!(market.trade_count, 1);
    }

    #[test]
    fn test_arbitrage_calculation() {
        let calc = LMSRCalculator::new(2, 100.0).unwrap();
        let current_quantities = vec![0.0, 0.0];
        let external_prices = vec![0.6, 0.4];
        
        let arbitrage_trades = calc.optimal_arbitrage_trade(&current_quantities, &external_prices).unwrap();
        
        // Should suggest buying outcome 0 (higher external price than market price of 0.5)
        assert!(arbitrage_trades[0] > 0.0);
        assert!(arbitrage_trades[1] < 0.0);
    }

    #[test]
    fn test_numerical_stability() {
        let calc = LMSRCalculator::new(2, 1.0).unwrap();
        
        // Test with very large quantities
        let large_quantities = vec![1000.0, 1000.0];
        let prices = calc.all_marginal_prices(&large_quantities).unwrap();
        assert!(prices.iter().all(|&p| p.is_finite()));
        
        // Test with very small liquidity parameter
        let calc_small = LMSRCalculator::new(2, 0.001).unwrap();
        let prices = calc_small.all_marginal_prices(&[0.1, 0.1]).unwrap();
        assert!(prices.iter().all(|&p| p.is_finite()));
    }
}