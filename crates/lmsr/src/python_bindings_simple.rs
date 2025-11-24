//! Simplified PyO3 Python bindings for LMSR functionality
//! 
//! This module provides a basic Python interface to the LMSR implementation
//! without advanced features that cause compilation issues.

use crate::market::{Market, MarketFactory};
use crate::lmsr::LMSRCalculator;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Simple Python wrapper for Market
#[pyclass(name = "LMSRMarket")]
#[derive(Clone)]
pub struct PyLMSRMarket {
    market: Market,
}

#[pymethods]
impl PyLMSRMarket {
    /// Create a new LMSR market
    #[new]
    fn new(num_outcomes: usize, liquidity_parameter: f64) -> PyResult<Self> {
        let market = Market::new(num_outcomes, liquidity_parameter)?;
        Ok(Self { market })
    }
    
    /// Create a binary market (yes/no)
    #[staticmethod]
    fn create_binary(name: String, description: String, liquidity: f64) -> PyResult<Self> {
        let market = MarketFactory::create_binary_market(name, description, liquidity)?;
        Ok(Self { market })
    }
    
    /// Create a categorical market
    #[staticmethod]
    fn create_categorical(
        name: String, 
        description: String, 
        outcomes: Vec<String>, 
        liquidity: f64
    ) -> PyResult<Self> {
        let market = MarketFactory::create_categorical_market(name, description, outcomes, liquidity)?;
        Ok(Self { market })
    }
    
    /// Execute a trade
    fn trade(&self, trader_id: String, quantities: Vec<f64>) -> PyResult<f64> {
        let trade_record = self.market.execute_trade(trader_id, &quantities)?;
        Ok(trade_record.cost)
    }
    
    /// Get current market prices
    fn get_prices(&self) -> PyResult<Vec<f64>> {
        Ok(self.market.get_prices()?)
    }
    
    /// Get price for specific outcome
    fn get_price(&self, outcome: usize) -> PyResult<f64> {
        Ok(self.market.get_price(outcome)?)
    }
    
    /// Calculate cost of potential trade without executing
    fn calculate_cost(&self, quantities: Vec<f64>) -> PyResult<f64> {
        Ok(self.market.calculate_trade_cost(&quantities)?)
    }
    
    /// Get market name
    fn get_name(&self) -> String {
        self.market.get_metadata().name.clone()
    }
    
    /// Get market outcomes
    fn get_outcomes(&self) -> Vec<String> {
        self.market.get_metadata().outcomes.clone()
    }
    
    /// Check if market is closed
    fn is_closed(&self) -> bool {
        self.market.is_closed()
    }
    
    /// Reset market to initial state
    fn reset(&self) -> PyResult<()> {
        Ok(self.market.reset()?)
    }
}

/// Standalone function to calculate LMSR price
#[pyfunction]
pub fn py_calculate_price(quantities: Vec<f64>, outcome: usize, liquidity: f64) -> PyResult<f64> {
    let calculator = LMSRCalculator::new(quantities.len(), liquidity)?;
    Ok(calculator.marginal_price(&quantities, outcome)?)
}

/// Standalone function to calculate LMSR cost
#[pyfunction]
pub fn py_calculate_cost(current_quantities: Vec<f64>, buy_amounts: Vec<f64>, liquidity: f64) -> PyResult<f64> {
    let calculator = LMSRCalculator::new(current_quantities.len(), liquidity)?;
    Ok(calculator.calculate_buy_cost(&current_quantities, &buy_amounts)?)
}

/// Simple market simulation class
#[pyclass(name = "MarketSimulation")]
pub struct PyMarketSimulation {
    markets: Vec<PyLMSRMarket>,
    traders: HashMap<String, f64>, // trader_id -> balance
}

#[pymethods]
impl PyMarketSimulation {
    #[new]
    fn new() -> Self {
        Self {
            markets: Vec::new(),
            traders: HashMap::new(),
        }
    }
    
    /// Add a new market to the simulation
    fn add_market(&mut self, market: PyLMSRMarket) -> usize {
        self.markets.push(market);
        self.markets.len() - 1
    }
    
    /// Add a trader with initial balance
    fn add_trader(&mut self, trader_id: String, initial_balance: f64) {
        self.traders.insert(trader_id, initial_balance);
    }
    
    /// Execute trade with balance checking
    fn execute_trade(
        &mut self, 
        market_id: usize, 
        trader_id: String, 
        quantities: Vec<f64>
    ) -> PyResult<f64> {
        if market_id >= self.markets.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Invalid market ID"));
        }
        
        let cost = self.markets[market_id].calculate_cost(quantities.clone())?;
        
        // Check trader balance
        let balance = self.traders.get(&trader_id).unwrap_or(&0.0);
        if *balance < cost {
            return Err(pyo3::exceptions::PyValueError::new_err("Insufficient balance"));
        }
        
        // Execute trade
        let actual_cost = self.markets[market_id].trade(trader_id.clone(), quantities)?;
        
        // Update balance
        if let Some(trader_balance) = self.traders.get_mut(&trader_id) {
            *trader_balance -= actual_cost;
        }
        
        Ok(actual_cost)
    }
    
    /// Get trader balance
    fn get_balance(&self, trader_id: String) -> f64 {
        *self.traders.get(&trader_id).unwrap_or(&0.0)
    }
    
    /// Get all market prices
    fn get_all_prices(&self) -> PyResult<Vec<Vec<f64>>> {
        let mut all_prices = Vec::new();
        for market in &self.markets {
            all_prices.push(market.get_prices()?);
        }
        Ok(all_prices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_market_creation() {
        let market = PyLMSRMarket::new(2, 100.0).unwrap();
        let prices = market.get_prices().unwrap();
        assert_eq!(prices.len(), 2);
    }

    #[test]
    fn test_python_trading() {
        let market = PyLMSRMarket::new(2, 100.0).unwrap();
        let cost = market.trade("trader1".to_string(), vec![10.0, 0.0]).unwrap();
        assert!(cost > 0.0);
    }
}