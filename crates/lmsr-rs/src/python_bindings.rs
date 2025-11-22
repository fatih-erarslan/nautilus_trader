//! PyO3 Python bindings for LMSR functionality
//! 
//! This module provides a Python interface to the high-performance Rust LMSR
//! implementation, enabling seamless integration with Python trading systems.

use crate::errors::Result;
use crate::market::{Market, MarketFactory, MarketMetadata, Position, MarketState, PositionManager};
use crate::lmsr::{LMSRCalculator, MarketStatistics};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::sync::Arc;

/// Python wrapper for Market
#[pyclass(name = "LMSRMarket")]
pub struct PyLMSRMarket {
    market: Market,
    position_manager: Arc<PositionManager>,
}

#[pymethods]
impl PyLMSRMarket {
    /// Create a new LMSR market
    #[new]
    fn new(num_outcomes: usize, liquidity_parameter: f64) -> PyResult<Self> {
        let market = Market::new(num_outcomes, liquidity_parameter)?;
        let position_manager = Arc::new(PositionManager::new());
        
        Ok(Self {
            market,
            position_manager,
        })
    }
    
    /// Create a binary market (yes/no)
    #[staticmethod]
    fn create_binary(name: String, description: String, liquidity: f64) -> PyResult<Self> {
        let market = MarketFactory::create_binary_market(name, description, liquidity)?;
        let position_manager = Arc::new(PositionManager::new());
        
        Ok(Self {
            market,
            position_manager,
        })
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
        let position_manager = Arc::new(PositionManager::new());
        
        Ok(Self {
            market,
            position_manager,
        })
    }
    
    /// Execute a trade
    fn trade(&mut self, trader_id: String, quantities: Vec<f64>) -> PyResult<f64> {
        let trade_record = self.market.execute_trade(trader_id.clone(), &quantities)?;
        
        // Update position
        self.position_manager.update_position(
            trader_id,
            &quantities,
            trade_record.cost
        )?;
        
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
    
    /// Get market statistics
    fn get_statistics(&self) -> PyResult<PyMarketStatistics> {
        let stats = self.market.get_statistics()?;
        Ok(PyMarketStatistics { inner: stats })
    }
    
    /// Get market metadata
    fn get_metadata(&self) -> PyResult<PyDict> {
        let py = Python::acquire_gil();
        let py = py.python();
        let dict = PyDict::new(py);
        
        let metadata = self.market.get_metadata();
        dict.set_item("id", &metadata.id)?;
        dict.set_item("name", &metadata.name)?;
        dict.set_item("description", &metadata.description)?;
        dict.set_item("outcomes", &metadata.outcomes)?;
        dict.set_item("category", &metadata.category)?;
        dict.set_item("tags", &metadata.tags)?;
        
        Ok(dict.to_object(py))
    }
    
    /// Get trader position
    fn get_position(&self, trader_id: String) -> PyResult<Option<PyPosition>> {
        if let Some(position) = self.position_manager.get_position(&trader_id) {
            Ok(Some(PyPosition { inner: position }))
        } else {
            Ok(None)
        }
    }
    
    /// Calculate position value at current prices
    fn calculate_position_value(&self, trader_id: String) -> PyResult<f64> {
        let prices = self.market.get_prices()?;
        Ok(self.position_manager.calculate_position_value(&trader_id, &prices)?)
    }
    
    /// Get all positions
    fn get_all_positions(&self) -> PyResult<HashMap<String, PyPosition>> {
        let positions = self.position_manager.get_all_positions();
        let py_positions = positions.into_iter()
            .map(|(k, v)| (k, PyPosition { inner: v }))
            .collect();
        Ok(py_positions)
    }
    
    /// Check if market is closed
    fn is_closed(&self) -> bool {
        self.market.is_closed()
    }
    
    /// Reset market to initial state
    fn reset(&self) -> PyResult<()> {
        Ok(self.market.reset()?)
    }
    
    /// Get implied probabilities
    fn get_probabilities(&self) -> PyResult<Vec<f64>> {
        // Same as prices for LMSR
        self.get_prices()
    }
    
    /// Batch trading for multiple traders
    fn batch_trade(&mut self, trades: Vec<(String, Vec<f64>)>) -> PyResult<Vec<f64>> {
        let mut costs = Vec::new();
        
        for (trader_id, quantities) in trades {
            let cost = self.trade(trader_id, quantities)?;
            costs.push(cost);
        }
        
        Ok(costs)
    }
    
    /// Get market depth (liquidity at different price levels)
    fn get_market_depth(&self, price_levels: Vec<f64>) -> PyResult<Vec<(f64, f64)>> {
        let mut depth = Vec::new();
        let current_prices = self.get_prices()?;
        
        for target_price in price_levels {
            // Simplified market depth calculation
            // In reality, this would involve more complex calculations
            let liquidity = 100.0 / (target_price - current_prices[0]).abs().max(0.01);
            depth.push((target_price, liquidity));
        }
        
        Ok(depth)
    }
}

/// Python wrapper for MarketStatistics
#[pyclass(name = "MarketStatistics")]
pub struct PyMarketStatistics {
    inner: MarketStatistics,
}

#[pymethods]
impl PyMarketStatistics {
    #[getter]
    fn current_prices(&self) -> Vec<f64> {
        self.inner.current_prices.clone()
    }
    
    #[getter]
    fn total_cost(&self) -> f64 {
        self.inner.total_cost
    }
    
    #[getter]
    fn total_volume(&self) -> f64 {
        self.inner.total_volume
    }
    
    #[getter]
    fn trade_count(&self) -> u64 {
        self.inner.trade_count
    }
    
    #[getter]
    fn liquidity_parameter(&self) -> f64 {
        self.inner.liquidity_parameter
    }
    
    #[getter]
    fn quantities(&self) -> Vec<f64> {
        self.inner.quantities.clone()
    }
    
    fn __repr__(&self) -> String {
        format!(
            "MarketStatistics(prices={:?}, volume={:.2}, trades={})",
            self.inner.current_prices,
            self.inner.total_volume,
            self.inner.trade_count
        )
    }
}

/// Python wrapper for Position
#[pyclass(name = "Position")]
pub struct PyPosition {
    inner: Position,
}

#[pymethods]
impl PyPosition {
    #[getter]
    fn trader_id(&self) -> String {
        self.inner.trader_id.clone()
    }
    
    #[getter]
    fn quantities(&self) -> Vec<f64> {
        self.inner.quantities.clone()
    }
    
    #[getter]
    fn average_costs(&self) -> Vec<f64> {
        self.inner.average_costs.clone()
    }
    
    #[getter]
    fn total_invested(&self) -> f64 {
        self.inner.total_invested
    }
    
    #[getter]
    fn last_updated(&self) -> u64 {
        self.inner.last_updated
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    fn __repr__(&self) -> String {
        format!(
            "Position(trader={}, quantities={:?}, invested={:.2})",
            self.inner.trader_id,
            self.inner.quantities,
            self.inner.total_invested
        )
    }
}

/// Python wrapper for MarketState
#[pyclass(name = "MarketState")]
pub struct PyMarketState {
    inner: MarketState,
}

#[pymethods]
impl PyMarketState {
    #[getter]
    fn statistics(&self) -> PyMarketStatistics {
        PyMarketStatistics {
            inner: self.inner.statistics.clone()
        }
    }
    
    #[getter]
    fn timestamp(&self) -> u64 {
        self.inner.timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    fn __repr__(&self) -> String {
        format!(
            "MarketState(trades={}, volume={:.2})",
            self.inner.statistics.trade_count,
            self.inner.statistics.total_volume
        )
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

/// High-level market simulation class
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
    
    /// Run random trading simulation
    fn run_random_simulation(&mut self, num_trades: usize) -> PyResult<Vec<f64>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut total_costs = Vec::new();
        
        for _ in 0..num_trades {
            if self.markets.is_empty() || self.traders.is_empty() {
                break;
            }
            
            let market_id = rng.gen_range(0..self.markets.len());
            let trader_keys: Vec<_> = self.traders.keys().cloned().collect();
            let trader_id = trader_keys[rng.gen_range(0..trader_keys.len())].clone();
            
            let market = &self.markets[market_id];
            let prices = market.get_prices()?;
            let num_outcomes = prices.len();
            
            // Generate random trade quantities
            let mut quantities = vec![0.0; num_outcomes];
            let outcome = rng.gen_range(0..num_outcomes);
            quantities[outcome] = rng.gen_range(1.0..10.0);
            
            match self.execute_trade(market_id, trader_id, quantities) {
                Ok(cost) => total_costs.push(cost),
                Err(_) => {} // Ignore failed trades (insufficient balance, etc.)
            }
        }
        
        Ok(total_costs)
    }
}

/// Performance benchmarking utilities
#[pyclass(name = "LMSRBenchmark")]
pub struct PyLMSRBenchmark;

#[pymethods]
impl PyLMSRBenchmark {
    #[new]
    fn new() -> Self {
        Self
    }
    
    /// Benchmark price calculations
    #[staticmethod]
    fn benchmark_prices(
        num_outcomes: usize, 
        liquidity: f64, 
        num_iterations: usize
    ) -> PyResult<f64> {
        use std::time::Instant;
        
        let calculator = LMSRCalculator::new(num_outcomes, liquidity)?;
        let quantities = vec![10.0; num_outcomes];
        
        let start = Instant::now();
        
        for _ in 0..num_iterations {
            let _ = calculator.all_marginal_prices(&quantities)?;
        }
        
        let duration = start.elapsed();
        Ok(duration.as_secs_f64())
    }
    
    /// Benchmark trade cost calculations
    #[staticmethod]
    fn benchmark_costs(
        num_outcomes: usize, 
        liquidity: f64, 
        num_iterations: usize
    ) -> PyResult<f64> {
        use std::time::Instant;
        
        let calculator = LMSRCalculator::new(num_outcomes, liquidity)?;
        let current_quantities = vec![10.0; num_outcomes];
        let buy_amounts = vec![1.0; num_outcomes];
        
        let start = Instant::now();
        
        for _ in 0..num_iterations {
            let _ = calculator.calculate_buy_cost(&current_quantities, &buy_amounts)?;
        }
        
        let duration = start.elapsed();
        Ok(duration.as_secs_f64())
    }
    
    /// Benchmark full market operations
    #[staticmethod]
    fn benchmark_market_operations(
        num_outcomes: usize, 
        liquidity: f64, 
        num_trades: usize
    ) -> PyResult<f64> {
        use std::time::Instant;
        
        let mut market = PyLMSRMarket::new(num_outcomes, liquidity)?;
        let quantities = vec![1.0; num_outcomes];
        
        let start = Instant::now();
        
        for i in 0..num_trades {
            let trader_id = format!("trader_{}", i);
            let _ = market.trade(trader_id, quantities.clone())?;
        }
        
        let duration = start.elapsed();
        Ok(duration.as_secs_f64())
    }
}

#[cfg(test)]
mod python_tests {
    use super::*;

    #[test]
    fn test_python_market_creation() {
        let market = PyLMSRMarket::new(2, 100.0).unwrap();
        let prices = market.get_prices().unwrap();
        assert_eq!(prices.len(), 2);
    }

    #[test]
    fn test_python_trading() {
        let mut market = PyLMSRMarket::new(2, 100.0).unwrap();
        let cost = market.trade("trader1".to_string(), vec![10.0, 0.0]).unwrap();
        assert!(cost > 0.0);
        
        let position = market.get_position("trader1".to_string()).unwrap().unwrap();
        assert_eq!(position.quantities(), vec![10.0, 0.0]);
    }

    #[test]
    fn test_market_simulation() {
        let mut sim = PyMarketSimulation::new();
        let market = PyLMSRMarket::new(2, 100.0).unwrap();
        
        sim.add_market(market);
        sim.add_trader("trader1".to_string(), 1000.0);
        
        let cost = sim.execute_trade(0, "trader1".to_string(), vec![10.0, 0.0]).unwrap();
        assert!(cost > 0.0);
        assert!(sim.get_balance("trader1".to_string()) < 1000.0);
    }
}