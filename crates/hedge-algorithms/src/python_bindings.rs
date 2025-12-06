//! Python Bindings for Quantum Hedge Algorithms
//!
//! This module provides comprehensive Python bindings for all quantum hedge
//! algorithm functionality, enabling seamless integration with Python trading systems.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict, PyTuple};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use std::collections::HashMap;
use crate::{
    QuantumHedgeManager, QuantumHedgeConfig, QuantumHedgeMode,
    ExpertSpecialization, QuantumHedgeMetrics, QuantumHedgeError,
    HedgeResult, ExpertPrediction, QuantumState, MarketData,
};

/// Python wrapper for QuantumHedgeManager
#[pyclass(name = "QuantumHedgeManager")]
pub struct PyQuantumHedgeManager {
    inner: QuantumHedgeManager,
}

#[pymethods]
impl PyQuantumHedgeManager {
    /// Create new quantum hedge manager
    #[new]
    pub fn new(config: Option<PyDict>) -> PyResult<Self> {
        let config = if let Some(config_dict) = config {
            Self::parse_hedge_config(config_dict)?
        } else {
            QuantumHedgeConfig::default()
        };

        let inner = QuantumHedgeManager::new(config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create hedge manager: {}", e)))?;
        Ok(Self { inner })
    }

    /// Initialize quantum hedge system
    pub fn initialize(&mut self) -> PyResult<()> {
        self.inner.initialize()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize: {}", e)))?;
        Ok(())
    }

    /// Add expert to hedge ensemble
    pub fn add_expert(&mut self, specialization: &str, weight: f64) -> PyResult<()> {
        let spec = Self::parse_specialization(specialization)?;
        self.inner.add_expert(spec, weight)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to add expert: {}", e)))?;
        Ok(())
    }

    /// Update hedge weights based on performance
    pub fn update_weights(&mut self, performance_data: PyReadonlyArray1<f64>) -> PyResult<()> {
        let performance = performance_data.as_slice()?;
        self.inner.update_weights(performance.to_vec())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to update weights: {}", e)))?;
        Ok(())
    }

    /// Get expert predictions
    pub fn get_expert_predictions(&self, market_data: PyDict) -> PyResult<PyDict> {
        let market_data = Self::parse_market_data(market_data)?;
        let predictions = self.inner.get_expert_predictions(&market_data)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get predictions: {}", e)))?;

        let py_dict = PyDict::new(Python::acquire_gil().python());
        for prediction in predictions {
            let expert_name = format!("{:?}", prediction.expert);
            let prediction_data = PyDict::new(Python::acquire_gil().python());
            
            prediction_data.set_item("confidence", prediction.confidence)?;
            prediction_data.set_item("direction", prediction.direction)?;
            prediction_data.set_item("magnitude", prediction.magnitude)?;
            prediction_data.set_item("risk_level", prediction.risk_level)?;
            prediction_data.set_item("timestamp", prediction.timestamp.timestamp())?;
            
            py_dict.set_item(expert_name, prediction_data)?;
        }

        Ok(py_dict)
    }

    /// Execute quantum multiplicative weights update
    pub fn quantum_multiplicative_weights_update(
        &mut self,
        losses: PyReadonlyArray1<f64>,
        learning_rate: f64
    ) -> PyResult<PyReadonlyArray1<f64>> {
        let losses = losses.as_slice()?.to_vec();
        let new_weights = self.inner.quantum_multiplicative_weights_update(losses, learning_rate)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to update weights: {}", e)))?;

        Python::with_gil(|py| {
            Ok(PyArray1::from_vec(py, new_weights).readonly())
        })
    }

    /// Calculate quantum hedge ratio
    pub fn calculate_quantum_hedge_ratio(
        &self,
        portfolio_value: f64,
        risk_metrics: PyDict
    ) -> PyResult<f64> {
        let risk_data = Self::parse_risk_metrics(risk_metrics)?;
        let ratio = self.inner.calculate_quantum_hedge_ratio(portfolio_value, &risk_data)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate hedge ratio: {}", e)))?;
        Ok(ratio)
    }

    /// Execute quantum arbitrage detection
    pub fn detect_quantum_arbitrage(&self, price_data: PyReadonlyArray2<f64>) -> PyResult<PyList> {
        let price_matrix = Self::numpy_to_matrix(price_data)?;
        let opportunities = self.inner.detect_quantum_arbitrage(&price_matrix)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to detect arbitrage: {}", e)))?;

        Python::with_gil(|py| {
            let py_list = PyList::empty(py);
            for opportunity in opportunities {
                let opp_dict = PyDict::new(py);
                opp_dict.set_item("asset_pair", format!("{:?}", opportunity.asset_pair))?;
                opp_dict.set_item("expected_return", opportunity.expected_return)?;
                opp_dict.set_item("risk_level", opportunity.risk_level)?;
                opp_dict.set_item("confidence", opportunity.confidence)?;
                py_list.append(opp_dict)?;
            }
            Ok(py_list)
        })
    }

    /// Optimize hedge portfolio allocation
    pub fn optimize_hedge_allocation(
        &self,
        returns: PyReadonlyArray2<f64>,
        risk_aversion: f64
    ) -> PyResult<PyReadonlyArray1<f64>> {
        let returns_matrix = Self::numpy_to_matrix(returns)?;
        let allocation = self.inner.optimize_hedge_allocation(&returns_matrix, risk_aversion)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to optimize allocation: {}", e)))?;

        Python::with_gil(|py| {
            Ok(PyArray1::from_vec(py, allocation).readonly())
        })
    }

    /// Execute quantum risk parity strategy
    pub fn quantum_risk_parity(
        &self,
        covariance_matrix: PyReadonlyArray2<f64>
    ) -> PyResult<PyReadonlyArray1<f64>> {
        let cov_matrix = Self::numpy_to_matrix(covariance_matrix)?;
        let weights = self.inner.quantum_risk_parity(&cov_matrix)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate risk parity: {}", e)))?;

        Python::with_gil(|py| {
            Ok(PyArray1::from_vec(py, weights).readonly())
        })
    }

    /// Calculate dynamic hedging strategy
    pub fn dynamic_hedging_strategy(
        &mut self,
        market_conditions: PyDict,
        portfolio_state: PyDict
    ) -> PyResult<PyDict> {
        let market_data = Self::parse_market_data(market_conditions)?;
        let portfolio_data = Self::parse_portfolio_state(portfolio_state)?;
        
        let strategy = self.inner.dynamic_hedging_strategy(&market_data, &portfolio_data)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate dynamic hedging: {}", e)))?;

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("hedge_ratio", strategy.hedge_ratio)?;
            result_dict.set_item("rebalance_frequency", strategy.rebalance_frequency)?;
            result_dict.set_item("expected_return", strategy.expected_return)?;
            result_dict.set_item("risk_reduction", strategy.risk_reduction)?;
            result_dict.set_item("cost_estimate", strategy.cost_estimate)?;
            Ok(result_dict)
        })
    }

    /// Get quantum hedge performance metrics
    pub fn get_performance_metrics(&self) -> PyResult<PyDict> {
        let metrics = self.inner.get_performance_metrics()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get metrics: {}", e)))?;

        Python::with_gil(|py| {
            let py_dict = PyDict::new(py);
            py_dict.set_item("total_return", metrics.total_return)?;
            py_dict.set_item("sharpe_ratio", metrics.sharpe_ratio)?;
            py_dict.set_item("max_drawdown", metrics.max_drawdown)?;
            py_dict.set_item("volatility", metrics.volatility)?;
            py_dict.set_item("alpha", metrics.alpha)?;
            py_dict.set_item("beta", metrics.beta)?;
            py_dict.set_item("information_ratio", metrics.information_ratio)?;
            py_dict.set_item("calmar_ratio", metrics.calmar_ratio)?;
            py_dict.set_item("sortino_ratio", metrics.sortino_ratio)?;
            py_dict.set_item("quantum_advantage", metrics.quantum_advantage)?;
            Ok(py_dict)
        })
    }

    /// Execute quantum portfolio insurance
    pub fn quantum_portfolio_insurance(
        &self,
        portfolio_value: f64,
        floor_value: f64,
        time_horizon: f64
    ) -> PyResult<PyDict> {
        let insurance = self.inner.quantum_portfolio_insurance(portfolio_value, floor_value, time_horizon)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate portfolio insurance: {}", e)))?;

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("hedge_amount", insurance.hedge_amount)?;
            result_dict.set_item("delta", insurance.delta)?;
            result_dict.set_item("gamma", insurance.gamma)?;
            result_dict.set_item("theta", insurance.theta)?;
            result_dict.set_item("vega", insurance.vega)?;
            result_dict.set_item("protection_cost", insurance.protection_cost)?;
            Ok(result_dict)
        })
    }

    /// Simulate quantum hedge scenarios
    pub fn simulate_hedge_scenarios(
        &self,
        num_scenarios: usize,
        time_horizon: f64,
        market_params: PyDict
    ) -> PyResult<PyDict> {
        let params = Self::parse_market_params(market_params)?;
        let scenarios = self.inner.simulate_hedge_scenarios(num_scenarios, time_horizon, &params)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to simulate scenarios: {}", e)))?;

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            
            let returns = PyArray1::from_vec(py, scenarios.returns);
            let risks = PyArray1::from_vec(py, scenarios.risks);
            let hedge_effectiveness = PyArray1::from_vec(py, scenarios.hedge_effectiveness);
            
            result_dict.set_item("returns", returns)?;
            result_dict.set_item("risks", risks)?;
            result_dict.set_item("hedge_effectiveness", hedge_effectiveness)?;
            result_dict.set_item("mean_return", scenarios.mean_return)?;
            result_dict.set_item("std_return", scenarios.std_return)?;
            result_dict.set_item("var_95", scenarios.var_95)?;
            result_dict.set_item("cvar_95", scenarios.cvar_95)?;
            
            Ok(result_dict)
        })
    }

    /// Get current expert weights
    pub fn get_expert_weights(&self) -> PyResult<PyDict> {
        let weights = self.inner.get_expert_weights()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get expert weights: {}", e)))?;

        Python::with_gil(|py| {
            let py_dict = PyDict::new(py);
            for (expert, weight) in weights {
                let expert_name = format!("{:?}", expert);
                py_dict.set_item(expert_name, weight)?;
            }
            Ok(py_dict)
        })
    }

    /// Reset hedge manager state
    pub fn reset(&mut self) -> PyResult<()> {
        self.inner.reset()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to reset: {}", e)))?;
        Ok(())
    }

    /// Check if quantum advantage is achieved
    pub fn has_quantum_advantage(&self) -> bool {
        self.inner.has_quantum_advantage()
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!("QuantumHedgeManager(experts={}, quantum_enabled={})", 
                self.inner.num_experts(), 
                self.inner.is_quantum_enabled())
    }

    /// Representation
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

impl PyQuantumHedgeManager {
    fn parse_hedge_config(config_dict: &PyDict) -> PyResult<QuantumHedgeConfig> {
        let mut config = QuantumHedgeConfig::default();

        if let Some(processing_mode) = config_dict.get_item("processing_mode") {
            let mode_str = processing_mode.extract::<String>()?;
            config.processing_mode = match mode_str.as_str() {
                "classical" => QuantumHedgeMode::Classical,
                "quantum" => QuantumHedgeMode::Quantum,
                "hybrid" => QuantumHedgeMode::Hybrid,
                "auto" => QuantumHedgeMode::Auto,
                _ => return Err(PyValueError::new_err(format!("Unknown processing mode: {}", mode_str))),
            };
        }

        if let Some(num_qubits) = config_dict.get_item("num_qubits") {
            config.num_qubits = num_qubits.extract::<usize>()?;
        }

        if let Some(circuit_depth) = config_dict.get_item("circuit_depth") {
            config.circuit_depth = circuit_depth.extract::<usize>()?;
        }

        if let Some(learning_rate) = config_dict.get_item("learning_rate") {
            config.learning_rate = learning_rate.extract::<f64>()?;
        }

        if let Some(regularization) = config_dict.get_item("regularization") {
            config.regularization = regularization.extract::<f64>()?;
        }

        if let Some(rebalance_threshold) = config_dict.get_item("rebalance_threshold") {
            config.rebalance_threshold = rebalance_threshold.extract::<f64>()?;
        }

        if let Some(max_experts) = config_dict.get_item("max_experts") {
            config.max_experts = max_experts.extract::<usize>()?;
        }

        Ok(config)
    }

    fn parse_specialization(spec_str: &str) -> PyResult<ExpertSpecialization> {
        match spec_str {
            "trend_following" => Ok(ExpertSpecialization::TrendFollowing),
            "mean_reversion" => Ok(ExpertSpecialization::MeanReversion),
            "volatility_trading" => Ok(ExpertSpecialization::VolatilityTrading),
            "momentum" => Ok(ExpertSpecialization::Momentum),
            "sentiment_analysis" => Ok(ExpertSpecialization::SentimentAnalysis),
            "liquidity_provision" => Ok(ExpertSpecialization::LiquidityProvision),
            "correlation_trading" => Ok(ExpertSpecialization::CorrelationTrading),
            "cycle_analysis" => Ok(ExpertSpecialization::CycleAnalysis),
            "anomaly_detection" => Ok(ExpertSpecialization::AnomalyDetection),
            "risk_management" => Ok(ExpertSpecialization::RiskManagement),
            "options_trading" => Ok(ExpertSpecialization::OptionsTrading),
            "pairs_trading" => Ok(ExpertSpecialization::PairsTrading),
            "arbitrage_expert" => Ok(ExpertSpecialization::ArbitrageExpert),
            "high_frequency_trading" => Ok(ExpertSpecialization::HighFrequencyTrading),
            "macro_economic" => Ok(ExpertSpecialization::MacroEconomic),
            "technical_analysis" => Ok(ExpertSpecialization::TechnicalAnalysis),
            _ => Err(PyValueError::new_err(format!("Unknown specialization: {}", spec_str))),
        }
    }

    fn parse_market_data(market_dict: PyDict) -> PyResult<MarketData> {
        let mut market_data = MarketData::default();

        if let Some(prices) = market_dict.get_item("prices") {
            market_data.prices = prices.extract::<Vec<f64>>()?;
        }

        if let Some(volumes) = market_dict.get_item("volumes") {
            market_data.volumes = volumes.extract::<Vec<f64>>()?;
        }

        if let Some(volatility) = market_dict.get_item("volatility") {
            market_data.volatility = volatility.extract::<f64>()?;
        }

        if let Some(trend) = market_dict.get_item("trend") {
            market_data.trend = trend.extract::<f64>()?;
        }

        if let Some(momentum) = market_dict.get_item("momentum") {
            market_data.momentum = momentum.extract::<f64>()?;
        }

        Ok(market_data)
    }

    fn parse_risk_metrics(risk_dict: PyDict) -> PyResult<HashMap<String, f64>> {
        let mut risk_data = HashMap::new();

        for (key, value) in risk_dict.iter() {
            let key_str = key.extract::<String>()?;
            let value_f64 = value.extract::<f64>()?;
            risk_data.insert(key_str, value_f64);
        }

        Ok(risk_data)
    }

    fn parse_portfolio_state(portfolio_dict: PyDict) -> PyResult<HashMap<String, f64>> {
        let mut portfolio_data = HashMap::new();

        for (key, value) in portfolio_dict.iter() {
            let key_str = key.extract::<String>()?;
            let value_f64 = value.extract::<f64>()?;
            portfolio_data.insert(key_str, value_f64);
        }

        Ok(portfolio_data)
    }

    fn parse_market_params(params_dict: PyDict) -> PyResult<HashMap<String, f64>> {
        let mut params = HashMap::new();

        for (key, value) in params_dict.iter() {
            let key_str = key.extract::<String>()?;
            let value_f64 = value.extract::<f64>()?;
            params.insert(key_str, value_f64);
        }

        Ok(params)
    }

    fn numpy_to_matrix(array: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<f64>>> {
        let array = array.as_array();
        let mut matrix = Vec::new();
        
        for row in array.rows() {
            matrix.push(row.to_vec());
        }
        
        Ok(matrix)
    }
}

/// Quantum hedge utility functions
#[pyfunction]
pub fn create_quantum_hedge_expert(specialization: &str, initial_weight: f64) -> PyResult<PyDict> {
    let spec = PyQuantumHedgeManager::parse_specialization(specialization)?;
    
    Python::with_gil(|py| {
        let expert_dict = PyDict::new(py);
        expert_dict.set_item("specialization", specialization)?;
        expert_dict.set_item("weight", initial_weight)?;
        expert_dict.set_item("performance", 0.0)?;
        expert_dict.set_item("trades", 0)?;
        expert_dict.set_item("last_update", chrono::Utc::now().timestamp())?;
        Ok(expert_dict)
    })
}

/// Calculate optimal hedge ratio using Black-Scholes
#[pyfunction]
pub fn black_scholes_hedge_ratio(
    spot_price: f64,
    strike_price: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    volatility: f64
) -> PyResult<f64> {
    let ratio = crate::black_scholes_hedge_ratio(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate Black-Scholes hedge ratio: {}", e)))?;
    Ok(ratio)
}

/// Calculate Value at Risk using quantum methods
#[pyfunction]
pub fn quantum_value_at_risk(
    returns: PyReadonlyArray1<f64>,
    confidence_level: f64,
    num_qubits: usize
) -> PyResult<f64> {
    let returns_vec = returns.as_slice()?.to_vec();
    let var = crate::quantum_value_at_risk(&returns_vec, confidence_level, num_qubits)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate quantum VaR: {}", e)))?;
    Ok(var)
}

/// Optimize portfolio using quantum algorithms
#[pyfunction]
pub fn quantum_portfolio_optimization(
    expected_returns: PyReadonlyArray1<f64>,
    covariance_matrix: PyReadonlyArray2<f64>,
    risk_aversion: f64
) -> PyResult<PyReadonlyArray1<f64>> {
    let returns = expected_returns.as_slice()?.to_vec();
    let cov_matrix = PyQuantumHedgeManager::numpy_to_matrix(covariance_matrix)?;
    
    let weights = crate::quantum_portfolio_optimization(&returns, &cov_matrix, risk_aversion)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to optimize portfolio: {}", e)))?;

    Python::with_gil(|py| {
        Ok(PyArray1::from_vec(py, weights).readonly())
    })
}

/// Calculate correlation using quantum algorithms
#[pyfunction]
pub fn quantum_correlation_analysis(
    data1: PyReadonlyArray1<f64>,
    data2: PyReadonlyArray1<f64>
) -> PyResult<f64> {
    let data1_vec = data1.as_slice()?.to_vec();
    let data2_vec = data2.as_slice()?.to_vec();
    
    let correlation = crate::quantum_correlation_analysis(&data1_vec, &data2_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate quantum correlation: {}", e)))?;
    Ok(correlation)
}

/// Get quantum hedge module version
#[pyfunction]
pub fn get_version() -> String {
    crate::VERSION.to_string()
}

/// Initialize quantum hedge module
#[pyfunction]
pub fn initialize_quantum_hedge() -> PyResult<()> {
    crate::initialize()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize quantum hedge module: {}", e)))
}

/// Python module definition
#[pymodule]
fn hedge_algorithms(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyQuantumHedgeManager>()?;
    
    m.add_function(wrap_pyfunction!(create_quantum_hedge_expert, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_hedge_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(quantum_value_at_risk, m)?)?;
    m.add_function(wrap_pyfunction!(quantum_portfolio_optimization, m)?)?;
    m.add_function(wrap_pyfunction!(quantum_correlation_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(initialize_quantum_hedge, m)?)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_python_quantum_hedge_manager() {
        Python::with_gil(|py| {
            let manager = PyQuantumHedgeManager::new(None).unwrap();
            assert!(manager.__str__().contains("QuantumHedgeManager"));
        });
    }

    #[test]
    fn test_create_quantum_hedge_expert() {
        Python::with_gil(|py| {
            let expert = create_quantum_hedge_expert("trend_following", 0.1).unwrap();
            assert!(expert.contains("specialization").unwrap());
            assert!(expert.contains("weight").unwrap());
        });
    }

    #[test]
    fn test_quantum_utility_functions() {
        Python::with_gil(|py| {
            // Test version function
            let version = get_version();
            assert!(!version.is_empty());
            
            // Test Black-Scholes hedge ratio
            let ratio = black_scholes_hedge_ratio(100.0, 100.0, 0.25, 0.05, 0.2);
            assert!(ratio.is_ok());
        });
    }
}