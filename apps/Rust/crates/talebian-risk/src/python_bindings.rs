//! Python Bindings for Quantum Talebian Risk Management
//!
//! This module provides comprehensive Python bindings for all quantum Talebian
//! risk management functionality, enabling seamless integration with Python trading systems.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict, PyTuple};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use std::collections::HashMap;
use crate::{
    QuantumTalebianRisk, QuantumTalebianConfig, QuantumTalebianMode,
    AntifragilityType, BlackSwanEvent, TailRiskMetrics, AntifragilityMetrics,
    QuantumTalebianMetrics, TalebianError, TalebianResult,
};

/// Python wrapper for QuantumTalebianRisk
#[pyclass(name = "QuantumTalebianRisk")]
pub struct PyQuantumTalebianRisk {
    inner: QuantumTalebianRisk,
}

#[pymethods]
impl PyQuantumTalebianRisk {
    /// Create new quantum Talebian risk manager
    #[new]
    pub fn new(config: Option<PyDict>) -> PyResult<Self> {
        let config = if let Some(config_dict) = config {
            Self::parse_talebian_config(config_dict)?
        } else {
            QuantumTalebianConfig::default()
        };

        let inner = QuantumTalebianRisk::new(config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Talebian risk manager: {}", e)))?;
        Ok(Self { inner })
    }

    /// Initialize quantum Talebian risk system
    pub fn initialize(&mut self) -> PyResult<()> {
        self.inner.initialize()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize: {}", e)))?;
        Ok(())
    }

    /// Calculate antifragility score
    pub fn calculate_antifragility(
        &self,
        returns: PyReadonlyArray1<f64>,
        stress_events: PyList,
        antifragility_type: &str
    ) -> PyResult<f64> {
        let returns_vec = returns.as_slice()?.to_vec();
        let stress_events_vec = Self::parse_stress_events(stress_events)?;
        let af_type = Self::parse_antifragility_type(antifragility_type)?;

        let score = self.inner.calculate_antifragility(&returns_vec, &stress_events_vec, af_type)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate antifragility: {}", e)))?;
        Ok(score)
    }

    /// Detect black swan events
    pub fn detect_black_swan_events(
        &self,
        data: PyReadonlyArray1<f64>,
        threshold: f64
    ) -> PyResult<PyList> {
        let data_vec = data.as_slice()?.to_vec();
        let events = self.inner.detect_black_swan_events(&data_vec, threshold)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to detect black swan events: {}", e)))?;

        Python::with_gil(|py| {
            let py_list = PyList::empty(py);
            for event in events {
                let event_dict = PyDict::new(py);
                event_dict.set_item("magnitude", event.magnitude)?;
                event_dict.set_item("probability", event.probability)?;
                event_dict.set_item("impact", event.impact)?;
                event_dict.set_item("timestamp", event.timestamp.timestamp())?;
                event_dict.set_item("event_type", format!("{:?}", event.event_type))?;
                event_dict.set_item("description", &event.description)?;
                py_list.append(event_dict)?;
            }
            Ok(py_list)
        })
    }

    /// Calculate tail risk metrics
    pub fn calculate_tail_risk(
        &self,
        returns: PyReadonlyArray1<f64>,
        confidence_levels: PyReadonlyArray1<f64>
    ) -> PyResult<PyDict> {
        let returns_vec = returns.as_slice()?.to_vec();
        let confidence_vec = confidence_levels.as_slice()?.to_vec();

        let tail_risk = self.inner.calculate_tail_risk(&returns_vec, &confidence_vec)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate tail risk: {}", e)))?;

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("var_95", tail_risk.var_95)?;
            result_dict.set_item("var_99", tail_risk.var_99)?;
            result_dict.set_item("var_999", tail_risk.var_999)?;
            result_dict.set_item("cvar_95", tail_risk.cvar_95)?;
            result_dict.set_item("cvar_99", tail_risk.cvar_99)?;
            result_dict.set_item("cvar_999", tail_risk.cvar_999)?;
            result_dict.set_item("expected_shortfall", tail_risk.expected_shortfall)?;
            result_dict.set_item("maximum_loss", tail_risk.maximum_loss)?;
            result_dict.set_item("tail_expectation", tail_risk.tail_expectation)?;
            Ok(result_dict)
        })
    }

    /// Optimize convexity
    pub fn optimize_convexity(
        &self,
        portfolio_data: PyDict,
        market_conditions: PyDict
    ) -> PyResult<PyDict> {
        let portfolio = Self::parse_portfolio_data(portfolio_data)?;
        let market = Self::parse_market_conditions(market_conditions)?;

        let optimization = self.inner.optimize_convexity(&portfolio, &market)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to optimize convexity: {}", e)))?;

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("convexity_score", optimization.convexity_score)?;
            result_dict.set_item("optimal_allocation", PyArray1::from_vec(py, optimization.optimal_allocation))?;
            result_dict.set_item("expected_payoff", optimization.expected_payoff)?;
            result_dict.set_item("downside_protection", optimization.downside_protection)?;
            result_dict.set_item("upside_capture", optimization.upside_capture)?;
            result_dict.set_item("gamma_exposure", optimization.gamma_exposure)?;
            Ok(result_dict)
        })
    }

    /// Calculate barbell strategy allocation
    pub fn calculate_barbell_strategy(
        &self,
        safe_asset_return: f64,
        risky_asset_data: PyDict,
        risk_budget: f64
    ) -> PyResult<PyDict> {
        let risky_data = Self::parse_risky_asset_data(risky_asset_data)?;

        let barbell = self.inner.calculate_barbell_strategy(safe_asset_return, &risky_data, risk_budget)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate barbell strategy: {}", e)))?;

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("safe_allocation", barbell.safe_allocation)?;
            result_dict.set_item("risky_allocation", barbell.risky_allocation)?;
            result_dict.set_item("expected_return", barbell.expected_return)?;
            result_dict.set_item("maximum_loss", barbell.maximum_loss)?;
            result_dict.set_item("antifragility_score", barbell.antifragility_score)?;
            result_dict.set_item("asymmetry_ratio", barbell.asymmetry_ratio)?;
            Ok(result_dict)
        })
    }

    /// Perform quantum stress testing
    pub fn quantum_stress_test(
        &self,
        portfolio_data: PyDict,
        stress_scenarios: PyList,
        num_simulations: usize
    ) -> PyResult<PyDict> {
        let portfolio = Self::parse_portfolio_data(portfolio_data)?;
        let scenarios = Self::parse_stress_scenarios(stress_scenarios)?;

        let stress_results = self.inner.quantum_stress_test(&portfolio, &scenarios, num_simulations)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to perform stress test: {}", e)))?;

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("worst_case_loss", stress_results.worst_case_loss)?;
            result_dict.set_item("average_loss", stress_results.average_loss)?;
            result_dict.set_item("probability_of_ruin", stress_results.probability_of_ruin)?;
            result_dict.set_item("recovery_time", stress_results.recovery_time)?;
            result_dict.set_item("stress_correlation", stress_results.stress_correlation)?;
            result_dict.set_item("fragility_score", stress_results.fragility_score)?;
            result_dict.set_item("scenario_results", PyArray2::from_vec2(py, &stress_results.scenario_results)?)?;
            Ok(result_dict)
        })
    }

    /// Calculate fat tail protection
    pub fn calculate_fat_tail_protection(
        &self,
        returns: PyReadonlyArray1<f64>,
        protection_level: f64
    ) -> PyResult<PyDict> {
        let returns_vec = returns.as_slice()?.to_vec();

        let protection = self.inner.calculate_fat_tail_protection(&returns_vec, protection_level)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate fat tail protection: {}", e)))?;

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("hedge_ratio", protection.hedge_ratio)?;
            result_dict.set_item("protection_cost", protection.protection_cost)?;
            result_dict.set_item("expected_benefit", protection.expected_benefit)?;
            result_dict.set_item("tail_hedge_effectiveness", protection.tail_hedge_effectiveness)?;
            result_dict.set_item("skewness_adjustment", protection.skewness_adjustment)?;
            result_dict.set_item("kurtosis_adjustment", protection.kurtosis_adjustment)?;
            Ok(result_dict)
        })
    }

    /// Measure via negativa impact
    pub fn measure_via_negativa(
        &self,
        portfolio_data: PyDict,
        elimination_candidates: PyList
    ) -> PyResult<PyDict> {
        let portfolio = Self::parse_portfolio_data(portfolio_data)?;
        let candidates = Self::parse_elimination_candidates(elimination_candidates)?;

        let via_negativa = self.inner.measure_via_negativa(&portfolio, &candidates)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to measure via negativa: {}", e)))?;

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("risk_reduction", via_negativa.risk_reduction)?;
            result_dict.set_item("complexity_reduction", via_negativa.complexity_reduction)?;
            result_dict.set_item("cost_savings", via_negativa.cost_savings)?;
            result_dict.set_item("robustness_improvement", via_negativa.robustness_improvement)?;
            result_dict.set_item("elimination_priority", PyArray1::from_vec(py, via_negativa.elimination_priority))?;
            result_dict.set_item("net_benefit", via_negativa.net_benefit)?;
            Ok(result_dict)
        })
    }

    /// Calculate lindy effect strength
    pub fn calculate_lindy_effect(
        &self,
        asset_ages: PyReadonlyArray1<f64>,
        performance_data: PyReadonlyArray2<f64>
    ) -> PyResult<PyDict> {
        let ages = asset_ages.as_slice()?.to_vec();
        let performance = Self::numpy_to_matrix(performance_data)?;

        let lindy = self.inner.calculate_lindy_effect(&ages, &performance)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate Lindy effect: {}", e)))?;

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("lindy_strength", lindy.lindy_strength)?;
            result_dict.set_item("survival_probability", PyArray1::from_vec(py, lindy.survival_probability))?;
            result_dict.set_item("age_performance_correlation", lindy.age_performance_correlation)?;
            result_dict.set_item("longevity_premium", lindy.longevity_premium)?;
            result_dict.set_item("fragility_discount", lindy.fragility_discount)?;
            Ok(result_dict)
        })
    }

    /// Optimize skin in the game
    pub fn optimize_skin_in_game(
        &self,
        manager_data: PyDict,
        investor_data: PyDict,
        alignment_requirements: PyDict
    ) -> PyResult<PyDict> {
        let manager = Self::parse_manager_data(manager_data)?;
        let investor = Self::parse_investor_data(investor_data)?;
        let requirements = Self::parse_alignment_requirements(alignment_requirements)?;

        let optimization = self.inner.optimize_skin_in_game(&manager, &investor, &requirements)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to optimize skin in game: {}", e)))?;

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("optimal_manager_stake", optimization.optimal_manager_stake)?;
            result_dict.set_item("alignment_score", optimization.alignment_score)?;
            result_dict.set_item("risk_sharing_ratio", optimization.risk_sharing_ratio)?;
            result_dict.set_item("incentive_effectiveness", optimization.incentive_effectiveness)?;
            result_dict.set_item("moral_hazard_reduction", optimization.moral_hazard_reduction)?;
            Ok(result_dict)
        })
    }

    /// Get comprehensive Talebian metrics
    pub fn get_talebian_metrics(&self) -> PyResult<PyDict> {
        let metrics = self.inner.get_talebian_metrics()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get metrics: {}", e)))?;

        Python::with_gil(|py| {
            let py_dict = PyDict::new(py);
            py_dict.set_item("antifragility_score", metrics.antifragility_score)?;
            py_dict.set_item("black_swan_protection", metrics.black_swan_protection)?;
            py_dict.set_item("tail_risk_exposure", metrics.tail_risk_exposure)?;
            py_dict.set_item("convexity_measure", metrics.convexity_measure)?;
            py_dict.set_item("robustness_index", metrics.robustness_index)?;
            py_dict.set_item("fragility_index", metrics.fragility_index)?;
            py_dict.set_item("via_negativa_score", metrics.via_negativa_score)?;
            py_dict.set_item("lindy_strength", metrics.lindy_strength)?;
            py_dict.set_item("skin_in_game_score", metrics.skin_in_game_score)?;
            py_dict.set_item("quantum_advantage", metrics.quantum_advantage)?;
            Ok(py_dict)
        })
    }

    /// Simulate extreme scenarios
    pub fn simulate_extreme_scenarios(
        &self,
        portfolio_data: PyDict,
        num_scenarios: usize,
        severity_levels: PyReadonlyArray1<f64>
    ) -> PyResult<PyDict> {
        let portfolio = Self::parse_portfolio_data(portfolio_data)?;
        let severities = severity_levels.as_slice()?.to_vec();

        let simulation = self.inner.simulate_extreme_scenarios(&portfolio, num_scenarios, &severities)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to simulate scenarios: {}", e)))?;

        Python::with_gil(|py| {
            let result_dict = PyDict::new(py);
            result_dict.set_item("scenario_outcomes", PyArray2::from_vec2(py, &simulation.scenario_outcomes)?)?;
            result_dict.set_item("probability_weights", PyArray1::from_vec(py, simulation.probability_weights))?;
            result_dict.set_item("expected_losses", PyArray1::from_vec(py, simulation.expected_losses))?;
            result_dict.set_item("recovery_times", PyArray1::from_vec(py, simulation.recovery_times))?;
            result_dict.set_item("antifragile_benefit", simulation.antifragile_benefit)?;
            result_dict.set_item("fragile_damage", simulation.fragile_damage)?;
            Ok(result_dict)
        })
    }

    /// Reset Talebian risk system
    pub fn reset(&mut self) -> PyResult<()> {
        self.inner.reset()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to reset: {}", e)))?;
        Ok(())
    }

    /// Check if quantum advantage is achieved
    pub fn has_quantum_advantage(&self) -> bool {
        self.inner.has_quantum_advantage()
    }

    /// Get historical black swan events
    pub fn get_historical_black_swans(&self) -> PyResult<PyList> {
        let events = self.inner.get_historical_black_swans()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get historical events: {}", e)))?;

        Python::with_gil(|py| {
            let py_list = PyList::empty(py);
            for event in events {
                let event_dict = PyDict::new(py);
                event_dict.set_item("magnitude", event.magnitude)?;
                event_dict.set_item("probability", event.probability)?;
                event_dict.set_item("impact", event.impact)?;
                event_dict.set_item("timestamp", event.timestamp.timestamp())?;
                event_dict.set_item("event_type", format!("{:?}", event.event_type))?;
                event_dict.set_item("description", &event.description)?;
                py_list.append(event_dict)?;
            }
            Ok(py_list)
        })
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!("QuantumTalebianRisk(quantum_enabled={}, black_swans_detected={})", 
                self.inner.is_quantum_enabled(), 
                self.inner.black_swan_count())
    }

    /// Representation
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

impl PyQuantumTalebianRisk {
    fn parse_talebian_config(config_dict: &PyDict) -> PyResult<QuantumTalebianConfig> {
        let mut config = QuantumTalebianConfig::default();

        if let Some(processing_mode) = config_dict.get_item("processing_mode") {
            let mode_str = processing_mode.extract::<String>()?;
            config.processing_mode = match mode_str.as_str() {
                "classical" => QuantumTalebianMode::Classical,
                "quantum" => QuantumTalebianMode::Quantum,
                "hybrid" => QuantumTalebianMode::Hybrid,
                "auto" => QuantumTalebianMode::Auto,
                _ => return Err(PyValueError::new_err(format!("Unknown processing mode: {}", mode_str))),
            };
        }

        if let Some(num_qubits) = config_dict.get_item("num_qubits") {
            config.num_qubits = num_qubits.extract::<usize>()?;
        }

        if let Some(circuit_depth) = config_dict.get_item("circuit_depth") {
            config.circuit_depth = circuit_depth.extract::<usize>()?;
        }

        if let Some(threshold) = config_dict.get_item("black_swan_threshold") {
            config.black_swan_threshold = threshold.extract::<f64>()?;
        }

        if let Some(window) = config_dict.get_item("antifragility_window") {
            config.antifragility_window = window.extract::<usize>()?;
        }

        if let Some(percentile) = config_dict.get_item("tail_risk_percentile") {
            config.tail_risk_percentile = percentile.extract::<f64>()?;
        }

        Ok(config)
    }

    fn parse_antifragility_type(type_str: &str) -> PyResult<AntifragilityType> {
        match type_str {
            "volatility" => Ok(AntifragilityType::Volatility),
            "disorder" => Ok(AntifragilityType::Disorder),
            "stress" => Ok(AntifragilityType::Stress),
            "uncertainty" => Ok(AntifragilityType::Uncertainty),
            "tail_events" => Ok(AntifragilityType::TailEvents),
            "complexity" => Ok(AntifragilityType::Complexity),
            _ => Err(PyValueError::new_err(format!("Unknown antifragility type: {}", type_str))),
        }
    }

    fn parse_stress_events(events_list: PyList) -> PyResult<Vec<f64>> {
        let mut events = Vec::new();
        for item in events_list.iter() {
            events.push(item.extract::<f64>()?);
        }
        Ok(events)
    }

    fn parse_portfolio_data(portfolio_dict: PyDict) -> PyResult<HashMap<String, f64>> {
        let mut portfolio = HashMap::new();
        for (key, value) in portfolio_dict.iter() {
            let key_str = key.extract::<String>()?;
            let value_f64 = value.extract::<f64>()?;
            portfolio.insert(key_str, value_f64);
        }
        Ok(portfolio)
    }

    fn parse_market_conditions(market_dict: PyDict) -> PyResult<HashMap<String, f64>> {
        let mut market = HashMap::new();
        for (key, value) in market_dict.iter() {
            let key_str = key.extract::<String>()?;
            let value_f64 = value.extract::<f64>()?;
            market.insert(key_str, value_f64);
        }
        Ok(market)
    }

    fn parse_risky_asset_data(asset_dict: PyDict) -> PyResult<HashMap<String, f64>> {
        let mut asset_data = HashMap::new();
        for (key, value) in asset_dict.iter() {
            let key_str = key.extract::<String>()?;
            let value_f64 = value.extract::<f64>()?;
            asset_data.insert(key_str, value_f64);
        }
        Ok(asset_data)
    }

    fn parse_stress_scenarios(scenarios_list: PyList) -> PyResult<Vec<HashMap<String, f64>>> {
        let mut scenarios = Vec::new();
        for item in scenarios_list.iter() {
            let scenario_dict = item.downcast::<PyDict>()?;
            let mut scenario = HashMap::new();
            for (key, value) in scenario_dict.iter() {
                let key_str = key.extract::<String>()?;
                let value_f64 = value.extract::<f64>()?;
                scenario.insert(key_str, value_f64);
            }
            scenarios.push(scenario);
        }
        Ok(scenarios)
    }

    fn parse_elimination_candidates(candidates_list: PyList) -> PyResult<Vec<String>> {
        let mut candidates = Vec::new();
        for item in candidates_list.iter() {
            candidates.push(item.extract::<String>()?);
        }
        Ok(candidates)
    }

    fn parse_manager_data(manager_dict: PyDict) -> PyResult<HashMap<String, f64>> {
        let mut manager = HashMap::new();
        for (key, value) in manager_dict.iter() {
            let key_str = key.extract::<String>()?;
            let value_f64 = value.extract::<f64>()?;
            manager.insert(key_str, value_f64);
        }
        Ok(manager)
    }

    fn parse_investor_data(investor_dict: PyDict) -> PyResult<HashMap<String, f64>> {
        let mut investor = HashMap::new();
        for (key, value) in investor_dict.iter() {
            let key_str = key.extract::<String>()?;
            let value_f64 = value.extract::<f64>()?;
            investor.insert(key_str, value_f64);
        }
        Ok(investor)
    }

    fn parse_alignment_requirements(requirements_dict: PyDict) -> PyResult<HashMap<String, f64>> {
        let mut requirements = HashMap::new();
        for (key, value) in requirements_dict.iter() {
            let key_str = key.extract::<String>()?;
            let value_f64 = value.extract::<f64>()?;
            requirements.insert(key_str, value_f64);
        }
        Ok(requirements)
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

/// Utility functions for Talebian risk analysis
#[pyfunction]
pub fn calculate_black_swan_probability(
    historical_data: PyReadonlyArray1<f64>,
    threshold_sigma: f64
) -> PyResult<f64> {
    let data = historical_data.as_slice()?.to_vec();
    let probability = crate::calculate_black_swan_probability(&data, threshold_sigma)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate black swan probability: {}", e)))?;
    Ok(probability)
}

/// Calculate barbell strategy optimal allocation
#[pyfunction]
pub fn optimal_barbell_allocation(
    safe_return: f64,
    risky_return: f64,
    risky_volatility: f64,
    max_loss_tolerance: f64
) -> PyResult<PyTuple> {
    let (safe_alloc, risky_alloc) = crate::optimal_barbell_allocation(
        safe_return, risky_return, risky_volatility, max_loss_tolerance)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate optimal allocation: {}", e)))?;
    
    Python::with_gil(|py| {
        Ok(PyTuple::new(py, &[safe_alloc.into_py(py), risky_alloc.into_py(py)]))
    })
}

/// Calculate antifragility coefficient
#[pyfunction]
pub fn antifragility_coefficient(
    returns: PyReadonlyArray1<f64>,
    stress_indicator: PyReadonlyArray1<f64>
) -> PyResult<f64> {
    let returns_vec = returns.as_slice()?.to_vec();
    let stress_vec = stress_indicator.as_slice()?.to_vec();
    
    let coefficient = crate::antifragility_coefficient(&returns_vec, &stress_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate antifragility coefficient: {}", e)))?;
    Ok(coefficient)
}

/// Calculate via negativa benefit
#[pyfunction]
pub fn via_negativa_benefit(
    complexity_measure: f64,
    elimination_impact: PyReadonlyArray1<f64>
) -> PyResult<f64> {
    let impact_vec = elimination_impact.as_slice()?.to_vec();
    let benefit = crate::via_negativa_benefit(complexity_measure, &impact_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate via negativa benefit: {}", e)))?;
    Ok(benefit)
}

/// Calculate Lindy effect strength
#[pyfunction]
pub fn lindy_effect_strength(
    asset_ages: PyReadonlyArray1<f64>,
    survival_rates: PyReadonlyArray1<f64>
) -> PyResult<f64> {
    let ages = asset_ages.as_slice()?.to_vec();
    let survival = survival_rates.as_slice()?.to_vec();
    
    let strength = crate::lindy_effect_strength(&ages, &survival)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to calculate Lindy effect: {}", e)))?;
    Ok(strength)
}

/// Get Talebian risk module version
#[pyfunction]
pub fn get_version() -> String {
    crate::VERSION.to_string()
}

/// Initialize Talebian risk module
#[pyfunction]
pub fn initialize_talebian_risk() -> PyResult<()> {
    crate::initialize()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize Talebian risk module: {}", e)))
}

/// Python module definition
#[pymodule]
fn talebian_risk(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyQuantumTalebianRisk>()?;
    
    m.add_function(wrap_pyfunction!(calculate_black_swan_probability, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_barbell_allocation, m)?)?;
    m.add_function(wrap_pyfunction!(antifragility_coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(via_negativa_benefit, m)?)?;
    m.add_function(wrap_pyfunction!(lindy_effect_strength, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(initialize_talebian_risk, m)?)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_python_quantum_talebian_risk() {
        Python::with_gil(|py| {
            let risk_manager = PyQuantumTalebianRisk::new(None).unwrap();
            assert!(risk_manager.__str__().contains("QuantumTalebianRisk"));
        });
    }

    #[test]
    fn test_talebian_utility_functions() {
        Python::with_gil(|py| {
            let version = get_version();
            assert!(!version.is_empty());
            
            let returns = vec![0.01, -0.02, 0.015, -0.03, 0.02];
            let stress = vec![0.1, 0.8, 0.2, 0.9, 0.3];
            
            let returns_array = PyArray1::from_vec(py, returns);
            let stress_array = PyArray1::from_vec(py, stress);
            
            let coefficient = antifragility_coefficient(returns_array.readonly(), stress_array.readonly());
            assert!(coefficient.is_ok());
        });
    }

    #[test]
    fn test_black_swan_probability() {
        Python::with_gil(|py| {
            let data = vec![0.01, 0.02, -0.05, 0.01, 0.03, -0.08, 0.02];
            let data_array = PyArray1::from_vec(py, data);
            
            let probability = calculate_black_swan_probability(data_array.readonly(), 3.0);
            assert!(probability.is_ok());
            
            let prob_value = probability.unwrap();
            assert!(prob_value >= 0.0 && prob_value <= 1.0);
        });
    }
}