//! Python bindings for the Talebian Risk Management library
//!
//! This module provides Python bindings using PyO3 to make the Rust library
//! accessible from Python environments.

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
pub mod antifragility;
#[cfg(feature = "python")]
pub mod black_swan;
#[cfg(feature = "python")]
pub mod strategies;
#[cfg(feature = "python")]
pub mod distributions;
#[cfg(feature = "python")]
pub mod portfolio;

#[cfg(feature = "python")]
use crate::strategies::MarketRegime;
#[cfg(feature = "python")]
use crate::antifragility::{AntifragilityMeasurement, AntifragilityMeasurer, AntifragilityParams};
#[cfg(feature = "python")]
use crate::barbell::{AssetType, BarbellStrategy, BarbellParams, StrategyConfig};
#[cfg(feature = "python")]
use crate::black_swan::{BlackSwanDetector, BlackSwanEvent, MarketObservation, BlackSwanParams, BlackSwanType};
#[cfg(feature = "python")]
use crate::distributions::{ParetoDistribution, ExtremeValueStats, DistributionMoments};
#[cfg(feature = "python")]
use crate::error::TalebianError;

#[cfg(feature = "python")]
use chrono::{DateTime, Utc};
#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2};
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "python")]
use std::collections::HashMap;

/// Python module initialization
#[cfg(feature = "python")]
#[pymodule]
fn talebian_risk(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core types
    m.add_class::<PyAntifragilityMeasurer>()?;
    m.add_class::<PyBlackSwanDetector>()?;
    m.add_class::<PyBarbellStrategy>()?;
    m.add_class::<PyMarketData>()?;
    m.add_class::<PyRiskConfig>()?;
    
    // Distribution types
    m.add_class::<PyFatTailDistribution>()?;
    m.add_class::<PyDistributionMoments>()?;
    m.add_class::<PyExtremeValueStats>()?;
    
    // Strategy types
    m.add_class::<PyStrategyRiskMetrics>()?;
    m.add_class::<PyPerformanceAttribution>()?;
    m.add_class::<PyRobustnessAssessment>()?;
    
    // Utility functions
    m.add_function(wrap_pyfunction!(create_barbell_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(measure_antifragility, m)?)?;
    m.add_function(wrap_pyfunction!(detect_black_swan, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_risk_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(fit_fat_tail_distribution, m)?)?;
    
    // Constants
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    m.add("DESCRIPTION", env!("CARGO_PKG_DESCRIPTION"))?;
    
    Ok(())
}

/// Python wrapper for AntifragilityMeasurer
#[cfg(feature = "python")]
#[pyclass(name = "AntifragilityMeasurer")]
pub struct PyAntifragilityMeasurer {
    inner: AntifragilityMeasurer,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAntifragilityMeasurer {
    #[new]
    fn new(id: String, volatility_threshold: f64, convexity_sensitivity: f64) -> PyResult<Self> {
        let params = AntifragilityParams {
            volatility_threshold,
            convexity_sensitivity,
            ..Default::default()
        };
        
        let inner = AntifragilityMeasurer::new(id, params);
        Ok(Self { inner })
    }
    
    fn measure_antifragility(&mut self, returns: &PyArray1<f64>) -> PyResult<PyAntifragilityMeasurement> {
        let returns_slice = unsafe { returns.as_slice()? };
        let measurement = self.inner.measure_antifragility(returns_slice)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyAntifragilityMeasurement { inner: measurement })
    }
    
    fn get_trend(&self, periods: usize) -> PyResult<f64> {
        self.inner.get_trend(periods)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    fn update_regime(&mut self, regime: String) -> PyResult<()> {
        let regime = match regime.as_str() {
            "normal" => MarketRegime::Normal,
            "volatile" => MarketRegime::Volatile,
            "crisis" => MarketRegime::Crisis,
            "recovery" => MarketRegime::Recovery,
            "bubble" => MarketRegime::Bubble,
            "crash" => MarketRegime::Crash,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid regime")),
        };
        
        self.inner.update_regime(regime);
        Ok(())
    }
    
    fn clear_history(&mut self) {
        self.inner.clear_history();
    }
    
    fn get_id(&self) -> String {
        self.inner.id().to_string()
    }
}

/// Python wrapper for AntifragilityMeasurement
#[cfg(feature = "python")]
#[pyclass(name = "AntifragilityMeasurement")]
pub struct PyAntifragilityMeasurement {
    inner: AntifragilityMeasurement,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAntifragilityMeasurement {
    #[getter]
    fn overall_score(&self) -> f64 {
        self.inner.overall_score
    }
    
    #[getter]
    fn convexity(&self) -> f64 {
        self.inner.convexity
    }
    
    #[getter]
    fn volatility_benefit(&self) -> f64 {
        self.inner.volatility_benefit
    }
    
    #[getter]
    fn stress_response(&self) -> f64 {
        self.inner.stress_response
    }
    
    #[getter]
    fn hormesis_effect(&self) -> f64 {
        self.inner.hormesis_effect
    }
    
    #[getter]
    fn tail_benefit(&self) -> f64 {
        self.inner.tail_benefit
    }
    
    #[getter]
    fn regime_adaptation(&self) -> f64 {
        self.inner.regime_adaptation
    }
    
    fn is_antifragile(&self) -> bool {
        self.inner.is_antifragile()
    }
    
    fn is_fragile(&self) -> bool {
        self.inner.is_fragile()
    }
    
    fn level_description(&self) -> String {
        self.inner.level_description().to_string()
    }
    
    fn __str__(&self) -> String {
        format!("AntifragilityMeasurement(score={:.3}, level={})", 
                self.inner.overall_score, self.inner.level_description())
    }
}

/// Python wrapper for BlackSwanDetector
#[cfg(feature = "python")]
#[pyclass(name = "BlackSwanDetector")]
pub struct PyBlackSwanDetector {
    inner: BlackSwanDetector,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBlackSwanDetector {
    #[new]
    fn new(id: String, min_std_devs: f64, probability_threshold: f64) -> PyResult<Self> {
        let params = BlackSwanParams {
            min_std_devs,
            probability_threshold,
            ..Default::default()
        };
        
        let inner = BlackSwanDetector::new(id, params);
        Ok(Self { inner })
    }
    
    fn add_observation(&mut self, 
                      returns: &PyDict, 
                      volatilities: &PyDict, 
                      correlations: &PyArray2<f64>,
                      volumes: &PyDict,
                      regime: String) -> PyResult<()> {
        let returns_map = dict_to_hashmap_f64(returns)?;
        let volatilities_map = dict_to_hashmap_f64(volatilities)?;
        let volumes_map = dict_to_hashmap_f64(volumes)?;
        
        let regime = match regime.as_str() {
            "normal" => MarketRegime::Normal,
            "volatile" => MarketRegime::Volatile,
            "crisis" => MarketRegime::Crisis,
            "recovery" => MarketRegime::Recovery,
            "bubble" => MarketRegime::Bubble,
            "crash" => MarketRegime::Crash,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid regime")),
        };
        
        let correlations_array = correlations.to_owned_array();
        
        let observation = MarketObservation {
            timestamp: Utc::now(),
            returns: returns_map,
            volatilities: volatilities_map,
            correlations: correlations_array,
            volumes: volumes_map,
            regime,
        };
        
        self.inner.add_observation(observation)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        Ok(())
    }
    
    fn get_current_probability(&self) -> f64 {
        self.inner.get_current_probability()
    }
    
    fn get_alert_state(&self) -> String {
        format!("{:?}", self.inner.get_alert_state())
    }
    
    fn get_events(&self) -> Vec<PyBlackSwanEvent> {
        self.inner.get_events().iter()
            .map(|event| PyBlackSwanEvent { inner: event.clone() })
            .collect()
    }
    
    fn get_events_by_type(&self, event_type: String) -> PyResult<Vec<PyBlackSwanEvent>> {
        let event_type = match event_type.as_str() {
            "market_crash" => BlackSwanType::MarketCrash,
            "correlation_breakdown" => BlackSwanType::CorrelationBreakdown,
            "liquidity_crisis" => BlackSwanType::LiquidityCrisis,
            "volatility_spike" => BlackSwanType::VolatilitySpike,
            "regime_change" => BlackSwanType::RegimeChange,
            "systemic_risk" => BlackSwanType::SystemicRisk,
            "tail_risk" => BlackSwanType::TailRisk,
            "contagion_effect" => BlackSwanType::ContagionEffect,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid event type")),
        };
        
        let events = self.inner.get_events_by_type(event_type);
        Ok(events.iter()
            .map(|event| PyBlackSwanEvent { inner: (*event).clone() })
            .collect())
    }
    
    fn clear_events(&mut self) {
        self.inner.clear_events();
    }
    
    fn get_summary(&self) -> PyBlackSwanSummary {
        PyBlackSwanSummary { inner: self.inner.get_summary() }
    }
}

/// Python wrapper for BlackSwanEvent
#[cfg(feature = "python")]
#[pyclass(name = "BlackSwanEvent")]
pub struct PyBlackSwanEvent {
    inner: BlackSwanEvent,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBlackSwanEvent {
    #[getter]
    fn severity(&self) -> f64 {
        self.inner.severity
    }
    
    #[getter]
    fn probability(&self) -> f64 {
        self.inner.probability
    }
    
    #[getter]
    fn impact(&self) -> f64 {
        self.inner.impact
    }
    
    #[getter]
    fn affected_assets(&self) -> Vec<String> {
        self.inner.affected_assets.clone()
    }
    
    #[getter]
    fn event_type(&self) -> String {
        format!("{:?}", self.inner.event_type)
    }
    
    #[getter]
    fn correlation_breakdown(&self) -> f64 {
        self.inner.correlation_breakdown
    }
    
    #[getter]
    fn volatility_spike(&self) -> f64 {
        self.inner.volatility_spike
    }
    
    #[getter]
    fn contagion_effect(&self) -> f64 {
        self.inner.contagion_effect
    }
    
    fn __str__(&self) -> String {
        format!("BlackSwanEvent(type={:?}, severity={:.3}, probability={:.6})", 
                self.inner.event_type, self.inner.severity, self.inner.probability)
    }
}

/// Python wrapper for BlackSwanSummary
#[cfg(feature = "python")]
#[pyclass(name = "BlackSwanSummary")]
pub struct PyBlackSwanSummary {
    inner: BlackSwanSummary,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBlackSwanSummary {
    #[getter]
    fn total_events(&self) -> usize {
        self.inner.total_events
    }
    
    #[getter]
    fn average_severity(&self) -> f64 {
        self.inner.average_severity
    }
    
    #[getter]
    fn current_probability(&self) -> f64 {
        self.inner.current_probability
    }
    
    #[getter]
    fn alert_state(&self) -> String {
        format!("{:?}", self.inner.alert_state)
    }
    
    fn get_event_count(&self, event_type: String) -> PyResult<usize> {
        let event_type = match event_type.as_str() {
            "market_crash" => BlackSwanType::MarketCrash,
            "correlation_breakdown" => BlackSwanType::CorrelationBreakdown,
            "liquidity_crisis" => BlackSwanType::LiquidityCrisis,
            "volatility_spike" => BlackSwanType::VolatilitySpike,
            "regime_change" => BlackSwanType::RegimeChange,
            "systemic_risk" => BlackSwanType::SystemicRisk,
            "tail_risk" => BlackSwanType::TailRisk,
            "contagion_effect" => BlackSwanType::ContagionEffect,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid event type")),
        };
        
        Ok(self.inner.event_counts.get(&event_type).cloned().unwrap_or(0))
    }
}

/// Python wrapper for BarbellStrategy
#[cfg(feature = "python")]
#[pyclass(name = "BarbellStrategy")]
pub struct PyBarbellStrategy {
    inner: BarbellStrategy,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBarbellStrategy {
    #[new]
    fn new(id: String, safe_target: f64, risky_target: f64, max_position_size: f64) -> PyResult<Self> {
        let config = StrategyConfig {
            max_position_size,
            ..Default::default()
        };
        
        let params = BarbellParams {
            safe_target,
            risky_target,
            ..Default::default()
        };
        
        let inner = BarbellStrategy::new(id, config, params)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        Ok(Self { inner })
    }
    
    fn calculate_position_sizes(&self, assets: Vec<String>, market_data: &PyMarketData) -> PyResult<HashMap<String, f64>> {
        self.inner.calculate_position_sizes(&assets, &market_data.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    fn update_strategy(&mut self, market_data: &PyMarketData) -> PyResult<()> {
        self.inner.update_strategy(&market_data.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    fn expected_return(&self, market_data: &PyMarketData) -> PyResult<f64> {
        self.inner.expected_return(&market_data.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    fn risk_metrics(&self, market_data: &PyMarketData) -> PyResult<PyStrategyRiskMetrics> {
        let metrics = self.inner.risk_metrics(&market_data.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyStrategyRiskMetrics { inner: metrics })
    }
    
    fn get_barbell_metrics(&self) -> PyBarbellMetrics {
        PyBarbellMetrics { inner: self.inner.get_barbell_metrics() }
    }
    
    fn is_suitable(&self, market_data: &PyMarketData) -> PyResult<bool> {
        self.inner.is_suitable(&market_data.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    fn calculate_capacity(&self, market_data: &PyMarketData) -> PyResult<f64> {
        self.inner.calculate_capacity(&market_data.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    fn get_id(&self) -> String {
        self.inner.id().to_string()
    }
    
    fn get_name(&self) -> String {
        self.inner.name().to_string()
    }
}

/// Python wrapper for BarbellMetrics
#[cfg(feature = "python")]
#[pyclass(name = "BarbellMetrics")]
pub struct PyBarbellMetrics {
    inner: BarbellMetrics,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBarbellMetrics {
    #[getter]
    fn safe_allocation(&self) -> f64 {
        self.inner.safe_allocation
    }
    
    #[getter]
    fn risky_allocation(&self) -> f64 {
        self.inner.risky_allocation
    }
    
    #[getter]
    fn target_safe_allocation(&self) -> f64 {
        self.inner.target_safe_allocation
    }
    
    #[getter]
    fn target_risky_allocation(&self) -> f64 {
        self.inner.target_risky_allocation
    }
    
    #[getter]
    fn allocation_drift(&self) -> f64 {
        self.inner.allocation_drift
    }
    
    #[getter]
    fn barbell_ratio(&self) -> f64 {
        self.inner.barbell_ratio
    }
    
    #[getter]
    fn convexity_exposure(&self) -> f64 {
        self.inner.convexity_exposure
    }
    
    #[getter]
    fn safety_score(&self) -> f64 {
        self.inner.safety_score
    }
}

/// Python wrapper for MarketData
#[cfg(feature = "python")]
#[pyclass(name = "MarketData")]
pub struct PyMarketData {
    inner: MarketData,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMarketData {
    #[new]
    fn new(prices: &PyDict, 
           returns: &PyDict, 
           volatilities: &PyDict, 
           correlations: &PyDict,
           volumes: &PyDict, 
           asset_types: &PyDict,
           regime: String) -> PyResult<Self> {
        let prices_map = dict_to_hashmap_f64(prices)?;
        let returns_map = dict_to_hashmap_vec_f64(returns)?;
        let volatilities_map = dict_to_hashmap_f64(volatilities)?;
        let correlations_map = dict_to_hashmap_correlation(correlations)?;
        let volumes_map = dict_to_hashmap_f64(volumes)?;
        let asset_types_map = dict_to_hashmap_asset_type(asset_types)?;
        
        let regime = match regime.as_str() {
            "normal" => MarketRegime::Normal,
            "bull" => MarketRegime::Bull,
            "bear" => MarketRegime::Bear,
            "high_volatility" => MarketRegime::HighVolatility,
            "low_volatility" => MarketRegime::LowVolatility,
            "crisis" => MarketRegime::Crisis,
            "recovery" => MarketRegime::Recovery,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid regime")),
        };
        
        let inner = MarketData {
            prices: prices_map,
            returns: returns_map,
            volatilities: volatilities_map,
            correlations: correlations_map,
            volumes: volumes_map,
            asset_types: asset_types_map,
            timestamp: Utc::now(),
            regime,
        };
        
        Ok(Self { inner })
    }
}

/// Utility function to create a barbell strategy
#[cfg(feature = "python")]
#[pyfunction]
fn create_barbell_strategy(id: String, safe_target: f64, risky_target: f64) -> PyResult<PyBarbellStrategy> {
    let config = StrategyConfig::default();
    let params = BarbellParams {
        safe_target,
        risky_target,
        ..Default::default()
    };
    
    let inner = BarbellStrategy::new(id, config, params)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    Ok(PyBarbellStrategy { inner })
}

/// Utility function to measure antifragility
#[cfg(feature = "python")]
#[pyfunction]
fn measure_antifragility(returns: &PyArray1<f64>, volatility_threshold: f64) -> PyResult<PyAntifragilityMeasurement> {
    let params = AntifragilityParams {
        volatility_threshold,
        ..Default::default()
    };
    
    let mut measurer = AntifragilityMeasurer::new("utility", params);
    let returns_slice = unsafe { returns.as_slice()? };
    let measurement = measurer.measure_antifragility(returns_slice)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    Ok(PyAntifragilityMeasurement { inner: measurement })
}

/// Utility function to detect black swan events
#[cfg(feature = "python")]
#[pyfunction]
fn detect_black_swan(returns: &PyDict, volatilities: &PyDict, min_std_devs: f64) -> PyResult<Vec<PyBlackSwanEvent>> {
    let params = BlackSwanParams {
        min_std_devs,
        ..Default::default()
    };
    
    let mut detector = BlackSwanDetector::new("utility", params);
    
    let returns_map = dict_to_hashmap_f64(returns)?;
    let volatilities_map = dict_to_hashmap_f64(volatilities)?;
    
    // Create a simple observation
    let observation = MarketObservation {
        timestamp: Utc::now(),
        returns: returns_map,
        volatilities: volatilities_map,
        correlations: ndarray::Array2::eye(1),
        volumes: HashMap::new(),
        regime: MarketRegime::Normal,
    };
    
    detector.add_observation(observation)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    Ok(detector.get_events().iter()
        .map(|event| PyBlackSwanEvent { inner: event.clone() })
        .collect())
}

/// Additional wrapper types for completeness
#[cfg(feature = "python")]
#[pyclass(name = "StrategyRiskMetrics")]
pub struct PyStrategyRiskMetrics {
    inner: StrategyRiskMetrics,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyStrategyRiskMetrics {
    #[getter]
    fn var_95(&self) -> f64 { self.inner.var_95 }
    
    #[getter]
    fn cvar_95(&self) -> f64 { self.inner.cvar_95 }
    
    #[getter]
    fn max_drawdown(&self) -> f64 { self.inner.max_drawdown }
    
    #[getter]
    fn volatility(&self) -> f64 { self.inner.volatility }
    
    #[getter]
    fn sharpe_ratio(&self) -> f64 { self.inner.sortino_ratio }
    
    #[getter]
    fn antifragility_score(&self) -> f64 { self.inner.antifragility_score }
    
    #[getter]
    fn black_swan_probability(&self) -> f64 { self.inner.black_swan_probability }
}

/// Placeholder wrappers for distributions and other types
#[cfg(feature = "python")]
#[pyclass(name = "FatTailDistribution")]
pub struct PyFatTailDistribution {
    // This would contain the actual distribution implementation
}

#[cfg(feature = "python")]
#[pyclass(name = "DistributionMoments")]
pub struct PyDistributionMoments {
    // This would contain the actual moments implementation
}

#[cfg(feature = "python")]
#[pyclass(name = "ExtremeValueStats")]
pub struct PyExtremeValueStats {
    // This would contain the actual extreme value stats implementation
}

#[cfg(feature = "python")]
#[pyclass(name = "PerformanceAttribution")]
pub struct PyPerformanceAttribution {
    // This would contain the actual performance attribution implementation
}

#[cfg(feature = "python")]
#[pyclass(name = "RobustnessAssessment")]
pub struct PyRobustnessAssessment {
    // This would contain the actual robustness assessment implementation
}

#[cfg(feature = "python")]
#[pyclass(name = "RiskConfig")]
pub struct PyRiskConfig {
    // This would contain the actual risk config implementation
}

/// Utility functions for Python interop
#[cfg(feature = "python")]
#[pyfunction]
fn calculate_risk_metrics(returns: &PyArray1<f64>) -> PyResult<PyStrategyRiskMetrics> {
    // Placeholder implementation
    let inner = StrategyRiskMetrics {
        var_95: -0.05,
        cvar_95: -0.08,
        max_drawdown: 0.15,
        volatility: 0.2,
        downside_deviation: 0.15,
        tail_ratio: 1.2,
        sortino_ratio: 0.8,
        calmar_ratio: 0.5,
        antifragility_score: 0.3,
        black_swan_probability: 0.01,
    };
    
    Ok(PyStrategyRiskMetrics { inner })
}

#[cfg(feature = "python")]
#[pyfunction]
fn fit_fat_tail_distribution(data: &PyArray1<f64>, distribution_type: String) -> PyResult<PyFatTailDistribution> {
    // Placeholder implementation
    Ok(PyFatTailDistribution {})
}

/// Helper functions for converting between Python and Rust types
#[cfg(feature = "python")]
fn dict_to_hashmap_f64(dict: &PyDict) -> PyResult<HashMap<String, f64>> {
    let mut map = HashMap::new();
    for (key, value) in dict {
        let key_str = key.extract::<String>()?;
        let value_f64 = value.extract::<f64>()?;
        map.insert(key_str, value_f64);
    }
    Ok(map)
}

#[cfg(feature = "python")]
fn dict_to_hashmap_vec_f64(dict: &PyDict) -> PyResult<HashMap<String, Vec<f64>>> {
    let mut map = HashMap::new();
    for (key, value) in dict {
        let key_str = key.extract::<String>()?;
        let value_list = value.extract::<Vec<f64>>()?;
        map.insert(key_str, value_list);
    }
    Ok(map)
}

#[cfg(feature = "python")]
fn dict_to_hashmap_correlation(dict: &PyDict) -> PyResult<HashMap<(String, String), f64>> {
    let mut map = HashMap::new();
    for (key, value) in dict {
        let key_tuple = key.extract::<(String, String)>()?;
        let value_f64 = value.extract::<f64>()?;
        map.insert(key_tuple, value_f64);
    }
    Ok(map)
}

#[cfg(feature = "python")]
fn dict_to_hashmap_asset_type(dict: &PyDict) -> PyResult<HashMap<String, AssetType>> {
    let mut map = HashMap::new();
    for (key, value) in dict {
        let key_str = key.extract::<String>()?;
        let value_str = value.extract::<String>()?;
        let asset_type = match value_str.as_str() {
            "volatile" => AssetType::Volatile,
            "moderate" => AssetType::Moderate,
            "safe" => AssetType::Safe,
            "antifragile" => AssetType::Antifragile,
            "derivative" => AssetType::Derivative,
            "alternative" => AssetType::Alternative,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid asset type")),
        };
        map.insert(key_str, asset_type);
    }
    Ok(map)
}

#[cfg(not(feature = "python"))]
pub fn init_python_module() {
    // Empty function when Python feature is not enabled
}