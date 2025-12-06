//! Python bindings for hedge algorithms using PyO3

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "python")]
use numpy::PyArray1;
#[cfg(feature = "python")]
use std::collections::HashMap;

#[cfg(feature = "python")]
use crate::{
    HedgeAlgorithms, HedgeConfig, QuantumHedgeAlgorithm, MarketData, HedgeRecommendation,
    StandardFactorModel, ExpertSystem, PerformanceMetrics, RiskMetrics, AdvancedMetrics,
    PerformanceAttribution, RiskAdjustedMetrics, OptionsHedger, OptionType, Greeks,
    PairsTrader, VolatilityHedger, WhaleDetector, RegretMinimizer, HedgeError,
};

#[cfg(feature = "python")]
#[pyclass]
pub struct PyHedgeAlgorithms {
    inner: HedgeAlgorithms,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyHedgeAlgorithms {
    #[new]
    fn new(config: Option<PyHedgeConfig>) -> PyResult<Self> {
        let hedge_config = config.map(|c| c.inner).unwrap_or_default();
        let inner = HedgeAlgorithms::new(hedge_config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyHedgeAlgorithms { inner })
    }
    
    fn update_market_data(&mut self, market_data: PyMarketData) -> PyResult<()> {
        self.inner.update_market_data(market_data.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn get_hedge_recommendation(&self) -> PyResult<PyHedgeRecommendation> {
        let recommendation = self.inner.get_hedge_recommendation()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyHedgeRecommendation { inner: recommendation })
    }
    
    fn get_metrics(&self) -> PyResult<PyPerformanceMetrics> {
        let metrics = self.inner.get_metrics();
        Ok(PyPerformanceMetrics { inner: metrics })
    }
    
    fn reset(&self) -> PyResult<()> {
        self.inner.reset()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }
    
    fn save_state(&self, path: &str) -> PyResult<()> {
        self.inner.save_state(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }
    
    #[staticmethod]
    fn load_state(path: &str) -> PyResult<Self> {
        let inner = HedgeAlgorithms::load_state(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyHedgeAlgorithms { inner })
    }
}

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyHedgeConfig {
    inner: HedgeConfig,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyHedgeConfig {
    #[new]
    fn new(
        learning_rate: Option<f64>,
        weight_decay: Option<f64>,
        min_weight: Option<f64>,
        max_weight: Option<f64>,
        max_history: Option<usize>,
        confidence_threshold: Option<f64>,
        risk_tolerance: Option<f64>,
    ) -> Self {
        let mut config = HedgeConfig::default();
        
        if let Some(lr) = learning_rate {
            config.learning_rate = lr;
        }
        if let Some(wd) = weight_decay {
            config.weight_decay = wd;
        }
        if let Some(min_w) = min_weight {
            config.min_weight = min_w;
        }
        if let Some(max_w) = max_weight {
            config.max_weight = max_w;
        }
        if let Some(max_h) = max_history {
            config.max_history = max_h;
        }
        if let Some(ct) = confidence_threshold {
            config.confidence_threshold = ct;
        }
        if let Some(rt) = risk_tolerance {
            config.risk_tolerance = rt;
        }
        
        PyHedgeConfig { inner: config }
    }
    
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        // TODO: Implement from_file for HedgeConfig
        let config = HedgeConfig::default();
        Ok(PyHedgeConfig { inner: config })
    }
    
    fn to_file(&self, path: &str) -> PyResult<()> {
        // TODO: Implement to_file for HedgeConfig
        Ok(())
    }
    
    fn validate(&self) -> PyResult<()> {
        // TODO: Implement validate for HedgeConfig
        Ok(())
    }
    
    #[getter]
    fn learning_rate(&self) -> f64 {
        self.inner.learning_rate
    }
    
    #[setter]
    fn set_learning_rate(&mut self, value: f64) {
        self.inner.learning_rate = value;
    }
    
    #[getter]
    fn weight_decay(&self) -> f64 {
        self.inner.weight_decay
    }
    
    #[setter]
    fn set_weight_decay(&mut self, value: f64) {
        self.inner.weight_decay = value;
    }
}

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyMarketData {
    inner: MarketData,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMarketData {
    #[new]
    fn new(
        symbol: String,
        timestamp: f64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        bid: Option<f64>,
        ask: Option<f64>,
    ) -> Self {
        let timestamp = chrono::DateTime::from_timestamp(timestamp as i64, 0)
            .unwrap_or_else(chrono::Utc::now);
        
        let mut market_data = MarketData::new(symbol, timestamp, [open, high, low, close, volume]);
        market_data.bid = bid.unwrap_or(0.0);
        market_data.ask = ask.unwrap_or(0.0);
        
        PyMarketData { inner: market_data }
    }
    
    #[getter]
    fn symbol(&self) -> String {
        self.inner.symbol.clone()
    }
    
    #[getter]
    fn timestamp(&self) -> f64 {
        self.inner.timestamp.timestamp() as f64
    }
    
    #[getter]
    fn open(&self) -> f64 {
        self.inner.open()
    }
    
    #[getter]
    fn high(&self) -> f64 {
        self.inner.high()
    }
    
    #[getter]
    fn low(&self) -> f64 {
        self.inner.low()
    }
    
    #[getter]
    fn close(&self) -> f64 {
        self.inner.close()
    }
    
    #[getter]
    fn volume(&self) -> f64 {
        self.inner.volume()
    }
    
    #[getter]
    fn bid(&self) -> Option<f64> {
        Some(self.inner.bid)
    }
    
    #[getter]
    fn ask(&self) -> Option<f64> {
        Some(self.inner.ask)
    }
    
    fn typical_price(&self) -> f64 {
        self.inner.typical_price()
    }
    
    fn weighted_price(&self) -> f64 {
        self.inner.weighted_price()
    }
    
    fn spread(&self) -> f64 {
        self.inner.spread
    }
    
    fn mid_price(&self) -> f64 {
        self.inner.mid_price
    }
}

#[cfg(feature = "python")]
#[pyclass]
pub struct PyHedgeRecommendation {
    inner: HedgeRecommendation,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyHedgeRecommendation {
    #[getter]
    fn position_size(&self) -> f64 {
        self.inner.position_size
    }
    
    #[getter]
    fn hedge_ratio(&self) -> f64 {
        self.inner.hedge_ratio
    }
    
    #[getter]
    fn confidence(&self) -> f64 {
        self.inner.confidence
    }
    
    #[getter]
    fn expected_return(&self) -> f64 {
        self.inner.expected_return
    }
    
    #[getter]
    fn volatility(&self) -> f64 {
        self.inner.volatility
    }
    
    #[getter]
    fn max_drawdown(&self) -> f64 {
        self.inner.max_drawdown
    }
    
    #[getter]
    fn sharpe_ratio(&self) -> f64 {
        self.inner.sharpe_ratio
    }
    
    #[getter]
    fn timestamp(&self) -> f64 {
        self.inner.timestamp.timestamp() as f64
    }
    
    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }
    
    fn risk_adjusted_return(&self) -> f64 {
        self.inner.risk_adjusted_return()
    }
    
    fn position_value(&self, price: f64) -> f64 {
        self.inner.position_value(price)
    }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("position_size", self.inner.position_size)?;
        dict.set_item("hedge_ratio", self.inner.hedge_ratio)?;
        dict.set_item("confidence", self.inner.confidence)?;
        dict.set_item("expected_return", self.inner.expected_return)?;
        dict.set_item("volatility", self.inner.volatility)?;
        dict.set_item("max_drawdown", self.inner.max_drawdown)?;
        dict.set_item("sharpe_ratio", self.inner.sharpe_ratio)?;
        dict.set_item("timestamp", self.inner.timestamp.timestamp())?;
        
        Ok(dict.into())
    }
}

#[cfg(feature = "python")]
#[pyclass]
pub struct PyPerformanceMetrics {
    inner: PerformanceMetrics,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPerformanceMetrics {
    #[getter]
    fn total_trades(&self) -> usize {
        self.inner.total_trades.try_into().unwrap_or(0)
    }
    
    #[getter]
    fn winning_trades(&self) -> usize {
        self.inner.winning_trades.try_into().unwrap_or(0)
    }
    
    #[getter]
    fn losing_trades(&self) -> usize {
        self.inner.losing_trades.try_into().unwrap_or(0)
    }
    
    #[getter]
    fn total_return(&self) -> f64 {
        self.inner.total_return
    }
    
    #[getter]
    fn average_return(&self) -> f64 {
        self.inner.average_return
    }
    
    #[getter]
    fn hit_rate(&self) -> f64 {
        self.inner.hit_rate
    }
    
    #[getter]
    fn profit_factor(&self) -> f64 {
        self.inner.profit_factor
    }
    
    #[getter]
    fn sortino_ratio(&self) -> f64 {
        self.inner.sortino_ratio
    }
    
    #[getter]
    fn calmar_ratio(&self) -> f64 {
        self.inner.calmar_ratio
    }
    
    fn summary(&self, py: Python) -> PyResult<PyObject> {
        let summary = self.inner.summary();
        let dict = PyDict::new(py);
        
        // TODO: Fix summary method - summary returns String but we need key-value pairs
        dict.set_item("summary", summary)?;
        
        Ok(dict.into())
    }
}

#[cfg(feature = "python")]
#[pyclass]
pub struct PyOptionsHedger {
    inner: OptionsHedger,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyOptionsHedger {
    #[new]
    fn new(config: Option<PyHedgeConfig>) -> Self {
        let hedge_config = config.map(|c| c.inner).unwrap_or_default();
        let inner = OptionsHedger::new(hedge_config);
        
        PyOptionsHedger { inner }
    }
    
    fn black_scholes_price(
        &self,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
        is_call: bool,
    ) -> PyResult<f64> {
        let option_type = if is_call { OptionType::Call } else { OptionType::Put };
        
        let price = self.inner.black_scholes_price(spot, strike, time_to_expiry, volatility, option_type)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(price)
    }
    
    fn calculate_greeks(
        &self,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
        is_call: bool,
    ) -> PyResult<PyGreeks> {
        let option_type = if is_call { OptionType::Call } else { OptionType::Put };
        
        let greeks = self.inner.calculate_greeks(spot, strike, time_to_expiry, volatility, option_type)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyGreeks { inner: greeks })
    }
    
    fn calculate_implied_volatility(
        &self,
        market_price: f64,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        is_call: bool,
    ) -> PyResult<f64> {
        let option_type = if is_call { OptionType::Call } else { OptionType::Put };
        
        let iv = self.inner.calculate_implied_volatility(market_price, spot, strike, time_to_expiry, option_type)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(iv)
    }
}

#[cfg(feature = "python")]
#[pyclass]
pub struct PyGreeks {
    inner: Greeks,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGreeks {
    #[getter]
    fn delta(&self) -> f64 {
        self.inner.delta
    }
    
    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma
    }
    
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }
    
    #[getter]
    fn vega(&self) -> f64 {
        self.inner.vega
    }
    
    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho
    }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("delta", self.inner.delta)?;
        dict.set_item("gamma", self.inner.gamma)?;
        dict.set_item("theta", self.inner.theta)?;
        dict.set_item("vega", self.inner.vega)?;
        dict.set_item("rho", self.inner.rho)?;
        
        Ok(dict.into())
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn hedge_algorithms(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyHedgeAlgorithms>()?;
    m.add_class::<PyHedgeConfig>()?;
    m.add_class::<PyMarketData>()?;
    m.add_class::<PyHedgeRecommendation>()?;
    m.add_class::<PyPerformanceMetrics>()?;
    m.add_class::<PyOptionsHedger>()?;
    m.add_class::<PyGreeks>()?;
    
    Ok(())
}

#[cfg(feature = "python")]
pub fn create_python_module(py: Python) -> PyResult<&PyModule> {
    let module = PyModule::new(py, "hedge_algorithms")?;
    
    module.add_class::<PyHedgeAlgorithms>()?;
    module.add_class::<PyHedgeConfig>()?;
    module.add_class::<PyMarketData>()?;
    module.add_class::<PyHedgeRecommendation>()?;
    module.add_class::<PyPerformanceMetrics>()?;
    module.add_class::<PyOptionsHedger>()?;
    module.add_class::<PyGreeks>()?;
    
    Ok(module)
}

#[cfg(test)]
#[cfg(feature = "python")]
mod tests {
    use super::*;
    
    #[test]
    fn test_python_config() {
        let config = PyHedgeConfig::new(
            Some(0.01),
            Some(0.999),
            Some(0.001),
            Some(0.5),
            Some(10000),
            Some(0.7),
            Some(0.05),
        );
        
        assert_eq!(config.learning_rate(), 0.01);
        assert_eq!(config.weight_decay(), 0.999);
    }
    
    #[test]
    fn test_python_market_data() {
        let market_data = PyMarketData::new(
            "BTCUSD".to_string(),
            1640995200.0, // 2022-01-01 00:00:00 UTC
            100.0,
            105.0,
            95.0,
            102.0,
            1000.0,
            Some(101.8),
            Some(102.2),
        );
        
        assert_eq!(market_data.symbol(), "BTCUSD");
        assert_eq!(market_data.open(), 100.0);
        assert_eq!(market_data.close(), 102.0);
        assert_eq!(market_data.bid(), Some(101.8));
        assert_eq!(market_data.ask(), Some(102.2));
        assert_eq!(market_data.spread(), 0.4);
    }
}