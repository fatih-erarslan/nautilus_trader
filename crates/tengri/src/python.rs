//! Python integration module for Tengri trading strategy
//! 
//! Provides Python bindings for configuration, monitoring, and control
//! of the Tengri trading strategy from Python applications.

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "python")]
use std::collections::HashMap;
#[cfg(feature = "python")]
use tokio::runtime::Runtime;

#[cfg(feature = "python")]
use crate::{TengriStrategy, TengriConfig, Result, TengriError};
#[cfg(feature = "python")]
use crate::types::{PortfolioMetrics, RiskMetrics, Position, Order, TradingSession};

/// Python wrapper for TengriStrategy
#[cfg(feature = "python")]
#[pyclass]
pub struct PyTengriStrategy {
    strategy: Option<TengriStrategy>,
    runtime: Runtime,
}

/// Python wrapper for TengriConfig
#[cfg(feature = "python")]
#[pyclass]
pub struct PyTengriConfig {
    config: TengriConfig,
}

/// Python wrapper for PortfolioMetrics
#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyPortfolioMetrics {
    #[pyo3(get)]
    total_value: f64,
    #[pyo3(get)]
    cash_balance: f64,
    #[pyo3(get)]
    unrealized_pnl: f64,
    #[pyo3(get)]
    realized_pnl: f64,
    #[pyo3(get)]
    return_percentage: f64,
    #[pyo3(get)]
    max_drawdown: f64,
    #[pyo3(get)]
    sharpe_ratio: Option<f64>,
    #[pyo3(get)]
    var_95: f64,
    #[pyo3(get)]
    open_positions: usize,
    #[pyo3(get)]
    volatility: f64,
}

/// Python wrapper for Position
#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyPosition {
    #[pyo3(get)]
    symbol: String,
    #[pyo3(get)]
    side: String,
    #[pyo3(get)]
    size: f64,
    #[pyo3(get)]
    entry_price: f64,
    #[pyo3(get)]
    current_price: f64,
    #[pyo3(get)]
    unrealized_pnl: f64,
    #[pyo3(get)]
    value: f64,
}

/// Python wrapper for TradingSession
#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyTradingSession {
    #[pyo3(get)]
    session_id: String,
    #[pyo3(get)]
    start_time: String,
    #[pyo3(get)]
    session_pnl: f64,
    #[pyo3(get)]
    trade_count: u64,
    #[pyo3(get)]
    win_rate: f64,
    #[pyo3(get)]
    max_drawdown: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTengriStrategy {
    /// Create new strategy instance from configuration file
    #[new]
    #[pyo3(signature = (config_path = None))]
    fn new(config_path: Option<String>) -> PyResult<Self> {
        let runtime = Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create async runtime: {}", e)))?;

        let strategy = if let Some(path) = config_path {
            let config = TengriConfig::from_file(path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to load config: {}", e)))?;
            
            Some(runtime.block_on(async {
                TengriStrategy::new(config).await
            }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create strategy: {}", e)))?)
        } else {
            None
        };

        Ok(Self { strategy, runtime })
    }

    /// Initialize strategy with configuration
    fn initialize(&mut self, config: &PyTengriConfig) -> PyResult<()> {
        self.strategy = Some(
            self.runtime.block_on(async {
                TengriStrategy::new(config.config.clone()).await
            }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to initialize strategy: {}", e)))?
        );
        Ok(())
    }

    /// Start the trading strategy
    fn start(&mut self) -> PyResult<()> {
        if let Some(ref mut strategy) = self.strategy {
            self.runtime.block_on(async {
                strategy.start().await
            }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to start strategy: {}", e)))?;
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Strategy not initialized"));
        }
        Ok(())
    }

    /// Stop the trading strategy
    fn stop(&mut self) -> PyResult<()> {
        if let Some(ref mut strategy) = self.strategy {
            self.runtime.block_on(async {
                strategy.stop().await
            }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to stop strategy: {}", e)))?;
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Strategy not initialized"));
        }
        Ok(())
    }

    /// Get current portfolio metrics
    fn get_portfolio_metrics(&self) -> PyResult<PyPortfolioMetrics> {
        if let Some(ref strategy) = self.strategy {
            let metrics = self.runtime.block_on(async {
                strategy.get_portfolio_metrics().await
            });
            
            Ok(PyPortfolioMetrics::from(metrics))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Strategy not initialized"))
        }
    }

    /// Get current positions
    fn get_positions(&self) -> PyResult<Vec<PyPosition>> {
        if let Some(ref strategy) = self.strategy {
            let positions = self.runtime.block_on(async {
                strategy.get_positions().await
            });
            
            let py_positions: Vec<PyPosition> = positions.values()
                .map(|pos| PyPosition::from(pos.clone()))
                .collect();
            
            Ok(py_positions)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Strategy not initialized"))
        }
    }

    /// Get current trading session
    fn get_current_session(&self) -> PyResult<Option<PyTradingSession>> {
        if let Some(ref strategy) = self.strategy {
            let session = self.runtime.block_on(async {
                strategy.get_current_session().await
            });
            
            Ok(session.map(PyTradingSession::from))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Strategy not initialized"))
        }
    }

    /// Get strategy status
    fn get_status(&self) -> PyResult<HashMap<String, String>> {
        let mut status = HashMap::new();
        
        if self.strategy.is_some() {
            status.insert("initialized".to_string(), "true".to_string());
            status.insert("status".to_string(), "ready".to_string());
        } else {
            status.insert("initialized".to_string(), "false".to_string());
            status.insert("status".to_string(), "not_initialized".to_string());
        }
        
        Ok(status)
    }

    /// Create market order (for manual trading)
    fn create_market_order(&mut self, symbol: String, side: String, quantity: f64) -> PyResult<String> {
        if let Some(ref mut strategy) = self.strategy {
            let order_side = match side.to_lowercase().as_str() {
                "buy" => crate::types::OrderSide::Buy,
                "sell" => crate::types::OrderSide::Sell,
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid order side")),
            };

            let order = self.runtime.block_on(async {
                strategy.execution_engine.create_market_order(&symbol, order_side, quantity).await
            }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create order: {}", e)))?;

            Ok(order.id)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Strategy not initialized"))
        }
    }

    /// Get performance summary
    fn get_performance_summary(&self) -> PyResult<HashMap<String, f64>> {
        if let Some(ref strategy) = self.strategy {
            let metrics = self.runtime.block_on(async {
                strategy.get_portfolio_metrics().await
            });
            
            let mut summary = HashMap::new();
            summary.insert("total_value".to_string(), metrics.total_value);
            summary.insert("unrealized_pnl".to_string(), metrics.unrealized_pnl);
            summary.insert("realized_pnl".to_string(), metrics.realized_pnl);
            summary.insert("return_percentage".to_string(), metrics.return_percentage);
            summary.insert("max_drawdown".to_string(), metrics.max_drawdown);
            summary.insert("volatility".to_string(), metrics.volatility);
            
            if let Some(sharpe) = metrics.sharpe_ratio {
                summary.insert("sharpe_ratio".to_string(), sharpe);
            }
            
            Ok(summary)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Strategy not initialized"))
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTengriConfig {
    /// Create new configuration
    #[new]
    fn new() -> Self {
        Self {
            config: TengriConfig::default(),
        }
    }

    /// Load configuration from TOML file
    #[staticmethod]
    fn from_file(path: String) -> PyResult<Self> {
        let config = TengriConfig::from_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to load config: {}", e)))?;
        
        Ok(Self { config })
    }

    /// Save configuration to TOML file
    fn to_file(&self, path: String) -> PyResult<()> {
        self.config.to_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to save config: {}", e)))?;
        Ok(())
    }

    /// Set strategy name
    fn set_strategy_name(&mut self, name: String) {
        self.config.strategy.name = name;
    }

    /// Set base currency
    fn set_base_currency(&mut self, currency: String) {
        self.config.strategy.base_currency = currency;
    }

    /// Set quote currency
    fn set_quote_currency(&mut self, currency: String) {
        self.config.strategy.quote_currency = currency;
    }

    /// Set trading instruments
    fn set_instruments(&mut self, instruments: Vec<String>) {
        self.config.strategy.instruments = instruments;
    }

    /// Set maximum position size
    fn set_max_position_size(&mut self, size: f64) {
        self.config.strategy.parameters.max_position_size = size;
    }

    /// Set signal threshold
    fn set_signal_threshold(&mut self, threshold: f64) {
        self.config.strategy.parameters.signal_threshold = threshold;
    }

    /// Enable/disable data sources
    fn enable_databento(&mut self, enabled: bool, api_key: Option<String>) {
        self.config.data_sources.databento.enabled = enabled;
        if let Some(key) = api_key {
            self.config.data_sources.databento.api_key = key;
        }
    }

    fn enable_tardis(&mut self, enabled: bool, api_key: Option<String>) {
        self.config.data_sources.tardis.enabled = enabled;
        if let Some(key) = api_key {
            self.config.data_sources.tardis.api_key = key;
        }
    }

    fn enable_polymarket(&mut self, enabled: bool, api_key: Option<String>) {
        self.config.data_sources.polymarket.enabled = enabled;
        if let Some(key) = api_key {
            self.config.data_sources.polymarket.api_key = Some(key);
        }
    }

    /// Configure Binance credentials
    fn set_binance_credentials(&mut self, api_key: String, api_secret: String, testnet: Option<bool>) {
        self.config.exchanges.binance_spot.api_key = api_key.clone();
        self.config.exchanges.binance_spot.api_secret = api_secret.clone();
        self.config.exchanges.binance_futures.api_key = api_key;
        self.config.exchanges.binance_futures.api_secret = api_secret;
        
        if let Some(testnet_enabled) = testnet {
            self.config.exchanges.binance_spot.testnet = testnet_enabled;
            self.config.exchanges.binance_futures.testnet = testnet_enabled;
        }
    }

    /// Set risk management parameters
    fn set_risk_parameters(&mut self, max_portfolio_loss: f64, max_daily_loss: f64, max_position_size: f64) {
        self.config.risk.max_portfolio_loss = max_portfolio_loss;
        self.config.risk.max_daily_loss = max_daily_loss;
        self.config.risk.max_position_size = max_position_size;
    }

    /// Enable GPU acceleration
    fn enable_gpu(&mut self, enabled: bool) {
        self.config.performance.gpu.enabled = enabled;
    }

    /// Get configuration as dictionary
    fn to_dict(&self) -> PyResult<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            let mut dict = HashMap::new();
            
            // Strategy info
            dict.insert("strategy_name".to_string(), self.config.strategy.name.to_object(py));
            dict.insert("base_currency".to_string(), self.config.strategy.base_currency.to_object(py));
            dict.insert("instruments".to_string(), self.config.strategy.instruments.to_object(py));
            
            // Risk parameters
            dict.insert("max_portfolio_loss".to_string(), self.config.risk.max_portfolio_loss.to_object(py));
            dict.insert("max_daily_loss".to_string(), self.config.risk.max_daily_loss.to_object(py));
            
            // Data sources
            dict.insert("databento_enabled".to_string(), self.config.data_sources.databento.enabled.to_object(py));
            dict.insert("tardis_enabled".to_string(), self.config.data_sources.tardis.enabled.to_object(py));
            dict.insert("polymarket_enabled".to_string(), self.config.data_sources.polymarket.enabled.to_object(py));
            
            // Performance
            dict.insert("gpu_enabled".to_string(), self.config.performance.gpu.enabled.to_object(py));
            
            Ok(dict)
        })
    }
}

#[cfg(feature = "python")]
impl From<PortfolioMetrics> for PyPortfolioMetrics {
    fn from(metrics: PortfolioMetrics) -> Self {
        Self {
            total_value: metrics.total_value,
            cash_balance: metrics.cash_balance,
            unrealized_pnl: metrics.unrealized_pnl,
            realized_pnl: metrics.realized_pnl,
            return_percentage: metrics.return_percentage,
            max_drawdown: metrics.max_drawdown,
            sharpe_ratio: metrics.sharpe_ratio,
            var_95: metrics.var_95,
            open_positions: metrics.open_positions,
            volatility: metrics.volatility,
        }
    }
}

#[cfg(feature = "python")]
impl From<Position> for PyPosition {
    fn from(position: Position) -> Self {
        let symbol = position.symbol.clone();
        let side = format!("{:?}", position.side);
        let size = position.size;
        let entry_price = position.entry_price;
        let current_price = position.current_price;
        let unrealized_pnl = position.unrealized_pnl;
        let value = position.value();
        
        Self {
            symbol,
            side,
            size,
            entry_price,
            current_price,
            unrealized_pnl,
            value,
        }
    }
}

#[cfg(feature = "python")]
impl From<TradingSession> for PyTradingSession {
    fn from(session: TradingSession) -> Self {
        Self {
            session_id: session.id.to_string(),
            start_time: session.start_time.to_rfc3339(),
            session_pnl: session.session_pnl,
            trade_count: session.trade_count,
            win_rate: session.win_rate,
            max_drawdown: session.max_drawdown,
        }
    }
}

/// Utility functions for Python integration
#[cfg(feature = "python")]
#[pyfunction]
fn create_default_config() -> PyTengriConfig {
    PyTengriConfig::new()
}

#[cfg(feature = "python")]
#[pyfunction]
fn validate_config_file(path: String) -> PyResult<bool> {
    match TengriConfig::from_file(path) {
        Ok(config) => {
            match config.validate() {
                Ok(_) => Ok(true),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Config validation failed: {}", e))),
            }
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to load config: {}", e))),
    }
}

#[cfg(feature = "python")]
#[pyfunction]
fn get_version() -> String {
    crate::VERSION.to_string()
}

#[cfg(feature = "python")]
#[pyfunction]
fn get_supported_exchanges() -> Vec<String> {
    vec!["binance_spot".to_string(), "binance_futures".to_string()]
}

#[cfg(feature = "python")]
#[pyfunction]
fn get_supported_data_sources() -> Vec<String> {
    vec!["databento".to_string(), "tardis".to_string(), "polymarket".to_string()]
}

/// Python module definition
#[cfg(feature = "python")]
#[pymodule]
fn tengri(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyTengriStrategy>()?;
    m.add_class::<PyTengriConfig>()?;
    m.add_class::<PyPortfolioMetrics>()?;
    m.add_class::<PyPosition>()?;
    m.add_class::<PyTradingSession>()?;
    
    m.add_function(wrap_pyfunction!(create_default_config, m)?)?;
    m.add_function(wrap_pyfunction!(validate_config_file, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(get_supported_exchanges, m)?)?;
    m.add_function(wrap_pyfunction!(get_supported_data_sources, m)?)?;
    
    m.add("__version__", crate::VERSION)?;
    
    Ok(())
}

// If Python feature is not enabled, provide empty implementations
#[cfg(not(feature = "python"))]
pub struct PyTengriStrategy;

#[cfg(not(feature = "python"))]
pub struct PyTengriConfig;

#[cfg(not(feature = "python"))]
impl PyTengriStrategy {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(not(feature = "python"))]
impl PyTengriConfig {
    pub fn new() -> Self {
        Self
    }
}