//! Python bindings for FreqTrade integration using PyO3
//! 
//! Provides a Python interface to the Rust market analysis engine for seamless
//! integration with FreqTrade strategies and Python-based trading systems.

#![cfg(feature = "python")]

use crate::{
    MarketAnalyzer, Config, MarketData, MarketAnalysis, WhaleSignal, RegimeInfo,
    Pattern, Predictions, MicrostructureAnalysis, RiskMetrics, AnalysisSignal,
    Trade, TradeSide, TradeType, OrderBook, OrderBookLevel, Timeframe,
    error::{AnalysisError, Result},
    data::{DataPipeline, RawMarketData},
    performance::PerformanceMonitor,
};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use numpy::{PyArray1, PyArray2, ToPyArray};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use tokio::runtime::Runtime;
use std::sync::{Arc, Mutex};

/// Python wrapper for the MarketAnalyzer
#[pyclass(name = "MarketAnalyzer")]
#[derive(Debug)]
pub struct PyMarketAnalyzer {
    analyzer: Arc<Mutex<MarketAnalyzer>>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PyMarketAnalyzer {
    #[new]
    #[pyo3(signature = (config_dict = None))]
    fn new(config_dict: Option<&PyDict>) -> PyResult<Self> {
        let config = if let Some(dict) = config_dict {
            parse_config_from_dict(dict)?
        } else {
            Config::default()
        };
        
        let analyzer = MarketAnalyzer::new(config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create analyzer: {}", e)))?;
        
        let runtime = Arc::new(Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?);
        
        Ok(Self {
            analyzer: Arc::new(Mutex::new(analyzer)),
            runtime,
        })
    }
    
    /// Analyze market data and return comprehensive analysis
    #[pyo3(signature = (symbol, timestamp, prices, volumes, trades = None, order_book = None, metadata = None))]
    fn analyze_market(
        &self,
        symbol: String,
        timestamp: f64, // Unix timestamp
        prices: &PyArray1<f64>,
        volumes: &PyArray1<f64>,
        trades: Option<&PyList>,
        order_book: Option<&PyDict>,
        metadata: Option<&PyDict>,
    ) -> PyResult<PyMarketAnalysis> {
        let market_data = self.create_market_data(
            symbol, timestamp, prices, volumes, trades, order_book, metadata
        )?;
        
        let analyzer = self.analyzer.lock().unwrap();
        let analysis = self.runtime.block_on(async {
            analyzer.analyze_market(&market_data).await
        }).map_err(|e| PyRuntimeError::new_err(format!("Analysis failed: {}", e)))?;
        
        Ok(PyMarketAnalysis::from_rust(analysis))
    }
    
    /// Get current market state
    fn get_market_state(&self) -> PyResult<PyDict> {
        let analyzer = self.analyzer.lock().unwrap();
        let state = analyzer.get_market_state();
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            if let Some(regime) = state.current_regime {
                dict.set_item("current_regime", format!("{:?}", regime))?;
            }
            
            if let Some(vol_regime) = state.volatility_regime {
                dict.set_item("volatility_regime", format!("{:?}", vol_regime))?;
            }
            
            dict.set_item("trend_strength", state.trend_strength)?;
            dict.set_item("market_stress_level", state.market_stress_level)?;
            
            Ok(dict.into())
        })
    }
    
    /// Get performance metrics
    fn get_metrics(&self) -> PyResult<PyDict> {
        let analyzer = self.analyzer.lock().unwrap();
        let metrics = analyzer.get_metrics();
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            dict.set_item("total_analyses", metrics.total_analyses)?;
            dict.set_item("average_processing_time", metrics.average_processing_time)?;
            dict.set_item("cache_hit_rate", metrics.cache_hit_rate)?;
            dict.set_item("prediction_accuracy", metrics.prediction_accuracy)?;
            dict.set_item("whale_detection_accuracy", metrics.whale_detection_accuracy)?;
            dict.set_item("pattern_recognition_accuracy", metrics.pattern_recognition_accuracy)?;
            dict.set_item("regime_detection_accuracy", metrics.regime_detection_accuracy)?;
            dict.set_item("high_confidence_analyses", metrics.high_confidence_analyses)?;
            
            Ok(dict.into())
        })
    }
    
    /// Subscribe to analysis signals (returns a signal receiver)
    fn subscribe_signals(&self) -> PyResult<PySignalReceiver> {
        let analyzer = self.analyzer.lock().unwrap();
        let receiver = analyzer.subscribe_signals();
        
        Ok(PySignalReceiver {
            receiver: Arc::new(Mutex::new(receiver)),
            runtime: Arc::clone(&self.runtime),
        })
    }
    
    /// Update models with feedback for online learning
    fn update_models(&self, feedback_dict: &PyDict) -> PyResult<()> {
        let feedback = parse_model_feedback_from_dict(feedback_dict)?;
        
        let mut analyzer = self.analyzer.lock().unwrap();
        self.runtime.block_on(async {
            analyzer.update_models(feedback).await
        }).map_err(|e| PyRuntimeError::new_err(format!("Model update failed: {}", e)))?;
        
        Ok(())
    }
    
    // Helper methods
    
    fn create_market_data(
        &self,
        symbol: String,
        timestamp: f64,
        prices: &PyArray1<f64>,
        volumes: &PyArray1<f64>,
        trades: Option<&PyList>,
        order_book: Option<&PyDict>,
        metadata: Option<&PyDict>,
    ) -> PyResult<MarketData> {
        let timestamp_dt = DateTime::from_timestamp(timestamp as i64, 0)
            .ok_or_else(|| PyValueError::new_err("Invalid timestamp"))?;
        
        let prices_vec = prices.to_vec().map_err(|e| PyRuntimeError::new_err(format!("Failed to convert prices: {}", e)))?;
        let volumes_vec = volumes.to_vec().map_err(|e| PyRuntimeError::new_err(format!("Failed to convert volumes: {}", e)))?;
        
        let mut market_data = MarketData::new(symbol, Timeframe::OneMinute);
        market_data.timestamp = timestamp_dt;
        market_data.prices = prices_vec;
        market_data.volumes = volumes_vec;
        
        // Parse trades if provided
        if let Some(trades_list) = trades {
            for trade_obj in trades_list.iter() {
                let trade = parse_trade_from_py(trade_obj)?;
                market_data.add_trade(trade);
            }
        }
        
        // Parse order book if provided
        if let Some(order_book_dict) = order_book {
            let order_book = parse_order_book_from_dict(order_book_dict)?;
            market_data.set_order_book(order_book);
        }
        
        // Parse metadata if provided
        if let Some(metadata_dict) = metadata {
            market_data.metadata = parse_metadata_from_dict(metadata_dict)?;
        }
        
        Ok(market_data)
    }
}

/// Python wrapper for analysis results
#[pyclass(name = "MarketAnalysis")]
#[derive(Debug, Clone)]
pub struct PyMarketAnalysis {
    pub timestamp: f64,
    pub symbol: String,
    pub whale_signals: Vec<PyWhaleSignal>,
    pub regime_info: PyRegimeInfo,
    pub patterns: Vec<PyPattern>,
    pub predictions: PyPredictions,
    pub microstructure: PyMicrostructureAnalysis,
    pub confidence_score: f64,
    pub risk_metrics: PyRiskMetrics,
}

impl PyMarketAnalysis {
    fn from_rust(analysis: MarketAnalysis) -> Self {
        Self {
            timestamp: analysis.timestamp.timestamp() as f64,
            symbol: analysis.symbol,
            whale_signals: analysis.whale_signals.into_iter().map(PyWhaleSignal::from_rust).collect(),
            regime_info: PyRegimeInfo::from_rust(analysis.regime_info),
            patterns: analysis.patterns.into_iter().map(PyPattern::from_rust).collect(),
            predictions: PyPredictions::from_rust(analysis.predictions),
            microstructure: PyMicrostructureAnalysis::from_rust(analysis.microstructure),
            confidence_score: analysis.confidence_score,
            risk_metrics: PyRiskMetrics::from_rust(analysis.risk_metrics),
        }
    }
}

#[pymethods]
impl PyMarketAnalysis {
    #[getter]
    fn timestamp(&self) -> f64 { self.timestamp }
    
    #[getter]
    fn symbol(&self) -> String { self.symbol.clone() }
    
    #[getter]
    fn whale_signals(&self) -> Vec<PyWhaleSignal> { self.whale_signals.clone() }
    
    #[getter]
    fn regime_info(&self) -> PyRegimeInfo { self.regime_info.clone() }
    
    #[getter]
    fn patterns(&self) -> Vec<PyPattern> { self.patterns.clone() }
    
    #[getter]
    fn predictions(&self) -> PyPredictions { self.predictions.clone() }
    
    #[getter]
    fn microstructure(&self) -> PyMicrostructureAnalysis { self.microstructure.clone() }
    
    #[getter]
    fn confidence_score(&self) -> f64 { self.confidence_score }
    
    #[getter]
    fn risk_metrics(&self) -> PyRiskMetrics { self.risk_metrics.clone() }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        dict.set_item("timestamp", self.timestamp)?;
        dict.set_item("symbol", &self.symbol)?;
        dict.set_item("confidence_score", self.confidence_score)?;
        
        // Convert whale signals to list of dicts
        let whale_signals_list = PyList::empty(py);
        for signal in &self.whale_signals {
            whale_signals_list.append(signal.to_dict(py)?)?;
        }
        dict.set_item("whale_signals", whale_signals_list)?;
        
        dict.set_item("regime_info", self.regime_info.to_dict(py)?)?;
        dict.set_item("predictions", self.predictions.to_dict(py)?)?;
        dict.set_item("risk_metrics", self.risk_metrics.to_dict(py)?)?;
        
        Ok(dict.into())
    }
}

/// Python wrapper for whale signals
#[pyclass(name = "WhaleSignal")]
#[derive(Debug, Clone)]
pub struct PyWhaleSignal {
    pub signal_type: String,
    pub strength: f64,
    pub confidence: f64,
    pub timestamp: f64,
}

impl PyWhaleSignal {
    fn from_rust(signal: WhaleSignal) -> Self {
        Self {
            signal_type: format!("{:?}", signal.signal_type),
            strength: signal.strength,
            confidence: signal.confidence,
            timestamp: signal.timestamp.timestamp() as f64,
        }
    }
}

#[pymethods]
impl PyWhaleSignal {
    #[getter]
    fn signal_type(&self) -> String { self.signal_type.clone() }
    
    #[getter]
    fn strength(&self) -> f64 { self.strength }
    
    #[getter]
    fn confidence(&self) -> f64 { self.confidence }
    
    #[getter]
    fn timestamp(&self) -> f64 { self.timestamp }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("signal_type", &self.signal_type)?;
        dict.set_item("strength", self.strength)?;
        dict.set_item("confidence", self.confidence)?;
        dict.set_item("timestamp", self.timestamp)?;
        Ok(dict.into())
    }
}

/// Python wrapper for regime information
#[pyclass(name = "RegimeInfo")]
#[derive(Debug, Clone)]
pub struct PyRegimeInfo {
    pub current_regime: String,
    pub confidence: f64,
    pub regime_duration_minutes: i64,
    pub volatility_regime: String,
}

impl PyRegimeInfo {
    fn from_rust(info: RegimeInfo) -> Self {
        Self {
            current_regime: format!("{:?}", info.current_regime),
            confidence: info.confidence,
            regime_duration_minutes: info.regime_duration.num_minutes(),
            volatility_regime: format!("{:?}", info.volatility_regime),
        }
    }
}

#[pymethods]
impl PyRegimeInfo {
    #[getter]
    fn current_regime(&self) -> String { self.current_regime.clone() }
    
    #[getter]
    fn confidence(&self) -> f64 { self.confidence }
    
    #[getter]
    fn regime_duration_minutes(&self) -> i64 { self.regime_duration_minutes }
    
    #[getter]
    fn volatility_regime(&self) -> String { self.volatility_regime.clone() }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("current_regime", &self.current_regime)?;
        dict.set_item("confidence", self.confidence)?;
        dict.set_item("regime_duration_minutes", self.regime_duration_minutes)?;
        dict.set_item("volatility_regime", &self.volatility_regime)?;
        Ok(dict.into())
    }
}

/// Python wrapper for patterns
#[pyclass(name = "Pattern")]
#[derive(Debug, Clone)]
pub struct PyPattern {
    pub pattern_type: String,
    pub confidence: f64,
    pub start_time: f64,
    pub end_time: Option<f64>,
    pub price_levels: Vec<f64>,
    pub volume_confirmation: bool,
    pub breakout_target: Option<f64>,
    pub stop_loss: Option<f64>,
}

impl PyPattern {
    fn from_rust(pattern: Pattern) -> Self {
        Self {
            pattern_type: format!("{:?}", pattern.pattern_type),
            confidence: pattern.confidence,
            start_time: pattern.start_time.timestamp() as f64,
            end_time: pattern.end_time.map(|t| t.timestamp() as f64),
            price_levels: pattern.price_levels,
            volume_confirmation: pattern.volume_confirmation,
            breakout_target: pattern.breakout_target,
            stop_loss: pattern.stop_loss,
        }
    }
}

#[pymethods]
impl PyPattern {
    #[getter]
    fn pattern_type(&self) -> String { self.pattern_type.clone() }
    
    #[getter]
    fn confidence(&self) -> f64 { self.confidence }
    
    #[getter]
    fn start_time(&self) -> f64 { self.start_time }
    
    #[getter]
    fn end_time(&self) -> Option<f64> { self.end_time }
    
    #[getter]
    fn price_levels(&self) -> Vec<f64> { self.price_levels.clone() }
    
    #[getter]
    fn volume_confirmation(&self) -> bool { self.volume_confirmation }
    
    #[getter]
    fn breakout_target(&self) -> Option<f64> { self.breakout_target }
    
    #[getter]
    fn stop_loss(&self) -> Option<f64> { self.stop_loss }
}

/// Python wrapper for predictions
#[pyclass(name = "Predictions")]
#[derive(Debug, Clone)]
pub struct PyPredictions {
    pub short_term: Vec<PyPricePrediction>,
    pub medium_term: Vec<PyPricePrediction>,
    pub volatility_forecast: PyVolatilityForecast,
}

impl PyPredictions {
    fn from_rust(predictions: Predictions) -> Self {
        Self {
            short_term: predictions.short_term.into_iter().map(PyPricePrediction::from_rust).collect(),
            medium_term: predictions.medium_term.into_iter().map(PyPricePrediction::from_rust).collect(),
            volatility_forecast: PyVolatilityForecast::from_rust(predictions.volatility_forecast),
        }
    }
}

#[pymethods]
impl PyPredictions {
    #[getter]
    fn short_term(&self) -> Vec<PyPricePrediction> { self.short_term.clone() }
    
    #[getter]
    fn medium_term(&self) -> Vec<PyPricePrediction> { self.medium_term.clone() }
    
    #[getter]
    fn volatility_forecast(&self) -> PyVolatilityForecast { self.volatility_forecast.clone() }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        let short_term_list = PyList::empty(py);
        for pred in &self.short_term {
            short_term_list.append(pred.to_dict(py)?)?;
        }
        dict.set_item("short_term", short_term_list)?;
        
        let medium_term_list = PyList::empty(py);
        for pred in &self.medium_term {
            medium_term_list.append(pred.to_dict(py)?)?;
        }
        dict.set_item("medium_term", medium_term_list)?;
        
        dict.set_item("volatility_forecast", self.volatility_forecast.to_dict(py)?)?;
        
        Ok(dict.into())
    }
}

/// Python wrapper for price predictions
#[pyclass(name = "PricePrediction")]
#[derive(Debug, Clone)]
pub struct PyPricePrediction {
    pub horizon_minutes: i64,
    pub predicted_price: f64,
    pub confidence_interval_low: f64,
    pub confidence_interval_high: f64,
    pub model_uncertainty: f64,
}

impl PyPricePrediction {
    fn from_rust(prediction: crate::types::PricePrediction) -> Self {
        Self {
            horizon_minutes: prediction.horizon.num_minutes(),
            predicted_price: prediction.predicted_price,
            confidence_interval_low: prediction.confidence_interval.0,
            confidence_interval_high: prediction.confidence_interval.1,
            model_uncertainty: prediction.model_uncertainty,
        }
    }
}

#[pymethods]
impl PyPricePrediction {
    #[getter]
    fn horizon_minutes(&self) -> i64 { self.horizon_minutes }
    
    #[getter]
    fn predicted_price(&self) -> f64 { self.predicted_price }
    
    #[getter]
    fn confidence_interval_low(&self) -> f64 { self.confidence_interval_low }
    
    #[getter]
    fn confidence_interval_high(&self) -> f64 { self.confidence_interval_high }
    
    #[getter]
    fn model_uncertainty(&self) -> f64 { self.model_uncertainty }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("horizon_minutes", self.horizon_minutes)?;
        dict.set_item("predicted_price", self.predicted_price)?;
        dict.set_item("confidence_interval_low", self.confidence_interval_low)?;
        dict.set_item("confidence_interval_high", self.confidence_interval_high)?;
        dict.set_item("model_uncertainty", self.model_uncertainty)?;
        Ok(dict.into())
    }
}

/// Python wrapper for volatility forecast
#[pyclass(name = "VolatilityForecast")]
#[derive(Debug, Clone)]
pub struct PyVolatilityForecast {
    pub current_volatility: f64,
    pub forecasted_volatility: Vec<(i64, f64)>, // (minutes, volatility)
    pub volatility_regime_change_probability: f64,
}

impl PyVolatilityForecast {
    fn from_rust(forecast: crate::types::VolatilityForecast) -> Self {
        Self {
            current_volatility: forecast.current_volatility,
            forecasted_volatility: forecast.forecasted_volatility
                .into_iter()
                .map(|(duration, vol)| (duration.num_minutes(), vol))
                .collect(),
            volatility_regime_change_probability: forecast.volatility_regime_change_probability,
        }
    }
}

#[pymethods]
impl PyVolatilityForecast {
    #[getter]
    fn current_volatility(&self) -> f64 { self.current_volatility }
    
    #[getter]
    fn forecasted_volatility(&self) -> Vec<(i64, f64)> { self.forecasted_volatility.clone() }
    
    #[getter]
    fn volatility_regime_change_probability(&self) -> f64 { self.volatility_regime_change_probability }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("current_volatility", self.current_volatility)?;
        dict.set_item("forecasted_volatility", &self.forecasted_volatility)?;
        dict.set_item("volatility_regime_change_probability", self.volatility_regime_change_probability)?;
        Ok(dict.into())
    }
}

/// Python wrapper for microstructure analysis
#[pyclass(name = "MicrostructureAnalysis")]
#[derive(Debug, Clone)]
pub struct PyMicrostructureAnalysis {
    pub bid_ask_spread: f64,
    pub total_bid_depth: f64,
    pub total_ask_depth: f64,
    pub depth_imbalance: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub net_flow: f64,
    pub market_resilience: f64,
}

impl PyMicrostructureAnalysis {
    fn from_rust(analysis: MicrostructureAnalysis) -> Self {
        Self {
            bid_ask_spread: analysis.bid_ask_spread,
            total_bid_depth: analysis.market_depth.total_bid_depth,
            total_ask_depth: analysis.market_depth.total_ask_depth,
            depth_imbalance: analysis.market_depth.depth_imbalance,
            buy_volume: analysis.order_flow.buy_volume,
            sell_volume: analysis.order_flow.sell_volume,
            net_flow: analysis.order_flow.net_flow,
            market_resilience: analysis.market_depth.market_resilience,
        }
    }
}

#[pymethods]
impl PyMicrostructureAnalysis {
    #[getter]
    fn bid_ask_spread(&self) -> f64 { self.bid_ask_spread }
    
    #[getter]
    fn total_bid_depth(&self) -> f64 { self.total_bid_depth }
    
    #[getter]
    fn total_ask_depth(&self) -> f64 { self.total_ask_depth }
    
    #[getter]
    fn depth_imbalance(&self) -> f64 { self.depth_imbalance }
    
    #[getter]
    fn buy_volume(&self) -> f64 { self.buy_volume }
    
    #[getter]
    fn sell_volume(&self) -> f64 { self.sell_volume }
    
    #[getter]
    fn net_flow(&self) -> f64 { self.net_flow }
    
    #[getter]
    fn market_resilience(&self) -> f64 { self.market_resilience }
}

/// Python wrapper for risk metrics
#[pyclass(name = "RiskMetrics")]
#[derive(Debug, Clone)]
pub struct PyRiskMetrics {
    pub value_at_risk_95: f64,
    pub expected_shortfall_95: f64,
    pub maximum_drawdown: f64,
    pub volatility_regime: String,
    pub tail_ratio: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl PyRiskMetrics {
    fn from_rust(metrics: RiskMetrics) -> Self {
        Self {
            value_at_risk_95: metrics.value_at_risk_95,
            expected_shortfall_95: metrics.expected_shortfall_95,
            maximum_drawdown: metrics.maximum_drawdown,
            volatility_regime: metrics.volatility_regime,
            tail_ratio: metrics.tail_ratio,
            skewness: metrics.skewness,
            kurtosis: metrics.kurtosis,
        }
    }
}

#[pymethods]
impl PyRiskMetrics {
    #[getter]
    fn value_at_risk_95(&self) -> f64 { self.value_at_risk_95 }
    
    #[getter]
    fn expected_shortfall_95(&self) -> f64 { self.expected_shortfall_95 }
    
    #[getter]
    fn maximum_drawdown(&self) -> f64 { self.maximum_drawdown }
    
    #[getter]
    fn volatility_regime(&self) -> String { self.volatility_regime.clone() }
    
    #[getter]
    fn tail_ratio(&self) -> f64 { self.tail_ratio }
    
    #[getter]
    fn skewness(&self) -> f64 { self.skewness }
    
    #[getter]
    fn kurtosis(&self) -> f64 { self.kurtosis }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("value_at_risk_95", self.value_at_risk_95)?;
        dict.set_item("expected_shortfall_95", self.expected_shortfall_95)?;
        dict.set_item("maximum_drawdown", self.maximum_drawdown)?;
        dict.set_item("volatility_regime", &self.volatility_regime)?;
        dict.set_item("tail_ratio", self.tail_ratio)?;
        dict.set_item("skewness", self.skewness)?;
        dict.set_item("kurtosis", self.kurtosis)?;
        Ok(dict.into())
    }
}

/// Python wrapper for signal receiver
#[pyclass(name = "SignalReceiver")]
pub struct PySignalReceiver {
    receiver: Arc<Mutex<tokio::sync::broadcast::Receiver<AnalysisSignal>>>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PySignalReceiver {
    /// Try to receive a signal (non-blocking)
    fn try_recv(&self) -> PyResult<Option<PyDict>> {
        let mut receiver = self.receiver.lock().unwrap();
        
        match receiver.try_recv() {
            Ok(signal) => {
                Python::with_gil(|py| {
                    let dict = signal_to_dict(py, &signal)?;
                    Ok(Some(dict))
                })
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!("Signal receive error: {}", e))),
        }
    }
    
    /// Receive a signal (blocking with timeout)
    fn recv_timeout(&self, timeout_ms: u64) -> PyResult<Option<PyDict>> {
        let mut receiver = self.receiver.lock().unwrap();
        
        let result = self.runtime.block_on(async {
            tokio::time::timeout(
                tokio::time::Duration::from_millis(timeout_ms),
                receiver.recv()
            ).await
        });
        
        match result {
            Ok(Ok(signal)) => {
                Python::with_gil(|py| {
                    let dict = signal_to_dict(py, &signal)?;
                    Ok(Some(dict))
                })
            }
            Ok(Err(_)) => Err(PyRuntimeError::new_err("Signal channel closed")),
            Err(_) => Ok(None), // Timeout
        }
    }
}

/// Python wrapper for data pipeline
#[pyclass(name = "DataPipeline")]
pub struct PyDataPipeline {
    pipeline: Arc<Mutex<DataPipeline>>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PyDataPipeline {
    #[new]
    #[pyo3(signature = (config_dict = None))]
    fn new(config_dict: Option<&PyDict>) -> PyResult<Self> {
        let config = if let Some(dict) = config_dict {
            parse_config_from_dict(dict)?
        } else {
            Config::default()
        };
        
        let pipeline = DataPipeline::new(config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create pipeline: {}", e)))?;
        
        let runtime = Arc::new(Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?);
        
        Ok(Self {
            pipeline: Arc::new(Mutex::new(pipeline)),
            runtime,
        })
    }
    
    /// Start the data pipeline
    fn start(&self) -> PyResult<()> {
        let pipeline = self.pipeline.lock().unwrap();
        self.runtime.block_on(async {
            pipeline.start().await
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to start pipeline: {}", e)))?;
        
        Ok(())
    }
    
    /// Stop the data pipeline
    fn stop(&self) -> PyResult<()> {
        let pipeline = self.pipeline.lock().unwrap();
        self.runtime.block_on(async {
            pipeline.stop().await
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to stop pipeline: {}", e)))?;
        
        Ok(())
    }
    
    /// Ingest raw market data
    #[pyo3(signature = (symbol, timestamp, price, volume, bid = None, ask = None, trade_id = None, trade_side = None, metadata = None))]
    fn ingest(
        &self,
        symbol: String,
        timestamp: f64,
        price: f64,
        volume: f64,
        bid: Option<f64>,
        ask: Option<f64>,
        trade_id: Option<String>,
        trade_side: Option<String>,
        metadata: Option<&PyDict>,
    ) -> PyResult<()> {
        let timestamp_dt = DateTime::from_timestamp(timestamp as i64, 0)
            .ok_or_else(|| PyValueError::new_err("Invalid timestamp"))?;
        
        let side = if let Some(side_str) = trade_side {
            Some(match side_str.as_str() {
                "buy" | "Buy" => TradeSide::Buy,
                "sell" | "Sell" => TradeSide::Sell,
                _ => return Err(PyValueError::new_err("Invalid trade side")),
            })
        } else {
            None
        };
        
        let metadata_map = if let Some(meta_dict) = metadata {
            parse_metadata_from_dict(meta_dict)?
        } else {
            HashMap::new()
        };
        
        let raw_data = RawMarketData {
            symbol,
            timestamp: timestamp_dt,
            price,
            volume,
            bid,
            ask,
            trade_id,
            trade_side: side,
            metadata: metadata_map,
        };
        
        let pipeline = self.pipeline.lock().unwrap();
        self.runtime.block_on(async {
            pipeline.ingest(raw_data).await
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to ingest data: {}", e)))?;
        
        Ok(())
    }
    
    /// Get pipeline metrics
    fn get_metrics(&self) -> PyResult<PyDict> {
        let pipeline = self.pipeline.lock().unwrap();
        let metrics = pipeline.get_metrics();
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            dict.set_item("total_messages_received", metrics.total_messages_received)?;
            dict.set_item("total_messages_processed", metrics.total_messages_processed)?;
            dict.set_item("total_messages_dropped", metrics.total_messages_dropped)?;
            dict.set_item("average_processing_latency_ms", metrics.average_processing_latency_ms)?;
            dict.set_item("peak_processing_latency_ms", metrics.peak_processing_latency_ms)?;
            dict.set_item("buffer_utilization", metrics.buffer_utilization)?;
            dict.set_item("throughput_per_second", metrics.throughput_per_second)?;
            dict.set_item("error_count", metrics.error_count)?;
            
            Ok(dict.into())
        })
    }
}

// Helper functions for parsing Python objects

fn parse_config_from_dict(dict: &PyDict) -> PyResult<Config> {
    // For now, return default config
    // In a full implementation, you would parse each configuration section
    Ok(Config::default())
}

fn parse_model_feedback_from_dict(dict: &PyDict) -> PyResult<crate::types::ModelFeedback> {
    // Simplified feedback parsing
    use crate::types::*;
    
    Ok(ModelFeedback {
        whale_feedback: WhaleFeedback {
            true_positives: dict.get_item("whale_true_positives")
                .and_then(|v| v.extract().ok()).unwrap_or(0),
            false_positives: dict.get_item("whale_false_positives")
                .and_then(|v| v.extract().ok()).unwrap_or(0),
            false_negatives: dict.get_item("whale_false_negatives")
                .and_then(|v| v.extract().ok()).unwrap_or(0),
            parameter_adjustments: HashMap::new(),
        },
        regime_feedback: RegimeFeedback {
            correct_classifications: dict.get_item("regime_correct")
                .and_then(|v| v.extract().ok()).unwrap_or(0),
            incorrect_classifications: dict.get_item("regime_incorrect")
                .and_then(|v| v.extract().ok()).unwrap_or(0),
            regime_transition_accuracy: dict.get_item("regime_transition_accuracy")
                .and_then(|v| v.extract().ok()).unwrap_or(0.0),
            model_updates: HashMap::new(),
        },
        pattern_feedback: PatternFeedback {
            pattern_accuracy: HashMap::new(),
            breakout_success_rate: dict.get_item("breakout_success_rate")
                .and_then(|v| v.extract().ok()).unwrap_or(0.0),
            false_pattern_rate: dict.get_item("false_pattern_rate")
                .and_then(|v| v.extract().ok()).unwrap_or(0.0),
            weight_adjustments: HashMap::new(),
        },
        prediction_feedback: PredictionFeedback {
            price_prediction_accuracy: dict.get_item("price_prediction_accuracy")
                .and_then(|v| v.extract().ok()).unwrap_or(0.0),
            volatility_prediction_accuracy: dict.get_item("volatility_prediction_accuracy")
                .and_then(|v| v.extract().ok()).unwrap_or(0.0),
            directional_accuracy: dict.get_item("directional_accuracy")
                .and_then(|v| v.extract().ok()).unwrap_or(0.0),
            model_performance_metrics: HashMap::new(),
        },
    })
}

fn parse_trade_from_py(trade_obj: &PyAny) -> PyResult<Trade> {
    let dict = trade_obj.downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("Trade must be a dictionary"))?;
    
    let id = dict.get_item("id")
        .ok_or_else(|| PyValueError::new_err("Trade ID is required"))?
        .extract::<String>()?;
    
    let timestamp = dict.get_item("timestamp")
        .ok_or_else(|| PyValueError::new_err("Trade timestamp is required"))?
        .extract::<f64>()?;
    
    let price = dict.get_item("price")
        .ok_or_else(|| PyValueError::new_err("Trade price is required"))?
        .extract::<f64>()?;
    
    let quantity = dict.get_item("quantity")
        .ok_or_else(|| PyValueError::new_err("Trade quantity is required"))?
        .extract::<f64>()?;
    
    let side_str = dict.get_item("side")
        .ok_or_else(|| PyValueError::new_err("Trade side is required"))?
        .extract::<String>()?;
    
    let side = match side_str.as_str() {
        "buy" | "Buy" => TradeSide::Buy,
        "sell" | "Sell" => TradeSide::Sell,
        _ => return Err(PyValueError::new_err("Invalid trade side")),
    };
    
    let trade_type = dict.get_item("type")
        .and_then(|v| v.extract::<String>().ok())
        .map(|type_str| match type_str.as_str() {
            "market" => TradeType::Market,
            "limit" => TradeType::Limit,
            "stop" => TradeType::Stop,
            _ => TradeType::Market,
        })
        .unwrap_or(TradeType::Market);
    
    let timestamp_dt = DateTime::from_timestamp(timestamp as i64, 0)
        .ok_or_else(|| PyValueError::new_err("Invalid timestamp"))?;
    
    Ok(Trade {
        id,
        timestamp: timestamp_dt,
        price,
        quantity,
        side,
        trade_type,
    })
}

fn parse_order_book_from_dict(dict: &PyDict) -> PyResult<OrderBook> {
    let timestamp = dict.get_item("timestamp")
        .ok_or_else(|| PyValueError::new_err("Order book timestamp is required"))?
        .extract::<f64>()?;
    
    let timestamp_dt = DateTime::from_timestamp(timestamp as i64, 0)
        .ok_or_else(|| PyValueError::new_err("Invalid timestamp"))?;
    
    let bids_list = dict.get_item("bids")
        .ok_or_else(|| PyValueError::new_err("Order book bids are required"))?
        .downcast::<PyList>()?;
    
    let asks_list = dict.get_item("asks")
        .ok_or_else(|| PyValueError::new_err("Order book asks are required"))?
        .downcast::<PyList>()?;
    
    let mut bids = Vec::new();
    for bid_item in bids_list.iter() {
        let bid_tuple = bid_item.downcast::<PyTuple>()?;
        if bid_tuple.len() >= 2 {
            let price = bid_tuple.get_item(0)?.extract::<f64>()?;
            let quantity = bid_tuple.get_item(1)?.extract::<f64>()?;
            bids.push(OrderBookLevel {
                price,
                quantity,
                order_count: None,
            });
        }
    }
    
    let mut asks = Vec::new();
    for ask_item in asks_list.iter() {
        let ask_tuple = ask_item.downcast::<PyTuple>()?;
        if ask_tuple.len() >= 2 {
            let price = ask_tuple.get_item(0)?.extract::<f64>()?;
            let quantity = ask_tuple.get_item(1)?.extract::<f64>()?;
            asks.push(OrderBookLevel {
                price,
                quantity,
                order_count: None,
            });
        }
    }
    
    let sequence = dict.get_item("sequence")
        .and_then(|v| v.extract().ok())
        .unwrap_or(0);
    
    Ok(OrderBook {
        timestamp: timestamp_dt,
        bids,
        asks,
        sequence,
    })
}

fn parse_metadata_from_dict(dict: &PyDict) -> PyResult<HashMap<String, serde_json::Value>> {
    let mut metadata = HashMap::new();
    
    for (key, value) in dict.iter() {
        let key_str = key.extract::<String>()?;
        
        let json_value = if let Ok(s) = value.extract::<String>() {
            serde_json::Value::String(s)
        } else if let Ok(f) = value.extract::<f64>() {
            serde_json::Value::Number(serde_json::Number::from_f64(f).unwrap())
        } else if let Ok(i) = value.extract::<i64>() {
            serde_json::Value::Number(serde_json::Number::from(i))
        } else if let Ok(b) = value.extract::<bool>() {
            serde_json::Value::Bool(b)
        } else {
            serde_json::Value::Null
        };
        
        metadata.insert(key_str, json_value);
    }
    
    Ok(metadata)
}

fn signal_to_dict(py: Python, signal: &AnalysisSignal) -> PyResult<PyDict> {
    let dict = PyDict::new(py);
    
    match signal {
        AnalysisSignal::WhaleActivity { symbol, signal, timestamp } => {
            dict.set_item("type", "whale_activity")?;
            dict.set_item("symbol", symbol)?;
            dict.set_item("signal_type", format!("{:?}", signal.signal_type))?;
            dict.set_item("strength", signal.strength)?;
            dict.set_item("confidence", signal.confidence)?;
            dict.set_item("timestamp", timestamp.timestamp())?;
        }
        AnalysisSignal::RegimeChange { symbol, old_regime, new_regime, confidence, timestamp } => {
            dict.set_item("type", "regime_change")?;
            dict.set_item("symbol", symbol)?;
            if let Some(old) = old_regime {
                dict.set_item("old_regime", format!("{:?}", old))?;
            }
            dict.set_item("new_regime", format!("{:?}", new_regime))?;
            dict.set_item("confidence", confidence)?;
            dict.set_item("timestamp", timestamp.timestamp())?;
        }
        AnalysisSignal::PatternDetected { symbol, pattern, timestamp } => {
            dict.set_item("type", "pattern_detected")?;
            dict.set_item("symbol", symbol)?;
            dict.set_item("pattern_type", format!("{:?}", pattern.pattern_type))?;
            dict.set_item("confidence", pattern.confidence)?;
            dict.set_item("timestamp", timestamp.timestamp())?;
        }
        AnalysisSignal::RiskAlert { symbol, risk_type, severity, timestamp } => {
            dict.set_item("type", "risk_alert")?;
            dict.set_item("symbol", symbol)?;
            dict.set_item("risk_type", format!("{:?}", risk_type))?;
            dict.set_item("severity", severity)?;
            dict.set_item("timestamp", timestamp.timestamp())?;
        }
        AnalysisSignal::LiquidityAlert { symbol, liquidity_level, threshold, timestamp } => {
            dict.set_item("type", "liquidity_alert")?;
            dict.set_item("symbol", symbol)?;
            dict.set_item("liquidity_level", liquidity_level)?;
            dict.set_item("threshold", threshold)?;
            dict.set_item("timestamp", timestamp.timestamp())?;
        }
    }
    
    Ok(dict)
}

/// Python module definition
#[pymodule]
fn rust_market_analysis(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMarketAnalyzer>()?;
    m.add_class::<PyDataPipeline>()?;
    m.add_class::<PyMarketAnalysis>()?;
    m.add_class::<PyWhaleSignal>()?;
    m.add_class::<PyRegimeInfo>()?;
    m.add_class::<PyPattern>()?;
    m.add_class::<PyPredictions>()?;
    m.add_class::<PyPricePrediction>()?;
    m.add_class::<PyVolatilityForecast>()?;
    m.add_class::<PyMicrostructureAnalysis>()?;
    m.add_class::<PyRiskMetrics>()?;
    m.add_class::<PySignalReceiver>()?;
    
    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}