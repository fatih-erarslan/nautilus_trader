//! Advanced Market Analysis System
//! 
//! This Rust library provides sophisticated market analysis capabilities including:
//! - Antifragility detection and measurement
//! - Whale activity detection and tracking
//! - Self-Organized Criticality analysis
//! - Panarchy cycle detection
//! - Fibonacci pattern recognition
//! - Black Swan event detection
//!
//! Designed for high-performance integration with the neuro_trader system.

use std::collections::HashMap;

pub mod antifragility;
pub mod whale_detection;
pub mod soc_criticality;
pub mod panarchy;
pub mod fibonacci;
pub mod black_swan;
pub mod market_data;
pub mod utils;

use crate::antifragility::AntifragilityAnalyzer;
use crate::whale_detection::{WhaleDetector, WhaleSignals};
use crate::soc_criticality::SOCAnalyzer;
use crate::panarchy::PanarchyDetector;
use crate::fibonacci::FibonacciAnalyzer;
use crate::black_swan::BlackSwanDetector;
use crate::market_data::MarketData;

/// Analysis results structure
#[derive(Debug, Clone)]
pub struct AnalysisResults {
    pub antifragility_score: f64,
    pub whale_signals: WhaleSignals,
    pub soc_level: f64,
    pub panarchy_phase: String,
    pub fibonacci_levels: Vec<f64>,
    pub black_swan_prob: f64,
    pub overall_score: f64,
    pub trade_action: String,
    pub position_size_multiplier: f64,
    pub risk_level: String,
}

/// Trade signals structure
#[derive(Debug, Clone)]
pub struct TradeSignals {
    pub enter_long: bool,
    pub enter_short: bool,
    pub confidence: f64,
    pub exit_long: bool,
    pub exit_short: bool,
    pub urgency: String,
}

/// Main market analysis engine that coordinates all subsystems
#[cfg_attr(feature = "python-bindings", pyo3::pyclass)]
pub struct MarketAnalysisEngine {
    pub antifragility: AntifragilityAnalyzer,
    pub whale_detector: WhaleDetector,
    pub soc_analyzer: SOCAnalyzer,
    pub panarchy: PanarchyDetector,
    pub fibonacci: FibonacciAnalyzer,
    pub black_swan: BlackSwanDetector,
}

impl MarketAnalysisEngine {
    pub fn new() -> Self {
        Self {
            antifragility: AntifragilityAnalyzer::new(),
            whale_detector: WhaleDetector::new(),
            soc_analyzer: SOCAnalyzer::new(),
            panarchy: PanarchyDetector::new(),
            fibonacci: FibonacciAnalyzer::new(),
            black_swan: BlackSwanDetector::new(),
        }
    }

    /// Run comprehensive analysis combining all detection systems
    pub fn run_comprehensive_analysis(&mut self, data: &MarketData) -> anyhow::Result<AnalysisResults> {
        // Run all analysis systems
        let antifragility_score = self.antifragility.calculate_antifragility(data)?;
        let whale_signals = self.whale_detector.detect_whale_activity(data)?;
        let soc_level = self.soc_analyzer.analyze_criticality(data)?;
        let panarchy_phase = self.panarchy.detect_cycle_phase(data)?;
        let fibonacci_levels = self.fibonacci.find_fibonacci_levels(data)?;
        let black_swan_prob = self.black_swan.calculate_probability(data)?;
        
        // Calculate overall score and recommendations
        let overall_score = self.calculate_overall_score(
            antifragility_score, &whale_signals, soc_level, 
            &panarchy_phase, black_swan_prob
        );
        
        let trade_action = self.determine_trade_action(&whale_signals, &panarchy_phase, black_swan_prob);
        let position_size_multiplier = self.calculate_position_size(antifragility_score, black_swan_prob);
        let risk_level = self.assess_risk_level(soc_level, black_swan_prob);
        
        Ok(AnalysisResults {
            antifragility_score,
            whale_signals,
            soc_level,
            panarchy_phase,
            fibonacci_levels,
            black_swan_prob,
            overall_score,
            trade_action,
            position_size_multiplier,
            risk_level,
        })
    }

    /// Generate trade signals based on analysis
    pub fn generate_trade_signals(&mut self, data: &MarketData) -> anyhow::Result<TradeSignals> {
        let results = self.run_comprehensive_analysis(data)?;
        
        // Entry signals
        let should_enter = self.should_enter_trade(&results);
        let should_exit = self.should_exit_trade(&results);
        
        Ok(TradeSignals {
            enter_long: should_enter.enter_long,
            enter_short: should_enter.enter_short,
            confidence: should_enter.confidence,
            exit_long: should_exit.exit_long,
            exit_short: should_exit.exit_short,
            urgency: should_exit.urgency,
        })
    }

    /// Rank pairs by profitability potential
    pub fn rank_pairs(&mut self, pairs_data: HashMap<String, MarketData>) -> anyhow::Result<Vec<String>> {
        let mut pair_scores = Vec::new();
        
        for (pair, data) in pairs_data.iter() {
            let results = self.run_comprehensive_analysis(data)?;
            pair_scores.push((pair.clone(), results.overall_score));
        }
        
        // Sort by score descending
        pair_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top pairs
        Ok(pair_scores.into_iter()
            .take(10)
            .map(|(pair, _)| pair)
            .collect())
    }

    fn calculate_overall_score(&self, antifragility: f64, whale_signals: &WhaleSignals, 
                              soc_level: f64, panarchy_phase: &str, black_swan_prob: f64) -> f64 {
        let whale_factor = if whale_signals.major_whale_detected {
            whale_signals.whale_strength * whale_signals.whale_direction
        } else {
            0.0
        };

        let panarchy_factor = match panarchy_phase {
            "growth" => 0.8,
            "conservation" => 0.6,
            "release" => -0.2,
            "reorganization" => 0.4,
            _ => 0.5,
        };

        let score = (
            antifragility * 0.3 +
            whale_factor * 0.25 +
            panarchy_factor * 0.2 +
            (1.0 - soc_level) * 0.15 +
            (1.0 - black_swan_prob) * 0.1
        ).max(0.0).min(1.0);

        score
    }

    fn determine_trade_action(&self, whale_signals: &WhaleSignals, panarchy_phase: &str, black_swan_prob: f64) -> String {
        if black_swan_prob > 0.7 {
            return "AVOID".to_string();
        }

        if whale_signals.major_whale_detected && whale_signals.whale_direction > 0.5 {
            return "BUY".to_string();
        }

        match panarchy_phase {
            "growth" => "BUY".to_string(),
            "conservation" => "HOLD".to_string(),
            "release" => "SELL".to_string(),
            "reorganization" => "WAIT".to_string(),
            _ => "HOLD".to_string(),
        }
    }

    fn calculate_position_size(&self, antifragility: f64, black_swan_prob: f64) -> f64 {
        let base_size = 1.0;
        let antifragility_bonus = antifragility * 0.5;
        let black_swan_penalty = black_swan_prob * 0.8;
        
        (base_size + antifragility_bonus - black_swan_penalty).max(0.1).min(2.0)
    }

    fn assess_risk_level(&self, soc_level: f64, black_swan_prob: f64) -> String {
        let risk_score = (soc_level + black_swan_prob) / 2.0;
        
        if risk_score > 0.7 {
            "HIGH".to_string()
        } else if risk_score > 0.4 {
            "MEDIUM".to_string()
        } else {
            "LOW".to_string()
        }
    }

    fn should_enter_trade(&self, results: &AnalysisResults) -> TradeSignals {
        let enter_long = results.overall_score > 0.7 && results.black_swan_prob < 0.3;
        let enter_short = results.overall_score < 0.3 && results.black_swan_prob < 0.5;
        let confidence = if enter_long || enter_short {
            results.overall_score * 100.0
        } else {
            0.0
        };

        TradeSignals {
            enter_long,
            enter_short,
            confidence,
            exit_long: false,
            exit_short: false,
            urgency: "NORMAL".to_string(),
        }
    }

    fn should_exit_trade(&self, results: &AnalysisResults) -> TradeSignals {
        let exit_long = results.overall_score < 0.4 || results.black_swan_prob > 0.6;
        let exit_short = results.overall_score > 0.6 || results.black_swan_prob > 0.6;
        let urgency = if results.black_swan_prob > 0.8 {
            "HIGH".to_string()
        } else {
            "NORMAL".to_string()
        };

        TradeSignals {
            enter_long: false,
            enter_short: false,
            confidence: 0.0,
            exit_long,
            exit_short,
            urgency,
        }
    }
}

impl Default for MarketAnalysisEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Python bindings module (conditional compilation)
#[cfg(feature = "python-bindings")]
pub mod python_bindings {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use std::collections::HashMap;

    #[pymethods]
    impl MarketAnalysisEngine {
        #[new]
        pub fn py_new() -> Self {
            Self::new()
        }

        /// Direct access to run_comprehensive_analysis method for Python
        #[pyo3(name = "run_comprehensive_analysis")]
        pub fn py_run_comprehensive_analysis(&mut self, py: Python, data: &PyDict) -> PyResult<PyObject> {
            let market_data = parse_market_data(data)?;
            
            let results = self.run_comprehensive_analysis(&market_data)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
            // Convert results to Python dictionary
            let py_dict = PyDict::new(py);
            
            py_dict.set_item("antifragility_score", results.antifragility_score)?;
            
            let whale_dict = PyDict::new(py);
            whale_dict.set_item("major_whale_detected", results.whale_signals.major_whale_detected)?;
            whale_dict.set_item("whale_direction", results.whale_signals.whale_direction)?;
            whale_dict.set_item("whale_strength", results.whale_signals.whale_strength)?;
            py_dict.set_item("whale_signals", whale_dict)?;
            
            py_dict.set_item("soc_level", results.soc_level)?;
            py_dict.set_item("panarchy_phase", results.panarchy_phase)?;
            py_dict.set_item("fibonacci_levels", results.fibonacci_levels)?;
            py_dict.set_item("black_swan_prob", results.black_swan_prob)?;
            py_dict.set_item("overall_score", results.overall_score)?;
            py_dict.set_item("trade_action", results.trade_action)?;
            py_dict.set_item("position_size_multiplier", results.position_size_multiplier)?;
            py_dict.set_item("risk_level", results.risk_level)?;
            
            Ok(py_dict.into())
        }

        /// Comprehensive market analysis combining all detection systems (legacy method name)
        pub fn analyze_market(&mut self, py: Python, data: &PyDict) -> PyResult<PyObject> {
            let market_data = parse_market_data(data)?;
            
            let results = self.run_comprehensive_analysis(&market_data)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
            // Convert results to Python dictionary with legacy field names
            let py_dict = PyDict::new(py);
            
            py_dict.set_item("antifragility", results.antifragility_score)?;
            
            let whale_dict = PyDict::new(py);
            whale_dict.set_item("major_whale_detected", results.whale_signals.major_whale_detected)?;
            whale_dict.set_item("whale_direction", results.whale_signals.whale_direction)?;
            whale_dict.set_item("whale_strength", results.whale_signals.whale_strength)?;
            py_dict.set_item("whale_activity", whale_dict)?;
            
            py_dict.set_item("criticality_level", results.soc_level)?;
            py_dict.set_item("panarchy_phase", results.panarchy_phase)?;
            py_dict.set_item("fibonacci_levels", results.fibonacci_levels)?;
            py_dict.set_item("black_swan_probability", results.black_swan_prob)?;
            py_dict.set_item("overall_score", results.overall_score)?;
            py_dict.set_item("trade_recommendation", results.trade_action)?;
            
            Ok(py_dict.into())
        }

        /// Get profitable pairs ranked by comprehensive analysis
        pub fn get_profitable_pairs(&mut self, py: Python, pairs_data: &PyDict) -> PyResult<PyObject> {
            let mut pairs_map = HashMap::new();
            
            for (pair, data) in pairs_data.iter() {
                let pair_name: String = pair.extract()?;
                let market_data = parse_market_data(data.downcast()?)?;
                pairs_map.insert(pair_name, market_data);
            }
            
            let top_pairs = self.rank_pairs(pairs_map)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                
            Ok(top_pairs.to_object(py))
        }

        /// Real-time trade signal generation
        pub fn py_generate_trade_signals(&mut self, py: Python, data: &PyDict) -> PyResult<PyObject> {
            let market_data = parse_market_data(data)?;
            
            let signals = self.generate_trade_signals(&market_data)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
            let signals_dict = PyDict::new(py);
            
            signals_dict.set_item("enter_long", signals.enter_long)?;
            signals_dict.set_item("enter_short", signals.enter_short)?;
            signals_dict.set_item("confidence", signals.confidence)?;
            signals_dict.set_item("exit_long", signals.exit_long)?;
            signals_dict.set_item("exit_short", signals.exit_short)?;
            signals_dict.set_item("urgency", signals.urgency)?;
            
            Ok(signals_dict.into())
        }
    }

    fn parse_market_data(data: &PyDict) -> PyResult<MarketData> {
        let prices: Vec<f64> = data.get_item("close")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'close' data"))?
            .extract()?;
            
        let volumes: Vec<f64> = data.get_item("volume")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'volume' data"))?
            .extract()?;
            
        let highs: Vec<f64> = data.get_item("high")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'high' data"))?
            .extract()?;
            
        let lows: Vec<f64> = data.get_item("low")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'low' data"))?
            .extract()?;
        
        Ok(MarketData::new(prices, volumes, highs, lows))
    }

    /// Python module definition
    #[pymodule]
    pub fn market_analysis(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<MarketAnalysisEngine>()?;
        Ok(())
    }
}

// Re-export the Python module function if feature is enabled
#[cfg(feature = "python-bindings")]
pub use python_bindings::market_analysis;