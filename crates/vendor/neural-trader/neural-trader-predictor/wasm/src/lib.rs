//! WebAssembly bindings for Neural Trader Predictor
//!
//! Provides JavaScript-friendly wrappers around Rust conformal prediction algorithms

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

// Use `wasm_bindgen`'s `prelude::*` for all wasm_bindgen imports
use wasm_bindgen::JsValue;

// Import neural-trader-predictor types
use neural_trader_predictor::conformal::{SplitConformalPredictor, AdaptiveConformalPredictor};
use neural_trader_predictor::core::{PredictionInterval, AdaptiveConfig};
use neural_trader_predictor::scores::AbsoluteScore;

// Re-export the panic hook
pub use console_error_panic_hook::set_once as set_panic_hook;

/// Helper to convert errors to JsValue
fn error_to_js(err: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&err.to_string())
}

/// Initialize panic hook for better error messages in console
#[wasm_bindgen(start)]
pub fn init_wasm() {
    console_error_panic_hook::set_once();
}

/// TypeScript-compatible prediction interval
#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct WasmPredictionInterval {
    point: f64,
    lower: f64,
    upper: f64,
    alpha: f64,
    quantile: f64,
    timestamp: f64,
}

#[wasm_bindgen]
impl WasmPredictionInterval {
    /// Create a new prediction interval
    #[wasm_bindgen(constructor)]
    pub fn new(
        point: f64,
        lower: f64,
        upper: f64,
        alpha: f64,
        quantile: f64,
        timestamp: Option<f64>,
    ) -> WasmPredictionInterval {
        WasmPredictionInterval {
            point,
            lower,
            upper,
            alpha,
            quantile,
            timestamp: timestamp.unwrap_or_else(|| js_sys::Date::now()),
        }
    }

    /// Get point prediction
    #[wasm_bindgen(getter)]
    pub fn point(&self) -> f64 {
        self.point
    }

    /// Get lower bound
    #[wasm_bindgen(getter)]
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// Get upper bound
    #[wasm_bindgen(getter)]
    pub fn upper(&self) -> f64 {
        self.upper
    }

    /// Get alpha (miscoverage rate)
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get quantile threshold
    #[wasm_bindgen(getter)]
    pub fn quantile(&self) -> f64 {
        self.quantile
    }

    /// Get timestamp
    #[wasm_bindgen(getter)]
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }

    /// Calculate interval width
    #[wasm_bindgen]
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Check if value is contained in interval
    #[wasm_bindgen]
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Calculate relative width as percentage
    #[wasm_bindgen]
    pub fn relative_width(&self) -> f64 {
        if self.point.abs() < f64::EPSILON {
            f64::INFINITY
        } else {
            (self.width() / self.point.abs()) * 100.0
        }
    }

    /// Get expected coverage (1 - alpha)
    #[wasm_bindgen]
    pub fn coverage(&self) -> f64 {
        1.0 - self.alpha
    }

    /// Convert to JSON string
    #[wasm_bindgen]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

impl From<&PredictionInterval> for WasmPredictionInterval {
    fn from(interval: &PredictionInterval) -> Self {
        // Convert DateTime<Utc> to milliseconds since epoch
        let timestamp_ms = interval.timestamp.timestamp_millis() as f64;

        WasmPredictionInterval {
            point: interval.point,
            lower: interval.lower,
            upper: interval.upper,
            alpha: interval.alpha,
            quantile: interval.quantile,
            timestamp: timestamp_ms,
        }
    }
}

/// Configuration for Split Conformal Predictor
#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct WasmPredictorConfig {
    #[serde(default = "default_alpha")]
    alpha: f64,
    #[serde(default = "default_calibration_size")]
    calibration_size: usize,
    #[serde(default = "default_max_interval_width")]
    max_interval_width_pct: f64,
    #[serde(default = "default_recalibration_freq")]
    recalibration_freq: usize,
}

fn default_alpha() -> f64 {
    0.1
}

fn default_calibration_size() -> usize {
    2000
}

fn default_max_interval_width() -> f64 {
    5.0
}

fn default_recalibration_freq() -> usize {
    100
}

#[wasm_bindgen]
impl WasmPredictorConfig {
    /// Create a new predictor configuration with defaults
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: Option<f64>) -> WasmPredictorConfig {
        WasmPredictorConfig {
            alpha: alpha.unwrap_or(0.1),
            calibration_size: 2000,
            max_interval_width_pct: 5.0,
            recalibration_freq: 100,
        }
    }

    /// Set alpha value
    #[wasm_bindgen]
    pub fn with_alpha(mut self, alpha: f64) -> WasmPredictorConfig {
        self.alpha = alpha;
        self
    }

    /// Set calibration size
    #[wasm_bindgen]
    pub fn with_calibration_size(mut self, size: usize) -> WasmPredictorConfig {
        self.calibration_size = size;
        self
    }

    /// Get alpha value
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get calibration size
    #[wasm_bindgen(getter)]
    pub fn calibration_size(&self) -> usize {
        self.calibration_size
    }
}

/// WebAssembly wrapper for Split Conformal Predictor
#[wasm_bindgen]
pub struct WasmConformalPredictor {
    predictor: SplitConformalPredictor<AbsoluteScore>,
}

#[wasm_bindgen]
impl WasmConformalPredictor {
    /// Create a new conformal predictor
    #[wasm_bindgen(constructor)]
    pub fn new(config: Option<WasmPredictorConfig>) -> WasmConformalPredictor {
        let cfg = config.unwrap_or_else(|| WasmPredictorConfig::new(None));
        WasmConformalPredictor {
            predictor: SplitConformalPredictor::new(cfg.alpha, AbsoluteScore),
        }
    }

    /// Calibrate the predictor with predictions and actuals
    #[wasm_bindgen]
    pub fn calibrate(
        &mut self,
        predictions: Vec<f64>,
        actuals: Vec<f64>,
    ) -> Result<(), JsValue> {
        if predictions.len() != actuals.len() {
            return Err(JsValue::from_str("Predictions and actuals must have same length"));
        }

        if predictions.is_empty() {
            return Err(JsValue::from_str("At least one calibration sample required"));
        }

        self.predictor
            .calibrate(&predictions, &actuals)
            .map_err(error_to_js)
    }

    /// Make a prediction with confidence interval
    #[wasm_bindgen]
    pub fn predict(&mut self, point_prediction: f64) -> WasmPredictionInterval {
        let interval = self.predictor.predict(point_prediction);
        WasmPredictionInterval::from(&interval)
    }

    /// Update predictor with new observation (binary search insertion)
    #[wasm_bindgen]
    pub fn update(
        &mut self,
        prediction: f64,
        actual: f64,
    ) -> Result<(), JsValue> {
        self.predictor
            .update(prediction, actual)
            .map_err(error_to_js)
    }

    /// Get empirical coverage from calibration set
    #[wasm_bindgen]
    pub fn get_empirical_coverage(
        &self,
        predictions: Vec<f64>,
        actuals: Vec<f64>,
    ) -> Result<f64, JsValue> {
        if predictions.len() != actuals.len() {
            return Err(JsValue::from_str("Predictions and actuals must have same length"));
        }

        if predictions.is_empty() {
            return Ok(0.0);
        }

        let mut covered = 0;
        for i in 0..predictions.len() {
            // Create temporary mutable copy for prediction
            let mut temp_predictor = SplitConformalPredictor::new(self.predictor.alpha(), AbsoluteScore);
            let _ = temp_predictor.calibrate(&predictions, &actuals);
            let interval = temp_predictor.predict(predictions[i]);
            if interval.contains(actuals[i]) {
                covered += 1;
            }
        }

        Ok(covered as f64 / predictions.len() as f64)
    }

    /// Get the number of calibration samples
    #[wasm_bindgen]
    pub fn n_calibration(&self) -> usize {
        self.predictor.n_calibration()
    }

    /// Get the number of predictions made
    #[wasm_bindgen]
    pub fn n_predictions(&self) -> usize {
        self.predictor.n_predictions()
    }

    /// Get current quantile value
    #[wasm_bindgen]
    pub fn get_quantile(&self) -> f64 {
        self.predictor.get_quantile()
    }

    /// Get alpha value
    #[wasm_bindgen]
    pub fn get_alpha(&self) -> f64 {
        self.predictor.alpha()
    }

    /// Check if predictor is calibrated
    #[wasm_bindgen]
    pub fn is_calibrated(&self) -> bool {
        self.predictor.is_calibrated()
    }

    /// Get calibration statistics as JSON
    #[wasm_bindgen]
    pub fn get_stats(&self) -> String {
        let stats = serde_json::json!({
            "n_calibration": self.predictor.n_calibration(),
            "alpha": self.predictor.alpha(),
            "quantile": self.predictor.get_quantile(),
            "n_predictions": self.predictor.n_predictions(),
            "is_calibrated": self.predictor.is_calibrated(),
        });
        serde_json::to_string(&stats).unwrap_or_else(|_| "{}".to_string())
    }

    /// Reset the predictor
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        // Create a new predictor to reset state
        let alpha = self.predictor.alpha();
        self.predictor = SplitConformalPredictor::new(alpha, AbsoluteScore);
    }
}

/// Configuration for Adaptive Conformal Predictor
#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct WasmAdaptiveConfig {
    #[serde(default = "default_target_coverage")]
    target_coverage: f64,
    #[serde(default = "default_gamma")]
    gamma: f64,
    #[serde(default = "default_coverage_window")]
    coverage_window: usize,
    #[serde(default = "default_alpha_min")]
    alpha_min: f64,
    #[serde(default = "default_alpha_max")]
    alpha_max: f64,
}

fn default_target_coverage() -> f64 {
    0.90
}

fn default_gamma() -> f64 {
    0.02
}

fn default_coverage_window() -> usize {
    200
}

fn default_alpha_min() -> f64 {
    0.01
}

fn default_alpha_max() -> f64 {
    0.30
}

#[wasm_bindgen]
impl WasmAdaptiveConfig {
    /// Create adaptive configuration with defaults
    #[wasm_bindgen(constructor)]
    pub fn new(target_coverage: Option<f64>, gamma: Option<f64>) -> WasmAdaptiveConfig {
        WasmAdaptiveConfig {
            target_coverage: target_coverage.unwrap_or(0.90),
            gamma: gamma.unwrap_or(0.02),
            coverage_window: 200,
            alpha_min: 0.01,
            alpha_max: 0.30,
        }
    }

    /// Get target coverage
    #[wasm_bindgen(getter)]
    pub fn target_coverage(&self) -> f64 {
        self.target_coverage
    }

    /// Get gamma (learning rate)
    #[wasm_bindgen(getter)]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

impl From<WasmAdaptiveConfig> for AdaptiveConfig {
    fn from(wasm_cfg: WasmAdaptiveConfig) -> Self {
        AdaptiveConfig {
            target_coverage: wasm_cfg.target_coverage,
            gamma: wasm_cfg.gamma,
            coverage_window: wasm_cfg.coverage_window,
            alpha_min: wasm_cfg.alpha_min,
            alpha_max: wasm_cfg.alpha_max,
        }
    }
}

/// WebAssembly wrapper for Adaptive Conformal Predictor
#[wasm_bindgen]
pub struct WasmAdaptivePredictor {
    predictor: AdaptiveConformalPredictor<AbsoluteScore>,
}

#[wasm_bindgen]
impl WasmAdaptivePredictor {
    /// Create a new adaptive conformal predictor
    #[wasm_bindgen(constructor)]
    pub fn new(config: Option<WasmAdaptiveConfig>) -> WasmAdaptivePredictor {
        let cfg = config
            .unwrap_or_else(|| WasmAdaptiveConfig::new(None, None))
            .into();
        WasmAdaptivePredictor {
            predictor: AdaptiveConformalPredictor::new(cfg, AbsoluteScore),
        }
    }

    /// Calibrate with initial data
    #[wasm_bindgen]
    pub fn calibrate(
        &mut self,
        predictions: Vec<f64>,
        actuals: Vec<f64>,
    ) -> Result<(), JsValue> {
        if predictions.len() != actuals.len() {
            return Err(JsValue::from_str("Predictions and actuals must have same length"));
        }

        if predictions.is_empty() {
            return Err(JsValue::from_str("At least one calibration sample required"));
        }

        self.predictor
            .calibrate(&predictions, &actuals)
            .map_err(error_to_js)
    }

    /// Make prediction with adaptive alpha
    #[wasm_bindgen]
    pub fn predict(&mut self, point_prediction: f64) -> WasmPredictionInterval {
        let interval = self.predictor.predict(point_prediction);
        WasmPredictionInterval::from(&interval)
    }

    /// Make prediction and adapt alpha based on actual value
    #[wasm_bindgen]
    pub fn predict_and_adapt(
        &mut self,
        point_prediction: f64,
        actual: Option<f64>,
    ) -> WasmPredictionInterval {
        let interval = self.predictor.predict_and_adapt(point_prediction, actual);
        WasmPredictionInterval::from(&interval)
    }

    /// Observe actual value and adapt alpha separately
    #[wasm_bindgen]
    pub fn observe_and_adapt(
        &mut self,
        interval: &WasmPredictionInterval,
        actual: f64,
    ) -> f64 {
        let rust_interval = PredictionInterval::new(
            interval.point,
            interval.lower,
            interval.upper,
            interval.alpha,
            interval.quantile,
        );
        self.predictor.observe_and_adapt(&rust_interval, actual)
    }

    /// Get empirical coverage
    #[wasm_bindgen]
    pub fn empirical_coverage(&self) -> f64 {
        self.predictor.empirical_coverage()
    }

    /// Get current alpha
    #[wasm_bindgen]
    pub fn get_current_alpha(&self) -> f64 {
        self.predictor.alpha()
    }

    /// Get target coverage
    #[wasm_bindgen]
    pub fn get_target_coverage(&self) -> f64 {
        self.predictor.target_coverage()
    }

    /// Get coverage error (empirical - target)
    #[wasm_bindgen]
    pub fn get_coverage_error(&self) -> f64 {
        self.predictor.coverage_error()
    }

    /// Get number of adaptations
    #[wasm_bindgen]
    pub fn get_n_adaptations(&self) -> usize {
        self.predictor.n_adaptations()
    }

    /// Get coverage history length
    #[wasm_bindgen]
    pub fn get_history_size(&self) -> usize {
        self.predictor.history_size()
    }

    /// Get statistics including coverage metrics as JSON
    #[wasm_bindgen]
    pub fn get_stats(&self) -> String {
        let stats = serde_json::json!({
            "alpha": self.predictor.alpha(),
            "target_coverage": self.predictor.target_coverage(),
            "empirical_coverage": self.predictor.empirical_coverage(),
            "coverage_error": self.predictor.coverage_error(),
            "n_adaptations": self.predictor.n_adaptations(),
            "history_size": self.predictor.history_size(),
        });
        serde_json::to_string(&stats).unwrap_or_else(|_| "{}".to_string())
    }

    /// Reset the adaptive state
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        let config = AdaptiveConfig {
            target_coverage: self.predictor.target_coverage(),
            gamma: 0.02, // Use default gamma for reset
            coverage_window: 200,
            alpha_min: 0.01,
            alpha_max: 0.30,
        };
        self.predictor = AdaptiveConformalPredictor::new(config, AbsoluteScore);
    }
}
