//! Calibrated FANN Backend with Uncertainty Quantification
//!
//! Wraps FannBackend with conformal prediction intervals for uncertainty-aware inference.

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hyperphysics_reasoning_router::{
    BackendPool, LatencyTier, ProblemDomain, RouterResult,
    backend::{BackendCapability, BackendId, BackendMetrics, ReasoningBackend, ReasoningResult, ResultValue},
    problem::{Problem, ProblemSignature, ProblemType},
};

use crate::fann::{FannConfig, FannNetwork};
use super::conformal::{ConformalConfig, FastConformalPredictor, UncertaintyBounds};
use super::calibration::{CalibrationConfig, NeuralCalibrator, CalibratedPrediction};

/// Result with uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyAwareResult {
    /// Base prediction value
    pub prediction: ResultValue,
    /// Uncertainty bounds
    pub uncertainty: Option<UncertaintyBounds>,
    /// Calibration info
    pub calibration: Option<CalibratedPrediction>,
    /// Combined confidence (base * calibration * uncertainty)
    pub combined_confidence: f64,
    /// ATS calibration score
    pub ats_score: f64,
    /// Total latency in nanoseconds
    pub total_latency_ns: u64,
}

impl UncertaintyAwareResult {
    /// Check if prediction is reliable (high confidence + narrow uncertainty)
    pub fn is_reliable(&self, confidence_threshold: f64, max_relative_uncertainty: f64) -> bool {
        if self.combined_confidence < confidence_threshold {
            return false;
        }

        if let Some(ref unc) = self.uncertainty {
            if unc.relative_uncertainty() > max_relative_uncertainty {
                return false;
            }
        }

        true
    }

    /// Get relative uncertainty if available
    pub fn relative_uncertainty(&self) -> Option<f64> {
        self.uncertainty.as_ref().map(|u| u.relative_uncertainty())
    }
}

/// Configuration for calibrated FANN backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedFannConfig {
    /// FANN network configuration
    pub fann: FannConfig,
    /// Conformal prediction configuration
    pub conformal: ConformalConfig,
    /// Temperature scaling configuration
    pub calibration: CalibrationConfig,
    /// Enable conformal prediction
    pub enable_conformal: bool,
    /// Enable temperature scaling
    pub enable_temperature_scaling: bool,
    /// Target combined latency
    pub target_latency_us: u64,
}

impl Default for CalibratedFannConfig {
    fn default() -> Self {
        Self {
            fann: FannConfig::default(),
            conformal: ConformalConfig::default(),
            calibration: CalibrationConfig::default(),
            enable_conformal: true,
            enable_temperature_scaling: true,
            target_latency_us: 50, // 50μs total (FANN + conformal + calibration)
        }
    }
}

impl CalibratedFannConfig {
    /// HFT-optimized configuration
    pub fn hft(input_dim: usize, hidden_dims: &[usize], output_dim: usize) -> Self {
        Self {
            fann: FannConfig::hft(input_dim, hidden_dims, output_dim),
            conformal: ConformalConfig::hft(),
            calibration: CalibrationConfig::hft(),
            enable_conformal: true,
            enable_temperature_scaling: true,
            target_latency_us: 25, // 25μs total for HFT
        }
    }

    /// Fast inference without uncertainty (just calibration)
    pub fn fast(input_dim: usize, hidden_dims: &[usize], output_dim: usize) -> Self {
        Self {
            fann: FannConfig::hft(input_dim, hidden_dims, output_dim),
            conformal: ConformalConfig::default(),
            calibration: CalibrationConfig::hft(),
            enable_conformal: false,
            enable_temperature_scaling: true,
            target_latency_us: 15,
        }
    }
}

/// Calibrated FANN backend with uncertainty quantification
pub struct CalibratedFannBackend {
    /// Backend identifier
    id: BackendId,
    /// Backend name
    name: String,
    /// Underlying FANN network
    network: Arc<Mutex<FannNetwork>>,
    /// Conformal predictor
    conformal: Arc<Mutex<FastConformalPredictor>>,
    /// Temperature calibrator
    calibrator: Arc<Mutex<NeuralCalibrator>>,
    /// Configuration
    config: CalibratedFannConfig,
    /// Capabilities
    capabilities: HashSet<BackendCapability>,
    /// Metrics
    metrics: Mutex<BackendMetrics>,
    /// Supported domains
    domains: Vec<ProblemDomain>,
    /// Supported problem types
    problem_types: Vec<ProblemType>,
}

impl CalibratedFannBackend {
    /// Create new calibrated FANN backend
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        config: CalibratedFannConfig,
    ) -> Result<Self, crate::fann::FannError> {
        let network = FannNetwork::new(config.fann.clone())?;

        let mut capabilities = HashSet::new();
        capabilities.insert(BackendCapability::Deterministic);
        capabilities.insert(BackendCapability::Differentiable);
        capabilities.insert(BackendCapability::SimdOptimized);
        capabilities.insert(BackendCapability::UncertaintyQuantification);

        Ok(Self {
            id: BackendId::new(id),
            name: name.into(),
            network: Arc::new(Mutex::new(network)),
            conformal: Arc::new(Mutex::new(FastConformalPredictor::new(config.conformal.clone()))),
            calibrator: Arc::new(Mutex::new(NeuralCalibrator::new(config.calibration.clone()))),
            config,
            capabilities,
            metrics: Mutex::new(BackendMetrics::default()),
            domains: vec![
                ProblemDomain::Financial,
                ProblemDomain::Physics,
                ProblemDomain::General,
            ],
            problem_types: vec![
                ProblemType::Prediction,
                ProblemType::Estimation,
                ProblemType::Classification,
            ],
        })
    }

    /// Create HFT-optimized calibrated backend
    pub fn hft(
        id: impl Into<String>,
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
    ) -> Result<Self, crate::fann::FannError> {
        let config = CalibratedFannConfig::hft(input_dim, hidden_dims, output_dim);
        Self::new(id, "CalibratedFANN-HFT", config)
    }

    /// Add calibration scores from historical predictions
    pub fn add_calibration_scores(&self, scores: &[f64]) {
        let mut conformal = self.conformal.lock();
        conformal.add_calibration_batch(scores);
    }

    /// Update with new observation (for online calibration)
    pub fn update(&self, prediction: f64, actual: f64) {
        let mut conformal = self.conformal.lock();
        conformal.update(prediction, actual);
    }

    /// Optimize temperature using validation data
    pub fn optimize_temperature(
        &self,
        predictions: &[f64],
        targets: &[f64],
    ) -> crate::error::NeuralResult<f64> {
        let mut calibrator = self.calibrator.lock();
        calibrator.optimize_temperature(predictions, targets)
    }

    /// Run inference with uncertainty quantification
    pub fn infer_with_uncertainty(
        &self,
        input: &[f64],
    ) -> crate::error::NeuralResult<UncertaintyAwareResult> {
        let start = Instant::now();

        // Step 1: FANN forward pass
        let output = {
            let mut network = self.network.lock();
            network.forward(input).map_err(|e| crate::error::NeuralError::FannError(e.to_string()))?
        };

        // Get primary prediction (scalar for simplicity)
        let primary_pred = if output.len() == 1 {
            output[0]
        } else {
            // For multi-output, use mean or first element
            output.iter().sum::<f64>() / output.len() as f64
        };

        // Step 2: Conformal prediction (if enabled)
        let uncertainty = if self.config.enable_conformal {
            let mut conformal = self.conformal.lock();
            match conformal.predict(primary_pred) {
                Ok(bounds) => Some(bounds),
                Err(_) => None, // Not enough calibration data
            }
        } else {
            None
        };

        // Step 3: Temperature scaling (if enabled)
        let calibration = if self.config.enable_temperature_scaling {
            let mut calibrator = self.calibrator.lock();
            Some(calibrator.calibrate(primary_pred)?)
        } else {
            None
        };

        let total_latency_ns = start.elapsed().as_nanos() as u64;

        // Compute combined confidence
        let base_confidence = self.compute_confidence(&output);
        let calibration_factor = calibration.as_ref().map(|c| c.confidence).unwrap_or(1.0);
        let uncertainty_factor = uncertainty.as_ref()
            .map(|u| 1.0 / (1.0 + u.relative_uncertainty()))
            .unwrap_or(1.0);

        let combined_confidence = base_confidence * calibration_factor * uncertainty_factor;
        let ats_score = calibration.as_ref().map(|c| c.ats_score).unwrap_or(base_confidence);

        // Build result
        let prediction = if output.len() == 1 {
            ResultValue::Scalar(output[0])
        } else {
            ResultValue::Vector(output)
        };

        Ok(UncertaintyAwareResult {
            prediction,
            uncertainty,
            calibration,
            combined_confidence,
            ats_score,
            total_latency_ns,
        })
    }

    /// Batch inference with uncertainty
    pub fn infer_batch_with_uncertainty(
        &self,
        inputs: &[Vec<f64>],
    ) -> crate::error::NeuralResult<Vec<UncertaintyAwareResult>> {
        inputs.iter()
            .map(|input| self.infer_with_uncertainty(input))
            .collect()
    }

    /// Extract features from problem (similar to FannBackend)
    fn extract_features(&self, problem: &Problem) -> RouterResult<Vec<f64>> {
        use hyperphysics_reasoning_router::RouterError;
        use hyperphysics_reasoning_router::problem::ProblemData;

        let input_dim = self.network.lock().input_dim();

        match &problem.data {
            ProblemData::Vector(nums) => {
                let mut features = nums.clone();
                features.resize(input_dim, 0.0);
                Ok(features)
            }
            ProblemData::TimeSeries { values, .. } => {
                let n = values.len().min(input_dim);
                let mut features = vec![0.0; input_dim];
                for (i, &v) in values.iter().rev().take(n).enumerate() {
                    features[i] = v;
                }
                Ok(features)
            }
            ProblemData::PhysicsState { positions, velocities, masses } => {
                let mut features = Vec::new();
                features.extend(positions.iter().copied());
                features.extend(velocities.iter().copied());
                features.extend(masses.iter().copied());
                features.resize(input_dim, 0.0);
                Ok(features)
            }
            ProblemData::Matrix { data, .. } => {
                let mut features = data.clone();
                features.resize(input_dim, 0.0);
                Ok(features)
            }
            _ => Err(RouterError::InvalidProblem("Unsupported problem data type".into())),
        }
    }

    /// Compute confidence from output
    fn compute_confidence(&self, output: &[f64]) -> f64 {
        let mean = output.iter().sum::<f64>() / output.len() as f64;
        let variance = output
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / output.len() as f64;

        (1.0 / (1.0 + variance)).max(0.5)
    }

    /// Get conformal predictor statistics
    pub fn conformal_stats(&self) -> (usize, f64) {
        let conformal = self.conformal.lock();
        (conformal.calibration_size(), conformal.avg_latency_ns())
    }

    /// Get calibrator statistics
    pub fn calibrator_stats(&self) -> (f64, f64) {
        let calibrator = self.calibrator.lock();
        (calibrator.temperature(), calibrator.avg_latency_ns())
    }

    /// Get network reference
    pub fn network(&self) -> &Arc<Mutex<FannNetwork>> {
        &self.network
    }

    /// Get configuration
    pub fn config(&self) -> &CalibratedFannConfig {
        &self.config
    }
}

#[async_trait]
impl ReasoningBackend for CalibratedFannBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn pool(&self) -> BackendPool {
        BackendPool::Optimization
    }

    fn supported_domains(&self) -> &[ProblemDomain] {
        &self.domains
    }

    fn capabilities(&self) -> &HashSet<BackendCapability> {
        &self.capabilities
    }

    fn latency_tier(&self) -> LatencyTier {
        if self.config.target_latency_us <= 25 {
            LatencyTier::UltraFast
        } else if self.config.target_latency_us < 1000 {
            LatencyTier::Fast
        } else if self.config.target_latency_us < 10000 {
            LatencyTier::Medium
        } else {
            LatencyTier::Slow
        }
    }

    fn can_handle(&self, signature: &ProblemSignature) -> bool {
        self.problem_types.contains(&signature.problem_type)
            && self.domains.contains(&signature.domain)
    }

    fn estimate_latency(&self, _signature: &ProblemSignature) -> Duration {
        Duration::from_micros(self.config.target_latency_us)
    }

    async fn execute(&self, problem: &Problem) -> RouterResult<ReasoningResult> {
        use hyperphysics_reasoning_router::RouterError;

        let start = Instant::now();

        // Extract features
        let features = self.extract_features(problem)?;

        // Run inference with uncertainty
        let result = self.infer_with_uncertainty(&features)
            .map_err(|e| RouterError::BackendFailed {
                backend_id: self.id.0.clone(),
                message: e.to_string(),
            })?;

        let latency = start.elapsed();

        // Update metrics
        {
            let mut metrics = self.metrics.lock();
            metrics.record(latency, true, Some(result.combined_confidence));
        }

        // Check latency
        if latency > Duration::from_micros(self.config.target_latency_us) {
            tracing::warn!(
                "CalibratedFANN inference exceeded latency: {:?} > {}μs",
                latency,
                self.config.target_latency_us
            );
        }

        let network = self.network.lock();
        Ok(ReasoningResult {
            value: result.prediction,
            confidence: result.combined_confidence,
            quality: result.ats_score,
            latency,
            backend_id: self.id.clone(),
            metadata: json!({
                "backend": "calibrated-fann",
                "network": self.name,
                "input_dim": network.input_dim(),
                "output_dim": network.output_dim(),
                "uncertainty": result.uncertainty.map(|u| json!({
                    "lower": u.lower,
                    "upper": u.upper,
                    "width": u.width,
                    "confidence": u.confidence,
                })),
                "calibration": result.calibration.map(|c| json!({
                    "temperature": c.temperature,
                    "ats_score": c.ats_score,
                })),
                "combined_confidence": result.combined_confidence,
                "latency_ns": result.total_latency_ns,
            }),
        })
    }

    fn metrics(&self) -> BackendMetrics {
        self.metrics.lock().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperphysics_reasoning_router::problem::ProblemData;

    #[test]
    fn test_calibrated_fann_creation() {
        let config = CalibratedFannConfig::hft(10, &[32, 16], 3);
        let backend = CalibratedFannBackend::new("test", "Test", config).unwrap();

        assert_eq!(backend.id().0, "test");
        assert!(backend.capabilities().contains(&BackendCapability::UncertaintyQuantification));
    }

    #[test]
    fn test_calibrated_hft_backend() {
        let backend = CalibratedFannBackend::hft("hft-test", 10, &[32], 3).unwrap();

        assert_eq!(backend.latency_tier(), LatencyTier::UltraFast);
        assert_eq!(backend.config().target_latency_us, 25);
    }

    #[test]
    fn test_infer_without_calibration_data() {
        let config = CalibratedFannConfig::hft(10, &[16], 3);
        let backend = CalibratedFannBackend::new("test", "Test", config).unwrap();

        let input = vec![1.0; 10];
        let result = backend.infer_with_uncertainty(&input).unwrap();

        // Should work but without uncertainty (not enough calibration)
        assert!(result.uncertainty.is_none());
        assert!(result.calibration.is_some());
        assert!(result.combined_confidence > 0.0);
    }

    #[test]
    fn test_infer_with_calibration_data() {
        let mut config = CalibratedFannConfig::hft(10, &[16], 3);
        config.conformal.min_calibration_size = 5;

        let backend = CalibratedFannBackend::new("test", "Test", config).unwrap();

        // Add calibration scores
        backend.add_calibration_scores(&[0.1, 0.2, 0.15, 0.25, 0.18, 0.22, 0.12, 0.28, 0.19, 0.21]);

        let input = vec![1.0; 10];
        let result = backend.infer_with_uncertainty(&input).unwrap();

        // Should now have uncertainty bounds
        assert!(result.uncertainty.is_some());
        let bounds = result.uncertainty.unwrap();
        assert!(bounds.lower < bounds.prediction);
        assert!(bounds.upper > bounds.prediction);
    }

    #[test]
    fn test_uncertainty_aware_result_reliability() {
        let result = UncertaintyAwareResult {
            prediction: ResultValue::Scalar(100.0),
            uncertainty: Some(UncertaintyBounds {
                prediction: 100.0,
                lower: 95.0,
                upper: 105.0,
                confidence: 0.95,
                width: 10.0,
                latency_ns: 100,
            }),
            calibration: None,
            combined_confidence: 0.9,
            ats_score: 0.85,
            total_latency_ns: 500,
        };

        // 10% relative uncertainty with 0.9 confidence
        assert!(result.is_reliable(0.8, 0.15));
        assert!(!result.is_reliable(0.95, 0.15)); // Confidence too low
        assert!(!result.is_reliable(0.8, 0.05)); // Uncertainty too high
    }

    #[test]
    fn test_online_calibration_update() {
        let mut config = CalibratedFannConfig::hft(10, &[16], 3);
        config.conformal.min_calibration_size = 3;

        let backend = CalibratedFannBackend::new("test", "Test", config).unwrap();

        // Simulate predictions and actuals
        for i in 0..10 {
            let pred = i as f64;
            let actual = pred + (i as f64 * 0.05);
            backend.update(pred, actual);
        }

        let (size, _) = backend.conformal_stats();
        assert!(size >= 3);
    }

    #[tokio::test]
    async fn test_execute_via_reasoning_backend() {
        let mut config = CalibratedFannConfig::hft(10, &[16], 3);
        config.conformal.min_calibration_size = 5;

        let backend = CalibratedFannBackend::new("test", "Test", config).unwrap();
        backend.add_calibration_scores(&[0.1, 0.2, 0.15, 0.25, 0.18, 0.22]);

        let sig = ProblemSignature::new(ProblemType::Prediction, ProblemDomain::Financial);
        let problem = Problem::new(sig, ProblemData::Vector(vec![1.0; 10]));

        let result = backend.execute(&problem).await.unwrap();

        assert!(result.confidence > 0.0);
        assert!(result.latency.as_nanos() > 0);
        assert!(result.metadata.get("uncertainty").is_some());
    }

    #[test]
    fn test_latency_tiers() {
        let hft_config = CalibratedFannConfig::hft(10, &[16], 3);
        let hft = CalibratedFannBackend::new("hft", "HFT", hft_config).unwrap();
        assert_eq!(hft.latency_tier(), LatencyTier::UltraFast);

        let mut slow_config = CalibratedFannConfig::default();
        slow_config.target_latency_us = 5000;
        slow_config.fann = FannConfig::regression(10, &[16], 3);
        let slow = CalibratedFannBackend::new("slow", "Slow", slow_config).unwrap();
        assert_eq!(slow.latency_tier(), LatencyTier::Medium);
    }
}
