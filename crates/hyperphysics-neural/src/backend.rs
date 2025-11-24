//! Neural backend adapter for reasoning router integration
//!
//! Implements ReasoningBackend trait to allow neural networks to participate
//! in the reasoning router's backend selection and execution system.

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hyperphysics_reasoning_router::{
    BackendPool, LatencyTier, ProblemDomain, RouterError, RouterResult,
    backend::{BackendCapability, BackendId, BackendMetrics, ReasoningBackend, ReasoningResult, ResultValue},
    problem::{Problem, ProblemData, ProblemSignature, ProblemType},
};

use crate::core::{Tensor, TensorShape};
use crate::network::Network;

/// Neural network backend for reasoning router
pub struct NeuralBackend {
    /// Backend identifier
    id: BackendId,
    /// Backend name
    name: String,
    /// Neural network model
    network: Arc<Mutex<Network>>,
    /// Backend capabilities
    capabilities: HashSet<BackendCapability>,
    /// Performance metrics
    metrics: Mutex<BackendMetrics>,
    /// Configuration
    config: NeuralBackendConfig,
}

/// Neural backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralBackendConfig {
    /// Maximum inference latency (microseconds)
    pub max_latency_us: u64,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Batch size for batch inference
    pub batch_size: usize,
    /// Problem domains this backend handles
    pub domains: Vec<ProblemDomain>,
    /// Problem types this backend handles
    pub problem_types: Vec<ProblemType>,
}

impl Default for NeuralBackendConfig {
    fn default() -> Self {
        Self {
            max_latency_us: 1000, // 1ms target
            min_confidence: 0.5,
            batch_size: 32,
            domains: vec![
                ProblemDomain::Financial,
                ProblemDomain::Physics,
                ProblemDomain::General,
            ],
            problem_types: vec![
                ProblemType::Prediction,
                ProblemType::Estimation,
                ProblemType::Simulation,
            ],
        }
    }
}

impl NeuralBackend {
    /// Create new neural backend
    pub fn new(id: impl Into<String>, name: impl Into<String>, network: Network) -> Self {
        Self::with_config(id, name, network, NeuralBackendConfig::default())
    }

    /// Create new neural backend with configuration
    pub fn with_config(
        id: impl Into<String>,
        name: impl Into<String>,
        network: Network,
        config: NeuralBackendConfig,
    ) -> Self {
        let mut capabilities = HashSet::new();
        capabilities.insert(BackendCapability::Deterministic);
        capabilities.insert(BackendCapability::Differentiable);
        capabilities.insert(BackendCapability::SimdOptimized);

        Self {
            id: BackendId::new(id),
            name: name.into(),
            network: Arc::new(Mutex::new(network)),
            capabilities,
            metrics: Mutex::new(BackendMetrics::default()),
            config,
        }
    }

    /// Add capability
    pub fn with_capability(mut self, cap: BackendCapability) -> Self {
        self.capabilities.insert(cap);
        self
    }

    /// Convert problem data to tensor input
    fn problem_to_tensor(&self, problem: &Problem) -> RouterResult<Tensor> {
        // Extract features from problem based on type
        let features = self.extract_features(problem)?;
        let shape = TensorShape::d2(1, features.len());

        Tensor::new(features, shape)
            .map_err(|e| RouterError::InvalidProblem(e.to_string()))
    }

    /// Extract feature vector from problem
    fn extract_features(&self, problem: &Problem) -> RouterResult<Vec<f64>> {
        let input_dim = self.network.lock().input_dim();

        match &problem.data {
            ProblemData::Vector(nums) => {
                let mut features = nums.clone();
                features.resize(input_dim, 0.0);
                Ok(features)
            },
            ProblemData::TimeSeries { values, .. } => {
                // Use last N values as features
                let n = values.len().min(input_dim);
                let mut features = vec![0.0; input_dim];
                for (i, &v) in values.iter().rev().take(n).enumerate() {
                    features[i] = v;
                }
                Ok(features)
            },
            ProblemData::PhysicsState {
                positions, velocities, masses,
            } => {
                // Concatenate physics state into feature vector
                let mut features = Vec::new();
                features.extend(positions.iter().copied());
                features.extend(velocities.iter().copied());
                features.extend(masses.iter().copied());

                // Pad or truncate to expected input size
                features.resize(input_dim, 0.0);
                Ok(features)
            },
            ProblemData::Matrix { data, .. } => {
                let mut features = data.clone();
                features.resize(input_dim, 0.0);
                Ok(features)
            },
            ProblemData::SparseMatrix { values, .. } => {
                let mut features = values.clone();
                features.resize(input_dim, 0.0);
                Ok(features)
            },
            ProblemData::Graph { edges, .. } => {
                // Flatten edge weights as features
                let mut features: Vec<f64> = edges.iter().map(|(_, _, w)| *w).collect();
                features.resize(input_dim, 0.0);
                Ok(features)
            },
            ProblemData::Json(data) => {
                // Try to extract numerical features from JSON
                let mut features = Vec::new();
                Self::extract_json_features(data, &mut features);
                features.resize(input_dim, 0.0);
                Ok(features)
            },
        }
    }

    /// Recursively extract numerical features from JSON
    fn extract_json_features(value: &serde_json::Value, features: &mut Vec<f64>) {
        match value {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    features.push(f);
                }
            },
            serde_json::Value::Array(arr) => {
                for v in arr {
                    Self::extract_json_features(v, features);
                }
            },
            serde_json::Value::Object(map) => {
                for v in map.values() {
                    Self::extract_json_features(v, features);
                }
            },
            _ => {}
        }
    }

    /// Convert tensor output to result value
    fn tensor_to_result(&self, output: &Tensor, problem_type: ProblemType) -> ResultValue {
        let data = output.data();

        match problem_type {
            ProblemType::Optimization | ProblemType::Estimation => {
                ResultValue::Solution {
                    parameters: data.to_vec(),
                    fitness: data.iter().sum::<f64>() / data.len() as f64,
                }
            },
            ProblemType::Prediction => {
                if data.len() == 1 {
                    ResultValue::Scalar(data[0])
                } else {
                    ResultValue::Vector(data.to_vec())
                }
            },
            ProblemType::Simulation => {
                ResultValue::PhysicsState {
                    positions: data.to_vec(),
                    velocities: vec![0.0; data.len()],
                    energy: data.iter().map(|x| x * x).sum::<f64>().sqrt(),
                }
            },
            ProblemType::Classification => {
                // Find argmax
                let (max_idx, _) = data.iter()
                    .enumerate()
                    .fold((0, f64::NEG_INFINITY), |(idx, max), (i, &v)| {
                        if v > max { (i, v) } else { (idx, max) }
                    });
                ResultValue::Classification {
                    class: max_idx as u32,
                    probabilities: data.to_vec(),
                }
            },
            _ => ResultValue::Vector(data.to_vec()),
        }
    }

    /// Compute confidence from network output
    fn compute_confidence(&self, output: &Tensor) -> f64 {
        let data = output.data();

        // Use output variance as inverse confidence measure
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;

        // Higher variance = lower confidence
        // Map variance to [0, 1] confidence range
        (1.0 / (1.0 + variance)).max(self.config.min_confidence)
    }

    /// Get network reference
    pub fn network(&self) -> &Arc<Mutex<Network>> {
        &self.network
    }

    /// Get configuration
    pub fn config(&self) -> &NeuralBackendConfig {
        &self.config
    }
}

#[async_trait]
impl ReasoningBackend for NeuralBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn pool(&self) -> BackendPool {
        // Neural backends can serve as optimization, physics, or general purpose
        BackendPool::Optimization
    }

    fn supported_domains(&self) -> &[ProblemDomain] {
        &self.config.domains
    }

    fn capabilities(&self) -> &HashSet<BackendCapability> {
        &self.capabilities
    }

    fn latency_tier(&self) -> LatencyTier {
        // Neural inference is typically fast
        if self.config.max_latency_us < 10 {
            LatencyTier::UltraFast
        } else if self.config.max_latency_us < 1000 {
            LatencyTier::Fast
        } else if self.config.max_latency_us < 10000 {
            LatencyTier::Medium
        } else {
            LatencyTier::Slow
        }
    }

    fn can_handle(&self, signature: &ProblemSignature) -> bool {
        self.config.problem_types.contains(&signature.problem_type) &&
        self.config.domains.contains(&signature.domain)
    }

    fn estimate_latency(&self, _signature: &ProblemSignature) -> Duration {
        Duration::from_micros(self.config.max_latency_us)
    }

    async fn execute(&self, problem: &Problem) -> RouterResult<ReasoningResult> {
        let start = Instant::now();

        // Convert problem to input tensor
        let input = self.problem_to_tensor(problem)?;

        // Run inference
        let output = {
            let network = self.network.lock();
            network.predict(&input)
                .map_err(|e| RouterError::BackendFailed {
                    backend_id: self.id.0.clone(),
                    message: e.to_string(),
                })?
        };

        let latency = start.elapsed();

        // Convert output to result
        let value = self.tensor_to_result(&output, problem.signature.problem_type);
        let confidence = self.compute_confidence(&output);
        let quality = confidence; // Use confidence as quality metric

        // Update metrics
        {
            let mut metrics = self.metrics.lock();
            metrics.record(latency, true, Some(quality));
        }

        // Check latency constraint
        if latency > Duration::from_micros(self.config.max_latency_us) {
            tracing::warn!(
                "Neural inference exceeded latency target: {:?} > {}μs",
                latency,
                self.config.max_latency_us
            );
        }

        Ok(ReasoningResult {
            value,
            confidence,
            quality,
            latency,
            backend_id: self.id.clone(),
            metadata: json!({
                "network": self.name,
                "input_dim": self.network.lock().input_dim(),
                "output_dim": self.network.lock().output_dim(),
                "num_params": self.network.lock().num_params(),
                "latency_us": latency.as_micros(),
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
    use crate::network::mlp;

    #[tokio::test]
    async fn test_neural_backend_creation() {
        let network = mlp(10, &[32, 16], 5).unwrap();
        let backend = NeuralBackend::new("test-neural", "Test Neural Network", network);

        assert_eq!(backend.id().0, "test-neural");
        assert_eq!(backend.name(), "Test Neural Network");
        assert!(backend.capabilities().contains(&BackendCapability::Deterministic));
    }

    #[tokio::test]
    async fn test_neural_backend_inference() {
        let network = mlp(10, &[32], 3).unwrap();
        let backend = NeuralBackend::new("test", "Test", network);

        let sig = ProblemSignature::new(ProblemType::Prediction, ProblemDomain::General);
        let problem = Problem::new(sig, ProblemData::Vector(vec![1.0; 10]));

        let result = backend.execute(&problem).await.unwrap();

        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.latency.as_micros() > 0);
    }

    #[test]
    fn test_can_handle() {
        let network = mlp(10, &[32], 3).unwrap();
        let backend = NeuralBackend::new("test", "Test", network);

        let sig = ProblemSignature::new(ProblemType::Prediction, ProblemDomain::Financial);
        assert!(backend.can_handle(&sig));

        let sig2 = ProblemSignature::new(ProblemType::Verification, ProblemDomain::Verification);
        assert!(!backend.can_handle(&sig2));
    }

    #[test]
    fn test_latency_tier() {
        let network = mlp(10, &[32], 3).unwrap();

        let fast_config = NeuralBackendConfig {
            max_latency_us: 100,
            ..Default::default()
        };
        let fast_backend = NeuralBackend::with_config("fast", "Fast", network.clone(), fast_config);
        assert_eq!(fast_backend.latency_tier(), LatencyTier::Fast);

        let slow_config = NeuralBackendConfig {
            max_latency_us: 50000,
            ..Default::default()
        };
        let slow_backend = NeuralBackend::with_config("slow", "Slow", network, slow_config);
        assert_eq!(slow_backend.latency_tier(), LatencyTier::Slow);
    }
}
