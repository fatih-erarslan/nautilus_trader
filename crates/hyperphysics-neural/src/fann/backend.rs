//! FANN-based reasoning backend for router integration
//!
//! Implements ReasoningBackend trait using ruv-FANN for high-performance inference.

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hyperphysics_reasoning_router::{
    BackendPool, LatencyTier, ProblemDomain, RouterError, RouterResult,
    backend::{BackendCapability, BackendId, BackendMetrics, ReasoningBackend, ReasoningResult, ResultValue},
    problem::{Problem, ProblemData, ProblemSignature, ProblemType},
};

use super::network::{FannConfig, FannNetwork};

/// FANN-based reasoning backend
pub struct FannBackend {
    /// Backend identifier
    id: BackendId,
    /// Backend name
    name: String,
    /// FANN network
    network: Arc<Mutex<FannNetwork>>,
    /// Capabilities
    capabilities: HashSet<BackendCapability>,
    /// Metrics
    metrics: Mutex<BackendMetrics>,
    /// Supported domains
    domains: Vec<ProblemDomain>,
    /// Supported problem types
    problem_types: Vec<ProblemType>,
    /// Maximum latency target
    max_latency_us: u64,
}

impl FannBackend {
    /// Create new FANN backend
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        network: FannNetwork,
    ) -> Self {
        let max_latency_us = network.config().max_latency_us.unwrap_or(1000);

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
            max_latency_us,
        }
    }

    /// Create HFT-optimized FANN backend
    pub fn hft(
        id: impl Into<String>,
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
    ) -> Result<Self, super::network::FannError> {
        let config = FannConfig::hft(input_dim, hidden_dims, output_dim);
        let network = FannNetwork::new(config)?;

        let mut backend = Self::new(id, "FANN-HFT", network);
        backend.max_latency_us = 10; // 10μs for HFT
        Ok(backend)
    }

    /// Create classification FANN backend
    pub fn classifier(
        id: impl Into<String>,
        input_dim: usize,
        hidden_dims: &[usize],
        num_classes: usize,
    ) -> Result<Self, super::network::FannError> {
        let config = FannConfig::classification(input_dim, hidden_dims, num_classes);
        let network = FannNetwork::new(config)?;

        let mut backend = Self::new(id, "FANN-Classifier", network);
        backend.problem_types = vec![ProblemType::Classification];
        Ok(backend)
    }

    /// Add capability
    pub fn with_capability(mut self, cap: BackendCapability) -> Self {
        self.capabilities.insert(cap);
        self
    }

    /// Set supported domains
    pub fn with_domains(mut self, domains: Vec<ProblemDomain>) -> Self {
        self.domains = domains;
        self
    }

    /// Extract features from problem
    fn extract_features(&self, problem: &Problem) -> RouterResult<Vec<f64>> {
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
            ProblemData::PhysicsState {
                positions,
                velocities,
                masses,
            } => {
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
            ProblemData::SparseMatrix { values, .. } => {
                let mut features = values.clone();
                features.resize(input_dim, 0.0);
                Ok(features)
            }
            ProblemData::Graph { edges, .. } => {
                let mut features: Vec<f64> = edges.iter().map(|(_, _, w)| *w).collect();
                features.resize(input_dim, 0.0);
                Ok(features)
            }
            ProblemData::Json(data) => {
                let mut features = Vec::new();
                Self::extract_json_features(data, &mut features);
                features.resize(input_dim, 0.0);
                Ok(features)
            }
        }
    }

    /// Recursively extract numerical features from JSON
    fn extract_json_features(value: &serde_json::Value, features: &mut Vec<f64>) {
        match value {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    features.push(f);
                }
            }
            serde_json::Value::Array(arr) => {
                for v in arr {
                    Self::extract_json_features(v, features);
                }
            }
            serde_json::Value::Object(map) => {
                for v in map.values() {
                    Self::extract_json_features(v, features);
                }
            }
            _ => {}
        }
    }

    /// Convert output to result value
    fn output_to_result(&self, output: &[f64], problem_type: ProblemType) -> ResultValue {
        match problem_type {
            ProblemType::Classification => {
                let (max_idx, _) = output
                    .iter()
                    .enumerate()
                    .fold((0, f64::NEG_INFINITY), |(idx, max), (i, &v)| {
                        if v > max {
                            (i, v)
                        } else {
                            (idx, max)
                        }
                    });
                ResultValue::Classification {
                    class: max_idx as u32,
                    probabilities: output.to_vec(),
                }
            }
            ProblemType::Prediction | ProblemType::Estimation => {
                if output.len() == 1 {
                    ResultValue::Scalar(output[0])
                } else {
                    ResultValue::Vector(output.to_vec())
                }
            }
            ProblemType::Simulation => ResultValue::PhysicsState {
                positions: output.to_vec(),
                velocities: vec![0.0; output.len()],
                energy: output.iter().map(|x| x * x).sum::<f64>().sqrt(),
            },
            ProblemType::Optimization => ResultValue::Solution {
                parameters: output.to_vec(),
                fitness: output.iter().sum::<f64>() / output.len() as f64,
            },
            _ => ResultValue::Vector(output.to_vec()),
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

    /// Get network reference
    pub fn network(&self) -> &Arc<Mutex<FannNetwork>> {
        &self.network
    }
}

#[async_trait]
impl ReasoningBackend for FannBackend {
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
        if self.max_latency_us <= 10 {
            LatencyTier::UltraFast
        } else if self.max_latency_us < 1000 {
            LatencyTier::Fast
        } else if self.max_latency_us < 10000 {
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
        Duration::from_micros(self.max_latency_us)
    }

    async fn execute(&self, problem: &Problem) -> RouterResult<ReasoningResult> {
        let start = Instant::now();

        // Extract features
        let features = self.extract_features(problem)?;

        // Run inference
        let output = {
            let mut network = self.network.lock();
            network.forward(&features).map_err(|e| RouterError::BackendFailed {
                backend_id: self.id.0.clone(),
                message: e.to_string(),
            })?
        };

        let latency = start.elapsed();

        // Convert output
        let value = self.output_to_result(&output, problem.signature.problem_type);
        let confidence = self.compute_confidence(&output);
        let quality = confidence;

        // Update metrics
        {
            let mut metrics = self.metrics.lock();
            metrics.record(latency, true, Some(quality));
        }

        // Check latency
        if latency > Duration::from_micros(self.max_latency_us) {
            tracing::warn!(
                "FANN inference exceeded latency: {:?} > {}μs",
                latency,
                self.max_latency_us
            );
        }

        let network = self.network.lock();
        Ok(ReasoningResult {
            value,
            confidence,
            quality,
            latency,
            backend_id: self.id.clone(),
            metadata: json!({
                "backend": "ruv-fann",
                "network": self.name,
                "input_dim": network.input_dim(),
                "output_dim": network.output_dim(),
                "connections": network.num_connections(),
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

    #[tokio::test]
    async fn test_fann_backend_creation() {
        let network = FannNetwork::mlp(10, &[32, 16], 5).unwrap();
        let backend = FannBackend::new("fann-test", "Test FANN", network);

        assert_eq!(backend.id().0, "fann-test");
        assert_eq!(backend.name(), "Test FANN");
        assert!(backend.capabilities().contains(&BackendCapability::Deterministic));
    }

    #[tokio::test]
    async fn test_fann_hft_backend() {
        let backend = FannBackend::hft("fann-hft", 10, &[32], 3).unwrap();

        assert_eq!(backend.latency_tier(), LatencyTier::UltraFast);
        assert_eq!(backend.max_latency_us, 10);
    }

    #[tokio::test]
    async fn test_fann_backend_inference() {
        let network = FannNetwork::mlp(10, &[16], 3).unwrap();
        let backend = FannBackend::new("test", "Test", network);

        let sig = ProblemSignature::new(ProblemType::Prediction, ProblemDomain::General);
        let problem = Problem::new(sig, ProblemData::Vector(vec![1.0; 10]));

        let result = backend.execute(&problem).await.unwrap();

        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.latency.as_micros() > 0);
    }

    #[test]
    fn test_fann_can_handle() {
        let network = FannNetwork::mlp(10, &[16], 3).unwrap();
        let backend = FannBackend::new("test", "Test", network);

        let sig = ProblemSignature::new(ProblemType::Prediction, ProblemDomain::Financial);
        assert!(backend.can_handle(&sig));

        let sig2 = ProblemSignature::new(ProblemType::Verification, ProblemDomain::Verification);
        assert!(!backend.can_handle(&sig2));
    }
}
