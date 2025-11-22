//! Quantum Machine Learning Framework for High-Frequency Trading
//!
//! This module provides quantum-enhanced ML algorithms optimized for trading predictions
//! with sub-100μs inference targets and comprehensive uncertainty quantification.

pub mod quantum_gates;
pub mod qlstm;
pub mod quantum_snn;
pub mod qats_cp;
pub mod webgpu_acceleration;
pub mod uncertainty_quantification;

use crate::TENGRIError;
use nalgebra::{DMatrix, DVector};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};

/// Quantum ML framework configuration
#[derive(Debug, Clone)]
pub struct QuantumMLConfig {
    pub use_quantum_gates: bool,
    pub max_qubits: usize,
    pub inference_timeout_us: u64,
    pub uncertainty_threshold: f64,
    pub webgpu_enabled: bool,
    pub batch_size: usize,
}

impl Default for QuantumMLConfig {
    fn default() -> Self {
        Self {
            use_quantum_gates: true,
            max_qubits: 16,
            inference_timeout_us: 100,
            uncertainty_threshold: 0.95,
            webgpu_enabled: true,
            batch_size: 32,
        }
    }
}

/// Quantum ML prediction result
#[derive(Debug, Clone)]
pub struct QuantumPrediction {
    pub prediction: f64,
    pub confidence: f64,
    pub uncertainty_bounds: (f64, f64),
    pub quantum_state_entropy: f64,
    pub inference_time_ns: u64,
    pub timestamp: DateTime<Utc>,
}

/// Main quantum ML framework
pub struct QuantumMLFramework {
    config: QuantumMLConfig,
    qlstm: Arc<RwLock<qlstm::QuantumLSTM>>,
    quantum_snn: Arc<RwLock<quantum_snn::QuantumSNN>>,
    qats_cp: Arc<RwLock<qats_cp::QuantumATS>>,
    webgpu_accel: Arc<RwLock<webgpu_acceleration::WebGPUAccelerator>>,
    uncertainty_quantifier: Arc<RwLock<uncertainty_quantification::QuantumUncertaintyQuantifier>>,
}

impl QuantumMLFramework {
    /// Initialize quantum ML framework
    pub async fn new(config: QuantumMLConfig) -> Result<Self, TENGRIError> {
        let qlstm = Arc::new(RwLock::new(
            qlstm::QuantumLSTM::new(config.max_qubits, config.batch_size).await?
        ));
        
        let quantum_snn = Arc::new(RwLock::new(
            quantum_snn::QuantumSNN::new(config.max_qubits, config.batch_size).await?
        ));
        
        let qats_cp = Arc::new(RwLock::new(
            qats_cp::QuantumATS::new(config.uncertainty_threshold).await?
        ));
        
        let webgpu_accel = Arc::new(RwLock::new(
            webgpu_acceleration::WebGPUAccelerator::new(config.webgpu_enabled).await?
        ));
        
        let uncertainty_quantifier = Arc::new(RwLock::new(
            uncertainty_quantification::QuantumUncertaintyQuantifier::new(config.max_qubits).await?
        ));

        Ok(Self {
            config,
            qlstm,
            quantum_snn,
            qats_cp,
            webgpu_accel,
            uncertainty_quantifier,
        })
    }

    /// Perform quantum-enhanced prediction with sub-100μs target
    pub async fn predict(&self, input: &DMatrix<f64>) -> Result<QuantumPrediction, TENGRIError> {
        let start_time = std::time::Instant::now();
        
        // Parallel quantum predictions
        let (qlstm_pred, snn_pred, quantum_uncertainty) = tokio::try_join!(
            self.qlstm.read().await.predict(input),
            self.quantum_snn.read().await.predict(input),
            self.uncertainty_quantifier.read().await.quantify_uncertainty(input)
        )?;

        // Combine predictions using quantum adaptive temperature scaling
        let combined_prediction = self.qats_cp.read().await.combine_predictions(
            &[qlstm_pred, snn_pred], 
            &quantum_uncertainty
        ).await?;

        let inference_time = start_time.elapsed();
        
        // Ensure sub-100μs target
        if inference_time.as_micros() > self.config.inference_timeout_us as u128 {
            tracing::warn!(
                "Quantum ML inference exceeded {}μs target: {}μs",
                self.config.inference_timeout_us,
                inference_time.as_micros()
            );
        }

        Ok(QuantumPrediction {
            prediction: combined_prediction.value,
            confidence: combined_prediction.confidence,
            uncertainty_bounds: combined_prediction.uncertainty_bounds,
            quantum_state_entropy: quantum_uncertainty.entropy,
            inference_time_ns: inference_time.as_nanos() as u64,
            timestamp: Utc::now(),
        })
    }

    /// Update models with new market data
    pub async fn update_models(&self, training_data: &DMatrix<f64>, targets: &DVector<f64>) -> Result<(), TENGRIError> {
        // Parallel model updates
        tokio::try_join!(
            self.qlstm.write().await.update(training_data, targets),
            self.quantum_snn.write().await.update(training_data, targets),
            self.qats_cp.write().await.update(training_data, targets)
        )?;

        Ok(())
    }

    /// Get current quantum state metrics
    pub async fn get_quantum_metrics(&self) -> Result<QuantumMetrics, TENGRIError> {
        let qlstm_metrics = self.qlstm.read().await.get_metrics().await?;
        let snn_metrics = self.quantum_snn.read().await.get_metrics().await?;
        let uncertainty_metrics = self.uncertainty_quantifier.read().await.get_metrics().await?;

        Ok(QuantumMetrics {
            qlstm_quantum_fidelity: qlstm_metrics.quantum_fidelity,
            snn_spike_coherence: snn_metrics.spike_coherence,
            uncertainty_entropy: uncertainty_metrics.entropy,
            overall_quantum_advantage: (qlstm_metrics.quantum_fidelity + snn_metrics.spike_coherence) / 2.0,
        })
    }
}

/// Quantum ML metrics
#[derive(Debug, Clone)]
pub struct QuantumMetrics {
    pub qlstm_quantum_fidelity: f64,
    pub snn_spike_coherence: f64,
    pub uncertainty_entropy: f64,
    pub overall_quantum_advantage: f64,
}

/// Combined prediction result
#[derive(Debug, Clone)]
pub struct CombinedPrediction {
    pub value: f64,
    pub confidence: f64,
    pub uncertainty_bounds: (f64, f64),
}

/// Quantum uncertainty result
#[derive(Debug, Clone)]
pub struct QuantumUncertainty {
    pub entropy: f64,
    pub variance: f64,
    pub confidence_interval: (f64, f64),
}