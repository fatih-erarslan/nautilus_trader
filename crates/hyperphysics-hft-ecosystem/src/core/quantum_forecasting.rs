//! Quantum-Inspired Forecasting Integration
//!
//! Integrates hyperphysics-ml quantum-inspired models into the HFT pipeline.
//! Provides BioCognitive LSTM and complex-valued forecasting for market prediction.
//!
//! # Architecture
//!
//! ```text
//! Market Data → StateEncoder → QuantumInspiredLSTM → Forecast
//!                    ↓
//!              BioCognitiveLSTM → Biological Dynamics Signal
//! ```

#[cfg(feature = "quantum-forecasting")]
use hyperphysics_ml::quantum::{
    BioCognitiveLSTM, BioCognitiveConfig, BioCognitiveState,
    ComplexLSTM, ComplexLSTMState,
    StateEncoder, EncodingType,
    Complex, CoherenceMetrics,
};

use crate::{EcosystemError, Result};
use std::time::Instant;

/// Quantum-inspired forecast result
#[derive(Debug, Clone)]
pub struct QuantumForecastResult {
    /// Primary prediction (expected return direction)
    pub prediction: f64,
    /// Prediction confidence [0, 1]
    pub confidence: f64,
    /// Phase coherence metric (higher = more stable prediction)
    pub phase_coherence: f64,
    /// Biological rhythm signal (gamma/theta oscillation strength)
    pub rhythm_signal: f64,
    /// Quantum tunneling events (barrier crossings)
    pub tunneling_events: usize,
    /// Inference latency in microseconds
    pub latency_us: u64,
    /// Model type used
    pub model_type: QuantumModelType,
}

/// Types of quantum-inspired models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumModelType {
    /// BioCognitive LSTM with biological dynamics
    BioCognitive,
    /// Complex-valued LSTM with phase coherence
    ComplexLSTM,
    /// Ensemble of both models
    Ensemble,
}

/// Configuration for quantum forecasting
#[derive(Debug, Clone)]
pub struct QuantumForecastConfig {
    /// Input window size (number of time steps)
    pub window_size: usize,
    /// Hidden state dimension
    pub hidden_size: usize,
    /// Number of LSTM layers
    pub num_layers: usize,
    /// State encoding type
    pub encoding_type: QuantumEncodingType,
    /// Enable biological effects (oscillations, plasticity)
    pub biological_effects: bool,
    /// Decoherence rate for quantum-inspired dynamics
    pub decoherence_rate: f32,
    /// Model type to use
    pub model_type: QuantumModelType,
    /// Confidence threshold for trading decisions
    pub confidence_threshold: f64,
}

/// Encoding types (mirror of hyperphysics-ml types)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumEncodingType {
    /// Data values become amplitudes
    Amplitude,
    /// Data values become rotation angles
    Angle,
    /// Data values become phase factors
    Phase,
    /// Combination of amplitude and phase
    Hybrid,
    /// Instantaneous Quantum Polynomial
    IQP,
}

impl Default for QuantumForecastConfig {
    fn default() -> Self {
        Self {
            window_size: 20,
            hidden_size: 64,
            num_layers: 2,
            encoding_type: QuantumEncodingType::Amplitude,
            biological_effects: true,
            decoherence_rate: 0.05,
            model_type: QuantumModelType::BioCognitive,
            confidence_threshold: 0.6,
        }
    }
}

impl QuantumForecastConfig {
    /// Create HFT-optimized configuration
    pub fn hft_optimized() -> Self {
        Self {
            window_size: 10,        // Shorter window for HFT
            hidden_size: 32,        // Smaller for speed
            num_layers: 1,          // Single layer for minimal latency
            encoding_type: QuantumEncodingType::Amplitude,
            biological_effects: true,
            decoherence_rate: 0.1,  // Faster decoherence for adaptability
            model_type: QuantumModelType::BioCognitive,
            confidence_threshold: 0.5,
        }
    }

    /// Create high-accuracy configuration
    pub fn high_accuracy() -> Self {
        Self {
            window_size: 50,
            hidden_size: 128,
            num_layers: 3,
            encoding_type: QuantumEncodingType::Hybrid,
            biological_effects: true,
            decoherence_rate: 0.02,
            model_type: QuantumModelType::Ensemble,
            confidence_threshold: 0.7,
        }
    }
}

/// Quantum-inspired forecast engine
///
/// Integrates BioCognitive LSTM and complex-valued models for market prediction.
#[cfg(feature = "quantum-forecasting")]
pub struct QuantumForecastEngine {
    /// Configuration
    config: QuantumForecastConfig,
    /// BioCognitive LSTM model
    bio_model: BioCognitiveLSTM,
    /// BioCognitive hidden state
    bio_state: BioCognitiveState,
    /// Complex LSTM model (for ensemble)
    complex_model: ComplexLSTM,
    /// Complex LSTM state
    complex_state: ComplexLSTMState,
    /// State encoder
    encoder: StateEncoder,
    /// Inference count
    inference_count: u64,
    /// Total latency for averaging
    total_latency_us: u64,
}

#[cfg(feature = "quantum-forecasting")]
impl QuantumForecastEngine {
    /// Create new quantum forecast engine
    pub fn new(config: QuantumForecastConfig) -> Self {
        // Create BioCognitive model
        let bio_config = BioCognitiveConfig::new(config.window_size, config.hidden_size)
            .with_num_layers(config.num_layers)
            .with_oscillations(config.biological_effects, config.biological_effects);

        let bio_model = BioCognitiveLSTM::from_config(bio_config);
        let bio_state = BioCognitiveState::new(config.hidden_size);

        // Create Complex LSTM model
        let complex_model = ComplexLSTM::with_dims(config.window_size, config.hidden_size);
        let complex_state = ComplexLSTMState::new(config.hidden_size);

        // Create encoder based on config
        let encoding = match config.encoding_type {
            QuantumEncodingType::Amplitude => EncodingType::Amplitude,
            QuantumEncodingType::Angle => EncodingType::Angle,
            QuantumEncodingType::Phase => EncodingType::Phase,
            QuantumEncodingType::Hybrid => EncodingType::Hybrid,
            QuantumEncodingType::IQP => EncodingType::IQP,
        };
        let num_qubits = ((config.window_size as f32).log2().ceil() as usize).max(2);
        let encoder = StateEncoder::new(encoding, num_qubits);

        Self {
            config,
            bio_model,
            bio_state,
            complex_model,
            complex_state,
            encoder,
            inference_count: 0,
            total_latency_us: 0,
        }
    }

    /// Create HFT-optimized engine
    pub fn hft_optimized() -> Self {
        Self::new(QuantumForecastConfig::hft_optimized())
    }

    /// Run forecast on market returns
    pub fn forecast(&mut self, returns: &[f64]) -> Result<QuantumForecastResult> {
        let start = Instant::now();

        // Convert to f32 for model
        let input: Vec<f32> = returns.iter().map(|&x| x as f32).collect();

        // Pad or truncate to window size
        let padded = self.prepare_input(&input);

        // Run appropriate model(s)
        let result = match self.config.model_type {
            QuantumModelType::BioCognitive => self.run_bio_cognitive(&padded),
            QuantumModelType::ComplexLSTM => self.run_complex_lstm(&padded),
            QuantumModelType::Ensemble => self.run_ensemble(&padded),
        };

        let latency_us = start.elapsed().as_micros() as u64;

        // Update statistics
        self.inference_count += 1;
        self.total_latency_us += latency_us;

        Ok(QuantumForecastResult {
            prediction: result.0,
            confidence: result.1,
            phase_coherence: result.2,
            rhythm_signal: result.3,
            tunneling_events: result.4,
            latency_us,
            model_type: self.config.model_type,
        })
    }

    /// Prepare input to match window size
    fn prepare_input(&self, input: &[f32]) -> Vec<f32> {
        let window = self.config.window_size;
        if input.len() >= window {
            input[input.len() - window..].to_vec()
        } else {
            // Pad with zeros at the beginning
            let mut padded = vec![0.0f32; window - input.len()];
            padded.extend_from_slice(input);
            padded
        }
    }

    /// Run BioCognitive LSTM
    fn run_bio_cognitive(&mut self, input: &[f32]) -> (f64, f64, f64, f64, usize) {
        // Process through BioCognitive LSTM
        let (output, new_state) = self.bio_model.forward_simple(input, &self.bio_state);
        self.bio_state = new_state;

        // Extract prediction from output (last hidden state)
        let prediction = if !output.is_empty() {
            output.iter().sum::<f32>() / output.len() as f32
        } else {
            0.0
        };

        // Compute confidence from membrane potential variance
        let membrane_mean: f32 = self.bio_state.membrane.iter().sum::<f32>()
            / self.bio_state.membrane.len() as f32;
        let membrane_var: f32 = self.bio_state.membrane.iter()
            .map(|&x| (x - membrane_mean).powi(2))
            .sum::<f32>() / self.bio_state.membrane.len() as f32;
        let confidence = (1.0 - membrane_var.sqrt().min(1.0)) as f64;

        // Phase coherence from hidden state
        let h_sum: f32 = self.bio_state.h.iter().sum();
        let h_norm: f32 = self.bio_state.h.iter().map(|x| x.abs()).sum();
        let phase_coherence = if h_norm > 1e-6 {
            (h_sum.abs() / h_norm) as f64
        } else {
            0.0
        };

        // Rhythm signal from recent activity
        let rhythm_signal = self.bio_state.membrane.iter()
            .map(|&x| x.abs())
            .sum::<f32>() as f64 / self.bio_state.membrane.len() as f64;

        // Tunneling events from spike history (count neurons that have spiked)
        let tunneling_events = self.bio_state.spike_history.iter()
            .filter(|spikes| !spikes.is_empty())
            .count();

        (
            prediction as f64,
            confidence,
            phase_coherence,
            rhythm_signal,
            tunneling_events,
        )
    }

    /// Run Complex LSTM
    fn run_complex_lstm(&mut self, input: &[f32]) -> (f64, f64, f64, f64, usize) {
        // Process through Complex LSTM
        let (output, new_state) = self.complex_model.forward(input, &self.complex_state);
        self.complex_state = new_state;

        // Extract prediction from output
        let prediction = if !output.is_empty() {
            output.iter().sum::<f32>() / output.len() as f32
        } else {
            0.0
        };

        // Compute phase coherence from complex hidden state
        let phase_coherence = self.complex_state.phase_coherence() as f64;

        // Confidence from output magnitude
        let output_magnitude: f32 = output.iter().map(|x| x.abs()).sum::<f32>()
            / output.len().max(1) as f32;
        let confidence = output_magnitude.tanh() as f64;

        // No biological rhythm for pure complex LSTM
        let rhythm_signal = 0.0;
        let tunneling_events = 0;

        (
            prediction as f64,
            confidence,
            phase_coherence,
            rhythm_signal,
            tunneling_events,
        )
    }

    /// Run ensemble of both models
    fn run_ensemble(&mut self, input: &[f32]) -> (f64, f64, f64, f64, usize) {
        let bio_result = self.run_bio_cognitive(input);
        let complex_result = self.run_complex_lstm(input);

        // Weighted average based on confidence
        let total_conf = bio_result.1 + complex_result.1;
        let bio_weight = if total_conf > 0.0 { bio_result.1 / total_conf } else { 0.5 };
        let complex_weight = 1.0 - bio_weight;

        (
            bio_result.0 * bio_weight + complex_result.0 * complex_weight,
            (bio_result.1 + complex_result.1) / 2.0,  // Average confidence
            (bio_result.2 + complex_result.2) / 2.0,  // Average phase coherence
            bio_result.3,  // Only bio has rhythm signal
            bio_result.4,  // Only bio has tunneling
        )
    }

    /// Reset model states
    pub fn reset(&mut self) {
        self.bio_state = BioCognitiveState::new(self.config.hidden_size);
        self.complex_state = ComplexLSTMState::new(self.config.hidden_size);
    }

    /// Get average inference latency
    pub fn avg_latency_us(&self) -> f64 {
        if self.inference_count > 0 {
            self.total_latency_us as f64 / self.inference_count as f64
        } else {
            0.0
        }
    }

    /// Get total inference count
    pub fn inference_count(&self) -> u64 {
        self.inference_count
    }
}

/// Stub implementation when quantum-forecasting feature is disabled
#[cfg(not(feature = "quantum-forecasting"))]
pub struct QuantumForecastEngine {
    config: QuantumForecastConfig,
}

#[cfg(not(feature = "quantum-forecasting"))]
impl QuantumForecastEngine {
    /// Create new quantum forecast engine (no-op)
    pub fn new(_config: QuantumForecastConfig) -> Self {
        Self {
            config: QuantumForecastConfig::default(),
        }
    }

    /// Create HFT-optimized engine (no-op)
    pub fn hft_optimized() -> Self {
        Self::new(QuantumForecastConfig::hft_optimized())
    }

    /// Run forecast (no-op)
    pub fn forecast(&mut self, _returns: &[f64]) -> Result<QuantumForecastResult> {
        Ok(QuantumForecastResult {
            prediction: 0.0,
            confidence: 0.0,
            phase_coherence: 0.0,
            rhythm_signal: 0.0,
            tunneling_events: 0,
            latency_us: 0,
            model_type: QuantumModelType::BioCognitive,
        })
    }

    /// Reset model states (no-op)
    pub fn reset(&mut self) {}

    /// Get average inference latency (no-op)
    pub fn avg_latency_us(&self) -> f64 {
        0.0
    }

    /// Get total inference count (no-op)
    pub fn inference_count(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_forecast_config_default() {
        let config = QuantumForecastConfig::default();
        assert_eq!(config.window_size, 20);
        assert_eq!(config.hidden_size, 64);
        assert!(config.biological_effects);
    }

    #[test]
    fn test_quantum_forecast_config_hft() {
        let config = QuantumForecastConfig::hft_optimized();
        assert_eq!(config.window_size, 10);
        assert_eq!(config.hidden_size, 32);
        assert_eq!(config.num_layers, 1);
    }

    #[cfg(feature = "quantum-forecasting")]
    #[test]
    fn test_quantum_forecast_engine_creation() {
        let config = QuantumForecastConfig::default();
        let engine = QuantumForecastEngine::new(config);
        assert_eq!(engine.inference_count(), 0);
    }

    #[cfg(feature = "quantum-forecasting")]
    #[test]
    fn test_quantum_forecast_inference() {
        let mut engine = QuantumForecastEngine::hft_optimized();
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.003, 0.01, 0.005, -0.002];

        let result = engine.forecast(&returns);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.phase_coherence >= 0.0);
        assert_eq!(result.model_type, QuantumModelType::BioCognitive);
        assert_eq!(engine.inference_count(), 1);
    }

    #[cfg(feature = "quantum-forecasting")]
    #[test]
    fn test_quantum_forecast_latency() {
        let mut engine = QuantumForecastEngine::hft_optimized();
        let returns = vec![0.01; 10];

        // Run multiple inferences
        for _ in 0..10 {
            let _ = engine.forecast(&returns);
        }

        // Check latency tracking
        assert!(engine.avg_latency_us() > 0.0);
        assert_eq!(engine.inference_count(), 10);
    }

    #[cfg(feature = "quantum-forecasting")]
    #[test]
    fn test_quantum_forecast_reset() {
        let mut engine = QuantumForecastEngine::hft_optimized();
        let returns = vec![0.01; 10];

        // Run inference to update state
        let _ = engine.forecast(&returns);

        // Reset and verify
        engine.reset();
        // State should be reset (can't easily verify without exposing internals)
    }

    #[cfg(feature = "quantum-forecasting")]
    #[test]
    fn test_ensemble_model() {
        let config = QuantumForecastConfig {
            model_type: QuantumModelType::Ensemble,
            ..Default::default()
        };
        let mut engine = QuantumForecastEngine::new(config);
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];

        let result = engine.forecast(&returns);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().model_type, QuantumModelType::Ensemble);
    }
}
