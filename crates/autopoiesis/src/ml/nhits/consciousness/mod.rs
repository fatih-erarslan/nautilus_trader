/// NHITS Consciousness Integration Module
/// 
/// This module implements consciousness-aware neural hierarchical interpolation for time series.
/// It integrates autopoietic consciousness principles with NHITS forecasting for enhanced
/// temporal pattern recognition and prediction accuracy.

pub mod syntergic_forecasting;
pub mod consciousness_attention;
pub mod adaptive_learning;
pub mod quantum_forecasting;
pub mod market_consciousness;
pub mod temporal_consciousness;
pub mod field_synchronization;
pub mod syntergic_ensemble;
pub mod consciousness_anomaly;

pub use syntergic_forecasting::SyntergicForecaster;
pub use consciousness_attention::ConsciousnessAttention;
pub use adaptive_learning::AdaptiveLearning;
pub use quantum_forecasting::QuantumForecaster;
pub use market_consciousness::MarketConsciousness;
pub use temporal_consciousness::TemporalConsciousness;
pub use field_synchronization::FieldSynchronization;
pub use syntergic_ensemble::SyntergicEnsemble;
pub use consciousness_anomaly::ConsciousnessAnomaly;

// Add missing types needed by tests
pub type ConsciousnessIntegration = NHITSConsciousness;
pub type AttentionMechanism = ConsciousnessAttention;

use ndarray::{Array2, Array1};
use nalgebra::{DMatrix, DVector};
use crate::consciousness::field_coherence::QuantumField;
use crate::consciousness::core::ConsciousnessState;

/// Main NHITS Consciousness Integration System
pub struct NHITSConsciousness {
    pub syntergic_forecaster: SyntergicForecaster,
    pub attention: ConsciousnessAttention,
    pub adaptive_learning: AdaptiveLearning,
    pub quantum_forecaster: QuantumForecaster,
    pub market_consciousness: MarketConsciousness,
    pub temporal_consciousness: TemporalConsciousness,
    pub field_sync: FieldSynchronization,
    pub ensemble: SyntergicEnsemble,
    pub anomaly_detector: ConsciousnessAnomaly,
    pub consciousness_state: ConsciousnessState,
}

impl NHITSConsciousness {
    pub fn new(input_size: usize, forecast_horizon: usize) -> Self {
        Self {
            syntergic_forecaster: SyntergicForecaster::new(input_size, forecast_horizon),
            attention: ConsciousnessAttention::new(input_size),
            adaptive_learning: AdaptiveLearning::new(0.01),
            quantum_forecaster: QuantumForecaster::new(input_size),
            market_consciousness: MarketConsciousness::new(input_size),
            temporal_consciousness: TemporalConsciousness::new(input_size),
            field_sync: FieldSynchronization::new(),
            ensemble: SyntergicEnsemble::new(5),
            anomaly_detector: ConsciousnessAnomaly::new(input_size),
            consciousness_state: ConsciousnessState::new(),
        }
    }

    /// Perform consciousness-aware forecasting
    pub fn forecast(&mut self, input: &Array2<f64>) -> Array1<f64> {
        // Update consciousness state
        self.consciousness_state.update_state(input);
        
        // Synchronize field coherence
        self.field_sync.synchronize(&self.consciousness_state);
        
        // Apply consciousness attention
        let attended_input = self.attention.apply_attention(input, &self.consciousness_state);
        
        // Generate forecasts from multiple consciousness models
        let syntergic_forecast = self.syntergic_forecaster.forecast(&attended_input);
        let quantum_forecast = self.quantum_forecaster.forecast(&attended_input);
        let market_forecast = self.market_consciousness.predict(&attended_input);
        let temporal_forecast = self.temporal_consciousness.evolve_patterns(&attended_input);
        
        // Ensemble predictions with consciousness coherence
        let ensemble_forecast = self.ensemble.combine_forecasts(&[
            syntergic_forecast,
            quantum_forecast,
            market_forecast,
            temporal_forecast
        ], &self.consciousness_state);
        
        // Detect anomalies in consciousness patterns
        self.anomaly_detector.detect_anomalies(&ensemble_forecast, &self.consciousness_state);
        
        // Adaptive learning from consciousness feedback
        self.adaptive_learning.learn_from_consciousness(&ensemble_forecast, &self.consciousness_state);
        
        ensemble_forecast
    }
}