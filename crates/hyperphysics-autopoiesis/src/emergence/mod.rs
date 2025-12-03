//! Emergence patterns and collective states in autopoietic systems
//!
//! This module defines the data structures and analysis tools for identifying
//! and characterizing emergent phenomena in complex adaptive systems.
//!
//! ## Theoretical Background
//!
//! Emergence refers to higher-order properties that arise from component interactions
//! but cannot be reduced to individual components. Key signatures include:
//!
//! - **Downward causation**: Collective states influencing component behavior
//! - **Multiple realizability**: Same emergent pattern from different substrates
//! - **Irreducibility**: Cannot be predicted from component properties alone
//!
//! ## References
//! - Bedau (1997) "Weak Emergence"
//! - Kauffman (2000) "Investigations"
//! - Holland (1998) "Emergence: From Chaos to Order"

use std::collections::HashMap;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Categories of emergent patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternCategory {
    /// Spatial patterns (Turing patterns, spirals)
    Spatial,
    /// Temporal patterns (oscillations, rhythms)
    Temporal,
    /// Spatiotemporal patterns (traveling waves)
    Spatiotemporal,
    /// Network patterns (communities, hierarchies)
    Network,
    /// Information patterns (compression, integration)
    Informational,
    /// Behavioral patterns (collective motion, swarming)
    Behavioral,
}

/// Stability classification for emergent patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternStability {
    /// Pattern persists indefinitely
    Stable,
    /// Pattern persists but with fluctuations
    Metastable,
    /// Pattern is transient
    Transient,
    /// Pattern exhibits critical dynamics
    Critical,
}

/// Representation of an emergent pattern
///
/// Emergent patterns are higher-order regularities that arise from
/// component interactions in autopoietic systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentPattern {
    /// Unique identifier
    pub id: Uuid,
    /// Pattern category
    pub category: PatternCategory,
    /// Stability classification
    pub stability: PatternStability,
    /// Characteristic scale (spatial or temporal)
    pub scale: f64,
    /// Coherence measure (0 = noise, 1 = perfect pattern)
    pub coherence: f64,
    /// Participation ratio (how many components involved)
    pub participation: f64,
    /// Pattern signature vector
    pub signature: Vec<f64>,
    /// When pattern was first detected
    pub detected_at: chrono::DateTime<chrono::Utc>,
    /// Duration pattern has persisted
    pub duration_ms: u64,
    /// Associated metadata
    pub metadata: HashMap<String, String>,
}

impl EmergentPattern {
    /// Create a new emergent pattern
    pub fn new(category: PatternCategory, stability: PatternStability) -> Self {
        Self {
            id: Uuid::new_v4(),
            category,
            stability,
            scale: 1.0,
            coherence: 0.0,
            participation: 0.0,
            signature: Vec::new(),
            detected_at: chrono::Utc::now(),
            duration_ms: 0,
            metadata: HashMap::new(),
        }
    }

    /// Set pattern scale
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Set pattern coherence
    pub fn with_coherence(mut self, coherence: f64) -> Self {
        self.coherence = coherence.clamp(0.0, 1.0);
        self
    }

    /// Set participation ratio
    pub fn with_participation(mut self, participation: f64) -> Self {
        self.participation = participation;
        self
    }

    /// Set pattern signature
    pub fn with_signature(mut self, signature: Vec<f64>) -> Self {
        self.signature = signature;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Update duration based on current time
    pub fn update_duration(&mut self) {
        let now = chrono::Utc::now();
        self.duration_ms = (now - self.detected_at).num_milliseconds() as u64;
    }

    /// Check if pattern is considered robust (stable and coherent)
    pub fn is_robust(&self) -> bool {
        self.coherence > 0.7
            && (self.stability == PatternStability::Stable
                || self.stability == PatternStability::Metastable)
    }
}

/// Types of emergence events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergenceEventType {
    /// New pattern appeared
    PatternFormation,
    /// Existing pattern dissolved
    PatternDissolution,
    /// Pattern transitioned to different type
    PatternTransition,
    /// Multiple patterns synchronized
    PatternSynchronization,
    /// Pattern split into multiple patterns
    PatternBifurcation,
    /// Patterns merged into single pattern
    PatternMerger,
}

/// Record of an emergence event
///
/// Tracks significant changes in the emergent landscape of the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceEvent {
    /// Unique identifier
    pub id: Uuid,
    /// Event type
    pub event_type: EmergenceEventType,
    /// When event occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Patterns involved (by ID)
    pub pattern_ids: Vec<Uuid>,
    /// Control parameter value at event
    pub control_parameter: Option<f64>,
    /// Energy change associated with event
    pub energy_delta: Option<f64>,
    /// Entropy change associated with event
    pub entropy_delta: Option<f64>,
    /// Event description
    pub description: String,
}

impl EmergenceEvent {
    /// Create new emergence event
    pub fn new(event_type: EmergenceEventType, description: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            event_type,
            timestamp: chrono::Utc::now(),
            pattern_ids: Vec::new(),
            control_parameter: None,
            energy_delta: None,
            entropy_delta: None,
            description: description.to_string(),
        }
    }

    /// Add pattern to event
    pub fn with_pattern(mut self, pattern_id: Uuid) -> Self {
        self.pattern_ids.push(pattern_id);
        self
    }

    /// Set control parameter
    pub fn with_control_parameter(mut self, value: f64) -> Self {
        self.control_parameter = Some(value);
        self
    }

    /// Set thermodynamic changes
    pub fn with_thermodynamics(mut self, energy: f64, entropy: f64) -> Self {
        self.energy_delta = Some(energy);
        self.entropy_delta = Some(entropy);
        self
    }
}

/// Collective state of the system
///
/// Represents the macroscopic state emerging from microscopic components,
/// capturing order parameters and collective observables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveState {
    /// Primary order parameter (e.g., magnetization, polarization)
    pub order_parameter: f64,
    /// Secondary order parameters
    pub secondary_orders: HashMap<String, f64>,
    /// Susceptibility (response to perturbations)
    pub susceptibility: f64,
    /// Correlation length
    pub correlation_length: f64,
    /// Entropy of collective state
    pub entropy: f64,
    /// Free energy estimate
    pub free_energy: f64,
    /// Active patterns in this state
    pub active_patterns: Vec<Uuid>,
    /// State phase classification
    pub phase: CollectivePhase,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Default for CollectiveState {
    fn default() -> Self {
        Self {
            order_parameter: 0.0,
            secondary_orders: HashMap::new(),
            susceptibility: 1.0,
            correlation_length: 1.0,
            entropy: 0.0,
            free_energy: 0.0,
            active_patterns: Vec::new(),
            phase: CollectivePhase::Disordered,
            timestamp: chrono::Utc::now(),
        }
    }
}

impl CollectiveState {
    /// Create new collective state with given order parameter
    pub fn new(order_parameter: f64) -> Self {
        let mut state = Self::default();
        state.order_parameter = order_parameter;
        state.phase = Self::classify_phase(order_parameter, 1.0);
        state
    }

    /// Classify phase based on order parameter and susceptibility
    fn classify_phase(order: f64, susceptibility: f64) -> CollectivePhase {
        if order.abs() < 0.1 {
            if susceptibility > 10.0 {
                CollectivePhase::Critical
            } else {
                CollectivePhase::Disordered
            }
        } else if order.abs() > 0.8 {
            CollectivePhase::Ordered
        } else {
            CollectivePhase::Intermediate
        }
    }

    /// Update phase classification
    pub fn update_phase(&mut self) {
        self.phase = Self::classify_phase(self.order_parameter, self.susceptibility);
    }

    /// Set secondary order parameter
    pub fn set_secondary_order(&mut self, name: &str, value: f64) {
        self.secondary_orders.insert(name.to_string(), value);
    }

    /// Add active pattern
    pub fn add_pattern(&mut self, pattern_id: Uuid) {
        if !self.active_patterns.contains(&pattern_id) {
            self.active_patterns.push(pattern_id);
        }
    }

    /// Remove pattern
    pub fn remove_pattern(&mut self, pattern_id: &Uuid) {
        self.active_patterns.retain(|id| id != pattern_id);
    }

    /// Compute distance from criticality
    ///
    /// Returns 0 at critical point, increases away from it
    pub fn distance_from_criticality(&self) -> f64 {
        // Critical point characterized by:
        // - Order parameter ≈ 0
        // - Susceptibility → ∞
        // - Correlation length → ∞

        let order_contribution = self.order_parameter.abs();
        let susceptibility_contribution = 1.0 / (self.susceptibility + 1.0);

        (order_contribution + susceptibility_contribution) / 2.0
    }

    /// Check if state is near critical point
    pub fn is_critical(&self) -> bool {
        self.phase == CollectivePhase::Critical || self.distance_from_criticality() < 0.1
    }
}

/// Phase classifications for collective states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollectivePhase {
    /// High symmetry, low order
    Disordered,
    /// Broken symmetry, high order
    Ordered,
    /// Near phase transition
    Critical,
    /// Between ordered and disordered
    Intermediate,
    /// Coexisting phases
    MixedPhase,
}

/// Analyzer for emergent patterns and collective states
///
/// Provides tools for detecting, tracking, and characterizing emergence.
#[derive(Debug)]
pub struct EmergenceAnalyzer {
    /// Active patterns
    patterns: HashMap<Uuid, EmergentPattern>,
    /// Event history
    events: Vec<EmergenceEvent>,
    /// Current collective state
    collective_state: CollectiveState,
    /// Detection thresholds
    coherence_threshold: f64,
    pattern_timeout_ms: u64,
}

impl Default for EmergenceAnalyzer {
    fn default() -> Self {
        Self::new(0.5, 10000)
    }
}

impl EmergenceAnalyzer {
    /// Create new emergence analyzer
    pub fn new(coherence_threshold: f64, pattern_timeout_ms: u64) -> Self {
        Self {
            patterns: HashMap::new(),
            events: Vec::new(),
            collective_state: CollectiveState::default(),
            coherence_threshold,
            pattern_timeout_ms,
        }
    }

    /// Detect patterns from covariance structure
    pub fn detect_from_covariance(&mut self, covariance: &DMatrix<f64>) -> Vec<EmergentPattern> {
        let mut detected = Vec::new();

        // Eigendecomposition for pattern detection
        let eigenvalues: Vec<f64> = covariance
            .clone()
            .symmetric_eigenvalues()
            .iter()
            .copied()
            .collect();

        let total: f64 = eigenvalues.iter().filter(|&&e| e > 0.0).sum();
        if total < 1e-10 {
            return detected;
        }

        // Normalized eigenvalues
        let normalized: Vec<f64> = eigenvalues.iter().map(|e| e.max(0.0) / total).collect();

        // Detect dominant modes as patterns
        for (i, &eigenvalue) in normalized.iter().enumerate() {
            if eigenvalue > self.coherence_threshold / normalized.len() as f64 {
                let pattern = EmergentPattern::new(
                    PatternCategory::Informational,
                    PatternStability::Metastable,
                )
                .with_coherence(eigenvalue)
                .with_scale(i as f64 + 1.0)
                .with_participation(1.0 / normalized.iter().map(|p| p * p).sum::<f64>());

                detected.push(pattern);
            }
        }

        // Register detected patterns
        for pattern in &detected {
            self.register_pattern(pattern.clone());
        }

        detected
    }

    /// Detect patterns from time series
    pub fn detect_from_timeseries(&mut self, data: &[f64], sample_rate: f64) -> Vec<EmergentPattern> {
        let mut detected = Vec::new();

        if data.len() < 4 {
            return detected;
        }

        // Compute autocorrelation for temporal patterns
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

        if variance < 1e-10 {
            return detected;
        }

        // Look for periodicity in autocorrelation
        let max_lag = data.len() / 2;
        let mut autocorr = Vec::with_capacity(max_lag);

        for lag in 0..max_lag {
            let mut corr = 0.0;
            for i in 0..(data.len() - lag) {
                corr += (data[i] - mean) * (data[i + lag] - mean);
            }
            corr /= (data.len() - lag) as f64 * variance;
            autocorr.push(corr);
        }

        // Find peaks in autocorrelation (indicates periodicity)
        for i in 1..(autocorr.len() - 1) {
            if autocorr[i] > autocorr[i - 1]
                && autocorr[i] > autocorr[i + 1]
                && autocorr[i] > self.coherence_threshold
            {
                let period = i as f64 / sample_rate;
                let pattern = EmergentPattern::new(
                    PatternCategory::Temporal,
                    PatternStability::Stable,
                )
                .with_coherence(autocorr[i])
                .with_scale(period)
                .with_metadata("period_samples", &i.to_string());

                detected.push(pattern);
            }
        }

        for pattern in &detected {
            self.register_pattern(pattern.clone());
        }

        detected
    }

    /// Register a new pattern
    pub fn register_pattern(&mut self, pattern: EmergentPattern) {
        let event = EmergenceEvent::new(
            EmergenceEventType::PatternFormation,
            &format!("New {:?} pattern detected", pattern.category),
        )
        .with_pattern(pattern.id);

        self.events.push(event);
        self.collective_state.add_pattern(pattern.id);
        self.patterns.insert(pattern.id, pattern);
    }

    /// Update collective state from component states
    pub fn update_collective_state(&mut self, component_states: &[f64]) {
        if component_states.is_empty() {
            return;
        }

        // Compute order parameter (mean absolute deviation from mean)
        let mean: f64 = component_states.iter().sum::<f64>() / component_states.len() as f64;
        let variance: f64 = component_states
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / component_states.len() as f64;

        // Order parameter: normalized standard deviation
        self.collective_state.order_parameter = variance.sqrt() / (mean.abs() + 1.0);

        // Susceptibility approximation from variance
        self.collective_state.susceptibility = variance * component_states.len() as f64;

        // Entropy from distribution
        self.collective_state.entropy = self.estimate_entropy(component_states);

        // Update phase classification
        self.collective_state.update_phase();
        self.collective_state.timestamp = chrono::Utc::now();
    }

    /// Estimate entropy from samples
    fn estimate_entropy(&self, samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }

        // Histogram-based entropy estimation
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < 1e-10 {
            return 0.0;
        }

        let n_bins = (samples.len() as f64).sqrt().ceil() as usize;
        let bin_width = (max - min) / n_bins as f64;

        let mut histogram = vec![0usize; n_bins];
        for &sample in samples {
            let bin = ((sample - min) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            histogram[bin] += 1;
        }

        // Shannon entropy
        let n = samples.len() as f64;
        histogram
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / n;
                -p * p.ln()
            })
            .sum()
    }

    /// Prune expired patterns
    pub fn prune_expired(&mut self) {
        let now = chrono::Utc::now();
        let timeout = chrono::Duration::milliseconds(self.pattern_timeout_ms as i64);

        let expired: Vec<Uuid> = self
            .patterns
            .iter()
            .filter(|(_, p)| {
                p.stability == PatternStability::Transient && (now - p.detected_at) > timeout
            })
            .map(|(id, _)| *id)
            .collect();

        for id in expired {
            if let Some(pattern) = self.patterns.remove(&id) {
                let event = EmergenceEvent::new(
                    EmergenceEventType::PatternDissolution,
                    &format!("{:?} pattern dissolved", pattern.category),
                )
                .with_pattern(id);

                self.events.push(event);
                self.collective_state.remove_pattern(&id);
            }
        }
    }

    /// Get active patterns
    pub fn active_patterns(&self) -> impl Iterator<Item = &EmergentPattern> {
        self.patterns.values()
    }

    /// Get pattern by ID
    pub fn get_pattern(&self, id: &Uuid) -> Option<&EmergentPattern> {
        self.patterns.get(id)
    }

    /// Get current collective state
    pub fn collective_state(&self) -> &CollectiveState {
        &self.collective_state
    }

    /// Get event history
    pub fn events(&self) -> &[EmergenceEvent] {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_emergent_pattern_creation() {
        let pattern = EmergentPattern::new(PatternCategory::Temporal, PatternStability::Stable)
            .with_coherence(0.85)
            .with_scale(10.0)
            .with_metadata("source", "oscillator");

        assert_eq!(pattern.category, PatternCategory::Temporal);
        assert_relative_eq!(pattern.coherence, 0.85, epsilon = 1e-10);
        assert_eq!(pattern.metadata.get("source"), Some(&"oscillator".to_string()));
    }

    #[test]
    fn test_collective_state_phase_classification() {
        let mut state = CollectiveState::new(0.9);
        assert_eq!(state.phase, CollectivePhase::Ordered);

        state.order_parameter = 0.05;
        state.susceptibility = 100.0;
        state.update_phase();
        assert_eq!(state.phase, CollectivePhase::Critical);

        state.order_parameter = 0.02;
        state.susceptibility = 1.0;
        state.update_phase();
        assert_eq!(state.phase, CollectivePhase::Disordered);
    }

    #[test]
    fn test_emergence_analyzer_covariance_detection() {
        let mut analyzer = EmergenceAnalyzer::new(0.1, 10000);

        // Covariance with clear dominant mode
        let mut cov = DMatrix::zeros(4, 4);
        cov[(0, 0)] = 10.0;
        cov[(1, 1)] = 1.0;
        cov[(2, 2)] = 1.0;
        cov[(3, 3)] = 1.0;

        let patterns = analyzer.detect_from_covariance(&cov);
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_emergence_analyzer_timeseries_detection() {
        let mut analyzer = EmergenceAnalyzer::new(0.3, 10000);

        // Periodic signal
        let data: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1 * std::f64::consts::PI).sin())
            .collect();

        let patterns = analyzer.detect_from_timeseries(&data, 1.0);
        // Should detect periodic pattern
        assert!(patterns.iter().any(|p| p.category == PatternCategory::Temporal));
    }

    #[test]
    fn test_collective_state_entropy() {
        let mut analyzer = EmergenceAnalyzer::default();

        // Uniform distribution has higher entropy
        let uniform: Vec<f64> = (0..100).map(|i| i as f64).collect();
        analyzer.update_collective_state(&uniform);
        let uniform_entropy = analyzer.collective_state().entropy;

        // Concentrated distribution has lower entropy
        let concentrated: Vec<f64> = vec![50.0; 100];
        analyzer.update_collective_state(&concentrated);
        let concentrated_entropy = analyzer.collective_state().entropy;

        assert!(uniform_entropy > concentrated_entropy);
    }
}
