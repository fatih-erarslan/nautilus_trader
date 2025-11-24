//! Temporal consciousness structure
//!
//! Implements Husserlian temporal phenomenology:
//! - Retention: Just-past experiences fading into memory
//! - Primal Impression: The living present moment
//! - Protention: Anticipation of immediate future
//!
//! # pbRTCA Integration
//!
//! The temporal layer provides "thickness" to conscious experience,
//! allowing the agent to integrate past, present, and future into
//! a coherent experiential flow.

use nalgebra as na;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Temporal consciousness implementing Husserlian time structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConsciousness {
    /// Retention buffer: fading past experiences
    retention: VecDeque<RetentionFrame>,
    /// Maximum retention depth
    retention_depth: usize,
    /// Protention: anticipated future states
    protention: Vec<ProtentionFrame>,
    /// Protention horizon (how far ahead to anticipate)
    protention_horizon: usize,
    /// Current primal impression (the "now")
    primal_impression: Option<PrimalImpression>,
    /// Temporal decay rate for retention (exponential)
    retention_decay: f64,
    /// Cumulative temporal thickness
    temporal_thickness: f64,
}

/// A retained past experience with decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionFrame {
    /// The retained belief state
    pub belief: na::DVector<f64>,
    /// Temporal distance from present (in steps)
    pub temporal_distance: usize,
    /// Retention strength (1.0 = vivid, 0.0 = forgotten)
    pub strength: f64,
    /// Associated free energy at that moment
    pub free_energy: f64,
}

/// The primal impression - the living present
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimalImpression {
    /// Current belief state
    pub belief: na::DVector<f64>,
    /// Current observation
    pub observation: Option<na::DVector<f64>>,
    /// Phenomenal intensity
    pub intensity: f64,
    /// Timestamp (step number)
    pub timestamp: u64,
}

/// An anticipated future state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtentionFrame {
    /// Predicted belief state
    pub predicted_belief: na::DVector<f64>,
    /// Temporal distance into future
    pub temporal_distance: usize,
    /// Confidence in prediction
    pub confidence: f64,
    /// Expected free energy
    pub expected_free_energy: f64,
}

impl TemporalConsciousness {
    /// Create new temporal consciousness structure
    ///
    /// # Arguments
    /// * `retention_depth` - How many past frames to retain
    pub fn new(retention_depth: usize) -> Self {
        Self {
            retention: VecDeque::with_capacity(retention_depth),
            retention_depth,
            protention: Vec::with_capacity(4),
            protention_horizon: 4,
            primal_impression: None,
            retention_decay: 0.8, // 20% decay per step
            temporal_thickness: 0.0,
        }
    }

    /// Create with custom parameters
    pub fn with_params(
        retention_depth: usize,
        protention_horizon: usize,
        retention_decay: f64,
    ) -> Self {
        Self {
            retention: VecDeque::with_capacity(retention_depth),
            retention_depth,
            protention: Vec::with_capacity(protention_horizon),
            protention_horizon,
            primal_impression: None,
            retention_decay: retention_decay.clamp(0.0, 1.0),
            temporal_thickness: 0.0,
        }
    }

    /// Update temporal consciousness with new belief state
    ///
    /// This performs the fundamental temporal flow:
    /// 1. Current primal impression → retention
    /// 2. New belief → primal impression
    /// 3. Decay all retentions
    /// 4. Update protentions
    pub fn update(&mut self, new_belief: &na::DVector<f64>) {
        // 1. Move current primal impression to retention
        if let Some(current) = self.primal_impression.take() {
            let retention_frame = RetentionFrame {
                belief: current.belief,
                temporal_distance: 1,
                strength: 1.0,
                free_energy: 0.0, // Will be updated externally
            };
            self.retention.push_front(retention_frame);

            // Trim if exceeding depth
            while self.retention.len() > self.retention_depth {
                self.retention.pop_back();
            }
        }

        // 2. Update all retention frames (increase distance, decay strength)
        for frame in self.retention.iter_mut() {
            frame.temporal_distance += 1;
            frame.strength *= self.retention_decay;
        }

        // Remove faded retentions (strength < threshold)
        self.retention.retain(|f| f.strength > 0.01);

        // 3. Create new primal impression
        let timestamp = self.primal_impression
            .as_ref()
            .map(|p| p.timestamp + 1)
            .unwrap_or(0);

        self.primal_impression = Some(PrimalImpression {
            belief: new_belief.clone(),
            observation: None,
            intensity: 1.0,
            timestamp,
        });

        // 4. Update temporal thickness
        self.temporal_thickness = self.compute_temporal_thickness();
    }

    /// Update with full context (belief + observation + free energy)
    pub fn update_full(
        &mut self,
        new_belief: &na::DVector<f64>,
        observation: &na::DVector<f64>,
        free_energy: f64,
    ) {
        self.update(new_belief);

        // Update primal impression with observation
        if let Some(ref mut primal) = self.primal_impression {
            primal.observation = Some(observation.clone());
            primal.intensity = (-free_energy).exp().min(1.0);
        }

        // Update most recent retention's free energy
        if let Some(frame) = self.retention.front_mut() {
            frame.free_energy = free_energy;
        }
    }

    /// Generate protentions (future anticipations)
    ///
    /// Uses transition dynamics to predict future states
    pub fn generate_protentions(
        &mut self,
        transition: &na::DMatrix<f64>,
        expected_free_energies: &[f64],
    ) {
        self.protention.clear();

        let Some(ref primal) = self.primal_impression else {
            return;
        };

        let mut predicted = primal.belief.clone();

        for i in 0..self.protention_horizon.min(expected_free_energies.len()) {
            // Propagate belief through transition
            predicted = transition * &predicted;

            // Normalize
            let sum = predicted.sum();
            if sum > 1e-10 {
                predicted /= sum;
            }

            // Compute confidence (decreases with distance)
            let confidence = (0.9_f64).powi(i as i32 + 1);

            self.protention.push(ProtentionFrame {
                predicted_belief: predicted.clone(),
                temporal_distance: i + 1,
                confidence,
                expected_free_energy: expected_free_energies[i],
            });
        }
    }

    /// Get temporal thickness (phenomenological depth)
    ///
    /// Measures the "richness" of temporal experience:
    /// - Sum of retention strengths weighted by information content
    /// - Plus protention confidence
    pub fn get_temporal_thickness(&self) -> f64 {
        self.temporal_thickness
    }

    /// Compute temporal thickness
    fn compute_temporal_thickness(&self) -> f64 {
        // Retention contribution
        let retention_contrib: f64 = self.retention.iter()
            .map(|f| f.strength * self.belief_entropy(&f.belief))
            .sum();

        // Protention contribution
        let protention_contrib: f64 = self.protention.iter()
            .map(|f| f.confidence * self.belief_entropy(&f.predicted_belief))
            .sum();

        // Primal impression contribution
        let primal_contrib = self.primal_impression.as_ref()
            .map(|p| p.intensity * self.belief_entropy(&p.belief))
            .unwrap_or(0.0);

        retention_contrib + primal_contrib + protention_contrib
    }

    /// Compute entropy of a belief distribution
    fn belief_entropy(&self, belief: &na::DVector<f64>) -> f64 {
        -belief.iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Get retention summary
    pub fn retention_summary(&self) -> Vec<(usize, f64)> {
        self.retention.iter()
            .map(|f| (f.temporal_distance, f.strength))
            .collect()
    }

    /// Get protention summary
    pub fn protention_summary(&self) -> Vec<(usize, f64)> {
        self.protention.iter()
            .map(|f| (f.temporal_distance, f.confidence))
            .collect()
    }

    /// Get current primal impression
    pub fn get_primal_impression(&self) -> Option<&PrimalImpression> {
        self.primal_impression.as_ref()
    }

    /// Compute temporal integration measure
    ///
    /// How well past and future are integrated in the present
    pub fn temporal_integration(&self) -> f64 {
        let Some(ref primal) = self.primal_impression else {
            return 0.0;
        };

        // Measure similarity between past retentions and current state
        let past_integration: f64 = self.retention.iter()
            .map(|f| f.strength * primal.belief.dot(&f.belief))
            .sum::<f64>()
            / self.retention.len().max(1) as f64;

        // Measure similarity between protentions and current state
        let future_integration: f64 = self.protention.iter()
            .map(|f| f.confidence * primal.belief.dot(&f.predicted_belief))
            .sum::<f64>()
            / self.protention.len().max(1) as f64;

        (past_integration + future_integration) / 2.0
    }

    /// Get the "specious present" - the phenomenologically unified now
    ///
    /// Integrates recent retentions with primal impression
    pub fn specious_present(&self) -> Option<na::DVector<f64>> {
        let primal = self.primal_impression.as_ref()?;

        let mut integrated = primal.belief.clone() * primal.intensity;
        let mut total_weight = primal.intensity;

        // Add recent retentions (last 3 steps typically)
        for frame in self.retention.iter().take(3) {
            integrated += &frame.belief * frame.strength;
            total_weight += frame.strength;
        }

        if total_weight > 1e-10 {
            Some(integrated / total_weight)
        } else {
            Some(primal.belief.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_consciousness_creation() {
        let tc = TemporalConsciousness::new(16);
        assert_eq!(tc.retention_depth, 16);
        assert!(tc.primal_impression.is_none());
    }

    #[test]
    fn test_temporal_update() {
        let mut tc = TemporalConsciousness::new(8);

        let belief1 = na::DVector::from_vec(vec![0.5, 0.5]);
        let belief2 = na::DVector::from_vec(vec![0.7, 0.3]);

        tc.update(&belief1);
        assert!(tc.primal_impression.is_some());
        assert!(tc.retention.is_empty());

        tc.update(&belief2);
        assert_eq!(tc.retention.len(), 1);
        assert_eq!(tc.retention[0].temporal_distance, 2);
    }

    #[test]
    fn test_retention_decay() {
        let mut tc = TemporalConsciousness::with_params(8, 4, 0.5);

        let belief = na::DVector::from_vec(vec![0.5, 0.5]);

        // Multiple updates
        for _ in 0..5 {
            tc.update(&belief);
        }

        // Check decay
        for (i, frame) in tc.retention.iter().enumerate() {
            let expected_strength = (0.5_f64).powi(i as i32 + 1);
            assert!((frame.strength - expected_strength).abs() < 0.01);
        }
    }

    #[test]
    fn test_temporal_thickness() {
        let mut tc = TemporalConsciousness::new(8);

        let belief = na::DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]);

        // Build up temporal structure
        for _ in 0..5 {
            tc.update(&belief);
        }

        let thickness = tc.get_temporal_thickness();
        assert!(thickness > 0.0);
        assert!(thickness.is_finite());
    }

    #[test]
    fn test_specious_present() {
        let mut tc = TemporalConsciousness::new(8);

        let belief = na::DVector::from_vec(vec![0.5, 0.5]);
        tc.update(&belief);
        tc.update(&belief);

        let sp = tc.specious_present();
        assert!(sp.is_some());

        let sp = sp.unwrap();
        assert!((sp.sum() - 1.0).abs() < 0.01);
    }
}
