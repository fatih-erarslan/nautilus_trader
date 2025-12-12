//! Shared types for cognition system

use serde::{Deserialize, Serialize};

/// Lorentz point in H^11 (12D: 1 time + 11 spatial dimensions)
pub type LorentzPoint = [f64; 12];

/// Timestamp in milliseconds
pub type Timestamp = u64;

/// Neuron/node identifier
pub type NodeId = u64;

/// Cognition phase in the self-referential loop
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitionPhase {
    /// Perceiving sensory input
    Perceiving,
    /// Processing cognitive representations
    Cognizing,
    /// Deliberating in neocortex
    Deliberating,
    /// Forming intentions (agency)
    Intending,
    /// Integrating consciousness
    Integrating,
    /// Executing actions
    Acting,
}

impl CognitionPhase {
    /// Get the next phase in the loop
    pub fn next(self) -> Self {
        match self {
            Self::Perceiving => Self::Cognizing,
            Self::Cognizing => Self::Deliberating,
            Self::Deliberating => Self::Intending,
            Self::Intending => Self::Integrating,
            Self::Integrating => Self::Acting,
            Self::Acting => Self::Perceiving,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Perceiving => "Perception",
            Self::Cognizing => "Cognition",
            Self::Deliberating => "Neocortex",
            Self::Intending => "Agency",
            Self::Integrating => "Consciousness",
            Self::Acting => "Action",
        }
    }
}

/// Arousal level (0.0 = deep sleep, 1.0 = maximal arousal)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ArousalLevel(pub f64);

impl ArousalLevel {
    /// Create new arousal level (clamped to [0, 1])
    pub fn new(level: f64) -> Self {
        Self(level.clamp(0.0, 1.0))
    }

    /// Get raw value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if in dream state (arousal < 0.3)
    pub fn is_dream_state(&self) -> bool {
        self.0 < 0.3
    }

    /// Check if in waking state (arousal > 0.5)
    pub fn is_waking_state(&self) -> bool {
        self.0 > 0.5
    }

    /// Check if in transition state (0.3 <= arousal <= 0.5)
    pub fn is_transition_state(&self) -> bool {
        (0.3..=0.5).contains(&self.0)
    }
}

impl Default for ArousalLevel {
    fn default() -> Self {
        Self::new(0.7) // Default to waking state
    }
}

/// Cognitive load (0.0 = minimal, 1.0 = maximal)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CognitiveLoad(pub f64);

impl CognitiveLoad {
    /// Create new cognitive load (clamped to [0, 1])
    pub fn new(load: f64) -> Self {
        Self(load.clamp(0.0, 1.0))
    }

    /// Get raw value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if overloaded (load > 0.8)
    pub fn is_overloaded(&self) -> bool {
        self.0 > 0.8
    }

    /// Check if underutilized (load < 0.2)
    pub fn is_underutilized(&self) -> bool {
        self.0 < 0.2
    }
}

impl Default for CognitiveLoad {
    fn default() -> Self {
        Self::new(0.5) // Default to moderate load
    }
}

/// Attention bandwidth (inversely proportional to curvature)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AttentionBandwidth {
    /// Bandwidth value (arbitrary units)
    pub value: f64,
    /// Associated curvature
    pub curvature: f64,
}

impl AttentionBandwidth {
    /// Compute bandwidth from curvature: BW = k / κ
    pub fn from_curvature(curvature: f64, constant: f64) -> Self {
        Self {
            value: constant / curvature,
            curvature,
        }
    }

    /// Check if narrow focus (high curvature, low bandwidth)
    pub fn is_narrow_focus(&self) -> bool {
        self.curvature > 3.0
    }

    /// Check if broad awareness (low curvature, high bandwidth)
    pub fn is_broad_awareness(&self) -> bool {
        self.curvature < 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognition_phase_cycle() {
        let mut phase = CognitionPhase::Perceiving;
        phase = phase.next();
        assert_eq!(phase, CognitionPhase::Cognizing);
        phase = phase.next();
        assert_eq!(phase, CognitionPhase::Deliberating);
        phase = phase.next();
        assert_eq!(phase, CognitionPhase::Intending);
        phase = phase.next();
        assert_eq!(phase, CognitionPhase::Integrating);
        phase = phase.next();
        assert_eq!(phase, CognitionPhase::Acting);
        phase = phase.next();
        assert_eq!(phase, CognitionPhase::Perceiving); // Loops back
    }

    #[test]
    fn test_arousal_level() {
        let sleep = ArousalLevel::new(0.1);
        assert!(sleep.is_dream_state());
        assert!(!sleep.is_waking_state());

        let awake = ArousalLevel::new(0.8);
        assert!(!awake.is_dream_state());
        assert!(awake.is_waking_state());

        let transition = ArousalLevel::new(0.4);
        assert!(transition.is_transition_state());
    }

    #[test]
    fn test_cognitive_load() {
        let overload = CognitiveLoad::new(0.9);
        assert!(overload.is_overloaded());

        let underutilized = CognitiveLoad::new(0.1);
        assert!(underutilized.is_underutilized());
    }

    #[test]
    fn test_attention_bandwidth() {
        let narrow = AttentionBandwidth::from_curvature(5.0, 10.0);
        assert!(narrow.is_narrow_focus());
        assert!(!narrow.is_broad_awareness());

        let broad = AttentionBandwidth::from_curvature(0.3, 10.0);
        assert!(!broad.is_narrow_focus());
        assert!(broad.is_broad_awareness());
    }
}
