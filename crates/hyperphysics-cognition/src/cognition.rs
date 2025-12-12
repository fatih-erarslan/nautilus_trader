//! Unified cognition system integrating all components

use crate::error::{CognitionError, Result};
use crate::types::*;
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{debug, info, warn};

#[cfg(feature = "attention")]
use crate::attention::HyperbolicAttention;

#[cfg(feature = "loops")]
use crate::loop_coordinator::SelfReferentialLoop;

#[cfg(feature = "dream")]
use crate::dream::DreamConsolidator;

#[cfg(feature = "learning")]
use crate::learning::BatesonLearner;

#[cfg(feature = "integration")]
use crate::integration::CorticalBusIntegration;

/// Cognition system configuration
#[derive(Debug, Clone)]
pub struct CognitionConfig {
    /// Enable hyperbolic attention
    pub enable_attention: bool,

    /// Enable self-referential loops
    pub enable_loops: bool,

    /// Enable dream state consolidation
    pub enable_dream: bool,

    /// Enable Bateson learning
    pub enable_learning: bool,

    /// Enable cortical bus integration
    pub enable_integration: bool,

    /// Default attention curvature
    pub default_curvature: f64,

    /// Loop frequency (Hz)
    pub loop_frequency: f64,

    /// Dream consolidation threshold (arousal level)
    pub dream_threshold: f64,
}

impl Default for CognitionConfig {
    fn default() -> Self {
        Self {
            enable_attention: true,
            enable_loops: true,
            enable_dream: true,
            enable_learning: true,
            enable_integration: true,
            default_curvature: crate::DEFAULT_CURVATURE,
            loop_frequency: crate::GAMMA_FREQUENCY_HZ,
            dream_threshold: 0.3,
        }
    }
}

/// Unified cognition system
///
/// Integrates:
/// - Hyperbolic attention (curvature-modulated focus)
/// - Self-referential loops (6-stage cycle at 40Hz)
/// - Dream state consolidation (offline learning)
/// - Bateson's learning levels (meta-learning)
/// - Cortical bus integration (message routing)
pub struct CognitionSystem {
    /// Configuration
    config: CognitionConfig,

    /// Current arousal level
    arousal: Arc<RwLock<ArousalLevel>>,

    /// Current cognitive load
    load: Arc<RwLock<CognitiveLoad>>,

    /// Hyperbolic attention mechanism
    #[cfg(feature = "attention")]
    attention: Option<HyperbolicAttention>,

    /// Self-referential loop coordinator
    #[cfg(feature = "loops")]
    loop_coordinator: Option<SelfReferentialLoop>,

    /// Dream state consolidator
    #[cfg(feature = "dream")]
    dream: Option<DreamConsolidator>,

    /// Bateson learner
    #[cfg(feature = "learning")]
    learner: Option<BatesonLearner>,

    /// Cortical bus integration
    #[cfg(feature = "integration")]
    integration: Option<CorticalBusIntegration>,
}

impl CognitionSystem {
    /// Create new cognition system
    pub fn new(config: CognitionConfig) -> Result<Self> {
        info!("🧠 Initializing HyperPhysics Cognition System");

        #[cfg(feature = "attention")]
        let attention = if config.enable_attention {
            debug!("  ✓ Hyperbolic attention enabled");
            Some(HyperbolicAttention::new(config.default_curvature)?)
        } else {
            None
        };

        #[cfg(feature = "loops")]
        let loop_coordinator = if config.enable_loops {
            debug!("  ✓ Self-referential loops enabled ({}Hz)", config.loop_frequency);
            Some(SelfReferentialLoop::new(config.loop_frequency)?)
        } else {
            None
        };

        #[cfg(feature = "dream")]
        let dream = if config.enable_dream {
            debug!("  ✓ Dream consolidation enabled (threshold: {:.2})", config.dream_threshold);
            Some(DreamConsolidator::new(config.dream_threshold)?)
        } else {
            None
        };

        #[cfg(feature = "learning")]
        let learner = if config.enable_learning {
            debug!("  ✓ Bateson learning enabled");
            Some(BatesonLearner::new()?)
        } else {
            None
        };

        #[cfg(feature = "integration")]
        let integration = if config.enable_integration {
            debug!("  ✓ Cortical bus integration enabled");
            Some(CorticalBusIntegration::new()?)
        } else {
            None
        };

        info!("✅ Cognition system initialized");

        Ok(Self {
            config,
            arousal: Arc::new(RwLock::new(ArousalLevel::default())),
            load: Arc::new(RwLock::new(CognitiveLoad::default())),
            #[cfg(feature = "attention")]
            attention,
            #[cfg(feature = "loops")]
            loop_coordinator,
            #[cfg(feature = "dream")]
            dream,
            #[cfg(feature = "learning")]
            learner,
            #[cfg(feature = "integration")]
            integration,
        })
    }

    /// Get current arousal level
    pub fn arousal(&self) -> ArousalLevel {
        *self.arousal.read()
    }

    /// Set arousal level
    pub fn set_arousal(&self, level: ArousalLevel) {
        let mut arousal = self.arousal.write();
        *arousal = level;
        debug!("Arousal level set to {:.2}", level.value());

        // Check if we should enter/exit dream state
        #[cfg(feature = "dream")]
        if let Some(ref dream) = self.dream {
            if level.is_dream_state() && !dream.is_active() {
                debug!("Entering dream state (arousal: {:.2})", level.value());
            } else if level.is_waking_state() && dream.is_active() {
                debug!("Exiting dream state (arousal: {:.2})", level.value());
            }
        }
    }

    /// Get current cognitive load
    pub fn cognitive_load(&self) -> CognitiveLoad {
        *self.load.read()
    }

    /// Set cognitive load
    pub fn set_cognitive_load(&self, load: CognitiveLoad) {
        let mut current = self.load.write();
        *current = load;

        if load.is_overloaded() {
            warn!("Cognitive load overloaded: {:.2}", load.value());
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CognitionConfig {
        &self.config
    }

    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        let arousal_ok = {
            let arousal = self.arousal.read();
            arousal.value() >= 0.0 && arousal.value() <= 1.0
        };

        let load_ok = {
            let load = self.load.read();
            !load.is_overloaded()
        };

        arousal_ok && load_ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognition_system_creation() {
        let config = CognitionConfig::default();
        let system = CognitionSystem::new(config).unwrap();
        assert!(system.is_healthy());
    }

    #[test]
    fn test_arousal_control() {
        let config = CognitionConfig::default();
        let system = CognitionSystem::new(config).unwrap();

        system.set_arousal(ArousalLevel::new(0.9));
        assert_eq!(system.arousal().value(), 0.9);

        system.set_arousal(ArousalLevel::new(0.2));
        assert_eq!(system.arousal().value(), 0.2);
    }

    #[test]
    fn test_cognitive_load() {
        let config = CognitionConfig::default();
        let system = CognitionSystem::new(config).unwrap();

        system.set_cognitive_load(CognitiveLoad::new(0.5));
        assert_eq!(system.cognitive_load().value(), 0.5);

        system.set_cognitive_load(CognitiveLoad::new(0.95));
        assert!(system.cognitive_load().is_overloaded());
    }
}
