//! # HyperPhysics Cognition
//!
//! Bio-Digital Isomorphic Cognition System implementing:
//!
//! - **Hyperbolic Attention**: Curvature-modulated attention in H^11 Lorentz space
//! - **Self-Referential Loops**: Perception → Cognition → Neocortex → Agency → Consciousness → Action
//! - **Dream State Consolidation**: Offline learning during low arousal states
//! - **Bateson's Learning Levels**: Hierarchical learning (0-III)
//! - **Cortical Bus Integration**: Ultra-low-latency message routing
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    HYPERPHYSICS COGNITION SYSTEM                        │
//! │                                                                         │
//! │  ┌────────────────────┐      ┌────────────────────┐                    │
//! │  │ Hyperbolic         │      │ Self-Referential   │                    │
//! │  │ Attention          │◄────►│ Loop Coordinator   │                    │
//! │  │ (H^11 curvature)   │      │ (40Hz gamma)       │                    │
//! │  └──────┬─────────────┘      └──────┬─────────────┘                    │
//! │         │                           │                                  │
//! │         ▼                           ▼                                  │
//! │  ┌────────────────────┐      ┌────────────────────┐                    │
//! │  │ Dream State        │      │ Bateson Learning   │                    │
//! │  │ Consolidation      │◄────►│ Levels (0-III)     │                    │
//! │  │ (Offline learning) │      │ (Meta-learning)    │                    │
//! │  └──────┬─────────────┘      └──────┬─────────────┘                    │
//! │         │                           │                                  │
//! │         └───────────┬───────────────┘                                  │
//! │                     ▼                                                  │
//! │              ┌─────────────┐                                           │
//! │              │ Cortical    │                                           │
//! │              │ Bus         │                                           │
//! │              │ Integration │                                           │
//! │              └─────────────┘                                           │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use hyperphysics_cognition::prelude::*;
//!
//! // Create cognition system
//! let config = CognitionConfig::default();
//! let mut cognition = CognitionSystem::new(config)?;
//!
//! // Modulate attention via hyperbolic curvature
//! cognition.set_attention_curvature(5.0)?; // Narrow focus
//!
//! // Process perception through self-referential loop
//! let perception = PerceptionInput::new(sensory_data);
//! let action = cognition.process_loop(perception).await?;
//!
//! // Dream state consolidation (offline learning)
//! cognition.enter_dream_state()?;
//! cognition.consolidate_memories(replay_buffer).await?;
//! cognition.exit_dream_state()?;
//!
//! // Meta-learning with Bateson Level II
//! cognition.update_learning_strategy(context)?;
//! ```
//!
//! ## Scientific Grounding
//!
//! Based on peer-reviewed research:
//!
//! - **Hyperbolic Geometry**: Chami et al. (2020) "Hyperbolic Graph Neural Networks", NeurIPS
//! - **Attention Mechanism**: Vaswani et al. (2017) "Attention Is All You Need"
//! - **Self-Organized Criticality**: Bak et al. (1987) "Self-organized criticality"
//! - **Active Inference**: Friston (2010) "The free-energy principle: a unified brain theory?"
//! - **Bateson's Learning**: Bateson (1972) "Steps to an Ecology of Mind"
//! - **Dream Consolidation**: Wilson & McNaughton (1994) "Reactivation of hippocampal ensemble memories"
//! - **Integrated Information Theory**: Tononi (2004) "An information integration theory of consciousness"
//!
//! ## Performance
//!
//! - **Loop Frequency**: 40Hz gamma rhythm (25ms period)
//! - **Attention Modulation**: <1ms hyperbolic curvature update
//! - **Message Routing**: <50ns via cortical bus
//! - **Dream Consolidation**: Configurable replay rate
//!
//! ## Features
//!
//! - `full` (default): All features enabled
//! - `attention`: Hyperbolic attention mechanism
//! - `loops`: Self-referential loop coordinator
//! - `dream`: Dream state consolidation
//! - `learning`: Bateson's learning levels
//! - `integration`: Cortical bus integration
//!

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

// ============================================================================
// Core Modules
// ============================================================================

/// Hyperbolic attention mechanism (H^11 Lorentz model)
#[cfg(feature = "attention")]
#[cfg_attr(docsrs, doc(cfg(feature = "attention")))]
pub mod attention;

/// Self-referential loop coordinator (40Hz gamma rhythm)
#[cfg(feature = "loops")]
#[cfg_attr(docsrs, doc(cfg(feature = "loops")))]
pub mod loop_coordinator;

/// Dream state consolidation (offline learning)
#[cfg(feature = "dream")]
#[cfg_attr(docsrs, doc(cfg(feature = "dream")))]
pub mod dream;

/// Bateson's Learning Levels (0-III hierarchical learning)
#[cfg(feature = "learning")]
#[cfg_attr(docsrs, doc(cfg(feature = "learning")))]
pub mod learning;

/// Cortical bus integration (ultra-low-latency messaging)
#[cfg(feature = "integration")]
#[cfg_attr(docsrs, doc(cfg(feature = "integration")))]
pub mod integration;

/// Cognitive action matrix (biomimetic algorithms)
pub mod actions;

// ============================================================================
// Shared Types
// ============================================================================

pub mod types;
pub mod error;

// ============================================================================
// Re-exports
// ============================================================================

pub use error::{CognitionError, Result};
pub use types::*;

#[cfg(feature = "attention")]
pub use attention::{
    AttentionState, AttentionConfig, HyperbolicAttention,
    CurvatureModulator, LocusCoeruleusGain,
};

#[cfg(feature = "loops")]
pub use loop_coordinator::{
    SelfReferentialLoop, LoopState, LoopConfig, LoopMessage,
    PerceptionInput, CognitionOutput, NeocortexState, AgencyIntent, ConsciousnessIntegration,
};

#[cfg(feature = "dream")]
pub use dream::{
    DreamState, DreamConfig, DreamConsolidator,
    ReplayBuffer, EpisodicMemory, ConsolidationMetrics,
};

#[cfg(feature = "learning")]
pub use learning::{
    LearningLevel, BatesonLearner, LearningContext,
    ProtoLearning, Learning, DeuteroLearning, LearningIII,
};

#[cfg(feature = "integration")]
pub use integration::{
    CorticalBusIntegration, MessageRouter, RouteConfig,
};

// ============================================================================
// Unified Cognition System
// ============================================================================

mod cognition;
pub use cognition::{CognitionSystem, CognitionConfig};

// ============================================================================
// Prelude Module
// ============================================================================

/// Prelude module - import everything you need
pub mod prelude {
    pub use crate::error::{CognitionError, Result};
    pub use crate::types::*;
    pub use crate::cognition::{CognitionSystem, CognitionConfig};

    #[cfg(feature = "attention")]
    pub use crate::attention::{
        AttentionState, HyperbolicAttention, CurvatureModulator,
    };

    #[cfg(feature = "loops")]
    pub use crate::loop_coordinator::{
        SelfReferentialLoop, LoopState, PerceptionInput, LoopMessage,
    };

    #[cfg(feature = "dream")]
    pub use crate::dream::{
        DreamState, DreamConsolidator, ReplayBuffer,
    };

    #[cfg(feature = "learning")]
    pub use crate::learning::{
        LearningLevel, BatesonLearner, LearningContext,
    };

    #[cfg(feature = "integration")]
    pub use crate::integration::{
        CorticalBusIntegration, MessageRouter,
    };
}

// ============================================================================
// Version Info
// ============================================================================

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get version information
pub fn version() -> &'static str {
    VERSION
}

/// Get feature summary
pub fn features() -> Vec<&'static str> {
    let mut features = vec!["core"];

    #[cfg(feature = "attention")]
    features.push("attention");

    #[cfg(feature = "loops")]
    features.push("loops");

    #[cfg(feature = "dream")]
    features.push("dream");

    #[cfg(feature = "learning")]
    features.push("learning");

    #[cfg(feature = "integration")]
    features.push("integration");

    #[cfg(feature = "cortical-bus")]
    features.push("cortical-bus");

    features
}

// ============================================================================
// Constants
// ============================================================================

/// Gamma rhythm frequency (40 Hz)
pub const GAMMA_FREQUENCY_HZ: f64 = 40.0;

/// Loop period (25ms for 40Hz)
pub const LOOP_PERIOD_MS: f64 = 1000.0 / GAMMA_FREQUENCY_HZ;

/// Hyperbolic dimension (H^11)
pub const HYPERBOLIC_DIM: usize = 11;

/// Lorentz dimension (12D: 1 time + 11 spatial)
pub const LORENTZ_DIM: usize = 12;

/// Ising critical temperature (Onsager solution)
pub const ISING_CRITICAL_TEMP: f64 = 2.269185314;

/// Default attention curvature (moderate focus)
pub const DEFAULT_CURVATURE: f64 = 1.0;

/// Curvature range [min, max]
pub const CURVATURE_RANGE: (f64, f64) = (0.1, 10.0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }

    #[test]
    fn test_features() {
        let f = features();
        assert!(f.contains(&"core"));
    }

    #[test]
    fn test_constants() {
        assert_eq!(GAMMA_FREQUENCY_HZ, 40.0);
        assert_eq!(LOOP_PERIOD_MS, 25.0);
        assert_eq!(HYPERBOLIC_DIM, 11);
        assert_eq!(LORENTZ_DIM, 12);
    }
}
