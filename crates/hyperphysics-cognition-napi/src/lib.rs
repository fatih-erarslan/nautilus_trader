//! Node.js/Bun.js NAPI bindings for HyperPhysics Cognition System
//!
//! This crate provides zero-copy NAPI bindings for JavaScript/TypeScript access
//! to the Bio-Digital Isomorphic Cognition System.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │     BUN.JS / NODE.JS APPLICATION                    │
//! │  (Web Dashboard, API Server, CLI)                   │
//! └─────────────────┬───────────────────────────────────┘
//!                   │
//!                   ▼
//! ┌─────────────────────────────────────────────────────┐
//! │      TYPESCRIPT/JAVASCRIPT BINDINGS                 │
//! │  CognitionSystem class (ergonomic API)              │
//! └─────────────────┬───────────────────────────────────┘
//!                   │
//!                   ▼
//! ┌─────────────────────────────────────────────────────┐
//! │         NAPI LAYER (THIS CRATE)                     │
//! │  Zero-copy bindings via napi-rs                     │
//! └─────────────────┬───────────────────────────────────┘
//!                   │
//!                   ▼
//! ┌─────────────────────────────────────────────────────┐
//! │    RUST COGNITION SYSTEM                            │
//! │  hyperphysics-cognition crate                       │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage (Bun.js)
//!
//! ```typescript
//! import { CognitionSystem } from 'hyperphysics-cognition'
//!
//! // Create cognition system
//! const cognition = new CognitionSystem({
//!   enableAttention: true,
//!   enableLoops: true,
//!   enableDream: true,
//!   enableLearning: true,
//!   enableIntegration: true,
//!   defaultCurvature: 1.0,
//!   loopFrequency: 40.0,  // 40Hz gamma
//!   dreamThreshold: 0.3
//! })
//!
//! // Set arousal level
//! cognition.setArousal(0.8)
//!
//! // Check health
//! console.log(`Healthy: ${cognition.isHealthy()}`)
//!
//! // Get current phase
//! console.log(`Phase: ${cognition.currentPhase()}`)
//! ```

use hyperphysics_cognition::prelude::*;
use napi::{Error, Status};
use napi_derive::napi;
use std::sync::Arc;
use tracing::{error, info};

/// Initialize tracing (call once at startup)
#[napi]
pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter("hyperphysics_cognition=debug")
        .init();
    info!("🧠 HyperPhysics Cognition NAPI initialized");
}

// ============================================================================
// Configuration
// ============================================================================

/// Cognition system configuration (JavaScript object)
#[napi(object)]
#[derive(Debug, Clone)]
pub struct JsCognitionConfig {
    pub enable_attention: bool,
    pub enable_loops: bool,
    pub enable_dream: bool,
    pub enable_learning: bool,
    pub enable_integration: bool,
    pub default_curvature: f64,
    pub loop_frequency: f64,
    pub dream_threshold: f64,
}

impl From<JsCognitionConfig> for CognitionConfig {
    fn from(js: JsCognitionConfig) -> Self {
        Self {
            enable_attention: js.enable_attention,
            enable_loops: js.enable_loops,
            enable_dream: js.enable_dream,
            enable_learning: js.enable_learning,
            enable_integration: js.enable_integration,
            default_curvature: js.default_curvature,
            loop_frequency: js.loop_frequency,
            dream_threshold: js.dream_threshold,
        }
    }
}

impl From<CognitionConfig> for JsCognitionConfig {
    fn from(rust: CognitionConfig) -> Self {
        Self {
            enable_attention: rust.enable_attention,
            enable_loops: rust.enable_loops,
            enable_dream: rust.enable_dream,
            enable_learning: rust.enable_learning,
            enable_integration: rust.enable_integration,
            default_curvature: rust.default_curvature,
            loop_frequency: rust.loop_frequency,
            dream_threshold: rust.dream_threshold,
        }
    }
}

/// Get default cognition configuration
#[napi]
pub fn default_config() -> JsCognitionConfig {
    CognitionConfig::default().into()
}

// ============================================================================
// Cognition Phase
// ============================================================================

/// Cognition phase enum
#[napi]
pub enum JsCognitionPhase {
    Perceiving,
    Cognizing,
    Deliberating,
    Intending,
    Integrating,
    Acting,
}

impl From<CognitionPhase> for JsCognitionPhase {
    fn from(phase: CognitionPhase) -> Self {
        match phase {
            CognitionPhase::Perceiving => JsCognitionPhase::Perceiving,
            CognitionPhase::Cognizing => JsCognitionPhase::Cognizing,
            CognitionPhase::Deliberating => JsCognitionPhase::Deliberating,
            CognitionPhase::Intending => JsCognitionPhase::Intending,
            CognitionPhase::Integrating => JsCognitionPhase::Integrating,
            CognitionPhase::Acting => JsCognitionPhase::Acting,
        }
    }
}

impl From<JsCognitionPhase> for CognitionPhase {
    fn from(phase: JsCognitionPhase) -> Self {
        match phase {
            JsCognitionPhase::Perceiving => CognitionPhase::Perceiving,
            JsCognitionPhase::Cognizing => CognitionPhase::Cognizing,
            JsCognitionPhase::Deliberating => CognitionPhase::Deliberating,
            JsCognitionPhase::Intending => CognitionPhase::Intending,
            JsCognitionPhase::Integrating => CognitionPhase::Integrating,
            JsCognitionPhase::Acting => CognitionPhase::Acting,
        }
    }
}

/// Get next phase in loop
#[napi]
pub fn next_phase(phase: JsCognitionPhase) -> JsCognitionPhase {
    let rust_phase: CognitionPhase = phase.into();
    rust_phase.next().into()
}

/// Get phase name
#[napi]
pub fn phase_name(phase: JsCognitionPhase) -> String {
    let rust_phase: CognitionPhase = phase.into();
    rust_phase.name().to_string()
}

// ============================================================================
// Cognition System
// ============================================================================

/// Cognition system class (JavaScript)
#[napi]
pub struct CognitionSystem {
    /// Internal Rust cognition system (Arc for thread-safety)
    inner: Arc<hyperphysics_cognition::CognitionSystem>,
}

#[napi]
impl CognitionSystem {
    /// Create new cognition system
    #[napi(constructor)]
    pub fn new(config: JsCognitionConfig) -> napi::Result<Self> {
        let rust_config: CognitionConfig = config.into();

        match hyperphysics_cognition::CognitionSystem::new(rust_config) {
            Ok(system) => {
                info!("🧠 CognitionSystem created via NAPI");
                Ok(Self {
                    inner: Arc::new(system),
                })
            }
            Err(e) => {
                error!("Failed to create cognition system: {}", e);
                Err(Error::new(
                    Status::GenericFailure,
                    format!("Cognition system creation failed: {}", e)
                ))
            }
        }
    }

    /// Get current arousal level
    #[napi]
    pub fn get_arousal(&self) -> f64 {
        self.inner.arousal().value()
    }

    /// Set arousal level
    #[napi]
    pub fn set_arousal(&self, level: f64) {
        self.inner.set_arousal(ArousalLevel::new(level));
    }

    /// Get current cognitive load
    #[napi]
    pub fn get_load(&self) -> f64 {
        self.inner.cognitive_load().value()
    }

    /// Set cognitive load
    #[napi]
    pub fn set_load(&self, load: f64) {
        self.inner.set_cognitive_load(CognitiveLoad::new(load));
    }

    /// Check if system is healthy
    #[napi]
    pub fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }

    /// Get configuration
    #[napi]
    pub fn config(&self) -> JsCognitionConfig {
        self.inner.config().clone().into()
    }
}

// ============================================================================
// Version Info
// ============================================================================

/// Get version string
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_napi_config_conversion() {
        let js_config = JsCognitionConfig {
            enable_attention: true,
            enable_loops: true,
            enable_dream: true,
            enable_learning: true,
            enable_integration: true,
            default_curvature: 1.0,
            loop_frequency: 40.0,
            dream_threshold: 0.3,
        };

        let rust_config: CognitionConfig = js_config.clone().into();
        assert_eq!(rust_config.default_curvature, 1.0);
        assert_eq!(rust_config.loop_frequency, 40.0);

        let js_config_back: JsCognitionConfig = rust_config.into();
        assert_eq!(js_config_back.default_curvature, js_config.default_curvature);
    }

    #[test]
    fn test_phase_conversion() {
        let js_phase = JsCognitionPhase::Perceiving;
        let rust_phase: CognitionPhase = js_phase.into();
        assert_eq!(rust_phase, CognitionPhase::Perceiving);

        let js_phase_back: JsCognitionPhase = rust_phase.into();
        assert!(matches!(js_phase_back, JsCognitionPhase::Perceiving));
    }
}
