//! Core PADS system implementation
//!
//! This module contains the main PanarchyAdaptiveDecisionSystem implementation
//! that orchestrates all components and provides the primary API.

pub mod pads;
pub mod cache;
pub mod fusion;
pub mod overrides;
pub mod feedback;
pub mod config;
pub mod history;
pub mod summary;
pub mod recovery;
pub mod factory;

// Re-export main types
pub use pads::PanarchyAdaptiveDecisionSystem;
pub use cache::CircuitCache;
pub use fusion::DecisionFusion;
pub use overrides::DecisionOverrides;
pub use feedback::FeedbackProcessor;
pub use history::DecisionHistory;
pub use summary::SystemSummary;
pub use recovery::SystemRecovery;
pub use factory::PadsFactory;