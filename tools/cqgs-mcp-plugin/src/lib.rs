//! # CQGS MCP Plugin
//!
//! Code Quality Governance Sentinels Model Context Protocol Plugin
//!
//! ## Architecture
//!
//! Security-first design with post-quantum Dilithium ML-DSA-65 authentication,
//! following HyperPhysics plugin pattern:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                       CQGS MCP PLUGIN v1.0                              │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
//! │  │  Dilithium  │  │   49 Core   │  │    MCP      │  │   NAPI/     │    │
//! │  │   ML-DSA    │  │  Sentinels  │  │  Protocol   │  │   WASM      │    │
//! │  │  Security   │  │  Exposure   │  │  Server     │  │  Bindings   │    │
//! │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
//! │         │                │                │                │           │
//! │         └────────────────┼────────────────┼────────────────┘           │
//! │                          │                │                            │
//! │  ┌─────────────┐  ┌──────▼──────┐  ┌──────▼──────┐  ┌─────────────┐    │
//! │  │   Wolfram   │  │  Hyperbolic │  │  Symbolic   │  │  Quality    │    │
//! │  │Integration  │  │  Geometry   │  │  Compute    │  │  Metrics    │    │
//! │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - `full` (default): All features enabled
//! - `dilithium`: Post-quantum cryptography (ML-DSA-65)
//! - `mcp`: Model Context Protocol server
//! - `napi`: Node.js/Bun.JS bindings
//! - `wasm`: WebAssembly bindings
//! - `hyperbolic`: H^11 hyperbolic geometry
//! - `symbolic`: Symbolic computation engine
//! - `wolfram`: WolframLLM integration for optimization
//!
//! ## Exposed Sentinels (49 Total)
//!
//! ### Core Governance (17)
//! - Mock Detection, Framework Analysis, Runtime Verification
//! - Reward Hacking Prevention, Policy Enforcement
//! - Real Data Validation, Semantic Analysis, Behavioral Monitoring
//! - Cross-Scale Analysis, Audit Trails, Neural Pattern Detection
//! - Zero-Synthetic Enforcement, Self-Healing, Regression Detection
//! - AST Pattern Matching, Code Quality Metrics
//!
//! ### Security & Performance (12)
//! - Memory Safety, Thread Safety, Type Safety
//! - Security Vulnerability Detection, Performance Optimization
//! - Dependency Analysis, Integration Testing
//! - API Contract Validation, Documentation Quality
//! - Error Handling, Concurrency Patterns, Resource Management
//!
//! ### Infrastructure (10)
//! - Configuration Management, Build Process Validation
//! - CI/CD Pipeline Integration, Deployment Safety
//! - Monitoring & Alerting, Logging Standards
//! - Database Schema Validation, API Versioning
//! - Feature Flag Management, A/B Testing Safety
//!
//! ### Advanced (10)
//! - Distributed System Coordination, Event Sourcing
//! - CQRS Pattern Validation, Microservice Communication
//! - Message Queue Safety, Service Mesh Integration
//! - Cloud Resource Optimization, Container Security
//! - Kubernetes Manifest Validation, Infrastructure as Code

#![allow(clippy::module_name_repetitions)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// ============================================================================
// Core Modules
// ============================================================================

/// Unified sentinel interface - exposes all 49 sentinels
pub mod sentinels;

/// Cognitive capabilities for sentinels (learning, memory, evolution)
/// Based on IIT (Tononi), FEP (Friston), STDP (Bi & Poo)
pub mod cognition;

/// MCP tool definitions and server integration
#[cfg(feature = "mcp")]
pub mod mcp_tools;

// ============================================================================
// Security Layer (Security-First Architecture)
// ============================================================================

/// Dilithium ML-DSA-65 authentication and authorization
#[cfg(feature = "dilithium")]
pub mod dilithium_auth;

// ============================================================================
// Computation Modules
// ============================================================================

/// Hyperbolic geometry integration from cqgs-core
#[cfg(feature = "hyperbolic")]
pub mod hyperbolic;

/// Symbolic computation integration from cqgs-core
#[cfg(feature = "symbolic")]
pub mod symbolic;

/// WolframLLM integration for algorithm optimization
#[cfg(feature = "wolfram")]
pub mod wolfram;

/// Fibonacci-scaled thresholds module
pub mod fibonacci;

// ============================================================================
// Language Bindings
// ============================================================================

/// NAPI bindings for Node.js/Bun.JS
#[cfg(feature = "napi")]
pub mod napi_bindings;

/// WebAssembly bindings
#[cfg(feature = "wasm")]
pub mod wasm_bindings;

// ============================================================================
// Re-exports
// ============================================================================

pub use sentinels::*;

pub use cognition::{
    CognitiveSentinelSystem, CollectiveIntelligence, ConsciousnessMetrics,
    WorkingMemory, EpisodicMemory, SemanticMemory, StdpLearning,
    PHI_IIT, FREE_ENERGY_F, STDP_WEIGHT_CHANGE,
};

#[cfg(feature = "dilithium")]
pub use dilithium_auth::{DilithiumAuth, AuthToken, ClientCredentials};

#[cfg(feature = "mcp")]
pub use mcp_tools::{McpToolRegistry, ToolDefinition, ToolResponse};

#[cfg(feature = "hyperbolic")]
pub use crate::hyperbolic::{
    hyperbolic_distance, lorentz_inner, compute_distance,
    HYPERBOLIC_DIM, LORENTZ_DIM,
};

#[cfg(feature = "hyperbolic")]
pub use crate::hyperbolic::PHI as HYPERBOLIC_PHI;

#[cfg(feature = "symbolic")]
pub use crate::symbolic::{
    shannon_entropy, compute_entropy,
};

pub use crate::fibonacci::{
    TechnicalDebtMinutes, ComplexityThresholds, FileSizeThresholds, EntropyThresholds,
    FibonacciThresholds, fibonacci, golden_power, PHI, PHI_INV,
};

// ============================================================================
// Error Types
// ============================================================================

/// Error types for CQGS MCP operations
#[derive(thiserror::Error, Debug)]
pub enum CqgsMcpError {
    /// Authentication error
    #[error("Authentication failed: {0}")]
    Authentication(String),

    /// Authorization error
    #[error("Authorization denied: {0}")]
    Authorization(String),

    /// Sentinel violation detected
    #[error("Sentinel violation: {0}")]
    SentinelViolation(String),

    /// MCP protocol error
    #[error("MCP protocol error: {0}")]
    Protocol(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Computation error
    #[error("Computation error: {0}")]
    Computation(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Result type for CQGS MCP operations
pub type Result<T> = std::result::Result<T, CqgsMcpError>;

// ============================================================================
// Version Info
// ============================================================================

/// Plugin version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get version information
pub fn version() -> &'static str {
    VERSION
}

/// Get enabled features
pub fn features() -> Vec<&'static str> {
    let mut features = vec!["sentinels"];

    #[cfg(feature = "dilithium")]
    features.push("dilithium");

    #[cfg(feature = "mcp")]
    features.push("mcp");

    #[cfg(feature = "napi")]
    features.push("napi");

    #[cfg(feature = "wasm")]
    features.push("wasm");

    #[cfg(feature = "hyperbolic")]
    features.push("hyperbolic");

    #[cfg(feature = "symbolic")]
    features.push("symbolic");

    #[cfg(feature = "wolfram")]
    features.push("wolfram");

    features.push("fibonacci");

    features
}

/// Get sentinel count
pub fn sentinel_count() -> usize {
    49
}

// ============================================================================
// Prelude Module
// ============================================================================

/// Prelude module - import everything you need
pub mod prelude {
    pub use crate::sentinels::*;

    // Cognitive capabilities
    pub use crate::cognition::{
        CognitiveSentinelSystem, CollectiveIntelligence, ConsciousnessMetrics,
        WorkingMemory, EpisodicMemory, SemanticMemory, StdpLearning,
        PHI_IIT, FREE_ENERGY_F, STDP_WEIGHT_CHANGE,
    };

    #[cfg(feature = "dilithium")]
    pub use crate::dilithium_auth::{DilithiumAuth, AuthToken};

    #[cfg(feature = "mcp")]
    pub use crate::mcp_tools::{McpToolRegistry, ToolDefinition};

    #[cfg(feature = "hyperbolic")]
    pub use crate::hyperbolic::{
        lorentz_inner, hyperbolic_distance, compute_distance,
        HYPERBOLIC_DIM, LORENTZ_DIM, PHI as HYPERBOLIC_PHI,
    };

    #[cfg(feature = "symbolic")]
    pub use crate::symbolic::{
        shannon_entropy, compute_entropy,
        PHI as SYMBOLIC_PHI,
    };

    pub use crate::fibonacci::{
        TechnicalDebtMinutes, ComplexityThresholds, FileSizeThresholds,
        EntropyThresholds, FibonacciThresholds, fibonacci, golden_power,
        PHI, PHI_INV,
    };
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }

    #[test]
    fn test_sentinel_count() {
        assert_eq!(sentinel_count(), 49);
    }

    #[test]
    fn test_features() {
        let f = features();
        assert!(f.contains(&"sentinels"));
    }
}
