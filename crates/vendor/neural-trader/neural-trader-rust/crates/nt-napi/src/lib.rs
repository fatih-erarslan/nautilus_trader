//! Neural Trader NAPI-RS Native Addon
//!
//! This crate provides Node.js bindings for the neural-trader Rust implementation,
//! exposing 99 trading tools across 9 categories via NAPI-RS.

#![warn(clippy::all)]
#![allow(clippy::new_without_default)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

// Module declarations
mod trading;
mod neural;
mod sports;
mod syndicate;
mod prediction;
mod e2b;
mod fantasy;
mod news;
mod portfolio;

// Error handling utilities
mod error;

// Re-export all public APIs
pub use trading::*;
pub use neural::*;
pub use sports::*;
pub use syndicate::*;
pub use prediction::*;
pub use e2b::*;
pub use fantasy::*;
pub use news::*;
pub use portfolio::*;

/// Initialize the neural-trader native module
/// Sets up logging, validates configuration, and prepares runtime
#[napi]
pub async fn init_neural_trader(config: Option<String>) -> Result<String> {
    // Initialize tracing/logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    tracing::info!("Initializing neural-trader native module");

    // Parse configuration if provided
    let config_data = if let Some(cfg) = config {
        serde_json::from_str(&cfg)
            .map_err(|e| Error::from_reason(format!("Invalid config JSON: {}", e)))?
    } else {
        serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "mode": "development"
        })
    };

    Ok(format!("Neural Trader initialized: {}", config_data))
}

/// Get system information and capabilities
#[napi]
pub fn get_system_info() -> Result<SystemInfo> {
    Ok(SystemInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        rust_version: option_env!("CARGO_PKG_RUST_VERSION").unwrap_or("unknown").to_string(),
        build_timestamp: option_env!("BUILD_TIMESTAMP").unwrap_or("unknown").to_string(),
        features: vec![
            "trading".to_string(),
            "neural".to_string(),
            "sports-betting".to_string(),
            "syndicates".to_string(),
            "prediction-markets".to_string(),
            "e2b-deployment".to_string(),
            "fantasy-sports".to_string(),
            "news-analysis".to_string(),
            "portfolio-management".to_string(),
        ],
        total_tools: 99,
    })
}

/// System information structure
#[napi(object)]
pub struct SystemInfo {
    pub version: String,
    pub rust_version: String,
    pub build_timestamp: String,
    pub features: Vec<String>,
    pub total_tools: u32,
}

/// Health check endpoint
#[napi]
pub async fn health_check() -> Result<HealthStatus> {
    Ok(HealthStatus {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        uptime_seconds: 0, // TODO: Track actual uptime
    })
}

/// Health status structure
#[napi(object)]
pub struct HealthStatus {
    pub status: String,
    pub timestamp: String,
    pub uptime_seconds: i64,
}

/// Shutdown the neural-trader module gracefully
#[napi]
pub async fn shutdown() -> Result<String> {
    tracing::info!("Shutting down neural-trader native module");
    Ok("Shutdown complete".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_info() {
        let info = get_system_info().unwrap();
        assert_eq!(info.total_tools, 99);
        assert!(info.features.contains(&"trading".to_string()));
    }
}
