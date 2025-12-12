//! Configuration management for QBMIA Unified
//!
//! TENGRI-compliant configuration system for all QBMIA components.

use crate::types::*;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main QBMIA configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QbmiaConfig {
    /// GPU device selection preferences
    pub gpu_preferences: GpuPreferences,
    /// Real market data API configurations
    pub market_apis: MarketApiConfig,
    /// Biological intelligence parameters
    pub biological_config: BiologicalConfig,
    /// Performance monitoring settings
    pub performance_config: PerformanceConfig,
    /// TENGRI compliance enforcement
    pub tengri_validation: TengriConfig,
    /// Quantum simulation settings
    pub quantum_config: QuantumConfig,
}

/// Quantum simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    /// Maximum qubits for simulation
    pub max_qubits: u32,
    /// Default circuit depth limit
    pub max_circuit_depth: u32,
    /// Preferred GPU backend for quantum
    pub preferred_backend: Option<GpuBackend>,
    /// Enable quantum error correction
    pub error_correction: bool,
    /// Measurement shots for algorithms
    pub default_shots: u32,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            max_qubits: 20,
            max_circuit_depth: 1000,
            preferred_backend: None,
            error_correction: true,
            default_shots: 1024,
        }
    }
}

impl QbmiaConfig {
    /// Load configuration from file with TENGRI validation
    pub async fn load_from_file(path: &str) -> Result<Self> {
        let contents = tokio::fs::read_to_string(path).await?;
        let config: Self = toml::from_str(&contents)
            .map_err(|e| crate::error::QbmiaError::Parse(e.to_string()))?;
        
        config.validate_tengri_compliance().await?;
        Ok(config)
    }

    /// Save configuration to file
    pub async fn save_to_file(&self, path: &str) -> Result<()> {
        let contents = toml::to_string_pretty(self)
            .map_err(|e| crate::error::QbmiaError::SerializationError {
                format: "toml".to_string(),
                reason: e.to_string(),
            })?;
        
        tokio::fs::write(path, contents).await?;
        Ok(())
    }

    /// Validate TENGRI compliance of configuration
    async fn validate_tengri_compliance(&self) -> Result<()> {
        // Ensure no mock data is allowed
        if self.market_apis.allow_mock_data {
            return Err(crate::error::QbmiaError::TengriViolation {
                violation: "Mock data is not allowed in TENGRI-compliant configuration".to_string(),
            });
        }

        // Ensure real hardware is required
        if !self.gpu_preferences.require_real_hardware {
            return Err(crate::error::QbmiaError::TengriViolation {
                violation: "Real hardware detection is required for TENGRI compliance".to_string(),
            });
        }

        // Ensure biological networks are authentic
        if !self.biological_config.use_real_neural_patterns {
            return Err(crate::error::QbmiaError::TengriViolation {
                violation: "Authentic neural patterns are required for TENGRI compliance".to_string(),
            });
        }

        Ok(())
    }
}

/// Configuration validation utilities
pub mod validation {
    use super::*;

    /// Validate API configuration
    pub fn validate_api_config(config: &MarketApiConfig) -> Result<()> {
        if config.allow_mock_data {
            return Err(crate::error::QbmiaError::TengriViolation {
                violation: "Mock data not allowed in market API configuration".to_string(),
            });
        }

        // Check for realistic timeout values
        if config.timeout_ms > 300000 { // 5 minutes max
            return Err(crate::error::QbmiaError::ConfigurationError {
                config: "market_apis".to_string(),
                reason: "Timeout too high for real-time trading".to_string(),
            });
        }

        Ok(())
    }

    /// Validate GPU configuration
    pub fn validate_gpu_config(config: &GpuPreferences) -> Result<()> {
        if config.allow_simulation {
            return Err(crate::error::QbmiaError::TengriViolation {
                violation: "GPU simulation not allowed - real hardware only".to_string(),
            });
        }

        Ok(())
    }
}