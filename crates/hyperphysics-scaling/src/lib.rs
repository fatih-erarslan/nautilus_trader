//! Auto-Scaling Orchestrator for HyperPhysics
//!
//! Dynamically scales simulation configuration based on system capabilities.
//! Supports 48 nodes → 1 billion nodes with intelligent resource allocation.

pub mod config;
pub mod workload;

use hyperphysics_core::{EngineConfig, Result, EngineError};
use sysinfo::System;

/// Auto-scaling orchestrator
pub struct AutoScaler {
    system: System,
}

impl AutoScaler {
    /// Create new auto-scaler
    pub fn new() -> Self {
        Self {
            system: System::new_all(),
        }
    }

    /// Get current system capabilities
    pub fn system_capabilities(&mut self) -> SystemCapabilities {
        self.system.refresh_all();

        SystemCapabilities {
            cpu_count: num_cpus::get(),
            total_memory_gb: self.system.total_memory() as f64 / 1e9,
            available_memory_gb: self.system.available_memory() as f64 / 1e9,
            has_gpu: false, // TODO: Detect GPU via hyperphysics-gpu
        }
    }

    /// Recommend optimal configuration for target node count
    ///
    /// # Arguments
    /// * `target_nodes` - Desired number of pBits in lattice
    ///
    /// # Returns
    /// Optimal `EngineConfig` matching system capabilities
    pub fn recommend_config(&mut self, target_nodes: usize) -> Result<EngineConfig> {
        let caps = self.system_capabilities();

        // ROI 48: 48 nodes (default)
        if target_nodes <= 48 {
            return Ok(EngineConfig::roi_48(1.0));
        }

        // ROI 128×128: 16,384 nodes
        if target_nodes <= 16_384 {
            // Check memory requirements: ~500 MB for 16K nodes
            let required_memory_gb = 0.5;
            if caps.available_memory_gb < required_memory_gb {
                return Err(EngineError::Configuration {
                    message: format!(
                        "Insufficient memory: {} GB available, {} GB required",
                        caps.available_memory_gb, required_memory_gb
                    ),
                });
            }
            return Ok(EngineConfig::roi_48(1.0)); // TODO: Create roi_128x128 config
        }

        // ROI 1024×1024: 1,048,576 nodes
        if target_nodes <= 1_048_576 {
            let required_memory_gb = 8.0;
            if caps.available_memory_gb < required_memory_gb {
                return Err(EngineError::Configuration {
                    message: format!(
                        "Insufficient memory: {} GB available, {} GB required",
                        caps.available_memory_gb, required_memory_gb
                    ),
                });
            }
            if !caps.has_gpu {
                return Err(EngineError::Configuration {
                    message: "GPU required for 1M+ nodes".to_string(),
                });
            }
            return Ok(EngineConfig::roi_48(1.0)); // TODO: Create roi_1024x1024 config
        }

        // ROI 32K×32K: 1 billion nodes
        if target_nodes <= 1_073_741_824 {
            let required_memory_gb = 128.0;
            if caps.available_memory_gb < required_memory_gb {
                return Err(EngineError::Configuration {
                    message: format!(
                        "Insufficient memory: {} GB available, {} GB required",
                        caps.available_memory_gb, required_memory_gb
                    ),
                });
            }
            if !caps.has_gpu {
                return Err(EngineError::Configuration {
                    message: "High-end GPU required for 1B nodes".to_string(),
                });
            }
            return Ok(EngineConfig::roi_48(1.0)); // TODO: Create roi_32kx32k config
        }

        Err(EngineError::Configuration {
            message: format!("Node count {} exceeds maximum supported (1B)", target_nodes),
        })
    }
}

impl Default for AutoScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// System capability snapshot
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    /// Number of CPU cores
    pub cpu_count: usize,
    /// Total system memory in GB
    pub total_memory_gb: f64,
    /// Available system memory in GB
    pub available_memory_gb: f64,
    /// Whether GPU is available
    pub has_gpu: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_capabilities() {
        let mut scaler = AutoScaler::new();
        let caps = scaler.system_capabilities();

        assert!(caps.cpu_count > 0);
        assert!(caps.total_memory_gb > 0.0);
        assert!(caps.available_memory_gb > 0.0);
        assert!(caps.available_memory_gb <= caps.total_memory_gb);
    }

    #[test]
    fn test_recommend_config_small() {
        let mut scaler = AutoScaler::new();
        let config = scaler.recommend_config(48);
        assert!(config.is_ok());
    }

    #[test]
    fn test_recommend_config_too_large() {
        let mut scaler = AutoScaler::new();
        let config = scaler.recommend_config(2_000_000_000);
        assert!(config.is_err());
    }
}
