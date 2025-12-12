//! # HyperPhysics-Varlociraptor Integration
//!
//! Integration bridge between HyperPhysics consciousness/optimization framework
//! and Varlociraptor genomic variant calling engine.
//!
//! ## Features
//!
//! - **Subprocess Interface**: Execute varlociraptor as external process
//! - **Bayesian Integration**: Integrate variant calling with consciousness models
//! - **Hyperbolic Variant Space**: Map genetic variants to hyperbolic geometry
//! - **Optimization**: Use HyperPhysics optimization for parameter tuning
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │                  HyperPhysics Core                          │
//! │   (Consciousness, Optimization, Hyperbolic Geometry)        │
//! └────────────────────┬───────────────────────────────────────┘
//!                      │
//!                      ▼
//! ┌────────────────────────────────────────────────────────────┐
//! │           Varlociraptor Integration Layer                   │
//! │  - Subprocess execution                                     │
//! │  - VCF parsing and generation                               │
//! │  - Bayesian parameter optimization                          │
//! │  - Hyperbolic variant clustering                            │
//! └────────────────────┬───────────────────────────────────────┘
//!                      │
//!                      ▼
//! ┌────────────────────────────────────────────────────────────┐
//! │              Varlociraptor Engine                           │
//! │  (Genomic Variant Calling - External Binary)                │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use hyperphysics_varlociraptor::{VarlociraptorBridge, VariantCallConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize bridge to varlociraptor
//!     let bridge = VarlociraptorBridge::new()?;
//!
//!     // Configure variant calling
//!     let config = VariantCallConfig {
//!         reference_genome: "hg38.fa".into(),
//!         tumor_bam: "tumor.bam".into(),
//!         normal_bam: Some("normal.bam".into()),
//!         output_vcf: "variants.vcf".into(),
//!         ..Default::default()
//!     };
//!
//!     // Execute variant calling with HyperPhysics optimization
//!     let variants = bridge.call_variants(config).await?;
//!
//!     println!("Called {} variants", variants.len());
//!     Ok(())
//! }
//! ```

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tracing::{debug, info, warn};

pub mod bridge;
pub mod config;
pub mod hyperbolic;
pub mod optimization;
pub mod vcf;

pub use bridge::VarlociraptorBridge;
pub use config::VariantCallConfig;
pub use hyperbolic::HyperbolicVariantSpace;
pub use optimization::BayesianParameterOptimizer;

/// Varlociraptor execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Execute varlociraptor as subprocess
    Subprocess,
    /// Direct FFI bindings (requires varlociraptor library)
    Ffi,
}

/// Check if varlociraptor binary is available in PATH
pub async fn check_varlociraptor_available() -> Result<PathBuf> {
    which::which("varlociraptor")
        .context("Varlociraptor binary not found. Install with: cargo install varlociraptor")
}

/// Get varlociraptor version
pub async fn get_varlociraptor_version() -> Result<String> {
    let varlociraptor_path = check_varlociraptor_available().await?;

    let output = Command::new(varlociraptor_path)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("Failed to execute varlociraptor --version")?;

    if !output.status.success() {
        anyhow::bail!("Varlociraptor version check failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    let version = String::from_utf8(output.stdout)
        .context("Invalid UTF-8 in varlociraptor version output")?
        .trim()
        .to_string();

    Ok(version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "Requires varlociraptor installation"]
    async fn test_varlociraptor_available() {
        let result = check_varlociraptor_available().await;
        assert!(result.is_ok(), "Varlociraptor should be available");
    }

    #[tokio::test]
    #[ignore = "Requires varlociraptor installation"]
    async fn test_varlociraptor_version() {
        let version = get_varlociraptor_version().await;
        assert!(version.is_ok(), "Should get varlociraptor version");
        println!("Varlociraptor version: {}", version.unwrap());
    }
}
