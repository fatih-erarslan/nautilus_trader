//! Bridge between HyperPhysics and Varlociraptor
//!
//! Provides high-level interface for genomic variant calling integrated with
//! HyperPhysics consciousness and optimization frameworks.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;
use tracing::{debug, info};

use crate::config::VariantCallConfig;
use crate::vcf::VcfVariant;

/// Bridge to Varlociraptor variant calling engine
#[derive(Debug)]
pub struct VarlociraptorBridge {
    /// Path to varlociraptor binary
    varlociraptor_path: PathBuf,
    /// Working directory for temporary files
    work_dir: PathBuf,
}

impl VarlociraptorBridge {
    /// Create new bridge instance
    pub fn new() -> Result<Self> {
        let varlociraptor_path = which::which("varlociraptor")
            .context("Varlociraptor not found. Install with: cargo install varlociraptor")?;

        let work_dir = std::env::temp_dir().join("hyperphysics-varlociraptor");
        std::fs::create_dir_all(&work_dir)?;

        Ok(Self {
            varlociraptor_path,
            work_dir,
        })
    }

    /// Create bridge with custom varlociraptor binary path
    pub fn with_path(path: PathBuf) -> Result<Self> {
        if !path.exists() {
            anyhow::bail!("Varlociraptor binary not found at: {}", path.display());
        }

        let work_dir = std::env::temp_dir().join("hyperphysics-varlociraptor");
        std::fs::create_dir_all(&work_dir)?;

        Ok(Self {
            varlociraptor_path: path,
            work_dir,
        })
    }

    /// Call variants with given configuration
    pub async fn call_variants(&self, config: VariantCallConfig) -> Result<Vec<VcfVariant>> {
        info!("Calling variants with Varlociraptor");
        debug!("Config: {:?}", config);

        // Build varlociraptor command
        let mut cmd = Command::new(&self.varlociraptor_path);
        cmd.arg("call")
            .arg("variants")
            .arg("generic")
            .arg("--obs")
            .arg(config.tumor_bam.as_path());

        if let Some(normal_bam) = &config.normal_bam {
            cmd.arg("--obs").arg(normal_bam.as_path());
        }

        cmd.arg("--scenario")
            .arg(config.scenario_file.as_path())
            .arg("--output")
            .arg(config.output_vcf.as_path());

        // Execute command
        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .context("Failed to execute varlociraptor")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Varlociraptor variant calling failed: {}", stderr);
        }

        // Parse VCF output
        let variants = crate::vcf::parse_vcf_file(&config.output_vcf)
            .context("Failed to parse VCF output")?;

        info!("Called {} variants", variants.len());
        Ok(variants)
    }

    /// Preprocess BAM file for variant calling
    pub async fn preprocess_bam(&self, bam_path: &Path) -> Result<PathBuf> {
        info!("Preprocessing BAM file: {}", bam_path.display());

        let output_path = self.work_dir.join(
            bam_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .replace(".bam", ".preprocessed.bam"),
        );

        let mut cmd = Command::new(&self.varlociraptor_path);
        cmd.arg("preprocess")
            .arg("variants")
            .arg(bam_path)
            .arg("--output")
            .arg(&output_path);

        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .context("Failed to execute varlociraptor preprocess")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Varlociraptor preprocessing failed: {}", stderr);
        }

        Ok(output_path)
    }
}

impl Default for VarlociraptorBridge {
    fn default() -> Self {
        Self::new().expect("Failed to create VarlociraptorBridge")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        // This will fail if varlociraptor is not installed
        let result = VarlociraptorBridge::new();
        if result.is_err() {
            eprintln!("Note: Varlociraptor not installed. Install with: cargo install varlociraptor");
        }
    }
}
