//! Configuration for Varlociraptor variant calling

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for variant calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantCallConfig {
    /// Path to reference genome FASTA
    pub reference_genome: PathBuf,

    /// Path to tumor/case BAM file
    pub tumor_bam: PathBuf,

    /// Optional path to normal/control BAM file
    pub normal_bam: Option<PathBuf>,

    /// Path to varlociraptor scenario file
    pub scenario_file: PathBuf,

    /// Output VCF file path
    pub output_vcf: PathBuf,

    /// Minimum variant allele frequency
    pub min_vaf: f64,

    /// Minimum coverage depth
    pub min_depth: u32,

    /// Bayesian prior probability
    pub prior_probability: f64,

    /// Use HyperPhysics optimization for parameter tuning
    pub hyperphysics_optimization: bool,

    /// Map variants to hyperbolic space for clustering
    pub hyperbolic_clustering: bool,
}

impl Default for VariantCallConfig {
    fn default() -> Self {
        Self {
            reference_genome: PathBuf::from("reference.fa"),
            tumor_bam: PathBuf::from("tumor.bam"),
            normal_bam: None,
            scenario_file: PathBuf::from("scenario.yaml"),
            output_vcf: PathBuf::from("variants.vcf"),
            min_vaf: 0.05,
            min_depth: 10,
            prior_probability: 0.001,
            hyperphysics_optimization: true,
            hyperbolic_clustering: false,
        }
    }
}

impl VariantCallConfig {
    /// Create new configuration
    pub fn new(tumor_bam: PathBuf, output_vcf: PathBuf) -> Self {
        Self {
            tumor_bam,
            output_vcf,
            ..Default::default()
        }
    }

    /// Set reference genome
    pub fn with_reference(mut self, reference: PathBuf) -> Self {
        self.reference_genome = reference;
        self
    }

    /// Set normal BAM file
    pub fn with_normal(mut self, normal: PathBuf) -> Self {
        self.normal_bam = Some(normal);
        self
    }

    /// Set scenario file
    pub fn with_scenario(mut self, scenario: PathBuf) -> Self {
        self.scenario_file = scenario;
        self
    }

    /// Enable HyperPhysics optimization
    pub fn with_optimization(mut self, enabled: bool) -> Self {
        self.hyperphysics_optimization = enabled;
        self
    }

    /// Enable hyperbolic clustering
    pub fn with_hyperbolic_clustering(mut self, enabled: bool) -> Self {
        self.hyperbolic_clustering = enabled;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        if !self.tumor_bam.exists() {
            anyhow::bail!("Tumor BAM file does not exist: {}", self.tumor_bam.display());
        }

        if let Some(normal) = &self.normal_bam {
            if !normal.exists() {
                anyhow::bail!("Normal BAM file does not exist: {}", normal.display());
            }
        }

        if self.min_vaf < 0.0 || self.min_vaf > 1.0 {
            anyhow::bail!("Invalid min_vaf: must be between 0.0 and 1.0");
        }

        if self.prior_probability < 0.0 || self.prior_probability > 1.0 {
            anyhow::bail!("Invalid prior_probability: must be between 0.0 and 1.0");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = VariantCallConfig::default();
        assert_eq!(config.min_vaf, 0.05);
        assert_eq!(config.min_depth, 10);
        assert!(config.hyperphysics_optimization);
    }

    #[test]
    fn test_config_builder() {
        let config = VariantCallConfig::new(
            PathBuf::from("tumor.bam"),
            PathBuf::from("out.vcf"),
        )
        .with_reference(PathBuf::from("ref.fa"))
        .with_normal(PathBuf::from("normal.bam"))
        .with_optimization(true)
        .with_hyperbolic_clustering(true);

        assert!(config.hyperphysics_optimization);
        assert!(config.hyperbolic_clustering);
        assert!(config.normal_bam.is_some());
    }
}
