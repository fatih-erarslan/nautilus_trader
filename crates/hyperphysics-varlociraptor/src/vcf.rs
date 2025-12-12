//! VCF (Variant Call Format) parsing and manipulation

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// VCF variant record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VcfVariant {
    /// Chromosome
    pub chrom: String,

    /// Position (1-based)
    pub pos: u64,

    /// Variant ID
    pub id: Option<String>,

    /// Reference allele
    pub ref_allele: String,

    /// Alternate alleles
    pub alt_alleles: Vec<String>,

    /// Quality score
    pub qual: Option<f64>,

    /// Filter status
    pub filter: String,

    /// Info fields
    pub info: HashMap<String, String>,

    /// Format specification
    pub format: Option<String>,

    /// Sample data
    pub samples: Vec<HashMap<String, String>>,
}

impl VcfVariant {
    /// Parse variant from VCF line
    pub fn from_vcf_line(line: &str) -> Result<Self> {
        let fields: Vec<&str> = line.split('\t').collect();

        if fields.len() < 8 {
            anyhow::bail!("Invalid VCF line: insufficient fields");
        }

        let chrom = fields[0].to_string();
        let pos = fields[1].parse::<u64>()
            .context("Failed to parse position")?;
        let id = if fields[2] == "." {
            None
        } else {
            Some(fields[2].to_string())
        };
        let ref_allele = fields[3].to_string();
        let alt_alleles = fields[4].split(',').map(|s| s.to_string()).collect();
        let qual = if fields[5] == "." {
            None
        } else {
            Some(fields[5].parse::<f64>()
                .context("Failed to parse quality")?)
        };
        let filter = fields[6].to_string();

        // Parse INFO field
        let mut info = HashMap::new();
        for item in fields[7].split(';') {
            if let Some((key, value)) = item.split_once('=') {
                info.insert(key.to_string(), value.to_string());
            } else {
                info.insert(item.to_string(), "true".to_string());
            }
        }

        // Parse FORMAT and samples if present
        let (format, samples) = if fields.len() >= 10 {
            let format_str = fields[8];
            let format_fields: Vec<&str> = format_str.split(':').collect();

            let mut samples = Vec::new();
            for sample_data in &fields[9..] {
                let sample_values: Vec<&str> = sample_data.split(':').collect();
                let mut sample_map = HashMap::new();

                for (i, field) in format_fields.iter().enumerate() {
                    if i < sample_values.len() {
                        sample_map.insert(field.to_string(), sample_values[i].to_string());
                    }
                }
                samples.push(sample_map);
            }

            (Some(format_str.to_string()), samples)
        } else {
            (None, Vec::new())
        };

        Ok(Self {
            chrom,
            pos,
            id,
            ref_allele,
            alt_alleles,
            qual,
            filter,
            info,
            format,
            samples,
        })
    }

    /// Get variant allele frequency (VAF) from sample
    pub fn get_vaf(&self, sample_idx: usize) -> Option<f64> {
        if sample_idx >= self.samples.len() {
            return None;
        }

        // Try to get VAF from PROB_SOMATIC (varlociraptor format)
        if let Some(prob_som) = self.samples[sample_idx].get("PROB_SOMATIC") {
            return prob_som.parse::<f64>().ok();
        }

        // Try to get from AF field
        if let Some(af) = self.samples[sample_idx].get("AF") {
            return af.parse::<f64>().ok();
        }

        None
    }

    /// Get coverage depth from sample
    pub fn get_depth(&self, sample_idx: usize) -> Option<u32> {
        if sample_idx >= self.samples.len() {
            return None;
        }

        if let Some(dp) = self.samples[sample_idx].get("DP") {
            return dp.parse::<u32>().ok();
        }

        None
    }
}

/// Parse VCF file
pub fn parse_vcf_file(path: &Path) -> Result<Vec<VcfVariant>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open VCF file: {}", path.display()))?;

    let reader = BufReader::new(file);
    let mut variants = Vec::new();

    for line in reader.lines() {
        let line = line.context("Failed to read VCF line")?;

        // Skip header lines
        if line.starts_with('#') {
            continue;
        }

        let variant = VcfVariant::from_vcf_line(&line)
            .context("Failed to parse VCF variant")?;
        variants.push(variant);
    }

    Ok(variants)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vcf_line() {
        let line = "chr1\t12345\trs123\tA\tG\t30.0\tPASS\tDP=50;AF=0.3\tGT:DP:AF\t0/1:50:0.3";
        let variant = VcfVariant::from_vcf_line(line).unwrap();

        assert_eq!(variant.chrom, "chr1");
        assert_eq!(variant.pos, 12345);
        assert_eq!(variant.ref_allele, "A");
        assert_eq!(variant.alt_alleles[0], "G");
        assert_eq!(variant.qual, Some(30.0));
        assert_eq!(variant.filter, "PASS");
    }

    #[test]
    fn test_get_vaf() {
        let line = "chr1\t12345\t.\tA\tG\t.\tPASS\tDP=50\tGT:DP:AF\t0/1:50:0.3";
        let variant = VcfVariant::from_vcf_line(line).unwrap();

        let vaf = variant.get_vaf(0);
        assert_eq!(vaf, Some(0.3));
    }
}
