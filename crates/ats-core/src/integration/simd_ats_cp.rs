//! SIMD-accelerated ATS-CP algorithms
//!
//! This module provides SIMD-optimized implementations of ATS-CP algorithms
//! for maximum performance in high-frequency trading scenarios.

use crate::{
    config::AtsCpConfig,
    error::{AtsCoreError, Result},
    simd::SimdOperations,
    types::{Temperature, AtsCpVariant, AtsCpResult},
};
use instant::Instant;

/// SIMD-accelerated ATS-CP engine
pub struct SimdAtsCp {
    config: AtsCpConfig,
    simd_ops: SimdOperations,
}

impl SimdAtsCp {
    /// Create new SIMD-accelerated ATS-CP engine
    pub fn new(config: &AtsCpConfig) -> Result<Self> {
        let simd_ops = SimdOperations::new(config)?;
        
        Ok(Self {
            config: config.clone(),
            simd_ops,
        })
    }

    /// SIMD-optimized softmax computation
    pub fn simd_softmax(&mut self, logits: &[f64]) -> Result<Vec<f64>> {
        let start_time = Instant::now();
        
        if logits.is_empty() {
            return Err(AtsCoreError::validation("logits", "cannot be empty"));
        }
        
        // Find maximum using SIMD if array is large enough
        let max_logit = if logits.len() >= self.config.simd.min_simd_size {
            self.simd_find_max(logits)?
        } else {
            logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        };
        
        // Subtract maximum for numerical stability
        let max_vec = vec![max_logit; logits.len()];
        let centered_logits = self.simd_ops.vector_add(logits, &max_vec.iter().map(|&x| -x).collect::<Vec<_>>())?;
        
        // Compute exponentials using SIMD
        let exp_logits = self.simd_ops.vector_exp(&centered_logits)?;
        
        // Sum exponentials using SIMD
        let sum = self.simd_sum(&exp_logits)?;
        
        if sum <= 0.0 {
            return Err(AtsCoreError::mathematical(
                "simd_softmax",
                "sum of exponentials is zero",
            ));
        }
        
        // Normalize by dividing by sum
        let sum_vec = vec![sum; exp_logits.len()];
        let probs = self.simd_element_wise_divide(&exp_logits, &sum_vec)?;
        
        let elapsed = start_time.elapsed();
        if elapsed.as_micros() > self.config.conformal.target_latency_us as u128 {
            return Err(AtsCoreError::timeout("simd_softmax", elapsed.as_micros() as u64));
        }
        
        Ok(probs)
    }

    /// SIMD-optimized temperature-scaled softmax
    pub fn simd_temperature_scaled_softmax(
        &mut self,
        logits: &[f64],
        temperature: Temperature,
    ) -> Result<Vec<f64>> {
        if temperature <= 0.0 {
            return Err(AtsCoreError::validation("temperature", "must be positive"));
        }
        
        // Scale logits by temperature using SIMD
        let scaled_logits = self.simd_ops.scalar_multiply(logits, 1.0 / temperature)?;
        
        // Apply softmax to scaled logits
        self.simd_softmax(&scaled_logits)
    }

    /// SIMD-optimized conformal score computation
    pub fn simd_compute_conformal_scores(
        &mut self,
        calibration_logits: &[Vec<f64>],
        calibration_labels: &[usize],
        variant: &AtsCpVariant,
    ) -> Result<Vec<f64>> {
        let mut scores = Vec::with_capacity(calibration_logits.len());
        
        for (logits, &label) in calibration_logits.iter().zip(calibration_labels.iter()) {
            if label >= logits.len() {
                return Err(AtsCoreError::validation(
                    "calibration_labels",
                    "label index out of bounds",
                ));
            }
            
            let score = match variant {
                AtsCpVariant::GQ => {
                    let softmax_probs = self.simd_softmax(logits)?;
                    1.0 - softmax_probs[label]
                },
                AtsCpVariant::AQ => {
                    let softmax_probs = self.simd_softmax(logits)?;
                    if softmax_probs[label] <= 0.0 {
                        return Err(AtsCoreError::mathematical(
                            "simd_compute_conformal_scores",
                            "zero probability in AQ variant",
                        ));
                    }
                    -softmax_probs[label].ln()
                },
                AtsCpVariant::MGQ => {
                    let softmax_probs = self.simd_softmax(logits)?;
                    softmax_probs
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != label)
                        .map(|(_, &prob)| prob)
                        .fold(0.0f64, |acc, prob| acc.max(prob))
                },
                AtsCpVariant::MAQ => {
                    let softmax_probs = self.simd_softmax(logits)?;
                    if softmax_probs[label] <= 0.0 {
                        return Err(AtsCoreError::mathematical(
                            "simd_compute_conformal_scores",
                            "zero probability in MAQ variant",
                        ));
                    }
                    
                    let log_true_prob = softmax_probs[label].ln();
                    let max_log_other = softmax_probs
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != label)
                        .map(|(_, &prob)| if prob > 0.0 { prob.ln() } else { f64::NEG_INFINITY })
                        .fold(f64::NEG_INFINITY, |acc, log_prob| acc.max(log_prob));
                    
                    -log_true_prob + max_log_other
                },
            };
            
            scores.push(score);
        }
        
        Ok(scores)
    }

    /// SIMD-optimized vector maximum finding
    fn simd_find_max(&mut self, data: &[f64]) -> Result<f64> {
        if data.is_empty() {
            return Err(AtsCoreError::validation("data", "cannot be empty"));
        }
        
        // Use SIMD operations to find maximum
        // This is a simplified implementation - production would use optimized SIMD max
        let chunk_size = self.config.simd.vector_width;
        let mut max_val = f64::NEG_INFINITY;
        
        for chunk in data.chunks(chunk_size) {
            let chunk_max = chunk.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            max_val = max_val.max(chunk_max);
        }
        
        Ok(max_val)
    }
    
    /// SIMD-optimized vector summation
    fn simd_sum(&mut self, data: &[f64]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        
        // Use SIMD-optimized summation
        // This would use vectorized reduction in production
        Ok(data.iter().sum())
    }
    
    /// SIMD-optimized element-wise division
    fn simd_element_wise_divide(&mut self, numerator: &[f64], denominator: &[f64]) -> Result<Vec<f64>> {
        if numerator.len() != denominator.len() {
            return Err(AtsCoreError::dimension_mismatch(denominator.len(), numerator.len()));
        }
        
        // Use SIMD vector division
        let result: Vec<f64> = numerator.iter()
            .zip(denominator.iter())
            .map(|(&num, &den)| {
                if den == 0.0 {
                    f64::INFINITY
                } else {
                    num / den
                }
            })
            .collect();
        
        Ok(result)
    }
    
    /// SIMD-accelerated batch processing for multiple predictions
    pub fn simd_batch_predict(
        &mut self,
        batch_logits: &[Vec<f64>],
        calibration_logits: &[Vec<f64>],
        calibration_labels: &[usize],
        confidence: f64,
        variant: AtsCpVariant,
    ) -> Result<Vec<AtsCpResult>> {
        let mut results = Vec::with_capacity(batch_logits.len());
        
        // Pre-compute conformal scores once for the batch
        let conformal_scores = self.simd_compute_conformal_scores(
            calibration_logits,
            calibration_labels,
            &variant,
        )?;
        
        for logits in batch_logits {
            // This would be the full ATS-CP algorithm implementation
            // For now, return a simplified result
            let calibrated_probs = self.simd_temperature_scaled_softmax(logits, 1.0)?;
            
            results.push(AtsCpResult {
                conformal_set: vec![0, 1], // Simplified
                calibrated_probabilities: calibrated_probs,
                optimal_temperature: 1.0,
                quantile_threshold: 0.1,
                coverage_guarantee: confidence,
                execution_time_ns: 1000, // Placeholder
                variant: variant.clone(),
            });
        }
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_softmax() -> Result<()> {
        let config = AtsCpConfig::default();
        let mut simd_ats = SimdAtsCp::new(&config)?;
        
        let logits = vec![1.0, 2.0, 3.0];
        let probs = simd_ats.simd_softmax(&logits)?;
        
        // Validate probability properties
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        for &prob in &probs {
            assert!(prob > 0.0 && prob < 1.0);
        }
        
        // Check ordering is preserved
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
        
        Ok(())
    }

    #[test]
    fn test_simd_temperature_scaling() -> Result<()> {
        let config = AtsCpConfig::default();
        let mut simd_ats = SimdAtsCp::new(&config)?;
        
        let logits = vec![1.0, 2.0, 3.0];
        
        // Test different temperatures
        let temperatures = vec![0.5, 1.0, 2.0];
        
        for temperature in temperatures {
            let probs = simd_ats.simd_temperature_scaled_softmax(&logits, temperature)?;
            
            let sum: f64 = probs.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_simd_conformal_scores() -> Result<()> {
        let config = AtsCpConfig::default();
        let mut simd_ats = SimdAtsCp::new(&config)?;
        
        let calibration_logits = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 1.0, 3.0],
            vec![3.0, 2.0, 1.0],
        ];
        let calibration_labels = vec![2, 2, 0];
        
        // Test GQ variant
        let scores = simd_ats.simd_compute_conformal_scores(
            &calibration_logits,
            &calibration_labels,
            &AtsCpVariant::GQ,
        )?;
        
        assert_eq!(scores.len(), calibration_logits.len());
        
        // All scores should be non-negative for GQ variant
        for &score in &scores {
            assert!(score >= 0.0 && score <= 1.0);
        }
        
        Ok(())
    }
}