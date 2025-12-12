//! Prospect Theory implementation with SIMD optimization

use std::simd::*;
use std::arch::x86_64::*;

/// Prospect Theory parameters
#[derive(Debug, Clone)]
pub struct ProspectTheory {
    /// Risk aversion parameter for gains
    pub alpha: f64,
    /// Risk aversion parameter for losses
    pub beta: f64,
    /// Loss aversion coefficient
    pub lambda_loss: f64,
    /// Probability weighting parameter
    pub delta: f64,
}

impl Default for ProspectTheory {
    fn default() -> Self {
        Self {
            alpha: 0.88,
            beta: 0.88,
            lambda_loss: 2.25,
            delta: 0.65,
        }
    }
}

impl ProspectTheory {
    /// Compute subjective value for outcome x
    #[inline]
    pub fn value(&self, x: f64) -> f64 {
        if x >= 0.0 {
            x.powf(self.alpha)
        } else {
            -self.lambda_loss * (-x).powf(self.beta)
        }
    }
    
    /// Compute subjective probability weight
    #[inline]
    pub fn probability_weight(&self, p: f64) -> f64 {
        let p_delta = p.powf(self.delta);
        let q_delta = (1.0 - p).powf(self.delta);
        p_delta / (p_delta + q_delta).powf(1.0 / self.delta)
    }
    
    /// Compute prospect value (value * weighted probability)
    #[inline]
    pub fn prospect_value(&self, outcome: f64, probability: f64) -> f64 {
        self.value(outcome) * self.probability_weight(probability)
    }
    
    /// SIMD-optimized batch value computation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn value_simd(&self, outcomes: &[f64]) -> Vec<f64> {
        let mut results = vec![0.0; outcomes.len()];
        
        // Process 4 values at a time using AVX2
        let chunks = outcomes.chunks_exact(4);
        let remainder = chunks.remainder();
        
        let alpha = _mm256_set1_pd(self.alpha);
        let beta = _mm256_set1_pd(self.beta);
        let lambda_loss = _mm256_set1_pd(self.lambda_loss);
        let zero = _mm256_setzero_pd();
        
        for (chunk, out_chunk) in chunks.zip(results.chunks_exact_mut(4)) {
            let values = _mm256_loadu_pd(chunk.as_ptr());
            
            // Create mask for positive values
            let positive_mask = _mm256_cmp_pd(values, zero, _CMP_GE_OQ);
            
            // Compute positive branch: x^alpha
            let abs_values = _mm256_andnot_pd(_mm256_set1_pd(-0.0), values);
            // Note: AVX2 doesn't have native pow, so we'd use approximation or fall back
            // For demonstration, using scalar fallback
            let mut pos_results = [0.0; 4];
            for i in 0..4 {
                pos_results[i] = chunk[i].powf(self.alpha);
            }
            let positive_result = _mm256_loadu_pd(pos_results.as_ptr());
            
            // Compute negative branch: -lambda * (-x)^beta
            let mut neg_results = [0.0; 4];
            for i in 0..4 {
                if chunk[i] < 0.0 {
                    neg_results[i] = -self.lambda_loss * (-chunk[i]).powf(self.beta);
                }
            }
            let negative_result = _mm256_loadu_pd(neg_results.as_ptr());
            
            // Select based on mask
            let result = _mm256_blendv_pd(negative_result, positive_result, positive_mask);
            _mm256_storeu_pd(out_chunk.as_mut_ptr(), result);
        }
        
        // Process remainder
        for (i, &x) in remainder.iter().enumerate() {
            results[outcomes.len() - remainder.len() + i] = self.value(x);
        }
        
        results
    }
    
    /// SIMD-optimized batch probability weight computation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn probability_weight_simd(&self, probabilities: &[f64]) -> Vec<f64> {
        let mut results = vec![0.0; probabilities.len()];
        
        // For simplicity, using scalar computation
        // Full SIMD implementation would require approximating pow function
        for (i, &p) in probabilities.iter().enumerate() {
            results[i] = self.probability_weight(p);
        }
        
        results
    }
}

/// Normalized signal computation
#[inline]
pub fn normalize_signal(signal: f64, min_signal: f64, max_signal: f64) -> f64 {
    std::f64::consts::PI * (2.0 * (signal - min_signal) / (max_signal - min_signal) - 1.0)
}

/// SIMD-optimized signal normalization
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn normalize_signal_simd(
    signals: &[f64],
    min_signal: f64,
    max_signal: f64,
) -> Vec<f64> {
    let mut results = vec![0.0; signals.len()];
    
    let pi = _mm256_set1_pd(std::f64::consts::PI);
    let two = _mm256_set1_pd(2.0);
    let one = _mm256_set1_pd(1.0);
    let min = _mm256_set1_pd(min_signal);
    let range = _mm256_set1_pd(max_signal - min_signal);
    
    let chunks = signals.chunks_exact(4);
    let remainder = chunks.remainder();
    
    for (chunk, out_chunk) in chunks.zip(results.chunks_exact_mut(4)) {
        let signal = _mm256_loadu_pd(chunk.as_ptr());
        
        // (signal - min) / range
        let normalized = _mm256_div_pd(_mm256_sub_pd(signal, min), range);
        
        // 2 * normalized - 1
        let scaled = _mm256_sub_pd(_mm256_mul_pd(two, normalized), one);
        
        // pi * scaled
        let result = _mm256_mul_pd(pi, scaled);
        
        _mm256_storeu_pd(out_chunk.as_mut_ptr(), result);
    }
    
    // Process remainder
    for (i, &s) in remainder.iter().enumerate() {
        results[signals.len() - remainder.len() + i] = normalize_signal(s, min_signal, max_signal);
    }
    
    results
}

/// Prospect value result
#[derive(Debug, Clone)]
pub struct ProspectValue {
    pub value: f64,
    pub weighted_probability: f64,
    pub prospect_value: f64,
}

impl ProspectValue {
    /// Create new prospect value
    pub fn new(theory: &ProspectTheory, outcome: f64, probability: f64) -> Self {
        let value = theory.value(outcome);
        let weighted_probability = theory.probability_weight(probability);
        let prospect_value = value * weighted_probability;
        
        Self {
            value,
            weighted_probability,
            prospect_value,
        }
    }
}

/// Batch prospect computation result
pub struct BatchProspectResult {
    pub values: Vec<f64>,
    pub weighted_probabilities: Vec<f64>,
    pub prospect_values: Vec<f64>,
}

impl ProspectTheory {
    /// Compute prospect values for batch of outcomes and probabilities
    pub fn batch_prospect_values(
        &self,
        outcomes: &[f64],
        probabilities: &[f64],
    ) -> BatchProspectResult {
        assert_eq!(outcomes.len(), probabilities.len());
        
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx2") {
                let values = self.value_simd(outcomes);
                let weighted_probs = self.probability_weight_simd(probabilities);
                let prospect_values: Vec<f64> = values
                    .iter()
                    .zip(weighted_probs.iter())
                    .map(|(&v, &w)| v * w)
                    .collect();
                
                return BatchProspectResult {
                    values,
                    weighted_probabilities: weighted_probs,
                    prospect_values,
                };
            }
        }
        
        // Fallback to scalar computation
        let values: Vec<f64> = outcomes.iter().map(|&x| self.value(x)).collect();
        let weighted_probs: Vec<f64> = probabilities
            .iter()
            .map(|&p| self.probability_weight(p))
            .collect();
        let prospect_values: Vec<f64> = values
            .iter()
            .zip(weighted_probs.iter())
            .map(|(&v, &w)| v * w)
            .collect();
        
        BatchProspectResult {
            values,
            weighted_probabilities: weighted_probs,
            prospect_values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prospect_theory() {
        let theory = ProspectTheory::default();
        
        // Test positive outcome
        let pos_value = theory.value(10.0);
        assert!(pos_value > 0.0);
        assert!(pos_value < 10.0); // Due to alpha < 1
        
        // Test negative outcome
        let neg_value = theory.value(-10.0);
        assert!(neg_value < 0.0);
        assert!(neg_value.abs() > pos_value); // Loss aversion
        
        // Test probability weighting
        let weight = theory.probability_weight(0.5);
        assert!(weight > 0.0 && weight < 1.0);
        
        // Test edge cases
        assert_eq!(theory.probability_weight(0.0), 0.0);
        assert_eq!(theory.probability_weight(1.0), 1.0);
    }
    
    #[test]
    fn test_normalize_signal() {
        let normalized = normalize_signal(0.0, -2.0, 2.0);
        assert!((normalized - 0.0).abs() < 1e-10);
        
        let normalized_max = normalize_signal(2.0, -2.0, 2.0);
        assert!((normalized_max - std::f64::consts::PI).abs() < 1e-10);
        
        let normalized_min = normalize_signal(-2.0, -2.0, 2.0);
        assert!((normalized_min + std::f64::consts::PI).abs() < 1e-10);
    }
    
    #[test]
    fn test_batch_computation() {
        let theory = ProspectTheory::default();
        let outcomes = vec![10.0, -10.0, 5.0, -5.0, 0.0];
        let probabilities = vec![0.8, 0.2, 0.5, 0.5, 0.5];
        
        let result = theory.batch_prospect_values(&outcomes, &probabilities);
        
        assert_eq!(result.values.len(), outcomes.len());
        assert_eq!(result.weighted_probabilities.len(), probabilities.len());
        assert_eq!(result.prospect_values.len(), outcomes.len());
        
        // Verify first value manually
        let expected_value = theory.value(outcomes[0]);
        let expected_weight = theory.probability_weight(probabilities[0]);
        assert!((result.values[0] - expected_value).abs() < 1e-10);
        assert!((result.weighted_probabilities[0] - expected_weight).abs() < 1e-10);
    }
}