use ndarray::{Array1, ArrayView1};
use std::collections::HashMap;

/// Sample Entropy calculation
/// 
/// Measures the complexity and regularity of time series data
pub struct SampleEntropy;

impl SampleEntropy {
    /// Calculate sample entropy of a time series
    /// 
    /// Parameters:
    /// - signal: The time series data
    /// - m: Pattern length
    /// - r: Tolerance for pattern matching (usually 0.1-0.2 * std)
    pub fn calculate(signal: &ArrayView1<f64>, m: usize, r: f64) -> Result<f64, &'static str> {
        let n = signal.len();
        
        if n < m + 1 {
            return Err("Signal too short for given pattern length");
        }
        
        if m == 0 {
            return Err("Pattern length must be at least 1");
        }
        
        if r <= 0.0 {
            return Err("Tolerance must be positive");
        }
        
        // Count pattern matches for length m
        let b_m = Self::count_patterns(signal, m, r)?;
        
        // Count pattern matches for length m+1
        let b_m1 = Self::count_patterns(signal, m + 1, r)?;
        
        // Calculate sample entropy
        if b_m1 == 0.0 {
            Ok(f64::INFINITY)
        } else {
            Ok(-(b_m1 / b_m).ln())
        }
    }
    
    /// Count matching patterns in the signal
    fn count_patterns(signal: &ArrayView1<f64>, m: usize, r: f64) -> Result<f64, &'static str> {
        let n = signal.len();
        let n_patterns = n - m + 1;
        
        if n_patterns < 2 {
            return Ok(0.0);
        }
        
        let mut count = 0.0;
        
        for i in 0..n_patterns {
            for j in (i + 1)..n_patterns {
                // Check if patterns match within tolerance
                let mut matches = true;
                for k in 0..m {
                    if (signal[i + k] - signal[j + k]).abs() > r {
                        matches = false;
                        break;
                    }
                }
                
                if matches {
                    count += 2.0; // Count both (i,j) and (j,i)
                }
            }
        }
        
        Ok(count / (n_patterns * (n_patterns - 1)) as f64)
    }
    
    /// Fast sample entropy using optimized algorithm
    pub fn calculate_fast(signal: &ArrayView1<f64>, m: usize, r: f64) -> Result<f64, &'static str> {
        let n = signal.len();
        
        if n < m + 1 {
            return Err("Signal too short for given pattern length");
        }
        
        // Create templates
        let mut templates_m = Vec::new();
        let mut templates_m1 = Vec::new();
        
        for i in 0..=(n - m) {
            templates_m.push(signal.slice(ndarray::s![i..i + m]).to_vec());
            if i <= n - m - 1 {
                templates_m1.push(signal.slice(ndarray::s![i..i + m + 1]).to_vec());
            }
        }
        
        // Count matches using Chebyshev distance
        let mut matches_m = 0;
        let mut matches_m1 = 0;
        
        for i in 0..templates_m.len() {
            for j in (i + 1)..templates_m.len() {
                let dist_m = Self::chebyshev_distance(&templates_m[i], &templates_m[j]);
                if dist_m <= r {
                    matches_m += 1;
                    
                    // Check m+1 length patterns
                    if i < templates_m1.len() && j < templates_m1.len() {
                        let dist_m1 = Self::chebyshev_distance(&templates_m1[i], &templates_m1[j]);
                        if dist_m1 <= r {
                            matches_m1 += 1;
                        }
                    }
                }
            }
        }
        
        if matches_m == 0 {
            Ok(f64::INFINITY)
        } else {
            let phi_m = matches_m as f64 / ((templates_m.len() * (templates_m.len() - 1)) / 2) as f64;
            let phi_m1 = matches_m1 as f64 / ((templates_m1.len() * (templates_m1.len() - 1)) / 2) as f64;
            
            if phi_m1 == 0.0 {
                Ok(f64::INFINITY)
            } else {
                Ok(-(phi_m1 / phi_m).ln())
            }
        }
    }
    
    /// Chebyshev distance (maximum absolute difference)
    fn chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }
    
    /// Multiscale sample entropy
    pub fn multiscale(
        signal: &ArrayView1<f64>, 
        m: usize, 
        r: f64, 
        scales: &[usize]
    ) -> Result<Vec<f64>, &'static str> {
        let mut entropies = Vec::new();
        
        for &scale in scales {
            if scale == 0 {
                return Err("Scale must be at least 1");
            }
            
            // Coarse-grain the signal
            let coarse_signal = Self::coarse_grain(signal, scale);
            
            // Calculate sample entropy for this scale
            let entropy = Self::calculate(&coarse_signal.view(), m, r)?;
            entropies.push(entropy);
        }
        
        Ok(entropies)
    }
    
    /// Coarse-grain signal for multiscale entropy
    fn coarse_grain(signal: &ArrayView1<f64>, scale: usize) -> Array1<f64> {
        let n = signal.len();
        let coarse_len = n / scale;
        let mut coarse_signal = Array1::zeros(coarse_len);
        
        for i in 0..coarse_len {
            let start = i * scale;
            let end = (i + 1) * scale;
            coarse_signal[i] = signal.slice(ndarray::s![start..end]).mean().unwrap();
        }
        
        coarse_signal
    }
}

/// Approximate Entropy (predecessor to Sample Entropy)
pub struct ApproximateEntropy;

impl ApproximateEntropy {
    /// Calculate approximate entropy
    pub fn calculate(signal: &ArrayView1<f64>, m: usize, r: f64) -> Result<f64, &'static str> {
        let n = signal.len();
        
        if n < m + 1 {
            return Err("Signal too short for given pattern length");
        }
        
        let phi_m = Self::calculate_phi(signal, m, r)?;
        let phi_m1 = Self::calculate_phi(signal, m + 1, r)?;
        
        Ok(phi_m - phi_m1)
    }
    
    fn calculate_phi(signal: &ArrayView1<f64>, m: usize, r: f64) -> Result<f64, &'static str> {
        let n = signal.len();
        let n_patterns = n - m + 1;
        
        let mut phi_sum = 0.0;
        
        for i in 0..n_patterns {
            let mut count = 0;
            
            for j in 0..n_patterns {
                let mut matches = true;
                for k in 0..m {
                    if (signal[i + k] - signal[j + k]).abs() > r {
                        matches = false;
                        break;
                    }
                }
                
                if matches {
                    count += 1;
                }
            }
            
            phi_sum += (count as f64 / n_patterns as f64).ln();
        }
        
        Ok(phi_sum / n_patterns as f64)
    }
}

/// Permutation Entropy
pub struct PermutationEntropy;

impl PermutationEntropy {
    /// Calculate permutation entropy
    pub fn calculate(signal: &ArrayView1<f64>, order: usize, delay: usize) -> Result<f64, &'static str> {
        let n = signal.len();
        
        if order < 2 {
            return Err("Order must be at least 2");
        }
        
        if delay < 1 {
            return Err("Delay must be at least 1");
        }
        
        let n_vectors = n - (order - 1) * delay;
        if n_vectors < 1 {
            return Err("Signal too short for given order and delay");
        }
        
        // Count permutation patterns
        let mut pattern_counts: HashMap<Vec<usize>, usize> = HashMap::new();
        
        for i in 0..n_vectors {
            // Extract embedded vector
            let mut vector = Vec::new();
            for j in 0..order {
                vector.push(signal[i + j * delay]);
            }
            
            // Get permutation pattern
            let pattern = Self::get_permutation_pattern(&vector);
            *pattern_counts.entry(pattern).or_insert(0) += 1;
        }
        
        // Calculate entropy
        let mut entropy = 0.0;
        let total = n_vectors as f64;
        
        for count in pattern_counts.values() {
            let p = *count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        
        // Normalize by maximum possible entropy
        let max_entropy = (1..=order).product::<usize>() as f64;
        Ok(entropy / max_entropy.ln())
    }
    
    /// Get permutation pattern of a vector
    fn get_permutation_pattern(vector: &[f64]) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = vector.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let mut pattern = vec![0; vector.len()];
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            pattern[*idx] = rank;
        }
        
        pattern
    }
}

/// Shannon Entropy and related measures
pub struct ShannonEntropy;

impl ShannonEntropy {
    /// Calculate Shannon entropy of a probability distribution
    pub fn calculate(probabilities: &ArrayView1<f64>) -> Result<f64, &'static str> {
        let sum = probabilities.sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err("Probabilities must sum to 1");
        }
        
        let mut entropy = 0.0;
        for &p in probabilities.iter() {
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        
        Ok(entropy)
    }
    
    /// Calculate entropy of a discrete signal
    pub fn from_signal(signal: &ArrayView1<i32>) -> Result<f64, &'static str> {
        if signal.is_empty() {
            return Err("Signal cannot be empty");
        }
        
        // Count occurrences
        let mut counts: HashMap<i32, usize> = HashMap::new();
        for &value in signal.iter() {
            *counts.entry(value).or_insert(0) += 1;
        }
        
        // Convert to probabilities
        let total = signal.len() as f64;
        let probabilities: Vec<f64> = counts.values()
            .map(|&count| count as f64 / total)
            .collect();
        
        let prob_array = Array1::from_vec(probabilities);
        Self::calculate(&prob_array.view())
    }
    
    /// Differential entropy for continuous signals (using binning)
    pub fn differential(signal: &ArrayView1<f64>, n_bins: usize) -> Result<f64, &'static str> {
        if signal.is_empty() {
            return Err("Signal cannot be empty");
        }
        
        if n_bins < 2 {
            return Err("Number of bins must be at least 2");
        }
        
        // Create histogram
        let min_val = signal.fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = signal.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max_val - min_val).abs() < f64::EPSILON {
            return Ok(0.0); // No variation
        }
        
        let bin_width = (max_val - min_val) / n_bins as f64;
        let mut hist = vec![0; n_bins];
        
        for &value in signal.iter() {
            let bin = ((value - min_val) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            hist[bin] += 1;
        }
        
        // Convert to probabilities and calculate entropy
        let total = signal.len() as f64;
        let mut entropy = 0.0;
        
        for count in hist {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.ln();
            }
        }
        
        // Add bin width correction for differential entropy
        Ok(entropy + bin_width.ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_sample_entropy() {
        let signal = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let entropy = SampleEntropy::calculate(&signal.view(), 2, 0.5).unwrap();
        
        // Highly regular signal should have low entropy
        assert!(entropy < 0.5);
    }
    
    #[test]
    fn test_sample_entropy_random() {
        let signal = array![1.2, 3.4, 0.5, 2.8, 1.9, 4.2, 0.3, 3.1];
        let entropy = SampleEntropy::calculate(&signal.view(), 2, 1.0).unwrap();
        
        // More random signal should have higher entropy
        assert!(entropy > 0.5);
    }
    
    #[test]
    fn test_multiscale_entropy() {
        let signal = array![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0
        ];
        let scales = vec![1, 2, 4];
        
        let entropies = SampleEntropy::multiscale(&signal.view(), 2, 2.0, &scales).unwrap();
        assert_eq!(entropies.len(), 3);
    }
    
    #[test]
    fn test_permutation_entropy() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0];
        let entropy = PermutationEntropy::calculate(&signal.view(), 3, 1).unwrap();
        
        assert!(entropy >= 0.0 && entropy <= 1.0);
    }
    
    #[test]
    fn test_shannon_entropy() {
        let probabilities = array![0.25, 0.25, 0.25, 0.25];
        let entropy = ShannonEntropy::calculate(&probabilities.view()).unwrap();
        
        // Maximum entropy for 4 outcomes
        let expected = 4.0_f64.ln();
        assert!((entropy - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_differential_entropy() {
        let signal = Array1::linspace(0.0, 10.0, 100);
        let entropy = ShannonEntropy::differential(&signal.view(), 10).unwrap();
        
        // Uniform distribution should have relatively high entropy
        assert!(entropy > 1.0);
    }
}