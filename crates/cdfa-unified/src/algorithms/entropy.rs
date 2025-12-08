use ndarray::{Array1, ArrayView1};
use std::collections::HashMap;

/// Sample Entropy calculation
/// 
/// Measures the complexity and regularity of time series data
/// Enhanced version with additional optimizations and features
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
        if b_m1 == 0.0 || b_m == 0.0 {
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
    
    /// Composite multiscale sample entropy (CMSE)
    pub fn composite_multiscale(
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
            
            // Generate multiple coarse-grained series
            let mut coarse_signals = Vec::new();
            for j in 0..scale {
                let coarse_signal = Self::coarse_grain_with_offset(signal, scale, j);
                if !coarse_signal.is_empty() {
                    coarse_signals.push(coarse_signal);
                }
            }
            
            if coarse_signals.is_empty() {
                entropies.push(f64::INFINITY);
                continue;
            }
            
            // Concatenate all coarse-grained series
            let mut concatenated = Vec::new();
            for signal in &coarse_signals {
                concatenated.extend(signal.iter());
            }
            
            let concatenated_array = Array1::from_vec(concatenated);
            let entropy = Self::calculate(&concatenated_array.view(), m, r)?;
            entropies.push(entropy);
        }
        
        Ok(entropies)
    }
    
    /// Coarse-grain signal with offset
    fn coarse_grain_with_offset(signal: &ArrayView1<f64>, scale: usize, offset: usize) -> Array1<f64> {
        let n = signal.len();
        if offset >= n {
            return Array1::zeros(0);
        }
        
        let mut coarse_signal = Vec::new();
        let mut i = offset;
        
        while i + scale <= n {
            let mean = signal.slice(ndarray::s![i..i + scale]).mean().unwrap();
            coarse_signal.push(mean);
            i += scale;
        }
        
        Array1::from_vec(coarse_signal)
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
            
            if count > 0 {
                phi_sum += (count as f64 / n_patterns as f64).ln();
            }
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
        
        // Handle equal values by using original indices for tie-breaking
        indexed.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        
        let mut pattern = vec![0; vector.len()];
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            pattern[*idx] = rank;
        }
        
        pattern
    }
    
    /// Weighted permutation entropy (considers relative variance)
    pub fn weighted(signal: &ArrayView1<f64>, order: usize, delay: usize) -> Result<f64, &'static str> {
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
        
        let mut pattern_weights: HashMap<Vec<usize>, f64> = HashMap::new();
        
        for i in 0..n_vectors {
            // Extract embedded vector
            let mut vector = Vec::new();
            for j in 0..order {
                vector.push(signal[i + j * delay]);
            }
            
            // Calculate relative variance as weight
            let mean = vector.iter().sum::<f64>() / vector.len() as f64;
            let variance = vector.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / vector.len() as f64;
            
            let pattern = Self::get_permutation_pattern(&vector);
            *pattern_weights.entry(pattern).or_insert(0.0) += variance;
        }
        
        // Calculate weighted entropy
        let total_weight: f64 = pattern_weights.values().sum();
        if total_weight == 0.0 {
            return Ok(0.0);
        }
        
        let mut entropy = 0.0;
        for weight in pattern_weights.values() {
            let p = weight / total_weight;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        
        // Normalize by maximum possible entropy
        let max_entropy = (1..=order).product::<usize>() as f64;
        Ok(entropy / max_entropy.ln())
    }
    
    /// Multiscale permutation entropy
    pub fn multiscale(
        signal: &ArrayView1<f64>, 
        order: usize, 
        delay: usize, 
        scales: &[usize]
    ) -> Result<Vec<f64>, &'static str> {
        let mut entropies = Vec::new();
        
        for &scale in scales {
            if scale == 0 {
                return Err("Scale must be at least 1");
            }
            
            let coarse_signal = SampleEntropy::coarse_grain(signal, scale);
            let entropy = Self::calculate(&coarse_signal.view(), order, delay)?;
            entropies.push(entropy);
        }
        
        Ok(entropies)
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
    
    /// Conditional entropy H(Y|X)
    pub fn conditional(
        x_signal: &ArrayView1<i32>, 
        y_signal: &ArrayView1<i32>
    ) -> Result<f64, &'static str> {
        if x_signal.len() != y_signal.len() {
            return Err("Signals must have same length");
        }
        
        if x_signal.is_empty() {
            return Err("Signals cannot be empty");
        }
        
        // Count joint and marginal occurrences
        let mut joint_counts: HashMap<(i32, i32), usize> = HashMap::new();
        let mut x_counts: HashMap<i32, usize> = HashMap::new();
        
        for (&x, &y) in x_signal.iter().zip(y_signal.iter()) {
            *joint_counts.entry((x, y)).or_insert(0) += 1;
            *x_counts.entry(x).or_insert(0) += 1;
        }
        
        let total = x_signal.len() as f64;
        let mut conditional_entropy = 0.0;
        
        for (&(x, y), &joint_count) in &joint_counts {
            let x_count = x_counts[&x];
            let p_xy = joint_count as f64 / total;
            let p_y_given_x = joint_count as f64 / x_count as f64;
            
            if p_y_given_x > 0.0 {
                conditional_entropy -= p_xy * p_y_given_x.ln();
            }
        }
        
        Ok(conditional_entropy)
    }
    
    /// Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)
    pub fn mutual_information(
        x_signal: &ArrayView1<i32>, 
        y_signal: &ArrayView1<i32>
    ) -> Result<f64, &'static str> {
        if x_signal.len() != y_signal.len() {
            return Err("Signals must have same length");
        }
        
        let h_x = Self::from_signal(x_signal)?;
        let h_y = Self::from_signal(y_signal)?;
        
        // Calculate joint entropy
        let mut joint_signal = Vec::new();
        let mut value_map = HashMap::new();
        let mut next_value = 0;
        
        for (&x, &y) in x_signal.iter().zip(y_signal.iter()) {
            let joint_key = (x, y);
            let mapped_value = *value_map.entry(joint_key).or_insert_with(|| {
                let val = next_value;
                next_value += 1;
                val
            });
            joint_signal.push(mapped_value);
        }
        
        let joint_array = Array1::from_vec(joint_signal);
        let h_xy = Self::from_signal(&joint_array.view())?;
        
        Ok(h_x + h_y - h_xy)
    }
}

/// Tsallis Entropy (generalization of Shannon entropy)
pub struct TsallisEntropy;

impl TsallisEntropy {
    /// Calculate Tsallis entropy with parameter q
    pub fn calculate(probabilities: &ArrayView1<f64>, q: f64) -> Result<f64, &'static str> {
        let sum = probabilities.sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err("Probabilities must sum to 1");
        }
        
        if (q - 1.0).abs() < f64::EPSILON {
            // Special case: q = 1 gives Shannon entropy
            return ShannonEntropy::calculate(probabilities);
        }
        
        let mut sum_pq = 0.0;
        for &p in probabilities.iter() {
            if p > 0.0 {
                sum_pq += p.powf(q);
            }
        }
        
        Ok((1.0 - sum_pq) / (q - 1.0))
    }
}

/// Rényi Entropy (another generalization of Shannon entropy)
pub struct RenyiEntropy;

impl RenyiEntropy {
    /// Calculate Rényi entropy with parameter alpha
    pub fn calculate(probabilities: &ArrayView1<f64>, alpha: f64) -> Result<f64, &'static str> {
        let sum = probabilities.sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err("Probabilities must sum to 1");
        }
        
        if alpha < 0.0 {
            return Err("Alpha must be non-negative");
        }
        
        if (alpha - 1.0).abs() < f64::EPSILON {
            // Special case: alpha = 1 gives Shannon entropy
            return ShannonEntropy::calculate(probabilities);
        }
        
        if alpha.is_infinite() {
            // Special case: alpha = infinity gives min-entropy
            let max_prob: f64 = probabilities.fold(0.0, |a, &b| a.max(b));
            return Ok(-max_prob.ln());
        }
        
        let mut sum_p_alpha = 0.0;
        for &p in probabilities.iter() {
            if p > 0.0 {
                sum_p_alpha += p.powf(alpha);
            }
        }
        
        if sum_p_alpha <= 0.0 {
            Ok(f64::INFINITY)
        } else {
            Ok(sum_p_alpha.ln() / (1.0 - alpha))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
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
        assert!(entropy > 0.0);
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
    fn test_composite_multiscale_entropy() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scales = vec![1, 2];
        
        let entropies = SampleEntropy::composite_multiscale(&signal.view(), 2, 1.0, &scales).unwrap();
        assert_eq!(entropies.len(), 2);
    }
    
    #[test]
    fn test_approximate_entropy() {
        let signal = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let entropy = ApproximateEntropy::calculate(&signal.view(), 2, 0.5).unwrap();
        
        assert!(entropy >= 0.0);
    }
    
    #[test]
    fn test_permutation_entropy() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0];
        let entropy = PermutationEntropy::calculate(&signal.view(), 3, 1).unwrap();
        
        assert!(entropy >= 0.0 && entropy <= 1.0);
    }
    
    #[test]
    fn test_weighted_permutation_entropy() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0];
        let entropy = PermutationEntropy::weighted(&signal.view(), 3, 1).unwrap();
        
        assert!(entropy >= 0.0 && entropy <= 1.0);
    }
    
    #[test]
    fn test_shannon_entropy() {
        let probabilities = array![0.25, 0.25, 0.25, 0.25];
        let entropy = ShannonEntropy::calculate(&probabilities.view()).unwrap();
        
        // Maximum entropy for 4 outcomes
        let expected = 4.0_f64.ln();
        assert_abs_diff_eq!(entropy, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_differential_entropy() {
        let signal = Array1::linspace(0.0, 10.0, 100);
        let entropy = ShannonEntropy::differential(&signal.view(), 10).unwrap();
        
        // Uniform distribution should have relatively high entropy
        assert!(entropy > 1.0);
    }
    
    #[test]
    fn test_conditional_entropy() {
        let x = array![1, 1, 2, 2, 3, 3];
        let y = array![1, 1, 2, 2, 3, 4];
        
        let h_y_given_x = ShannonEntropy::conditional(&x.view(), &y.view()).unwrap();
        assert!(h_y_given_x >= 0.0);
    }
    
    #[test]
    fn test_mutual_information() {
        let x = array![1, 2, 1, 2, 1, 2];
        let y = array![1, 2, 1, 2, 1, 2];
        
        let mi = ShannonEntropy::mutual_information(&x.view(), &y.view()).unwrap();
        
        // Perfect correlation should give high mutual information
        assert!(mi > 0.5);
    }
    
    #[test]
    fn test_tsallis_entropy() {
        let probabilities = array![0.5, 0.3, 0.2];
        let entropy = TsallisEntropy::calculate(&probabilities.view(), 2.0).unwrap();
        
        assert!(entropy >= 0.0);
    }
    
    #[test]
    fn test_renyi_entropy() {
        let probabilities = array![0.5, 0.3, 0.2];
        let entropy = RenyiEntropy::calculate(&probabilities.view(), 2.0).unwrap();
        
        assert!(entropy >= 0.0);
    }
    
    #[test]
    fn test_renyi_entropy_limit_cases() {
        let probabilities = array![0.5, 0.3, 0.2];
        
        // Alpha = 1 should give Shannon entropy
        let shannon = ShannonEntropy::calculate(&probabilities.view()).unwrap();
        let renyi_1 = RenyiEntropy::calculate(&probabilities.view(), 1.0).unwrap();
        assert_abs_diff_eq!(shannon, renyi_1, epsilon = 1e-10);
        
        // Alpha = infinity should give min-entropy
        let renyi_inf = RenyiEntropy::calculate(&probabilities.view(), f64::INFINITY).unwrap();
        let max_prob: f64 = probabilities.fold(0.0, |a, &b| a.max(b));
        let expected_min_entropy = -max_prob.ln();
        assert_abs_diff_eq!(renyi_inf, expected_min_entropy, epsilon = 1e-10);
    }
}