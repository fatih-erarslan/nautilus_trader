//! Utility functions and helper types

use crate::error::{QBMIAError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Statistical utilities
pub mod stats {
    use super::*;
    
    /// Calculate mean of a data slice
    pub fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            0.0
        } else {
            data.iter().sum::<f64>() / data.len() as f64
        }
    }
    
    /// Calculate standard deviation
    pub fn std_dev(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let m = mean(data);
        let variance = data.iter()
            .map(|x| (x - m).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
        
        variance.sqrt()
    }
    
    /// Calculate percentile
    pub fn percentile(data: &[f64], p: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

/// Time-related utilities
pub mod time {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    /// Get current timestamp as seconds since Unix epoch
    pub fn now_timestamp() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }
    
    /// Format timestamp as ISO 8601 string
    pub fn format_timestamp(timestamp: f64) -> String {
        chrono::DateTime::from_timestamp(timestamp as i64, 0)
            .unwrap_or_default()
            .to_rfc3339()
    }
}

/// Array utilities
pub mod arrays {
    use super::*;
    
    /// Normalize array to unit norm
    pub fn normalize(mut array: Array1<f64>) -> Result<Array1<f64>> {
        let norm = array.dot(&array).sqrt();
        if norm < 1e-12 {
            return Err(QBMIAError::numerical("Cannot normalize zero vector"));
        }
        array /= norm;
        Ok(array)
    }
    
    /// Calculate cosine similarity between two arrays
    pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64> {
        if a.len() != b.len() {
            return Err(QBMIAError::validation("Array dimension mismatch"));
        }
        
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();
        
        if norm_a * norm_b < 1e-12 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm_a * norm_b))
        }
    }
}

/// Hardware detection utilities
pub mod hardware {
    /// Detect available SIMD features
    pub fn detect_simd_features() -> Vec<String> {
        let mut features = Vec::new();
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                features.push("avx2".to_string());
            }
            if is_x86_feature_detected!("sse4.1") {
                features.push("sse4.1".to_string());
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                features.push("neon".to_string());
            }
        }
        
        features
    }
    
    /// Get number of available CPU cores
    pub fn cpu_cores() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(stats::mean(&data), 3.0);
        assert!((stats::std_dev(&data) - 1.5811388300841898).abs() < 1e-10);
        assert_eq!(stats::percentile(&data, 50.0), 3.0);
    }
    
    #[test]
    fn test_arrays() {
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        
        let normalized = arrays::normalize(a.clone()).unwrap();
        assert!((normalized.dot(&normalized) - 1.0).abs() < 1e-10);
        
        let similarity = arrays::cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 0.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_hardware() {
        let features = hardware::detect_simd_features();
        println!("SIMD features: {:?}", features);
        
        let cores = hardware::cpu_cores();
        assert!(cores > 0);
        println!("CPU cores: {}", cores);
    }
}