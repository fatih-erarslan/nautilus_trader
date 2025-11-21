//! Fuzz tests for ATS-Core
//!
//! These tests use random inputs to discover edge cases and verify robustness.

use ats_core::{config::AtsCpConfig, prelude::*};

#[test]
fn fuzz_temperature_scaling() {
    let config = AtsCpConfig::default();
    let engine = AtsCpEngine::new(config).unwrap();
    
    // Generate random test cases
    for seed in 0..1000 {
        let predictions = generate_fuzz_data(seed, 1, 200);
        let temperature = generate_fuzz_temperature(seed);
        
        if temperature > 0.0 && !predictions.is_empty() {
            let result = engine.temperature_scale(&predictions, temperature);
            
            // Should either succeed or fail gracefully
            match result {
                Ok(scaled) => {
                    assert_eq!(scaled.len(), predictions.len());
                    for value in scaled {
                        assert!(value.is_finite());
                    }
                },
                Err(_) => {
                    // Error is acceptable for edge cases
                }
            }
        }
    }
}

#[test]
fn fuzz_conformal_prediction() {
    let config = AtsCpConfig::default();
    let engine = AtsCpEngine::new(config).unwrap();
    
    for seed in 0..500 {
        let predictions = generate_fuzz_data(seed, 1, 100);
        let calibration_data = generate_fuzz_data(seed + 1000, 10, 1000);
        
        if !predictions.is_empty() && calibration_data.len() >= 10 {
            let result = engine.conformal_predict(&predictions, &calibration_data);
            
            match result {
                Ok(intervals) => {
                    assert_eq!(intervals.len(), predictions.len());
                    for (lower, upper) in intervals {
                        assert!(lower.is_finite());
                        assert!(upper.is_finite());
                        assert!(lower <= upper);
                    }
                },
                Err(_) => {
                    // Error is acceptable for edge cases
                }
            }
        }
    }
}

#[test]
fn fuzz_simd_operations() {
    let config = AtsCpConfig::default();
    let engine = AtsCpEngine::new(config).unwrap();
    
    for seed in 0..500 {
        let a = generate_fuzz_data(seed, 0, 300);
        let b = generate_fuzz_data(seed + 500, 0, 300);
        
        if a.len() == b.len() && !a.is_empty() {
            let result = engine.simd_vector_add(&a, &b);
            
            match result {
                Ok(sum) => {
                    assert_eq!(sum.len(), a.len());
                    for (i, &value) in sum.iter().enumerate() {
                        if a[i].is_finite() && b[i].is_finite() {
                            let expected = a[i] + b[i];
                            if expected.is_finite() {
                                assert!((value - expected).abs() < 1e-10);
                            }
                        }
                    }
                },
                Err(_) => {
                    // Error is acceptable for dimension mismatches
                }
            }
        }
    }
}

/// Generates fuzz test data with controlled randomness
fn generate_fuzz_data(seed: u64, min_len: usize, max_len: usize) -> Vec<f64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut state = hasher.finish();
    
    // Generate length
    state = state.wrapping_mul(1103515245).wrapping_add(12345);
    let len = min_len + (state as usize) % (max_len - min_len + 1);
    
    // Generate values
    (0..len).map(|_| {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let val = (state as f64) / (u64::MAX as f64) * 20.0 - 10.0;
        
        // Occasionally introduce special values
        match state % 100 {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            3 => 0.0,
            4 => -0.0,
            5 => f64::MIN_POSITIVE,
            6 => f64::MAX,
            7 => f64::MIN,
            _ => val,
        }
    }).collect()
}

/// Generates fuzz test temperatures
fn generate_fuzz_temperature(seed: u64) -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let state = hasher.finish();
    
    match state % 20 {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => f64::NEG_INFINITY,
        3 => 0.0,
        4 => -1.0,
        5 => f64::MIN_POSITIVE,
        6 => f64::MAX,
        _ => ((state as f64) / (u64::MAX as f64)) * 10.0 + 0.1,
    }
}