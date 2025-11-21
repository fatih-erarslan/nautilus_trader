//! Test utilities for ATS-Core

/// Generates test calibration data
pub fn generate_calibration_data(size: usize) -> Vec<f64> {
    (0..size).map(|i| (i as f64) * 0.01).collect()
}

/// Generates test prediction data
pub fn generate_predictions(size: usize) -> Vec<f64> {
    (0..size).map(|i| (i as f64) * 0.1 + 0.5).collect()
}

/// Generates random test data
pub fn generate_random_data(size: usize, seed: u64) -> Vec<f64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut state = hasher.finish();
    
    (0..size).map(|_| {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        (state as f64 / u64::MAX as f64) * 2.0 - 1.0
    }).collect()
}