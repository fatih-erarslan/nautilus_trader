//! Utility functions for quantum LSTM

use crate::types::*;
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Generate cache key from data
pub fn generate_cache_key<T: Hash>(data: &T) -> CacheKey {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

/// Normalize vector to unit norm
pub fn normalize_vector(vec: &mut Array1<f64>) {
    let norm = vec.dot(vec).sqrt();
    if norm > 1e-10 {
        vec.mapv_inplace(|x| x / norm);
    }
}

/// Calculate quantum fidelity between two states
pub fn quantum_fidelity(state1: &QuantumState, state2: &QuantumState) -> f64 {
    let inner_product: ComplexNum = state1.amplitudes.iter()
        .zip(state2.amplitudes.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    
    inner_product.norm_sqr()
}