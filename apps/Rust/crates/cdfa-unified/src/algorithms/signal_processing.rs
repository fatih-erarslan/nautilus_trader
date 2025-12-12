//! Signal processing utilities

use ndarray::Array1;
use crate::error::CdfaResult;

/// Apply low-pass filter
pub fn low_pass_filter(signal: &Array1<f64>, cutoff: f64) -> CdfaResult<Array1<f64>> {
    // Simple exponential moving average implementation
    let alpha = 2.0 / (1.0 + cutoff);
    let mut filtered = Array1::zeros(signal.len());
    
    if !signal.is_empty() {
        filtered[0] = signal[0];
        for i in 1..signal.len() {
            filtered[i] = alpha * signal[i] + (1.0 - alpha) * filtered[i - 1];
        }
    }
    
    Ok(filtered)
}

/// Calculate power spectral density
pub fn power_spectral_density(signal: &Array1<f64>) -> CdfaResult<Array1<f64>> {
    // Simplified PSD calculation
    let n = signal.len();
    let mut psd = Array1::zeros(n / 2);
    
    for i in 0..psd.len() {
        let end_idx = std::cmp::min(i + 2, signal.len());
        let freq_component = signal.slice(ndarray::s![i..end_idx]).mapv(|x| x * x).sum();
        psd[i] = freq_component;
    }
    
    Ok(psd)
}