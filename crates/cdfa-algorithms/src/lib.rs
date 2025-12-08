//! CDFA Algorithms
//! 
//! Advanced signal processing and analysis algorithms for CDFA

pub mod wavelet;
pub mod entropy;
pub mod volatility;
pub mod pbit_algorithms;

// Re-export main types and functions
pub use wavelet::{WaveletTransform, WaveletPacket};
pub use entropy::{SampleEntropy, ApproximateEntropy, PermutationEntropy, ShannonEntropy};
pub use volatility::{VolatilityClustering, VolatilityRegime};

/// Common utilities for signal processing
pub mod utils {
    use ndarray::{Array1, ArrayView1};
    
    /// Normalize signal to zero mean and unit variance
    pub fn normalize(signal: &ArrayView1<f64>) -> Array1<f64> {
        let mean = signal.mean().unwrap_or(0.0);
        let std = signal.std(0.0);
        
        if std < f64::EPSILON {
            Array1::zeros(signal.len())
        } else {
            (signal - mean) / std
        }
    }
    
    /// Detrend signal using linear regression
    pub fn detrend(signal: &ArrayView1<f64>) -> Array1<f64> {
        let n = signal.len();
        if n < 2 {
            return signal.to_owned();
        }
        
        // Create time indices
        let t: Array1<f64> = Array1::range(0.0, n as f64, 1.0);
        let t_mean = t.mean().unwrap();
        let signal_mean = signal.mean().unwrap();
        
        // Calculate slope and intercept
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            num += (t[i] - t_mean) * (signal[i] - signal_mean);
            den += (t[i] - t_mean) * (t[i] - t_mean);
        }
        
        let slope = if den > f64::EPSILON { num / den } else { 0.0 };
        let intercept = signal_mean - slope * t_mean;
        
        // Remove trend
        signal - (slope * t + intercept)
    }
    
    /// Apply Savitzky-Golay filter for smoothing
    pub fn savgol_filter(signal: &ArrayView1<f64>, window: usize, poly_order: usize) -> Result<Array1<f64>, &'static str> {
        if window % 2 == 0 {
            return Err("Window size must be odd");
        }
        
        if window <= poly_order {
            return Err("Window size must be greater than polynomial order");
        }
        
        if signal.len() < window {
            return Err("Signal length must be at least window size");
        }
        
        let n = signal.len();
        let half_window = window / 2;
        let mut filtered = Array1::zeros(n);
        
        // Simple moving average for now (full Savitzky-Golay would require matrix operations)
        for i in 0..n {
            let start = if i >= half_window { i - half_window } else { 0 };
            let end = if i + half_window < n { i + half_window + 1 } else { n };
            
            let window_data = signal.slice(ndarray::s![start..end]);
            filtered[i] = window_data.mean().unwrap();
        }
        
        Ok(filtered)
    }
    
    /// Calculate signal-to-noise ratio
    pub fn snr(signal: &ArrayView1<f64>, noise: &ArrayView1<f64>) -> Result<f64, &'static str> {
        if signal.len() != noise.len() {
            return Err("Signal and noise must have same length");
        }
        
        let signal_power = signal.mapv(|x| x * x).sum();
        let noise_power = noise.mapv(|x| x * x).sum();
        
        if noise_power < f64::EPSILON {
            Ok(f64::INFINITY)
        } else {
            Ok(10.0 * (signal_power / noise_power).log10())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_integration() {
        // Test that all modules are accessible
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        // Wavelet
        let (approx, detail) = WaveletTransform::dwt_haar(&signal.view()).unwrap();
        assert_eq!(approx.len(), 4);
        
        // Entropy
        let entropy = SampleEntropy::calculate(&signal.view(), 2, 2.0).unwrap();
        assert!(entropy >= 0.0);
        
        // Volatility
        let volatility = VolatilityClustering::ewma_volatility(&signal.view(), 0.94).unwrap();
        assert_eq!(volatility.len(), signal.len());
    }
    
    #[test]
    fn test_utils() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Normalize
        let normalized = utils::normalize(&signal.view());
        assert!((normalized.mean().unwrap()).abs() < 1e-10);
        assert!((normalized.std(0.0) - 1.0).abs() < 1e-10);
        
        // Detrend
        let detrended = utils::detrend(&signal.view());
        assert!(detrended.std(0.0) < signal.std(0.0));
    }
}