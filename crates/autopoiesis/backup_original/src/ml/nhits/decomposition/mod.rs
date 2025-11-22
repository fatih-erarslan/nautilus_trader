//! Multi-scale Time Series Decomposition
//! Decomposes time series into trend, seasonal, and residual components

use ndarray::{Array3, Array2, Array1, s};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for decomposer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposerConfig {
    pub decomposition_type: DecompositionType,
    pub num_scales: usize,
    pub seasonal_periods: Vec<usize>,
    pub trend_filter_size: usize,
    pub use_stl: bool,
    pub robust: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionType {
    Additive,
    Multiplicative,
    STL, // Seasonal-Trend decomposition using Loess
    Wavelet,
    EMD, // Empirical Mode Decomposition
    Hybrid,
}

/// Multi-scale time series decomposer
#[derive(Debug, Clone)]
pub struct MultiScaleDecomposer {
    config: DecomposerConfig,
    
    // Wavelet filters
    wavelet_filters: Vec<WaveletFilter>,
    
    // Seasonal patterns
    seasonal_patterns: Vec<SeasonalPattern>,
    
    // Trend extraction parameters
    trend_smoother: TrendSmoother,
    
    // Learned decomposition parameters
    decomposition_weights: Array2<f64>,
}

/// Wavelet filter for multi-scale decomposition
#[derive(Debug, Clone)]
struct WaveletFilter {
    scale: usize,
    low_pass: Vec<f64>,
    high_pass: Vec<f64>,
}

/// Seasonal pattern detector
#[derive(Debug, Clone)]
struct SeasonalPattern {
    period: usize,
    amplitude: f64,
    phase: f64,
    harmonics: Vec<(f64, f64)>, // (frequency, amplitude) pairs
}

/// Trend smoothing
#[derive(Debug, Clone)]
struct TrendSmoother {
    window_size: usize,
    smoother_type: SmootherType,
    adaptive_weights: Vec<f64>,
}

#[derive(Debug, Clone)]
enum SmootherType {
    MovingAverage,
    Loess,
    HodrickPrescott { lambda: f64 },
    Kalman,
}

impl MultiScaleDecomposer {
    pub fn new(config: &DecomposerConfig) -> Self {
        let wavelet_filters = Self::initialize_wavelet_filters(config.num_scales);
        let seasonal_patterns = config.seasonal_periods.iter()
            .map(|&period| SeasonalPattern {
                period,
                amplitude: 1.0,
                phase: 0.0,
                harmonics: vec![(1.0, 1.0)],
            })
            .collect();
        
        let trend_smoother = TrendSmoother {
            window_size: config.trend_filter_size,
            smoother_type: SmootherType::Loess,
            adaptive_weights: vec![1.0 / config.trend_filter_size as f64; config.trend_filter_size],
        };
        
        let decomposition_weights = Array2::from_shape_fn(
            (config.num_scales, 3), // 3 components: trend, seasonal, residual
            |_| rand::random::<f64>(),
        );
        
        Self {
            config: config.clone(),
            wavelet_filters,
            seasonal_patterns,
            trend_smoother,
            decomposition_weights,
        }
    }
    
    /// Decompose time series into multiple components
    pub fn decompose(&self, input: &Array3<f64>) -> Result<DecomposedSeries, DecompositionError> {
        let (batch_size, seq_len, features) = input.shape();
        
        let mut trend = Array3::zeros((batch_size, seq_len, features));
        let mut seasonal = Array3::zeros((batch_size, seq_len, features));
        let mut residual = Array3::zeros((batch_size, seq_len, features));
        
        match self.config.decomposition_type {
            DecompositionType::Additive => {
                self.additive_decompose(input, &mut trend, &mut seasonal, &mut residual)?;
            }
            DecompositionType::Multiplicative => {
                self.multiplicative_decompose(input, &mut trend, &mut seasonal, &mut residual)?;
            }
            DecompositionType::STL => {
                self.stl_decompose(input, &mut trend, &mut seasonal, &mut residual)?;
            }
            DecompositionType::Wavelet => {
                self.wavelet_decompose(input, &mut trend, &mut seasonal, &mut residual)?;
            }
            DecompositionType::EMD => {
                self.emd_decompose(input, &mut trend, &mut seasonal, &mut residual)?;
            }
            DecompositionType::Hybrid => {
                self.hybrid_decompose(input, &mut trend, &mut seasonal, &mut residual)?;
            }
        }
        
        // Apply multi-scale decomposition
        let scales = self.extract_scales(input)?;
        
        Ok(DecomposedSeries {
            original: input.clone(),
            trend,
            seasonal,
            residual,
            scales,
        })
    }
    
    /// Initialize wavelet filters for different scales
    fn initialize_wavelet_filters(num_scales: usize) -> Vec<WaveletFilter> {
        let mut filters = Vec::new();
        
        for scale in 0..num_scales {
            let filter_length = 2_usize.pow(scale as u32 + 2);
            
            // Daubechies wavelet coefficients (simplified)
            let mut low_pass = vec![0.0; filter_length];
            let mut high_pass = vec![0.0; filter_length];
            
            // Initialize with basic wavelet structure
            for i in 0..filter_length {
                let t = i as f64 / filter_length as f64;
                low_pass[i] = (1.0 - t).sqrt();
                high_pass[i] = if i < filter_length / 2 {
                    t.sqrt()
                } else {
                    -(1.0 - t).sqrt()
                };
            }
            
            // Normalize filters
            let low_sum: f64 = low_pass.iter().map(|x| x * x).sum::<f64>().sqrt();
            let high_sum: f64 = high_pass.iter().map(|x| x * x).sum::<f64>().sqrt();
            
            for i in 0..filter_length {
                low_pass[i] /= low_sum;
                high_pass[i] /= high_sum;
            }
            
            filters.push(WaveletFilter {
                scale: scale + 1,
                low_pass,
                high_pass,
            });
        }
        
        filters
    }
    
    /// Additive decomposition: Y = Trend + Seasonal + Residual
    fn additive_decompose(
        &self,
        input: &Array3<f64>,
        trend: &mut Array3<f64>,
        seasonal: &mut Array3<f64>,
        residual: &mut Array3<f64>,
    ) -> Result<(), DecompositionError> {
        let (batch_size, seq_len, features) = input.shape();
        
        for b in 0..batch_size {
            for f in 0..features {
                let series = input.slice(s![b, .., f]);
                
                // Extract trend
                let trend_component = self.extract_trend(&series)?;
                trend.slice_mut(s![b, .., f]).assign(&trend_component);
                
                // Detrend the series
                let detrended = &series - &trend_component;
                
                // Extract seasonal component
                let seasonal_component = self.extract_seasonal(&detrended)?;
                seasonal.slice_mut(s![b, .., f]).assign(&seasonal_component);
                
                // Calculate residual
                let residual_component = &detrended - &seasonal_component;
                residual.slice_mut(s![b, .., f]).assign(&residual_component);
            }
        }
        
        Ok(())
    }
    
    /// Multiplicative decomposition: Y = Trend * Seasonal * Residual
    fn multiplicative_decompose(
        &self,
        input: &Array3<f64>,
        trend: &mut Array3<f64>,
        seasonal: &mut Array3<f64>,
        residual: &mut Array3<f64>,
    ) -> Result<(), DecompositionError> {
        let (batch_size, _, features) = input.shape();
        
        for b in 0..batch_size {
            for f in 0..features {
                let series = input.slice(s![b, .., f]);
                
                // Extract trend
                let trend_component = self.extract_trend(&series)?;
                trend.slice_mut(s![b, .., f]).assign(&trend_component);
                
                // Detrend the series multiplicatively
                let detrended = &series / &trend_component.mapv(|x| x.max(1e-10));
                
                // Extract seasonal component
                let seasonal_component = self.extract_seasonal(&detrended)?;
                seasonal.slice_mut(s![b, .., f]).assign(&seasonal_component);
                
                // Calculate residual
                let residual_component = &detrended / &seasonal_component.mapv(|x| x.max(1e-10));
                residual.slice_mut(s![b, .., f]).assign(&residual_component);
            }
        }
        
        Ok(())
    }
    
    /// STL decomposition
    fn stl_decompose(
        &self,
        input: &Array3<f64>,
        trend: &mut Array3<f64>,
        seasonal: &mut Array3<f64>,
        residual: &mut Array3<f64>,
    ) -> Result<(), DecompositionError> {
        // Simplified STL implementation
        // Full implementation would use iterative loess smoothing
        
        self.additive_decompose(input, trend, seasonal, residual)?;
        
        if self.config.robust {
            // Apply robust weights based on residual magnitude
            self.apply_robust_weights(input, trend, seasonal, residual)?;
        }
        
        Ok(())
    }
    
    /// Wavelet decomposition
    fn wavelet_decompose(
        &self,
        input: &Array3<f64>,
        trend: &mut Array3<f64>,
        seasonal: &mut Array3<f64>,
        residual: &mut Array3<f64>,
    ) -> Result<(), DecompositionError> {
        let (batch_size, seq_len, features) = input.shape();
        
        for b in 0..batch_size {
            for f in 0..features {
                let series = input.slice(s![b, .., f]).to_owned();
                
                // Apply wavelet transform
                let mut approximation = series.clone();
                let mut details = Vec::new();
                
                for filter in &self.wavelet_filters {
                    let (low, high) = self.apply_wavelet_filter(&approximation, filter)?;
                    details.push(high);
                    approximation = low;
                }
                
                // Trend is the final approximation
                trend.slice_mut(s![b, .., f]).assign(&approximation);
                
                // Seasonal is sum of selected detail coefficients
                let mut seasonal_sum = Array1::zeros(seq_len);
                for (i, detail) in details.iter().enumerate() {
                    if i < details.len() / 2 {
                        seasonal_sum = seasonal_sum + detail;
                    }
                }
                seasonal.slice_mut(s![b, .., f]).assign(&seasonal_sum);
                
                // Residual is remaining details
                let residual_component = &series - &approximation - &seasonal_sum;
                residual.slice_mut(s![b, .., f]).assign(&residual_component);
            }
        }
        
        Ok(())
    }
    
    /// EMD (Empirical Mode Decomposition)
    fn emd_decompose(
        &self,
        input: &Array3<f64>,
        trend: &mut Array3<f64>,
        seasonal: &mut Array3<f64>,
        residual: &mut Array3<f64>,
    ) -> Result<(), DecompositionError> {
        // Simplified EMD implementation
        let (batch_size, _, features) = input.shape();
        
        for b in 0..batch_size {
            for f in 0..features {
                let series = input.slice(s![b, .., f]);
                
                // Extract IMFs (Intrinsic Mode Functions)
                let imfs = self.extract_imfs(&series)?;
                
                // Assign IMFs to components
                if !imfs.is_empty() {
                    // Last IMF is trend
                    trend.slice_mut(s![b, .., f]).assign(&imfs[imfs.len() - 1]);
                    
                    // Middle IMFs are seasonal
                    let mut seasonal_sum = Array1::zeros(series.len());
                    for i in 1..imfs.len() - 1 {
                        seasonal_sum = seasonal_sum + &imfs[i];
                    }
                    seasonal.slice_mut(s![b, .., f]).assign(&seasonal_sum);
                    
                    // First IMF is residual
                    residual.slice_mut(s![b, .., f]).assign(&imfs[0]);
                }
            }
        }
        
        Ok(())
    }
    
    /// Hybrid decomposition combining multiple methods
    fn hybrid_decompose(
        &self,
        input: &Array3<f64>,
        trend: &mut Array3<f64>,
        seasonal: &mut Array3<f64>,
        residual: &mut Array3<f64>,
    ) -> Result<(), DecompositionError> {
        // First pass: wavelet decomposition
        self.wavelet_decompose(input, trend, seasonal, residual)?;
        
        // Second pass: refine seasonal component with STL
        let mut refined_seasonal = seasonal.clone();
        self.stl_decompose(seasonal, trend, &mut refined_seasonal, residual)?;
        seasonal.assign(&refined_seasonal);
        
        // Adjust residual
        *residual = input - trend - seasonal;
        
        Ok(())
    }
    
    /// Extract trend component
    fn extract_trend(&self, series: &ndarray::ArrayView1<f64>) -> Result<Array1<f64>, DecompositionError> {
        match self.trend_smoother.smoother_type {
            SmootherType::MovingAverage => self.moving_average(series, self.trend_smoother.window_size),
            SmootherType::Loess => self.loess_smooth(series, self.trend_smoother.window_size),
            SmootherType::HodrickPrescott { lambda } => self.hp_filter(series, lambda),
            SmootherType::Kalman => self.kalman_smooth(series),
        }
    }
    
    /// Moving average smoothing
    fn moving_average(
        &self,
        series: &ndarray::ArrayView1<f64>,
        window_size: usize,
    ) -> Result<Array1<f64>, DecompositionError> {
        let n = series.len();
        let mut smoothed = Array1::zeros(n);
        
        for i in 0..n {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(n);
            
            let sum: f64 = series.slice(s![start..end]).sum();
            smoothed[i] = sum / (end - start) as f64;
        }
        
        Ok(smoothed)
    }
    
    /// LOESS smoothing
    fn loess_smooth(
        &self,
        series: &ndarray::ArrayView1<f64>,
        window_size: usize,
    ) -> Result<Array1<f64>, DecompositionError> {
        let n = series.len();
        let mut smoothed = Array1::zeros(n);
        
        for i in 0..n {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(n);
            
            // Weighted regression in local window
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;
            
            for j in start..end {
                let dist = ((i as f64 - j as f64) / (window_size as f64 / 2.0)).abs();
                let weight = (1.0 - dist.powi(3)).powi(3).max(0.0); // Tricubic weight
                
                weighted_sum += series[j] * weight;
                weight_sum += weight;
            }
            
            smoothed[i] = weighted_sum / weight_sum.max(1e-10);
        }
        
        Ok(smoothed)
    }
    
    /// Hodrick-Prescott filter
    fn hp_filter(
        &self,
        series: &ndarray::ArrayView1<f64>,
        lambda: f64,
    ) -> Result<Array1<f64>, DecompositionError> {
        // Simplified HP filter
        // Full implementation would solve the optimization problem
        let n = series.len();
        let mut trend = series.to_owned();
        
        // Iterative smoothing
        for _ in 0..10 {
            let mut new_trend = trend.clone();
            
            for i in 1..n - 1 {
                let smooth_term = (trend[i - 1] + trend[i + 1]) / 2.0;
                let data_term = series[i];
                new_trend[i] = (lambda * smooth_term + data_term) / (lambda + 1.0);
            }
            
            trend = new_trend;
        }
        
        Ok(trend)
    }
    
    /// Kalman smoothing
    fn kalman_smooth(&self, series: &ndarray::ArrayView1<f64>) -> Result<Array1<f64>, DecompositionError> {
        // Simplified Kalman filter for trend extraction
        let n = series.len();
        let mut filtered = Array1::zeros(n);
        
        let mut x = series[0]; // State estimate
        let mut p = 1.0; // Error covariance
        let q = 0.01; // Process noise
        let r = 0.1; // Measurement noise
        
        for i in 0..n {
            // Prediction
            let x_pred = x;
            let p_pred = p + q;
            
            // Update
            let k = p_pred / (p_pred + r); // Kalman gain
            x = x_pred + k * (series[i] - x_pred);
            p = (1.0 - k) * p_pred;
            
            filtered[i] = x;
        }
        
        Ok(filtered)
    }
    
    /// Extract seasonal component
    fn extract_seasonal(&self, detrended: &ndarray::ArrayView1<f64>) -> Result<Array1<f64>, DecompositionError> {
        let n = detrended.len();
        let mut seasonal = Array1::zeros(n);
        
        for pattern in &self.seasonal_patterns {
            let period = pattern.period;
            
            // Compute seasonal averages
            let mut seasonal_avgs = vec![0.0; period];
            let mut counts = vec![0; period];
            
            for i in 0..n {
                let season_idx = i % period;
                seasonal_avgs[season_idx] += detrended[i];
                counts[season_idx] += 1;
            }
            
            for i in 0..period {
                if counts[i] > 0 {
                    seasonal_avgs[i] /= counts[i] as f64;
                }
            }
            
            // Apply seasonal pattern
            for i in 0..n {
                seasonal[i] += seasonal_avgs[i % period] * pattern.amplitude;
                
                // Add harmonics
                for (freq, amp) in &pattern.harmonics {
                    let phase = 2.0 * PI * freq * (i as f64) / (period as f64) + pattern.phase;
                    seasonal[i] += amp * phase.sin();
                }
            }
        }
        
        Ok(seasonal)
    }
    
    /// Apply wavelet filter
    fn apply_wavelet_filter(
        &self,
        signal: &Array1<f64>,
        filter: &WaveletFilter,
    ) -> Result<(Array1<f64>, Array1<f64>), DecompositionError> {
        let n = signal.len();
        let filter_len = filter.low_pass.len();
        
        let mut low_freq = Array1::zeros(n);
        let mut high_freq = Array1::zeros(n);
        
        for i in 0..n {
            for j in 0..filter_len {
                let idx = (i + j).wrapping_sub(filter_len / 2) % n;
                low_freq[i] += signal[idx] * filter.low_pass[j];
                high_freq[i] += signal[idx] * filter.high_pass[j];
            }
        }
        
        Ok((low_freq, high_freq))
    }
    
    /// Extract IMFs for EMD
    fn extract_imfs(&self, series: &ndarray::ArrayView1<f64>) -> Result<Vec<Array1<f64>>, DecompositionError> {
        let mut imfs = Vec::new();
        let mut residue = series.to_owned();
        
        // Simplified IMF extraction
        for _ in 0..3 {
            // Find extrema
            let (maxima, minima) = self.find_extrema(&residue)?;
            
            if maxima.len() < 3 || minima.len() < 3 {
                break;
            }
            
            // Interpolate envelopes
            let upper_env = self.interpolate_extrema(&maxima, residue.len())?;
            let lower_env = self.interpolate_extrema(&minima, residue.len())?;
            
            // Compute mean
            let mean = (&upper_env + &lower_env) / 2.0;
            
            // Extract IMF
            let imf = &residue - &mean;
            imfs.push(imf.clone());
            
            // Update residue
            residue = mean;
        }
        
        imfs.push(residue); // Final residue is last IMF
        Ok(imfs)
    }
    
    /// Find local extrema
    fn find_extrema(&self, signal: &Array1<f64>) -> Result<(Vec<(usize, f64)>, Vec<(usize, f64)>), DecompositionError> {
        let mut maxima = Vec::new();
        let mut minima = Vec::new();
        
        for i in 1..signal.len() - 1 {
            if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
                maxima.push((i, signal[i]));
            } else if signal[i] < signal[i - 1] && signal[i] < signal[i + 1] {
                minima.push((i, signal[i]));
            }
        }
        
        Ok((maxima, minima))
    }
    
    /// Interpolate extrema points
    fn interpolate_extrema(
        &self,
        extrema: &[(usize, f64)],
        length: usize,
    ) -> Result<Array1<f64>, DecompositionError> {
        let mut interpolated = Array1::zeros(length);
        
        // Simple linear interpolation between extrema
        for window in extrema.windows(2) {
            let (idx1, val1) = window[0];
            let (idx2, val2) = window[1];
            
            for i in idx1..=idx2 {
                let t = (i - idx1) as f64 / (idx2 - idx1) as f64;
                interpolated[i] = val1 * (1.0 - t) + val2 * t;
            }
        }
        
        // Extrapolate boundaries
        if let Some(&(idx, val)) = extrema.first() {
            for i in 0..idx {
                interpolated[i] = val;
            }
        }
        
        if let Some(&(idx, val)) = extrema.last() {
            for i in idx + 1..length {
                interpolated[i] = val;
            }
        }
        
        Ok(interpolated)
    }
    
    /// Apply robust weights
    fn apply_robust_weights(
        &self,
        input: &Array3<f64>,
        trend: &mut Array3<f64>,
        seasonal: &mut Array3<f64>,
        residual: &mut Array3<f64>,
    ) -> Result<(), DecompositionError> {
        let (batch_size, _, features) = input.shape();
        
        for b in 0..batch_size {
            for f in 0..features {
                let res = residual.slice(s![b, .., f]);
                
                // Compute robust weights based on residual magnitude
                let mad = self.compute_mad(&res)?; // Median Absolute Deviation
                let weights: Array1<f64> = res.mapv(|r| {
                    let z = r.abs() / (mad * 1.4826); // 1.4826 makes MAD consistent with std dev
                    if z <= 1.0 {
                        1.0
                    } else if z <= 2.0 {
                        2.0 - z
                    } else {
                        0.0
                    }
                });
                
                // Re-decompose with weights
                let weighted_input = &input.slice(s![b, .., f]) * &weights;
                let weighted_trend = self.extract_trend(&weighted_input.view())?;
                let weighted_detrended = &weighted_input - &weighted_trend;
                let weighted_seasonal = self.extract_seasonal(&weighted_detrended.view())?;
                
                trend.slice_mut(s![b, .., f]).assign(&weighted_trend);
                seasonal.slice_mut(s![b, .., f]).assign(&weighted_seasonal);
                residual.slice_mut(s![b, .., f]).assign(&(&weighted_input - &weighted_trend - &weighted_seasonal));
            }
        }
        
        Ok(())
    }
    
    /// Compute Median Absolute Deviation
    fn compute_mad(&self, values: &ndarray::ArrayView1<f64>) -> Result<f64, DecompositionError> {
        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        
        let deviations: Vec<f64> = values.iter()
            .map(|&v| (v - median).abs())
            .collect();
        
        let mut sorted_dev = deviations;
        sorted_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mad = if sorted_dev.len() % 2 == 0 {
            (sorted_dev[sorted_dev.len() / 2 - 1] + sorted_dev[sorted_dev.len() / 2]) / 2.0
        } else {
            sorted_dev[sorted_dev.len() / 2]
        };
        
        Ok(mad.max(1e-10))
    }
    
    /// Extract multiple scales
    fn extract_scales(&self, input: &Array3<f64>) -> Result<Vec<Array3<f64>>, DecompositionError> {
        let mut scales = Vec::new();
        let (batch_size, seq_len, features) = input.shape();
        
        for scale_idx in 0..self.config.num_scales {
            let scale_factor = 2_usize.pow(scale_idx as u32);
            
            if seq_len % scale_factor != 0 {
                continue;
            }
            
            let scaled_len = seq_len / scale_factor;
            let mut scaled = Array3::zeros((batch_size, scaled_len, features));
            
            // Downsample to create scale
            for b in 0..batch_size {
                for t in 0..scaled_len {
                    for f in 0..features {
                        let start_idx = t * scale_factor;
                        let end_idx = start_idx + scale_factor;
                        
                        // Average pooling for downsampling
                        let sum: f64 = input.slice(s![b, start_idx..end_idx, f]).sum();
                        scaled[[b, t, f]] = sum / scale_factor as f64;
                    }
                }
            }
            
            scales.push(scaled);
        }
        
        Ok(scales)
    }
}

/// Result of decomposition
#[derive(Debug, Clone)]
pub struct DecomposedSeries {
    pub original: Array3<f64>,
    pub trend: Array3<f64>,
    pub seasonal: Array3<f64>,
    pub residual: Array3<f64>,
    pub scales: Vec<Array3<f64>>,
}

#[derive(Debug, thiserror::Error)]
pub enum DecompositionError {
    #[error("Invalid input shape: {0}")]
    InvalidShape(String),
    
    #[error("Decomposition failed: {0}")]
    DecompositionFailed(String),
    
    #[error("Insufficient data points: {0}")]
    InsufficientData(String),
}

extern crate rand;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decomposition() {
        // Test implementation
    }
}