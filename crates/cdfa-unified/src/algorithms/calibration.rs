use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Time series calibration and normalization algorithms
/// 
/// This module provides methods for calibrating time series data
/// including various normalization and scaling techniques
pub struct TimeSeriesCalibration;

impl TimeSeriesCalibration {
    /// Z-score normalization (standardization)
    pub fn z_score_normalize(data: &ArrayView1<f64>) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(0.0);
        
        if std < f64::EPSILON {
            // If standard deviation is zero, return zeros
            Ok(Array1::zeros(data.len()))
        } else {
            Ok((data - mean) / std)
        }
    }
    
    /// Min-max normalization to [0, 1] range
    pub fn min_max_normalize(data: &ArrayView1<f64>) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let min_val = data.fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max_val - min_val).abs() < f64::EPSILON {
            // If range is zero, return zeros
            Ok(Array1::zeros(data.len()))
        } else {
            Ok((data - min_val) / (max_val - min_val))
        }
    }
    
    /// Min-max normalization to custom range [a, b]
    pub fn min_max_normalize_range(
        data: &ArrayView1<f64>, 
        target_min: f64, 
        target_max: f64
    ) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if target_min >= target_max {
            return Err("Target min must be less than target max");
        }
        
        let normalized = Self::min_max_normalize(data)?;
        let range = target_max - target_min;
        
        Ok(normalized * range + target_min)
    }
    
    /// Robust scaling using median and IQR
    pub fn robust_scale(data: &ArrayView1<f64>) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_data.len();
        let median = if n % 2 == 0 {
            (sorted_data[n/2 - 1] + sorted_data[n/2]) / 2.0
        } else {
            sorted_data[n/2]
        };
        
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        let q1 = sorted_data[q1_idx];
        let q3 = sorted_data[q3_idx];
        let iqr = q3 - q1;
        
        if iqr < f64::EPSILON {
            Ok(Array1::zeros(data.len()))
        } else {
            Ok((data - median) / iqr)
        }
    }
    
    /// Unit vector scaling (L2 normalization)
    pub fn unit_vector_scale(data: &ArrayView1<f64>) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let l2_norm = data.mapv(|x| x * x).sum().sqrt();
        
        if l2_norm < f64::EPSILON {
            Ok(Array1::zeros(data.len()))
        } else {
            Ok(data / l2_norm)
        }
    }
    
    /// Quantile uniform normalization
    pub fn quantile_uniform_normalize(data: &ArrayView1<f64>) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let n = data.len();
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
        
        let mut result = Array1::zeros(n);
        for (rank, &original_idx) in sorted_indices.iter().enumerate() {
            result[original_idx] = rank as f64 / (n - 1) as f64;
        }
        
        Ok(result)
    }
    
    /// Power transformation (Box-Cox like)
    pub fn power_transform(data: &ArrayView1<f64>, lambda: f64) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        // Check for non-positive values if lambda <= 0
        if lambda <= 0.0 && data.iter().any(|&x| x <= 0.0) {
            return Err("Data must be positive for lambda <= 0");
        }
        
        let transformed = if lambda.abs() < f64::EPSILON {
            // lambda = 0: log transformation
            data.mapv(|x| x.ln())
        } else {
            // Box-Cox transformation
            data.mapv(|x| (x.powf(lambda) - 1.0) / lambda)
        };
        
        Ok(transformed)
    }
    
    /// Yeo-Johnson transformation (handles negative values)
    pub fn yeo_johnson_transform(data: &ArrayView1<f64>, lambda: f64) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let transformed = data.mapv(|x| {
            if x >= 0.0 {
                if lambda.abs() < f64::EPSILON {
                    (x + 1.0).ln()
                } else {
                    ((x + 1.0).powf(lambda) - 1.0) / lambda
                }
            } else {
                if (lambda - 2.0).abs() < f64::EPSILON {
                    -((-x + 1.0).ln())
                } else {
                    -((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
                }
            }
        });
        
        Ok(transformed)
    }
    
    /// Sigmoid normalization
    pub fn sigmoid_normalize(data: &ArrayView1<f64>, steepness: f64) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if steepness <= 0.0 {
            return Err("Steepness must be positive");
        }
        
        let mean = data.mean().unwrap_or(0.0);
        let transformed = data.mapv(|x| 1.0 / (1.0 + (-steepness * (x - mean)).exp()));
        
        Ok(transformed)
    }
    
    /// Tanh normalization
    pub fn tanh_normalize(data: &ArrayView1<f64>, steepness: f64) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if steepness <= 0.0 {
            return Err("Steepness must be positive");
        }
        
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(0.0);
        
        let normalized = if std > f64::EPSILON {
            data.mapv(|x| (steepness * (x - mean) / std).tanh())
        } else {
            Array1::zeros(data.len())
        };
        
        Ok(normalized)
    }
    
    /// Winsorization (clip outliers to percentiles)
    pub fn winsorize(
        data: &ArrayView1<f64>, 
        lower_percentile: f64, 
        upper_percentile: f64
    ) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if lower_percentile < 0.0 || upper_percentile > 100.0 || lower_percentile >= upper_percentile {
            return Err("Invalid percentiles");
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_data.len();
        let lower_idx = ((lower_percentile / 100.0) * (n - 1) as f64) as usize;
        let upper_idx = ((upper_percentile / 100.0) * (n - 1) as f64) as usize;
        
        let lower_bound = sorted_data[lower_idx];
        let upper_bound = sorted_data[upper_idx];
        
        let winsorized = data.mapv(|x| {
            if x < lower_bound {
                lower_bound
            } else if x > upper_bound {
                upper_bound
            } else {
                x
            }
        });
        
        Ok(winsorized)
    }
    
    /// Detrending using linear regression
    pub fn linear_detrend(data: &ArrayView1<f64>) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let n = data.len();
        if n < 2 {
            return Ok(data.to_owned());
        }
        
        // Create time indices
        let t: Array1<f64> = Array1::range(0.0, n as f64, 1.0);
        let t_mean = t.mean().unwrap();
        let data_mean = data.mean().unwrap();
        
        // Calculate slope and intercept
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..n {
            numerator += (t[i] - t_mean) * (data[i] - data_mean);
            denominator += (t[i] - t_mean).powi(2);
        }
        
        let slope = if denominator > f64::EPSILON { 
            numerator / denominator 
        } else { 
            0.0 
        };
        let intercept = data_mean - slope * t_mean;
        
        // Remove trend
        let trend = slope * &t + intercept;
        Ok(data - trend)
    }
    
    /// Polynomial detrending
    pub fn polynomial_detrend(data: &ArrayView1<f64>, degree: usize) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let n = data.len();
        if degree >= n {
            return Err("Polynomial degree must be less than data length");
        }
        
        if degree == 1 {
            return Self::linear_detrend(data);
        }
        
        // For higher degrees, use simple approximation
        // In practice, this would solve the least squares problem
        let mut detrended = data.to_owned();
        
        // Simple smoothing approach for polynomial detrending
        let window_size = (n / (degree + 1)).max(3);
        for _ in 0..degree {
            let smoothed = Self::moving_average(&detrended.view(), window_size)?;
            detrended = &detrended - &smoothed;
        }
        
        Ok(detrended)
    }
    
    /// Moving average smoothing
    pub fn moving_average(data: &ArrayView1<f64>, window: usize) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if window == 0 {
            return Err("Window size must be positive");
        }
        
        let n = data.len();
        let mut smoothed = Array1::zeros(n);
        
        for i in 0..n {
            let start = if i >= window { i - window + 1 } else { 0 };
            let end = i + 1;
            
            let window_data = data.slice(ndarray::s![start..end]);
            smoothed[i] = window_data.mean().unwrap_or(0.0);
        }
        
        Ok(smoothed)
    }
    
    /// Exponential smoothing
    pub fn exponential_smoothing(data: &ArrayView1<f64>, alpha: f64) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if alpha <= 0.0 || alpha > 1.0 {
            return Err("Alpha must be between 0 and 1");
        }
        
        let n = data.len();
        let mut smoothed = Array1::zeros(n);
        
        smoothed[0] = data[0];
        for i in 1..n {
            smoothed[i] = alpha * data[i] + (1.0 - alpha) * smoothed[i-1];
        }
        
        Ok(smoothed)
    }
    
    /// Seasonal decomposition (simple additive model)
    pub fn seasonal_decompose(
        data: &ArrayView1<f64>, 
        period: usize
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if period == 0 || period >= data.len() {
            return Err("Invalid period");
        }
        
        let n = data.len();
        
        // Calculate trend using moving average
        let trend = Self::moving_average(data, period)?;
        
        // Calculate seasonal component
        let detrended = data - &trend;
        let mut seasonal = Array1::zeros(n);
        
        // Calculate average for each season
        let mut seasonal_sums = vec![0.0; period];
        let mut seasonal_counts = vec![0; period];
        
        for i in 0..n {
            let season_idx = i % period;
            seasonal_sums[season_idx] += detrended[i];
            seasonal_counts[season_idx] += 1;
        }
        
        // Create seasonal pattern
        for i in 0..n {
            let season_idx = i % period;
            if seasonal_counts[season_idx] > 0 {
                seasonal[i] = seasonal_sums[season_idx] / seasonal_counts[season_idx] as f64;
            }
        }
        
        // Calculate residual
        let residual = data - &trend - &seasonal;
        
        Ok((trend, seasonal, residual))
    }
    
    /// Multivariate scaling (column-wise operations)
    pub fn multivariate_scale(data: &ArrayView2<f64>, method: ScalingMethod) -> Result<Array2<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let (n_rows, n_cols) = data.dim();
        let mut scaled = Array2::zeros((n_rows, n_cols));
        
        for col in 0..n_cols {
            let column = data.column(col);
            let scaled_column = match method {
                ScalingMethod::ZScore => Self::z_score_normalize(&column)?,
                ScalingMethod::MinMax => Self::min_max_normalize(&column)?,
                ScalingMethod::Robust => Self::robust_scale(&column)?,
                ScalingMethod::UnitVector => Self::unit_vector_scale(&column)?,
            };
            
            scaled.column_mut(col).assign(&scaled_column);
        }
        
        Ok(scaled)
    }
}

/// Scaling methods for calibration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalingMethod {
    ZScore,
    MinMax,
    Robust,
    UnitVector,
}

impl ScalingMethod {
    /// Get scaling method from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "zscore" | "standard" | "standardize" => Some(ScalingMethod::ZScore),
            "minmax" | "min_max" => Some(ScalingMethod::MinMax),
            "robust" => Some(ScalingMethod::Robust),
            "unit" | "unit_vector" | "l2" => Some(ScalingMethod::UnitVector),
            _ => None,
        }
    }
    
    /// Get scaling method name
    pub fn name(&self) -> &'static str {
        match self {
            ScalingMethod::ZScore => "z_score",
            ScalingMethod::MinMax => "min_max",
            ScalingMethod::Robust => "robust",
            ScalingMethod::UnitVector => "unit_vector",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_z_score_normalize() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = TimeSeriesCalibration::z_score_normalize(&data.view()).unwrap();
        
        // Check mean is approximately 0 and std is approximately 1
        assert_abs_diff_eq!(normalized.mean().unwrap(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(normalized.std(0.0), 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_min_max_normalize() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = TimeSeriesCalibration::min_max_normalize(&data.view()).unwrap();
        
        // Check range is [0, 1]
        assert_abs_diff_eq!(normalized[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(normalized[4], 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_min_max_normalize_range() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = TimeSeriesCalibration::min_max_normalize_range(
            &data.view(), -1.0, 1.0
        ).unwrap();
        
        // Check range is [-1, 1]
        assert_abs_diff_eq!(normalized[0], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(normalized[4], 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_robust_scale() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // With outlier
        let scaled = TimeSeriesCalibration::robust_scale(&data.view()).unwrap();
        
        // Robust scaling should be less affected by the outlier
        assert!(scaled.len() == data.len());
    }
    
    #[test]
    fn test_unit_vector_scale() {
        let data = array![3.0, 4.0]; // 3-4-5 triangle
        let scaled = TimeSeriesCalibration::unit_vector_scale(&data.view()).unwrap();
        
        // Check L2 norm is 1
        let norm = scaled.mapv(|x| x * x).sum().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_quantile_uniform_normalize() {
        let data = array![5.0, 1.0, 3.0, 2.0, 4.0];
        let normalized = TimeSeriesCalibration::quantile_uniform_normalize(&data.view()).unwrap();
        
        // Check range is [0, 1] and values are uniform quantiles
        let min_val = normalized.fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = normalized.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        assert_abs_diff_eq!(min_val, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(max_val, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_power_transform() {
        let data = array![1.0, 4.0, 9.0, 16.0]; // Perfect squares
        let transformed = TimeSeriesCalibration::power_transform(&data.view(), 0.5).unwrap();
        
        // Square root transformation
        let expected = array![0.0, 1.0, 2.0, 3.0]; // (x^0.5 - 1) / 0.5
        for i in 0..data.len() {
            assert_abs_diff_eq!(transformed[i], expected[i], epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_yeo_johnson_transform() {
        let data = array![-2.0, -1.0, 0.0, 1.0, 2.0]; // Mixed positive/negative
        let transformed = TimeSeriesCalibration::yeo_johnson_transform(&data.view(), 0.0).unwrap();
        
        // Should handle negative values
        assert_eq!(transformed.len(), data.len());
    }
    
    #[test]
    fn test_sigmoid_normalize() {
        let data = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let normalized = TimeSeriesCalibration::sigmoid_normalize(&data.view(), 1.0).unwrap();
        
        // Check range is (0, 1)
        assert!(normalized.iter().all(|&x| x > 0.0 && x < 1.0));
        assert_abs_diff_eq!(normalized[2], 0.5, epsilon = 1e-10); // Sigmoid(0) = 0.5
    }
    
    #[test]
    fn test_tanh_normalize() {
        let data = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let normalized = TimeSeriesCalibration::tanh_normalize(&data.view(), 1.0).unwrap();
        
        // Check range is (-1, 1)
        assert!(normalized.iter().all(|&x| x > -1.0 && x < 1.0));
        assert_abs_diff_eq!(normalized[2], 0.0, epsilon = 1e-10); // tanh(0) = 0
    }
    
    #[test]
    fn test_winsorize() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // With outlier
        let winsorized = TimeSeriesCalibration::winsorize(&data.view(), 10.0, 90.0).unwrap();
        
        // The outlier should be clipped
        assert!(winsorized[5] < 100.0);
    }
    
    #[test]
    fn test_linear_detrend() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0]; // Linear trend
        let detrended = TimeSeriesCalibration::linear_detrend(&data.view()).unwrap();
        
        // Should remove the linear trend, leaving small residuals
        let mean = detrended.mean().unwrap();
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_moving_average() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let smoothed = TimeSeriesCalibration::moving_average(&data.view(), 3).unwrap();
        
        assert_eq!(smoothed.len(), data.len());
        // First value should be just the first data point
        assert_eq!(smoothed[0], 1.0);
        // Third value should be average of first three
        assert_abs_diff_eq!(smoothed[2], 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_exponential_smoothing() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let smoothed = TimeSeriesCalibration::exponential_smoothing(&data.view(), 0.5).unwrap();
        
        assert_eq!(smoothed.len(), data.len());
        assert_eq!(smoothed[0], 1.0); // First value unchanged
    }
    
    #[test]
    fn test_seasonal_decompose() {
        let data = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]; // Period 2
        let (trend, seasonal, residual) = TimeSeriesCalibration::seasonal_decompose(
            &data.view(), 2
        ).unwrap();
        
        assert_eq!(trend.len(), data.len());
        assert_eq!(seasonal.len(), data.len());
        assert_eq!(residual.len(), data.len());
        
        // Check decomposition: data = trend + seasonal + residual
        let reconstructed = &trend + &seasonal + &residual;
        for i in 0..data.len() {
            assert_abs_diff_eq!(reconstructed[i], data[i], epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_multivariate_scale() {
        let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let scaled = TimeSeriesCalibration::multivariate_scale(
            &data.view(), ScalingMethod::ZScore
        ).unwrap();
        
        assert_eq!(scaled.dim(), data.dim());
        
        // Each column should be z-score normalized
        let col0 = scaled.column(0);
        let col1 = scaled.column(1);
        
        assert_abs_diff_eq!(col0.mean().unwrap(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(col1.mean().unwrap(), 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_scaling_method_from_str() {
        assert_eq!(ScalingMethod::from_str("zscore"), Some(ScalingMethod::ZScore));
        assert_eq!(ScalingMethod::from_str("minmax"), Some(ScalingMethod::MinMax));
        assert_eq!(ScalingMethod::from_str("robust"), Some(ScalingMethod::Robust));
        assert_eq!(ScalingMethod::from_str("unit"), Some(ScalingMethod::UnitVector));
        assert_eq!(ScalingMethod::from_str("invalid"), None);
    }
    
    #[test]
    fn test_scaling_method_name() {
        assert_eq!(ScalingMethod::ZScore.name(), "z_score");
        assert_eq!(ScalingMethod::MinMax.name(), "min_max");
        assert_eq!(ScalingMethod::Robust.name(), "robust");
        assert_eq!(ScalingMethod::UnitVector.name(), "unit_vector");
    }
}