use ndarray::{Array1, ArrayView1};

/// Advanced statistical functions for time series analysis
pub struct Statistics;

impl Statistics {
    /// Calculate descriptive statistics
    pub fn descriptive_stats(data: &ArrayView1<f64>) -> Result<DescriptiveStats, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let n = data.len();
        let mean = data.mean().unwrap_or(0.0);
        let variance = data.var(0.0);
        let std_dev = variance.sqrt();
        
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median = if n % 2 == 0 {
            (sorted[n/2 - 1] + sorted[n/2]) / 2.0
        } else {
            sorted[n/2]
        };
        
        let min = sorted[0];
        let max = sorted[n-1];
        let range = max - min;
        
        // Quartiles
        let q1 = sorted[n/4];
        let q3 = sorted[3*n/4];
        let iqr = q3 - q1;
        
        // Skewness
        let skewness = if std_dev > 0.0 {
            data.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n as f64
        } else {
            0.0
        };
        
        // Kurtosis
        let kurtosis = if std_dev > 0.0 {
            data.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n as f64 - 3.0
        } else {
            0.0
        };
        
        Ok(DescriptiveStats {
            count: n,
            mean,
            median,
            std_dev,
            variance,
            min,
            max,
            range,
            q1,
            q3,
            iqr,
            skewness,
            kurtosis,
        })
    }
    
    /// Calculate rolling statistics
    pub fn rolling_stats(
        data: &ArrayView1<f64>, 
        window: usize, 
        stat_type: StatType
    ) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if window == 0 {
            return Err("Window size must be positive");
        }
        
        let n = data.len();
        let mut result = Array1::zeros(n);
        
        for i in 0..n {
            let start = if i >= window { i - window + 1 } else { 0 };
            let window_data = data.slice(ndarray::s![start..=i]);
            
            result[i] = match stat_type {
                StatType::Mean => window_data.mean().unwrap_or(0.0),
                StatType::Std => window_data.std(0.0),
                StatType::Var => window_data.var(0.0),
                StatType::Min => window_data.fold(f64::INFINITY, |a, &b| a.min(b)),
                StatType::Max => window_data.fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                StatType::Median => {
                    let mut sorted = window_data.to_vec();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let len = sorted.len();
                    if len % 2 == 0 {
                        (sorted[len/2 - 1] + sorted[len/2]) / 2.0
                    } else {
                        sorted[len/2]
                    }
                },
                StatType::Skewness => {
                    let mean = window_data.mean().unwrap_or(0.0);
                    let std = window_data.std(0.0);
                    if std > 0.0 {
                        window_data.iter()
                            .map(|&x| ((x - mean) / std).powi(3))
                            .sum::<f64>() / window_data.len() as f64
                    } else {
                        0.0
                    }
                },
                StatType::Kurtosis => {
                    let mean = window_data.mean().unwrap_or(0.0);
                    let std = window_data.std(0.0);
                    if std > 0.0 {
                        window_data.iter()
                            .map(|&x| ((x - mean) / std).powi(4))
                            .sum::<f64>() / window_data.len() as f64 - 3.0
                    } else {
                        0.0
                    }
                },
            };
        }
        
        Ok(result)
    }
    
    /// Calculate autocorrelation function
    pub fn autocorrelation(data: &ArrayView1<f64>, max_lag: usize) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if max_lag >= data.len() {
            return Err("Max lag must be less than data length");
        }
        
        let n = data.len();
        let mean = data.mean().unwrap_or(0.0);
        let variance = data.var(0.0);
        
        if variance < f64::EPSILON {
            return Ok(Array1::ones(max_lag + 1));
        }
        
        let mut autocorr = Array1::zeros(max_lag + 1);
        autocorr[0] = 1.0; // Lag 0 is always 1
        
        for lag in 1..=max_lag {
            let mut sum = 0.0;
            let count = n - lag;
            
            for i in 0..count {
                sum += (data[i] - mean) * (data[i + lag] - mean);
            }
            
            autocorr[lag] = sum / (count as f64 * variance);
        }
        
        Ok(autocorr)
    }
    
    /// Calculate partial autocorrelation function
    pub fn partial_autocorrelation(data: &ArrayView1<f64>, max_lag: usize) -> Result<Array1<f64>, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if max_lag >= data.len() {
            return Err("Max lag must be less than data length");
        }
        
        let autocorr = Self::autocorrelation(data, max_lag)?;
        let mut pacf = Array1::zeros(max_lag + 1);
        pacf[0] = 1.0;
        
        if max_lag == 0 {
            return Ok(pacf);
        }
        
        pacf[1] = autocorr[1];
        
        // Durbin-Levinson algorithm
        let mut phi = vec![vec![0.0; max_lag + 1]; max_lag + 1];
        phi[1][1] = autocorr[1];
        
        for k in 2..=max_lag {
            let mut num = autocorr[k];
            for j in 1..k {
                num -= phi[k-1][j] * autocorr[k-j];
            }
            
            phi[k][k] = num;
            pacf[k] = phi[k][k];
            
            for j in 1..k {
                phi[k][j] = phi[k-1][j] - phi[k][k] * phi[k-1][k-j];
            }
        }
        
        Ok(pacf)
    }
    
    /// Cross-correlation between two series
    pub fn cross_correlation(
        x: &ArrayView1<f64>, 
        y: &ArrayView1<f64>, 
        max_lag: usize
    ) -> Result<Array1<f64>, &'static str> {
        if x.len() != y.len() {
            return Err("Series must have same length");
        }
        
        if x.is_empty() {
            return Err("Series cannot be empty");
        }
        
        let n = x.len();
        if max_lag >= n {
            return Err("Max lag must be less than series length");
        }
        
        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(0.0);
        let x_std = x.std(0.0);
        let y_std = y.std(0.0);
        
        if x_std < f64::EPSILON || y_std < f64::EPSILON {
            return Err("Series have zero variance");
        }
        
        let total_lags = 2 * max_lag + 1;
        let mut cross_corr = Array1::zeros(total_lags);
        
        for i in 0..total_lags {
            let lag = i as i32 - max_lag as i32;
            let mut sum = 0.0;
            let mut count = 0;
            
            for j in 0..n {
                let k = j as i32 + lag;
                if k >= 0 && k < n as i32 {
                    let x_norm = (x[j] - x_mean) / x_std;
                    let y_norm = (y[k as usize] - y_mean) / y_std;
                    sum += x_norm * y_norm;
                    count += 1;
                }
            }
            
            if count > 0 {
                cross_corr[i] = sum / count as f64;
            }
        }
        
        Ok(cross_corr)
    }
    
    /// Ljung-Box test for autocorrelation
    pub fn ljung_box_test(data: &ArrayView1<f64>, lags: usize) -> Result<(f64, f64), &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if lags == 0 || lags >= data.len() {
            return Err("Invalid number of lags");
        }
        
        let autocorr = Self::autocorrelation(data, lags)?;
        let n = data.len() as f64;
        
        let mut lb_stat = 0.0;
        for k in 1..=lags {
            lb_stat += autocorr[k].powi(2) / (n - k as f64);
        }
        lb_stat *= n * (n + 2.0);
        
        // Chi-square degrees of freedom
        let dof = lags as f64;
        
        Ok((lb_stat, dof))
    }
    
    /// Jarque-Bera test for normality
    pub fn jarque_bera_test(data: &ArrayView1<f64>) -> Result<(f64, f64), &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let stats = Self::descriptive_stats(data)?;
        let n = data.len() as f64;
        
        let jb_stat = (n / 6.0) * (stats.skewness.powi(2) + stats.kurtosis.powi(2) / 4.0);
        let dof = 2.0; // 2 degrees of freedom
        
        Ok((jb_stat, dof))
    }
    
    /// Augmented Dickey-Fuller test (simplified)
    pub fn adf_test(data: &ArrayView1<f64>, lags: usize) -> Result<(f64, bool), &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if lags >= data.len() - 1 {
            return Err("Too many lags for data length");
        }
        
        let n = data.len();
        let mut diff_data = Array1::zeros(n - 1);
        
        for i in 0..n-1 {
            diff_data[i] = data[i+1] - data[i];
        }
        
        // Simple regression: diff(y_t) = alpha + beta * y_{t-1} + error
        let y_lagged = data.slice(ndarray::s![..n-1]);
        let mean_y_lag = y_lagged.mean().unwrap_or(0.0);
        let mean_diff = diff_data.mean().unwrap_or(0.0);
        
        let mut num = 0.0;
        let mut den = 0.0;
        
        for i in 0..n-1 {
            num += (y_lagged[i] - mean_y_lag) * (diff_data[i] - mean_diff);
            den += (y_lagged[i] - mean_y_lag).powi(2);
        }
        
        let beta = if den > f64::EPSILON { num / den } else { 0.0 };
        
        // Calculate t-statistic (simplified)
        let mut sse = 0.0;
        for i in 0..n-1 {
            let predicted = mean_diff + beta * (y_lagged[i] - mean_y_lag);
            sse += (diff_data[i] - predicted).powi(2);
        }
        
        let mse = sse / (n - 3) as f64; // Adjust for parameters
        let se_beta = if den > f64::EPSILON { (mse / den).sqrt() } else { f64::INFINITY };
        
        let t_stat = if se_beta > f64::EPSILON { beta / se_beta } else { 0.0 };
        
        // Simplified critical value (approximately -2.86 for 5% significance)
        let is_stationary = t_stat < -2.86;
        
        Ok((t_stat, is_stationary))
    }
    
    /// KPSS test for stationarity
    pub fn kpss_test(data: &ArrayView1<f64>) -> Result<(f64, bool), &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let n = data.len();
        if n < 3 {
            return Err("Need at least 3 observations");
        }
        
        // Detrend data
        let mean = data.mean().unwrap_or(0.0);
        let detrended: Array1<f64> = data - mean;
        
        // Calculate partial sums
        let mut partial_sums = Array1::zeros(n);
        partial_sums[0] = detrended[0];
        for i in 1..n {
            partial_sums[i] = partial_sums[i-1] + detrended[i];
        }
        
        // Calculate test statistic
        let sum_sq_partial = partial_sums.mapv(|x| x.powi(2)).sum();
        let sigma_sq = detrended.mapv(|x| x.powi(2)).sum() / n as f64;
        
        let kpss_stat = if sigma_sq > f64::EPSILON {
            sum_sq_partial / (n as f64 * n as f64 * sigma_sq)
        } else {
            0.0
        };
        
        // Critical value for 5% significance (approximately 0.463)
        let is_stationary = kpss_stat < 0.463;
        
        Ok((kpss_stat, is_stationary))
    }
    
    /// Calculate Hurst exponent
    pub fn hurst_exponent(data: &ArrayView1<f64>) -> Result<f64, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let n = data.len();
        if n < 4 {
            return Err("Need at least 4 observations");
        }
        
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(0.0);
        
        if std < f64::EPSILON {
            return Ok(0.5); // Random walk for constant series
        }
        
        // R/S analysis
        let mut rs_values = Vec::new();
        let mut sizes = Vec::new();
        
        let mut size = 4;
        while size <= n / 2 {
            let num_windows = n / size;
            let mut rs_sum = 0.0;
            
            for i in 0..num_windows {
                let start = i * size;
                let end = start + size;
                let window = data.slice(ndarray::s![start..end]);
                
                // Calculate cumulative deviations from mean
                let window_mean = window.mean().unwrap_or(0.0);
                let mut cum_dev = Array1::zeros(size);
                cum_dev[0] = window[0] - window_mean;
                
                for j in 1..size {
                    cum_dev[j] = cum_dev[j-1] + (window[j] - window_mean);
                }
                
                // Calculate range and standard deviation
                let range = cum_dev.fold(f64::NEG_INFINITY, |a, &b| a.max(b)) 
                          - cum_dev.fold(f64::INFINITY, |a, &b| a.min(b));
                let window_std = window.std(0.0);
                
                if window_std > f64::EPSILON {
                    rs_sum += range / window_std;
                }
            }
            
            if num_windows > 0 {
                rs_values.push((rs_sum / num_windows as f64).ln());
                sizes.push((size as f64).ln());
            }
            
            size *= 2;
        }
        
        if rs_values.len() < 2 {
            return Ok(0.5); // Default to random walk
        }
        
        // Linear regression to find Hurst exponent
        let mean_size = sizes.iter().sum::<f64>() / sizes.len() as f64;
        let mean_rs = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
        
        let mut num = 0.0;
        let mut den = 0.0;
        
        for i in 0..sizes.len() {
            num += (sizes[i] - mean_size) * (rs_values[i] - mean_rs);
            den += (sizes[i] - mean_size).powi(2);
        }
        
        let hurst = if den > f64::EPSILON { num / den } else { 0.5 };
        
        Ok(hurst.max(0.0).min(1.0)) // Clamp to valid range
    }
    
    /// Calculate maximum drawdown
    pub fn maximum_drawdown(data: &ArrayView1<f64>) -> Result<(f64, usize, usize), &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        let mut max_dd = 0.0;
        let mut peak_idx = 0;
        let mut trough_idx = 0;
        let mut running_max = data[0];
        let mut running_max_idx = 0;
        
        for i in 1..data.len() {
            if data[i] > running_max {
                running_max = data[i];
                running_max_idx = i;
            }
            
            let drawdown = (running_max - data[i]) / running_max;
            if drawdown > max_dd {
                max_dd = drawdown;
                peak_idx = running_max_idx;
                trough_idx = i;
            }
        }
        
        Ok((max_dd, peak_idx, trough_idx))
    }
    
    /// Calculate Value at Risk (VaR)
    pub fn value_at_risk(data: &ArrayView1<f64>, confidence_level: f64) -> Result<f64, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err("Confidence level must be between 0 and 1");
        }
        
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((1.0 - confidence_level) * sorted.len() as f64) as usize;
        let var = -sorted[index.min(sorted.len() - 1)]; // Negative for loss
        
        Ok(var)
    }
    
    /// Calculate Conditional Value at Risk (CVaR)
    pub fn conditional_var(data: &ArrayView1<f64>, confidence_level: f64) -> Result<f64, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err("Confidence level must be between 0 and 1");
        }
        
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let cutoff_index = ((1.0 - confidence_level) * sorted.len() as f64) as usize;
        
        if cutoff_index == 0 {
            return Ok(-sorted[0]); // Only one value in tail
        }
        
        let tail_mean = sorted[..cutoff_index].iter().sum::<f64>() / cutoff_index as f64;
        let cvar = -tail_mean; // Negative for loss
        
        Ok(cvar)
    }
}

/// Descriptive statistics structure
#[derive(Debug, Clone)]
pub struct DescriptiveStats {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Types of statistics for rolling calculations
#[derive(Debug, Clone, Copy)]
pub enum StatType {
    Mean,
    Std,
    Var,
    Min,
    Max,
    Median,
    Skewness,
    Kurtosis,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_descriptive_stats() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = Statistics::descriptive_stats(&data.view()).unwrap();
        
        assert_eq!(stats.count, 5);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.range, 4.0);
    }
    
    #[test]
    fn test_rolling_stats() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let rolling_mean = Statistics::rolling_stats(&data.view(), 3, StatType::Mean).unwrap();
        
        assert_eq!(rolling_mean.len(), 5);
        assert_eq!(rolling_mean[0], 1.0); // First value
        assert_eq!(rolling_mean[1], 1.5); // (1+2)/2
        assert_eq!(rolling_mean[2], 2.0); // (1+2+3)/3
    }
    
    #[test]
    fn test_autocorrelation() {
        let data = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]; // Periodic pattern
        let autocorr = Statistics::autocorrelation(&data.view(), 3).unwrap();
        
        assert_eq!(autocorr.len(), 4);
        assert_eq!(autocorr[0], 1.0); // Lag 0 is always 1
        assert!(autocorr[2] > 0.8); // Should have high correlation at lag 2
    }
    
    #[test]
    fn test_cross_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0]; // Same series
        
        let cross_corr = Statistics::cross_correlation(&x.view(), &y.view(), 2).unwrap();
        
        assert_eq!(cross_corr.len(), 5); // 2*max_lag + 1
        assert_abs_diff_eq!(cross_corr[2], 1.0, epsilon = 1e-10); // Perfect correlation at lag 0
    }
    
    #[test]
    fn test_ljung_box_test() {
        let data = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]; // Autocorrelated
        let (lb_stat, dof) = Statistics::ljung_box_test(&data.view(), 2).unwrap();
        
        assert!(lb_stat > 0.0);
        assert_eq!(dof, 2.0);
    }
    
    #[test]
    fn test_jarque_bera_test() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0]; // Uniform distribution
        let (jb_stat, dof) = Statistics::jarque_bera_test(&data.view()).unwrap();
        
        assert!(jb_stat >= 0.0);
        assert_eq!(dof, 2.0);
    }
    
    #[test]
    fn test_adf_test() {
        let data = array![1.0, 1.1, 1.2, 1.3, 1.4]; // Trending series
        let (_t_stat, is_stationary) = Statistics::adf_test(&data.view(), 1).unwrap();
        
        // Trending series should not be stationary
        assert!(!is_stationary);
    }
    
    #[test]
    fn test_kpss_test() {
        let data = array![0.1, -0.2, 0.15, -0.1, 0.05]; // Stationary-like
        let (kpss_stat, _is_stationary) = Statistics::kpss_test(&data.view()).unwrap();
        
        assert!(kpss_stat >= 0.0);
    }
    
    #[test]
    fn test_hurst_exponent() {
        let data = Array1::range(1.0, 17.0, 1.0); // Trending series
        let hurst = Statistics::hurst_exponent(&data.view()).unwrap();
        
        assert!(hurst >= 0.0 && hurst <= 1.0);
        assert!(hurst > 0.5); // Should indicate persistence for trending data
    }
    
    #[test]
    fn test_maximum_drawdown() {
        let data = array![100.0, 110.0, 105.0, 90.0, 95.0, 120.0];
        let (max_dd, peak_idx, trough_idx) = Statistics::maximum_drawdown(&data.view()).unwrap();
        
        assert!(max_dd > 0.0);
        assert_eq!(peak_idx, 1); // Peak at 110.0
        assert_eq!(trough_idx, 3); // Trough at 90.0
        assert_abs_diff_eq!(max_dd, (110.0 - 90.0) / 110.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_value_at_risk() {
        let data = array![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]; // Losses are negative
        let var_95 = Statistics::value_at_risk(&data.view(), 0.95).unwrap();
        
        // 95% VaR should be around the 5th percentile
        assert!(var_95 > 0.0); // VaR is positive (represents loss magnitude)
    }
    
    #[test]
    fn test_conditional_var() {
        let data = array![-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0]; 
        let cvar_95 = Statistics::conditional_var(&data.view(), 0.95).unwrap();
        
        // CVaR should be larger than VaR
        let var_95 = Statistics::value_at_risk(&data.view(), 0.95).unwrap();
        assert!(cvar_95 >= var_95);
    }
    
    #[test]
    fn test_partial_autocorrelation() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pacf = Statistics::partial_autocorrelation(&data.view(), 3).unwrap();
        
        assert_eq!(pacf.len(), 4);
        assert_eq!(pacf[0], 1.0); // Lag 0 is always 1
    }
}