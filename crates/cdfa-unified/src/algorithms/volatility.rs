use ndarray::{Array1, ArrayView1};

/// Volatility clustering detection and analysis
/// 
/// Enhanced version with additional volatility models and regime detection
pub struct VolatilityClustering;

impl VolatilityClustering {
    /// GARCH(1,1) volatility estimation
    /// 
    /// Parameters:
    /// - returns: Asset returns
    /// - omega: Constant term
    /// - alpha: ARCH coefficient  
    /// - beta: GARCH coefficient
    pub fn garch_11(
        returns: &ArrayView1<f64>, 
        omega: f64, 
        alpha: f64, 
        beta: f64
    ) -> Result<Array1<f64>, &'static str> {
        if returns.is_empty() {
            return Err("Returns cannot be empty");
        }
        
        if omega <= 0.0 || alpha < 0.0 || beta < 0.0 {
            return Err("Parameters must be non-negative (omega must be positive)");
        }
        
        if alpha + beta >= 1.0 {
            return Err("Alpha + beta must be less than 1 for stationarity");
        }
        
        let n = returns.len();
        let mut volatility = Array1::zeros(n);
        
        // Initialize with unconditional variance
        let unconditional_var = omega / (1.0 - alpha - beta);
        volatility[0] = unconditional_var.sqrt();
        
        // GARCH recursion
        for t in 1..n {
            let variance = omega 
                         + alpha * returns[t-1].powi(2) 
                         + beta * volatility[t-1].powi(2);
            volatility[t] = variance.sqrt();
        }
        
        Ok(volatility)
    }
    
    /// EWMA (Exponentially Weighted Moving Average) volatility
    pub fn ewma_volatility(
        returns: &ArrayView1<f64>, 
        lambda: f64
    ) -> Result<Array1<f64>, &'static str> {
        if returns.is_empty() {
            return Err("Returns cannot be empty");
        }
        
        if lambda <= 0.0 || lambda > 1.0 {
            return Err("Lambda must be between 0 and 1");
        }
        
        let n = returns.len();
        let mut volatility = Array1::zeros(n);
        
        // Initialize with first return squared
        volatility[0] = returns[0].abs();
        
        // EWMA recursion
        for t in 1..n {
            let variance = lambda * volatility[t-1].powi(2) 
                         + (1.0 - lambda) * returns[t-1].powi(2);
            volatility[t] = variance.sqrt();
        }
        
        Ok(volatility)
    }
    
    /// Realized volatility calculation
    pub fn realized_volatility(
        returns: &ArrayView1<f64>, 
        window: usize
    ) -> Result<Array1<f64>, &'static str> {
        if returns.is_empty() {
            return Err("Returns cannot be empty");
        }
        
        if window == 0 {
            return Err("Window must be positive");
        }
        
        let n = returns.len();
        let mut volatility = Array1::zeros(n);
        
        for t in 0..n {
            let start = if t >= window { t - window + 1 } else { 0 };
            let window_returns = returns.slice(ndarray::s![start..=t]);
            
            // Calculate standard deviation
            if window_returns.len() > 1 {
                volatility[t] = window_returns.std(0.0);
            } else {
                volatility[t] = window_returns[0].abs();
            }
        }
        
        Ok(volatility)
    }
    
    /// Parkinson volatility estimator (using high-low range)
    pub fn parkinson_volatility(
        high: &ArrayView1<f64>, 
        low: &ArrayView1<f64>, 
        window: usize
    ) -> Result<Array1<f64>, &'static str> {
        if high.len() != low.len() {
            return Err("High and low arrays must have same length");
        }
        
        if high.is_empty() {
            return Err("Price arrays cannot be empty");
        }
        
        if window == 0 {
            return Err("Window must be positive");
        }
        
        let n = high.len();
        let mut volatility = Array1::zeros(n);
        let factor = 1.0 / (4.0 * 2.0_f64.ln());
        
        for t in 0..n {
            let start = if t >= window { t - window + 1 } else { 0 };
            
            let mut sum_sq_log_ratio = 0.0;
            let count = t - start + 1;
            
            for i in start..=t {
                if low[i] > 0.0 && high[i] > low[i] {
                    let log_ratio = (high[i] / low[i]).ln();
                    sum_sq_log_ratio += log_ratio * log_ratio;
                }
            }
            
            volatility[t] = (factor * sum_sq_log_ratio / count as f64).sqrt();
        }
        
        Ok(volatility)
    }
    
    /// Garman-Klass volatility estimator
    pub fn garman_klass_volatility(
        open: &ArrayView1<f64>,
        high: &ArrayView1<f64>,
        low: &ArrayView1<f64>,
        close: &ArrayView1<f64>,
        window: usize
    ) -> Result<Array1<f64>, &'static str> {
        let n = open.len();
        
        if n != high.len() || n != low.len() || n != close.len() {
            return Err("All price arrays must have same length");
        }
        
        if n == 0 {
            return Err("Price arrays cannot be empty");
        }
        
        if window == 0 {
            return Err("Window must be positive");
        }
        
        let mut volatility = Array1::zeros(n);
        
        for t in 0..n {
            let start = if t >= window { t - window + 1 } else { 0 };
            
            let mut sum_gk = 0.0;
            let count = t - start + 1;
            
            for i in start..=t {
                if high[i] > 0.0 && low[i] > 0.0 && close[i] > 0.0 && open[i] > 0.0 {
                    let hl_term = 0.5 * (high[i] / low[i]).ln().powi(2);
                    let co_term = (2.0 * 2.0_f64.ln() - 1.0) * (close[i] / open[i]).ln().powi(2);
                    sum_gk += hl_term - co_term;
                }
            }
            
            volatility[t] = (sum_gk / count as f64).max(0.0).sqrt();
        }
        
        Ok(volatility)
    }
    
    /// Rogers-Satchell volatility estimator
    pub fn rogers_satchell_volatility(
        open: &ArrayView1<f64>,
        high: &ArrayView1<f64>,
        low: &ArrayView1<f64>,
        close: &ArrayView1<f64>,
        window: usize
    ) -> Result<Array1<f64>, &'static str> {
        let n = open.len();
        
        if n != high.len() || n != low.len() || n != close.len() {
            return Err("All price arrays must have same length");
        }
        
        if n == 0 {
            return Err("Price arrays cannot be empty");
        }
        
        if window == 0 {
            return Err("Window must be positive");
        }
        
        let mut volatility = Array1::zeros(n);
        
        for t in 0..n {
            let start = if t >= window { t - window + 1 } else { 0 };
            
            let mut sum_rs = 0.0;
            let count = t - start + 1;
            
            for i in start..=t {
                if high[i] > 0.0 && low[i] > 0.0 && close[i] > 0.0 && open[i] > 0.0 {
                    let log_ho = (high[i] / open[i]).ln();
                    let log_hc = (high[i] / close[i]).ln();
                    let log_lo = (low[i] / open[i]).ln();
                    let log_lc = (low[i] / close[i]).ln();
                    
                    sum_rs += log_ho * log_hc + log_lo * log_lc;
                }
            }
            
            volatility[t] = (sum_rs / count as f64).max(0.0).sqrt();
        }
        
        Ok(volatility)
    }
    
    /// Yang-Zhang volatility estimator (combines overnight and intraday components)
    pub fn yang_zhang_volatility(
        prev_close: &ArrayView1<f64>,
        open: &ArrayView1<f64>,
        high: &ArrayView1<f64>,
        low: &ArrayView1<f64>,
        close: &ArrayView1<f64>,
        window: usize
    ) -> Result<Array1<f64>, &'static str> {
        let n = open.len();
        
        if n != high.len() || n != low.len() || n != close.len() || n != prev_close.len() {
            return Err("All price arrays must have same length");
        }
        
        if n == 0 {
            return Err("Price arrays cannot be empty");
        }
        
        if window == 0 {
            return Err("Window must be positive");
        }
        
        let mut volatility = Array1::zeros(n);
        let k = 0.34 / (1.34 + (window + 1) as f64 / (window - 1) as f64);
        
        for t in 0..n {
            let start = if t >= window { t - window + 1 } else { 0 };
            let count = t - start + 1;
            
            let mut overnight_sum = 0.0;
            let mut rs_sum = 0.0;
            let mut open_close_sum = 0.0;
            
            // Calculate mean returns for bias correction
            let mut overnight_returns = Vec::new();
            let mut open_close_returns = Vec::new();
            
            for i in start..=t {
                if i > 0 && prev_close[i] > 0.0 && open[i] > 0.0 {
                    let overnight_ret = (open[i] / prev_close[i]).ln();
                    overnight_returns.push(overnight_ret);
                }
                
                if open[i] > 0.0 && close[i] > 0.0 {
                    let open_close_ret = (close[i] / open[i]).ln();
                    open_close_returns.push(open_close_ret);
                }
            }
            
            let mean_overnight = if !overnight_returns.is_empty() {
                overnight_returns.iter().sum::<f64>() / overnight_returns.len() as f64
            } else { 0.0 };
            
            let mean_open_close = if !open_close_returns.is_empty() {
                open_close_returns.iter().sum::<f64>() / open_close_returns.len() as f64
            } else { 0.0 };
            
            for i in start..=t {
                // Overnight component
                if i > 0 && prev_close[i] > 0.0 && open[i] > 0.0 {
                    let overnight_ret = (open[i] / prev_close[i]).ln() - mean_overnight;
                    overnight_sum += overnight_ret * overnight_ret;
                }
                
                // Rogers-Satchell component
                if high[i] > 0.0 && low[i] > 0.0 && close[i] > 0.0 && open[i] > 0.0 {
                    let log_ho = (high[i] / open[i]).ln();
                    let log_hc = (high[i] / close[i]).ln();
                    let log_lo = (low[i] / open[i]).ln();
                    let log_lc = (low[i] / close[i]).ln();
                    
                    rs_sum += log_ho * log_hc + log_lo * log_lc;
                }
                
                // Open-to-close component
                if open[i] > 0.0 && close[i] > 0.0 {
                    let open_close_ret = (close[i] / open[i]).ln() - mean_open_close;
                    open_close_sum += open_close_ret * open_close_ret;
                }
            }
            
            let overnight_vol = overnight_sum / count as f64;
            let rs_vol = rs_sum / count as f64;
            let open_close_vol = open_close_sum / count as f64;
            
            let yz_vol = overnight_vol + k * open_close_vol + (1.0 - k) * rs_vol;
            volatility[t] = yz_vol.max(0.0).sqrt();
        }
        
        Ok(volatility)
    }
    
    /// Detect volatility clusters using threshold method
    pub fn detect_clusters(
        volatility: &ArrayView1<f64>, 
        threshold_multiplier: f64
    ) -> Result<Vec<(usize, usize)>, &'static str> {
        if volatility.is_empty() {
            return Err("Volatility series cannot be empty");
        }
        
        let mean_vol = volatility.mean().unwrap();
        let threshold = mean_vol * threshold_multiplier;
        
        let mut clusters = Vec::new();
        let mut in_cluster = false;
        let mut cluster_start = 0;
        
        for (i, &vol) in volatility.iter().enumerate() {
            if vol > threshold && !in_cluster {
                in_cluster = true;
                cluster_start = i;
            } else if vol <= threshold && in_cluster {
                in_cluster = false;
                clusters.push((cluster_start, i - 1));
            }
        }
        
        // Handle case where series ends in a cluster
        if in_cluster {
            clusters.push((cluster_start, volatility.len() - 1));
        }
        
        Ok(clusters)
    }
    
    /// Volatility persistence test (autocorrelation of squared returns)
    pub fn persistence_test(
        returns: &ArrayView1<f64>, 
        max_lag: usize
    ) -> Result<Array1<f64>, &'static str> {
        if returns.is_empty() {
            return Err("Returns cannot be empty");
        }
        
        if max_lag >= returns.len() {
            return Err("Max lag must be less than series length");
        }
        
        let squared_returns: Array1<f64> = returns.mapv(|r| r * r);
        let mean_sq = squared_returns.mean().unwrap();
        
        let mut autocorrelations = Array1::zeros(max_lag + 1);
        autocorrelations[0] = 1.0; // Lag 0 is always 1
        
        // Calculate variance
        let variance = squared_returns.mapv(|r| (r - mean_sq).powi(2)).sum() / returns.len() as f64;
        
        if variance < f64::EPSILON {
            return Ok(autocorrelations); // No variance, all correlations are 0
        }
        
        // Calculate autocorrelations
        for lag in 1..=max_lag {
            let mut sum = 0.0;
            let count = returns.len() - lag;
            
            for i in 0..count {
                sum += (squared_returns[i] - mean_sq) * (squared_returns[i + lag] - mean_sq);
            }
            
            autocorrelations[lag] = sum / (count as f64 * variance);
        }
        
        Ok(autocorrelations)
    }
    
    /// Regime-switching volatility model (simplified 2-state model)
    pub fn regime_switching_volatility(
        returns: &ArrayView1<f64>,
        low_vol: f64,
        high_vol: f64,
        prob_stay_low: f64,
        prob_stay_high: f64
    ) -> Result<(Array1<f64>, Array1<f64>), &'static str> {
        if returns.is_empty() {
            return Err("Returns cannot be empty");
        }
        
        if low_vol <= 0.0 || high_vol <= 0.0 {
            return Err("Volatilities must be positive");
        }
        
        if low_vol >= high_vol {
            return Err("Low volatility must be less than high volatility");
        }
        
        if prob_stay_low <= 0.0 || prob_stay_low >= 1.0 || 
           prob_stay_high <= 0.0 || prob_stay_high >= 1.0 {
            return Err("Probabilities must be between 0 and 1");
        }
        
        let n = returns.len();
        let mut volatility = Array1::zeros(n);
        let mut regime_probs = Array1::zeros(n); // Probability of being in high regime
        
        // Transition probabilities
        let prob_low_to_high = 1.0 - prob_stay_low;
        let prob_high_to_low = 1.0 - prob_stay_high;
        
        // Initialize with unconditional probability
        let uncond_prob_high = prob_low_to_high / (prob_low_to_high + prob_high_to_low);
        regime_probs[0] = uncond_prob_high;
        
        // Forward recursion
        for t in 0..n {
            // Likelihood of observation given each regime
            let likelihood_low = Self::normal_pdf(returns[t], 0.0, low_vol);
            let likelihood_high = Self::normal_pdf(returns[t], 0.0, high_vol);
            
            if t > 0 {
                // Prediction step
                let pred_prob_high = regime_probs[t-1] * prob_stay_high 
                                   + (1.0 - regime_probs[t-1]) * prob_low_to_high;
                
                // Update step (Bayes' rule)
                let numerator = likelihood_high * pred_prob_high;
                let denominator = numerator + likelihood_low * (1.0 - pred_prob_high);
                
                regime_probs[t] = if denominator > 0.0 { numerator / denominator } else { 0.5 };
            }
            
            // Weighted volatility
            volatility[t] = (1.0 - regime_probs[t]) * low_vol + regime_probs[t] * high_vol;
        }
        
        Ok((volatility, regime_probs))
    }
    
    /// Normal probability density function
    fn normal_pdf(x: f64, mean: f64, std: f64) -> f64 {
        let variance = std * std;
        let exp_term = -0.5 * ((x - mean) * (x - mean)) / variance;
        (1.0 / (std * (2.0 * std::f64::consts::PI).sqrt())) * exp_term.exp()
    }
    
    /// HAR-RV (Heterogeneous Autoregressive Realized Volatility) model
    pub fn har_rv(
        daily_rv: &ArrayView1<f64>,
        weekly_rv: &ArrayView1<f64>,
        monthly_rv: &ArrayView1<f64>
    ) -> Result<Array1<f64>, &'static str> {
        let n = daily_rv.len();
        
        if n != weekly_rv.len() || n != monthly_rv.len() {
            return Err("All RV series must have same length");
        }
        
        if n < 22 {  // Need at least a month of data
            return Err("Need at least 22 observations");
        }
        
        // Simple HAR model coefficients (can be estimated from data)
        let beta_daily = 0.35;
        let beta_weekly = 0.35;
        let beta_monthly = 0.30;
        
        let har_forecast = beta_daily * daily_rv + beta_weekly * weekly_rv + beta_monthly * monthly_rv;
        
        Ok(har_forecast)
    }
    
    /// Stochastic Volatility Model (simplified)
    pub fn stochastic_volatility(
        returns: &ArrayView1<f64>,
        mu: f64,
        phi: f64,
        sigma_v: f64
    ) -> Result<Array1<f64>, &'static str> {
        if returns.is_empty() {
            return Err("Returns cannot be empty");
        }
        
        if phi <= -1.0 || phi >= 1.0 {
            return Err("Phi must be between -1 and 1 for stationarity");
        }
        
        if sigma_v <= 0.0 {
            return Err("Sigma_v must be positive");
        }
        
        let n = returns.len();
        let mut log_volatility = Array1::zeros(n);
        let mut volatility = Array1::zeros(n);
        
        // Initialize log volatility
        log_volatility[0] = mu;
        volatility[0] = log_volatility[0].exp();
        
        // Stochastic volatility evolution (simplified - without Kalman filter)
        for t in 1..n {
            // Simple approximation: use squared returns to infer log volatility
            let implied_log_vol = if returns[t-1].abs() > 1e-10 {
                (returns[t-1].abs()).ln()
            } else {
                mu
            };
            
            // AR(1) process for log volatility
            log_volatility[t] = mu + phi * (log_volatility[t-1] - mu) 
                              + 0.1 * (implied_log_vol - log_volatility[t-1]); // Simple update
            
            volatility[t] = log_volatility[t].exp();
        }
        
        Ok(volatility)
    }
}

/// Volatility regime classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolatilityRegime {
    Low,
    Medium,
    High,
    Extreme,
}

impl VolatilityRegime {
    /// Classify volatility level based on percentiles
    pub fn classify(volatility: f64, percentiles: &[f64; 3]) -> Self {
        if volatility < percentiles[0] {
            VolatilityRegime::Low
        } else if volatility < percentiles[1] {
            VolatilityRegime::Medium
        } else if volatility < percentiles[2] {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Extreme
        }
    }
    
    /// Get percentiles from historical volatility
    pub fn compute_percentiles(historical_vol: &ArrayView1<f64>) -> [f64; 3] {
        let mut sorted = historical_vol.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted.len();
        [
            sorted[n * 25 / 100],  // 25th percentile
            sorted[n * 75 / 100],  // 75th percentile
            sorted[n * 95 / 100],  // 95th percentile
        ]
    }
    
    /// Convert regime to numeric value
    pub fn to_numeric(&self) -> f64 {
        match self {
            VolatilityRegime::Low => 1.0,
            VolatilityRegime::Medium => 2.0,
            VolatilityRegime::High => 3.0,
            VolatilityRegime::Extreme => 4.0,
        }
    }
    
    /// Create regime from numeric value
    pub fn from_numeric(value: f64) -> Option<Self> {
        match value as i32 {
            1 => Some(VolatilityRegime::Low),
            2 => Some(VolatilityRegime::Medium),
            3 => Some(VolatilityRegime::High),
            4 => Some(VolatilityRegime::Extreme),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_garch_volatility() {
        let returns = array![0.01, -0.02, 0.015, -0.01, 0.02, -0.025, 0.01, -0.005];
        let volatility = VolatilityClustering::garch_11(&returns.view(), 0.00001, 0.1, 0.85).unwrap();
        
        assert_eq!(volatility.len(), returns.len());
        assert!(volatility.iter().all(|&v| v > 0.0));
    }
    
    #[test]
    fn test_ewma_volatility() {
        let returns = array![0.01, -0.02, 0.015, -0.01, 0.02, -0.025, 0.01, -0.005];
        let volatility = VolatilityClustering::ewma_volatility(&returns.view(), 0.94).unwrap();
        
        assert_eq!(volatility.len(), returns.len());
        assert!(volatility.iter().all(|&v| v >= 0.0));
    }
    
    #[test]
    fn test_realized_volatility() {
        let returns = array![0.01, -0.02, 0.015, -0.01, 0.02, -0.025, 0.01, -0.005];
        let volatility = VolatilityClustering::realized_volatility(&returns.view(), 3).unwrap();
        
        assert_eq!(volatility.len(), returns.len());
    }
    
    #[test]
    fn test_detect_clusters() {
        let volatility = array![0.01, 0.02, 0.05, 0.06, 0.05, 0.02, 0.01, 0.02];
        let clusters = VolatilityClustering::detect_clusters(&volatility.view(), 2.0).unwrap();
        
        // Should detect the high volatility period in the middle
        assert!(!clusters.is_empty());
        assert_eq!(clusters[0], (2, 4)); // Indices 2, 3, 4 have high volatility
    }
    
    #[test]
    fn test_persistence() {
        let returns = array![0.01, -0.02, 0.015, -0.01, 0.02, -0.025, 0.01, -0.005];
        let autocorr = VolatilityClustering::persistence_test(&returns.view(), 3).unwrap();
        
        assert_eq!(autocorr.len(), 4);
        assert_eq!(autocorr[0], 1.0); // Lag 0 is always 1
    }
    
    #[test]
    fn test_regime_switching() {
        let returns = array![0.001, -0.002, 0.001, 0.05, -0.04, 0.06, 0.002, -0.001];
        let (volatility, regime_probs) = VolatilityClustering::regime_switching_volatility(
            &returns.view(), 0.01, 0.05, 0.9, 0.8
        ).unwrap();
        
        assert_eq!(volatility.len(), returns.len());
        assert_eq!(regime_probs.len(), returns.len());
        
        // During high volatility period, regime probability should be higher
        assert!(regime_probs[4] > regime_probs[0]);
    }
    
    #[test]
    fn test_parkinson_volatility() {
        let high = array![101.0, 102.5, 103.0, 102.0, 104.0, 103.5, 102.5, 101.5];
        let low = array![99.0, 100.0, 101.0, 100.5, 101.5, 101.0, 100.5, 100.0];
        
        let volatility = VolatilityClustering::parkinson_volatility(&high.view(), &low.view(), 3).unwrap();
        assert_eq!(volatility.len(), high.len());
    }
    
    #[test]
    fn test_rogers_satchell_volatility() {
        let open = array![100.0, 101.0, 102.0, 101.0, 103.0, 102.0, 101.0, 100.0];
        let high = array![101.0, 102.5, 103.0, 102.0, 104.0, 103.5, 102.5, 101.5];
        let low = array![99.0, 100.0, 101.0, 100.5, 101.5, 101.0, 100.5, 100.0];
        let close = array![100.5, 102.0, 101.5, 102.5, 102.0, 101.5, 101.0, 100.5];
        
        let volatility = VolatilityClustering::rogers_satchell_volatility(
            &open.view(), &high.view(), &low.view(), &close.view(), 3
        ).unwrap();
        assert_eq!(volatility.len(), open.len());
    }
    
    #[test]
    fn test_yang_zhang_volatility() {
        let prev_close = array![99.5, 100.5, 102.0, 101.5, 102.5, 102.0, 101.5, 101.0];
        let open = array![100.0, 101.0, 102.0, 101.0, 103.0, 102.0, 101.0, 100.0];
        let high = array![101.0, 102.5, 103.0, 102.0, 104.0, 103.5, 102.5, 101.5];
        let low = array![99.0, 100.0, 101.0, 100.5, 101.5, 101.0, 100.5, 100.0];
        let close = array![100.5, 102.0, 101.5, 102.5, 102.0, 101.5, 101.0, 100.5];
        
        let volatility = VolatilityClustering::yang_zhang_volatility(
            &prev_close.view(), &open.view(), &high.view(), &low.view(), &close.view(), 3
        ).unwrap();
        assert_eq!(volatility.len(), open.len());
    }
    
    #[test]
    fn test_stochastic_volatility() {
        let returns = array![0.01, -0.02, 0.015, -0.01, 0.02, -0.025, 0.01, -0.005];
        let volatility = VolatilityClustering::stochastic_volatility(
            &returns.view(), -2.0, 0.9, 0.1
        ).unwrap();
        
        assert_eq!(volatility.len(), returns.len());
        assert!(volatility.iter().all(|&v| v > 0.0));
    }
    
    #[test]
    fn test_volatility_regime() {
        let historical_vol = array![0.01, 0.02, 0.05, 0.03, 0.01, 0.04, 0.06, 0.02];
        let percentiles = VolatilityRegime::compute_percentiles(&historical_vol.view());
        
        let regime_low = VolatilityRegime::classify(0.015, &percentiles);
        let regime_high = VolatilityRegime::classify(0.055, &percentiles);
        
        assert_eq!(regime_low, VolatilityRegime::Low);
        assert_eq!(regime_high, VolatilityRegime::Extreme);
        
        // Test numeric conversion
        assert_eq!(regime_low.to_numeric(), 1.0);
        assert_eq!(VolatilityRegime::from_numeric(1.0), Some(VolatilityRegime::Low));
    }
    
    #[test]
    fn test_har_rv() {
        let daily_rv = array![0.01, 0.02, 0.015, 0.025, 0.018, 0.022, 0.016];
        let weekly_rv = array![0.015, 0.018, 0.017, 0.022, 0.020, 0.019, 0.018];
        let monthly_rv = array![0.020, 0.021, 0.020, 0.023, 0.021, 0.020, 0.019];
        
        let forecast = VolatilityClustering::har_rv(
            &daily_rv.view(), &weekly_rv.view(), &monthly_rv.view()
        ).unwrap();
        
        assert_eq!(forecast.len(), daily_rv.len());
        assert!(forecast.iter().all(|&v| v > 0.0));
    }
}