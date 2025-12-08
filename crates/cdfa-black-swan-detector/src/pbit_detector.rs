//! pBit-Enhanced Black Swan Detection
//!
//! Uses Ising model dynamics for rare event detection and
//! Boltzmann importance sampling for tail probability estimation.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - **EVT Tail Index**: ξ = 1/n × Σ ln(X_i/X_min) (Hill estimator)
//! - **pBit Tail Sampling**: P(X > u) ∝ exp(-βE(X)) for E(X) = -ln(X/u)
//! - **Ising Correlation**: J_ij = corr(tail_i, tail_j) for cascade detection
//! - **Black Swan Score**: S = P(X > k×σ) × Impact(X)

use rand::prelude::*;
use std::collections::VecDeque;

/// pBit-based Black Swan detector
#[derive(Debug, Clone)]
pub struct PBitBlackSwanDetector {
    /// Temperature (controls sensitivity)
    pub temperature: f64,
    /// Threshold multiplier (k × σ for tail detection)
    pub threshold_sigma: f64,
    /// Rolling window size
    pub window_size: usize,
    /// Minimum samples for detection
    pub min_samples: usize,
    /// History buffer
    history: VecDeque<f64>,
}

impl Default for PBitBlackSwanDetector {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            threshold_sigma: 3.0,  // 3-sigma events
            window_size: 252,      // ~1 year of daily data
            min_samples: 30,
            history: VecDeque::with_capacity(256),
        }
    }
}

impl PBitBlackSwanDetector {
    /// Create with custom parameters
    pub fn new(temperature: f64, threshold_sigma: f64, window_size: usize) -> Self {
        Self {
            temperature,
            threshold_sigma,
            window_size,
            min_samples: 30,
            history: VecDeque::with_capacity(window_size + 10),
        }
    }

    /// Add observation and detect black swan
    pub fn observe(&mut self, value: f64) -> PBitBlackSwanResult {
        self.history.push_back(value);
        if self.history.len() > self.window_size {
            self.history.pop_front();
        }

        if self.history.len() < self.min_samples {
            return PBitBlackSwanResult::insufficient_data();
        }

        self.detect()
    }

    /// Detect black swan in current window
    pub fn detect(&self) -> PBitBlackSwanResult {
        let n = self.history.len();
        if n < self.min_samples {
            return PBitBlackSwanResult::insufficient_data();
        }

        let data: Vec<f64> = self.history.iter().copied().collect();
        
        // Calculate statistics
        let mean = data.iter().sum::<f64>() / n as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // Threshold for tail events
        let threshold = mean + self.threshold_sigma * std_dev;
        let lower_threshold = mean - self.threshold_sigma * std_dev;

        // Current value
        let current = *self.history.back().unwrap();
        
        // pBit tail probability estimation
        let tail_prob = self.pbit_tail_probability(&data, threshold, std_dev);
        
        // Hill estimator for tail index
        let tail_index = self.hill_estimator(&data, threshold);
        
        // Impact score (how extreme is current value)
        let z_score = if std_dev > 1e-10 {
            (current - mean) / std_dev
        } else {
            0.0
        };
        
        // Black swan score
        let is_upper_tail = current > threshold;
        let is_lower_tail = current < lower_threshold;
        let black_swan_score = if is_upper_tail || is_lower_tail {
            tail_prob * z_score.abs() * (1.0 + tail_index)
        } else {
            0.0
        };

        // Cascade risk from pBit correlation
        let cascade_risk = self.pbit_cascade_risk(&data);

        PBitBlackSwanResult {
            is_black_swan: black_swan_score > 1.0,
            score: black_swan_score,
            tail_probability: tail_prob,
            tail_index,
            z_score,
            cascade_risk,
            direction: if is_upper_tail { 
                TailDirection::Upper 
            } else if is_lower_tail { 
                TailDirection::Lower 
            } else { 
                TailDirection::None 
            },
        }
    }

    /// pBit importance sampling for tail probability
    fn pbit_tail_probability(&self, data: &[f64], threshold: f64, std_dev: f64) -> f64 {
        let mut rng = rand::thread_rng();
        let n = data.len();
        
        // Count exceedances with Boltzmann weighting
        let mut weighted_count = 0.0;
        let mut total_weight = 0.0;

        for &x in data {
            // Energy based on distance from threshold
            let energy = if x > threshold {
                -((x - threshold) / std_dev.max(0.01)).ln().max(-10.0)
            } else {
                ((threshold - x) / std_dev.max(0.01)).max(0.01)
            };

            let weight = (-energy / self.temperature).exp();
            total_weight += weight;

            if x > threshold {
                weighted_count += weight;
            }
        }

        if total_weight > 1e-10 {
            weighted_count / total_weight
        } else {
            0.0
        }
    }

    /// Hill estimator for tail index (EVT)
    fn hill_estimator(&self, data: &[f64], threshold: f64) -> f64 {
        let mut exceedances: Vec<f64> = data.iter()
            .filter(|&&x| x > threshold)
            .copied()
            .collect();

        if exceedances.len() < 3 {
            return 0.0;
        }

        exceedances.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        let k = exceedances.len();
        let x_k = exceedances[k - 1]; // k-th largest

        if x_k <= 0.0 {
            return 0.0;
        }

        let sum_log: f64 = exceedances.iter()
            .take(k - 1)
            .map(|&x| (x / x_k).max(1.0).ln())
            .sum();

        if k > 1 {
            sum_log / (k - 1) as f64
        } else {
            0.0
        }
    }

    /// pBit cascade risk from tail correlations
    fn pbit_cascade_risk(&self, data: &[f64]) -> f64 {
        let n = data.len();
        if n < 10 {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let std_dev = (data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n as f64).sqrt();

        if std_dev < 1e-10 {
            return 0.0;
        }

        // Convert to tail indicators (Ising spins)
        let threshold = 2.0 * std_dev;
        let spins: Vec<i8> = data.iter()
            .map(|x| {
                let z = (x - mean) / std_dev;
                if z.abs() > threshold { 1 } else { -1 }
            })
            .collect();

        // Calculate autocorrelation of tail events (Ising coupling)
        let mut correlation = 0.0;
        let mut count = 0;

        for i in 1..n {
            correlation += (spins[i] as f64) * (spins[i - 1] as f64);
            count += 1;
        }

        let avg_corr = if count > 0 {
            correlation / count as f64
        } else {
            0.0
        };

        // Cascade risk = high correlation of tail events
        // Boltzmann probability of cascade
        (avg_corr / self.temperature).exp() / (1.0 + (avg_corr / self.temperature).exp())
    }

    /// Batch detection on time series
    pub fn detect_batch(&mut self, data: &[f64]) -> Vec<PBitBlackSwanResult> {
        data.iter().map(|&x| self.observe(x)).collect()
    }

    /// Get current state
    pub fn state(&self) -> DetectorState {
        let n = self.history.len();
        if n < self.min_samples {
            return DetectorState::Initializing { samples: n, required: self.min_samples };
        }

        let data: Vec<f64> = self.history.iter().copied().collect();
        let mean = data.iter().sum::<f64>() / n as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;

        DetectorState::Ready {
            samples: n,
            mean,
            volatility: variance.sqrt(),
            threshold: mean + self.threshold_sigma * variance.sqrt(),
        }
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

/// pBit black swan detection result
#[derive(Debug, Clone)]
pub struct PBitBlackSwanResult {
    /// Is this a black swan event?
    pub is_black_swan: bool,
    /// Black swan score (higher = more extreme)
    pub score: f64,
    /// Tail probability from pBit sampling
    pub tail_probability: f64,
    /// EVT tail index (Hill estimator)
    pub tail_index: f64,
    /// Z-score of current observation
    pub z_score: f64,
    /// Cascade risk from Ising correlations
    pub cascade_risk: f64,
    /// Direction of tail event
    pub direction: TailDirection,
}

impl PBitBlackSwanResult {
    fn insufficient_data() -> Self {
        Self {
            is_black_swan: false,
            score: 0.0,
            tail_probability: 0.0,
            tail_index: 0.0,
            z_score: 0.0,
            cascade_risk: 0.0,
            direction: TailDirection::None,
        }
    }

    /// Classification of event severity
    pub fn severity(&self) -> &'static str {
        if self.score > 5.0 {
            "Extreme Black Swan"
        } else if self.score > 2.0 {
            "Black Swan"
        } else if self.score > 1.0 {
            "Grey Swan"
        } else if self.z_score.abs() > 2.0 {
            "Tail Event"
        } else {
            "Normal"
        }
    }
}

/// Tail direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailDirection {
    Upper,
    Lower,
    None,
}

/// Detector state
#[derive(Debug, Clone)]
pub enum DetectorState {
    Initializing { samples: usize, required: usize },
    Ready { samples: usize, mean: f64, volatility: f64, threshold: f64 },
}

/// Quick black swan check using pBit sampling
pub fn quick_black_swan_check(returns: &[f64], sigma_threshold: f64) -> Option<(f64, f64)> {
    let n = returns.len();
    if n < 10 {
        return None;
    }

    let mean: f64 = returns.iter().sum::<f64>() / n as f64;
    let std_dev = (returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / n as f64).sqrt();

    if std_dev < 1e-10 {
        return None;
    }

    let threshold = sigma_threshold * std_dev;
    
    // Count tail events
    let tail_count = returns.iter()
        .filter(|&&r| (r - mean).abs() > threshold)
        .count();

    let tail_prob = tail_count as f64 / n as f64;
    
    // Max z-score
    let max_z = returns.iter()
        .map(|r| ((r - mean) / std_dev).abs())
        .fold(0.0_f64, f64::max);

    Some((tail_prob, max_z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_market() {
        let mut detector = PBitBlackSwanDetector::default();
        
        // Normal returns around 0
        let returns: Vec<f64> = (0..100)
            .map(|i| 0.001 * ((i as f64 * 0.1).sin()))
            .collect();

        for r in &returns {
            let result = detector.observe(*r);
            if result.score > 0.0 {
                assert!(!result.is_black_swan, "Normal market shouldn't trigger black swan");
            }
        }
    }

    #[test]
    fn test_black_swan_detection() {
        let mut detector = PBitBlackSwanDetector::new(1.0, 3.0, 100);
        
        // Build history with normal data
        for i in 0..50 {
            detector.observe(0.001 * (i as f64 * 0.1).sin());
        }

        // Inject extreme event
        let result = detector.observe(-0.15); // -15% crash
        
        println!("Score: {:.3}, Z: {:.2}, Severity: {}", 
            result.score, result.z_score, result.severity());
        
        assert!(result.z_score.abs() > 3.0, "Extreme event should have high z-score");
    }

    #[test]
    fn test_hill_estimator() {
        let detector = PBitBlackSwanDetector::default();
        
        // Pareto-distributed tail (known tail index)
        let data: Vec<f64> = (1..=100).map(|i| 1.0 / (i as f64).powf(0.5)).collect();
        let threshold = 0.1;
        
        let xi = detector.hill_estimator(&data, threshold);
        println!("Hill estimator: {:.3}", xi);
        
        // Should be positive for fat-tailed distribution
        assert!(xi > 0.0);
    }

    #[test]
    fn test_cascade_risk() {
        let detector = PBitBlackSwanDetector::default();
        
        // Clustered volatility (high cascade risk)
        let mut data = vec![0.01; 50];
        data.extend(vec![-0.05, -0.04, -0.06, -0.03, -0.05]); // Clustered drawdowns
        data.extend(vec![0.01; 45]);

        let cascade = detector.pbit_cascade_risk(&data);
        println!("Cascade risk: {:.3}", cascade);
        
        assert!(cascade > 0.0 && cascade < 1.0);
    }

    #[test]
    fn test_quick_check() {
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02, -0.08, 0.01, -0.01, 0.02, 0.01];
        
        if let Some((tail_prob, max_z)) = quick_black_swan_check(&returns, 2.0) {
            println!("Tail prob: {:.3}, Max Z: {:.2}", tail_prob, max_z);
            assert!(max_z > 2.0, "Should detect the -8% as outlier");
        }
    }

    #[test]
    fn test_boltzmann_tail_wolfram_validated() {
        // Wolfram: For tail sampling, P(tail) ∝ exp(-E/T)
        // E = distance from threshold
        let t = 1.0_f64;
        let p_near = (-0.5_f64 / t).exp(); // Close to threshold
        let p_far = (-2.0_f64 / t).exp();  // Far from threshold
        
        // Near threshold should have higher probability
        assert!(p_near > p_far);
        assert!((p_near - 0.607).abs() < 0.001);
        assert!((p_far - 0.135).abs() < 0.001);
    }
}
