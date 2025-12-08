//! SOC regime classification and detection

use crate::{SocParameters, SocRegime};
use rayon::prelude::*;

/// Regime classifier for SOC states
pub struct RegimeClassifier {
    params: SocParameters,
}

impl RegimeClassifier {
    /// Create a new regime classifier
    pub fn new(params: SocParameters) -> Self {
        Self { params }
    }
    
    /// Classify SOC regime based on equilibrium, fragility, and complexity
    pub fn classify(
        &self,
        equilibrium: &[f32],
        fragility: &[f32],
        complexity: &[f32],
    ) -> Vec<SocRegime> {
        if equilibrium.len() != fragility.len() || equilibrium.len() != complexity.len() {
            // Return default regime for mismatched lengths
            return vec![SocRegime::Normal; equilibrium.len()];
        }
        
        let n = equilibrium.len();
        let mut regimes = Vec::with_capacity(n);
        
        // Process in parallel for large datasets
        if n > 1000 && cfg!(feature = "parallel") {
            regimes = (0..n)
                .into_par_iter()
                .map(|i| self.classify_point(equilibrium[i], fragility[i], complexity[i]))
                .collect();
        } else {
            // Sequential processing
            for i in 0..n {
                regimes.push(self.classify_point(equilibrium[i], fragility[i], complexity[i]));
            }
        }
        
        regimes
    }
    
    /// Classify a single point
    fn classify_point(&self, eq: f32, frag: f32, comp: f32) -> SocRegime {
        // Critical regime: high equilibrium, high fragility, high complexity
        if eq > self.params.critical_threshold_equilibrium
            && frag > self.params.critical_threshold_fragility
            && comp > self.params.critical_threshold_complexity
        {
            return SocRegime::Critical;
        }
        
        // Near-critical: close to critical thresholds
        if eq > self.params.critical_threshold_equilibrium * 0.9
            && frag > self.params.critical_threshold_fragility * 0.9
            && comp > self.params.critical_threshold_complexity * 0.9
        {
            return SocRegime::NearCritical;
        }
        
        // Unstable: low equilibrium, high fragility
        if eq < self.params.unstable_threshold_equilibrium
            && frag > self.params.unstable_threshold_fragility
        {
            return SocRegime::Unstable;
        }
        
        // Stable: high equilibrium, low fragility
        if eq > self.params.stable_threshold_equilibrium
            && frag < self.params.stable_threshold_fragility
        {
            return SocRegime::Stable;
        }
        
        // Default: Normal
        SocRegime::Normal
    }
    
    /// Detect regime transitions
    pub fn detect_transitions(&self, regimes: &[SocRegime]) -> Vec<RegimeTransition> {
        let mut transitions = Vec::new();
        
        if regimes.len() < 2 {
            return transitions;
        }
        
        let mut current_regime = regimes[0];
        let mut regime_start = 0;
        
        for (i, &regime) in regimes.iter().enumerate().skip(1) {
            if regime != current_regime {
                // Transition detected
                transitions.push(RegimeTransition {
                    from_regime: current_regime,
                    to_regime: regime,
                    transition_idx: i,
                    from_duration: i - regime_start,
                });
                
                current_regime = regime;
                regime_start = i;
            }
        }
        
        transitions
    }
    
    /// Calculate regime stability (how long system stays in each regime)
    pub fn calculate_regime_stability(&self, regimes: &[SocRegime]) -> RegimeStability {
        let mut regime_durations = std::collections::HashMap::new();
        let mut current_regime = None;
        let mut current_duration = 0;
        
        for &regime in regimes {
            if current_regime == Some(regime) {
                current_duration += 1;
            } else {
                if let Some(prev_regime) = current_regime {
                    regime_durations
                        .entry(prev_regime)
                        .or_insert_with(Vec::new)
                        .push(current_duration);
                }
                current_regime = Some(regime);
                current_duration = 1;
            }
        }
        
        // Don't forget the last regime
        if let Some(prev_regime) = current_regime {
            regime_durations
                .entry(prev_regime)
                .or_insert_with(Vec::new)
                .push(current_duration);
        }
        
        // Calculate statistics for each regime
        let mut regime_stats = std::collections::HashMap::new();
        
        for (regime, durations) in &regime_durations {
            let mean_duration = durations.iter().sum::<usize>() as f32 / durations.len() as f32;
            let max_duration = *durations.iter().max().unwrap_or(&0);
            let total_time = durations.iter().sum::<usize>();
            
            regime_stats.insert(
                *regime,
                RegimeDurationStats {
                    mean_duration,
                    max_duration,
                    total_time,
                    occurrence_count: durations.len(),
                },
            );
        }
        
        RegimeStability {
            regime_stats,
            total_transitions: regime_durations.values().map(|v| v.len()).sum::<usize>() - 1,
        }
    }
    
    /// Calculate critical slowing down indicators
    pub fn calculate_critical_slowing_down(
        &self,
        data: &[f32],
        window: usize,
    ) -> Vec<CriticalSlowingDown> {
        let n = data.len();
        if n < window * 2 {
            return vec![CriticalSlowingDown::default(); n];
        }
        
        let mut indicators = vec![CriticalSlowingDown::default(); n];
        
        for i in window..n {
            let window_data = &data[(i - window)..i];
            
            // Calculate variance
            let mean = window_data.iter().sum::<f32>() / window as f32;
            let variance = window_data
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / window as f32;
            
            // Calculate lag-1 autocorrelation
            let autocorr = calculate_autocorrelation(window_data, 1);
            
            // Calculate skewness
            let skewness = if variance > 0.0 {
                let std_dev = variance.sqrt();
                window_data
                    .iter()
                    .map(|&x| ((x - mean) / std_dev).powi(3))
                    .sum::<f32>()
                    / window as f32
            } else {
                0.0
            };
            
            // Calculate kurtosis
            let kurtosis = if variance > 0.0 {
                let std_dev = variance.sqrt();
                window_data
                    .iter()
                    .map(|&x| ((x - mean) / std_dev).powi(4))
                    .sum::<f32>()
                    / window as f32
                    - 3.0 // Excess kurtosis
            } else {
                0.0
            };
            
            indicators[i] = CriticalSlowingDown {
                variance,
                autocorrelation: autocorr,
                skewness,
                kurtosis,
                detrended_fluctuation: calculate_dfa(window_data),
            };
        }
        
        indicators
    }
}

/// Regime transition information
#[derive(Debug, Clone)]
pub struct RegimeTransition {
    pub from_regime: SocRegime,
    pub to_regime: SocRegime,
    pub transition_idx: usize,
    pub from_duration: usize,
}

/// Regime stability analysis
#[derive(Debug, Clone)]
pub struct RegimeStability {
    pub regime_stats: std::collections::HashMap<SocRegime, RegimeDurationStats>,
    pub total_transitions: usize,
}

/// Duration statistics for a regime
#[derive(Debug, Clone)]
pub struct RegimeDurationStats {
    pub mean_duration: f32,
    pub max_duration: usize,
    pub total_time: usize,
    pub occurrence_count: usize,
}

/// Critical slowing down indicators
#[derive(Debug, Clone, Default)]
pub struct CriticalSlowingDown {
    pub variance: f32,
    pub autocorrelation: f32,
    pub skewness: f32,
    pub kurtosis: f32,
    pub detrended_fluctuation: f32,
}

/// Calculate autocorrelation at given lag
fn calculate_autocorrelation(data: &[f32], lag: usize) -> f32 {
    let n = data.len();
    if n <= lag {
        return 0.0;
    }
    
    let mean = data.iter().sum::<f32>() / n as f32;
    
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..(n - lag) {
        numerator += (data[i] - mean) * (data[i + lag] - mean);
    }
    
    for &x in data {
        denominator += (x - mean).powi(2);
    }
    
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Calculate detrended fluctuation analysis (DFA)
fn calculate_dfa(data: &[f32]) -> f32 {
    let n = data.len();
    if n < 10 {
        return 0.5;
    }
    
    // Integrate the series
    let mean = data.iter().sum::<f32>() / n as f32;
    let mut integrated = vec![0.0f32; n];
    integrated[0] = data[0] - mean;
    
    for i in 1..n {
        integrated[i] = integrated[i - 1] + data[i] - mean;
    }
    
    // Calculate fluctuation for different box sizes
    let min_box = 4;
    let max_box = n / 4;
    
    if max_box <= min_box {
        return 0.5;
    }
    
    let mut log_n = Vec::new();
    let mut log_f = Vec::new();
    
    for box_size in min_box..=max_box {
        let n_boxes = n / box_size;
        let mut fluctuation = 0.0;
        
        for i in 0..n_boxes {
            let start = i * box_size;
            let end = start + box_size;
            let segment = &integrated[start..end];
            
            // Fit linear trend
            let x_mean = (box_size - 1) as f32 / 2.0;
            let mut sum_xy = 0.0;
            let mut sum_xx = 0.0;
            
            for (j, &y) in segment.iter().enumerate() {
                let x = j as f32;
                sum_xy += (x - x_mean) * y;
                sum_xx += (x - x_mean).powi(2);
            }
            
            let slope = if sum_xx > 0.0 { sum_xy / sum_xx } else { 0.0 };
            let intercept = segment.iter().sum::<f32>() / box_size as f32;
            
            // Calculate detrended fluctuation
            for (j, &y) in segment.iter().enumerate() {
                let trend = intercept + slope * (j as f32 - x_mean);
                fluctuation += (y - trend).powi(2);
            }
        }
        
        fluctuation = (fluctuation / (n_boxes * box_size) as f32).sqrt();
        
        if fluctuation > 0.0 {
            log_n.push((box_size as f32).ln());
            log_f.push(fluctuation.ln());
        }
    }
    
    // Fit log-log relationship
    if log_n.len() < 2 {
        return 0.5;
    }
    
    let n_points = log_n.len() as f32;
    let sum_x: f32 = log_n.iter().sum();
    let sum_y: f32 = log_f.iter().sum();
    let sum_xy: f32 = log_n.iter().zip(&log_f).map(|(x, y)| x * y).sum();
    let sum_xx: f32 = log_n.iter().map(|x| x * x).sum();
    
    let dfa_exponent = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_xx - sum_x * sum_x);
    
    // Normalize to [0, 1] range (DFA exponent typically ranges from 0.5 to 1.5)
    ((dfa_exponent - 0.5) / 1.0).min(1.0).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_regime_classification() {
        let params = SocParameters::default();
        let classifier = RegimeClassifier::new(params);
        
        let equilibrium = vec![0.8, 0.3, 0.7, 0.2, 0.9];
        let fragility = vec![0.7, 0.8, 0.2, 0.9, 0.8];
        let complexity = vec![0.8, 0.5, 0.6, 0.4, 0.9];
        
        let regimes = classifier.classify(&equilibrium, &fragility, &complexity);
        
        assert_eq!(regimes.len(), 5);
        assert_eq!(regimes[0], SocRegime::Critical);
        assert_eq!(regimes[1], SocRegime::Unstable);
        assert_eq!(regimes[2], SocRegime::Stable);
    }
    
    #[test]
    fn test_regime_transitions() {
        let params = SocParameters::default();
        let classifier = RegimeClassifier::new(params);
        
        let regimes = vec![
            SocRegime::Normal,
            SocRegime::Normal,
            SocRegime::Critical,
            SocRegime::Critical,
            SocRegime::Unstable,
        ];
        
        let transitions = classifier.detect_transitions(&regimes);
        
        assert_eq!(transitions.len(), 2);
        assert_eq!(transitions[0].from_regime, SocRegime::Normal);
        assert_eq!(transitions[0].to_regime, SocRegime::Critical);
        assert_eq!(transitions[0].transition_idx, 2);
    }
    
    #[test]
    fn test_autocorrelation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let autocorr = calculate_autocorrelation(&data, 1);
        
        // Should be positive for this pattern
        assert!(autocorr > 0.0);
        assert!(autocorr <= 1.0);
    }
}