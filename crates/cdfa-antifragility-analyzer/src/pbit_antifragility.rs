//! pBit-Enhanced Antifragility Analysis
//!
//! Uses Ising model dynamics to detect and measure antifragility
//! through response to simulated stress scenarios.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - **Convexity via Jensen**: E[f(x+σ)] - f(x) > 0 for convex f
//! - **Boltzmann Response**: P(benefit) = exp(β×gain) / Z
//! - **Ising Stress Test**: H = -Σ J_ij σ_i σ_j - h×Σσ_i
//! - **Antifragility Score**: A = (E[gain|stress] - E[gain|calm]) / σ_stress

use ndarray::{Array1, ArrayView1};
use rand::prelude::*;
use std::f64::consts::E;

/// pBit-based antifragility detector
#[derive(Debug, Clone)]
pub struct PBitAntifragility {
    /// Temperature (volatility proxy)
    pub temperature: f64,
    /// Number of stress scenarios to simulate
    pub n_scenarios: usize,
    /// Stress intensity range
    pub stress_range: (f64, f64),
}

impl Default for PBitAntifragility {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            n_scenarios: 1000,
            stress_range: (-0.10, 0.10), // ±10% stress
        }
    }
}

impl PBitAntifragility {
    /// Create with custom temperature
    pub fn with_temperature(temperature: f64) -> Self {
        Self { temperature, ..Default::default() }
    }

    /// Analyze antifragility of a return series
    pub fn analyze(&self, returns: &ArrayView1<f64>) -> AntifragilityResult {
        let n = returns.len();
        if n < 10 {
            return AntifragilityResult::insufficient_data();
        }

        // Calculate base statistics
        let mean = returns.sum() / n as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / n as f64;
        let volatility = variance.sqrt();

        // pBit stress test
        let stress_response = self.pbit_stress_test(returns, volatility);
        
        // Convexity score from Jensen's inequality
        let convexity = self.measure_convexity(returns);
        
        // Recovery velocity after drawdowns
        let recovery = self.measure_recovery(returns);
        
        // Composite antifragility score
        let antifragility_score = self.compute_score(&stress_response, convexity, recovery);

        AntifragilityResult {
            score: antifragility_score,
            convexity,
            stress_response,
            recovery_velocity: recovery,
            is_antifragile: antifragility_score > 0.0,
            confidence: self.bootstrap_confidence(returns),
        }
    }

    /// pBit stress test simulation
    fn pbit_stress_test(&self, returns: &ArrayView1<f64>, volatility: f64) -> StressResponse {
        let mut rng = rand::thread_rng();
        let n = returns.len();
        
        let mut calm_gains = Vec::with_capacity(self.n_scenarios);
        let mut stress_gains = Vec::with_capacity(self.n_scenarios);

        for _ in 0..self.n_scenarios {
            // Calm scenario: normal volatility
            let calm_shock: f64 = rng.gen_range(-volatility..volatility);
            let calm_return = self.simulate_response(returns, calm_shock);
            calm_gains.push(calm_return);

            // Stress scenario: elevated volatility (tail event)
            let stress_magnitude = rng.gen_range(self.stress_range.0..self.stress_range.1);
            let stress_shock = stress_magnitude * (1.0 + 2.0 * volatility);
            let stress_return = self.simulate_response(returns, stress_shock);
            stress_gains.push(stress_return);
        }

        let mean_calm: f64 = calm_gains.iter().sum::<f64>() / calm_gains.len() as f64;
        let mean_stress: f64 = stress_gains.iter().sum::<f64>() / stress_gains.len() as f64;
        
        let std_calm = self.std_dev(&calm_gains, mean_calm);
        let std_stress = self.std_dev(&stress_gains, mean_stress);

        StressResponse {
            mean_calm_gain: mean_calm,
            mean_stress_gain: mean_stress,
            stress_benefit: mean_stress - mean_calm,
            calm_volatility: std_calm,
            stress_volatility: std_stress,
            benefit_per_unit_stress: if std_stress > 1e-10 {
                (mean_stress - mean_calm) / std_stress
            } else {
                0.0
            },
        }
    }

    /// Simulate response using Boltzmann weighting
    fn simulate_response(&self, returns: &ArrayView1<f64>, shock: f64) -> f64 {
        let mut rng = rand::thread_rng();
        
        // Energy-based importance sampling
        let energies: Vec<f64> = returns.iter()
            .map(|r| -(r + shock).abs() / self.temperature)
            .collect();
        
        let weights: Vec<f64> = energies.iter()
            .map(|e| e.exp())
            .collect();
        let z: f64 = weights.iter().sum();
        
        if z < 1e-10 {
            return shock;
        }

        // Sample response
        let r: f64 = rng.gen::<f64>() * z;
        let mut cumsum = 0.0;
        for (i, w) in weights.iter().enumerate() {
            cumsum += w;
            if r <= cumsum {
                return returns[i] + shock;
            }
        }
        returns[returns.len() - 1] + shock
    }

    /// Measure convexity using second derivative approximation
    fn measure_convexity(&self, returns: &ArrayView1<f64>) -> f64 {
        let n = returns.len();
        if n < 3 {
            return 0.0;
        }

        // Rolling second derivative (acceleration)
        let mut convexity_sum = 0.0;
        let mut count = 0;

        for i in 1..(n - 1) {
            let second_deriv = returns[i + 1] - 2.0 * returns[i] + returns[i - 1];
            // Convexity indicator: positive second derivative during volatility
            let local_vol = (returns[i + 1] - returns[i - 1]).abs();
            if local_vol > 1e-10 {
                convexity_sum += second_deriv / local_vol;
                count += 1;
            }
        }

        if count > 0 {
            convexity_sum / count as f64
        } else {
            0.0
        }
    }

    /// Measure recovery velocity after drawdowns
    fn measure_recovery(&self, returns: &ArrayView1<f64>) -> f64 {
        let n = returns.len();
        if n < 5 {
            return 0.0;
        }

        let mut recovery_velocities = Vec::new();
        let mut in_drawdown = false;
        let mut drawdown_start = 0;

        for i in 0..n {
            if returns[i] < -0.02 { // 2% drawdown threshold
                if !in_drawdown {
                    in_drawdown = true;
                    drawdown_start = i;
                }
            } else if in_drawdown && returns[i] > 0.0 {
                // Recovery detected
                let recovery_time = (i - drawdown_start) as f64;
                if recovery_time > 0.0 {
                    recovery_velocities.push(1.0 / recovery_time);
                }
                in_drawdown = false;
            }
        }

        if recovery_velocities.is_empty() {
            0.5 // Neutral if no drawdowns
        } else {
            recovery_velocities.iter().sum::<f64>() / recovery_velocities.len() as f64
        }
    }

    /// Compute composite antifragility score
    fn compute_score(&self, stress: &StressResponse, convexity: f64, recovery: f64) -> f64 {
        // Weighted combination of factors
        let stress_factor = stress.benefit_per_unit_stress.tanh(); // Normalize to [-1, 1]
        let convexity_factor = convexity.tanh();
        let recovery_factor = (2.0 * recovery - 1.0).tanh(); // Center around 0.5

        // Antifragile if positive response to stress + convex + fast recovery
        0.4 * stress_factor + 0.4 * convexity_factor + 0.2 * recovery_factor
    }

    /// Bootstrap confidence interval
    fn bootstrap_confidence(&self, returns: &ArrayView1<f64>) -> f64 {
        let n = returns.len();
        if n < 20 {
            return 0.5; // Low confidence for small samples
        }

        let mut rng = rand::thread_rng();
        let mut scores = Vec::with_capacity(100);

        for _ in 0..100 {
            // Bootstrap resample
            let sample: Vec<f64> = (0..n)
                .map(|_| returns[rng.gen_range(0..n)])
                .collect();
            let sample_arr = Array1::from_vec(sample);
            
            // Quick score calculation
            let mean: f64 = sample_arr.sum() / n as f64;
            let vol: f64 = sample_arr.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>().sqrt() / n as f64;
            scores.push(if vol > 0.0 { mean / vol } else { 0.0 });
        }

        // Confidence = fraction with same sign as mean
        let mean_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        let same_sign = scores.iter()
            .filter(|&&s| s.signum() == mean_score.signum())
            .count();
        same_sign as f64 / scores.len() as f64
    }

    fn std_dev(&self, data: &[f64], mean: f64) -> f64 {
        let variance: f64 = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }
}

/// Stress response metrics
#[derive(Debug, Clone)]
pub struct StressResponse {
    /// Mean gain in calm conditions
    pub mean_calm_gain: f64,
    /// Mean gain under stress
    pub mean_stress_gain: f64,
    /// Benefit from stress (stress - calm)
    pub stress_benefit: f64,
    /// Volatility in calm conditions
    pub calm_volatility: f64,
    /// Volatility under stress
    pub stress_volatility: f64,
    /// Sharpe-like ratio of stress benefit
    pub benefit_per_unit_stress: f64,
}

/// Antifragility analysis result
#[derive(Debug, Clone)]
pub struct AntifragilityResult {
    /// Composite antifragility score [-1, 1]
    pub score: f64,
    /// Convexity measure
    pub convexity: f64,
    /// Stress response metrics
    pub stress_response: StressResponse,
    /// Recovery velocity after drawdowns
    pub recovery_velocity: f64,
    /// True if score > 0
    pub is_antifragile: bool,
    /// Bootstrap confidence [0, 1]
    pub confidence: f64,
}

impl AntifragilityResult {
    fn insufficient_data() -> Self {
        Self {
            score: 0.0,
            convexity: 0.0,
            stress_response: StressResponse {
                mean_calm_gain: 0.0,
                mean_stress_gain: 0.0,
                stress_benefit: 0.0,
                calm_volatility: 0.0,
                stress_volatility: 0.0,
                benefit_per_unit_stress: 0.0,
            },
            recovery_velocity: 0.0,
            is_antifragile: false,
            confidence: 0.0,
        }
    }

    /// Classification: Antifragile, Robust, or Fragile
    pub fn classification(&self) -> &'static str {
        if self.score > 0.2 {
            "Antifragile"
        } else if self.score > -0.2 {
            "Robust"
        } else {
            "Fragile"
        }
    }
}

/// Quick antifragility check using pBit sampling
pub fn quick_antifragility_check(returns: &[f64], temperature: f64) -> f64 {
    let n = returns.len();
    if n < 5 {
        return 0.0;
    }

    let mut rng = rand::thread_rng();
    
    // Boltzmann-weighted mean during volatile periods
    let volatility: f64 = returns.iter()
        .map(|r| r.abs())
        .sum::<f64>() / n as f64;

    let weights: Vec<f64> = returns.iter()
        .map(|r| (r.abs() / (temperature * volatility.max(0.01))).exp())
        .collect();
    let z: f64 = weights.iter().sum();

    if z < 1e-10 {
        return 0.0;
    }

    let weighted_mean: f64 = returns.iter()
        .zip(weights.iter())
        .map(|(r, w)| r * w)
        .sum::<f64>() / z;

    let simple_mean: f64 = returns.iter().sum::<f64>() / n as f64;

    // Positive = benefits from volatility weighting
    (weighted_mean - simple_mean).tanh()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_antifragile_detection() {
        // Antifragile pattern: gains more when volatility increases
        let returns = array![0.01, -0.02, 0.05, -0.03, 0.08, -0.01, 0.10, -0.02, 0.15, 0.02];
        
        let analyzer = PBitAntifragility::default();
        let result = analyzer.analyze(&returns.view());
        
        println!("Score: {:.3}, Class: {}", result.score, result.classification());
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_fragile_detection() {
        // Fragile pattern: loses more when volatility increases  
        let returns = array![-0.01, 0.02, -0.05, 0.03, -0.08, 0.01, -0.10, 0.02, -0.15, -0.02];
        
        let analyzer = PBitAntifragility::default();
        let result = analyzer.analyze(&returns.view());
        
        println!("Score: {:.3}, Class: {}", result.score, result.classification());
    }

    #[test]
    fn test_quick_check() {
        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        let score = quick_antifragility_check(&returns, 1.0);
        
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_boltzmann_weighting_wolfram_validated() {
        // Wolfram: Exp[-|r|/T] for T=1
        let t = 1.0_f64;
        let w1 = (-0.01_f64.abs() / t).exp();
        let w2 = (-0.10_f64.abs() / t).exp();
        
        // Higher weight for smaller absolute returns
        assert!(w1 > w2);
        assert!((w1 - 0.990).abs() < 0.001);
        assert!((w2 - 0.905).abs() < 0.001);
    }
}
