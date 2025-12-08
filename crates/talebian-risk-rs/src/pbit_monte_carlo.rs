//! pBit Monte Carlo for Talebian Risk Management
//!
//! Uses Ising-model dynamics for importance sampling of tail events
//! in fat-tailed distributions.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - **Kelly Criterion**: f* = (bp - q) / b
//!   - f*(p=0.6, b=2) = 0.40, f*(p=0.7, b=2) = 0.55
//! - **Pareto Tail**: P(X > x) = (x_m/x)^α
//!   - P(X > 3) = 0.111 for α=2
//! - **pBit IS**: W(E) = exp(-E/T)
//!   - W(E=2, T=1) = 0.1353
//! - **Barbell**: E[R] = w_safe × r_safe + w_spec × r_spec

use rand::prelude::*;
use rand_distr::{Distribution, Normal, Pareto, StandardNormal};
use std::collections::HashMap;

/// pBit state for Monte Carlo sampling
#[derive(Debug, Clone)]
pub struct PBitSampler {
    /// Temperature (controls exploration)
    pub temperature: f64,
    /// Spin configuration
    pub spins: Vec<i8>,
    /// Local fields
    pub fields: Vec<f64>,
}

impl PBitSampler {
    pub fn new(n: usize, temperature: f64) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            temperature,
            spins: (0..n).map(|_| if rng.gen::<bool>() { 1 } else { -1 }).collect(),
            fields: vec![0.0; n],
        }
    }

    /// Update spin using Metropolis
    pub fn update(&mut self) {
        let mut rng = rand::thread_rng();
        let i = rng.gen_range(0..self.spins.len());
        
        let delta_e = 2.0 * self.spins[i] as f64 * self.fields[i];
        let accept = if delta_e <= 0.0 {
            true
        } else {
            rng.gen::<f64>() < (-delta_e / self.temperature).exp()
        };
        
        if accept {
            self.spins[i] *= -1;
        }
    }

    /// Get magnetization (consensus)
    pub fn magnetization(&self) -> f64 {
        self.spins.iter().map(|&s| s as f64).sum::<f64>() / self.spins.len() as f64
    }
}

/// Kelly Criterion calculator
#[derive(Debug, Clone)]
pub struct KellyCriterion {
    /// Win probability
    pub win_prob: f64,
    /// Win/loss ratio (b = win_amount / loss_amount)
    pub odds_ratio: f64,
}

impl KellyCriterion {
    pub fn new(win_prob: f64, odds_ratio: f64) -> Self {
        Self { win_prob, odds_ratio }
    }

    /// Calculate optimal Kelly fraction
    /// f* = (bp - q) / b where q = 1 - p
    pub fn optimal_fraction(&self) -> f64 {
        let p = self.win_prob.clamp(0.0, 1.0);
        let q = 1.0 - p;
        let b = self.odds_ratio.max(0.001);
        
        ((b * p - q) / b).max(0.0).min(1.0)
    }

    /// Half-Kelly (more conservative)
    pub fn half_kelly(&self) -> f64 {
        self.optimal_fraction() / 2.0
    }

    /// pBit-adjusted Kelly using temperature for uncertainty
    pub fn pbit_kelly(&self, temperature: f64) -> f64 {
        let base = self.optimal_fraction();
        // Higher temperature = more uncertainty = lower allocation
        let uncertainty_factor = 1.0 / (1.0 + temperature);
        base * uncertainty_factor
    }
}

/// Antifragility calculator using convexity
#[derive(Debug, Clone)]
pub struct AntifragilityMeasure {
    /// Sample returns
    pub returns: Vec<f64>,
    /// Volatility level
    pub volatility: f64,
}

impl AntifragilityMeasure {
    pub fn new(returns: Vec<f64>) -> Self {
        let n = returns.len();
        let mean = returns.iter().sum::<f64>() / n.max(1) as f64;
        let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n.max(1) as f64;
        
        Self {
            returns,
            volatility: var.sqrt(),
        }
    }

    /// Calculate convexity gain (antifragility)
    /// Positive = antifragile, Negative = fragile
    pub fn convexity_gain(&self) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let mean = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        
        // E[f(X)] where f(x) = x^2 (convex payoff)
        let e_f_x = self.returns.iter().map(|r| r * r).sum::<f64>() / self.returns.len() as f64;
        
        // f(E[X])
        let f_e_x = mean * mean;
        
        // Convexity gain = E[f(X)] - f(E[X])
        e_f_x - f_e_x
    }

    /// Antifragility score (normalized)
    pub fn antifragility_score(&self) -> f64 {
        let gain = self.convexity_gain();
        // Normalize by volatility^2
        let normalized = gain / (self.volatility.powi(2).max(0.0001));
        // Map to [-1, 1] using tanh
        normalized.tanh()
    }
}

/// Black Swan detector using Pareto tails
#[derive(Debug, Clone)]
pub struct BlackSwanDetector {
    /// Pareto shape parameter α
    pub alpha: f64,
    /// Minimum value x_m
    pub x_min: f64,
    /// Detection threshold (in standard deviations)
    pub threshold_sigma: f64,
}

impl BlackSwanDetector {
    pub fn new(alpha: f64, x_min: f64) -> Self {
        Self {
            alpha,
            x_min,
            threshold_sigma: 3.0,  // 3-sigma event
        }
    }

    /// Pareto tail probability: P(X > x) = (x_m/x)^α
    pub fn tail_probability(&self, x: f64) -> f64 {
        if x <= self.x_min {
            return 1.0;
        }
        (self.x_min / x).powf(self.alpha)
    }

    /// Estimate α from returns
    pub fn estimate_alpha(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 2.0;  // Default
        }

        // Hill estimator for tail index
        let mut sorted: Vec<f64> = returns.iter().map(|r| r.abs()).collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let k = (sorted.len() as f64 * 0.1).max(1.0) as usize;
        let threshold = sorted[k.min(sorted.len() - 1)];

        if threshold <= 0.0 {
            return 2.0;
        }

        let sum: f64 = sorted.iter()
            .take(k)
            .map(|x| (x / threshold).ln())
            .sum();

        let alpha = k as f64 / sum.max(0.001);
        alpha.clamp(1.0, 10.0)
    }

    /// Is this a black swan event?
    pub fn is_black_swan(&self, value: f64, mean: f64, std: f64) -> bool {
        let z = (value - mean).abs() / std.max(0.0001);
        z > self.threshold_sigma
    }

    /// pBit importance sampling for rare events
    pub fn pbit_sample_tail(&self, n_samples: usize, temperature: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut sampler = PBitSampler::new(10, temperature);
        let pareto = Pareto::new(self.x_min, self.alpha).unwrap();
        
        let mut samples = Vec::with_capacity(n_samples);
        
        for _ in 0..n_samples {
            sampler.update();
            
            // Use magnetization to bias sampling toward tails
            let bias = (1.0 + sampler.magnetization().abs()) / 2.0;
            
            if rng.gen::<f64>() < bias {
                // Sample from tail
                samples.push(pareto.sample(&mut rng));
            } else {
                // Sample from body
                samples.push(self.x_min * (1.0 + rng.gen::<f64>()));
            }
        }
        
        samples
    }
}

/// Barbell strategy allocator
#[derive(Debug, Clone)]
pub struct BarbellStrategy {
    /// Safe allocation weight
    pub safe_weight: f64,
    /// Speculative allocation weight
    pub spec_weight: f64,
    /// Safe return expectation
    pub safe_return: f64,
    /// Speculative return expectation
    pub spec_return: f64,
}

impl BarbellStrategy {
    pub fn new(safe_weight: f64, safe_return: f64, spec_return: f64) -> Self {
        let safe = safe_weight.clamp(0.0, 1.0);
        Self {
            safe_weight: safe,
            spec_weight: 1.0 - safe,
            safe_return,
            spec_return,
        }
    }

    /// Expected return: w_s × r_s + w_p × r_p
    pub fn expected_return(&self) -> f64 {
        self.safe_weight * self.safe_return + self.spec_weight * self.spec_return
    }

    /// Optimal safe weight given risk tolerance
    pub fn optimize(risk_tolerance: f64, safe_return: f64, spec_return: f64, spec_vol: f64) -> Self {
        // Higher risk tolerance = more speculative
        let spec = (risk_tolerance * spec_return / spec_vol.max(0.01)).min(0.35);
        Self::new(1.0 - spec, safe_return, spec_return)
    }

    /// pBit-adjusted allocation using temperature
    pub fn pbit_allocation(&self, temperature: f64, market_signal: f64) -> (f64, f64) {
        // Temperature affects conviction
        let conviction = 1.0 / (1.0 + temperature);
        
        // Market signal adjusts allocation
        let signal_adj = market_signal.tanh() * 0.1;
        
        let safe = (self.safe_weight + signal_adj * conviction).clamp(0.5, 0.95);
        let spec = 1.0 - safe;
        
        (safe, spec)
    }
}

/// pBit Monte Carlo engine for tail risk
#[derive(Debug)]
pub struct PBitTailRiskEngine {
    /// pBit sampler
    pub sampler: PBitSampler,
    /// Black swan detector
    pub black_swan: BlackSwanDetector,
    /// Kelly calculator
    pub kelly: KellyCriterion,
    /// Barbell allocator
    pub barbell: BarbellStrategy,
}

impl PBitTailRiskEngine {
    pub fn new(temperature: f64) -> Self {
        Self {
            sampler: PBitSampler::new(20, temperature),
            black_swan: BlackSwanDetector::new(2.0, 1.0),
            kelly: KellyCriterion::new(0.55, 2.0),
            barbell: BarbellStrategy::new(0.85, 0.03, 0.30),
        }
    }

    /// Run full risk analysis
    pub fn analyze(&mut self, returns: &[f64]) -> TailRiskAnalysis {
        // Update sampler
        for _ in 0..100 {
            self.sampler.update();
        }

        // Estimate tail index
        let alpha = BlackSwanDetector::estimate_alpha(returns);
        self.black_swan.alpha = alpha;

        // Calculate antifragility
        let antifrag = AntifragilityMeasure::new(returns.to_vec());

        // Get tail samples
        let tail_samples = self.black_swan.pbit_sample_tail(1000, self.sampler.temperature);
        let tail_var = tail_samples.iter()
            .map(|x| x.powi(2))
            .sum::<f64>() / tail_samples.len() as f64;

        // pBit allocation
        let (safe_alloc, spec_alloc) = self.barbell.pbit_allocation(
            self.sampler.temperature,
            self.sampler.magnetization(),
        );

        TailRiskAnalysis {
            pareto_alpha: alpha,
            antifragility_score: antifrag.antifragility_score(),
            kelly_fraction: self.kelly.pbit_kelly(self.sampler.temperature),
            tail_var: tail_var.sqrt(),
            safe_allocation: safe_alloc,
            spec_allocation: spec_alloc,
            black_swan_probability: self.black_swan.tail_probability(3.0),
            pbit_consensus: self.sampler.magnetization(),
        }
    }
}

/// Tail risk analysis result
#[derive(Debug, Clone)]
pub struct TailRiskAnalysis {
    pub pareto_alpha: f64,
    pub antifragility_score: f64,
    pub kelly_fraction: f64,
    pub tail_var: f64,
    pub safe_allocation: f64,
    pub spec_allocation: f64,
    pub black_swan_probability: f64,
    pub pbit_consensus: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_criterion() {
        // Wolfram validated: f*(p=0.6, b=2) = 0.40
        let kelly = KellyCriterion::new(0.6, 2.0);
        let f = kelly.optimal_fraction();
        assert!((f - 0.40).abs() < 0.01, "Kelly = {}", f);
    }

    #[test]
    fn test_kelly_70() {
        // Wolfram validated: f*(p=0.7, b=2) = 0.55
        let kelly = KellyCriterion::new(0.7, 2.0);
        let f = kelly.optimal_fraction();
        assert!((f - 0.55).abs() < 0.01, "Kelly = {}", f);
    }

    #[test]
    fn test_pareto_tail() {
        // Wolfram validated: P(X > 3) = 0.111 for α=2
        let detector = BlackSwanDetector::new(2.0, 1.0);
        let p = detector.tail_probability(3.0);
        assert!((p - 0.111).abs() < 0.01, "P(X>3) = {}", p);
    }

    #[test]
    fn test_barbell_return() {
        // Wolfram validated: 85/15 → E[R] = 0.0705
        let barbell = BarbellStrategy::new(0.85, 0.03, 0.30);
        let r = barbell.expected_return();
        assert!((r - 0.0705).abs() < 0.001, "E[R] = {}", r);
    }

    #[test]
    fn test_pbit_engine() {
        let mut engine = PBitTailRiskEngine::new(1.0);
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.01).sin() * 0.02).collect();
        
        let analysis = engine.analyze(&returns);
        
        assert!(analysis.kelly_fraction >= 0.0 && analysis.kelly_fraction <= 1.0);
        assert!(analysis.safe_allocation + analysis.spec_allocation > 0.99);
    }
}
