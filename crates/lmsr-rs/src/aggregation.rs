//! Probability aggregation methods for LMSR

use crate::errors::{LMSRError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Methods for aggregating probability distributions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Arithmetic mean aggregation
    Arithmetic,
    /// Geometric mean aggregation
    Geometric,
    /// Log-odds aggregation
    LogOdds,
    /// Quantum probability aggregation
    Quantum,
}

impl AggregationMethod {
    /// Get all available aggregation methods
    pub fn all() -> &'static [AggregationMethod] {
        &[
            AggregationMethod::Arithmetic,
            AggregationMethod::Geometric,
            AggregationMethod::LogOdds,
            AggregationMethod::Quantum,
        ]
    }

    /// Get method description
    pub fn description(&self) -> &'static str {
        match self {
            AggregationMethod::Arithmetic => "Simple arithmetic mean of probabilities",
            AggregationMethod::Geometric => "Geometric mean with normalization",
            AggregationMethod::LogOdds => "Log-odds aggregation with conversion back to probabilities",
            AggregationMethod::Quantum => "Quantum probability amplitude aggregation",
        }
    }

    /// Check if method is commutative
    pub fn is_commutative(&self) -> bool {
        match self {
            AggregationMethod::Arithmetic => true,
            AggregationMethod::Geometric => true,
            AggregationMethod::LogOdds => true,
            AggregationMethod::Quantum => true,
        }
    }

    /// Check if method is associative
    pub fn is_associative(&self) -> bool {
        match self {
            AggregationMethod::Arithmetic => true,
            AggregationMethod::Geometric => true,
            AggregationMethod::LogOdds => true,
            AggregationMethod::Quantum => false, // Due to normalization
        }
    }
}

impl fmt::Display for AggregationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregationMethod::Arithmetic => write!(f, "arithmetic"),
            AggregationMethod::Geometric => write!(f, "geometric"),
            AggregationMethod::LogOdds => write!(f, "log_odds"),
            AggregationMethod::Quantum => write!(f, "quantum"),
        }
    }
}

impl std::str::FromStr for AggregationMethod {
    type Err = LMSRError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "arithmetic" | "mean" => Ok(AggregationMethod::Arithmetic),
            "geometric" | "geo" => Ok(AggregationMethod::Geometric),
            "log_odds" | "logodds" | "log-odds" => Ok(AggregationMethod::LogOdds),
            "quantum" => Ok(AggregationMethod::Quantum),
            _ => Err(LMSRError::invalid_input(format!(
                "Unknown aggregation method: {}",
                s
            ))),
        }
    }
}

/// Result of probability aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    /// The aggregated probability distribution
    pub probabilities: Vec<f64>,
    /// Confidence score of the aggregation (0.0 to 1.0)
    pub confidence: f64,
    /// Method used for aggregation
    pub method: AggregationMethod,
    /// Number of distributions aggregated
    pub n_distributions: usize,
}

impl AggregationResult {
    /// Create a new aggregation result
    pub fn new(
        probabilities: Vec<f64>,
        confidence: f64,
        method: AggregationMethod,
        n_distributions: usize,
    ) -> Self {
        Self {
            probabilities,
            confidence,
            method,
            n_distributions,
        }
    }

    /// Get the entropy of the aggregated distribution
    pub fn entropy(&self) -> f64 {
        let mut entropy = 0.0;
        for &p in &self.probabilities {
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Get the maximum probability
    pub fn max_probability(&self) -> f64 {
        self.probabilities.iter().fold(0.0, |a, &b| a.max(b))
    }

    /// Get the minimum probability
    pub fn min_probability(&self) -> f64 {
        self.probabilities.iter().fold(1.0, |a, &b| a.min(b))
    }

    /// Get the index of the most likely outcome
    pub fn argmax(&self) -> usize {
        self.probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get the index of the least likely outcome
    pub fn argmin(&self) -> usize {
        self.probabilities
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Check if the distribution is uniform
    pub fn is_uniform(&self, tolerance: f64) -> bool {
        if self.probabilities.is_empty() {
            return true;
        }
        
        let expected = 1.0 / self.probabilities.len() as f64;
        self.probabilities.iter().all(|&p| (p - expected).abs() < tolerance)
    }

    /// Get the effective number of outcomes (based on entropy)
    pub fn effective_outcomes(&self) -> f64 {
        self.entropy().exp()
    }

    /// Calculate the Gini coefficient (inequality measure)
    pub fn gini_coefficient(&self) -> f64 {
        let n = self.probabilities.len();
        if n <= 1 {
            return 0.0;
        }
        
        let mut sorted = self.probabilities.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let sum: f64 = sorted.iter().sum();
        if sum == 0.0 {
            return 0.0;
        }
        
        let mut gini = 0.0;
        for (i, &p) in sorted.iter().enumerate() {
            gini += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * p;
        }
        
        gini / (n as f64 * sum)
    }

    /// Calculate the concentration ratio (sum of top k probabilities)
    pub fn concentration_ratio(&self, k: usize) -> f64 {
        let mut sorted = self.probabilities.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        sorted.iter().take(k).sum()
    }

    /// Calculate the Herfindahl-Hirschman Index (HHI)
    pub fn hhi(&self) -> f64 {
        self.probabilities.iter().map(|&p| p * p).sum()
    }

    /// Convert to summary statistics
    pub fn summary(&self) -> AggregationSummary {
        AggregationSummary {
            method: self.method,
            n_distributions: self.n_distributions,
            n_outcomes: self.probabilities.len(),
            confidence: self.confidence,
            entropy: self.entropy(),
            max_probability: self.max_probability(),
            min_probability: self.min_probability(),
            most_likely_outcome: self.argmax(),
            effective_outcomes: self.effective_outcomes(),
            gini_coefficient: self.gini_coefficient(),
            hhi: self.hhi(),
        }
    }
}

/// Summary statistics for aggregation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationSummary {
    /// Aggregation method used
    pub method: AggregationMethod,
    /// Number of distributions aggregated
    pub n_distributions: usize,
    /// Number of outcomes
    pub n_outcomes: usize,
    /// Confidence score
    pub confidence: f64,
    /// Entropy of the distribution
    pub entropy: f64,
    /// Maximum probability
    pub max_probability: f64,
    /// Minimum probability
    pub min_probability: f64,
    /// Index of most likely outcome
    pub most_likely_outcome: usize,
    /// Effective number of outcomes
    pub effective_outcomes: f64,
    /// Gini coefficient
    pub gini_coefficient: f64,
    /// Herfindahl-Hirschman Index
    pub hhi: f64,
}

impl fmt::Display for AggregationSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AggregationSummary(method={}, n_dist={}, n_out={}, conf={:.3}, ent={:.3}, max={:.3})",
            self.method, self.n_distributions, self.n_outcomes, 
            self.confidence, self.entropy, self.max_probability
        )
    }
}

/// Aggregation engine for combining multiple probability distributions
#[derive(Debug, Clone)]
pub struct AggregationEngine {
    /// Default aggregation method
    pub default_method: AggregationMethod,
    /// Confidence threshold for accepting results
    pub confidence_threshold: f64,
    /// Enable parallel processing
    pub parallel: bool,
}

impl AggregationEngine {
    /// Create a new aggregation engine
    pub fn new() -> Self {
        Self {
            default_method: AggregationMethod::LogOdds,
            confidence_threshold: 0.5,
            parallel: true,
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        default_method: AggregationMethod,
        confidence_threshold: f64,
        parallel: bool,
    ) -> Self {
        Self {
            default_method,
            confidence_threshold,
            parallel,
        }
    }

    /// Aggregate multiple distributions using the default method
    pub fn aggregate(&self, distributions: &[Vec<f64>]) -> Result<AggregationResult> {
        self.aggregate_with_method(distributions, self.default_method)
    }

    /// Aggregate with a specific method
    pub fn aggregate_with_method(
        &self,
        distributions: &[Vec<f64>],
        method: AggregationMethod,
    ) -> Result<AggregationResult> {
        if distributions.is_empty() {
            return Err(LMSRError::invalid_input("No distributions provided"));
        }

        // Validate all distributions have the same length
        let n_outcomes = distributions[0].len();
        for (i, dist) in distributions.iter().enumerate() {
            if dist.len() != n_outcomes {
                return Err(LMSRError::dimension_mismatch(n_outcomes, dist.len()));
            }
            
            // Validate probabilities
            for &p in dist {
                if p < 0.0 || p > 1.0 || p.is_nan() {
                    return Err(LMSRError::invalid_probability(p));
                }
            }
            
            // Check if distribution sums to 1.0
            let sum: f64 = dist.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(LMSRError::invalid_input(format!(
                    "Distribution {} does not sum to 1.0 (sum = {})",
                    i, sum
                )));
            }
        }

        match method {
            AggregationMethod::Arithmetic => self.arithmetic_aggregation(distributions),
            AggregationMethod::Geometric => self.geometric_aggregation(distributions),
            AggregationMethod::LogOdds => self.log_odds_aggregation(distributions),
            AggregationMethod::Quantum => self.quantum_aggregation(distributions),
        }
    }

    /// Compare multiple aggregation methods
    pub fn compare_methods(&self, distributions: &[Vec<f64>]) -> Result<Vec<AggregationResult>> {
        let mut results = Vec::new();
        
        for &method in AggregationMethod::all() {
            let result = self.aggregate_with_method(distributions, method)?;
            results.push(result);
        }
        
        Ok(results)
    }

    /// Find the best aggregation method based on confidence
    pub fn best_method(&self, distributions: &[Vec<f64>]) -> Result<AggregationResult> {
        let results = self.compare_methods(distributions)?;
        
        results
            .into_iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .ok_or_else(|| LMSRError::invalid_input("No valid aggregation results"))
    }

    /// Arithmetic mean aggregation
    fn arithmetic_aggregation(&self, distributions: &[Vec<f64>]) -> Result<AggregationResult> {
        let n_outcomes = distributions[0].len();
        let mut aggregated = vec![0.0; n_outcomes];
        
        // Calculate arithmetic mean
        for dist in distributions {
            for (i, &p) in dist.iter().enumerate() {
                aggregated[i] += p;
            }
        }
        
        let n_dists = distributions.len() as f64;
        for p in &mut aggregated {
            *p /= n_dists;
        }
        
        let confidence = self.calculate_confidence(distributions)?;
        
        Ok(AggregationResult::new(
            aggregated,
            confidence,
            AggregationMethod::Arithmetic,
            distributions.len(),
        ))
    }

    /// Geometric mean aggregation
    fn geometric_aggregation(&self, distributions: &[Vec<f64>]) -> Result<AggregationResult> {
        let n_outcomes = distributions[0].len();
        let mut aggregated = vec![1.0; n_outcomes];
        
        // Calculate geometric mean
        for dist in distributions {
            for (i, &p) in dist.iter().enumerate() {
                if p <= 0.0 {
                    // Handle zero probabilities by using small epsilon
                    aggregated[i] *= 1e-10;
                } else {
                    aggregated[i] *= p;
                }
            }
        }
        
        let n_dists = distributions.len() as f64;
        for p in &mut aggregated {
            *p = p.powf(1.0 / n_dists);
        }
        
        // Normalize to ensure sum equals 1.0
        let sum: f64 = aggregated.iter().sum();
        if sum > 0.0 {
            for p in &mut aggregated {
                *p /= sum;
            }
        }
        
        let confidence = self.calculate_confidence(distributions)?;
        
        Ok(AggregationResult::new(
            aggregated,
            confidence,
            AggregationMethod::Geometric,
            distributions.len(),
        ))
    }

    /// Log-odds aggregation
    fn log_odds_aggregation(&self, distributions: &[Vec<f64>]) -> Result<AggregationResult> {
        let n_outcomes = distributions[0].len();
        let mut aggregated_log_odds = vec![0.0; n_outcomes];
        
        // Convert to log-odds and sum
        for dist in distributions {
            for (i, &p) in dist.iter().enumerate() {
                let log_odds = if p <= 0.0 {
                    -10.0 // Very negative log-odds for zero probability
                } else if p >= 1.0 {
                    10.0 // Very positive log-odds for probability 1
                } else {
                    (p / (1.0 - p)).ln()
                };
                aggregated_log_odds[i] += log_odds;
            }
        }
        
        // Average the log-odds
        let n_dists = distributions.len() as f64;
        for lo in &mut aggregated_log_odds {
            *lo /= n_dists;
        }
        
        // Convert back to probabilities
        let mut aggregated = Vec::with_capacity(n_outcomes);
        for &lo in &aggregated_log_odds {
            let p = if lo > 10.0 {
                1.0 - 1e-10
            } else if lo < -10.0 {
                1e-10
            } else {
                let exp_lo = lo.exp();
                exp_lo / (1.0 + exp_lo)
            };
            aggregated.push(p);
        }
        
        // Normalize to ensure sum equals 1.0
        let sum: f64 = aggregated.iter().sum();
        if sum > 0.0 {
            for p in &mut aggregated {
                *p /= sum;
            }
        }
        
        let confidence = self.calculate_confidence(distributions)?;
        
        Ok(AggregationResult::new(
            aggregated,
            confidence,
            AggregationMethod::LogOdds,
            distributions.len(),
        ))
    }

    /// Quantum aggregation (amplitude-based)
    fn quantum_aggregation(&self, distributions: &[Vec<f64>]) -> Result<AggregationResult> {
        let n_outcomes = distributions[0].len();
        let mut aggregated = vec![0.0; n_outcomes];
        
        // Convert probabilities to amplitudes and sum
        for dist in distributions {
            for (i, &p) in dist.iter().enumerate() {
                aggregated[i] += p.sqrt();
            }
        }
        
        // Average amplitudes and square to get probabilities
        let n_dists = distributions.len() as f64;
        for p in &mut aggregated {
            *p = (*p / n_dists).powi(2);
        }
        
        // Normalize to ensure sum equals 1.0
        let sum: f64 = aggregated.iter().sum();
        if sum > 0.0 {
            for p in &mut aggregated {
                *p /= sum;
            }
        }
        
        let confidence = self.calculate_confidence(distributions)?;
        
        Ok(AggregationResult::new(
            aggregated,
            confidence,
            AggregationMethod::Quantum,
            distributions.len(),
        ))
    }

    /// Calculate confidence score for aggregation
    fn calculate_confidence(&self, distributions: &[Vec<f64>]) -> Result<f64> {
        if distributions.len() < 2 {
            return Ok(1.0);
        }
        
        let n_outcomes = distributions[0].len();
        let mut total_kl = 0.0;
        let mut count = 0;
        
        // Calculate average pairwise KL divergence
        for i in 0..distributions.len() {
            for j in (i + 1)..distributions.len() {
                let kl = self.kl_divergence(&distributions[i], &distributions[j])?;
                total_kl += kl;
                count += 1;
            }
        }
        
        let avg_kl = total_kl / count as f64;
        
        // Convert to confidence score (lower KL = higher confidence)
        let confidence = 1.0 / (1.0 + avg_kl);
        
        Ok(confidence)
    }

    /// Calculate KL divergence between two distributions
    fn kl_divergence(&self, p: &[f64], q: &[f64]) -> Result<f64> {
        let mut kl = 0.0;
        
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi > 0.0 && qi > 0.0 {
                kl += pi * (pi / qi).ln();
            } else if pi > 0.0 && qi == 0.0 {
                return Ok(f64::INFINITY);
            }
        }
        
        Ok(kl)
    }
}

impl Default for AggregationEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_aggregation_method_display() {
        assert_eq!(AggregationMethod::Arithmetic.to_string(), "arithmetic");
        assert_eq!(AggregationMethod::Geometric.to_string(), "geometric");
        assert_eq!(AggregationMethod::LogOdds.to_string(), "log_odds");
        assert_eq!(AggregationMethod::Quantum.to_string(), "quantum");
    }

    #[test]
    fn test_aggregation_method_from_str() {
        assert_eq!("arithmetic".parse::<AggregationMethod>().unwrap(), AggregationMethod::Arithmetic);
        assert_eq!("geometric".parse::<AggregationMethod>().unwrap(), AggregationMethod::Geometric);
        assert_eq!("log_odds".parse::<AggregationMethod>().unwrap(), AggregationMethod::LogOdds);
        assert_eq!("quantum".parse::<AggregationMethod>().unwrap(), AggregationMethod::Quantum);
    }

    #[test]
    fn test_aggregation_result_methods() {
        let result = AggregationResult::new(
            vec![0.5, 0.3, 0.2],
            0.8,
            AggregationMethod::Arithmetic,
            3,
        );
        
        assert_eq!(result.argmax(), 0);
        assert_eq!(result.argmin(), 2);
        assert_eq!(result.max_probability(), 0.5);
        assert_eq!(result.min_probability(), 0.2);
        assert!(!result.is_uniform(0.1));
        assert!(result.entropy() > 0.0);
    }

    #[test]
    fn test_aggregation_engine_arithmetic() {
        let engine = AggregationEngine::new();
        let distributions = vec![
            vec![0.4, 0.3, 0.3],
            vec![0.6, 0.2, 0.2],
            vec![0.5, 0.3, 0.2],
        ];
        
        let result = engine.arithmetic_aggregation(&distributions).unwrap();
        
        // Should be average of inputs
        assert_relative_eq!(result.probabilities[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(result.probabilities[1], 0.266666666666667, epsilon = 1e-10);
        assert_relative_eq!(result.probabilities[2], 0.233333333333333, epsilon = 1e-10);
        
        // Should sum to 1.0
        let sum: f64 = result.probabilities.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_aggregation_engine_geometric() {
        let engine = AggregationEngine::new();
        let distributions = vec![
            vec![0.4, 0.3, 0.3],
            vec![0.6, 0.2, 0.2],
            vec![0.5, 0.3, 0.2],
        ];
        
        let result = engine.geometric_aggregation(&distributions).unwrap();
        
        // Should sum to 1.0
        let sum: f64 = result.probabilities.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // All probabilities should be positive
        for p in &result.probabilities {
            assert!(*p > 0.0);
        }
    }

    #[test]
    fn test_aggregation_engine_log_odds() {
        let engine = AggregationEngine::new();
        let distributions = vec![
            vec![0.4, 0.3, 0.3],
            vec![0.6, 0.2, 0.2],
            vec![0.5, 0.3, 0.2],
        ];
        
        let result = engine.log_odds_aggregation(&distributions).unwrap();
        
        // Should sum to 1.0
        let sum: f64 = result.probabilities.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // All probabilities should be positive
        for p in &result.probabilities {
            assert!(*p > 0.0);
        }
    }

    #[test]
    fn test_aggregation_engine_quantum() {
        let engine = AggregationEngine::new();
        let distributions = vec![
            vec![0.4, 0.3, 0.3],
            vec![0.6, 0.2, 0.2],
            vec![0.5, 0.3, 0.2],
        ];
        
        let result = engine.quantum_aggregation(&distributions).unwrap();
        
        // Should sum to 1.0
        let sum: f64 = result.probabilities.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // All probabilities should be positive
        for p in &result.probabilities {
            assert!(*p > 0.0);
        }
    }

    #[test]
    fn test_aggregation_engine_compare_methods() {
        let engine = AggregationEngine::new();
        let distributions = vec![
            vec![0.4, 0.3, 0.3],
            vec![0.6, 0.2, 0.2],
            vec![0.5, 0.3, 0.2],
        ];
        
        let results = engine.compare_methods(&distributions).unwrap();
        assert_eq!(results.len(), 4);
        
        // All should sum to 1.0
        for result in &results {
            let sum: f64 = result.probabilities.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_aggregation_engine_best_method() {
        let engine = AggregationEngine::new();
        let distributions = vec![
            vec![0.4, 0.3, 0.3],
            vec![0.6, 0.2, 0.2],
            vec![0.5, 0.3, 0.2],
        ];
        
        let result = engine.best_method(&distributions).unwrap();
        
        // Should have valid probabilities
        let sum: f64 = result.probabilities.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // Should have reasonable confidence
        assert!(result.confidence > 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn test_aggregation_summary() {
        let result = AggregationResult::new(
            vec![0.5, 0.3, 0.2],
            0.8,
            AggregationMethod::Arithmetic,
            3,
        );
        
        let summary = result.summary();
        assert_eq!(summary.method, AggregationMethod::Arithmetic);
        assert_eq!(summary.n_distributions, 3);
        assert_eq!(summary.n_outcomes, 3);
        assert_eq!(summary.confidence, 0.8);
        assert_eq!(summary.most_likely_outcome, 0);
        assert!(summary.entropy > 0.0);
        assert!(summary.gini_coefficient >= 0.0);
        assert!(summary.hhi > 0.0);
    }
}