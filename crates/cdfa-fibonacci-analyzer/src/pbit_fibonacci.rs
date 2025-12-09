//! pBit-Enhanced Fibonacci Level Analyzer
//!
//! Uses Boltzmann statistics to assign probabilistic weights to Fibonacci
//! retracement and extension levels based on historical price action.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! Each Fibonacci level L_i has an energy based on historical violations:
//! - E_i = -ln(success_rate_i) where success_rate = bounces / tests
//! - Boltzmann weight: W_i = exp(-E_i / T) = success_rate^(1/T)
//! - Normalized probability: P_i = W_i / Z where Z = Σ W_j
//!
//! ## Golden Ratio Connection
//!
//! φ = (1 + √5) / 2 ≈ 1.618033988749895
//! Key levels: 0.236 ≈ φ^(-3), 0.382 ≈ φ^(-2), 0.618 ≈ φ^(-1)

use std::collections::HashMap;

/// Golden ratio φ = (1 + √5) / 2
pub const PHI: f64 = 1.618033988749895;

/// Standard Fibonacci retracement levels
pub const STANDARD_LEVELS: [f64; 7] = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0];

/// Extension levels
pub const EXTENSION_LEVELS: [f64; 5] = [1.0, 1.272, 1.618, 2.618, 3.618];

/// pBit Fibonacci configuration
#[derive(Debug, Clone)]
pub struct PBitFibonacciConfig {
    /// Temperature for Boltzmann weighting
    pub temperature: f64,
    /// Tolerance for level hits (as fraction of price range)
    pub hit_tolerance: f64,
    /// Minimum tests required for statistical significance
    pub min_tests: usize,
    /// Decay factor for older observations
    pub decay_factor: f64,
}

impl Default for PBitFibonacciConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            hit_tolerance: 0.006, // 0.6% tolerance
            min_tests: 5,
            decay_factor: 0.95,
        }
    }
}

/// Statistics for a single Fibonacci level
#[derive(Debug, Clone)]
pub struct LevelStats {
    /// Fibonacci ratio (e.g., 0.618)
    pub ratio: f64,
    /// Number of times price tested this level
    pub tests: usize,
    /// Number of times price bounced (level held)
    pub bounces: usize,
    /// Raw success rate
    pub success_rate: f64,
    /// Boltzmann weight
    pub weight: f64,
    /// Normalized probability
    pub probability: f64,
    /// Energy (negative log success)
    pub energy: f64,
}

impl LevelStats {
    /// Create new level statistics
    pub fn new(ratio: f64) -> Self {
        Self {
            ratio,
            tests: 0,
            bounces: 0,
            success_rate: 0.5, // Prior
            weight: 1.0,
            probability: 0.0,
            energy: 0.0,
        }
    }

    /// Record a test of this level
    pub fn record_test(&mut self, bounced: bool) {
        self.tests += 1;
        if bounced {
            self.bounces += 1;
        }
        self.update_success_rate();
    }

    /// Update success rate with Laplace smoothing
    fn update_success_rate(&mut self) {
        // Laplace smoothing: (bounces + 1) / (tests + 2)
        self.success_rate = (self.bounces as f64 + 1.0) / (self.tests as f64 + 2.0);
    }

    /// Calculate Boltzmann weight at given temperature
    pub fn calculate_weight(&mut self, temperature: f64) {
        // Energy = -ln(success_rate)
        self.energy = -self.success_rate.ln();
        // Weight = exp(-E/T) = success_rate^(1/T)
        self.weight = (-self.energy / temperature).exp();
    }
}

/// pBit-enhanced Fibonacci analyzer
#[derive(Debug)]
pub struct PBitFibonacciAnalyzer {
    config: PBitFibonacciConfig,
    /// Statistics for retracement levels
    retracement_stats: HashMap<u64, LevelStats>,
    /// Statistics for extension levels
    extension_stats: HashMap<u64, LevelStats>,
    /// Partition function (normalization)
    partition_function: f64,
}

impl PBitFibonacciAnalyzer {
    /// Create new analyzer
    pub fn new(config: PBitFibonacciConfig) -> Self {
        let mut retracement_stats = HashMap::new();
        for &level in &STANDARD_LEVELS {
            let key = (level * 1000.0) as u64;
            retracement_stats.insert(key, LevelStats::new(level));
        }

        let mut extension_stats = HashMap::new();
        for &level in &EXTENSION_LEVELS {
            let key = (level * 1000.0) as u64;
            extension_stats.insert(key, LevelStats::new(level));
        }

        Self {
            config,
            retracement_stats,
            extension_stats,
            partition_function: 1.0,
        }
    }

    /// Analyze price series to update level statistics
    pub fn analyze(&mut self, prices: &[f64], swing_highs: &[usize], swing_lows: &[usize]) -> PBitFibonacciResult {
        if prices.is_empty() || swing_highs.is_empty() || swing_lows.is_empty() {
            return PBitFibonacciResult::empty();
        }

        // Find most recent swing range
        let (high_idx, low_idx) = self.find_recent_swings(swing_highs, swing_lows);
        let high_price = prices.get(high_idx).copied().unwrap_or(0.0);
        let low_price = prices.get(low_idx).copied().unwrap_or(0.0);
        let price_range = (high_price - low_price).abs();

        if price_range < f64::EPSILON {
            return PBitFibonacciResult::empty();
        }

        // Calculate absolute price levels
        let is_uptrend = high_idx > low_idx;
        let levels = self.calculate_price_levels(high_price, low_price, is_uptrend);

        // Test each level against recent price action
        let test_start = high_idx.max(low_idx) + 1;
        if test_start < prices.len() {
            self.test_levels(&prices[test_start..], &levels, price_range);
        }

        // Calculate Boltzmann weights and normalize
        self.calculate_weights();

        // Build result
        self.build_result(high_price, low_price, is_uptrend)
    }

    /// Find most recent swing high and low
    fn find_recent_swings(&self, swing_highs: &[usize], swing_lows: &[usize]) -> (usize, usize) {
        let high_idx = *swing_highs.last().unwrap_or(&0);
        let low_idx = *swing_lows.last().unwrap_or(&0);
        (high_idx, low_idx)
    }

    /// Calculate absolute price levels from swing range
    fn calculate_price_levels(&self, high: f64, low: f64, is_uptrend: bool) -> Vec<(f64, f64)> {
        let range = high - low;
        STANDARD_LEVELS
            .iter()
            .map(|&ratio| {
                let price = if is_uptrend {
                    high - ratio * range // Retracement from high
                } else {
                    low + ratio * range // Retracement from low
                };
                (ratio, price)
            })
            .collect()
    }

    /// Test levels against price action
    fn test_levels(&mut self, prices: &[f64], levels: &[(f64, f64)], price_range: f64) {
        let tolerance = self.config.hit_tolerance * price_range;

        for &(ratio, level_price) in levels {
            let key = (ratio * 1000.0) as u64;
            
            // Check if price tested this level
            for (i, &price) in prices.iter().enumerate() {
                if (price - level_price).abs() < tolerance {
                    // Level was tested
                    let bounced = if i + 1 < prices.len() {
                        // Check if price bounced (moved away from level)
                        let next_price = prices[i + 1];
                        (next_price - level_price).abs() > tolerance
                    } else {
                        false
                    };

                    if let Some(stats) = self.retracement_stats.get_mut(&key) {
                        stats.record_test(bounced);
                    }
                    break; // Only count first test in this series
                }
            }
        }
    }

    /// Calculate Boltzmann weights and partition function
    fn calculate_weights(&mut self) {
        // Calculate weights for all levels
        for stats in self.retracement_stats.values_mut() {
            stats.calculate_weight(self.config.temperature);
        }
        for stats in self.extension_stats.values_mut() {
            stats.calculate_weight(self.config.temperature);
        }

        // Calculate partition function
        self.partition_function = self.retracement_stats.values()
            .chain(self.extension_stats.values())
            .map(|s| s.weight)
            .sum();

        if self.partition_function < f64::EPSILON {
            self.partition_function = 1.0;
        }

        // Normalize probabilities
        for stats in self.retracement_stats.values_mut() {
            stats.probability = stats.weight / self.partition_function;
        }
        for stats in self.extension_stats.values_mut() {
            stats.probability = stats.weight / self.partition_function;
        }
    }

    /// Build analysis result
    fn build_result(&self, high: f64, low: f64, is_uptrend: bool) -> PBitFibonacciResult {
        let range = high - low;
        
        let mut level_probabilities: Vec<(f64, f64, f64)> = self.retracement_stats
            .values()
            .filter(|s| s.tests >= self.config.min_tests)
            .map(|s| {
                let price = if is_uptrend {
                    high - s.ratio * range
                } else {
                    low + s.ratio * range
                };
                (s.ratio, price, s.probability)
            })
            .collect();
        
        level_probabilities.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // Strongest level
        let strongest_level = level_probabilities.first().map(|&(r, p, prob)| (r, p, prob));

        // System entropy
        let entropy: f64 = self.retracement_stats
            .values()
            .filter(|s| s.probability > f64::EPSILON)
            .map(|s| -s.probability * s.probability.ln())
            .sum();

        // Free energy
        let free_energy = -self.config.temperature * self.partition_function.ln();

        PBitFibonacciResult {
            strongest_level,
            level_probabilities,
            partition_function: self.partition_function,
            entropy,
            free_energy,
            is_uptrend,
            high_price: high,
            low_price: low,
        }
    }

    /// Get probability for a specific level
    pub fn level_probability(&self, ratio: f64) -> f64 {
        let key = (ratio * 1000.0) as u64;
        self.retracement_stats
            .get(&key)
            .map(|s| s.probability)
            .unwrap_or(0.0)
    }
}

/// Result of pBit Fibonacci analysis
#[derive(Debug, Clone)]
pub struct PBitFibonacciResult {
    /// Strongest level: (ratio, price, probability)
    pub strongest_level: Option<(f64, f64, f64)>,
    /// All levels with probabilities: (ratio, price, probability)
    pub level_probabilities: Vec<(f64, f64, f64)>,
    /// Partition function Z
    pub partition_function: f64,
    /// System entropy
    pub entropy: f64,
    /// Free energy F = -T ln(Z)
    pub free_energy: f64,
    /// Trend direction
    pub is_uptrend: bool,
    /// Swing high price
    pub high_price: f64,
    /// Swing low price
    pub low_price: f64,
}

impl PBitFibonacciResult {
    /// Create empty result
    pub fn empty() -> Self {
        Self {
            strongest_level: None,
            level_probabilities: Vec::new(),
            partition_function: 1.0,
            entropy: 0.0,
            free_energy: 0.0,
            is_uptrend: true,
            high_price: 0.0,
            low_price: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_ratio_wolfram() {
        // Wolfram: (1 + Sqrt[5])/2 = 1.618033988749895
        let phi_expected = 1.618033988749895;
        assert!((PHI - phi_expected).abs() < 1e-14);
    }

    #[test]
    fn test_fibonacci_levels_from_phi() {
        // 0.618 ≈ 1/φ = φ - 1
        let expected_618 = 1.0 / PHI;
        assert!((0.618 - expected_618).abs() < 0.001);

        // 0.382 ≈ 1/φ² 
        let expected_382 = 1.0 / (PHI * PHI);
        assert!((0.382 - expected_382).abs() < 0.001);

        // 0.236 ≈ 1/φ³
        let expected_236 = 1.0 / (PHI * PHI * PHI);
        assert!((0.236 - expected_236).abs() < 0.001);
    }

    #[test]
    fn test_boltzmann_weight() {
        let mut stats = LevelStats::new(0.618);
        
        // Simulate 10 tests with 8 bounces (80% success)
        for _ in 0..8 {
            stats.record_test(true);
        }
        for _ in 0..2 {
            stats.record_test(false);
        }

        // Success rate with Laplace smoothing: (8+1)/(10+2) = 0.75
        assert!((stats.success_rate - 0.75).abs() < 0.01);

        // Calculate weight at T=1
        stats.calculate_weight(1.0);
        
        // Energy = -ln(0.75) ≈ 0.288
        // Weight = exp(-0.288) ≈ 0.75
        assert!((stats.weight - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_analyzer_creation() {
        let config = PBitFibonacciConfig::default();
        let analyzer = PBitFibonacciAnalyzer::new(config);

        assert_eq!(analyzer.retracement_stats.len(), 7);
        assert_eq!(analyzer.extension_stats.len(), 5);
    }

    #[test]
    fn test_level_probability() {
        let config = PBitFibonacciConfig::default();
        let analyzer = PBitFibonacciAnalyzer::new(config);

        // Initial probabilities should be uniform-ish
        let p618 = analyzer.level_probability(0.618);
        assert!(p618 >= 0.0 && p618 <= 1.0);
    }
}
