//! pBit-Enhanced Advanced Detectors
//!
//! Uses Ising model and Boltzmann statistics for probabilistic pattern detection.
//!
//! ## Mathematical Foundation
//!
//! Market phases modeled as Ising spins:
//! - **Accumulation**: Ordered low-energy state (spins aligned, M → +1)
//! - **Distribution**: Ordered high-energy state (spins anti-aligned, M → -1)
//! - **Bubble**: Critical regime with high susceptibility (χ diverges)
//! - **Normal**: Disordered paramagnetic state (M ≈ 0)
//!
//! Phase transitions detected via:
//! - Magnetization changes
//! - Susceptibility spikes (χ = dM/dH)
//! - Correlation length divergence

use crate::{DetectionResult, DetectorError, MarketData, Result};
use std::collections::HashMap;

/// Critical temperature for phase transitions
pub const CRITICAL_TEMPERATURE: f32 = 2.269;

/// pBit detector configuration
#[derive(Debug, Clone)]
pub struct PBitDetectorConfig {
    /// Temperature for Boltzmann dynamics
    pub temperature: f32,
    /// Coupling strength between price movements
    pub coupling_j: f32,
    /// External field (market bias)
    pub external_h: f32,
    /// Lookback window for magnetization
    pub lookback: usize,
    /// Threshold for phase detection
    pub phase_threshold: f32,
}

impl Default for PBitDetectorConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            coupling_j: 1.0,
            external_h: 0.0,
            lookback: 20,
            phase_threshold: 0.6,
        }
    }
}

/// Market phase from Ising perspective
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IsingMarketPhase {
    /// Ferromagnetic ordering - accumulation
    Accumulation,
    /// Anti-ferromagnetic ordering - distribution
    Distribution,
    /// Critical point - bubble formation
    Critical,
    /// Paramagnetic disorder - normal trading
    Normal,
}

/// pBit-enhanced pattern detector
#[derive(Debug)]
pub struct PBitPatternDetector {
    config: PBitDetectorConfig,
    /// Spin history (+1 for up, -1 for down)
    spin_history: Vec<i8>,
    /// Magnetization history
    magnetization_history: Vec<f32>,
    /// Susceptibility history
    susceptibility_history: Vec<f32>,
}

impl PBitPatternDetector {
    /// Create new detector
    pub fn new(config: PBitDetectorConfig) -> Self {
        Self {
            config,
            spin_history: Vec::with_capacity(10000),
            magnetization_history: Vec::with_capacity(10000),
            susceptibility_history: Vec::with_capacity(10000),
        }
    }

    /// Convert price change to Ising spin
    fn price_to_spin(&self, current: f32, previous: f32) -> i8 {
        if current > previous { 1 } else { -1 }
    }

    /// Calculate magnetization over window
    fn magnetization(&self, spins: &[i8]) -> f32 {
        if spins.is_empty() {
            return 0.0;
        }
        spins.iter().map(|&s| s as f32).sum::<f32>() / spins.len() as f32
    }

    /// Calculate susceptibility (variance of magnetization)
    fn susceptibility(&self, spins: &[i8], temperature: f32) -> f32 {
        if spins.len() < 2 {
            return 0.0;
        }

        let m = self.magnetization(spins);
        let m2: f32 = spins.iter().map(|&s| (s as f32).powi(2)).sum::<f32>() / spins.len() as f32;
        
        // χ = (⟨M²⟩ - ⟨M⟩²) / T
        (m2 - m.powi(2)) / temperature.max(0.01)
    }

    /// Calculate correlation between spins at distance r
    fn correlation(&self, spins: &[i8], r: usize) -> f32 {
        if spins.len() <= r {
            return 0.0;
        }

        let n = spins.len() - r;
        let mut sum = 0.0;
        for i in 0..n {
            sum += (spins[i] * spins[i + r]) as f32;
        }
        sum / n as f32
    }

    /// Detect market phase using Ising dynamics
    pub fn detect_phase(&mut self, data: &MarketData) -> Result<PBitPhaseResult> {
        if data.len() < self.config.lookback + 1 {
            return Err(DetectorError::InsufficientData {
                required: self.config.lookback + 1,
                actual: data.len(),
            });
        }

        // Convert prices to spins
        let spins: Vec<i8> = data.prices.windows(2)
            .map(|w| self.price_to_spin(w[1], w[0]))
            .collect();

        self.spin_history.extend(spins.iter().copied());

        // Calculate magnetization over lookback window
        let recent_spins = if spins.len() > self.config.lookback {
            &spins[spins.len() - self.config.lookback..]
        } else {
            &spins
        };

        let m = self.magnetization(recent_spins);
        let chi = self.susceptibility(recent_spins, self.config.temperature);
        
        self.magnetization_history.push(m);
        self.susceptibility_history.push(chi);

        // Determine phase
        let phase = self.classify_phase(m, chi);

        // Calculate additional metrics
        let correlation_1 = self.correlation(recent_spins, 1);
        let correlation_2 = self.correlation(recent_spins, 2);
        
        // Energy per spin
        let energy = self.calculate_energy(recent_spins);

        // Order parameter (absolute magnetization)
        let order_parameter = m.abs();

        // Phase transition probability
        let transition_prob = self.transition_probability(chi);

        Ok(PBitPhaseResult {
            phase,
            magnetization: m,
            susceptibility: chi,
            order_parameter,
            energy,
            correlation_1,
            correlation_2,
            transition_probability: transition_prob,
            confidence: self.phase_confidence(m, chi, phase),
        })
    }

    /// Classify phase from magnetization and susceptibility
    fn classify_phase(&self, m: f32, chi: f32) -> IsingMarketPhase {
        let threshold = self.config.phase_threshold;

        // High susceptibility = critical point (bubble)
        if chi > 2.0 {
            IsingMarketPhase::Critical
        }
        // Strong positive magnetization = accumulation
        else if m > threshold {
            IsingMarketPhase::Accumulation
        }
        // Strong negative magnetization = distribution
        else if m < -threshold {
            IsingMarketPhase::Distribution
        }
        // Low magnetization = normal
        else {
            IsingMarketPhase::Normal
        }
    }

    /// Calculate Ising energy per spin
    fn calculate_energy(&self, spins: &[i8]) -> f32 {
        if spins.len() < 2 {
            return 0.0;
        }

        let j = self.config.coupling_j;
        let h = self.config.external_h;

        let mut energy = 0.0;

        // Nearest-neighbor interaction
        for i in 0..spins.len() - 1 {
            energy -= j * (spins[i] * spins[i + 1]) as f32;
        }

        // External field
        for &s in spins {
            energy -= h * s as f32;
        }

        energy / spins.len() as f32
    }

    /// Calculate transition probability based on susceptibility
    fn transition_probability(&self, chi: f32) -> f32 {
        // Higher susceptibility = higher transition probability
        // Using sigmoid to bound to [0, 1]
        1.0 / (1.0 + (-chi + 1.0).exp())
    }

    /// Calculate confidence in phase classification
    fn phase_confidence(&self, m: f32, chi: f32, phase: IsingMarketPhase) -> f32 {
        match phase {
            IsingMarketPhase::Accumulation | IsingMarketPhase::Distribution => {
                // Confidence from magnetization strength
                m.abs().min(1.0)
            }
            IsingMarketPhase::Critical => {
                // Confidence from susceptibility
                (chi / 3.0).min(1.0)
            }
            IsingMarketPhase::Normal => {
                // Confidence from low magnetization
                1.0 - m.abs()
            }
        }
    }

    /// Detect bubble formation using critical dynamics
    pub fn detect_bubble(&mut self, data: &MarketData) -> Result<DetectionResult> {
        let phase_result = self.detect_phase(data)?;

        let is_bubble = phase_result.phase == IsingMarketPhase::Critical 
            && phase_result.susceptibility > 2.0;

        let mut result = DetectionResult::new(
            is_bubble,
            phase_result.susceptibility / 3.0, // Normalize strength
            phase_result.confidence,
        );

        result.metadata.insert("magnetization".to_string(), phase_result.magnetization);
        result.metadata.insert("susceptibility".to_string(), phase_result.susceptibility);
        result.metadata.insert("energy".to_string(), phase_result.energy);
        result.metadata.insert("transition_prob".to_string(), phase_result.transition_probability);

        Ok(result)
    }

    /// Detect accumulation using ferromagnetic ordering
    pub fn detect_accumulation(&mut self, data: &MarketData) -> Result<DetectionResult> {
        let phase_result = self.detect_phase(data)?;

        let is_accumulation = phase_result.phase == IsingMarketPhase::Accumulation;

        let mut result = DetectionResult::new(
            is_accumulation,
            phase_result.magnetization.max(0.0), // Positive magnetization strength
            phase_result.confidence,
        );

        result.metadata.insert("magnetization".to_string(), phase_result.magnetization);
        result.metadata.insert("order_parameter".to_string(), phase_result.order_parameter);

        Ok(result)
    }

    /// Detect distribution using anti-ferromagnetic ordering
    pub fn detect_distribution(&mut self, data: &MarketData) -> Result<DetectionResult> {
        let phase_result = self.detect_phase(data)?;

        let is_distribution = phase_result.phase == IsingMarketPhase::Distribution;

        let mut result = DetectionResult::new(
            is_distribution,
            (-phase_result.magnetization).max(0.0), // Negative magnetization strength
            phase_result.confidence,
        );

        result.metadata.insert("magnetization".to_string(), phase_result.magnetization);
        result.metadata.insert("order_parameter".to_string(), phase_result.order_parameter);

        Ok(result)
    }
}

/// Result of pBit phase detection
#[derive(Debug, Clone)]
pub struct PBitPhaseResult {
    /// Detected phase
    pub phase: IsingMarketPhase,
    /// Magnetization M ∈ [-1, 1]
    pub magnetization: f32,
    /// Magnetic susceptibility χ
    pub susceptibility: f32,
    /// Order parameter |M|
    pub order_parameter: f32,
    /// Energy per spin
    pub energy: f32,
    /// Nearest-neighbor correlation
    pub correlation_1: f32,
    /// Next-nearest-neighbor correlation
    pub correlation_2: f32,
    /// Phase transition probability
    pub transition_probability: f32,
    /// Confidence in classification
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_market_data(prices: Vec<f32>) -> MarketData {
        let n = prices.len();
        MarketData {
            prices,
            volumes: vec![1000.0; n],
            timestamps: (0..n as i64).collect(),
            highs: vec![0.0; n],
            lows: vec![0.0; n],
            opens: vec![0.0; n],
        }
    }

    #[test]
    fn test_spin_conversion() {
        let config = PBitDetectorConfig::default();
        let detector = PBitPatternDetector::new(config);

        assert_eq!(detector.price_to_spin(105.0, 100.0), 1);
        assert_eq!(detector.price_to_spin(95.0, 100.0), -1);
    }

    #[test]
    fn test_magnetization() {
        let config = PBitDetectorConfig::default();
        let detector = PBitPatternDetector::new(config);

        // All up
        let spins_up = vec![1, 1, 1, 1, 1];
        assert_eq!(detector.magnetization(&spins_up), 1.0);

        // All down
        let spins_down = vec![-1, -1, -1, -1, -1];
        assert_eq!(detector.magnetization(&spins_down), -1.0);

        // Mixed
        let spins_mixed = vec![1, -1, 1, -1];
        assert_eq!(detector.magnetization(&spins_mixed), 0.0);
    }

    #[test]
    fn test_accumulation_detection() {
        let config = PBitDetectorConfig {
            lookback: 10,
            ..Default::default()
        };
        let mut detector = PBitPatternDetector::new(config);

        // Consistent upward movement (accumulation)
        let prices: Vec<f32> = (0..25).map(|i| 100.0 + i as f32).collect();
        let data = make_market_data(prices);

        let result = detector.detect_accumulation(&data).unwrap();
        assert!(result.detected);
        assert!(result.strength > 0.5);
    }

    #[test]
    fn test_distribution_detection() {
        let config = PBitDetectorConfig {
            lookback: 10,
            ..Default::default()
        };
        let mut detector = PBitPatternDetector::new(config);

        // Consistent downward movement (distribution)
        let prices: Vec<f32> = (0..25).map(|i| 200.0 - i as f32).collect();
        let data = make_market_data(prices);

        let result = detector.detect_distribution(&data).unwrap();
        assert!(result.detected);
        assert!(result.strength > 0.5);
    }

    #[test]
    fn test_critical_temperature() {
        // Onsager solution: T_c = 2/ln(1 + √2) ≈ 2.269
        assert!((CRITICAL_TEMPERATURE - 2.269).abs() < 0.001);
    }
}
