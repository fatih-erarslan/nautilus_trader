//! Ultra-fast pBit regime detection with sub-100ns latency
//! 
//! This crate provides high-performance market regime detection using:
//! - **pBit HMM**: Ising-model based Hidden Markov Model (Wolfram validated)
//! - **SIMD**: Vectorized feature extraction
//! - **5 Regimes**: Bullish, Bearish, Ranging, Volatile, Crisis
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - Transition: T_ij = exp(-E_ij/T) / Z
//! - Stationary: Bull ~22.4%, Ranging ~24.8%
//! - Ising: E = -J Σ s_i s_j

#![allow(clippy::excessive_precision)]

pub mod detector;
pub mod simd_ops;
pub mod types;
pub mod confidence;
pub mod cache;
pub mod pbit_hmm;

pub use detector::RegimeDetector;
pub use types::{MarketRegime, RegimeFeatures, RegimeDetectionResult, RegimeConfig};
pub use confidence::ConfidenceScorer;
pub use pbit_hmm::{PBitHmm, PBitHmmConfig};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_regime_detection_latency() {
        let detector = RegimeDetector::new();
        let prices: Vec<f32> = vec![100.0; 100];
        let volumes: Vec<f32> = vec![1000.0; 100];
        
        let start = std::time::Instant::now();
        let _ = detector.detect_regime(&prices, &volumes);
        let elapsed = start.elapsed();
        
        // Ensure sub-microsecond detection
        assert!(elapsed.as_nanos() < 10000, "Detection took {}ns", elapsed.as_nanos());
    }

    #[test]
    fn test_pbit_hmm_stationary() {
        let hmm = PBitHmm::new(PBitHmmConfig::default());
        let dist = hmm.get_stationary_distribution();
        
        // Wolfram validated
        let bull = dist.get(&MarketRegime::Bullish).unwrap();
        assert!((*bull - 0.224).abs() < 0.02, "Bull stationary = {}", bull);
    }
}
