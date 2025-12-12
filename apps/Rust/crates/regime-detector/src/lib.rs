//! Ultra-fast regime detection with sub-100ns latency using SIMD optimization
//! 
//! This crate provides high-performance market regime detection for HFT systems,
//! identifying 5 key market regimes with confidence scoring.

#![allow(clippy::excessive_precision)]

pub mod detector;
pub mod simd_ops;
pub mod types;
pub mod confidence;
pub mod cache;

pub use detector::RegimeDetector;
pub use types::{MarketRegime, RegimeFeatures, RegimeDetectionResult};
pub use confidence::ConfidenceScorer;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_regime_detection_latency() {
        let detector = RegimeDetector::new();
        let prices = vec![100.0; 100];
        let volumes = vec![1000.0; 100];
        
        let start = std::time::Instant::now();
        let _ = detector.detect_regime(&prices, &volumes);
        let elapsed = start.elapsed();
        
        // Ensure sub-microsecond detection
        assert!(elapsed.as_nanos() < 1000, "Detection took {}ns, expected < 1000ns", elapsed.as_nanos());
    }
}
