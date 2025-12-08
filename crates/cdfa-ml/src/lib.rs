//! Machine learning integrations for CDFA
//! 
//! This crate provides ML-based feature alignment including:
//! - Neural network-based alignment
//! - Classical ML algorithms
//! - Deep learning integration with Candle
//! - Pre-trained model support

pub mod neural;
pub mod classical;
pub mod models;
pub mod training;

pub use neural::*;
pub use classical::*;
pub use models::*;
pub use training::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_initialization() {
        // Placeholder test
        assert!(true);
    }
}