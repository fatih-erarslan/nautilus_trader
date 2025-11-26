//! Performance monitoring

use crate::models::BetPosition;

/// Performance monitor for tracking betting results
pub struct PerformanceMonitor;

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self
    }

    /// Calculate ROI from positions
    pub fn calculate_roi(&self, positions: &[BetPosition]) -> f64 {
        if positions.is_empty() {
            return 0.0;
        }

        let total_stake: f64 = positions.iter()
            .filter_map(|p| p.stake.to_string().parse::<f64>().ok())
            .sum();

        let total_return: f64 = positions.iter()
            .filter_map(|p| p.actual_payout.and_then(|payout| payout.to_string().parse::<f64>().ok()))
            .sum();

        if total_stake == 0.0 {
            0.0
        } else {
            (total_return - total_stake) / total_stake * 100.0
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}
