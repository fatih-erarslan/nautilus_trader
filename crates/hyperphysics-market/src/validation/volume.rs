//! Volume Validation and Anomaly Detection
//!
//! Validates trading volume data against statistical thresholds and
//! detects anomalies that may indicate data quality issues or market events.
//!
//! # Scientific Foundation
//!
//! Based on:
//! - Order flow analysis (Easley & O'Hara, 1987)
//! - Volume-weighted statistics
//! - Anomaly detection using z-scores and MAD (Median Absolute Deviation)
//!
//! # References
//!
//! - Easley, D., & O'Hara, M. (1987). Price, trade size, and information in securities markets
//! - Leys, C., et al. (2013). Detecting outliers: Do not use standard deviation around the mean

use crate::data::Bar;
use crate::data::tick::Tick;
use crate::error::{MarketError, MarketResult};

/// Volume validator with anomaly detection
#[derive(Debug)]
pub struct VolumeValidator {
    /// Minimum volume threshold
    min_threshold: u64,
}

impl VolumeValidator {
    /// Create new volume validator
    pub fn new(min_threshold: u64) -> Self {
        Self { min_threshold }
    }

    /// Validate bar volume
    pub fn validate_bar_volume(&self, bar: &Bar) -> MarketResult<()> {
        // Minimum volume check
        if bar.volume < self.min_threshold {
            return Err(MarketError::ValidationError(format!(
                "Volume {} below minimum threshold {}",
                bar.volume, self.min_threshold
            )));
        }

        // Zero volume check (should never happen for valid bars)
        if bar.volume == 0 {
            return Err(MarketError::ValidationError(
                "Zero volume detected".to_string()
            ));
        }

        // Check trade count consistency (if available)
        if let Some(trade_count) = bar.trade_count {
            if trade_count == 0 && bar.volume > 0 {
                return Err(MarketError::ValidationError(
                    "Zero trades but non-zero volume".to_string()
                ));
            }
        }

        Ok(())
    }

    /// Validate tick size
    pub fn validate_tick_size(&self, tick: &Tick) -> MarketResult<()> {
        // Positive size constraint
        if tick.size <= 0.0 {
            return Err(MarketError::ValidationError(format!(
                "Non-positive tick size: {}",
                tick.size
            )));
        }

        // Check for NaN or infinite
        if !tick.size.is_finite() {
            return Err(MarketError::ValidationError(
                "NaN or infinite tick size".to_string()
            ));
        }

        Ok(())
    }

    /// Detect volume anomalies using z-score
    pub fn detect_volume_anomalies(&self, volumes: &[u64], threshold: f64) -> Vec<usize> {
        if volumes.len() < 3 {
            return Vec::new();
        }

        // Calculate mean and standard deviation
        let mean = volumes.iter().map(|&v| v as f64).sum::<f64>() / volumes.len() as f64;
        let variance = volumes.iter()
            .map(|&v| {
                let diff = v as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / volumes.len() as f64;
        let std_dev = variance.sqrt();

        // Find anomalies (z-score > threshold)
        volumes.iter()
            .enumerate()
            .filter(|(_, &v)| {
                let z_score = ((v as f64 - mean) / std_dev).abs();
                z_score > threshold
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Detect volume anomalies using MAD (Median Absolute Deviation)
    /// More robust to outliers than z-score
    pub fn detect_volume_anomalies_mad(&self, volumes: &[u64], threshold: f64) -> Vec<usize> {
        if volumes.len() < 3 {
            return Vec::new();
        }

        // Calculate median
        let mut sorted = volumes.to_vec();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2] as f64;

        // Calculate MAD
        let mut deviations: Vec<f64> = volumes.iter()
            .map(|&v| (v as f64 - median).abs())
            .collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = deviations[deviations.len() / 2];

        // MAD-based z-score (modified z-score)
        let mad_z_factor = 1.4826;  // Consistency constant for normal distribution

        volumes.iter()
            .enumerate()
            .filter(|(_, &v)| {
                if mad == 0.0 {
                    return false;
                }
                let mad_z = ((v as f64 - median) / (mad_z_factor * mad)).abs();
                mad_z > threshold
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Validate volume-weighted average price (VWAP) consistency
    pub fn validate_vwap(&self, bars: &[Bar]) -> MarketResult<()> {
        for (i, bar) in bars.iter().enumerate() {
            if let Some(vwap) = bar.vwap {
                // VWAP should be within [low, high]
                if vwap < bar.low || vwap > bar.high {
                    return Err(MarketError::ValidationError(format!(
                        "Bar {}: VWAP {} outside [low {}, high {}]",
                        i, vwap, bar.low, bar.high
                    )));
                }

                // VWAP should be reasonable (closer to high if buying pressure)
                // This is a soft check, not a hard constraint
                let price_range = bar.high - bar.low;
                if price_range > 0.0 {
                    let vwap_position = (vwap - bar.low) / price_range;
                    if vwap_position < 0.0 || vwap_position > 1.0 {
                        return Err(MarketError::ValidationError(format!(
                            "Bar {}: VWAP position {} outside [0, 1]",
                            i, vwap_position
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Calculate volume-weighted return
    pub fn volume_weighted_return(&self, bars: &[Bar]) -> f64 {
        if bars.len() < 2 {
            return 0.0;
        }

        let mut total_volume = 0u64;
        let mut weighted_sum = 0.0;

        for i in 1..bars.len() {
            let log_return = (bars[i].close / bars[i-1].close).ln();
            weighted_sum += log_return * bars[i].volume as f64;
            total_volume += bars[i].volume;
        }

        if total_volume == 0 {
            0.0
        } else {
            weighted_sum / total_volume as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_valid_volume() {
        let validator = VolumeValidator::new(100);

        let bar = Bar {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            open: 100.0,
            high: 105.0,
            low: 98.0,
            close: 103.0,
            volume: 1000,
            vwap: Some(101.5),
            trade_count: Some(50),
        };

        assert!(validator.validate_bar_volume(&bar).is_ok());
    }

    #[test]
    fn test_zero_volume() {
        let validator = VolumeValidator::new(100);

        let bar = Bar {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            open: 100.0,
            high: 105.0,
            low: 98.0,
            close: 103.0,
            volume: 0,  // Invalid
            vwap: None,
            trade_count: None,
        };

        assert!(validator.validate_bar_volume(&bar).is_err());
    }

    #[test]
    fn test_volume_anomaly_detection() {
        let validator = VolumeValidator::new(100);

        // Note: z-score method is sensitive to outliers inflating std_dev
        // With [1000, 1100, 1050, 10000, 1020, 1080]:
        // mean ≈ 2542, std_dev ≈ 3416 (inflated by outlier)
        // z-score(10000) ≈ 2.18 < 3.0
        // Use threshold of 2.0 to detect the outlier with z-score method
        let volumes = vec![1000, 1100, 1050, 10000, 1020, 1080];
        let anomalies = validator.detect_volume_anomalies(&volumes, 2.0);

        assert!(anomalies.contains(&3));  // 10000 is an anomaly at threshold 2.0
    }

    #[test]
    fn test_mad_anomaly_detection() {
        let validator = VolumeValidator::new(100);

        let volumes = vec![1000, 1100, 1050, 10000, 1020, 1080];
        let anomalies = validator.detect_volume_anomalies_mad(&volumes, 3.5);

        assert!(anomalies.contains(&3));  // 10000 is an anomaly
    }
}
