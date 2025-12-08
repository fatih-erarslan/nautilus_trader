//! Utility functions for the data pipeline

use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Time utilities
pub mod time {
    use super::*;

    /// Get current timestamp in milliseconds
    pub fn current_timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_millis() as u64
    }

    /// Convert timestamp to DateTime
    pub fn timestamp_to_datetime(timestamp_ms: u64) -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::from_timestamp((timestamp_ms / 1000) as i64, 0)
            .unwrap_or_else(chrono::Utc::now)
    }

    /// Format duration for display
    pub fn format_duration(duration: Duration) -> String {
        if duration.as_millis() < 1000 {
            format!("{:.2}ms", duration.as_millis() as f64)
        } else if duration.as_secs() < 60 {
            format!("{:.2}s", duration.as_secs_f64())
        } else {
            format!("{:.1}m", duration.as_secs_f64() / 60.0)
        }
    }
}

/// Mathematical utilities
pub mod math {
    use super::*;

    /// Calculate moving average
    pub fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return vec![];
        }

        let mut result = Vec::new();
        for i in window..=data.len() {
            let sum: f64 = data[i-window..i].iter().sum();
            result.push(sum / window as f64);
        }
        result
    }

    /// Calculate standard deviation
    pub fn standard_deviation(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }

    /// Normalize data to [0, 1] range
    pub fn normalize(data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return vec![];
        }

        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max - min).abs() < f64::EPSILON {
            return vec![0.5; data.len()];
        }

        data.iter()
            .map(|&x| (x - min) / (max - min))
            .collect()
    }

    /// Calculate z-score
    pub fn z_score(value: f64, mean: f64, std_dev: f64) -> f64 {
        if std_dev == 0.0 {
            return 0.0;
        }
        (value - mean) / std_dev
    }
}

/// Data validation utilities
pub mod validation {
    use super::*;

    /// Check if a value is within a valid range
    pub fn is_in_range(value: f64, min: f64, max: f64) -> bool {
        value >= min && value <= max
    }

    /// Check if a value is a valid price
    pub fn is_valid_price(price: f64) -> bool {
        price > 0.0 && price.is_finite()
    }

    /// Check if a value is a valid volume
    pub fn is_valid_volume(volume: f64) -> bool {
        volume >= 0.0 && volume.is_finite()
    }

    /// Check if timestamp is within reasonable bounds
    pub fn is_valid_timestamp(timestamp: chrono::DateTime<chrono::Utc>) -> bool {
        let now = chrono::Utc::now();
        let one_year_ago = now - chrono::Duration::days(365);
        let one_hour_future = now + chrono::Duration::hours(1);
        
        timestamp >= one_year_ago && timestamp <= one_hour_future
    }
}

/// String utilities
pub mod string {
    use super::*;

    /// Clean text for processing
    pub fn clean_text(text: &str) -> String {
        text.trim()
            .replace('\n', " ")
            .replace('\r', " ")
            .replace('\t', " ")
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
    }

    /// Extract numbers from text
    pub fn extract_numbers(text: &str) -> Vec<f64> {
        let number_regex = regex::Regex::new(r"-?\d+\.?\d*").unwrap();
        number_regex
            .find_iter(text)
            .filter_map(|m| m.as_str().parse::<f64>().ok())
            .collect()
    }

    /// Truncate text to maximum length
    pub fn truncate_text(text: &str, max_length: usize) -> String {
        if text.len() <= max_length {
            text.to_string()
        } else {
            format!("{}...", &text[..max_length-3])
        }
    }
}

/// Memory utilities
pub mod memory {
    use super::*;

    /// Get current memory usage in bytes
    pub fn get_memory_usage() -> u64 {
        // Placeholder implementation
        1024 * 1024 * 100 // 100MB
    }

    /// Format bytes for display
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

/// Configuration utilities
pub mod config {
    use super::*;

    /// Load configuration from file
    pub fn load_config<T>(path: &str) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let content = std::fs::read_to_string(path)?;
        let config: T = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save_config<T>(config: &T, path: &str) -> Result<()>
    where
        T: Serialize,
    {
        let content = toml::to_string_pretty(config)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Merge configurations with priority
    pub fn merge_configs<T>(base: T, override_config: T) -> T
    where
        T: Serialize + for<'de> Deserialize<'de>,
    {
        // Simple implementation - in practice, would do deep merge
        override_config
    }
}

/// Hashing utilities
pub mod hash {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Calculate hash of a value
    pub fn calculate_hash<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Generate deterministic ID from data
    pub fn generate_id(symbol: &str, timestamp: u64) -> String {
        format!("{}_{}", symbol, timestamp)
    }

    /// Calculate checksum of data
    pub fn calculate_checksum(data: &[u8]) -> String {
        format!("{:x}", md5::compute(data))
    }
}

/// Performance utilities
pub mod perf {
    use super::*;
    use std::time::Instant;

    /// Simple timer for measuring execution time
    pub struct Timer {
        start: Instant,
        name: String,
    }

    impl Timer {
        pub fn new(name: &str) -> Self {
            Self {
                start: Instant::now(),
                name: name.to_string(),
            }
        }

        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }

        pub fn elapsed_ms(&self) -> f64 {
            self.start.elapsed().as_millis() as f64
        }
    }

    impl Drop for Timer {
        fn drop(&mut self) {
            tracing::debug!("{} took {:?}", self.name, self.elapsed());
        }
    }

    /// Measure execution time of a function
    pub fn measure_time<F, R>(name: &str, f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        tracing::debug!("{} took {:?}", name, duration);
        (result, duration)
    }
}

/// Error utilities
pub mod error {
    use super::*;
    use crate::error::DataPipelineError;

    /// Convert any error to pipeline error with context
    pub fn with_context<E: std::error::Error + Send + Sync + 'static>(
        error: E,
        context: &str,
    ) -> DataPipelineError {
        DataPipelineError::Unknown(format!("{}: {}", context, error))
    }

    /// Chain errors for better error reporting
    pub fn chain_errors(errors: Vec<DataPipelineError>) -> DataPipelineError {
        if errors.is_empty() {
            return DataPipelineError::Unknown("No errors".to_string());
        }

        if errors.len() == 1 {
            return errors.into_iter().next().unwrap();
        }

        let error_messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        DataPipelineError::Unknown(format!("Multiple errors: {}", error_messages.join("; ")))
    }

    /// Extract root cause from error chain
    pub fn extract_root_cause(error: &DataPipelineError) -> String {
        // Simple implementation - in practice, would traverse error chain
        error.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = math::moving_average(&data, 3);
        assert_eq!(ma, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_standard_deviation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std_dev = math::standard_deviation(&data);
        assert!((std_dev - 1.4142135623730951).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let data = vec![0.0, 5.0, 10.0];
        let normalized = math::normalize(&data);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_validation() {
        assert!(validation::is_valid_price(10.5));
        assert!(!validation::is_valid_price(-1.0));
        assert!(!validation::is_valid_price(f64::NAN));
        
        assert!(validation::is_valid_volume(0.0));
        assert!(validation::is_valid_volume(1000.0));
        assert!(!validation::is_valid_volume(-1.0));
    }

    #[test]
    fn test_clean_text() {
        let text = "  Hello\n\tWorld  \r\n  ";
        let cleaned = string::clean_text(text);
        assert_eq!(cleaned, "Hello World");
    }

    #[test]
    fn test_extract_numbers() {
        let text = "Price: $123.45, Volume: 1000, Change: -2.5%";
        let numbers = string::extract_numbers(text);
        assert_eq!(numbers, vec![123.45, 1000.0, -2.5]);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(memory::format_bytes(1024), "1.00 KB");
        assert_eq!(memory::format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(memory::format_bytes(1536), "1.50 KB");
    }

    #[test]
    fn test_timer() {
        let timer = perf::Timer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10.0);
    }
}