//! Utility functions and helpers for Tengri trading strategy
//! 
//! Provides common utilities for mathematical calculations, data processing,
//! logging, configuration helpers, and other shared functionality.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc, TimeZone, Timelike};
use serde::{Deserialize, Serialize};
use tokio::fs;
use uuid::Uuid;

use crate::{Result, TengriError};

/// Mathematical utility functions
pub mod math {
    use super::*;
    
    /// Calculate percentage change between two values
    pub fn percentage_change(old_value: f64, new_value: f64) -> f64 {
        if old_value == 0.0 {
            return 0.0;
        }
        ((new_value - old_value) / old_value) * 100.0
    }
    
    /// Calculate compound annual growth rate (CAGR)
    pub fn cagr(initial_value: f64, final_value: f64, years: f64) -> f64 {
        if initial_value <= 0.0 || final_value <= 0.0 || years <= 0.0 {
            return 0.0;
        }
        ((final_value / initial_value).powf(1.0 / years) - 1.0) * 100.0
    }
    
    /// Calculate Sharpe ratio
    pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let excess_return = mean_return - risk_free_rate;
        
        if returns.len() < 2 {
            return 0.0;
        }
        
        let variance = returns.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        excess_return / std_dev
    }
    
    /// Calculate Sortino ratio (downside deviation only)
    pub fn sortino_ratio(returns: &[f64], target_return: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let excess_return = mean_return - target_return;
        
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < target_return)
            .map(|&r| r - target_return)
            .collect();
        
        if downside_returns.is_empty() {
            return f64::INFINITY;
        }
        
        let downside_variance = downside_returns.iter()
            .map(|&r| r.powi(2))
            .sum::<f64>() / downside_returns.len() as f64;
        
        let downside_deviation = downside_variance.sqrt();
        
        if downside_deviation == 0.0 {
            return f64::INFINITY;
        }
        
        excess_return / downside_deviation
    }
    
    /// Calculate maximum drawdown from a series of values
    pub fn max_drawdown(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mut peak = values[0];
        let mut max_dd = 0.0;
        
        for &value in values.iter().skip(1) {
            if value > peak {
                peak = value;
            } else {
                let drawdown = (peak - value) / peak;
                if drawdown > max_dd {
                    max_dd = drawdown;
                }
            }
        }
        
        -max_dd // Return as negative percentage
    }
    
    /// Calculate correlation coefficient between two series
    pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }
        
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            return 0.0;
        }
        
        numerator / denominator
    }
    
    /// Calculate z-score for a value in a series
    pub fn z_score(value: f64, series: &[f64]) -> f64 {
        if series.is_empty() {
            return 0.0;
        }
        
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let variance = series.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / series.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        (value - mean) / std_dev
    }
    
    /// Round to specified decimal places
    pub fn round_to_decimals(value: f64, decimals: u32) -> f64 {
        let multiplier = 10_f64.powi(decimals as i32);
        (value * multiplier).round() / multiplier
    }
    
    /// Calculate the geometric mean of a series
    pub fn geometric_mean(values: &[f64]) -> f64 {
        if values.is_empty() || values.iter().any(|&x| x <= 0.0) {
            return 0.0;
        }
        
        let product: f64 = values.iter().product();
        product.powf(1.0 / values.len() as f64)
    }
    
    /// Calculate standard deviation
    pub fn standard_deviation(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        variance.sqrt()
    }
}

/// Time utility functions
pub mod time {
    use super::*;
    
    /// Convert timestamp to DateTime<Utc>
    pub fn timestamp_to_datetime(timestamp: u64) -> DateTime<Utc> {
        Utc.timestamp_opt(timestamp as i64, 0).single().unwrap_or_else(Utc::now)
    }
    
    /// Convert millisecond timestamp to DateTime<Utc>
    pub fn timestamp_ms_to_datetime(timestamp_ms: u64) -> DateTime<Utc> {
        let secs = timestamp_ms / 1000;
        let nanos = ((timestamp_ms % 1000) * 1_000_000) as u32;
        Utc.timestamp_opt(secs as i64, nanos).single().unwrap_or_else(Utc::now)
    }
    
    /// Get current timestamp in seconds
    pub fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    /// Get current timestamp in milliseconds
    pub fn current_timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
    
    /// Calculate elapsed time in milliseconds
    pub fn elapsed_ms(start: SystemTime) -> u64 {
        SystemTime::now()
            .duration_since(start)
            .unwrap_or_default()
            .as_millis() as u64
    }
    
    /// Check if a timestamp is within the last N seconds
    pub fn is_recent(timestamp: u64, max_age_seconds: u64) -> bool {
        let now = current_timestamp();
        now.saturating_sub(timestamp) <= max_age_seconds
    }
    
    /// Format duration as human-readable string
    pub fn format_duration(duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        let millis = duration.subsec_millis();
        
        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}.{}s", seconds, millis / 100)
        } else {
            format!("{}ms", millis)
        }
    }
    
    /// Parse ISO 8601 datetime string
    pub fn parse_iso8601(datetime_str: &str) -> Result<DateTime<Utc>> {
        datetime_str.parse::<DateTime<Utc>>()
            .map_err(|e| TengriError::Config(format!("Failed to parse datetime: {}", e)))
    }
    
    /// Get start of trading day (UTC)
    pub fn trading_day_start(date: DateTime<Utc>) -> DateTime<Utc> {
        date.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc()
    }
    
    /// Check if current time is within trading hours (simplified)
    pub fn is_trading_hours() -> bool {
        let now = Utc::now();
        let hour = now.hour();
        
        // Simplified: assume 24/7 crypto trading
        // For traditional markets, would check weekdays and specific hours
        true
    }
}

/// File system utilities
pub mod fs_utils {
    use super::*;
    
    /// Ensure directory exists, create if it doesn't
    pub async fn ensure_dir_exists<P: AsRef<Path>>(path: P) -> Result<()> {
        if !path.as_ref().exists() {
            fs::create_dir_all(path).await
                .map_err(|e| TengriError::Io(e))?;
        }
        Ok(())
    }
    
    /// Read JSON file
    pub async fn read_json_file<T, P>(path: P) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
        P: AsRef<Path>,
    {
        let contents = fs::read_to_string(path).await
            .map_err(|e| TengriError::Io(e))?;
        
        serde_json::from_str(&contents)
            .map_err(|e| TengriError::Serialization(e))
    }
    
    /// Write JSON file
    pub async fn write_json_file<T, P>(path: P, data: &T) -> Result<()>
    where
        T: Serialize,
        P: AsRef<Path>,
    {
        let contents = serde_json::to_string_pretty(data)
            .map_err(|e| TengriError::Serialization(e))?;
        
        // Ensure parent directory exists
        if let Some(parent) = path.as_ref().parent() {
            ensure_dir_exists(parent).await?;
        }
        
        fs::write(path, contents).await
            .map_err(|e| TengriError::Io(e))?;
        
        Ok(())
    }
    
    /// Get file size in bytes
    pub async fn file_size<P: AsRef<Path>>(path: P) -> Result<u64> {
        let metadata = fs::metadata(path).await
            .map_err(|e| TengriError::Io(e))?;
        Ok(metadata.len())
    }
    
    /// Check if file exists
    pub async fn file_exists<P: AsRef<Path>>(path: P) -> bool {
        fs::metadata(path).await.is_ok()
    }
    
    /// Create backup of a file
    pub async fn backup_file<P: AsRef<Path>>(path: P) -> Result<String> {
        let path_ref = path.as_ref();
        let backup_path = format!("{}.backup.{}", 
            path_ref.to_string_lossy(), 
            time::current_timestamp()
        );
        
        fs::copy(path_ref, &backup_path).await
            .map_err(|e| TengriError::Io(e))?;
        
        Ok(backup_path)
    }
    
    /// Clean up old backup files
    pub async fn cleanup_backups<P: AsRef<Path>>(dir: P, max_age_days: u64) -> Result<usize> {
        let mut removed_count = 0;
        let max_age_secs = max_age_days * 24 * 60 * 60;
        let now = time::current_timestamp();
        
        let mut entries = fs::read_dir(dir).await
            .map_err(|e| TengriError::Io(e))?;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| TengriError::Io(e))? {
            
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();
            
            if file_name_str.contains(".backup.") {
                if let Some(timestamp_str) = file_name_str.split(".backup.").nth(1) {
                    if let Ok(timestamp) = timestamp_str.parse::<u64>() {
                        if now.saturating_sub(timestamp) > max_age_secs {
                            if fs::remove_file(entry.path()).await.is_ok() {
                                removed_count += 1;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(removed_count)
    }
}

/// String utilities
pub mod string_utils {
    use super::*;
    
    /// Generate random string of specified length
    pub fn random_string(length: usize) -> String {
        use rand::Rng;
        const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let mut rng = rand::thread_rng();
        
        (0..length)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }
    
    /// Truncate string to max length with ellipsis
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else if max_len <= 3 {
            "...".to_string()
        } else {
            format!("{}...", &s[..max_len - 3])
        }
    }
    
    /// Convert camelCase to snake_case
    pub fn camel_to_snake(s: &str) -> String {
        let mut result = String::new();
        for (i, c) in s.chars().enumerate() {
            if c.is_uppercase() && i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap_or(c));
        }
        result
    }
    
    /// Convert snake_case to camelCase
    pub fn snake_to_camel(s: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = false;
        
        for c in s.chars() {
            if c == '_' {
                capitalize_next = true;
            } else if capitalize_next {
                result.push(c.to_uppercase().next().unwrap_or(c));
                capitalize_next = false;
            } else {
                result.push(c);
            }
        }
        
        result
    }
    
    /// Format number with thousands separators
    pub fn format_number(n: f64, decimals: usize) -> String {
        let formatted = format!("{:.decimals$}", n, decimals = decimals);
        let parts: Vec<&str> = formatted.split('.').collect();
        
        let integer_part = parts[0];
        let decimal_part = if parts.len() > 1 { parts[1] } else { "" };
        
        let mut result = String::new();
        for (i, c) in integer_part.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        
        let formatted_integer: String = result.chars().rev().collect();
        
        if decimal_part.is_empty() || decimals == 0 {
            formatted_integer
        } else {
            format!("{}.{}", formatted_integer, decimal_part)
        }
    }
    
    /// Extract number from string
    pub fn extract_number(s: &str) -> Option<f64> {
        let number_str: String = s.chars()
            .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+')
            .collect();
        
        number_str.parse().ok()
    }
    
    /// Sanitize filename
    pub fn sanitize_filename(filename: &str) -> String {
        filename.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }
}

/// Validation utilities
pub mod validation {
    use super::*;
    
    /// Validate symbol format (e.g., "BTCUSDT")
    pub fn validate_symbol(symbol: &str) -> bool {
        symbol.len() >= 6 && 
        symbol.len() <= 12 && 
        symbol.chars().all(|c| c.is_ascii_uppercase())
    }
    
    /// Validate price value
    pub fn validate_price(price: f64) -> bool {
        price > 0.0 && price.is_finite()
    }
    
    /// Validate quantity value
    pub fn validate_quantity(quantity: f64) -> bool {
        quantity > 0.0 && quantity.is_finite()
    }
    
    /// Validate percentage value (0-100)
    pub fn validate_percentage(percentage: f64) -> bool {
        percentage >= 0.0 && percentage <= 100.0 && percentage.is_finite()
    }
    
    /// Validate ratio value (0-1)
    pub fn validate_ratio(ratio: f64) -> bool {
        ratio >= 0.0 && ratio <= 1.0 && ratio.is_finite()
    }
    
    /// Validate email address (simple check)
    pub fn validate_email(email: &str) -> bool {
        email.contains('@') && email.contains('.') && email.len() > 5
    }
    
    /// Validate API key format
    pub fn validate_api_key(api_key: &str) -> bool {
        api_key.len() >= 16 && api_key.chars().all(|c| c.is_alphanumeric())
    }
    
    /// Validate timeframe in seconds
    pub fn validate_timeframe(timeframe: u64) -> bool {
        matches!(timeframe, 1 | 5 | 15 | 30 | 60 | 300 | 900 | 1800 | 3600 | 14400 | 86400)
    }
}

/// Logging utilities
pub mod logging {
    use super::*;
    use tracing::{info, warn, error};
    
    /// Log performance metrics
    pub fn log_performance(metrics: &HashMap<String, f64>) {
        info!("Performance metrics:");
        for (key, value) in metrics {
            info!("  {}: {:.4}", key, value);
        }
    }
    
    /// Log trade execution
    pub fn log_trade(symbol: &str, side: &str, quantity: f64, price: f64, order_id: &str) {
        info!("Trade executed: {} {} {} @ {} (Order: {})", 
            side, quantity, symbol, price, order_id);
    }
    
    /// Log error with context
    pub fn log_error_with_context(error: &str, context: &HashMap<String, String>) {
        error!("Error: {}", error);
        for (key, value) in context {
            error!("  {}: {}", key, value);
        }
    }
    
    /// Log system startup
    pub fn log_startup(version: &str, config_path: &str) {
        info!("Starting Tengri Trading Strategy v{}", version);
        info!("Configuration loaded from: {}", config_path);
        info!("System initialized successfully");
    }
    
    /// Log system shutdown
    pub fn log_shutdown(reason: &str) {
        info!("Shutting down Tengri Trading Strategy");
        info!("Shutdown reason: {}", reason);
        info!("System shutdown complete");
    }
}

/// Configuration utilities
pub mod config_utils {
    use super::*;
    
    /// Merge two HashMaps, with values from the second map taking precedence
    pub fn merge_configs<T: Clone>(base: &HashMap<String, T>, override_map: &HashMap<String, T>) -> HashMap<String, T> {
        let mut result = base.clone();
        for (key, value) in override_map {
            result.insert(key.clone(), value.clone());
        }
        result
    }
    
    /// Generate default configuration filename
    pub fn default_config_filename() -> String {
        format!("tengri_config_{}.toml", time::current_timestamp())
    }
    
    /// Validate configuration directory
    pub async fn ensure_config_dir() -> Result<String> {
        let config_dir = dirs::config_dir()
            .map(|mut path| {
                path.push("tengri");
                path
            })
            .unwrap_or_else(|| Path::new(".").join("config"));
        
        fs_utils::ensure_dir_exists(&config_dir).await?;
        
        Ok(config_dir.to_string_lossy().to_string())
    }
    
    /// Get environment variable with default
    pub fn get_env_var(key: &str, default: &str) -> String {
        std::env::var(key).unwrap_or_else(|_| default.to_string())
    }
    
    /// Parse environment variable as boolean
    pub fn get_env_bool(key: &str, default: bool) -> bool {
        match std::env::var(key) {
            Ok(val) => matches!(val.to_lowercase().as_str(), "true" | "1" | "yes" | "on"),
            Err(_) => default,
        }
    }
    
    /// Parse environment variable as number
    pub fn get_env_number<T>(key: &str, default: T) -> T
    where
        T: std::str::FromStr + Copy,
    {
        std::env::var(key)
            .ok()
            .and_then(|val| val.parse().ok())
            .unwrap_or(default)
    }
}

/// Performance utilities
pub mod performance {
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
        
        pub fn elapsed_ms(&self) -> u64 {
            self.start.elapsed().as_millis() as u64
        }
        
        pub fn log_elapsed(&self) {
            tracing::debug!("{} took {}ms", self.name, self.elapsed_ms());
        }
    }
    
    impl Drop for Timer {
        fn drop(&mut self) {
            self.log_elapsed();
        }
    }
    
    /// Measure and log function execution time
    pub async fn time_async<F, T>(name: &str, future: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();
        let result = future.await;
        let elapsed = start.elapsed();
        
        tracing::debug!("{} completed in {}", name, time::format_duration(elapsed));
        
        result
    }
    
    /// Calculate throughput (operations per second)
    pub fn calculate_throughput(operations: u64, duration: Duration) -> f64 {
        if duration.as_secs() == 0 {
            return 0.0;
        }
        operations as f64 / duration.as_secs_f64()
    }
    
    /// Memory usage monitoring (simplified)
    pub fn get_memory_usage_mb() -> f64 {
        // This would use proper system APIs in production
        0.0 // Placeholder
    }
}

/// Generate unique identifier
pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

/// Generate short ID (8 characters)
pub fn generate_short_id() -> String {
    Uuid::new_v4().to_string()[..8].to_string()
}

/// Check if string is valid JSON
pub fn is_valid_json(s: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(s).is_ok()
}

/// Retry function with exponential backoff
pub async fn retry_with_backoff<F, T, E>(
    mut operation: F,
    max_retries: usize,
    initial_delay: Duration,
) -> Result<T>
where
    F: FnMut() -> std::result::Result<T, E>,
    E: std::fmt::Display,
{
    let mut delay = initial_delay;
    
    for attempt in 0..=max_retries {
        match operation() {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempt == max_retries {
                    return Err(TengriError::Strategy(format!("Operation failed after {} retries: {}", max_retries, e)));
                }
                
                tracing::warn!("Operation failed (attempt {}), retrying in {}: {}", 
                    attempt + 1, time::format_duration(delay), e);
                
                tokio::time::sleep(delay).await;
                delay = std::cmp::min(delay * 2, Duration::from_secs(60)); // Cap at 1 minute
            }
        }
    }
    
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentage_change() {
        assert_eq!(math::percentage_change(100.0, 110.0), 10.0);
        assert_eq!(math::percentage_change(100.0, 90.0), -10.0);
        assert_eq!(math::percentage_change(0.0, 100.0), 0.0);
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.1, 0.05, -0.02, 0.08, 0.03];
        let sharpe = math::sharpe_ratio(&returns, 0.02);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let values = vec![100.0, 110.0, 95.0, 105.0, 90.0, 120.0];
        let dd = math::max_drawdown(&values);
        assert!(dd < 0.0); // Should be negative
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = math::correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001); // Perfect positive correlation
    }

    #[test]
    fn test_validation() {
        assert!(validation::validate_symbol("BTCUSDT"));
        assert!(!validation::validate_symbol("btcusdt"));
        assert!(!validation::validate_symbol("BTC"));
        
        assert!(validation::validate_price(100.5));
        assert!(!validation::validate_price(-100.0));
        assert!(!validation::validate_price(f64::NAN));
        
        assert!(validation::validate_percentage(50.0));
        assert!(!validation::validate_percentage(150.0));
    }

    #[test]
    fn test_string_utils() {
        assert_eq!(string_utils::camel_to_snake("camelCase"), "camel_case");
        assert_eq!(string_utils::snake_to_camel("snake_case"), "snakeCase");
        
        let truncated = string_utils::truncate("This is a long string", 10);
        assert_eq!(truncated, "This is...");
        
        let formatted = string_utils::format_number(1234567.89, 2);
        assert_eq!(formatted, "1,234,567.89");
    }

    #[test]
    fn test_time_utils() {
        let now = time::current_timestamp();
        assert!(now > 0);
        
        let recent = time::is_recent(now, 10);
        assert!(recent);
        
        let old = time::is_recent(now - 100, 10);
        assert!(!old);
    }

    #[tokio::test]
    async fn test_performance_timer() {
        let timer = performance::Timer::new("test");
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(timer.elapsed_ms() >= 10);
    }
}