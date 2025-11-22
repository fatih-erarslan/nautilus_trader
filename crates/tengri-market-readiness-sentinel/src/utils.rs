//! Utility functions for TENGRI Market Readiness Sentinel

use anyhow::Result;
use chrono::{DateTime, Utc};
use std::time::{Duration, Instant};
use tracing::debug;

/// Measure execution time of an async function
pub async fn measure_async<F, T>(operation: F) -> Result<(T, Duration)>
where
    F: std::future::Future<Output = Result<T>>,
{
    let start = Instant::now();
    let result = operation.await?;
    let duration = start.elapsed();
    Ok((result, duration))
}

/// Measure execution time of a sync function
pub fn measure_sync<F, T>(operation: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = operation();
    let duration = start.elapsed();
    (result, duration)
}

/// Format duration for human-readable display
pub fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    
    if total_seconds < 60 {
        format!("{}s", total_seconds)
    } else if total_seconds < 3600 {
        let minutes = total_seconds / 60;
        let seconds = total_seconds % 60;
        format!("{}m {}s", minutes, seconds)
    } else {
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        format!("{}h {}m {}s", hours, minutes, seconds)
    }
}

/// Calculate percentage with safe division
pub fn calculate_percentage(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        (numerator as f64 / denominator as f64) * 100.0
    }
}

/// Validate URL format
pub fn validate_url(url: &str) -> Result<()> {
    url::Url::parse(url)?;
    Ok(())
}

/// Retry an operation with exponential backoff
pub async fn retry_with_backoff<F, T, E>(
    mut operation: F,
    max_attempts: u32,
    initial_delay: Duration,
) -> Result<T, E>
where
    F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, E>> + Send>>,
    E: std::fmt::Debug,
{
    let mut delay = initial_delay;
    
    for attempt in 1..=max_attempts {
        debug!("Attempting operation (attempt {}/{})", attempt, max_attempts);
        
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempt == max_attempts {
                    debug!("Operation failed after {} attempts: {:?}", max_attempts, e);
                    return Err(e);
                }
                
                debug!("Operation failed, retrying in {:?}", delay);
                tokio::time::sleep(delay).await;
                delay = std::cmp::min(delay * 2, Duration::from_secs(60)); // Cap at 1 minute
            }
        }
    }
    
    unreachable!()
}

/// Generate a timestamp string for logging
pub fn timestamp_string() -> String {
    Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

/// Convert bytes to human-readable format
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    
    if bytes == 0 {
        return "0 B".to_string();
    }
    
    let base = 1024u64;
    let exp = (bytes as f64).log(base as f64).floor() as usize;
    let unit = UNITS.get(exp).unwrap_or(&"PB");
    let size = bytes as f64 / (base.pow(exp as u32) as f64);
    
    format!("{:.1} {}", size, unit)
}

/// Sanitize a string for safe logging
pub fn sanitize_for_logging(input: &str) -> String {
    input
        .chars()
        .map(|c| if c.is_control() { '_' } else { c })
        .collect()
}

/// Check if a port is available
pub async fn is_port_available(port: u16) -> bool {
    tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port))
        .await
        .is_ok()
}

/// Generate a random alphanumeric string
pub fn generate_random_string(length: usize) -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                            abcdefghijklmnopqrstuvwxyz\
                            0123456789";
    
    let mut rng = rand::thread_rng();
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// Calculate exponential moving average
pub fn calculate_ema(current_value: f64, previous_ema: f64, alpha: f64) -> f64 {
    alpha * current_value + (1.0 - alpha) * previous_ema
}

/// Calculate standard deviation
pub fn calculate_std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    
    variance.sqrt()
}

/// Calculate percentile
pub fn calculate_percentile(mut values: Vec<f64>, percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let index = (percentile / 100.0) * (values.len() - 1) as f64;
    let lower_index = index.floor() as usize;
    let upper_index = index.ceil() as usize;
    
    if lower_index == upper_index {
        values[lower_index]
    } else {
        let weight = index - lower_index as f64;
        values[lower_index] * (1.0 - weight) + values[upper_index] * weight
    }
}

/// Truncate string to specified length with ellipsis
pub fn truncate_string(s: &str, max_length: usize) -> String {
    if s.len() <= max_length {
        s.to_string()
    } else {
        format!("{}...", &s[..max_length.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m 1s");
    }

    #[test]
    fn test_calculate_percentage() {
        assert_eq!(calculate_percentage(50, 100), 50.0);
        assert_eq!(calculate_percentage(0, 100), 0.0);
        assert_eq!(calculate_percentage(100, 0), 0.0);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
    }

    #[test]
    fn test_calculate_std_dev() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std_dev = calculate_std_dev(&values);
        assert!((std_dev - 1.414).abs() < 0.01); // sqrt(2) ≈ 1.414
    }

    #[test]
    fn test_calculate_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(calculate_percentile(values.clone(), 0.0), 1.0);
        assert_eq!(calculate_percentile(values.clone(), 50.0), 3.0);
        assert_eq!(calculate_percentile(values.clone(), 100.0), 5.0);
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("hello", 10), "hello");
        assert_eq!(truncate_string("hello world", 8), "hello...");
    }

    #[test]
    fn test_sanitize_for_logging() {
        let input = "hello\nworld\ttab";
        let sanitized = sanitize_for_logging(input);
        assert_eq!(sanitized, "hello_world_tab");
    }

    #[tokio::test]
    async fn test_measure_async() {
        let operation = async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok::<_, anyhow::Error>(42)
        };

        let (result, duration) = measure_async(operation).await.unwrap();
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(90)); // Allow some variance
    }

    #[test]
    fn test_measure_sync() {
        let operation = || {
            std::thread::sleep(Duration::from_millis(10));
            42
        };

        let (result, duration) = measure_sync(operation);
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(5)); // Allow some variance
    }
}