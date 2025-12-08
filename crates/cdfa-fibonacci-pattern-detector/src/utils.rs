/*!
# Utilities Module

Common utilities and helper functions for pattern detection.
Provides mathematical functions, data validation, and performance utilities.

## Features

- **Mathematical Utilities**: Common math functions for pattern analysis
- **Data Validation**: Input data validation and sanitization
- **Performance Monitoring**: Timing and profiling utilities
- **Statistical Functions**: Statistical analysis helpers
- **Memory Management**: Efficient memory usage utilities
*/

use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::{PatternResult, PatternError};

/// Timer for performance monitoring
#[derive(Debug)]
pub struct Timer {
    start_time: Instant,
    label: String,
}

impl Timer {
    /// Create a new timer
    pub fn new(label: &str) -> Self {
        Self {
            start_time: Instant::now(),
            label: label.to_string(),
        }
    }
    
    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> u64 {
        self.elapsed().as_micros() as u64
    }
    
    /// Get elapsed time in nanoseconds
    pub fn elapsed_ns(&self) -> u64 {
        self.elapsed().as_nanos() as u64
    }
    
    /// Stop timer and return elapsed time
    pub fn stop(self) -> Duration {
        let elapsed = self.elapsed();
        log::debug!("Timer '{}' elapsed: {:?}", self.label, elapsed);
        elapsed
    }
}

/// Scoped timer that logs on drop
pub struct ScopedTimer {
    timer: Timer,
}

impl ScopedTimer {
    /// Create a new scoped timer
    pub fn new(label: &str) -> Self {
        Self {
            timer: Timer::new(label),
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        log::debug!("Scoped timer '{}' elapsed: {:?}", self.timer.label, self.timer.elapsed());
    }
}

/// Performance profiler for detailed timing
#[derive(Debug, Default)]
pub struct Profiler {
    timers: HashMap<String, Vec<Duration>>,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Start timing an operation
    pub fn start_timer(&self, label: &str) -> Timer {
        Timer::new(label)
    }
    
    /// Record timing for an operation
    pub fn record(&mut self, label: &str, duration: Duration) {
        self.timers.entry(label.to_string()).or_insert_with(Vec::new).push(duration);
    }
    
    /// Get average time for an operation
    pub fn get_average(&self, label: &str) -> Option<Duration> {
        if let Some(times) = self.timers.get(label) {
            if !times.is_empty() {
                let total: Duration = times.iter().sum();
                Some(total / times.len() as u32)
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Get total time for an operation
    pub fn get_total(&self, label: &str) -> Option<Duration> {
        self.timers.get(label).map(|times| times.iter().sum())
    }
    
    /// Get call count for an operation
    pub fn get_count(&self, label: &str) -> usize {
        self.timers.get(label).map(|times| times.len()).unwrap_or(0)
    }
    
    /// Get all statistics
    pub fn get_stats(&self) -> HashMap<String, ProfileStats> {
        let mut stats = HashMap::new();
        
        for (label, times) in &self.timers {
            if !times.is_empty() {
                let total: Duration = times.iter().sum();
                let average = total / times.len() as u32;
                let min = *times.iter().min().unwrap();
                let max = *times.iter().max().unwrap();
                
                stats.insert(label.clone(), ProfileStats {
                    count: times.len(),
                    total,
                    average,
                    min,
                    max,
                });
            }
        }
        
        stats
    }
    
    /// Clear all timing data
    pub fn clear(&mut self) {
        self.timers.clear();
    }
    
    /// Print statistics
    pub fn print_stats(&self) {
        println!("Performance Statistics:");
        println!("{:<20} {:>8} {:>12} {:>12} {:>12} {:>12}", 
                 "Operation", "Count", "Total (ms)", "Avg (μs)", "Min (μs)", "Max (μs)");
        println!("{}", "-".repeat(84));
        
        for (label, stats) in self.get_stats() {
            println!("{:<20} {:>8} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
                     label,
                     stats.count,
                     stats.total.as_secs_f64() * 1000.0,
                     stats.average.as_secs_f64() * 1_000_000.0,
                     stats.min.as_secs_f64() * 1_000_000.0,
                     stats.max.as_secs_f64() * 1_000_000.0);
        }
    }
}

/// Profile statistics for an operation
#[derive(Debug, Clone)]
pub struct ProfileStats {
    pub count: usize,
    pub total: Duration,
    pub average: Duration,
    pub min: Duration,
    pub max: Duration,
}

/// Data validation utilities
pub struct DataValidator;

impl DataValidator {
    /// Validate price arrays
    pub fn validate_prices(highs: &[f64], lows: &[f64], closes: &[f64]) -> PatternResult<()> {
        if highs.is_empty() || lows.is_empty() || closes.is_empty() {
            return Err(PatternError::InvalidInput("Empty price arrays".to_string()));
        }
        
        if highs.len() != lows.len() || highs.len() != closes.len() {
            return Err(PatternError::InvalidInput("Price arrays must have same length".to_string()));
        }
        
        // Check for valid price values
        for (i, (&high, &low, &close)) in highs.iter().zip(lows.iter()).zip(closes.iter()).enumerate() {
            if !high.is_finite() || !low.is_finite() || !close.is_finite() {
                return Err(PatternError::InvalidInput(format!("Invalid price at index {}", i)));
            }
            
            if high < 0.0 || low < 0.0 || close < 0.0 {
                return Err(PatternError::InvalidInput(format!("Negative price at index {}", i)));
            }
            
            if high < low {
                return Err(PatternError::InvalidInput(format!("High < Low at index {}", i)));
            }
            
            if close < low || close > high {
                return Err(PatternError::InvalidInput(format!("Close outside high-low range at index {}", i)));
            }
        }
        
        Ok(())
    }
    
    /// Validate swing indices
    pub fn validate_swing_indices(swing_highs: &[usize], swing_lows: &[usize], data_length: usize) -> PatternResult<()> {
        // Check bounds
        for &idx in swing_highs {
            if idx >= data_length {
                return Err(PatternError::InvalidInput(format!("Swing high index {} out of bounds", idx)));
            }
        }
        
        for &idx in swing_lows {
            if idx >= data_length {
                return Err(PatternError::InvalidInput(format!("Swing low index {} out of bounds", idx)));
            }
        }
        
        // Check for duplicates
        let mut seen_highs = std::collections::HashSet::new();
        for &idx in swing_highs {
            if !seen_highs.insert(idx) {
                return Err(PatternError::InvalidInput(format!("Duplicate swing high index {}", idx)));
            }
        }
        
        let mut seen_lows = std::collections::HashSet::new();
        for &idx in swing_lows {
            if !seen_lows.insert(idx) {
                return Err(PatternError::InvalidInput(format!("Duplicate swing low index {}", idx)));
            }
        }
        
        Ok(())
    }
    
    /// Validate pattern configuration
    pub fn validate_config(min_pattern_size: f64, max_pattern_size: f64, ratio_tolerance: f64) -> PatternResult<()> {
        if min_pattern_size <= 0.0 || min_pattern_size > 1.0 {
            return Err(PatternError::ConfigError("min_pattern_size must be between 0 and 1".to_string()));
        }
        
        if max_pattern_size <= 0.0 || max_pattern_size > 1.0 {
            return Err(PatternError::ConfigError("max_pattern_size must be between 0 and 1".to_string()));
        }
        
        if min_pattern_size >= max_pattern_size {
            return Err(PatternError::ConfigError("min_pattern_size must be less than max_pattern_size".to_string()));
        }
        
        if ratio_tolerance <= 0.0 || ratio_tolerance > 1.0 {
            return Err(PatternError::ConfigError("ratio_tolerance must be between 0 and 1".to_string()));
        }
        
        Ok(())
    }
}

/// Mathematical utilities
pub struct MathUtils;

impl MathUtils {
    /// Calculate standard deviation
    pub fn std_dev(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variance.sqrt()
    }
    
    /// Calculate correlation coefficient
    pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        
        let n = x.len() as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>();
        let sum_x2 = x.iter().map(|a| a * a).sum::<f64>();
        let sum_y2 = y.iter().map(|b| b * b).sum::<f64>();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    /// Calculate linear regression slope
    pub fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        
        let n = x.len() as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>();
        let sum_x2 = x.iter().map(|a| a * a).sum::<f64>();
        
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator == 0.0 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }
    
    /// Calculate exponential moving average
    pub fn ema(values: &[f64], period: usize) -> Vec<f64> {
        if values.is_empty() || period == 0 {
            return Vec::new();
        }
        
        let alpha = 2.0 / (period + 1) as f64;
        let mut ema = Vec::with_capacity(values.len());
        ema.push(values[0]);
        
        for i in 1..values.len() {
            let new_ema = alpha * values[i] + (1.0 - alpha) * ema[i - 1];
            ema.push(new_ema);
        }
        
        ema
    }
    
    /// Calculate simple moving average
    pub fn sma(values: &[f64], period: usize) -> Vec<f64> {
        if values.is_empty() || period == 0 {
            return Vec::new();
        }
        
        let mut sma = Vec::new();
        
        for i in 0..values.len() {
            let start = if i >= period { i - period + 1 } else { 0 };
            let end = i + 1;
            let sum = values[start..end].iter().sum::<f64>();
            sma.push(sum / (end - start) as f64);
        }
        
        sma
    }
    
    /// Calculate relative strength index
    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 {
            return Vec::new();
        }
        
        let mut gains = Vec::new();
        let mut losses = Vec::new();
        
        // Calculate price changes
        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            gains.push(change.max(0.0));
            losses.push((-change).max(0.0));
        }
        
        // Calculate RSI
        let mut rsi = Vec::new();
        let mut avg_gain = gains[..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss = losses[..period].iter().sum::<f64>() / period as f64;
        
        for i in period..gains.len() {
            if avg_loss == 0.0 {
                rsi.push(100.0);
            } else {
                let rs = avg_gain / avg_loss;
                rsi.push(100.0 - (100.0 / (1.0 + rs)));
            }
            
            // Update averages
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
        }
        
        rsi
    }
    
    /// Normalize values to 0-1 range
    pub fn normalize(values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }
        
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_val == min_val {
            vec![0.5; values.len()]
        } else {
            values.iter()
                .map(|&x| (x - min_val) / (max_val - min_val))
                .collect()
        }
    }
    
    /// Calculate percentile
    pub fn percentile(values: &[f64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (percentile / 100.0) * (sorted.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        
        if lower == upper {
            sorted[lower]
        } else {
            let weight = index - lower as f64;
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }
}

/// Memory utilities
pub struct MemoryUtils;

impl MemoryUtils {
    /// Get current memory usage in bytes
    pub fn get_memory_usage() -> usize {
        // Platform-specific memory usage calculation
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(contents) = fs::read_to_string("/proc/self/status") {
                for line in contents.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(size_str) = line.split_whitespace().nth(1) {
                            if let Ok(size_kb) = size_str.parse::<usize>() {
                                return size_kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            use std::mem;
            use std::ptr;
            
            // macOS implementation using task_info
            unsafe {
                let mut info: libc::mach_task_basic_info_data_t = mem::zeroed();
                let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
                
                let result = libc::task_info(
                    libc::mach_task_self(),
                    libc::MACH_TASK_BASIC_INFO,
                    &mut info as *mut _ as *mut i32,
                    &mut count,
                );
                
                if result == libc::KERN_SUCCESS {
                    return info.resident_size as usize;
                }
            }
        }
        
        // Fallback: return 0 if platform-specific implementation fails
        0
    }
    
    /// Get memory usage in a human-readable format
    pub fn format_memory_usage(bytes: usize) -> String {
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

/// String utilities
pub struct StringUtils;

impl StringUtils {
    /// Convert snake_case to camelCase
    pub fn snake_to_camel(snake_str: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = false;
        
        for ch in snake_str.chars() {
            if ch == '_' {
                capitalize_next = true;
            } else if capitalize_next {
                result.push(ch.to_uppercase().next().unwrap_or(ch));
                capitalize_next = false;
            } else {
                result.push(ch);
            }
        }
        
        result
    }
    
    /// Convert camelCase to snake_case
    pub fn camel_to_snake(camel_str: &str) -> String {
        let mut result = String::new();
        
        for ch in camel_str.chars() {
            if ch.is_uppercase() {
                if !result.is_empty() {
                    result.push('_');
                }
                result.push(ch.to_lowercase().next().unwrap_or(ch));
            } else {
                result.push(ch);
            }
        }
        
        result
    }
    
    /// Format duration in human-readable format
    pub fn format_duration(duration: Duration) -> String {
        let nanos = duration.as_nanos() as u64;
        
        if nanos < 1_000 {
            format!("{} ns", nanos)
        } else if nanos < 1_000_000 {
            format!("{:.2} μs", nanos as f64 / 1_000.0)
        } else if nanos < 1_000_000_000 {
            format!("{:.2} ms", nanos as f64 / 1_000_000.0)
        } else {
            format!("{:.2} s", nanos as f64 / 1_000_000_000.0)
        }
    }
}

/// Create a scoped timer
pub fn time_scope(label: &str) -> ScopedTimer {
    ScopedTimer::new(label)
}

/// Benchmark a function
pub fn benchmark<F, R>(label: &str, mut f: F) -> (R, Duration)
where
    F: FnMut() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    
    log::info!("Benchmark '{}' took: {}", label, StringUtils::format_duration(elapsed));
    
    (result, elapsed)
}

/// Benchmark a function multiple times
pub fn benchmark_multiple<F, R>(label: &str, iterations: usize, mut f: F) -> (Vec<R>, Duration)
where
    F: FnMut() -> R,
{
    let mut results = Vec::with_capacity(iterations);
    let start = Instant::now();
    
    for _ in 0..iterations {
        results.push(f());
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed / iterations as u32;
    
    log::info!("Benchmark '{}' - {} iterations, avg: {}, total: {}", 
              label, iterations, 
              StringUtils::format_duration(avg_time), 
              StringUtils::format_duration(elapsed));
    
    (results, elapsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_timer() {
        let timer = Timer::new("test");
        std::thread::sleep(Duration::from_millis(1));
        let elapsed = timer.elapsed();
        assert!(elapsed.as_millis() >= 1);
    }
    
    #[test]
    fn test_profiler() {
        let mut profiler = Profiler::new();
        
        let timer = profiler.start_timer("test_operation");
        std::thread::sleep(Duration::from_millis(1));
        profiler.record("test_operation", timer.elapsed());
        
        let avg = profiler.get_average("test_operation");
        assert!(avg.is_some());
        assert!(avg.unwrap().as_millis() >= 1);
        
        let count = profiler.get_count("test_operation");
        assert_eq!(count, 1);
    }
    
    #[test]
    fn test_data_validation() {
        let highs = vec![100.0, 105.0, 110.0];
        let lows = vec![95.0, 98.0, 102.0];
        let closes = vec![98.0, 103.0, 108.0];
        
        let result = DataValidator::validate_prices(&highs, &lows, &closes);
        assert!(result.is_ok());
        
        // Test mismatched lengths
        let result = DataValidator::validate_prices(&highs, &lows, &[100.0]);
        assert!(result.is_err());
        
        // Test invalid prices
        let invalid_highs = vec![100.0, 105.0, f64::NAN];
        let result = DataValidator::validate_prices(&invalid_highs, &lows, &closes);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_math_utils() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let std_dev = MathUtils::std_dev(&values);
        assert_relative_eq!(std_dev, 1.4142135623730951, epsilon = 1e-10);
        
        let normalized = MathUtils::normalize(&values);
        assert_relative_eq!(normalized[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(normalized[4], 1.0, epsilon = 1e-10);
        
        let percentile = MathUtils::percentile(&values, 50.0);
        assert_relative_eq!(percentile, 3.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let correlation = MathUtils::correlation(&x, &y);
        assert_relative_eq!(correlation, 1.0, epsilon = 1e-10);
        
        let y_inverse = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let correlation = MathUtils::correlation(&x, &y_inverse);
        assert_relative_eq!(correlation, -1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_moving_averages() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let sma = MathUtils::sma(&values, 3);
        assert_eq!(sma.len(), 5);
        assert_relative_eq!(sma[2], 2.0, epsilon = 1e-10); // (1+2+3)/3
        assert_relative_eq!(sma[4], 4.0, epsilon = 1e-10); // (3+4+5)/3
        
        let ema = MathUtils::ema(&values, 3);
        assert_eq!(ema.len(), 5);
        assert_eq!(ema[0], 1.0);
    }
    
    #[test]
    fn test_string_utils() {
        assert_eq!(StringUtils::snake_to_camel("hello_world"), "helloWorld");
        assert_eq!(StringUtils::camel_to_snake("helloWorld"), "hello_world");
        
        let duration = Duration::from_millis(1500);
        let formatted = StringUtils::format_duration(duration);
        assert!(formatted.contains("1.50 s"));
    }
    
    #[test]
    fn test_memory_utils() {
        let memory_usage = MemoryUtils::get_memory_usage();
        println!("Memory usage: {}", MemoryUtils::format_memory_usage(memory_usage));
        
        let formatted = MemoryUtils::format_memory_usage(1024 * 1024); // 1 MB
        assert!(formatted.contains("1.00 MB"));
    }
    
    #[test]
    fn test_scoped_timer() {
        let _timer = time_scope("test_scope");
        std::thread::sleep(Duration::from_millis(1));
        // Timer will automatically log when dropped
    }
    
    #[test]
    fn test_benchmark() {
        let (result, duration) = benchmark("test_benchmark", || {
            std::thread::sleep(Duration::from_millis(1));
            42
        });
        
        assert_eq!(result, 42);
        assert!(duration.as_millis() >= 1);
    }
    
    #[test]
    fn test_benchmark_multiple() {
        let (results, duration) = benchmark_multiple("test_multiple", 3, || {
            std::thread::sleep(Duration::from_millis(1));
            42
        });
        
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|&x| x == 42));
        assert!(duration.as_millis() >= 3);
    }
}