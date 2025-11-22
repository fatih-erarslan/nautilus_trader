//! Test module for the autopoiesis framework
//! Provides comprehensive testing infrastructure for all system components

pub mod integration;
pub mod unit;
pub mod benchmarks;
pub mod test_data;
pub mod test_runner;

// Common test utilities
pub mod common {
    use std::time::{Duration, Instant};
    use tokio::time::sleep;
    
    /// Test timeout wrapper
    pub async fn with_timeout<F, R>(duration: Duration, future: F) -> Result<R, &'static str>
    where
        F: std::future::Future<Output = R>,
    {
        match tokio::time::timeout(duration, future).await {
            Ok(result) => Ok(result),
            Err(_) => Err("Test timed out"),
        }
    }
    
    /// Retry mechanism for flaky tests
    pub async fn with_retry<F, R, E>(max_attempts: usize, mut test_fn: F) -> Result<R, E>
    where
        F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<R, E>>>>,
        E: std::fmt::Debug,
    {
        let mut last_error = None;
        
        for attempt in 1..=max_attempts {
            match test_fn().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if attempt < max_attempts {
                        println!("Test attempt {} failed: {:?}, retrying...", attempt, error);
                        sleep(Duration::from_millis(100 * attempt as u64)).await;
                    }
                    last_error = Some(error);
                }
            }
        }
        
        Err(last_error.unwrap())
    }
    
    /// Performance measurement utility
    pub struct PerfTimer {
        start: Instant,
        name: String,
    }
    
    impl PerfTimer {
        pub fn new(name: &str) -> Self {
            Self {
                start: Instant::now(),
                name: name.to_string(),
            }
        }
        
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }
        
        pub fn check_threshold(&self, threshold: Duration) -> bool {
            self.elapsed() <= threshold
        }
    }
    
    impl Drop for PerfTimer {
        fn drop(&mut self) {
            let elapsed = self.elapsed();
            println!("⏱️  {} completed in {:?}", self.name, elapsed);
        }
    }
    
    /// Memory usage estimation
    pub fn estimate_memory_usage() -> usize {
        // Simplified memory usage estimation
        // In a real implementation, you would use actual memory profiling
        std::mem::size_of::<usize>() * 1000 // Placeholder
    }
    
    /// Generate test data with specific patterns
    pub fn generate_test_series(length: usize, pattern: TestPattern) -> Vec<f64> {
        match pattern {
            TestPattern::Linear => (0..length).map(|i| i as f64).collect(),
            TestPattern::Exponential => (0..length).map(|i| (i as f64).exp()).collect(),
            TestPattern::Sinusoidal => {
                (0..length).map(|i| (i as f64 * 0.1).sin()).collect()
            },
            TestPattern::Random => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                (0..length).map(|_| rng.gen_range(0.0..1.0)).collect()
            },
            TestPattern::Step => {
                (0..length).map(|i| if i < length / 2 { 0.0 } else { 1.0 }).collect()
            },
        }
    }
    
    #[derive(Clone, Debug)]
    pub enum TestPattern {
        Linear,
        Exponential,
        Sinusoidal,
        Random,
        Step,
    }
    
    /// Assertion helpers
    pub fn assert_within_range<T>(value: T, min: T, max: T, message: &str)
    where
        T: PartialOrd + std::fmt::Debug,
    {
        assert!(
            value >= min && value <= max,
            "{}: {:?} not in range [{:?}, {:?}]",
            message, value, min, max
        );
    }
    
    pub fn assert_approximately_equal(a: f64, b: f64, epsilon: f64, message: &str) {
        assert!(
            (a - b).abs() <= epsilon,
            "{}: {} and {} differ by more than {}",
            message, a, b, epsilon
        );
    }
    
    /// Test fixture management
    pub struct TestFixture {
        pub temp_dir: tempfile::TempDir,
        pub config: TestConfig,
    }
    
    #[derive(Clone, Debug)]
    pub struct TestConfig {
        pub timeout: Duration,
        pub retry_count: usize,
        pub performance_threshold: Duration,
        pub memory_limit: usize,
    }
    
    impl Default for TestConfig {
        fn default() -> Self {
            Self {
                timeout: Duration::from_secs(30),
                retry_count: 3,
                performance_threshold: Duration::from_millis(100),
                memory_limit: 1024 * 1024 * 100, // 100MB
            }
        }
    }
    
    impl TestFixture {
        pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let temp_dir = tempfile::tempdir()?;
            
            Ok(Self {
                temp_dir,
                config: TestConfig::default(),
            })
        }
        
        pub fn with_config(config: TestConfig) -> Result<Self, Box<dyn std::error::Error>> {
            let temp_dir = tempfile::tempdir()?;
            
            Ok(Self {
                temp_dir,
                config,
            })
        }
        
        pub fn temp_path(&self) -> &std::path::Path {
            self.temp_dir.path()
        }
    }
}

// Test categories
pub mod categories {
    /// Unit test marker
    pub struct Unit;
    
    /// Integration test marker
    pub struct Integration;
    
    /// Performance test marker
    pub struct Performance;
    
    /// Property-based test marker
    pub struct Property;
    
    /// Chaos engineering test marker
    pub struct Chaos;
}

// Test macros
#[macro_export]
macro_rules! autopoiesis_test {
    ($name:ident, $category:ty, $test_fn:expr) => {
        #[tokio::test]
        #[allow(non_snake_case)]
        async fn $name() {
            use $crate::tests::common::*;
            
            let fixture = TestFixture::new().expect("Failed to create test fixture");
            let timer = PerfTimer::new(stringify!($name));
            
            let result = with_timeout(
                fixture.config.timeout,
                $test_fn(fixture)
            ).await;
            
            match result {
                Ok(test_result) => {
                    if !timer.check_threshold(fixture.config.performance_threshold) {
                        println!("⚠️  Test {} exceeded performance threshold", stringify!($name));
                    }
                    test_result
                },
                Err(timeout_msg) => {
                    panic!("Test {} failed: {}", stringify!($name), timeout_msg);
                }
            }
        }
    };
}

#[macro_export]
macro_rules! chaos_test {
    ($name:ident, $chaos_fn:expr) => {
        autopoiesis_test!($name, $crate::tests::categories::Chaos, $chaos_fn);
    };
}

#[macro_export]
macro_rules! property_test {
    ($name:ident, $property_fn:expr) => {
        autopoiesis_test!($name, $crate::tests::categories::Property, $property_fn);
    };
}

#[macro_export]
macro_rules! performance_test {
    ($name:ident, $perf_fn:expr) => {
        autopoiesis_test!($name, $crate::tests::categories::Performance, $perf_fn);
    };
}

// Re-exports for convenience
pub use test_data::generators::*;
pub use test_runner::{TestRunner, TestConfig as RunnerConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use super::common::*;
    
    #[test]
    fn test_generate_test_series() {
        let linear = generate_test_series(5, TestPattern::Linear);
        assert_eq!(linear, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        
        let step = generate_test_series(4, TestPattern::Step);
        assert_eq!(step, vec![0.0, 0.0, 1.0, 1.0]);
        
        let sinusoidal = generate_test_series(10, TestPattern::Sinusoidal);
        assert_eq!(sinusoidal.len(), 10);
        
        let random = generate_test_series(5, TestPattern::Random);
        assert_eq!(random.len(), 5);
        assert!(random.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
    
    #[test]
    fn test_assertion_helpers() {
        assert_within_range(5, 0, 10, "Value should be in range");
        assert_approximately_equal(1.0, 1.001, 0.01, "Values should be approximately equal");
    }
    
    #[tokio::test]
    async fn test_timeout_wrapper() {
        use std::time::Duration;
        
        // Test successful completion
        let result = with_timeout(
            Duration::from_millis(100),
            async { "success" }
        ).await;
        assert_eq!(result, Ok("success"));
        
        // Test timeout
        let result = with_timeout(
            Duration::from_millis(10),
            async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                "should timeout"
            }
        ).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_retry_mechanism() {
        let mut attempt_count = 0;
        
        let result = with_retry(3, || {
            attempt_count += 1;
            Box::pin(async move {
                if attempt_count < 3 {
                    Err("not yet")
                } else {
                    Ok("success")
                }
            })
        }).await;
        
        assert_eq!(result, Ok("success"));
        assert_eq!(attempt_count, 3);
    }
    
    #[test]
    fn test_perf_timer() {
        let timer = PerfTimer::new("test_operation");
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        assert!(timer.elapsed() >= std::time::Duration::from_millis(10));
        assert!(!timer.check_threshold(std::time::Duration::from_millis(5)));
        assert!(timer.check_threshold(std::time::Duration::from_millis(100)));
    }
    
    #[test]
    fn test_fixture_creation() {
        let fixture = TestFixture::new().expect("Should create fixture");
        assert!(fixture.temp_path().exists());
        
        let custom_config = TestConfig {
            timeout: Duration::from_secs(60),
            retry_count: 5,
            ..TestConfig::default()
        };
        
        let custom_fixture = TestFixture::with_config(custom_config.clone())
            .expect("Should create custom fixture");
        assert_eq!(custom_fixture.config.timeout, Duration::from_secs(60));
        assert_eq!(custom_fixture.config.retry_count, 5);
    }
}