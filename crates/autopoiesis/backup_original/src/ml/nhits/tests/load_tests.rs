use super::*;
use crate::ml::nhits::{NHITSModel, NHITSConfig};
use crate::api::nhits_api::{NHITSService, PredictionRequest, TrainingRequest};
use tokio;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use std::time::{Duration, Instant};
use futures::future::join_all;
use ndarray::Array2;
use tokio::sync::Semaphore;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Comprehensive load testing suite for NHITS API and model performance
pub struct LoadTestSuite {
    pub config: LoadTestConfig,
    pub metrics: Arc<LoadTestMetrics>,
}

#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    pub max_concurrent_requests: usize,
    pub total_requests: usize,
    pub ramp_up_duration: Duration,
    pub test_duration: Duration,
    pub request_timeout: Duration,
    pub model_config: NHITSConfig,
    pub payload_size: PayloadSize,
    pub test_scenarios: Vec<TestScenario>,
}

#[derive(Debug, Clone)]
pub enum PayloadSize {
    Small,   // 24 input features, 12 output features
    Medium,  // 168 input features, 24 output features  
    Large,   // 336 input features, 48 output features
    XLarge,  // 720 input features, 168 output features
    Custom { input_size: usize, output_size: usize },
}

#[derive(Debug, Clone)]
pub struct TestScenario {
    pub name: String,
    pub weight: f32, // Percentage of total requests
    pub request_type: RequestType,
    pub expected_latency_ms: u64,
    pub expected_throughput_rps: f32,
}

#[derive(Debug, Clone)]
pub enum RequestType {
    Prediction,
    Training,
    BatchPrediction { batch_size: usize },
    ConsciousnessPrediction,
    ModelInfo,
    HealthCheck,
}

pub struct LoadTestMetrics {
    pub total_requests: AtomicUsize,
    pub successful_requests: AtomicUsize,
    pub failed_requests: AtomicUsize,
    pub timeout_requests: AtomicUsize,
    pub total_latency_ms: AtomicUsize,
    pub max_latency_ms: AtomicUsize,
    pub min_latency_ms: AtomicUsize,
    pub start_time: Instant,
    pub response_times: tokio::sync::Mutex<Vec<u64>>,
    pub error_counts: tokio::sync::Mutex<HashMap<String, usize>>,
    pub throughput_samples: tokio::sync::Mutex<Vec<(Instant, usize)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResult {
    pub test_name: String,
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub timeout_requests: usize,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub max_latency_ms: u64,
    pub min_latency_ms: u64,
    pub throughput_rps: f64,
    pub error_rate: f64,
    pub errors_by_type: HashMap<String, usize>,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub test_duration: Duration,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        LoadTestConfig {
            max_concurrent_requests: 50,
            total_requests: 1000,
            ramp_up_duration: Duration::from_secs(10),
            test_duration: Duration::from_secs(60),
            request_timeout: Duration::from_secs(5),
            model_config: NHITSConfig::default(),
            payload_size: PayloadSize::Medium,
            test_scenarios: vec![
                TestScenario {
                    name: "prediction".to_string(),
                    weight: 0.7,
                    request_type: RequestType::Prediction,
                    expected_latency_ms: 100,
                    expected_throughput_rps: 50.0,
                },
                TestScenario {
                    name: "batch_prediction".to_string(),
                    weight: 0.2,
                    request_type: RequestType::BatchPrediction { batch_size: 10 },
                    expected_latency_ms: 200,
                    expected_throughput_rps: 10.0,
                },
                TestScenario {
                    name: "health_check".to_string(),
                    weight: 0.1,
                    request_type: RequestType::HealthCheck,
                    expected_latency_ms: 10,
                    expected_throughput_rps: 100.0,
                },
            ],
        }
    }
}

impl LoadTestMetrics {
    pub fn new() -> Arc<Self> {
        Arc::new(LoadTestMetrics {
            total_requests: AtomicUsize::new(0),
            successful_requests: AtomicUsize::new(0),
            failed_requests: AtomicUsize::new(0),
            timeout_requests: AtomicUsize::new(0),
            total_latency_ms: AtomicUsize::new(0),
            max_latency_ms: AtomicUsize::new(0),
            min_latency_ms: AtomicUsize::new(u64::MAX as usize),
            start_time: Instant::now(),
            response_times: tokio::sync::Mutex::new(Vec::new()),
            error_counts: tokio::sync::Mutex::new(HashMap::new()),
            throughput_samples: tokio::sync::Mutex::new(Vec::new()),
        })
    }
    
    pub async fn record_request(&self, latency_ms: u64, success: bool, error_type: Option<String>) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        
        if success {
            self.successful_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_requests.fetch_add(1, Ordering::Relaxed);
            
            if let Some(error) = error_type {
                let mut error_counts = self.error_counts.lock().await;
                *error_counts.entry(error).or_insert(0) += 1;
            }
        }
        
        // Update latency metrics
        self.total_latency_ms.fetch_add(latency_ms as usize, Ordering::Relaxed);
        
        // Update min latency
        let mut current_min = self.min_latency_ms.load(Ordering::Relaxed);
        while latency_ms < current_min as u64 {
            match self.min_latency_ms.compare_exchange_weak(
                current_min,
                latency_ms as usize,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }
        
        // Update max latency
        let mut current_max = self.max_latency_ms.load(Ordering::Relaxed);
        while latency_ms > current_max as u64 {
            match self.max_latency_ms.compare_exchange_weak(
                current_max,
                latency_ms as usize,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
        
        // Store response time for percentile calculations
        let mut response_times = self.response_times.lock().await;
        response_times.push(latency_ms);
        
        // Sample throughput periodically
        if response_times.len() % 100 == 0 {
            let mut throughput_samples = self.throughput_samples.lock().await;
            throughput_samples.push((Instant::now(), self.total_requests.load(Ordering::Relaxed)));
        }
    }
    
    pub async fn generate_result(&self, test_name: String, test_duration: Duration) -> LoadTestResult {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let successful_requests = self.successful_requests.load(Ordering::Relaxed);
        let failed_requests = self.failed_requests.load(Ordering::Relaxed);
        let timeout_requests = self.timeout_requests.load(Ordering::Relaxed);
        let total_latency_ms = self.total_latency_ms.load(Ordering::Relaxed);
        
        let avg_latency_ms = if total_requests > 0 {
            total_latency_ms as f64 / total_requests as f64
        } else {
            0.0
        };
        
        let throughput_rps = if test_duration.as_secs() > 0 {
            total_requests as f64 / test_duration.as_secs_f64()
        } else {
            0.0
        };
        
        let error_rate = if total_requests > 0 {
            failed_requests as f64 / total_requests as f64
        } else {
            0.0
        };
        
        // Calculate percentiles
        let mut response_times = self.response_times.lock().await;
        response_times.sort_unstable();
        
        let p50_latency_ms = self.calculate_percentile(&response_times, 50.0);
        let p95_latency_ms = self.calculate_percentile(&response_times, 95.0);
        let p99_latency_ms = self.calculate_percentile(&response_times, 99.0);
        
        let errors_by_type = self.error_counts.lock().await.clone();
        
        LoadTestResult {
            test_name,
            total_requests,
            successful_requests,
            failed_requests,
            timeout_requests,
            avg_latency_ms,
            p50_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            max_latency_ms: self.max_latency_ms.load(Ordering::Relaxed) as u64,
            min_latency_ms: if self.min_latency_ms.load(Ordering::Relaxed) == usize::MAX {
                0
            } else {
                self.min_latency_ms.load(Ordering::Relaxed) as u64
            },
            throughput_rps,
            error_rate,
            errors_by_type,
            memory_usage_mb: self.get_memory_usage_mb(),
            cpu_usage_percent: self.get_cpu_usage_percent(),
            test_duration,
        }
    }
    
    fn calculate_percentile(&self, sorted_times: &[u64], percentile: f64) -> f64 {
        if sorted_times.is_empty() {
            return 0.0;
        }
        
        let index = (percentile / 100.0 * (sorted_times.len() - 1) as f64).round() as usize;
        sorted_times[index.min(sorted_times.len() - 1)] as f64
    }
    
    fn get_memory_usage_mb(&self) -> f64 {
        // Simplified memory usage estimation
        // In a real implementation, you would use system-specific memory monitoring
        100.0 // Placeholder
    }
    
    fn get_cpu_usage_percent(&self) -> f64 {
        // Simplified CPU usage estimation
        // In a real implementation, you would use system-specific CPU monitoring
        50.0 // Placeholder
    }
}

impl LoadTestSuite {
    pub fn new(config: LoadTestConfig) -> Self {
        LoadTestSuite {
            config,
            metrics: LoadTestMetrics::new(),
        }
    }
    
    /// Run comprehensive load test suite
    pub async fn run_load_tests(&mut self) -> Vec<LoadTestResult> {
        let mut results = Vec::new();
        
        // Test each scenario independently
        for scenario in &self.config.test_scenarios.clone() {
            println!("Running load test scenario: {}", scenario.name);
            
            let result = self.run_scenario_test(scenario).await;
            results.push(result);
            
            // Brief pause between scenarios
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        
        // Run combined load test
        println!("Running combined load test");
        let combined_result = self.run_combined_load_test().await;
        results.push(combined_result);
        
        results
    }
    
    async fn run_scenario_test(&self, scenario: &TestScenario) -> LoadTestResult {
        let metrics = LoadTestMetrics::new();
        let start_time = Instant::now();
        
        // Calculate number of requests for this scenario
        let scenario_requests = (self.config.total_requests as f32 * scenario.weight) as usize;
        
        // Create semaphore to limit concurrent requests
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent_requests));
        
        // Create NHITS service for testing
        let service = Arc::new(NHITSService::new(self.config.model_config.clone()));
        
        // Generate request tasks
        let mut tasks = Vec::new();
        
        for request_id in 0..scenario_requests {
            let semaphore = Arc::clone(&semaphore);
            let metrics = Arc::clone(&metrics);
            let service = Arc::clone(&service);
            let scenario = scenario.clone();
            let config = self.config.clone();
            
            let task = tokio::spawn(async move {
                // Acquire semaphore permit
                let _permit = semaphore.acquire().await.unwrap();
                
                // Add ramp-up delay
                let ramp_up_delay = config.ramp_up_duration.as_millis() as u64 * request_id as u64 / scenario_requests as u64;
                tokio::time::sleep(Duration::from_millis(ramp_up_delay)).await;
                
                // Execute request
                let request_start = Instant::now();
                let result = Self::execute_request(&service, &scenario.request_type, &config).await;
                let latency_ms = request_start.elapsed().as_millis() as u64;
                
                // Record metrics
                match result {
                    Ok(_) => {
                        metrics.record_request(latency_ms, true, None).await;
                    }
                    Err(error) => {
                        let error_type = format!("{:?}", error);
                        metrics.record_request(latency_ms, false, Some(error_type)).await;
                    }
                }
            });
            
            tasks.push(task);
        }
        
        // Wait for all tasks to complete or timeout
        let timeout_future = tokio::time::sleep(self.config.test_duration);
        tokio::select! {
            _ = join_all(tasks) => {
                println!("All requests completed for scenario: {}", scenario.name);
            }
            _ = timeout_future => {
                println!("Test duration exceeded for scenario: {}", scenario.name);
            }
        }
        
        let test_duration = start_time.elapsed();
        metrics.generate_result(scenario.name.clone(), test_duration).await
    }
    
    async fn run_combined_load_test(&self) -> LoadTestResult {
        let metrics = LoadTestMetrics::new();
        let start_time = Instant::now();
        
        // Create semaphore to limit concurrent requests
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent_requests));
        
        // Create NHITS service for testing
        let service = Arc::new(NHITSService::new(self.config.model_config.clone()));
        
        // Generate mixed workload based on scenario weights
        let mut tasks = Vec::new();
        let mut request_counter = 0;
        
        for scenario in &self.config.test_scenarios {
            let scenario_requests = (self.config.total_requests as f32 * scenario.weight) as usize;
            
            for _ in 0..scenario_requests {
                let semaphore = Arc::clone(&semaphore);
                let metrics = Arc::clone(&metrics);
                let service = Arc::clone(&service);
                let scenario = scenario.clone();
                let config = self.config.clone();
                let request_id = request_counter;
                request_counter += 1;
                
                let task = tokio::spawn(async move {
                    // Acquire semaphore permit
                    let _permit = semaphore.acquire().await.unwrap();
                    
                    // Add ramp-up delay
                    let ramp_up_delay = config.ramp_up_duration.as_millis() as u64 * request_id as u64 / config.total_requests as u64;
                    tokio::time::sleep(Duration::from_millis(ramp_up_delay)).await;
                    
                    // Execute request
                    let request_start = Instant::now();
                    let result = Self::execute_request(&service, &scenario.request_type, &config).await;
                    let latency_ms = request_start.elapsed().as_millis() as u64;
                    
                    // Record metrics
                    match result {
                        Ok(_) => {
                            metrics.record_request(latency_ms, true, None).await;
                        }
                        Err(error) => {
                            let error_type = format!("{:?}", error);
                            metrics.record_request(latency_ms, false, Some(error_type)).await;
                        }
                    }
                });
                
                tasks.push(task);
            }
        }
        
        // Wait for all tasks to complete or timeout
        let timeout_future = tokio::time::sleep(self.config.test_duration);
        tokio::select! {
            _ = join_all(tasks) => {
                println!("Combined load test completed");
            }
            _ = timeout_future => {
                println!("Combined load test duration exceeded");
            }
        }
        
        let test_duration = start_time.elapsed();
        metrics.generate_result("combined_load_test".to_string(), test_duration).await
    }
    
    async fn execute_request(
        service: &NHITSService,
        request_type: &RequestType,
        config: &LoadTestConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (input_size, output_size) = Self::get_payload_dimensions(&config.payload_size);
        
        match request_type {
            RequestType::Prediction => {
                let request = PredictionRequest {
                    input_data: vec![vec![1.0; input_size]],
                    model_id: Some("load_test_model".to_string()),
                    return_attention: false,
                    return_consciousness: false,
                };
                
                let response = tokio::time::timeout(
                    config.request_timeout,
                    service.predict(request)
                ).await??;
                
                if response.predictions.is_empty() {
                    return Err("Empty prediction response".into());
                }
                
                Ok(())
            }
            
            RequestType::BatchPrediction { batch_size } => {
                let request = PredictionRequest {
                    input_data: vec![vec![1.0; input_size]; *batch_size],
                    model_id: Some("load_test_model".to_string()),
                    return_attention: false,
                    return_consciousness: false,
                };
                
                let response = tokio::time::timeout(
                    config.request_timeout,
                    service.predict(request)
                ).await??;
                
                if response.predictions.len() != *batch_size {
                    return Err("Incorrect batch prediction response size".into());
                }
                
                Ok(())
            }
            
            RequestType::ConsciousnessPrediction => {
                let request = PredictionRequest {
                    input_data: vec![vec![1.0; input_size]],
                    model_id: Some("load_test_model".to_string()),
                    return_attention: true,
                    return_consciousness: true,
                };
                
                let response = tokio::time::timeout(
                    config.request_timeout,
                    service.predict(request)
                ).await??;
                
                if response.predictions.is_empty() || response.consciousness_state.is_none() {
                    return Err("Invalid consciousness prediction response".into());
                }
                
                Ok(())
            }
            
            RequestType::Training => {
                let request = TrainingRequest {
                    train_data: vec![vec![1.0; input_size]; 10],
                    train_targets: vec![vec![0.0; output_size]; 10],
                    val_data: Some(vec![vec![1.0; input_size]; 2]),
                    val_targets: Some(vec![vec![0.0; output_size]; 2]),
                    config_override: None,
                    model_id: "load_test_model".to_string(),
                };
                
                let response = tokio::time::timeout(
                    config.request_timeout.mul_f32(5.0), // Training takes longer
                    service.train(request)
                ).await??;
                
                if response.final_loss.is_nan() || response.final_loss.is_infinite() {
                    return Err("Invalid training response".into());
                }
                
                Ok(())
            }
            
            RequestType::ModelInfo => {
                // Simulate model info request (would be actual API call in real implementation)
                tokio::time::sleep(Duration::from_millis(5)).await;
                Ok(())
            }
            
            RequestType::HealthCheck => {
                // Simulate health check (would be actual API call in real implementation)
                tokio::time::sleep(Duration::from_millis(1)).await;
                Ok(())
            }
        }
    }
    
    fn get_payload_dimensions(payload_size: &PayloadSize) -> (usize, usize) {
        match payload_size {
            PayloadSize::Small => (24, 12),
            PayloadSize::Medium => (168, 24),
            PayloadSize::Large => (336, 48),
            PayloadSize::XLarge => (720, 168),
            PayloadSize::Custom { input_size, output_size } => (*input_size, *output_size),
        }
    }
    
    /// Run stress test with gradually increasing load
    pub async fn run_stress_test(&mut self) -> StressTestResult {
        println!("Starting stress test");
        
        let mut stress_results = Vec::new();
        let base_concurrent_requests = 10;
        let max_concurrent_requests = 200;
        let step_size = 20;
        let step_duration = Duration::from_secs(30);
        
        // Create NHITS service
        let service = Arc::new(NHITSService::new(self.config.model_config.clone()));
        
        for concurrent_requests in (base_concurrent_requests..=max_concurrent_requests).step_by(step_size) {
            println!("Testing with {} concurrent requests", concurrent_requests);
            
            let metrics = LoadTestMetrics::new();
            let start_time = Instant::now();
            
            // Create semaphore for this stress level
            let semaphore = Arc::new(Semaphore::new(concurrent_requests));
            let mut tasks = Vec::new();
            
            // Generate requests for this stress level
            let requests_per_step = concurrent_requests * 3; // 3x concurrent requests for sustained load
            
            for request_id in 0..requests_per_step {
                let semaphore = Arc::clone(&semaphore);
                let metrics = Arc::clone(&metrics);
                let service = Arc::clone(&service);
                let config = self.config.clone();
                
                let task = tokio::spawn(async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    
                    let request_start = Instant::now();
                    let result = Self::execute_request(&service, &RequestType::Prediction, &config).await;
                    let latency_ms = request_start.elapsed().as_millis() as u64;
                    
                    match result {
                        Ok(_) => metrics.record_request(latency_ms, true, None).await,
                        Err(error) => {
                            let error_type = format!("{:?}", error);
                            metrics.record_request(latency_ms, false, Some(error_type)).await;
                        }
                    }
                });
                
                tasks.push(task);
            }
            
            // Wait for step duration or all tasks to complete
            let timeout_future = tokio::time::sleep(step_duration);
            tokio::select! {
                _ = join_all(tasks) => {
                    println!("All requests completed for {} concurrent", concurrent_requests);
                }
                _ = timeout_future => {
                    println!("Step duration exceeded for {} concurrent", concurrent_requests);
                }
            }
            
            let step_duration_actual = start_time.elapsed();
            let step_result = metrics.generate_result(
                format!("stress_test_{}_concurrent", concurrent_requests),
                step_duration_actual
            ).await;
            
            stress_results.push(StressTestStepResult {
                concurrent_requests,
                result: step_result,
                breaking_point: false,
            });
            
            // Check if we've hit the breaking point
            let latest_result = &stress_results.last().unwrap().result;
            if latest_result.error_rate > 0.1 || latest_result.p95_latency_ms > 5000.0 {
                println!("Breaking point reached at {} concurrent requests", concurrent_requests);
                stress_results.last_mut().unwrap().breaking_point = true;
                break;
            }
            
            // Brief pause between stress levels
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        
        StressTestResult {
            step_results: stress_results,
            max_stable_concurrent_requests: self.find_max_stable_concurrent_requests(&stress_results),
            breaking_point_concurrent_requests: self.find_breaking_point(&stress_results),
        }
    }
    
    fn find_max_stable_concurrent_requests(&self, results: &[StressTestStepResult]) -> usize {
        results.iter()
            .filter(|r| r.result.error_rate < 0.05 && r.result.p95_latency_ms < 2000.0)
            .map(|r| r.concurrent_requests)
            .max()
            .unwrap_or(0)
    }
    
    fn find_breaking_point(&self, results: &[StressTestStepResult]) -> Option<usize> {
        results.iter()
            .find(|r| r.breaking_point)
            .map(|r| r.concurrent_requests)
    }
    
    /// Generate comprehensive load test report
    pub fn generate_report(&self, results: &[LoadTestResult]) -> String {
        let mut report = String::from("NHITS Load Test Report\n");
        report.push_str("======================\n\n");
        
        report.push_str(&format!("Test Configuration:\n"));
        report.push_str(&format!("  Max Concurrent Requests: {}\n", self.config.max_concurrent_requests));
        report.push_str(&format!("  Total Requests: {}\n", self.config.total_requests));
        report.push_str(&format!("  Test Duration: {:?}\n", self.config.test_duration));
        report.push_str(&format!("  Payload Size: {:?}\n", self.config.payload_size));
        report.push_str("\n");
        
        for result in results {
            report.push_str(&format!("Test: {}\n", result.test_name));
            report.push_str(&format!("  Total Requests: {}\n", result.total_requests));
            report.push_str(&format!("  Success Rate: {:.2}%\n", (1.0 - result.error_rate) * 100.0));
            report.push_str(&format!("  Avg Latency: {:.2}ms\n", result.avg_latency_ms));
            report.push_str(&format!("  P95 Latency: {:.2}ms\n", result.p95_latency_ms));
            report.push_str(&format!("  P99 Latency: {:.2}ms\n", result.p99_latency_ms));
            report.push_str(&format!("  Throughput: {:.2} RPS\n", result.throughput_rps));
            report.push_str(&format!("  Memory Usage: {:.2}MB\n", result.memory_usage_mb));
            report.push_str(&format!("  CPU Usage: {:.2}%\n", result.cpu_usage_percent));
            
            if !result.errors_by_type.is_empty() {
                report.push_str("  Errors:\n");
                for (error_type, count) in &result.errors_by_type {
                    report.push_str(&format!("    {}: {}\n", error_type, count));
                }
            }
            
            report.push_str("\n");
        }
        
        report.push_str(&format!("Report generated at: {}\n", chrono::Utc::now().to_rfc3339()));
        report
    }
}

#[derive(Debug, Clone)]
pub struct StressTestResult {
    pub step_results: Vec<StressTestStepResult>,
    pub max_stable_concurrent_requests: usize,
    pub breaking_point_concurrent_requests: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct StressTestStepResult {
    pub concurrent_requests: usize,
    pub result: LoadTestResult,
    pub breaking_point: bool,
}

#[cfg(test)]
mod load_tests {
    use super::*;

    #[tokio::test]
    async fn test_load_test_metrics() {
        let metrics = LoadTestMetrics::new();
        
        // Record some test metrics
        metrics.record_request(100, true, None).await;
        metrics.record_request(200, true, None).await;
        metrics.record_request(150, false, Some("timeout".to_string())).await;
        
        let result = metrics.generate_result("test".to_string(), Duration::from_secs(10)).await;
        
        assert_eq!(result.total_requests, 3);
        assert_eq!(result.successful_requests, 2);
        assert_eq!(result.failed_requests, 1);
        assert!(result.avg_latency_ms > 0.0);
        assert!(result.throughput_rps > 0.0);
    }

    #[tokio::test]
    async fn test_load_test_configuration() {
        let config = LoadTestConfig {
            max_concurrent_requests: 10,
            total_requests: 100,
            payload_size: PayloadSize::Small,
            ..Default::default()
        };
        
        let suite = LoadTestSuite::new(config);
        
        assert_eq!(suite.config.max_concurrent_requests, 10);
        assert_eq!(suite.config.total_requests, 100);
        
        let (input_size, output_size) = LoadTestSuite::get_payload_dimensions(&suite.config.payload_size);
        assert_eq!(input_size, 24);
        assert_eq!(output_size, 12);
    }

    #[tokio::test]
    async fn test_request_execution_mock() {
        let config = LoadTestConfig::default();
        let service = Arc::new(NHITSService::new(config.model_config.clone()));
        
        // This would normally test actual request execution
        // For now, we just test the function structure
        let result = LoadTestSuite::execute_request(
            &service,
            &RequestType::HealthCheck,
            &config
        ).await;
        
        // Health check should succeed (it's just a sleep)
        assert!(result.is_ok());
    }

    #[test]
    fn test_payload_size_dimensions() {
        assert_eq!(LoadTestSuite::get_payload_dimensions(&PayloadSize::Small), (24, 12));
        assert_eq!(LoadTestSuite::get_payload_dimensions(&PayloadSize::Medium), (168, 24));
        assert_eq!(LoadTestSuite::get_payload_dimensions(&PayloadSize::Large), (336, 48));
        assert_eq!(LoadTestSuite::get_payload_dimensions(&PayloadSize::XLarge), (720, 168));
        
        let custom = PayloadSize::Custom { input_size: 100, output_size: 50 };
        assert_eq!(LoadTestSuite::get_payload_dimensions(&custom), (100, 50));
    }

    #[test]
    fn test_scenario_weight_calculation() {
        let config = LoadTestConfig {
            total_requests: 1000,
            test_scenarios: vec![
                TestScenario {
                    name: "scenario1".to_string(),
                    weight: 0.7,
                    request_type: RequestType::Prediction,
                    expected_latency_ms: 100,
                    expected_throughput_rps: 50.0,
                },
                TestScenario {
                    name: "scenario2".to_string(),
                    weight: 0.3,
                    request_type: RequestType::BatchPrediction { batch_size: 5 },
                    expected_latency_ms: 200,
                    expected_throughput_rps: 20.0,
                },
            ],
            ..Default::default()
        };
        
        let scenario1_requests = (config.total_requests as f32 * config.test_scenarios[0].weight) as usize;
        let scenario2_requests = (config.total_requests as f32 * config.test_scenarios[1].weight) as usize;
        
        assert_eq!(scenario1_requests, 700);
        assert_eq!(scenario2_requests, 300);
    }

    #[test]
    fn test_load_test_report_generation() {
        let config = LoadTestConfig::default();
        let suite = LoadTestSuite::new(config);
        
        let results = vec![
            LoadTestResult {
                test_name: "test1".to_string(),
                total_requests: 100,
                successful_requests: 95,
                failed_requests: 5,
                timeout_requests: 0,
                avg_latency_ms: 150.0,
                p50_latency_ms: 120.0,
                p95_latency_ms: 250.0,
                p99_latency_ms: 300.0,
                max_latency_ms: 350,
                min_latency_ms: 50,
                throughput_rps: 10.0,
                error_rate: 0.05,
                errors_by_type: HashMap::new(),
                memory_usage_mb: 100.0,
                cpu_usage_percent: 50.0,
                test_duration: Duration::from_secs(10),
            }
        ];
        
        let report = suite.generate_report(&results);
        
        assert!(report.contains("NHITS Load Test Report"));
        assert!(report.contains("test1"));
        assert!(report.contains("Success Rate: 95.00%"));
        assert!(report.contains("Throughput: 10.00 RPS"));
    }
}