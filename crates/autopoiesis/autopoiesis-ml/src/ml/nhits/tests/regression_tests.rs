use super::*;
use crate::ml::nhits::{NHITSModel, NHITSConfig};
use crate::ml::nhits::consciousness::ConsciousnessIntegration;
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};

/// Comprehensive regression testing suite for NHITS models
pub struct RegressionTestSuite {
    reference_outputs: HashMap<String, ReferenceOutput>,
    test_configurations: Vec<TestConfiguration>,
    tolerance_config: ToleranceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceOutput {
    pub config_hash: String,
    pub input_hash: String,
    pub output: Array2<f32>,
    pub loss: f32,
    pub attention_weights: Option<Vec<Array3<f32>>>,
    pub consciousness_state: Option<HashMap<String, f32>>,
    pub metadata: HashMap<String, String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct TestDataConfig {
    pub sequence_length: usize,
    pub prediction_length: usize,
    pub num_samples: usize,
    pub input_features: usize,
}

#[derive(Debug, Clone)]
pub struct TestConfiguration {
    pub name: String,
    pub model_config: NHITSConfig,
    pub test_data_config: TestDataConfig,
    pub enable_consciousness: bool,
    pub training_steps: usize,
    pub expected_performance: PerformanceThresholds,
}

#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub max_loss: f32,
    pub min_r2_score: f32,
    pub max_inference_time_ms: f64,
    pub max_memory_mb: f64,
}

#[derive(Debug, Clone)]
pub struct ToleranceConfig {
    pub output_tolerance: f32,
    pub loss_tolerance: f32,
    pub attention_tolerance: f32,
    pub consciousness_tolerance: f32,
    pub performance_tolerance: f32,
}

#[derive(Debug, Clone)]
pub struct RegressionTestResult {
    pub test_name: String,
    pub passed: bool,
    pub output_match: bool,
    pub loss_match: bool,
    pub attention_match: bool,
    pub consciousness_match: bool,
    pub performance_match: bool,
    pub error_message: Option<String>,
    pub output_diff: f32,
    pub loss_diff: f32,
    pub execution_time: std::time::Duration,
    pub memory_usage: u64,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        ToleranceConfig {
            output_tolerance: 1e-5,
            loss_tolerance: 1e-6,
            attention_tolerance: 1e-4,
            consciousness_tolerance: 1e-4,
            performance_tolerance: 0.1, // 10% tolerance
        }
    }
}

impl RegressionTestSuite {
    pub fn new() -> Self {
        let mut suite = RegressionTestSuite {
            reference_outputs: HashMap::new(),
            test_configurations: Vec::new(),
            tolerance_config: ToleranceConfig::default(),
        };
        
        suite.setup_standard_tests();
        suite
    }
    
    pub fn with_tolerance(mut self, tolerance: ToleranceConfig) -> Self {
        self.tolerance_config = tolerance;
        self
    }
    
    fn generate_test_data(&self, config: &TestDataConfig) -> Array3<f32> {
        // Generate simple deterministic test data
        let mut data = Array3::<f32>::zeros((
            config.num_samples,
            config.sequence_length,
            config.input_features,
        ));
        
        // Fill with simple pattern for reproducibility
        for i in 0..config.num_samples {
            for j in 0..config.sequence_length {
                for k in 0..config.input_features {
                    data[[i, j, k]] = ((i + j + k) as f32).sin() * 0.5 + 0.5;
                }
            }
        }
        
        data
    }
    
    fn setup_standard_tests(&mut self) {
        // Basic functionality test
        self.test_configurations.push(TestConfiguration {
            name: "basic_forward_pass".to_string(),
            model_config: NHITSConfig {
                input_size: 168,
                output_size: 24,
                num_stacks: 2,
                num_blocks: vec![1, 1],
                num_layers: vec![2, 2],
                layer_widths: vec![128, 128],
                pooling_kernel_sizes: vec![2, 2],
                n_freq_downsample: vec![4, 2],
                dropout: 0.1,
                max_steps: 10,
                learning_rate: 1e-3,
                batch_size: 8,
                ..Default::default()
            },
            test_data_config: TestDataConfig {
                num_samples: 32,
                sequence_length: 168,
                prediction_length: 24,
                input_features: 1,
            },
            enable_consciousness: false,
            training_steps: 5,
            expected_performance: PerformanceThresholds {
                max_loss: 2.0,
                min_r2_score: -1.0, // Allow negative for untrained model
                max_inference_time_ms: 100.0,
                max_memory_mb: 100.0,
            },
        });
        
        // Consciousness integration test  
        self.test_configurations.push(TestConfiguration {
            name: "consciousness_integration".to_string(),
            model_config: NHITSConfig {
                input_size: 168,
                output_size: 24,
                num_stacks: 2,
                num_blocks: vec![1, 1],
                num_layers: vec![2, 2],
                layer_widths: vec![256, 256],
                pooling_kernel_sizes: vec![2, 2],
                n_freq_downsample: vec![4, 2],
                dropout: 0.1,
                max_steps: 10,
                learning_rate: 1e-3,
                batch_size: 4,
                ..Default::default()
            },
            test_data_config: TestDataConfig {
                num_samples: 16,
                sequence_length: 168,
                prediction_length: 24,
                input_features: 1,
            },
            enable_consciousness: true,
            training_steps: 5,
            expected_performance: PerformanceThresholds {
                max_loss: 2.5,
                min_r2_score: -1.0,
                max_inference_time_ms: 200.0,
                max_memory_mb: 150.0,
            },
        });
        
        // Complex pattern test
        self.test_configurations.push(TestConfiguration {
            name: "complex_pattern_learning".to_string(),
            model_config: NHITSConfig {
                input_size: 168,
                output_size: 24,
                num_stacks: 3,
                num_blocks: vec![1, 1, 1],
                num_layers: vec![3, 3, 3],
                layer_widths: vec![256, 256, 256],
                pooling_kernel_sizes: vec![2, 2, 2],
                n_freq_downsample: vec![8, 4, 2],
                dropout: 0.1,
                max_steps: 20,
                learning_rate: 1e-3,
                batch_size: 16,
                ..Default::default()
            },
            test_data_config: TestDataConfig {
                num_samples: 64,
                sequence_length: 168,
                prediction_length: 24,
                input_features: 1,
            },
            enable_consciousness: false,
            training_steps: 15,
            expected_performance: PerformanceThresholds {
                max_loss: 1.5,
                min_r2_score: 0.1,
                max_inference_time_ms: 150.0,
                max_memory_mb: 200.0,
            },
        });
        
        // Small model efficiency test
        self.test_configurations.push(TestConfiguration {
            name: "small_model_efficiency".to_string(),
            model_config: NHITSConfig {
                input_size: 24,
                output_size: 12,
                num_stacks: 1,
                num_blocks: vec![1],
                num_layers: vec![2],
                layer_widths: vec![64],
                pooling_kernel_sizes: vec![2],
                n_freq_downsample: vec![2],
                dropout: 0.0,
                max_steps: 10,
                learning_rate: 1e-2,
                batch_size: 32,
                ..Default::default()
            },
            test_data_config: TestDataConfig {
                num_samples: 128,
                sequence_length: 24,
                prediction_length: 12,
                input_features: 1,
            },
            enable_consciousness: false,
            training_steps: 10,
            expected_performance: PerformanceThresholds {
                max_loss: 1.0,
                min_r2_score: 0.3,
                max_inference_time_ms: 20.0,
                max_memory_mb: 20.0,
            },
        });
        
        // Batch size consistency test
        self.test_configurations.push(TestConfiguration {
            name: "batch_size_consistency".to_string(),
            model_config: NHITSConfig {
                input_size: 48,
                output_size: 12,
                num_stacks: 2,
                num_blocks: vec![1, 1],
                num_layers: vec![2, 2],
                layer_widths: vec![128, 128],
                pooling_kernel_sizes: vec![2, 2],
                n_freq_downsample: vec![4, 2],
                dropout: 0.1,
                max_steps: 5,
                learning_rate: 1e-3,
                batch_size: 1, // Single sample
                ..Default::default()
            },
            test_data_config: TestDataConfig {
                num_samples: 8,
                sequence_length: 48,
                prediction_length: 12,
                input_features: 1,
            },
            enable_consciousness: false,
            training_steps: 3,
            expected_performance: PerformanceThresholds {
                max_loss: 2.0,
                min_r2_score: -1.0,
                max_inference_time_ms: 50.0,
                max_memory_mb: 50.0,
            },
        });
    }
    
    /// Generate reference outputs for all test configurations
    pub fn generate_reference_outputs(&mut self) {
        for config in &self.test_configurations {
            println!("Generating reference output for: {}", config.name);
            
            let reference = self.generate_single_reference(config);
            self.reference_outputs.insert(config.name.clone(), reference);
        }
    }
    
    fn generate_single_reference(&self, config: &TestConfiguration) -> ReferenceOutput {
        // Generate deterministic test data
        let x_data = self.generate_test_data(&config.test_data_config);
        let y_data = Array2::<f32>::zeros((config.test_data_config.num_samples, config.test_data_config.prediction_length));
        
        // Create and configure model
        let mut model = NHITSModel::new(config.model_config.clone());
        if config.enable_consciousness {
            model.enable_consciousness(256, 8, 4);
        }
        
        // Perform training steps
        for _ in 0..config.training_steps {
            model.train_step(&x_data, &y_data);
        }
        
        // Generate reference output
        let output = model.forward(&x_data);
        let loss = model.compute_loss(&output, &y_data);
        
        // Extract attention weights if available
        let attention_weights = if model.has_attention() {
            let (_, attention) = model.forward_with_attention(&x_data);
            Some(attention)
        } else {
            None
        };
        
        // Extract consciousness state if enabled
        let consciousness_state = if config.enable_consciousness {
            model.get_consciousness_state()
        } else {
            None
        };
        
        // Compute hashes for reproducibility
        let config_hash = self.compute_config_hash(&config.model_config);
        let input_hash = self.compute_array_hash(&x_data);
        
        // Generate metadata
        let mut metadata = HashMap::new();
        metadata.insert("model_parameters".to_string(), model.get_parameters().len().to_string());
        metadata.insert("training_steps".to_string(), config.training_steps.to_string());
        metadata.insert("dataset_size".to_string(), x_data.len().to_string());
        
        ReferenceOutput {
            config_hash,
            input_hash,
            output,
            loss,
            attention_weights,
            consciousness_state,
            metadata,
            created_at: chrono::Utc::now(),
        }
    }
    
    /// Run all regression tests against reference outputs
    pub fn run_regression_tests(&self) -> Vec<RegressionTestResult> {
        let mut results = Vec::new();
        
        for config in &self.test_configurations {
            println!("Running regression test: {}", config.name);
            
            if let Some(reference) = self.reference_outputs.get(&config.name) {
                let result = self.run_single_regression_test(config, reference);
                results.push(result);
            } else {
                results.push(RegressionTestResult {
                    test_name: config.name.clone(),
                    passed: false,
                    output_match: false,
                    loss_match: false,
                    attention_match: false,
                    consciousness_match: false,
                    performance_match: false,
                    error_message: Some("No reference output found".to_string()),
                    output_diff: f32::INFINITY,
                    loss_diff: f32::INFINITY,
                    execution_time: std::time::Duration::from_secs(0),
                    memory_usage: 0,
                });
            }
        }
        
        results
    }
    
    fn run_single_regression_test(
        &self,
        config: &TestConfiguration,
        reference: &ReferenceOutput,
    ) -> RegressionTestResult {
        let start_time = std::time::Instant::now();
        let start_memory = self.get_memory_usage();
        
        // Generate same test data as reference
        let x_data = self.generate_test_data(&config.test_data_config);
        let y_data = Array2::<f32>::zeros((config.test_data_config.num_samples, config.test_data_config.prediction_length));
        
        // Verify input consistency
        let input_hash = self.compute_array_hash(&x_data);
        if input_hash != reference.input_hash {
            return RegressionTestResult {
                test_name: config.name.clone(),
                passed: false,
                output_match: false,
                loss_match: false,
                attention_match: false,
                consciousness_match: false,
                performance_match: false,
                error_message: Some("Input data hash mismatch".to_string()),
                output_diff: f32::INFINITY,
                loss_diff: f32::INFINITY,
                execution_time: start_time.elapsed(),
                memory_usage: 0,
            };
        }
        
        // Create and configure model
        let mut model = NHITSModel::new(config.model_config.clone());
        if config.enable_consciousness {
            model.enable_consciousness(256, 8, 4);
        }
        
        // Perform same training steps
        for _ in 0..config.training_steps {
            model.train_step(&x_data, &y_data);
        }
        
        // Generate current output
        let current_output = model.forward(&x_data);
        let current_loss = model.compute_loss(&current_output, &y_data);
        
        // Compare outputs
        let output_match = self.compare_arrays(&current_output, &reference.output, self.tolerance_config.output_tolerance);
        let output_diff = self.compute_array_diff(&current_output, &reference.output);
        
        // Compare losses
        let loss_match = (current_loss - reference.loss).abs() < self.tolerance_config.loss_tolerance;
        let loss_diff = (current_loss - reference.loss).abs();
        
        // Compare attention weights if available
        let attention_match = if let Some(ref reference_attention) = reference.attention_weights {
            if model.has_attention() {
                let (_, current_attention) = model.forward_with_attention(&dataset.x_data);
                self.compare_attention_weights(&current_attention, reference_attention)
            } else {
                false
            }
        } else {
            true // No attention to compare
        };
        
        // Compare consciousness state if enabled
        let consciousness_match = if let Some(ref reference_consciousness) = reference.consciousness_state {
            if let Some(current_consciousness) = model.get_consciousness_state() {
                self.compare_consciousness_states(&current_consciousness, reference_consciousness)
            } else {
                false
            }
        } else {
            true // No consciousness to compare
        };
        
        // Check performance thresholds
        let execution_time = start_time.elapsed();
        let memory_usage = self.get_memory_usage() - start_memory;
        
        let performance_match = 
            current_loss <= config.expected_performance.max_loss &&
            execution_time.as_millis() as f64 <= config.expected_performance.max_inference_time_ms &&
            memory_usage as f64 / 1024.0 / 1024.0 <= config.expected_performance.max_memory_mb;
        
        // Overall pass/fail
        let passed = output_match && loss_match && attention_match && consciousness_match && performance_match;
        
        let error_message = if !passed {
            let mut errors = Vec::new();
            if !output_match { errors.push("Output mismatch"); }
            if !loss_match { errors.push("Loss mismatch"); }
            if !attention_match { errors.push("Attention mismatch"); }
            if !consciousness_match { errors.push("Consciousness mismatch"); }
            if !performance_match { errors.push("Performance threshold exceeded"); }
            Some(errors.join(", "))
        } else {
            None
        };
        
        RegressionTestResult {
            test_name: config.name.clone(),
            passed,
            output_match,
            loss_match,
            attention_match,
            consciousness_match,
            performance_match,
            error_message,
            output_diff,
            loss_diff,
            execution_time,
            memory_usage,
        }
    }
    
    /// Save reference outputs to disk for persistence
    pub fn save_reference_outputs(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json_data = serde_json::to_string_pretty(&self.reference_outputs)?;
        std::fs::write(path, json_data)?;
        Ok(())
    }
    
    /// Load reference outputs from disk
    pub fn load_reference_outputs(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json_data = std::fs::read_to_string(path)?;
        self.reference_outputs = serde_json::from_str(&json_data)?;
        Ok(())
    }
    
    // Helper methods
    fn compare_arrays(&self, a: &Array2<f32>, b: &Array2<f32>, tolerance: f32) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        
        for (val_a, val_b) in a.iter().zip(b.iter()) {
            if (val_a - val_b).abs() > tolerance {
                return false;
            }
        }
        
        true
    }
    
    fn compute_array_diff(&self, a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        if a.shape() != b.shape() {
            return f32::INFINITY;
        }
        
        let diff_sum: f32 = a.iter().zip(b.iter())
            .map(|(val_a, val_b)| (val_a - val_b).abs())
            .sum();
        
        diff_sum / a.len() as f32
    }
    
    fn compare_attention_weights(&self, current: &[Array3<f32>], reference: &[Array3<f32>]) -> bool {
        if current.len() != reference.len() {
            return false;
        }
        
        for (curr_attn, ref_attn) in current.iter().zip(reference.iter()) {
            if curr_attn.shape() != ref_attn.shape() {
                return false;
            }
            
            for (curr_val, ref_val) in curr_attn.iter().zip(ref_attn.iter()) {
                if (curr_val - ref_val).abs() > self.tolerance_config.attention_tolerance {
                    return false;
                }
            }
        }
        
        true
    }
    
    fn compare_consciousness_states(
        &self,
        current: &HashMap<String, f32>,
        reference: &HashMap<String, f32>,
    ) -> bool {
        if current.len() != reference.len() {
            return false;
        }
        
        for (key, ref_val) in reference.iter() {
            if let Some(&curr_val) = current.get(key) {
                if (curr_val - ref_val).abs() > self.tolerance_config.consciousness_tolerance {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        true
    }
    
    fn compute_config_hash(&self, config: &NHITSConfig) -> String {
        let config_str = format!(
            "{}_{}_{}_{:?}_{:?}_{:?}_{:?}_{:?}_{}_{}_{}_{}",
            config.input_size,
            config.output_size,
            config.num_stacks,
            config.num_blocks,
            config.num_layers,
            config.layer_widths,
            config.pooling_kernel_sizes,
            config.n_freq_downsample,
            config.dropout,
            config.learning_rate,
            config.batch_size,
            config.activation
        );
        
        let mut hasher = Sha256::new();
        hasher.update(config_str.as_bytes());
        format!("{:x}", hasher.finalize())
    }
    
    fn compute_array_hash(&self, array: &Array2<f32>) -> String {
        let bytes: Vec<u8> = array.iter()
            .flat_map(|&f| f.to_be_bytes().to_vec())
            .collect();
        
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        format!("{:x}", hasher.finalize())
    }
    
    fn get_memory_usage(&self) -> u64 {
        // Simplified memory usage estimation
        // In a real implementation, you would use system-specific memory tracking
        use std::alloc::{GlobalAlloc, Layout, System};
        
        // This is a placeholder - actual memory tracking would be more sophisticated
        0
    }
    
    /// Generate a comprehensive test report
    pub fn generate_report(&self, results: &[RegressionTestResult]) -> String {
        let mut report = String::from("NHITS Regression Test Report\n");
        report.push_str("=====================================\n\n");
        
        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;
        
        report.push_str(&format!("Total Tests: {}\n", total_tests));
        report.push_str(&format!("Passed: {} ({:.1}%)\n", passed_tests, passed_tests as f32 / total_tests as f32 * 100.0));
        report.push_str(&format!("Failed: {} ({:.1}%)\n\n", failed_tests, failed_tests as f32 / total_tests as f32 * 100.0));
        
        report.push_str("Detailed Results:\n");
        report.push_str("-----------------\n");
        
        for result in results {
            let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
            report.push_str(&format!("{} {}\n", status, result.test_name));
            
            if !result.passed {
                if let Some(ref error) = result.error_message {
                    report.push_str(&format!("  Error: {}\n", error));
                }
                report.push_str(&format!("  Output diff: {:.6}\n", result.output_diff));
                report.push_str(&format!("  Loss diff: {:.6}\n", result.loss_diff));
            }
            
            report.push_str(&format!("  Execution time: {:.2}ms\n", result.execution_time.as_millis()));
            report.push_str(&format!("  Memory usage: {:.2}MB\n", result.memory_usage as f64 / 1024.0 / 1024.0));
            report.push_str("\n");
        }
        
        // Performance summary
        report.push_str("Performance Summary:\n");
        report.push_str("--------------------\n");
        
        let avg_execution_time = results.iter()
            .map(|r| r.execution_time.as_millis() as f64)
            .sum::<f64>() / results.len() as f64;
        
        let avg_memory_usage = results.iter()
            .map(|r| r.memory_usage as f64)
            .sum::<f64>() / results.len() as f64;
        
        report.push_str(&format!("Average execution time: {:.2}ms\n", avg_execution_time));
        report.push_str(&format!("Average memory usage: {:.2}MB\n", avg_memory_usage / 1024.0 / 1024.0));
        
        report.push_str(&format!("\nReport generated at: {}\n", chrono::Utc::now().to_rfc3339()));
        
        report
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn test_regression_test_suite_creation() {
        let suite = RegressionTestSuite::new();
        
        assert!(!suite.test_configurations.is_empty());
        assert_eq!(suite.tolerance_config.output_tolerance, 1e-5);
    }

    #[test]
    fn test_reference_output_generation() {
        let mut suite = RegressionTestSuite::new();
        
        // Generate reference for first test only
        if let Some(config) = suite.test_configurations.first() {
            let reference = suite.generate_single_reference(config);
            
            assert!(!reference.config_hash.is_empty());
            assert!(!reference.input_hash.is_empty());
            assert_eq!(reference.output.shape()[1], config.model_config.output_size);
            assert!(reference.loss.is_finite());
        }
    }

    #[test]
    fn test_array_comparison() {
        let suite = RegressionTestSuite::new();
        
        let array1 = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let array2 = Array2::from_shape_vec((2, 3), vec![1.0001, 2.0001, 3.0001, 4.0001, 5.0001, 6.0001]).unwrap();
        let array3 = Array2::from_shape_vec((2, 3), vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1]).unwrap();
        
        // Should match with high tolerance
        assert!(suite.compare_arrays(&array1, &array2, 1e-3));
        
        // Should not match with low tolerance
        assert!(!suite.compare_arrays(&array1, &array2, 1e-5));
        
        // Should not match with significant difference
        assert!(!suite.compare_arrays(&array1, &array3, 1e-3));
    }

    #[test]
    fn test_config_hash_consistency() {
        let suite = RegressionTestSuite::new();
        
        let config1 = NHITSConfig {
            input_size: 168,
            output_size: 24,
            ..Default::default()
        };
        
        let config2 = config1.clone();
        
        let config3 = NHITSConfig {
            input_size: 169, // Different
            output_size: 24,
            ..Default::default()
        };
        
        let hash1 = suite.compute_config_hash(&config1);
        let hash2 = suite.compute_config_hash(&config2);
        let hash3 = suite.compute_config_hash(&config3);
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_array_hash_consistency() {
        let suite = RegressionTestSuite::new();
        
        let array1 = Array2::ones((3, 4));
        let array2 = Array2::ones((3, 4));
        let array3 = Array2::zeros((3, 4));
        
        let hash1 = suite.compute_array_hash(&array1);
        let hash2 = suite.compute_array_hash(&array2);
        let hash3 = suite.compute_array_hash(&array3);
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_regression_test_with_simple_model() {
        let mut suite = RegressionTestSuite::new();
        
        // Create a simple test configuration
        let simple_config = TestConfiguration {
            name: "simple_test".to_string(),
            model_config: NHITSConfig {
                input_size: 10,
                output_size: 5,
                num_stacks: 1,
                num_blocks: vec![1],
                num_layers: vec![1],
                layer_widths: vec![32],
                pooling_kernel_sizes: vec![1],
                n_freq_downsample: vec![1],
                dropout: 0.0,
                max_steps: 2,
                learning_rate: 1e-2,
                batch_size: 4,
                ..Default::default()
            },
            test_data_config: TestDataConfig {
                num_samples: 8,
                sequence_length: 10,
                prediction_length: 5,
                input_features: 1,
            },
            enable_consciousness: false,
            training_steps: 1,
            expected_performance: PerformanceThresholds {
                max_loss: 10.0,
                min_r2_score: -2.0,
                max_inference_time_ms: 1000.0,
                max_memory_mb: 100.0,
            },
        };
        
        // Generate reference
        let reference = suite.generate_single_reference(&simple_config);
        
        // Run test against itself (should pass)
        let result = suite.run_single_regression_test(&simple_config, &reference);
        
        assert!(result.passed, "Regression test should pass when run against its own reference");
        assert!(result.output_match);
        assert!(result.loss_match);
    }

    #[test]
    fn test_tolerance_configuration() {
        let strict_tolerance = ToleranceConfig {
            output_tolerance: 1e-8,
            loss_tolerance: 1e-9,
            attention_tolerance: 1e-7,
            consciousness_tolerance: 1e-7,
            performance_tolerance: 0.01,
        };
        
        let suite = RegressionTestSuite::new().with_tolerance(strict_tolerance);
        
        assert_eq!(suite.tolerance_config.output_tolerance, 1e-8);
        assert_eq!(suite.tolerance_config.loss_tolerance, 1e-9);
    }
}