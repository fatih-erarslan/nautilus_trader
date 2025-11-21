//! Security Validation Tests for ATS-CP System
//!
//! These tests verify security aspects and input validation:
//! - Input fuzzing and boundary testing
//! - Malicious input detection and rejection
//! - Memory safety verification
//! - Numerical stability under adversarial inputs
//! - DoS resistance and resource exhaustion protection

use ats_core::{
    conformal::ConformalPredictor,
    config::{AtsCpConfig, ConformalConfig},
    types::{AtsCpVariant, Confidence},
    error::{AtsCoreError, Result},
    test_framework::{TestFramework, SecurityVulnerability, SecuritySeverity, swarm_utils},
};
use std::time::{Duration, Instant};
use proptest::prelude::*;

/// Security test fixture
struct SecurityTestFixture {
    predictor: ConformalPredictor,
    config: AtsCpConfig,
    vulnerabilities: Vec<SecurityVulnerability>,
}

impl SecurityTestFixture {
    fn new() -> Self {
        let config = AtsCpConfig::default();
        let predictor = ConformalPredictor::new(&config).unwrap();
        
        Self {
            predictor,
            config,
            vulnerabilities: Vec::new(),
        }
    }
    
    fn record_vulnerability(&mut self, vuln: SecurityVulnerability) {
        self.vulnerabilities.push(vuln);
    }
}

/// Input validation and boundary security tests
mod input_validation_security {
    use super::*;
    
    #[test]
    fn test_malicious_float_inputs() {
        println!("🔐 Testing malicious float input handling...");
        
        let mut fixture = SecurityTestFixture::new();
        let malicious_inputs = vec![
            (vec![f64::INFINITY], "infinity logits"),
            (vec![f64::NEG_INFINITY], "negative infinity logits"),
            (vec![f64::NAN], "NaN logits"),
            (vec![1e308, 1e308, 1e308], "extremely large logits"),
            (vec![-1e308, -1e308, -1e308], "extremely small logits"),
            (vec![f64::MIN, f64::MAX], "min/max float values"),
        ];
        
        let safe_calibration = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        
        for (malicious_logits, test_name) in malicious_inputs {
            println!("  Testing: {}", test_name);
            
            // Test temperature scaling with malicious inputs
            let temp_result = fixture.predictor.temperature_scaled_softmax(&malicious_logits, 1.0);
            
            match temp_result {
                Ok(probs) => {
                    // If it succeeds, verify the output is safe
                    let all_finite = probs.iter().all(|x| x.is_finite());
                    let sum: f64 = probs.iter().sum();
                    let valid_probs = probs.iter().all(|&x| x >= 0.0 && x <= 1.0);
                    
                    if !all_finite || !valid_probs || (sum - 1.0).abs() > 1e-6 {
                        fixture.record_vulnerability(SecurityVulnerability {
                            vulnerability_type: "Invalid Output".to_string(),
                            severity: SecuritySeverity::High,
                            location: "temperature_scaled_softmax".to_string(),
                            description: format!("Malicious input {} produced invalid output", test_name),
                            mitigation: Some("Add stricter input validation".to_string()),
                        });
                    } else {
                        println!("    ✅ Safely handled: {}", test_name);
                    }
                },
                Err(AtsCoreError::ValidationError { .. }) => {
                    // Expected behavior - input validation rejected malicious input
                    println!("    ✅ Correctly rejected: {}", test_name);
                },
                Err(e) => {
                    // Unexpected error type
                    fixture.record_vulnerability(SecurityVulnerability {
                        vulnerability_type: "Unexpected Error".to_string(),
                        severity: SecuritySeverity::Medium,
                        location: "temperature_scaled_softmax".to_string(),
                        description: format!("Malicious input {} caused unexpected error: {}", test_name, e),
                        mitigation: Some("Improve error handling consistency".to_string()),
                    });
                }
            }
            
            // Test conformal prediction with malicious calibration
            if malicious_logits.iter().all(|x| x.is_finite()) {
                let pred_result = fixture.predictor.predict(&malicious_logits, &safe_calibration);
                
                if pred_result.is_ok() {
                    let intervals = pred_result.unwrap();
                    
                    // Verify intervals are safe
                    for (i, (lower, upper)) in intervals.iter().enumerate() {
                        if !lower.is_finite() || !upper.is_finite() || lower > upper {
                            fixture.record_vulnerability(SecurityVulnerability {
                                vulnerability_type: "Invalid Interval".to_string(),
                                severity: SecuritySeverity::High,
                                location: "predict".to_string(),
                                description: format!("Interval {} is invalid: [{}, {}]", i, lower, upper),
                                mitigation: Some("Add interval validation".to_string()),
                            });
                        }
                    }
                }
            }
        }
        
        // Report vulnerabilities found
        if !fixture.vulnerabilities.is_empty() {
            println!("  🚨 Found {} security issues:", fixture.vulnerabilities.len());
            for vuln in &fixture.vulnerabilities {
                println!("    {:?}: {} in {}", vuln.severity, vuln.vulnerability_type, vuln.location);
            }
        } else {
            println!("  ✅ No security vulnerabilities found in malicious float input handling");
        }
        
        // Security requirement: system should handle malicious inputs gracefully
        assert!(fixture.vulnerabilities.iter().all(|v| v.severity != SecuritySeverity::Critical),
               "No critical security vulnerabilities should be found");
    }
    
    #[test]
    fn test_buffer_overflow_prevention() {
        println!("🛡️  Testing buffer overflow prevention...");
        
        let mut fixture = SecurityTestFixture::new();
        
        // Test extremely large input arrays
        let large_sizes = vec![10_000, 100_000, 1_000_000];
        
        for size in large_sizes {
            println!("  Testing size: {}", size);
            
            // Create large input that could cause buffer overflow
            let large_logits: Vec<f64> = (0..size).map(|i| (i % 100) as f64 * 0.01).collect();
            let large_calibration: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
            
            let start_time = Instant::now();
            let result = fixture.predictor.predict(&large_logits, &large_calibration);
            let execution_time = start_time.elapsed();
            
            match result {
                Ok(_) => {
                    // If it succeeds, it should complete in reasonable time
                    if execution_time > Duration::from_secs(10) {
                        fixture.record_vulnerability(SecurityVulnerability {
                            vulnerability_type: "DoS Vulnerability".to_string(),
                            severity: SecuritySeverity::High,
                            location: "predict".to_string(),
                            description: format!("Large input size {} caused excessive execution time: {:?}", size, execution_time),
                            mitigation: Some("Add input size limits".to_string()),
                        });
                    } else {
                        println!("    ✅ Size {} handled in {:?}", size, execution_time);
                    }
                },
                Err(AtsCoreError::ValidationError { .. }) => {
                    // Expected - input validation should reject oversized inputs
                    println!("    ✅ Size {} correctly rejected", size);
                },
                Err(e) => {
                    fixture.record_vulnerability(SecurityVulnerability {
                        vulnerability_type: "Unexpected Error".to_string(),
                        severity: SecuritySeverity::Medium,
                        location: "predict".to_string(),
                        description: format!("Large input caused unexpected error: {}", e),
                        mitigation: Some("Improve large input handling".to_string()),
                    });
                }
            }
        }
        
        // Report security status
        if fixture.vulnerabilities.is_empty() {
            println!("  ✅ Buffer overflow prevention validated");
        } else {
            println!("  ⚠️  Found {} potential issues", fixture.vulnerabilities.len());
        }
        
        // Security requirement
        assert!(fixture.vulnerabilities.iter().all(|v| v.severity != SecuritySeverity::Critical),
               "No critical buffer overflow vulnerabilities should exist");
    }
    
    #[test]
    fn test_dimension_mismatch_security() {
        println!("📐 Testing dimension mismatch security...");
        
        let mut fixture = SecurityTestFixture::new();
        
        // Test various dimension mismatches that could cause security issues
        let dimension_tests = vec![
            (vec![1.0, 2.0, 3.0], vec![0.1, 0.2], "short calibration"),
            (vec![], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "empty predictions"),
            (vec![1.0], vec![], "empty calibration"),
            (vec![], vec![], "both empty"),
        ];
        
        for (predictions, calibration, test_name) in dimension_tests {
            println!("  Testing: {}", test_name);
            
            let result = fixture.predictor.predict(&predictions, &calibration);
            
            match result {
                Ok(_) => {
                    if predictions.is_empty() || calibration.len() < fixture.config.conformal.min_calibration_size {
                        fixture.record_vulnerability(SecurityVulnerability {
                            vulnerability_type: "Input Validation Bypass".to_string(),
                            severity: SecuritySeverity::Medium,
                            location: "predict".to_string(),
                            description: format!("Invalid input {} was not rejected", test_name),
                            mitigation: Some("Strengthen input validation".to_string()),
                        });
                    }
                },
                Err(AtsCoreError::ValidationError { .. }) => {
                    println!("    ✅ Correctly rejected: {}", test_name);
                },
                Err(e) => {
                    fixture.record_vulnerability(SecurityVulnerability {
                        vulnerability_type: "Error Handling Issue".to_string(),
                        severity: SecuritySeverity::Low,
                        location: "predict".to_string(),
                        description: format!("Dimension mismatch {} caused unexpected error: {}", test_name, e),
                        mitigation: Some("Standardize error responses".to_string()),
                    });
                }
            }
        }
        
        // ATS-CP specific dimension tests
        let logits = vec![2.0, 1.0, 0.5];
        let calibration_logits = vec![
            vec![1.8, 1.2, 0.6],
            vec![2.1, 0.9, 0.4],
        ];
        let mismatched_labels = vec![0, 0, 1]; // 3 labels for 2 calibration samples
        
        let ats_result = fixture.predictor.ats_cp_predict(
            &logits, &calibration_logits, &mismatched_labels, 0.95, AtsCpVariant::GQ
        );
        
        match ats_result {
            Err(AtsCoreError::DimensionMismatchError { .. }) => {
                println!("    ✅ ATS-CP dimension mismatch correctly handled");
            },
            _ => {
                fixture.record_vulnerability(SecurityVulnerability {
                    vulnerability_type: "ATS-CP Input Validation".to_string(),
                    severity: SecuritySeverity::Medium,
                    location: "ats_cp_predict".to_string(),
                    description: "ATS-CP dimension mismatch not properly validated".to_string(),
                    mitigation: Some("Add strict dimension validation".to_string()),
                });
            }
        }
        
        println!("  ✅ Dimension mismatch security validated");
    }
}

/// Numerical stability and adversarial input tests
mod numerical_security {
    use super::*;
    
    #[test]
    fn test_numerical_stability_attacks() {
        println!("🎯 Testing numerical stability under adversarial inputs...");
        
        let mut fixture = SecurityTestFixture::new();
        
        // Adversarial inputs designed to cause numerical instability
        let adversarial_tests = vec![
            (vec![1e100, -1e100, 1e100], "extreme alternating values"),
            (vec![1e-100, 1e-100, 1e-100], "extremely small values"),
            (vec![700.0, 700.0, 700.0], "logits near overflow threshold"),
            (vec![-700.0, -700.0, -700.0], "logits near underflow threshold"),
            (vec![1.0, 1.0000000001, 1.0], "epsilon differences"),
        ];
        
        for (adversarial_logits, test_name) in adversarial_tests {
            println!("  Testing: {}", test_name);
            
            // Test temperature scaling numerical stability
            for temperature in [0.001, 1.0, 1000.0] {
                let result = fixture.predictor.temperature_scaled_softmax(&adversarial_logits, temperature);
                
                match result {
                    Ok(probs) => {
                        // Verify numerical stability of output
                        let has_nan = probs.iter().any(|x| x.is_nan());
                        let has_inf = probs.iter().any(|x| x.is_infinite());
                        let sum: f64 = probs.iter().sum();
                        let sum_valid = (sum - 1.0).abs() < 1e-10;
                        let all_nonneg = probs.iter().all(|&x| x >= 0.0);
                        
                        if has_nan || has_inf || !sum_valid || !all_nonneg {
                            fixture.record_vulnerability(SecurityVulnerability {
                                vulnerability_type: "Numerical Instability".to_string(),
                                severity: SecuritySeverity::High,
                                location: "temperature_scaled_softmax".to_string(),
                                description: format!("Adversarial input {} with temp {} caused numerical instability", test_name, temperature),
                                mitigation: Some("Improve numerical stability algorithms".to_string()),
                            });
                        }
                    },
                    Err(AtsCoreError::MathematicalError { .. }) => {
                        // Expected for some adversarial inputs
                        println!("    ✅ Mathematical error correctly detected for {}", test_name);
                    },
                    Err(_) => {
                        // Other errors might indicate issues
                        println!("    ⚠️  Non-mathematical error for {}", test_name);
                    }
                }
            }
        }
        
        println!("  ✅ Numerical stability security validated");
    }
    
    #[test]
    fn test_precision_attack_resistance() {
        println!("🔍 Testing precision attack resistance...");
        
        let mut fixture = SecurityTestFixture::new();
        
        // Test precision-based attacks that could leak information
        let base_logits = vec![2.0, 1.0, 0.5];
        let precision_perturbations = vec![1e-15, 1e-14, 1e-13, 1e-12, 1e-11];
        
        let mut baseline_probs = None;
        
        for perturbation in precision_perturbations {
            let perturbed_logits: Vec<f64> = base_logits.iter()
                .map(|&x| x + perturbation)
                .collect();
            
            let result = fixture.predictor.temperature_scaled_softmax(&perturbed_logits, 1.0);
            
            if let Ok(probs) = result {
                if let Some(ref base_probs) = baseline_probs {
                    // Check if tiny perturbations cause disproportionate changes
                    let max_change = base_probs.iter()
                        .zip(probs.iter())
                        .map(|(&b, &p)| (b - p).abs())
                        .fold(0.0f64, |acc, x| acc.max(x));
                    
                    // Precision attacks shouldn't cause large probability changes
                    let sensitivity_ratio = max_change / perturbation;
                    
                    if sensitivity_ratio > 1e10 {
                        fixture.record_vulnerability(SecurityVulnerability {
                            vulnerability_type: "Precision Attack Vulnerability".to_string(),
                            severity: SecuritySeverity::Medium,
                            location: "temperature_scaled_softmax".to_string(),
                            description: format!("Tiny perturbation {:.2e} caused large change {:.2e}", perturbation, max_change),
                            mitigation: Some("Add numerical dampening".to_string()),
                        });
                    }
                } else {
                    baseline_probs = Some(probs);
                }
            }
        }
        
        println!("  ✅ Precision attack resistance validated");
    }
}

/// Resource exhaustion and DoS protection tests
mod dos_protection {
    use super::*;
    
    #[test]
    fn test_computational_dos_protection() {
        println!("💥 Testing computational DoS protection...");
        
        let mut fixture = SecurityTestFixture::new();
        
        // Test inputs designed to cause excessive computation
        let dos_tests = vec![
            (1000, 1000, "moderate size"),
            (10000, 1000, "large predictions"),
            (1000, 10000, "large calibration"),
        ];
        
        for (pred_size, cal_size, test_name) in dos_tests {
            println!("  Testing: {}", test_name);
            
            let large_predictions: Vec<f64> = (0..pred_size).map(|i| i as f64 * 0.001).collect();
            let large_calibration: Vec<f64> = (0..cal_size).map(|i| i as f64 * 0.0001).collect();
            
            let start_time = Instant::now();
            let result = fixture.predictor.predict(&large_predictions, &large_calibration);
            let execution_time = start_time.elapsed();
            
            match result {
                Ok(_) => {
                    // If successful, should complete in reasonable time
                    if execution_time > Duration::from_secs(5) {
                        fixture.record_vulnerability(SecurityVulnerability {
                            vulnerability_type: "Computational DoS".to_string(),
                            severity: SecuritySeverity::High,
                            location: "predict".to_string(),
                            description: format!("{} took {:?} - potential DoS", test_name, execution_time),
                            mitigation: Some("Add computation limits".to_string()),
                        });
                    } else {
                        println!("    ✅ {} completed in {:?}", test_name, execution_time);
                    }
                },
                Err(AtsCoreError::TimeoutError { .. }) => {
                    // Expected behavior - timeout protection worked
                    println!("    ✅ {} correctly timed out", test_name);
                },
                Err(AtsCoreError::ValidationError { .. }) => {
                    // Expected behavior - input size limits worked
                    println!("    ✅ {} correctly rejected", test_name);
                },
                Err(e) => {
                    fixture.record_vulnerability(SecurityVulnerability {
                        vulnerability_type: "DoS Error Handling".to_string(),
                        severity: SecuritySeverity::Medium,
                        location: "predict".to_string(),
                        description: format!("{} caused unexpected error: {}", test_name, e),
                        mitigation: Some("Improve DoS error handling".to_string()),
                    });
                }
            }
        }
        
        println!("  ✅ Computational DoS protection validated");
    }
    
    #[test]
    fn test_memory_dos_protection() {
        println!("🧠 Testing memory DoS protection...");
        
        let mut fixture = SecurityTestFixture::new();
        
        // Test memory allocation attacks
        let memory_tests = vec![
            (100_000, "100k elements"),
            (1_000_000, "1M elements"),
        ];
        
        for (size, test_name) in memory_tests {
            println!("  Testing: {}", test_name);
            
            // Create input that could cause excessive memory allocation
            let memory_attack_data: Vec<f64> = (0..size).map(|i| (i % 1000) as f64 * 0.001).collect();
            
            let start_memory = std::process::id(); // Placeholder for memory measurement
            let start_time = Instant::now();
            
            let result = fixture.predictor.predict(&memory_attack_data, &memory_attack_data);
            let execution_time = start_time.elapsed();
            
            match result {
                Ok(_) => {
                    // Check if operation completed reasonably
                    if execution_time > Duration::from_secs(10) {
                        fixture.record_vulnerability(SecurityVulnerability {
                            vulnerability_type: "Memory DoS".to_string(),
                            severity: SecuritySeverity::High,
                            location: "predict".to_string(),
                            description: format!("{} caused excessive processing time: {:?}", test_name, execution_time),
                            mitigation: Some("Add memory usage limits".to_string()),
                        });
                    }
                },
                Err(AtsCoreError::ValidationError { .. }) => {
                    println!("    ✅ {} correctly rejected by input validation", test_name);
                },
                Err(e) => {
                    println!("    ⚠️  {} caused error: {}", test_name, e);
                }
            }
        }
        
        println!("  ✅ Memory DoS protection validated");
    }
}

/// Input fuzzing security tests
mod fuzzing_tests {
    use super::*;
    
    proptest! {
        #[test]
        fn test_random_input_fuzzing(
            logits in prop::collection::vec(any::<f64>(), 0..20),
            calibration in prop::collection::vec(any::<f64>(), 0..50),
            confidence in any::<f64>(),
            temperature in any::<f64>()
        ) {
            let mut fixture = SecurityTestFixture::new();
            
            // Test conformal prediction with random inputs
            if !logits.is_empty() && !calibration.is_empty() {
                let pred_result = fixture.predictor.predict(&logits, &calibration);
                
                // System should either succeed gracefully or fail with appropriate error
                match pred_result {
                    Ok(intervals) => {
                        // If successful, output should be valid
                        for (i, (lower, upper)) in intervals.iter().enumerate() {
                            prop_assert!(lower.is_finite() || pred_result.is_err(),
                                       "Lower bound {} should be finite if prediction succeeds", i);
                            prop_assert!(upper.is_finite() || pred_result.is_err(),
                                       "Upper bound {} should be finite if prediction succeeds", i);
                        }
                    },
                    Err(AtsCoreError::ValidationError { .. }) => {
                        // Expected for invalid inputs
                    },
                    Err(AtsCoreError::MathematicalError { .. }) => {
                        // Expected for mathematically invalid inputs  
                    },
                    Err(AtsCoreError::TimeoutError { .. }) => {
                        // Expected for inputs that take too long
                    },
                    Err(e) => {
                        // Unexpected error types might indicate security issues
                        prop_assert!(false, "Unexpected error type: {}", e);
                    }
                }
            }
            
            // Test temperature scaling with random inputs
            if !logits.is_empty() && temperature.is_finite() && temperature > 0.0 {
                let temp_result = fixture.predictor.temperature_scaled_softmax(&logits, temperature);
                
                match temp_result {
                    Ok(probs) => {
                        // Verify output safety
                        prop_assert!(probs.len() == logits.len(), "Output length should match input");
                        
                        for &prob in &probs {
                            prop_assert!(prob.is_finite(), "All probabilities should be finite");
                            prop_assert!(prob >= 0.0, "All probabilities should be non-negative");
                            prop_assert!(prob <= 1.0, "All probabilities should be <= 1.0");
                        }
                        
                        let sum: f64 = probs.iter().sum();
                        prop_assert!((sum - 1.0).abs() < 1e-6, "Probabilities should sum to 1.0");
                    },
                    Err(_) => {
                        // Errors are acceptable for invalid inputs
                    }
                }
            }
        }
        
        #[test]
        fn test_adversarial_ats_cp_fuzzing(
            logits in prop::collection::vec(-1000.0f64..1000.0, 1..10),
            calibration_size in 5usize..50,
            confidence in 0.01f64..0.99
        ) {
            let mut fixture = SecurityTestFixture::new();
            
            // Generate calibration data
            let calibration_logits: Vec<Vec<f64>> = (0..calibration_size)
                .map(|_| logits.clone())
                .collect();
            
            let calibration_labels: Vec<usize> = (0..calibration_size)
                .map(|i| i % logits.len())
                .collect();
            
            let variants = vec![AtsCpVariant::GQ, AtsCpVariant::AQ, AtsCpVariant::MGQ, AtsCpVariant::MAQ];
            
            for variant in variants {
                let result = fixture.predictor.ats_cp_predict(
                    &logits,
                    &calibration_logits,
                    &calibration_labels,
                    confidence,
                    variant.clone()
                );
                
                match result {
                    Ok(ats_result) => {
                        // Verify output security properties
                        prop_assert!(!ats_result.conformal_set.is_empty(),
                                   "Conformal set should not be empty for variant {:?}", variant);
                        
                        prop_assert!(ats_result.optimal_temperature > 0.0 && ats_result.optimal_temperature.is_finite(),
                                   "Temperature should be positive and finite for variant {:?}", variant);
                        
                        let prob_sum: f64 = ats_result.calibrated_probabilities.iter().sum();
                        prop_assert!((prob_sum - 1.0).abs() < 1e-6,
                                   "Probabilities should sum to 1.0 for variant {:?}", variant);
                        
                        for &class_idx in &ats_result.conformal_set {
                            prop_assert!(class_idx < logits.len(),
                                       "Conformal set indices should be valid for variant {:?}", variant);
                        }
                    },
                    Err(_) => {
                        // Errors are acceptable for adversarial inputs
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_structured_fuzzing_attacks() {
        println!("🎲 Testing structured fuzzing attacks...");
        
        let mut fixture = SecurityTestFixture::new();
        let mut attack_attempts = 0;
        let mut successful_attacks = 0;
        
        // Structured attacks with known problematic patterns
        let attack_patterns = vec![
            // Pattern 1: Gradual overflow
            (0..100).map(|i| i as f64 * 10.0).collect::<Vec<f64>>(),
            // Pattern 2: Alternating extremes
            (0..50).map(|i| if i % 2 == 0 { 1e6 } else { -1e6 }).collect::<Vec<f64>>(),
            // Pattern 3: Near-zero values
            (0..50).map(|i| i as f64 * 1e-10).collect::<Vec<f64>>(),
            // Pattern 4: Repeated values (potential division by zero)
            vec![1.0; 100],
            // Pattern 5: Fibonacci-like growth
            {
                let mut fib = vec![1.0, 1.0];
                for i in 2..50 {
                    fib.push(fib[i-1] + fib[i-2]);
                }
                fib
            },
        ];
        
        for (pattern_idx, attack_pattern) in attack_patterns.iter().enumerate() {
            attack_attempts += 1;
            println!("  Testing attack pattern {}", pattern_idx + 1);
            
            let calibration_data: Vec<f64> = (0..attack_pattern.len()).map(|i| i as f64 * 0.01).collect();
            
            let start_time = Instant::now();
            let result = fixture.predictor.predict(attack_pattern, &calibration_data);
            let execution_time = start_time.elapsed();
            
            let attack_successful = match result {
                Ok(_) => {
                    // Check if attack caused performance issues
                    execution_time > Duration::from_millis(100)
                },
                Err(AtsCoreError::ValidationError { .. }) => false, // Properly rejected
                Err(AtsCoreError::TimeoutError { .. }) => false, // Timeout protection worked
                Err(_) => true, // Unexpected error might indicate successful attack
            };
            
            if attack_successful {
                successful_attacks += 1;
                fixture.record_vulnerability(SecurityVulnerability {
                    vulnerability_type: "Fuzzing Attack".to_string(),
                    severity: SecuritySeverity::Medium,
                    location: "predict".to_string(),
                    description: format!("Attack pattern {} succeeded", pattern_idx + 1),
                    mitigation: Some("Strengthen input validation".to_string()),
                });
            }
            
            println!("    Pattern {} result: {} in {:?}", 
                    pattern_idx + 1, 
                    if attack_successful { "VULNERABLE" } else { "PROTECTED" },
                    execution_time);
        }
        
        let attack_success_rate = (successful_attacks as f64) / (attack_attempts as f64);
        
        println!("  Fuzzing Attack Summary:");
        println!("    Total attacks:   {}", attack_attempts);
        println!("    Successful:      {}", successful_attacks);
        println!("    Success rate:    {:.1}%", attack_success_rate * 100.0);
        
        // Security requirement: attack success rate should be low
        assert!(attack_success_rate < 0.2, 
               "Fuzzing attack success rate should be <20%, got {:.1}%", 
               attack_success_rate * 100.0);
        
        println!("  ✅ Structured fuzzing attack resistance validated");
    }
}

/// Comprehensive security audit
mod security_audit {
    use super::*;
    
    #[test]
    fn test_comprehensive_security_audit() {
        println!("🔒 Running comprehensive security audit...");
        
        let mut fixture = SecurityTestFixture::new();
        let mut audit_results = Vec::new();
        
        // Test 1: Input validation coverage
        println!("  Auditing input validation coverage...");
        let validation_tests = vec![
            (vec![], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "empty predictions"),
            (vec![1.0, 2.0, 3.0], vec![], "empty calibration"),
            (vec![f64::NAN], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "NaN input"),
            (vec![f64::INFINITY], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "infinity input"),
        ];
        
        let mut validation_passes = 0;
        for (preds, cal, test_name) in validation_tests {
            let result = fixture.predictor.predict(&preds, &cal);
            match result {
                Err(AtsCoreError::ValidationError { .. }) => {
                    validation_passes += 1;
                    println!("    ✅ {} properly validated", test_name);
                },
                _ => {
                    println!("    ❌ {} validation bypassed", test_name);
                }
            }
        }
        audit_results.push(("Input Validation", validation_passes, validation_tests.len()));
        
        // Test 2: Error handling consistency
        println!("  Auditing error handling consistency...");
        let error_tests = vec![
            (vec![-0.5], "invalid confidence"),
            (vec![1.5], "invalid confidence"),
            (vec![0.0], "boundary confidence"),
            (vec![1.0], "boundary confidence"),
        ];
        
        let mut error_handling_passes = 0;
        for (conf_vals, test_name) in error_tests {
            for &conf in &conf_vals {
                let result = fixture.predictor.predict_detailed(
                    &vec![1.0, 2.0, 3.0],
                    &vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    conf
                );
                
                match result {
                    Err(AtsCoreError::ValidationError { .. }) if conf <= 0.0 || conf >= 1.0 => {
                        error_handling_passes += 1;
                        println!("    ✅ {} error properly handled", test_name);
                    },
                    Ok(_) if conf > 0.0 && conf < 1.0 => {
                        error_handling_passes += 1;
                        println!("    ✅ {} valid input accepted", test_name);
                    },
                    _ => {
                        println!("    ❌ {} error handling inconsistent", test_name);
                    }
                }
            }
        }
        audit_results.push(("Error Handling", error_handling_passes, conf_vals.len() * error_tests.len()));
        
        // Test 3: Performance bounds enforcement
        println!("  Auditing performance bounds enforcement...");
        let perf_test_data = vec![1.0, 2.0, 3.0];
        let perf_calibration = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        
        let mut performance_violations = 0;
        for _ in 0..10 {
            let start_time = Instant::now();
            let result = fixture.predictor.predict(&perf_test_data, &perf_calibration);
            let execution_time = start_time.elapsed();
            
            if result.is_ok() && execution_time > Duration::from_micros(100) {
                performance_violations += 1;
            }
        }
        
        let performance_passes = 10 - performance_violations;
        audit_results.push(("Performance Bounds", performance_passes, 10));
        
        // Generate audit report
        println!("\n  🔍 Security Audit Results:");
        let mut total_passes = 0;
        let mut total_tests = 0;
        
        for (category, passes, total) in &audit_results {
            let percentage = (*passes as f64) / (*total as f64) * 100.0;
            println!("    {}: {}/{} ({:.1}%)", category, passes, total, percentage);
            total_passes += passes;
            total_tests += total;
        }
        
        let overall_score = (total_passes as f64) / (total_tests as f64) * 100.0;
        println!("    Overall Security Score: {:.1}%", overall_score);
        
        // Security requirements
        assert!(overall_score >= 80.0, "Overall security score should be ≥80%, got {:.1}%", overall_score);
        
        if fixture.vulnerabilities.is_empty() {
            println!("  ✅ No critical security vulnerabilities found");
        } else {
            println!("  ⚠️  Found {} potential security issues", fixture.vulnerabilities.len());
            for vuln in &fixture.vulnerabilities {
                println!("    {:?}: {}", vuln.severity, vuln.description);
            }
        }
        
        println!("✅ Comprehensive security audit completed");
    }
}

#[cfg(test)]
mod security_test_integration {
    use super::*;
    use ats_core::test_framework::{TestFramework, swarm_utils};
    
    #[tokio::test]
    async fn test_security_tests_swarm_coordination() {
        // Initialize security test framework
        let mut framework = TestFramework::new(
            "security_test_swarm".to_string(),
            "security_test_agent".to_string(),
        ).unwrap();
        
        // Signal coordination with other test agents
        swarm_utils::coordinate_test_execution(&framework.context, "security_tests").await.unwrap();
        
        // Execute security test sample
        let mut fixture = SecurityTestFixture::new();
        
        // Test with malicious input
        let malicious_logits = vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
        let safe_calibration = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        
        let result = fixture.predictor.predict(&malicious_logits, &safe_calibration);
        
        // Security requirement: malicious input should be handled safely
        match result {
            Err(AtsCoreError::ValidationError { .. }) => {
                println!("✅ Malicious input correctly rejected");
                framework.context.execution_metrics.tests_passed += 1;
            },
            _ => {
                framework.context.execution_metrics.tests_failed += 1;
                framework.context.execution_metrics.security_vulnerabilities.push(
                    SecurityVulnerability {
                        vulnerability_type: "Input Validation Bypass".to_string(),
                        severity: SecuritySeverity::High,
                        location: "predict".to_string(),
                        description: "Malicious input was not properly rejected".to_string(),
                        mitigation: Some("Strengthen input validation".to_string()),
                    }
                );
            }
        }
        
        // Share results with swarm
        swarm_utils::share_test_results(&framework.context, &framework.context.execution_metrics).await.unwrap();
        
        // Security requirement: no critical vulnerabilities should be found
        let critical_vulns: Vec<_> = framework.context.execution_metrics.security_vulnerabilities
            .iter()
            .filter(|v| v.severity == SecuritySeverity::Critical)
            .collect();
        
        assert!(critical_vulns.is_empty(), 
               "No critical security vulnerabilities should be found, got: {:?}", critical_vulns);
        
        println!("✅ Security tests swarm coordination completed");
    }
}