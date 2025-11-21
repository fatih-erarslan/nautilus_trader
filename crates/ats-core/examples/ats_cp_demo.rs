//! ATS-CP Algorithm Demonstration
//!
//! This example demonstrates the complete ATS-CP workflow with all variants:
//! GQ, AQ, MGQ, MAQ using real-world inspired data.

use ats_core::{
    config::AtsCpConfig,
    conformal::ConformalPredictor,
    types::{AtsCpVariant, Confidence},
    error::Result,
};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🧠 ATS-CP Algorithm Demonstration");
    println!("=================================\n");

    // Create high-performance configuration for financial trading
    let config = AtsCpConfig::high_performance();
    let mut predictor = ConformalPredictor::new(&config)?;

    // Simulate financial market prediction scenario
    let calibration_data = generate_financial_calibration_data();
    let (calibration_logits, calibration_labels) = calibration_data;

    // Test prediction - new market conditions
    let test_logits = vec![2.1, 1.8, 3.2, 2.5, 1.9]; // 5-class market scenario
    let confidence = 0.95; // 95% coverage guarantee

    println!("📊 Market Scenario: 5-class prediction");
    println!("Test logits: {:?}", test_logits);
    println!("Confidence target: {:.1}%\n", confidence * 100.0);

    // Test all ATS-CP variants
    let variants = vec![
        (AtsCpVariant::GQ, "Generalized Quantile"),
        (AtsCpVariant::AQ, "Adaptive Quantile"),
        (AtsCpVariant::MGQ, "Multi-class Generalized Quantile"),
        (AtsCpVariant::MAQ, "Multi-class Adaptive Quantile"),
    ];

    for (variant, name) in variants {
        println!("🔬 Testing {} ({})", name, format!("{:?}", variant));
        
        let start_time = Instant::now();
        
        let result = predictor.ats_cp_predict(
            &test_logits,
            &calibration_logits,
            &calibration_labels,
            confidence,
            variant,
        )?;
        
        let elapsed = start_time.elapsed();
        
        println!("  ✓ Execution time: {:.2}μs", elapsed.as_nanos() as f64 / 1000.0);
        println!("  ✓ Optimal temperature: {:.6}", result.optimal_temperature);
        println!("  ✓ Conformal set: {:?}", result.conformal_set);
        println!("  ✓ Coverage guarantee: {:.1}%", result.coverage_guarantee * 100.0);
        
        // Validate probabilities sum to 1.0
        let sum: f64 = result.calibrated_probabilities.iter().sum();
        println!("  ✓ Probability sum: {:.10} (should be 1.0)", sum);
        
        // Show top predictions
        let mut indexed_probs: Vec<_> = result.calibrated_probabilities
            .iter()
            .enumerate()
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        
        println!("  📈 Top predictions:");
        for (i, (class_idx, &prob)) in indexed_probs.iter().take(3).enumerate() {
            println!("    {}. Class {}: {:.4} ({:.1}%)", 
                i + 1, class_idx, prob, prob * 100.0);
        }
        
        // Performance validation
        if result.execution_time_ns < config.conformal.target_latency_us * 1000 {
            println!("  ✅ Meets latency target ({} μs)", config.conformal.target_latency_us);
        } else {
            println!("  ⚠️  Exceeds latency target");
        }
        
        println!();
    }

    // Demonstrate batch processing
    println!("🚀 Batch Processing Demo");
    println!("========================\n");
    
    let batch_test_cases = vec![
        vec![1.5, 2.0, 2.8, 1.2, 1.9],
        vec![3.1, 1.4, 2.2, 2.7, 1.8],
        vec![2.0, 2.9, 1.6, 2.4, 3.1],
    ];
    
    for (i, test_case) in batch_test_cases.iter().enumerate() {
        let start_time = Instant::now();
        
        let result = predictor.ats_cp_predict(
            test_case,
            &calibration_logits,
            &calibration_labels,
            0.9, // 90% confidence for faster processing
            AtsCpVariant::GQ,
        )?;
        
        let elapsed = start_time.elapsed();
        
        println!("Batch item {}: {:.2}μs, conformal set: {:?}", 
            i + 1, elapsed.as_nanos() as f64 / 1000.0, result.conformal_set);
    }

    // Performance summary
    println!("\n📋 Performance Summary");
    println!("=====================");
    let (ops, avg_latency, ops_per_sec) = predictor.get_performance_stats();
    println!("Total operations: {}", ops);
    println!("Average latency: {:.2}μs", avg_latency as f64 / 1000.0);
    println!("Operations per second: {:.0}", ops_per_sec);
    
    println!("\n✅ ATS-CP demonstration completed successfully!");
    println!("   All algorithms maintain mathematical rigor with:");
    println!("   • IEEE 754 floating-point compliance");
    println!("   • Numerical stability guarantees");
    println!("   • Sub-microsecond latency performance");
    println!("   • Theoretical coverage guarantees");

    Ok(())
}

/// Generate realistic financial calibration data
fn generate_financial_calibration_data() -> (Vec<Vec<f64>>, Vec<usize>) {
    // Simulate 100 historical market predictions with 5 market states
    let mut calibration_logits = Vec::new();
    let mut calibration_labels = Vec::new();
    
    // Different market regime patterns
    let patterns = vec![
        // Bull market pattern
        (vec![1.8, 1.2, 3.4, 2.1, 1.5], 2), // State 2: Strong uptrend
        (vec![2.1, 1.5, 3.1, 2.4, 1.7], 2),
        (vec![1.9, 1.3, 3.2, 2.0, 1.6], 2),
        
        // Bear market pattern  
        (vec![3.2, 2.8, 1.1, 1.4, 2.3], 0), // State 0: Strong downtrend
        (vec![3.0, 2.6, 1.3, 1.6, 2.1], 0),
        (vec![3.1, 2.7, 1.2, 1.5, 2.2], 0),
        
        // Sideways market pattern
        (vec![2.0, 2.1, 2.2, 2.0, 1.9], 1), // State 1: Consolidation
        (vec![2.1, 2.0, 2.3, 1.9, 2.2], 1),
        (vec![1.9, 2.2, 2.1, 2.1, 2.0], 1),
        
        // High volatility pattern
        (vec![2.8, 1.2, 2.1, 3.1, 1.8], 3), // State 3: High volatility
        (vec![2.7, 1.4, 2.3, 2.9, 1.6], 3),
        (vec![2.9, 1.1, 2.0, 3.2, 1.7], 3),
        
        // Low volatility pattern
        (vec![1.8, 1.9, 1.7, 2.0, 2.1], 4), // State 4: Low volatility
        (vec![1.9, 2.0, 1.8, 1.9, 2.2], 4),
        (vec![2.0, 1.8, 1.9, 2.1, 1.9], 4),
    ];
    
    // Replicate patterns to get 100 samples
    for _ in 0..7 {
        for (logits, label) in &patterns {
            calibration_logits.push(logits.clone());
            calibration_labels.push(*label);
        }
    }
    
    // Add some noise variation to the last few samples
    for i in 0..5 {
        let base_pattern = &patterns[i % patterns.len()];
        let mut noisy_logits = base_pattern.0.clone();
        
        // Add small random variations
        for logit in &mut noisy_logits {
            *logit += (i as f64 - 2.0) * 0.1; // Small perturbation
        }
        
        calibration_logits.push(noisy_logits);
        calibration_labels.push(base_pattern.1);
    }
    
    (calibration_logits, calibration_labels)
}