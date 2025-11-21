//! Performance Bottleneck Analyzer Demo
//!
//! This example demonstrates how to use the Performance Bottleneck Analyzer Agent
//! to monitor, detect, and optimize performance issues in ATS-Core conformal prediction.

use std::time::{Duration, SystemTime};
use ats_core::{
    AtsCpConfig,
    performance_bottleneck_analyzer::*,
    performance_analysis_integration::*,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🔍 ATS-Core Performance Bottleneck Analyzer Demo");
    println!("================================================");
    
    // Demo 1: Basic Performance Analysis
    println!("\n📊 Demo 1: Basic Performance Analysis");
    demo_basic_analysis()?;
    
    // Demo 2: Bottleneck Detection
    println!("\n🚨 Demo 2: Bottleneck Detection");
    demo_bottleneck_detection()?;
    
    // Demo 3: Integrated Monitoring
    println!("\n📈 Demo 3: Integrated Performance Monitoring");
    demo_integrated_monitoring()?;
    
    // Demo 4: Optimization Recommendations
    println!("\n💡 Demo 4: Optimization Recommendations");
    demo_optimization_recommendations()?;
    
    // Demo 5: Performance Trends
    println!("\n📉 Demo 5: Performance Trend Analysis");
    demo_trend_analysis()?;
    
    println!("\n✅ Performance Bottleneck Analyzer Demo Complete!");
    println!("🎯 Key Takeaways:");
    println!("   • Real-time bottleneck detection with ML-based prediction");
    println!("   • Automated optimization recommendations with safety guardrails");
    println!("   • Comprehensive performance trend analysis");
    println!("   • Seamless integration with existing ATS-Core optimizations");
    
    Ok(())
}

/// Demo 1: Basic Performance Analysis Setup
fn demo_basic_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("Setting up Performance Bottleneck Analyzer...");
    
    let config = AnalyzerConfig {
        max_history_size: 1000,
        detection_sensitivity: 0.15, // 15% degradation threshold
        baseline_update_threshold: 0.05,
        enable_ml_prediction: true,
        reporting_interval: Duration::from_secs(30),
        auto_optimization: false,
    };
    
    let analyzer = PerformanceBottleneckAnalyzer::new(config)?;
    
    // Record some sample metrics
    let good_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(15), // Good: sub-20μs
        cpu_utilization: 0.85,
        memory_usage: 2 * 1024 * 1024, // 2MB
        cache_misses: 25,
        simd_utilization: 0.90,
        throughput: 1000.0,
        operation_type: "softmax".to_string(),
        input_size: 16,
    };
    
    analyzer.record_metrics(good_metrics)?;
    
    println!("✅ Baseline performance recorded:");
    println!("   • Execution Time: 15μs (✅ under 20μs target)");
    println!("   • CPU Utilization: 85%");
    println!("   • SIMD Utilization: 90%");
    println!("   • Cache Misses: 25");
    
    Ok(())
}

/// Demo 2: Bottleneck Detection with Various Issues
fn demo_bottleneck_detection() -> Result<(), Box<dyn std::error::Error>> {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config)?;
    
    // Establish baseline first
    let baseline_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(10),
        cpu_utilization: 0.85,
        memory_usage: 1024 * 1024,
        cache_misses: 20,
        simd_utilization: 0.90,
        throughput: 1000.0,
        operation_type: "quantile_computation".to_string(),
        input_size: 100,
    };
    analyzer.record_metrics(baseline_metrics)?;
    
    println!("Simulating various performance issues...");
    
    // Issue 1: High execution time
    let slow_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(50), // 5x slower!
        cpu_utilization: 0.30, // Low CPU utilization suggests blocking
        memory_usage: 1024 * 1024,
        cache_misses: 20,
        simd_utilization: 0.20, // Low vectorization
        throughput: 200.0,
        operation_type: "quantile_computation".to_string(),
        input_size: 100,
    };
    
    let bottlenecks = analyzer.analyze_for_bottlenecks(&slow_metrics)?;
    
    println!("\n🚨 Detected {} bottlenecks:", bottlenecks.len());
    for bottleneck in &bottlenecks {
        println!("   • {:?} ({:?}): {}", 
            bottleneck.bottleneck_type,
            bottleneck.severity,
            bottleneck.description
        );
        println!("     Root Cause: {}", bottleneck.root_cause);
        println!("     Impact: {:.1}% performance degradation", bottleneck.impact_estimate);
        
        if !bottleneck.recommendations.is_empty() {
            println!("     Top Recommendation: {}", 
                bottleneck.recommendations[0].description);
        }
        println!();
    }
    
    // Issue 2: Memory bottleneck
    let memory_heavy_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(30),
        cpu_utilization: 0.85,
        memory_usage: 600 * 1024 * 1024, // 600MB - very high
        cache_misses: 1500, // High cache misses
        simd_utilization: 0.85,
        throughput: 300.0,
        operation_type: "quantile_computation".to_string(),
        input_size: 100,
    };
    
    let memory_bottlenecks = analyzer.analyze_for_bottlenecks(&memory_heavy_metrics)?;
    
    println!("🧠 Memory-related bottlenecks:");
    for bottleneck in &memory_bottlenecks {
        if bottleneck.bottleneck_type == BottleneckType::MemoryAccess || 
           bottleneck.bottleneck_type == BottleneckType::CacheMisses {
            println!("   • Memory Issue: {}", bottleneck.description);
            for rec in &bottleneck.recommendations {
                println!("     → {}", rec.description);
            }
        }
    }
    
    Ok(())
}

/// Demo 3: Integrated Performance Monitoring
fn demo_integrated_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    let config = AtsCpConfig::high_performance();
    let monitoring_config = MonitoringConfig {
        enable_detailed_profiling: true,
        collect_memory_metrics: true,
        collect_cpu_metrics: true,
        collect_cache_metrics: true,
        auto_optimize_threshold: 0.20,
        reporting_frequency: 5, // Report every 5 operations for demo
    };
    
    let mut monitored_predictor = MonitoredOptimizedConformalPredictor::new(&config, monitoring_config)?;
    
    println!("Running monitored conformal predictions...");
    
    // Generate test data
    let logits = vec![2.3, 1.8, 3.1, 0.9, 2.7, 1.5, 3.8, 2.1];
    let calibration_scores: Vec<f64> = (0..200).map(|i| (i as f64 * 0.005) + 0.1).collect();
    let alpha = 0.1;
    
    // Run multiple predictions with monitoring
    for i in 0..10 {
        let result = monitored_predictor.predict_monitored(&logits, &calibration_scores, alpha)?;
        
        println!("Prediction {}: {:.2}μs", 
            i + 1, 
            result.execution_time.as_nanos() as f64 / 1000.0
        );
        
        if !result.detected_issues.is_empty() {
            println!("   ⚠️  Issues detected: {}", result.detected_issues.len());
        }
        
        if !result.applied_optimizations.is_empty() {
            println!("   🔧 Auto-optimizations: {:?}", result.applied_optimizations);
        }
        
        // Demonstrate ATS-CP monitoring
        if i % 3 == 0 {
            let ats_result = monitored_predictor.ats_cp_predict_monitored(
                &logits, 
                &calibration_scores, 
                alpha,
                Some((0.1, 2.0))
            )?;
            
            println!("   ATS-CP: {:.2}μs, Temperature: {:.3}", 
                ats_result.execution_time.as_nanos() as f64 / 1000.0,
                ats_result.result.optimal_temperature
            );
        }
    }
    
    // Get critical bottlenecks
    let critical_bottlenecks = monitored_predictor.get_critical_bottlenecks();
    if !critical_bottlenecks.is_empty() {
        println!("\n🔴 Critical bottlenecks requiring attention:");
        for bottleneck in &critical_bottlenecks {
            println!("   • {}: {}", 
                bottleneck.description,
                bottleneck.root_cause
            );
        }
    } else {
        println!("\n✅ No critical bottlenecks detected - system performing optimally!");
    }
    
    Ok(())
}

/// Demo 4: Optimization Recommendations Engine
fn demo_optimization_recommendations() -> Result<(), Box<dyn std::error::Error>> {
    let config = AtsCpConfig::high_performance();
    let monitoring_config = MonitoringConfig::default();
    let monitored_predictor = MonitoredOptimizedConformalPredictor::new(&config, monitoring_config)?;
    
    // Simulate some performance issues to generate recommendations
    let analyzer = monitored_predictor.get_analyzer();
    
    // Create metrics that will trigger various optimization recommendations
    let scenarios = vec![
        ("High Memory Usage", PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time: Duration::from_micros(25),
            cpu_utilization: 0.80,
            memory_usage: 800 * 1024 * 1024, // 800MB
            cache_misses: 100,
            simd_utilization: 0.85,
            throughput: 400.0,
            operation_type: "memory_intensive_op".to_string(),
            input_size: 1000,
        }),
        ("Poor Vectorization", PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time: Duration::from_micros(40),
            cpu_utilization: 0.60,
            memory_usage: 50 * 1024 * 1024,
            cache_misses: 50,
            simd_utilization: 0.15, // Very low SIMD utilization
            throughput: 250.0,
            operation_type: "non_vectorized_op".to_string(),
            input_size: 64,
        }),
        ("Cache Inefficiency", PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time: Duration::from_micros(35),
            cpu_utilization: 0.90,
            memory_usage: 100 * 1024 * 1024,
            cache_misses: 2000, // Very high cache misses
            simd_utilization: 0.80,
            throughput: 300.0,
            operation_type: "cache_unfriendly_op".to_string(),
            input_size: 500,
        }),
    ];
    
    println!("Analyzing different performance scenarios...\n");
    
    for (scenario_name, metrics) in scenarios {
        println!("📊 Scenario: {}", scenario_name);
        analyzer.record_metrics(metrics)?;
        
        let bottlenecks = analyzer.analyze_for_bottlenecks(&metrics)?;
        
        for bottleneck in bottlenecks {
            println!("   🔍 Issue: {}", bottleneck.description);
            println!("   📈 Impact: {:.1}% performance loss", bottleneck.impact_estimate);
            
            println!("   💡 Recommendations:");
            for (i, rec) in bottleneck.recommendations.iter().take(2).enumerate() {
                println!("     {}. {} ({:?} effort)", 
                    i + 1,
                    rec.description,
                    rec.implementation_effort
                );
                println!("        Expected improvement: {:.1}%", rec.estimated_improvement);
                if let Some(ref example) = rec.code_example {
                    println!("        Code hint: {}", example);
                }
            }
        }
        println!();
    }
    
    // Generate comprehensive recommendations
    let recommendations = monitored_predictor.get_optimization_recommendations()?;
    
    println!("🎯 Top System-wide Optimization Recommendations:");
    for (i, rec) in recommendations.iter().take(5).enumerate() {
        println!("{}. {} ({:?} priority)", 
            i + 1, 
            rec.description, 
            rec.priority
        );
        println!("   • Expected improvement: {:.1}%", rec.estimated_improvement);
        println!("   • Implementation effort: {:?}", rec.implementation_effort);
        println!();
    }
    
    Ok(())
}

/// Demo 5: Performance Trend Analysis
fn demo_trend_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config)?;
    
    println!("Simulating performance trends over time...");
    
    // Simulate improving performance trend
    println!("\n📈 Scenario 1: Performance Improvement Trend");
    for i in 0..20 {
        let execution_time = if i < 10 {
            Duration::from_micros(30 - i) // Getting faster
        } else {
            Duration::from_micros(20) // Stabilized at good performance
        };
        
        let metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time,
            cpu_utilization: 0.80 + (i as f64 * 0.01),
            memory_usage: (20 - i) * 1024 * 1024,
            cache_misses: (50 - i * 2) as u64,
            simd_utilization: 0.70 + (i as f64 * 0.015),
            throughput: 500.0 + (i as f64 * 25.0),
            operation_type: "improving_operation".to_string(),
            input_size: 100,
        };
        
        analyzer.record_metrics(metrics)?;
    }
    
    let improving_report = analyzer.generate_report()?;
    println!("Trend: {:?}", improving_report.performance_trend);
    println!("Health Score: {:.1}/100", improving_report.system_health_score);
    println!("Average Execution Time: {:.2}μs", 
        improving_report.average_execution_time.as_nanos() as f64 / 1000.0);
    
    // Reset and simulate degrading performance
    println!("\n📉 Scenario 2: Performance Degradation Trend");
    let analyzer2 = PerformanceBottleneckAnalyzer::new(AnalyzerConfig::default())?;
    
    for i in 0..20 {
        let execution_time = Duration::from_micros(15 + i); // Getting slower
        
        let metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time,
            cpu_utilization: 0.90 - (i as f64 * 0.02),
            memory_usage: (10 + i * 5) * 1024 * 1024,
            cache_misses: (20 + i * 10) as u64,
            simd_utilization: 0.90 - (i as f64 * 0.02),
            throughput: 1000.0 - (i as f64 * 30.0),
            operation_type: "degrading_operation".to_string(),
            input_size: 100,
        };
        
        analyzer2.record_metrics(metrics)?;
    }
    
    let degrading_report = analyzer2.generate_report()?;
    println!("Trend: {:?}", degrading_report.performance_trend);
    println!("Health Score: {:.1}/100", degrading_report.system_health_score);
    println!("Average Execution Time: {:.2}μs", 
        degrading_report.average_execution_time.as_nanos() as f64 / 1000.0);
    
    if !degrading_report.detected_bottlenecks.is_empty() {
        println!("🚨 Detected bottlenecks in degrading system:");
        for bottleneck in &degrading_report.detected_bottlenecks {
            println!("   • {:?}: {}", bottleneck.bottleneck_type, bottleneck.description);
        }
    }
    
    // Show comparison
    println!("\n📊 Performance Comparison:");
    println!("   Improving System Health: {:.1}/100", improving_report.system_health_score);
    println!("   Degrading System Health: {:.1}/100", degrading_report.system_health_score);
    println!("   Health Difference: {:.1} points", 
        improving_report.system_health_score - degrading_report.system_health_score);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_functions() {
        // Test that our demo functions don't panic
        assert!(demo_basic_analysis().is_ok());
        assert!(demo_bottleneck_detection().is_ok());
        assert!(demo_integrated_monitoring().is_ok());
        assert!(demo_optimization_recommendations().is_ok());
        assert!(demo_trend_analysis().is_ok());
    }
}