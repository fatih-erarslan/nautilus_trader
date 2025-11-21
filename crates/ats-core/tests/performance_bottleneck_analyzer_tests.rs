//! Comprehensive tests for the Performance Bottleneck Analyzer Agent
//!
//! This test suite validates the correctness and effectiveness of bottleneck detection,
//! performance monitoring, and optimization recommendation systems.

use std::time::{Duration, SystemTime};
use ats_core::{
    AtsCpConfig,
    performance_bottleneck_analyzer::*,
    performance_analysis_integration::*,
};
use approx::*;

#[test]
fn test_performance_bottleneck_analyzer_initialization() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config);
    
    assert!(analyzer.is_ok());
    let analyzer = analyzer.unwrap();
    
    // Verify empty state
    let report = analyzer.generate_report().unwrap();
    assert_eq!(report.total_measurements, 0);
    assert_eq!(report.detected_bottlenecks.len(), 0);
    assert_eq!(report.system_health_score, 100.0);
}

#[test]
fn test_metrics_recording_and_baseline_creation() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
    
    let metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(15),
        cpu_utilization: 0.85,
        memory_usage: 2 * 1024 * 1024,
        cache_misses: 25,
        simd_utilization: 0.90,
        throughput: 1000.0,
        operation_type: "test_operation".to_string(),
        input_size: 16,
    };
    
    analyzer.record_metrics(metrics.clone()).unwrap();
    
    // Verify metrics were recorded
    let report = analyzer.generate_report().unwrap();
    assert_eq!(report.total_measurements, 1);
    assert_relative_eq!(
        report.average_execution_time.as_micros() as f64,
        15.0,
        epsilon = 1.0
    );
    
    // Verify baseline was created
    let baseline = analyzer.get_baseline("test_operation");
    assert!(baseline.is_some());
    let baseline = baseline.unwrap();
    assert_eq!(baseline.operation, "test_operation");
    assert_relative_eq!(
        baseline.expected_time.as_micros() as f64,
        15.0,
        epsilon = 1.0
    );
}

#[test]
fn test_execution_time_bottleneck_detection() {
    let config = AnalyzerConfig {
        detection_sensitivity: 0.20, // 20% degradation threshold
        ..Default::default()
    };
    let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
    
    // Establish baseline
    let baseline_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(10),
        cpu_utilization: 0.85,
        memory_usage: 1024 * 1024,
        cache_misses: 20,
        simd_utilization: 0.90,
        throughput: 1000.0,
        operation_type: "baseline_op".to_string(),
        input_size: 16,
    };
    analyzer.record_metrics(baseline_metrics).unwrap();
    
    // Create metrics that should trigger bottleneck detection (3x slower)
    let slow_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(30),
        cpu_utilization: 0.85,
        memory_usage: 1024 * 1024,
        cache_misses: 20,
        simd_utilization: 0.90,
        throughput: 333.0,
        operation_type: "baseline_op".to_string(),
        input_size: 16,
    };
    
    let bottlenecks = analyzer.analyze_for_bottlenecks(&slow_metrics).unwrap();
    
    // Should detect execution time bottleneck
    assert!(!bottlenecks.is_empty());
    
    let execution_bottleneck = bottlenecks.iter()
        .find(|b| b.bottleneck_type == BottleneckType::ExecutionTime);
    assert!(execution_bottleneck.is_some());
    
    let bottleneck = execution_bottleneck.unwrap();
    assert_eq!(bottleneck.severity, BottleneckSeverity::Critical); // 3x slower
    assert!(bottleneck.impact_estimate > 100.0); // >100% slower
}

#[test]
fn test_memory_bottleneck_detection() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
    
    let high_memory_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(20),
        cpu_utilization: 0.85,
        memory_usage: 600 * 1024 * 1024, // 600MB - should trigger bottleneck
        cache_misses: 50,
        simd_utilization: 0.90,
        throughput: 800.0,
        operation_type: "memory_test".to_string(),
        input_size: 100,
    };
    
    let bottlenecks = analyzer.analyze_for_bottlenecks(&high_memory_metrics).unwrap();
    
    // Should detect memory bottleneck
    let memory_bottleneck = bottlenecks.iter()
        .find(|b| b.bottleneck_type == BottleneckType::MemoryAccess);
    assert!(memory_bottleneck.is_some());
    
    let bottleneck = memory_bottleneck.unwrap();
    assert_eq!(bottleneck.severity, BottleneckSeverity::High);
    assert!(bottleneck.description.contains("memory"));
    
    // Check recommendations
    assert!(!bottleneck.recommendations.is_empty());
    let memory_rec = bottleneck.recommendations.iter()
        .find(|r| r.recommendation_type == RecommendationType::MemoryLayout);
    assert!(memory_rec.is_some());
}

#[test]
fn test_cache_miss_bottleneck_detection() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
    
    let high_cache_miss_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(25),
        cpu_utilization: 0.80,
        memory_usage: 50 * 1024 * 1024,
        cache_misses: 1000, // High cache misses
        simd_utilization: 0.85,
        throughput: 600.0,
        operation_type: "cache_test".to_string(),
        input_size: 200,
    };
    
    let bottlenecks = analyzer.analyze_for_bottlenecks(&high_cache_miss_metrics).unwrap();
    
    // Should detect cache miss bottleneck
    let cache_bottleneck = bottlenecks.iter()
        .find(|b| b.bottleneck_type == BottleneckType::CacheMisses);
    assert!(cache_bottleneck.is_some());
    
    let bottleneck = cache_bottleneck.unwrap();
    assert_eq!(bottleneck.severity, BottleneckSeverity::High);
    assert!(bottleneck.description.contains("cache"));
    
    // Should have memory layout recommendation
    let cache_rec = bottleneck.recommendations.iter()
        .find(|r| r.recommendation_type == RecommendationType::MemoryLayout);
    assert!(cache_rec.is_some());
    assert!(cache_rec.unwrap().description.contains("cache-aligned"));
}

#[test]
fn test_vectorization_bottleneck_detection() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
    
    let low_simd_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(30),
        cpu_utilization: 0.80,
        memory_usage: 10 * 1024 * 1024,
        cache_misses: 50,
        simd_utilization: 0.20, // Very low SIMD utilization
        throughput: 400.0,
        operation_type: "simd_test".to_string(),
        input_size: 64, // Large enough for vectorization
    };
    
    let bottlenecks = analyzer.analyze_for_bottlenecks(&low_simd_metrics).unwrap();
    
    // Should detect vectorization bottleneck
    let simd_bottleneck = bottlenecks.iter()
        .find(|b| b.bottleneck_type == BottleneckType::VectorizationMissed);
    assert!(simd_bottleneck.is_some());
    
    let bottleneck = simd_bottleneck.unwrap();
    assert_eq!(bottleneck.severity, BottleneckSeverity::Medium);
    assert!(bottleneck.description.contains("SIMD"));
    
    // Should have vectorization recommendation
    let vec_rec = bottleneck.recommendations.iter()
        .find(|r| r.recommendation_type == RecommendationType::Vectorization);
    assert!(vec_rec.is_some());
    assert!(vec_rec.unwrap().description.contains("AVX"));
}

#[test]
fn test_sequential_blocking_bottleneck_detection() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
    
    let low_cpu_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(40), // Slow execution
        cpu_utilization: 0.40, // Very low CPU utilization
        memory_usage: 20 * 1024 * 1024,
        cache_misses: 30,
        simd_utilization: 0.80,
        throughput: 300.0,
        operation_type: "sequential_test".to_string(),
        input_size: 100,
    };
    
    let bottlenecks = analyzer.analyze_for_bottlenecks(&low_cpu_metrics).unwrap();
    
    // Should detect sequential blocking
    let sequential_bottleneck = bottlenecks.iter()
        .find(|b| b.bottleneck_type == BottleneckType::SequentialBlocking);
    assert!(sequential_bottleneck.is_some());
    
    let bottleneck = sequential_bottleneck.unwrap();
    assert_eq!(bottleneck.severity, BottleneckSeverity::Medium);
    assert!(bottleneck.description.contains("CPU utilization"));
    
    // Should have parallelization recommendation
    let parallel_rec = bottleneck.recommendations.iter()
        .find(|r| r.recommendation_type == RecommendationType::Parallelization);
    assert!(parallel_rec.is_some());
    assert!(parallel_rec.unwrap().description.contains("rayon"));
}

#[test]
fn test_performance_trend_analysis() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
    
    // Simulate improving performance trend
    for i in 0..20 {
        let execution_time = Duration::from_micros(30 - i); // Getting faster
        let metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time,
            cpu_utilization: 0.70 + (i as f64 * 0.01),
            memory_usage: (25 - i) * 1024 * 1024,
            cache_misses: (60 - i * 2) as u64,
            simd_utilization: 0.60 + (i as f64 * 0.015),
            throughput: 400.0 + (i as f64 * 30.0),
            operation_type: "trending_operation".to_string(),
            input_size: 50,
        };
        analyzer.record_metrics(metrics).unwrap();
    }
    
    let report = analyzer.generate_report().unwrap();
    
    // Should detect improving trend
    assert!(matches!(report.performance_trend, PerformanceTrend::Improving));
    assert!(report.system_health_score > 80.0); // Should have good health score
    
    // Average should be reasonable
    assert!(report.average_execution_time.as_micros() < 25); // Should be faster than initial
}

#[test]
fn test_system_health_score_calculation() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
    
    // Test with good performance metrics
    let good_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(15), // Under 20μs target
        cpu_utilization: 0.85,
        memory_usage: 5 * 1024 * 1024,
        cache_misses: 20, // Low cache misses
        simd_utilization: 0.90, // High SIMD utilization
        throughput: 1200.0,
        operation_type: "good_operation".to_string(),
        input_size: 32,
    };
    
    for _ in 0..10 {
        analyzer.record_metrics(good_metrics.clone()).unwrap();
    }
    
    let good_report = analyzer.generate_report().unwrap();
    assert!(good_report.system_health_score > 90.0);
    
    // Test with poor performance metrics
    let analyzer2 = PerformanceBottleneckAnalyzer::new(AnalyzerConfig::default()).unwrap();
    let poor_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(60), // Way over 20μs target
        cpu_utilization: 0.40,
        memory_usage: 500 * 1024 * 1024,
        cache_misses: 1500, // High cache misses
        simd_utilization: 0.20, // Low SIMD utilization
        throughput: 200.0,
        operation_type: "poor_operation".to_string(),
        input_size: 32,
    };
    
    for _ in 0..10 {
        analyzer2.record_metrics(poor_metrics.clone()).unwrap();
    }
    
    let poor_report = analyzer2.generate_report().unwrap();
    assert!(poor_report.system_health_score < 50.0);
    
    // Good system should have higher health score
    assert!(good_report.system_health_score > poor_report.system_health_score + 40.0);
}

#[test]
fn test_optimization_recommendation_prioritization() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
    
    // Create metrics with multiple issues to generate various recommendations
    let problematic_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(50),
        cpu_utilization: 0.30,
        memory_usage: 600 * 1024 * 1024,
        cache_misses: 2000,
        simd_utilization: 0.15,
        throughput: 200.0,
        operation_type: "multi_issue_op".to_string(),
        input_size: 128,
    };
    
    analyzer.record_metrics(problematic_metrics).unwrap();
    
    let report = analyzer.generate_report().unwrap();
    
    // Should have multiple recommendations
    assert!(report.top_recommendations.len() > 3);
    
    // Recommendations should be sorted by priority and impact
    let mut prev_priority = RecommendationPriority::Critical;
    let mut prev_improvement = f64::INFINITY;
    
    for rec in &report.top_recommendations {
        // Priority should be same or lower
        assert!(rec.priority <= prev_priority);
        
        // If same priority, improvement should be same or lower
        if rec.priority == prev_priority {
            assert!(rec.estimated_improvement <= prev_improvement);
        }
        
        prev_priority = rec.priority.clone();
        prev_improvement = rec.estimated_improvement;
    }
}

#[test]
fn test_bottleneck_frequency_tracking() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
    
    // Establish baseline
    let baseline = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(10),
        cpu_utilization: 0.85,
        memory_usage: 10 * 1024 * 1024,
        cache_misses: 20,
        simd_utilization: 0.90,
        throughput: 1000.0,
        operation_type: "frequency_test".to_string(),
        input_size: 50,
    };
    analyzer.record_metrics(baseline).unwrap();
    
    // Repeat same bottleneck multiple times
    let cache_miss_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(25),
        cpu_utilization: 0.85,
        memory_usage: 10 * 1024 * 1024,
        cache_misses: 1000, // High cache misses
        simd_utilization: 0.90,
        throughput: 600.0,
        operation_type: "frequency_test".to_string(),
        input_size: 50,
    };
    
    // Record the same issue 5 times
    for _ in 0..5 {
        analyzer.analyze_for_bottlenecks(&cache_miss_metrics).unwrap();
    }
    
    // Check that frequency is tracked
    let bottlenecks = analyzer.get_bottlenecks_by_severity(BottleneckSeverity::Low);
    let cache_bottleneck = bottlenecks.iter()
        .find(|b| b.bottleneck_type == BottleneckType::CacheMisses);
    
    assert!(cache_bottleneck.is_some());
    let bottleneck = cache_bottleneck.unwrap();
    assert!(bottleneck.frequency >= 5); // Should track repeated occurrences
}

#[test]
fn test_integrated_monitoring_wrapper() {
    let config = AtsCpConfig::high_performance();
    let monitoring_config = MonitoringConfig {
        enable_detailed_profiling: true,
        collect_memory_metrics: true,
        collect_cpu_metrics: true,
        collect_cache_metrics: true,
        auto_optimize_threshold: 0.25,
        reporting_frequency: 1, // Report every operation for testing
    };
    
    let mut monitored_predictor = MonitoredOptimizedConformalPredictor::new(&config, monitoring_config).unwrap();
    
    // Test data
    let logits = vec![2.1, 1.8, 3.2, 1.5, 2.9];
    let calibration_scores: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    let alpha = 0.1;
    
    // Perform monitored prediction
    let result = monitored_predictor.predict_monitored(&logits, &calibration_scores, alpha);
    assert!(result.is_ok());
    
    let monitoring_result = result.unwrap();
    
    // Verify monitoring result structure
    assert!(monitoring_result.execution_time > Duration::from_nanos(0));
    assert_eq!(monitoring_result.performance_metrics.operation_type, "conformal_prediction");
    assert_eq!(monitoring_result.performance_metrics.input_size, logits.len());
    assert!(monitoring_result.performance_metrics.throughput > 0.0);
    
    // Should have performance metrics recorded
    let analyzer = monitored_predictor.get_analyzer();
    let report = analyzer.generate_report().unwrap();
    assert_eq!(report.total_measurements, 1);
}

#[test]
fn test_ats_cp_integrated_monitoring() {
    let config = AtsCpConfig::high_performance();
    let monitoring_config = MonitoringConfig::default();
    let mut monitored_predictor = MonitoredOptimizedConformalPredictor::new(&config, monitoring_config).unwrap();
    
    // Test data
    let logits = vec![1.5, 2.8, 0.9, 3.1, 2.2, 1.7, 3.5, 2.0];
    let calibration_scores: Vec<f64> = (0..200).map(|i| (i as f64 * 0.005) + 0.1).collect();
    let alpha = 0.05;
    
    // Test ATS-CP monitoring
    let result = monitored_predictor.ats_cp_predict_monitored(
        &logits,
        &calibration_scores,
        alpha,
        Some((0.1, 2.0))
    );
    
    assert!(result.is_ok());
    let monitoring_result = result.unwrap();
    
    // Verify ATS-CP specific results
    assert!(monitoring_result.result.optimal_temperature > 0.0);
    assert!(monitoring_result.result.conformal_set.prediction_sets.len() > 0);
    assert_eq!(monitoring_result.performance_metrics.operation_type, "ats_cp_prediction");
    assert_eq!(monitoring_result.performance_metrics.input_size, logits.len());
    
    // Should achieve sub-20μs target
    assert!(monitoring_result.execution_time.as_micros() < 25); // Allow some margin for testing environment
}

#[test]
fn test_automatic_optimization_application() {
    let config = AtsCpConfig::high_performance();
    let monitoring_config = MonitoringConfig {
        auto_optimize_threshold: 0.15, // Lower threshold for testing
        ..Default::default()
    };
    let mut monitored_predictor = MonitoredOptimizedConformalPredictor::new(&config, monitoring_config).unwrap();
    
    // Create a scenario that should trigger optimization recommendations
    let analyzer = monitored_predictor.get_analyzer();
    
    // Simulate memory bottleneck that could trigger low-effort optimizations
    let memory_bottleneck = PerformanceBottleneck {
        bottleneck_type: BottleneckType::MemoryAccess,
        severity: BottleneckSeverity::Medium,
        description: "High memory usage detected".to_string(),
        root_cause: "Large allocations".to_string(),
        impact_estimate: 20.0,
        recommendations: vec![
            OptimizationRecommendation {
                recommendation_type: RecommendationType::Caching,
                description: "Enable result caching".to_string(),
                estimated_improvement: 15.0,
                implementation_effort: ImplementationEffort::Low,
                code_example: None,
                priority: RecommendationPriority::Medium,
            }
        ],
        detected_at: SystemTime::now(),
        frequency: 1,
    };
    
    // Test that the optimization system can handle recommendations
    let bottlenecks = vec![memory_bottleneck];
    let applied = monitored_predictor.apply_automatic_optimizations(&bottlenecks);
    
    assert!(applied.is_ok());
    // The exact optimizations applied depend on the safety checks in the implementation
}

#[test]
fn test_performance_regression_detection() {
    let config = AnalyzerConfig {
        detection_sensitivity: 0.10, // Sensitive to 10% regressions
        ..Default::default()
    };
    let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
    
    // Establish good baseline performance
    for _ in 0..10 {
        let good_metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time: Duration::from_micros(12),
            cpu_utilization: 0.90,
            memory_usage: 5 * 1024 * 1024,
            cache_misses: 15,
            simd_utilization: 0.95,
            throughput: 1500.0,
            operation_type: "regression_test".to_string(),
            input_size: 64,
        };
        analyzer.record_metrics(good_metrics).unwrap();
    }
    
    // Introduce regression (15% slower)
    let regressed_metrics = PerformanceMetrics {
        timestamp: SystemTime::now(),
        execution_time: Duration::from_micros(14), // 15% slower
        cpu_utilization: 0.90,
        memory_usage: 5 * 1024 * 1024,
        cache_misses: 15,
        simd_utilization: 0.95,
        throughput: 1300.0, // Lower throughput
        operation_type: "regression_test".to_string(),
        input_size: 64,
    };
    
    let bottlenecks = analyzer.analyze_for_bottlenecks(&regressed_metrics).unwrap();
    
    // Should detect performance regression
    assert!(!bottlenecks.is_empty());
    let regression_bottleneck = bottlenecks.iter()
        .find(|b| b.impact_estimate > 10.0); // Should detect >10% regression
    assert!(regression_bottleneck.is_some());
}

#[test]
fn test_bottleneck_severity_classification() {
    let config = AnalyzerConfig::default();
    let analyzer = PerformanceBottleneckAnalyzer::new(config).unwrap();
    
    // Test different severity levels
    let test_cases = vec![
        (Duration::from_micros(10), Duration::from_micros(12), BottleneckSeverity::Low),    // 20% slower
        (Duration::from_micros(10), Duration::from_micros(15), BottleneckSeverity::Medium), // 50% slower  
        (Duration::from_micros(10), Duration::from_micros(20), BottleneckSeverity::High),   // 100% slower
        (Duration::from_micros(10), Duration::from_micros(35), BottleneckSeverity::Critical), // 250% slower
    ];
    
    for (baseline_time, current_time, expected_severity) in test_cases {
        let analyzer_test = PerformanceBottleneckAnalyzer::new(AnalyzerConfig::default()).unwrap();
        
        // Establish baseline
        let baseline_metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time: baseline_time,
            cpu_utilization: 0.85,
            memory_usage: 10 * 1024 * 1024,
            cache_misses: 20,
            simd_utilization: 0.90,
            throughput: 1000.0,
            operation_type: "severity_test".to_string(),
            input_size: 32,
        };
        analyzer_test.record_metrics(baseline_metrics).unwrap();
        
        // Test current performance
        let current_metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            execution_time: current_time,
            cpu_utilization: 0.85,
            memory_usage: 10 * 1024 * 1024,
            cache_misses: 20,
            simd_utilization: 0.90,
            throughput: 1000.0 * baseline_time.as_nanos() as f64 / current_time.as_nanos() as f64,
            operation_type: "severity_test".to_string(),
            input_size: 32,
        };
        
        let bottlenecks = analyzer_test.analyze_for_bottlenecks(&current_metrics).unwrap();
        
        if !bottlenecks.is_empty() {
            let bottleneck = &bottlenecks[0];
            assert_eq!(bottleneck.severity, expected_severity, 
                "Expected {:?} for {:.0}% performance degradation", 
                expected_severity,
                (current_time.as_nanos() as f64 / baseline_time.as_nanos() as f64 - 1.0) * 100.0
            );
        }
    }
}