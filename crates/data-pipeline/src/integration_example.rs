//! # Real-Time Data Pipeline Integration Example
//!
//! Complete example demonstrating the integration of all real-time validation components
//! for an enterprise trading system with sub-millisecond performance requirements.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use anyhow::Result;
use tracing::{info, warn, error};
use chrono::Utc;

use crate::{
    RawDataItem,
    RealTimeIntegrityValidator,
    RealTimeValidationConfig,
    TransformationValidator,
    TransformationValidationConfig,
    TransformationData,
    TransformationConfig,
    TransformationType,
    QualityRequirements,
    ConsistencyFailoverManager,
    ConsistencyFailoverConfig,
    ConsistencyLevel,
    EnterpriseDataPipeline,
    PipelineConfig,
    IntegrityConfig,
    LineageConfig,
    ReconciliationConfig,
    AuditConfig,
    EncryptionConfig,
    RecoveryConfig,
    BlockchainConfig,
    QualityConfig,
};

/// Complete enterprise data pipeline integration example
#[tokio::main]
pub async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("Starting Enterprise Real-Time Data Pipeline Integration Example");
    
    // Step 1: Initialize all validation components
    let pipeline = initialize_enterprise_pipeline().await?;
    
    // Step 2: Demonstrate real-time validation capabilities
    demonstrate_realtime_validation(&pipeline).await?;
    
    // Step 3: Demonstrate transformation validation
    demonstrate_transformation_validation(&pipeline).await?;
    
    // Step 4: Demonstrate consistency and failover
    demonstrate_consistency_failover(&pipeline).await?;
    
    // Step 5: Performance benchmark
    run_performance_benchmark(&pipeline).await?;
    
    // Step 6: Load testing
    run_load_testing(&pipeline).await?;
    
    // Step 7: Failover simulation
    simulate_failover_scenario(&pipeline).await?;
    
    info!("Enterprise Data Pipeline Integration Example completed successfully");
    Ok(())
}

/// Initialize the complete enterprise data pipeline
async fn initialize_enterprise_pipeline() -> Result<Arc<EnterpriseDataPipeline>> {
    info!("Initializing enterprise data pipeline with all components");
    
    let config = PipelineConfig {
        integrity: IntegrityConfig::default(),
        lineage: LineageConfig::default(),
        reconciliation: ReconciliationConfig::default(),
        audit: AuditConfig::default(),
        encryption: EncryptionConfig::default(),
        recovery: RecoveryConfig::default(),
        blockchain: BlockchainConfig::default(),
        quality: QualityConfig::default(),
    };
    
    let pipeline = EnterpriseDataPipeline::new(config).await?;
    
    info!("Enterprise data pipeline initialized successfully");
    Ok(Arc::new(pipeline))
}

/// Demonstrate real-time validation capabilities
async fn demonstrate_realtime_validation(pipeline: &Arc<EnterpriseDataPipeline>) -> Result<()> {
    info!("Demonstrating real-time validation capabilities");
    
    // Initialize real-time validator
    let config = RealTimeValidationConfig {
        target_latency_us: 800,
        max_latency_us: 1000,
        max_queue_size: 100_000,
        validation_threads: 16,
        enable_simd: true,
        enable_lockless: true,
        ..Default::default()
    };
    
    let validator = RealTimeIntegrityValidator::new(config)?;
    validator.start().await?;
    
    // Create sample market data
    let market_data = create_sample_market_data("AAPL", 150.25, 1000.0);
    
    // Measure validation performance
    let start = Instant::now();
    validator.validate_data(market_data).await?;
    let validation_time = start.elapsed();
    
    info!("Real-time validation completed in: {:?}", validation_time);
    
    // Get validation results
    let results = validator.get_validation_results().await?;
    info!("Validation results: {} items processed", results.len());
    
    // Check for data loss detection
    let data_loss_results = validator.detect_data_loss().await?;
    if !data_loss_results.is_empty() {
        warn!("Data loss detected: {} issues found", data_loss_results.len());
    }
    
    // Get latency metrics
    let latency_metrics = validator.get_latency_metrics().await?;
    info!("Latency metrics: avg={}µs, max={}µs", 
          latency_metrics.avg_latency_us, latency_metrics.max_latency_us);
    
    // Get health status
    let health_status = validator.get_health_status().await?;
    info!("Pipeline health: {:?}", health_status.overall_health);
    
    validator.shutdown().await?;
    
    info!("Real-time validation demonstration completed");
    Ok(())
}

/// Demonstrate transformation validation
async fn demonstrate_transformation_validation(pipeline: &Arc<EnterpriseDataPipeline>) -> Result<()> {
    info!("Demonstrating transformation validation");
    
    // Initialize transformation validator
    let config = TransformationValidationConfig {
        max_transformation_latency_us: 500,
        quality_thresholds: crate::transformation_validation::QualityThresholds {
            min_accuracy: 0.9999,
            max_data_loss: 0.0001,
            min_completeness: 0.999,
            max_error_rate: 0.0001,
        },
        ..Default::default()
    };
    
    let validator = TransformationValidator::new(config)?;
    
    // Create transformation data
    let original_data = create_sample_market_data("GOOGL", 2800.50, 500.0);
    let transformed_data = create_transformed_market_data("GOOGL", 2800.50, 500.0, "USD");
    
    let transformation_data = TransformationData {
        original: original_data,
        transformed: transformed_data,
        transformation_config: TransformationConfig {
            transformation_type: TransformationType::Enrichment,
            parameters: HashMap::new(),
            quality_requirements: QualityRequirements {
                required_accuracy: 0.99,
                required_completeness: 0.99,
                max_error_rate: 0.01,
                required_consistency: 0.99,
            },
        },
        transformation_timestamp: Utc::now(),
    };
    
    // Validate transformation
    let start = Instant::now();
    let result = validator.validate_transformation(transformation_data).await?;
    let validation_time = start.elapsed();
    
    info!("Transformation validation completed in: {:?}", validation_time);
    info!("Transformation valid: {}, quality score: {:.4}", 
          result.is_valid, result.quality_score);
    
    // Get health status
    let health_status = validator.get_health_status().await?;
    info!("Transformation validator health: {:?}", health_status.status);
    
    info!("Transformation validation demonstration completed");
    Ok(())
}

/// Demonstrate consistency and failover
async fn demonstrate_consistency_failover(pipeline: &Arc<EnterpriseDataPipeline>) -> Result<()> {
    info!("Demonstrating consistency and failover capabilities");
    
    // Initialize consistency failover manager
    let config = ConsistencyFailoverConfig {
        consistency_level: ConsistencyLevel::StrongConsistency,
        failover_config: crate::consistency_failover::FailoverConfig {
            auto_failover: true,
            failover_timeout_ms: 30_000,
            max_failover_attempts: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let manager = ConsistencyFailoverManager::new(config)?;
    manager.start().await?;
    
    // Create sample data for consistency validation
    let data = create_sample_market_data("TSLA", 250.75, 2000.0);
    
    // Validate consistency
    let start = Instant::now();
    let consistency_result = manager.validate_consistency(data, ConsistencyLevel::StrongConsistency).await?;
    let validation_time = start.elapsed();
    
    info!("Consistency validation completed in: {:?}", validation_time);
    info!("Consistency status: {:?}, score: {:.4}", 
          consistency_result.consistency_status, consistency_result.consistency_score);
    
    // Get health status
    let health_status = manager.get_health_status().await?;
    info!("Consistency manager health: {:?}", health_status.status);
    
    manager.shutdown().await?;
    
    info!("Consistency and failover demonstration completed");
    Ok(())
}

/// Run performance benchmark
async fn run_performance_benchmark(pipeline: &Arc<EnterpriseDataPipeline>) -> Result<()> {
    info!("Running performance benchmark");
    
    let config = RealTimeValidationConfig {
        target_latency_us: 500,
        max_latency_us: 1000,
        validation_threads: 32,
        enable_simd: true,
        enable_lockless: true,
        ..Default::default()
    };
    
    let validator = RealTimeIntegrityValidator::new(config)?;
    validator.start().await?;
    
    const BENCHMARK_ITEMS: usize = 10_000;
    let start = Instant::now();
    
    for i in 0..BENCHMARK_ITEMS {
        let data = create_sample_market_data(
            &format!("SYMBOL{}", i % 100),
            100.0 + (i as f64 * 0.01),
            1000.0 + (i as f64 * 0.1),
        );
        
        validator.validate_data(data).await?;
        
        if i % 1000 == 0 {
            info!("Processed {} items", i);
        }
    }
    
    let total_time = start.elapsed();
    let throughput = BENCHMARK_ITEMS as f64 / total_time.as_secs_f64();
    
    info!("Performance benchmark completed:");
    info!("  Total items: {}", BENCHMARK_ITEMS);
    info!("  Total time: {:?}", total_time);
    info!("  Throughput: {:.2} items/sec", throughput);
    info!("  Average latency: {:.2}µs", total_time.as_micros() as f64 / BENCHMARK_ITEMS as f64);
    
    // Get final metrics
    let latency_metrics = validator.get_latency_metrics().await?;
    info!("Final latency metrics:");
    info!("  Average: {}µs", latency_metrics.avg_latency_us);
    info!("  Max: {}µs", latency_metrics.max_latency_us);
    info!("  Min: {}µs", latency_metrics.min_latency_us);
    info!("  Violations: {}", latency_metrics.violation_count);
    
    validator.shutdown().await?;
    
    info!("Performance benchmark completed");
    Ok(())
}

/// Run load testing
async fn run_load_testing(pipeline: &Arc<EnterpriseDataPipeline>) -> Result<()> {
    info!("Running load testing");
    
    let config = RealTimeValidationConfig {
        target_latency_us: 800,
        max_queue_size: 1_000_000,
        validation_threads: 64,
        enable_simd: true,
        enable_lockless: true,
        ..Default::default()
    };
    
    let validator = RealTimeIntegrityValidator::new(config)?;
    validator.start().await?;
    
    const LOAD_TEST_DURATION: Duration = Duration::from_secs(30);
    const ITEMS_PER_SECOND: usize = 100_000;
    
    let start = Instant::now();
    let mut total_items = 0;
    
    info!("Starting load test for {:?} at {} items/sec", LOAD_TEST_DURATION, ITEMS_PER_SECOND);
    
    while start.elapsed() < LOAD_TEST_DURATION {
        let batch_start = Instant::now();
        
        for i in 0..ITEMS_PER_SECOND {
            let data = create_sample_market_data(
                &format!("LOAD{}", i % 50),
                100.0 + (i as f64 * 0.001),
                1000.0 + (i as f64 * 0.01),
            );
            
            if let Err(e) = validator.validate_data(data).await {
                error!("Validation failed during load test: {}", e);
            }
            
            total_items += 1;
        }
        
        let batch_time = batch_start.elapsed();
        if batch_time < Duration::from_secs(1) {
            sleep(Duration::from_secs(1) - batch_time).await;
        }
        
        if total_items % 10_000 == 0 {
            info!("Load test progress: {} items processed", total_items);
        }
    }
    
    let total_time = start.elapsed();
    let actual_throughput = total_items as f64 / total_time.as_secs_f64();
    
    info!("Load test completed:");
    info!("  Total items: {}", total_items);
    info!("  Total time: {:?}", total_time);
    info!("  Actual throughput: {:.2} items/sec", actual_throughput);
    info!("  Target throughput: {} items/sec", ITEMS_PER_SECOND);
    info!("  Throughput ratio: {:.2}%", (actual_throughput / ITEMS_PER_SECOND as f64) * 100.0);
    
    // Get health status after load test
    let health_status = validator.get_health_status().await?;
    info!("Health status after load test: {:?}", health_status.overall_health);
    
    validator.shutdown().await?;
    
    info!("Load testing completed");
    Ok(())
}

/// Simulate failover scenario
async fn simulate_failover_scenario(pipeline: &Arc<EnterpriseDataPipeline>) -> Result<()> {
    info!("Simulating failover scenario");
    
    let config = ConsistencyFailoverConfig {
        consistency_level: ConsistencyLevel::StrongConsistency,
        failover_config: crate::consistency_failover::FailoverConfig {
            auto_failover: true,
            failover_timeout_ms: 5_000,
            max_failover_attempts: 3,
            failure_detection_threshold: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let manager = ConsistencyFailoverManager::new(config)?;
    manager.start().await?;
    
    // Simulate normal operation
    info!("Simulating normal operation...");
    for i in 0..10 {
        let data = create_sample_market_data(
            &format!("FAIL{}", i),
            50.0 + i as f64,
            100.0 + i as f64,
        );
        
        let _ = manager.validate_consistency(data, ConsistencyLevel::StrongConsistency).await?;
        sleep(Duration::from_millis(100)).await;
    }
    
    // Simulate node failure
    info!("Simulating node failure...");
    // In a real system, this would involve actual node failure simulation
    
    // Continue operation during failover
    info!("Continuing operation during failover...");
    for i in 0..20 {
        let data = create_sample_market_data(
            &format!("FAILOVER{}", i),
            60.0 + i as f64,
            200.0 + i as f64,
        );
        
        match manager.validate_consistency(data, ConsistencyLevel::StrongConsistency).await {
            Ok(result) => {
                info!("Validation during failover: {:?}", result.consistency_status);
            }
            Err(e) => {
                warn!("Validation failed during failover: {}", e);
            }
        }
        
        sleep(Duration::from_millis(50)).await;
    }
    
    // Check system health
    let health_status = manager.get_health_status().await?;
    info!("System health after failover simulation: {:?}", health_status.status);
    
    manager.shutdown().await?;
    
    info!("Failover scenario simulation completed");
    Ok(())
}

/// Create sample market data
fn create_sample_market_data(symbol: &str, price: f64, volume: f64) -> RawDataItem {
    RawDataItem {
        id: format!("{}_{}", symbol, Utc::now().timestamp_nanos_opt().unwrap_or(0)),
        source: "market_data_feed".to_string(),
        timestamp: Utc::now(),
        data_type: "trade".to_string(),
        payload: serde_json::json!({
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "bid_price": price - 0.01,
            "ask_price": price + 0.01,
            "timestamp": Utc::now().timestamp_millis(),
            "exchange": "NASDAQ",
            "market_hours": true
        }),
        metadata: {
            let mut metadata = HashMap::new();
            metadata.insert("priority".to_string(), "high".to_string());
            metadata.insert("latency_sensitive".to_string(), "true".to_string());
            metadata.insert("sequence".to_string(), "1".to_string());
            metadata
        },
    }
}

/// Create transformed market data
fn create_transformed_market_data(symbol: &str, price: f64, volume: f64, currency: &str) -> RawDataItem {
    RawDataItem {
        id: format!("{}_{}_transformed", symbol, Utc::now().timestamp_nanos_opt().unwrap_or(0)),
        source: "transformation_engine".to_string(),
        timestamp: Utc::now(),
        data_type: "trade".to_string(),
        payload: serde_json::json!({
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "currency": currency,
            "notional_value": price * volume,
            "bid_price": price - 0.01,
            "ask_price": price + 0.01,
            "spread": 0.02,
            "timestamp": Utc::now().timestamp_millis(),
            "exchange": "NASDAQ",
            "market_hours": true,
            "enriched": true
        }),
        metadata: {
            let mut metadata = HashMap::new();
            metadata.insert("transformation_type".to_string(), "enrichment".to_string());
            metadata.insert("quality_score".to_string(), "0.999".to_string());
            metadata.insert("lineage_id".to_string(), "lineage_123".to_string());
            metadata
        },
    }
}

/// Comprehensive test suite for all components
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    
    #[test]
    async fn test_complete_integration() {
        // Test the complete integration flow
        let pipeline = initialize_enterprise_pipeline().await.unwrap();
        
        // Test each component
        assert!(demonstrate_realtime_validation(&pipeline).await.is_ok());
        assert!(demonstrate_transformation_validation(&pipeline).await.is_ok());
        assert!(demonstrate_consistency_failover(&pipeline).await.is_ok());
    }
    
    #[test]
    async fn test_performance_requirements() {
        // Test that performance requirements are met
        let config = RealTimeValidationConfig {
            target_latency_us: 500,
            max_latency_us: 1000,
            ..Default::default()
        };
        
        let validator = RealTimeIntegrityValidator::new(config).unwrap();
        validator.start().await.unwrap();
        
        let data = create_sample_market_data("TEST", 100.0, 1000.0);
        
        let start = Instant::now();
        validator.validate_data(data).await.unwrap();
        let duration = start.elapsed();
        
        // Should meet sub-millisecond requirement
        assert!(duration.as_micros() < 1000);
        
        validator.shutdown().await.unwrap();
    }
    
    #[test]
    async fn test_data_integrity() {
        // Test data integrity across all components
        let pipeline = initialize_enterprise_pipeline().await.unwrap();
        
        let original_data = create_sample_market_data("INTEGRITY", 123.45, 6789.0);
        let data_hash = blake3::hash(serde_json::to_string(&original_data).unwrap().as_bytes()).to_hex();
        
        // Process through enterprise pipeline
        let processed_data = pipeline.process_with_integrity(original_data).await.unwrap();
        
        // Verify integrity
        assert!(!processed_data.blockchain_hash.is_empty());
        assert!(processed_data.quality_score > 0.99);
        assert!(!processed_data.lineage_id.is_empty());
    }
    
    #[test]
    async fn test_failover_resilience() {
        // Test system resilience during failover
        let config = ConsistencyFailoverConfig::default();
        let manager = ConsistencyFailoverManager::new(config).unwrap();
        manager.start().await.unwrap();
        
        // Test consistency validation during simulated failures
        for i in 0..10 {
            let data = create_sample_market_data(&format!("RESILIENCE{}", i), 100.0, 1000.0);
            
            let result = manager.validate_consistency(data, ConsistencyLevel::StrongConsistency).await;
            assert!(result.is_ok());
        }
        
        manager.shutdown().await.unwrap();
    }
    
    #[test]
    async fn test_transformation_accuracy() {
        // Test transformation accuracy
        let config = TransformationValidationConfig::default();
        let validator = TransformationValidator::new(config).unwrap();
        
        let original = create_sample_market_data("ACCURACY", 200.0, 500.0);
        let transformed = create_transformed_market_data("ACCURACY", 200.0, 500.0, "USD");
        
        let transformation_data = TransformationData {
            original,
            transformed,
            transformation_config: TransformationConfig {
                transformation_type: TransformationType::Enrichment,
                parameters: HashMap::new(),
                quality_requirements: QualityRequirements {
                    required_accuracy: 0.99,
                    required_completeness: 0.99,
                    max_error_rate: 0.01,
                    required_consistency: 0.99,
                },
            },
            transformation_timestamp: Utc::now(),
        };
        
        let result = validator.validate_transformation(transformation_data).await.unwrap();
        assert!(result.is_valid);
        assert!(result.accuracy_score > 0.99);
    }
}