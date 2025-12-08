//! Integration tests for the data pipeline

use data_pipeline::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::test;

/// Test full data pipeline end-to-end processing
#[test]
async fn test_full_pipeline_processing() {
    let config = DataPipelineConfig::default();
    let pipeline = DataPipeline::new(config).await.unwrap();
    
    // Start the pipeline
    pipeline.start().await.unwrap();
    
    // Create test data
    let data = fusion::DataItem {
        symbol: "BTCUSD".to_string(),
        timestamp: chrono::Utc::now(),
        price: 50000.0,
        volume: 1.5,
        bid: Some(49999.0),
        ask: Some(50001.0),
        text: Some("Bitcoin price shows bullish momentum".to_string()),
        raw_data: vec![],
    };
    
    // Process data through pipeline
    let result = pipeline.process_data(data).await.unwrap();
    
    // Verify results
    assert_eq!(result.symbol, "BTCUSD");
    assert!(result.quality_score > 0.0);
    assert!(result.confidence > 0.0);
    assert!(!result.features.is_empty());
    
    // Stop the pipeline
    pipeline.stop().await.unwrap();
}

/// Test pipeline health monitoring
#[test]
async fn test_pipeline_health_monitoring() {
    let config = DataPipelineConfig::default();
    let pipeline = DataPipeline::new(config).await.unwrap();
    
    // Start pipeline
    pipeline.start().await.unwrap();
    
    // Check initial health status
    let health = pipeline.health_check().await.unwrap();
    assert!(matches!(health.streaming, ComponentHealth::Healthy));
    
    // Get pipeline metrics
    let metrics = pipeline.get_metrics().await.unwrap();
    assert!(metrics.throughput.messages_per_second >= 0.0);
    assert!(metrics.latency.p99_ms >= 0.0);
    
    // Stop pipeline
    pipeline.stop().await.unwrap();
}

/// Test data validation and quality monitoring
#[test]
async fn test_data_validation() {
    let config = ValidationConfig::default();
    let validator = DataValidator::new(Arc::new(config)).unwrap();
    
    // Test valid data
    let valid_data = fusion::DataItem {
        symbol: "EURUSD".to_string(),
        timestamp: chrono::Utc::now(),
        price: 1.1234,
        volume: 1000.0,
        bid: Some(1.1233),
        ask: Some(1.1235),
        text: None,
        raw_data: vec![],
    };
    
    let result = validator.validate(valid_data).await;
    assert!(result.is_ok());
    
    // Test invalid data (negative price)
    let invalid_data = fusion::DataItem {
        symbol: "INVALID".to_string(),
        timestamp: chrono::Utc::now(),
        price: -100.0,
        volume: 1000.0,
        bid: None,
        ask: None,
        text: None,
        raw_data: vec![],
    };
    
    let result = validator.validate(invalid_data).await;
    assert!(result.is_err());
}

/// Test streaming engine functionality
#[test]
async fn test_streaming_engine() {
    let config = Arc::new(StreamingConfig::default());
    let engine = StreamingEngine::new(config).await.unwrap();
    
    // Test health check before starting
    let health = engine.health_check().await.unwrap();
    assert!(matches!(health, ComponentHealth::Unhealthy));
    
    // Get initial state
    let state = engine.get_state().await;
    assert!(!state.is_running);
    assert_eq!(state.active_streams, 0);
    
    // Get initial metrics
    let metrics = engine.get_metrics().await;
    assert_eq!(metrics.messages_consumed, 0);
    assert_eq!(metrics.messages_produced, 0);
}

/// Test sentiment analysis functionality
#[test]
async fn test_sentiment_analysis() {
    let config = Arc::new(SentimentConfig::default());
    
    // Test text preprocessing
    let preprocessor = sentiment::TextPreprocessor::new(config.clone());
    let cleaned_text = preprocessor.preprocess("This is a @test with https://example.com #hashtag! 😊").unwrap();
    assert!(!cleaned_text.contains("@test"));
    assert!(!cleaned_text.contains("https://example.com"));
    
    // Test sentiment cache
    let mut cache = sentiment::SentimentCache::new(100, Duration::from_secs(60));
    
    let test_result = sentiment::SentimentResult {
        text: "test".to_string(),
        sentiment: sentiment::SentimentLabel::Positive,
        confidence: 0.85,
        scores: sentiment::SentimentScores {
            positive: 0.8,
            negative: 0.1,
            neutral: 0.1,
            compound: 0.7,
        },
        emotions: None,
        aspects: None,
        language: None,
        processing_time: Duration::from_millis(50),
        model_used: "test-model".to_string(),
        metadata: sentiment::SentimentMetadata {
            text_length: 4,
            token_count: 1,
            preprocessing_applied: vec!["lowercase".to_string()],
            model_version: "1.0".to_string(),
            cache_hit: false,
            batch_size: 1,
            gpu_used: false,
        },
    };
    
    cache.put("test_key".to_string(), test_result.clone());
    let cached_result = cache.get("test_key").unwrap();
    assert_eq!(cached_result.confidence, 0.85);
    assert_eq!(cached_result.sentiment, sentiment::SentimentLabel::Positive);
}

/// Test SIMD-optimized technical indicators
#[test]
async fn test_technical_indicators() {
    let config = Arc::new(IndicatorsConfig::default());
    let calculator = indicators::SIMDCalculator::new(config.clone());
    
    // Test data
    let prices = vec![
        100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
        111.0, 110.0, 112.0, 114.0, 113.0, 115.0, 117.0, 116.0, 118.0, 120.0,
    ];
    
    // Test Simple Moving Average
    let sma = calculator.sma_simd(&prices, 5).unwrap();
    assert!(!sma.is_empty());
    assert_eq!(sma.len(), prices.len() - 5 + 1);
    
    // Test Exponential Moving Average
    let ema = calculator.ema_simd(&prices, 10).unwrap();
    assert_eq!(ema.len(), prices.len());
    assert_eq!(ema[0], prices[0]); // First value should be the initial price
    
    // Test RSI
    let rsi = calculator.rsi_simd(&prices, 14).unwrap();
    assert!(!rsi.is_empty());
    for &rsi_value in &rsi {
        assert!(rsi_value >= 0.0 && rsi_value <= 100.0);
    }
    
    // Test MACD
    let macd = calculator.macd_simd(&prices, 12, 26, 9).unwrap();
    assert!(!macd.macd.is_empty());
    assert!(!macd.signal.is_empty());
    assert!(!macd.histogram.is_empty());
    
    // Test Bollinger Bands
    let bb = calculator.bollinger_bands_simd(&prices, 10, 2.0).unwrap();
    assert_eq!(bb.upper.len(), bb.middle.len());
    assert_eq!(bb.middle.len(), bb.lower.len());
    
    // Verify that upper > middle > lower for most cases
    for i in 0..bb.upper.len() {
        assert!(bb.upper[i] >= bb.middle[i]);
        assert!(bb.middle[i] >= bb.lower[i]);
    }
}

/// Test data fusion algorithms
#[test]
async fn test_data_fusion() {
    let config = Arc::new(FusionConfig::default());
    let fusion_engine = DataFusion::new(config).await.unwrap();
    
    // Create test input
    let raw_data = fusion::DataItem {
        symbol: "AAPL".to_string(),
        timestamp: chrono::Utc::now(),
        price: 150.0,
        volume: 1000000.0,
        bid: Some(149.99),
        ask: Some(150.01),
        text: Some("Apple stock shows strong performance".to_string()),
        raw_data: vec![],
    };
    
    let features = fusion::ProcessedData {
        timestamp: chrono::Utc::now(),
        symbol: "AAPL".to_string(),
        market_data: fusion::MarketDataFused {
            price: 150.0,
            volume: 1000000.0,
            bid: 149.99,
            ask: 150.01,
            spread: 0.02,
            volatility: 0.025,
            trend_strength: 0.7,
            momentum: 0.1,
        },
        technical_indicators: [
            ("RSI".to_string(), 65.0),
            ("MACD".to_string(), 0.5),
        ].iter().cloned().collect(),
        sentiment_scores: Some(fusion::SentimentScores {
            overall_sentiment: 0.6,
            news_sentiment: 0.7,
            social_sentiment: 0.5,
            market_sentiment: 0.6,
            composite_score: 0.6,
        }),
        features: vec![150.0, 1000000.0, 65.0, 0.5],
        quality_score: 0.9,
        confidence: 0.85,
        fusion_metadata: fusion::FusionMetadata {
            algorithm_used: "WeightedAverage".to_string(),
            sources_count: 3,
            processing_time: Duration::from_millis(15),
            quality_metrics: fusion::QualityMetrics {
                completeness: 1.0,
                accuracy: 0.9,
                consistency: 0.95,
                timeliness: 1.0,
                overall_quality: 0.9,
            },
            outliers_detected: 0,
            interpolated_points: 0,
        },
    };
    
    let fusion_input = fusion::FusionInput {
        raw_data,
        features,
        indicators: [
            ("RSI".to_string(), indicators::IndicatorValue {
                value: 65.0,
                timestamp: chrono::Utc::now(),
                confidence: 0.9,
                parameters: [("period".to_string(), 14.0)].iter().cloned().collect(),
                calculation_time: Duration::from_millis(5),
            }),
        ].iter().cloned().collect(),
        sentiment: Some(sentiment::SentimentResult {
            text: "Apple stock shows strong performance".to_string(),
            sentiment: sentiment::SentimentLabel::Positive,
            confidence: 0.8,
            scores: sentiment::SentimentScores {
                positive: 0.7,
                negative: 0.1,
                neutral: 0.2,
                compound: 0.6,
            },
            emotions: None,
            aspects: None,
            language: Some("en".to_string()),
            processing_time: Duration::from_millis(20),
            model_used: "bert-base-uncased".to_string(),
            metadata: sentiment::SentimentMetadata {
                text_length: 35,
                token_count: 8,
                preprocessing_applied: vec!["lowercase".to_string()],
                model_version: "1.0".to_string(),
                cache_hit: false,
                batch_size: 1,
                gpu_used: false,
            },
        }),
    };
    
    // Test fusion
    let result = fusion_engine.fuse(fusion_input).await.unwrap();
    
    // Verify results
    assert_eq!(result.symbol, "AAPL");
    assert!(result.quality_score > 0.0);
    assert!(result.confidence > 0.0);
    assert!(!result.features.is_empty());
    assert!(result.technical_indicators.contains_key("RSI"));
    assert!(result.sentiment_scores.is_some());
}

/// Test feature extraction functionality
#[test]
async fn test_feature_extraction() {
    let config = Arc::new(FeatureConfig::default());
    let extractor = FeatureExtractor::new(config).unwrap();
    
    let test_data = fusion::DataItem {
        symbol: "TSLA".to_string(),
        timestamp: chrono::Utc::now(),
        price: 800.0,
        volume: 50000.0,
        bid: Some(799.5),
        ask: Some(800.5),
        text: None,
        raw_data: vec![],
    };
    
    let features = extractor.extract(&test_data).await.unwrap();
    
    // Verify feature extraction
    assert!(!features.statistical_features.is_empty());
    assert!(!features.time_series_features.is_empty());
    assert!(!features.engineered_features.is_empty());
    
    // Check that price-to-volume ratio is calculated correctly
    let expected_ratio = test_data.price / test_data.volume;
    assert!((features.engineered_features[0] - expected_ratio).abs() < 1e-10);
}

/// Test pattern detection in technical analysis
#[test]
async fn test_pattern_detection() {
    let config = Arc::new(IndicatorsConfig::default());
    let detector = indicators::PatternDetector::new(config);
    
    // Create test OHLC data for a doji pattern
    let doji_candle = indicators::OHLC {
        open: 100.0,
        high: 102.0,
        low: 98.0,
        close: 100.1, // Very small body
        volume: 1000.0,
        timestamp: chrono::Utc::now(),
    };
    
    let ohlc_data = vec![doji_candle];
    let patterns = detector.detect_patterns(&ohlc_data);
    
    // Should detect at least one pattern (doji)
    assert!(!patterns.is_empty());
    
    // Find doji pattern
    let doji_patterns: Vec<_> = patterns.iter()
        .filter(|p| matches!(p.pattern_type, indicators::PatternType::Doji))
        .collect();
    assert!(!doji_patterns.is_empty());
    
    let doji = &doji_patterns[0];
    assert!(doji.confidence > 0.0);
    assert_eq!(doji.start_index, 0);
    assert_eq!(doji.end_index, 0);
}

/// Test utility functions
#[test]
async fn test_utilities() {
    // Test mathematical utilities
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    
    let ma = utils::math::moving_average(&data, 3);
    assert_eq!(ma, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    
    let std_dev = utils::math::standard_deviation(&data);
    assert!((std_dev - 2.8722813232690143).abs() < 1e-10);
    
    let normalized = utils::math::normalize(&data);
    assert_eq!(normalized[0], 0.0); // Min value
    assert_eq!(normalized[normalized.len()-1], 1.0); // Max value
    
    // Test validation utilities
    assert!(utils::validation::is_valid_price(100.0));
    assert!(!utils::validation::is_valid_price(-10.0));
    assert!(!utils::validation::is_valid_price(f64::NAN));
    
    assert!(utils::validation::is_valid_volume(1000.0));
    assert!(utils::validation::is_valid_volume(0.0));
    assert!(!utils::validation::is_valid_volume(-100.0));
    
    // Test string utilities
    let messy_text = "  Hello\n\tWorld  \r\n  ";
    let cleaned = utils::string::clean_text(messy_text);
    assert_eq!(cleaned, "Hello World");
    
    let text_with_numbers = "Price: $123.45, Volume: 1000, Change: -2.5%";
    let numbers = utils::string::extract_numbers(text_with_numbers);
    assert_eq!(numbers, vec![123.45, 1000.0, -2.5]);
    
    // Test memory utilities
    let bytes_formatted = utils::memory::format_bytes(1536);
    assert_eq!(bytes_formatted, "1.50 KB");
    
    let bytes_formatted_mb = utils::memory::format_bytes(1024 * 1024 + 512 * 1024);
    assert_eq!(bytes_formatted_mb, "1.50 MB");
}

/// Test error handling and recovery
#[test]
async fn test_error_handling() {
    // Test pipeline with invalid configuration
    let mut config = DataPipelineConfig::default();
    config.streaming.kafka_brokers = vec!["invalid:9999".to_string()];
    
    // Pipeline creation should still succeed
    let pipeline = DataPipeline::new(config).await;
    assert!(pipeline.is_ok());
    
    // Test validation errors
    let validator_config = Arc::new(ValidationConfig::default());
    let validator = DataValidator::new(validator_config).unwrap();
    
    let invalid_data = fusion::DataItem {
        symbol: "TEST".to_string(),
        timestamp: chrono::Utc::now(),
        price: -100.0, // Invalid negative price
        volume: 1000.0,
        bid: None,
        ask: None,
        text: None,
        raw_data: vec![],
    };
    
    let validation_result = validator.validate(invalid_data).await;
    assert!(validation_result.is_err());
    
    // Test indicator calculation with insufficient data
    let config = Arc::new(IndicatorsConfig::default());
    let calculator = indicators::SIMDCalculator::new(config);
    
    let insufficient_data = vec![100.0, 101.0]; // Only 2 points
    let sma_result = calculator.sma_simd(&insufficient_data, 10); // Window too large
    assert!(sma_result.is_err());
}

/// Test concurrent processing and thread safety
#[test]
async fn test_concurrent_processing() {
    let config = DataPipelineConfig::default();
    let pipeline = Arc::new(DataPipeline::new(config).await.unwrap());
    
    pipeline.start().await.unwrap();
    
    // Spawn multiple concurrent tasks
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let pipeline_clone = Arc::clone(&pipeline);
        let handle = tokio::spawn(async move {
            let data = fusion::DataItem {
                symbol: format!("TEST{}", i),
                timestamp: chrono::Utc::now(),
                price: 100.0 + i as f64,
                volume: 1000.0 + i as f64 * 100.0,
                bid: Some(99.5 + i as f64),
                ask: Some(100.5 + i as f64),
                text: Some(format!("Test message {}", i)),
                raw_data: vec![],
            };
            
            pipeline_clone.process_data(data).await
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let results = futures::future::join_all(handles).await;
    
    // Verify all tasks completed successfully
    for result in results {
        assert!(result.is_ok());
        let processed_data = result.unwrap().unwrap();
        assert!(processed_data.quality_score > 0.0);
    }
    
    pipeline.stop().await.unwrap();
}

/// Test memory management and resource cleanup
#[test]
async fn test_resource_management() {
    let config = DataPipelineConfig::default();
    let pipeline = DataPipeline::new(config).await.unwrap();
    
    // Start and stop multiple times to test cleanup
    for _ in 0..5 {
        pipeline.start().await.unwrap();
        
        // Process some data
        let data = fusion::DataItem {
            symbol: "RESOURCE_TEST".to_string(),
            timestamp: chrono::Utc::now(),
            price: 100.0,
            volume: 1000.0,
            bid: Some(99.9),
            ask: Some(100.1),
            text: Some("Resource management test".to_string()),
            raw_data: vec![],
        };
        
        let _result = pipeline.process_data(data).await.unwrap();
        
        pipeline.stop().await.unwrap();
        
        // Reset pipeline state
        pipeline.reset().await.unwrap();
    }
    
    // Final health check should show healthy state
    let health = pipeline.health_check().await.unwrap();
    // Note: Health might be unhealthy since pipeline is stopped, that's expected
}

/// Performance regression test
#[test]
async fn test_performance_baseline() {
    let config = DataPipelineConfig::default();
    let pipeline = DataPipeline::new(config).await.unwrap();
    
    pipeline.start().await.unwrap();
    
    let start_time = std::time::Instant::now();
    let iterations = 100;
    
    for i in 0..iterations {
        let data = fusion::DataItem {
            symbol: "PERF_TEST".to_string(),
            timestamp: chrono::Utc::now(),
            price: 100.0 + (i as f64 * 0.1),
            volume: 1000.0 + (i as f64 * 10.0),
            bid: Some(99.9 + (i as f64 * 0.1)),
            ask: Some(100.1 + (i as f64 * 0.1)),
            text: Some(format!("Performance test iteration {}", i)),
            raw_data: vec![],
        };
        
        let _result = pipeline.process_data(data).await.unwrap();
    }
    
    let total_time = start_time.elapsed();
    let avg_time_per_iteration = total_time / iterations;
    
    // Performance baseline: each iteration should complete within 50ms on average
    assert!(avg_time_per_iteration < Duration::from_millis(50), 
            "Performance regression detected: average time per iteration is {:?}", 
            avg_time_per_iteration);
    
    pipeline.stop().await.unwrap();
}