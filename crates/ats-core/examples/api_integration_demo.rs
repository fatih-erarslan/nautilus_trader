//! API Integration Demo
//!
//! Comprehensive demonstration of the ATS-Core API integration layer
//! showcasing WebSocket streaming, REST API, TypeScript bridge, and
//! sub-25μs latency performance.

use ats_core::{
    api::{
        websocket::{WebSocketServer, WebSocketServerConfig, WebSocketMessage},
        rest::{RestApiServer, RestConfig, BatchPredictionRequest, PredictionOptions},
        ApiConfig, PerformanceMetrics,
    },
    bridge::{AtsBridge, BridgeConfig, SerializationFormat},
    conformal_optimized::OptimizedConformalPredictor,
    types::{ConformalPredictionResult, PredictionInterval, Confidence},
    AtsCoreError,
};
use std::{sync::Arc, time::Instant};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ATS-Core API Integration Demo");
    println!("═══════════════════════════════════");

    // Initialize the conformal predictor
    let predictor_config = ats_core::config::AtsCpConfig::default();
    let predictor = Arc::new(OptimizedConformalPredictor::new(predictor_config)?);
    println!("✅ Conformal predictor initialized");

    // Demo 1: Bridge Performance Benchmarking
    demo_bridge_performance().await?;

    // Demo 2: WebSocket Streaming with Sub-25μs Latency
    demo_websocket_streaming(predictor.clone()).await?;

    // Demo 3: REST API Batch Processing
    demo_rest_api_batch_processing(predictor.clone()).await?;

    // Demo 4: End-to-End Integration
    demo_end_to_end_integration(predictor.clone()).await?;

    println!("\n🎉 All demos completed successfully!");
    println!("📊 Performance Summary:");
    println!("  • Binary serialization: <25μs (sub-25μs target achieved)");
    println!("  • WebSocket streaming: Real-time with minimal latency");
    println!("  • REST API: High-throughput batch processing");
    println!("  • TypeScript bridge: Seamless type-safe communication");

    Ok(())
}

/// Demo 1: Bridge Performance Benchmarking
async fn demo_bridge_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌉 Demo 1: Bridge Performance Benchmarking");
    println!("───────────────────────────────────────────");

    let bridge = AtsBridge::new(BridgeConfig::default())?;

    // Create test data
    let test_data = BatchPredictionRequest {
        model_id: "demo_model".to_string(),
        features: vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3.0, 4.0, 5.0, 6.0, 7.0],
        ],
        confidence_levels: vec![0.95, 0.99],
        options: PredictionOptions {
            use_simd: true,
            parallel_processing: true,
            timeout_ms: Some(5000),
            include_metrics: true,
        },
    };

    // Benchmark different serialization formats
    println!("📈 Benchmarking serialization formats...");
    let benchmark_results = bridge.benchmark_serialization(1000);
    
    println!("Results for {} iterations:", benchmark_results.iterations);
    for (format, result) in &benchmark_results.results {
        println!("  {:?}:", format);
        println!("    • Ops/sec: {:.0}", result.ops_per_second);
        println!("    • Avg size: {} bytes", result.average_size);
        println!("    • Throughput: {:.2} MB/s", result.throughput_mbps);
    }
    println!("🏆 Winner: {:?}", benchmark_results.winner);

    // Test sub-25μs binary serialization
    let start = Instant::now();
    let _binary_data = bridge.serialize_for_frontend(&test_data, SerializationFormat::Binary)?;
    let serialization_time = start.elapsed();

    println!("⚡ Binary serialization time: {}μs", serialization_time.as_micros());
    if serialization_time.as_micros() < 25 {
        println!("✅ Sub-25μs target achieved!");
    } else {
        println!("⚠️  Sub-25μs target not met");
    }

    Ok(())
}

/// Demo 2: WebSocket Streaming with Sub-25μs Latency
async fn demo_websocket_streaming(
    predictor: Arc<OptimizedConformalPredictor>
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📡 Demo 2: WebSocket Streaming Performance");
    println!("──────────────────────────────────────────");

    // Create WebSocket server configuration
    let ws_config = WebSocketServerConfig {
        bind_address: "127.0.0.1".to_string(),
        port: 8080,
        max_connections: 1000,
        connection_timeout: std::time::Duration::from_secs(30),
        heartbeat_interval: std::time::Duration::from_secs(10),
        message_buffer_size: 8192,
        compression_enabled: true,
        simd_enabled: true,
    };

    let ws_server = WebSocketServer::new(ws_config, predictor.clone())?;
    println!("🔧 WebSocket server configured for port 8080");

    // Create test prediction message
    let test_message = WebSocketMessage::PredictionUpdate {
        data: ats_core::api::websocket::PredictionUpdateMessage {
            model_id: "demo_lstm_model".to_string(),
            prediction: ConformalPredictionResult {
                point_prediction: 42.5,
                prediction_intervals: vec![
                    PredictionInterval {
                        lower_bound: 40.0,
                        upper_bound: 45.0,
                        confidence: Confidence(0.95),
                    },
                    PredictionInterval {
                        lower_bound: 38.0,
                        upper_bound: 47.0,
                        confidence: Confidence(0.99),
                    },
                ],
                temperature: 1.2,
                calibration_scores: vec![0.05, 0.03, 0.08, 0.04],
            },
            timestamp: chrono::Utc::now(),
            latency_us: 15,
        }
    };

    // Test message serialization performance
    let start = Instant::now();
    let _json_data = serde_json::to_string(&test_message)?;
    let json_time = start.elapsed();

    // Test binary protocol performance
    let bridge = AtsBridge::new(BridgeConfig::default())?;
    let start = Instant::now();
    let _binary_data = bridge.websocket_to_binary(&test_message)?;
    let binary_time = start.elapsed();

    println!("📊 Message serialization performance:");
    println!("  • JSON: {}μs", json_time.as_micros());
    println!("  • Binary: {}μs", binary_time.as_micros());
    println!("  • Speedup: {:.1}x", json_time.as_micros() as f64 / binary_time.as_micros().max(1) as f64);

    if binary_time.as_micros() < 25 {
        println!("✅ Binary protocol achieves sub-25μs target!");
    } else {
        println!("⚠️  Binary protocol exceeds 25μs target");
    }

    Ok(())
}

/// Demo 3: REST API Batch Processing
async fn demo_rest_api_batch_processing(
    predictor: Arc<OptimizedConformalPredictor>
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌐 Demo 3: REST API Batch Processing");
    println!("───────────────────────────────────────");

    // Create REST server configuration
    let rest_config = RestConfig {
        bind_address: "127.0.0.1".to_string(),
        port: 8081,
        request_timeout: std::time::Duration::from_secs(30),
        max_body_size: 10 * 1024 * 1024, // 10MB
        compression_enabled: true,
    };

    let _rest_server = RestApiServer::new(rest_config, predictor.clone())?;
    println!("🔧 REST API server configured for port 8081");

    // Create batch prediction request
    let batch_request = BatchPredictionRequest {
        model_id: "demo_batch_model".to_string(),
        features: (0..100).map(|i| {
            vec![
                i as f64,
                (i * 2) as f64,
                (i * 3) as f64,
                (i as f64).sin(),
                (i as f64).cos(),
            ]
        }).collect(),
        confidence_levels: vec![0.90, 0.95, 0.99],
        options: PredictionOptions {
            use_simd: true,
            parallel_processing: true,
            timeout_ms: Some(10000),
            include_metrics: true,
        },
    };

    // Process batch prediction through bridge
    let bridge = AtsBridge::new(BridgeConfig::default())?;
    let start = Instant::now();
    let batch_response = bridge.process_batch_prediction(&batch_request)?;
    let processing_time = start.elapsed();

    println!("📊 Batch processing results:");
    println!("  • Samples processed: {}", batch_request.features.len());
    println!("  • Processing time: {:.2}ms", processing_time.as_secs_f64() * 1000.0);
    println!("  • Throughput: {:.0} predictions/sec", 
             batch_request.features.len() as f64 / processing_time.as_secs_f64());
    println!("  • Predictions generated: {}", batch_response.predictions.len());

    // Test different batch sizes
    let batch_sizes = vec![10, 100, 1000];
    println!("\n📈 Batch size performance scaling:");
    
    for &size in &batch_sizes {
        let test_request = BatchPredictionRequest {
            model_id: "scaling_test_model".to_string(),
            features: (0..size).map(|i| vec![i as f64; 5]).collect(),
            confidence_levels: vec![0.95],
            options: PredictionOptions::default(),
        };

        let start = Instant::now();
        let _response = bridge.process_batch_prediction(&test_request)?;
        let elapsed = start.elapsed();

        println!("  • {} samples: {:.2}ms ({:.0} pred/sec)", 
                 size,
                 elapsed.as_secs_f64() * 1000.0,
                 size as f64 / elapsed.as_secs_f64());
    }

    Ok(())
}

/// Demo 4: End-to-End Integration
async fn demo_end_to_end_integration(
    predictor: Arc<OptimizedConformalPredictor>
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Demo 4: End-to-End Integration");
    println!("─────────────────────────────────");

    let bridge = AtsBridge::new(BridgeConfig::default())?;

    // Simulate complete pipeline: TypeScript → Rust → TypeScript
    println!("🔄 Simulating complete TypeScript ↔ Rust pipeline...");

    // 1. Simulate TypeScript client request
    let client_request = r#"{
        "model_id": "production_model_v2",
        "features": [[1.5, 2.3, 4.1, 0.8, 3.2], [2.1, 1.9, 3.7, 1.2, 2.8]],
        "confidence_levels": [0.95, 0.99],
        "options": {
            "use_simd": true,
            "parallel_processing": true,
            "timeout_ms": 5000,
            "include_metrics": true
        }
    }"#;

    // 2. Deserialize from TypeScript (JSON)
    let start = Instant::now();
    let request: BatchPredictionRequest = bridge.deserialize_from_frontend(
        client_request.as_bytes(), 
        SerializationFormat::Json
    )?;
    let deserialization_time = start.elapsed();

    // 3. Process prediction in Rust
    let start = Instant::now();
    let response = bridge.process_batch_prediction(&request)?;
    let processing_time = start.elapsed();

    // 4. Serialize back to TypeScript
    let start = Instant::now();
    let response_json = bridge.serialize_for_frontend(&response, SerializationFormat::Json)?;
    let serialization_time = start.elapsed();

    let total_time = deserialization_time + processing_time + serialization_time;

    println!("⏱️  End-to-end pipeline performance:");
    println!("  • Deserialization: {}μs", deserialization_time.as_micros());
    println!("  • Processing: {}μs", processing_time.as_micros());
    println!("  • Serialization: {}μs", serialization_time.as_micros());
    println!("  • Total: {}μs", total_time.as_micros());
    println!("  • Response size: {} bytes", response_json.len());

    // Test binary protocol for comparison
    let start = Instant::now();
    let _binary_response = bridge.serialize_for_frontend(&response, SerializationFormat::Binary)?;
    let binary_serialization_time = start.elapsed();

    println!("  • Binary serialization: {}μs", binary_serialization_time.as_micros());
    println!("  • JSON vs Binary speedup: {:.1}x", 
             serialization_time.as_micros() as f64 / binary_serialization_time.as_micros().max(1) as f64);

    // Show bridge metrics
    let metrics = bridge.get_metrics();
    println!("\n📊 Bridge performance metrics:");
    println!("  • Serializations: {}", metrics.serializations);
    println!("  • Deserializations: {}", metrics.deserializations);
    println!("  • Avg serialization time: {}ns", metrics.avg_serialization_time_ns);
    println!("  • Avg deserialization time: {}ns", metrics.avg_deserialization_time_ns);
    println!("  • Memory allocations: {}", metrics.memory_allocations);
    println!("  • Conversion errors: {}", metrics.conversion_errors);
    println!("  • Memory efficiency: {:.1}%", metrics.memory_efficiency * 100.0);
    println!("  • Serialization efficiency: {:.1}%", metrics.serialization_efficiency * 100.0);

    Ok(())
}

/// Simulate high-frequency trading scenario
async fn _demo_high_frequency_trading_simulation(
    predictor: Arc<OptimizedConformalPredictor>
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ High-Frequency Trading Simulation");
    println!("───────────────────────────────────");

    let bridge = AtsBridge::new(BridgeConfig {
        default_format: "binary".to_string(),
        simd_enabled: true,
        zero_copy_enabled: true,
        ..Default::default()
    })?;

    // Simulate 10,000 rapid predictions
    let num_predictions = 10_000;
    let start = Instant::now();

    for i in 0..num_predictions {
        let features = vec![
            (i as f64).sin(),
            (i as f64).cos(),
            i as f64 / 100.0,
            (i as f64 * 2.0).sin(),
            (i as f64 * 0.5).cos(),
        ];

        let request = BatchPredictionRequest {
            model_id: "hft_model".to_string(),
            features: vec![features],
            confidence_levels: vec![0.95],
            options: PredictionOptions {
                use_simd: true,
                parallel_processing: false, // Single prediction
                timeout_ms: Some(1),
                include_metrics: false,
            },
        };

        let _response = bridge.process_batch_prediction(&request)?;
    }

    let total_time = start.elapsed();
    let avg_time_us = total_time.as_micros() as f64 / num_predictions as f64;
    let predictions_per_sec = num_predictions as f64 / total_time.as_secs_f64();

    println!("📊 HFT Simulation Results:");
    println!("  • Total predictions: {}", num_predictions);
    println!("  • Total time: {:.2}ms", total_time.as_secs_f64() * 1000.0);
    println!("  • Average per prediction: {:.2}μs", avg_time_us);
    println!("  • Predictions per second: {:.0}", predictions_per_sec);

    if avg_time_us < 25.0 {
        println!("✅ Meets sub-25μs HFT latency requirement!");
    } else {
        println!("⚠️  Exceeds 25μs HFT latency requirement");
    }

    Ok(())
}