//! Quantum Security Performance Benchmarks
//!
//! Comprehensive benchmarks for all quantum security operations to ensure
//! sub-100μs latency targets are met.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use quantum_security::*;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark quantum security engine initialization
fn bench_engine_initialization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("engine_initialization", |b| {
        b.to_async(&rt).iter(|| async {
            let config = black_box(QuantumSecurityConfig::default());
            let engine = QuantumSecurityEngine::new(config).await.unwrap();
            black_box(engine)
        });
    });
}

/// Benchmark session initialization
fn bench_session_initialization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = QuantumSecurityConfig::default();
    let engine = rt.block_on(QuantumSecurityEngine::new(config)).unwrap();
    
    c.bench_function("session_initialization", |b| {
        b.to_async(&rt).iter(|| async {
            let agent_id = black_box("test_agent");
            let session_id = engine.initialize_session(agent_id).await.unwrap();
            black_box(session_id)
        });
    });
}

/// Benchmark post-quantum cryptographic operations
fn bench_post_quantum_crypto(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = QuantumSecurityConfig::default();
    let engine = rt.block_on(QuantumSecurityEngine::new(config)).unwrap();
    let session_id = rt.block_on(engine.initialize_session("bench_agent")).unwrap();
    
    let mut group = c.benchmark_group("post_quantum_crypto");
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark encryption operations
    for size in [64, 256, 1024, 4096].iter() {
        let data = vec![0u8; *size];
        
        group.bench_with_input(BenchmarkId::new("encrypt", size), size, |b, _| {
            b.to_async(&rt).iter(|| async {
                let data = black_box(data.clone());
                let encrypted = engine.encrypt_data(session_id, &data, None).await.unwrap();
                black_box(encrypted)
            });
        });
    }
    
    // Benchmark decryption operations
    let test_data = vec![0u8; 1024];
    let encrypted_data = rt.block_on(engine.encrypt_data(session_id, &test_data, None)).unwrap();
    
    group.bench_function("decrypt_1024", |b| {
        b.to_async(&rt).iter(|| async {
            let encrypted = black_box(&encrypted_data);
            let decrypted = engine.decrypt_data(session_id, encrypted).await.unwrap();
            black_box(decrypted)
        });
    });
    
    // Benchmark digital signatures
    let message = vec![0u8; 256];
    
    group.bench_function("sign_dilithium", |b| {
        b.to_async(&rt).iter(|| async {
            let msg = black_box(&message);
            let signature = engine.sign_data(session_id, msg, SignatureType::Dilithium).await.unwrap();
            black_box(signature)
        });
    });
    
    let signature = rt.block_on(engine.sign_data(session_id, &message, SignatureType::Dilithium)).unwrap();
    
    group.bench_function("verify_dilithium", |b| {
        b.to_async(&rt).iter(|| async {
            let msg = black_box(&message);
            let sig = black_box(&signature);
            let valid = engine.verify_signature(session_id, msg, sig).await.unwrap();
            black_box(valid)
        });
    });
    
    group.finish();
}

/// Benchmark quantum key distribution
fn bench_quantum_key_distribution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = QKDConfig::default();
    let manager = rt.block_on(QuantumKeyDistributionManager::new(config)).unwrap();
    
    let mut group = c.benchmark_group("quantum_key_distribution");
    group.measurement_time(Duration::from_secs(15));
    
    group.bench_function("establish_keys", |b| {
        b.to_async(&rt).iter(|| async {
            let alice = black_box("alice");
            let bob = black_box("bob");
            let keys = manager.establish_keys(alice, bob).await.unwrap();
            black_box(keys)
        });
    });
    
    group.finish();
}

/// Benchmark authentication operations
fn bench_authentication(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = QuantumSecurityConfig::default();
    let auth_manager = rt.block_on(QuantumAuthManager::new(config.authentication.clone())).unwrap();
    
    let mut group = c.benchmark_group("authentication");
    
    group.bench_function("create_context", |b| {
        b.to_async(&rt).iter(|| async {
            let agent_id = black_box("test_agent");
            let context = auth_manager.create_context(agent_id).await.unwrap();
            black_box(context)
        });
    });
    
    group.finish();
}

/// Benchmark threat detection
fn bench_threat_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = ThreatDetectionConfig::default();
    let detector = rt.block_on(QuantumThreatDetector::new(config)).unwrap();
    
    let mut group = c.benchmark_group("threat_detection");
    
    // Create sample threat detection event
    let event = ThreatDetectionEvent {
        event_id: uuid::Uuid::new_v4(),
        timestamp: chrono::Utc::now(),
        event_type: ThreatEventType::SuspiciousNetworkActivity,
        source: "test_source".to_string(),
        target: Some("test_target".to_string()),
        threat_level: ThreatLevel::Medium,
        threat_category: ThreatCategory::NetworkIntrusion,
        raw_data: vec![0u8; 1024],
        processed_data: std::collections::HashMap::new(),
        context: EventContext {
            session_id: None,
            agent_id: Some("test_agent".to_string()),
            operation_type: Some("network_activity".to_string()),
            source_ip: Some("192.168.1.1".to_string()),
            user_agent: None,
            geo_location: None,
            device_fingerprint: None,
            risk_factors: vec!["unusual_activity".to_string()],
        },
    };
    
    group.bench_function("assess_threat_level", |b| {
        b.to_async(&rt).iter(|| async {
            let agent_id = black_box("test_agent");
            let threat_level = detector.assess_threat_level(agent_id).await.unwrap();
            black_box(threat_level)
        });
    });
    
    group.bench_function("analyze_event", |b| {
        b.to_async(&rt).iter(|| async {
            let evt = black_box(&event);
            let result = detector.analyze_threat_event(evt).await.unwrap();
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark HSM operations
fn bench_hsm_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = HSMConfiguration::default();
    let manager = rt.block_on(HSMManager::new(config)).unwrap();
    
    let mut group = c.benchmark_group("hsm_operations");
    
    group.bench_function("generate_key", |b| {
        b.to_async(&rt).iter(|| async {
            let key_type = black_box(HSMKeyType::AES);
            let attributes = black_box(std::collections::HashMap::new());
            let operation = HSMOperation::GenerateKey {
                key_type,
                key_size: 256,
                attributes,
            };
            let result = manager.execute_operation(operation).await.unwrap();
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark communication encryption
fn bench_communication_encryption(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = QuantumSecurityConfig::default();
    let comm_manager = rt.block_on(QuantumCommunicationManager::new(config.communication.clone())).unwrap();
    
    let mut group = c.benchmark_group("communication");
    group.measurement_time(Duration::from_secs(10));
    
    // Test different message sizes
    for size in [64, 256, 1024, 4096, 16384].iter() {
        let message = vec![0u8; *size];
        
        group.bench_with_input(BenchmarkId::new("encrypt_message", size), size, |b, _| {
            b.to_async(&rt).iter(|| async {
                let msg = black_box(message.clone());
                let encrypted = comm_manager.encrypt_message("sender", "receiver", &msg).await.unwrap();
                black_box(encrypted)
            });
        });
    }
    
    group.finish();
}

/// Benchmark TENGRI integration
fn bench_tengri_integration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let quantum_config = QuantumSecurityConfig::default();
    let tengri_config = TENGRIConfig::default();
    let integration = rt.block_on(TENGRIQuantumIntegration::new(quantum_config, tengri_config)).unwrap();
    
    // Initialize agent
    rt.block_on(integration.initialize_agent_security("bench_agent")).unwrap();
    
    let mut group = c.benchmark_group("tengri_integration");
    
    let operation = TradingOperation {
        id: uuid::Uuid::new_v4(),
        operation_type: "buy_order".to_string(),
        agent_id: "bench_agent".to_string(),
        timestamp: chrono::Utc::now(),
        data: std::collections::HashMap::new(),
        security_level: SecurityLevel::High,
        requires_quantum_verification: true,
    };
    
    group.bench_function("validate_trading_operation", |b| {
        b.to_async(&rt).iter(|| async {
            let op = black_box(&operation);
            let result = integration.validate_trading_operation(op).await.unwrap();
            black_box(result)
        });
    });
    
    let message = vec![0u8; 1024];
    
    group.bench_function("secure_agent_communication", |b| {
        b.to_async(&rt).iter(|| async {
            let msg = black_box(&message);
            let secure_msg = integration.secure_agent_communication("bench_agent", "target_agent", msg).await.unwrap();
            black_box(secure_msg)
        });
    });
    
    group.finish();
}

/// Benchmark end-to-end operations to validate sub-100μs targets
fn bench_end_to_end_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = QuantumSecurityConfig::default();
    let engine = rt.block_on(QuantumSecurityEngine::new(config)).unwrap();
    let session_id = rt.block_on(engine.initialize_session("latency_test_agent")).unwrap();
    
    let mut group = c.benchmark_group("end_to_end_latency");
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(5));
    
    // Target: sub-100μs for critical operations
    let small_data = vec![0u8; 64]; // Small trading message
    
    group.bench_function("encrypt_decrypt_64bytes", |b| {
        b.to_async(&rt).iter(|| async {
            let data = black_box(&small_data);
            let encrypted = engine.encrypt_data(session_id, data, None).await.unwrap();
            let decrypted = engine.decrypt_data(session_id, &encrypted).await.unwrap();
            black_box(decrypted)
        });
    });
    
    group.bench_function("sign_verify_64bytes", |b| {
        b.to_async(&rt).iter(|| async {
            let data = black_box(&small_data);
            let signature = engine.sign_data(session_id, data, SignatureType::Dilithium).await.unwrap();
            let valid = engine.verify_signature(session_id, data, &signature).await.unwrap();
            black_box(valid)
        });
    });
    
    // Test concurrent operations
    group.bench_function("concurrent_encrypt_4_threads", |b| {
        b.to_async(&rt).iter(|| async {
            let futures = (0..4).map(|_| {
                let data = small_data.clone();
                engine.encrypt_data(session_id, &data, None)
            });
            
            let results = futures::future::join_all(futures).await;
            black_box(results)
        });
    });
    
    group.finish();
}

/// Memory usage benchmarks
fn bench_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_usage");
    
    group.bench_function("engine_memory_footprint", |b| {
        b.iter(|| {
            let config = black_box(QuantumSecurityConfig::default());
            let engine = rt.block_on(QuantumSecurityEngine::new(config)).unwrap();
            
            // Simulate typical usage
            let session_id = rt.block_on(engine.initialize_session("memory_test")).unwrap();
            let data = vec![0u8; 1024];
            let _encrypted = rt.block_on(engine.encrypt_data(session_id, &data, None)).unwrap();
            
            black_box(engine)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_engine_initialization,
    bench_session_initialization,
    bench_post_quantum_crypto,
    bench_quantum_key_distribution,
    bench_authentication,
    bench_threat_detection,
    bench_hsm_operations,
    bench_communication_encryption,
    bench_tengri_integration,
    bench_end_to_end_latency,
    bench_memory_usage
);

criterion_main!(benches)