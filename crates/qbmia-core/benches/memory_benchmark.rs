//! Performance benchmarks for biological memory system

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use qbmia_core::{
    memory::{BiologicalMemory, MemoryConfig},
    config::HardwareConfig,
};
use std::collections::HashMap;

fn generate_test_experience(id: usize) -> HashMap<String, serde_json::Value> {
    let mut experience = HashMap::new();
    
    experience.insert("market_snapshot".to_string(), serde_json::json!({
        "price": 50000.0 + id as f64,
        "volume": 1000000.0,
        "volatility": 0.02,
        "trend": 0.1
    }));
    
    experience.insert("integrated_decision".to_string(), serde_json::json!({
        "action": "buy",
        "confidence": 0.8
    }));
    
    experience.insert("component_results".to_string(), serde_json::json!({
        "quantum_nash": {
            "equilibrium": {
                "convergence_score": 0.9
            }
        },
        "machiavellian": {
            "manipulation_detected": {
                "detected": false,
                "confidence": 0.2
            }
        }
    }));
    
    experience
}

fn bench_memory_storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_storage");
    
    for experience_count in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("store_experiences", experience_count),
            experience_count,
            |b, &experience_count| {
                b.iter(|| {
                    let memory_config = MemoryConfig {
                        capacity: 1000,
                        short_term_size: 100,
                        ..MemoryConfig::default()
                    };
                    let hardware_config = HardwareConfig::default();
                    let mut memory = BiologicalMemory::new(memory_config, hardware_config).unwrap();
                    
                    for i in 0..experience_count {
                        let experience = generate_test_experience(i);
                        memory.store_experience(&experience).unwrap();
                    }
                    
                    black_box(memory.get_usage_stats())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_recall");
    
    for memory_size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("recall_similar", memory_size),
            memory_size,
            |b, &memory_size| {
                // Pre-populate memory
                let memory_config = MemoryConfig {
                    capacity: memory_size,
                    short_term_size: memory_size / 10,
                    ..MemoryConfig::default()
                };
                let hardware_config = HardwareConfig::default();
                let mut memory = BiologicalMemory::new(memory_config, hardware_config).unwrap();
                
                for i in 0..memory_size / 2 {
                    let experience = generate_test_experience(i);
                    memory.store_experience(&experience).unwrap();
                }
                
                let query = generate_test_experience(42);
                
                b.iter(|| {
                    let results = memory.recall_similar_experiences(&query, 5).unwrap();
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_feature_extraction(c: &mut Criterion) {
    let memory_config = MemoryConfig::default();
    let hardware_config = HardwareConfig::default();
    let memory = BiologicalMemory::new(memory_config, hardware_config).unwrap();
    
    c.bench_function("feature_extraction", |b| {
        let experience = generate_test_experience(1);
        
        b.iter(|| {
            let features = memory.extract_features(&experience).unwrap();
            black_box(features)
        });
    });
}

fn bench_consolidation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_consolidation");
    
    for short_term_size in [10, 20, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("consolidate_memories", short_term_size),
            short_term_size,
            |b, &short_term_size| {
                b.iter(|| {
                    let memory_config = MemoryConfig {
                        capacity: 1000,
                        short_term_size,
                        ..MemoryConfig::default()
                    };
                    let hardware_config = HardwareConfig::default();
                    let mut memory = BiologicalMemory::new(memory_config, hardware_config).unwrap();
                    
                    // Fill short-term memory to trigger consolidation
                    for i in 0..short_term_size + 5 {
                        let experience = generate_test_experience(i);
                        memory.store_experience(&experience).unwrap();
                    }
                    
                    black_box(memory.get_usage_stats())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_attention_mechanism(c: &mut Criterion) {
    let memory_config = MemoryConfig::default();
    let hardware_config = HardwareConfig::default();
    let mut memory = BiologicalMemory::new(memory_config, hardware_config).unwrap();
    
    c.bench_function("attention_mechanism", |b| {
        let focus_areas = vec!["volatility".to_string(), "manipulation".to_string()];
        
        b.iter(|| {
            memory.apply_attention(&focus_areas);
            black_box(())
        });
    });
}

criterion_group!(
    benches,
    bench_memory_storage,
    bench_memory_recall,
    bench_feature_extraction,
    bench_consolidation,
    bench_attention_mechanism
);
criterion_main!(benches);