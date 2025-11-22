//! Comprehensive benchmarks for TENGRI Event System
//! 
//! Tests performance against targets:
//! - Insertion latency: <1 microsecond
//! - Extraction latency: <1 microsecond
//! - Throughput: >1M events/second
//! - Memory efficiency: <100MB for 1M events

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

use tengri::events::{
    EventQueue, EventProcessor, EventEnvelope, NeuralEvent, EventFilter, ProcessorConfig
};

fn bench_queue_insertion(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("queue_insertion");
    group.throughput(Throughput::Elements(1));
    
    for queue_size in [1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("insert_spike_event", queue_size),
            queue_size,
            |b, &size| {
                let queue = EventQueue::new(size * 2); // Extra capacity
                
                b.to_async(&rt).iter(|| async {
                    let event = EventEnvelope::new(
                        black_box(NeuralEvent::Spike {
                            neuron_id: 12345,
                            layer_id: 2,
                            activation_value: 750,
                            metadata: HashMap::new(),
                        }),
                        black_box(200),
                        black_box("benchmark".to_string()),
                    );
                    
                    queue.insert(black_box(event)).await.unwrap();
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("insert_weight_update", queue_size),
            queue_size,
            |b, &size| {
                let queue = EventQueue::new(size * 2);
                
                b.to_async(&rt).iter(|| async {
                    let event = EventEnvelope::new(
                        black_box(NeuralEvent::WeightUpdate {
                            layer_id: 1,
                            from_neuron: 100,
                            to_neuron: 200,
                            weight_delta: -25,
                            learning_rate: 100,
                        }),
                        black_box(150),
                        black_box("benchmark".to_string()),
                    );
                    
                    queue.insert(black_box(event)).await.unwrap();
                });
            },
        );
    }
    
    group.finish();
}

fn bench_queue_extraction(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("queue_extraction");
    group.throughput(Throughput::Elements(1));
    
    for queue_size in [1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("extract_prioritized", queue_size),
            queue_size,
            |b, &size| {
                b.to_async(&rt).iter_batched(
                    || {
                        // Setup: Fill queue with events
                        let queue = EventQueue::new(size * 2);
                        rt.block_on(async {
                            for i in 0..size {
                                let event = EventEnvelope::new(
                                    NeuralEvent::Spike {
                                        neuron_id: i as u64,
                                        layer_id: (i % 10) as u32,
                                        activation_value: 500 + (i % 500) as u32,
                                        metadata: HashMap::new(),
                                    },
                                    (i % 255) as u8,
                                    format!("neuron_{}", i),
                                );
                                queue.insert(event).await.unwrap();
                            }
                            queue
                        })
                    },
                    |queue| async move {
                        // Benchmark: Extract single event
                        black_box(queue.extract().await.unwrap());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

fn bench_mixed_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("mixed_operations");
    group.throughput(Throughput::Elements(100));
    
    group.bench_function("insert_extract_cycle", |b| {
        let queue = EventQueue::new(50000);
        
        b.to_async(&rt).iter(|| async {
            // Insert 100 events with different priorities
            for i in 0..100 {
                let event = match i % 4 {
                    0 => EventEnvelope::new(
                        black_box(NeuralEvent::Spike {
                            neuron_id: i as u64,
                            layer_id: (i % 5) as u32,
                            activation_value: 500 + (i % 500) as u32,
                            metadata: HashMap::new(),
                        }),
                        black_box(200),
                        black_box(format!("spike_{}", i)),
                    ),
                    1 => EventEnvelope::new(
                        black_box(NeuralEvent::WeightUpdate {
                            layer_id: (i % 3) as u32,
                            from_neuron: i as u64,
                            to_neuron: (i + 1) as u64,
                            weight_delta: (i % 200) as i32 - 100,
                            learning_rate: 50 + (i % 50) as u32,
                        }),
                        black_box(150),
                        black_box(format!("weight_{}", i)),
                    ),
                    2 => EventEnvelope::new(
                        black_box(NeuralEvent::PredictionError {
                            output_id: (i % 10) as u32,
                            expected_value: 800,
                            actual_value: 600 + (i % 200) as u32,
                            error_magnitude: (i % 300) as u32,
                        }),
                        black_box(250),
                        black_box(format!("error_{}", i)),
                    ),
                    _ => EventEnvelope::new(
                        black_box(NeuralEvent::SystemEvent {
                            event_type: (i % 4) as u8,
                            metric_value: 500 + (i % 400) as u32,
                            threshold_breached: i % 3 == 0,
                        }),
                        black_box(100),
                        black_box(format!("system_{}", i)),
                    ),
                };
                
                queue.insert(black_box(event)).await.unwrap();
            }
            
            // Extract 50 events (leaving some in queue)
            for _ in 0..50 {
                black_box(queue.extract().await.unwrap());
            }
        });
    });
    
    group.finish();
}

fn bench_event_processor_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("processor_throughput");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(20);
    
    for batch_size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("process_neural_events", batch_size),
            batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter(|| async {
                    let queue = Arc::new(EventQueue::new(100000));
                    let config = ProcessorConfig {
                        max_batch_size: batch_size,
                        max_batch_age_ns: 1_000_000, // 1ms
                        processing_interval_ms: 1,
                        max_concurrent_processors: 4,
                        event_timeout_ms: 10,
                    };
                    
                    let processor = EventProcessor::new(queue.clone(), config);
                    processor.start().await.unwrap();
                    
                    let filter = EventFilter::accept_all();
                    let mut rx = processor.register_handler(filter).await;
                    
                    // Generate events
                    for i in 0..batch_size {
                        let event = EventEnvelope::new(
                            black_box(NeuralEvent::Spike {
                                neuron_id: i as u64,
                                layer_id: (i % 10) as u32,
                                activation_value: 500 + (i % 500) as u32,
                                metadata: HashMap::new(),
                            }),
                            black_box(150),
                            black_box(format!("bench_{}", i)),
                        );
                        
                        queue.insert(black_box(event)).await.unwrap();
                    }
                    
                    // Process events
                    let mut total_processed = 0;
                    while total_processed < batch_size {
                        if let Some(batch) = rx.recv().await {
                            total_processed += batch.events.len();
                            black_box(&batch);
                        } else {
                            break;
                        }
                    }
                    
                    processor.stop().await;
                    black_box(total_processed);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_event_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_filtering");
    group.throughput(Throughput::Elements(1));
    
    // Create test events
    let events = vec![
        EventEnvelope::new(
            NeuralEvent::Spike { neuron_id: 1, layer_id: 1, activation_value: 800, metadata: HashMap::new() },
            200,
            "layer_1".to_string(),
        ).with_tags(vec!["critical".to_string(), "layer_1".to_string()]),
        
        EventEnvelope::new(
            NeuralEvent::WeightUpdate { layer_id: 2, from_neuron: 1, to_neuron: 2, weight_delta: 50, learning_rate: 100 },
            150,
            "learning_engine".to_string(),
        ).with_tags(vec!["learning".to_string(), "layer_2".to_string()]),
        
        EventEnvelope::new(
            NeuralEvent::SystemEvent { event_type: 0, metric_value: 900, threshold_breached: true },
            100,
            "system_monitor".to_string(),
        ).with_tags(vec!["system".to_string(), "alert".to_string()]),
        
        EventEnvelope::new(
            NeuralEvent::PredictionError { output_id: 1, expected_value: 800, actual_value: 200, error_magnitude: 600 },
            250,
            "validator".to_string(),
        ).with_tags(vec!["critical".to_string(), "error".to_string()]),
    ];
    
    // Different filter complexities
    let simple_filter = EventFilter {
        event_types: vec!["spike".to_string()],
        required_tags: vec![],
        excluded_tags: vec![],
        priority_range: None,
        source_patterns: vec![],
    };
    
    let complex_filter = EventFilter {
        event_types: vec!["spike".to_string(), "prediction_error".to_string()],
        required_tags: vec!["critical".to_string()],
        excluded_tags: vec!["system".to_string()],
        priority_range: Some((150, 255)),
        source_patterns: vec!["layer".to_string()],
    };
    
    group.bench_function("simple_filter", |b| {
        b.iter(|| {
            for event in &events {
                black_box(simple_filter.matches(black_box(event)));
            }
        });
    });
    
    group.bench_function("complex_filter", |b| {
        b.iter(|| {
            for event in &events {
                black_box(complex_filter.matches(black_box(event)));
            }
        });
    });
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_usage");
    group.throughput(Throughput::Elements(10000));
    
    group.bench_function("queue_memory_footprint", |b| {
        b.to_async(&rt).iter(|| async {
            let queue = EventQueue::new(50000);
            
            // Fill queue with 10,000 events
            for i in 0..10000 {
                let event = EventEnvelope::new(
                    black_box(NeuralEvent::Spike {
                        neuron_id: i as u64,
                        layer_id: (i % 20) as u32,
                        activation_value: 500 + (i % 500) as u32,
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert("benchmark".to_string(), i as u32);
                            meta.insert("layer_info".to_string(), (i % 20) as u32);
                            black_box(meta)
                        },
                    }),
                    black_box((i % 255) as u8),
                    black_box(format!("neuron_{}", i)),
                ).with_tags(black_box(vec![
                    "benchmark".to_string(),
                    format!("layer_{}", i % 20),
                ]));
                
                queue.insert(black_box(event)).await.unwrap();
            }
            
            // Extract half of them
            for _ in 0..5000 {
                black_box(queue.extract().await.unwrap());
            }
            
            black_box(queue.size().await);
        });
    });
    
    group.finish();
}

fn bench_concurrent_access(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_access");
    group.throughput(Throughput::Elements(1000));
    
    group.bench_function("multi_producer_single_consumer", |b| {
        b.to_async(&rt).iter(|| async {
            let queue = Arc::new(EventQueue::new(10000));
            
            let mut producers = Vec::new();
            
            // Spawn 4 producer tasks
            for producer_id in 0..4 {
                let queue_clone = queue.clone();
                
                let producer = tokio::spawn(async move {
                    for i in 0..250 {
                        let event = EventEnvelope::new(
                            black_box(NeuralEvent::Spike {
                                neuron_id: (producer_id * 1000 + i) as u64,
                                layer_id: producer_id as u32,
                                activation_value: 500 + i as u32,
                                metadata: HashMap::new(),
                            }),
                            black_box(100 + (i % 155) as u8),
                            black_box(format!("producer_{}", producer_id)),
                        );
                        
                        queue_clone.insert(black_box(event)).await.unwrap();
                    }
                });
                
                producers.push(producer);
            }
            
            // Wait for all producers to finish
            for producer in producers {
                producer.await.unwrap();
            }
            
            // Single consumer extracts all events
            let mut extracted_count = 0;
            while let Some(_) = queue.extract().await.unwrap() {
                extracted_count += 1;
            }
            
            black_box(extracted_count);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_queue_insertion,
    bench_queue_extraction,
    bench_mixed_operations,
    bench_event_processor_throughput,
    bench_event_filtering,
    bench_memory_usage,
    bench_concurrent_access
);

criterion_main!(benches);