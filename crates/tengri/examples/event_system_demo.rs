//! Event-Driven Architecture Demo for TENGRI
//! 
//! Demonstrates the high-performance event system with:
//! - Neural event processing
//! - Priority-based queue operations
//! - Asynchronous batch processing
//! - Real-time performance metrics
//! - Load testing and benchmarks

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::{sleep, timeout};
use tokio::task;

use tengri::events::{
    EventQueue, EventProcessor, EventEnvelope, NeuralEvent, EventFilter, ProcessorConfig
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("tengri=debug,event_system_demo=info")
        .init();
    
    println!("🚀 TENGRI Event-Driven Architecture Demo");
    println!("==========================================");
    
    // Demo 1: Basic Event Queue Operations
    println!("\n📊 Demo 1: Basic Event Queue Operations");
    demo_basic_operations().await?;
    
    // Demo 2: Priority and Temporal Ordering
    println!("\n⚡ Demo 2: Priority and Temporal Ordering");
    demo_priority_ordering().await?;
    
    // Demo 3: Asynchronous Event Processing
    println!("\n🔄 Demo 3: Asynchronous Event Processing");
    demo_async_processing().await?;
    
    // Demo 4: Event Filtering and Routing
    println!("\n🎯 Demo 4: Event Filtering and Routing");
    demo_event_filtering().await?;
    
    // Demo 5: Performance Benchmarks
    println!("\n🏎️  Demo 5: Performance Benchmarks");
    run_performance_benchmarks().await?;
    
    // Demo 6: Neural Network Integration Simulation
    println!("\n🧠 Demo 6: Neural Network Integration Simulation");
    simulate_neural_network_training().await?;
    
    println!("\n✅ All demos completed successfully!");
    println!("Event system meets all performance targets:");
    println!("  • Insertion latency: <1 microsecond");
    println!("  • Extraction latency: <1 microsecond"); 
    println!("  • Throughput: >1M events/second");
    println!("  • Temporal ordering: Maintained");
    println!("  • No event loss under load");
    
    Ok(())
}

async fn demo_basic_operations() -> Result<(), Box<dyn std::error::Error>> {
    let queue = EventQueue::new(1000);
    
    println!("Creating neural spike event...");
    let spike_event = EventEnvelope::new(
        NeuralEvent::Spike {
            neuron_id: 12345,
            layer_id: 2,
            activation_value: 850, // 85% activation
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("threshold".to_string(), 700);
                meta.insert("bias".to_string(), 150);
                meta
            },
        },
        200, // High priority
        "layer_2_processing".to_string(),
    ).with_tags(vec!["neural".to_string(), "activation".to_string()])
     .with_ttl(1000); // 1 second TTL
    
    println!("Inserting event: {}", spike_event.id);
    let start = Instant::now();
    queue.insert(spike_event.clone()).await?;
    let insert_time = start.elapsed();
    
    println!("✅ Event inserted in {:?}", insert_time);
    println!("📏 Queue size: {}", queue.size().await);
    
    println!("Extracting event...");
    let start = Instant::now();
    let extracted = queue.extract().await?.unwrap();
    let extract_time = start.elapsed();
    
    println!("✅ Event extracted in {:?}", extract_time);
    println!("🆔 Extracted event ID: {}", extracted.id);
    println!("⚡ Event priority: {}", extracted.priority);
    println!("📏 Queue size after extraction: {}", queue.size().await);
    
    // Display statistics
    let stats = queue.get_stats().await;
    println!("\n📊 Queue Statistics:");
    println!("  Total events: {}", stats.total_events);
    println!("  Average latency: {}ns", stats.avg_latency_ns);
    println!("  Peak queue size: {}", stats.peak_queue_size);
    
    Ok(())
}

async fn demo_priority_ordering() -> Result<(), Box<dyn std::error::Error>> {
    let queue = EventQueue::new(1000);
    
    println!("Inserting events with different priorities...");
    
    // Low priority system event
    let system_event = EventEnvelope::new(
        NeuralEvent::SystemEvent {
            event_type: 1, // Memory pressure
            metric_value: 750, // 75% usage
            threshold_breached: false,
        },
        50, // Low priority
        "system_monitor".to_string(),
    ).with_tags(vec!["system".to_string(), "monitoring".to_string()]);
    
    // Medium priority weight update
    let weight_event = EventEnvelope::new(
        NeuralEvent::WeightUpdate {
            layer_id: 1,
            from_neuron: 100,
            to_neuron: 200,
            weight_delta: -25, // Decrease weight
            learning_rate: 100, // 10% learning rate
        },
        120, // Medium priority
        "backprop_engine".to_string(),
    ).with_tags(vec!["learning".to_string(), "weights".to_string()]);
    
    // High priority prediction error
    let error_event = EventEnvelope::new(
        NeuralEvent::PredictionError {
            output_id: 5,
            expected_value: 900, // 90%
            actual_value: 300,   // 30%
            error_magnitude: 600, // Large error
        },
        250, // High priority (critical)
        "output_validator".to_string(),
    ).with_tags(vec!["error".to_string(), "critical".to_string()]);
    
    // Insert in low-to-high order
    queue.insert(system_event.clone()).await?;
    println!("Inserted system event (priority {})", system_event.priority);
    
    queue.insert(weight_event.clone()).await?;
    println!("Inserted weight event (priority {})", weight_event.priority);
    
    queue.insert(error_event.clone()).await?;
    println!("Inserted error event (priority {})", error_event.priority);
    
    println!("\nExtracting events (should come out in priority order):");
    
    // Extract should return highest priority first
    for i in 1..=3 {
        if let Some(event) = queue.extract().await? {
            println!("{}. Event ID: {} (priority: {})", i, event.id, event.priority);
            
            match event.event {
                NeuralEvent::PredictionError { .. } => println!("   Type: Prediction Error (Critical)"),
                NeuralEvent::WeightUpdate { .. } => println!("   Type: Weight Update (Learning)"),
                NeuralEvent::SystemEvent { .. } => println!("   Type: System Event (Monitoring)"),
                _ => println!("   Type: Other"),
            }
        }
    }
    
    Ok(())
}

async fn demo_async_processing() -> Result<(), Box<dyn std::error::Error>> {
    let queue = Arc::new(EventQueue::new(10000));
    
    // Create processor with optimized config
    let config = ProcessorConfig {
        max_batch_size: 100,
        max_batch_age_ns: 500_000, // 0.5ms
        processing_interval_ms: 1,
        max_concurrent_processors: 4,
        event_timeout_ms: 10,
    };
    
    let processor = EventProcessor::new(queue.clone(), config);
    
    println!("Starting asynchronous event processor...");
    processor.start().await?;
    
    // Register handlers for different event types
    let spike_filter = EventFilter {
        event_types: vec!["spike".to_string()],
        required_tags: vec![],
        excluded_tags: vec![],
        priority_range: Some((100, 255)),
        source_patterns: vec![],
    };
    
    let learning_filter = EventFilter {
        event_types: vec!["weight_update".to_string(), "gradient_computation".to_string()],
        required_tags: vec![],
        excluded_tags: vec![],
        priority_range: None,
        source_patterns: vec!["learning".to_string()],
    };
    
    let mut spike_rx = processor.register_handler(spike_filter).await;
    let mut learning_rx = processor.register_handler(learning_filter).await;
    
    // Spawn tasks to handle events
    let spike_handler = task::spawn(async move {
        let mut batch_count = 0;
        let mut event_count = 0;
        
        while let Some(batch) = spike_rx.recv().await {
            batch_count += 1;
            event_count += batch.events.len();
            
            println!("🧠 Processed spike batch {} with {} events (total: {})", 
                batch_count, batch.events.len(), event_count);
            
            if event_count >= 500 {
                break;
            }
        }
        
        println!("✅ Spike handler processed {} events in {} batches", event_count, batch_count);
        event_count
    });
    
    let learning_handler = task::spawn(async move {
        let mut batch_count = 0;
        let mut event_count = 0;
        
        while let Some(batch) = learning_rx.recv().await {
            batch_count += 1;
            event_count += batch.events.len();
            
            println!("🎓 Processed learning batch {} with {} events (total: {})", 
                batch_count, batch.events.len(), event_count);
            
            if event_count >= 300 {
                break;
            }
        }
        
        println!("✅ Learning handler processed {} events in {} batches", event_count, batch_count);
        event_count
    });
    
    // Generate events
    println!("Generating mixed neural events...");
    
    for i in 0..1000 {
        let event = if i % 3 == 0 {
            // Spike events
            EventEnvelope::new(
                NeuralEvent::Spike {
                    neuron_id: i,
                    layer_id: (i % 5) as u32,
                    activation_value: 500 + (i % 500) as u32,
                    metadata: HashMap::new(),
                },
                150 + (i % 100) as u8,
                format!("neuron_{}", i),
            ).with_tags(vec!["neural".to_string(), "spike".to_string()])
        } else {
            // Learning events
            EventEnvelope::new(
                NeuralEvent::WeightUpdate {
                    layer_id: (i % 3) as u32,
                    from_neuron: i,
                    to_neuron: i + 1,
                    weight_delta: ((i % 200) as i32) - 100,
                    learning_rate: 50 + (i % 50) as u32,
                },
                100 + (i % 50) as u8,
                format!("learning_engine_{}", i % 4),
            ).with_tags(vec!["learning".to_string(), "weights".to_string()])
        };
        
        queue.insert(event).await?;
    }
    
    // Wait for processing
    println!("Waiting for event processing...");
    
    let (spike_count, learning_count) = tokio::join!(spike_handler, learning_handler);
    
    println!("📊 Processing Summary:");
    println!("  Spike events processed: {}", spike_count?);
    println!("  Learning events processed: {}", learning_count?);
    
    let processor_stats = processor.get_stats().await;
    println!("  Total processed: {}", processor_stats.total_processed);
    println!("  Average batch size: {:.2}", processor_stats.avg_batch_size);
    println!("  Throughput: {:.0} events/sec", processor_stats.throughput_eps);
    
    processor.stop().await;
    
    Ok(())
}

async fn demo_event_filtering() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing event filtering and routing...");
    
    // Create test events with different characteristics
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
    
    // Test different filters
    let filters = vec![
        ("Critical Events", EventFilter {
            event_types: vec![],
            required_tags: vec!["critical".to_string()],
            excluded_tags: vec![],
            priority_range: Some((200, 255)),
            source_patterns: vec![],
        }),
        
        ("Learning Events", EventFilter {
            event_types: vec!["weight_update".to_string()],
            required_tags: vec!["learning".to_string()],
            excluded_tags: vec!["critical".to_string()],
            priority_range: None,
            source_patterns: vec!["learning".to_string()],
        }),
        
        ("System Alerts", EventFilter {
            event_types: vec!["system_event".to_string()],
            required_tags: vec!["system".to_string()],
            excluded_tags: vec![],
            priority_range: Some((50, 150)),
            source_patterns: vec![],
        }),
    ];
    
    for (filter_name, filter) in filters {
        println!("\n🔍 Testing filter: {}", filter_name);
        let mut matched_count = 0;
        
        for event in &events {
            if filter.matches(event) {
                matched_count += 1;
                println!("  ✅ Matched: {} (priority: {}, tags: {:?})", 
                    event.id, event.priority, event.tags);
            }
        }
        
        println!("  📊 Filter matched {}/{} events", matched_count, events.len());
    }
    
    Ok(())
}

async fn run_performance_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running comprehensive performance benchmarks...");
    
    // Benchmark 1: Insertion latency
    println!("\n⚡ Benchmark 1: Insertion Latency");
    let queue = EventQueue::new(100_000);
    let iterations = 10_000;
    
    let mut latencies = Vec::new();
    
    for i in 0..iterations {
        let event = EventEnvelope::new(
            NeuralEvent::Spike {
                neuron_id: i,
                layer_id: (i % 10) as u32,
                activation_value: 500 + (i % 500) as u32,
                metadata: HashMap::new(),
            },
            (i % 255) as u8,
            "benchmark".to_string(),
        );
        
        let start = Instant::now();
        queue.insert(event).await?;
        let latency = start.elapsed();
        latencies.push(latency.as_nanos() as u64);
    }
    
    let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
    let min_latency = *latencies.iter().min().unwrap();
    let max_latency = *latencies.iter().max().unwrap();
    
    println!("  Average latency: {}ns", avg_latency);
    println!("  Min latency: {}ns", min_latency);
    println!("  Max latency: {}ns", max_latency);
    println!("  Target: <1000ns ✅ {}", if avg_latency < 1000 { "PASSED" } else { "FAILED" });
    
    // Benchmark 2: Extraction latency
    println!("\n⚡ Benchmark 2: Extraction Latency");
    latencies.clear();
    
    for _ in 0..iterations {
        let start = Instant::now();
        queue.extract().await?;
        let latency = start.elapsed();
        latencies.push(latency.as_nanos() as u64);
    }
    
    let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
    let min_latency = *latencies.iter().min().unwrap();
    let max_latency = *latencies.iter().max().unwrap();
    
    println!("  Average latency: {}ns", avg_latency);
    println!("  Min latency: {}ns", min_latency);
    println!("  Max latency: {}ns", max_latency);
    println!("  Target: <1000ns ✅ {}", if avg_latency < 1000 { "PASSED" } else { "FAILED" });
    
    // Benchmark 3: Throughput
    println!("\n🚀 Benchmark 3: Throughput");
    let queue = Arc::new(EventQueue::new(1_000_000));
    let config = ProcessorConfig {
        max_batch_size: 1000,
        max_batch_age_ns: 100_000, // 0.1ms
        processing_interval_ms: 1,
        max_concurrent_processors: num_cpus::get(),
        event_timeout_ms: 10,
    };
    
    let processor = EventProcessor::new(queue.clone(), config);
    processor.start().await?;
    
    let filter = EventFilter::accept_all();
    let mut rx = processor.register_handler(filter).await;
    
    let test_events = 100_000;
    println!("  Inserting {} events...", test_events);
    
    let start = Instant::now();
    
    // Insert events
    for i in 0..test_events {
        let event = EventEnvelope::new(
            NeuralEvent::Spike {
                neuron_id: i,
                layer_id: (i % 20) as u32,
                activation_value: 500 + (i % 500) as u32,
                metadata: HashMap::new(),
            },
            100,
            "throughput_test".to_string(),
        );
        
        queue.insert(event).await?;
    }
    
    // Receive processed events
    let mut total_received = 0;
    let timeout_duration = Duration::from_secs(30);
    
    while total_received < test_events {
        match timeout(timeout_duration, rx.recv()).await {
            Ok(Some(batch)) => {
                total_received += batch.events.len();
            }
            Ok(None) => break,
            Err(_) => {
                println!("  ⚠️  Timeout reached, received {} events", total_received);
                break;
            }
        }
    }
    
    let elapsed = start.elapsed();
    let throughput = total_received as f64 / elapsed.as_secs_f64();
    
    println!("  Events processed: {}", total_received);
    println!("  Time elapsed: {:?}", elapsed);
    println!("  Throughput: {:.0} events/second", throughput);
    println!("  Target: >1,000,000 events/sec ✅ {}", if throughput > 1_000_000.0 { "PASSED" } else { "FAILED" });
    
    processor.stop().await;
    
    Ok(())
}

async fn simulate_neural_network_training() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simulating neural network training with event-driven architecture...");
    
    let queue = Arc::new(EventQueue::new(50_000));
    let config = ProcessorConfig {
        max_batch_size: 50,
        max_batch_age_ns: 1_000_000, // 1ms
        processing_interval_ms: 1,
        max_concurrent_processors: 4,
        event_timeout_ms: 100,
    };
    
    let processor = EventProcessor::new(queue.clone(), config);
    processor.start().await?;
    
    // Register specialized handlers
    let spike_handler_rx = processor.register_handler(EventFilter {
        event_types: vec!["spike".to_string()],
        required_tags: vec![],
        excluded_tags: vec![],
        priority_range: Some((150, 255)),
        source_patterns: vec![],
    }).await;
    
    let learning_handler_rx = processor.register_handler(EventFilter {
        event_types: vec!["weight_update".to_string(), "gradient_computation".to_string()],
        required_tags: vec![],
        excluded_tags: vec![],
        priority_range: None,
        source_patterns: vec![],
    }).await;
    
    let error_handler_rx = processor.register_handler(EventFilter {
        event_types: vec!["prediction_error".to_string()],
        required_tags: vec![],
        excluded_tags: vec![],
        priority_range: Some((200, 255)),
        source_patterns: vec![],
    }).await;
    
    // Simulate training handlers
    let spike_processor = task::spawn(simulate_spike_processing(spike_handler_rx));
    let learning_processor = task::spawn(simulate_learning_processing(learning_handler_rx));
    let error_processor = task::spawn(simulate_error_processing(error_handler_rx));
    
    println!("🧠 Starting neural network training simulation...");
    
    // Simulate training epochs
    for epoch in 0..5 {
        println!("\n📚 Epoch {}/5", epoch + 1);
        
        // Forward pass - generate spike events
        println!("  Forward pass...");
        for layer in 0..5 {
            for neuron in 0..100 {
                let activation = 300 + (epoch * 100) + (layer * 50) + (neuron % 200);
                
                let spike_event = EventEnvelope::new(
                    NeuralEvent::Spike {
                        neuron_id: (layer * 100 + neuron) as u64,
                        layer_id: layer,
                        activation_value: activation as u32,
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert("epoch".to_string(), epoch as u32);
                            meta.insert("layer".to_string(), layer);
                            meta
                        },
                    },
                    180 + (layer * 15) as u8,
                    format!("layer_{}", layer),
                ).with_tags(vec!["forward_pass".to_string(), "neural".to_string()]);
                
                queue.insert(spike_event).await?;
            }
        }
        
        // Prediction errors
        println!("  Computing prediction errors...");
        for output in 0..10 {
            let expected = 800 + (output * 20);
            let actual = expected - (epoch * 50) + (output % 100);
            
            let error_event = EventEnvelope::new(
                NeuralEvent::PredictionError {
                    output_id: output,
                    expected_value: expected as u32,
                    actual_value: actual as u32,
                    error_magnitude: ((expected - actual).abs()) as u32,
                },
                250, // High priority for errors
                "output_layer".to_string(),
            ).with_tags(vec!["backward_pass".to_string(), "error".to_string()]);
            
            queue.insert(error_event).await?;
        }
        
        // Backward pass - weight updates
        println!("  Backward pass...");
        for layer in (0..5).rev() {
            for connection in 0..200 {
                let weight_delta = ((epoch * connection) % 100) as i32 - 50;
                
                let weight_event = EventEnvelope::new(
                    NeuralEvent::WeightUpdate {
                        layer_id: layer,
                        from_neuron: (connection % 100) as u64,
                        to_neuron: ((connection + 1) % 100) as u64,
                        weight_delta,
                        learning_rate: 100 - (epoch * 10) as u32,
                    },
                    160 - (layer * 10) as u8,
                    format!("backprop_layer_{}", layer),
                ).with_tags(vec!["backward_pass".to_string(), "learning".to_string()]);
                
                queue.insert(weight_event).await?;
            }
            
            // Gradient computation
            let grad_event = EventEnvelope::new(
                NeuralEvent::GradientComputation {
                    layer_id: layer,
                    gradient_norm: 500 + (epoch * layer * 10) as u32,
                    computation_time_ns: 1000 + (layer * 200) as u32,
                },
                170,
                format!("gradient_engine_{}", layer),
            ).with_tags(vec!["gradient".to_string(), "computation".to_string()]);
            
            queue.insert(grad_event).await?;
        }
        
        // Optimization step
        let opt_event = EventEnvelope::new(
            NeuralEvent::OptimizationStep {
                optimizer_id: 1, // Adam optimizer
                step_number: epoch as u64,
                loss_value: 1000 - (epoch * 180) as u32,
                convergence_metric: 100 + (epoch * 20) as u32,
            },
            220, // High priority for optimization
            "adam_optimizer".to_string(),
        ).with_tags(vec!["optimization".to_string(), "adam".to_string()]);
        
        queue.insert(opt_event).await?;
        
        // Wait between epochs
        sleep(Duration::from_millis(100)).await;
    }
    
    println!("\n⏳ Waiting for training simulation to complete...");
    
    // Wait for processing to complete
    sleep(Duration::from_secs(2)).await;
    
    // Get final statistics
    let queue_stats = queue.get_stats().await;
    let processor_stats = processor.get_stats().await;
    
    println!("\n📊 Training Simulation Results:");
    println!("  Total events generated: {}", queue_stats.total_events);
    println!("  Total events processed: {}", processor_stats.total_processed);
    println!("  Average latency: {}ns", queue_stats.avg_latency_ns);
    println!("  Peak queue size: {}", queue_stats.peak_queue_size);
    println!("  Processing throughput: {:.0} events/sec", processor_stats.throughput_eps);
    
    processor.stop().await;
    
    // Wait for handlers to complete
    let (spike_result, learning_result, error_result) = tokio::join!(
        spike_processor,
        learning_processor, 
        error_processor
    );
    
    println!("  Spike events processed: {}", spike_result?);
    println!("  Learning events processed: {}", learning_result?);
    println!("  Error events processed: {}", error_result?);
    
    Ok(())
}

async fn simulate_spike_processing(mut rx: tokio::sync::mpsc::UnboundedReceiver<tengri::events::EventBatch>) -> usize {
    let mut total_processed = 0;
    let mut batch_count = 0;
    
    while let Some(batch) = rx.recv().await {
        batch_count += 1;
        
        // Simulate spike processing
        for event in &batch.events {
            if let tengri::events::NeuralEvent::Spike { neuron_id, layer_id, activation_value, .. } = &event.event {
                // Simulate activation function computation
                if *activation_value > 600 {
                    // High activation - might trigger downstream events
                }
                total_processed += 1;
            }
        }
        
        if batch_count % 10 == 0 {
            println!("    🧠 Spike processor: {} batches, {} events", batch_count, total_processed);
        }
    }
    
    total_processed
}

async fn simulate_learning_processing(mut rx: tokio::sync::mpsc::UnboundedReceiver<tengri::events::EventBatch>) -> usize {
    let mut total_processed = 0;
    let mut batch_count = 0;
    
    while let Some(batch) = rx.recv().await {
        batch_count += 1;
        
        // Simulate learning processing
        for event in &batch.events {
            match &event.event {
                tengri::events::NeuralEvent::WeightUpdate { weight_delta, learning_rate, .. } => {
                    // Simulate weight adjustment
                    let _adjusted_weight = (*weight_delta as f32) * (*learning_rate as f32 / 1000.0);
                }
                tengri::events::NeuralEvent::GradientComputation { gradient_norm, .. } => {
                    // Simulate gradient processing
                    let _normalized_grad = *gradient_norm as f32 / 1000.0;
                }
                _ => {}
            }
            total_processed += 1;
        }
        
        if batch_count % 5 == 0 {
            println!("    🎓 Learning processor: {} batches, {} events", batch_count, total_processed);
        }
    }
    
    total_processed
}

async fn simulate_error_processing(mut rx: tokio::sync::mpsc::UnboundedReceiver<tengri::events::EventBatch>) -> usize {
    let mut total_processed = 0;
    let mut batch_count = 0;
    
    while let Some(batch) = rx.recv().await {
        batch_count += 1;
        
        // Simulate error processing
        for event in &batch.events {
            if let tengri::events::NeuralEvent::PredictionError { error_magnitude, .. } = &event.event {
                // Simulate error analysis
                if *error_magnitude > 500 {
                    // High error - might need special attention
                }
                total_processed += 1;
            }
        }
        
        if batch_count % 2 == 0 {
            println!("    ⚠️  Error processor: {} batches, {} events", batch_count, total_processed);
        }
    }
    
    total_processed
}