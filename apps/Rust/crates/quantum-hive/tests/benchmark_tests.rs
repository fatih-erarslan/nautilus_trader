//! Benchmark tests using criterion for detailed performance analysis

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use quantum_hive::*;
use std::hint::black_box;

fn bench_quantum_strategy_lut(c: &mut Criterion) {
    let lut = QuantumStrategyLUT::default();
    
    c.bench_function("strategy_lut_safe_lookup", |b| {
        b.iter(|| {
            let price = black_box(0.5);
            black_box(lut.get_action_safe(price))
        })
    });
    
    c.bench_function("strategy_lut_unsafe_lookup", |b| {
        b.iter(|| {
            let index = black_box(32768u16);
            unsafe {
                black_box(lut.get_action(index))
            }
        })
    });
    
    // Benchmark different batch sizes
    let mut group = c.benchmark_group("strategy_lut_batch");
    for size in [1, 10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("safe_lookup", size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    let price = (i as f64) / (size as f64);
                    black_box(lut.get_action_safe(black_box(price)));
                }
            })
        });
    }
    group.finish();
}

fn bench_circular_buffer(c: &mut Criterion) {
    let mut buffer = CircularBuffer::<f64>::new(1000);
    
    c.bench_function("circular_buffer_push", |b| {
        b.iter(|| {
            buffer.push(black_box(42.0));
        })
    });
    
    c.bench_function("circular_buffer_latest", |b| {
        b.iter(|| {
            black_box(buffer.latest())
        })
    });
    
    // Benchmark different buffer sizes
    let mut group = c.benchmark_group("circular_buffer_sizes");
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("operations", size), size, |b, &size| {
            let mut buffer = CircularBuffer::<f64>::new(size);
            b.iter(|| {
                buffer.push(black_box(1.0));
                black_box(buffer.latest());
            })
        });
    }
    group.finish();
}

fn bench_lattice_node_operations(c: &mut Criterion) {
    let mut node = LatticeNode::new(0, [0.0, 0.0, 0.0], vec![1, 2, 3]);
    let tick = MarketTick {
        symbol: [b'B', b'T', b'C', b'U', b'S', b'D', b'T', 0],
        price: 50000.0,
        volume: 1.0,
        timestamp: chrono::Utc::now().timestamp() as u64,
        bid: 49999.0,
        ask: 50001.0,
    };
    
    c.bench_function("node_tick_processing", |b| {
        b.iter(|| {
            node.process_tick(black_box(tick));
        })
    });
    
    c.bench_function("node_trade_execution", |b| {
        b.iter(|| {
            // Add a trade first
            {
                let mut trades = node.pending_trades.lock();
                trades.push(TradeAction {
                    action_type: ActionType::Buy,
                    quantity: 1.0,
                    confidence: 0.8,
                    risk_factor: 0.2,
                });
            }
            node.execute_pending_trades();
        })
    });
}

fn bench_hyperbolic_lattice_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperbolic_lattice");
    
    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("creation", size), size, |b, &size| {
            b.iter(|| {
                let nodes = AutopoieticHive::create_hyperbolic_lattice(black_box(size));
                black_box(nodes);
            })
        });
    }
    group.finish();
}

fn bench_swarm_intelligence(c: &mut Criterion) {
    let hive = AutopoieticHive::with_config(HiveConfig {
        node_count: 100,
        checkpoint_interval: std::time::Duration::from_secs(60),
        quantum_job_batch_size: 32,
        enable_gpu: false,
    });
    
    c.bench_function("swarm_update", |b| {
        b.iter(|| {
            // Create a mutable copy for the benchmark
            let mut swarm = SwarmIntelligence::new();
            swarm.update(black_box(&hive.nodes));
        })
    });
    
    c.bench_function("detect_successful_clusters", |b| {
        b.iter(|| {
            let clusters = hive.swarm_intelligence.detect_successful_clusters(black_box(&hive.nodes));
            black_box(clusters);
        })
    });
}

fn bench_quantum_queen_operations(c: &mut Criterion) {
    let hive = AutopoieticHive::new();
    let tick = MarketTick {
        symbol: [b'E', b'T', b'H', b'U', b'S', b'D', b'T', 0],
        price: 3000.0,
        volume: 2.0,
        timestamp: chrono::Utc::now().timestamp() as u64,
        bid: 2999.0,
        ask: 3001.0,
    };
    
    c.bench_function("quantum_queen_decision", |b| {
        b.iter(|| {
            let decision = hive.queen.emergency_decision_sync(black_box(&tick));
            black_box(decision);
        })
    });
}

fn bench_memory_operations(c: &mut Criterion) {
    c.bench_function("quantum_state_creation", |b| {
        b.iter(|| {
            let state = QuantumState {
                amplitude: black_box([0.7071, 0.7071]),
                phase: black_box(std::f64::consts::PI / 4.0),
                entanglement_strength: black_box(0.5),
            };
            black_box(state);
        })
    });
    
    c.bench_function("trade_action_creation", |b| {
        b.iter(|| {
            let action = TradeAction {
                action_type: black_box(ActionType::Buy),
                quantity: black_box(1.5),
                confidence: black_box(0.85),
                risk_factor: black_box(0.15),
            };
            black_box(action);
        })
    });
}

criterion_group!(
    benches,
    bench_quantum_strategy_lut,
    bench_circular_buffer,
    bench_lattice_node_operations,
    bench_hyperbolic_lattice_creation,
    bench_swarm_intelligence,
    bench_quantum_queen_operations,
    bench_memory_operations
);

criterion_main!(benches);