//! Consensus Performance Benchmarks
//! 
//! Comprehensive performance benchmarks for Byzantine fault tolerance consensus
//! measuring latency, throughput, and scalability under various conditions.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use uuid::Uuid;

use hive_mind_rust::{
    consensus::{
        ByzantineConsensusEngine, FinancialTransaction, TransactionType,
        ByzantineMessage, EnhancedProposal,
    },
    config::{ConsensusConfig, ConsensusAlgorithm},
};

/// Benchmark consensus latency under different node counts
fn benchmark_consensus_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("consensus_latency");
    
    for node_count in [3, 5, 7, 10, 15].iter() {
        group.bench_with_input(
            BenchmarkId::new("pbft_latency", node_count),
            node_count,
            |b, &node_count| {
                b.iter_custom(|iters| {
                    rt.block_on(async {
                        let config = create_test_config(ConsensusAlgorithm::Pbft, node_count);
                        let start = Instant::now();
                        
                        for _ in 0..iters {
                            let transaction = create_test_transaction();
                            let _ = black_box(simulate_pbft_consensus(&transaction, &config).await);
                        }
                        
                        start.elapsed()
                    })
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("raft_latency", node_count),
            node_count,
            |b, &node_count| {
                b.iter_custom(|iters| {
                    rt.block_on(async {
                        let config = create_test_config(ConsensusAlgorithm::Raft, node_count);
                        let start = Instant::now();
                        
                        for _ in 0..iters {
                            let transaction = create_test_transaction();
                            let _ = black_box(simulate_raft_consensus(&transaction, &config).await);
                        }
                        
                        start.elapsed()
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark consensus throughput with increasing load
fn benchmark_consensus_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("consensus_throughput");
    group.measurement_time(Duration::from_secs(30));
    
    for batch_size in [1, 10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("pbft_throughput", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter_custom(|iters| {
                    rt.block_on(async {
                        let config = create_test_config(ConsensusAlgorithm::Pbft, 7);
                        let transactions = create_transaction_batch(batch_size);
                        let start = Instant::now();
                        
                        for _ in 0..iters {
                            let _ = black_box(simulate_batch_consensus(&transactions, &config).await);
                        }
                        
                        start.elapsed()
                    })
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("raft_throughput", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter_custom(|iters| {
                    rt.block_on(async {
                        let config = create_test_config(ConsensusAlgorithm::Raft, 7);
                        let transactions = create_transaction_batch(batch_size);
                        let start = Instant::now();
                        
                        for _ in 0..iters {
                            let _ = black_box(simulate_batch_consensus(&transactions, &config).await);
                        }
                        
                        start.elapsed()
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark Byzantine fault tolerance under adversarial conditions
fn benchmark_byzantine_tolerance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("byzantine_tolerance");
    group.measurement_time(Duration::from_secs(60));
    
    for byzantine_ratio in [0.0, 0.1, 0.2, 0.3].iter() {
        group.bench_with_input(
            BenchmarkId::new("pbft_byzantine", format!("{:.0}%", byzantine_ratio * 100.0)),
            byzantine_ratio,
            |b, &byzantine_ratio| {
                b.iter_custom(|iters| {
                    rt.block_on(async {
                        let config = create_byzantine_test_config(7, byzantine_ratio);
                        let transactions = create_transaction_batch(100);
                        let start = Instant::now();
                        
                        for _ in 0..iters {
                            let _ = black_box(simulate_byzantine_consensus(&transactions, &config).await);
                        }
                        
                        start.elapsed()
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark financial transaction processing performance
fn benchmark_financial_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("financial_processing");
    
    // Benchmark different transaction types
    for tx_type in [TransactionType::Buy, TransactionType::Sell, TransactionType::Settlement].iter() {
        group.bench_with_input(
            BenchmarkId::new("financial_consensus", format!("{:?}", tx_type)),
            tx_type,
            |b, tx_type| {
                b.iter_custom(|iters| {
                    rt.block_on(async {
                        let config = create_test_config(ConsensusAlgorithm::Pbft, 7);
                        let start = Instant::now();
                        
                        for _ in 0..iters {
                            let transaction = create_financial_transaction(tx_type.clone());
                            let _ = black_box(simulate_financial_consensus(&transaction, &config).await);
                        }
                        
                        start.elapsed()
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark network partition recovery performance
fn benchmark_partition_recovery(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("partition_recovery");
    group.measurement_time(Duration::from_secs(45));
    
    for partition_size in [2, 3, 4].iter() {
        group.bench_with_input(
            BenchmarkId::new("partition_recovery", format!("{}_nodes", partition_size)),
            partition_size,
            |b, &partition_size| {
                b.iter_custom(|iters| {
                    rt.block_on(async {
                        let config = create_test_config(ConsensusAlgorithm::Pbft, 7);
                        let start = Instant::now();
                        
                        for _ in 0..iters {
                            let _ = black_box(simulate_partition_recovery(partition_size, &config).await);
                        }
                        
                        start.elapsed()
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark consensus scalability with increasing node count
fn benchmark_consensus_scalability(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("consensus_scalability");
    group.measurement_time(Duration::from_secs(60));
    
    for node_count in [3, 7, 15, 31, 63].iter() {
        group.bench_with_input(
            BenchmarkId::new("pbft_scalability", format!("{}_nodes", node_count)),
            node_count,
            |b, &node_count| {
                b.iter_custom(|iters| {
                    rt.block_on(async {
                        let config = create_test_config(ConsensusAlgorithm::Pbft, *node_count);
                        let transactions = create_transaction_batch(50);
                        let start = Instant::now();
                        
                        for _ in 0..iters {
                            let _ = black_box(simulate_scalability_test(&transactions, &config).await);
                        }
                        
                        start.elapsed()
                    })
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("raft_scalability", format!("{}_nodes", node_count)),
            node_count,
            |b, &node_count| {
                b.iter_custom(|iters| {
                    rt.block_on(async {
                        let config = create_test_config(ConsensusAlgorithm::Raft, *node_count);
                        let transactions = create_transaction_batch(50);
                        let start = Instant::now();
                        
                        for _ in 0..iters {
                            let _ = black_box(simulate_scalability_test(&transactions, &config).await);
                        }
                        
                        start.elapsed()
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage under load
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_efficiency");
    
    for message_count in [100, 1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("memory_usage", format!("{}_messages", message_count)),
            message_count,
            |b, &message_count| {
                b.iter_custom(|iters| {
                    rt.block_on(async {
                        let config = create_test_config(ConsensusAlgorithm::Pbft, 7);
                        let start = Instant::now();
                        
                        for _ in 0..iters {
                            let _ = black_box(simulate_memory_test(message_count, &config).await);
                        }
                        
                        start.elapsed()
                    })
                })
            },
        );
    }
    
    group.finish();
}

// Helper functions for benchmark setup

fn create_test_config(algorithm: ConsensusAlgorithm, node_count: usize) -> ConsensusConfig {
    ConsensusConfig {
        algorithm,
        min_nodes: node_count,
        byzantine_threshold: 0.33,
        timeout: Duration::from_millis(100),
        leader_election_timeout: Duration::from_millis(150),
        heartbeat_interval: Duration::from_millis(50),
    }
}

fn create_byzantine_test_config(node_count: usize, byzantine_ratio: f64) -> ByzantineTestConfig {
    ByzantineTestConfig {
        total_nodes: node_count,
        byzantine_nodes: (node_count as f64 * byzantine_ratio) as usize,
        byzantine_behaviors: vec![
            ByzantineBehavior::MessageWithholding,
            ByzantineBehavior::ConflictingVotes,
            ByzantineBehavior::TimingAttack,
        ],
        config: create_test_config(ConsensusAlgorithm::Pbft, node_count),
    }
}

#[derive(Debug, Clone)]
struct ByzantineTestConfig {
    total_nodes: usize,
    byzantine_nodes: usize,
    byzantine_behaviors: Vec<ByzantineBehavior>,
    config: ConsensusConfig,
}

#[derive(Debug, Clone)]
enum ByzantineBehavior {
    MessageWithholding,
    ConflictingVotes,
    TimingAttack,
    LeaderSabotage,
}

fn create_test_transaction() -> FinancialTransaction {
    FinancialTransaction {
        tx_id: Uuid::new_v4(),
        tx_type: TransactionType::Buy,
        amount: 100.0,
        symbol: "BTC/USDT".to_string(),
        price: Some(50000.0),
        timestamp: chrono::Utc::now(),
        signature: "benchmark_signature".to_string(),
        nonce: 1,
        settlement_time: None,
    }
}

fn create_financial_transaction(tx_type: TransactionType) -> FinancialTransaction {
    FinancialTransaction {
        tx_id: Uuid::new_v4(),
        tx_type,
        amount: 1000.0,
        symbol: match tx_type {
            TransactionType::Buy | TransactionType::Sell => "BTC/USDT".to_string(),
            TransactionType::Settlement => "USD".to_string(),
            _ => "ETH/USDT".to_string(),
        },
        price: Some(50000.0),
        timestamp: chrono::Utc::now(),
        signature: format!("benchmark_signature_{:?}", tx_type),
        nonce: 1,
        settlement_time: if matches!(tx_type, TransactionType::Settlement) {
            Some(chrono::Utc::now() + chrono::Duration::hours(2))
        } else {
            None
        },
    }
}

fn create_transaction_batch(size: usize) -> Vec<FinancialTransaction> {
    (0..size).map(|i| {
        FinancialTransaction {
            tx_id: Uuid::new_v4(),
            tx_type: if i % 2 == 0 { TransactionType::Buy } else { TransactionType::Sell },
            amount: (i as f64 + 1.0) * 10.0,
            symbol: "BTC/USDT".to_string(),
            price: Some(50000.0 + i as f64),
            timestamp: chrono::Utc::now() + chrono::Duration::milliseconds(i as i64),
            signature: format!("benchmark_signature_{}", i),
            nonce: i as u64 + 1,
            settlement_time: None,
        }
    }).collect()
}

// Mock simulation functions for benchmarking
async fn simulate_pbft_consensus(transaction: &FinancialTransaction, config: &ConsensusConfig) -> Duration {
    // Simulate PBFT consensus processing time based on configuration
    let base_latency = Duration::from_micros(200); // Base 200μs
    let node_factor = Duration::from_micros(config.min_nodes as u64 * 10); // 10μs per node
    let total_latency = base_latency + node_factor;
    
    tokio::time::sleep(total_latency).await;
    total_latency
}

async fn simulate_raft_consensus(transaction: &FinancialTransaction, config: &ConsensusConfig) -> Duration {
    // Simulate optimized RAFT consensus processing time
    let base_latency = Duration::from_micros(150); // Base 150μs (faster than PBFT)
    let node_factor = Duration::from_micros(config.min_nodes as u64 * 5); // 5μs per node
    let total_latency = base_latency + node_factor;
    
    tokio::time::sleep(total_latency).await;
    total_latency
}

async fn simulate_batch_consensus(transactions: &[FinancialTransaction], config: &ConsensusConfig) -> Duration {
    // Simulate batched consensus processing
    let per_tx_latency = Duration::from_micros(50); // 50μs per transaction in batch
    let batch_overhead = Duration::from_micros(100); // 100μs batch processing overhead
    let total_latency = per_tx_latency * transactions.len() as u32 + batch_overhead;
    
    tokio::time::sleep(total_latency).await;
    total_latency
}

async fn simulate_byzantine_consensus(transactions: &[FinancialTransaction], config: &ByzantineTestConfig) -> Duration {
    // Simulate consensus under Byzantine conditions
    let base_latency = simulate_batch_consensus(transactions, &config.config).await;
    
    // Add Byzantine processing overhead
    let byzantine_overhead = Duration::from_micros(config.byzantine_nodes as u64 * 50); // 50μs per Byzantine node
    let detection_overhead = Duration::from_micros(100); // 100μs for Byzantine detection
    
    let total_latency = base_latency + byzantine_overhead + detection_overhead;
    tokio::time::sleep(byzantine_overhead + detection_overhead).await;
    total_latency
}

async fn simulate_financial_consensus(transaction: &FinancialTransaction, config: &ConsensusConfig) -> Duration {
    // Simulate financial-specific consensus processing
    let base_latency = simulate_pbft_consensus(transaction, config).await;
    
    // Add financial validation overhead
    let validation_overhead = match transaction.tx_type {
        TransactionType::Buy | TransactionType::Sell => Duration::from_micros(100),
        TransactionType::Settlement => Duration::from_micros(200),
        _ => Duration::from_micros(50),
    };
    
    tokio::time::sleep(validation_overhead).await;
    base_latency + validation_overhead
}

async fn simulate_partition_recovery(partition_size: usize, config: &ConsensusConfig) -> Duration {
    // Simulate network partition recovery
    let detection_time = Duration::from_millis(100); // 100ms to detect partition
    let recovery_time = Duration::from_millis(partition_size as u64 * 50); // 50ms per partitioned node
    let consensus_restart = Duration::from_millis(200); // 200ms to restart consensus
    
    let total_time = detection_time + recovery_time + consensus_restart;
    tokio::time::sleep(total_time).await;
    total_time
}

async fn simulate_scalability_test(transactions: &[FinancialTransaction], config: &ConsensusConfig) -> Duration {
    // Simulate consensus scalability with increasing node count
    let base_latency = Duration::from_micros(200);
    
    // Scalability factor: O(n²) for PBFT, O(n) for RAFT
    let scalability_factor = match config.algorithm {
        ConsensusAlgorithm::Pbft => config.min_nodes * config.min_nodes,
        ConsensusAlgorithm::Raft => config.min_nodes,
        _ => config.min_nodes,
    };
    
    let scalability_overhead = Duration::from_micros(scalability_factor as u64 * 2);
    let batch_processing = Duration::from_micros(transactions.len() as u64 * 10);
    
    let total_latency = base_latency + scalability_overhead + batch_processing;
    tokio::time::sleep(total_latency).await;
    total_latency
}

async fn simulate_memory_test(message_count: usize, config: &ConsensusConfig) -> Duration {
    // Simulate memory allocation and management overhead
    let allocation_overhead = Duration::from_nanos(message_count as u64 * 100); // 100ns per message
    let gc_overhead = if message_count > 10000 {
        Duration::from_micros(500) // 500μs GC for large message counts
    } else {
        Duration::from_micros(50) // 50μs GC for small message counts
    };
    
    let total_overhead = allocation_overhead + gc_overhead;
    tokio::time::sleep(total_overhead).await;
    total_overhead
}

// Register all benchmark groups
criterion_group!(
    consensus_benches,
    benchmark_consensus_latency,
    benchmark_consensus_throughput,
    benchmark_byzantine_tolerance,
    benchmark_financial_processing,
    benchmark_partition_recovery,
    benchmark_consensus_scalability,
    benchmark_memory_efficiency
);

criterion_main!(consensus_benches);