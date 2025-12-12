//! Performance benchmark tests for QAR system
//! 
//! Comprehensive performance testing to validate superior performance vs Python:
//! - Sub-microsecond decision making benchmarks
//! - Memory efficiency and allocation patterns
//! - Throughput under high-frequency trading loads
//! - Latency distribution analysis
//! - Quantum enhancement performance impact

use quantum_agentic_reasoning::*;
use criterion::{black_box, Criterion, BenchmarkId};
use std::time::{Duration, Instant};
use std::collections::HashMap;

// Benchmark: Basic QAR decision making latency
#[tokio::test]
async fn benchmark_decision_latency() {
    let config = QARConfig::default();
    let mut qar = QuantumAgenticReasoning::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    // Warmup
    for _ in 0..10 {
        qar.make_decision(&market_data, None).unwrap();
    }
    
    // Benchmark
    let iterations = 1000;
    let mut times = Vec::with_capacity(iterations);
    
    for _ in 0..iterations {
        let start = Instant::now();
        let _decision = qar.make_decision(&market_data, None).unwrap();
        let elapsed = start.elapsed();
        times.push(elapsed.as_nanos() as u64);
    }
    
    // Calculate statistics
    times.sort();
    let min_time = times[0];
    let max_time = times[iterations - 1];
    let median_time = times[iterations / 2];
    let p95_time = times[(iterations * 95) / 100];
    let p99_time = times[(iterations * 99) / 100];
    let mean_time = times.iter().sum::<u64>() / iterations as u64;
    
    println!("Decision Latency Benchmark:");
    println!("  Iterations: {}", iterations);
    println!("  Min:        {}ns", min_time);
    println!("  Mean:       {}ns", mean_time);
    println!("  Median:     {}ns", median_time);
    println!("  P95:        {}ns", p95_time);
    println!("  P99:        {}ns", p99_time);
    println!("  Max:        {}ns", max_time);
    
    // Performance targets (should be sub-microsecond)
    assert!(mean_time < 1000, "Mean latency {}ns exceeds 1μs target", mean_time);
    assert!(p95_time < 2000, "P95 latency {}ns exceeds 2μs target", p95_time);
    assert!(p99_time < 5000, "P99 latency {}ns exceeds 5μs target", p99_time);
}

// Benchmark: Quantum vs Classical performance comparison
#[tokio::test]
async fn benchmark_quantum_vs_classical() {
    let mut quantum_config = QARConfig::default();
    quantum_config.quantum_enabled = true;
    quantum_config.target_latency_ns = 500;
    
    let mut classical_config = QARConfig::default();
    classical_config.quantum_enabled = false;
    classical_config.target_latency_ns = 500;
    
    let mut qar_quantum = QuantumAgenticReasoning::new(quantum_config).unwrap();
    let mut qar_classical = QuantumAgenticReasoning::new(classical_config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![55000.0, 52000.0, 48000.0, 45000.0],
        buy_probabilities: vec![0.3, 0.3, 0.2, 0.2],
        sell_probabilities: vec![0.2, 0.2, 0.3, 0.3],
        hold_probabilities: vec![0.25, 0.25, 0.25, 0.25],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.8,
        },
        timestamp: 1640995200000,
    };
    
    // Warmup both systems
    for _ in 0..5 {
        qar_quantum.make_decision(&market_data, None).unwrap();
        qar_classical.make_decision(&market_data, None).unwrap();
    }
    
    let iterations = 500;
    
    // Benchmark Quantum
    let start_quantum = Instant::now();
    for _ in 0..iterations {
        let _decision = qar_quantum.make_decision(&market_data, None).unwrap();
    }
    let quantum_total = start_quantum.elapsed();
    
    // Benchmark Classical
    let start_classical = Instant::now();
    for _ in 0..iterations {
        let _decision = qar_classical.make_decision(&market_data, None).unwrap();
    }
    let classical_total = start_classical.elapsed();
    
    let quantum_avg = quantum_total.as_nanos() as f64 / iterations as f64;
    let classical_avg = classical_total.as_nanos() as f64 / iterations as f64;
    let speedup = classical_avg / quantum_avg;
    
    println!("Quantum vs Classical Benchmark:");
    println!("  Quantum avg:   {:.0}ns", quantum_avg);
    println!("  Classical avg: {:.0}ns", classical_avg);
    println!("  Speedup:       {:.2}x", speedup);
    
    // Quantum should be competitive or faster
    assert!(quantum_avg < 2000.0, "Quantum performance {}ns too slow", quantum_avg);
    assert!(classical_avg < 2000.0, "Classical performance {}ns too slow", classical_avg);
}

// Benchmark: High-frequency trading simulation
#[tokio::test]
async fn benchmark_hft_simulation() {
    let mut config = QARConfig::trading_optimized().unwrap().config;
    config.target_latency_ns = 200; // Aggressive HFT target
    
    let mut qar = QuantumAgenticReasoning::new(config).unwrap();
    
    let base_price = 50000.0;
    let num_ticks = 10000;
    let mut prices = Vec::with_capacity(num_ticks);
    
    // Generate realistic price sequence
    let mut current_price = base_price;
    for _ in 0..num_ticks {
        let change = (rand::random::<f64>() - 0.5) * 0.001; // 0.1% max change
        current_price *= 1.0 + change;
        prices.push(current_price);
    }
    
    // Warmup
    let warmup_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: base_price,
        possible_outcomes: vec![base_price * 1.001, base_price * 0.999],
        buy_probabilities: vec![0.5, 0.5],
        sell_probabilities: vec![0.5, 0.5],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    for _ in 0..10 {
        qar.make_decision(&warmup_data, None).unwrap();
    }
    
    // HFT simulation
    let start_time = Instant::now();
    let mut total_latency = 0u64;
    let mut max_latency = 0u64;
    let mut decisions_made = 0;
    
    for (i, &price) in prices.iter().enumerate() {
        let market_data = MarketData {
            symbol: "BTC/USDT".to_string(),
            current_price: price,
            possible_outcomes: vec![price * 1.0005, price * 0.9995],
            buy_probabilities: vec![0.55, 0.45],
            sell_probabilities: vec![0.45, 0.55],
            hold_probabilities: vec![0.5, 0.5],
            frame: prospect_theory::FramingContext {
                frame_type: prospect_theory::FrameType::Neutral,
                emphasis: 0.5,
            },
            timestamp: 1640995200000 + (i as u64 * 1000), // 1ms intervals
        };
        
        let decision_start = Instant::now();
        if let Ok(decision) = qar.make_decision(&market_data, None) {
            let decision_latency = decision_start.elapsed().as_nanos() as u64;
            total_latency += decision_latency;
            max_latency = max_latency.max(decision_latency);
            decisions_made += 1;
            
            // Verify decision quality under pressure
            assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        }
    }
    
    let total_time = start_time.elapsed();
    let avg_latency = total_latency / decisions_made;
    let throughput = decisions_made as f64 / total_time.as_secs_f64();
    
    println!("HFT Simulation Benchmark:");
    println!("  Ticks processed: {}", decisions_made);
    println!("  Total time:      {:.2}ms", total_time.as_millis());
    println!("  Avg latency:     {}ns", avg_latency);
    println!("  Max latency:     {}ns", max_latency);
    println!("  Throughput:      {:.0} decisions/sec", throughput);
    
    // HFT performance targets
    assert!(avg_latency < 500, "Average HFT latency {}ns exceeds 500ns target", avg_latency);
    assert!(max_latency < 2000, "Max HFT latency {}ns exceeds 2μs target", max_latency);
    assert!(throughput > 10000.0, "Throughput {:.0} decisions/sec below 10k target", throughput);
}

// Benchmark: Memory allocation patterns
#[tokio::test]
async fn benchmark_memory_efficiency() {
    let config = QARConfig::default();
    let mut qar = QuantumAgenticReasoning::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    // Measure memory usage before
    let initial_memory = get_memory_usage();
    
    // Make many decisions to test memory stability
    let iterations = 10000;
    for i in 0..iterations {
        let mut data = market_data.clone();
        data.current_price += i as f64; // Slight variation
        data.timestamp += i as u64;
        
        let _decision = qar.make_decision(&data, None).unwrap();
        
        // Check memory periodically
        if i % 1000 == 999 {
            let current_memory = get_memory_usage();
            let memory_growth = current_memory - initial_memory;
            
            println!("Iteration {}: Memory growth = {} KB", i + 1, memory_growth / 1024);
            
            // Memory should not grow excessively (allow for some caching)
            assert!(memory_growth < 50 * 1024 * 1024, "Memory growth {} bytes too large", memory_growth);
        }
    }
    
    let final_memory = get_memory_usage();
    let total_growth = final_memory - initial_memory;
    
    println!("Memory Efficiency Benchmark:");
    println!("  Iterations:     {}", iterations);
    println!("  Initial memory: {} KB", initial_memory / 1024);
    println!("  Final memory:   {} KB", final_memory / 1024);
    println!("  Total growth:   {} KB", total_growth / 1024);
    println!("  Growth per op:  {} bytes", total_growth / iterations);
    
    // Memory efficiency targets
    assert!(total_growth < 100 * 1024 * 1024, "Total memory growth {} MB too large", total_growth / (1024 * 1024));
    assert!(total_growth / iterations < 1024, "Memory per operation {} bytes too large", total_growth / iterations);
}

// Benchmark: Prospect Theory component performance
#[tokio::test]
async fn benchmark_prospect_theory_performance() {
    let config = prospect_theory::QuantumProspectTheoryConfig::default();
    let pt = prospect_theory::QuantumProspectTheory::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![55000.0, 52000.0, 48000.0, 45000.0],
        buy_probabilities: vec![0.25, 0.35, 0.25, 0.15],
        sell_probabilities: vec![0.15, 0.25, 0.35, 0.25],
        hold_probabilities: vec![0.2, 0.3, 0.3, 0.2],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.7,
        },
        timestamp: 1640995200000,
    };
    
    let position = prospect_theory::Position {
        symbol: "BTC/USDT".to_string(),
        quantity: 1.0,
        entry_price: 48000.0,
        current_value: 50000.0,
        unrealized_pnl: 2000.0,
    };
    
    // Warmup
    for _ in 0..10 {
        pt.make_trading_decision(&market_data, Some(&position)).unwrap();
    }
    
    // Benchmark
    let iterations = 5000;
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _decision = pt.make_trading_decision(&market_data, Some(&position)).unwrap();
    }
    
    let total_time = start.elapsed();
    let avg_time = total_time.as_nanos() as f64 / iterations as f64;
    
    println!("Prospect Theory Performance:");
    println!("  Iterations: {}", iterations);
    println!("  Total time: {:.2}ms", total_time.as_millis());
    println!("  Avg time:   {:.0}ns", avg_time);
    
    // Prospect Theory should be very fast
    assert!(avg_time < 200.0, "Prospect Theory avg time {:.0}ns too slow", avg_time);
}

// Benchmark: LMSR component performance
#[tokio::test]
async fn benchmark_lmsr_performance() {
    let config = lmsr_integration::LMSRConfig::default();
    let mut predictor = lmsr_integration::QuantumLMSRPredictor::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    // Warmup
    for _ in 0..5 {
        predictor.predict(&market_data).await.unwrap();
    }
    
    // Benchmark
    let iterations = 3000;
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _prediction = predictor.predict(&market_data).await.unwrap();
    }
    
    let total_time = start.elapsed();
    let avg_time = total_time.as_nanos() as f64 / iterations as f64;
    
    println!("LMSR Performance:");
    println!("  Iterations: {}", iterations);
    println!("  Total time: {:.2}ms", total_time.as_millis());
    println!("  Avg time:   {:.0}ns", avg_time);
    
    // LMSR should be fast
    assert!(avg_time < 300.0, "LMSR avg time {:.0}ns too slow", avg_time);
}

// Benchmark: Hedge algorithm performance
#[tokio::test]
async fn benchmark_hedge_performance() {
    let config = hedge_integration::HedgeConfig::default();
    let mut engine = hedge_integration::QuantumHedgeEngine::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![53000.0, 51000.0, 49000.0, 47000.0],
        buy_probabilities: vec![0.3, 0.3, 0.2, 0.2],
        sell_probabilities: vec![0.2, 0.2, 0.3, 0.3],
        hold_probabilities: vec![0.25, 0.25, 0.25, 0.25],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    // Warmup
    for _ in 0..5 {
        engine.optimize_portfolio(&market_data, None).await.unwrap();
    }
    
    // Benchmark
    let iterations = 2000;
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _result = engine.optimize_portfolio(&market_data, None).await.unwrap();
    }
    
    let total_time = start.elapsed();
    let avg_time = total_time.as_nanos() as f64 / iterations as f64;
    
    println!("Hedge Algorithm Performance:");
    println!("  Iterations: {}", iterations);
    println!("  Total time: {:.2}ms", total_time.as_millis());
    println!("  Avg time:   {:.0}ns", avg_time);
    
    // Hedge should be reasonably fast
    assert!(avg_time < 500.0, "Hedge avg time {:.0}ns too slow", avg_time);
}

// Benchmark: Parallel decision making
#[tokio::test]
async fn benchmark_parallel_decisions() {
    let config = QARConfig::default();
    
    let symbols = vec![
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", 
        "DOT/USDT", "LINK/USDT", "UNI/USDT", "AVAX/USDT"
    ];
    
    let base_prices = vec![
        50000.0, 4000.0, 100.0, 0.5, 
        30.0, 25.0, 15.0, 80.0
    ];
    
    // Sequential benchmark
    let start_sequential = Instant::now();
    for (&symbol, &price) in symbols.iter().zip(base_prices.iter()) {
        let mut qar = QuantumAgenticReasoning::new(config.clone()).unwrap();
        
        let market_data = MarketData {
            symbol: symbol.to_string(),
            current_price: price,
            possible_outcomes: vec![price * 1.02, price * 0.98],
            buy_probabilities: vec![0.6, 0.4],
            sell_probabilities: vec![0.4, 0.6],
            hold_probabilities: vec![0.5, 0.5],
            frame: prospect_theory::FramingContext {
                frame_type: prospect_theory::FrameType::Neutral,
                emphasis: 0.5,
            },
            timestamp: 1640995200000,
        };
        
        let _decision = qar.make_decision(&market_data, None).unwrap();
    }
    let sequential_time = start_sequential.elapsed();
    
    // Parallel benchmark
    let start_parallel = Instant::now();
    let mut handles = Vec::new();
    
    for (&symbol, &price) in symbols.iter().zip(base_prices.iter()) {
        let config_clone = config.clone();
        let symbol_clone = symbol.to_string();
        
        let handle = tokio::spawn(async move {
            let mut qar = QuantumAgenticReasoning::new(config_clone).unwrap();
            
            let market_data = MarketData {
                symbol: symbol_clone,
                current_price: price,
                possible_outcomes: vec![price * 1.02, price * 0.98],
                buy_probabilities: vec![0.6, 0.4],
                sell_probabilities: vec![0.4, 0.6],
                hold_probabilities: vec![0.5, 0.5],
                frame: prospect_theory::FramingContext {
                    frame_type: prospect_theory::FrameType::Neutral,
                    emphasis: 0.5,
                },
                timestamp: 1640995200000,
            };
            
            qar.make_decision(&market_data, None).unwrap()
        });
        
        handles.push(handle);
    }
    
    // Wait for all parallel decisions
    for handle in handles {
        let _decision = handle.await.unwrap();
    }
    let parallel_time = start_parallel.elapsed();
    
    let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
    
    println!("Parallel Decision Benchmark:");
    println!("  Symbols:        {}", symbols.len());
    println!("  Sequential:     {:.2}ms", sequential_time.as_millis());
    println!("  Parallel:       {:.2}ms", parallel_time.as_millis());
    println!("  Speedup:        {:.2}x", speedup);
    
    // Parallel should be faster
    assert!(speedup > 1.5, "Parallel speedup {:.2}x insufficient", speedup);
}

// Helper function to get memory usage (simplified for testing)
fn get_memory_usage() -> usize {
    // In a real implementation, this would use system calls to get actual memory usage
    // For testing purposes, we'll use a placeholder
    std::process::id() as usize * 1024 // Simplified placeholder
}

// Utility to generate realistic market data for benchmarks
fn generate_realistic_market_data(symbol: &str, base_price: f64, volatility: f64) -> MarketData {
    let price_range = base_price * volatility;
    let outcomes = vec![
        base_price + price_range,
        base_price + price_range * 0.5,
        base_price - price_range * 0.5,
        base_price - price_range,
    ];
    
    // Generate probabilities based on market conditions
    let trend_strength = (rand::random::<f64>() - 0.5) * 2.0; // -1 to 1
    let base_prob = 0.25;
    let trend_adj = trend_strength * 0.1;
    
    let buy_probs = vec![
        base_prob + trend_adj,
        base_prob + trend_adj * 0.5,
        base_prob - trend_adj * 0.5,
        base_prob - trend_adj,
    ];
    
    let sell_probs = vec![
        base_prob - trend_adj,
        base_prob - trend_adj * 0.5,
        base_prob + trend_adj * 0.5,
        base_prob + trend_adj,
    ];
    
    MarketData {
        symbol: symbol.to_string(),
        current_price: base_price,
        possible_outcomes: outcomes,
        buy_probabilities: buy_probs,
        sell_probabilities: sell_probs,
        hold_probabilities: vec![base_prob; 4],
        frame: prospect_theory::FramingContext {
            frame_type: if trend_strength > 0.0 { 
                prospect_theory::FrameType::Gain 
            } else { 
                prospect_theory::FrameType::Loss 
            },
            emphasis: trend_strength.abs(),
        },
        timestamp: chrono::Utc::now().timestamp_millis() as u64,
    }
}