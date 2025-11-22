//! # Grover Search Integration Example
//!
//! This example demonstrates how to integrate the quantum Grover search algorithm
//! into a parasitic trading system for pattern detection and organism optimization.

use chrono::Utc;
use std::sync::Arc;
use uuid::Uuid;

use parasitic::quantum::{
    ExploitationStrategy, GroverDemo, GroverSearchConfig, GroverSearchEngine,
    MarketOpportunityOracle, OrganismConfigOracle, ProfitablePatternOracle, QuantumConfig,
    QuantumMode, TradeOutcome, TradingPattern,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("🧬 Parasitic Trading System - Quantum Grover Search Integration");
    println!("================================================================");

    // Initialize quantum runtime
    QuantumMode::set_global(QuantumMode::Enhanced);
    println!("🔧 Initialized quantum runtime in Enhanced mode");

    // Run the comprehensive demo
    let demo = GroverDemo::new();
    demo.run_full_demo().await?;

    // Advanced integration example
    advanced_integration_example().await?;

    Ok(())
}

/// Advanced integration example showing real-world usage
async fn advanced_integration_example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("\n\n🔬 Advanced Integration Example");
    println!("==============================");

    // Create a high-performance search engine
    let grover_config = GroverSearchConfig {
        max_iterations: 50,
        match_threshold: 0.8,
        profit_threshold: 0.15,
        max_risk_score: 0.3,
        result_limit: 5,
        enable_amplitude_amplification: true,
        max_circuit_depth: 75,
    };

    let quantum_config = QuantumConfig {
        max_qubits: 15,
        max_circuit_depth: 150,
        ..QuantumConfig::default()
    };

    let engine = GroverSearchEngine::new(grover_config, quantum_config);

    // Load realistic trading patterns
    load_realistic_patterns(&engine).await?;

    // 1. High-frequency trading pattern search
    println!("\n1️⃣  High-Frequency Trading Pattern Search");
    println!("   Target: Sub-millisecond execution patterns");

    let hft_oracle = Arc::new(ProfitablePatternOracle::new(
        0.7,                                           // High profit requirement
        0.25,                                          // Low risk tolerance
        vec!["virus".to_string(), "wasp".to_string()], // Fast organisms
    ));

    let hft_conditions = vec![0.9, 0.8, 0.9, 0.7, 0.85]; // High-speed conditions
    QuantumMode::set_global(QuantumMode::Full);

    let hft_result = engine.search_patterns(hft_oracle, &hft_conditions).await?;

    println!("   ⚡ Search time: {} μs", hft_result.execution_time_us);
    println!("   🔍 Algorithm: {:?}", hft_result.algorithm_type);
    println!(
        "   🏆 Quantum advantage: {:.2}x",
        hft_result.quantum_advantage
    );

    for pattern in &hft_result.patterns {
        println!(
            "     • {} pattern: {:.1}% profit, {:.1}% risk",
            pattern.organism_type,
            pattern.profit_potential * 100.0,
            pattern.risk_score * 100.0
        );
    }

    // 2. Market regime change detection
    println!("\n2️⃣  Market Regime Change Detection");
    println!("   Target: Patterns for volatile market conditions");

    let volatility_conditions = vec![0.95, 0.9, 0.8, 0.9, 0.95]; // High volatility
    let regime_oracle = Arc::new(MarketOpportunityOracle::new(
        volatility_conditions.clone(),
        0.75,
    ));

    let regime_result = engine
        .search_patterns(regime_oracle, &volatility_conditions)
        .await?;

    println!(
        "   📊 Found {} patterns for high volatility",
        regime_result.patterns.len()
    );
    println!(
        "   🔄 Search efficiency: {} oracle queries",
        regime_result.oracle_queries
    );

    // 3. Organism configuration optimization
    println!("\n3️⃣  Cordyceps Neural Control Optimization");
    println!("   Target: Optimal neural control configurations");

    let neural_oracle = Arc::new(OrganismConfigOracle::new(
        "cordyceps".to_string(),
        ExploitationStrategy::Cordyceps,
        0.8, // Very high success rate requirement
    ));

    let neural_conditions = vec![0.9, 0.85, 0.9, 0.8, 0.9];
    let neural_result = engine
        .search_patterns(neural_oracle, &neural_conditions)
        .await?;

    println!(
        "   🧠 Optimal configurations found: {}",
        neural_result.patterns.len()
    );
    println!(
        "   🎯 Success probability: {:.1}%",
        neural_result.success_probability * 100.0
    );

    // 4. Performance comparison across quantum modes
    println!("\n4️⃣  Performance Comparison Across Quantum Modes");
    performance_comparison(&engine).await?;

    // 5. Real-time pattern monitoring simulation
    println!("\n5️⃣  Real-Time Pattern Monitoring Simulation");
    real_time_monitoring_simulation(&engine).await?;

    // Final statistics
    println!("\n📈 Final Performance Report");
    let final_stats = engine.get_stats().await;
    println!(
        "   🔢 Total searches performed: {}",
        final_stats.total_searches
    );
    println!(
        "   ⏱️  Average search time: {:.1} μs",
        final_stats.average_search_time_us
    );
    println!(
        "   🎯 Patterns discovered: {}",
        final_stats.total_patterns_found
    );
    println!(
        "   ✅ Overall success rate: {:.1}%",
        final_stats.success_rate * 100.0
    );

    Ok(())
}

/// Load realistic trading patterns for demonstration
async fn load_realistic_patterns(
    engine: &GroverSearchEngine,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let patterns = vec![
        // Ultra-fast HFT virus pattern
        TradingPattern {
            id: Uuid::new_v4(),
            organism_type: "virus".to_string(),
            feature_vector: vec![0.95, 0.9, 0.85, 0.9, 0.95],
            success_history: generate_trading_history(&[0.12, 0.15, 0.18, 0.14], &[25, 20, 30, 22]),
            market_conditions: vec![0.9, 0.85, 0.9, 0.8, 0.95],
            exploitation_strategy: ExploitationStrategy::FrontRun,
            profit_score: 0.85,
            risk_score: 0.2,
            last_seen: Utc::now(),
        },
        // Sophisticated Cordyceps neural control
        TradingPattern {
            id: Uuid::new_v4(),
            organism_type: "cordyceps".to_string(),
            feature_vector: vec![0.88, 0.92, 0.9, 0.95, 0.87],
            success_history: generate_trading_history(
                &[0.16, 0.18, 0.14, 0.19, 0.17],
                &[40, 35, 45, 38, 42],
            ),
            market_conditions: vec![0.85, 0.9, 0.88, 0.92, 0.85],
            exploitation_strategy: ExploitationStrategy::Cordyceps,
            profit_score: 0.92,
            risk_score: 0.15,
            last_seen: Utc::now(),
        },
        // Opportunistic arbitrage wasp
        TradingPattern {
            id: Uuid::new_v4(),
            organism_type: "wasp".to_string(),
            feature_vector: vec![0.82, 0.8, 0.9, 0.75, 0.85],
            success_history: generate_trading_history(&[0.06, 0.08, 0.05, 0.07], &[15, 18, 12, 16]),
            market_conditions: vec![0.75, 0.8, 0.85, 0.9, 0.8],
            exploitation_strategy: ExploitationStrategy::Arbitrage,
            profit_score: 0.7,
            risk_score: 0.1,
            last_seen: Utc::now(),
        },
        // Stealth shadow pattern
        TradingPattern {
            id: Uuid::new_v4(),
            organism_type: "cuckoo".to_string(),
            feature_vector: vec![0.75, 0.85, 0.8, 0.8, 0.78],
            success_history: generate_trading_history(&[0.10, 0.12, 0.09, 0.11], &[60, 55, 65, 58]),
            market_conditions: vec![0.7, 0.8, 0.75, 0.85, 0.8],
            exploitation_strategy: ExploitationStrategy::Shadow,
            profit_score: 0.8,
            risk_score: 0.25,
            last_seen: Utc::now(),
        },
        // Conservative leech pattern
        TradingPattern {
            id: Uuid::new_v4(),
            organism_type: "bacteria".to_string(),
            feature_vector: vec![0.6, 0.65, 0.7, 0.6, 0.65],
            success_history: generate_trading_history(
                &[0.04, 0.05, 0.03, 0.04],
                &[120, 130, 115, 125],
            ),
            market_conditions: vec![0.5, 0.6, 0.65, 0.7, 0.6],
            exploitation_strategy: ExploitationStrategy::Leech,
            profit_score: 0.5,
            risk_score: 0.05,
            last_seen: Utc::now(),
        },
        // Adaptive mimic pattern
        TradingPattern {
            id: Uuid::new_v4(),
            organism_type: "platypus".to_string(),
            feature_vector: vec![0.78, 0.82, 0.75, 0.8, 0.79],
            success_history: generate_trading_history(&[0.09, 0.11, 0.08, 0.10], &[50, 48, 52, 51]),
            market_conditions: vec![0.75, 0.8, 0.78, 0.82, 0.77],
            exploitation_strategy: ExploitationStrategy::Mimic,
            profit_score: 0.75,
            risk_score: 0.18,
            last_seen: Utc::now(),
        },
    ];

    for pattern in patterns {
        engine.add_pattern(pattern).await?;
    }

    println!(
        "   ✅ Loaded {} realistic trading patterns",
        engine.pattern_count().await
    );
    Ok(())
}

/// Generate realistic trading history
fn generate_trading_history(profits: &[f64], times: &[u64]) -> Vec<TradeOutcome> {
    profits
        .iter()
        .zip(times.iter())
        .map(|(&profit, &time)| {
            TradeOutcome {
                profit_pct: profit,
                execution_time_ms: time,
                market_impact: profit * 0.01,
                slippage: time as f64 * 0.0001,
                timestamp: Utc::now() - chrono::Duration::hours(fastrand::i64(1..168)), // Random time in last week
            }
        })
        .collect()
}

/// Compare performance across different quantum modes
async fn performance_comparison(
    engine: &GroverSearchEngine,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let oracle = Arc::new(ProfitablePatternOracle::new(0.6, 0.4, vec![]));
    let conditions = vec![0.8, 0.7, 0.8, 0.75, 0.8];

    let modes = [
        QuantumMode::Classical,
        QuantumMode::Enhanced,
        QuantumMode::Full,
    ];
    let mut results = Vec::new();

    for &mode in &modes {
        QuantumMode::set_global(mode);

        // Run multiple searches to get average performance
        let mut total_time = 0u64;
        let mut total_advantage = 0f64;
        let runs = 3;

        for _ in 0..runs {
            let result = engine.search_patterns(oracle.clone(), &conditions).await?;
            total_time += result.execution_time_us;
            total_advantage += result.quantum_advantage;
        }

        results.push((
            mode,
            total_time / runs as u64,
            total_advantage / runs as f64,
        ));
    }

    println!("   Mode Comparison Results:");
    for (mode, avg_time, avg_advantage) in results {
        println!(
            "   • {:?}: {} μs avg, {:.2}x advantage",
            mode, avg_time, avg_advantage
        );
    }

    Ok(())
}

/// Simulate real-time pattern monitoring
async fn real_time_monitoring_simulation(
    engine: &GroverSearchEngine,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("   Simulating 10 real-time pattern searches...");

    let oracle = Arc::new(ProfitablePatternOracle::new(0.5, 0.5, vec![]));
    QuantumMode::set_global(QuantumMode::Enhanced);

    let mut total_time = 0u64;
    let mut sub_ms_count = 0;

    for i in 1..=10 {
        // Simulate changing market conditions
        let conditions = vec![
            0.5 + (i as f64 * 0.05),
            0.6 + (i as f64 * 0.04),
            0.7 + (i as f64 * 0.03),
            0.8 - (i as f64 * 0.02),
            0.9 - (i as f64 * 0.01),
        ];

        let result = engine.search_patterns(oracle.clone(), &conditions).await?;
        total_time += result.execution_time_us;

        if result.execution_time_us < 1000 {
            sub_ms_count += 1;
        }

        if i % 3 == 0 {
            println!(
                "     Search {}: {} μs, {} patterns found",
                i,
                result.execution_time_us,
                result.patterns.len()
            );
        }
    }

    println!("   📊 Real-time monitoring results:");
    println!("     Average search time: {} μs", total_time / 10);
    println!(
        "     Sub-millisecond searches: {}/10 ({:.0}%)",
        sub_ms_count,
        sub_ms_count as f64 * 10.0
    );
    println!(
        "     ✅ Real-time capability: {}",
        if sub_ms_count >= 8 {
            "EXCELLENT"
        } else {
            "GOOD"
        }
    );

    Ok(())
}
