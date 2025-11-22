//! # Grover Search Demo
//!
//! Demonstrates the quantum Grover search algorithm for pattern detection
//! in parasitic trading systems.

use chrono::Utc;
use std::sync::Arc;
use uuid::Uuid;

use crate::quantum::{
    grover::{
        ExploitationStrategy, GroverSearchConfig, GroverSearchEngine, MarketOpportunityOracle,
        OrganismConfigOracle, ProfitablePatternOracle, TradeOutcome, TradingPattern,
    },
    QuantumConfig, QuantumMode,
};

/// Demo configuration for Grover search
pub struct GroverDemo {
    engine: GroverSearchEngine,
}

impl GroverDemo {
    /// Create a new Grover search demo
    pub fn new() -> Self {
        let grover_config = GroverSearchConfig {
            max_iterations: 100,
            match_threshold: 0.7,
            profit_threshold: 0.1,
            max_risk_score: 0.5,
            result_limit: 10,
            enable_amplitude_amplification: true,
            max_circuit_depth: 50,
        };

        let quantum_config = QuantumConfig {
            max_qubits: 10,
            max_circuit_depth: 100,
            ..QuantumConfig::default()
        };

        let engine = GroverSearchEngine::new(grover_config, quantum_config);

        Self { engine }
    }

    /// Load demo trading patterns
    pub async fn load_demo_patterns(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let patterns = vec![
            // High-profit Cordyceps pattern
            TradingPattern {
                id: Uuid::new_v4(),
                organism_type: "cordyceps".to_string(),
                feature_vector: vec![0.9, 0.8, 0.7, 0.9, 0.6],
                success_history: vec![
                    TradeOutcome {
                        profit_pct: 0.15,
                        execution_time_ms: 50,
                        market_impact: 0.001,
                        slippage: 0.002,
                        timestamp: Utc::now(),
                    },
                    TradeOutcome {
                        profit_pct: 0.12,
                        execution_time_ms: 45,
                        market_impact: 0.001,
                        slippage: 0.001,
                        timestamp: Utc::now(),
                    },
                ],
                market_conditions: vec![0.8, 0.7, 0.9, 0.6, 0.8],
                exploitation_strategy: ExploitationStrategy::Cordyceps,
                profit_score: 0.9,
                risk_score: 0.2,
                last_seen: Utc::now(),
            },
            // Medium-profit Cuckoo pattern
            TradingPattern {
                id: Uuid::new_v4(),
                organism_type: "cuckoo".to_string(),
                feature_vector: vec![0.7, 0.6, 0.8, 0.5, 0.7],
                success_history: vec![TradeOutcome {
                    profit_pct: 0.08,
                    execution_time_ms: 75,
                    market_impact: 0.002,
                    slippage: 0.003,
                    timestamp: Utc::now(),
                }],
                market_conditions: vec![0.6, 0.5, 0.7, 0.8, 0.6],
                exploitation_strategy: ExploitationStrategy::Mimic,
                profit_score: 0.7,
                risk_score: 0.3,
                last_seen: Utc::now(),
            },
            // High-risk Shadow pattern
            TradingPattern {
                id: Uuid::new_v4(),
                organism_type: "virus".to_string(),
                feature_vector: vec![0.6, 0.9, 0.5, 0.8, 0.9],
                success_history: vec![TradeOutcome {
                    profit_pct: 0.20,
                    execution_time_ms: 30,
                    market_impact: 0.005,
                    slippage: 0.008,
                    timestamp: Utc::now(),
                }],
                market_conditions: vec![0.9, 0.8, 0.6, 0.7, 0.9],
                exploitation_strategy: ExploitationStrategy::Shadow,
                profit_score: 0.8,
                risk_score: 0.7, // Too risky for most searches
                last_seen: Utc::now(),
            },
            // Low-profit safe pattern
            TradingPattern {
                id: Uuid::new_v4(),
                organism_type: "bacteria".to_string(),
                feature_vector: vec![0.4, 0.5, 0.6, 0.3, 0.5],
                success_history: vec![TradeOutcome {
                    profit_pct: 0.03,
                    execution_time_ms: 150,
                    market_impact: 0.0001,
                    slippage: 0.0005,
                    timestamp: Utc::now(),
                }],
                market_conditions: vec![0.3, 0.4, 0.5, 0.6, 0.4],
                exploitation_strategy: ExploitationStrategy::Leech,
                profit_score: 0.4,
                risk_score: 0.1,
                last_seen: Utc::now(),
            },
            // Arbitrage pattern
            TradingPattern {
                id: Uuid::new_v4(),
                organism_type: "wasp".to_string(),
                feature_vector: vec![0.8, 0.7, 0.9, 0.7, 0.8],
                success_history: vec![TradeOutcome {
                    profit_pct: 0.05,
                    execution_time_ms: 25,
                    market_impact: 0.0005,
                    slippage: 0.001,
                    timestamp: Utc::now(),
                }],
                market_conditions: vec![0.7, 0.8, 0.8, 0.9, 0.7],
                exploitation_strategy: ExploitationStrategy::Arbitrage,
                profit_score: 0.6,
                risk_score: 0.15,
                last_seen: Utc::now(),
            },
        ];

        // Load patterns into the search engine
        for pattern in patterns {
            self.engine.add_pattern(pattern).await?;
        }

        Ok(())
    }

    /// Run profitable pattern search demo
    pub async fn demo_profitable_search(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("🔍 Running Profitable Pattern Search Demo");
        println!("==========================================");

        // Create oracle for high-profit, low-risk patterns
        let oracle = Arc::new(ProfitablePatternOracle::new(
            0.5, // Minimum 50% profit score
            0.4, // Maximum 40% risk score
            vec![
                "cordyceps".to_string(),
                "cuckoo".to_string(),
                "wasp".to_string(),
            ],
        ));

        let query_conditions = vec![0.7, 0.6, 0.8, 0.7, 0.7];

        // Test different quantum modes
        for mode in [
            QuantumMode::Classical,
            QuantumMode::Enhanced,
            QuantumMode::Full,
        ] {
            QuantumMode::set_global(mode);
            println!("\n🧮 Testing with {:?} mode:", mode);

            let result = self
                .engine
                .search_patterns(oracle.clone(), &query_conditions)
                .await?;

            println!(
                "  ⏱️  Execution time: {} microseconds",
                result.execution_time_us
            );
            println!("  🎯 Algorithm used: {:?}", result.algorithm_type);
            println!("  ⚡ Quantum advantage: {:.2}x", result.quantum_advantage);
            println!(
                "  🎲 Success probability: {:.1}%",
                result.success_probability * 100.0
            );
            println!("  📋 Oracle queries: {}", result.oracle_queries);
            println!("  📊 Patterns found: {}", result.patterns.len());

            for (i, pattern) in result.patterns.iter().enumerate() {
                println!(
                    "    {}. {} ({:.1}% match, {:.1}% profit, {:.1}% risk)",
                    i + 1,
                    pattern.organism_type,
                    pattern.match_probability * 100.0,
                    pattern.profit_potential * 100.0,
                    pattern.risk_score * 100.0
                );
            }
        }

        Ok(())
    }

    /// Run market opportunity search demo
    pub async fn demo_market_opportunity_search(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("\n\n🌊 Running Market Opportunity Search Demo");
        println!("=========================================");

        // Search for patterns matching high volatility conditions
        let target_conditions = vec![0.8, 0.7, 0.9, 0.8, 0.9];
        let oracle = Arc::new(MarketOpportunityOracle::new(target_conditions.clone(), 0.7));

        QuantumMode::set_global(QuantumMode::Enhanced);

        let result = self
            .engine
            .search_patterns(oracle, &target_conditions)
            .await?;

        println!("  🎯 Target market conditions: {:?}", target_conditions);
        println!(
            "  ⏱️  Search time: {} microseconds",
            result.execution_time_us
        );
        println!("  🔄 Algorithm: {:?}", result.algorithm_type);
        println!("  📊 Opportunities found: {}", result.patterns.len());

        for (i, pattern) in result.patterns.iter().enumerate() {
            println!(
                "    {}. {} strategy using {} (similarity: {:.1}%)",
                i + 1,
                format!("{:?}", pattern.exploitation_vector),
                pattern.organism_type,
                pattern.match_probability * 100.0
            );
        }

        Ok(())
    }

    /// Run organism configuration optimization demo
    pub async fn demo_organism_config_search(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("\n\n🧬 Running Organism Configuration Search Demo");
        println!("=============================================");

        // Search for optimal Cordyceps configurations
        let oracle = Arc::new(OrganismConfigOracle::new(
            "cordyceps".to_string(),
            ExploitationStrategy::Cordyceps,
            0.6, // Minimum 60% success rate
        ));

        let query_conditions = vec![0.9, 0.8, 0.8, 0.9, 0.7];

        QuantumMode::set_global(QuantumMode::Full);

        let result = self
            .engine
            .search_patterns(oracle, &query_conditions)
            .await?;

        println!("  🎯 Target organism: Cordyceps with neural control strategy");
        println!("  📊 Configurations found: {}", result.patterns.len());
        println!("  ⚡ Quantum speedup: {:.2}x", result.quantum_advantage);

        for (i, pattern) in result.patterns.iter().enumerate() {
            println!(
                "    {}. Configuration {} (profit: {:.1}%, risk: {:.1}%)",
                i + 1,
                pattern.pattern_id,
                pattern.profit_potential * 100.0,
                pattern.risk_score * 100.0
            );
        }

        Ok(())
    }

    /// Display performance statistics
    pub async fn show_performance_stats(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("\n\n📊 Performance Statistics");
        println!("=========================");

        let stats = self.engine.get_stats().await;

        println!("  🔢 Total searches: {}", stats.total_searches);
        println!("  🏛️  Classical searches: {}", stats.classical_searches);
        println!("  🔬 Enhanced searches: {}", stats.enhanced_searches);
        println!("  ⚛️  Quantum searches: {}", stats.quantum_searches);
        println!(
            "  ⏱️  Average search time: {:.1} microseconds",
            stats.average_search_time_us
        );
        println!(
            "  ⚡ Average quantum advantage: {:.2}x",
            stats.average_quantum_advantage
        );
        println!("  🎯 Total patterns found: {}", stats.total_patterns_found);
        println!("  ✅ Success rate: {:.1}%", stats.success_rate * 100.0);

        Ok(())
    }

    /// Run complete Grover search demonstration
    pub async fn run_full_demo(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("🚀 Quantum Grover Search Algorithm Demonstration");
        println!("================================================");
        println!("Implementing O(√N) pattern detection for parasitic trading systems");
        println!("Features: Amplitude amplification, sub-millisecond search, classical fallback");

        // Load demo data
        println!("\n📥 Loading demo trading patterns...");
        self.load_demo_patterns().await?;
        println!(
            "✅ Loaded {} patterns into search database",
            self.engine.pattern_count().await
        );

        // Run different search types
        self.demo_profitable_search().await?;
        self.demo_market_opportunity_search().await?;
        self.demo_organism_config_search().await?;

        // Show performance statistics
        self.show_performance_stats().await?;

        println!("\n\n🎉 Grover Search Demo Complete!");
        println!("The quantum algorithm successfully demonstrated O(√N) pattern detection");
        println!("with sub-millisecond performance and reliable classical fallback.");

        // Reset quantum mode
        QuantumMode::set_global(QuantumMode::Classical);

        Ok(())
    }
}

impl Default for GroverDemo {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_grover_demo_creation() {
        let demo = GroverDemo::new();
        assert_eq!(demo.engine.pattern_count().await, 0);
    }

    #[tokio::test]
    async fn test_load_demo_patterns() {
        let demo = GroverDemo::new();
        demo.load_demo_patterns().await.unwrap();
        assert_eq!(demo.engine.pattern_count().await, 5);
    }

    #[tokio::test]
    async fn test_profitable_search_demo() {
        let demo = GroverDemo::new();
        demo.load_demo_patterns().await.unwrap();

        // Should not panic
        demo.demo_profitable_search().await.unwrap();

        let stats = demo.engine.get_stats().await;
        assert!(stats.total_searches > 0);
    }

    #[tokio::test]
    async fn test_full_demo_run() {
        let demo = GroverDemo::new();

        // Should complete without errors
        demo.run_full_demo().await.unwrap();

        let stats = demo.engine.get_stats().await;
        assert!(stats.total_searches >= 3); // At least 3 search types
        assert!(stats.total_patterns_found > 0);
    }

    #[tokio::test]
    async fn test_performance_tracking() {
        let demo = GroverDemo::new();
        demo.load_demo_patterns().await.unwrap();

        // Run multiple searches
        let oracle = Arc::new(ProfitablePatternOracle::new(0.1, 0.9, vec![]));
        let conditions = vec![0.5, 0.5, 0.5, 0.5, 0.5];

        for _ in 0..3 {
            demo.engine
                .search_patterns(oracle.clone(), &conditions)
                .await
                .unwrap();
        }

        let stats = demo.engine.get_stats().await;
        assert_eq!(stats.total_searches, 3);
        assert!(stats.average_search_time_us > 0.0);
    }
}
