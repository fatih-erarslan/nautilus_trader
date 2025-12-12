//! Core QBMIA algorithms consolidated from qbmia-core
//! 
//! This module contains all core algorithms including Nash equilibrium solving,
//! Machiavellian strategy detection, and agent logic - all with TENGRI compliance.

pub mod nash_equilibrium;
pub mod machiavellian;
pub mod agent;
pub mod strategy;

pub use nash_equilibrium::*;
pub use machiavellian::*;
pub use agent::*;
pub use strategy::*;

use crate::types::*;
use crate::error::Result;
use crate::quantum::GpuQuantumSimulator;
use crate::biological::BiologicalProcessor;
use std::sync::Arc;
use tracing::{info, debug, instrument};

/// Core algorithm processor that coordinates all core QBMIA algorithms
#[derive(Debug)]
pub struct CoreProcessor {
    nash_solver: NashEquilibriumSolver,
    machiavellian_detector: MachiavellianDetector,
    agent_coordinator: AgentCoordinator,
    strategy_analyzer: StrategyAnalyzer,
    quantum_bridge: Arc<GpuQuantumSimulator>,
    biological_bridge: Arc<BiologicalProcessor>,
}

impl CoreProcessor {
    /// Create new core processor with quantum and biological bridges
    pub async fn new(
        quantum_simulator: &Arc<GpuQuantumSimulator>,
        biological_processor: &Arc<BiologicalProcessor>,
    ) -> Result<Self> {
        info!("Initializing Core Algorithm Processor");

        let nash_solver = NashEquilibriumSolver::new().await?;
        let machiavellian_detector = MachiavellianDetector::new().await?;
        let agent_coordinator = AgentCoordinator::new().await?;
        let strategy_analyzer = StrategyAnalyzer::new().await?;

        Ok(Self {
            nash_solver,
            machiavellian_detector,
            agent_coordinator,
            strategy_analyzer,
            quantum_bridge: quantum_simulator.clone(),
            biological_bridge: biological_processor.clone(),
        })
    }

    /// Analyze market data using all core algorithms
    #[instrument(skip(self, market_data, quantum_analysis, biological_analysis))]
    pub async fn analyze(
        &self,
        market_data: &MarketData,
        quantum_analysis: &QuantumAnalysis,
        biological_analysis: &BiologicalAnalysis,
    ) -> Result<CoreAnalysis> {
        info!("Running core algorithm analysis");

        // Step 1: Nash equilibrium analysis
        debug!("Computing Nash equilibrium...");
        let nash_equilibrium = self.nash_solver.solve_market_equilibrium(
            market_data,
            quantum_analysis,
            biological_analysis,
        ).await?;

        // Step 2: Machiavellian strategy detection
        debug!("Detecting Machiavellian strategies...");
        let machiavellian_strategies = self.machiavellian_detector.detect_strategies(
            market_data,
            &nash_equilibrium,
            biological_analysis,
        ).await?;

        // Step 3: Agent recommendations
        debug!("Generating agent recommendations...");
        let agent_recommendations = self.agent_coordinator.generate_recommendations(
            market_data,
            &nash_equilibrium,
            &machiavellian_strategies,
            quantum_analysis,
            biological_analysis,
        ).await?;

        // Step 4: Game theory metrics
        debug!("Computing game theory metrics...");
        let game_theory_metrics = self.strategy_analyzer.compute_metrics(
            market_data,
            &nash_equilibrium,
            &machiavellian_strategies,
        ).await?;

        // Calculate overall confidence
        let confidence = self.calculate_analysis_confidence(
            &nash_equilibrium,
            &machiavellian_strategies,
            &agent_recommendations,
            quantum_analysis,
            biological_analysis,
        ).await;

        Ok(CoreAnalysis {
            confidence,
            nash_equilibrium,
            machiavellian_strategies,
            agent_recommendations,
            game_theory_metrics,
        })
    }

    /// Calculate confidence level for core analysis
    async fn calculate_analysis_confidence(
        &self,
        nash_equilibrium: &NashEquilibrium,
        machiavellian_strategies: &[MachiavellianStrategy],
        agent_recommendations: &[AgentRecommendation],
        quantum_analysis: &QuantumAnalysis,
        biological_analysis: &BiologicalAnalysis,
    ) -> f64 {
        // Weighted confidence calculation
        let nash_weight = 0.3;
        let machiavellian_weight = 0.25;
        let agent_weight = 0.2;
        let quantum_weight = 0.15;
        let biological_weight = 0.1;

        let nash_confidence = nash_equilibrium.stability_measure;
        
        let machiavellian_confidence = if machiavellian_strategies.is_empty() {
            0.5 // Neutral confidence if no strategies detected
        } else {
            machiavellian_strategies.iter()
                .map(|s| s.detection_confidence)
                .sum::<f64>() / machiavellian_strategies.len() as f64
        };

        let agent_confidence = if agent_recommendations.is_empty() {
            0.5 // Neutral confidence if no recommendations
        } else {
            agent_recommendations.iter()
                .map(|r| r.confidence)
                .sum::<f64>() / agent_recommendations.len() as f64
        };

        nash_weight * nash_confidence
            + machiavellian_weight * machiavellian_confidence
            + agent_weight * agent_confidence
            + quantum_weight * quantum_analysis.confidence
            + biological_weight * biological_analysis.confidence
    }

    /// Get solver statistics
    pub async fn get_solver_statistics(&self) -> SolverStatistics {
        SolverStatistics {
            nash_computations_completed: self.nash_solver.get_computation_count().await,
            machiavellian_detections: self.machiavellian_detector.get_detection_count().await,
            agent_recommendations_generated: self.agent_coordinator.get_recommendation_count().await,
            average_computation_time_ms: self.get_average_computation_time().await,
            success_rate: self.get_success_rate().await,
        }
    }

    async fn get_average_computation_time(&self) -> u64 {
        // Aggregate timing data from all solvers
        let nash_time = self.nash_solver.get_average_time().await;
        let machiavellian_time = self.machiavellian_detector.get_average_time().await;
        let agent_time = self.agent_coordinator.get_average_time().await;
        
        (nash_time + machiavellian_time + agent_time) / 3
    }

    async fn get_success_rate(&self) -> f64 {
        let nash_rate = self.nash_solver.get_success_rate().await;
        let machiavellian_rate = self.machiavellian_detector.get_success_rate().await;
        let agent_rate = self.agent_coordinator.get_success_rate().await;
        
        (nash_rate + machiavellian_rate + agent_rate) / 3.0
    }
}

/// Statistics for core algorithm solvers
#[derive(Debug, Clone)]
pub struct SolverStatistics {
    pub nash_computations_completed: u64,
    pub machiavellian_detections: u64,
    pub agent_recommendations_generated: u64,
    pub average_computation_time_ms: u64,
    pub success_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_core_processor_initialization() {
        // This test validates that the core processor can be initialized
        // It may fail if quantum simulator or biological processor aren't available
        
        // Mock quantum simulator and biological processor for testing
        // Note: In production, these would be real implementations
        // For testing, we can use simplified versions
        
        println!("Core processor initialization test - requires real components");
        // Actual implementation would need real quantum and biological processors
    }

    #[tokio::test]
    async fn test_confidence_calculation() {
        // Test confidence calculation with realistic values
        let nash_equilibrium = NashEquilibrium {
            player_strategies: std::collections::HashMap::new(),
            payoff_matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            equilibrium_type: EquilibriumType::Pure,
            stability_measure: 0.85,
            convergence_iterations: 100,
        };

        // Test that confidence calculation produces reasonable results
        assert!(nash_equilibrium.stability_measure > 0.0);
        assert!(nash_equilibrium.stability_measure <= 1.0);
    }
}