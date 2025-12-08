//! State persistence for the quantum hive

use crate::{AutopoieticHive, quantum_queen::QuantumQueenState};
use serde::{Serialize, Deserialize};
use std::time::{Duration, Instant};
use anyhow::Result;
use tokio::fs;
use tracing::{info, error};

/// State persistence manager
pub struct StatePersistence {
    checkpoint_interval: Duration,
    last_checkpoint: Instant,
    checkpoint_path: String,
}

impl StatePersistence {
    pub fn new(checkpoint_interval: Duration) -> Self {
        Self {
            checkpoint_interval,
            last_checkpoint: Instant::now(),
            checkpoint_path: "quantum_hive_state.json".to_string(),
        }
    }
    
    /// Check if checkpoint is needed
    pub fn needs_checkpoint(&self) -> bool {
        self.last_checkpoint.elapsed() >= self.checkpoint_interval
    }
    
    /// Save hive state to JSON
    pub async fn checkpoint(&mut self, hive: &AutopoieticHive) -> Result<()> {
        info!("Creating checkpoint...");
        
        let state = HiveState {
            timestamp: chrono::Utc::now().timestamp(),
            queen_state: self.extract_queen_state(hive),
            swarm_metrics: self.extract_swarm_metrics(hive),
            performance_metrics: self.extract_performance_metrics(hive),
        };
        
        let json = serde_json::to_string_pretty(&state)?;
        fs::write(&self.checkpoint_path, json).await?;
        
        self.last_checkpoint = Instant::now();
        info!("Checkpoint saved to {}", self.checkpoint_path);
        
        Ok(())
    }
    
    /// Save snapshot to JSON (simpler version for borrowing compatibility)
    pub async fn checkpoint_snapshot(&mut self, snapshot: crate::QuantumHiveSnapshot) -> Result<()> {
        info!("Creating snapshot checkpoint...");
        
        let json = serde_json::to_string_pretty(&snapshot)?;
        let snapshot_path = format!("quantum_hive_snapshot_{}.json", snapshot.timestamp.timestamp());
        fs::write(&snapshot_path, json).await?;
        
        self.last_checkpoint = Instant::now();
        info!("Snapshot saved to {}", snapshot_path);
        
        Ok(())
    }
    
    /// Extract queen state
    fn extract_queen_state(&self, hive: &AutopoieticHive) -> QuantumQueenState {
        QuantumQueenState {
            strategy_generation: hive.queen.strategy_generation,
            market_regime: hive.queen.market_regime,
            performance_metrics: crate::quantum_queen::PerformanceMetrics {
                total_decisions: 0,
                successful_trades: 0,
                total_pnl: 0.0,
                sharpe_ratio: 0.0,
                quantum_advantage: 0.0,
            },
        }
    }
    
    /// Extract swarm metrics
    fn extract_swarm_metrics(&self, hive: &AutopoieticHive) -> SwarmMetrics {
        let total_trades: u64 = hive.nodes.iter()
            .map(|n| n.execution_stats.lock().trades_executed)
            .sum();
            
        let total_pnl: f64 = hive.nodes.iter()
            .map(|n| n.execution_stats.lock().total_pnl)
            .sum();
        
        SwarmMetrics {
            total_nodes: hive.nodes.len(),
            total_trades,
            total_pnl,
            active_pheromone_trails: hive.swarm_intelligence.pheromone_trails.len(),
        }
    }
    
    /// Extract performance metrics
    fn extract_performance_metrics(&self, hive: &AutopoieticHive) -> PerformanceMetrics {
        PerformanceMetrics {
            iterations: hive.performance_tracker.iterations,
            avg_latency_ns: hive.performance_tracker.avg_latency_ns,
        }
    }
}

/// Serializable hive state
#[derive(Debug, Serialize, Deserialize)]
struct HiveState {
    timestamp: i64,
    queen_state: QuantumQueenState,
    swarm_metrics: SwarmMetrics,
    performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
struct SwarmMetrics {
    total_nodes: usize,
    total_trades: u64,
    total_pnl: f64,
    active_pheromone_trails: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct PerformanceMetrics {
    iterations: u64,
    avg_latency_ns: u64,
}