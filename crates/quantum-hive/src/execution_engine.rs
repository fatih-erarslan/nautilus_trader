//! Ultra-low latency execution engine

use crate::core::*;
use crate::lattice::LatticeNode;
use std::sync::Arc;
use parking_lot::RwLock;
use rayon::prelude::*;

/// Execution engine for nanosecond-latency trading
pub struct ExecutionEngine {
    /// Global strategy LUT for ultra-fast lookups
    global_strategy: Arc<RwLock<QuantumStrategyLUT>>,
    
    /// Execution statistics
    stats: ExecutionStats,
}

impl ExecutionEngine {
    pub fn new(global_strategy: Arc<RwLock<QuantumStrategyLUT>>) -> Self {
        Self {
            global_strategy,
            stats: ExecutionStats::default(),
        }
    }
    
    /// Execute trades across all nodes in parallel
    pub fn execute_batch(&mut self, nodes: &mut [LatticeNode], market_data: &[MarketTick]) {
        // Process each tick through all nodes in parallel
        for tick in market_data {
            nodes.par_iter_mut().for_each(|node| {
                node.process_tick(*tick);
            });
        }
        
        // Execute pending trades
        nodes.par_iter().for_each(|node| {
            node.execute_pending_trades_parallel();
        });
        
        // Update statistics
        self.update_statistics(nodes);
    }
    
    /// Update execution statistics
    fn update_statistics(&mut self, nodes: &[LatticeNode]) {
        for node in nodes {
            let node_stats = node.execution_stats.lock();
            self.stats.trades_executed += node_stats.trades_executed;
            self.stats.total_pnl += node_stats.total_pnl;
            self.stats.error_count += node_stats.error_count;
        }
        
        if self.stats.trades_executed > 0 {
            self.stats.success_rate = 
                (self.stats.trades_executed - self.stats.error_count) as f64 / 
                self.stats.trades_executed as f64;
        }
    }
    
    /// Get current execution statistics
    pub fn get_stats(&self) -> &ExecutionStats {
        &self.stats
    }
}