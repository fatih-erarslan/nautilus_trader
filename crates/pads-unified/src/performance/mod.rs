//! Performance Metrics
//! 
//! Tracks system performance and decision quality

use std::time::Duration;

pub struct PerformanceMetrics {
    decision_latencies: Vec<Duration>,
    decisions: Vec<crate::types::TradingDecision>,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            decision_latencies: Vec::new(),
            decisions: Vec::new(),
        }
    }
    
    pub fn record_decision_latency(&mut self, latency: Duration) {
        self.decision_latencies.push(latency);
        
        // Keep only last 1000 measurements
        if self.decision_latencies.len() > 1000 {
            self.decision_latencies.remove(0);
        }
    }
    
    pub fn record_decision_outcome(&mut self, decision: &crate::types::TradingDecision) {
        self.decisions.push(decision.clone());
        
        // Keep only last 1000 decisions
        if self.decisions.len() > 1000 {
            self.decisions.remove(0);
        }
    }
    
    pub fn record_feedback(
        &mut self,
        _decision: &crate::types::TradingDecision,
        _outcome: bool,
        _metrics: Option<&std::collections::HashMap<String, f64>>
    ) {
        // Implementation placeholder
    }
    
    pub fn avg_decision_latency(&self) -> Duration {
        if self.decision_latencies.is_empty() {
            return Duration::from_millis(0);
        }
        
        let total: Duration = self.decision_latencies.iter().sum();
        total / (self.decision_latencies.len() as u32)
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}