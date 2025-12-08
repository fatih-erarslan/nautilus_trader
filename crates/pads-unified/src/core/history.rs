//! Decision history management for PADS system

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::SystemTime;
use crate::types::TradingDecision;

/// Decision history manager
#[derive(Debug, Clone)]
pub struct DecisionHistory {
    /// History of decisions
    history: VecDeque<HistoryEntry>,
    /// Maximum history size
    max_size: usize,
    /// Statistics
    stats: HistoryStats,
}

/// Individual history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// The decision that was made
    pub decision: TradingDecision,
    /// Timestamp when decision was made
    pub timestamp: SystemTime,
    /// Outcome if known
    pub outcome: Option<bool>,
    /// Performance metrics
    pub metrics: Option<std::collections::HashMap<String, f64>>,
}

/// History statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryStats {
    /// Total decisions made
    pub total_decisions: usize,
    /// Success rate (if outcomes available)
    pub success_rate: Option<f64>,
    /// Average decision confidence
    pub avg_confidence: f64,
    /// Decision frequency (decisions per second)
    pub decision_frequency: f64,
}

impl Default for HistoryStats {
    fn default() -> Self {
        Self {
            total_decisions: 0,
            success_rate: None,
            avg_confidence: 0.0,
            decision_frequency: 0.0,
        }
    }
}

impl DecisionHistory {
    /// Create new decision history
    pub fn new(max_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_size),
            max_size,
            stats: HistoryStats::default(),
        }
    }
    
    /// Add a decision to history
    pub fn add_decision(&mut self, decision: TradingDecision) {
        let entry = HistoryEntry {
            decision,
            timestamp: SystemTime::now(),
            outcome: None,
            metrics: None,
        };
        
        if self.history.len() >= self.max_size {
            self.history.pop_front();
        }
        
        self.history.push_back(entry);
        self.update_stats();
    }
    
    /// Update outcome for a decision
    pub fn update_outcome(&mut self, decision_id: &str, outcome: bool, metrics: Option<std::collections::HashMap<String, f64>>) {
        for entry in self.history.iter_mut() {
            if entry.decision.id == decision_id {
                entry.outcome = Some(outcome);
                entry.metrics = metrics;
                break;
            }
        }
        self.update_stats();
    }
    
    /// Get recent decisions
    pub fn get_recent(&self, count: usize) -> Vec<&HistoryEntry> {
        self.history.iter()
            .rev()
            .take(count)
            .collect()
    }
    
    /// Get decisions within time range
    pub fn get_since(&self, since: SystemTime) -> Vec<&HistoryEntry> {
        self.history.iter()
            .filter(|entry| entry.timestamp > since)
            .collect()
    }
    
    /// Get all decisions
    pub fn get_all(&self) -> Vec<&HistoryEntry> {
        self.history.iter().collect()
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> &HistoryStats {
        &self.stats
    }
    
    /// Clear history
    pub fn clear(&mut self) {
        self.history.clear();
        self.stats = HistoryStats::default();
    }
    
    /// Update internal statistics
    fn update_stats(&mut self) {
        self.stats.total_decisions = self.history.len();
        
        if !self.history.is_empty() {
            // Calculate average confidence
            let total_confidence: f64 = self.history.iter()
                .map(|entry| entry.decision.confidence)
                .sum();
            self.stats.avg_confidence = total_confidence / self.history.len() as f64;
            
            // Calculate success rate if outcomes are available
            let outcomes: Vec<bool> = self.history.iter()
                .filter_map(|entry| entry.outcome)
                .collect();
            
            if !outcomes.is_empty() {
                let success_count = outcomes.iter().filter(|&&outcome| outcome).count();
                self.stats.success_rate = Some(success_count as f64 / outcomes.len() as f64);
            }
            
            // Calculate decision frequency
            if self.history.len() > 1 {
                let first_time = self.history.front().unwrap().timestamp;
                let last_time = self.history.back().unwrap().timestamp;
                
                if let Ok(duration) = last_time.duration_since(first_time) {
                    let duration_secs = duration.as_secs_f64();
                    if duration_secs > 0.0 {
                        self.stats.decision_frequency = (self.history.len() - 1) as f64 / duration_secs;
                    }
                }
            }
        }
    }
}

impl Default for DecisionHistory {
    fn default() -> Self {
        Self::new(10_000)
    }
}