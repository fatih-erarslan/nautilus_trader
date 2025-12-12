//! Feedback processing for PADS system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::PadsResult;
use crate::types::TradingDecision;

/// Feedback processor for learning from outcomes
#[derive(Debug, Clone)]
pub struct FeedbackProcessor {
    /// Learning rate for feedback integration
    learning_rate: f64,
    /// Feedback history
    feedback_history: Vec<FeedbackEntry>,
    /// Performance metrics
    metrics: FeedbackMetrics,
}

/// Individual feedback entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEntry {
    /// Decision that was made
    pub decision: TradingDecision,
    /// Outcome (success/failure)
    pub outcome: bool,
    /// Additional metrics
    pub metrics: Option<HashMap<String, f64>>,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Feedback metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackMetrics {
    /// Total feedback entries
    pub total_entries: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average feedback latency
    pub avg_latency: std::time::Duration,
}

impl Default for FeedbackMetrics {
    fn default() -> Self {
        Self {
            total_entries: 0,
            success_rate: 0.0,
            avg_latency: std::time::Duration::from_nanos(0),
        }
    }
}

impl FeedbackProcessor {
    /// Create new feedback processor
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            feedback_history: Vec::new(),
            metrics: FeedbackMetrics::default(),
        }
    }
    
    /// Process feedback for a decision
    pub fn process_feedback(
        &mut self,
        decision: &TradingDecision,
        outcome: bool,
        metrics: Option<&HashMap<String, f64>>,
    ) -> PadsResult<()> {
        let entry = FeedbackEntry {
            decision: decision.clone(),
            outcome,
            metrics: metrics.cloned(),
            timestamp: std::time::SystemTime::now(),
        };
        
        self.feedback_history.push(entry);
        self.update_metrics();
        
        Ok(())
    }
    
    /// Update internal metrics
    fn update_metrics(&mut self) {
        self.metrics.total_entries = self.feedback_history.len();
        
        if !self.feedback_history.is_empty() {
            let success_count = self.feedback_history.iter()
                .filter(|entry| entry.outcome)
                .count();
            
            self.metrics.success_rate = success_count as f64 / self.feedback_history.len() as f64;
        }
    }
    
    /// Get feedback metrics
    pub fn get_metrics(&self) -> &FeedbackMetrics {
        &self.metrics
    }
    
    /// Clear feedback history
    pub fn clear_history(&mut self) {
        self.feedback_history.clear();
        self.metrics = FeedbackMetrics::default();
    }
}

impl Default for FeedbackProcessor {
    fn default() -> Self {
        Self::new(0.1)
    }
}