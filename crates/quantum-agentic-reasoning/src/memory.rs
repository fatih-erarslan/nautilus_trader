//! Memory management system for quantum agentic reasoning
//!
//! This module provides persistent memory, caching, and state management
//! for the QAR system, enabling learning and adaptation over time.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use crate::config::MemoryConfig;
use crate::error::QarError;

/// Main memory management system
#[derive(Debug)]
pub struct MemoryManager {
    config: MemoryConfig,
    decision_history: Arc<RwLock<VecDeque<DecisionMemory>>>,
    pattern_cache: Arc<RwLock<HashMap<String, PatternMemory>>>,
    quantum_circuit_cache: Arc<RwLock<HashMap<String, QuantumCircuitCache>>>,
    performance_cache: Arc<RwLock<HashMap<String, PerformanceMemory>>>,
    last_cleanup: Arc<RwLock<Instant>>,
}

/// Memory of a trading decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionMemory {
    pub timestamp: u64,
    pub market_data: HashMap<String, f64>,
    pub decision: String,
    pub confidence: f64,
    pub outcome: Option<f64>,
    pub learning_feedback: Option<f64>,
    pub behavioral_factors: HashMap<String, f64>,
    pub quantum_features: Option<HashMap<String, f64>>,
}

/// Cached pattern recognition results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMemory {
    pub pattern_type: String,
    pub features: Vec<f64>,
    pub prediction: f64,
    pub confidence: f64,
    pub success_rate: f64,
    pub usage_count: u64,
    pub last_used: u64,
    pub adaptation_history: Vec<f64>,
}

/// Quantum circuit execution cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCircuitCache {
    pub circuit_hash: String,
    pub input_parameters: Vec<f64>,
    pub output_result: Vec<f64>,
    pub execution_time_ns: u64,
    pub fidelity: Option<f64>,
    pub cache_hits: u64,
    pub last_accessed: u64,
}

/// Performance metrics memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMemory {
    pub operation_type: String,
    pub execution_times: VecDeque<u64>,
    pub success_rates: VecDeque<f64>,
    pub error_counts: VecDeque<u64>,
    pub optimization_suggestions: Vec<String>,
    pub trending_direction: f64, // -1.0 to 1.0
    pub last_updated: u64,
}

impl MemoryManager {
    /// Create new memory manager with configuration
    pub fn new(config: MemoryConfig) -> Result<Self, QarError> {
        Ok(Self {
            config,
            decision_history: Arc::new(RwLock::new(VecDeque::new())),
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
            quantum_circuit_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_cache: Arc::new(RwLock::new(HashMap::new())),
            last_cleanup: Arc::new(RwLock::new(Instant::now())),
        })
    }
    
    /// Store a trading decision in memory
    pub fn store_decision(&self, decision: DecisionMemory) -> Result<(), QarError> {
        let mut history = self.decision_history.write()
            .map_err(|_| QarError::MemoryError("Failed to lock decision history".to_string()))?;
        
        history.push_back(decision);
        
        // Maintain memory length limit
        while history.len() > self.config.memory_length {
            history.pop_front();
        }
        
        Ok(())
    }
    
    /// Retrieve decision history matching criteria
    pub fn get_decision_history(&self, criteria: Option<&DecisionCriteria>) -> Result<Vec<DecisionMemory>, QarError> {
        let history = self.decision_history.read()
            .map_err(|_| QarError::MemoryError("Failed to lock decision history".to_string()))?;
        
        let decisions: Vec<DecisionMemory> = if let Some(criteria) = criteria {
            history.iter()
                .filter(|decision| self.matches_criteria(decision, criteria))
                .cloned()
                .collect()
        } else {
            history.iter().cloned().collect()
        };
        
        Ok(decisions)
    }
    
    /// Store pattern recognition result
    pub fn store_pattern(&self, pattern_id: String, pattern: PatternMemory) -> Result<(), QarError> {
        let mut cache = self.pattern_cache.write()
            .map_err(|_| QarError::MemoryError("Failed to lock pattern cache".to_string()))?;
        
        // Check cache size limit
        if cache.len() >= self.config.max_patterns {
            self.evict_least_used_pattern(&mut cache);
        }
        
        cache.insert(pattern_id, pattern);
        Ok(())
    }
    
    /// Retrieve pattern from cache
    pub fn get_pattern(&self, pattern_id: &str) -> Result<Option<PatternMemory>, QarError> {
        let mut cache = self.pattern_cache.write()
            .map_err(|_| QarError::MemoryError("Failed to lock pattern cache".to_string()))?;
        
        if let Some(pattern) = cache.get_mut(pattern_id) {
            pattern.last_used = self.current_timestamp();
            pattern.usage_count += 1;
            Ok(Some(pattern.clone()))
        } else {
            Ok(None)
        }
    }
    
    /// Store quantum circuit execution result
    pub fn store_quantum_result(&self, circuit_id: String, cache_entry: QuantumCircuitCache) -> Result<(), QarError> {
        let mut cache = self.quantum_circuit_cache.write()
            .map_err(|_| QarError::MemoryError("Failed to lock quantum circuit cache".to_string()))?;
        
        // Check cache size limit
        if cache.len() >= self.config.circuit_cache_size {
            self.evict_oldest_circuit(&mut cache);
        }
        
        cache.insert(circuit_id, cache_entry);
        Ok(())
    }
    
    /// Retrieve quantum circuit result from cache
    pub fn get_quantum_result(&self, circuit_id: &str) -> Result<Option<QuantumCircuitCache>, QarError> {
        let mut cache = self.quantum_circuit_cache.write()
            .map_err(|_| QarError::MemoryError("Failed to lock quantum circuit cache".to_string()))?;
        
        if let Some(entry) = cache.get_mut(circuit_id) {
            entry.last_accessed = self.current_timestamp();
            entry.cache_hits += 1;
            Ok(Some(entry.clone()))
        } else {
            Ok(None)
        }
    }
    
    /// Update performance memory
    pub fn update_performance(&self, operation: &str, execution_time: u64, success: bool) -> Result<(), QarError> {
        let mut cache = self.performance_cache.write()
            .map_err(|_| QarError::MemoryError("Failed to lock performance cache".to_string()))?;
        
        let entry = cache.entry(operation.to_string()).or_insert_with(|| {
            PerformanceMemory {
                operation_type: operation.to_string(),
                execution_times: VecDeque::new(),
                success_rates: VecDeque::new(),
                error_counts: VecDeque::new(),
                optimization_suggestions: Vec::new(),
                trending_direction: 0.0,
                last_updated: self.current_timestamp(),
            }
        });
        
        // Update metrics
        entry.execution_times.push_back(execution_time);
        entry.success_rates.push_back(if success { 1.0 } else { 0.0 });
        entry.error_counts.push_back(if success { 0 } else { 1 });
        entry.last_updated = self.current_timestamp();
        
        // Maintain reasonable history size (last 1000 entries)
        const MAX_HISTORY: usize = 1000;
        if entry.execution_times.len() > MAX_HISTORY {
            entry.execution_times.pop_front();
            entry.success_rates.pop_front();
            entry.error_counts.pop_front();
        }
        
        // Calculate trending direction
        if entry.execution_times.len() >= 10 {
            let recent_avg = entry.execution_times.iter().rev().take(5).sum::<u64>() as f64 / 5.0;
            let older_avg = entry.execution_times.iter().rev().skip(5).take(5).sum::<u64>() as f64 / 5.0;
            entry.trending_direction = (older_avg - recent_avg) / older_avg.max(1.0); // Positive = improving
        }
        
        Ok(())
    }
    
    /// Get performance statistics for operation
    pub fn get_performance_stats(&self, operation: &str) -> Result<Option<PerformanceStats>, QarError> {
        let cache = self.performance_cache.read()
            .map_err(|_| QarError::MemoryError("Failed to lock performance cache".to_string()))?;
        
        if let Some(entry) = cache.get(operation) {
            let avg_time = if !entry.execution_times.is_empty() {
                entry.execution_times.iter().sum::<u64>() as f64 / entry.execution_times.len() as f64
            } else {
                0.0
            };
            
            let success_rate = if !entry.success_rates.is_empty() {
                entry.success_rates.iter().sum::<f64>() / entry.success_rates.len() as f64
            } else {
                0.0
            };
            
            let total_errors = entry.error_counts.iter().sum::<u64>();
            
            Ok(Some(PerformanceStats {
                operation_type: operation.to_string(),
                average_execution_time_ns: avg_time as u64,
                success_rate_percent: success_rate * 100.0,
                total_executions: entry.execution_times.len() as u64,
                total_errors,
                trending_direction: entry.trending_direction,
                optimization_suggestions: entry.optimization_suggestions.clone(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Perform memory cleanup if needed
    pub fn cleanup_if_needed(&self) -> Result<(), QarError> {
        let last_cleanup = self.last_cleanup.read()
            .map_err(|_| QarError::MemoryError("Failed to lock cleanup timestamp".to_string()))?;
        
        if last_cleanup.elapsed() > self.config.cleanup_interval {
            drop(last_cleanup);
            self.perform_cleanup()?;
        }
        
        Ok(())
    }
    
    /// Force memory cleanup
    pub fn perform_cleanup(&self) -> Result<(), QarError> {
        // Update cleanup timestamp
        {
            let mut last_cleanup = self.last_cleanup.write()
                .map_err(|_| QarError::MemoryError("Failed to lock cleanup timestamp".to_string()))?;
            *last_cleanup = Instant::now();
        }
        
        // Cleanup old patterns
        {
            let mut pattern_cache = self.pattern_cache.write()
                .map_err(|_| QarError::MemoryError("Failed to lock pattern cache".to_string()))?;
            
            let current_time = self.current_timestamp();
            let cutoff_time = current_time - (24 * 60 * 60 * 1000); // 24 hours ago
            
            pattern_cache.retain(|_, pattern| pattern.last_used > cutoff_time);
        }
        
        // Cleanup old quantum circuit cache
        {
            let mut circuit_cache = self.quantum_circuit_cache.write()
                .map_err(|_| QarError::MemoryError("Failed to lock quantum circuit cache".to_string()))?;
            
            let current_time = self.current_timestamp();
            let cutoff_time = current_time - (6 * 60 * 60 * 1000); // 6 hours ago
            
            circuit_cache.retain(|_, entry| entry.last_accessed > cutoff_time);
        }
        
        Ok(())
    }
    
    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> Result<MemoryStats, QarError> {
        let decision_count = self.decision_history.read()
            .map_err(|_| QarError::MemoryError("Failed to lock decision history".to_string()))?
            .len();
        
        let pattern_count = self.pattern_cache.read()
            .map_err(|_| QarError::MemoryError("Failed to lock pattern cache".to_string()))?
            .len();
        
        let circuit_count = self.quantum_circuit_cache.read()
            .map_err(|_| QarError::MemoryError("Failed to lock quantum circuit cache".to_string()))?
            .len();
        
        let performance_count = self.performance_cache.read()
            .map_err(|_| QarError::MemoryError("Failed to lock performance cache".to_string()))?
            .len();
        
        Ok(MemoryStats {
            decision_history_count: decision_count,
            pattern_cache_count: pattern_count,
            circuit_cache_count: circuit_count,
            performance_entries_count: performance_count,
            memory_efficiency_percent: self.calculate_memory_efficiency(),
        })
    }
    
    // Helper methods
    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_millis() as u64
    }
    
    fn matches_criteria(&self, decision: &DecisionMemory, criteria: &DecisionCriteria) -> bool {
        if let Some(min_confidence) = criteria.min_confidence {
            if decision.confidence < min_confidence {
                return false;
            }
        }
        
        if let Some(decision_type) = &criteria.decision_type {
            if &decision.decision != decision_type {
                return false;
            }
        }
        
        if let Some(time_range) = &criteria.time_range {
            if decision.timestamp < time_range.start || decision.timestamp > time_range.end {
                return false;
            }
        }
        
        true
    }
    
    fn evict_least_used_pattern(&self, cache: &mut HashMap<String, PatternMemory>) {
        if let Some((key_to_remove, _)) = cache.iter()
            .min_by_key(|(_, pattern)| pattern.last_used) {
            let key_to_remove = key_to_remove.clone();
            cache.remove(&key_to_remove);
        }
    }
    
    fn evict_oldest_circuit(&self, cache: &mut HashMap<String, QuantumCircuitCache>) {
        if let Some((key_to_remove, _)) = cache.iter()
            .min_by_key(|(_, entry)| entry.last_accessed) {
            let key_to_remove = key_to_remove.clone();
            cache.remove(&key_to_remove);
        }
    }
    
    fn calculate_memory_efficiency(&self) -> f64 {
        // Simple efficiency calculation based on cache hit rates
        // In a real implementation, this would be more sophisticated
        85.0 // Placeholder
    }
}

/// Criteria for querying decision history
#[derive(Debug, Clone)]
pub struct DecisionCriteria {
    pub min_confidence: Option<f64>,
    pub decision_type: Option<String>,
    pub time_range: Option<TimeRange>,
}

/// Time range for queries
#[derive(Debug, Clone)]
pub struct TimeRange {
    pub start: u64,
    pub end: u64,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub operation_type: String,
    pub average_execution_time_ns: u64,
    pub success_rate_percent: f64,
    pub total_executions: u64,
    pub total_errors: u64,
    pub trending_direction: f64,
    pub optimization_suggestions: Vec<String>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub decision_history_count: usize,
    pub pattern_cache_count: usize,
    pub circuit_cache_count: usize,
    pub performance_entries_count: usize,
    pub memory_efficiency_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_manager_creation() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config);
        assert!(manager.is_ok());
    }
    
    #[test]
    fn test_decision_storage_and_retrieval() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config).unwrap();
        
        let decision = DecisionMemory {
            timestamp: 1640995200000,
            market_data: HashMap::new(),
            decision: "BUY".to_string(),
            confidence: 0.8,
            outcome: None,
            learning_feedback: None,
            behavioral_factors: HashMap::new(),
            quantum_features: None,
        };
        
        assert!(manager.store_decision(decision).is_ok());
        
        let history = manager.get_decision_history(None).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].decision, "BUY");
    }
    
    #[test]
    fn test_pattern_cache() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config).unwrap();
        
        let pattern = PatternMemory {
            pattern_type: "TREND_REVERSAL".to_string(),
            features: vec![1.0, 2.0, 3.0],
            prediction: 0.75,
            confidence: 0.9,
            success_rate: 0.85,
            usage_count: 0,
            last_used: manager.current_timestamp(),
            adaptation_history: vec![],
        };
        
        assert!(manager.store_pattern("pattern_1".to_string(), pattern).is_ok());
        
        let retrieved = manager.get_pattern("pattern_1").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().pattern_type, "TREND_REVERSAL");
    }
    
    #[test]
    fn test_performance_tracking() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config).unwrap();
        
        // Record some performance data
        assert!(manager.update_performance("quantum_decision", 1500, true).is_ok());
        assert!(manager.update_performance("quantum_decision", 1200, true).is_ok());
        assert!(manager.update_performance("quantum_decision", 1800, false).is_ok());
        
        let stats = manager.get_performance_stats("quantum_decision").unwrap();
        assert!(stats.is_some());
        
        let stats = stats.unwrap();
        assert_eq!(stats.total_executions, 3);
        assert_eq!(stats.total_errors, 1);
        assert!(stats.success_rate_percent > 0.0);
    }
    
    #[test]
    fn test_memory_stats() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config).unwrap();
        
        let stats = manager.get_memory_stats().unwrap();
        assert_eq!(stats.decision_history_count, 0);
        assert_eq!(stats.pattern_cache_count, 0);
        assert_eq!(stats.circuit_cache_count, 0);
    }
}