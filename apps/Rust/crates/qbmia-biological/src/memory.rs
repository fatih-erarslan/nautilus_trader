//! QBMIA Biological Memory - Memory patterns and consolidation
//!
//! This module implements biological memory patterns including short-term, long-term,
//! and episodic memory with automatic consolidation and recall mechanisms.

use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;

use crate::{
    ComponentHealth, HealthStatus, MemoryUsage, MarketData, IntegratedDecision,
    hardware::HardwareOptimizer,
};

/// Memory entry types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    ShortTerm,
    LongTerm,
    Episodic,
    Semantic,
    Procedural,
}

/// Memory consolidation algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsolidationAlgorithm {
    Hebbian,
    SpikeTiming,
    Homeostatic,
    Competitive,
    Cooperative,
}

/// Memory entry structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub timestamp: SystemTime,
    pub memory_type: MemoryType,
    pub content: serde_json::Value,
    pub strength: f64,
    pub access_count: u64,
    pub last_accessed: SystemTime,
    pub consolidation_level: f64,
    pub associations: Vec<String>,
    pub decay_rate: f64,
    pub importance: f64,
    pub emotional_valence: f64,
    pub context_tags: Vec<String>,
}

/// Memory consolidation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationState {
    pub entry_id: String,
    pub progress: f64,
    pub algorithm: ConsolidationAlgorithm,
    pub started_at: SystemTime,
    pub last_update: SystemTime,
    pub success_rate: f64,
    pub interference_level: f64,
}

/// Memory recall result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub entries: Vec<MemoryEntry>,
    pub recall_strength: f64,
    pub context_match: f64,
    pub temporal_relevance: f64,
    pub associative_strength: f64,
    pub confidence: f64,
}

/// Memory pattern recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternResult {
    pub pattern_id: String,
    pub pattern_type: String,
    pub strength: f64,
    pub occurrences: Vec<SystemTime>,
    pub predictive_power: f64,
    pub associated_outcomes: Vec<String>,
}

/// Biological memory system
#[derive(Debug)]
pub struct BiologicalMemory {
    // Core memory stores
    short_term: Arc<RwLock<VecDeque<MemoryEntry>>>,
    long_term: Arc<RwLock<HashMap<String, MemoryEntry>>>,
    episodic: Arc<RwLock<HashMap<String, MemoryEntry>>>,
    semantic: Arc<RwLock<HashMap<String, MemoryEntry>>>,
    procedural: Arc<RwLock<HashMap<String, MemoryEntry>>>,
    
    // Memory management
    associations: Arc<RwLock<HashMap<String, Vec<String>>>>,
    consolidation_queue: Arc<RwLock<VecDeque<ConsolidationState>>>,
    access_patterns: Arc<RwLock<HashMap<String, Vec<SystemTime>>>>,
    
    // Configuration
    capacity: usize,
    consolidation_rate: f64,
    recall_threshold: f64,
    decay_rate: f64,
    
    // Hardware optimization
    hardware_optimizer: Arc<HardwareOptimizer>,
    
    // Performance tracking
    consolidation_stats: Arc<RwLock<ConsolidationStats>>,
    recall_stats: Arc<RwLock<RecallStats>>,
    
    // State management
    is_running: Arc<RwLock<bool>>,
}

/// Consolidation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationStats {
    pub total_consolidations: u64,
    pub successful_consolidations: u64,
    pub failed_consolidations: u64,
    pub average_consolidation_time: Duration,
    pub memory_efficiency: f64,
    pub interference_rate: f64,
}

/// Recall statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallStats {
    pub total_recalls: u64,
    pub successful_recalls: u64,
    pub average_recall_time: Duration,
    pub average_recall_strength: f64,
    pub context_match_rate: f64,
    pub false_positive_rate: f64,
}

impl BiologicalMemory {
    /// Create new biological memory system
    pub async fn new(
        capacity: usize,
        consolidation_rate: f64,
        recall_threshold: f64,
        hardware_optimizer: Arc<HardwareOptimizer>,
    ) -> Result<Self> {
        info!("Initializing QBMIA Biological Memory System");
        
        Ok(Self {
            short_term: Arc::new(RwLock::new(VecDeque::with_capacity(capacity / 10))),
            long_term: Arc::new(RwLock::new(HashMap::with_capacity(capacity))),
            episodic: Arc::new(RwLock::new(HashMap::with_capacity(capacity / 4))),
            semantic: Arc::new(RwLock::new(HashMap::with_capacity(capacity / 2))),
            procedural: Arc::new(RwLock::new(HashMap::with_capacity(capacity / 8))),
            
            associations: Arc::new(RwLock::new(HashMap::new())),
            consolidation_queue: Arc::new(RwLock::new(VecDeque::new())),
            access_patterns: Arc::new(RwLock::new(HashMap::new())),
            
            capacity,
            consolidation_rate,
            recall_threshold,
            decay_rate: 0.99,
            
            hardware_optimizer,
            
            consolidation_stats: Arc::new(RwLock::new(ConsolidationStats {
                total_consolidations: 0,
                successful_consolidations: 0,
                failed_consolidations: 0,
                average_consolidation_time: Duration::from_millis(0),
                memory_efficiency: 1.0,
                interference_rate: 0.0,
            })),
            
            recall_stats: Arc::new(RwLock::new(RecallStats {
                total_recalls: 0,
                successful_recalls: 0,
                average_recall_time: Duration::from_millis(0),
                average_recall_strength: 0.0,
                context_match_rate: 0.0,
                false_positive_rate: 0.0,
            })),
            
            is_running: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Start memory system
    pub async fn start(&self) -> Result<()> {
        info!("Starting QBMIA Biological Memory System");
        *self.is_running.write().await = true;
        
        // Start background consolidation process
        let consolidation_memory = Arc::clone(&self.consolidation_queue);
        let consolidation_stats = Arc::clone(&self.consolidation_stats);
        let long_term = Arc::clone(&self.long_term);
        let short_term = Arc::clone(&self.short_term);
        let is_running = Arc::clone(&self.is_running);
        let consolidation_rate = self.consolidation_rate;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            
            while *is_running.read().await {
                interval.tick().await;
                
                // Process consolidation queue
                let mut queue = consolidation_memory.write().await;
                let mut stats = consolidation_stats.write().await;
                
                while let Some(mut state) = queue.pop_front() {
                    let start_time = Instant::now();
                    
                    // Perform consolidation
                    let success = Self::consolidate_memory_entry(&state).await;
                    
                    stats.total_consolidations += 1;
                    if success {
                        stats.successful_consolidations += 1;
                    } else {
                        stats.failed_consolidations += 1;
                    }
                    
                    // Update average consolidation time
                    let duration = start_time.elapsed();
                    stats.average_consolidation_time = Duration::from_millis(
                        (stats.average_consolidation_time.as_millis() as u64 + duration.as_millis() as u64) / 2
                    );
                    
                    // Rate limiting
                    if queue.len() as f64 > consolidation_rate * 100.0 {
                        break;
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Stop memory system
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping QBMIA Biological Memory System");
        *self.is_running.write().await = false;
        Ok(())
    }
    
    /// Store experience in memory
    pub async fn store_experience(&self, market_data: &MarketData, decision: &IntegratedDecision) -> Result<String> {
        let entry_id = format!("exp_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos());
        
        let experience = serde_json::json!({
            "market_data": market_data,
            "decision": decision,
            "outcome": "pending"
        });
        
        let memory_entry = MemoryEntry {
            id: entry_id.clone(),
            timestamp: SystemTime::now(),
            memory_type: MemoryType::Episodic,
            content: experience,
            strength: decision.confidence,
            access_count: 1,
            last_accessed: SystemTime::now(),
            consolidation_level: 0.0,
            associations: Vec::new(),
            decay_rate: self.decay_rate,
            importance: self.calculate_importance(market_data, decision),
            emotional_valence: self.calculate_emotional_valence(decision),
            context_tags: self.generate_context_tags(market_data),
        };
        
        // Store in short-term memory first
        let mut short_term = self.short_term.write().await;
        short_term.push_back(memory_entry.clone());
        
        // Maintain capacity
        if short_term.len() > self.capacity / 10 {
            short_term.pop_front();
        }
        
        // Queue for consolidation if important enough
        if memory_entry.importance > self.recall_threshold {
            self.queue_for_consolidation(memory_entry).await?;
        }
        
        debug!("Stored experience in memory: {}", entry_id);
        Ok(entry_id)
    }
    
    /// Recall memories based on context
    pub async fn recall(&self, context: &str, max_results: usize) -> Result<RecallResult> {
        let start_time = Instant::now();
        
        // Update recall stats
        {
            let mut stats = self.recall_stats.write().await;
            stats.total_recalls += 1;
        }
        
        // Search across all memory stores
        let mut candidates = Vec::new();
        
        // Search episodic memory
        {
            let episodic = self.episodic.read().await;
            for entry in episodic.values() {
                let relevance = self.calculate_relevance(entry, context);
                if relevance > self.recall_threshold {
                    candidates.push((entry.clone(), relevance));
                }
            }
        }
        
        // Search semantic memory
        {
            let semantic = self.semantic.read().await;
            for entry in semantic.values() {
                let relevance = self.calculate_relevance(entry, context);
                if relevance > self.recall_threshold {
                    candidates.push((entry.clone(), relevance));
                }
            }
        }
        
        // Search long-term memory
        {
            let long_term = self.long_term.read().await;
            for entry in long_term.values() {
                let relevance = self.calculate_relevance(entry, context);
                if relevance > self.recall_threshold {
                    candidates.push((entry.clone(), relevance));
                }
            }
        }
        
        // Sort by relevance and take top results
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(max_results);
        
        let entries: Vec<MemoryEntry> = candidates.iter().map(|(entry, _)| entry.clone()).collect();
        let recall_strength = if !candidates.is_empty() {
            candidates.iter().map(|(_, relevance)| relevance).sum::<f64>() / candidates.len() as f64
        } else {
            0.0
        };
        
        // Update access patterns
        for (entry, _) in &candidates {
            self.update_access_pattern(&entry.id).await?;
        }
        
        // Calculate result metrics
        let context_match = self.calculate_context_match(&entries, context);
        let temporal_relevance = self.calculate_temporal_relevance(&entries);
        let associative_strength = self.calculate_associative_strength(&entries);
        let confidence = recall_strength * context_match * temporal_relevance;
        
        // Update recall stats
        {
            let mut stats = self.recall_stats.write().await;
            let duration = start_time.elapsed();
            stats.average_recall_time = Duration::from_millis(
                (stats.average_recall_time.as_millis() as u64 + duration.as_millis() as u64) / 2
            );
            stats.average_recall_strength = 
                (stats.average_recall_strength + recall_strength) / 2.0;
            stats.context_match_rate = 
                (stats.context_match_rate + context_match) / 2.0;
            
            if !entries.is_empty() {
                stats.successful_recalls += 1;
            }
        }
        
        Ok(RecallResult {
            entries,
            recall_strength,
            context_match,
            temporal_relevance,
            associative_strength,
            confidence,
        })
    }
    
    /// Get memory utilization metrics
    pub async fn get_utilization(&self) -> Result<f64> {
        let short_term_size = self.short_term.read().await.len();
        let long_term_size = self.long_term.read().await.len();
        let episodic_size = self.episodic.read().await.len();
        let semantic_size = self.semantic.read().await.len();
        let procedural_size = self.procedural.read().await.len();
        
        let total_used = short_term_size + long_term_size + episodic_size + semantic_size + procedural_size;
        Ok(total_used as f64 / self.capacity as f64)
    }
    
    /// Get memory usage statistics
    pub async fn get_usage_stats(&self) -> Result<MemoryUsage> {
        let short_term_size = self.short_term.read().await.len();
        let long_term_size = self.long_term.read().await.len();
        let episodic_size = self.episodic.read().await.len();
        
        let total_used = short_term_size + long_term_size + episodic_size;
        let capacity_used = total_used as f64 / self.capacity as f64;
        
        Ok(MemoryUsage {
            short_term_size,
            long_term_size,
            episodic_size,
            capacity_used,
            consolidation_rate: self.consolidation_rate,
        })
    }
    
    /// Recognize patterns in memory
    pub async fn recognize_patterns(&self, pattern_type: &str) -> Result<Vec<PatternResult>> {
        let mut patterns = Vec::new();
        
        // Analyze episodic memory for patterns
        let episodic = self.episodic.read().await;
        let entries: Vec<&MemoryEntry> = episodic.values().collect();
        
        // Time-based patterns
        if pattern_type == "temporal" || pattern_type == "all" {
            patterns.extend(self.find_temporal_patterns(&entries).await?);
        }
        
        // Content-based patterns
        if pattern_type == "content" || pattern_type == "all" {
            patterns.extend(self.find_content_patterns(&entries).await?);
        }
        
        // Associative patterns
        if pattern_type == "associative" || pattern_type == "all" {
            patterns.extend(self.find_associative_patterns(&entries).await?);
        }
        
        // Behavioral patterns
        if pattern_type == "behavioral" || pattern_type == "all" {
            patterns.extend(self.find_behavioral_patterns(&entries).await?);
        }
        
        // Sort by strength
        patterns.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
        
        Ok(patterns)
    }
    
    /// Health check
    pub async fn health_check(&self) -> Result<ComponentHealth> {
        let is_running = *self.is_running.read().await;
        let utilization = self.get_utilization().await?;
        let consolidation_stats = self.consolidation_stats.read().await;
        let recall_stats = self.recall_stats.read().await;
        
        let performance_score = if consolidation_stats.total_consolidations > 0 {
            let consolidation_success_rate = consolidation_stats.successful_consolidations as f64 / 
                consolidation_stats.total_consolidations as f64;
            let recall_success_rate = if recall_stats.total_recalls > 0 {
                recall_stats.successful_recalls as f64 / recall_stats.total_recalls as f64
            } else {
                1.0
            };
            let memory_efficiency = 1.0 - utilization.min(1.0);
            
            (consolidation_success_rate * 0.4 + recall_success_rate * 0.4 + memory_efficiency * 0.2)
        } else {
            0.8 // Default score when no operations yet
        };
        
        Ok(ComponentHealth {
            status: if is_running && performance_score > 0.7 {
                HealthStatus::Healthy
            } else if is_running && performance_score > 0.5 {
                HealthStatus::Degraded
            } else if is_running {
                HealthStatus::Critical
            } else {
                HealthStatus::Offline
            },
            last_update: std::time::SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64,
            error_count: consolidation_stats.failed_consolidations,
            performance_score,
        })
    }
    
    // Private helper methods
    
    async fn queue_for_consolidation(&self, entry: MemoryEntry) -> Result<()> {
        let consolidation_state = ConsolidationState {
            entry_id: entry.id.clone(),
            progress: 0.0,
            algorithm: ConsolidationAlgorithm::Hebbian,
            started_at: SystemTime::now(),
            last_update: SystemTime::now(),
            success_rate: 0.0,
            interference_level: 0.0,
        };
        
        let mut queue = self.consolidation_queue.write().await;
        queue.push_back(consolidation_state);
        
        Ok(())
    }
    
    async fn consolidate_memory_entry(state: &ConsolidationState) -> bool {
        // Simulate consolidation process
        // In real implementation, this would use neural network training
        // or other sophisticated consolidation algorithms
        
        // Simple success simulation based on entry importance
        let success_probability = 0.8; // 80% success rate
        rand::random::<f64>() < success_probability
    }
    
    fn calculate_importance(&self, market_data: &MarketData, decision: &IntegratedDecision) -> f64 {
        // Calculate importance based on various factors
        let volatility_factor = market_data.snapshot.volatility;
        let confidence_factor = decision.confidence;
        let risk_factor = decision.risk_assessment.risk_score;
        
        (volatility_factor * 0.3 + confidence_factor * 0.4 + risk_factor * 0.3)
            .min(1.0)
            .max(0.0)
    }
    
    fn calculate_emotional_valence(&self, decision: &IntegratedDecision) -> f64 {
        // Calculate emotional valence based on decision outcome expectations
        match decision.action {
            crate::TradingAction::Buy => 0.6,
            crate::TradingAction::Sell => -0.2,
            crate::TradingAction::Hold => 0.0,
            crate::TradingAction::Wait => -0.1,
        }
    }
    
    fn generate_context_tags(&self, market_data: &MarketData) -> Vec<String> {
        let mut tags = Vec::new();
        
        // Market regime tags
        tags.push(format!("regime_{:?}", market_data.conditions.regime));
        tags.push(format!("volatility_{:?}", market_data.conditions.volatility_state));
        tags.push(format!("liquidity_{:?}", market_data.conditions.liquidity_state));
        
        // Price level tags
        let price = market_data.snapshot.price;
        if price > 1000.0 {
            tags.push("high_price".to_string());
        } else if price < 100.0 {
            tags.push("low_price".to_string());
        } else {
            tags.push("medium_price".to_string());
        }
        
        // Volume tags
        let volume = market_data.snapshot.volume;
        if volume > 10000.0 {
            tags.push("high_volume".to_string());
        } else if volume < 1000.0 {
            tags.push("low_volume".to_string());
        } else {
            tags.push("medium_volume".to_string());
        }
        
        tags
    }
    
    fn calculate_relevance(&self, entry: &MemoryEntry, context: &str) -> f64 {
        // Calculate relevance based on context tags, content, and temporal factors
        let mut relevance = 0.0;
        
        // Context tag matching
        let context_lower = context.to_lowercase();
        for tag in &entry.context_tags {
            if context_lower.contains(&tag.to_lowercase()) {
                relevance += 0.2;
            }
        }
        
        // Content matching (simplified)
        if entry.content.to_string().to_lowercase().contains(&context_lower) {
            relevance += 0.3;
        }
        
        // Temporal decay
        if let Ok(elapsed) = entry.last_accessed.elapsed() {
            let decay_factor = (-elapsed.as_secs_f64() / 86400.0 * 0.1).exp(); // Daily decay
            relevance *= decay_factor;
        }
        
        // Strength and access count factors
        relevance *= entry.strength;
        relevance *= (1.0 + entry.access_count as f64).ln() / 10.0;
        
        relevance.min(1.0).max(0.0)
    }
    
    async fn update_access_pattern(&self, entry_id: &str) -> Result<()> {
        let mut patterns = self.access_patterns.write().await;
        patterns.entry(entry_id.to_string())
            .or_insert_with(Vec::new)
            .push(SystemTime::now());
        
        Ok(())
    }
    
    fn calculate_context_match(&self, entries: &[MemoryEntry], context: &str) -> f64 {
        if entries.is_empty() {
            return 0.0;
        }
        
        let total_match: f64 = entries.iter()
            .map(|entry| self.calculate_relevance(entry, context))
            .sum();
        
        total_match / entries.len() as f64
    }
    
    fn calculate_temporal_relevance(&self, entries: &[MemoryEntry]) -> f64 {
        if entries.is_empty() {
            return 0.0;
        }
        
        let now = SystemTime::now();
        let total_relevance: f64 = entries.iter()
            .map(|entry| {
                if let Ok(elapsed) = now.duration_since(entry.timestamp) {
                    (-elapsed.as_secs_f64() / 86400.0 * 0.1).exp()
                } else {
                    0.0
                }
            })
            .sum();
        
        total_relevance / entries.len() as f64
    }
    
    fn calculate_associative_strength(&self, entries: &[MemoryEntry]) -> f64 {
        if entries.is_empty() {
            return 0.0;
        }
        
        let total_strength: f64 = entries.iter()
            .map(|entry| entry.strength * (1.0 + entry.associations.len() as f64).ln())
            .sum();
        
        total_strength / entries.len() as f64
    }
    
    async fn find_temporal_patterns(&self, entries: &[&MemoryEntry]) -> Result<Vec<PatternResult>> {
        let mut patterns = Vec::new();
        
        // Group entries by time intervals
        let mut time_groups: BTreeMap<u64, Vec<&MemoryEntry>> = BTreeMap::new();
        
        for entry in entries {
            if let Ok(duration) = entry.timestamp.duration_since(UNIX_EPOCH) {
                let hour_bucket = duration.as_secs() / 3600; // Group by hour
                time_groups.entry(hour_bucket).or_insert_with(Vec::new).push(entry);
            }
        }
        
        // Find recurring patterns
        for (hour, group_entries) in time_groups {
            if group_entries.len() >= 3 {
                let pattern = PatternResult {
                    pattern_id: format!("temporal_{}", hour),
                    pattern_type: "temporal".to_string(),
                    strength: group_entries.len() as f64 / entries.len() as f64,
                    occurrences: group_entries.iter().map(|e| e.timestamp).collect(),
                    predictive_power: 0.7,
                    associated_outcomes: Vec::new(),
                };
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    async fn find_content_patterns(&self, entries: &[&MemoryEntry]) -> Result<Vec<PatternResult>> {
        let mut patterns = Vec::new();
        
        // Analyze content similarities
        let mut content_groups: HashMap<String, Vec<&MemoryEntry>> = HashMap::new();
        
        for entry in entries {
            // Extract key content features
            if let Some(content_str) = entry.content.as_str() {
                let words: Vec<&str> = content_str.split_whitespace().collect();
                for word in words {
                    if word.len() > 3 {
                        content_groups.entry(word.to_string())
                            .or_insert_with(Vec::new)
                            .push(entry);
                    }
                }
            }
        }
        
        // Find significant content patterns
        for (content, group_entries) in content_groups {
            if group_entries.len() >= 2 {
                let pattern = PatternResult {
                    pattern_id: format!("content_{}", content),
                    pattern_type: "content".to_string(),
                    strength: group_entries.len() as f64 / entries.len() as f64,
                    occurrences: group_entries.iter().map(|e| e.timestamp).collect(),
                    predictive_power: 0.6,
                    associated_outcomes: Vec::new(),
                };
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    async fn find_associative_patterns(&self, entries: &[&MemoryEntry]) -> Result<Vec<PatternResult>> {
        let mut patterns = Vec::new();
        
        // Analyze association networks
        let mut association_strength: HashMap<String, f64> = HashMap::new();
        
        for entry in entries {
            for association in &entry.associations {
                *association_strength.entry(association.clone()).or_insert(0.0) += entry.strength;
            }
        }
        
        // Find strong associations
        for (association, strength) in association_strength {
            if strength > 1.0 {
                let pattern = PatternResult {
                    pattern_id: format!("association_{}", association),
                    pattern_type: "associative".to_string(),
                    strength: strength / entries.len() as f64,
                    occurrences: Vec::new(),
                    predictive_power: 0.8,
                    associated_outcomes: vec![association.clone()],
                };
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    async fn find_behavioral_patterns(&self, entries: &[&MemoryEntry]) -> Result<Vec<PatternResult>> {
        let mut patterns = Vec::new();
        
        // Analyze behavioral sequences
        let mut behavior_chains: HashMap<String, u32> = HashMap::new();
        
        for entry in entries {
            for tag in &entry.context_tags {
                if tag.starts_with("regime_") || tag.starts_with("volatility_") {
                    *behavior_chains.entry(tag.clone()).or_insert(0) += 1;
                }
            }
        }
        
        // Find significant behavioral patterns
        for (behavior, count) in behavior_chains {
            if count >= 3 {
                let pattern = PatternResult {
                    pattern_id: format!("behavior_{}", behavior),
                    pattern_type: "behavioral".to_string(),
                    strength: count as f64 / entries.len() as f64,
                    occurrences: Vec::new(),
                    predictive_power: 0.75,
                    associated_outcomes: vec![behavior.clone()],
                };
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;
    
    #[tokio::test]
    async fn test_memory_initialization() {
        let hardware_optimizer = Arc::new(HardwareOptimizer::new(true, false).unwrap());
        let memory = BiologicalMemory::new(1000, 0.1, 0.7, hardware_optimizer).await;
        assert!(memory.is_ok());
    }
    
    #[tokio::test]
    async fn test_memory_start_stop() {
        let hardware_optimizer = Arc::new(HardwareOptimizer::new(true, false).unwrap());
        let memory = BiologicalMemory::new(1000, 0.1, 0.7, hardware_optimizer).await.unwrap();
        
        assert!(memory.start().await.is_ok());
        assert!(*memory.is_running.read().await);
        
        assert!(memory.stop().await.is_ok());
        assert!(!*memory.is_running.read().await);
    }
    
    #[tokio::test]
    async fn test_memory_utilization() {
        let hardware_optimizer = Arc::new(HardwareOptimizer::new(true, false).unwrap());
        let memory = BiologicalMemory::new(1000, 0.1, 0.7, hardware_optimizer).await.unwrap();
        
        let utilization = memory.get_utilization().await.unwrap();
        assert!(utilization >= 0.0 && utilization <= 1.0);
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let hardware_optimizer = Arc::new(HardwareOptimizer::new(true, false).unwrap());
        let memory = BiologicalMemory::new(1000, 0.1, 0.7, hardware_optimizer).await.unwrap();
        
        let health = memory.health_check().await.unwrap();
        assert!(matches!(health.status, HealthStatus::Offline));
        
        memory.start().await.unwrap();
        let health = memory.health_check().await.unwrap();
        assert!(matches!(health.status, HealthStatus::Healthy | HealthStatus::Degraded));
    }
}