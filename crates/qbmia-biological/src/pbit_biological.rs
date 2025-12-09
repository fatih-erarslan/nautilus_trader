//! pBit-Enhanced Biological Learning
//!
//! Uses Boltzmann statistics for biologically-inspired memory consolidation
//! and learning dynamics.
//!
//! ## Mathematical Foundation
//!
//! Memory consolidation follows Boltzmann dynamics:
//! - Memory energy: E_m = -importance × recency × coherence
//! - Consolidation probability: P(consolidate) = σ(E_m / T)
//! - Recall strength: W_m = exp(-E_m / T) / Z
//!
//! Neural adaptation uses STDP-like Hebbian rules:
//! - ΔW = η × pre × post × exp(-|Δt| / τ)
//! - Weight bounded by pBit probability: W ∈ [0, 1]

use std::collections::HashMap;

/// Boltzmann constant for biological dynamics
pub const BIO_BOLTZMANN_K: f64 = 1.0;

/// pBit biological memory configuration
#[derive(Debug, Clone)]
pub struct PBitBiologicalConfig {
    /// Temperature for memory consolidation
    pub temperature: f64,
    /// Learning rate for adaptation
    pub learning_rate: f64,
    /// Memory decay time constant
    pub decay_tau: f64,
    /// Consolidation threshold
    pub consolidation_threshold: f64,
    /// STDP time constant
    pub stdp_tau: f64,
}

impl Default for PBitBiologicalConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            learning_rate: 0.01,
            decay_tau: 100.0,
            consolidation_threshold: 0.5,
            stdp_tau: 20.0,
        }
    }
}

/// Memory item with Boltzmann weight
#[derive(Debug, Clone)]
pub struct PBitMemory {
    /// Unique identifier
    pub id: String,
    /// Content embedding
    pub embedding: Vec<f64>,
    /// Importance score
    pub importance: f64,
    /// Recency (time since creation/access)
    pub recency: f64,
    /// Coherence with other memories
    pub coherence: f64,
    /// Energy (computed from above)
    pub energy: f64,
    /// Boltzmann weight
    pub weight: f64,
    /// Access count
    pub access_count: u64,
}

impl PBitMemory {
    /// Create new memory item
    pub fn new(id: String, embedding: Vec<f64>, importance: f64) -> Self {
        Self {
            id,
            embedding,
            importance,
            recency: 1.0,
            coherence: 0.5,
            energy: 0.0,
            weight: 1.0,
            access_count: 0,
        }
    }

    /// Calculate memory energy
    pub fn calculate_energy(&mut self) {
        // Lower energy = more stable memory
        self.energy = -(self.importance * self.recency * self.coherence);
    }

    /// Calculate Boltzmann weight at temperature T
    pub fn calculate_weight(&mut self, temperature: f64) {
        self.calculate_energy();
        self.weight = (-self.energy / temperature.max(0.01)).exp();
    }

    /// Decay recency over time
    pub fn decay(&mut self, decay_rate: f64) {
        self.recency *= decay_rate;
    }

    /// Reinforce on access
    pub fn access(&mut self) {
        self.access_count += 1;
        self.recency = 1.0; // Reset recency
        self.importance *= 1.1; // Slight importance boost
        self.importance = self.importance.min(10.0); // Cap
    }
}

/// pBit-enhanced biological memory system
#[derive(Debug)]
pub struct PBitMemorySystem {
    config: PBitBiologicalConfig,
    /// Short-term memory (high recency, variable importance)
    short_term: Vec<PBitMemory>,
    /// Long-term memory (consolidated)
    long_term: Vec<PBitMemory>,
    /// Partition function
    partition_function: f64,
    /// Total consolidation events
    consolidation_count: u64,
}

impl PBitMemorySystem {
    /// Create new memory system
    pub fn new(config: PBitBiologicalConfig) -> Self {
        Self {
            config,
            short_term: Vec::new(),
            long_term: Vec::new(),
            partition_function: 1.0,
            consolidation_count: 0,
        }
    }

    /// Add memory to short-term storage
    pub fn store(&mut self, id: String, embedding: Vec<f64>, importance: f64) {
        let memory = PBitMemory::new(id, embedding, importance);
        self.short_term.push(memory);
    }

    /// Consolidate memories from short-term to long-term
    pub fn consolidate(&mut self) -> ConsolidationResult {
        let mut consolidated = 0;
        let mut forgotten = 0;

        // Update weights for all short-term memories
        for memory in &mut self.short_term {
            memory.calculate_weight(self.config.temperature);
        }

        // Calculate partition function
        self.partition_function = self.short_term.iter()
            .map(|m| m.weight)
            .sum::<f64>()
            .max(1.0);

        // Consolidate based on Boltzmann probability
        let mut to_consolidate = Vec::new();
        let mut to_forget = Vec::new();

        for (i, memory) in self.short_term.iter().enumerate() {
            let prob = memory.weight / self.partition_function;
            
            if prob > self.config.consolidation_threshold {
                to_consolidate.push(i);
            } else if memory.recency < 0.1 {
                to_forget.push(i);
            }
        }

        // Move to long-term (reverse order to preserve indices)
        for &i in to_consolidate.iter().rev() {
            let memory = self.short_term.remove(i);
            self.long_term.push(memory);
            consolidated += 1;
            self.consolidation_count += 1;
        }

        // Remove forgotten (reverse order)
        for &i in to_forget.iter().rev() {
            if i < self.short_term.len() {
                self.short_term.remove(i);
                forgotten += 1;
            }
        }

        ConsolidationResult {
            consolidated,
            forgotten,
            short_term_count: self.short_term.len(),
            long_term_count: self.long_term.len(),
            partition_function: self.partition_function,
        }
    }

    /// Recall memory by ID
    pub fn recall(&mut self, id: &str) -> Option<&PBitMemory> {
        // Search long-term first (more stable)
        for memory in &mut self.long_term {
            if memory.id == id {
                memory.access();
                return Some(memory);
            }
        }

        // Then short-term
        for memory in &mut self.short_term {
            if memory.id == id {
                memory.access();
                return Some(memory);
            }
        }

        None
    }

    /// Recall by similarity (Boltzmann-weighted)
    pub fn recall_similar(&self, query: &[f64], k: usize) -> Vec<(&PBitMemory, f64)> {
        let mut results: Vec<(&PBitMemory, f64)> = Vec::new();

        // Combine all memories
        for memory in self.long_term.iter().chain(self.short_term.iter()) {
            let similarity = self.cosine_similarity(query, &memory.embedding);
            // Weight by Boltzmann factor
            let weighted_sim = similarity * memory.weight;
            results.push((memory, weighted_sim));
        }

        // Sort by weighted similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }

    /// Cosine similarity between vectors
    fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Decay all memories
    pub fn decay_all(&mut self) {
        let decay_rate = (-1.0 / self.config.decay_tau).exp();
        
        for memory in &mut self.short_term {
            memory.decay(decay_rate);
        }
        
        // Long-term decays more slowly
        for memory in &mut self.long_term {
            memory.decay(decay_rate.powf(0.1));
        }
    }

    /// Get system entropy
    pub fn entropy(&self) -> f64 {
        let all_memories: Vec<&PBitMemory> = self.short_term.iter()
            .chain(self.long_term.iter())
            .collect();

        if all_memories.is_empty() {
            return 0.0;
        }

        let z: f64 = all_memories.iter().map(|m| m.weight).sum();
        
        all_memories.iter()
            .map(|m| {
                let p = m.weight / z;
                if p > f64::EPSILON {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }
}

/// Result of consolidation
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    pub consolidated: usize,
    pub forgotten: usize,
    pub short_term_count: usize,
    pub long_term_count: usize,
    pub partition_function: f64,
}

/// pBit-enhanced Hebbian learning
#[derive(Debug)]
pub struct PBitHebbianLearner {
    config: PBitBiologicalConfig,
    /// Synaptic weights (connection strengths)
    weights: HashMap<(usize, usize), f64>,
    /// Eligibility traces
    traces: HashMap<(usize, usize), f64>,
}

impl PBitHebbianLearner {
    /// Create new learner
    pub fn new(config: PBitBiologicalConfig) -> Self {
        Self {
            config,
            weights: HashMap::new(),
            traces: HashMap::new(),
        }
    }

    /// STDP update based on spike timing
    pub fn stdp_update(&mut self, pre: usize, post: usize, delta_t: f64) {
        let key = (pre, post);
        
        // Get current weight
        let w = *self.weights.get(&key).unwrap_or(&0.5);
        
        // STDP function: positive for pre-before-post, negative for post-before-pre
        let dw = if delta_t > 0.0 {
            // LTP: pre before post
            self.config.learning_rate * (1.0 - w) * (-delta_t / self.config.stdp_tau).exp()
        } else {
            // LTD: post before pre
            -self.config.learning_rate * w * (delta_t / self.config.stdp_tau).exp()
        };

        // Update weight with pBit bounds
        let new_w = (w + dw).clamp(0.0, 1.0);
        self.weights.insert(key, new_w);
    }

    /// Get connection weight
    pub fn get_weight(&self, pre: usize, post: usize) -> f64 {
        *self.weights.get(&(pre, post)).unwrap_or(&0.5)
    }

    /// Decay all weights toward baseline
    pub fn decay_weights(&mut self, decay_rate: f64) {
        let baseline = 0.5;
        for weight in self.weights.values_mut() {
            *weight = *weight * decay_rate + baseline * (1.0 - decay_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_energy() {
        let mut memory = PBitMemory::new("test".to_string(), vec![1.0, 0.0], 0.8);
        memory.coherence = 0.9;
        memory.calculate_energy();
        
        // Energy should be negative (stable)
        assert!(memory.energy < 0.0);
    }

    #[test]
    fn test_memory_consolidation() {
        let config = PBitBiologicalConfig::default();
        let mut system = PBitMemorySystem::new(config);

        // Add memories with varying importance
        system.store("high".to_string(), vec![1.0], 0.9);
        system.store("low".to_string(), vec![0.0], 0.1);

        let result = system.consolidate();
        assert!(result.consolidated > 0 || result.short_term_count > 0);
    }

    #[test]
    fn test_recall_similar() {
        let config = PBitBiologicalConfig::default();
        let mut system = PBitMemorySystem::new(config);

        system.store("a".to_string(), vec![1.0, 0.0, 0.0], 0.8);
        system.store("b".to_string(), vec![0.0, 1.0, 0.0], 0.8);
        system.store("c".to_string(), vec![0.9, 0.1, 0.0], 0.8);

        let query = vec![1.0, 0.0, 0.0];
        let results = system.recall_similar(&query, 2);

        assert_eq!(results.len(), 2);
        // First should be most similar to query
        assert_eq!(results[0].0.id, "a");
    }

    #[test]
    fn test_stdp_ltp() {
        let config = PBitBiologicalConfig::default();
        let mut learner = PBitHebbianLearner::new(config);

        let initial = learner.get_weight(0, 1);
        
        // Pre before post = LTP
        learner.stdp_update(0, 1, 5.0);
        
        assert!(learner.get_weight(0, 1) > initial);
    }

    #[test]
    fn test_stdp_ltd() {
        let config = PBitBiologicalConfig::default();
        let mut learner = PBitHebbianLearner::new(config);

        // Set initial weight high
        learner.weights.insert((0, 1), 0.8);
        let initial = learner.get_weight(0, 1);
        
        // Post before pre = LTD
        learner.stdp_update(0, 1, -5.0);
        
        assert!(learner.get_weight(0, 1) < initial);
    }
}
