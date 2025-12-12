//! # Layer 4: Learning & Plasticity API
//!
//! Synaptic plasticity, knowledge consolidation, and reasoning.
//!
//! ## Scientific Foundation
//!
//! **Three-Factor STDP** (Spike-Timing Dependent Plasticity):
//! - Hebbian: "Neurons that fire together, wire together"
//! - Timing: Pre-before-post → potentiation, post-before-pre → depression
//! - Dopamine: Neuromodulatory signal for reward-based learning
//!
//! ## Key Equations
//!
//! ```text
//! STDP Weight Change:
//!   Δw = η · A(Δt) · D
//!
//!   where:
//!   A(Δt) = A_+ exp(-Δt/τ_+)  if Δt > 0  (potentiation)
//!         = -A_- exp(Δt/τ_-)  if Δt < 0  (depression)
//!   D = dopamine signal
//!   η = learning rate
//!
//! Fibonacci STDP:
//!   τ = φ (golden ratio) for optimal temporal credit assignment
//! ```
//!
//! ## References
//! - Bi & Poo (1998). Synaptic modifications in cultured hippocampal neurons.
//! - Izhikevich (2007). Solving the distal reward problem through linkage of STDP and dopamine.

use crate::{Result, QksError};
use std::collections::HashMap;

/// Golden ratio τ for Fibonacci STDP
pub const FIBONACCI_TAU: f64 = 1.618_033_988_749_895;

/// Default learning rate
pub const DEFAULT_LEARNING_RATE: f64 = 0.01;

/// STDP time window (ms)
pub const STDP_WINDOW: f64 = 20.0;

/// Potentiation amplitude
pub const A_PLUS: f64 = 0.1;

/// Depression amplitude
pub const A_MINUS: f64 = 0.105;

/// Synapse with learning capability
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Synaptic weight
    pub weight: f64,
    /// Pre-synaptic neuron ID
    pub pre_neuron: usize,
    /// Post-synaptic neuron ID
    pub post_neuron: usize,
    /// Eligibility trace
    pub eligibility: f64,
    /// Last update time
    pub last_update: f64,
}

/// Learning event (spike pair)
#[derive(Debug, Clone)]
pub struct SpikeEvent {
    /// Neuron ID
    pub neuron_id: usize,
    /// Spike time (ms)
    pub time: f64,
    /// Is presynaptic?
    pub is_pre: bool,
}

/// Knowledge item in reasoning bank
#[derive(Debug, Clone)]
pub struct KnowledgeItem {
    /// Unique identifier
    pub id: String,
    /// Content representation
    pub content: Vec<f64>,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Usage count
    pub usage_count: usize,
    /// Last access time
    pub last_access: f64,
}

/// Reasoning bank (long-term knowledge storage)
#[derive(Debug, Clone)]
pub struct ReasoningBank {
    /// Stored knowledge items
    items: HashMap<String, KnowledgeItem>,
    /// Consolidation threshold
    consolidation_threshold: f64,
}

impl ReasoningBank {
    /// Create new reasoning bank
    pub fn new() -> Self {
        Self {
            items: HashMap::new(),
            consolidation_threshold: 0.7,
        }
    }

    /// Store knowledge item
    pub fn store(&mut self, item: KnowledgeItem) {
        self.items.insert(item.id.clone(), item);
    }

    /// Retrieve knowledge by ID
    pub fn retrieve(&mut self, id: &str) -> Option<&mut KnowledgeItem> {
        if let Some(item) = self.items.get_mut(id) {
            item.usage_count += 1;
            Some(item)
        } else {
            None
        }
    }

    /// Consolidate: Strengthen high-usage items, forget low-usage
    pub fn consolidate(&mut self) {
        // Remove items below threshold
        self.items.retain(|_, item| {
            item.confidence >= self.consolidation_threshold || item.usage_count > 5
        });

        // Boost frequently used items
        for item in self.items.values_mut() {
            if item.usage_count > 10 {
                item.confidence = (item.confidence * 1.1).min(1.0);
            }
        }
    }

    /// Get all items
    pub fn items(&self) -> &HashMap<String, KnowledgeItem> {
        &self.items
    }
}

impl Default for ReasoningBank {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute STDP weight change
///
/// # Arguments
/// * `delta_t` - Spike time difference (post - pre) in milliseconds
/// * `dopamine` - Dopamine signal (0-1)
/// * `learning_rate` - Learning rate η
///
/// # Returns
/// Weight change Δw
///
/// # Formula
/// ```text
/// Δw = η · A(Δt) · D
///
/// A(Δt) = A_+ exp(-Δt/τ_+)  if Δt > 0
///       = -A_- exp(Δt/τ_-)  if Δt < 0
/// ```
///
/// # Example
/// ```rust,ignore
/// // Pre fires 5ms before post → potentiation
/// let dw = apply_stdp(5.0, 1.0, 0.01)?;
/// assert!(dw > 0.0);
///
/// // Post fires 5ms before pre → depression
/// let dw = apply_stdp(-5.0, 1.0, 0.01)?;
/// assert!(dw < 0.0);
/// ```
pub fn apply_stdp(delta_t: f64, dopamine: f64, learning_rate: f64) -> Result<f64> {
    if !(0.0..=1.0).contains(&dopamine) {
        return Err(QksError::InvalidConfig("Dopamine must be in [0,1]".to_string()));
    }

    let tau = FIBONACCI_TAU; // Use golden ratio for optimal learning

    let amplitude = if delta_t > 0.0 {
        // Potentiation: pre before post
        A_PLUS * (-delta_t / tau).exp()
    } else {
        // Depression: post before pre
        -A_MINUS * (delta_t / tau).exp()
    };

    let weight_change = learning_rate * amplitude * dopamine;

    Ok(weight_change)
}

/// Compute STDP weight change using Fibonacci time constant
///
/// # Arguments
/// * `delta_t` - Spike time difference (ms)
/// * `reward` - Reward signal (can be negative)
///
/// # Returns
/// Weight change Δw
pub fn stdp_weight_change(delta_t: f64, reward: f64) -> f64 {
    let dopamine = reward.max(0.0).min(1.0); // Clamp to [0,1]
    apply_stdp(delta_t, dopamine, DEFAULT_LEARNING_RATE).unwrap_or(0.0)
}

/// Update synapse weights based on spike pairs
///
/// # Arguments
/// * `synapses` - Mutable slice of synapses
/// * `spike_events` - Recent spike events
/// * `dopamine` - Global dopamine signal
///
/// # Example
/// ```rust,ignore
/// update_synapses(&mut synapses, &spike_events, 0.8)?;
/// ```
pub fn update_synapses(
    synapses: &mut [Synapse],
    spike_events: &[SpikeEvent],
    dopamine: f64,
) -> Result<()> {
    for synapse in synapses {
        // Find relevant pre and post spikes
        let pre_spikes: Vec<f64> = spike_events
            .iter()
            .filter(|e| e.is_pre && e.neuron_id == synapse.pre_neuron)
            .map(|e| e.time)
            .collect();

        let post_spikes: Vec<f64> = spike_events
            .iter()
            .filter(|e| !e.is_pre && e.neuron_id == synapse.post_neuron)
            .map(|e| e.time)
            .collect();

        // Compute weight change for all spike pairs
        let mut total_dw = 0.0;
        for &t_pre in &pre_spikes {
            for &t_post in &post_spikes {
                let delta_t = t_post - t_pre;
                if delta_t.abs() < STDP_WINDOW {
                    total_dw += apply_stdp(delta_t, dopamine, DEFAULT_LEARNING_RATE)?;
                }
            }
        }

        // Update weight with bounds
        synapse.weight = (synapse.weight + total_dw).clamp(0.0, 1.0);
    }

    Ok(())
}

/// Consolidate working memory into reasoning bank
///
/// # Arguments
/// * `items` - Items to consolidate
/// * `threshold` - Minimum confidence for consolidation
///
/// # Returns
/// Number of items consolidated
///
/// # Example
/// ```rust,ignore
/// let count = consolidate(&working_memory_items, 0.7)?;
/// println!("Consolidated {} items", count);
/// ```
pub fn consolidate(items: &[KnowledgeItem], threshold: f64) -> Result<usize> {
    let mut count = 0;

    for item in items {
        if item.confidence >= threshold {
            // TODO: Store in reasoning bank
            count += 1;
        }
    }

    Ok(count)
}

/// Transfer knowledge between domains (transfer learning)
///
/// # Arguments
/// * `source_domain` - Source knowledge representation
/// * `target_domain` - Target domain mapping
///
/// # Returns
/// Transferred knowledge
pub fn transfer_knowledge(
    source_domain: &[f64],
    target_domain: &[f64],
) -> Result<Vec<f64>> {
    if source_domain.len() != target_domain.len() {
        return Err(QksError::InvalidConfig("Domain dimension mismatch".to_string()));
    }

    // Simple linear combination for now
    let transferred: Vec<f64> = source_domain
        .iter()
        .zip(target_domain.iter())
        .map(|(s, t)| 0.7 * s + 0.3 * t)
        .collect();

    Ok(transferred)
}

/// Compute eligibility trace for delayed reward
///
/// # Arguments
/// * `time_since_spike` - Time since spike (ms)
/// * `tau` - Time constant (ms)
///
/// # Returns
/// Eligibility trace value
///
/// # Formula
/// ```text
/// e(t) = exp(-t/τ)
/// ```
pub fn eligibility_trace(time_since_spike: f64, tau: f64) -> f64 {
    (-time_since_spike / tau).exp()
}

/// Update eligibility traces for all synapses
///
/// # Arguments
/// * `synapses` - Mutable slice of synapses
/// * `current_time` - Current simulation time
/// * `decay_tau` - Decay time constant
pub fn update_eligibility_traces(
    synapses: &mut [Synapse],
    current_time: f64,
    decay_tau: f64,
) {
    for synapse in synapses {
        let dt = current_time - synapse.last_update;
        synapse.eligibility *= eligibility_trace(dt, decay_tau);
        synapse.last_update = current_time;
    }
}

/// Meta-learning: Learn optimal learning rate
///
/// # Arguments
/// * `performance_history` - Recent performance metrics
///
/// # Returns
/// Updated learning rate
pub fn meta_learn_rate(performance_history: &[f64]) -> f64 {
    if performance_history.len() < 2 {
        return DEFAULT_LEARNING_RATE;
    }

    // If improving, increase rate; if degrading, decrease
    let recent_improvement =
        performance_history[performance_history.len() - 1]
            - performance_history[performance_history.len() - 2];

    if recent_improvement > 0.0 {
        (DEFAULT_LEARNING_RATE * 1.1).min(0.1)
    } else {
        (DEFAULT_LEARNING_RATE * 0.9).max(0.001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_stdp_potentiation() {
        // Pre fires before post → potentiation
        let dw = apply_stdp(5.0, 1.0, 0.01).unwrap();
        assert!(dw > 0.0);
    }

    #[test]
    fn test_stdp_depression() {
        // Post fires before pre → depression
        let dw = apply_stdp(-5.0, 1.0, 0.01).unwrap();
        assert!(dw < 0.0);
    }

    #[test]
    fn test_fibonacci_tau() {
        assert_relative_eq!(FIBONACCI_TAU, 1.618034, epsilon = 1e-5);
    }

    #[test]
    fn test_eligibility_trace() {
        let e = eligibility_trace(0.0, 1.0);
        assert_relative_eq!(e, 1.0, epsilon = 1e-10);

        let e = eligibility_trace(1.0, 1.0);
        assert_relative_eq!(e, 1.0_f64.exp().recip(), epsilon = 1e-10);
    }

    #[test]
    fn test_reasoning_bank() {
        let mut bank = ReasoningBank::new();

        let item = KnowledgeItem {
            id: "fact_1".to_string(),
            content: vec![0.5, 0.3],
            confidence: 0.9,
            usage_count: 0,
            last_access: 0.0,
        };

        bank.store(item);
        assert_eq!(bank.items().len(), 1);

        let retrieved = bank.retrieve("fact_1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().usage_count, 1);
    }
}
