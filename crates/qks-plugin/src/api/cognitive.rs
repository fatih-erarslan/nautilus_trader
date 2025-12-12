//! # Layer 2: Cognitive Processing API
//!
//! Basic cognitive functions: attention allocation, working memory, and pattern recognition.
//!
//! ## Scientific Foundation
//!
//! - **Attention**: Competitive selection based on salience
//! - **Working Memory**: Limited capacity buffer (7±2 items)
//! - **Pattern Recognition**: Template matching and feature extraction
//! - **Predictive Coding**: Error minimization through prediction
//!
//! ## Key Concepts
//!
//! ```text
//! Attention Allocation:
//!   salience_i = w_i * novelty_i * relevance_i
//!   attention_i = softmax(salience)
//!
//! Working Memory Capacity:
//!   N_items ≈ 7 ± 2 (Miller's Law)
//!
//! Pattern Matching:
//!   similarity = cosine(pattern, template)
//! ```

use crate::{Result, QksError};
use std::collections::HashMap;

/// Miller's Law: Working memory capacity
pub const WORKING_MEMORY_CAPACITY: usize = 7;

/// Attention focus threshold (0-1)
pub const ATTENTION_THRESHOLD: f64 = 0.1;

/// Memory item with activation level
#[derive(Debug, Clone)]
pub struct MemoryItem {
    /// Unique identifier
    pub id: String,
    /// Content representation (feature vector)
    pub content: Vec<f64>,
    /// Activation level (0-1)
    pub activation: f64,
    /// Age in timesteps
    pub age: usize,
    /// Number of retrievals
    pub retrievals: usize,
}

/// Attention focus state
#[derive(Debug, Clone)]
pub struct AttentionState {
    /// Currently focused items (id -> weight)
    pub focus: HashMap<String, f64>,
    /// Total attention capacity (sums to 1.0)
    pub capacity: f64,
    /// Distraction level
    pub distraction: f64,
}

/// Pattern recognition result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Pattern identifier
    pub pattern_id: String,
    /// Similarity score (0-1)
    pub similarity: f64,
    /// Matched features
    pub features: Vec<usize>,
    /// Confidence level
    pub confidence: f64,
}

/// Working memory state
#[derive(Debug, Clone)]
pub struct WorkingMemory {
    /// Items in working memory
    items: Vec<MemoryItem>,
    /// Maximum capacity
    capacity: usize,
}

impl WorkingMemory {
    /// Create new working memory with default capacity
    pub fn new() -> Self {
        Self::with_capacity(WORKING_MEMORY_CAPACITY)
    }

    /// Create working memory with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Check if memory is full
    pub fn is_full(&self) -> bool {
        self.items.len() >= self.capacity
    }

    /// Get number of items in memory
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if memory is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Store item in working memory
    pub fn store(&mut self, item: MemoryItem) -> Result<()> {
        if self.is_full() {
            // Evict least active item
            if let Some(min_idx) = self.items
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.activation.partial_cmp(&b.activation).unwrap())
                .map(|(i, _)| i)
            {
                self.items.remove(min_idx);
            }
        }
        self.items.push(item);
        Ok(())
    }

    /// Retrieve item by id
    pub fn retrieve(&mut self, id: &str) -> Option<&mut MemoryItem> {
        self.items.iter_mut().find(|item| item.id == id)
    }

    /// Get all items
    pub fn items(&self) -> &[MemoryItem] {
        &self.items
    }

    /// Decay all activation levels
    pub fn decay(&mut self, rate: f64) {
        for item in &mut self.items {
            item.activation *= 1.0 - rate;
            item.age += 1;
        }
    }
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Allocate attention to multiple stimuli
///
/// # Arguments
/// * `saliences` - Salience values for each stimulus
///
/// # Returns
/// Attention weights (sum to 1.0)
///
/// # Example
/// ```rust,ignore
/// let saliences = vec![0.5, 0.3, 0.9];
/// let attention = focus_attention(&saliences)?;
/// // attention ≈ [0.24, 0.15, 0.61]
/// ```
pub fn focus_attention(saliences: &[f64]) -> Result<Vec<f64>> {
    if saliences.is_empty() {
        return Err(QksError::InvalidConfig("Empty salience array".to_string()));
    }

    // Softmax normalization
    let max_sal = saliences.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = saliences
        .iter()
        .map(|&s| (s - max_sal).exp())
        .sum();

    Ok(saliences
        .iter()
        .map(|&s| (s - max_sal).exp() / exp_sum)
        .collect())
}

/// Compute salience for a stimulus
///
/// # Arguments
/// * `novelty` - Novelty score (0-1)
/// * `relevance` - Task relevance (0-1)
/// * `intensity` - Stimulus intensity (0-1)
///
/// # Returns
/// Combined salience score
pub fn compute_salience(novelty: f64, relevance: f64, intensity: f64) -> f64 {
    // Weighted combination
    0.4 * novelty + 0.4 * relevance + 0.2 * intensity
}

/// Store memory with activation
///
/// # Arguments
/// * `id` - Memory identifier
/// * `content` - Feature vector
/// * `activation` - Initial activation (0-1)
///
/// # Example
/// ```rust,ignore
/// store_memory("event_123", vec![0.5, 0.3, 0.8], 1.0)?;
/// ```
pub fn store_memory(id: &str, content: Vec<f64>, activation: f64) -> Result<()> {
    // TODO: Interface with actual memory system
    if activation < 0.0 || activation > 1.0 {
        return Err(QksError::InvalidConfig("Activation must be in [0,1]".to_string()));
    }
    Ok(())
}

/// Retrieve memory by ID with activation boost
///
/// # Arguments
/// * `id` - Memory identifier
///
/// # Returns
/// Memory content and current activation
pub fn retrieve_memory(id: &str) -> Result<(Vec<f64>, f64)> {
    // TODO: Interface with actual memory system
    Ok((vec![], 0.0))
}

/// Recognize pattern in input
///
/// # Arguments
/// * `input` - Input feature vector
/// * `templates` - Known pattern templates
///
/// # Returns
/// Best matching pattern
///
/// # Example
/// ```rust,ignore
/// let input = vec![0.5, 0.3, 0.8];
/// let templates = vec![
///     ("pattern_A", vec![0.5, 0.3, 0.9]),
///     ("pattern_B", vec![0.1, 0.2, 0.3]),
/// ];
/// let matches = recognize_pattern(&input, &templates)?;
/// ```
pub fn recognize_pattern(
    input: &[f64],
    templates: &[(&str, Vec<f64>)],
) -> Result<Vec<PatternMatch>> {
    if input.is_empty() {
        return Err(QksError::InvalidConfig("Empty input vector".to_string()));
    }

    let mut matches = Vec::new();

    for (id, template) in templates {
        let similarity = cosine_similarity(input, template);

        matches.push(PatternMatch {
            pattern_id: id.to_string(),
            similarity,
            features: vec![], // TODO: Extract matched features
            confidence: similarity,
        });
    }

    // Sort by similarity (descending)
    matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

    Ok(matches)
}

/// Compute cosine similarity between two vectors
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity (-1 to 1)
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Update working memory with decay
///
/// # Arguments
/// * `decay_rate` - Decay rate per timestep (0-1)
///
/// # Example
/// ```rust,ignore
/// update_working_memory(0.1)?; // 10% decay per step
/// ```
pub fn update_working_memory(decay_rate: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&decay_rate) {
        return Err(QksError::InvalidConfig("Decay rate must be in [0,1]".to_string()));
    }
    // TODO: Interface with actual working memory
    Ok(())
}

/// Predict next stimulus using predictive coding
///
/// # Arguments
/// * `history` - Temporal sequence of observations
///
/// # Returns
/// Predicted next observation
pub fn predict_next(history: &[Vec<f64>]) -> Result<Vec<f64>> {
    if history.is_empty() {
        return Err(QksError::InvalidConfig("Empty history".to_string()));
    }

    // Simple linear extrapolation for now
    if history.len() == 1 {
        return Ok(history[0].clone());
    }

    let dim = history[0].len();
    let mut prediction = vec![0.0; dim];

    // Linear extrapolation: x_t+1 = 2*x_t - x_t-1
    for i in 0..dim {
        let x_t = history[history.len() - 1][i];
        let x_t_1 = history[history.len() - 2][i];
        prediction[i] = 2.0 * x_t - x_t_1;
    }

    Ok(prediction)
}

/// Compute prediction error
///
/// # Arguments
/// * `predicted` - Predicted observation
/// * `actual` - Actual observation
///
/// # Returns
/// Mean squared error
pub fn prediction_error(predicted: &[f64], actual: &[f64]) -> f64 {
    if predicted.len() != actual.len() {
        return f64::INFINITY;
    }

    let mse: f64 = predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f64>()
        / predicted.len() as f64;

    mse
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_working_memory_capacity() {
        let mut mem = WorkingMemory::new();
        assert_eq!(mem.capacity, WORKING_MEMORY_CAPACITY);
        assert!(mem.is_empty());
    }

    #[test]
    fn test_focus_attention() {
        let saliences = vec![0.5, 0.3, 0.9];
        let attention = focus_attention(&saliences).unwrap();

        // Check sum to 1
        let sum: f64 = attention.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);

        // Check highest salience gets most attention
        let max_idx = attention
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 2);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_relative_eq!(cosine_similarity(&a, &b), 1.0, epsilon = 1e-10);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_relative_eq!(cosine_similarity(&a, &b), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_working_memory_store() {
        let mut mem = WorkingMemory::with_capacity(2);

        let item1 = MemoryItem {
            id: "item1".to_string(),
            content: vec![0.5],
            activation: 1.0,
            age: 0,
            retrievals: 0,
        };

        mem.store(item1).unwrap();
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn test_prediction_error() {
        let predicted = vec![1.0, 2.0, 3.0];
        let actual = vec![1.1, 2.1, 3.1];
        let error = prediction_error(&predicted, &actual);
        assert_relative_eq!(error, 0.01, epsilon = 1e-10);
    }
}
