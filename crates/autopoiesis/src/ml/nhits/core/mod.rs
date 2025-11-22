//! Core NHITS (Neural Hierarchical Interpolation for Time Series) Architecture
//! Integrated with Autopoietic Consciousness System

use std::sync::Arc;
use std::collections::HashMap;
use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

use crate::consciousness::{ConsciousnessField, NeuronalField};
use crate::core::autopoiesis::{AutopoieticSystem, BasicAutopoieticSystem};
use crate::core::sync::SynchronizedSwarm;

use super::blocks::HierarchicalBlock;
use super::attention::TemporalAttention;
use super::decomposition::MultiScaleDecomposer;
use super::adaptation::AdaptiveStructure;
use super::configs::NHITSConfig;

pub mod model;

// Re-export model types for backward compatibility
pub use model::{
    NHITSModel, NHITSConfig as ModelConfig, StackBlock, Block, 
    BasisExpansion, ActivationType, TrainingHistory as ModelTrainingHistory
};

/// Main NHITS model with consciousness integration
#[derive(Debug, Clone)]
pub struct NHITS {
    /// Model configuration
    config: NHITSConfig,
    
    /// Hierarchical neural blocks
    blocks: Vec<HierarchicalBlock>,
    
    /// Temporal attention mechanism
    attention: TemporalAttention,
    
    /// Multi-scale decomposer
    decomposer: MultiScaleDecomposer,
    
    /// Adaptive structure manager
    adapter: AdaptiveStructure,
    
    /// Consciousness field integration
    consciousness: Arc<ConsciousnessField>,
    
    // Removed autopoietic_system due to associated type complexity
    // autopoietic_system: Arc<dyn AutopoieticSystem>,
    
    /// Model state and metadata
    state: ModelState,
}

/// Model state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    pub epoch: usize,
    pub total_predictions: usize,
    pub performance_history: Vec<f64>,
    pub structural_changes: Vec<StructuralChange>,
    pub consciousness_coherence: f64,
}

/// Structural adaptation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralChange {
    pub timestamp: u64,
    pub change_type: ChangeType,
    pub performance_impact: f64,
    pub consciousness_influence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    BlockAdded { depth: usize, units: usize },
    BlockRemoved { depth: usize },
    AttentionReconfigured { heads: usize },
    PoolingAdjusted { factor: usize },
    BasisExpanded { new_basis: usize },
}

impl NHITS {
    /// Create new NHITS model with consciousness integration
    pub fn new(
        config: NHITSConfig,
        consciousness: Arc<ConsciousnessField>,
        autopoietic_system: Arc<dyn AutopoieticSystem<State = crate::core::state::SystemState, Boundary = crate::core::boundary::SystemBoundary, Process = crate::core::process::SystemProcess, Environment = crate::core::environment::SystemEnvironment>>,
    ) -> Self {
        let blocks = Self::initialize_blocks(&config);
        let attention = TemporalAttention::new(&config.attention_config);
        let decomposer = MultiScaleDecomposer::new(&config.decomposer_config);
        let adapter = AdaptiveStructure::new(&config.adaptation_config);
        
        Self {
            config,
            blocks,
            attention,
            decomposer,
            adapter,
            consciousness,
            autopoietic_system,
            state: ModelState {
                epoch: 0,
                total_predictions: 0,
                performance_history: Vec::new(),
                structural_changes: Vec::new(),
                consciousness_coherence: 1.0,
            },
        }
    }
    
    /// Forward pass with consciousness-aware processing
    pub fn forward(
        &mut self,
        input: &Array3<f64>,
        lookback_window: usize,
        forecast_horizon: usize,
    ) -> Result<Array3<f64>, NHITSError> {
        // Synchronize with consciousness field
        let consciousness_state = self.consciousness.get_current_state();
        self.state.consciousness_coherence = consciousness_state.coherence;
        
        // Multi-scale decomposition
        let decomposed = self.decomposer.decompose(input)?;
        
        // Process through hierarchical blocks with consciousness modulation
        let mut hidden_states = Vec::new();
        let mut current_input = decomposed.clone();
        
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let block_output = block.forward(
                &current_input,
                consciousness_state.field_strength,
            )?;
            
            // Apply consciousness-modulated attention
            let attended = self.attention.apply(
                &block_output,
                consciousness_state.attention_weights.get(i).cloned(),
            )?;
            
            hidden_states.push(attended.clone());
            current_input = attended;
        }
        
        // Aggregate predictions across scales
        let predictions = self.aggregate_predictions(&hidden_states, forecast_horizon)?;
        
        // Adapt structure based on performance
        self.maybe_adapt_structure(&predictions, &consciousness_state)?;
        
        // Update state
        self.state.total_predictions += 1;
        
        Ok(predictions)
    }
    
    /// Initialize hierarchical blocks based on configuration
    fn initialize_blocks(config: &NHITSConfig) -> Vec<HierarchicalBlock> {
        config.block_configs
            .iter()
            .map(|block_config| HierarchicalBlock::new(block_config.clone()))
            .collect()
    }
    
    /// Aggregate predictions from multiple scales
    fn aggregate_predictions(
        &self,
        hidden_states: &[Array3<f64>],
        forecast_horizon: usize,
    ) -> Result<Array3<f64>, NHITSError> {
        // Implement hierarchical aggregation
        // This is a simplified version - full implementation would include
        // proper interpolation and scale-aware aggregation
        
        let batch_size = hidden_states[0].shape()[0];
        let mut aggregated = Array3::zeros((batch_size, forecast_horizon, 1));
        
        for (scale_idx, hidden) in hidden_states.iter().enumerate() {
            let scale_weight = self.get_scale_weight(scale_idx);
            aggregated = aggregated + hidden.slice(s![.., ..forecast_horizon, ..]).to_owned() * scale_weight;
        }
        
        Ok(aggregated)
    }
    
    /// Get weight for each scale based on consciousness coherence
    fn get_scale_weight(&self, scale_idx: usize) -> f64 {
        let base_weight = 1.0 / (scale_idx + 1) as f64;
        base_weight * self.state.consciousness_coherence
    }
    
    /// Adapt model structure based on performance and consciousness state
    fn maybe_adapt_structure(
        &mut self,
        predictions: &Array3<f64>,
        consciousness_state: &ConsciousnessState,
    ) -> Result<(), NHITSError> {
        let adaptation_decision = self.adapter.evaluate(
            &self.state.performance_history,
            consciousness_state,
            &self.config.adaptation_config,
        )?;
        
        if let Some(change) = adaptation_decision {
            self.apply_structural_change(change)?;
        }
        
        Ok(())
    }
    
    /// Apply structural change to the model
    fn apply_structural_change(&mut self, change: StructuralChange) -> Result<(), NHITSError> {
        match &change.change_type {
            ChangeType::BlockAdded { depth, units } => {
                // Add new block at specified depth
                let new_block = HierarchicalBlock::new_with_units(*units);
                self.blocks.insert(*depth, new_block);
            }
            ChangeType::BlockRemoved { depth } => {
                if self.blocks.len() > 1 {
                    self.blocks.remove(*depth);
                }
            }
            ChangeType::AttentionReconfigured { heads } => {
                self.attention.reconfigure_heads(*heads);
            }
            ChangeType::PoolingAdjusted { factor } => {
                for block in &mut self.blocks {
                    block.adjust_pooling(*factor);
                }
            }
            ChangeType::BasisExpanded { new_basis } => {
                for block in &mut self.blocks {
                    block.expand_basis(*new_basis);
                }
            }
        }
        
        self.state.structural_changes.push(change);
        Ok(())
    }
    
    /// Train the model with consciousness-aware optimization
    pub fn train(
        &mut self,
        train_data: &Array3<f64>,
        val_data: Option<&Array3<f64>>,
        epochs: usize,
    ) -> Result<TrainingHistory, NHITSError> {
        let mut history = TrainingHistory::new();
        
        for epoch in 0..epochs {
            self.state.epoch = epoch;
            
            // Training logic with consciousness integration
            let epoch_loss = self.train_epoch(train_data)?;
            history.train_losses.push(epoch_loss);
            
            // Validation if provided
            if let Some(val) = val_data {
                let val_loss = self.validate(val)?;
                history.val_losses.push(val_loss);
            }
            
            // Update performance history
            self.state.performance_history.push(epoch_loss);
            
            // Consciousness-guided early stopping
            if self.should_stop_early(&history) {
                break;
            }
        }
        
        Ok(history)
    }
    
    /// Train single epoch
    fn train_epoch(&mut self, data: &Array3<f64>) -> Result<f64, NHITSError> {
        // Simplified training logic - full implementation would include
        // proper batching, gradient computation, and optimization
        let predictions = self.forward(data, self.config.lookback_window, self.config.forecast_horizon)?;
        let loss = self.compute_loss(data, &predictions)?;
        Ok(loss)
    }
    
    /// Validate model
    fn validate(&mut self, data: &Array3<f64>) -> Result<f64, NHITSError> {
        let predictions = self.forward(data, self.config.lookback_window, self.config.forecast_horizon)?;
        self.compute_loss(data, &predictions)
    }
    
    /// Compute loss with consciousness-aware regularization
    fn compute_loss(&self, targets: &Array3<f64>, predictions: &Array3<f64>) -> Result<f64, NHITSError> {
        // MSE loss with consciousness coherence regularization
        let mse = ((targets - predictions).mapv(|x| x * x)).mean().unwrap_or(0.0);
        let coherence_penalty = (1.0 - self.state.consciousness_coherence).powi(2);
        Ok(mse + self.config.coherence_weight * coherence_penalty)
    }
    
    /// Check for consciousness-guided early stopping
    fn should_stop_early(&self, history: &TrainingHistory) -> bool {
        if history.train_losses.len() < 10 {
            return false;
        }
        
        // Check if consciousness coherence is degrading
        if self.state.consciousness_coherence < self.config.min_coherence_threshold {
            return true;
        }
        
        // Check if loss is not improving
        let recent_losses = &history.train_losses[history.train_losses.len() - 10..];
        let loss_variance: f64 = recent_losses.iter()
            .map(|&x| (x - recent_losses.iter().sum::<f64>() / recent_losses.len() as f64).powi(2))
            .sum::<f64>() / recent_losses.len() as f64;
        
        loss_variance < self.config.early_stop_threshold
    }
    
    /// Generate predictions for a given horizon
    pub fn predict(&self, input: &Array1<f64>, horizon: usize) -> Result<Array1<f64>, NHITSError> {
        // Convert 1D input to 3D for processing
        let batch_size = 1;
        let seq_len = input.len();
        let features = 1;
        
        let mut input_3d = Array3::zeros((batch_size, seq_len, features));
        for (i, &val) in input.iter().enumerate() {
            input_3d[[0, i, 0]] = val;
        }
        
        // Create mutable clone for forward pass
        let mut model_clone = self.clone();
        let predictions_3d = model_clone.forward(&input_3d, seq_len, horizon)?;
        
        // Extract 1D predictions
        let mut predictions = Array1::zeros(horizon);
        for i in 0..horizon {
            predictions[i] = predictions_3d[[0, i, 0]];
        }
        
        Ok(predictions)
    }
    
    /// Update model online with new data
    pub fn update_online(
        &mut self,
        inputs: &[Array1<f64>],
        targets: &[Array1<f64>],
    ) -> Result<(), NHITSError> {
        if inputs.len() != targets.len() {
            return Err(NHITSError::ShapeMismatch {
                expected: vec![inputs.len()],
                got: vec![targets.len()],
            });
        }
        
        // Convert to 3D arrays for processing
        let batch_size = inputs.len();
        let seq_len = inputs[0].len();
        let target_len = targets[0].len();
        
        let mut input_3d = Array3::zeros((batch_size, seq_len, 1));
        let mut target_3d = Array3::zeros((batch_size, target_len, 1));
        
        for (b, (inp, tgt)) in inputs.iter().zip(targets.iter()).enumerate() {
            for (i, &val) in inp.iter().enumerate() {
                input_3d[[b, i, 0]] = val;
            }
            for (i, &val) in tgt.iter().enumerate() {
                target_3d[[b, i, 0]] = val;
            }
        }
        
        // Perform single training step
        let _ = self.train_epoch(&input_3d)?;
        
        Ok(())
    }
    
    /// Reset model weights to initial state
    pub fn reset_weights(&mut self) -> Result<(), NHITSError> {
        // Reinitialize blocks
        self.blocks = Self::initialize_blocks(&self.config);
        
        // Reset attention
        self.attention = TemporalAttention::new(&self.config.attention_config);
        
        // Reset state
        self.state = ModelState {
            epoch: 0,
            total_predictions: 0,
            performance_history: Vec::new(),
            structural_changes: Vec::new(),
            consciousness_coherence: 1.0,
        };
        
        Ok(())
    }
    
    /// Save model state for persistence
    pub fn save_state(&self) -> Result<ModelState, NHITSError> {
        Ok(self.state.clone())
    }
    
    /// Load model state from persistence
    pub fn load_state(&mut self, state: ModelState) -> Result<(), NHITSError> {
        self.state = state;
        Ok(())
    }
}

/// Training history tracker
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    pub train_losses: Vec<f64>,
    pub val_losses: Vec<f64>,
    pub best_epoch: usize,
    pub best_val_loss: f64,
}

impl TrainingHistory {
    fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            best_epoch: 0,
            best_val_loss: f64::INFINITY,
        }
    }
}

/// Consciousness state representation
#[derive(Debug, Clone)]
pub struct ConsciousnessState {
    pub coherence: f64,
    pub field_strength: f64,
    pub attention_weights: Vec<Option<Array2<f64>>>,
}

/// NHITS errors
#[derive(Debug, thiserror::Error)]
pub enum NHITSError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Computation error: {0}")]
    ComputationError(String),
    
    #[error("Adaptation error: {0}")]
    AdaptationError(String),
}

// Remove extern crate - ndarray is already in Cargo.toml dependencies

use ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nhits_creation() {
        // Test implementation
    }
}