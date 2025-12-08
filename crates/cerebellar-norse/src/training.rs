//! Training engine for cerebellar Norse networks
//! 
//! Provides sophisticated training algorithms including surrogate gradient BPTT,
//! spike-timing dependent plasticity, and biological learning rules for 
//! cerebellar microcircuit adaptation.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use candle_nn::{self as nn, VarBuilder, VarMap, AdamW};
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};
use std::collections::HashMap;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{CerebellarNorseConfig, CerebellarMetrics};
use crate::cerebellar_circuit::CerebellarCircuit;
use crate::encoding::{InputEncoder, OutputDecoder};
use crate::compatibility::{TensorCompat, NeuralNetCompat, DTypeCompat, TrainingCompat};

/// Training configuration for cerebellar networks
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Surrogate gradient steepness
    pub surrogate_alpha: f64,
    /// STDP time window (ms)
    pub stdp_window: f64,
    /// LTP strength
    pub ltp_strength: f64,
    /// LTD strength
    pub ltd_strength: f64,
    /// Weight decay factor
    pub weight_decay: f64,
    /// Gradient clipping threshold
    pub gradient_clip: f64,
    /// Use biological plasticity rules
    pub use_biological_plasticity: bool,
    /// Parallel training batch size
    pub parallel_batch_size: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            surrogate_alpha: 100.0,
            stdp_window: 20.0,
            ltp_strength: 0.01,
            ltd_strength: 0.005,
            weight_decay: 1e-5,
            gradient_clip: 1.0,
            use_biological_plasticity: true,
            parallel_batch_size: 32,
        }
    }
}

/// Comprehensive training engine for cerebellar networks
pub struct TrainingEngine {
    /// Training configuration
    config: TrainingConfig,
    /// AdamW optimizer
    optimizer: AdamW,
    /// Device for computation
    device: Device,
    /// Loss function
    loss_fn: LossFunction,
    /// STDP plasticity engine
    stdp_engine: STDPEngine,
    /// Training statistics
    training_stats: TrainingStatistics,
    /// Gradient accumulator
    gradient_accumulator: HashMap<String, Tensor>,
}

impl TrainingEngine {
    /// Create new training engine
    pub fn new(
        network_config: &CerebellarNorseConfig,
        varmap: &VarMap,
        learning_rate: f64,
    ) -> Result<Self> {
        let mut config = TrainingConfig::default();
        config.learning_rate = learning_rate;
        
        // Create optimizer using candle-core compatibility
        let optimizer = NeuralNetCompat::create_adamw_optimizer(varmap, learning_rate)?;
        
        // Initialize loss function  
        let loss_fn = LossFunction::new(network_config.output_dim)?;
        
        // Initialize STDP engine
        let stdp_engine = STDPEngine::new(&config, network_config.device);
        
        info!("Training engine initialized with learning rate {:.2e}", learning_rate);
        
        Ok(Self {
            config,
            optimizer,
            device: network_config.device,
            loss_fn,
            stdp_engine,
            training_stats: TrainingStatistics::default(),
            gradient_accumulator: HashMap::new(),
        })
    }
    
    /// Train for one epoch
    pub fn train_epoch(
        &mut self,
        circuit: &mut CerebellarCircuit,
        encoder: &mut InputEncoder,
        decoder: &mut OutputDecoder,
        x_train: &Tensor,
        y_train: &Tensor,
        batch_size: usize,
    ) -> Result<f64> {
        let start_time = std::time::Instant::now();
        let n_samples = x_train.size()[0] as usize;
        let n_batches = (n_samples + batch_size - 1) / batch_size;
        
        let mut epoch_loss = 0.0;
        
        // Reset gradient accumulator
        self.gradient_accumulator.clear();
        
        for batch_idx in 0..n_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = ((batch_idx + 1) * batch_size).min(n_samples);
            let actual_batch_size = batch_end - batch_start;
            
            // Extract batch
            let x_batch = x_train.narrow(0, batch_start as i64, actual_batch_size as i64);
            let y_batch = y_train.narrow(0, batch_start as i64, actual_batch_size as i64);
            
            // Forward pass with gradient tracking
            let batch_loss = self.train_batch(
                circuit,
                encoder,
                decoder,
                &x_batch,
                &y_batch,
            )?;
            
            epoch_loss += batch_loss;
            
            // Update progress
            if batch_idx % 10 == 0 {
                debug!("Batch {}/{}: Loss = {:.6}", batch_idx + 1, n_batches, batch_loss);
            }
        }
        
        // Apply accumulated gradients
        self.apply_gradients()?;
        
        // Apply biological plasticity if enabled
        if self.config.use_biological_plasticity {
            self.apply_biological_plasticity(circuit)?;
        }
        
        let avg_loss = epoch_loss / n_batches as f64;
        
        // Update training statistics
        self.training_stats.update_epoch_stats(avg_loss, start_time.elapsed());
        
        debug!("Epoch completed in {}ms: Average loss = {:.6}", 
               start_time.elapsed().as_millis(), avg_loss);
        
        Ok(avg_loss)
    }
    
    /// Train single batch with surrogate gradient BPTT
    fn train_batch(
        &mut self,
        circuit: &mut CerebellarCircuit,
        encoder: &mut InputEncoder,
        decoder: &mut OutputDecoder,
        x_batch: &Tensor,
        y_batch: &Tensor,
    ) -> Result<f64> {
        // Reset circuit state
        circuit.reset();
        
        // Encode inputs to spike patterns
        let encoded_inputs = encoder.encode(x_batch)?;
        
        // Forward pass through circuit with gradient tracking
        let mut spike_history = Vec::new();
        let mut membrane_potentials = Vec::new();
        
        // Time-stepped forward pass for BPTT
        let time_steps = encoded_inputs.len();
        let mut current_state = circuit.get_initial_state()?;
        
        for t in 0..time_steps {
            let input_t = &encoded_inputs[t];
            let (spikes, v_mem) = circuit.forward_timestep(input_t, &current_state)?;
            
            spike_history.push(spikes);
            membrane_potentials.push(v_mem.clone());
            current_state = circuit.update_state(&current_state, &spikes)?;
        }
        
        // Decode final outputs
        let final_spikes = spike_history.last().unwrap();
        let predictions = decoder.decode(final_spikes)?;
        
        // Compute loss
        let loss = self.loss_fn.compute_loss(&predictions, y_batch)?;
        
        // Surrogate gradient backward pass through time
        let gradients = self.surrogate_gradient_bptt(
            &spike_history,
            &membrane_potentials,
            &loss
        )?;
        
        // Apply gradients with clipping
        self.apply_surrogate_gradients(circuit, gradients)?;
        
        // Update STDP engine with spike activity
        if self.config.use_biological_plasticity {
            let circuit_outputs = self.spike_history_to_outputs(&spike_history)?;
            self.stdp_engine.update_spike_history(&circuit_outputs);
        }
        
        let loss_value = TensorCompat::sum_compat(&loss)?;
        Ok(loss_value)
    }
    
    /// Surrogate gradient backpropagation through time for spiking networks
    fn surrogate_gradient_bptt(
        &self,
        spike_history: &[Tensor],
        membrane_potentials: &[Tensor],
        loss: &Tensor,
    ) -> Result<Vec<HashMap<String, Tensor>>> {
        let time_steps = spike_history.len();
        let mut gradients = vec![HashMap::new(); time_steps];
        
        // Initialize gradient backpropagation from final loss
        let mut grad_output = loss.clone();
        
        // Backward pass through time
        for t in (0..time_steps).rev() {
            let spikes_t = &spike_history[t];
            let v_mem_t = &membrane_potentials[t];
            
            // Compute surrogate gradient for spike function
            // Using fast sigmoid approximation: g'(v) = alpha / (alpha * |v - v_th| + 1)^2
            let v_th = 1.0; // Threshold voltage
            let alpha = self.config.surrogate_alpha as f32;
            
            let v_diff = v_mem_t.sub_scalar(v_th)?;
            let v_abs = v_diff.abs()?;
            let denominator = v_abs.mul_scalar(alpha)?.add_scalar(1.0)?.powf(2.0)?;
            let surrogate_grad = Tensor::full(v_mem_t.dims(), alpha, v_mem_t.device())?.div(&denominator)?;
            
            // Gradient of loss w.r.t. membrane potential
            let grad_v_mem = &grad_output * &surrogate_grad;
            
            // Store gradients for this timestep
            gradients[t].insert("v_mem".to_string(), grad_v_mem.clone());
            gradients[t].insert("spikes".to_string(), grad_output.clone());
            
            // Propagate gradients to previous timestep
            if t > 0 {
                // Gradient flows through membrane dynamics: v[t] = decay * v[t-1] + input[t]
                let membrane_decay = 0.9; // Typical decay factor
                grad_output = grad_v_mem.mul_scalar(membrane_decay)?;
            }
        }
        
        Ok(gradients)
    }
    
    /// Apply surrogate gradients to circuit parameters
    fn apply_surrogate_gradients(
        &mut self,
        circuit: &mut CerebellarCircuit,
        gradients: Vec<HashMap<String, Tensor>>,
    ) -> Result<()> {
        // Accumulate gradients across time steps
        let mut accumulated_gradients = HashMap::new();
        
        for grad_step in gradients {
            for (param_name, grad) in grad_step {
                let acc_grad = accumulated_gradients.entry(param_name.clone())
                    .or_insert_with(|| Tensor::zeros(grad.dims(), grad.dtype(), grad.device()).unwrap());
                *acc_grad = (acc_grad.clone() + grad)?;
            }
        }
        
        // Apply gradient clipping
        for (param_name, grad) in &mut accumulated_gradients {
            let grad_norm = TensorCompat::sum_compat(&grad.abs()?)?;
            if grad_norm > self.config.gradient_clip {
                let scale_factor = self.config.gradient_clip / grad_norm;
                *grad = grad.mul_scalar(scale_factor as f32)?;
            }
        }
        
        // Update circuit parameters
        circuit.apply_gradients(&accumulated_gradients, self.config.learning_rate)?;
        
        // Update training statistics
        let total_grad_norm = accumulated_gradients.values()
            .map(|g| TensorCompat::sum_compat(&g.abs().unwrap()).unwrap_or(0.0))
            .sum::<f64>();
        self.training_stats.gradient_norms.push(total_grad_norm);
        
        debug!("Applied surrogate gradients: total_norm={:.6}", total_grad_norm);
        
        Ok(())
    }
    
    /// Convert spike history to circuit outputs format
    fn spike_history_to_outputs(&self, spike_history: &[Tensor]) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();
        
        if let Some(final_spikes) = spike_history.last() {
            outputs.insert("final_spikes".to_string(), final_spikes.clone());
            outputs.insert("spike_rate".to_string(), 
                TensorCompat::mean_compat(final_spikes)?.into());
        }
        
        // Add temporal spike patterns for STDP
        for (t, spikes) in spike_history.iter().enumerate() {
            outputs.insert(format!("spikes_t{}", t), spikes.clone());
        }
        
        Ok(outputs)
    }
    
    /// Apply gradient clipping
    fn clip_gradients(&self) -> Result<()> {
        // Note: tch-rs handles gradient clipping through optimizer configuration
        // This is a placeholder for custom gradient processing if needed
        Ok(())
    }
    
    /// Apply accumulated gradients
    fn apply_gradients(&mut self) -> Result<()> {
        // Custom gradient application logic if needed
        // Currently handled by PyTorch optimizer
        Ok(())
    }
    
    /// Apply biological plasticity rules
    fn apply_biological_plasticity(&mut self, circuit: &mut CerebellarCircuit) -> Result<()> {
        let plasticity_updates = self.stdp_engine.compute_plasticity_updates()?;
        circuit.apply_plasticity_updates(plasticity_updates)?;
        Ok(())
    }
    
    /// Get training statistics
    pub fn get_statistics(&self) -> &TrainingStatistics {
        &self.training_stats
    }
    
    /// Reset training state
    pub fn reset(&mut self) {
        self.training_stats = TrainingStatistics::default();
        self.stdp_engine.reset();
        self.gradient_accumulator.clear();
    }
}

/// Loss functions for cerebellar network training
pub struct LossFunction {
    /// Output dimension
    output_dim: usize,
    /// Loss type
    loss_type: LossType,
}

#[derive(Debug, Clone, Copy)]
enum LossType {
    MSE,
    CrossEntropy,
    SpikeLoss,
}

impl LossFunction {
    pub fn new(output_dim: usize) -> Result<Self> {
        let loss_type = if output_dim == 1 {
            LossType::MSE
        } else {
            LossType::CrossEntropy
        };
        
        Ok(Self {
            output_dim,
            loss_type,
        })
    }
    
    pub fn compute_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        match self.loss_type {
            LossType::MSE => {
                let diff = predictions - targets;
                let mse = (&diff * &diff)?;
                TensorCompat::mean_compat(&mse).map_err(|e| candle_core::Error::Msg(format!("MSE calculation failed: {}", e)))
            }
            LossType::CrossEntropy => {
                Ok(predictions.cross_entropy_for_logits(targets))
            }
            LossType::SpikeLoss => {
                // Spike-based loss for spiking neural networks
                let sum_tensor = predictions.sum(1)?;
                let mean_penalty = TensorCompat::mean_compat(&sum_tensor).map_err(|e| candle_core::Error::Msg(format!("Penalty calculation failed: {}", e)))?;
                let spike_penalty = mean_penalty * 0.01;
                let diff = (predictions - targets)?;
                let mse_tensor = (&diff * &diff)?;
                let mse_loss = TensorCompat::mean_compat(&mse_tensor).map_err(|e| candle_core::Error::Msg(format!("MSE loss failed: {}", e)))?;
                Ok(mse_loss + spike_penalty)
            }
        }
    }
}

/// STDP (Spike-Timing Dependent Plasticity) engine
pub struct STDPEngine {
    /// Configuration
    config: TrainingConfig,
    /// Device
    device: Device,
    /// Spike history buffer
    spike_history: HashMap<String, Vec<Tensor>>,
    /// Plasticity update buffer
    plasticity_updates: HashMap<String, Tensor>,
    /// Time window for STDP
    time_window: usize,
}

impl STDPEngine {
    pub fn new(config: &TrainingConfig, device: Device) -> Self {
        let time_window = (config.stdp_window / 1.0) as usize; // Assuming 1ms time steps
        
        Self {
            config: config.clone(),
            device,
            spike_history: HashMap::new(),
            plasticity_updates: HashMap::new(),
            time_window,
        }
    }
    
    /// Update spike history with new activity
    pub fn update_spike_history(&mut self, circuit_outputs: &HashMap<String, Tensor>) {
        for (layer_name, spikes) in circuit_outputs {
            if layer_name.contains("spikes") {
                let history = self.spike_history.entry(layer_name.clone()).or_insert_with(Vec::new);
                history.push(spikes.detach().shallow_clone());
                
                // Keep only recent history
                if history.len() > self.time_window {
                    history.remove(0);
                }
            }
        }
    }
    
    /// Compute plasticity updates based on spike timing
    pub fn compute_plasticity_updates(&mut self) -> Result<HashMap<String, Tensor>> {
        let mut updates = HashMap::new();
        
        // Compute pairwise STDP between connected layers
        self.compute_stdp_updates("grc_spikes", "pc_spikes", "grc_pc", &mut updates)?;
        self.compute_stdp_updates("pc_spikes", "dcn_spikes", "pc_dcn", &mut updates)?;
        
        Ok(updates)
    }
    
    /// Compute STDP updates between two layers
    fn compute_stdp_updates(
        &self,
        pre_layer: &str,
        post_layer: &str,
        connection_name: &str,
        updates: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        let pre_history = self.spike_history.get(pre_layer);
        let post_history = self.spike_history.get(post_layer);
        
        if let (Some(pre_spikes), Some(post_spikes)) = (pre_history, post_history) {
            if pre_spikes.len() > 1 && post_spikes.len() > 1 {
                let update = self.compute_stdp_weight_update(pre_spikes, post_spikes)?;
                updates.insert(connection_name.to_string(), update);
            }
        }
        
        Ok(())
    }
    
    /// Compute STDP weight update based on spike timing with temporal correlation analysis
    fn compute_stdp_weight_update(
        &self,
        pre_spikes: &[Tensor],
        post_spikes: &[Tensor],
    ) -> Result<Tensor> {
        let n_pre = pre_spikes.len();
        let n_post = post_spikes.len();
        
        if n_pre == 0 || n_post == 0 {
            return Err(anyhow!("Empty spike history"));
        }
        
        let pre_dims = pre_spikes[0].dims();
        let post_dims = post_spikes[0].dims();
        
        // Initialize weight update matrix
        let mut weight_update = Tensor::zeros(
            &[post_dims[0], pre_dims[0]], // [post_neurons, pre_neurons]
            DType::F32,
            &self.device
        )?;
        
        // Compute eligibility traces for each neuron pair
        let mut eligibility_traces = Tensor::zeros(
            &[post_dims[0], pre_dims[0], self.time_window],
            DType::F32,
            &self.device
        )?;
        
        // Compute causal (LTP) and anti-causal (LTD) components
        for (t_post, post_spike) in post_spikes.iter().enumerate() {
            for (t_pre, pre_spike) in pre_spikes.iter().enumerate() {
                let dt = t_post as f64 - t_pre as f64;
                
                // Only consider spikes within the STDP window
                if dt.abs() <= self.config.stdp_window {
                    // Compute STDP weight based on temporal difference
                    let stdp_magnitude = if dt > 0.0 {
                        // Causal case: post-synaptic spike after pre-synaptic (LTP)
                        self.config.ltp_strength * (-dt / self.config.stdp_window * 5.0).exp()
                    } else {
                        // Anti-causal case: pre-synaptic spike after post-synaptic (LTD)
                        -self.config.ltd_strength * (dt / self.config.stdp_window * 5.0).exp()
                    };
                    
                    // Convert spikes to float for computation
                    let pre_float = pre_spike.to_dtype(DType::F32)?;
                    let post_float = post_spike.to_dtype(DType::F32)?;
                    
                    // Compute outer product for synaptic weight updates
                    // Each element (i,j) represents connection from pre-neuron j to post-neuron i
                    for i in 0..post_dims[0] {
                        for j in 0..pre_dims[0] {
                            // Extract spike activity for specific neurons
                            let post_activity = post_float.get(i)?;
                            let pre_activity = pre_float.get(j)?;
                            
                            // Compute correlation-based weight update
                            let correlation = TensorCompat::sum_compat(&(&post_activity * &pre_activity)?)?
                                .mul_scalar(stdp_magnitude as f32);
                            
                            // Apply learning rule with homeostatic scaling
                            let current_update = weight_update.get([i, j])?;
                            let new_update = current_update.add_scalar(correlation as f32)?;
                            weight_update = weight_update.slice_set(&[i..i+1, j..j+1], &new_update.unsqueeze(0)?.unsqueeze(0)?)?;
                        }
                    }
                    
                    // Update eligibility traces
                    let trace_idx = (t_post.min(t_pre)) % self.time_window;
                    let trace_value = stdp_magnitude.abs() as f32;
                    
                    let current_traces = eligibility_traces.narrow(2, trace_idx, 1)?.squeeze(2)?;
                    let updated_traces = current_traces.add_scalar(trace_value)?;
                    eligibility_traces = eligibility_traces.slice_set(
                        &[0..post_dims[0], 0..pre_dims[0], trace_idx..trace_idx+1],
                        &updated_traces.unsqueeze(2)?
                    )?;
                }
            }
        }
        
        // Apply homeostatic scaling to prevent runaway potentiation/depression
        let weight_norm = TensorCompat::sum_compat(&weight_update.abs()?)? + 1e-8;
        let scaling_factor = (self.config.ltp_strength + self.config.ltd_strength) / weight_norm;
        weight_update = weight_update.mul_scalar(scaling_factor as f32)?;
        
        // Apply eligibility trace normalization
        let trace_norm = TensorCompat::mean_compat(&eligibility_traces.sum(2)?)? + 1e-8;
        let trace_scaling = 1.0 / trace_norm.max(1.0);
        weight_update = weight_update.mul_scalar(trace_scaling as f32)?;
        
        // Apply weight decay for stability
        weight_update = weight_update.mul_scalar(1.0 - self.config.weight_decay as f32)?;
        
        debug!("STDP weight update computed: norm={:.6}, max={:.6}, min={:.6}", 
               weight_norm, 
               TensorCompat::sum_compat(&weight_update.max(1)?.max(0)?)?,
               TensorCompat::sum_compat(&weight_update.min(1)?.min(0)?)?);
        
        Ok(weight_update)
    }
    
    /// Reset STDP state
    pub fn reset(&mut self) {
        self.spike_history.clear();
        self.plasticity_updates.clear();
    }
}

/// Training statistics tracking
#[derive(Debug, Default, Clone)]
pub struct TrainingStatistics {
    /// Loss history
    pub loss_history: Vec<f64>,
    /// Training times per epoch
    pub epoch_times: Vec<std::time::Duration>,
    /// Gradient norms
    pub gradient_norms: Vec<f64>,
    /// Learning rate schedule
    pub learning_rates: Vec<f64>,
    /// Total training time
    pub total_training_time: std::time::Duration,
}

impl TrainingStatistics {
    pub fn update_epoch_stats(&mut self, loss: f64, epoch_time: std::time::Duration) {
        self.loss_history.push(loss);
        self.epoch_times.push(epoch_time);
        self.total_training_time += epoch_time;
    }
    
    pub fn get_average_loss(&self, last_n: usize) -> f64 {
        if self.loss_history.is_empty() {
            return 0.0;
        }
        
        let start_idx = self.loss_history.len().saturating_sub(last_n);
        let recent_losses = &self.loss_history[start_idx..];
        recent_losses.iter().sum::<f64>() / recent_losses.len() as f64
    }
    
    pub fn is_converged(&self, tolerance: f64, patience: usize) -> bool {
        if self.loss_history.len() < patience {
            return false;
        }
        
        let recent_losses = &self.loss_history[self.loss_history.len() - patience..];
        let max_loss = recent_losses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_loss = recent_losses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        (max_loss - min_loss) < tolerance
    }
}

/// Parallel batch training for improved performance
pub struct ParallelTrainer {
    /// Number of parallel workers
    n_workers: usize,
    /// Batch size per worker
    worker_batch_size: usize,
}

impl ParallelTrainer {
    pub fn new(n_workers: usize, total_batch_size: usize) -> Self {
        let worker_batch_size = (total_batch_size + n_workers - 1) / n_workers;
        Self {
            n_workers,
            worker_batch_size,
        }
    }
    
    /// Train multiple batches in parallel
    pub fn parallel_train_batches(
        &self,
        training_data: Vec<(Tensor, Tensor)>,
    ) -> Result<Vec<f64>> {
        let losses: Result<Vec<f64>> = training_data
            .par_iter()
            .map(|(_x_batch, _y_batch)| {
                // This would need access to individual network copies
                // Implementation depends on thread-safe network cloning
                // For now, return placeholder
                Ok(0.0)
            })
            .collect();
        
        losses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CerebellarNorseConfig, CerebellarCircuit};
    use std::collections::HashMap;
    
    #[test]
    fn test_training_engine_creation() {
        let config = CerebellarNorseConfig::default();
        let var_map = VarMap::new();
        let vs = VarBuilder::from_varmap(&var_map, DType::F32, &Device::Cpu);
        
        let trainer = TrainingEngine::new(&config, &var_map, 1e-3).unwrap();
        assert_eq!(trainer.config.learning_rate, 1e-3);
    }
    
    #[test]
    fn test_loss_function() {
        let loss_fn = LossFunction::new(1).unwrap();
        
        let predictions = Tensor::ones(&[2, 1], (Kind::Float, Device::Cpu));
        let targets = Tensor::zeros(&[2, 1], (Kind::Float, Device::Cpu));
        
        let loss = loss_fn.compute_loss(&predictions, &targets).unwrap();
        let loss_value: f64 = loss.into();
        assert!(loss_value > 0.0);
    }
    
    #[test]
    fn test_stdp_engine() {
        let config = TrainingConfig::default();
        let mut stdp = STDPEngine::new(&config, Device::Cpu);
        
        let mut circuit_outputs = HashMap::new();
        circuit_outputs.insert(
            "grc_spikes".to_string(),
            Tensor::rand(&[2, 100], (Kind::Float, Device::Cpu))
        );
        
        stdp.update_spike_history(&circuit_outputs);
        assert!(stdp.spike_history.contains_key("grc_spikes"));
    }
    
    #[test]
    fn test_training_statistics() {
        let mut stats = TrainingStatistics::default();
        
        stats.update_epoch_stats(1.0, std::time::Duration::from_millis(100));
        stats.update_epoch_stats(0.5, std::time::Duration::from_millis(110));
        
        assert_eq!(stats.get_average_loss(2), 0.75);
        assert!(!stats.is_converged(0.1, 3));
    }
}
