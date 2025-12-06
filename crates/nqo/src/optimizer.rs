//! Main NQO optimizer implementation

use crate::cache::{OptimizationCache, CacheKey};
use crate::error::NqoResult;
use crate::hardware;
use crate::neural_network::NeuralNetwork;
use crate::quantum_circuits::{QuantumBackend, QuantumOptimizationCircuits};
use crate::types::*;
use ndarray::Array1;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Main Neuromorphic Quantum Optimizer
pub struct NeuromorphicQuantumOptimizer {
    /// Configuration
    config: NqoConfig,
    /// Quantum backend
    quantum_backend: Arc<Mutex<QuantumBackend>>,
    /// Quantum circuits
    quantum_circuits: Arc<QuantumOptimizationCircuits>,
    /// Neural network
    neural_network: Arc<NeuralNetwork>,
    /// Optimization cache
    cache: Arc<OptimizationCache>,
    /// Optimization history
    optimization_history: Arc<RwLock<VecDeque<OptimizationResult>>>,
    /// Execution times
    execution_times: Arc<RwLock<Vec<f64>>>,
    /// Initialization status
    is_initialized: bool,
}

impl NeuromorphicQuantumOptimizer {
    /// Create a new NQO instance
    pub async fn new(config: NqoConfig) -> NqoResult<Self> {
        // Initialize logging
        Self::init_logging(&config.log_level)?;
        
        info!("Initializing NQO with {} neurons and {} qubits", config.neurons, config.qubits);
        
        // Detect hardware
        let hw_caps = hardware::detect_hardware();
        info!("Hardware capabilities: {:?}", hw_caps);
        
        // Create quantum backend
        let quantum_backend = QuantumBackend::new(config.qubits, config.quantum_shots)?;
        
        // Create quantum circuits
        let quantum_circuits = QuantumOptimizationCircuits::new(config.qubits);
        
        // Create neural network with flexible input dimension
        // Use max of 20 to handle most optimization problems
        let neural_network = NeuralNetwork::new(
            config.neurons,
            20, // Flexible input dimension for various problem sizes
            10, // Output dimension for gradient processing
            config.use_gpu && (hw_caps.has_cuda || hw_caps.has_apple_gpu),
        )?;
        
        // Create cache
        let cache = OptimizationCache::new(config.cache_size as u64);
        
        let mut optimizer = Self {
            config,
            quantum_backend: Arc::new(Mutex::new(quantum_backend)),
            quantum_circuits: Arc::new(quantum_circuits),
            neural_network: Arc::new(neural_network),
            cache: Arc::new(cache),
            optimization_history: Arc::new(RwLock::new(VecDeque::with_capacity(50))),
            execution_times: Arc::new(RwLock::new(Vec::with_capacity(100))),
            is_initialized: false,
        };
        
        // Initialize
        optimizer.initialize().await?;
        
        Ok(optimizer)
    }
    
    /// Initialize the optimizer
    async fn initialize(&mut self) -> NqoResult<()> {
        if self.is_initialized {
            return Ok(());
        }
        
        info!("Initializing NQO optimizer");
        
        // Reset neural network hidden state
        self.neural_network.reset_hidden_state()?;
        
        self.is_initialized = true;
        info!("NQO optimizer initialized successfully");
        
        Ok(())
    }
    
    /// Optimize parameters for an objective function
    pub async fn optimize_parameters<F>(
        &self,
        objective: F,
        initial_params: Vec<f64>,
        iterations: usize,
    ) -> NqoResult<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync,
    {
        let start = Instant::now();
        info!("Starting parameter optimization with {} iterations", iterations);
        
        let mut params = initial_params.clone();
        let mut best_params = params.clone();
        let mut best_value = objective(&params);
        let initial_value = best_value;
        let mut history = vec![best_value];
        
        for i in 0..iterations {
            // Evaluate objective
            let value = objective(&params);
            
            // Update best if improved
            if value < best_value {
                best_value = value;
                best_params = params.clone();
            }
            
            // Calculate gradient
            let gradient = self.calculate_gradient(&objective, &params).await?;
            
            // Apply neuromorphic processing
            let processed_gradient = self.neuromorphic_processing(&gradient).await?;
            
            // Apply quantum optimization step
            params = self.quantum_optimization_step(&params, &processed_gradient).await?;
            
            // Update neural network if we have history
            if i > 0 {
                self.update_neural_network(&params, value, &history).await?;
            }
            
            history.push(value);
        }
        
        // Try quantum enhancement as final step
        if iterations > 0 && history.len() > 2 {
            let enhanced_params = self.quantum_enhance_parameters(&best_params, &history).await?;
            let enhanced_value = objective(&enhanced_params);
            
            if enhanced_value < best_value {
                best_params = enhanced_params;
                best_value = enhanced_value;
                history.push(enhanced_value);
            }
        }
        
        // Calculate metrics
        let improvement = (initial_value - best_value) / (initial_value.abs() + 1e-10);
        let confidence = (0.2 + 0.75 * improvement).clamp(0.1, 0.95);
        let execution_time = start.elapsed().as_millis() as f64;
        
        // Track execution time
        self.track_execution_time(execution_time);
        
        // Create result
        let result = OptimizationResult {
            params: best_params,
            value: best_value,
            initial_value,
            history,
            iterations,
            confidence,
            execution_time_ms: execution_time,
        };
        
        // Store in history
        self.store_optimization_result(result.clone());
        
        info!("Optimization completed in {:.2}ms. Final value: {:.6}", execution_time, best_value);
        
        Ok(result)
    }
    
    /// Calculate gradient numerically
    async fn calculate_gradient<F>(&self, objective: &F, params: &[f64]) -> NqoResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync,
    {
        let epsilon = 1e-5;
        let mut gradient = vec![0.0; params.len()];
        let base_value = objective(params);
        
        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            params_plus[i] += epsilon;
            let value_plus = objective(&params_plus);
            gradient[i] = (value_plus - base_value) / epsilon;
        }
        
        Ok(gradient)
    }
    
    /// Apply neuromorphic processing to gradient
    async fn neuromorphic_processing(&self, gradient: &[f64]) -> NqoResult<Vec<f64>> {
        let num_elements = gradient.len().min(self.config.neurons);
        debug!("Applying gradient update to {} neural elements", num_elements);
        
        // Select largest gradient elements
        let mut indexed_gradient: Vec<(usize, f64)> = gradient.iter()
            .copied()
            .enumerate()
            .collect();
        indexed_gradient.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut processed = vec![0.0; gradient.len()];
        
        // Process through neural network - pad input to expected dimension
        let mut padded_gradient = gradient.to_vec();
        padded_gradient.resize(20, 0.0); // Match neural network input dimension
        let input = Array1::from_vec(padded_gradient);
        let tensor = self.neural_network.array_to_tensor(&input)?;
        let output = self.neural_network.forward(&tensor)?;
        let nn_output = self.neural_network.tensor_to_array(&output)?;
        
        // Apply adaptivity
        for (i, &grad) in gradient.iter().enumerate() {
            let nn_factor = if i < nn_output.len() { nn_output[i] } else { 0.0 };
            processed[i] = self.config.adaptivity * nn_factor + (1.0 - self.config.adaptivity) * grad;
        }
        
        Ok(processed)
    }
    
    /// Apply quantum optimization step
    async fn quantum_optimization_step(&self, params: &[f64], gradient: &[f64]) -> NqoResult<Vec<f64>> {
        // Check cache
        let cache_key = CacheKey::new("qaoa", &[params, gradient].concat());
        
        if let Some(cached) = self.cache.get(&cache_key).await {
            return Ok(cached);
        }
        
        // Normalize inputs
        let params_norm = self.normalize_vector(params);
        let grad_norm = self.normalize_vector(gradient);
        
        // Select subset for quantum processing
        let subset_size = self.config.qubits.min(params.len());
        let mut largest_indices: Vec<usize> = (0..gradient.len())
            .collect::<Vec<_>>();
        largest_indices.sort_by(|&a, &b| {
            gradient[b].abs().partial_cmp(&gradient[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
        largest_indices.truncate(subset_size);
        
        let selected_params: Vec<f64> = largest_indices.iter()
            .map(|&i| params_norm[i])
            .collect();
        let selected_grads: Vec<f64> = largest_indices.iter()
            .map(|&i| grad_norm[i])
            .collect();
        
        // Build and execute QAOA circuit
        let circuit = self.quantum_circuits.build_qaoa_circuit(&selected_params, &selected_grads)?;
        
        let mut backend = self.quantum_backend.lock().await;
        let measurements = backend.execute(&circuit)?;
        
        // Convert measurements to parameter updates
        let mut new_params = params.to_vec();
        for (i, &idx) in largest_indices.iter().enumerate() {
            if i < measurements.len() {
                let update = (1.0 - measurements[i]) / 2.0;
                new_params[idx] -= self.config.learning_rate * update * gradient[idx];
            }
        }
        
        // Cache result
        self.cache.insert(cache_key, new_params.clone()).await;
        
        Ok(new_params)
    }
    
    /// Enhance parameters using quantum circuits
    async fn quantum_enhance_parameters(&self, best_params: &[f64], history: &[f64]) -> NqoResult<Vec<f64>> {
        // Execute exploration circuit
        let circuit = self.quantum_circuits.build_parameter_exploration_circuit(best_params)?;
        
        let mut backend = self.quantum_backend.lock().await;
        let quantum_result = backend.execute(&circuit)?;
        
        // Determine optimal direction
        let optimal_direction: Vec<f64> = quantum_result.iter()
            .map(|&v| v * 2.0 - 1.0)
            .collect();
        
        // Get improvement direction from history
        let recent_improvement = if history.len() >= 3 {
            let recent_avg = history[history.len()-3..].iter().sum::<f64>() / 3.0;
            let older_avg = history[0..3.min(history.len())].iter().sum::<f64>() / 3.0;
            (older_avg - recent_avg) / (older_avg.abs() + 1e-10)
        } else {
            0.0
        };
        
        // Apply refinement
        let mut enhanced = best_params.to_vec();
        for i in 0..enhanced.len().min(optimal_direction.len()) {
            enhanced[i] += optimal_direction[i] * recent_improvement * 0.1;
        }
        
        Ok(enhanced)
    }
    
    /// Update neural network based on optimization progress
    async fn update_neural_network(&self, params: &[f64], value: f64, history: &[f64]) -> NqoResult<()> {
        // Prepare input features
        let mut features = params.to_vec();
        features.push(value);
        features.push(history.iter().sum::<f64>() / history.len() as f64);
        features.push(history.iter().fold(0.0f64, |a, &b| a.max(b)) - history.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
        
        // Ensure correct dimension - match neural network input
        features.resize(20, 0.0);
        
        let input = Array1::from_vec(features);
        let tensor = self.neural_network.array_to_tensor(&input)?;
        
        // Forward pass
        let output = self.neural_network.forward(&tensor)?;
        
        // Simple gradient based on improvement
        let improvement = if history.len() >= 2 {
            (history[history.len()-2] - history[history.len()-1]) / (history[history.len()-2].abs() + 1e-10)
        } else {
            0.0
        };
        
        // Create pseudo-gradient for learning
        let gradient = candle_core::Tensor::new(&[improvement], &output.device())?;
        
        // Update weights
        self.neural_network.update_weights(&gradient, self.config.learning_rate)?;
        
        Ok(())
    }
    
    /// Optimize trading parameters
    pub async fn optimize_trading_parameters(
        &self,
        potential_matches: std::collections::HashMap<String, f64>,
        _order_book_data: Option<std::collections::HashMap<String, serde_json::Value>>,
    ) -> NqoResult<TradingParameters> {
        // Define objective function
        let matches = potential_matches.clone();
        let objective = move |params: &[f64]| -> f64 {
            let entry_threshold = params[0];
            let stop_loss = params[1];
            let take_profit = params[2];
            
            // Calculate expected value
            let probability = matches.values().sum::<f64>() / matches.len().max(1) as f64;
            let risk = stop_loss;
            let reward = take_profit;
            
            let expected_value = probability * reward - (1.0 - probability) * risk;
            
            // Penalties
            let mut penalty = 0.0;
            if entry_threshold < 0.3 || entry_threshold > 0.9 {
                penalty += 10.0;
            }
            if stop_loss < 0.01 || stop_loss > 0.1 {
                penalty += 10.0;
            }
            if take_profit < stop_loss * 1.5 {
                penalty += 10.0;
            }
            
            -expected_value + penalty
        };
        
        let initial_params = vec![0.6, 0.03, 0.06];
        let result = self.optimize_parameters(objective, initial_params, 5).await?;
        
        Ok(TradingParameters {
            entry_threshold: result.params[0],
            stop_loss: result.params[1],
            take_profit: result.params[2],
            confidence: (1.0 - result.value).clamp(0.0, 1.0),
        })
    }
    
    /// Optimize allocation
    pub async fn optimize_allocation(
        &self,
        _pair: &str,
        edge: f64,
        win_rate: f64,
        market_data: std::collections::HashMap<String, f64>,
    ) -> NqoResult<AllocationResult> {
        let objective = move |params: &[f64]| -> f64 {
            let allocation = params[0];
            
            // Kelly criterion
            let kelly = win_rate - (1.0 - win_rate) / (edge / 0.03);
            let optimal_allocation = (kelly / 2.0).max(0.0);
            
            // Deviation penalty
            let deviation_penalty = (allocation - optimal_allocation).powi(2);
            
            // Risk penalty
            let volatility = market_data.get("volatility").copied().unwrap_or(0.5);
            let risk_penalty = allocation * volatility;
            
            deviation_penalty + risk_penalty
        };
        
        let initial_params = vec![0.05];
        let result = self.optimize_parameters(objective, initial_params, 5).await?;
        
        Ok(AllocationResult {
            allocation: result.params[0],
            confidence: (1.0 - result.value).clamp(0.0, 1.0),
        })
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        let history = self.optimization_history.read();
        
        if history.is_empty() {
            return PerformanceMetrics {
                mean_improvement: 0.0,
                success_rate: 0.0,
                sample_size: 0,
            };
        }
        
        let mut improvements = Vec::new();
        let mut successes = 0;
        
        for result in history.iter() {
            let improvement = (result.initial_value - result.value) / (result.initial_value.abs() + 1e-10);
            improvements.push(improvement);
            
            if result.value < result.initial_value {
                successes += 1;
            }
        }
        
        PerformanceMetrics {
            mean_improvement: improvements.iter().sum::<f64>() / improvements.len() as f64,
            success_rate: successes as f64 / history.len() as f64,
            sample_size: history.len(),
        }
    }
    
    /// Get execution statistics
    pub fn get_execution_stats(&self) -> ExecutionStats {
        let times = self.execution_times.read();
        
        if times.is_empty() {
            return ExecutionStats::default();
        }
        
        let sum: f64 = times.iter().sum();
        let avg = sum / times.len() as f64;
        let min = times.iter().copied().fold(f64::INFINITY, f64::min);
        let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        
        ExecutionStats {
            avg_time_ms: avg,
            min_time_ms: min,
            max_time_ms: max,
            count: times.len(),
        }
    }
    
    /// Clear cache
    pub async fn clear_cache(&self) {
        self.cache.clear().await;
        info!("Cache cleared");
    }
    
    /// Reset optimizer
    pub async fn reset(&mut self) -> NqoResult<()> {
        info!("Resetting NQO optimizer");
        
        // Clear history
        self.optimization_history.write().clear();
        self.execution_times.write().clear();
        
        // Reset neural network
        self.neural_network.reset_hidden_state()?;
        
        // Clear cache
        self.clear_cache().await;
        
        Ok(())
    }
    
    /// Main optimization method for OptimizationProblem
    pub async fn optimize(&self, problem: &OptimizationProblem) -> NqoResult<OptimizationResult> {
        let start = Instant::now();
        info!("Starting optimization for {}-dimensional problem", problem.dimension);
        
        // Convert to internal parameter optimization
        let objective = |params: &[f64]| -> f64 {
            (problem.objective)(params)
        };
        
        // Use the existing optimize_parameters method
        let result = self.optimize_parameters(
            objective,
            problem.initial_params.clone(),
            self.config.epochs,
        ).await?;
        
        let execution_time = start.elapsed().as_secs_f64() * 1000.0;
        self.track_execution_time(execution_time);
        
        info!("Optimization completed in {:.2}ms", execution_time);
        Ok(result)
    }
    
    /// Track execution time
    fn track_execution_time(&self, time_ms: f64) {
        let mut times = self.execution_times.write();
        times.push(time_ms);
        
        if times.len() > 100 {
            times.remove(0);
        }
    }
    
    /// Store optimization result
    fn store_optimization_result(&self, result: OptimizationResult) {
        let mut history = self.optimization_history.write();
        history.push_back(result);
        
        if history.len() > self.config.max_history {
            history.pop_front();
        }
    }
    
    /// Normalize vector
    fn normalize_vector(&self, vec: &[f64]) -> Vec<f64> {
        let max_abs = vec.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        
        if max_abs > 0.0 {
            vec.iter().map(|v| v / max_abs).collect()
        } else {
            vec.to_vec()
        }
    }
    
    /// Initialize logging
    fn init_logging(level: &str) -> NqoResult<()> {
        use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
        
        let level = match level.to_uppercase().as_str() {
            "ERROR" => tracing::Level::ERROR,
            "WARN" => tracing::Level::WARN,
            "INFO" => tracing::Level::INFO,
            "DEBUG" => tracing::Level::DEBUG,
            "TRACE" => tracing::Level::TRACE,
            _ => tracing::Level::INFO,
        };
        
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::new(
                format!("nqo={},roqoqo=warn", level)
            ))
            .with(tracing_subscriber::fmt::layer())
            .try_init()
            .ok();
            
        Ok(())
    }
}

// Implement Send + Sync for async usage
unsafe impl Send for NeuromorphicQuantumOptimizer {}
unsafe impl Sync for NeuromorphicQuantumOptimizer {}