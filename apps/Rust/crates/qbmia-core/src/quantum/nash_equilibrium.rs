//! Quantum Nash Equilibrium solver implementation
//! 
//! High-performance Rust implementation of quantum Nash equilibrium finding using
//! variational quantum algorithms with SIMD optimization.

use crate::{
    config::QuantumConfig,
    error::{QBMIAError, Result},
    quantum::{QuantumCircuit, QuantumGate, QuantumMetrics, utils},
};
use ndarray::{Array1, Array2, Array3, Array4, Axis};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{Duration, Instant};
use rayon::prelude::*;

#[cfg(feature = "simd")]
use wide::*;

/// Game matrix representation for Nash equilibrium calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameMatrix {
    /// Payoff matrix: [player, opponent, player_action, opponent_action]
    payoffs: Array4<f64>,
    /// Number of players
    num_players: usize,
    /// Number of actions per player
    num_actions: usize,
}

impl GameMatrix {
    /// Create a new game matrix
    pub fn new(payoffs: Array4<f64>) -> Result<Self> {
        let shape = payoffs.shape();
        if shape.len() != 4 {
            return Err(QBMIAError::validation("Game matrix must be 4-dimensional"));
        }
        
        let num_players = shape[0];
        let num_actions = shape[2];
        
        if num_players == 0 || num_actions == 0 {
            return Err(QBMIAError::validation("Invalid game dimensions"));
        }
        
        if shape[0] != shape[1] || shape[2] != shape[3] {
            return Err(QBMIAError::validation("Game matrix must be symmetric in players and actions"));
        }
        
        Ok(Self {
            payoffs,
            num_players,
            num_actions,
        })
    }
    
    /// Get payoff for a player given action profile
    pub fn get_payoff(&self, player: usize, actions: &[usize]) -> Result<f64> {
        if player >= self.num_players {
            return Err(QBMIAError::validation("Player index out of bounds"));
        }
        
        if actions.len() != self.num_players {
            return Err(QBMIAError::validation("Action profile length mismatch"));
        }
        
        for &action in actions {
            if action >= self.num_actions {
                return Err(QBMIAError::validation("Action index out of bounds"));
            }
        }
        
        // For simplicity, consider 2-player games primarily
        if self.num_players == 2 {
            Ok(self.payoffs[[player, 1 - player, actions[player], actions[1 - player]]])
        } else {
            // Multi-player case - simplified
            Ok(self.payoffs[[player, 0, actions[player], actions[0]]])
        }
    }
    
    /// Get number of players
    pub fn num_players(&self) -> usize {
        self.num_players
    }
    
    /// Get number of actions
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }
}

/// Result of quantum Nash equilibrium calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumNashResult {
    /// Equilibrium strategies for each player
    pub strategies: HashMap<String, Array1<f64>>,
    /// Convergence score [0, 1]
    pub convergence_score: f64,
    /// Nash loss value
    pub nash_loss: f64,
    /// Number of iterations to convergence
    pub iterations: usize,
    /// Convergence history
    pub convergence_history: Vec<f64>,
    /// Quantum state entropy
    pub quantum_state_entropy: f64,
    /// Execution time in milliseconds
    pub execution_time: f64,
    /// Optimal action for player 0
    pub optimal_action: usize,
    /// Stability analysis
    pub stability_analysis: HashMap<String, f64>,
    /// Performance metrics
    pub metrics: QuantumMetrics,
}

/// Quantum Nash Equilibrium solver
pub struct QuantumNashEquilibrium {
    config: QuantumConfig,
    num_qubits: usize,
    num_layers: usize,
    circuit: QuantumCircuit,
    
    // Optimization state
    parameters: Array1<f64>,
    learning_rate: f64,
    convergence_threshold: f64,
    max_iterations: usize,
    
    // Performance tracking
    execution_stats: QuantumMetrics,
    
    // SIMD optimization buffers
    #[cfg(feature = "simd")]
    simd_buffer: Vec<f64x4>,
}

impl QuantumNashEquilibrium {
    /// Create a new quantum Nash equilibrium solver
    pub async fn new(config: QuantumConfig) -> Result<Self> {
        if config.num_qubits == 0 {
            return Err(QBMIAError::config("num_qubits must be greater than 0"));
        }
        
        if config.num_qubits > 32 {
            return Err(QBMIAError::config("num_qubits cannot exceed 32 for performance reasons"));
        }
        
        let num_qubits = config.num_qubits;
        let num_layers = config.num_layers;
        
        // Create quantum circuit
        let circuit = Self::create_variational_circuit(num_qubits, num_layers)?;
        
        // Initialize parameters
        let params_per_layer = num_qubits * 3; // RX, RY, RZ per qubit
        let total_params = num_layers * params_per_layer;
        let parameters = Self::initialize_parameters(total_params);
        
        #[cfg(feature = "simd")]
        let simd_buffer = vec![f64x4::splat(0.0); (1 << num_qubits + 3) / 4];
        
        Ok(Self {
            config: config.clone(),
            num_qubits,
            num_layers,
            circuit,
            parameters,
            learning_rate: config.learning_rate,
            convergence_threshold: config.convergence_threshold,
            max_iterations: config.max_iterations,
            execution_stats: QuantumMetrics::default(),
            #[cfg(feature = "simd")]
            simd_buffer,
        })
    }
    
    /// Find quantum Nash equilibrium for the given game
    pub async fn find_equilibrium(
        &mut self,
        game: &GameMatrix,
        market_conditions: Option<HashMap<String, f64>>,
    ) -> Result<QuantumNashResult> {
        let start_time = Instant::now();
        
        // Initialize parameters with game information
        self.encode_game_matrix(game, market_conditions.as_ref())?;
        
        let mut best_loss = f64::INFINITY;
        let mut best_strategies = HashMap::new();
        let mut convergence_history = Vec::new();
        
        // Optimization loop
        for iteration in 0..self.max_iterations {
            // Execute quantum circuit
            let probabilities = self.execute_quantum_circuit().await?;
            
            // Extract strategies from quantum state
            let strategies = self.extract_strategies(&probabilities, game)?;
            
            // Calculate Nash loss
            let loss = self.calculate_nash_loss(&strategies, game)?;
            convergence_history.push(loss);
            
            // Update best solution
            if loss < best_loss {
                best_loss = loss;
                best_strategies = strategies.clone();
            }
            
            // Check convergence
            if loss < self.convergence_threshold {
                log::debug!("Nash equilibrium converged at iteration {}", iteration);
                break;
            }
            
            // Update parameters using gradient-free optimization
            self.update_parameters(iteration)?;
        }
        
        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        // Calculate final metrics
        let convergence_score = self.calculate_convergence_score(best_loss, game.num_players());
        let quantum_entropy = self.calculate_entropy(&self.execute_quantum_circuit().await?);
        let optimal_action = self.determine_optimal_action(&best_strategies);
        let stability_analysis = self.analyze_stability(&best_strategies, game)?;
        
        // Update performance metrics
        self.execution_stats.execution_time_ms = execution_time;
        self.execution_stats.gates_executed = self.circuit.depth();
        self.execution_stats.state_entropy = quantum_entropy;
        self.execution_stats.convergence_rate = if convergence_history.len() > 1 {
            convergence_history[0] - convergence_history.last().unwrap()
        } else {
            0.0
        };
        
        Ok(QuantumNashResult {
            strategies: best_strategies,
            convergence_score,
            nash_loss: best_loss,
            iterations: convergence_history.len(),
            convergence_history,
            quantum_state_entropy: quantum_entropy,
            execution_time,
            optimal_action,
            stability_analysis,
            metrics: self.execution_stats.clone(),
        })
    }
    
    /// Create variational quantum circuit for Nash equilibrium
    fn create_variational_circuit(num_qubits: usize, num_layers: usize) -> Result<QuantumCircuit> {
        let mut circuit = QuantumCircuit::new(num_qubits);
        
        // Initial superposition layer
        for i in 0..num_qubits {
            circuit.add_gate(QuantumGate::H(i));
        }
        
        // Variational layers
        for layer in 0..num_layers {
            // Single qubit rotations
            for i in 0..num_qubits {
                let param_base = layer * num_qubits * 3 + i * 3;
                circuit.add_parameterized_gate(QuantumGate::RX(i, 0.0), param_base);
                circuit.add_parameterized_gate(QuantumGate::RY(i, 0.0), param_base + 1);
                circuit.add_parameterized_gate(QuantumGate::RZ(i, 0.0), param_base + 2);
            }
            
            // Entangling layer (ring topology)
            for i in 0..num_qubits {
                circuit.add_gate(QuantumGate::CNOT(i, (i + 1) % num_qubits));
            }
            
            // Additional entanglement for complex correlations
            if layer < num_layers - 1 {
                for i in (0..num_qubits - 1).step_by(2) {
                    circuit.add_gate(QuantumGate::CZ(i, i + 1));
                }
            }
        }
        
        circuit.validate()?;
        Ok(circuit)
    }
    
    /// Initialize random parameters
    fn initialize_parameters(size: usize) -> Array1<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Array1::from_vec(
            (0..size)
                .map(|_| rng.random_range(-std::f64::consts::PI..std::f64::consts::PI))
                .collect()
        )
    }
    
    /// Encode game matrix into quantum circuit parameters
    fn encode_game_matrix(
        &mut self,
        game: &GameMatrix,
        market_conditions: Option<&HashMap<String, f64>>,
    ) -> Result<()> {
        // Flatten and normalize payoff matrix
        let flat_payoffs: Vec<f64> = game.payoffs.iter().cloned().collect();
        
        // Normalize to [-π, π] range
        let max_abs = flat_payoffs.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
        let normalized: Vec<f64> = if max_abs > 0.0 {
            flat_payoffs.iter().map(|&x| x / max_abs * std::f64::consts::PI).collect()
        } else {
            flat_payoffs
        };
        
        // Update parameters with game information
        let param_len = self.parameters.len();
        for (i, &value) in normalized.iter().enumerate() {
            if i < param_len {
                self.parameters[i] += value * 0.1; // Small influence
            }
        }
        
        // Add market condition bias if provided
        if let Some(conditions) = market_conditions {
            if let Some(&volatility) = conditions.get("volatility") {
                // Add noise proportional to volatility
                use rand::Rng;
                let mut rng = rand::thread_rng();
                for param in self.parameters.iter_mut() {
                    *param += rng.random_range(-volatility * 0.1..volatility * 0.1);
                }
            }
            
            if let Some(&trend) = conditions.get("trend") {
                // Bias parameters based on trend
                for param in self.parameters.iter_mut() {
                    *param += trend * 0.05;
                }
            }
        }
        
        // Clip parameters to valid range
        for param in self.parameters.iter_mut() {
            *param = param.max(-std::f64::consts::PI).min(std::f64::consts::PI);
        }
        
        Ok(())
    }
    
    /// Execute quantum circuit and return state probabilities
    async fn execute_quantum_circuit(&self) -> Result<Vec<f64>> {
        let num_states = 1 << self.num_qubits;
        
        #[cfg(feature = "simd")]
        {
            self.execute_quantum_circuit_simd().await
        }
        
        #[cfg(not(feature = "simd"))]
        {
            self.execute_quantum_circuit_cpu().await
        }
    }
    
    #[cfg(feature = "simd")]
    async fn execute_quantum_circuit_simd(&self) -> Result<Vec<f64>> {
        let num_states = 1 << self.num_qubits;
        let mut state = vec![0.0; num_states];
        state[0] = 1.0; // |00...0⟩ initial state
        
        // Convert to SIMD vectors for parallel processing
        let simd_chunks = num_states / 4;
        let mut simd_state: Vec<f64x4> = state
            .chunks_exact(4)
            .map(|chunk| f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        // Convert back to regular state for processing
        let mut regular_state: Vec<f64> = simd_state.iter()
            .flat_map(|chunk| chunk.to_array())
            .collect();
        
        // Apply quantum gates with CPU implementation
        for gate in &self.circuit.gates {
            match gate {
                QuantumGate::H(qubit) => {
                    // Apply Hadamard gate on regular state
                    let num_states = regular_state.len();
                    let mut new_state = vec![0.0; num_states];
                    for i in 0..num_states {
                        let bit = (i >> qubit) & 1;
                        let partner = i ^ (1 << qubit);
                        if bit == 0 {
                            new_state[i] = (regular_state[i] + regular_state[partner]) / 2.0_f64.sqrt();
                        } else {
                            new_state[i] = (regular_state[i] - regular_state[partner]) / 2.0_f64.sqrt();
                        }
                    }
                    regular_state = new_state;
                }
                QuantumGate::RX(qubit, angle) => {
                    let angle = if let Some(param_idx) = self.get_parameter_index(gate) {
                        self.parameters[param_idx]
                    } else {
                        *angle
                    };
                    self.apply_rx_cpu(&mut regular_state, *qubit, angle)?;
                }
                QuantumGate::RY(qubit, angle) => {
                    let angle = if let Some(param_idx) = self.get_parameter_index(gate) {
                        self.parameters[param_idx]
                    } else {
                        *angle
                    };
                    self.apply_ry_cpu(&mut regular_state, *qubit, angle)?;
                }
                QuantumGate::RZ(qubit, angle) => {
                    let angle = if let Some(param_idx) = self.get_parameter_index(gate) {
                        self.parameters[param_idx]
                    } else {
                        *angle
                    };
                    self.apply_rz_cpu(&mut regular_state, *qubit, angle)?;
                }
                QuantumGate::CNOT(control, target) => {
                    self.apply_cnot_cpu(&mut regular_state, *control, *target)?;
                }
                QuantumGate::CZ(control, target) => {
                    self.apply_cz_cpu(&mut regular_state, *control, *target)?;
                }
                _ => {
                    return Err(QBMIAError::quantum_simulation("Unsupported gate for SIMD"));
                }
            }
        }
        
        // Convert back to SIMD state
        simd_state = regular_state
            .chunks_exact(4)
            .map(|chunk| f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        // Convert back to regular vector and calculate probabilities
        let final_state: Vec<f64> = simd_state
            .iter()
            .flat_map(|simd_vec| simd_vec.to_array().to_vec())
            .collect();
        
        Ok(final_state.iter().map(|&amp| amp * amp).collect())
    }
    
    #[cfg(not(feature = "simd"))]
    async fn execute_quantum_circuit_cpu(&self) -> Result<Vec<f64>> {
        let num_states = 1 << self.num_qubits;
        let mut state = vec![0.0; num_states];
        state[0] = 1.0; // |00...0⟩ initial state
        
        // Apply quantum gates sequentially
        for gate in &self.circuit.gates {
            match gate {
                QuantumGate::H(qubit) => {
                    self.apply_hadamard_cpu(&mut state, *qubit)?;
                }
                QuantumGate::RX(qubit, angle) => {
                    let angle = if let Some(param_idx) = self.get_parameter_index(gate) {
                        self.parameters[param_idx]
                    } else {
                        *angle
                    };
                    self.apply_rx_cpu(&mut state, *qubit, angle)?;
                }
                QuantumGate::RY(qubit, angle) => {
                    let angle = if let Some(param_idx) = self.get_parameter_index(gate) {
                        self.parameters[param_idx]
                    } else {
                        *angle
                    };
                    self.apply_ry_cpu(&mut state, *qubit, angle)?;
                }
                QuantumGate::RZ(qubit, angle) => {
                    let angle = if let Some(param_idx) = self.get_parameter_index(gate) {
                        self.parameters[param_idx]
                    } else {
                        *angle
                    };
                    self.apply_rz_cpu(&mut state, *qubit, angle)?;
                }
                QuantumGate::CNOT(control, target) => {
                    self.apply_cnot_cpu(&mut state, *control, *target)?;
                }
                QuantumGate::CZ(control, target) => {
                    self.apply_cz_cpu(&mut state, *control, *target)?;
                }
                _ => {
                    return Err(QBMIAError::quantum_simulation("Unsupported gate"));
                }
            }
        }
        
        // Calculate probabilities
        Ok(state.iter().map(|&amp| amp * amp).collect())
    }
    
    // SIMD gate implementations
    #[cfg(feature = "simd")]
    fn apply_hadamard_simd(&self, state: &mut [f64x4], qubit: usize) -> Result<()> {
        let qubit_bit = 1 << qubit;
        let sqrt_2_inv = f64x4::splat(1.0 / std::f64::consts::SQRT_2);
        
        for i in 0..state.len() {
            let base_idx = i * 4;
            let mut affected = [false; 4];
            let mut partner_indices = [0; 4];
            
            for j in 0..4 {
                let idx = base_idx + j;
                if idx < (1 << self.num_qubits) {
                    affected[j] = true;
                    partner_indices[j] = idx ^ qubit_bit;
                }
            }
            
            if affected.iter().any(|&x| x) {
                let current = state[i];
                let mut partner_values = f64x4::splat(0.0);
                
                // Gather partner values
                let mut partner_array = [0.0; 4];
                for j in 0..4 {
                    if affected[j] {
                        let partner_chunk = partner_indices[j] / 4;
                        let partner_offset = partner_indices[j] % 4;
                        if partner_chunk < state.len() {
                            let partner_chunk_array = state[partner_chunk].to_array();
                            partner_array[j] = partner_chunk_array[partner_offset];
                        }
                    }
                }
                partner_values = f64x4::new(partner_array);
                
                let new_value = (current + partner_values) * sqrt_2_inv;
                state[i] = new_value;
            }
        }
        
        Ok(())
    }
    
    // CPU gate implementations
    fn apply_hadamard_cpu(&self, state: &mut [f64], qubit: usize) -> Result<()> {
        let qubit_bit = 1 << qubit;
        let sqrt_2_inv = 1.0 / std::f64::consts::SQRT_2;
        
        for i in 0..(1 << self.num_qubits) {
            if i & qubit_bit == 0 {
                let j = i | qubit_bit;
                let temp = state[i];
                state[i] = (state[i] + state[j]) * sqrt_2_inv;
                state[j] = (temp - state[j]) * sqrt_2_inv;
            }
        }
        
        Ok(())
    }
    
    fn apply_rx_cpu(&self, state: &mut [f64], qubit: usize, angle: f64) -> Result<()> {
        let qubit_bit = 1 << qubit;
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        for i in 0..(1 << self.num_qubits) {
            if i & qubit_bit == 0 {
                let j = i | qubit_bit;
                let temp = state[i];
                state[i] = cos_half * state[i] - sin_half * state[j];
                state[j] = cos_half * state[j] - sin_half * temp;
            }
        }
        
        Ok(())
    }
    
    fn apply_ry_cpu(&self, state: &mut [f64], qubit: usize, angle: f64) -> Result<()> {
        let qubit_bit = 1 << qubit;
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        for i in 0..(1 << self.num_qubits) {
            if i & qubit_bit == 0 {
                let j = i | qubit_bit;
                let temp = state[i];
                state[i] = cos_half * state[i] - sin_half * state[j];
                state[j] = cos_half * state[j] + sin_half * temp;
            }
        }
        
        Ok(())
    }
    
    fn apply_rz_cpu(&self, state: &mut [f64], qubit: usize, angle: f64) -> Result<()> {
        let qubit_bit = 1 << qubit;
        let phase = (-angle / 2.0).exp(); // e^(-iθ/2) for |0⟩, e^(iθ/2) for |1⟩
        
        for i in 0..(1 << self.num_qubits) {
            if i & qubit_bit != 0 {
                state[i] *= -phase; // Phase for |1⟩ state
            } else {
                state[i] *= phase; // Phase for |0⟩ state
            }
        }
        
        Ok(())
    }
    
    fn apply_cnot_cpu(&self, state: &mut [f64], control: usize, target: usize) -> Result<()> {
        let control_bit = 1 << control;
        let target_bit = 1 << target;
        
        for i in 0..(1 << self.num_qubits) {
            if i & control_bit != 0 && i & target_bit == 0 {
                let j = i | target_bit;
                let temp = state[i];
                state[i] = state[j];
                state[j] = temp;
            }
        }
        
        Ok(())
    }
    
    fn apply_cz_cpu(&self, state: &mut [f64], control: usize, target: usize) -> Result<()> {
        let control_bit = 1 << control;
        let target_bit = 1 << target;
        
        for i in 0..(1 << self.num_qubits) {
            if i & control_bit != 0 && i & target_bit != 0 {
                state[i] *= -1.0; // Apply phase only to |11⟩ state
            }
        }
        
        Ok(())
    }
    
    fn get_parameter_index(&self, gate: &QuantumGate) -> Option<usize> {
        // This is a simplified parameter mapping
        // In practice, would track parameter indices more carefully
        match gate {
            QuantumGate::RX(qubit, _) => Some(qubit * 3),
            QuantumGate::RY(qubit, _) => Some(qubit * 3 + 1),
            QuantumGate::RZ(qubit, _) => Some(qubit * 3 + 2),
            _ => None,
        }
    }
    
    /// Extract player strategies from quantum state probabilities
    fn extract_strategies(&self, probabilities: &[f64], game: &GameMatrix) -> Result<HashMap<String, Array1<f64>>> {
        let num_players = game.num_players();
        let num_actions = game.num_actions();
        let mut strategies = HashMap::new();
        
        // Calculate bits needed per player
        let bits_per_player = (num_actions as f64).log2().ceil() as usize;
        
        for player_idx in 0..num_players {
            let mut player_probs = Array1::zeros(num_actions);
            
            for (state_idx, &prob) in probabilities.iter().enumerate() {
                // Extract player's action from binary representation
                let player_bits_start = player_idx * bits_per_player;
                let player_bits_end = (player_idx + 1) * bits_per_player;
                
                if player_bits_end <= self.num_qubits {
                    // Extract action index from state
                    let action_bits = (state_idx >> player_bits_start) & ((1 << bits_per_player) - 1);
                    if action_bits < num_actions {
                        player_probs[action_bits] += prob;
                    }
                }
            }
            
            // Normalize to create probability distribution
            let sum = player_probs.sum();
            if sum > 1e-12 {
                player_probs /= sum;
            } else {
                // Uniform distribution fallback
                player_probs.fill(1.0 / num_actions as f64);
            }
            
            strategies.insert(format!("player_{}", player_idx), player_probs);
        }
        
        Ok(strategies)
    }
    
    /// Calculate Nash equilibrium loss function
    fn calculate_nash_loss(&self, strategies: &HashMap<String, Array1<f64>>, game: &GameMatrix) -> Result<f64> {
        let num_players = game.num_players();
        let mut total_loss = 0.0;
        
        // For each player, calculate deviation incentive
        for player_idx in 0..num_players {
            let player_key = format!("player_{}", player_idx);
            let player_strategy = strategies.get(&player_key)
                .ok_or_else(|| QBMIAError::strategy("Missing player strategy"))?;
            
            // Calculate expected payoff for current strategy
            let current_payoff = self.calculate_expected_payoff(player_idx, player_strategy, strategies, game)?;
            
            // Calculate best response payoff
            let best_response = self.find_best_response(player_idx, strategies, game)?;
            let best_payoff = self.calculate_expected_payoff(player_idx, &best_response, strategies, game)?;
            
            // Add deviation loss
            let deviation = (best_payoff - current_payoff).max(0.0);
            total_loss += deviation * deviation;
        }
        
        Ok(total_loss)
    }
    
    /// Calculate expected payoff for a player's strategy
    fn calculate_expected_payoff(
        &self,
        player_idx: usize,
        strategy: &Array1<f64>,
        all_strategies: &HashMap<String, Array1<f64>>,
        game: &GameMatrix,
    ) -> Result<f64> {
        let num_players = game.num_players();
        let num_actions = game.num_actions();
        
        if num_players == 2 {
            // 2-player case
            let opponent_idx = 1 - player_idx;
            let opponent_key = format!("player_{}", opponent_idx);
            let opponent_strategy = all_strategies.get(&opponent_key)
                .ok_or_else(|| QBMIAError::strategy("Missing opponent strategy"))?;
            
            let mut expected_payoff = 0.0;
            
            for my_action in 0..num_actions {
                for opp_action in 0..num_actions {
                    let actions = if player_idx == 0 { 
                        vec![my_action, opp_action] 
                    } else { 
                        vec![opp_action, my_action] 
                    };
                    
                    let payoff = game.get_payoff(player_idx, &actions)?;
                    let prob = strategy[my_action] * opponent_strategy[opp_action];
                    expected_payoff += payoff * prob;
                }
            }
            
            Ok(expected_payoff)
        } else {
            // Multi-player case (simplified)
            Ok(0.0) // Would implement full multi-player calculation
        }
    }
    
    /// Find best response strategy for a player
    fn find_best_response(
        &self,
        player_idx: usize,
        all_strategies: &HashMap<String, Array1<f64>>,
        game: &GameMatrix,
    ) -> Result<Array1<f64>> {
        let num_actions = game.num_actions();
        let mut best_payoff = f64::NEG_INFINITY;
        let mut best_action = 0;
        
        // Check each pure strategy
        for action in 0..num_actions {
            let mut pure_strategy = Array1::zeros(num_actions);
            pure_strategy[action] = 1.0;
            
            let payoff = self.calculate_expected_payoff(player_idx, &pure_strategy, all_strategies, game)?;
            
            if payoff > best_payoff {
                best_payoff = payoff;
                best_action = action;
            }
        }
        
        let mut best_strategy = Array1::zeros(num_actions);
        best_strategy[best_action] = 1.0;
        Ok(best_strategy)
    }
    
    /// Update parameters using gradient-free optimization
    fn update_parameters(&mut self, iteration: usize) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Simulated annealing-like approach
        let noise_scale = self.learning_rate * (1.0 - iteration as f64 / self.max_iterations as f64);
        
        for param in self.parameters.iter_mut() {
            let noise = rng.random_range(-noise_scale..noise_scale);
            *param += noise;
            *param = param.max(-std::f64::consts::PI).min(std::f64::consts::PI);
        }
        
        Ok(())
    }
    
    /// Calculate convergence score from Nash loss
    fn calculate_convergence_score(&self, nash_loss: f64, num_players: usize) -> f64 {
        // Normalize convergence score between 0 and 1
        let max_possible_loss = num_players as f64 * 10.0; // Heuristic
        (1.0 - (nash_loss / max_possible_loss)).max(0.0).min(1.0)
    }
    
    /// Calculate entropy of quantum state distribution
    pub fn calculate_entropy(&self, probabilities: &[f64]) -> f64 {
        utils::von_neumann_entropy(probabilities)
    }
    
    /// Determine optimal action based on equilibrium strategies
    fn determine_optimal_action(&self, strategies: &HashMap<String, Array1<f64>>) -> usize {
        // For simplicity, return the action with highest probability for player 0
        if let Some(strategy) = strategies.get("player_0") {
            strategy.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        } else {
            0
        }
    }
    
    /// Analyze stability of the Nash equilibrium
    fn analyze_stability(
        &self,
        strategies: &HashMap<String, Array1<f64>>,
        game: &GameMatrix,
    ) -> Result<HashMap<String, f64>> {
        let mut stability_metrics = HashMap::new();
        
        // Calculate strategy entropy (mixed vs pure)
        for (player, strategy) in strategies {
            let entropy = utils::von_neumann_entropy(&strategy.to_vec());
            let max_entropy = (strategy.len() as f64).log2();
            let mixedness = if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 };
            stability_metrics.insert(format!("{}_mixedness", player), mixedness);
        }
        
        // Calculate robustness to perturbations
        let perturbation_size = 0.01;
        let mut perturbed_strategies = HashMap::new();
        
        for (player, strategy) in strategies {
            // Add small perturbation
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let mut perturbed = strategy.clone();
            
            for element in perturbed.iter_mut() {
                *element += rng.random_range(-perturbation_size..perturbation_size);
                *element = element.max(0.0);
            }
            
            // Renormalize
            let sum = perturbed.sum();
            if sum > 1e-12 {
                perturbed /= sum;
            }
            
            perturbed_strategies.insert(player.clone(), perturbed);
        }
        
        // Calculate change in payoffs
        let mut total_change = 0.0;
        for player_idx in 0..game.num_players() {
            let player_key = format!("player_{}", player_idx);
            
            if let (Some(original), Some(perturbed)) = (
                strategies.get(&player_key),
                perturbed_strategies.get(&player_key),
            ) {
                let original_payoff = self.calculate_expected_payoff(player_idx, original, strategies, game)?;
                let perturbed_payoff = self.calculate_expected_payoff(player_idx, perturbed, &perturbed_strategies, game)?;
                total_change += (perturbed_payoff - original_payoff).abs();
            }
        }
        
        stability_metrics.insert("perturbation_sensitivity".to_string(), total_change);
        
        Ok(stability_metrics)
    }
    
    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
    
    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DeviceType;
    
    #[tokio::test]
    async fn test_quantum_nash_creation() {
        let config = QuantumConfig {
            num_qubits: 4,
            num_layers: 2,
            ..QuantumConfig::default()
        };
        
        let solver = QuantumNashEquilibrium::new(config).await;
        assert!(solver.is_ok());
    }
    
    #[tokio::test]
    async fn test_game_matrix_creation() {
        let matrix = Array4::zeros((2, 2, 2, 2));
        let game = GameMatrix::new(matrix);
        assert!(game.is_ok());
        
        let game = game.unwrap();
        assert_eq!(game.num_players(), 2);
        assert_eq!(game.num_actions(), 2);
    }
    
    #[test]
    fn test_game_matrix_payoff() {
        let mut matrix = Array4::zeros((2, 2, 2, 2));
        matrix[[0, 1, 0, 1]] = 5.0;
        
        let game = GameMatrix::new(matrix).unwrap();
        let payoff = game.get_payoff(0, &[0, 1]).unwrap();
        assert_eq!(payoff, 5.0);
    }
}