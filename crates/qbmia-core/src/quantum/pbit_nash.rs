//! pBit-based Quantum Nash Equilibrium Solver
//!
//! Uses pBit probabilistic computing for finding Nash equilibria
//! in market games via simulated annealing and Boltzmann dynamics.
//!
//! ## Key Mappings
//!
//! - Quantum superposition → pBit probability distribution
//! - Variational circuit → pBit annealing with adaptive biases
//! - Nash loss → Ising Hamiltonian energy
//! - Strategy extraction → Boltzmann sampling

use crate::{
    config::QuantumConfig,
    error::{QBMIAError, Result},
    quantum::{GameMatrix, QuantumMetrics, utils},
};
use quantum_core::{PBitState, PBitConfig, PBitCoupling};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// pBit-based Nash Equilibrium result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBitNashResult {
    /// Equilibrium strategies for each player
    pub strategies: HashMap<String, Array1<f64>>,
    /// Convergence score [0, 1]
    pub convergence_score: f64,
    /// Nash loss value
    pub nash_loss: f64,
    /// Number of annealing iterations
    pub iterations: usize,
    /// Convergence history
    pub convergence_history: Vec<f64>,
    /// pBit state entropy
    pub entropy: f64,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Optimal action for player 0
    pub optimal_action: usize,
    /// Final temperature
    pub final_temperature: f64,
}

/// pBit-based Nash Equilibrium solver
pub struct PBitNashEquilibrium {
    /// Number of pBits per player
    pbits_per_player: usize,
    /// Number of players
    num_players: usize,
    /// pBit state
    pbit_state: PBitState,
    /// Learning rate for bias updates
    learning_rate: f64,
    /// Convergence threshold
    convergence_threshold: f64,
    /// Maximum iterations
    max_iterations: usize,
    /// Initial temperature
    initial_temperature: f64,
    /// Final temperature
    final_temperature: f64,
    /// Sweeps per iteration
    sweeps_per_iteration: usize,
}

impl PBitNashEquilibrium {
    /// Create a new pBit Nash equilibrium solver
    pub fn new(config: &QuantumConfig) -> Result<Self> {
        let num_qubits = config.num_qubits.max(4);
        let num_players = 2; // Default to 2-player games
        let pbits_per_player = num_qubits / num_players;
        
        let pbit_config = PBitConfig {
            temperature: 10.0,
            coupling_strength: 1.0,
            external_field: 0.0,
            seed: None,
        };
        
        let total_pbits = num_players * pbits_per_player;
        let pbit_state = PBitState::with_config(total_pbits, pbit_config)
            .map_err(|e| QBMIAError::quantum_simulation(e.to_string()))?;
        
        Ok(Self {
            pbits_per_player,
            num_players,
            pbit_state,
            learning_rate: config.learning_rate,
            convergence_threshold: config.convergence_threshold,
            max_iterations: config.max_iterations,
            initial_temperature: 10.0,
            final_temperature: 0.1,
            sweeps_per_iteration: 10,
        })
    }
    
    /// Find Nash equilibrium using pBit annealing
    pub fn find_equilibrium(
        &mut self,
        game: &GameMatrix,
        market_conditions: Option<&HashMap<String, f64>>,
    ) -> Result<PBitNashResult> {
        let start_time = Instant::now();
        
        // Initialize pBit state based on game structure
        self.initialize_from_game(game, market_conditions)?;
        
        let mut best_loss = f64::INFINITY;
        let mut best_strategies = HashMap::new();
        let mut convergence_history = Vec::new();
        
        // Temperature schedule
        let cooling_rate = (self.final_temperature / self.initial_temperature)
            .powf(1.0 / self.max_iterations as f64);
        let mut temperature = self.initial_temperature;
        
        for iteration in 0..self.max_iterations {
            // Update pBit biases based on game payoffs
            self.update_biases_from_game(game, temperature)?;
            
            // Perform pBit sweeps (equilibration)
            for _ in 0..self.sweeps_per_iteration {
                self.pbit_state.sweep();
            }
            
            // Extract strategies from pBit state
            let strategies = self.extract_strategies(game)?;
            
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
                break;
            }
            
            // Cool down
            temperature *= cooling_rate;
        }
        
        let execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let entropy = self.pbit_state.entropy();
        let convergence_score = self.calculate_convergence_score(best_loss, game.num_players());
        let optimal_action = self.determine_optimal_action(&best_strategies);
        
        Ok(PBitNashResult {
            strategies: best_strategies,
            convergence_score,
            nash_loss: best_loss,
            iterations: convergence_history.len(),
            convergence_history,
            entropy,
            execution_time_ms,
            optimal_action,
            final_temperature: temperature,
        })
    }
    
    /// Initialize pBit state from game structure
    fn initialize_from_game(
        &mut self,
        game: &GameMatrix,
        market_conditions: Option<&HashMap<String, f64>>,
    ) -> Result<()> {
        let num_actions = game.num_actions();
        
        // Initialize all pBits to equal superposition
        for i in 0..self.pbit_state.num_qubits() {
            if let Some(pbit) = self.pbit_state.get_pbit_mut(i) {
                pbit.probability_up = 0.5;
                pbit.bias = 0.0;
            }
        }
        
        // Add couplings within each player's action space
        for player in 0..self.num_players {
            let start_idx = player * self.pbits_per_player;
            
            // Couplings within player's pBits (represent correlated actions)
            for i in 0..self.pbits_per_player.saturating_sub(1) {
                let idx1 = start_idx + i;
                let idx2 = start_idx + i + 1;
                self.pbit_state.add_coupling(
                    PBitCoupling::bell_coupling(idx1, idx2, 0.3)
                );
            }
        }
        
        // Add cross-player couplings (competitive dynamics)
        if self.num_players >= 2 {
            for i in 0..self.pbits_per_player {
                let p1_idx = i;
                let p2_idx = self.pbits_per_player + i;
                
                if p2_idx < self.pbit_state.num_qubits() {
                    // Anti-ferromagnetic coupling for competitive games
                    self.pbit_state.add_coupling(
                        PBitCoupling::anti_bell_coupling(p1_idx, p2_idx, 0.5)
                    );
                }
            }
        }
        
        // Apply market condition biases
        if let Some(conditions) = market_conditions {
            if let Some(&volatility) = conditions.get("volatility") {
                // Higher volatility → more exploration (higher effective temperature)
                for i in 0..self.pbit_state.num_qubits() {
                    if let Some(pbit) = self.pbit_state.get_pbit_mut(i) {
                        pbit.bias *= 1.0 - volatility.clamp(0.0, 0.5);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Update pBit biases based on game payoffs
    fn update_biases_from_game(&mut self, game: &GameMatrix, temperature: f64) -> Result<()> {
        let num_actions = game.num_actions();
        
        for player in 0..self.num_players.min(game.num_players()) {
            let start_idx = player * self.pbits_per_player;
            
            // Calculate expected payoffs for each action
            for action in 0..num_actions.min(self.pbits_per_player) {
                let mut expected_payoff = 0.0;
                
                // Average payoff against uniform opponent
                for opp_action in 0..num_actions {
                    let actions = if player == 0 {
                        vec![action, opp_action]
                    } else {
                        vec![opp_action, action]
                    };
                    
                    if let Ok(payoff) = game.get_payoff(player, &actions) {
                        expected_payoff += payoff / num_actions as f64;
                    }
                }
                
                // Set bias based on expected payoff (normalized by temperature)
                let pbit_idx = start_idx + action;
                if let Some(pbit) = self.pbit_state.get_pbit_mut(pbit_idx) {
                    // Higher payoff → higher probability of selecting this action
                    pbit.bias = expected_payoff / temperature;
                }
            }
        }
        
        Ok(())
    }
    
    /// Extract strategies from pBit state
    fn extract_strategies(&self, game: &GameMatrix) -> Result<HashMap<String, Array1<f64>>> {
        let mut strategies = HashMap::new();
        let num_actions = game.num_actions();
        
        for player in 0..self.num_players.min(game.num_players()) {
            let start_idx = player * self.pbits_per_player;
            let mut action_probs = Array1::zeros(num_actions);
            
            // Convert pBit probabilities to action probabilities
            for action in 0..num_actions.min(self.pbits_per_player) {
                if let Some(pbit) = self.pbit_state.get_pbit(start_idx + action) {
                    action_probs[action] = pbit.probability_up;
                }
            }
            
            // Normalize to valid probability distribution
            let sum: f64 = action_probs.sum();
            if sum > 1e-10 {
                action_probs /= sum;
            } else {
                // Uniform distribution if all zero
                action_probs.fill(1.0 / num_actions as f64);
            }
            
            strategies.insert(format!("player_{}", player), action_probs);
        }
        
        Ok(strategies)
    }
    
    /// Calculate Nash loss (deviation incentive)
    fn calculate_nash_loss(
        &self,
        strategies: &HashMap<String, Array1<f64>>,
        game: &GameMatrix,
    ) -> Result<f64> {
        let mut total_loss = 0.0;
        let num_actions = game.num_actions();
        
        for player in 0..self.num_players.min(game.num_players()) {
            let player_key = format!("player_{}", player);
            let strategy = strategies.get(&player_key)
                .ok_or_else(|| QBMIAError::quantum_simulation("Missing strategy"))?;
            
            // Calculate expected payoff under current strategy
            let mut current_payoff = 0.0;
            let mut best_deviation_payoff = f64::NEG_INFINITY;
            
            for action in 0..num_actions {
                // Expected payoff for this action against opponent's strategy
                let mut action_payoff = 0.0;
                
                let opp_key = format!("player_{}", 1 - player);
                let opp_strategy = strategies.get(&opp_key)
                    .ok_or_else(|| QBMIAError::quantum_simulation("Missing opponent strategy"))?;
                
                for opp_action in 0..num_actions {
                    let actions = if player == 0 {
                        vec![action, opp_action]
                    } else {
                        vec![opp_action, action]
                    };
                    
                    if let Ok(payoff) = game.get_payoff(player, &actions) {
                        action_payoff += payoff * opp_strategy[opp_action];
                    }
                }
                
                current_payoff += action_payoff * strategy[action];
                best_deviation_payoff = best_deviation_payoff.max(action_payoff);
            }
            
            // Nash loss = potential gain from deviating
            let deviation_gain = (best_deviation_payoff - current_payoff).max(0.0);
            total_loss += deviation_gain;
        }
        
        Ok(total_loss)
    }
    
    /// Calculate convergence score
    fn calculate_convergence_score(&self, loss: f64, num_players: usize) -> f64 {
        let max_loss = num_players as f64; // Rough upper bound
        (1.0 - (loss / max_loss).min(1.0)).max(0.0)
    }
    
    /// Determine optimal action from strategies
    fn determine_optimal_action(&self, strategies: &HashMap<String, Array1<f64>>) -> usize {
        if let Some(strategy) = strategies.get("player_0") {
            strategy.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;
    
    fn create_test_config() -> QuantumConfig {
        QuantumConfig {
            num_qubits: 4,
            num_layers: 2,
            learning_rate: 0.01,
            convergence_threshold: 0.1,
            max_iterations: 100,
            ..Default::default()
        }
    }
    
    fn create_prisoners_dilemma() -> GameMatrix {
        // Prisoner's dilemma payoff matrix
        // Actions: 0 = Cooperate, 1 = Defect
        let mut payoffs = Array4::zeros((2, 2, 2, 2));
        
        // Player 0 payoffs
        payoffs[[0, 1, 0, 0]] = 3.0; // Both cooperate
        payoffs[[0, 1, 0, 1]] = 0.0; // P0 cooperates, P1 defects
        payoffs[[0, 1, 1, 0]] = 5.0; // P0 defects, P1 cooperates
        payoffs[[0, 1, 1, 1]] = 1.0; // Both defect
        
        // Player 1 payoffs (symmetric)
        payoffs[[1, 0, 0, 0]] = 3.0;
        payoffs[[1, 0, 1, 0]] = 0.0;
        payoffs[[1, 0, 0, 1]] = 5.0;
        payoffs[[1, 0, 1, 1]] = 1.0;
        
        GameMatrix::new(payoffs).unwrap()
    }
    
    #[test]
    fn test_pbit_nash_creation() {
        let config = create_test_config();
        let solver = PBitNashEquilibrium::new(&config);
        assert!(solver.is_ok());
    }
    
    #[test]
    fn test_find_equilibrium() {
        let config = create_test_config();
        let mut solver = PBitNashEquilibrium::new(&config).unwrap();
        let game = create_prisoners_dilemma();
        
        let result = solver.find_equilibrium(&game, None);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.strategies.contains_key("player_0"));
        assert!(result.strategies.contains_key("player_1"));
        assert!(result.nash_loss >= 0.0);
    }
}
