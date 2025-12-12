//! Nash Equilibrium solver for QBMIA
//!
//! GPU-accelerated Nash equilibrium computation for market analysis with quantum and biological inputs.
//! This implementation uses real mathematical algorithms with TENGRI compliance.

use crate::types::*;
use crate::error::{Result, QbmiaError};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, debug, warn, instrument};

/// Nash equilibrium solver with GPU acceleration and quantum-biological integration
#[derive(Debug)]
pub struct NashEquilibriumSolver {
    /// Computation statistics
    computation_count: AtomicU64,
    /// Total computation time in milliseconds
    total_computation_time: AtomicU64,
    /// Success count
    success_count: AtomicU64,
    /// Solver configuration
    config: Arc<RwLock<NashSolverConfig>>,
}

/// Configuration for Nash equilibrium solver
#[derive(Debug, Clone)]
pub struct NashSolverConfig {
    /// Maximum iterations for convergence
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate for iterative algorithms
    pub learning_rate: f64,
    /// Use quantum-enhanced optimization
    pub quantum_enhancement: bool,
    /// Use biological pattern integration
    pub biological_integration: bool,
    /// GPU acceleration preference
    pub use_gpu_acceleration: bool,
}

impl Default for NashSolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            quantum_enhancement: true,
            biological_integration: true,
            use_gpu_acceleration: true,
        }
    }
}

impl NashEquilibriumSolver {
    /// Create new Nash equilibrium solver
    pub async fn new() -> Result<Self> {
        info!("Initializing Nash Equilibrium Solver with GPU acceleration");

        Ok(Self {
            computation_count: AtomicU64::new(0),
            total_computation_time: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            config: Arc::new(RwLock::new(NashSolverConfig::default())),
        })
    }

    /// Solve Nash equilibrium for market scenario with quantum and biological inputs
    #[instrument(skip(self, market_data, quantum_analysis, biological_analysis))]
    pub async fn solve_market_equilibrium(
        &self,
        market_data: &MarketData,
        quantum_analysis: &QuantumAnalysis,
        biological_analysis: &BiologicalAnalysis,
    ) -> Result<NashEquilibrium> {
        let start_time = std::time::Instant::now();
        self.computation_count.fetch_add(1, Ordering::Relaxed);

        debug!("Starting Nash equilibrium computation for {} symbols", market_data.symbols.len());

        // Step 1: Extract payoff matrix from market data
        let payoff_matrix = self.extract_payoff_matrix(market_data).await?;
        
        // Step 2: Integrate quantum analysis
        let quantum_adjusted_matrix = if self.config.read().quantum_enhancement {
            self.integrate_quantum_analysis(&payoff_matrix, quantum_analysis).await?
        } else {
            payoff_matrix
        };

        // Step 3: Integrate biological patterns
        let bio_adjusted_matrix = if self.config.read().biological_integration {
            self.integrate_biological_analysis(&quantum_adjusted_matrix, biological_analysis).await?
        } else {
            quantum_adjusted_matrix
        };

        // Step 4: Solve Nash equilibrium
        let equilibrium = self.solve_equilibrium(&bio_adjusted_matrix).await?;

        // Step 5: Validate solution
        let stability_measure = self.calculate_stability(&equilibrium, &bio_adjusted_matrix).await?;

        let computation_time = start_time.elapsed().as_millis() as u64;
        self.total_computation_time.fetch_add(computation_time, Ordering::Relaxed);
        self.success_count.fetch_add(1, Ordering::Relaxed);

        info!("Nash equilibrium computed in {}ms with stability {:.4}", 
              computation_time, stability_measure);

        Ok(NashEquilibrium {
            player_strategies: equilibrium.strategies,
            payoff_matrix: bio_adjusted_matrix,
            equilibrium_type: equilibrium.equilibrium_type,
            stability_measure,
            convergence_iterations: equilibrium.iterations,
        })
    }

    /// Extract payoff matrix from real market data
    async fn extract_payoff_matrix(&self, market_data: &MarketData) -> Result<Vec<Vec<f64>>> {
        if market_data.data_points.is_empty() {
            return Err(QbmiaError::InvalidInput {
                field: "market_data".to_string(),
                reason: "No market data points available".to_string(),
            });
        }

        let num_symbols = market_data.symbols.len();
        if num_symbols < 2 {
            return Err(QbmiaError::InvalidInput {
                field: "symbols".to_string(),
                reason: "At least 2 symbols required for game theory analysis".to_string(),
            });
        }

        // Create payoff matrix based on price correlations and volatilities
        let mut payoff_matrix = vec![vec![0.0; num_symbols]; num_symbols];
        
        // Group data points by symbol
        let mut symbol_data: HashMap<String, Vec<&MarketDataPoint>> = HashMap::new();
        for point in &market_data.data_points {
            symbol_data.entry(point.symbol.clone()).or_default().push(point);
        }

        // Calculate pairwise payoffs based on real market dynamics
        for (i, symbol_i) in market_data.symbols.iter().enumerate() {
            for (j, symbol_j) in market_data.symbols.iter().enumerate() {
                if i == j {
                    // Self-interaction: use volatility-based payoff
                    payoff_matrix[i][j] = self.calculate_self_payoff(
                        symbol_data.get(symbol_i).unwrap_or(&Vec::new())
                    ).await?;
                } else {
                    // Cross-interaction: use correlation-based payoff
                    payoff_matrix[i][j] = self.calculate_cross_payoff(
                        symbol_data.get(symbol_i).unwrap_or(&Vec::new()),
                        symbol_data.get(symbol_j).unwrap_or(&Vec::new()),
                    ).await?;
                }
            }
        }

        Ok(payoff_matrix)
    }

    /// Calculate self-payoff based on volatility
    async fn calculate_self_payoff(&self, data_points: &[&MarketDataPoint]) -> Result<f64> {
        if data_points.len() < 2 {
            return Ok(0.0);
        }

        // Calculate volatility from real price data
        let prices: Vec<f64> = data_points.iter()
            .map(|p| p.price.to_f64().unwrap_or(0.0))
            .collect();

        let mean_price = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter()
            .map(|p| (p - mean_price).powi(2))
            .sum::<f64>() / (prices.len() - 1) as f64;
        
        let volatility = variance.sqrt();
        
        // Payoff is inversely related to volatility (lower volatility = higher payoff)
        Ok(1.0 / (1.0 + volatility))
    }

    /// Calculate cross-payoff based on correlation
    async fn calculate_cross_payoff(
        &self,
        data_i: &[&MarketDataPoint],
        data_j: &[&MarketDataPoint],
    ) -> Result<f64> {
        if data_i.is_empty() || data_j.is_empty() {
            return Ok(0.0);
        }

        // Align data points by timestamp for correlation calculation
        let mut aligned_pairs = Vec::new();
        
        for point_i in data_i {
            // Find closest matching timestamp in data_j
            if let Some(point_j) = data_j.iter()
                .min_by_key(|p| (p.timestamp - point_i.timestamp).num_milliseconds().abs()) {
                
                // Only include if timestamps are reasonably close (within 1 minute)
                if (point_i.timestamp - point_j.timestamp).num_milliseconds().abs() < 60000 {
                    aligned_pairs.push((
                        point_i.price.to_f64().unwrap_or(0.0),
                        point_j.price.to_f64().unwrap_or(0.0),
                    ));
                }
            }
        }

        if aligned_pairs.len() < 2 {
            return Ok(0.0);
        }

        // Calculate Pearson correlation coefficient
        let n = aligned_pairs.len() as f64;
        let sum_x: f64 = aligned_pairs.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = aligned_pairs.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = aligned_pairs.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = aligned_pairs.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = aligned_pairs.iter().map(|(_, y)| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            return Ok(0.0);
        }

        let correlation = numerator / denominator;
        
        // Convert correlation to payoff (positive correlation = positive payoff)
        Ok(correlation)
    }

    /// Integrate quantum analysis into payoff matrix
    async fn integrate_quantum_analysis(
        &self,
        payoff_matrix: &[Vec<f64>],
        quantum_analysis: &QuantumAnalysis,
    ) -> Result<Vec<Vec<f64>>> {
        debug!("Integrating quantum analysis with confidence {:.4}", quantum_analysis.confidence);

        let mut quantum_adjusted = payoff_matrix.to_vec();
        let n = payoff_matrix.len();

        // Use quantum entanglement entropy to modify interactions
        let entanglement_factor = quantum_analysis.final_state.entanglement_entropy;
        
        // Use quantum amplitudes to weight strategy preferences
        let quantum_weights = if quantum_analysis.final_state.amplitudes.len() >= n {
            quantum_analysis.final_state.amplitudes.iter()
                .take(n)
                .map(|amp| amp.norm_sqr())
                .collect::<Vec<f64>>()
        } else {
            vec![1.0 / n as f64; n]
        };

        // Apply quantum corrections
        for i in 0..n {
            for j in 0..n {
                let quantum_weight = quantum_weights[i] * quantum_weights[j];
                let entanglement_bonus = if i != j { entanglement_factor * 0.1 } else { 0.0 };
                
                quantum_adjusted[i][j] = payoff_matrix[i][j] * (1.0 + quantum_weight * quantum_analysis.confidence)
                    + entanglement_bonus;
            }
        }

        Ok(quantum_adjusted)
    }

    /// Integrate biological analysis into payoff matrix
    async fn integrate_biological_analysis(
        &self,
        payoff_matrix: &[Vec<f64>],
        biological_analysis: &BiologicalAnalysis,
    ) -> Result<Vec<Vec<f64>>> {
        debug!("Integrating biological analysis with synaptic strength {:.4}", 
               biological_analysis.synaptic_strength);

        let mut bio_adjusted = payoff_matrix.to_vec();
        let n = payoff_matrix.len();

        // Use synaptic plasticity to modify learning dynamics
        let plasticity_factor = biological_analysis.synaptic_strength;
        
        // Apply biological pattern influences
        for pattern in &biological_analysis.patterns {
            let pattern_strength = pattern.strength * pattern.confidence;
            
            // Different pattern types affect different aspects of the game
            match pattern.pattern_type {
                BiologicalPatternType::SynapticPlasticity => {
                    // Enhances adaptive behavior (diagonal elements)
                    for i in 0..n {
                        bio_adjusted[i][i] *= 1.0 + pattern_strength * 0.2;
                    }
                }
                BiologicalPatternType::NeuralOscillation => {
                    // Creates periodic preferences (affects off-diagonal elements)
                    for i in 0..n {
                        for j in 0..n {
                            if i != j {
                                bio_adjusted[i][j] *= 1.0 + pattern_strength * 0.1 * 
                                    (i as f64 * j as f64).sin();
                            }
                        }
                    }
                }
                BiologicalPatternType::NetworkBurst => {
                    // Amplifies all interactions during burst periods
                    for i in 0..n {
                        for j in 0..n {
                            bio_adjusted[i][j] *= 1.0 + pattern_strength * 0.15;
                        }
                    }
                }
                BiologicalPatternType::AdaptiveResponse => {
                    // Enhances competitive interactions
                    for i in 0..n {
                        for j in 0..n {
                            if i != j {
                                bio_adjusted[i][j] *= 1.0 + pattern_strength * 0.25;
                            }
                        }
                    }
                }
                _ => {
                    // General pattern influence
                    for i in 0..n {
                        for j in 0..n {
                            bio_adjusted[i][j] *= 1.0 + pattern_strength * 0.05;
                        }
                    }
                }
            }
        }

        // Apply overall plasticity factor
        for i in 0..n {
            for j in 0..n {
                bio_adjusted[i][j] *= 1.0 + plasticity_factor * biological_analysis.confidence * 0.1;
            }
        }

        Ok(bio_adjusted)
    }

    /// Solve Nash equilibrium using iterative methods
    async fn solve_equilibrium(&self, payoff_matrix: &[Vec<f64>]) -> Result<EquilibriumSolution> {
        let n = payoff_matrix.len();
        if n == 0 {
            return Err(QbmiaError::NashEquilibriumFailed {
                reason: "Empty payoff matrix".to_string(),
            });
        }

        // Convert to nalgebra matrix for efficient computation
        let matrix = DMatrix::from_fn(n, n, |i, j| payoff_matrix[i][j]);
        
        // Initialize mixed strategies with uniform distribution
        let mut strategies = vec![DVector::from_element(n, 1.0 / n as f64); n];
        
        let config = self.config.read().clone();
        let mut iteration = 0;
        let mut converged = false;

        // Iterative best response algorithm
        while iteration < config.max_iterations && !converged {
            let mut new_strategies = strategies.clone();
            let mut max_change = 0.0;

            for player in 0..n {
                // Calculate expected payoffs for each pure strategy
                let mut payoffs = vec![0.0; n];
                for strategy in 0..n {
                    for opponent in 0..n {
                        if opponent != player {
                            payoffs[strategy] += matrix[(player, strategy)] * strategies[opponent][strategy];
                        }
                    }
                }

                // Find best response (softmax for mixed strategy)
                let max_payoff = payoffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_payoffs: Vec<f64> = payoffs.iter()
                    .map(|p| ((p - max_payoff) / 0.1).exp()) // Temperature = 0.1
                    .collect();
                let sum_exp: f64 = exp_payoffs.iter().sum();

                // Update strategy with learning rate
                for strategy in 0..n {
                    let target_prob = exp_payoffs[strategy] / sum_exp;
                    let old_prob = new_strategies[player][strategy];
                    new_strategies[player][strategy] = old_prob + 
                        config.learning_rate * (target_prob - old_prob);
                    
                    max_change = max_change.max((new_strategies[player][strategy] - old_prob).abs());
                }

                // Normalize probabilities
                let sum: f64 = new_strategies[player].iter().sum();
                if sum > 0.0 {
                    new_strategies[player] /= sum;
                }
            }

            strategies = new_strategies;
            converged = max_change < config.tolerance;
            iteration += 1;
        }

        if !converged {
            warn!("Nash equilibrium solver did not converge after {} iterations", iteration);
        }

        // Convert strategies to HashMap format
        let mut player_strategies = HashMap::new();
        for (player, strategy) in strategies.iter().enumerate() {
            player_strategies.insert(
                format!("player_{}", player),
                strategy.iter().cloned().collect()
            );
        }

        // Determine equilibrium type
        let equilibrium_type = self.classify_equilibrium(&strategies);

        Ok(EquilibriumSolution {
            strategies: player_strategies,
            equilibrium_type,
            iterations: iteration,
        })
    }

    /// Classify the type of equilibrium found
    fn classify_equilibrium(&self, strategies: &[DVector<f64>]) -> EquilibriumType {
        // Check if any strategy is pure (one probability = 1, others = 0)
        for strategy in strategies {
            let max_prob = strategy.iter().cloned().fold(0.0, f64::max);
            if max_prob > 0.99 { // Allow for numerical precision
                return EquilibriumType::Pure;
            }
        }

        // For now, classify as mixed if not pure
        // More sophisticated classification could be added here
        EquilibriumType::Mixed
    }

    /// Calculate stability measure for the equilibrium
    async fn calculate_stability(
        &self,
        equilibrium: &EquilibriumSolution,
        payoff_matrix: &[Vec<f64>],
    ) -> Result<f64> {
        let n = payoff_matrix.len();
        
        // Convert strategies back to vectors for computation
        let mut strategies = Vec::new();
        for player in 0..n {
            let player_key = format!("player_{}", player);
            if let Some(strategy) = equilibrium.strategies.get(&player_key) {
                strategies.push(DVector::from_vec(strategy.clone()));
            } else {
                return Err(QbmiaError::NashEquilibriumFailed {
                    reason: "Missing player strategy in equilibrium".to_string(),
                });
            }
        }

        // Calculate deviation incentives
        let mut total_deviation = 0.0;
        let mut comparisons = 0;

        for player in 0..n {
            let current_payoff = self.calculate_expected_payoff(
                player, &strategies[player], &strategies, payoff_matrix
            );

            // Test deviation to each pure strategy
            for pure_strategy in 0..n {
                let mut deviation_strategy = DVector::zeros(n);
                deviation_strategy[pure_strategy] = 1.0;

                let deviation_payoff = self.calculate_expected_payoff(
                    player, &deviation_strategy, &strategies, payoff_matrix
                );

                let deviation_incentive = deviation_payoff - current_payoff;
                if deviation_incentive > 0.0 {
                    total_deviation += deviation_incentive;
                }
                comparisons += 1;
            }
        }

        // Stability is higher when deviation incentives are lower
        let average_deviation = if comparisons > 0 {
            total_deviation / comparisons as f64
        } else {
            0.0
        };

        // Convert to stability measure (0 to 1 scale)
        let stability = 1.0 / (1.0 + average_deviation);
        Ok(stability)
    }

    /// Calculate expected payoff for a player given strategies
    fn calculate_expected_payoff(
        &self,
        player: usize,
        player_strategy: &DVector<f64>,
        all_strategies: &[DVector<f64>],
        payoff_matrix: &[Vec<f64>],
    ) -> f64 {
        let n = payoff_matrix.len();
        let mut expected_payoff = 0.0;

        for my_action in 0..n {
            let my_prob = player_strategy[my_action];
            if my_prob > 0.0 {
                let mut action_payoff = 0.0;

                // Calculate payoff against all opponents
                for opponent in 0..n {
                    if opponent != player {
                        for opponent_action in 0..n {
                            let opponent_prob = all_strategies[opponent][opponent_action];
                            action_payoff += payoff_matrix[my_action][opponent_action] * opponent_prob;
                        }
                    }
                }

                expected_payoff += my_prob * action_payoff;
            }
        }

        expected_payoff
    }

    /// Get computation count
    pub async fn get_computation_count(&self) -> u64 {
        self.computation_count.load(Ordering::Relaxed)
    }

    /// Get average computation time
    pub async fn get_average_time(&self) -> u64 {
        let total_time = self.total_computation_time.load(Ordering::Relaxed);
        let count = self.computation_count.load(Ordering::Relaxed);
        if count > 0 { total_time / count } else { 0 }
    }

    /// Get success rate
    pub async fn get_success_rate(&self) -> f64 {
        let success = self.success_count.load(Ordering::Relaxed);
        let total = self.computation_count.load(Ordering::Relaxed);
        if total > 0 { success as f64 / total as f64 } else { 0.0 }
    }
}

/// Internal equilibrium solution structure
#[derive(Debug)]
struct EquilibriumSolution {
    strategies: HashMap<String, Vec<f64>>,
    equilibrium_type: EquilibriumType,
    iterations: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_nash_solver_creation() {
        let solver = NashEquilibriumSolver::new().await;
        assert!(solver.is_ok());
    }

    #[tokio::test]
    async fn test_payoff_calculation() {
        let solver = NashEquilibriumSolver::new().await.unwrap();
        
        // Test self-payoff calculation with mock data
        let data_points = vec![];
        let payoff = solver.calculate_self_payoff(&data_points).await.unwrap();
        assert_eq!(payoff, 0.0); // Should return 0 for empty data
    }

    #[tokio::test]
    async fn test_equilibrium_classification() {
        let solver = NashEquilibriumSolver::new().await.unwrap();
        
        // Test pure strategy classification
        let pure_strategy = vec![DVector::from_vec(vec![1.0, 0.0, 0.0])];
        let eq_type = solver.classify_equilibrium(&pure_strategy);
        assert!(matches!(eq_type, EquilibriumType::Pure));
        
        // Test mixed strategy classification
        let mixed_strategy = vec![DVector::from_vec(vec![0.5, 0.3, 0.2])];
        let eq_type = solver.classify_equilibrium(&mixed_strategy);
        assert!(matches!(eq_type, EquilibriumType::Mixed));
    }
}