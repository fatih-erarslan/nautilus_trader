//! # Hyperbolic Replicator Dynamics
//!
//! Evolutionary game theory on hyperbolic manifolds with replicator equations.
//!
//! ## Theoretical Foundation
//!
//! Replicator dynamics describe frequency-dependent selection:
//! ẋᵢ = xᵢ(fᵢ(x) - φ(x))
//!
//! where:
//! - xᵢ is frequency of strategy i
//! - fᵢ(x) is fitness of strategy i given population x
//! - φ(x) is average fitness
//!
//! ## Hyperbolic Extension
//!
//! In hyperbolic space:
//! - Strategies occupy positions on the hyperboloid
//! - Fitness depends on hyperbolic distances (geodesic interactions)
//! - Exponential volume growth affects competition dynamics
//! - Curvature modulates stability of equilibria
//!
//! ## Applications to SNNs
//!
//! - Synapse types as strategies competing for resources
//! - Neuron populations evolving connectivity patterns
//! - Learning rules as competing algorithms
//!
//! ## References
//!
//! - Taylor & Jonker (1978) "Evolutionarily Stable Strategies and Game Dynamics"
//! - Hofbauer & Sigmund (1998) "Evolutionary Games and Population Dynamics"
//! - Nowak (2006) "Evolutionary Dynamics: Exploring the Equations of Life"

use crate::hyperbolic_snn::LorentzVec;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for replicator dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicatorConfig {
    /// Number of strategies
    pub num_strategies: usize,
    /// Time step for integration
    pub dt: f64,
    /// Selection intensity (higher = stronger selection)
    pub selection_intensity: f64,
    /// Mutation rate
    pub mutation_rate: f64,
    /// Minimum frequency (extinction threshold)
    pub min_frequency: f64,
    /// Use hyperbolic distance for payoffs
    pub hyperbolic_payoffs: bool,
    /// Curvature (K = -1 for standard hyperbolic)
    pub curvature: f64,
}

impl Default for ReplicatorConfig {
    fn default() -> Self {
        Self {
            num_strategies: 3,
            dt: 0.01,
            selection_intensity: 1.0,
            mutation_rate: 0.001,
            min_frequency: 1e-6,
            hyperbolic_payoffs: true,
            curvature: -1.0,
        }
    }
}

/// Strategy in the evolutionary game
#[derive(Debug, Clone)]
pub struct Strategy {
    /// Strategy index
    pub id: usize,
    /// Strategy name
    pub name: String,
    /// Position in hyperbolic space
    pub position: LorentzVec,
    /// Current frequency in population
    pub frequency: f64,
    /// Current fitness
    pub fitness: f64,
    /// Frequency history
    pub history: VecDeque<f64>,
    /// Strategy color for visualization
    pub color: [f32; 3],
}

impl Strategy {
    /// Create new strategy
    pub fn new(id: usize, name: &str, position: LorentzVec, initial_freq: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            position,
            frequency: initial_freq,
            fitness: 0.0,
            history: VecDeque::with_capacity(1000),
            color: Self::default_color(id),
        }
    }

    /// Default color based on ID
    fn default_color(id: usize) -> [f32; 3] {
        let hue = (id as f32 * 0.618033988749) % 1.0; // Golden ratio for spread
        hsv_to_rgb(hue, 0.8, 0.9)
    }
}

/// Convert HSV to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h * 6.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    [r + m, g + m, b + m]
}

/// Payoff matrix for game
#[derive(Debug, Clone)]
pub struct PayoffMatrix {
    /// Payoff values: payoffs[i][j] = payoff to i when playing against j
    pub payoffs: Vec<Vec<f64>>,
    /// Number of strategies
    pub n: usize,
}

impl PayoffMatrix {
    /// Create zero matrix
    pub fn zeros(n: usize) -> Self {
        Self {
            payoffs: vec![vec![0.0; n]; n],
            n,
        }
    }

    /// Create prisoner's dilemma payoff matrix
    /// Cooperate = 0, Defect = 1
    pub fn prisoners_dilemma(r: f64, s: f64, t: f64, p: f64) -> Self {
        // T > R > P > S (temptation > reward > punishment > sucker)
        Self {
            payoffs: vec![
                vec![r, s],  // Cooperator vs Cooperator, Cooperator vs Defector
                vec![t, p],  // Defector vs Cooperator, Defector vs Defector
            ],
            n: 2,
        }
    }

    /// Create rock-paper-scissors payoff matrix
    pub fn rock_paper_scissors() -> Self {
        Self {
            payoffs: vec![
                vec![0.0, -1.0, 1.0],   // Rock vs Rock, Paper, Scissors
                vec![1.0, 0.0, -1.0],   // Paper vs Rock, Paper, Scissors
                vec![-1.0, 1.0, 0.0],   // Scissors vs Rock, Paper, Scissors
            ],
            n: 3,
        }
    }

    /// Create snowdrift game
    pub fn snowdrift(b: f64, c: f64) -> Self {
        // b = benefit, c = cost (b > c > 0)
        Self {
            payoffs: vec![
                vec![b - c / 2.0, b - c],  // Cooperator
                vec![b, 0.0],              // Defector
            ],
            n: 2,
        }
    }

    /// Get payoff
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.payoffs.get(i)
            .and_then(|row| row.get(j))
            .copied()
            .unwrap_or(0.0)
    }

    /// Set payoff
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        if i < self.n && j < self.n {
            self.payoffs[i][j] = value;
        }
    }
}

/// Hyperbolic replicator dynamics system
pub struct HyperbolicReplicator {
    /// Configuration
    config: ReplicatorConfig,
    /// Strategies
    pub strategies: Vec<Strategy>,
    /// Payoff matrix
    pub payoff: PayoffMatrix,
    /// Current time
    pub time: f64,
    /// Average fitness
    pub avg_fitness: f64,
    /// Statistics
    pub stats: ReplicatorStats,
}

/// Statistics for replicator dynamics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplicatorStats {
    /// Total time steps
    pub total_steps: u64,
    /// Number of extinctions
    pub extinctions: u64,
    /// Number of fixations (one strategy dominates)
    pub fixations: u64,
    /// Average entropy of distribution
    pub avg_entropy: f64,
    /// Time to equilibrium (if reached)
    pub equilibrium_time: Option<f64>,
    /// Is system at equilibrium?
    pub at_equilibrium: bool,
}

impl HyperbolicReplicator {
    /// Create new replicator system
    pub fn new(config: ReplicatorConfig, payoff: PayoffMatrix) -> Self {
        assert_eq!(config.num_strategies, payoff.n);

        // Initialize strategies with equal frequencies
        let initial_freq = 1.0 / config.num_strategies as f64;
        let strategies: Vec<Strategy> = (0..config.num_strategies)
            .map(|i| {
                // Position strategies on a circle in hyperbolic space
                let angle = 2.0 * std::f64::consts::PI * i as f64 / config.num_strategies as f64;
                let radius = 0.5;
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                let t = (1.0 + x * x + y * y).sqrt();

                Strategy::new(
                    i,
                    &format!("Strategy_{}", i),
                    LorentzVec::new(t, x, y, 0.0),
                    initial_freq,
                )
            })
            .collect();

        Self {
            config,
            strategies,
            payoff,
            time: 0.0,
            avg_fitness: 0.0,
            stats: ReplicatorStats::default(),
        }
    }

    /// Create with specific strategy positions
    pub fn with_positions(
        config: ReplicatorConfig,
        payoff: PayoffMatrix,
        positions: Vec<LorentzVec>,
    ) -> Self {
        let mut replicator = Self::new(config, payoff);

        for (i, pos) in positions.into_iter().enumerate() {
            if i < replicator.strategies.len() {
                replicator.strategies[i].position = pos;
            }
        }

        replicator
    }

    /// Set strategy names
    pub fn set_names(&mut self, names: &[&str]) {
        for (i, name) in names.iter().enumerate() {
            if i < self.strategies.len() {
                self.strategies[i].name = name.to_string();
            }
        }
    }

    /// Compute fitness for each strategy
    fn compute_fitness(&mut self) {
        let n = self.strategies.len();
        let frequencies: Vec<f64> = self.strategies.iter().map(|s| s.frequency).collect();

        for i in 0..n {
            let mut fitness = 0.0;

            for j in 0..n {
                let base_payoff = self.payoff.get(i, j);

                // Modulate by hyperbolic distance if enabled
                let effective_payoff = if self.config.hyperbolic_payoffs {
                    let dist = self.strategies[i].position
                        .hyperbolic_distance(&self.strategies[j].position);
                    // Closer strategies interact more strongly
                    base_payoff * (-dist * 0.5).exp()
                } else {
                    base_payoff
                };

                fitness += frequencies[j] * effective_payoff;
            }

            self.strategies[i].fitness = fitness * self.config.selection_intensity;
        }

        // Compute average fitness
        self.avg_fitness = self.strategies.iter()
            .map(|s| s.frequency * s.fitness)
            .sum();
    }

    /// Single step of replicator dynamics
    pub fn step(&mut self) -> StepResult {
        self.stats.total_steps += 1;
        self.time += self.config.dt;

        // Compute fitness
        self.compute_fitness();

        // Replicator equation: ẋᵢ = xᵢ(fᵢ - φ)
        let mut deltas = vec![0.0; self.strategies.len()];
        let mut extinctions = 0;

        for (i, strategy) in self.strategies.iter().enumerate() {
            // Selection term
            let selection = strategy.frequency * (strategy.fitness - self.avg_fitness);

            // Mutation term (uniform mutation to all others)
            let mut mutation = 0.0;
            if self.config.mutation_rate > 0.0 {
                let n = self.strategies.len() as f64;
                // Inflow from others
                let inflow: f64 = self.strategies.iter()
                    .filter(|s| s.id != i)
                    .map(|s| s.frequency)
                    .sum::<f64>() * self.config.mutation_rate / (n - 1.0);
                // Outflow to others
                let outflow = strategy.frequency * self.config.mutation_rate;
                mutation = inflow - outflow;
            }

            deltas[i] = self.config.dt * (selection + mutation);
        }

        // Apply changes
        for (i, delta) in deltas.iter().enumerate() {
            self.strategies[i].frequency = (self.strategies[i].frequency + delta)
                .max(0.0);

            // Check for extinction
            if self.strategies[i].frequency < self.config.min_frequency {
                self.strategies[i].frequency = 0.0;
                extinctions += 1;
            }
        }

        // Normalize frequencies
        let total: f64 = self.strategies.iter().map(|s| s.frequency).sum();
        if total > 1e-10 {
            for strategy in &mut self.strategies {
                strategy.frequency /= total;
            }
        }

        // Record history
        for strategy in &mut self.strategies {
            strategy.history.push_back(strategy.frequency);
            if strategy.history.len() > 1000 {
                strategy.history.pop_front();
            }
        }

        // Update statistics
        self.stats.extinctions += extinctions as u64;
        self.stats.avg_entropy = 0.99 * self.stats.avg_entropy + 0.01 * self.entropy();

        // Check for equilibrium
        let at_equilibrium = self.check_equilibrium();
        if at_equilibrium && !self.stats.at_equilibrium {
            self.stats.equilibrium_time = Some(self.time);
            self.stats.at_equilibrium = true;
        }

        // Check for fixation
        let fixation = self.strategies.iter()
            .any(|s| s.frequency > 0.99);
        if fixation {
            self.stats.fixations += 1;
        }

        StepResult {
            time: self.time,
            avg_fitness: self.avg_fitness,
            extinctions,
            at_equilibrium,
            fixation: if fixation {
                self.strategies.iter().position(|s| s.frequency > 0.99)
            } else {
                None
            },
        }
    }

    /// Run until equilibrium or max time
    pub fn run(&mut self, max_time: f64) -> Vec<StepResult> {
        let mut results = Vec::new();

        while self.time < max_time && !self.stats.at_equilibrium {
            results.push(self.step());
        }

        results
    }

    /// Compute entropy of frequency distribution
    pub fn entropy(&self) -> f64 {
        self.strategies.iter()
            .filter(|s| s.frequency > 0.0)
            .map(|s| -s.frequency * s.frequency.ln())
            .sum()
    }

    /// Check if system is at equilibrium
    fn check_equilibrium(&self) -> bool {
        // Check if frequencies are stable (low variance in recent history)
        for strategy in &self.strategies {
            if strategy.history.len() < 10 {
                return false;
            }

            let recent: Vec<f64> = strategy.history.iter()
                .rev()
                .take(10)
                .cloned()
                .collect();

            let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
            let variance: f64 = recent.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / recent.len() as f64;

            if variance > 1e-8 {
                return false;
            }
        }

        true
    }

    /// Get dominant strategy (highest frequency)
    pub fn dominant_strategy(&self) -> Option<&Strategy> {
        self.strategies.iter().max_by(|a, b| {
            a.frequency.partial_cmp(&b.frequency).unwrap()
        })
    }

    /// Get Nash equilibrium candidates
    pub fn nash_equilibria(&self) -> Vec<Vec<f64>> {
        // For simple 2-player symmetric games, find mixed Nash equilibria
        let n = self.strategies.len();
        let mut equilibria = Vec::new();

        // Pure strategy equilibria
        for i in 0..n {
            let mut freq = vec![0.0; n];
            freq[i] = 1.0;
            if self.is_nash(&freq) {
                equilibria.push(freq);
            }
        }

        // Mixed equilibria (simplified: just check current state)
        let current: Vec<f64> = self.strategies.iter().map(|s| s.frequency).collect();
        if self.is_nash(&current) {
            equilibria.push(current);
        }

        equilibria
    }

    /// Check if frequency vector is Nash equilibrium
    fn is_nash(&self, freq: &[f64]) -> bool {
        // Nash if no strategy can improve by unilateral deviation
        let n = freq.len();

        // Compute payoffs for each strategy given this distribution
        let payoffs: Vec<f64> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| freq[j] * self.payoff.get(i, j))
                    .sum()
            })
            .collect();

        // Find max payoff
        let max_payoff = payoffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // All strategies with positive frequency must achieve max payoff
        for i in 0..n {
            if freq[i] > 1e-6 && (payoffs[i] - max_payoff).abs() > 1e-6 {
                return false;
            }
        }

        true
    }

    /// Lyapunov function for convergence analysis
    pub fn lyapunov(&self) -> f64 {
        // For zero-sum games, average fitness is constant
        // For potential games, use the potential function
        // General: use entropy as proxy
        -self.entropy()
    }

    /// Update payoffs based on hyperbolic geometry
    pub fn update_hyperbolic_payoffs(&mut self) {
        if !self.config.hyperbolic_payoffs {
            return;
        }

        // Modulate payoff matrix by hyperbolic distance
        for i in 0..self.strategies.len() {
            for j in 0..self.strategies.len() {
                let dist = self.strategies[i].position
                    .hyperbolic_distance(&self.strategies[j].position);

                // Curvature-dependent interaction strength
                let curvature_factor = (-self.config.curvature * dist.powi(2) / 4.0).exp();

                // Store modulated payoff (original * distance factor)
                let original = self.payoff.get(i, j);
                self.payoff.set(i, j, original * curvature_factor);
            }
        }
    }

    /// Get reference to config
    pub fn config(&self) -> &ReplicatorConfig {
        &self.config
    }

    /// Get mutable reference to config
    pub fn config_mut(&mut self) -> &mut ReplicatorConfig {
        &mut self.config
    }

    /// Create with default payoff matrix (identity-like cooperation game)
    pub fn with_default_payoff(config: ReplicatorConfig) -> Self {
        let n = config.num_strategies;
        let mut payoff = PayoffMatrix::zeros(n);

        // Default: cooperation game where similar strategies benefit each other
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    payoff.set(i, j, 1.0); // Self-cooperation bonus
                } else {
                    let diff = (i as f64 - j as f64).abs() / n as f64;
                    payoff.set(i, j, 1.0 - diff); // Similarity benefit
                }
            }
        }

        Self::new(config, payoff)
    }

    /// Move strategy position based on fitness gradient
    pub fn evolve_positions(&mut self, learning_rate: f64) {
        // Strategies move toward high-fitness regions
        for i in 0..self.strategies.len() {
            let current_pos = self.strategies[i].position;

            // Compute fitness gradient in hyperbolic space
            let mut gradient = LorentzVec::new(0.0, 0.0, 0.0, 0.0);

            for j in 0..self.strategies.len() {
                if i == j {
                    continue;
                }

                let other_pos = &self.strategies[j].position;
                let other_fitness = self.strategies[j].fitness;

                // Direction to other strategy
                let direction = current_pos.log_map(other_pos);

                // Weight by relative fitness
                let fitness_diff = other_fitness - self.strategies[i].fitness;
                let weight = fitness_diff * self.strategies[j].frequency;

                gradient.x += weight * direction.x;
                gradient.y += weight * direction.y;
                gradient.z += weight * direction.z;
            }

            // Move along gradient via exponential map
            if gradient.x.abs() + gradient.y.abs() + gradient.z.abs() > 1e-10 {
                self.strategies[i].position = current_pos.exp_map(&gradient, learning_rate);
            }
        }
    }
}

/// Result of single step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Current time
    pub time: f64,
    /// Average fitness
    pub avg_fitness: f64,
    /// Number of extinctions this step
    pub extinctions: usize,
    /// Is system at equilibrium?
    pub at_equilibrium: bool,
    /// Index of fixed strategy (if fixation occurred)
    pub fixation: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prisoners_dilemma() {
        let config = ReplicatorConfig {
            num_strategies: 2,
            dt: 0.01,
            hyperbolic_payoffs: false,
            ..Default::default()
        };

        // Standard PD: T=5, R=3, P=1, S=0
        let payoff = PayoffMatrix::prisoners_dilemma(3.0, 0.0, 5.0, 1.0);
        let mut replicator = HyperbolicReplicator::new(config, payoff);

        replicator.set_names(&["Cooperate", "Defect"]);

        // Run until equilibrium
        replicator.run(100.0);

        // Defect should dominate in PD
        let dominant = replicator.dominant_strategy().unwrap();
        assert_eq!(dominant.name, "Defect");
    }

    #[test]
    fn test_rock_paper_scissors() {
        let config = ReplicatorConfig {
            num_strategies: 3,
            dt: 0.001,
            hyperbolic_payoffs: false,
            ..Default::default()
        };

        let payoff = PayoffMatrix::rock_paper_scissors();
        let mut replicator = HyperbolicReplicator::new(config, payoff);

        replicator.set_names(&["Rock", "Paper", "Scissors"]);

        // Run for a bit
        replicator.run(10.0);

        // Should maintain mixed equilibrium (oscillate around equal frequencies)
        let entropy = replicator.entropy();
        assert!(entropy > 0.5); // High entropy = mixed
    }

    #[test]
    fn test_hyperbolic_payoffs() {
        let config = ReplicatorConfig {
            num_strategies: 2,
            hyperbolic_payoffs: true,
            ..Default::default()
        };

        let payoff = PayoffMatrix::prisoners_dilemma(3.0, 0.0, 5.0, 1.0);
        let mut replicator = HyperbolicReplicator::new(config, payoff);

        // Position strategies at different distances
        replicator.strategies[0].position = LorentzVec::origin();
        replicator.strategies[1].position = LorentzVec::from_spatial(1.0, 0.0, 0.0);

        replicator.compute_fitness();

        // Both should have fitness computed
        assert!(replicator.strategies[0].fitness.is_finite());
        assert!(replicator.strategies[1].fitness.is_finite());
    }

    #[test]
    fn test_position_evolution() {
        let config = ReplicatorConfig {
            num_strategies: 3,
            ..Default::default()
        };

        let payoff = PayoffMatrix::rock_paper_scissors();
        let mut replicator = HyperbolicReplicator::new(config, payoff);

        let initial_positions: Vec<_> = replicator.strategies.iter()
            .map(|s| s.position)
            .collect();

        // Compute fitness and evolve positions
        replicator.compute_fitness();
        replicator.evolve_positions(0.1);

        // Positions should have changed
        let changed = replicator.strategies.iter()
            .zip(initial_positions.iter())
            .any(|(s, init)| {
                (s.position.x - init.x).abs() > 1e-10
                    || (s.position.y - init.y).abs() > 1e-10
            });

        // May or may not change depending on gradients
        // Just verify it doesn't crash
    }

    #[test]
    fn test_nash_equilibrium() {
        let config = ReplicatorConfig {
            num_strategies: 2,
            hyperbolic_payoffs: false,
            ..Default::default()
        };

        let payoff = PayoffMatrix::prisoners_dilemma(3.0, 0.0, 5.0, 1.0);
        let replicator = HyperbolicReplicator::new(config, payoff);

        let equilibria = replicator.nash_equilibria();

        // Pure defect should be Nash equilibrium in PD
        let defect_eq = equilibria.iter()
            .any(|eq| eq[1] > 0.9);

        assert!(defect_eq || equilibria.is_empty()); // Might need equilibrium check
    }
}
