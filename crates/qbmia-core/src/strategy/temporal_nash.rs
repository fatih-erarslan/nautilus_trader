//! Temporal Biological Nash Equilibrium - Time-aware game theory with biological patterns

use crate::error::{QBMIAError, Result};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Temporal Biological Nash Equilibrium solver
#[derive(Debug, Clone)]
pub struct TemporalBiologicalNash {
    pub time_horizon: usize,
    pub decay_factor: f64,
    pub biological_patterns: Vec<BiologicalPattern>,
    pub temporal_weights: Vec<f64>,
}

/// Temporal equilibrium result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEquilibrium {
    pub strategies: HashMap<String, Vec<f64>>,
    pub payoffs: HashMap<String, f64>,
    pub convergence_time: usize,
    pub stability_score: f64,
    pub biological_influence: f64,
}

/// Biological pattern for temporal evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalPattern {
    pub pattern_type: BiologicalPatternType,
    pub strength: f64,
    pub temporal_evolution: Vec<f64>,
    pub parameters: HashMap<String, f64>,
}

/// Types of biological patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BiologicalPatternType {
    CircadianRhythm,
    SeasonalCycle,
    PopulationDynamics,
    PredatorPrey,
    CooperativeEvolution,
    CompetitiveSelection,
}

impl TemporalBiologicalNash {
    /// Create new temporal biological Nash solver
    pub fn new(time_horizon: usize, decay_factor: f64) -> Self {
        let temporal_weights = (0..time_horizon)
            .map(|t| (-decay_factor * t as f64).exp())
            .collect();

        Self {
            time_horizon,
            decay_factor,
            biological_patterns: Vec::new(),
            temporal_weights,
        }
    }

    /// Add biological pattern to the system
    pub fn add_biological_pattern(&mut self, pattern: BiologicalPattern) {
        self.biological_patterns.push(pattern);
    }

    /// Solve temporal Nash equilibrium with biological influences
    pub async fn solve_temporal_equilibrium(
        &self,
        payoff_matrices: &HashMap<String, DMatrix<f64>>,
        initial_strategies: &HashMap<String, DVector<f64>>,
        time_steps: usize,
    ) -> Result<TemporalEquilibrium> {
        let players: Vec<String> = payoff_matrices.keys().cloned().collect();
        let mut current_strategies = initial_strategies.clone();
        let mut convergence_time = 0;

        // Temporal evolution with biological influences
        for t in 0..time_steps {
            let mut next_strategies = HashMap::new();
            
            for player in &players {
                let strategy = self.update_strategy_with_biology(
                    player,
                    &current_strategies,
                    payoff_matrices,
                    t,
                ).await?;
                next_strategies.insert(player.clone(), strategy);
            }

            // Check convergence
            if self.check_convergence(&current_strategies, &next_strategies, 1e-6) {
                convergence_time = t;
                break;
            }

            current_strategies = next_strategies;
        }

        // Calculate final payoffs
        let payoffs = self.calculate_temporal_payoffs(&current_strategies, payoff_matrices).await?;
        
        // Calculate stability and biological influence
        let stability_score = self.calculate_stability_score(&current_strategies, payoff_matrices).await?;
        let biological_influence = self.calculate_biological_influence_score();

        // Convert strategies to vectors for serialization
        let strategies: HashMap<String, Vec<f64>> = current_strategies
            .into_iter()
            .map(|(k, v)| (k, v.as_slice().to_vec()))
            .collect();

        Ok(TemporalEquilibrium {
            strategies,
            payoffs,
            convergence_time,
            stability_score,
            biological_influence,
        })
    }

    /// Update strategy with biological influences
    async fn update_strategy_with_biology(
        &self,
        player: &str,
        current_strategies: &HashMap<String, DVector<f64>>,
        payoff_matrices: &HashMap<String, DMatrix<f64>>,
        time_step: usize,
    ) -> Result<DVector<f64>> {
        let current_strategy = current_strategies.get(player)
            .ok_or_else(|| QBMIAError::nash_convergence(format!("Strategy not found for player: {}", player)))?;

        // Calculate best response
        let best_response = self.calculate_best_response(
            player,
            current_strategies,
            payoff_matrices,
        ).await?;

        // Apply biological influences
        let biological_adjustment = self.apply_biological_patterns(
            current_strategy,
            time_step,
            player,
        ).await?;

        // Combine best response with biological patterns
        let learning_rate = 0.1 * self.temporal_weights.get(time_step % self.time_horizon).unwrap_or(&1.0);
        let updated_strategy = current_strategy + 
            &((&best_response - current_strategy) * learning_rate + &biological_adjustment * 0.05);

        // Normalize to ensure valid probability distribution
        let sum = updated_strategy.sum();
        if sum > 0.0 {
            Ok(updated_strategy / sum)
        } else {
            Ok(DVector::from_element(current_strategy.len(), 1.0 / current_strategy.len() as f64))
        }
    }

    /// Calculate best response for a player
    async fn calculate_best_response(
        &self,
        player: &str,
        current_strategies: &HashMap<String, DVector<f64>>,
        payoff_matrices: &HashMap<String, DMatrix<f64>>,
    ) -> Result<DVector<f64>> {
        let payoff_matrix = payoff_matrices.get(player)
            .ok_or_else(|| QBMIAError::nash_convergence(format!("Payoff matrix not found for player: {}", player)))?;

        // Calculate expected payoffs for each action
        let mut expected_payoffs = DVector::zeros(payoff_matrix.nrows());
        
        for (opponent, opponent_strategy) in current_strategies {
            if opponent != player {
                // Expected payoffs against this opponent
                let opponent_payoffs = payoff_matrix * opponent_strategy;
                expected_payoffs += opponent_payoffs;
            }
        }

        // Find best response (softmax to allow for exploration)
        let temperature = 0.1;
        let exp_payoffs = expected_payoffs.map(|x| (x / temperature).exp());
        let sum_exp = exp_payoffs.sum();
        
        if sum_exp > 0.0 {
            Ok(exp_payoffs / sum_exp)
        } else {
            Ok(DVector::from_element(expected_payoffs.len(), 1.0 / expected_payoffs.len() as f64))
        }
    }

    /// Apply biological patterns to strategy evolution
    async fn apply_biological_patterns(
        &self,
        current_strategy: &DVector<f64>,
        time_step: usize,
        player: &str,
    ) -> Result<DVector<f64>> {
        let mut biological_adjustment = DVector::zeros(current_strategy.len());

        for pattern in &self.biological_patterns {
            let pattern_influence = self.calculate_pattern_influence(
                pattern,
                time_step,
                player,
                current_strategy,
            ).await?;
            
            biological_adjustment += &pattern_influence;
        }

        Ok(biological_adjustment)
    }

    /// Calculate influence of a specific biological pattern
    async fn calculate_pattern_influence(
        &self,
        pattern: &BiologicalPattern,
        time_step: usize,
        _player: &str,
        current_strategy: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        let time_index = time_step % pattern.temporal_evolution.len();
        let temporal_value = pattern.temporal_evolution[time_index];
        
        let mut influence = DVector::zeros(current_strategy.len());

        match pattern.pattern_type {
            BiologicalPatternType::CircadianRhythm => {
                // Circadian patterns affect all actions uniformly with temporal oscillation
                let oscillation = (2.0 * std::f64::consts::PI * time_step as f64 / 24.0).sin();
                influence.fill(pattern.strength * temporal_value * oscillation * 0.01);
            },
            BiologicalPatternType::SeasonalCycle => {
                // Seasonal patterns with longer cycles
                let seasonal = (2.0 * std::f64::consts::PI * time_step as f64 / 365.0).sin();
                influence.fill(pattern.strength * temporal_value * seasonal * 0.005);
            },
            BiologicalPatternType::PopulationDynamics => {
                // Population dynamics affect strategy diversity
                let diversity = self.calculate_strategy_diversity(current_strategy);
                influence.fill(pattern.strength * temporal_value * (1.0 - diversity) * 0.02);
            },
            BiologicalPatternType::PredatorPrey => {
                // Predator-prey dynamics with alternating advantages
                let phase = (time_step as f64 * 0.1).sin();
                for i in 0..influence.len() {
                    influence[i] = pattern.strength * temporal_value * phase * 
                        if i % 2 == 0 { 0.01 } else { -0.01 };
                }
            },
            BiologicalPatternType::CooperativeEvolution => {
                // Cooperative patterns favor balanced strategies
                let balance_bonus = 1.0 - self.calculate_strategy_concentration(current_strategy);
                influence.fill(pattern.strength * temporal_value * balance_bonus * 0.015);
            },
            BiologicalPatternType::CompetitiveSelection => {
                // Competitive selection favors dominant strategies
                let max_index = current_strategy.argmax().0;
                influence[max_index] = pattern.strength * temporal_value * 0.02;
            },
        }

        Ok(influence)
    }

    /// Calculate strategy diversity
    fn calculate_strategy_diversity(&self, strategy: &DVector<f64>) -> f64 {
        // Shannon entropy as diversity measure
        let mut entropy = 0.0;
        for &prob in strategy.iter() {
            if prob > 0.0 {
                entropy -= prob * prob.ln();
            }
        }
        entropy / (strategy.len() as f64).ln()
    }

    /// Calculate strategy concentration
    fn calculate_strategy_concentration(&self, strategy: &DVector<f64>) -> f64 {
        // Herfindahl-Hirschman Index
        strategy.iter().map(|&x| x * x).sum()
    }

    /// Check convergence between strategy sets
    fn check_convergence(
        &self,
        current: &HashMap<String, DVector<f64>>,
        next: &HashMap<String, DVector<f64>>,
        tolerance: f64,
    ) -> bool {
        for (player, current_strategy) in current {
            if let Some(next_strategy) = next.get(player) {
                let diff = (current_strategy - next_strategy).norm();
                if diff > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Calculate temporal payoffs with time-weighted averaging
    async fn calculate_temporal_payoffs(
        &self,
        strategies: &HashMap<String, DVector<f64>>,
        payoff_matrices: &HashMap<String, DMatrix<f64>>,
    ) -> Result<HashMap<String, f64>> {
        let mut payoffs = HashMap::new();

        for (player, strategy) in strategies {
            if let Some(payoff_matrix) = payoff_matrices.get(player) {
                let mut total_payoff = 0.0;
                
                for (opponent, opponent_strategy) in strategies {
                    if opponent != player {
                        let expected_payoff = strategy.dot(&(payoff_matrix * opponent_strategy));
                        total_payoff += expected_payoff;
                    }
                }
                
                payoffs.insert(player.clone(), total_payoff);
            }
        }

        Ok(payoffs)
    }

    /// Calculate stability score of the equilibrium
    async fn calculate_stability_score(
        &self,
        strategies: &HashMap<String, DVector<f64>>,
        payoff_matrices: &HashMap<String, DMatrix<f64>>,
    ) -> Result<f64> {
        let mut stability_scores = Vec::new();

        for (player, strategy) in strategies {
            if let Some(payoff_matrix) = payoff_matrices.get(player) {
                // Calculate how much improvement is possible by deviating
                let current_payoff = self.calculate_strategy_payoff(player, strategy, strategies, payoff_matrices).await?;
                
                // Try all pure strategies as deviations
                let mut max_deviation_benefit: f64 = 0.0;
                for i in 0..strategy.len() {
                    let mut deviation_strategy = DVector::zeros(strategy.len());
                    deviation_strategy[i] = 1.0;
                    
                    let deviation_payoff = self.calculate_strategy_payoff(player, &deviation_strategy, strategies, payoff_matrices).await?;
                    let benefit = deviation_payoff - current_payoff;
                    max_deviation_benefit = max_deviation_benefit.max(benefit);
                }
                
                // Stability is higher when deviation benefits are lower
                let stability = (-max_deviation_benefit).exp();
                stability_scores.push(stability);
            }
        }

        Ok(stability_scores.iter().sum::<f64>() / stability_scores.len() as f64)
    }

    /// Calculate payoff for a specific strategy
    async fn calculate_strategy_payoff(
        &self,
        player: &str,
        strategy: &DVector<f64>,
        all_strategies: &HashMap<String, DVector<f64>>,
        payoff_matrices: &HashMap<String, DMatrix<f64>>,
    ) -> Result<f64> {
        if let Some(payoff_matrix) = payoff_matrices.get(player) {
            let mut total_payoff = 0.0;
            
            for (opponent, opponent_strategy) in all_strategies {
                if opponent != player {
                    let expected_payoff = strategy.dot(&(payoff_matrix * opponent_strategy));
                    total_payoff += expected_payoff;
                }
            }
            
            Ok(total_payoff)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate biological influence score
    fn calculate_biological_influence_score(&self) -> f64 {
        let total_strength: f64 = self.biological_patterns.iter().map(|p| p.strength).sum();
        (total_strength / (1.0 + total_strength)).min(1.0)
    }
}

impl Default for TemporalBiologicalNash {
    fn default() -> Self {
        Self::new(24, 0.1) // 24-hour horizon with 10% decay
    }
}

impl BiologicalPattern {
    /// Create circadian rhythm pattern
    pub fn circadian_rhythm(strength: f64) -> Self {
        let temporal_evolution: Vec<f64> = (0..24)
            .map(|hour| (2.0 * std::f64::consts::PI * hour as f64 / 24.0).sin())
            .collect();

        Self {
            pattern_type: BiologicalPatternType::CircadianRhythm,
            strength,
            temporal_evolution,
            parameters: HashMap::new(),
        }
    }

    /// Create seasonal cycle pattern
    pub fn seasonal_cycle(strength: f64) -> Self {
        let temporal_evolution: Vec<f64> = (0..365)
            .map(|day| (2.0 * std::f64::consts::PI * day as f64 / 365.0).sin())
            .collect();

        Self {
            pattern_type: BiologicalPatternType::SeasonalCycle,
            strength,
            temporal_evolution,
            parameters: HashMap::new(),
        }
    }

    /// Create predator-prey dynamics pattern
    pub fn predator_prey(strength: f64, cycle_length: usize) -> Self {
        let temporal_evolution: Vec<f64> = (0..cycle_length)
            .map(|t| {
                let phase = 2.0 * std::f64::consts::PI * t as f64 / cycle_length as f64;
                (phase.sin() + phase.cos()) / 2.0
            })
            .collect();

        Self {
            pattern_type: BiologicalPatternType::PredatorPrey,
            strength,
            temporal_evolution,
            parameters: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_temporal_nash_creation() {
        let nash = TemporalBiologicalNash::new(24, 0.1);
        assert_eq!(nash.time_horizon, 24);
        assert_eq!(nash.temporal_weights.len(), 24);
    }

    #[tokio::test]
    async fn test_biological_patterns() {
        let circadian = BiologicalPattern::circadian_rhythm(0.5);
        assert_eq!(circadian.temporal_evolution.len(), 24);
        assert!(matches!(circadian.pattern_type, BiologicalPatternType::CircadianRhythm));

        let seasonal = BiologicalPattern::seasonal_cycle(0.3);
        assert_eq!(seasonal.temporal_evolution.len(), 365);
    }

    #[tokio::test]
    async fn test_simple_equilibrium() {
        let mut nash = TemporalBiologicalNash::new(10, 0.1);
        nash.add_biological_pattern(BiologicalPattern::circadian_rhythm(0.1));

        // Simple 2x2 game
        let mut payoff_matrices = HashMap::new();
        let payoff_a = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 5.0, 1.0]);
        let payoff_b = DMatrix::from_row_slice(2, 2, &[3.0, 5.0, 0.0, 1.0]);
        
        payoff_matrices.insert("A".to_string(), payoff_a);
        payoff_matrices.insert("B".to_string(), payoff_b);

        let mut initial_strategies = HashMap::new();
        initial_strategies.insert("A".to_string(), DVector::from_vec(vec![0.5, 0.5]));
        initial_strategies.insert("B".to_string(), DVector::from_vec(vec![0.5, 0.5]));

        let result = nash.solve_temporal_equilibrium(&payoff_matrices, &initial_strategies, 50).await;
        assert!(result.is_ok());

        let equilibrium = result.unwrap();
        assert!(equilibrium.strategies.contains_key("A"));
        assert!(equilibrium.strategies.contains_key("B"));
        assert!(equilibrium.stability_score >= 0.0 && equilibrium.stability_score <= 1.0);
        assert!(equilibrium.biological_influence >= 0.0 && equilibrium.biological_influence <= 1.0);
    }
}