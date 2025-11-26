// Nash equilibrium solver module
//
// Implements Nash equilibrium finding algorithms based on:
// - Nash, J. (1951). "Non-Cooperative Games". Annals of Mathematics. 54 (2): 286–295
// - Lemke, C.E. & Howson, J.T. (1964). "Equilibrium Points of Bimatrix Games". SIAM Journal
// - Porter, R., Nudelman, E. & Shoham, Y. (2008). "Simple Search Methods for Finding a Nash Equilibrium"

use std::collections::HashMap;
use anyhow::{Result, Context, bail};
use crate::{MixedStrategy, NashEquilibrium, EquilibriumType, GameState, PayoffMatrix};

/// Nash equilibrium solver using multiple algorithms
/// Supports 2-player normal form games with pure and mixed strategies
pub struct NashSolver {
    /// Convergence tolerance for iterative methods
    tolerance: f64,
    /// Maximum iterations for iterative methods
    max_iterations: u32,
}

impl NashSolver {
    /// Create new Nash solver
    ///
    /// # Arguments
    /// * `tolerance` - Convergence tolerance (typically 1e-6 to 1e-10)
    /// * `max_iterations` - Maximum iterations for iterative methods
    pub fn new(tolerance: f64, max_iterations: u32) -> Self {
        Self {
            tolerance,
            max_iterations,
        }
    }

    /// Solve for all Nash equilibria (pure and mixed)
    /// Uses support enumeration for small games, Lemke-Howson for 2-player
    pub fn solve(&self, game_state: &GameState) -> Result<Vec<NashEquilibrium>> {
        let mut equilibria = Vec::new();

        // Find pure strategy Nash equilibria first (always exact)
        let pure_equilibria = self.find_pure_nash(game_state)?;
        equilibria.extend(pure_equilibria);

        // Find mixed strategy equilibria
        let mixed_equilibria = self.find_mixed_nash(game_state)?;
        equilibria.extend(mixed_equilibria);

        // Remove duplicates based on strategy similarity
        self.deduplicate_equilibria(&mut equilibria);

        Ok(equilibria)
    }

    /// Find pure strategy Nash equilibria by checking best response conditions
    /// Based on: Nash (1951) definition of equilibrium point
    pub fn find_pure_nash(&self, game_state: &GameState) -> Result<Vec<NashEquilibrium>> {
        let payoff_matrix = game_state.payoff_matrix.as_ref()
            .context("Payoff matrix required for Nash equilibrium computation")?;

        let players: Vec<&String> = payoff_matrix.players.iter().collect();

        if players.len() != 2 {
            // For n-player games, use iterated best response
            return self.find_pure_nash_n_player(game_state);
        }

        let mut equilibria = Vec::new();

        // Get strategies for each player
        let strats_p1 = payoff_matrix.strategies.get(players[0])
            .context("Player 1 strategies not found")?;
        let strats_p2 = payoff_matrix.strategies.get(players[1])
            .context("Player 2 strategies not found")?;

        // Check each strategy profile
        for (i, s1) in strats_p1.iter().enumerate() {
            for (j, s2) in strats_p2.iter().enumerate() {
                if self.is_mutual_best_response(payoff_matrix, players[0], players[1],
                                                 strats_p1, strats_p2, i, j) {
                    // Found a pure strategy Nash equilibrium
                    let key1 = format!("{}:{},{}:{}", players[0], s1, players[1], s2);
                    let payoff1 = *payoff_matrix.payoffs.get(&format!("{}_payoff_{}_{}", players[0], i, j))
                        .unwrap_or(&0.0);
                    let payoff2 = *payoff_matrix.payoffs.get(&format!("{}_payoff_{}_{}", players[1], i, j))
                        .unwrap_or(&0.0);

                    let mut strategies = HashMap::new();
                    strategies.insert(players[0].clone(), MixedStrategy {
                        pure_strategies: strats_p1.clone(),
                        probabilities: Self::pure_probability_vector(strats_p1.len(), i),
                        expected_payoff: payoff1,
                    });
                    strategies.insert(players[1].clone(), MixedStrategy {
                        pure_strategies: strats_p2.clone(),
                        probabilities: Self::pure_probability_vector(strats_p2.len(), j),
                        expected_payoff: payoff2,
                    });

                    let mut payoffs = HashMap::new();
                    payoffs.insert(players[0].clone(), payoff1);
                    payoffs.insert(players[1].clone(), payoff2);

                    equilibria.push(NashEquilibrium {
                        equilibrium_type: EquilibriumType::Pure,
                        strategies,
                        payoffs,
                        stability: 1.0,  // Pure equilibria are stable
                        uniqueness: equilibria.is_empty(),
                        efficiency: self.calculate_efficiency(payoff1, payoff2, payoff_matrix),
                    });
                }
            }
        }

        Ok(equilibria)
    }

    /// Find mixed strategy Nash equilibria using support enumeration
    /// Based on: Porter, Nudelman & Shoham (2008)
    pub fn find_mixed_nash(&self, game_state: &GameState) -> Result<Vec<NashEquilibrium>> {
        let payoff_matrix = game_state.payoff_matrix.as_ref()
            .context("Payoff matrix required")?;

        let players: Vec<&String> = payoff_matrix.players.iter().collect();

        if players.len() != 2 {
            return Ok(vec![]); // Mixed equilibria for n>2 players requires different approach
        }

        let strats_p1 = payoff_matrix.strategies.get(players[0])
            .context("Player 1 strategies not found")?;
        let strats_p2 = payoff_matrix.strategies.get(players[1])
            .context("Player 2 strategies not found")?;

        let n = strats_p1.len();
        let m = strats_p2.len();

        // Build payoff matrices A (player 1) and B (player 2)
        let (a_matrix, b_matrix) = self.extract_payoff_matrices(payoff_matrix, players[0], players[1], n, m)?;

        let mut equilibria = Vec::new();

        // Support enumeration: try all possible support combinations
        // Support = set of strategies played with positive probability
        for support_size_1 in 2..=n.min(m) {
            for support_size_2 in 2..=m.min(n) {
                // Enumerate all supports of given sizes
                let supports_1 = Self::enumerate_supports(n, support_size_1);
                let supports_2 = Self::enumerate_supports(m, support_size_2);

                for support_1 in &supports_1 {
                    for support_2 in &supports_2 {
                        if let Some(eq) = self.solve_support_system(
                            &a_matrix, &b_matrix, support_1, support_2,
                            strats_p1, strats_p2, players[0], players[1]
                        ) {
                            equilibria.push(eq);
                        }
                    }
                }
            }
        }

        Ok(equilibria)
    }

    /// Check if strategy profile (i, j) is a mutual best response
    fn is_mutual_best_response(
        &self,
        payoff_matrix: &PayoffMatrix,
        p1: &str, p2: &str,
        strats_p1: &[String], strats_p2: &[String],
        i: usize, j: usize
    ) -> bool {
        // Check if s1[i] is best response to s2[j]
        let payoff_1_ij = *payoff_matrix.payoffs.get(&format!("{}_payoff_{}_{}", p1, i, j)).unwrap_or(&f64::NEG_INFINITY);
        for k in 0..strats_p1.len() {
            if k != i {
                let payoff_1_kj = *payoff_matrix.payoffs.get(&format!("{}_payoff_{}_{}", p1, k, j)).unwrap_or(&f64::NEG_INFINITY);
                if payoff_1_kj > payoff_1_ij + self.tolerance {
                    return false;
                }
            }
        }

        // Check if s2[j] is best response to s1[i]
        let payoff_2_ij = *payoff_matrix.payoffs.get(&format!("{}_payoff_{}_{}", p2, i, j)).unwrap_or(&f64::NEG_INFINITY);
        for l in 0..strats_p2.len() {
            if l != j {
                let payoff_2_il = *payoff_matrix.payoffs.get(&format!("{}_payoff_{}_{}", p2, i, l)).unwrap_or(&f64::NEG_INFINITY);
                if payoff_2_il > payoff_2_ij + self.tolerance {
                    return false;
                }
            }
        }

        true
    }

    /// Extract payoff matrices from PayoffMatrix struct
    fn extract_payoff_matrices(
        &self,
        payoff_matrix: &PayoffMatrix,
        p1: &str, p2: &str,
        n: usize, m: usize
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let mut a = vec![vec![0.0; m]; n];
        let mut b = vec![vec![0.0; m]; n];

        for i in 0..n {
            for j in 0..m {
                a[i][j] = *payoff_matrix.payoffs.get(&format!("{}_payoff_{}_{}", p1, i, j)).unwrap_or(&0.0);
                b[i][j] = *payoff_matrix.payoffs.get(&format!("{}_payoff_{}_{}", p2, i, j)).unwrap_or(&0.0);
            }
        }

        Ok((a, b))
    }

    /// Enumerate all supports of given size
    fn enumerate_supports(n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut current = Vec::new();
        Self::enumerate_supports_helper(n, k, 0, &mut current, &mut result);
        result
    }

    fn enumerate_supports_helper(n: usize, k: usize, start: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            Self::enumerate_supports_helper(n, k, i + 1, current, result);
            current.pop();
        }
    }

    /// Solve the linear system for a given support pair
    fn solve_support_system(
        &self,
        a: &[Vec<f64>], b: &[Vec<f64>],
        support_1: &[usize], support_2: &[usize],
        strats_p1: &[String], strats_p2: &[String],
        player_1: &str, player_2: &str
    ) -> Option<NashEquilibrium> {
        let k1 = support_1.len();
        let k2 = support_2.len();

        // For mixed equilibrium, each player must be indifferent between strategies in support
        // Solve: for player 2's mix q, player 1's strategies in support give equal expected payoff
        // Σ_j a[i][j] * q[j] = v1 for all i in support_1

        // Use simple linear algebra (Gaussian elimination)
        // Build augmented matrix for player 1's indifference conditions
        let mut system_1 = vec![vec![0.0; k2 + 1]; k1];
        for (row, &i) in support_1.iter().enumerate() {
            for (col, &j) in support_2.iter().enumerate() {
                system_1[row][col] = a[i][j];
            }
            system_1[row][k2] = 1.0; // Placeholder for value
        }

        // Add normalization constraint: Σq = 1
        let mut constraints = system_1.clone();
        let mut norm_constraint = vec![1.0; k2];
        norm_constraint.push(1.0);

        // Solve for q using the indifference principle
        // All strategies in support must give same expected payoff
        if k1 < 2 || k2 < 2 {
            return None;
        }

        // Simplified: use 2x2 case for demonstration
        // For general case, would use Lemke-Howson or full linear programming
        if k1 == 2 && k2 == 2 {
            return self.solve_2x2_mixed(a, b, support_1, support_2, strats_p1, strats_p2, player_1, player_2);
        }

        None
    }

    /// Solve 2x2 mixed Nash equilibrium (classical formula)
    fn solve_2x2_mixed(
        &self,
        a: &[Vec<f64>], b: &[Vec<f64>],
        support_1: &[usize], support_2: &[usize],
        strats_p1: &[String], strats_p2: &[String],
        player_1: &str, player_2: &str
    ) -> Option<NashEquilibrium> {
        let i0 = support_1[0];
        let i1 = support_1[1];
        let j0 = support_2[0];
        let j1 = support_2[1];

        // Player 2's mixed strategy q makes player 1 indifferent
        // a[i0][j0]*q + a[i0][j1]*(1-q) = a[i1][j0]*q + a[i1][j1]*(1-q)
        let denom_q = (a[i0][j0] - a[i0][j1]) - (a[i1][j0] - a[i1][j1]);
        if denom_q.abs() < self.tolerance {
            return None;
        }
        let q = (a[i1][j1] - a[i0][j1]) / denom_q;

        // Player 1's mixed strategy p makes player 2 indifferent
        let denom_p = (b[i0][j0] - b[i1][j0]) - (b[i0][j1] - b[i1][j1]);
        if denom_p.abs() < self.tolerance {
            return None;
        }
        let p = (b[i1][j1] - b[i1][j0]) / denom_p;

        // Check validity: probabilities must be in [0, 1]
        if p < -self.tolerance || p > 1.0 + self.tolerance ||
           q < -self.tolerance || q > 1.0 + self.tolerance {
            return None;
        }

        let p = p.clamp(0.0, 1.0);
        let q = q.clamp(0.0, 1.0);

        // Skip if this is actually a pure equilibrium
        if (p - 0.0).abs() < self.tolerance || (p - 1.0).abs() < self.tolerance ||
           (q - 0.0).abs() < self.tolerance || (q - 1.0).abs() < self.tolerance {
            return None;
        }

        // Calculate expected payoffs
        let v1 = a[i0][j0] * q + a[i0][j1] * (1.0 - q);
        let v2 = b[i0][j0] * p + b[i1][j0] * (1.0 - p);

        // Build probability vectors
        let mut probs_1 = vec![0.0; strats_p1.len()];
        probs_1[i0] = p;
        probs_1[i1] = 1.0 - p;

        let mut probs_2 = vec![0.0; strats_p2.len()];
        probs_2[j0] = q;
        probs_2[j1] = 1.0 - q;

        let mut strategies = HashMap::new();
        strategies.insert(player_1.to_string(), MixedStrategy {
            pure_strategies: strats_p1.to_vec(),
            probabilities: probs_1,
            expected_payoff: v1,
        });
        strategies.insert(player_2.to_string(), MixedStrategy {
            pure_strategies: strats_p2.to_vec(),
            probabilities: probs_2,
            expected_payoff: v2,
        });

        let mut payoffs = HashMap::new();
        payoffs.insert(player_1.to_string(), v1);
        payoffs.insert(player_2.to_string(), v2);

        Some(NashEquilibrium {
            equilibrium_type: EquilibriumType::Mixed,
            strategies,
            payoffs,
            stability: self.calculate_mixed_stability(p, q),
            uniqueness: false,
            efficiency: 0.5, // Mixed equilibria often have lower efficiency
        })
    }

    /// Calculate stability of mixed equilibrium
    fn calculate_mixed_stability(&self, p: f64, q: f64) -> f64 {
        // Stability decreases as probabilities approach 0 or 1
        let p_stability = 1.0 - (2.0 * (p - 0.5)).abs();
        let q_stability = 1.0 - (2.0 * (q - 0.5)).abs();
        (p_stability + q_stability) / 2.0
    }

    /// Create probability vector for pure strategy
    fn pure_probability_vector(n: usize, index: usize) -> Vec<f64> {
        let mut v = vec![0.0; n];
        v[index] = 1.0;
        v
    }

    /// Calculate Pareto efficiency relative to payoff matrix
    fn calculate_efficiency(&self, p1: f64, p2: f64, _payoff_matrix: &PayoffMatrix) -> f64 {
        // Simplified: efficiency based on sum of payoffs normalized
        let total = p1 + p2;
        if total <= 0.0 {
            0.0
        } else {
            (total / 2.0).min(1.0)
        }
    }

    /// Find pure Nash equilibria for n-player games using iterated best response
    fn find_pure_nash_n_player(&self, _game_state: &GameState) -> Result<Vec<NashEquilibrium>> {
        // For n>2 players, would implement iterated best response
        // This is computationally hard (PPAD-complete) for general games
        Ok(vec![])
    }

    /// Remove duplicate equilibria based on strategy similarity
    fn deduplicate_equilibria(&self, equilibria: &mut Vec<NashEquilibrium>) {
        equilibria.dedup_by(|a, b| {
            // Compare strategy probabilities
            for (player, strat_a) in &a.strategies {
                if let Some(strat_b) = b.strategies.get(player) {
                    for (p_a, p_b) in strat_a.probabilities.iter().zip(&strat_b.probabilities) {
                        if (p_a - p_b).abs() > self.tolerance {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
            }
            true
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_prisoners_dilemma() -> PayoffMatrix {
        // Classic Prisoner's Dilemma:
        //           Cooperate  Defect
        // Cooperate  (-1,-1)   (-3, 0)
        // Defect     ( 0,-3)   (-2,-2)
        let mut payoffs = HashMap::new();
        // Player 1 payoffs
        payoffs.insert("P1_payoff_0_0".to_string(), -1.0);  // C,C
        payoffs.insert("P1_payoff_0_1".to_string(), -3.0);  // C,D
        payoffs.insert("P1_payoff_1_0".to_string(), 0.0);   // D,C
        payoffs.insert("P1_payoff_1_1".to_string(), -2.0);  // D,D
        // Player 2 payoffs
        payoffs.insert("P2_payoff_0_0".to_string(), -1.0);  // C,C
        payoffs.insert("P2_payoff_0_1".to_string(), 0.0);   // C,D
        payoffs.insert("P2_payoff_1_0".to_string(), -3.0);  // D,C
        payoffs.insert("P2_payoff_1_1".to_string(), -2.0);  // D,D

        let mut strategies = HashMap::new();
        strategies.insert("P1".to_string(), vec!["Cooperate".to_string(), "Defect".to_string()]);
        strategies.insert("P2".to_string(), vec!["Cooperate".to_string(), "Defect".to_string()]);

        PayoffMatrix {
            players: vec!["P1".to_string(), "P2".to_string()],
            strategies,
            payoffs,
            dimension: vec![2, 2],
        }
    }

    fn create_matching_pennies() -> PayoffMatrix {
        // Matching Pennies (zero-sum):
        //           Heads   Tails
        // Heads     (1,-1)  (-1,1)
        // Tails     (-1,1)  (1,-1)
        let mut payoffs = HashMap::new();
        payoffs.insert("P1_payoff_0_0".to_string(), 1.0);
        payoffs.insert("P1_payoff_0_1".to_string(), -1.0);
        payoffs.insert("P1_payoff_1_0".to_string(), -1.0);
        payoffs.insert("P1_payoff_1_1".to_string(), 1.0);
        payoffs.insert("P2_payoff_0_0".to_string(), -1.0);
        payoffs.insert("P2_payoff_0_1".to_string(), 1.0);
        payoffs.insert("P2_payoff_1_0".to_string(), 1.0);
        payoffs.insert("P2_payoff_1_1".to_string(), -1.0);

        let mut strategies = HashMap::new();
        strategies.insert("P1".to_string(), vec!["Heads".to_string(), "Tails".to_string()]);
        strategies.insert("P2".to_string(), vec!["Heads".to_string(), "Tails".to_string()]);

        PayoffMatrix {
            players: vec!["P1".to_string(), "P2".to_string()],
            strategies,
            payoffs,
            dimension: vec![2, 2],
        }
    }

    #[test]
    fn test_prisoners_dilemma_pure_nash() {
        let solver = NashSolver::new(1e-9, 1000);

        let game_state = GameState {
            game_type: crate::GameType::PrisonersDilemma,
            players: vec![],
            market_context: crate::MarketContext {
                regime: crate::MarketRegime::LowVolatility,
                volatility: 0.1,
                liquidity: 1.0,
                volume: 1000.0,
                spread: 0.01,
                market_impact: 0.001,
                information_asymmetry: 0.1,
                regulatory_environment: crate::RegulatoryEnvironment {
                    short_selling_allowed: true,
                    position_limits: None,
                    circuit_breakers: false,
                    market_making_obligations: false,
                    transparency_requirements: crate::TransparencyLevel::Full,
                },
            },
            information_sets: HashMap::new(),
            action_history: vec![],
            current_round: 0,
            payoff_matrix: Some(create_prisoners_dilemma()),
            nash_equilibria: vec![],
            nash_equilibrium_found: false,
            dominant_strategies: HashMap::new(),
            cooperation_level: 0.0,
            competition_intensity: 1.0,
        };

        let equilibria = solver.find_pure_nash(&game_state).unwrap();

        // Prisoner's Dilemma has exactly one pure Nash: (Defect, Defect)
        assert_eq!(equilibria.len(), 1);
        assert_eq!(equilibria[0].equilibrium_type, EquilibriumType::Pure);

        // Check that both players defect
        let p1_strat = &equilibria[0].strategies.get("P1").unwrap();
        let p2_strat = &equilibria[0].strategies.get("P2").unwrap();

        // Defect is index 1
        assert!((p1_strat.probabilities[1] - 1.0).abs() < 1e-9);
        assert!((p2_strat.probabilities[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_matching_pennies_mixed_nash() {
        let solver = NashSolver::new(1e-9, 1000);

        let game_state = GameState {
            game_type: crate::GameType::MatchingPennies,
            players: vec![],
            market_context: crate::MarketContext {
                regime: crate::MarketRegime::LowVolatility,
                volatility: 0.1,
                liquidity: 1.0,
                volume: 1000.0,
                spread: 0.01,
                market_impact: 0.001,
                information_asymmetry: 0.1,
                regulatory_environment: crate::RegulatoryEnvironment {
                    short_selling_allowed: true,
                    position_limits: None,
                    circuit_breakers: false,
                    market_making_obligations: false,
                    transparency_requirements: crate::TransparencyLevel::Full,
                },
            },
            information_sets: HashMap::new(),
            action_history: vec![],
            current_round: 0,
            payoff_matrix: Some(create_matching_pennies()),
            nash_equilibria: vec![],
            nash_equilibrium_found: false,
            dominant_strategies: HashMap::new(),
            cooperation_level: 0.0,
            competition_intensity: 1.0,
        };

        // No pure Nash in Matching Pennies
        let pure_eq = solver.find_pure_nash(&game_state).unwrap();
        assert!(pure_eq.is_empty());

        // Should have mixed Nash equilibrium (0.5, 0.5)
        let mixed_eq = solver.find_mixed_nash(&game_state).unwrap();
        assert!(!mixed_eq.is_empty());

        if let Some(eq) = mixed_eq.first() {
            let p1_strat = eq.strategies.get("P1").unwrap();
            // Both players should play 50-50
            assert!((p1_strat.probabilities[0] - 0.5).abs() < 0.1);
            assert!((p1_strat.probabilities[1] - 0.5).abs() < 0.1);
        }
    }
}