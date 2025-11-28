//! Game Theory Integration for CWTS Risk Management
//!
//! Implements Nash equilibrium analysis, multi-agent coordination, and
//! strategic risk assessment based on von Neumann & Morgenstern game theory.
//!
//! ## Key Concepts
//!
//! - **Nash Equilibrium**: Stable state where no player benefits from unilateral deviation
//! - **Multi-Agent Risk**: Strategic interactions between market participants
//! - **Machiavellian Tactics**: Adversarial modeling for worst-case analysis
//! - **Coalition Games**: Cooperative risk sharing structures
//!
//! ## Risk Applications
//!
//! - Identify dominant strategies of adversarial players
//! - Calculate optimal responses to market maker strategies
//! - Model whale manipulation risk
//! - Assess coalition formation impact on liquidity

use crate::core::{MarketRegime, RiskLevel, Symbol, Price};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

// Re-export from game-theory-engine when available
#[cfg(feature = "cwts-game-theory")]
use game_theory_engine::{
    GameState, Player, NashEquilibrium, Strategy, PlayerType, GameType,
    MixedStrategy, PayoffMatrix,
};

/// Nash equilibrium analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NashAnalysis {
    /// Identified equilibria
    pub equilibria: Vec<EquilibriumPoint>,
    /// Our optimal strategy
    pub optimal_strategy: OptimalStrategy,
    /// Stability score of current market state
    pub stability: f64,
    /// Expected payoff under optimal play
    pub expected_payoff: f64,
    /// Time to compute
    pub computation_time_us: u64,
}

/// A Nash equilibrium point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquilibriumPoint {
    /// Equilibrium type
    pub eq_type: EquilibriumType,
    /// Strategy profile
    pub strategies: HashMap<String, f64>,
    /// Associated payoffs
    pub payoffs: HashMap<String, f64>,
    /// Stability (0.0-1.0)
    pub stability: f64,
    /// Is Pareto optimal?
    pub is_pareto_optimal: bool,
}

/// Types of Nash equilibria
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EquilibriumType {
    /// Pure strategy equilibrium
    Pure,
    /// Mixed strategy equilibrium
    Mixed,
    /// Correlated equilibrium (requires mediator)
    Correlated,
    /// Evolutionarily stable strategy
    Evolutionary,
    /// Subgame perfect
    SubgamePerfect,
}

/// Our optimal strategy recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalStrategy {
    /// Strategy name
    pub name: String,
    /// Action probabilities
    pub actions: HashMap<String, f64>,
    /// Expected payoff
    pub expected_payoff: f64,
    /// Risk-adjusted payoff
    pub risk_adjusted_payoff: f64,
    /// Worst-case payoff (minimax)
    pub worst_case_payoff: f64,
    /// Confidence in strategy
    pub confidence: f64,
}

/// Strategic position in market game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategicPosition {
    /// Current player type classification
    pub player_type: MarketPlayerType,
    /// Estimated market power (0.0-1.0)
    pub market_power: f64,
    /// Information advantage (-1.0 to 1.0)
    pub information_advantage: f64,
    /// Exposed vulnerabilities
    pub vulnerabilities: Vec<Vulnerability>,
    /// Strategic options available
    pub available_strategies: Vec<StrategyOption>,
}

/// Market player types from game theory perspective
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketPlayerType {
    /// Price taker with no market power
    PriceTaker,
    /// Small strategic player
    SmallStrategic,
    /// Medium player with moderate influence
    MediumPlayer,
    /// Large player / whale
    Whale,
    /// Market maker
    MarketMaker,
    /// Arbitrageur
    Arbitrageur,
    /// Informed trader
    InformedTrader,
    /// Noise trader
    NoiseTrader,
}

impl MarketPlayerType {
    /// Get risk multiplier for this player type
    #[must_use]
    pub fn risk_multiplier(&self) -> f64 {
        match self {
            Self::PriceTaker => 1.0,
            Self::SmallStrategic => 1.1,
            Self::MediumPlayer => 1.2,
            Self::Whale => 0.8, // Whales have market power to reduce risk
            Self::MarketMaker => 0.9,
            Self::Arbitrageur => 0.7,
            Self::InformedTrader => 0.85,
            Self::NoiseTrader => 1.5,
        }
    }
}

/// Vulnerability in strategic position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    /// Vulnerability type
    pub vuln_type: VulnerabilityType,
    /// Severity (0.0-1.0)
    pub severity: f64,
    /// Potential loss if exploited
    pub potential_loss: f64,
    /// Mitigation strategy
    pub mitigation: String,
}

/// Types of strategic vulnerabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VulnerabilityType {
    /// Predictable order flow
    PredictableFlow,
    /// Large visible position
    PositionExposure,
    /// Concentrated liquidity need
    LiquidityNeed,
    /// Stop loss hunting target
    StopLossTarget,
    /// Correlation squeeze exposure
    CorrelationSqueeze,
    /// Gamma exposure
    GammaExposure,
    /// Funding rate exposure
    FundingExposure,
}

/// Strategic option available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyOption {
    /// Strategy name
    pub name: String,
    /// Expected payoff
    pub expected_payoff: f64,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Required conditions
    pub conditions: Vec<String>,
}

/// Multi-agent risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentRisk {
    /// Total strategic risk from other players
    pub total_strategic_risk: f64,
    /// Risk from each player type
    pub risk_by_player_type: HashMap<String, f64>,
    /// Adversarial scenarios
    pub adversarial_scenarios: Vec<AdversarialScenario>,
    /// Coalition risk
    pub coalition_risk: CoalitionRisk,
    /// Recommended defensive posture
    pub defensive_posture: DefensivePosture,
}

/// Adversarial scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialScenario {
    /// Scenario name
    pub name: String,
    /// Adversary type
    pub adversary: MarketPlayerType,
    /// Attack vector
    pub attack_vector: AttackVector,
    /// Probability
    pub probability: f64,
    /// Impact if occurs
    pub impact: f64,
    /// Defense strategy
    pub defense: String,
}

/// Attack vectors from game theory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttackVector {
    /// Front-running
    FrontRunning,
    /// Sandwich attack
    Sandwich,
    /// Spoofing / layering
    Spoofing,
    /// Stop hunting
    StopHunting,
    /// Liquidity squeeze
    LiquiditySqueeze,
    /// Information asymmetry exploitation
    InfoExploitation,
    /// Correlation manipulation
    CorrelationManipulation,
}

/// Coalition formation risk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoalitionRisk {
    /// Probability of adverse coalition forming
    pub formation_probability: f64,
    /// Potential coalition members
    pub potential_members: Vec<String>,
    /// Coalition bargaining power
    pub coalition_power: f64,
    /// Shapley value (our share if we join)
    pub shapley_value: f64,
}

/// Defensive posture recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DefensivePosture {
    /// Normal operations
    Normal,
    /// Heightened awareness
    Heightened,
    /// Active defense (hedging, diversification)
    ActiveDefense,
    /// Retreat (reduce exposure)
    Retreat,
    /// Fortress (minimal exposure, maximum protection)
    Fortress,
}

impl DefensivePosture {
    /// Convert to risk level
    #[must_use]
    pub fn to_risk_level(&self) -> RiskLevel {
        match self {
            Self::Normal => RiskLevel::Normal,
            Self::Heightened => RiskLevel::Elevated,
            Self::ActiveDefense => RiskLevel::High,
            Self::Retreat => RiskLevel::High,
            Self::Fortress => RiskLevel::Critical,
        }
    }
}

/// Configuration for game theory risk adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameTheoryConfig {
    /// Enable Machiavellian analysis
    pub enable_machiavellian: bool,
    /// Nash solver tolerance
    pub nash_tolerance: f64,
    /// Maximum solver iterations
    pub max_iterations: u32,
    /// Adversarial modeling depth
    pub adversarial_depth: u32,
    /// Coalition game analysis
    pub enable_coalition_analysis: bool,
    /// Risk aversion parameter (0.0 = risk neutral, 1.0 = very risk averse)
    pub risk_aversion: f64,
}

impl Default for GameTheoryConfig {
    fn default() -> Self {
        Self {
            enable_machiavellian: true,
            nash_tolerance: 1e-6,
            max_iterations: 1000,
            adversarial_depth: 3,
            enable_coalition_analysis: true,
            risk_aversion: 0.5,
        }
    }
}

/// Game theory based risk adapter
///
/// Integrates game theory into risk management:
/// - Solves Nash equilibria for market games
/// - Models adversarial player strategies
/// - Assesses coalition formation risk
/// - Recommends strategic defensive postures
pub struct GameTheoryRiskAdapter {
    config: GameTheoryConfig,
    current_position: RwLock<StrategicPosition>,
    nash_cache: RwLock<HashMap<u64, NashAnalysis>>,
    adversary_models: RwLock<HashMap<String, AdversaryModel>>,
    analysis_counter: AtomicU64,
}

/// Internal adversary model
#[derive(Debug, Clone)]
struct AdversaryModel {
    player_type: MarketPlayerType,
    estimated_capital: f64,
    aggression_level: f64,
    historical_actions: Vec<String>,
    predicted_strategy: String,
}

impl GameTheoryRiskAdapter {
    /// Create a new game theory risk adapter
    #[must_use]
    pub fn new(config: GameTheoryConfig) -> Self {
        Self {
            config,
            current_position: RwLock::new(StrategicPosition {
                player_type: MarketPlayerType::SmallStrategic,
                market_power: 0.01,
                information_advantage: 0.0,
                vulnerabilities: Vec::new(),
                available_strategies: Vec::new(),
            }),
            nash_cache: RwLock::new(HashMap::new()),
            adversary_models: RwLock::new(HashMap::new()),
            analysis_counter: AtomicU64::new(0),
        }
    }

    /// Get current strategic position
    #[must_use]
    pub fn position(&self) -> StrategicPosition {
        self.current_position.read().clone()
    }

    /// Analyze Nash equilibrium for current market situation
    pub fn analyze_nash(
        &self,
        market_state: &MarketState,
        our_position_size: f64,
        our_capital: f64,
    ) -> NashAnalysis {
        let start = std::time::Instant::now();

        // Construct payoff matrix
        let payoff_matrix = self.construct_payoff_matrix(market_state, our_position_size);

        // Find equilibria
        let equilibria = self.find_equilibria(&payoff_matrix);

        // Calculate optimal strategy
        let optimal = self.calculate_optimal_strategy(&equilibria, our_capital);

        // Calculate stability
        let stability = self.calculate_market_stability(&equilibria, market_state);

        NashAnalysis {
            equilibria,
            optimal_strategy: optimal.clone(),
            stability,
            expected_payoff: optimal.expected_payoff,
            computation_time_us: start.elapsed().as_micros() as u64,
        }
    }

    /// Construct payoff matrix for market game
    fn construct_payoff_matrix(
        &self,
        market_state: &MarketState,
        our_position: f64,
    ) -> PayoffMatrixInternal {
        // Simplified 2-player game: Us vs. Market
        // Strategies: Aggressive, Neutral, Defensive

        let volatility = market_state.volatility;
        let spread = market_state.spread;

        // Our payoffs depend on market response
        PayoffMatrixInternal {
            our_strategies: vec!["Aggressive", "Neutral", "Defensive"],
            market_strategies: vec!["Favorable", "Neutral", "Adverse"],
            payoffs: vec![
                // Aggressive vs Favorable, Neutral, Adverse
                vec![(0.15 - spread, 0.05), (0.08 - spread, -0.02), (-0.10, -0.15)],
                // Neutral vs Favorable, Neutral, Adverse
                vec![(0.08, 0.03), (0.04, 0.0), (-0.03, -0.05)],
                // Defensive vs Favorable, Neutral, Adverse
                vec![(0.03, 0.01), (0.02, 0.01), (-0.01, -0.02)],
            ],
            volatility_adjustment: volatility,
        }
    }

    /// Find Nash equilibria
    fn find_equilibria(&self, payoff_matrix: &PayoffMatrixInternal) -> Vec<EquilibriumPoint> {
        let mut equilibria = Vec::new();

        // Check for pure strategy equilibria (simplified)
        // In a real implementation, this would use the full Nash solver

        // For now, find best response for each strategy combination
        for (i, our_strat) in payoff_matrix.our_strategies.iter().enumerate() {
            for (j, market_strat) in payoff_matrix.market_strategies.iter().enumerate() {
                let (our_payoff, market_payoff) = payoff_matrix.payoffs[i][j];

                // Check if this is a Nash equilibrium (no beneficial deviation)
                let is_equilibrium = self.is_nash_equilibrium(
                    payoff_matrix,
                    i,
                    j,
                    our_payoff,
                    market_payoff,
                );

                if is_equilibrium {
                    equilibria.push(EquilibriumPoint {
                        eq_type: EquilibriumType::Pure,
                        strategies: HashMap::from([
                            ("us".to_string(), i as f64),
                            ("market".to_string(), j as f64),
                        ]),
                        payoffs: HashMap::from([
                            ("us".to_string(), our_payoff),
                            ("market".to_string(), market_payoff),
                        ]),
                        stability: 0.8,
                        is_pareto_optimal: self.is_pareto_optimal(payoff_matrix, i, j),
                    });
                }
            }
        }

        // If no pure equilibrium, compute mixed strategy equilibrium
        if equilibria.is_empty() {
            equilibria.push(self.compute_mixed_equilibrium(payoff_matrix));
        }

        equilibria
    }

    /// Check if strategy profile is Nash equilibrium
    fn is_nash_equilibrium(
        &self,
        matrix: &PayoffMatrixInternal,
        our_idx: usize,
        market_idx: usize,
        our_payoff: f64,
        market_payoff: f64,
    ) -> bool {
        // Check if we can improve by deviating
        for i in 0..matrix.our_strategies.len() {
            if i != our_idx && matrix.payoffs[i][market_idx].0 > our_payoff {
                return false;
            }
        }

        // Check if market can improve by deviating
        for j in 0..matrix.market_strategies.len() {
            if j != market_idx && matrix.payoffs[our_idx][j].1 > market_payoff {
                return false;
            }
        }

        true
    }

    /// Check if outcome is Pareto optimal
    fn is_pareto_optimal(&self, matrix: &PayoffMatrixInternal, our_idx: usize, market_idx: usize) -> bool {
        let (our_payoff, market_payoff) = matrix.payoffs[our_idx][market_idx];

        for i in 0..matrix.our_strategies.len() {
            for j in 0..matrix.market_strategies.len() {
                let (other_our, other_market) = matrix.payoffs[i][j];
                // Check if there's a Pareto improvement
                if other_our >= our_payoff && other_market >= market_payoff &&
                   (other_our > our_payoff || other_market > market_payoff) {
                    return false;
                }
            }
        }

        true
    }

    /// Compute mixed strategy equilibrium
    fn compute_mixed_equilibrium(&self, matrix: &PayoffMatrixInternal) -> EquilibriumPoint {
        // Simplified: assume uniform mixed strategy
        let n_our = matrix.our_strategies.len() as f64;
        let n_market = matrix.market_strategies.len() as f64;

        let mut strategies = HashMap::new();
        for (i, strat) in matrix.our_strategies.iter().enumerate() {
            strategies.insert(format!("us_{strat}"), 1.0 / n_our);
        }
        for (i, strat) in matrix.market_strategies.iter().enumerate() {
            strategies.insert(format!("market_{strat}"), 1.0 / n_market);
        }

        // Calculate expected payoff under uniform mixing
        let mut expected_us = 0.0;
        let mut expected_market = 0.0;
        for row in &matrix.payoffs {
            for (our_p, market_p) in row {
                expected_us += our_p / (n_our * n_market);
                expected_market += market_p / (n_our * n_market);
            }
        }

        EquilibriumPoint {
            eq_type: EquilibriumType::Mixed,
            strategies,
            payoffs: HashMap::from([
                ("us".to_string(), expected_us),
                ("market".to_string(), expected_market),
            ]),
            stability: 0.5,
            is_pareto_optimal: false,
        }
    }

    /// Calculate optimal strategy
    fn calculate_optimal_strategy(
        &self,
        equilibria: &[EquilibriumPoint],
        capital: f64,
    ) -> OptimalStrategy {
        // Select equilibrium with best risk-adjusted payoff for us
        let best = equilibria.iter()
            .max_by(|a, b| {
                let a_payoff = a.payoffs.get("us").unwrap_or(&0.0);
                let b_payoff = b.payoffs.get("us").unwrap_or(&0.0);
                a_payoff.partial_cmp(b_payoff).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let expected_payoff = *best.payoffs.get("us").unwrap_or(&0.0);

        // Calculate minimax (worst case)
        let worst_case = equilibria.iter()
            .filter_map(|eq| eq.payoffs.get("us"))
            .fold(f64::MAX, |a, &b| a.min(b));

        // Risk adjustment using CRRA utility
        let risk_adj = expected_payoff - self.config.risk_aversion * (expected_payoff - worst_case).abs();

        OptimalStrategy {
            name: self.strategy_name_from_equilibrium(best),
            actions: best.strategies.clone(),
            expected_payoff,
            risk_adjusted_payoff: risk_adj,
            worst_case_payoff: worst_case,
            confidence: best.stability,
        }
    }

    /// Get strategy name from equilibrium
    fn strategy_name_from_equilibrium(&self, eq: &EquilibriumPoint) -> String {
        match eq.eq_type {
            EquilibriumType::Pure => {
                let our_idx = eq.strategies.get("us").unwrap_or(&0.0).round() as usize;
                match our_idx {
                    0 => "Aggressive".to_string(),
                    1 => "Neutral".to_string(),
                    _ => "Defensive".to_string(),
                }
            }
            EquilibriumType::Mixed => "Mixed_Equilibrium".to_string(),
            _ => "Strategic".to_string(),
        }
    }

    /// Calculate market stability
    fn calculate_market_stability(&self, equilibria: &[EquilibriumPoint], state: &MarketState) -> f64 {
        // Stability based on:
        // 1. Number of equilibria (more = less stable)
        // 2. Stability of individual equilibria
        // 3. Market conditions

        let eq_stability = equilibria.iter()
            .map(|eq| eq.stability)
            .sum::<f64>() / equilibria.len().max(1) as f64;

        let volatility_penalty = state.volatility.min(1.0) * 0.3;

        (eq_stability - volatility_penalty).clamp(0.0, 1.0)
    }

    /// Assess multi-agent risk
    pub fn assess_multi_agent_risk(
        &self,
        market_state: &MarketState,
        detected_players: &[(String, MarketPlayerType, f64)], // (id, type, estimated_size)
    ) -> MultiAgentRisk {
        let mut risk_by_type: HashMap<String, f64> = HashMap::new();
        let mut adversarial_scenarios = Vec::new();

        for (id, player_type, size) in detected_players {
            // Calculate risk contribution
            let type_risk = self.calculate_player_type_risk(*player_type, *size, market_state);
            *risk_by_type.entry(format!("{player_type:?}")).or_default() += type_risk;

            // Generate adversarial scenarios for significant players
            if *size > 0.05 { // > 5% market share
                adversarial_scenarios.extend(
                    self.generate_adversarial_scenarios(*player_type, *size)
                );
            }
        }

        let total_risk = risk_by_type.values().sum();

        // Assess coalition risk if enabled
        let coalition_risk = if self.config.enable_coalition_analysis {
            self.assess_coalition_risk(detected_players)
        } else {
            CoalitionRisk {
                formation_probability: 0.0,
                potential_members: Vec::new(),
                coalition_power: 0.0,
                shapley_value: 0.0,
            }
        };

        // Determine defensive posture
        let defensive_posture = self.recommend_defensive_posture(total_risk, &coalition_risk);

        MultiAgentRisk {
            total_strategic_risk: total_risk,
            risk_by_player_type: risk_by_type,
            adversarial_scenarios,
            coalition_risk,
            defensive_posture,
        }
    }

    /// Calculate risk from specific player type
    fn calculate_player_type_risk(
        &self,
        player_type: MarketPlayerType,
        size: f64,
        state: &MarketState,
    ) -> f64 {
        let base_risk = match player_type {
            MarketPlayerType::Whale => size * 2.0, // Whales have outsized impact
            MarketPlayerType::MarketMaker => size * 0.5, // MMs generally stabilize
            MarketPlayerType::Arbitrageur => size * 0.3, // Arbs reduce inefficiency
            MarketPlayerType::InformedTrader => size * 1.5, // Adverse selection risk
            MarketPlayerType::NoiseTrader => size * 0.2, // Noise is less strategic
            _ => size,
        };

        // Adjust for market conditions
        base_risk * (1.0 + state.volatility)
    }

    /// Generate adversarial scenarios for a player
    fn generate_adversarial_scenarios(
        &self,
        player_type: MarketPlayerType,
        size: f64,
    ) -> Vec<AdversarialScenario> {
        let mut scenarios = Vec::new();

        match player_type {
            MarketPlayerType::Whale => {
                scenarios.push(AdversarialScenario {
                    name: "Whale Dump".to_string(),
                    adversary: player_type,
                    attack_vector: AttackVector::LiquiditySqueeze,
                    probability: 0.1 * size,
                    impact: size * 2.0,
                    defense: "Reduce exposure, increase stops".to_string(),
                });
                scenarios.push(AdversarialScenario {
                    name: "Stop Hunt".to_string(),
                    adversary: player_type,
                    attack_vector: AttackVector::StopHunting,
                    probability: 0.15 * size,
                    impact: size * 1.5,
                    defense: "Use hidden stops, wider ranges".to_string(),
                });
            }
            MarketPlayerType::MarketMaker => {
                scenarios.push(AdversarialScenario {
                    name: "Spread Widening".to_string(),
                    adversary: player_type,
                    attack_vector: AttackVector::Spoofing,
                    probability: 0.2,
                    impact: size * 0.5,
                    defense: "Limit order usage, TWAP execution".to_string(),
                });
            }
            MarketPlayerType::InformedTrader => {
                scenarios.push(AdversarialScenario {
                    name: "Info Exploitation".to_string(),
                    adversary: player_type,
                    attack_vector: AttackVector::InfoExploitation,
                    probability: 0.3 * size,
                    impact: size * 1.2,
                    defense: "Improve signal latency, diversify sources".to_string(),
                });
            }
            _ => {}
        }

        scenarios
    }

    /// Assess coalition formation risk
    fn assess_coalition_risk(
        &self,
        players: &[(String, MarketPlayerType, f64)],
    ) -> CoalitionRisk {
        // Identify potential coalition members
        let whales: Vec<_> = players.iter()
            .filter(|(_, t, s)| matches!(t, MarketPlayerType::Whale) && *s > 0.03)
            .collect();

        let mms: Vec<_> = players.iter()
            .filter(|(_, t, _)| matches!(t, MarketPlayerType::MarketMaker))
            .collect();

        // Coalition probability increases with number of large players
        let formation_prob = (whales.len() as f64 * 0.1 + mms.len() as f64 * 0.05).min(0.8);

        // Coalition power is sum of member sizes
        let coalition_power: f64 = whales.iter().map(|(_, _, s)| s).sum();

        // Simplified Shapley value calculation
        let total_value = 1.0;
        let our_contribution = 0.01; // Assume small player
        let shapley = our_contribution / (coalition_power + our_contribution) * total_value;

        CoalitionRisk {
            formation_probability: formation_prob,
            potential_members: whales.iter().map(|(id, _, _)| id.clone()).collect(),
            coalition_power,
            shapley_value: shapley,
        }
    }

    /// Recommend defensive posture based on risk assessment
    fn recommend_defensive_posture(
        &self,
        total_risk: f64,
        coalition_risk: &CoalitionRisk,
    ) -> DefensivePosture {
        let combined_risk = total_risk + coalition_risk.formation_probability * coalition_risk.coalition_power;

        if combined_risk > 0.8 {
            DefensivePosture::Fortress
        } else if combined_risk > 0.6 {
            DefensivePosture::Retreat
        } else if combined_risk > 0.4 {
            DefensivePosture::ActiveDefense
        } else if combined_risk > 0.2 {
            DefensivePosture::Heightened
        } else {
            DefensivePosture::Normal
        }
    }

    /// Get risk level from game theory analysis
    #[must_use]
    pub fn risk_level(&self) -> RiskLevel {
        let position = self.current_position.read();
        let vuln_risk: f64 = position.vulnerabilities.iter()
            .map(|v| v.severity)
            .sum::<f64>() / position.vulnerabilities.len().max(1) as f64;

        if vuln_risk > 0.8 {
            RiskLevel::Critical
        } else if vuln_risk > 0.6 {
            RiskLevel::High
        } else if vuln_risk > 0.3 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        }
    }

    /// Update strategic position
    pub fn update_position(&self, position: StrategicPosition) {
        *self.current_position.write() = position;
    }

    /// Assess strategic risk and return SubsystemRisk for coordinator.
    #[must_use]
    pub fn assess_strategic_risk(&self, portfolio: &crate::core::Portfolio) -> super::coordinator::SubsystemRisk {
        use super::coordinator::{SubsystemRisk, SubsystemId};
        use crate::core::Timestamp;

        let start = std::time::Instant::now();

        // Estimate market state from portfolio
        let drawdown_pct = portfolio.drawdown_pct();
        let volatility = (drawdown_pct / 100.0).abs() * 0.8 + 0.1;
        let market_state = MarketState {
            volatility: volatility.clamp(0.05, 0.5),
            spread: 0.001,
            depth: portfolio.total_value / 10.0,
            info_asymmetry: 0.15,
        };

        // Calculate position size from portfolio
        let position_size = portfolio.positions.iter()
            .map(|p| (p.quantity.as_f64() * p.current_price.as_f64()).abs())
            .sum::<f64>();

        // Run Nash analysis
        let nash = self.analyze_nash(&market_state, position_size, portfolio.total_value);

        // Calculate multi-agent risk
        let players = vec![
            ("market_maker".to_string(), MarketPlayerType::MarketMaker, 0.05),
            ("whale".to_string(), MarketPlayerType::Whale, 0.08),
        ];
        let multi_risk = self.assess_multi_agent_risk(&market_state, &players);

        let latency_ns = start.elapsed().as_nanos() as u64;

        // Determine risk level from Nash stability and multi-agent risk
        let combined_risk = (1.0 - nash.stability) * 0.5 + multi_risk.total_strategic_risk * 0.5;
        let risk_level = if combined_risk > 0.8 {
            RiskLevel::Critical
        } else if combined_risk > 0.6 {
            RiskLevel::High
        } else if combined_risk > 0.3 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        };

        let confidence = nash.stability;
        let position_factor = (1.0 - combined_risk * 0.8).clamp(0.3, 1.0);

        SubsystemRisk {
            subsystem: SubsystemId::GameTheory,
            risk_level,
            confidence,
            risk_score: combined_risk,
            position_factor,
            reasoning: format!(
                "Game Theory: Nash stability={:.2}, Multi-agent risk={:.2}",
                nash.stability, multi_risk.total_strategic_risk
            ),
            timestamp: Timestamp::now(),
            latency_ns,
        }
    }
}

impl Default for GameTheoryRiskAdapter {
    fn default() -> Self {
        Self::new(GameTheoryConfig::default())
    }
}

/// Internal payoff matrix representation
struct PayoffMatrixInternal {
    our_strategies: Vec<&'static str>,
    market_strategies: Vec<&'static str>,
    payoffs: Vec<Vec<(f64, f64)>>, // (our_payoff, market_payoff)
    volatility_adjustment: f64,
}

/// Market state for game theory analysis
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Current volatility
    pub volatility: f64,
    /// Bid-ask spread
    pub spread: f64,
    /// Market depth
    pub depth: f64,
    /// Information asymmetry estimate
    pub info_asymmetry: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defensive_posture_risk_levels() {
        assert_eq!(DefensivePosture::Normal.to_risk_level(), RiskLevel::Normal);
        assert_eq!(DefensivePosture::Fortress.to_risk_level(), RiskLevel::Critical);
    }

    #[test]
    fn test_player_type_risk_multiplier() {
        assert!(MarketPlayerType::Whale.risk_multiplier() < 1.0);
        assert!(MarketPlayerType::NoiseTrader.risk_multiplier() > 1.0);
    }

    #[test]
    fn test_adapter_creation() {
        let adapter = GameTheoryRiskAdapter::default();
        let position = adapter.position();
        assert_eq!(position.player_type, MarketPlayerType::SmallStrategic);
    }

    #[test]
    fn test_nash_analysis() {
        let adapter = GameTheoryRiskAdapter::default();

        let market_state = MarketState {
            volatility: 0.2,
            spread: 0.001,
            depth: 1000.0,
            info_asymmetry: 0.1,
        };

        let analysis = adapter.analyze_nash(&market_state, 100.0, 10000.0);

        assert!(!analysis.equilibria.is_empty());
        assert!(analysis.stability >= 0.0 && analysis.stability <= 1.0);
    }

    #[test]
    fn test_multi_agent_risk() {
        let adapter = GameTheoryRiskAdapter::default();

        let market_state = MarketState {
            volatility: 0.3,
            spread: 0.002,
            depth: 500.0,
            info_asymmetry: 0.2,
        };

        let players = vec![
            ("whale1".to_string(), MarketPlayerType::Whale, 0.1),
            ("mm1".to_string(), MarketPlayerType::MarketMaker, 0.05),
        ];

        let risk = adapter.assess_multi_agent_risk(&market_state, &players);

        assert!(risk.total_strategic_risk > 0.0);
        assert!(!risk.adversarial_scenarios.is_empty());
    }

    #[test]
    fn test_coalition_risk() {
        let adapter = GameTheoryRiskAdapter::default();

        let players = vec![
            ("whale1".to_string(), MarketPlayerType::Whale, 0.15),
            ("whale2".to_string(), MarketPlayerType::Whale, 0.10),
        ];

        let coalition_risk = adapter.assess_coalition_risk(&players);

        assert!(coalition_risk.formation_probability > 0.0);
        assert!(!coalition_risk.potential_members.is_empty());
    }
}
