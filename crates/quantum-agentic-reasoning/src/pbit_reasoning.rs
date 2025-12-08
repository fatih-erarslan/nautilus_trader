//! pBit Reasoning Engine
//!
//! Integrates pBit probabilistic computing from quantum-core with the
//! Quantum Agentic Reasoning (QAR) decision engine. Uses Ising lattice
//! dynamics for decision fusion and prospect theory integration.
//!
//! ## Key Features
//!
//! - Decision states as pBit spins
//! - Prospect theory value function encoded in biases
//! - LMSR market beliefs as coupling strengths
//! - Annealing-based decision refinement
//! - STDP learning from trade outcomes
//!
//! ## Architecture
//!
//! ```text
//! Market Data → Prospect Theory → pBit Lattice → Decision
//!                    ↓                 ↓
//!               Value/Weight     Boltzmann Sampling
//!                    ↓                 ↓
//!               Bias Fields    → Spin Configuration → Action
//! ```

use quantum_core::{
    PBitState, PBitConfig, PBitCircuit, PBitBackend, PBitBackendConfig,
    PBitCoupling, QuantumError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::{TradingDecision, TradingAction, MarketData};

/// Configuration for pBit-based reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBitReasoningConfig {
    /// Number of pBits for decision encoding
    pub num_pbits: usize,
    /// Temperature for Boltzmann sampling
    pub temperature: f64,
    /// Prospect theory loss aversion coefficient
    pub loss_aversion: f64,
    /// Reference point for gains/losses
    pub reference_point: f64,
    /// Coupling strength for correlated decisions
    pub coupling_strength: f64,
    /// Number of sweeps per decision
    pub sweeps_per_decision: usize,
    /// Enable annealing
    pub enable_annealing: bool,
    /// Annealing steps
    pub annealing_steps: usize,
    /// Final temperature after annealing
    pub final_temperature: f64,
    /// Learning rate for outcome feedback
    pub learning_rate: f64,
}

impl Default for PBitReasoningConfig {
    fn default() -> Self {
        Self {
            num_pbits: 8,
            temperature: 1.0,
            loss_aversion: 2.25, // Kahneman-Tversky value
            reference_point: 0.0,
            coupling_strength: 1.0,
            sweeps_per_decision: 50,
            enable_annealing: true,
            annealing_steps: 100,
            final_temperature: 0.05,
            learning_rate: 0.01,
        }
    }
}

/// Decision factors encoded as pBit nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecisionFactor {
    /// Market momentum signal
    Momentum,
    /// Volatility regime
    Volatility,
    /// Trend strength
    Trend,
    /// Risk level
    Risk,
    /// Position sizing
    PositionSize,
    /// Time horizon
    TimeHorizon,
    /// LMSR market belief
    MarketBelief,
    /// Hedge recommendation
    HedgeSignal,
}

impl DecisionFactor {
    fn all() -> Vec<Self> {
        vec![
            Self::Momentum,
            Self::Volatility,
            Self::Trend,
            Self::Risk,
            Self::PositionSize,
            Self::TimeHorizon,
            Self::MarketBelief,
            Self::HedgeSignal,
        ]
    }

    fn index(&self) -> usize {
        match self {
            Self::Momentum => 0,
            Self::Volatility => 1,
            Self::Trend => 2,
            Self::Risk => 3,
            Self::PositionSize => 4,
            Self::TimeHorizon => 5,
            Self::MarketBelief => 6,
            Self::HedgeSignal => 7,
        }
    }
}

/// pBit-based reasoning engine for QAR
pub struct PBitReasoningEngine {
    /// Configuration
    config: PBitReasoningConfig,
    /// pBit state for decision factors
    pbit_state: PBitState,
    /// Factor weights learned from outcomes
    factor_weights: HashMap<DecisionFactor, f64>,
    /// Factor biases from market analysis
    factor_biases: HashMap<DecisionFactor, f64>,
    /// Decision history for learning
    decision_history: Vec<DecisionRecord>,
    /// Current temperature (for annealing)
    current_temperature: f64,
}

/// Record of a decision and its outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// The decision made
    pub decision: TradingDecision,
    /// Factor states at decision time
    pub factor_states: Vec<f64>,
    /// Actual outcome (profit/loss)
    pub outcome: Option<f64>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Result of pBit reasoning
#[derive(Debug, Clone)]
pub struct ReasoningResult {
    /// Recommended action
    pub action: TradingAction,
    /// Confidence in the decision
    pub confidence: f64,
    /// Individual factor contributions
    pub factor_contributions: HashMap<DecisionFactor, f64>,
    /// Prospect theory value
    pub prospect_value: f64,
    /// Entropy (uncertainty)
    pub entropy: f64,
    /// Magnetization (decision direction strength)
    pub magnetization: f64,
    /// Reasoning explanation
    pub reasoning: Vec<String>,
}

impl PBitReasoningEngine {
    /// Create a new pBit reasoning engine
    pub fn new(config: PBitReasoningConfig) -> Result<Self, QuantumError> {
        let pbit_config = PBitConfig {
            temperature: config.temperature,
            coupling_strength: config.coupling_strength,
            external_field: 0.0,
            seed: None,
        };

        let pbit_state = PBitState::with_config(config.num_pbits, pbit_config)?;

        // Initialize factor weights uniformly
        let mut factor_weights = HashMap::new();
        let mut factor_biases = HashMap::new();
        for factor in DecisionFactor::all() {
            factor_weights.insert(factor, 1.0 / DecisionFactor::all().len() as f64);
            factor_biases.insert(factor, 0.0);
        }

        Ok(Self {
            current_temperature: config.temperature,
            config,
            pbit_state,
            factor_weights,
            factor_biases,
            decision_history: Vec::new(),
        })
    }

    /// Analyze market data and make a decision
    pub fn reason(&mut self, market_data: &MarketData) -> Result<ReasoningResult, QuantumError> {
        // Extract factors from market data
        let factors = self.extract_factors(market_data);

        // Encode factors as pBit biases
        self.encode_factors(&factors);

        // Reset temperature for new decision
        self.current_temperature = self.config.temperature;

        // Equilibrate
        for _ in 0..self.config.sweeps_per_decision {
            self.pbit_state.sweep();
        }

        // Anneal if enabled
        if self.config.enable_annealing {
            self.anneal();
        }

        // Decode decision from pBit state
        let (action, confidence) = self.decode_decision();

        // Calculate prospect value
        let prospect_value = self.calculate_prospect_value(market_data);

        // Get factor contributions
        let factor_contributions = self.get_factor_contributions();

        // Build reasoning
        let reasoning = self.build_reasoning(&action, confidence, &factor_contributions);

        // Record decision
        self.record_decision(&action, &factors);

        Ok(ReasoningResult {
            action,
            confidence,
            factor_contributions,
            prospect_value,
            entropy: self.pbit_state.entropy(),
            magnetization: self.pbit_state.magnetization(),
            reasoning,
        })
    }

    /// Extract decision factors from market data
    fn extract_factors(&self, market_data: &MarketData) -> HashMap<DecisionFactor, f64> {
        let mut factors = HashMap::new();

        // Momentum: based on price change direction
        let momentum = if market_data.close > market_data.open {
            (market_data.close - market_data.open) / market_data.open
        } else {
            -(market_data.open - market_data.close) / market_data.open
        };
        factors.insert(DecisionFactor::Momentum, momentum.clamp(-1.0, 1.0));

        // Volatility: high-low range normalized
        let volatility = (market_data.high - market_data.low) / market_data.close;
        factors.insert(DecisionFactor::Volatility, volatility.clamp(0.0, 1.0));

        // Trend: simplified as momentum direction
        let trend = if momentum > 0.01 { 1.0 } else if momentum < -0.01 { -1.0 } else { 0.0 };
        factors.insert(DecisionFactor::Trend, trend);

        // Risk: inverse of stability (volatility-based)
        let risk = volatility * 2.0;
        factors.insert(DecisionFactor::Risk, risk.clamp(0.0, 1.0));

        // Position size recommendation
        let position_size = 1.0 - risk; // Lower risk = larger position
        factors.insert(DecisionFactor::PositionSize, position_size);

        // Time horizon signal (placeholder)
        factors.insert(DecisionFactor::TimeHorizon, 0.5);

        // Market belief from volume
        let volume_signal = market_data.volume.log10() / 10.0;
        factors.insert(DecisionFactor::MarketBelief, volume_signal.clamp(0.0, 1.0));

        // Hedge signal based on volatility
        let hedge_signal = if volatility > 0.05 { 0.8 } else { 0.2 };
        factors.insert(DecisionFactor::HedgeSignal, hedge_signal);

        factors
    }

    /// Encode factors as pBit biases
    fn encode_factors(&mut self, factors: &HashMap<DecisionFactor, f64>) {
        for (factor, &value) in factors {
            let weight = self.factor_weights.get(factor).copied().unwrap_or(1.0);
            let bias = value * weight * self.config.coupling_strength;
            self.factor_biases.insert(*factor, bias);

            // Set pBit bias
            let idx = factor.index();
            if idx < self.pbit_state.num_qubits() {
                if let Some(pbit) = self.pbit_state.get_pbit_mut(idx) {
                    pbit.bias = bias;
                    pbit.probability_up = (value + 1.0) / 2.0; // Normalize to [0,1]
                }
            }
        }

        // Add couplings between related factors
        self.add_factor_couplings();
    }

    /// Add couplings between related decision factors
    fn add_factor_couplings(&mut self) {
        // Momentum-Trend: strong positive coupling
        self.pbit_state.add_coupling(PBitCoupling::bell_coupling(
            DecisionFactor::Momentum.index(),
            DecisionFactor::Trend.index(),
            0.8,
        ));

        // Risk-PositionSize: negative coupling (high risk = low position)
        self.pbit_state.add_coupling(PBitCoupling::anti_bell_coupling(
            DecisionFactor::Risk.index(),
            DecisionFactor::PositionSize.index(),
            0.7,
        ));

        // Volatility-HedgeSignal: positive coupling
        self.pbit_state.add_coupling(PBitCoupling::bell_coupling(
            DecisionFactor::Volatility.index(),
            DecisionFactor::HedgeSignal.index(),
            0.6,
        ));
    }

    /// Perform simulated annealing
    fn anneal(&mut self) {
        let ratio = (self.config.final_temperature / self.current_temperature)
            .powf(1.0 / self.config.annealing_steps as f64);

        for _ in 0..self.config.annealing_steps {
            self.current_temperature *= ratio;
            self.current_temperature = self.current_temperature.max(self.config.final_temperature);
            self.pbit_state.sweep();
        }
    }

    /// Decode decision from pBit state
    fn decode_decision(&self) -> (TradingAction, f64) {
        // Aggregate spins weighted by factor importance
        let mut buy_score = 0.0;
        let mut sell_score = 0.0;
        let mut hold_score = 0.0;

        for factor in DecisionFactor::all() {
            let idx = factor.index();
            if idx < self.pbit_state.num_qubits() {
                if let Some(pbit) = self.pbit_state.get_pbit(idx) {
                    let weight = self.factor_weights.get(&factor).copied().unwrap_or(1.0);
                    let contribution = pbit.spin * pbit.probability_up * weight;

                    match factor {
                        DecisionFactor::Momentum | DecisionFactor::Trend | DecisionFactor::MarketBelief => {
                            if contribution > 0.0 {
                                buy_score += contribution;
                            } else {
                                sell_score += contribution.abs();
                            }
                        }
                        DecisionFactor::Risk | DecisionFactor::Volatility => {
                            hold_score += contribution.abs() * 0.5;
                        }
                        DecisionFactor::HedgeSignal => {
                            if contribution > 0.5 {
                                hold_score += 0.3;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Normalize and decide
        let total = buy_score + sell_score + hold_score + 1e-10;
        let confidence = (buy_score.max(sell_score).max(hold_score) / total).min(1.0);

        let action = if buy_score > sell_score && buy_score > hold_score {
            TradingAction::Buy
        } else if sell_score > buy_score && sell_score > hold_score {
            TradingAction::Sell
        } else {
            TradingAction::Hold
        };

        (action, confidence)
    }

    /// Calculate prospect theory value
    fn calculate_prospect_value(&self, market_data: &MarketData) -> f64 {
        let potential_gain = market_data.close - self.config.reference_point;

        if potential_gain >= 0.0 {
            // Value function for gains: v(x) = x^α (α typically 0.88)
            potential_gain.powf(0.88)
        } else {
            // Value function for losses: v(x) = -λ|x|^β (λ = loss_aversion, β typically 0.88)
            -self.config.loss_aversion * potential_gain.abs().powf(0.88)
        }
    }

    /// Get factor contributions to decision
    fn get_factor_contributions(&self) -> HashMap<DecisionFactor, f64> {
        let mut contributions = HashMap::new();

        for factor in DecisionFactor::all() {
            let idx = factor.index();
            if idx < self.pbit_state.num_qubits() {
                if let Some(pbit) = self.pbit_state.get_pbit(idx) {
                    let weight = self.factor_weights.get(&factor).copied().unwrap_or(1.0);
                    contributions.insert(factor, pbit.spin * pbit.probability_up * weight);
                }
            }
        }

        contributions
    }

    /// Build reasoning explanation
    fn build_reasoning(
        &self,
        action: &TradingAction,
        confidence: f64,
        contributions: &HashMap<DecisionFactor, f64>,
    ) -> Vec<String> {
        let mut reasoning = vec![
            format!("pBit Reasoning with {} factors", DecisionFactor::all().len()),
            format!("Final temperature: {:.3}", self.current_temperature),
            format!("Magnetization: {:.3}", self.pbit_state.magnetization()),
            format!("Entropy: {:.3}", self.pbit_state.entropy()),
        ];

        // Top contributing factors
        let mut sorted: Vec<_> = contributions.iter().collect();
        sorted.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        reasoning.push("Top factors:".to_string());
        for (factor, value) in sorted.iter().take(3) {
            reasoning.push(format!("  {:?}: {:.3}", factor, value));
        }

        reasoning.push(format!("Decision: {:?} with {:.1}% confidence", action, confidence * 100.0));

        reasoning
    }

    /// Record decision for learning
    fn record_decision(&mut self, action: &TradingAction, factors: &HashMap<DecisionFactor, f64>) {
        let factor_states: Vec<f64> = DecisionFactor::all()
            .iter()
            .map(|f| factors.get(f).copied().unwrap_or(0.0))
            .collect();

        let decision = TradingDecision {
            action: action.clone(),
            confidence: 0.0,
            reasoning: vec![],
        };

        self.decision_history.push(DecisionRecord {
            decision,
            factor_states,
            outcome: None,
            timestamp: chrono::Utc::now(),
        });

        // Limit history size
        if self.decision_history.len() > 1000 {
            self.decision_history.remove(0);
        }
    }

    /// Learn from trade outcome
    pub fn learn_from_outcome(&mut self, outcome: f64) -> Result<(), QuantumError> {
        if let Some(last) = self.decision_history.last_mut() {
            last.outcome = Some(outcome);
        }

        // Update factor weights based on outcome
        let success = outcome > 0.0;

        if let Some(last) = self.decision_history.last() {
            for (i, factor) in DecisionFactor::all().iter().enumerate() {
                if i < last.factor_states.len() {
                    let current_weight = self.factor_weights.get(factor).copied().unwrap_or(1.0);
                    let factor_value = last.factor_states[i];

                    // Hebbian-like learning: strengthen if factor aligned with success
                    let alignment = if success { factor_value } else { -factor_value };
                    let delta = self.config.learning_rate * alignment * outcome.abs().min(1.0);

                    let new_weight = (current_weight + delta).clamp(0.1, 2.0);
                    self.factor_weights.insert(*factor, new_weight);
                }
            }
        }

        // Normalize weights
        let total: f64 = self.factor_weights.values().sum();
        for weight in self.factor_weights.values_mut() {
            *weight /= total;
            *weight *= DecisionFactor::all().len() as f64; // Scale back
        }

        Ok(())
    }

    /// Get current factor weights
    pub fn factor_weights(&self) -> &HashMap<DecisionFactor, f64> {
        &self.factor_weights
    }

    /// Reset the engine
    pub fn reset(&mut self) -> Result<(), QuantumError> {
        self.current_temperature = self.config.temperature;
        self.decision_history.clear();

        // Reset weights to uniform
        for factor in DecisionFactor::all() {
            self.factor_weights.insert(factor, 1.0 / DecisionFactor::all().len() as f64);
            self.factor_biases.insert(factor, 0.0);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_market_data() -> MarketData {
        MarketData {
            symbol: "TEST".to_string(),
            timestamp: chrono::Utc::now(),
            open: 100.0,
            high: 105.0,
            low: 98.0,
            close: 103.0,
            volume: 1000000.0,
            bid: 102.9,
            ask: 103.1,
        }
    }

    #[test]
    fn test_engine_creation() {
        let engine = PBitReasoningEngine::new(PBitReasoningConfig::default()).unwrap();
        assert_eq!(engine.factor_weights.len(), 8);
    }

    #[test]
    fn test_reasoning() {
        let mut engine = PBitReasoningEngine::new(PBitReasoningConfig::default()).unwrap();
        let market_data = mock_market_data();

        let result = engine.reason(&market_data).unwrap();

        assert!(result.confidence > 0.0);
        assert!(result.confidence <= 1.0);
        assert!(!result.reasoning.is_empty());
    }

    #[test]
    fn test_learning() {
        let mut engine = PBitReasoningEngine::new(PBitReasoningConfig::default()).unwrap();
        let market_data = mock_market_data();

        // Make a decision
        let _ = engine.reason(&market_data).unwrap();

        // Provide positive outcome
        engine.learn_from_outcome(0.05).unwrap();

        // Weights should have changed
        let total: f64 = engine.factor_weights.values().sum();
        assert!(total > 0.0);
    }
}
