//! Quantum-Enhanced Hedge Algorithms Implementation
//!
//! This module provides quantum computing enhancements to hedge algorithms including
//! multiplicative weights, adaptive learning, expert systems, and quantum portfolio optimization.

use quantum_core::{
    QuantumState, QuantumGate, QuantumCircuit, QuantumDevice, DeviceType,
    QuantumResult, QuantumError,
};
use crate::{MarketData};
// use nalgebra::{DVector, DMatrix};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{warn};
use chrono::{DateTime, Utc};
// use rand::Rng;
// use statrs::distribution::{Normal, Continuous, ContinuousCDF};

/// Quantum processing modes for hedge algorithms
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantumHedgeMode {
    /// Always use classical algorithms
    Classical,
    /// Use quantum algorithms when advantageous
    Quantum,
    /// Hybrid approach - use both quantum and classical
    Hybrid,
    /// Automatically select best approach
    Auto,
}

/// Configuration for quantum hedge algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumHedgeConfig {
    /// Quantum processing mode
    pub processing_mode: QuantumHedgeMode,
    /// Number of qubits for quantum circuits
    pub num_qubits: usize,
    /// Quantum circuit depth
    pub circuit_depth: usize,
    /// Quantum device type preference
    pub device_type: DeviceType,
    /// Enable quantum error correction
    pub enable_error_correction: bool,
    /// Learning rate for adaptive algorithms
    pub learning_rate: f64,
    /// Number of experts in the system
    pub num_experts: usize,
    /// Portfolio rebalancing frequency
    pub rebalance_frequency: u64,
    /// Risk tolerance level
    pub risk_tolerance: f64,
    /// Maximum leverage allowed
    pub max_leverage: f64,
    /// Enable quantum state caching
    pub enable_state_caching: bool,
    /// Cache size for quantum states
    pub cache_size: usize,
}

impl Default for QuantumHedgeConfig {
    fn default() -> Self {
        Self {
            processing_mode: QuantumHedgeMode::Auto,
            num_qubits: 8, // Standard 8-factor model
            circuit_depth: 5,
            device_type: DeviceType::Simulator,
            enable_error_correction: true,
            learning_rate: 0.1,
            num_experts: 16,
            rebalance_frequency: 24, // Hours
            risk_tolerance: 0.05,
            max_leverage: 3.0,
            enable_state_caching: true,
            cache_size: 1000,
        }
    }
}

/// Expert system for hedge algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumExpert {
    /// Unique expert identifier
    pub id: String,
    /// Expert name
    pub name: String,
    /// Expert weight in the ensemble
    pub weight: f64,
    /// Historical performance
    pub performance_history: Vec<f64>,
    /// Expert's current prediction
    pub current_prediction: f64,
    /// Expert's confidence level
    pub confidence: f64,
    /// Expert's specialization
    pub specialization: ExpertSpecialization,
    /// Number of correct predictions
    pub correct_predictions: u64,
    /// Total number of predictions
    pub total_predictions: u64,
    /// Expert's quantum state
    pub quantum_state: Option<QuantumState>,
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum parameter
    pub momentum: f64,
    /// Regularization parameter
    pub regularization: f64,
}

/// Expert specialization types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExpertSpecialization {
    /// Trend following expert
    TrendFollowing,
    /// Mean reversion expert
    MeanReversion,
    /// Volatility trading expert
    VolatilityTrading,
    /// Momentum expert
    Momentum,
    /// Sentiment analysis expert
    SentimentAnalysis,
    /// Liquidity provision expert
    LiquidityProvision,
    /// Correlation trading expert
    CorrelationTrading,
    /// Cycle analysis expert
    CycleAnalysis,
    /// Anomaly detection expert
    AnomalyDetection,
    /// Risk management expert
    RiskManagement,
    /// Options trading expert
    OptionsTrading,
    /// Pairs trading expert
    PairsTrading,
    /// Arbitrage expert
    ArbitrageExpert,
    /// High-frequency trading expert
    HighFrequencyTrading,
    /// Macro economic expert
    MacroEconomic,
    /// Technical analysis expert
    TechnicalAnalysis,
}

/// Market regime types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Bull market
    Bull,
    /// Bear market
    Bear,
    /// Sideways market
    Sideways,
    /// High volatility market
    HighVolatility,
    /// Low volatility market
    LowVolatility,
    /// Crisis market
    Crisis,
    /// Recovery market
    Recovery,
    /// Bubble market
    Bubble,
}

/// Hedge decision structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgeDecision {
    /// Decision timestamp
    pub timestamp: DateTime<Utc>,
    /// Action to take
    pub action: HedgeAction,
    /// Position size
    pub position_size: f64,
    /// Confidence level
    pub confidence: f64,
    /// Expected return
    pub expected_return: f64,
    /// Risk estimate
    pub risk_estimate: f64,
    /// Stop loss level
    pub stop_loss: Option<f64>,
    /// Take profit level
    pub take_profit: Option<f64>,
    /// Expert weights used
    pub expert_weights: HashMap<String, f64>,
    /// Market regime
    pub market_regime: MarketRegime,
}

/// Hedge action types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HedgeAction {
    /// Buy position
    Buy,
    /// Sell position
    Sell,
    /// Hold position
    Hold,
    /// Reduce position
    Reduce,
    /// Increase position
    Increase,
    /// Close position
    Close,
    /// Hedge position
    Hedge,
    /// Rebalance portfolio
    Rebalance,
}

/// Quantum-enhanced hedge algorithm implementation
pub struct QuantumHedgeAlgorithm {
    /// Configuration
    config: QuantumHedgeConfig,
    /// Quantum device
    quantum_device: Option<QuantumDevice>,
    /// Expert system
    experts: Vec<QuantumExpert>,
    /// Quantum state cache
    state_cache: Arc<Mutex<HashMap<String, QuantumState>>>,
    /// Quantum circuits cache
    circuit_cache: Arc<Mutex<HashMap<String, QuantumCircuit>>>,
    /// Performance metrics
    quantum_metrics: Arc<Mutex<QuantumHedgeMetrics>>,
    /// Market data history
    market_history: Vec<crate::MarketData>,
    /// Hedge decisions history
    decision_history: Vec<HedgeDecision>,
    /// Current market regime
    current_regime: MarketRegime,
    /// Portfolio state
    portfolio_state: PortfolioState,
}

/// Market data structure (using crate::MarketData)
// Note: Using the MarketData struct defined in lib.rs
// The MarketData struct is defined in lib.rs and imported via use crate::MarketData

/// Portfolio state
#[derive(Debug, Clone, Default)]
pub struct PortfolioState {
    /// Current positions
    pub positions: HashMap<String, f64>,
    /// Cash balance
    pub cash: f64,
    /// Total portfolio value
    pub portfolio_value: f64,
    /// Risk exposure
    pub risk_exposure: f64,
    /// Leverage
    pub leverage: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
}

/// Quantum hedge performance metrics
#[derive(Debug, Clone, Default)]
struct QuantumHedgeMetrics {
    quantum_executions: u64,
    classical_executions: u64,
    quantum_time_total_ms: u64,
    classical_time_total_ms: u64,
    quantum_errors: u64,
    expert_predictions: u64,
    hedge_decisions: u64,
    portfolio_rebalances: u64,
    risk_adjustments: u64,
}

impl QuantumHedgeAlgorithm {
    /// Create a new quantum hedge algorithm instance
    pub fn new(expert_names: Vec<String>, config: crate::HedgeConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Convert HedgeConfig to QuantumHedgeConfig
        let quantum_config = QuantumHedgeConfig {
            processing_mode: if config.enable_quantum { QuantumHedgeMode::Quantum } else { QuantumHedgeMode::Classical },
            num_qubits: config.quantum_config.num_qubits,
            circuit_depth: 5,
            device_type: quantum_core::DeviceType::Simulator,
            enable_error_correction: true,
            learning_rate: config.learning_rate,
            num_experts: 16,
            rebalance_frequency: 24,
            risk_tolerance: config.risk_tolerance,
            max_leverage: 3.0,
            enable_state_caching: true,
            cache_size: 1000,
        };
        // Initialize quantum device
        let quantum_device = match QuantumDevice::new_simple(quantum_config.device_type, quantum_config.num_qubits) {
            Ok(device) => Some(device),
            Err(e) => {
                warn!("Failed to initialize quantum device: {:?}", e);
                None
            }
        };

        // Initialize experts
        let mut experts = Vec::new();
        let specializations = [
            ExpertSpecialization::TrendFollowing,
            ExpertSpecialization::MeanReversion,
            ExpertSpecialization::VolatilityTrading,
            ExpertSpecialization::Momentum,
            ExpertSpecialization::SentimentAnalysis,
            ExpertSpecialization::LiquidityProvision,
            ExpertSpecialization::CorrelationTrading,
            ExpertSpecialization::CycleAnalysis,
            ExpertSpecialization::AnomalyDetection,
            ExpertSpecialization::RiskManagement,
            ExpertSpecialization::OptionsTrading,
            ExpertSpecialization::PairsTrading,
            ExpertSpecialization::ArbitrageExpert,
            ExpertSpecialization::HighFrequencyTrading,
            ExpertSpecialization::MacroEconomic,
            ExpertSpecialization::TechnicalAnalysis,
        ];

        for (i, name) in expert_names.iter().enumerate() {
            let specialization = specializations[i % specializations.len()];
            experts.push(QuantumExpert {
                id: format!("expert_{}", i),
                name: name.clone(),
                weight: 1.0 / expert_names.len() as f64,
                performance_history: Vec::new(),
                current_prediction: 0.0,
                confidence: 0.5,
                specialization,
                correct_predictions: 0,
                total_predictions: 0,
                quantum_state: None,
                learning_rate: quantum_config.learning_rate,
                momentum: 0.9,
                regularization: 0.01,
            });
        }

        Ok(Self {
            config: quantum_config,
            quantum_device,
            experts,
            state_cache: Arc::new(Mutex::new(HashMap::new())),
            circuit_cache: Arc::new(Mutex::new(HashMap::new())),
            quantum_metrics: Arc::new(Mutex::new(QuantumHedgeMetrics::default())),
            market_history: Vec::new(),
            decision_history: Vec::new(),
            current_regime: MarketRegime::Sideways,
            portfolio_state: PortfolioState::default(),
        })
    }

    /// Update market data and expert predictions
    pub fn update_market_data(&mut self, market_data: MarketData) -> Result<(), Box<dyn std::error::Error>> {
        // Store market data
        self.market_history.push(market_data.clone());
        
        // Keep only recent history
        if self.market_history.len() > 1000 {
            self.market_history.drain(0..100);
        }

        // Update market regime
        self.current_regime = self.detect_market_regime(&market_data)?;

        // Update experts based on market data
        self.update_experts(&market_data)?;

        // Update portfolio state
        self.update_portfolio_state(&market_data)?;

        Ok(())
    }

    /// Quantum multiplicative weights update
    pub fn quantum_multiplicative_weights_update(&mut self, returns: &[f64]) -> QuantumResult<()> {
        let _quantum_device = self.quantum_device.as_ref()
            .ok_or_else(|| QuantumError::device_error("quantum", "Quantum device not available"))?;

        // Create quantum state encoding expert weights
        let mut quantum_state = self.encode_expert_weights()?;

        // Apply quantum multiplicative weights circuit
        let mw_circuit = self.get_or_create_circuit(
            "multiplicative_weights",
            || self.create_multiplicative_weights_circuit(self.experts.len())
        )?;

        mw_circuit.execute(&mut quantum_state)?;

        // Update expert weights based on quantum state
        self.update_expert_weights_from_quantum_state(&quantum_state, returns)?;

        Ok(())
    }

    /// Quantum hedge decision making
    pub fn quantum_hedge_decision(&mut self, market_data: &MarketData) -> QuantumResult<HedgeDecision> {
        let _quantum_device = self.quantum_device.as_ref()
            .ok_or_else(|| QuantumError::device_error("quantum", "Quantum device not available"))?;

        // Create quantum state encoding market conditions
        let mut quantum_state = self.encode_market_conditions(market_data)?;

        // Apply quantum hedge decision circuit
        let hedge_circuit = self.get_or_create_circuit(
            "hedge_decision",
            || self.create_hedge_decision_circuit(market_data.factors.len())
        )?;

        hedge_circuit.execute(&mut quantum_state)?;

        // Extract hedge decision from quantum state
        let hedge_decision = self.extract_hedge_decision_from_state(&quantum_state, market_data)?;

        // Store decision
        self.decision_history.push(hedge_decision.clone());

        Ok(hedge_decision)
    }

    /// Quantum portfolio optimization
    pub fn quantum_portfolio_optimization(&mut self, assets: &[String]) -> QuantumResult<HashMap<String, f64>> {
        let _quantum_device = self.quantum_device.as_ref()
            .ok_or_else(|| QuantumError::device_error("quantum", "Quantum device not available"))?;

        // Create quantum state encoding portfolio constraints
        let mut quantum_state = self.encode_portfolio_constraints(assets)?;

        // Apply quantum portfolio optimization circuit
        let portfolio_circuit = self.get_or_create_circuit(
            "portfolio_optimization",
            || self.create_portfolio_optimization_circuit(assets.len())
        )?;

        portfolio_circuit.execute(&mut quantum_state)?;

        // Extract optimal portfolio allocation
        let allocation = self.extract_portfolio_allocation_from_state(&quantum_state, assets)?;

        Ok(allocation)
    }

    /// Quantum expert system prediction
    pub fn quantum_expert_prediction(&mut self, market_data: &MarketData) -> QuantumResult<f64> {
        let _quantum_device = self.quantum_device.as_ref()
            .ok_or_else(|| QuantumError::device_error("quantum", "Quantum device not available"))?;

        // Create quantum state encoding expert predictions
        let mut quantum_state = self.encode_expert_predictions(market_data)?;

        // Apply quantum expert aggregation circuit
        let expert_circuit = self.get_or_create_circuit(
            "expert_aggregation",
            || self.create_expert_aggregation_circuit(self.experts.len())
        )?;

        expert_circuit.execute(&mut quantum_state)?;

        // Extract aggregated prediction
        let prediction = self.extract_prediction_from_state(&quantum_state)?;

        Ok(prediction)
    }

    /// Quantum risk management
    pub fn quantum_risk_management(&mut self, market_data: &MarketData) -> QuantumResult<f64> {
        let _quantum_device = self.quantum_device.as_ref()
            .ok_or_else(|| QuantumError::device_error("quantum", "Quantum device not available"))?;

        // Create quantum state encoding risk factors
        let mut quantum_state = self.encode_risk_factors(market_data)?;

        // Apply quantum risk assessment circuit
        let risk_circuit = self.get_or_create_circuit(
            "risk_management",
            || self.create_risk_management_circuit(market_data.factors.len())
        )?;

        risk_circuit.execute(&mut quantum_state)?;

        // Extract risk assessment
        let risk_score = self.extract_risk_score_from_state(&quantum_state)?;

        Ok(risk_score)
    }

    // Private helper methods

    /// Encode expert weights into quantum state
    fn encode_expert_weights(&self) -> QuantumResult<QuantumState> {
        let num_qubits = (self.experts.len() as f64).log2().ceil() as usize;
        let mut quantum_state = QuantumState::new(num_qubits)?;

        // Encode weights as probability amplitudes
        for (i, expert) in self.experts.iter().enumerate() {
            let amplitude = Complex64::new(expert.weight.sqrt(), 0.0);
            quantum_state.set_amplitude(i, amplitude)?;
        }

        quantum_state.normalize()?;
        Ok(quantum_state)
    }

    /// Encode market conditions into quantum state
    fn encode_market_conditions(&self, market_data: &MarketData) -> QuantumResult<QuantumState> {
        let num_qubits = self.config.num_qubits;
        let mut quantum_state = QuantumState::new(num_qubits)?;

        // Encode market factors
        for (i, &factor) in market_data.factors.iter().enumerate() {
            if i < num_qubits {
                let amplitude = Complex64::new(factor.tanh(), 0.0);
                quantum_state.set_amplitude(i, amplitude)?;
            }
        }

        quantum_state.normalize()?;
        Ok(quantum_state)
    }

    /// Encode portfolio constraints into quantum state
    fn encode_portfolio_constraints(&self, assets: &[String]) -> QuantumResult<QuantumState> {
        let num_qubits = (assets.len() as f64).log2().ceil() as usize;
        let mut quantum_state = QuantumState::new(num_qubits)?;

        // Initialize equal weight distribution
        let weight = 1.0 / assets.len() as f64;
        for i in 0..assets.len() {
            let amplitude = Complex64::new(weight.sqrt(), 0.0);
            quantum_state.set_amplitude(i, amplitude)?;
        }

        quantum_state.normalize()?;
        Ok(quantum_state)
    }

    /// Encode expert predictions into quantum state
    fn encode_expert_predictions(&self, _market_data: &MarketData) -> QuantumResult<QuantumState> {
        let num_qubits = (self.experts.len() as f64).log2().ceil() as usize;
        let mut quantum_state = QuantumState::new(num_qubits)?;

        // Encode predictions weighted by confidence
        for (i, expert) in self.experts.iter().enumerate() {
            let prediction = expert.current_prediction * expert.confidence;
            let amplitude = Complex64::new(prediction.tanh(), 0.0);
            quantum_state.set_amplitude(i, amplitude)?;
        }

        quantum_state.normalize()?;
        Ok(quantum_state)
    }

    /// Encode risk factors into quantum state
    fn encode_risk_factors(&self, market_data: &MarketData) -> QuantumResult<QuantumState> {
        let num_qubits = self.config.num_qubits;
        let mut quantum_state = QuantumState::new(num_qubits)?;

        // Encode risk-relevant factors
        let risk_factors = [
            market_data.volatility,
            market_data.spread,
            market_data.factors[1], // volatility factor
            market_data.factors[7], // anomaly factor
            self.portfolio_state.risk_exposure,
            self.portfolio_state.leverage,
            self.portfolio_state.max_drawdown,
            1.0 / (self.portfolio_state.sharpe_ratio + 1.0),
        ];

        for (i, &factor) in risk_factors.iter().enumerate() {
            if i < num_qubits {
                let amplitude = Complex64::new(factor.tanh(), 0.0);
                quantum_state.set_amplitude(i, amplitude)?;
            }
        }

        quantum_state.normalize()?;
        Ok(quantum_state)
    }

    /// Create multiplicative weights quantum circuit
    fn create_multiplicative_weights_circuit(&self, num_experts: usize) -> QuantumResult<QuantumCircuit> {
        let num_qubits = (num_experts as f64).log2().ceil() as usize;
        let mut circuit = QuantumCircuit::new_simple(num_qubits);

        // Apply Hadamard gates for superposition
        for i in 0..num_qubits {
            circuit.add_gate_auto(QuantumGate::hadamard(i)?)?;
        }

        // Apply rotation gates for weight updates
        for i in 0..num_qubits {
            let angle = std::f64::consts::PI / (2.0 * (i + 1) as f64);
            circuit.add_gate_auto(QuantumGate::rotation_y(i, angle)?)?;
        }

        // Apply controlled operations for expert interactions
        for i in 0..num_qubits - 1 {
            circuit.add_gate_auto(QuantumGate::controlled_not(i, i + 1)?)?;
        }

        // Final rotation for normalization
        for i in 0..num_qubits {
            let angle = std::f64::consts::PI / 4.0;
            circuit.add_gate_auto(QuantumGate::rotation_z(i, angle)?)?;
        }

        Ok(circuit)
    }

    /// Create hedge decision quantum circuit
    fn create_hedge_decision_circuit(&self, _num_factors: usize) -> QuantumResult<QuantumCircuit> {
        let num_qubits = self.config.num_qubits;
        let mut circuit = QuantumCircuit::new_simple(num_qubits);

        // Initialize with market data encoding
        for i in 0..num_qubits {
            circuit.add_gate_auto(QuantumGate::rotation_x(i, std::f64::consts::PI / 4.0)?)?;
        }

        // Apply factor-based rotations
        for i in 0..num_qubits {
            let angle = std::f64::consts::PI / (2.0 * (i + 1) as f64);
            circuit.add_gate_auto(QuantumGate::rotation_y(i, angle)?)?;
        }

        // Add entanglement for factor interactions
        for i in 0..num_qubits - 1 {
            circuit.add_gate_auto(QuantumGate::controlled_not(i, i + 1)?)?;
        }

        // Decision optimization layer
        for i in 0..num_qubits {
            let angle = std::f64::consts::PI / 6.0;
            circuit.add_gate_auto(QuantumGate::rotation_z(i, angle)?)?;
        }

        Ok(circuit)
    }

    /// Create portfolio optimization quantum circuit
    fn create_portfolio_optimization_circuit(&self, num_assets: usize) -> QuantumResult<QuantumCircuit> {
        let num_qubits = (num_assets as f64).log2().ceil() as usize;
        let mut circuit = QuantumCircuit::new_simple(num_qubits);

        // Initialize equal superposition
        for i in 0..num_qubits {
            circuit.add_gate_auto(QuantumGate::hadamard(i)?)?;
        }

        // Apply risk-adjusted rotations
        for i in 0..num_qubits {
            let angle = std::f64::consts::PI / (3.0 * (i + 1) as f64);
            circuit.add_gate_auto(QuantumGate::rotation_y(i, angle)?)?;
        }

        // Add correlation effects
        for i in 0..num_qubits - 1 {
            circuit.add_gate_auto(QuantumGate::controlled_phase(i, i + 1, std::f64::consts::PI / 4.0)?)?;
        }

        // Optimization layer
        for i in 0..num_qubits {
            let angle = std::f64::consts::PI / 8.0;
            circuit.add_gate_auto(QuantumGate::rotation_z(i, angle)?)?;
        }

        Ok(circuit)
    }

    /// Create expert aggregation quantum circuit
    fn create_expert_aggregation_circuit(&self, num_experts: usize) -> QuantumResult<QuantumCircuit> {
        let num_qubits = (num_experts as f64).log2().ceil() as usize;
        let mut circuit = QuantumCircuit::new_simple(num_qubits);

        // Initialize with expert predictions
        for i in 0..num_qubits {
            circuit.add_gate_auto(QuantumGate::rotation_x(i, std::f64::consts::PI / 6.0)?)?;
        }

        // Apply confidence-weighted rotations
        for i in 0..num_qubits {
            let angle = std::f64::consts::PI / (4.0 * (i + 1) as f64);
            circuit.add_gate_auto(QuantumGate::rotation_y(i, angle)?)?;
        }

        // Add expert interactions
        for i in 0..num_qubits - 1 {
            circuit.add_gate_auto(QuantumGate::controlled_not(i, i + 1)?)?;
        }

        // Aggregation layer
        for i in 0..num_qubits {
            let angle = std::f64::consts::PI / 12.0;
            circuit.add_gate_auto(QuantumGate::rotation_z(i, angle)?)?;
        }

        Ok(circuit)
    }

    /// Create risk management quantum circuit
    fn create_risk_management_circuit(&self, _num_factors: usize) -> QuantumResult<QuantumCircuit> {
        let num_qubits = self.config.num_qubits;
        let mut circuit = QuantumCircuit::new_simple(num_qubits);

        // Initialize with risk factors
        for i in 0..num_qubits {
            circuit.add_gate_auto(QuantumGate::rotation_x(i, std::f64::consts::PI / 8.0)?)?;
        }

        // Apply risk-sensitive rotations
        for i in 0..num_qubits {
            let angle = std::f64::consts::PI / (5.0 * (i + 1) as f64);
            circuit.add_gate_auto(QuantumGate::rotation_y(i, angle)?)?;
        }

        // Add risk correlations
        for i in 0..num_qubits - 1 {
            circuit.add_gate_auto(QuantumGate::controlled_phase(i, i + 1, std::f64::consts::PI / 6.0)?)?;
        }

        // Risk assessment layer
        for i in 0..num_qubits {
            let angle = std::f64::consts::PI / 10.0;
            circuit.add_gate_auto(QuantumGate::rotation_z(i, angle)?)?;
        }

        Ok(circuit)
    }

    /// Get or create cached quantum circuit
    fn get_or_create_circuit<F>(&self, name: &str, creator: F) -> QuantumResult<QuantumCircuit>
    where
        F: FnOnce() -> QuantumResult<QuantumCircuit>,
    {
        if let Ok(mut cache) = self.circuit_cache.lock() {
            if let Some(circuit) = cache.get(name) {
                return Ok(circuit.clone());
            }
            
            let circuit = creator()?;
            cache.insert(name.to_string(), circuit.clone());
            Ok(circuit)
        } else {
            creator()
        }
    }

    /// Update expert weights from quantum state
    fn update_expert_weights_from_quantum_state(&mut self, quantum_state: &QuantumState, returns: &[f64]) -> QuantumResult<()> {
        let amplitudes = quantum_state.get_amplitudes();
        let mut total_weight = 0.0;

        // Update weights based on quantum amplitudes and returns
        for (i, expert) in self.experts.iter_mut().enumerate() {
            if i < amplitudes.len() {
                let amplitude = amplitudes[i];
                let quantum_weight = amplitude.norm_sqr();
                
                // Incorporate return performance
                let return_factor = if i < returns.len() {
                    (returns[i] * self.config.learning_rate).exp()
                } else {
                    1.0
                };

                expert.weight = quantum_weight * return_factor;
                total_weight += expert.weight;
            }
        }

        // Normalize weights
        if total_weight > 0.0 {
            for expert in &mut self.experts {
                expert.weight /= total_weight;
            }
        }

        Ok(())
    }

    /// Extract hedge decision from quantum state
    fn extract_hedge_decision_from_state(&self, quantum_state: &QuantumState, market_data: &MarketData) -> QuantumResult<HedgeDecision> {
        let amplitudes = quantum_state.get_amplitudes();
        
        // Extract decision parameters from quantum state
        let mut decision_value = 0.0;
        let mut confidence = 0.0;
        
        for (i, amplitude) in amplitudes.iter().enumerate() {
            let prob = amplitude.norm_sqr();
            decision_value += prob * (i as f64 - amplitudes.len() as f64 / 2.0);
            confidence += prob;
        }

        // Normalize decision value
        decision_value /= amplitudes.len() as f64;
        confidence = confidence.min(1.0);

        // Determine action based on decision value
        let action = if decision_value > 0.3 {
            HedgeAction::Buy
        } else if decision_value < -0.3 {
            HedgeAction::Sell
        } else if decision_value.abs() < 0.1 {
            HedgeAction::Hold
        } else {
            HedgeAction::Hedge
        };

        // Calculate position size based on confidence and risk tolerance
        let position_size = (confidence * self.config.risk_tolerance).min(1.0);

        // Calculate expected return and risk
        let expected_return = decision_value * market_data.factors[0]; // trend factor
        let risk_estimate = market_data.volatility * (1.0 - confidence);

        // Create expert weights map
        let mut expert_weights = HashMap::new();
        for expert in &self.experts {
            expert_weights.insert(expert.id.clone(), expert.weight);
        }

        Ok(HedgeDecision {
            timestamp: Utc::now(),
            action,
            position_size,
            confidence,
            expected_return,
            risk_estimate,
            stop_loss: Some(market_data.price * (1.0 - risk_estimate)),
            take_profit: Some(market_data.price * (1.0 + expected_return)),
            expert_weights,
            market_regime: self.current_regime,
        })
    }

    /// Extract portfolio allocation from quantum state
    fn extract_portfolio_allocation_from_state(&self, quantum_state: &QuantumState, assets: &[String]) -> QuantumResult<HashMap<String, f64>> {
        let amplitudes = quantum_state.get_amplitudes();
        let mut allocation = HashMap::new();
        let mut total_weight = 0.0;

        // Calculate weights from quantum amplitudes
        for (i, asset) in assets.iter().enumerate() {
            if i < amplitudes.len() {
                let weight = amplitudes[i].norm_sqr();
                allocation.insert(asset.clone(), weight);
                total_weight += weight;
            }
        }

        // Normalize allocation
        if total_weight > 0.0 {
            for (_, weight) in allocation.iter_mut() {
                *weight /= total_weight;
            }
        }

        Ok(allocation)
    }

    /// Extract prediction from quantum state
    fn extract_prediction_from_state(&self, quantum_state: &QuantumState) -> QuantumResult<f64> {
        let amplitudes = quantum_state.get_amplitudes();
        let mut prediction = 0.0;

        // Weighted average of amplitudes
        for (i, amplitude) in amplitudes.iter().enumerate() {
            let weight = amplitude.norm_sqr();
            prediction += weight * (i as f64 - amplitudes.len() as f64 / 2.0);
        }

        // Normalize prediction
        prediction /= amplitudes.len() as f64;
        Ok(prediction.tanh()) // Bounded between -1 and 1
    }

    /// Extract risk score from quantum state
    fn extract_risk_score_from_state(&self, quantum_state: &QuantumState) -> QuantumResult<f64> {
        let amplitudes = quantum_state.get_amplitudes();
        let mut risk_score = 0.0;

        // Calculate risk as variance of amplitudes
        let mean_amplitude = amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>() / amplitudes.len() as f64;
        
        for amplitude in amplitudes {
            let deviation = amplitude.norm_sqr() - mean_amplitude;
            risk_score += deviation * deviation;
        }

        risk_score = (risk_score / amplitudes.len() as f64).sqrt();
        Ok(risk_score.min(1.0))
    }

    /// Detect market regime
    fn detect_market_regime(&self, market_data: &MarketData) -> Result<MarketRegime, Box<dyn std::error::Error>> {
        let trend_factor = market_data.factors[0];
        let volatility_factor = market_data.factors[1];
        let anomaly_factor = market_data.factors[7];

        // Simple regime detection based on factors
        if anomaly_factor > 0.8 {
            Ok(MarketRegime::Crisis)
        } else if volatility_factor > 0.7 {
            Ok(MarketRegime::HighVolatility)
        } else if volatility_factor < 0.3 {
            Ok(MarketRegime::LowVolatility)
        } else if trend_factor > 0.6 {
            Ok(MarketRegime::Bull)
        } else if trend_factor < -0.6 {
            Ok(MarketRegime::Bear)
        } else {
            Ok(MarketRegime::Sideways)
        }
    }

    /// Update experts based on market data
    fn update_experts(&mut self, market_data: &MarketData) -> Result<(), Box<dyn std::error::Error>> {
        // Calculate predictions and confidences first to avoid borrowing conflicts
        let mut predictions = Vec::new();
        let mut confidences = Vec::new();
        
        for expert in &self.experts {
            let prediction = self.calculate_expert_prediction(expert, market_data)?;
            let confidence = self.calculate_expert_confidence(expert)?;
            predictions.push(prediction);
            confidences.push(confidence);
        }
        
        // Now update the experts
        for (i, expert) in self.experts.iter_mut().enumerate() {
            expert.current_prediction = predictions[i];
            expert.confidence = confidences[i];
            
            // Update learning parameters
            expert.learning_rate *= 0.999; // Decay learning rate
            expert.regularization *= 1.001; // Increase regularization
        }

        Ok(())
    }

    /// Calculate expert prediction based on specialization
    fn calculate_expert_prediction(&self, expert: &QuantumExpert, market_data: &MarketData) -> Result<f64, Box<dyn std::error::Error>> {
        let prediction = match expert.specialization {
            ExpertSpecialization::TrendFollowing => market_data.factors[0],
            ExpertSpecialization::MeanReversion => -market_data.factors[0],
            ExpertSpecialization::VolatilityTrading => market_data.factors[1],
            ExpertSpecialization::Momentum => market_data.factors[2],
            ExpertSpecialization::SentimentAnalysis => market_data.factors[3],
            ExpertSpecialization::LiquidityProvision => market_data.factors[4],
            ExpertSpecialization::CorrelationTrading => market_data.factors[5],
            ExpertSpecialization::CycleAnalysis => market_data.factors[6],
            ExpertSpecialization::AnomalyDetection => market_data.factors[7],
            ExpertSpecialization::RiskManagement => -market_data.volatility,
            ExpertSpecialization::OptionsTrading => market_data.volatility * market_data.factors[2],
            ExpertSpecialization::PairsTrading => market_data.factors[5] * market_data.factors[0],
            ExpertSpecialization::ArbitrageExpert => market_data.spread,
            ExpertSpecialization::HighFrequencyTrading => market_data.factors[4] * market_data.factors[1],
            ExpertSpecialization::MacroEconomic => market_data.factors[6] * market_data.factors[0],
            ExpertSpecialization::TechnicalAnalysis => market_data.factors[2] * market_data.factors[0],
        };

        Ok(prediction.tanh()) // Bounded between -1 and 1
    }

    /// Calculate expert confidence based on performance
    fn calculate_expert_confidence(&self, expert: &QuantumExpert) -> Result<f64, Box<dyn std::error::Error>> {
        if expert.total_predictions == 0 {
            return Ok(0.5);
        }

        let accuracy = expert.correct_predictions as f64 / expert.total_predictions as f64;
        let confidence = (accuracy - 0.5) * 2.0; // Scale to -1 to 1
        
        Ok(confidence.abs().min(1.0))
    }

    /// Update portfolio state
    fn update_portfolio_state(&mut self, market_data: &MarketData) -> Result<(), Box<dyn std::error::Error>> {
        // Update portfolio metrics based on market data
        let position_value = self.portfolio_state.positions.get(&market_data.symbol).unwrap_or(&0.0) * market_data.price;
        self.portfolio_state.portfolio_value = self.portfolio_state.cash + position_value;
        
        // Update risk metrics
        self.portfolio_state.risk_exposure = position_value / self.portfolio_state.portfolio_value;
        self.portfolio_state.leverage = self.portfolio_state.risk_exposure * self.config.max_leverage;
        
        self.portfolio_state.last_update = Utc::now();
        
        Ok(())
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> Result<QuantumHedgeMetrics, Box<dyn std::error::Error>> {
        Ok(self.quantum_metrics.lock().unwrap().clone())
    }

    /// Get expert weights
    pub fn get_expert_weights(&self) -> HashMap<String, f64> {
        self.experts.iter()
            .map(|e| (e.id.clone(), e.weight))
            .collect()
    }

    /// Get current market regime
    pub fn get_market_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get portfolio state
    pub fn get_portfolio_state(&self) -> &PortfolioState {
        &self.portfolio_state
    }

    /// Get decision history
    pub fn get_decision_history(&self) -> &[HedgeDecision] {
        &self.decision_history
    }
}

impl Default for QuantumExpert {
    fn default() -> Self {
        Self {
            id: "default".to_string(),
            name: "Default Expert".to_string(),
            weight: 0.0,
            performance_history: Vec::new(),
            current_prediction: 0.0,
            confidence: 0.5,
            specialization: ExpertSpecialization::TrendFollowing,
            correct_predictions: 0,
            total_predictions: 0,
            quantum_state: None,
            learning_rate: 0.1,
            momentum: 0.9,
            regularization: 0.01,
        }
    }
}

// MarketData implementation methods are in lib.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_hedge_algorithm_creation() {
        let config = QuantumHedgeConfig::default();
        let expert_names = vec!["Expert1".to_string(), "Expert2".to_string()];
        let algorithm = QuantumHedgeAlgorithm::new(expert_names, config);
        
        assert!(algorithm.is_ok());
        let algorithm = algorithm.unwrap();
        assert_eq!(algorithm.experts.len(), 2);
    }

    #[test]
    fn test_market_data_creation() {
        let factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let market_data = MarketData::new("BTCUSD".to_string(), 50000.0, 1000.0, factors);
        
        assert_eq!(market_data.symbol, "BTCUSD");
        assert_eq!(market_data.price, 50000.0);
        assert_eq!(market_data.volume, 1000.0);
        assert_eq!(market_data.factors, factors);
    }

    #[test]
    fn test_expert_specializations() {
        let expert = QuantumExpert {
            specialization: ExpertSpecialization::TrendFollowing,
            ..Default::default()
        };
        
        assert_eq!(expert.specialization, ExpertSpecialization::TrendFollowing);
    }

    #[test]
    fn test_hedge_decision_validity() {
        let decision = HedgeDecision {
            timestamp: Utc::now(),
            action: HedgeAction::Buy,
            position_size: 0.5,
            confidence: 0.8,
            expected_return: 0.02,
            risk_estimate: 0.01,
            stop_loss: Some(49000.0),
            take_profit: Some(51000.0),
            expert_weights: HashMap::new(),
            market_regime: MarketRegime::Bull,
        };

        assert_eq!(decision.action, HedgeAction::Buy);
        assert_eq!(decision.position_size, 0.5);
        assert_eq!(decision.confidence, 0.8);
    }
}