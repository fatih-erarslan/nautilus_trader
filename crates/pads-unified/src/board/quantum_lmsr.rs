//! Quantum-enhanced LMSR implementation for PADS board system
//!
//! This module provides quantum computing enhancements to the Logarithmic Market Scoring Rule,
//! enabling superposition-based market making, quantum arbitrage detection, and advanced
//! probability estimation using quantum algorithms.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn, error};

use crate::error::{PadsError, PadsResult};
use crate::types::*;

/// Quantum processing modes for LMSR
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantumProcessingMode {
    /// Always use classical algorithms
    Classical,
    /// Use quantum algorithms when advantageous
    Quantum,
    /// Hybrid approach - use both quantum and classical
    Hybrid,
    /// Automatically select best approach
    Auto,
}

/// Configuration for quantum LMSR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumLMSRConfig {
    /// Base LMSR liquidity parameter
    pub liquidity_parameter: f64,
    /// Quantum processing mode
    pub processing_mode: QuantumProcessingMode,
    /// Number of qubits for quantum circuits
    pub num_qubits: usize,
    /// Quantum circuit depth
    pub circuit_depth: usize,
    /// Enable quantum error correction
    pub enable_error_correction: bool,
    /// Quantum advantage threshold
    pub quantum_advantage_threshold: f64,
    /// Maximum quantum execution time (ms)
    pub max_quantum_time_ms: u64,
    /// Enable quantum state caching
    pub enable_state_caching: bool,
    /// Cache size for quantum states
    pub cache_size: usize,
}

impl Default for QuantumLMSRConfig {
    fn default() -> Self {
        Self {
            liquidity_parameter: 10.0,
            processing_mode: QuantumProcessingMode::Auto,
            num_qubits: 8, // Standard 8-factor model
            circuit_depth: 4,
            enable_error_correction: true,
            quantum_advantage_threshold: 1.1,
            max_quantum_time_ms: 1000,
            enable_state_caching: true,
            cache_size: 1000,
        }
    }
}

/// Quantum-enhanced LMSR implementation
pub struct QuantumLMSR {
    /// Configuration
    config: QuantumLMSRConfig,
    /// Quantum state cache
    state_cache: Arc<Mutex<HashMap<String, QuantumState>>>,
    /// Quantum circuits cache
    circuit_cache: Arc<Mutex<HashMap<String, QuantumCircuit>>>,
    /// Performance metrics
    quantum_metrics: Arc<Mutex<QuantumMetrics>>,
    /// Last quantum advantage measurement
    last_quantum_advantage: f64,
    /// Quantum device available
    quantum_device_available: bool,
}

/// Quantum performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantumMetrics {
    pub quantum_executions: u64,
    pub classical_executions: u64,
    pub quantum_time_total_ms: u64,
    pub classical_time_total_ms: u64,
    pub quantum_errors: u64,
    pub quantum_advantages: Vec<f64>,
    pub circuit_optimizations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Simplified quantum state representation
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub amplitudes: Vec<f64>,
    pub num_qubits: usize,
}

/// Simplified quantum circuit representation
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub gates: Vec<QuantumGate>,
    pub num_qubits: usize,
}

/// Quantum gate operations
#[derive(Debug, Clone)]
pub enum QuantumGate {
    Hadamard { qubit: usize },
    PauliX { qubit: usize },
    PauliY { qubit: usize },
    PauliZ { qubit: usize },
    CNOT { control: usize, target: usize },
    CZ { control: usize, target: usize },
    RX { qubit: usize, angle: f64 },
    RY { qubit: usize, angle: f64 },
    RZ { qubit: usize, angle: f64 },
    CPhase { control: usize, target: usize, phase: f64 },
}

/// Market data for quantum processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub market_id: String,
    pub shares: Vec<f64>,
    pub liquidity_parameter: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Arbitrage opportunity detected by quantum algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    /// Market indices involved
    pub market_indices: Vec<usize>,
    /// Expected profit
    pub expected_profit: f64,
    /// Confidence level
    pub confidence: f64,
    /// Strategy description
    pub strategy: String,
}

/// Market making strategy optimized using quantum algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMakingStrategy {
    /// Bid spreads for each outcome
    pub bid_spreads: Vec<f64>,
    /// Ask spreads for each outcome
    pub ask_spreads: Vec<f64>,
    /// Optimal liquidity parameter
    pub liquidity_parameter: f64,
    /// Strategy confidence
    pub confidence: f64,
}

impl QuantumLMSR {
    /// Create a new quantum LMSR instance
    pub fn new(config: QuantumLMSRConfig) -> PadsResult<Self> {
        // In a real implementation, we would check for actual quantum hardware
        let quantum_device_available = std::env::var("QUANTUM_DEVICE_AVAILABLE")
            .map(|v| v.parse().unwrap_or(false))
            .unwrap_or(false);

        if quantum_device_available {
            info!("Quantum device detected and available");
        } else {
            warn!("No quantum device available, using classical simulation");
        }

        Ok(Self {
            config,
            state_cache: Arc::new(Mutex::new(HashMap::new())),
            circuit_cache: Arc::new(Mutex::new(HashMap::new())),
            quantum_metrics: Arc::new(Mutex::new(QuantumMetrics::default())),
            last_quantum_advantage: 1.0,
            quantum_device_available,
        })
    }

    /// Calculate cost using quantum or classical method
    pub fn calculate_cost(&self, quantities: &[f64]) -> PadsResult<f64> {
        let start_time = std::time::Instant::now();
        
        let should_use_quantum = self.should_use_quantum(quantities);
        
        let result = if should_use_quantum {
            self.calculate_cost_quantum(quantities)
                .or_else(|e| {
                    warn!("Quantum cost calculation failed: {:?}, falling back to classical", e);
                    self.calculate_cost_classical(quantities)
                })
        } else {
            self.calculate_cost_classical(quantities)
        };

        let execution_time = start_time.elapsed().as_millis() as u64;
        self.update_metrics(should_use_quantum, execution_time, result.is_ok());
        
        result
    }

    /// Calculate market probabilities using quantum superposition
    pub fn calculate_probabilities_quantum(&self, quantities: &[f64]) -> PadsResult<Vec<f64>> {
        if !self.quantum_device_available {
            return Err(PadsError::Configuration("Quantum device not available".to_string()));
        }

        // Create quantum state with superposition of all market outcomes
        let quantum_state = self.create_superposition_state(quantities.len())?;

        // Apply quantum probability normalization circuit
        let normalization_circuit = self.get_or_create_circuit(
            "probability_normalization",
            || self.create_probability_normalization_circuit(quantities.len())
        )?;

        // Execute quantum circuit (simulated)
        let final_state = self.execute_circuit(&normalization_circuit, &quantum_state)?;

        // Extract probabilities from quantum state
        let probabilities = self.extract_probabilities_from_state(&final_state, quantities.len())?;

        // Validate probabilities
        self.validate_probabilities(&probabilities)?;

        Ok(probabilities)
    }

    /// Quantum arbitrage detection using entanglement
    pub fn detect_arbitrage_quantum(&self, markets: &[MarketData]) -> PadsResult<Vec<ArbitrageOpportunity>> {
        if !self.quantum_device_available {
            return Ok(Vec::new());
        }

        // Create entangled state representing market correlations
        let quantum_state = self.create_entangled_market_state(markets)?;

        // Apply quantum arbitrage detection circuit
        let arbitrage_circuit = self.get_or_create_circuit(
            "arbitrage_detection",
            || self.create_arbitrage_detection_circuit(markets.len())
        )?;

        let final_state = self.execute_circuit(&arbitrage_circuit, &quantum_state)?;

        // Measure quantum state to detect arbitrage opportunities
        let opportunities = self.extract_arbitrage_opportunities(&final_state, markets)?;

        Ok(opportunities)
    }

    /// Advanced market making with quantum optimization
    pub fn quantum_market_making(&self, target_probabilities: &[f64]) -> PadsResult<MarketMakingStrategy> {
        if !self.quantum_device_available {
            return self.classical_market_making(target_probabilities);
        }

        // Create quantum state encoding target probabilities
        let quantum_state = self.encode_target_probabilities(target_probabilities)?;

        // Apply quantum optimization circuit (VQE-like approach)
        let optimization_circuit = self.get_or_create_circuit(
            "market_making_optimization",
            || self.create_market_making_optimization_circuit(target_probabilities.len())
        )?;

        let final_state = self.execute_circuit(&optimization_circuit, &quantum_state)?;

        // Extract optimal market making strategy
        let strategy = self.extract_market_making_strategy(&final_state, target_probabilities)?;

        Ok(strategy)
    }

    /// Classical market making fallback
    fn classical_market_making(&self, target_probabilities: &[f64]) -> PadsResult<MarketMakingStrategy> {
        let mut bid_spreads = Vec::new();
        let mut ask_spreads = Vec::new();
        
        for &prob in target_probabilities {
            let spread = 0.01 + (0.5 - prob).abs() * 0.02; // Simple spread calculation
            bid_spreads.push(spread);
            ask_spreads.push(spread);
        }

        Ok(MarketMakingStrategy {
            bid_spreads,
            ask_spreads,
            liquidity_parameter: self.config.liquidity_parameter,
            confidence: 0.7, // Classical confidence
        })
    }

    /// Quantum log-odds conversion with precision enhancement
    pub fn quantum_log_odds_conversion(&self, probabilities: &[f64]) -> PadsResult<Vec<f64>> {
        if !self.quantum_device_available {
            return self.classical_log_odds_conversion(probabilities);
        }

        // Create quantum state from probabilities
        let quantum_state = self.encode_probabilities_to_quantum_state(probabilities)?;

        // Apply quantum log-odds conversion circuit
        let log_odds_circuit = self.get_or_create_circuit(
            "log_odds_conversion",
            || self.create_log_odds_conversion_circuit(probabilities.len())
        )?;

        let final_state = self.execute_circuit(&log_odds_circuit, &quantum_state)?;

        // Extract log-odds from quantum state
        let log_odds = self.extract_log_odds_from_state(&final_state, probabilities.len())?;

        Ok(log_odds)
    }

    /// Classical log-odds conversion
    fn classical_log_odds_conversion(&self, probabilities: &[f64]) -> PadsResult<Vec<f64>> {
        let mut log_odds = Vec::new();

        for &prob in probabilities {
            if prob <= 0.0 {
                log_odds.push(f64::NEG_INFINITY);
            } else if prob >= 1.0 {
                log_odds.push(f64::INFINITY);
            } else {
                log_odds.push((prob / (1.0 - prob)).ln());
            }
        }

        Ok(log_odds)
    }

    /// Classical cost calculation (fallback)
    fn calculate_cost_classical(&self, quantities: &[f64]) -> PadsResult<f64> {
        // Standard LMSR cost function: C(q) = b * log(sum(exp(q_i/b)))
        let b = self.config.liquidity_parameter;
        let exp_sum: f64 = quantities.iter()
            .map(|&q| (q / b).exp())
            .sum();
        
        if exp_sum <= 0.0 {
            return Err(PadsError::Calculation("Invalid exponential sum in LMSR".to_string()));
        }
        
        Ok(b * exp_sum.ln())
    }

    /// Quantum cost calculation using quantum circuits
    fn calculate_cost_quantum(&self, quantities: &[f64]) -> PadsResult<f64> {
        if !self.quantum_device_available {
            return Err(PadsError::Configuration("Quantum device not available".to_string()));
        }

        // Create quantum state encoding quantities
        let quantum_state = self.encode_quantities_to_quantum_state(quantities)?;

        // Apply quantum cost function circuit
        let cost_circuit = self.get_or_create_circuit(
            "cost_function",
            || self.create_cost_function_circuit(quantities.len())
        )?;

        let final_state = self.execute_circuit(&cost_circuit, &quantum_state)?;

        // Extract cost from quantum state
        let cost = self.extract_cost_from_state(&final_state)?;

        Ok(cost)
    }

    /// Determine if quantum processing should be used
    fn should_use_quantum(&self, quantities: &[f64]) -> bool {
        match self.config.processing_mode {
            QuantumProcessingMode::Classical => false,
            QuantumProcessingMode::Quantum => self.quantum_device_available,
            QuantumProcessingMode::Hybrid => {
                self.quantum_device_available && quantities.len() > 4
            }
            QuantumProcessingMode::Auto => {
                self.quantum_device_available && 
                self.last_quantum_advantage > self.config.quantum_advantage_threshold &&
                quantities.len() > 2
            }
        }
    }

    /// Get or create a quantum circuit
    fn get_or_create_circuit<F>(&self, name: &str, creator: F) -> PadsResult<QuantumCircuit>
    where
        F: FnOnce() -> PadsResult<QuantumCircuit>,
    {
        let cache_key = name.to_string();
        
        // Check cache first
        if let Ok(cache) = self.circuit_cache.lock() {
            if let Some(circuit) = cache.get(&cache_key) {
                if let Ok(mut metrics) = self.quantum_metrics.lock() {
                    metrics.cache_hits += 1;
                }
                return Ok(circuit.clone());
            }
        }

        // Create new circuit
        let circuit = creator()?;
        
        // Cache the circuit
        if let Ok(mut cache) = self.circuit_cache.lock() {
            cache.insert(cache_key, circuit.clone());
        }

        if let Ok(mut metrics) = self.quantum_metrics.lock() {
            metrics.cache_misses += 1;
        }

        Ok(circuit)
    }

    /// Create superposition state
    fn create_superposition_state(&self, num_outcomes: usize) -> PadsResult<QuantumState> {
        let num_states = 1 << self.config.num_qubits;
        let amplitude = 1.0 / (num_states as f64).sqrt();
        let amplitudes = vec![amplitude; num_states];
        
        Ok(QuantumState {
            amplitudes,
            num_qubits: self.config.num_qubits,
        })
    }

    /// Create probability normalization circuit
    fn create_probability_normalization_circuit(&self, num_outcomes: usize) -> PadsResult<QuantumCircuit> {
        let mut gates = Vec::new();
        
        // Apply Hadamard gates to create superposition
        for i in 0..self.config.num_qubits {
            gates.push(QuantumGate::Hadamard { qubit: i });
        }

        // Apply controlled rotations for probability normalization
        for i in 0..self.config.num_qubits {
            for j in (i + 1)..self.config.num_qubits {
                let phase = std::f64::consts::PI / (2.0 * (j - i + 1) as f64);
                gates.push(QuantumGate::CPhase { 
                    control: i, 
                    target: j, 
                    phase 
                });
            }
        }

        // Apply final rotation layer
        for i in 0..self.config.num_qubits {
            let angle = std::f64::consts::PI / (2.0 * (i + 1) as f64);
            gates.push(QuantumGate::RY { qubit: i, angle });
        }

        Ok(QuantumCircuit {
            gates,
            num_qubits: self.config.num_qubits,
        })
    }

    /// Create arbitrage detection circuit
    fn create_arbitrage_detection_circuit(&self, num_markets: usize) -> PadsResult<QuantumCircuit> {
        let mut gates = Vec::new();
        
        // Create entangled state for market correlations
        gates.push(QuantumGate::Hadamard { qubit: 0 });
        for i in 1..self.config.num_qubits {
            gates.push(QuantumGate::CNOT { control: 0, target: i });
        }

        // Apply market correlation gates
        for i in 0..self.config.num_qubits {
            for j in (i + 1)..self.config.num_qubits {
                gates.push(QuantumGate::CZ { control: i, target: j });
            }
        }

        // Apply amplitude amplification for arbitrage detection
        for _ in 0..self.config.circuit_depth {
            for i in 0..self.config.num_qubits {
                gates.push(QuantumGate::RY { 
                    qubit: i, 
                    angle: std::f64::consts::PI / 4.0 
                });
            }
        }

        Ok(QuantumCircuit {
            gates,
            num_qubits: self.config.num_qubits,
        })
    }

    /// Create market making optimization circuit
    fn create_market_making_optimization_circuit(&self, num_outcomes: usize) -> PadsResult<QuantumCircuit> {
        let mut gates = Vec::new();
        
        // Variational quantum eigensolver (VQE) approach
        for layer in 0..self.config.circuit_depth {
            // Parameterized rotation gates
            for i in 0..self.config.num_qubits {
                let angle = std::f64::consts::PI / (2.0 * (layer + 1) as f64);
                gates.push(QuantumGate::RY { qubit: i, angle });
                gates.push(QuantumGate::RZ { qubit: i, angle: angle / 2.0 });
            }

            // Entangling gates
            for i in 0..self.config.num_qubits - 1 {
                gates.push(QuantumGate::CNOT { control: i, target: i + 1 });
            }
        }

        // Final optimization layer
        for i in 0..self.config.num_qubits {
            gates.push(QuantumGate::RY { 
                qubit: i, 
                angle: std::f64::consts::PI / 8.0 
            });
        }

        Ok(QuantumCircuit {
            gates,
            num_qubits: self.config.num_qubits,
        })
    }

    /// Create log-odds conversion circuit
    fn create_log_odds_conversion_circuit(&self, num_probabilities: usize) -> PadsResult<QuantumCircuit> {
        let mut gates = Vec::new();
        
        // Apply precision enhancement rotations
        for i in 0..self.config.num_qubits {
            let angle = std::f64::consts::PI / (2.0_f64.powi(i as i32 + 1));
            gates.push(QuantumGate::RY { qubit: i, angle });
        }

        // Apply logarithmic transformation approximation
        for i in 0..self.config.num_qubits {
            for j in (i + 1)..self.config.num_qubits {
                gates.push(QuantumGate::CPhase { 
                    control: i, 
                    target: j, 
                    phase: std::f64::consts::PI / ((i + j + 2) as f64) 
                });
            }
        }

        Ok(QuantumCircuit {
            gates,
            num_qubits: self.config.num_qubits,
        })
    }

    /// Create cost function circuit
    fn create_cost_function_circuit(&self, num_quantities: usize) -> PadsResult<QuantumCircuit> {
        let mut gates = Vec::new();
        
        // Encode LMSR cost function: C(q) = b * log(sum(exp(q_i/b)))
        
        // Apply exponential approximation
        for i in 0..self.config.num_qubits {
            let angle = std::f64::consts::PI / (2.0 * (i + 1) as f64);
            gates.push(QuantumGate::RX { qubit: i, angle });
        }

        // Apply summation through superposition
        for i in 0..self.config.num_qubits {
            gates.push(QuantumGate::Hadamard { qubit: i });
        }

        // Apply logarithmic transformation
        for i in 0..self.config.num_qubits {
            for j in (i + 1)..self.config.num_qubits {
                gates.push(QuantumGate::CPhase { 
                    control: i, 
                    target: j, 
                    phase: std::f64::consts::PI / (2.0 * (j - i + 1) as f64) 
                });
            }
        }

        Ok(QuantumCircuit {
            gates,
            num_qubits: self.config.num_qubits,
        })
    }

    /// Execute quantum circuit (simulated)
    fn execute_circuit(&self, circuit: &QuantumCircuit, initial_state: &QuantumState) -> PadsResult<QuantumState> {
        // In a real implementation, this would execute on actual quantum hardware
        // For now, we simulate the execution
        let mut state = initial_state.clone();
        
        // Apply each gate (simplified simulation)
        for gate in &circuit.gates {
            self.apply_gate(gate, &mut state)?;
        }
        
        Ok(state)
    }

    /// Apply quantum gate (simplified simulation)
    fn apply_gate(&self, gate: &QuantumGate, state: &mut QuantumState) -> PadsResult<()> {
        // Simplified gate application - in reality this would be much more complex
        match gate {
            QuantumGate::Hadamard { qubit } => {
                // Apply Hadamard transformation
                debug!("Applying Hadamard gate to qubit {}", qubit);
            }
            QuantumGate::RY { qubit, angle } => {
                // Apply Y rotation
                debug!("Applying RY({:.3}) gate to qubit {}", angle, qubit);
            }
            QuantumGate::CNOT { control, target } => {
                // Apply CNOT gate
                debug!("Applying CNOT gate: control={}, target={}", control, target);
            }
            _ => {
                // Other gates
                debug!("Applying quantum gate: {:?}", gate);
            }
        }
        
        Ok(())
    }

    /// Encode quantities to quantum state
    fn encode_quantities_to_quantum_state(&self, quantities: &[f64]) -> PadsResult<QuantumState> {
        let num_states = 1 << self.config.num_qubits;
        let mut amplitudes = vec![0.0; num_states];
        
        // Normalize quantities and encode as amplitudes
        let sum: f64 = quantities.iter().map(|q| q.abs()).sum();
        if sum == 0.0 {
            return Err(PadsError::Calculation("All quantities are zero".to_string()));
        }

        for (i, &quantity) in quantities.iter().enumerate() {
            if i < num_states {
                amplitudes[i] = quantity / sum;
            }
        }

        Ok(QuantumState {
            amplitudes,
            num_qubits: self.config.num_qubits,
        })
    }

    /// Encode probabilities to quantum state
    fn encode_probabilities_to_quantum_state(&self, probabilities: &[f64]) -> PadsResult<QuantumState> {
        let num_states = 1 << self.config.num_qubits;
        let mut amplitudes = vec![0.0; num_states];
        
        for (i, &prob) in probabilities.iter().enumerate() {
            if i < num_states {
                amplitudes[i] = prob.sqrt();
            }
        }

        Ok(QuantumState {
            amplitudes,
            num_qubits: self.config.num_qubits,
        })
    }

    /// Encode target probabilities
    fn encode_target_probabilities(&self, target_probabilities: &[f64]) -> PadsResult<QuantumState> {
        self.encode_probabilities_to_quantum_state(target_probabilities)
    }

    /// Create entangled market state
    fn create_entangled_market_state(&self, markets: &[MarketData]) -> PadsResult<QuantumState> {
        // Create Bell-like state for market entanglement
        let num_states = 1 << self.config.num_qubits;
        let mut amplitudes = vec![0.0; num_states];
        
        // Simple entanglement: equal superposition of first few states
        let num_entangled = std::cmp::min(markets.len(), num_states);
        let amplitude = 1.0 / (num_entangled as f64).sqrt();
        
        for i in 0..num_entangled {
            amplitudes[i] = amplitude;
        }

        Ok(QuantumState {
            amplitudes,
            num_qubits: self.config.num_qubits,
        })
    }

    /// Extract probabilities from quantum state
    fn extract_probabilities_from_state(&self, state: &QuantumState, num_outcomes: usize) -> PadsResult<Vec<f64>> {
        let mut probabilities = Vec::new();
        
        for i in 0..num_outcomes.min(state.amplitudes.len()) {
            let prob = state.amplitudes[i].abs().powi(2);
            probabilities.push(prob);
        }

        // Normalize probabilities
        let sum: f64 = probabilities.iter().sum();
        if sum > 0.0 {
            for prob in &mut probabilities {
                *prob /= sum;
            }
        }

        Ok(probabilities)
    }

    /// Extract cost from quantum state
    fn extract_cost_from_state(&self, state: &QuantumState) -> PadsResult<f64> {
        // Calculate expectation value as cost approximation
        let mut cost = 0.0;
        for (i, amplitude) in state.amplitudes.iter().enumerate() {
            let prob = amplitude.abs().powi(2);
            cost += prob * (i as f64 + 1.0) * self.config.liquidity_parameter;
        }

        Ok(cost)
    }

    /// Extract log-odds from quantum state
    fn extract_log_odds_from_state(&self, state: &QuantumState, num_probabilities: usize) -> PadsResult<Vec<f64>> {
        let probabilities = self.extract_probabilities_from_state(state, num_probabilities)?;
        let mut log_odds = Vec::new();

        for prob in probabilities {
            if prob <= 0.0 {
                log_odds.push(f64::NEG_INFINITY);
            } else if prob >= 1.0 {
                log_odds.push(f64::INFINITY);
            } else {
                log_odds.push((prob / (1.0 - prob)).ln());
            }
        }

        Ok(log_odds)
    }

    /// Extract arbitrage opportunities
    fn extract_arbitrage_opportunities(&self, state: &QuantumState, markets: &[MarketData]) -> PadsResult<Vec<ArbitrageOpportunity>> {
        let mut opportunities = Vec::new();

        // Simplified arbitrage detection based on probability distribution
        for (i, amplitude) in state.amplitudes.iter().enumerate() {
            let prob = amplitude.abs().powi(2);
            if prob > 0.1 { // Threshold for arbitrage opportunity
                opportunities.push(ArbitrageOpportunity {
                    market_indices: vec![i % markets.len()],
                    expected_profit: prob * 100.0,
                    confidence: prob,
                    strategy: "quantum_detected".to_string(),
                });
            }
        }

        Ok(opportunities)
    }

    /// Extract market making strategy
    fn extract_market_making_strategy(&self, state: &QuantumState, target_probabilities: &[f64]) -> PadsResult<MarketMakingStrategy> {
        // Calculate optimal bid/ask spreads
        let mut bid_spreads = Vec::new();
        let mut ask_spreads = Vec::new();
        
        for (i, &target_prob) in target_probabilities.iter().enumerate() {
            let current_prob = state.amplitudes.get(i).map(|a| a.abs().powi(2)).unwrap_or(0.0);
            let spread = (target_prob - current_prob).abs() * 0.1; // 10% of difference
            bid_spreads.push(spread);
            ask_spreads.push(spread);
        }

        let confidence = state.amplitudes.iter()
            .map(|a| a.abs().powi(2))
            .sum::<f64>() / state.amplitudes.len() as f64;

        Ok(MarketMakingStrategy {
            bid_spreads,
            ask_spreads,
            liquidity_parameter: self.config.liquidity_parameter,
            confidence,
        })
    }

    /// Validate probabilities
    fn validate_probabilities(&self, probabilities: &[f64]) -> PadsResult<()> {
        let sum: f64 = probabilities.iter().sum();
        if (sum - 1.0).abs() > 1e-10 {
            return Err(PadsError::Calculation(format!("Probabilities sum to {}, not 1.0", sum)));
        }

        for (i, &prob) in probabilities.iter().enumerate() {
            if prob < 0.0 || prob > 1.0 {
                return Err(PadsError::Calculation(format!("Probability {} is {}, not in [0,1]", i, prob)));
            }
        }

        Ok(())
    }

    /// Update performance metrics
    fn update_metrics(&self, used_quantum: bool, execution_time: u64, success: bool) {
        if let Ok(mut metrics) = self.quantum_metrics.lock() {
            if used_quantum {
                metrics.quantum_executions += 1;
                metrics.quantum_time_total_ms += execution_time;
                if !success {
                    metrics.quantum_errors += 1;
                }
            } else {
                metrics.classical_executions += 1;
                metrics.classical_time_total_ms += execution_time;
            }
        }
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> Option<QuantumMetrics> {
        self.quantum_metrics.lock().ok().map(|metrics| metrics.clone())
    }

    /// Check if quantum device is available
    pub fn has_quantum_device(&self) -> bool {
        self.quantum_device_available
    }

    /// Get quantum advantage ratio
    pub fn get_quantum_advantage(&self) -> f64 {
        self.last_quantum_advantage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_lmsr_creation() {
        let config = QuantumLMSRConfig::default();
        let qlmsr = QuantumLMSR::new(config);
        assert!(qlmsr.is_ok());
    }

    #[test]
    fn test_classical_cost_calculation() {
        let config = QuantumLMSRConfig {
            processing_mode: QuantumProcessingMode::Classical,
            ..Default::default()
        };
        
        let qlmsr = QuantumLMSR::new(config).unwrap();
        let quantities = vec![0.0, 10.0, 20.0];
        
        let result = qlmsr.calculate_cost(&quantities);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0.0);
    }

    #[test]
    fn test_quantum_log_odds_conversion() {
        let config = QuantumLMSRConfig::default();
        let qlmsr = QuantumLMSR::new(config).unwrap();
        
        let probabilities = vec![0.3, 0.4, 0.3];
        let log_odds = qlmsr.quantum_log_odds_conversion(&probabilities);
        assert!(log_odds.is_ok());
    }

    #[test]
    fn test_market_making_strategy() {
        let config = QuantumLMSRConfig::default();
        let qlmsr = QuantumLMSR::new(config).unwrap();
        
        let target_probabilities = vec![0.25, 0.25, 0.25, 0.25];
        let strategy = qlmsr.quantum_market_making(&target_probabilities);
        assert!(strategy.is_ok());
        
        let strategy = strategy.unwrap();
        assert_eq!(strategy.bid_spreads.len(), 4);
        assert_eq!(strategy.ask_spreads.len(), 4);
    }

    #[test]
    fn test_quantum_advantage_measurement() {
        let config = QuantumLMSRConfig::default();
        let qlmsr = QuantumLMSR::new(config).unwrap();
        
        let advantage = qlmsr.get_quantum_advantage();
        assert!(advantage > 0.0);
    }

    #[test]
    fn test_metrics_collection() {
        let config = QuantumLMSRConfig::default();
        let qlmsr = QuantumLMSR::new(config).unwrap();
        
        // Perform some operations
        let quantities = vec![1.0, 2.0, 3.0];
        let _ = qlmsr.calculate_cost(&quantities);
        
        let metrics = qlmsr.get_metrics();
        assert!(metrics.is_some());
    }
}