//! pBit-based Quantum Enhancement for Q* Algorithm
//!
//! Replaces traditional quantum circuits with pBit probabilistic computing
//! for quantum-inspired decision making in trading systems.
//!
//! ## Key Features
//!
//! - Quantum superposition → pBit probability distribution
//! - Quantum entanglement → pBit ferromagnetic coupling
//! - Quantum measurement → Boltzmann sampling
//! - Quantum error correction → pBit majority voting

use q_star_core::{QStarError, MarketState, QStarAction};
use quantum_core::{PBitState, PBitConfig, PBitCoupling};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{debug, info};

/// pBit quantum state for Q* algorithm
#[derive(Debug, Clone)]
pub struct PBitQuantumState {
    /// Number of pBits (replaces qubits)
    num_pbits: usize,
    /// pBit state
    state: PBitState,
    /// State vector representation for compatibility
    amplitudes: Vec<f64>,
    /// Entanglement map (pBit pairs with coupling strength)
    entanglements: HashMap<(usize, usize), f64>,
}

impl PBitQuantumState {
    /// Create new pBit quantum state
    pub fn new(num_pbits: usize) -> Result<Self, QStarError> {
        let config = PBitConfig {
            temperature: 1.0,
            coupling_strength: 1.0,
            external_field: 0.0,
            seed: None,
        };
        
        let state = PBitState::with_config(num_pbits, config)
            .map_err(|e| QStarError::QuantumError(e.to_string()))?;
        
        // Initialize amplitudes (all equal superposition)
        let amplitudes = vec![1.0 / (num_pbits as f64).sqrt(); num_pbits];
        
        Ok(Self {
            num_pbits,
            state,
            amplitudes,
            entanglements: HashMap::new(),
        })
    }
    
    /// Apply Hadamard-like operation (set to equal superposition)
    pub fn hadamard(&mut self, target: usize) -> Result<(), QStarError> {
        if target >= self.num_pbits {
            return Err(QStarError::QuantumError("Target qubit out of range".into()));
        }
        
        if let Some(pbit) = self.state.get_pbit_mut(target) {
            pbit.probability_up = 0.5; // Equal superposition
            pbit.bias = 0.0;
        }
        
        Ok(())
    }
    
    /// Apply rotation (RY-like)
    pub fn rotate_y(&mut self, target: usize, theta: f64) -> Result<(), QStarError> {
        if target >= self.num_pbits {
            return Err(QStarError::QuantumError("Target qubit out of range".into()));
        }
        
        if let Some(pbit) = self.state.get_pbit_mut(target) {
            // RY(θ) → P(↑) = sin²(θ/2)
            pbit.probability_up = (theta / 2.0).sin().powi(2);
        }
        
        Ok(())
    }
    
    /// Apply CNOT-like entanglement
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<(), QStarError> {
        if control >= self.num_pbits || target >= self.num_pbits {
            return Err(QStarError::QuantumError("Qubit index out of range".into()));
        }
        
        // Add ferromagnetic coupling
        self.state.add_coupling(PBitCoupling::bell_coupling(control, target, 1.0));
        self.entanglements.insert((control, target), 1.0);
        
        Ok(())
    }
    
    /// Apply CZ-like phase entanglement
    pub fn cz(&mut self, q1: usize, q2: usize) -> Result<(), QStarError> {
        if q1 >= self.num_pbits || q2 >= self.num_pbits {
            return Err(QStarError::QuantumError("Qubit index out of range".into()));
        }
        
        // CZ creates anti-ferromagnetic-like coupling
        self.state.add_coupling(PBitCoupling::anti_bell_coupling(q1, q2, 0.5));
        self.entanglements.insert((q1, q2), -0.5);
        
        Ok(())
    }
    
    /// Measure all pBits (collapse to classical state)
    pub fn measure_all(&mut self, shots: usize) -> Result<Vec<Vec<i8>>, QStarError> {
        let mut results = Vec::with_capacity(shots);
        
        for _ in 0..shots {
            // Equilibrate
            for _ in 0..10 {
                self.state.sweep();
            }
            
            // Sample
            let measurement: Vec<i8> = (0..self.num_pbits)
                .map(|i| {
                    self.state.get_pbit(i)
                        .map(|p| if p.spin > 0.0 { 1 } else { 0 })
                        .unwrap_or(0)
                })
                .collect();
            
            results.push(measurement);
        }
        
        Ok(results)
    }
    
    /// Get expectation value for observable
    pub fn expectation(&self, observable_index: usize) -> f64 {
        if observable_index < self.num_pbits {
            self.state.get_pbit(observable_index)
                .map(|p| p.spin)
                .unwrap_or(0.0)
        } else {
            0.0
        }
    }
    
    /// Get magnetization (overall system state)
    pub fn magnetization(&self) -> f64 {
        self.state.magnetization()
    }
    
    /// Get entropy (uncertainty measure)
    pub fn entropy(&self) -> f64 {
        self.state.entropy()
    }
}

/// pBit quantum circuit for Q* decisions
pub struct PBitQuantumCircuit {
    /// Number of pBits
    num_pbits: usize,
    /// Operations to apply
    operations: Vec<QuantumOp>,
}

#[derive(Debug, Clone)]
enum QuantumOp {
    Hadamard(usize),
    RotateY(usize, f64),
    CNOT(usize, usize),
    CZ(usize, usize),
}

impl PBitQuantumCircuit {
    /// Create new quantum circuit
    pub fn new(num_pbits: usize) -> Self {
        Self {
            num_pbits,
            operations: Vec::new(),
        }
    }
    
    /// Add Hadamard gate
    pub fn h(&mut self, target: usize) -> &mut Self {
        self.operations.push(QuantumOp::Hadamard(target));
        self
    }
    
    /// Add RY rotation
    pub fn ry(&mut self, target: usize, theta: f64) -> &mut Self {
        self.operations.push(QuantumOp::RotateY(target, theta));
        self
    }
    
    /// Add CNOT gate
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.operations.push(QuantumOp::CNOT(control, target));
        self
    }
    
    /// Add CZ gate
    pub fn cz(&mut self, q1: usize, q2: usize) -> &mut Self {
        self.operations.push(QuantumOp::CZ(q1, q2));
        self
    }
    
    /// Execute circuit and return state
    pub fn execute(&self) -> Result<PBitQuantumState, QStarError> {
        let mut state = PBitQuantumState::new(self.num_pbits)?;
        
        for op in &self.operations {
            match op {
                QuantumOp::Hadamard(t) => state.hadamard(*t)?,
                QuantumOp::RotateY(t, theta) => state.rotate_y(*t, *theta)?,
                QuantumOp::CNOT(c, t) => state.cnot(*c, *t)?,
                QuantumOp::CZ(q1, q2) => state.cz(*q1, *q2)?,
            }
        }
        
        // Equilibrate the pBit state
        for _ in 0..20 {
            state.state.sweep();
        }
        
        Ok(state)
    }
}

/// pBit-based Q* decision engine
pub struct PBitQStarEngine {
    /// Configuration
    num_pbits: usize,
    /// Decision cache
    cache: HashMap<u64, QStarAction>,
    /// State history for learning
    history: Vec<(MarketState, QStarAction, f64)>,
}

impl PBitQStarEngine {
    /// Create new pBit Q* engine
    pub fn new(num_pbits: usize) -> Self {
        Self {
            num_pbits,
            cache: HashMap::new(),
            history: Vec::new(),
        }
    }
    
    /// Make trading decision using pBit quantum circuit
    pub fn decide(&mut self, market_state: &MarketState) -> Result<QStarAction, QStarError> {
        // Encode market state into pBit circuit
        let mut circuit = PBitQuantumCircuit::new(self.num_pbits);
        
        // Initialize in superposition
        for i in 0..self.num_pbits {
            circuit.h(i);
        }
        
        // Encode market features as rotations
        let features = self.encode_market_features(market_state);
        for (i, &feature) in features.iter().enumerate().take(self.num_pbits) {
            circuit.ry(i, feature * PI);
        }
        
        // Add entanglement for correlated decisions
        for i in 0..self.num_pbits.saturating_sub(1) {
            circuit.cnot(i, i + 1);
        }
        
        // Execute and measure
        let state = circuit.execute()?;
        let measurements = state.measure_all(100)?;
        
        // Decode decision from measurements
        let action = self.decode_action(&measurements);
        
        Ok(action)
    }
    
    /// Encode market state into feature vector
    fn encode_market_features(&self, market_state: &MarketState) -> Vec<f64> {
        // Normalize market features to [0, 1]
        let mut features = vec![
            (market_state.price / 100000.0).clamp(0.0, 1.0), // Normalize price
            (market_state.volume / 1000000.0).clamp(0.0, 1.0), // Normalize volume
            (market_state.volatility * 10.0).clamp(0.0, 1.0),
            market_state.rsi, // Already 0-1
            ((market_state.macd + 1.0) / 2.0).clamp(0.0, 1.0), // Normalize MACD
        ];
        
        // Add additional features
        features.extend(market_state.features.iter().take(3).map(|&f| f.clamp(0.0, 1.0)));
        
        features
    }
    
    /// Decode action from measurement results
    fn decode_action(&self, measurements: &[Vec<i8>]) -> QStarAction {
        // Count measurement outcomes
        let mut buy_count = 0;
        let mut sell_count = 0;
        let mut hold_count = 0;
        
        for measurement in measurements {
            // Use first few bits to encode action
            let action_bits: i32 = measurement.iter()
                .take(3)
                .enumerate()
                .map(|(i, &b)| (b as i32) << i)
                .sum();
            
            match action_bits % 3 {
                0 => hold_count += 1,
                1 => buy_count += 1,
                2 => sell_count += 1,
                _ => hold_count += 1,
            }
        }
        
        // Return majority action with confidence as amount
        let total = measurements.len() as f64;
        if buy_count >= sell_count && buy_count >= hold_count {
            QStarAction::Buy { amount: buy_count as f64 / total }
        } else if sell_count >= hold_count {
            QStarAction::Sell { amount: sell_count as f64 / total }
        } else {
            QStarAction::Hold
        }
    }
    
    /// Learn from reward
    pub fn learn(&mut self, market_state: &MarketState, action: QStarAction, reward: f64) {
        self.history.push((market_state.clone(), action, reward));
        
        // Limit history size
        if self.history.len() > 10000 {
            self.history.remove(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pbit_quantum_state() {
        let state = PBitQuantumState::new(4);
        assert!(state.is_ok());
        
        let state = state.unwrap();
        assert_eq!(state.num_pbits, 4);
    }
    
    #[test]
    fn test_quantum_circuit() {
        let mut circuit = PBitQuantumCircuit::new(4);
        circuit.h(0).h(1).cnot(0, 1).ry(2, PI / 4.0);
        
        let state = circuit.execute();
        assert!(state.is_ok());
    }
    
    #[test]
    fn test_measurement() {
        let mut state = PBitQuantumState::new(4).unwrap();
        state.hadamard(0).unwrap();
        state.hadamard(1).unwrap();
        state.cnot(0, 1).unwrap();
        
        let measurements = state.measure_all(10);
        assert!(measurements.is_ok());
        assert_eq!(measurements.unwrap().len(), 10);
    }
}
