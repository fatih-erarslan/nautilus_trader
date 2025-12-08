//! Quantum Circuit Builder for Pair Analysis
//!
//! This module provides specialized quantum circuit construction for 
//! trading pair analysis and optimization.

use std::collections::HashMap;
use std::f64::consts::PI;
use anyhow::{Result, Context};
use tracing::{info, debug, warn};
use serde::{Deserialize, Serialize};
use quantum_core::{
    QuantumCircuit, QuantumState, QuantumGate, CircuitBuilder, 
    ComplexAmplitude, QuantumResult, QuantumError
};

use crate::{PairMetrics, AnalyzerError};
use super::{QuantumConfig, QuantumProblem, QuantumProblemParameters};

/// Quantum circuit builder for pair analysis
#[derive(Debug)]
pub struct QuantumCircuitBuilder {
    config: QuantumConfig,
    circuit_cache: HashMap<String, QuantumCircuit>,
    gate_sequence_optimizer: GateSequenceOptimizer,
}

/// Gate sequence optimizer for circuit efficiency
#[derive(Debug)]
pub struct GateSequenceOptimizer {
    optimization_level: OptimizationLevel,
    gate_fusion_enabled: bool,
    decomposition_enabled: bool,
    commutation_analysis: bool,
}

/// Circuit optimization levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Intermediate,
    Advanced,
    Maximum,
}

/// Entanglement circuit types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EntanglementType {
    Bell,
    GHZ,
    W,
    ClusterState,
    GraphState,
    Custom,
}

/// Quantum feature encoding methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FeatureEncoding {
    Amplitude,
    Angle,
    Basis,
    Quantum,
    Hybrid,
}

impl QuantumCircuitBuilder {
    /// Create a new quantum circuit builder
    pub async fn new(config: QuantumConfig) -> Result<Self, AnalyzerError> {
        info!("Initializing quantum circuit builder");
        
        let gate_sequence_optimizer = GateSequenceOptimizer {
            optimization_level: OptimizationLevel::Intermediate,
            gate_fusion_enabled: true,
            decomposition_enabled: true,
            commutation_analysis: true,
        };
        
        Ok(Self {
            config,
            circuit_cache: HashMap::new(),
            gate_sequence_optimizer,
        })
    }
    
    /// Build optimization circuit for QAOA
    pub async fn build_optimization_circuit(
        &mut self,
        problem: &QuantumProblem,
    ) -> Result<QuantumCircuit, AnalyzerError> {
        let cache_key = format!("optimization_{}_{}", 
                               problem.parameters.num_qubits, 
                               problem.parameters.optimization_objective as u8);
        
        if let Some(cached_circuit) = self.circuit_cache.get(&cache_key) {
            debug!("Using cached optimization circuit");
            return Ok(cached_circuit.clone());
        }
        
        let circuit = self.build_optimization_circuit_impl(problem).await?;
        self.circuit_cache.insert(cache_key, circuit.clone());
        
        Ok(circuit)
    }
    
    /// Build entanglement measurement circuit
    pub async fn build_entanglement_circuit(
        &mut self,
        pair1: &PairMetrics,
        pair2: &PairMetrics,
    ) -> Result<QuantumCircuit, AnalyzerError> {
        debug!("Building entanglement circuit for pair correlation analysis");
        
        let mut circuit = QuantumCircuit::new("entanglement_measurement".to_string(), 4);
        
        // Encode pair metrics into quantum states
        self.encode_pair_metrics(&mut circuit, pair1, 0, 1).await?;
        self.encode_pair_metrics(&mut circuit, pair2, 2, 3).await?;
        
        // Create entanglement between pairs
        self.create_pair_entanglement(&mut circuit, EntanglementType::Bell).await?;
        
        // Add measurement basis rotation
        self.add_entanglement_measurement_basis(&mut circuit).await?;
        
        // Optimize circuit
        self.gate_sequence_optimizer.optimize_circuit(&mut circuit).await?;
        
        Ok(circuit)
    }
    
    /// Build portfolio optimization circuit
    pub async fn build_portfolio_circuit(
        &mut self,
        pairs: &[PairMetrics],
        max_portfolio_size: usize,
    ) -> Result<QuantumCircuit, AnalyzerError> {
        let num_pairs = pairs.len().min(self.config.max_qubits);
        let mut circuit = QuantumCircuit::new("portfolio_optimization".to_string(), num_pairs);
        
        // Initialize superposition
        for qubit in 0..num_pairs {
            circuit.add_hadamard(qubit)
                .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        }
        
        // Add portfolio constraints
        self.add_portfolio_constraints(&mut circuit, pairs, max_portfolio_size).await?;
        
        // Add risk-return optimization
        self.add_risk_return_optimization(&mut circuit, pairs).await?;
        
        // Add diversification constraints
        self.add_diversification_constraints(&mut circuit, pairs).await?;
        
        Ok(circuit)
    }
    
    /// Build quantum feature map circuit
    pub async fn build_feature_map_circuit(
        &mut self,
        features: &[f64],
        encoding: FeatureEncoding,
    ) -> Result<QuantumCircuit, AnalyzerError> {
        let num_features = features.len();
        let num_qubits = (num_features as f64).log2().ceil() as usize;
        let mut circuit = QuantumCircuit::new("feature_map".to_string(), num_qubits);
        
        match encoding {
            FeatureEncoding::Amplitude => {
                self.amplitude_encoding(&mut circuit, features).await?;
            }
            FeatureEncoding::Angle => {
                self.angle_encoding(&mut circuit, features).await?;
            }
            FeatureEncoding::Basis => {
                self.basis_encoding(&mut circuit, features).await?;
            }
            FeatureEncoding::Quantum => {
                self.quantum_encoding(&mut circuit, features).await?;
            }
            FeatureEncoding::Hybrid => {
                self.hybrid_encoding(&mut circuit, features).await?;
            }
        }
        
        Ok(circuit)
    }
    
    /// Build variational quantum eigensolver circuit
    pub async fn build_vqe_circuit(
        &mut self,
        hamiltonian: &[Vec<f64>],
        parameters: &[f64],
    ) -> Result<QuantumCircuit, AnalyzerError> {
        let num_qubits = hamiltonian.len();
        let mut circuit = QuantumCircuit::new("vqe_ansatz".to_string(), num_qubits);
        
        // Hardware-efficient ansatz
        self.add_hardware_efficient_ansatz(&mut circuit, parameters).await?;
        
        // Add problem-specific layers
        self.add_problem_specific_layers(&mut circuit, hamiltonian).await?;
        
        Ok(circuit)
    }
    
    /// Implementation of optimization circuit building
    async fn build_optimization_circuit_impl(
        &self,
        problem: &QuantumProblem,
    ) -> Result<QuantumCircuit, AnalyzerError> {
        let num_qubits = problem.parameters.num_qubits;
        let mut circuit = QuantumCircuit::new("optimization".to_string(), num_qubits);
        
        // Initial state preparation
        self.prepare_initial_state(&mut circuit, &problem.pair_metadata).await?;
        
        // Add cost function encoding
        self.encode_cost_function(&mut circuit, &problem.parameters.cost_matrix).await?;
        
        // Add constraint encoding
        self.encode_constraints(&mut circuit, &problem.parameters.constraint_matrices).await?;
        
        // Add variational layers
        self.add_variational_layers(&mut circuit, num_qubits).await?;
        
        Ok(circuit)
    }
    
    /// Encode pair metrics into quantum state
    async fn encode_pair_metrics(
        &self,
        circuit: &mut QuantumCircuit,
        pair: &PairMetrics,
        qubit1: usize,
        qubit2: usize,
    ) -> Result<(), AnalyzerError> {
        // Encode correlation as rotation angle
        let correlation_angle = pair.correlation_score * PI / 2.0;
        circuit.add_gate(
            QuantumGate::RY { qubit: qubit1, angle: correlation_angle },
            vec![qubit1],
        ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        
        // Encode volatility as phase
        let volatility_phase = pair.volatility_ratio * PI;
        circuit.add_gate(
            QuantumGate::RZ { qubit: qubit2, angle: volatility_phase },
            vec![qubit2],
        ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        
        // Encode liquidity as controlled rotation
        let liquidity_angle = pair.liquidity_ratio * PI / 4.0;
        circuit.add_cnot(qubit1, qubit2)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        circuit.add_gate(
            QuantumGate::RY { qubit: qubit2, angle: liquidity_angle },
            vec![qubit2],
        ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        circuit.add_cnot(qubit1, qubit2)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        
        Ok(())
    }
    
    /// Create entanglement between pairs
    async fn create_pair_entanglement(
        &self,
        circuit: &mut QuantumCircuit,
        entanglement_type: EntanglementType,
    ) -> Result<(), AnalyzerError> {
        match entanglement_type {
            EntanglementType::Bell => {
                // Bell state between qubits 0-1 and 2-3
                circuit.add_cnot(0, 1)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_cnot(2, 3)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
            EntanglementType::GHZ => {
                // GHZ state across all qubits
                circuit.add_cnot(0, 1)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_cnot(1, 2)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_cnot(2, 3)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
            EntanglementType::W => {
                // W state approximation
                circuit.add_gate(
                    QuantumGate::RY { qubit: 0, angle: std::f64::consts::FRAC_PI_3 },
                    vec![0],
                ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_cnot(0, 1)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_cnot(1, 2)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_cnot(2, 3)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
            EntanglementType::ClusterState => {
                // Linear cluster state
                circuit.add_hadamard(0)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_hadamard(1)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_hadamard(2)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_hadamard(3)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                
                circuit.add_gate(
                    QuantumGate::CZ { control: 0, target: 1 },
                    vec![0, 1],
                ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_gate(
                    QuantumGate::CZ { control: 1, target: 2 },
                    vec![1, 2],
                ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_gate(
                    QuantumGate::CZ { control: 2, target: 3 },
                    vec![2, 3],
                ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
            EntanglementType::GraphState => {
                // Complete graph state
                for i in 0..4 {
                    circuit.add_hadamard(i)
                        .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                }
                for i in 0..4 {
                    for j in (i + 1)..4 {
                        circuit.add_gate(
                            QuantumGate::CZ { control: i, target: j },
                            vec![i, j],
                        ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                    }
                }
            }
            EntanglementType::Custom => {
                // Custom entanglement pattern for trading pairs
                self.add_custom_trading_entanglement(circuit).await?;
            }
        }
        
        Ok(())
    }
    
    /// Add custom trading-specific entanglement
    async fn add_custom_trading_entanglement(
        &self,
        circuit: &mut QuantumCircuit,
    ) -> Result<(), AnalyzerError> {
        // Create correlation-based entanglement
        circuit.add_gate(
            QuantumGate::RY { qubit: 0, angle: PI / 4.0 },
            vec![0],
        ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        
        circuit.add_gate(
            QuantumGate::RY { qubit: 2, angle: PI / 4.0 },
            vec![2],
        ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        
        // Cross-correlation entanglement
        circuit.add_cnot(0, 2)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        circuit.add_cnot(1, 3)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        
        // Phase correlation
        circuit.add_gate(
            QuantumGate::CPhase { control: 0, target: 1, phase: PI / 8.0 },
            vec![0, 1],
        ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        
        circuit.add_gate(
            QuantumGate::CPhase { control: 2, target: 3, phase: PI / 8.0 },
            vec![2, 3],
        ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        
        Ok(())
    }
    
    /// Add measurement basis for entanglement
    async fn add_entanglement_measurement_basis(
        &self,
        circuit: &mut QuantumCircuit,
    ) -> Result<(), AnalyzerError> {
        // Bell basis measurement
        circuit.add_cnot(0, 1)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        circuit.add_hadamard(0)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        
        circuit.add_cnot(2, 3)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        circuit.add_hadamard(2)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        
        Ok(())
    }
    
    /// Prepare initial state for optimization
    async fn prepare_initial_state(
        &self,
        circuit: &mut QuantumCircuit,
        pairs: &[PairMetrics],
    ) -> Result<(), AnalyzerError> {
        let num_qubits = circuit.num_qubits;
        
        // Initialize with uniform superposition
        for qubit in 0..num_qubits {
            circuit.add_hadamard(qubit)
                .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        }
        
        // Add bias based on pair quality
        for (i, pair) in pairs.iter().enumerate().take(num_qubits) {
            let bias_angle = pair.composite_score * PI / 4.0;
            circuit.add_gate(
                QuantumGate::RY { qubit: i, angle: bias_angle },
                vec![i],
            ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        }
        
        Ok(())
    }
    
    /// Encode cost function into circuit
    async fn encode_cost_function(
        &self,
        circuit: &mut QuantumCircuit,
        cost_matrix: &nalgebra::DMatrix<f64>,
    ) -> Result<(), AnalyzerError> {
        let num_qubits = circuit.num_qubits;
        
        // Encode diagonal terms
        for i in 0..num_qubits {
            if i < cost_matrix.nrows() && i < cost_matrix.ncols() {
                let cost = cost_matrix[(i, i)];
                let angle = cost * PI / 4.0;
                circuit.add_gate(
                    QuantumGate::RZ { qubit: i, angle },
                    vec![i],
                ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
        }
        
        // Encode off-diagonal terms
        for i in 0..num_qubits {
            for j in (i + 1)..num_qubits {
                if i < cost_matrix.nrows() && j < cost_matrix.ncols() {
                    let coupling = cost_matrix[(i, j)];
                    if coupling.abs() > 1e-10 {
                        let angle = coupling * PI / 8.0;
                        circuit.add_cnot(i, j)
                            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                        circuit.add_gate(
                            QuantumGate::RZ { qubit: j, angle },
                            vec![j],
                        ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                        circuit.add_cnot(i, j)
                            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Encode constraints into circuit
    async fn encode_constraints(
        &self,
        circuit: &mut QuantumCircuit,
        constraint_matrices: &[nalgebra::DMatrix<f64>],
    ) -> Result<(), AnalyzerError> {
        // Add penalty terms for constraint violations
        for (k, constraint) in constraint_matrices.iter().enumerate() {
            let penalty_weight = 1.0 / (k + 1) as f64;
            
            for i in 0..circuit.num_qubits {
                if i < constraint.ncols() {
                    let constraint_value = constraint[(0, i)];
                    let penalty_angle = penalty_weight * constraint_value * PI / 16.0;
                    
                    circuit.add_gate(
                        QuantumGate::RZ { qubit: i, angle: penalty_angle },
                        vec![i],
                    ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Add variational layers
    async fn add_variational_layers(
        &self,
        circuit: &mut QuantumCircuit,
        num_qubits: usize,
    ) -> Result<(), AnalyzerError> {
        let num_layers = 2; // Configurable
        
        for layer in 0..num_layers {
            // Rotation layer
            for qubit in 0..num_qubits {
                circuit.add_gate(
                    QuantumGate::RY { qubit, angle: PI / 4.0 },
                    vec![qubit],
                ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
            
            // Entangling layer
            for qubit in 0..(num_qubits - 1) {
                circuit.add_cnot(qubit, qubit + 1)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
            
            // Wrap around for ring topology
            if num_qubits > 2 {
                circuit.add_cnot(num_qubits - 1, 0)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
        }
        
        Ok(())
    }
    
    /// Add portfolio constraints
    async fn add_portfolio_constraints(
        &self,
        circuit: &mut QuantumCircuit,
        pairs: &[PairMetrics],
        max_size: usize,
    ) -> Result<(), AnalyzerError> {
        let num_qubits = circuit.num_qubits;
        
        // Add size constraint penalty
        for i in 0..num_qubits {
            for j in (i + 1)..num_qubits {
                let penalty = if max_size > 0 { 1.0 / max_size as f64 } else { 0.1 };
                let angle = penalty * PI / 16.0;
                
                circuit.add_cnot(i, j)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_gate(
                    QuantumGate::RZ { qubit: j, angle },
                    vec![j],
                ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                circuit.add_cnot(i, j)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
        }
        
        Ok(())
    }
    
    /// Add risk-return optimization
    async fn add_risk_return_optimization(
        &self,
        circuit: &mut QuantumCircuit,
        pairs: &[PairMetrics],
    ) -> Result<(), AnalyzerError> {
        let num_qubits = circuit.num_qubits;
        
        for (i, pair) in pairs.iter().enumerate().take(num_qubits) {
            // Return term (positive)
            let return_angle = pair.expected_return * PI / 8.0;
            circuit.add_gate(
                QuantumGate::RZ { qubit: i, angle: -return_angle },
                vec![i],
            ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            
            // Risk term (negative)
            let risk_angle = pair.value_at_risk * PI / 8.0;
            circuit.add_gate(
                QuantumGate::RZ { qubit: i, angle: risk_angle },
                vec![i],
            ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        }
        
        Ok(())
    }
    
    /// Add diversification constraints
    async fn add_diversification_constraints(
        &self,
        circuit: &mut QuantumCircuit,
        pairs: &[PairMetrics],
    ) -> Result<(), AnalyzerError> {
        let num_qubits = circuit.num_qubits;
        
        // Add anti-correlation bonus
        for i in 0..num_qubits {
            for j in (i + 1)..num_qubits {
                if i < pairs.len() && j < pairs.len() {
                    let correlation = pairs[i].correlation_score;
                    if correlation < 0.0 {
                        let bonus_angle = correlation.abs() * PI / 16.0;
                        circuit.add_cnot(i, j)
                            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                        circuit.add_gate(
                            QuantumGate::RZ { qubit: j, angle: -bonus_angle },
                            vec![j],
                        ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                        circuit.add_cnot(i, j)
                            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Amplitude encoding
    async fn amplitude_encoding(
        &self,
        circuit: &mut QuantumCircuit,
        features: &[f64],
    ) -> Result<(), AnalyzerError> {
        // Normalize features
        let norm: f64 = features.iter().map(|f| f * f).sum::<f64>().sqrt();
        let normalized_features: Vec<f64> = features.iter().map(|f| f / norm).collect();
        
        // Use controlled rotations to encode amplitudes
        for (i, &feature) in normalized_features.iter().enumerate() {
            if i < circuit.num_qubits {
                let angle = 2.0 * feature.acos();
                circuit.add_gate(
                    QuantumGate::RY { qubit: i, angle },
                    vec![i],
                ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
        }
        
        Ok(())
    }
    
    /// Angle encoding
    async fn angle_encoding(
        &self,
        circuit: &mut QuantumCircuit,
        features: &[f64],
    ) -> Result<(), AnalyzerError> {
        for (i, &feature) in features.iter().enumerate() {
            if i < circuit.num_qubits {
                let angle = feature * PI;
                circuit.add_gate(
                    QuantumGate::RY { qubit: i, angle },
                    vec![i],
                ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
        }
        
        Ok(())
    }
    
    /// Basis encoding
    async fn basis_encoding(
        &self,
        circuit: &mut QuantumCircuit,
        features: &[f64],
    ) -> Result<(), AnalyzerError> {
        // Convert features to binary representation
        for (i, &feature) in features.iter().enumerate() {
            if i < circuit.num_qubits {
                if feature > 0.5 {
                    circuit.add_pauli_x(i)
                        .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Quantum encoding
    async fn quantum_encoding(
        &self,
        circuit: &mut QuantumCircuit,
        features: &[f64],
    ) -> Result<(), AnalyzerError> {
        // Quantum feature map with entanglement
        for (i, &feature) in features.iter().enumerate() {
            if i < circuit.num_qubits {
                let angle = feature * PI;
                circuit.add_gate(
                    QuantumGate::RY { qubit: i, angle },
                    vec![i],
                ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
        }
        
        // Add entanglement
        for i in 0..(circuit.num_qubits - 1) {
            circuit.add_cnot(i, i + 1)
                .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        }
        
        Ok(())
    }
    
    /// Hybrid encoding
    async fn hybrid_encoding(
        &self,
        circuit: &mut QuantumCircuit,
        features: &[f64],
    ) -> Result<(), AnalyzerError> {
        // Combine angle and amplitude encoding
        let half = features.len() / 2;
        
        // Angle encoding for first half
        self.angle_encoding(circuit, &features[..half]).await?;
        
        // Amplitude encoding for second half
        if half < features.len() {
            self.amplitude_encoding(circuit, &features[half..]).await?;
        }
        
        Ok(())
    }
    
    /// Add hardware-efficient ansatz
    async fn add_hardware_efficient_ansatz(
        &self,
        circuit: &mut QuantumCircuit,
        parameters: &[f64],
    ) -> Result<(), AnalyzerError> {
        let num_qubits = circuit.num_qubits;
        let num_layers = parameters.len() / (num_qubits * 3); // 3 parameters per qubit per layer
        
        for layer in 0..num_layers {
            // Rotation layer
            for qubit in 0..num_qubits {
                let base_idx = layer * num_qubits * 3 + qubit * 3;
                if base_idx + 2 < parameters.len() {
                    circuit.add_gate(
                        QuantumGate::RX { qubit, angle: parameters[base_idx] },
                        vec![qubit],
                    ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                    
                    circuit.add_gate(
                        QuantumGate::RY { qubit, angle: parameters[base_idx + 1] },
                        vec![qubit],
                    ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                    
                    circuit.add_gate(
                        QuantumGate::RZ { qubit, angle: parameters[base_idx + 2] },
                        vec![qubit],
                    ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                }
            }
            
            // Entangling layer
            for qubit in 0..(num_qubits - 1) {
                circuit.add_cnot(qubit, qubit + 1)
                    .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
            }
        }
        
        Ok(())
    }
    
    /// Add problem-specific layers
    async fn add_problem_specific_layers(
        &self,
        circuit: &mut QuantumCircuit,
        hamiltonian: &[Vec<f64>],
    ) -> Result<(), AnalyzerError> {
        let num_qubits = circuit.num_qubits;
        
        // Add Hamiltonian evolution
        for i in 0..num_qubits {
            for j in 0..num_qubits {
                if i < hamiltonian.len() && j < hamiltonian[i].len() {
                    let coupling = hamiltonian[i][j];
                    if coupling.abs() > 1e-10 {
                        let angle = coupling * PI / 8.0;
                        if i == j {
                            circuit.add_gate(
                                QuantumGate::RZ { qubit: i, angle },
                                vec![i],
                            ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                        } else {
                            circuit.add_cnot(i, j)
                                .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                            circuit.add_gate(
                                QuantumGate::RZ { qubit: j, angle },
                                vec![j],
                            ).map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                            circuit.add_cnot(i, j)
                                .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Clear circuit cache
    pub fn clear_cache(&mut self) {
        self.circuit_cache.clear();
    }
    
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("cache_size".to_string(), self.circuit_cache.len());
        stats.insert("total_circuits".to_string(), self.circuit_cache.len());
        stats
    }
}

impl GateSequenceOptimizer {
    /// Optimize quantum circuit
    pub async fn optimize_circuit(
        &self,
        circuit: &mut QuantumCircuit,
    ) -> Result<(), AnalyzerError> {
        match self.optimization_level {
            OptimizationLevel::None => Ok(()),
            OptimizationLevel::Basic => {
                self.basic_optimization(circuit).await
            }
            OptimizationLevel::Intermediate => {
                self.intermediate_optimization(circuit).await
            }
            OptimizationLevel::Advanced => {
                self.advanced_optimization(circuit).await
            }
            OptimizationLevel::Maximum => {
                self.maximum_optimization(circuit).await
            }
        }
    }
    
    /// Basic optimization
    async fn basic_optimization(
        &self,
        circuit: &mut QuantumCircuit,
    ) -> Result<(), AnalyzerError> {
        circuit.optimize(quantum_core::OptimizationLevel::Basic)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        Ok(())
    }
    
    /// Intermediate optimization
    async fn intermediate_optimization(
        &self,
        circuit: &mut QuantumCircuit,
    ) -> Result<(), AnalyzerError> {
        circuit.optimize(quantum_core::OptimizationLevel::Aggressive)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        Ok(())
    }
    
    /// Advanced optimization
    async fn advanced_optimization(
        &self,
        circuit: &mut QuantumCircuit,
    ) -> Result<(), AnalyzerError> {
        circuit.optimize(quantum_core::OptimizationLevel::Maximum)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        Ok(())
    }
    
    /// Maximum optimization
    async fn maximum_optimization(
        &self,
        circuit: &mut QuantumCircuit,
    ) -> Result<(), AnalyzerError> {
        circuit.optimize(quantum_core::OptimizationLevel::Maximum)
            .map_err(|e| AnalyzerError::QuantumError(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PairId;
    use chrono::Utc;
    
    #[tokio::test]
    async fn test_circuit_builder_creation() {
        let config = QuantumConfig::default();
        let builder = QuantumCircuitBuilder::new(config).await;
        assert!(builder.is_ok());
    }
    
    #[tokio::test]
    async fn test_entanglement_circuit_building() {
        let config = QuantumConfig::default();
        let mut builder = QuantumCircuitBuilder::new(config).await.unwrap();
        
        let pair1 = create_test_pair_metrics("BTC", "USD");
        let pair2 = create_test_pair_metrics("ETH", "USD");
        
        let circuit = builder.build_entanglement_circuit(&pair1, &pair2).await;
        assert!(circuit.is_ok());
        
        let circuit = circuit.unwrap();
        assert_eq!(circuit.num_qubits, 4);
        assert!(circuit.instructions.len() > 0);
    }
    
    #[tokio::test]
    async fn test_feature_map_circuit() {
        let config = QuantumConfig::default();
        let mut builder = QuantumCircuitBuilder::new(config).await.unwrap();
        
        let features = vec![0.5, 0.3, 0.8, 0.2];
        let circuit = builder.build_feature_map_circuit(&features, FeatureEncoding::Angle).await;
        assert!(circuit.is_ok());
        
        let circuit = circuit.unwrap();
        assert!(circuit.num_qubits > 0);
        assert!(circuit.instructions.len() > 0);
    }
    
    #[tokio::test]
    async fn test_portfolio_circuit() {
        let config = QuantumConfig::default();
        let mut builder = QuantumCircuitBuilder::new(config).await.unwrap();
        
        let pairs = vec![
            create_test_pair_metrics("BTC", "USD"),
            create_test_pair_metrics("ETH", "USD"),
            create_test_pair_metrics("ADA", "USD"),
        ];
        
        let circuit = builder.build_portfolio_circuit(&pairs, 2).await;
        assert!(circuit.is_ok());
        
        let circuit = circuit.unwrap();
        assert_eq!(circuit.num_qubits, 3);
        assert!(circuit.instructions.len() > 0);
    }
    
    #[tokio::test]
    async fn test_vqe_circuit() {
        let config = QuantumConfig::default();
        let mut builder = QuantumCircuitBuilder::new(config).await.unwrap();
        
        let hamiltonian = vec![
            vec![1.0, 0.5],
            vec![0.5, 1.0],
        ];
        let parameters = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        
        let circuit = builder.build_vqe_circuit(&hamiltonian, &parameters).await;
        assert!(circuit.is_ok());
        
        let circuit = circuit.unwrap();
        assert_eq!(circuit.num_qubits, 2);
        assert!(circuit.instructions.len() > 0);
    }
    
    #[tokio::test]
    async fn test_circuit_optimization() {
        let config = QuantumConfig::default();
        let mut builder = QuantumCircuitBuilder::new(config).await.unwrap();
        
        let mut circuit = QuantumCircuit::new("test".to_string(), 2);
        circuit.add_hadamard(0).unwrap();
        circuit.add_identity(0).unwrap();
        circuit.add_cnot(0, 1).unwrap();
        
        let initial_gates = circuit.instructions.len();
        
        builder.gate_sequence_optimizer.optimize_circuit(&mut circuit).await.unwrap();
        
        let final_gates = circuit.instructions.len();
        assert!(final_gates <= initial_gates);
    }
    
    #[test]
    fn test_entanglement_types() {
        let types = vec![
            EntanglementType::Bell,
            EntanglementType::GHZ,
            EntanglementType::W,
            EntanglementType::ClusterState,
            EntanglementType::GraphState,
            EntanglementType::Custom,
        ];
        
        assert_eq!(types.len(), 6);
    }
    
    #[test]
    fn test_feature_encoding_types() {
        let encodings = vec![
            FeatureEncoding::Amplitude,
            FeatureEncoding::Angle,
            FeatureEncoding::Basis,
            FeatureEncoding::Quantum,
            FeatureEncoding::Hybrid,
        ];
        
        assert_eq!(encodings.len(), 5);
    }
    
    fn create_test_pair_metrics(base: &str, quote: &str) -> PairMetrics {
        PairMetrics {
            pair_id: PairId::new(base, quote, "binance"),
            timestamp: Utc::now(),
            correlation_score: 0.5,
            cointegration_p_value: 0.01,
            volatility_ratio: 0.3,
            liquidity_ratio: 0.8,
            sentiment_divergence: 0.2,
            news_sentiment_score: 0.6,
            social_sentiment_score: 0.7,
            cuckoo_score: 0.0,
            firefly_score: 0.0,
            ant_colony_score: 0.0,
            quantum_entanglement: 0.0,
            quantum_advantage: 0.5,
            expected_return: 0.15,
            sharpe_ratio: 1.2,
            maximum_drawdown: 0.1,
            value_at_risk: 0.05,
            composite_score: 0.8,
            confidence: 0.9,
        }
    }
}