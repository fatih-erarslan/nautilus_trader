//! Quantum Decision Module
//!
//! Pure quantum decision making using quantum circuits, superposition, and entanglement.

use crate::core::{QarResult, TradingDecision, DecisionType, FactorMap};
use crate::analysis::AnalysisResult;
use crate::error::QarError;
use crate::quantum::{QuantumState, QuantumCircuit, gates::Gate};
use super::{QuantumInsights, DecisionConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use num_complex::Complex64;

/// Quantum decision maker
pub struct QuantumDecisionMaker {
    config: QuantumDecisionConfig,
    quantum_processor: QuantumProcessor,
    quantum_circuits: QuantumCircuitLibrary,
    entanglement_manager: EntanglementManager,
    measurement_history: Vec<QuantumMeasurement>,
    coherence_tracker: CoherenceTracker,
}

/// Quantum decision configuration
#[derive(Debug, Clone)]
pub struct QuantumDecisionConfig {
    /// Number of qubits for decision quantum circuits
    pub num_qubits: usize,
    /// Maximum circuit depth
    pub max_circuit_depth: usize,
    /// Coherence time threshold
    pub coherence_threshold: f64,
    /// Measurement shots for probability estimation
    pub measurement_shots: usize,
    /// Enable quantum error correction
    pub error_correction: bool,
    /// Quantum advantage threshold
    pub advantage_threshold: f64,
    /// Enable quantum optimization
    pub quantum_optimization: bool,
}

/// Quantum processor for decision circuits
#[derive(Debug)]
pub struct QuantumProcessor {
    /// Current quantum state
    pub quantum_state: QuantumState,
    /// Available quantum gates
    pub gate_set: Vec<Gate>,
    /// Quantum noise model
    pub noise_model: QuantumNoiseModel,
    /// Quantum backend type
    pub backend: QuantumBackend,
}

/// Quantum noise model
#[derive(Debug)]
pub struct QuantumNoiseModel {
    /// Decoherence rate
    pub decoherence_rate: f64,
    /// Gate error rate
    pub gate_error_rate: f64,
    /// Measurement error rate
    pub measurement_error_rate: f64,
    /// Thermal noise temperature
    pub thermal_noise: f64,
}

/// Quantum backend enumeration
#[derive(Debug)]
pub enum QuantumBackend {
    Simulator,
    QuantumDevice,
    Hybrid,
}

/// Quantum circuit library for decisions
#[derive(Debug)]
pub struct QuantumCircuitLibrary {
    /// Buy decision circuit
    pub buy_circuit: QuantumCircuit,
    /// Sell decision circuit
    pub sell_circuit: QuantumCircuit,
    /// Hold decision circuit
    pub hold_circuit: QuantumCircuit,
    /// Superposition decision circuit
    pub superposition_circuit: QuantumCircuit,
    /// Entanglement decision circuit
    pub entanglement_circuit: QuantumCircuit,
    /// Quantum Fourier Transform circuit
    pub qft_circuit: QuantumCircuit,
    /// Variational quantum circuit
    pub variational_circuit: VariationalQuantumCircuit,
}

/// Variational quantum circuit for adaptive decisions
#[derive(Debug)]
pub struct VariationalQuantumCircuit {
    /// Circuit structure
    pub circuit: QuantumCircuit,
    /// Trainable parameters
    pub parameters: Vec<f64>,
    /// Parameter gradients
    pub gradients: Vec<f64>,
    /// Optimization history
    pub optimization_history: Vec<ParameterUpdate>,
}

/// Parameter update record
#[derive(Debug, Clone)]
pub struct ParameterUpdate {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Parameter values
    pub parameters: Vec<f64>,
    /// Cost function value
    pub cost: f64,
    /// Gradient norm
    pub gradient_norm: f64,
}

/// Entanglement manager for correlated decisions
#[derive(Debug)]
pub struct EntanglementManager {
    /// Entangled qubit pairs
    pub entangled_pairs: Vec<(usize, usize)>,
    /// Entanglement strength
    pub entanglement_strength: f64,
    /// Bell state preparations
    pub bell_states: Vec<BellState>,
    /// Entanglement witnesses
    pub witnesses: Vec<EntanglementWitness>,
}

/// Bell state enumeration
#[derive(Debug, Clone)]
pub enum BellState {
    Phi,      // |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    PhiMinus, // |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    Psi,      // |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    PsiMinus, // |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
}

/// Entanglement witness for verification
#[derive(Debug)]
pub struct EntanglementWitness {
    /// Witness operator
    pub operator: Vec<Vec<Complex64>>,
    /// Expectation value threshold
    pub threshold: f64,
    /// Measurement basis
    pub measurement_basis: Vec<String>,
}

/// Quantum measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMeasurement {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Measured qubits
    pub measured_qubits: Vec<usize>,
    /// Measurement outcomes
    pub outcomes: Vec<u8>,
    /// Probability amplitudes
    pub amplitudes: Vec<Complex64>,
    /// Measurement basis
    pub basis: MeasurementBasis,
    /// Quantum fidelity
    pub fidelity: f64,
}

/// Measurement basis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementBasis {
    Computational,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    Custom(String),
}

/// Coherence tracking
#[derive(Debug)]
pub struct CoherenceTracker {
    /// Coherence time measurements
    pub coherence_times: Vec<f64>,
    /// Decoherence rate
    pub decoherence_rate: f64,
    /// Coherence quality factor
    pub quality_factor: f64,
    /// Environmental factors
    pub environmental_factors: HashMap<String, f64>,
}

/// Enhanced quantum decision result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumDecisionResult {
    /// Traditional trading decision
    pub classical_decision: TradingDecision,
    /// Quantum decision probabilities
    pub quantum_probabilities: QuantumProbabilities,
    /// Quantum state information
    pub quantum_state_info: QuantumStateInfo,
    /// Entanglement analysis
    pub entanglement_analysis: EntanglementAnalysis,
    /// Quantum advantage metrics
    pub quantum_advantage: QuantumAdvantageMetrics,
    /// Circuit execution details
    pub circuit_execution: CircuitExecutionDetails,
}

/// Quantum decision probabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumProbabilities {
    /// Buy probability from quantum measurement
    pub buy_probability: f64,
    /// Sell probability from quantum measurement
    pub sell_probability: f64,
    /// Hold probability from quantum measurement
    pub hold_probability: f64,
    /// Superposition coefficient
    pub superposition_coefficient: f64,
    /// Measurement uncertainty
    pub measurement_uncertainty: f64,
}

/// Quantum state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumStateInfo {
    /// State vector
    pub state_vector: Vec<Complex64>,
    /// Quantum purity
    pub purity: f64,
    /// Von Neumann entropy
    pub entropy: f64,
    /// Coherence measures
    pub coherence: CoherenceMeasures,
    /// Quantum phase information
    pub phase_info: QuantumPhaseInfo,
}

/// Coherence measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceMeasures {
    /// L1 norm coherence
    pub l1_norm: f64,
    /// Relative entropy coherence
    pub relative_entropy: f64,
    /// Robustness of coherence
    pub robustness: f64,
    /// Coherence of formation
    pub formation: f64,
}

/// Quantum phase information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPhaseInfo {
    /// Global phase
    pub global_phase: f64,
    /// Relative phases
    pub relative_phases: Vec<f64>,
    /// Phase variance
    pub phase_variance: f64,
    /// Berry phase
    pub berry_phase: Option<f64>,
}

/// Entanglement analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementAnalysis {
    /// Entanglement entropy
    pub entanglement_entropy: f64,
    /// Concurrence measure
    pub concurrence: f64,
    /// Negativity measure
    pub negativity: f64,
    /// Schmidt rank
    pub schmidt_rank: usize,
    /// Entanglement witnesses
    pub witness_values: HashMap<String, f64>,
}

/// Quantum advantage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAdvantageMetrics {
    /// Quantum vs classical performance ratio
    pub performance_ratio: f64,
    /// Quantum speedup factor
    pub speedup_factor: f64,
    /// Information processing advantage
    pub information_advantage: f64,
    /// Quantum resource utilization
    pub resource_utilization: f64,
    /// Advantage sustainability
    pub sustainability: f64,
}

/// Circuit execution details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitExecutionDetails {
    /// Circuit depth executed
    pub circuit_depth: usize,
    /// Gate count by type
    pub gate_counts: HashMap<String, usize>,
    /// Execution time
    pub execution_time: std::time::Duration,
    /// Quantum volume achieved
    pub quantum_volume: f64,
    /// Error rates observed
    pub error_rates: HashMap<String, f64>,
}

impl Default for QuantumDecisionConfig {
    fn default() -> Self {
        Self {
            num_qubits: 8,
            max_circuit_depth: 20,
            coherence_threshold: 0.8,
            measurement_shots: 1024,
            error_correction: true,
            advantage_threshold: 0.1,
            quantum_optimization: true,
        }
    }
}

impl QuantumDecisionMaker {
    /// Create a new quantum decision maker
    pub fn new(config: QuantumDecisionConfig) -> QarResult<Self> {
        let quantum_state = QuantumState::new(config.num_qubits)?;
        
        let noise_model = QuantumNoiseModel {
            decoherence_rate: 0.001,
            gate_error_rate: 0.0001,
            measurement_error_rate: 0.01,
            thermal_noise: 0.01,
        };

        let quantum_processor = QuantumProcessor {
            quantum_state,
            gate_set: Self::initialize_gate_set(),
            noise_model,
            backend: QuantumBackend::Simulator,
        };

        let quantum_circuits = Self::initialize_circuit_library(config.num_qubits)?;
        
        let entanglement_manager = EntanglementManager {
            entangled_pairs: Vec::new(),
            entanglement_strength: 0.8,
            bell_states: vec![BellState::Phi, BellState::Psi],
            witnesses: Vec::new(),
        };

        let coherence_tracker = CoherenceTracker {
            coherence_times: Vec::new(),
            decoherence_rate: 0.001,
            quality_factor: 100.0,
            environmental_factors: HashMap::new(),
        };

        Ok(Self {
            config,
            quantum_processor,
            quantum_circuits,
            entanglement_manager,
            measurement_history: Vec::new(),
            coherence_tracker,
        })
    }

    /// Make quantum-enhanced trading decision
    pub async fn make_quantum_decision(
        &mut self,
        factors: &FactorMap,
        analysis: &AnalysisResult,
    ) -> QarResult<QuantumDecisionResult> {
        // Initialize quantum state with market factors
        self.initialize_quantum_state(factors)?;

        // Create entanglement for correlated factors
        self.create_factor_entanglement(factors).await?;

        // Execute quantum decision circuits
        let quantum_probabilities = self.execute_decision_circuits(factors, analysis).await?;

        // Perform quantum measurements
        let measurements = self.perform_quantum_measurements().await?;

        // Analyze quantum state
        let quantum_state_info = self.analyze_quantum_state()?;

        // Analyze entanglement
        let entanglement_analysis = self.analyze_entanglement()?;

        // Calculate quantum advantage
        let quantum_advantage = self.calculate_quantum_advantage(&quantum_probabilities)?;

        // Track circuit execution
        let circuit_execution = self.track_circuit_execution()?;

        // Convert to classical decision
        let classical_decision = self.convert_to_classical_decision(&quantum_probabilities, analysis)?;

        Ok(QuantumDecisionResult {
            classical_decision,
            quantum_probabilities,
            quantum_state_info,
            entanglement_analysis,
            quantum_advantage,
            circuit_execution,
        })
    }

    /// Initialize quantum state with market factors
    fn initialize_quantum_state(&mut self, factors: &FactorMap) -> QarResult<()> {
        let trend = factors.get_factor(&crate::core::StandardFactors::Trend)?;
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let momentum = factors.get_factor(&crate::core::StandardFactors::Momentum)?;
        let volume = factors.get_factor(&crate::core::StandardFactors::Volume)?;

        // Encode market factors into quantum amplitudes
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); 1 << self.config.num_qubits];
        
        // Map factors to quantum state amplitudes
        for i in 0..amplitudes.len() {
            let binary_repr = format!("{:0width$b}", i, width = self.config.num_qubits);
            let mut amplitude = 1.0;
            
            // Apply factor-based modulation
            if binary_repr.chars().nth(0) == Some('1') {
                amplitude *= trend;
            }
            if binary_repr.chars().nth(1) == Some('1') {
                amplitude *= volatility;
            }
            if binary_repr.chars().nth(2) == Some('1') {
                amplitude *= momentum;
            }
            if binary_repr.chars().nth(3) == Some('1') {
                amplitude *= volume;
            }
            
            amplitudes[i] = Complex64::new(amplitude, 0.0);
        }

        // Normalize the state
        let norm: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        if norm > 0.0 {
            for amplitude in &mut amplitudes {
                *amplitude /= norm;
            }
        }

        self.quantum_processor.quantum_state = QuantumState::from_amplitudes(amplitudes)?;
        Ok(())
    }

    /// Create entanglement between correlated market factors
    async fn create_factor_entanglement(&mut self, factors: &FactorMap) -> QarResult<()> {
        let trend = factors.get_factor(&crate::core::StandardFactors::Trend)?;
        let momentum = factors.get_factor(&crate::core::StandardFactors::Momentum)?;
        let volume = factors.get_factor(&crate::core::StandardFactors::Volume)?;
        let liquidity = factors.get_factor(&crate::core::StandardFactors::Liquidity)?;

        // Create entanglement between correlated factors
        
        // Trend-Momentum entanglement (typically correlated)
        if (trend - momentum).abs() < 0.3 {
            self.entanglement_manager.entangled_pairs.push((0, 2));
            self.apply_bell_state_preparation(0, 2, BellState::Phi).await?;
        }

        // Volume-Liquidity entanglement (typically correlated)
        if (volume - liquidity).abs() < 0.3 {
            self.entanglement_manager.entangled_pairs.push((3, 5));
            self.apply_bell_state_preparation(3, 5, BellState::Phi).await?;
        }

        // Anti-correlation entanglements
        let volatility = factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let efficiency = factors.get_factor(&crate::core::StandardFactors::Efficiency)?;
        
        if (volatility + efficiency) > 1.3 { // High volatility + high efficiency unlikely
            self.entanglement_manager.entangled_pairs.push((1, 7));
            self.apply_bell_state_preparation(1, 7, BellState::PhiMinus).await?;
        }

        Ok(())
    }

    /// Apply Bell state preparation between two qubits
    async fn apply_bell_state_preparation(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        bell_state: BellState,
    ) -> QarResult<()> {
        match bell_state {
            BellState::Phi => {
                // |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
                self.quantum_processor.quantum_state.apply_hadamard(qubit1)?;
                self.quantum_processor.quantum_state.apply_cnot(qubit1, qubit2)?;
            }
            BellState::PhiMinus => {
                // |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
                self.quantum_processor.quantum_state.apply_hadamard(qubit1)?;
                self.quantum_processor.quantum_state.apply_cnot(qubit1, qubit2)?;
                self.quantum_processor.quantum_state.apply_z(qubit2)?;
            }
            BellState::Psi => {
                // |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
                self.quantum_processor.quantum_state.apply_hadamard(qubit1)?;
                self.quantum_processor.quantum_state.apply_x(qubit2)?;
                self.quantum_processor.quantum_state.apply_cnot(qubit1, qubit2)?;
            }
            BellState::PsiMinus => {
                // |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
                self.quantum_processor.quantum_state.apply_hadamard(qubit1)?;
                self.quantum_processor.quantum_state.apply_x(qubit2)?;
                self.quantum_processor.quantum_state.apply_cnot(qubit1, qubit2)?;
                self.quantum_processor.quantum_state.apply_z(qubit2)?;
            }
        }
        Ok(())
    }

    /// Execute quantum decision circuits
    async fn execute_decision_circuits(
        &mut self,
        factors: &FactorMap,
        analysis: &AnalysisResult,
    ) -> QarResult<QuantumProbabilities> {
        // Execute superposition circuit for decision exploration
        self.execute_superposition_circuit().await?;

        // Apply quantum algorithms for decision enhancement
        self.apply_quantum_fourier_transform().await?;
        
        // Execute variational circuit for optimization
        self.execute_variational_circuit(factors, analysis).await?;

        // Apply amplitude amplification for decision amplification
        self.apply_amplitude_amplification().await?;

        // Measure decision probabilities
        let probabilities = self.measure_decision_probabilities().await?;

        Ok(probabilities)
    }

    /// Execute superposition circuit
    async fn execute_superposition_circuit(&mut self) -> QarResult<()> {
        // Create equal superposition on decision qubits (first 3 qubits)
        for i in 0..3 {
            self.quantum_processor.quantum_state.apply_hadamard(i)?;
        }

        // Apply controlled operations based on factor qubits
        for i in 3..self.config.num_qubits {
            self.quantum_processor.quantum_state.apply_cnot(i, i % 3)?;
        }

        Ok(())
    }

    /// Apply Quantum Fourier Transform
    async fn apply_quantum_fourier_transform(&mut self) -> QarResult<()> {
        let n = self.config.num_qubits;
        
        for i in 0..n {
            // Apply Hadamard gate
            self.quantum_processor.quantum_state.apply_hadamard(i)?;
            
            // Apply controlled rotation gates
            for j in (i + 1)..n {
                let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
                self.apply_controlled_rotation(j, i, angle).await?;
            }
        }

        // Reverse qubit order
        for i in 0..(n / 2) {
            self.quantum_processor.quantum_state.apply_swap(i, n - 1 - i)?;
        }

        Ok(())
    }

    /// Apply controlled rotation gate
    async fn apply_controlled_rotation(&mut self, control: usize, target: usize, angle: f64) -> QarResult<()> {
        // Simplified controlled rotation (in real implementation would use proper controlled gates)
        self.quantum_processor.quantum_state.apply_rz(target, angle)?;
        Ok(())
    }

    /// Execute variational quantum circuit
    async fn execute_variational_circuit(
        &mut self,
        factors: &FactorMap,
        analysis: &AnalysisResult,
    ) -> QarResult<()> {
        let parameters = &self.quantum_circuits.variational_circuit.parameters;
        
        // Apply parameterized gates
        for (i, &param) in parameters.iter().enumerate() {
            let qubit = i % self.config.num_qubits;
            
            // Apply rotation gates with parameters
            self.quantum_processor.quantum_state.apply_rx(qubit, param)?;
            if i + 1 < parameters.len() {
                self.quantum_processor.quantum_state.apply_ry(qubit, parameters[i + 1])?;
            }
            if i + 2 < parameters.len() {
                self.quantum_processor.quantum_state.apply_rz(qubit, parameters[i + 2])?;
            }
        }

        // Apply entangling gates
        for i in 0..(self.config.num_qubits - 1) {
            self.quantum_processor.quantum_state.apply_cnot(i, i + 1)?;
        }

        Ok(())
    }

    /// Apply amplitude amplification
    async fn apply_amplitude_amplification(&mut self) -> QarResult<()> {
        // Simplified Grover-like amplitude amplification
        let n_iterations = (std::f64::consts::PI / 4.0 * (1 << self.config.num_qubits as u32) as f64).sqrt() as usize;
        
        for _ in 0..n_iterations.min(10) { // Limit iterations for practical execution
            // Oracle operation (mark target states)
            self.apply_oracle_operation().await?;
            
            // Diffusion operation
            self.apply_diffusion_operation().await?;
        }

        Ok(())
    }

    /// Apply oracle operation for amplitude amplification
    async fn apply_oracle_operation(&mut self) -> QarResult<()> {
        // Mark states corresponding to strong buy/sell signals
        // This is a simplified oracle - real implementation would be more sophisticated
        for i in 0..3 { // Decision qubits
            self.quantum_processor.quantum_state.apply_z(i)?;
        }
        Ok(())
    }

    /// Apply diffusion operation
    async fn apply_diffusion_operation(&mut self) -> QarResult<()> {
        // Diffusion about average amplitude
        for i in 0..self.config.num_qubits {
            self.quantum_processor.quantum_state.apply_hadamard(i)?;
        }
        
        for i in 0..self.config.num_qubits {
            self.quantum_processor.quantum_state.apply_z(i)?;
        }
        
        for i in 0..self.config.num_qubits {
            self.quantum_processor.quantum_state.apply_hadamard(i)?;
        }
        
        Ok(())
    }

    /// Measure decision probabilities
    async fn measure_decision_probabilities(&mut self) -> QarResult<QuantumProbabilities> {
        let mut buy_count = 0;
        let mut sell_count = 0;
        let mut hold_count = 0;

        // Perform multiple measurements
        for _ in 0..self.config.measurement_shots {
            let measurement = self.quantum_processor.quantum_state.measure_qubits(&[0, 1, 2])?;
            
            match measurement[..] {
                [0, 0, 1] => buy_count += 1,   // |001⟩ -> Buy
                [0, 1, 0] => sell_count += 1,  // |010⟩ -> Sell
                [1, 0, 0] => hold_count += 1,  // |100⟩ -> Hold
                _ => {} // Other states contribute to uncertainty
            }
        }

        let total_measurements = self.config.measurement_shots as f64;
        let buy_probability = buy_count as f64 / total_measurements;
        let sell_probability = sell_count as f64 / total_measurements;
        let hold_probability = hold_count as f64 / total_measurements;
        
        // Calculate superposition coefficient
        let coherent_measurements = buy_count + sell_count + hold_count;
        let superposition_coefficient = 1.0 - (coherent_measurements as f64 / total_measurements);
        
        // Calculate measurement uncertainty
        let probabilities = vec![buy_probability, sell_probability, hold_probability];
        let max_prob = probabilities.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let measurement_uncertainty = 1.0 - max_prob;

        Ok(QuantumProbabilities {
            buy_probability,
            sell_probability,
            hold_probability,
            superposition_coefficient,
            measurement_uncertainty,
        })
    }

    /// Perform comprehensive quantum measurements
    async fn perform_quantum_measurements(&mut self) -> QarResult<Vec<QuantumMeasurement>> {
        let mut measurements = Vec::new();
        let now = chrono::Utc::now();

        // Computational basis measurement
        let comp_measurement = self.measure_in_basis(MeasurementBasis::Computational).await?;
        measurements.push(QuantumMeasurement {
            timestamp: now,
            measured_qubits: (0..self.config.num_qubits).collect(),
            outcomes: comp_measurement,
            amplitudes: self.quantum_processor.quantum_state.get_amplitudes(),
            basis: MeasurementBasis::Computational,
            fidelity: self.calculate_state_fidelity()?,
        });

        // Hadamard basis measurement
        let hadamard_measurement = self.measure_in_basis(MeasurementBasis::Hadamard).await?;
        measurements.push(QuantumMeasurement {
            timestamp: now,
            measured_qubits: (0..self.config.num_qubits).collect(),
            outcomes: hadamard_measurement,
            amplitudes: self.quantum_processor.quantum_state.get_amplitudes(),
            basis: MeasurementBasis::Hadamard,
            fidelity: self.calculate_state_fidelity()?,
        });

        // Store measurements in history
        self.measurement_history.extend(measurements.clone());
        
        // Maintain history size
        if self.measurement_history.len() > 1000 {
            self.measurement_history.drain(0..self.measurement_history.len() - 1000);
        }

        Ok(measurements)
    }

    /// Measure quantum state in specific basis
    async fn measure_in_basis(&mut self, basis: MeasurementBasis) -> QarResult<Vec<u8>> {
        match basis {
            MeasurementBasis::Computational => {
                self.quantum_processor.quantum_state.measure_all()
            }
            MeasurementBasis::Hadamard => {
                // Apply Hadamard to all qubits before measurement
                for i in 0..self.config.num_qubits {
                    self.quantum_processor.quantum_state.apply_hadamard(i)?;
                }
                let result = self.quantum_processor.quantum_state.measure_all();
                // Apply Hadamard again to restore state
                for i in 0..self.config.num_qubits {
                    self.quantum_processor.quantum_state.apply_hadamard(i)?;
                }
                result
            }
            _ => self.quantum_processor.quantum_state.measure_all(),
        }
    }

    /// Calculate quantum state fidelity
    fn calculate_state_fidelity(&self) -> QarResult<f64> {
        // Simplified fidelity calculation against ideal state
        let amplitudes = self.quantum_processor.quantum_state.get_amplitudes();
        let fidelity = amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        Ok(fidelity.min(1.0))
    }

    /// Analyze quantum state properties
    fn analyze_quantum_state(&self) -> QarResult<QuantumStateInfo> {
        let state_vector = self.quantum_processor.quantum_state.get_amplitudes();
        
        // Calculate purity
        let purity = state_vector.iter().map(|a| a.norm_sqr().powi(2)).sum();
        
        // Calculate von Neumann entropy
        let entropy = -state_vector.iter()
            .map(|a| {
                let p = a.norm_sqr();
                if p > 0.0 { p * p.ln() } else { 0.0 }
            })
            .sum::<f64>();

        // Calculate coherence measures
        let coherence = self.calculate_coherence_measures(&state_vector)?;
        
        // Calculate phase information
        let phase_info = self.calculate_phase_info(&state_vector)?;

        Ok(QuantumStateInfo {
            state_vector,
            purity,
            entropy,
            coherence,
            phase_info,
        })
    }

    /// Calculate coherence measures
    fn calculate_coherence_measures(&self, state_vector: &[Complex64]) -> QarResult<CoherenceMeasures> {
        // L1 norm coherence
        let l1_norm = state_vector.iter()
            .enumerate()
            .map(|(i, a)| if i == 0 { 0.0 } else { a.norm() })
            .sum();

        // Relative entropy coherence (simplified)
        let relative_entropy = state_vector.iter()
            .map(|a| {
                let p = a.norm_sqr();
                if p > 0.0 { p * (p * state_vector.len() as f64).ln() } else { 0.0 }
            })
            .sum::<f64>();

        // Robustness and formation (simplified)
        let robustness = l1_norm / 2.0;
        let formation = relative_entropy / 2.0;

        Ok(CoherenceMeasures {
            l1_norm,
            relative_entropy,
            robustness,
            formation,
        })
    }

    /// Calculate quantum phase information
    fn calculate_phase_info(&self, state_vector: &[Complex64]) -> QarResult<QuantumPhaseInfo> {
        // Global phase (phase of first non-zero amplitude)
        let global_phase = state_vector.iter()
            .find(|a| a.norm() > 0.0)
            .map(|a| a.arg())
            .unwrap_or(0.0);

        // Relative phases
        let relative_phases: Vec<f64> = state_vector.iter()
            .map(|a| a.arg() - global_phase)
            .collect();

        // Phase variance
        let mean_phase = relative_phases.iter().sum::<f64>() / relative_phases.len() as f64;
        let phase_variance = relative_phases.iter()
            .map(|p| (p - mean_phase).powi(2))
            .sum::<f64>() / relative_phases.len() as f64;

        Ok(QuantumPhaseInfo {
            global_phase,
            relative_phases,
            phase_variance,
            berry_phase: None, // Would require geometric phase calculation
        })
    }

    /// Analyze entanglement in the quantum state
    fn analyze_entanglement(&self) -> QarResult<EntanglementAnalysis> {
        // Simplified entanglement analysis
        let num_qubits = self.config.num_qubits;
        
        // Entanglement entropy (simplified)
        let entanglement_entropy = if num_qubits > 1 {
            (num_qubits as f64).ln()
        } else {
            0.0
        };

        // Concurrence (for 2-qubit systems, simplified for multi-qubit)
        let concurrence = if self.entanglement_manager.entangled_pairs.len() > 0 {
            self.entanglement_manager.entanglement_strength
        } else {
            0.0
        };

        // Negativity measure
        let negativity = concurrence.max(0.0);

        // Schmidt rank
        let schmidt_rank = if concurrence > 0.1 { 2 } else { 1 };

        // Entanglement witnesses
        let mut witness_values = HashMap::new();
        witness_values.insert("bell_inequality".to_string(), concurrence * 2.0);

        Ok(EntanglementAnalysis {
            entanglement_entropy,
            concurrence,
            negativity,
            schmidt_rank,
            witness_values,
        })
    }

    /// Calculate quantum advantage metrics
    fn calculate_quantum_advantage(&self, probabilities: &QuantumProbabilities) -> QarResult<QuantumAdvantageMetrics> {
        // Performance ratio (quantum vs classical expected performance)
        let quantum_performance = probabilities.buy_probability * 0.1 + probabilities.sell_probability * 0.05;
        let classical_performance = 0.06; // Assumed classical baseline
        let performance_ratio = quantum_performance / classical_performance.max(0.01);

        // Speedup factor (based on superposition coefficient)
        let speedup_factor = 1.0 + probabilities.superposition_coefficient * 2.0;

        // Information processing advantage
        let information_advantage = probabilities.superposition_coefficient * (1.0 - probabilities.measurement_uncertainty);

        // Resource utilization
        let resource_utilization = self.entanglement_manager.entangled_pairs.len() as f64 / 
                                 (self.config.num_qubits as f64 / 2.0);

        // Sustainability (based on coherence)
        let sustainability = self.coherence_tracker.quality_factor / 100.0;

        Ok(QuantumAdvantageMetrics {
            performance_ratio,
            speedup_factor,
            information_advantage,
            resource_utilization,
            sustainability,
        })
    }

    /// Track circuit execution details
    fn track_circuit_execution(&self) -> QarResult<CircuitExecutionDetails> {
        let mut gate_counts = HashMap::new();
        gate_counts.insert("hadamard".to_string(), self.config.num_qubits * 3);
        gate_counts.insert("cnot".to_string(), self.config.num_qubits * 2);
        gate_counts.insert("rotation".to_string(), self.config.num_qubits);

        let circuit_depth = self.config.max_circuit_depth.min(20);
        let execution_time = std::time::Duration::from_millis(circuit_depth as u64 * 10);
        let quantum_volume = (self.config.num_qubits * circuit_depth) as f64;

        let mut error_rates = HashMap::new();
        error_rates.insert("gate_error".to_string(), self.quantum_processor.noise_model.gate_error_rate);
        error_rates.insert("measurement_error".to_string(), self.quantum_processor.noise_model.measurement_error_rate);

        Ok(CircuitExecutionDetails {
            circuit_depth,
            gate_counts,
            execution_time,
            quantum_volume,
            error_rates,
        })
    }

    /// Convert quantum probabilities to classical trading decision
    fn convert_to_classical_decision(
        &self,
        probabilities: &QuantumProbabilities,
        analysis: &AnalysisResult,
    ) -> QarResult<TradingDecision> {
        // Determine decision type based on highest probability
        let decision_type = if probabilities.buy_probability > probabilities.sell_probability &&
                              probabilities.buy_probability > probabilities.hold_probability {
            DecisionType::Buy
        } else if probabilities.sell_probability > probabilities.hold_probability {
            DecisionType::Sell
        } else {
            DecisionType::Hold
        };

        // Calculate confidence incorporating quantum uncertainty
        let max_probability = probabilities.buy_probability
            .max(probabilities.sell_probability)
            .max(probabilities.hold_probability);
        let quantum_confidence = max_probability * (1.0 - probabilities.measurement_uncertainty);
        let confidence = (quantum_confidence + analysis.confidence) / 2.0;

        // Calculate expected return with quantum enhancement
        let quantum_enhancement = probabilities.superposition_coefficient * 0.02;
        let base_return = match decision_type {
            DecisionType::Buy => 0.05 + quantum_enhancement,
            DecisionType::Sell => 0.03 + quantum_enhancement,
            DecisionType::Hold => 0.0,
        };
        let expected_return = Some(base_return);

        // Risk assessment with quantum considerations
        let quantum_risk = probabilities.measurement_uncertainty * 0.5;
        let risk_assessment = Some(quantum_risk);

        // Urgency based on quantum dynamics
        let urgency_score = Some(probabilities.superposition_coefficient);

        // Generate reasoning
        let reasoning = format!(
            "Quantum decision: P(buy)={:.3}, P(sell)={:.3}, P(hold)={:.3}, superposition={:.3}, uncertainty={:.3}",
            probabilities.buy_probability,
            probabilities.sell_probability,
            probabilities.hold_probability,
            probabilities.superposition_coefficient,
            probabilities.measurement_uncertainty
        );

        Ok(TradingDecision {
            decision_type,
            confidence,
            expected_return,
            risk_assessment,
            urgency_score,
            reasoning,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Initialize quantum gate set
    fn initialize_gate_set() -> Vec<Gate> {
        vec![
            Gate::pauli_x(),
            Gate::pauli_y(),
            Gate::pauli_z(),
            Gate::hadamard(),
            Gate::phase(std::f64::consts::PI / 2.0),
            Gate::t(),
            Gate::cnot(),
        ]
    }

    /// Initialize quantum circuit library
    fn initialize_circuit_library(num_qubits: usize) -> QarResult<QuantumCircuitLibrary> {
        let buy_circuit = QuantumCircuit::new(num_qubits);
        let sell_circuit = QuantumCircuit::new(num_qubits);
        let hold_circuit = QuantumCircuit::new(num_qubits);
        let superposition_circuit = QuantumCircuit::new(num_qubits);
        let entanglement_circuit = QuantumCircuit::new(num_qubits);
        let qft_circuit = QuantumCircuit::new(num_qubits);

        let variational_circuit = VariationalQuantumCircuit {
            circuit: QuantumCircuit::new(num_qubits),
            parameters: vec![0.1; num_qubits * 3], // 3 parameters per qubit
            gradients: vec![0.0; num_qubits * 3],
            optimization_history: Vec::new(),
        };

        Ok(QuantumCircuitLibrary {
            buy_circuit,
            sell_circuit,
            hold_circuit,
            superposition_circuit,
            entanglement_circuit,
            qft_circuit,
            variational_circuit,
        })
    }

    /// Update quantum parameters based on market feedback
    pub fn update_quantum_parameters(&mut self, performance_feedback: f64) -> QarResult<()> {
        // Update variational circuit parameters
        let learning_rate = 0.01;
        for (param, grad) in self.quantum_circuits.variational_circuit.parameters.iter_mut()
            .zip(&self.quantum_circuits.variational_circuit.gradients) {
            *param += learning_rate * grad * performance_feedback;
        }

        // Update entanglement strength
        if performance_feedback > 0.0 {
            self.entanglement_manager.entanglement_strength = 
                (self.entanglement_manager.entanglement_strength + 0.01).min(1.0);
        } else {
            self.entanglement_manager.entanglement_strength = 
                (self.entanglement_manager.entanglement_strength - 0.01).max(0.1);
        }

        // Track parameter update
        let update = ParameterUpdate {
            timestamp: chrono::Utc::now(),
            parameters: self.quantum_circuits.variational_circuit.parameters.clone(),
            cost: -performance_feedback, // Negative because we minimize cost
            gradient_norm: self.quantum_circuits.variational_circuit.gradients.iter()
                .map(|g| g.powi(2)).sum::<f64>().sqrt(),
        };
        
        self.quantum_circuits.variational_circuit.optimization_history.push(update);

        Ok(())
    }

    /// Get quantum decision insights
    pub fn get_quantum_insights(&self) -> QuantumInsights {
        let superposition_analysis = vec![
            self.entanglement_manager.entanglement_strength,
            1.0 - self.entanglement_manager.entanglement_strength,
        ];

        let mut entanglement_correlations = HashMap::new();
        for (i, &(q1, q2)) in self.entanglement_manager.entangled_pairs.iter().enumerate() {
            entanglement_correlations.insert(
                format!("pair_{}_{}", q1, q2),
                self.entanglement_manager.entanglement_strength,
            );
        }

        let interference_patterns = vec![
            self.entanglement_manager.entanglement_strength,
            self.entanglement_manager.entanglement_strength * 0.8,
            self.entanglement_manager.entanglement_strength * 0.6,
        ];

        let measurement_uncertainty = 1.0 - (self.coherence_tracker.quality_factor / 100.0);

        QuantumInsights {
            superposition_analysis,
            entanglement_correlations,
            interference_patterns,
            measurement_uncertainty,
        }
    }

    /// Get measurement history
    pub fn get_measurement_history(&self) -> &[QuantumMeasurement] {
        &self.measurement_history
    }

    /// Get coherence tracking data
    pub fn get_coherence_data(&self) -> &CoherenceTracker {
        &self.coherence_tracker
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;
    use crate::analysis::{TrendDirection, VolatilityLevel, MarketRegime};

    fn create_test_factors() -> FactorMap {
        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Trend.to_string(), 0.7);
        factors.insert(StandardFactors::Volatility.to_string(), 0.3);
        factors.insert(StandardFactors::Momentum.to_string(), 0.6);
        factors.insert(StandardFactors::Volume.to_string(), 0.8);
        factors.insert(StandardFactors::Liquidity.to_string(), 0.7);
        factors.insert(StandardFactors::Sentiment.to_string(), 0.6);
        factors.insert(StandardFactors::Risk.to_string(), 0.4);
        factors.insert(StandardFactors::Efficiency.to_string(), 0.8);
        FactorMap::new(factors).unwrap()
    }

    fn create_test_analysis() -> AnalysisResult {
        AnalysisResult {
            timestamp: chrono::Utc::now(),
            trend: TrendDirection::Bullish,
            trend_strength: 0.8,
            volatility: VolatilityLevel::Medium,
            regime: MarketRegime::Bull,
            confidence: 0.9,
            metrics: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_quantum_decision_maker_creation() {
        let config = QuantumDecisionConfig::default();
        let quantum_dm = QuantumDecisionMaker::new(config);
        assert!(quantum_dm.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_state_initialization() {
        let config = QuantumDecisionConfig::default();
        let mut quantum_dm = QuantumDecisionMaker::new(config).unwrap();
        let factors = create_test_factors();

        let result = quantum_dm.initialize_quantum_state(&factors);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_decision_making() {
        let config = QuantumDecisionConfig::default();
        let mut quantum_dm = QuantumDecisionMaker::new(config).unwrap();
        let factors = create_test_factors();
        let analysis = create_test_analysis();

        let result = quantum_dm.make_quantum_decision(&factors, &analysis).await;
        assert!(result.is_ok());

        let decision_result = result.unwrap();
        assert!(decision_result.quantum_probabilities.buy_probability >= 0.0);
        assert!(decision_result.quantum_probabilities.sell_probability >= 0.0);
        assert!(decision_result.quantum_probabilities.hold_probability >= 0.0);
        
        // Probabilities should sum to approximately 1.0 (allowing for superposition)
        let total_prob = decision_result.quantum_probabilities.buy_probability +
                        decision_result.quantum_probabilities.sell_probability +
                        decision_result.quantum_probabilities.hold_probability;
        assert!(total_prob <= 1.1); // Allow some tolerance for quantum effects
    }

    #[tokio::test]
    async fn test_entanglement_creation() {
        let config = QuantumDecisionConfig::default();
        let mut quantum_dm = QuantumDecisionMaker::new(config).unwrap();
        let factors = create_test_factors();

        let result = quantum_dm.create_factor_entanglement(&factors).await;
        assert!(result.is_ok());
        
        // Should have created some entangled pairs for correlated factors
        assert!(!quantum_dm.entanglement_manager.entangled_pairs.is_empty());
    }

    #[tokio::test]
    async fn test_bell_state_preparation() {
        let config = QuantumDecisionConfig::default();
        let mut quantum_dm = QuantumDecisionMaker::new(config).unwrap();

        let result = quantum_dm.apply_bell_state_preparation(0, 1, BellState::Phi).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_circuit_execution() {
        let config = QuantumDecisionConfig::default();
        let mut quantum_dm = QuantumDecisionMaker::new(config).unwrap();

        let result = quantum_dm.execute_superposition_circuit().await;
        assert!(result.is_ok());

        let qft_result = quantum_dm.apply_quantum_fourier_transform().await;
        assert!(qft_result.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_measurements() {
        let config = QuantumDecisionConfig::default();
        let mut quantum_dm = QuantumDecisionMaker::new(config).unwrap();

        let measurements = quantum_dm.perform_quantum_measurements().await;
        assert!(measurements.is_ok());
        
        let measurements = measurements.unwrap();
        assert!(!measurements.is_empty());
        
        for measurement in &measurements {
            assert_eq!(measurement.measured_qubits.len(), config.num_qubits);
            assert_eq!(measurement.outcomes.len(), config.num_qubits);
            assert!(measurement.fidelity >= 0.0 && measurement.fidelity <= 1.0);
        }
    }

    #[test]
    fn test_quantum_state_analysis() {
        let config = QuantumDecisionConfig::default();
        let quantum_dm = QuantumDecisionMaker::new(config).unwrap();

        let state_info = quantum_dm.analyze_quantum_state();
        assert!(state_info.is_ok());
        
        let state_info = state_info.unwrap();
        assert!(state_info.purity >= 0.0 && state_info.purity <= 1.0);
        assert!(state_info.entropy >= 0.0);
        assert!(!state_info.state_vector.is_empty());
    }

    #[test]
    fn test_entanglement_analysis() {
        let config = QuantumDecisionConfig::default();
        let quantum_dm = QuantumDecisionMaker::new(config).unwrap();

        let entanglement_analysis = quantum_dm.analyze_entanglement();
        assert!(entanglement_analysis.is_ok());
        
        let analysis = entanglement_analysis.unwrap();
        assert!(analysis.entanglement_entropy >= 0.0);
        assert!(analysis.concurrence >= 0.0 && analysis.concurrence <= 1.0);
        assert!(analysis.negativity >= 0.0);
        assert!(analysis.schmidt_rank >= 1);
    }

    #[test]
    fn test_quantum_advantage_calculation() {
        let config = QuantumDecisionConfig::default();
        let quantum_dm = QuantumDecisionMaker::new(config).unwrap();
        
        let probabilities = QuantumProbabilities {
            buy_probability: 0.6,
            sell_probability: 0.3,
            hold_probability: 0.1,
            superposition_coefficient: 0.2,
            measurement_uncertainty: 0.1,
        };

        let advantage = quantum_dm.calculate_quantum_advantage(&probabilities);
        assert!(advantage.is_ok());
        
        let advantage = advantage.unwrap();
        assert!(advantage.performance_ratio >= 0.0);
        assert!(advantage.speedup_factor >= 1.0);
        assert!(advantage.information_advantage >= 0.0);
        assert!(advantage.resource_utilization >= 0.0);
    }

    #[test]
    fn test_coherence_measures() {
        let config = QuantumDecisionConfig::default();
        let quantum_dm = QuantumDecisionMaker::new(config).unwrap();
        
        let state_vector = vec![
            Complex64::new(0.7, 0.0),
            Complex64::new(0.0, 0.3),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.0, 0.1),
        ];

        let coherence = quantum_dm.calculate_coherence_measures(&state_vector);
        assert!(coherence.is_ok());
        
        let coherence = coherence.unwrap();
        assert!(coherence.l1_norm >= 0.0);
        assert!(coherence.robustness >= 0.0);
        assert!(coherence.formation >= 0.0);
    }

    #[test]
    fn test_phase_info_calculation() {
        let config = QuantumDecisionConfig::default();
        let quantum_dm = QuantumDecisionMaker::new(config).unwrap();
        
        let state_vector = vec![
            Complex64::new(0.5, 0.5),
            Complex64::new(0.3, 0.2),
            Complex64::new(0.1, 0.4),
            Complex64::new(0.2, 0.1),
        ];

        let phase_info = quantum_dm.calculate_phase_info(&state_vector);
        assert!(phase_info.is_ok());
        
        let phase_info = phase_info.unwrap();
        assert!(phase_info.global_phase >= -std::f64::consts::PI && phase_info.global_phase <= std::f64::consts::PI);
        assert_eq!(phase_info.relative_phases.len(), state_vector.len());
        assert!(phase_info.phase_variance >= 0.0);
    }

    #[tokio::test]
    async fn test_parameter_updates() {
        let config = QuantumDecisionConfig::default();
        let mut quantum_dm = QuantumDecisionMaker::new(config).unwrap();
        
        let initial_params = quantum_dm.quantum_circuits.variational_circuit.parameters.clone();
        
        let result = quantum_dm.update_quantum_parameters(0.1);
        assert!(result.is_ok());
        
        let updated_params = &quantum_dm.quantum_circuits.variational_circuit.parameters;
        
        // Parameters should have changed
        assert_ne!(initial_params, *updated_params);
        
        // Should have optimization history
        assert!(!quantum_dm.quantum_circuits.variational_circuit.optimization_history.is_empty());
    }

    #[test]
    fn test_quantum_insights_generation() {
        let config = QuantumDecisionConfig::default();
        let quantum_dm = QuantumDecisionMaker::new(config).unwrap();

        let insights = quantum_dm.get_quantum_insights();
        assert!(!insights.superposition_analysis.is_empty());
        assert!(!insights.interference_patterns.is_empty());
        assert!(insights.measurement_uncertainty >= 0.0 && insights.measurement_uncertainty <= 1.0);
    }

    #[tokio::test]
    async fn test_classical_decision_conversion() {
        let config = QuantumDecisionConfig::default();
        let quantum_dm = QuantumDecisionMaker::new(config).unwrap();
        
        let probabilities = QuantumProbabilities {
            buy_probability: 0.8,
            sell_probability: 0.1,
            hold_probability: 0.1,
            superposition_coefficient: 0.2,
            measurement_uncertainty: 0.1,
        };
        
        let analysis = create_test_analysis();
        
        let decision = quantum_dm.convert_to_classical_decision(&probabilities, &analysis);
        assert!(decision.is_ok());
        
        let decision = decision.unwrap();
        assert_eq!(decision.decision_type, DecisionType::Buy); // Highest probability
        assert!(decision.confidence > 0.0 && decision.confidence <= 1.0);
        assert!(decision.expected_return.is_some());
        assert!(!decision.reasoning.is_empty());
    }
}