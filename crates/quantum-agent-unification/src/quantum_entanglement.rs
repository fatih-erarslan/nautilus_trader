//! Quantum entanglement coordination and correlation management

use crate::quantum_state::{QuantumState, QuantumBit};
use crate::{QuantumResult, QuantumError};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Quantum entanglement network for coordinating multiple agents
#[derive(Debug, Clone)]
pub struct QuantumEntanglement {
    /// Entanglement correlation matrix
    pub correlation_matrix: DMatrix<Complex64>,
    /// Bell state measurements
    pub bell_states: Vec<BellState>,
    /// Entanglement strength between agent pairs
    pub entanglement_strengths: HashMap<(usize, usize), f64>,
    /// Quantum communication channels
    pub quantum_channels: Vec<QuantumChannel>,
    /// EPR pair registry
    pub epr_pairs: Vec<EPRPair>,
    /// Entanglement entropy measurements
    pub entropy_history: Vec<f64>,
}

/// Bell state representation for maximum entanglement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BellState {
    /// Bell state type
    pub state_type: BellStateType,
    /// Qubit indices involved
    pub qubit_indices: (usize, usize),
    /// Correlation coefficient
    pub correlation: f64,
    /// Measurement fidelity
    pub fidelity: f64,
    /// State preparation time
    pub preparation_time: std::time::Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BellStateType {
    /// |Φ+⟩ = (|00⟩ + |11⟩)/√2
    PhiPlus,
    /// |Φ-⟩ = (|00⟩ - |11⟩)/√2
    PhiMinus,
    /// |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    PsiPlus,
    /// |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    PsiMinus,
}

/// Quantum communication channel for information transfer
#[derive(Debug, Clone)]
pub struct QuantumChannel {
    /// Source agent index
    pub source: usize,
    /// Target agent index
    pub target: usize,
    /// Channel capacity (qubits per second)
    pub capacity: f64,
    /// Channel fidelity
    pub fidelity: f64,
    /// Noise level
    pub noise_level: f64,
    /// Channel coherence time
    pub coherence_time: f64,
    /// Message queue
    pub message_queue: Vec<QuantumMessage>,
}

/// Quantum message for inter-agent communication
#[derive(Debug, Clone)]
pub struct QuantumMessage {
    /// Message content as quantum state
    pub quantum_payload: QuantumState,
    /// Classical metadata
    pub metadata: HashMap<String, String>,
    /// Message priority
    pub priority: MessagePriority,
    /// Transmission timestamp
    pub timestamp: std::time::Instant,
    /// Error correction codes
    pub error_correction: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Emergency,
}

/// Einstein-Podolsky-Rosen (EPR) pair for quantum correlation
#[derive(Debug, Clone)]
pub struct EPRPair {
    /// First qubit of the pair
    pub qubit_a: QuantumBit,
    /// Second qubit of the pair
    pub qubit_b: QuantumBit,
    /// Pair creation time
    pub creation_time: std::time::Instant,
    /// Correlation strength
    pub correlation_strength: f64,
    /// Decoherence rate
    pub decoherence_rate: f64,
    /// Measurement history
    pub measurement_history: Vec<(bool, bool, std::time::Instant)>,
}

impl QuantumEntanglement {
    /// Create new quantum entanglement network
    pub fn new(num_agents: usize) -> Self {
        Self {
            correlation_matrix: DMatrix::zeros(num_agents, num_agents),
            bell_states: Vec::new(),
            entanglement_strengths: HashMap::new(),
            quantum_channels: Vec::new(),
            epr_pairs: Vec::new(),
            entropy_history: Vec::new(),
        }
    }
    
    /// Create Bell state entanglement between two agents
    pub fn create_bell_state(&mut self, agent1: usize, agent2: usize, state_type: BellStateType) -> QuantumResult<BellState> {
        let start_time = std::time::Instant::now();
        
        // Create EPR pair
        let mut epr_pair = EPRPair::new();
        
        // Prepare Bell state based on type
        match state_type {
            BellStateType::PhiPlus => {
                epr_pair.create_phi_plus()?;
            },
            BellStateType::PhiMinus => {
                epr_pair.create_phi_minus()?;
            },
            BellStateType::PsiPlus => {
                epr_pair.create_psi_plus()?;
            },
            BellStateType::PsiMinus => {
                epr_pair.create_psi_minus()?;
            },
        }
        
        let preparation_time = start_time.elapsed();
        
        // Calculate correlation
        let correlation = self.measure_correlation(&epr_pair)?;
        
        let bell_state = BellState {
            state_type,
            qubit_indices: (agent1, agent2),
            correlation,
            fidelity: 0.99, // High fidelity for ideal preparation
            preparation_time,
        };
        
        // Update correlation matrix
        self.correlation_matrix[(agent1, agent2)] = Complex64::new(correlation, 0.0);
        self.correlation_matrix[(agent2, agent1)] = Complex64::new(correlation, 0.0);
        
        // Store entanglement strength
        self.entanglement_strengths.insert((agent1, agent2), correlation.abs());
        self.entanglement_strengths.insert((agent2, agent1), correlation.abs());
        
        // Add to EPR pair registry
        self.epr_pairs.push(epr_pair);
        self.bell_states.push(bell_state.clone());
        
        Ok(bell_state)
    }
    
    /// Measure quantum correlation using Bell test
    pub fn measure_correlation(&self, epr_pair: &EPRPair) -> QuantumResult<f64> {
        // Simulate Bell inequality test
        let mut correlation_sum = 0.0;
        let num_measurements = 1000;
        
        for _ in 0..num_measurements {
            // Random measurement angles
            let theta_a = rand::random::<f64>() * 2.0 * PI;
            let theta_b = rand::random::<f64>() * 2.0 * PI;
            
            // Simulate measurements at different angles
            let prob_correlation = self.calculate_quantum_correlation(theta_a, theta_b);
            correlation_sum += prob_correlation;
        }
        
        let average_correlation = correlation_sum / num_measurements as f64;
        
        // Check Bell inequality violation (quantum mechanics predicts violation)
        let bell_parameter = 2.0 * (2.0_f64).sqrt() * average_correlation;
        
        if bell_parameter > 2.0 {
            // Bell inequality violated - quantum entanglement confirmed
            Ok(average_correlation)
        } else {
            // Classical correlation only
            Ok(average_correlation * 0.5)
        }
    }
    
    /// Calculate quantum correlation for given measurement angles
    fn calculate_quantum_correlation(&self, theta_a: f64, theta_b: f64) -> f64 {
        // Quantum mechanics prediction for spin correlation
        -(theta_a - theta_b).cos()
    }
    
    /// Create quantum communication channel between agents
    pub fn create_quantum_channel(&mut self, source: usize, target: usize, capacity: f64) -> QuantumResult<usize> {
        let channel = QuantumChannel {
            source,
            target,
            capacity,
            fidelity: 0.95, // Realistic channel fidelity
            noise_level: 0.05,
            coherence_time: 1.0, // 1 second coherence
            message_queue: Vec::new(),
        };
        
        self.quantum_channels.push(channel);
        Ok(self.quantum_channels.len() - 1)
    }
    
    /// Send quantum message through entangled channel
    pub fn send_quantum_message(&mut self, channel_id: usize, message: QuantumMessage) -> QuantumResult<()> {
        if channel_id >= self.quantum_channels.len() {
            return Err(QuantumError::EntanglementError("Invalid channel ID".to_string()));
        }
        
        let channel = &mut self.quantum_channels[channel_id];
        
        // Check channel capacity
        if channel.message_queue.len() >= 100 { // Max queue size
            return Err(QuantumError::EntanglementError("Channel capacity exceeded".to_string()));
        }
        
        // Apply channel noise
        let mut noisy_message = message.clone();
        self.apply_channel_noise(&mut noisy_message.quantum_payload, channel.noise_level)?;
        
        channel.message_queue.push(noisy_message);
        Ok(())
    }
    
    /// Receive quantum message from channel
    pub fn receive_quantum_message(&mut self, channel_id: usize) -> QuantumResult<Option<QuantumMessage>> {
        if channel_id >= self.quantum_channels.len() {
            return Err(QuantumError::EntanglementError("Invalid channel ID".to_string()));
        }
        
        let channel = &mut self.quantum_channels[channel_id];
        
        // Sort by priority and timestamp
        channel.message_queue.sort_by(|a, b| {
            a.priority.cmp(&b.priority).then(a.timestamp.cmp(&b.timestamp))
        });
        
        Ok(channel.message_queue.pop())
    }
    
    /// Apply quantum error correction
    pub fn apply_error_correction(&mut self, channel_id: usize) -> QuantumResult<()> {
        if channel_id >= self.quantum_channels.len() {
            return Err(QuantumError::EntanglementError("Invalid channel ID".to_string()));
        }
        
        let channel = &mut self.quantum_channels[channel_id];
        
        for message in &mut channel.message_queue {
            if let Some(error_codes) = &message.error_correction {
                // Simulate quantum error correction
                self.correct_quantum_errors(&mut message.quantum_payload, error_codes)?;
            }
        }
        
        Ok(())
    }
    
    /// Measure entanglement entropy of the network
    pub fn measure_entanglement_entropy(&mut self) -> f64 {
        let mut total_entropy = 0.0;
        
        for epr_pair in &self.epr_pairs {
            let entropy = self.calculate_von_neumann_entropy(epr_pair);
            total_entropy += entropy;
        }
        
        let network_entropy = total_entropy / self.epr_pairs.len().max(1) as f64;
        self.entropy_history.push(network_entropy);
        
        network_entropy
    }
    
    /// Calculate von Neumann entropy for entanglement measure
    fn calculate_von_neumann_entropy(&self, epr_pair: &EPRPair) -> f64 {
        // Simplified von Neumann entropy calculation
        let p0_a = epr_pair.qubit_a.prob_zero();
        let p1_a = epr_pair.qubit_a.prob_one();
        
        let mut entropy = 0.0;
        if p0_a > 0.0 {
            entropy -= p0_a * p0_a.ln();
        }
        if p1_a > 0.0 {
            entropy -= p1_a * p1_a.ln();
        }
        
        entropy
    }
    
    /// Apply decoherence to all entangled pairs
    pub fn apply_decoherence(&mut self, time_step: f64) {
        for epr_pair in &mut self.epr_pairs {
            epr_pair.apply_decoherence(time_step);
        }
        
        // Update correlation matrix with decoherence effects
        for ((i, j), strength) in &mut self.entanglement_strengths {
            *strength *= (-time_step / 10.0).exp(); // Exponential decay
            self.correlation_matrix[(*i, *j)] *= (-time_step / 10.0).exp();
        }
    }
    
    /// Get entanglement strength between two agents
    pub fn get_entanglement_strength(&self, agent1: usize, agent2: usize) -> f64 {
        self.entanglement_strengths.get(&(agent1, agent2)).copied().unwrap_or(0.0)
    }
    
    /// Check if Bell inequality is violated (quantum behavior)
    pub fn check_bell_violation(&self) -> bool {
        if self.bell_states.is_empty() {
            return false;
        }
        
        let average_correlation = self.bell_states.iter()
            .map(|bs| bs.correlation.abs())
            .sum::<f64>() / self.bell_states.len() as f64;
        
        let bell_parameter = 2.0 * (2.0_f64).sqrt() * average_correlation;
        bell_parameter > 2.0 // Quantum mechanical bound
    }
    
    /// Apply channel noise to quantum state
    fn apply_channel_noise(&self, quantum_state: &mut QuantumState, noise_level: f64) -> QuantumResult<()> {
        for qubit in &mut quantum_state.qubits {
            // Apply amplitude damping noise
            let damping_rate = noise_level;
            let decay_factor = (1.0 - damping_rate).sqrt();
            
            qubit.alpha *= decay_factor;
            qubit.beta *= decay_factor;
            
            // Add random phase noise
            let phase_noise = rand::random::<f64>() * noise_level * PI;
            qubit.rotate_z(phase_noise);
        }
        
        Ok(())
    }
    
    /// Correct quantum errors using error correction codes
    fn correct_quantum_errors(&self, quantum_state: &mut QuantumState, error_codes: &[u8]) -> QuantumResult<()> {
        // Simplified quantum error correction (in practice, this would be much more complex)
        for (i, &error_bit) in error_codes.iter().enumerate() {
            if i < quantum_state.qubits.len() && error_bit == 1 {
                // Apply corrective operation
                quantum_state.qubits[i].rotate_x(PI); // Bit flip correction
            }
        }
        
        Ok(())
    }
    
    /// Generate entanglement network report
    pub fn generate_network_report(&self) -> String {
        format!(
            "=== Quantum Entanglement Network Report ===\n\
            Active EPR Pairs: {}\n\
            Bell States Created: {}\n\
            Quantum Channels: {}\n\
            Average Entanglement Entropy: {:.3}\n\
            Bell Inequality Violated: {}\n\
            Network Coherence: {:.3}\n\
            Active Correlations: {}\n",
            self.epr_pairs.len(),
            self.bell_states.len(),
            self.quantum_channels.len(),
            self.entropy_history.last().unwrap_or(&0.0),
            self.check_bell_violation(),
            self.calculate_network_coherence(),
            self.entanglement_strengths.len(),
        )
    }
    
    /// Calculate overall network coherence
    fn calculate_network_coherence(&self) -> f64 {
        if self.entanglement_strengths.is_empty() {
            return 0.0;
        }
        
        let total_strength: f64 = self.entanglement_strengths.values().sum();
        total_strength / self.entanglement_strengths.len() as f64
    }
}

impl EPRPair {
    /// Create new EPR pair in product state
    pub fn new() -> Self {
        Self {
            qubit_a: QuantumBit::new(),
            qubit_b: QuantumBit::new(),
            creation_time: std::time::Instant::now(),
            correlation_strength: 0.0,
            decoherence_rate: 0.01, // 1% per time step
            measurement_history: Vec::new(),
        }
    }
    
    /// Create |Φ+⟩ = (|00⟩ + |11⟩)/√2 Bell state
    pub fn create_phi_plus(&mut self) -> QuantumResult<()> {
        // Apply Hadamard to first qubit
        self.qubit_a.rotate_y(PI / 2.0);
        
        // Apply CNOT (controlled by first qubit)
        if self.qubit_a.prob_one() > 0.5 {
            self.qubit_b.rotate_x(PI);
        }
        
        self.correlation_strength = 1.0; // Maximum correlation
        Ok(())
    }
    
    /// Create |Φ-⟩ = (|00⟩ - |11⟩)/√2 Bell state
    pub fn create_phi_minus(&mut self) -> QuantumResult<()> {
        self.create_phi_plus()?;
        // Apply phase flip to create negative superposition
        self.qubit_b.rotate_z(PI);
        Ok(())
    }
    
    /// Create |Ψ+⟩ = (|01⟩ + |10⟩)/√2 Bell state
    pub fn create_psi_plus(&mut self) -> QuantumResult<()> {
        self.create_phi_plus()?;
        // Apply X gate to second qubit
        self.qubit_b.rotate_x(PI);
        Ok(())
    }
    
    /// Create |Ψ-⟩ = (|01⟩ - |10⟩)/√2 Bell state
    pub fn create_psi_minus(&mut self) -> QuantumResult<()> {
        self.create_psi_plus()?;
        // Apply phase flip
        self.qubit_b.rotate_z(PI);
        Ok(())
    }
    
    /// Measure both qubits and record correlation
    pub fn measure_correlation(&mut self) -> (bool, bool) {
        let result_a = self.qubit_a.measure();
        let result_b = self.qubit_b.measure();
        
        self.measurement_history.push((result_a, result_b, std::time::Instant::now()));
        
        (result_a, result_b)
    }
    
    /// Apply decoherence over time
    pub fn apply_decoherence(&mut self, time_step: f64) {
        let decoherence_factor = (-self.decoherence_rate * time_step).exp();
        
        // Reduce correlation strength
        self.correlation_strength *= decoherence_factor;
        
        // Apply random noise to qubits
        let noise_strength = self.decoherence_rate * time_step;
        
        self.qubit_a.rotate_z(rand::random::<f64>() * noise_strength);
        self.qubit_b.rotate_z(rand::random::<f64>() * noise_strength);
    }
}

impl MessagePriority {
    /// Get numeric priority value
    pub fn value(&self) -> u8 {
        match self {
            MessagePriority::Emergency => 0,
            MessagePriority::High => 1,
            MessagePriority::Normal => 2,
            MessagePriority::Low => 3,
        }
    }
}

impl Ord for MessagePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value().cmp(&other.value())
    }
}

impl PartialOrd for MessagePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Default for QuantumEntanglement {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Default for EPRPair {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bell_state_creation() {
        let mut entanglement = QuantumEntanglement::new(2);
        let bell_state = entanglement.create_bell_state(0, 1, BellStateType::PhiPlus);
        
        assert!(bell_state.is_ok());
        let state = bell_state.unwrap();
        assert_eq!(state.state_type, BellStateType::PhiPlus);
        assert!(state.correlation.abs() > 0.5); // Should show correlation
    }
    
    #[test]
    fn test_epr_pair() {
        let mut epr_pair = EPRPair::new();
        epr_pair.create_phi_plus().unwrap();
        
        assert_eq!(epr_pair.correlation_strength, 1.0);
        
        let (result_a, result_b) = epr_pair.measure_correlation();
        assert_eq!(epr_pair.measurement_history.len(), 1);
        assert_eq!(epr_pair.measurement_history[0].0, result_a);
        assert_eq!(epr_pair.measurement_history[0].1, result_b);
    }
    
    #[test]
    fn test_quantum_channel() {
        let mut entanglement = QuantumEntanglement::new(3);
        let channel_id = entanglement.create_quantum_channel(0, 1, 100.0).unwrap();
        
        assert_eq!(channel_id, 0);
        assert_eq!(entanglement.quantum_channels.len(), 1);
        assert_eq!(entanglement.quantum_channels[0].source, 0);
        assert_eq!(entanglement.quantum_channels[0].target, 1);
    }
    
    #[test]
    fn test_bell_violation() {
        let mut entanglement = QuantumEntanglement::new(2);
        
        // Create highly correlated Bell state
        entanglement.create_bell_state(0, 1, BellStateType::PhiPlus).unwrap();
        
        // Check if Bell inequality is violated (indicating quantum behavior)
        assert!(entanglement.check_bell_violation());
    }
}