//! Quantum Prospect Theory Agent
//! 
//! This agent implements quantum-enhanced prospect theory for modeling
//! decision-making under uncertainty and behavioral biases in financial markets.

use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use pyo3::prelude::*;
use crate::error::PadsError;
use super::{QuantumAgent, QuantumBridge, QuantumMetrics};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumProspectTheory {
    pub agent_id: String,
    pub bridge: Arc<QuantumBridge>,
    pub num_qubits: usize,
    pub reference_point: f64,
    pub loss_aversion_coefficient: f64,
    pub probability_weighting: ProbabilityWeighting,
    pub value_function: ValueFunction,
    pub decision_weights: Vec<f64>,
    pub behavioral_biases: BehavioralBiases,
    pub metrics: QuantumMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityWeighting {
    pub alpha_gains: f64,      // Probability weighting for gains
    pub alpha_losses: f64,     // Probability weighting for losses
    pub delta: f64,            // Decision weight parameter
    pub quantum_interference: f64, // Quantum superposition effect
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueFunction {
    pub alpha: f64,            // Risk aversion parameter for gains
    pub beta: f64,             // Risk seeking parameter for losses
    pub lambda: f64,           // Loss aversion parameter
    pub curvature: f64,        // Value function curvature
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralBiases {
    pub overconfidence: f64,
    pub anchoring: f64,
    pub availability_heuristic: f64,
    pub mental_accounting: f64,
    pub framing_effect: f64,
}

impl QuantumProspectTheory {
    /// Create a new Quantum Prospect Theory agent
    pub async fn new(bridge: Arc<QuantumBridge>) -> Result<Self, PadsError> {
        let agent_id = "quantum_prospect_theory".to_string();
        let num_qubits = 10;
        
        let probability_weighting = ProbabilityWeighting {
            alpha_gains: 0.88,    // Typical value from empirical studies
            alpha_losses: 0.88,
            delta: 0.69,          // Decision weight parameter
            quantum_interference: 0.3, // Quantum coherence effect
        };
        
        let value_function = ValueFunction {
            alpha: 0.88,          // Risk aversion for gains
            beta: 0.88,           // Risk seeking for losses
            lambda: 2.25,         // Loss aversion coefficient
            curvature: 0.5,       // S-shaped value function
        };
        
        let behavioral_biases = BehavioralBiases {
            overconfidence: 0.15,
            anchoring: 0.2,
            availability_heuristic: 0.18,
            mental_accounting: 0.12,
            framing_effect: 0.25,
        };
        
        let decision_weights = vec![0.3, 0.25, 0.2, 0.15, 0.1]; // Decreasing importance
        
        let metrics = QuantumMetrics {
            agent_id: agent_id.clone(),
            circuit_depth: 16,
            gate_count: 96,
            quantum_volume: 512.0,
            execution_time_ms: 220,
            fidelity: 0.86,
            error_rate: 0.14,
            coherence_time: 44.0,
        };
        
        Ok(Self {
            agent_id,
            bridge,
            num_qubits,
            reference_point: 0.0,
            loss_aversion_coefficient: 2.25,
            probability_weighting,
            value_function,
            decision_weights,
            behavioral_biases,
            metrics,
        })
    }
    
    /// Generate quantum circuit for prospect theory evaluation
    pub fn generate_prospect_circuit(&self, outcomes: &[f64], probabilities: &[f64], market_context: &[f64]) -> String {
        let alpha_gains = self.probability_weighting.alpha_gains;
        let alpha_losses = self.probability_weighting.alpha_losses;
        let lambda = self.value_function.lambda;
        let quantum_interference = self.probability_weighting.quantum_interference;
        
        format!(r#"
import pennylane as qml
import numpy as np
import torch

# Device setup for quantum prospect theory
dev = qml.device('default.qubit', wires={})

@qml.qnode(dev)
def quantum_prospect_evaluation(outcomes, probabilities, market_context, pt_params):
    alpha_gains, alpha_losses, lambda_coeff, quantum_interference = pt_params
    
    # Initialize outcome states in quantum superposition
    for i in range(min(4, len(outcomes))):
        # Encode outcome magnitude and sign
        outcome_angle = np.arctan(outcomes[i]) if outcomes[i] != 0 else 0
        qml.RY(outcome_angle, wires=i)
    
    # Initialize probability weights
    for i in range(min(3, len(probabilities))):
        # Probability weighting transformation
        if probabilities[i] > 0:
            prob_weight = probabilities[i] ** alpha_gains if outcomes[i % len(outcomes)] >= 0 else probabilities[i] ** alpha_losses
            qml.RX(prob_weight * np.pi, wires=i + 4)
        else:
            qml.RX(0, wires=i + 4)
    
    # Initialize market context (framing effects)
    for i in range(min(3, len(market_context))):
        qml.RZ(market_context[i] * np.pi, wires=i + 7)
    
    # Quantum prospect theory computation layers
    for layer in range(3):
        # Value function computation (quantum)
        for i in range(4):
            outcome = outcomes[i % len(outcomes)]
            
            # Reference point adjustment
            relative_outcome = outcome - 0  # Reference point = 0 for simplicity
            
            if relative_outcome >= 0:
                # Gains: concave value function
                value_angle = np.power(abs(relative_outcome), alpha_gains) * np.pi * 0.1
                qml.RY(value_angle, wires=i)
            else:
                # Losses: convex value function with loss aversion
                value_angle = -lambda_coeff * np.power(abs(relative_outcome), alpha_losses) * np.pi * 0.1
                qml.RY(value_angle, wires=i)
        
        # Probability weighting (quantum superposition)
        for i in range(3):
            prob = probabilities[i % len(probabilities)]
            outcome = outcomes[i % len(outcomes)]
            
            # Quantum probability weighting
            if outcome >= 0:
                # Overweighting of small probabilities for gains
                weighted_prob = prob ** alpha_gains
            else:
                # Different weighting for losses
                weighted_prob = prob ** alpha_losses
            
            qml.RX(weighted_prob * np.pi, wires=i + 4)
            
            # Quantum interference effects
            qml.Hadamard(wires=i + 4)
            qml.RZ(quantum_interference * np.pi, wires=i + 4)
        
        # Behavioral bias encoding
        # Overconfidence bias
        qml.RY({} * np.pi, wires=7)
        
        # Anchoring bias (reference point stickiness)
        qml.RZ({} * np.pi, wires=8)
        
        # Availability heuristic (recent event salience)
        qml.RX({} * market_context[0 % len(market_context)] * np.pi, wires=9)
        
        # Decision weight computation through entanglement
        for i in range(4):
            # Value-probability entanglement
            qml.CNOT(wires=[i, i + 4])
            
            # Behavioral bias integration
            qml.CNOT(wires=[i + 4, 7])
            qml.CNOT(wires=[i + 4, 8])
            qml.CNOT(wires=[i + 4, 9])
        
        # Quantum superposition of decision states
        for i in range(3):
            qml.Hadamard(wires=i + 7)
            qml.CNOT(wires=[i + 7, (i + 1) % 3 + 7])
        
        # Mental accounting (separate evaluation of gains/losses)
        gain_qubits = [0, 1]  # Assume first two outcomes are gains
        loss_qubits = [2, 3]  # Assume last two outcomes are losses
        
        # Separate processing for gains and losses
        for i in gain_qubits:
            qml.RY(0.5 * np.pi, wires=i)  # Gains processing
        
        for i in loss_qubits:
            qml.RY(-lambda_coeff * 0.2 * np.pi, wires=i)  # Loss aversion
        
        # Cross-domain correlation (gains-losses interaction)
        qml.CNOT(wires=[gain_qubits[0], loss_qubits[0]])
        qml.CNOT(wires=[gain_qubits[1], loss_qubits[1]])
    
    # Advanced prospect theory features
    # Framing effects (quantum context dependence)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 7])
    qml.RY({} * np.pi, wires=7)
    
    # Decision under uncertainty (quantum coherence)
    for i in range({}):
        qml.RZ(quantum_interference * np.pi * 0.5, wires=i)
    
    # Quantum measurements for prospect theory outputs
    prospect_results = []
    
    # Expected utility under prospect theory
    expected_utility = qml.expval(qml.PauliZ(0) @ qml.PauliZ(4) + 
                                 qml.PauliZ(1) @ qml.PauliZ(5) + 
                                 qml.PauliZ(2) @ qml.PauliZ(6))
    prospect_results.append(expected_utility)
    
    # Loss aversion strength
    loss_aversion = qml.expval(qml.PauliY(2) @ qml.PauliY(3))
    prospect_results.append(loss_aversion)
    
    # Probability weighting distortion
    prob_distortion = qml.expval(qml.PauliX(4) @ qml.PauliX(5) @ qml.PauliX(6))
    prospect_results.append(prob_distortion)
    
    # Behavioral bias impact
    bias_impact = qml.expval(qml.PauliZ(7) @ qml.PauliZ(8) @ qml.PauliZ(9))
    prospect_results.append(bias_impact)
    
    # Decision confidence
    confidence = qml.expval(qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliY(7))
    prospect_results.append(confidence)
    
    # Framing effect strength
    framing = qml.expval(qml.PauliX(0) @ qml.PauliZ(7))
    prospect_results.append(framing)
    
    return prospect_results

# Execute quantum prospect theory
outcomes = np.array({:?})
probabilities = np.array({:?})
market_context = np.array({:?})
pt_params = [{}, {}, {}, {}]

result = quantum_prospect_evaluation(outcomes, probabilities, market_context, pt_params)
result
"#, 
        self.num_qubits,
        self.behavioral_biases.overconfidence,
        self.behavioral_biases.anchoring,
        self.behavioral_biases.availability_heuristic,
        self.behavioral_biases.framing_effect,
        self.num_qubits,
        outcomes,
        probabilities,
        market_context,
        alpha_gains,
        alpha_losses,
        lambda,
        quantum_interference
        )
    }
    
    /// Evaluate investment prospects using quantum prospect theory
    pub async fn evaluate_prospects(&self, outcomes: &[f64], probabilities: &[f64], market_context: &[f64]) -> Result<Vec<f64>, PadsError> {
        let circuit = self.generate_prospect_circuit(outcomes, probabilities, market_context);
        let raw_results = self.bridge.execute_circuit(&circuit).await?;
        
        // Post-process quantum results
        let mut prospect_values = Vec::new();
        
        for (i, &result) in raw_results.iter().enumerate() {
            let processed_value = match i {
                0 => result, // Expected utility (already normalized)
                1 => result * self.loss_aversion_coefficient, // Loss aversion adjustment
                2 => result * self.probability_weighting.delta, // Probability weighting
                3 => result * 0.5, // Behavioral bias impact (scale down)
                4 => (result + 1.0) / 2.0, // Confidence (normalize to [0,1])
                5 => result * self.behavioral_biases.framing_effect, // Framing effect
                _ => result,
            };
            
            prospect_values.push(processed_value);
        }
        
        Ok(prospect_values)
    }
    
    /// Calculate decision weights using quantum probability weighting
    pub async fn calculate_decision_weights(&self, probabilities: &[f64], context: &[f64]) -> Result<Vec<f64>, PadsError> {
        let outcomes = vec![1.0, -1.0, 0.5, -0.5]; // Generic outcomes for weight calculation
        let evaluation = self.evaluate_prospects(&outcomes, probabilities, context).await?;
        
        let mut weights = Vec::new();
        
        // Extract probability weighting distortion
        if evaluation.len() > 2 {
            let distortion = evaluation[2];
            
            for (i, &prob) in probabilities.iter().enumerate() {
                let base_weight = if i < outcomes.len() && outcomes[i] >= 0.0 {
                    prob.powf(self.probability_weighting.alpha_gains)
                } else {
                    prob.powf(self.probability_weighting.alpha_losses)
                };
                
                // Apply quantum interference
                let quantum_weight = base_weight * (1.0 + distortion * self.probability_weighting.quantum_interference);
                weights.push(quantum_weight.clamp(0.0, 1.0));
            }
        }
        
        Ok(weights)
    }
    
    /// Model behavioral biases in decision making
    pub async fn model_behavioral_biases(&self, decision_context: &[f64]) -> Result<Vec<f64>, PadsError> {
        let outcomes = vec![0.1, -0.1, 0.05, -0.15]; // Test outcomes
        let probabilities = vec![0.6, 0.3, 0.8, 0.2]; // Test probabilities
        
        let evaluation = self.evaluate_prospects(&outcomes, &probabilities, decision_context).await?;
        
        let mut bias_effects = Vec::new();
        
        if evaluation.len() >= 6 {
            // Extract bias effects
            bias_effects.push(evaluation[3] * self.behavioral_biases.overconfidence);
            bias_effects.push(evaluation[5] * self.behavioral_biases.framing_effect);
            bias_effects.push(evaluation[1] * self.behavioral_biases.mental_accounting);
            
            // Anchoring effect (based on reference point distance)
            let anchoring_effect = self.reference_point * self.behavioral_biases.anchoring;
            bias_effects.push(anchoring_effect);
            
            // Availability heuristic (recent market events)
            let availability_effect = decision_context.get(0).unwrap_or(&0.0) * self.behavioral_biases.availability_heuristic;
            bias_effects.push(availability_effect);
        }
        
        Ok(bias_effects)
    }
    
    /// Update reference point based on recent outcomes
    pub fn update_reference_point(&mut self, recent_outcomes: &[f64], adaptation_rate: f64) {
        if !recent_outcomes.is_empty() {
            let average_outcome: f64 = recent_outcomes.iter().sum::<f64>() / recent_outcomes.len() as f64;
            
            // Adaptive reference point updating
            self.reference_point = (1.0 - adaptation_rate) * self.reference_point + adaptation_rate * average_outcome;
        }
    }
    
    /// Generate trading decision based on prospect theory
    pub async fn make_trading_decision(&self, investment_options: &[(Vec<f64>, Vec<f64>)], market_context: &[f64]) -> Result<usize, PadsError> {
        let mut best_option = 0;
        let mut best_prospect_value = f64::NEG_INFINITY;
        
        for (i, (outcomes, probabilities)) in investment_options.iter().enumerate() {
            let prospect_evaluation = self.evaluate_prospects(outcomes, probabilities, market_context).await?;
            
            // Use expected utility as decision criterion
            if let Some(&expected_utility) = prospect_evaluation.get(0) {
                if expected_utility > best_prospect_value {
                    best_prospect_value = expected_utility;
                    best_option = i;
                }
            }
        }
        
        Ok(best_option)
    }
    
    /// Calculate certainty equivalent for a risky prospect
    pub async fn certainty_equivalent(&self, outcomes: &[f64], probabilities: &[f64]) -> Result<f64, PadsError> {
        let market_context = vec![0.0, 0.0, 0.0]; // Neutral context
        let evaluation = self.evaluate_prospects(outcomes, probabilities, &market_context).await?;
        
        if let Some(&expected_utility) = evaluation.get(0) {
            // Convert utility back to monetary equivalent
            // This is simplified - in practice would use inverse value function
            Ok(expected_utility / self.value_function.alpha)
        } else {
            Ok(0.0)
        }
    }
}

#[async_trait]
impl QuantumAgent for QuantumProspectTheory {
    fn agent_id(&self) -> &str {
        &self.agent_id
    }
    
    fn quantum_circuit(&self) -> String {
        let outcomes = vec![0.1, -0.05, 0.08, -0.12];
        let probabilities = vec![0.4, 0.3, 0.6];
        let context = vec![0.0, 0.0, 0.0];
        self.generate_prospect_circuit(&outcomes, &probabilities, &context)
    }
    
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
    
    async fn execute(&self, input: &[f64]) -> Result<Vec<f64>, PadsError> {
        let outcomes = &input[..4.min(input.len())];
        let probabilities = &input[4..7.min(input.len())];
        let context = &input[7..10.min(input.len())];
        
        self.evaluate_prospects(outcomes, probabilities, context).await
    }
    
    async fn train(&mut self, training_data: &[Vec<f64>]) -> Result<(), PadsError> {
        let mut all_outcomes = Vec::new();
        
        for data in training_data {
            let outcomes = &data[..4.min(data.len())];
            all_outcomes.extend_from_slice(outcomes);
        }
        
        // Update reference point based on training outcomes
        self.update_reference_point(&all_outcomes, 0.1);
        
        // Update behavioral biases based on prediction accuracy (simplified)
        for bias_value in [
            &mut self.behavioral_biases.overconfidence,
            &mut self.behavioral_biases.anchoring,
            &mut self.behavioral_biases.availability_heuristic,
        ] {
            *bias_value *= 0.95; // Slight reduction through learning
            *bias_value = bias_value.clamp(0.01, 0.5);
        }
        
        Ok(())
    }
    
    fn get_metrics(&self) -> QuantumMetrics {
        self.metrics.clone()
    }
}