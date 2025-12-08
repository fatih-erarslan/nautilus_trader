//! Quantum Prospect Theory Agent
//! 
//! Implements behavioral finance modeling using quantum prospect theory,
//! quantum decision theory, and quantum behavioral economics for market psychology.

use super::{QuantumAgent, QuantumMetrics, QuantumBridge};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QProspectConfig {
    pub num_qubits: usize,
    pub behavioral_layers: usize,
    pub loss_aversion_factor: f64,
    pub probability_weighting_alpha: f64,
    pub reference_point: f64,
    pub decision_tree_depth: usize,
    pub quantum_interference_strength: f64,
}

impl Default for QProspectConfig {
    fn default() -> Self {
        Self {
            num_qubits: 8,
            behavioral_layers: 4,
            loss_aversion_factor: 2.25,  // Kahneman-Tversky empirical value
            probability_weighting_alpha: 0.88,  // Empirical probability weighting
            reference_point: 0.0,
            decision_tree_depth: 3,
            quantum_interference_strength: 0.1,
        }
    }
}

/// Quantum Prospect Theory Agent
/// 
/// Models behavioral finance using quantum decision theory, quantum prospect theory,
/// and quantum interference effects in financial decision making.
pub struct QuantumProspectTheory {
    config: QProspectConfig,
    reference_point: Arc<RwLock<f64>>,
    behavioral_weights: Arc<RwLock<Vec<f64>>>,
    decision_history: Arc<RwLock<Vec<DecisionAnalysis>>>,
    quantum_utility_function: Arc<RwLock<Vec<f64>>>,
    probability_distortions: Arc<RwLock<HashMap<String, f64>>>,
    bridge: Arc<QuantumBridge>,
    metrics: QuantumMetrics,
    behavioral_biases: Arc<RwLock<BehavioralBiases>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecisionAnalysis {
    timestamp: u64,
    input_data: Vec<f64>,
    gains_vs_losses: (f64, f64),
    probability_weights: Vec<f64>,
    utility_values: Vec<f64>,
    quantum_interference: f64,
    decision_value: f64,
    loss_aversion_effect: f64,
    certainty_effect: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BehavioralBiases {
    anchoring_bias: f64,
    availability_heuristic: f64,
    overconfidence: f64,
    herding_tendency: f64,
    framing_effect: f64,
    endowment_effect: f64,
    mental_accounting: f64,
}

impl QuantumProspectTheory {
    pub async fn new(bridge: Arc<QuantumBridge>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config = QProspectConfig::default();
        
        // Initialize behavioral weights
        let behavioral_weights = vec![1.0; config.num_qubits];
        
        // Initialize quantum utility function parameters
        let quantum_utility_function = vec![0.0; config.num_qubits * 2]; // For gains and losses
        
        let metrics = QuantumMetrics {
            agent_id: "QProspectTheory".to_string(),
            circuit_depth: config.behavioral_layers * 2,
            gate_count: config.num_qubits * config.behavioral_layers * 6,
            quantum_volume: (config.num_qubits * config.behavioral_layers) as f64 * 2.2,
            execution_time_ms: 0,
            fidelity: 0.86,
            error_rate: 0.14,
            coherence_time: 90.0,
        };
        
        let behavioral_biases = BehavioralBiases {
            anchoring_bias: 0.3,
            availability_heuristic: 0.4,
            overconfidence: 0.25,
            herding_tendency: 0.35,
            framing_effect: 0.2,
            endowment_effect: 0.3,
            mental_accounting: 0.15,
        };
        
        Ok(Self {
            config,
            reference_point: Arc::new(RwLock::new(config.reference_point)),
            behavioral_weights: Arc::new(RwLock::new(behavioral_weights)),
            decision_history: Arc::new(RwLock::new(Vec::new())),
            quantum_utility_function: Arc::new(RwLock::new(quantum_utility_function)),
            probability_distortions: Arc::new(RwLock::new(HashMap::new())),
            bridge,
            metrics,
            behavioral_biases: Arc::new(RwLock::new(behavioral_biases)),
        })
    }
    
    /// Generate quantum prospect theory circuit
    fn generate_quantum_prospect_circuit(&self, financial_data: &[f64], probabilities: &[f64]) -> String {
        format!(r#"
import pennylane as qml
import numpy as np
import torch

# Create quantum device for prospect theory modeling
dev = qml.device('default.qubit', wires={})

@qml.qnode(dev, interface='torch')
def quantum_prospect_theory_circuit(outcomes, probabilities, reference_point, loss_aversion):
    # Encode financial outcomes relative to reference point
    for i, outcome in enumerate(outcomes):
        if i < {}:
            # Encode gain/loss relative to reference point
            relative_outcome = outcome - reference_point
            
            if relative_outcome >= 0:
                # Gains domain - concave utility
                gain_utility = np.power(relative_outcome + 1e-8, 0.88)  # Diminishing sensitivity
                qml.RY(gain_utility * np.pi / 2, wires=i)
            else:
                # Loss domain - convex utility with loss aversion
                loss_utility = -loss_aversion * np.power(abs(relative_outcome) + 1e-8, 0.88)
                qml.RY(abs(loss_utility) * np.pi / 2, wires=i)
                qml.PauliZ(wires=i)  # Mark as loss
    
    # Probability weighting transformation
    for i, prob in enumerate(probabilities):
        if i < {}:
            # Kahneman-Tversky probability weighting function
            # w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)
            gamma = 0.61  # Empirical parameter for losses
            if prob > 0 and prob < 1:
                numerator = np.power(prob, gamma)
                denominator = np.power(numerator + np.power(1 - prob, gamma), 1/gamma)
                weighted_prob = numerator / (denominator + 1e-8)
            else:
                weighted_prob = prob
            
            # Encode weighted probability
            qml.RZ(weighted_prob * np.pi, wires=i)
    
    # Behavioral bias encoding layers
    for layer in range({}):
        # Anchoring bias (reference point dependency)
        for i in range({}):
            anchoring_strength = 0.3  # Typical anchoring effect
            qml.RX(anchoring_strength * reference_point * np.pi / 100, wires=i)
        
        # Framing effect (gain vs loss framing)
        for i in range({} - 1):
            # Quantum superposition for framing ambiguity
            qml.Hadamard(wires=i)
            qml.CRY(np.pi / 4, wires=[i, i + 1])  # Framing correlation
            qml.Hadamard(wires=i)
        
        # Overconfidence bias
        for i in range({}):
            overconfidence = 0.25
            qml.RY(overconfidence * np.pi / 4, wires=i)
    
    # Quantum interference effects in decision making
    # Implement quantum superposition of decision states
    decision_qubits = min(4, {})
    for i in range(decision_qubits):
        qml.Hadamard(wires=i)
    
    # Quantum interference between different choice scenarios
    for i in range(decision_qubits - 1):
        interference_strength = {}  # From config
        qml.CRY(interference_strength * np.pi, wires=[i, i + 1])
    
    # Entanglement for correlated behavioral effects
    for i in range(0, {}, 2):
        if i + 1 < {}:
            # Mental accounting correlations
            qml.CNOT(wires=[i, i + 1])
    
    # Quantum decision tree implementation
    tree_depth = min({}, 3)
    for depth in range(tree_depth):
        for i in range(min(2**depth, {})):
            if i < {}:
                # Decision branching based on utility comparison
                qml.RY(np.pi / (2**(depth + 1)), wires=i)
    
    # Measurements for prospect theory analysis
    measurements = []
    
    # Utility measurements (gains vs losses)
    for i in range(min(4, {})):
        measurements.append(qml.expval(qml.PauliZ(i)))  # Utility values
    
    # Probability weighting measurements
    for i in range(min(2, {})):
        measurements.append(qml.expval(qml.PauliX(i)))  # Probability distortions
    
    # Behavioral bias measurements
    if {} > 6:
        # Anchoring bias
        measurements.append(qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)))
        
        # Framing effect
        measurements.append(qml.expval(qml.PauliX(2) @ qml.PauliY(3)))
        
        # Overconfidence
        measurements.append(qml.expval(qml.PauliY(4) @ qml.PauliZ(5)))
    
    # Quantum interference measurement
    if {} > 7:
        measurements.append(qml.expval(qml.PauliX(6) @ qml.PauliX(7)))
    
    return measurements

# Execute quantum prospect theory circuit
outcomes_tensor = torch.tensor({}, dtype=torch.float32)
probs_tensor = torch.tensor({}, dtype=torch.float32)
ref_point = {}
loss_aversion = {}

result = quantum_prospect_theory_circuit(outcomes_tensor, probs_tensor, ref_point, loss_aversion)
[float(x) for x in result]
"#,
            self.config.num_qubits,
            self.config.num_qubits,
            self.config.num_qubits,
            self.config.behavioral_layers,
            self.config.num_qubits,
            self.config.num_qubits,
            self.config.num_qubits,
            self.config.num_qubits,
            self.config.quantum_interference_strength,
            self.config.num_qubits,
            self.config.num_qubits,
            self.config.decision_tree_depth,
            self.config.num_qubits,
            self.config.num_qubits,
            self.config.num_qubits,
            self.config.num_qubits,
            self.config.num_qubits,
            self.config.num_qubits,
            &financial_data[..financial_data.len().min(self.config.num_qubits)],
            &probabilities[..probabilities.len().min(self.config.num_qubits)],
            *self.reference_point.try_read().unwrap(),
            self.config.loss_aversion_factor
        )
    }
    
    /// Compute quantum utility function based on prospect theory
    async fn compute_quantum_utility(&self, outcomes: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let utility_code = format!(r#"
import numpy as np
import math

def quantum_prospect_utility(outcomes, reference_point, loss_aversion, alpha=0.88):
    """Compute quantum-enhanced prospect theory utility"""
    utilities = []
    
    for outcome in outcomes:
        relative_outcome = outcome - reference_point
        
        if relative_outcome >= 0:
            # Gains: utility = x^α (concave, diminishing sensitivity)
            utility = np.power(relative_outcome + 1e-10, alpha)
        else:
            # Losses: utility = -λ * |x|^α (convex in losses, loss aversion)
            utility = -loss_aversion * np.power(abs(relative_outcome) + 1e-10, alpha)
        
        # Add quantum interference effect
        quantum_phase = math.sin(outcome * math.pi / 10)  # Quantum oscillation
        quantum_correction = 0.05 * quantum_phase  # 5% quantum effect
        
        utilities.append(utility * (1 + quantum_correction))
    
    return utilities

def probability_weighting_function(probabilities, gamma=0.61):
    """Kahneman-Tversky probability weighting function"""
    weighted_probs = []
    
    for p in probabilities:
        if p <= 0:
            weighted_probs.append(0.0)
        elif p >= 1:
            weighted_probs.append(1.0)
        else:
            # w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)
            numerator = np.power(p, gamma)
            denominator_term = numerator + np.power(1 - p, gamma)
            denominator = np.power(denominator_term, 1/gamma)
            
            if denominator > 0:
                weighted_prob = numerator / denominator
            else:
                weighted_prob = p
            
            weighted_probs.append(weighted_prob)
    
    return weighted_probs

def compute_prospect_value(utilities, weighted_probabilities):
    """Compute overall prospect value"""
    if len(utilities) != len(weighted_probabilities):
        return 0.0
    
    prospect_value = 0.0
    for utility, weight in zip(utilities, weighted_probabilities):
        prospect_value += utility * weight
    
    return prospect_value

# Compute utilities and prospect value
outcomes = {}
reference_point = {}
loss_aversion = {}

utilities = quantum_prospect_utility(outcomes, reference_point, loss_aversion)

# Generate some probabilities for demonstration
probabilities = [1.0 / len(outcomes)] * len(outcomes) if outcomes else []
weighted_probs = probability_weighting_function(probabilities)

prospect_value = compute_prospect_value(utilities, weighted_probs)

{{
    "utilities": utilities,
    "weighted_probabilities": weighted_probs,
    "prospect_value": prospect_value
}}
"#,
            outcomes,
            *self.reference_point.read().await,
            self.config.loss_aversion_factor
        );
        
        let result = py.eval(&utility_code, None, None)?;
        let utility_data: HashMap<String, PyObject> = result.extract()?;
        
        let utilities: Vec<f64> = utility_data.get("utilities").unwrap().extract(py)?;
        
        // Update quantum utility function
        {
            let mut quf = self.quantum_utility_function.write().await;
            if utilities.len() <= quf.len() {
                for (i, &utility) in utilities.iter().enumerate() {
                    quf[i] = utility;
                }
            }
        }
        
        Ok(utilities)
    }
    
    /// Analyze behavioral biases in financial decisions
    async fn analyze_behavioral_biases(&self, decision_data: &[f64]) -> Result<HashMap<String, f64>, Box<dyn std::error::Error + Send + Sync>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let bias_code = format!(r#"
import numpy as np
import math

def analyze_behavioral_biases(decision_data, reference_point):
    biases = {{}}
    
    if len(decision_data) == 0:
        return biases
    
    # Anchoring bias - tendency to rely on first piece of information
    if len(decision_data) > 1:
        first_value = decision_data[0]
        subsequent_values = decision_data[1:]
        
        # Measure how much subsequent decisions are influenced by first value
        correlations = [abs(val - first_value) for val in subsequent_values]
        avg_correlation = np.mean(correlations) if correlations else 0
        
        # Normalize anchoring bias (0 = no bias, 1 = complete anchoring)
        max_range = max(decision_data) - min(decision_data) if len(decision_data) > 1 else 1
        biases['anchoring_bias'] = 1 - (avg_correlation / (max_range + 1e-8))
    
    # Loss aversion - greater sensitivity to losses than gains
    gains = [val - reference_point for val in decision_data if val > reference_point]
    losses = [reference_point - val for val in decision_data if val < reference_point]
    
    if gains and losses:
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        # Loss aversion ratio (should be around 2.25 according to research)
        loss_aversion_ratio = avg_loss / (avg_gain + 1e-8)
        biases['loss_aversion'] = min(loss_aversion_ratio / 2.25, 2.0)  # Normalize
    else:
        biases['loss_aversion'] = 1.0  # Default assumption
    
    # Overconfidence - overestimation of accuracy
    if len(decision_data) > 5:
        # Measure volatility as proxy for overconfidence
        volatility = np.std(decision_data)
        mean_value = np.mean(decision_data)
        
        if mean_value != 0:
            coefficient_of_variation = volatility / abs(mean_value)
            # Higher volatility suggests more overconfident decisions
            biases['overconfidence'] = min(coefficient_of_variation, 1.0)
        else:
            biases['overconfidence'] = 0.5
    
    # Herding tendency - following the crowd
    if len(decision_data) > 3:
        # Measure trend-following behavior
        trends = []
        for i in range(1, len(decision_data)):
            if decision_data[i] > decision_data[i-1]:
                trends.append(1)  # Up trend
            elif decision_data[i] < decision_data[i-1]:
                trends.append(-1)  # Down trend
            else:
                trends.append(0)  # No change
        
        if trends:
            # Measure consistency in trend direction
            trend_consistency = abs(sum(trends)) / len(trends)
            biases['herding_tendency'] = trend_consistency
    
    # Availability heuristic - judging probability by ease of recall
    if len(decision_data) > 2:
        # Measure recency bias (recent values having more influence)
        recent_weight = 0.7
        older_weight = 0.3
        
        recent_data = decision_data[-len(decision_data)//3:] if len(decision_data) > 3 else decision_data[-1:]
        older_data = decision_data[:len(decision_data)//3] if len(decision_data) > 3 else decision_data[:-1]
        
        if recent_data and older_data:
            recent_mean = np.mean(recent_data)
            older_mean = np.mean(older_data)
            
            # Measure how much recent data differs from older data
            recency_effect = abs(recent_mean - older_mean) / (abs(older_mean) + 1e-8)
            biases['availability_heuristic'] = min(recency_effect, 1.0)
    
    # Framing effect - different decisions based on how options are presented
    if len(decision_data) > 4:
        # Measure inconsistency in decisions (proxy for framing susceptibility)
        mid_point = len(decision_data) // 2
        first_half = decision_data[:mid_point]
        second_half = decision_data[mid_point:]
        
        if first_half and second_half:
            first_mean = np.mean(first_half)
            second_mean = np.mean(second_half)
            
            # Inconsistency suggests susceptibility to framing
            inconsistency = abs(first_mean - second_mean) / (abs(first_mean) + abs(second_mean) + 1e-8)
            biases['framing_effect'] = min(inconsistency, 1.0)
    
    return biases

# Analyze biases in decision data
decision_data = {}
reference_point = {}

behavioral_biases = analyze_behavioral_biases(decision_data, reference_point)
behavioral_biases
"#,
            decision_data,
            *self.reference_point.read().await
        );
        
        let result = py.eval(&bias_code, None, None)?;
        let biases: HashMap<String, f64> = result.extract()?;
        
        // Update behavioral biases
        {
            let mut bb = self.behavioral_biases.write().await;
            if let Some(&anchoring) = biases.get("anchoring_bias") {
                bb.anchoring_bias = anchoring;
            }
            if let Some(&loss_aversion) = biases.get("loss_aversion") {
                bb.endowment_effect = loss_aversion; // Related concept
            }
            if let Some(&overconfidence) = biases.get("overconfidence") {
                bb.overconfidence = overconfidence;
            }
            if let Some(&herding) = biases.get("herding_tendency") {
                bb.herding_tendency = herding;
            }
            if let Some(&availability) = biases.get("availability_heuristic") {
                bb.availability_heuristic = availability;
            }
            if let Some(&framing) = biases.get("framing_effect") {
                bb.framing_effect = framing;
            }
        }
        
        Ok(biases)
    }
    
    /// Update reference point based on market adaptation
    async fn update_reference_point(&self, market_data: &[f64]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if market_data.is_empty() {
            return Ok(());
        }
        
        // Use exponential moving average to update reference point
        let alpha = 0.1; // Learning rate
        let current_value = market_data.iter().sum::<f64>() / market_data.len() as f64;
        
        let mut ref_point = self.reference_point.write().await;
        *ref_point = alpha * current_value + (1.0 - alpha) * *ref_point;
        
        Ok(())
    }
}

impl QuantumAgent for QuantumProspectTheory {
    fn agent_id(&self) -> &str {
        "QProspectTheory"
    }
    
    fn quantum_circuit(&self) -> String {
        let dummy_outcomes = vec![100.0, 95.0, 105.0, 90.0, 110.0, 85.0, 115.0, 80.0];
        let dummy_probs = vec![0.2, 0.15, 0.25, 0.1, 0.15, 0.05, 0.08, 0.02];
        self.generate_quantum_prospect_circuit(&dummy_outcomes, &dummy_probs)
    }
    
    fn num_qubits(&self) -> usize {
        self.config.num_qubits
    }
    
    async fn execute(&self, input: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();
        
        // Split input into outcomes and probabilities
        let mid_point = input.len() / 2;
        let outcomes = &input[..mid_point];
        let probabilities = if input.len() > mid_point {
            &input[mid_point..]
        } else {
            // Generate uniform probabilities if not provided
            &vec![1.0 / outcomes.len() as f64; outcomes.len()]
        };
        
        // Generate and execute quantum prospect theory circuit
        let circuit_code = self.generate_quantum_prospect_circuit(outcomes, probabilities);
        let quantum_result = self.bridge.execute_circuit(&circuit_code).await?;
        
        // Compute quantum utilities
        let utilities = self.compute_quantum_utility(outcomes).await?;
        
        // Analyze behavioral biases
        let biases = self.analyze_behavioral_biases(input).await?;
        
        // Update reference point based on current data
        self.update_reference_point(outcomes).await?;
        
        // Prepare comprehensive result
        let mut result = quantum_result;
        
        // Add utility values
        result.extend(utilities);
        
        // Add behavioral bias measurements
        result.push(biases.get("anchoring_bias").unwrap_or(&0.3).clone());
        result.push(biases.get("loss_aversion").unwrap_or(&2.25).clone());
        result.push(biases.get("overconfidence").unwrap_or(&0.25).clone());
        result.push(biases.get("herding_tendency").unwrap_or(&0.35).clone());
        result.push(biases.get("availability_heuristic").unwrap_or(&0.4).clone());
        result.push(biases.get("framing_effect").unwrap_or(&0.2).clone());
        
        // Add prospect theory specific metrics
        let bb = self.behavioral_biases.read().await;
        result.push(bb.endowment_effect);
        result.push(bb.mental_accounting);
        result.push(*self.reference_point.read().await);
        result.push(self.config.loss_aversion_factor);
        
        // Record decision analysis
        let decision_analysis = DecisionAnalysis {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            input_data: input.to_vec(),
            gains_vs_losses: {
                let ref_point = *self.reference_point.read().await;
                let gains = outcomes.iter().filter(|&&x| x > ref_point).sum::<f64>();
                let losses = outcomes.iter().filter(|&&x| x < ref_point).sum::<f64>();
                (gains, losses)
            },
            probability_weights: probabilities.to_vec(),
            utility_values: utilities.clone(),
            quantum_interference: self.config.quantum_interference_strength,
            decision_value: result.iter().take(4).sum::<f64>() / 4.0,
            loss_aversion_effect: biases.get("loss_aversion").unwrap_or(&2.25).clone(),
            certainty_effect: biases.get("overconfidence").unwrap_or(&0.25).clone(),
        };
        
        {
            let mut history = self.decision_history.write().await;
            history.push(decision_analysis);
            
            if history.len() > 1000 {
                history.remove(0);
            }
        }
        
        Ok(result)
    }
    
    async fn train(&mut self, training_data: &[Vec<f64>]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Train behavioral model on decision data
        let mut all_outcomes = Vec::new();
        
        for data_point in training_data {
            all_outcomes.extend(data_point);
            
            // Update reference point incrementally
            self.update_reference_point(data_point).await?;
        }
        
        // Analyze biases across all training data
        let _biases = self.analyze_behavioral_biases(&all_outcomes).await?;
        
        // Adjust loss aversion based on training data
        if all_outcomes.len() > 10 {
            let ref_point = *self.reference_point.read().await;
            let gains: Vec<f64> = all_outcomes.iter().filter(|&&x| x > ref_point).cloned().collect();
            let losses: Vec<f64> = all_outcomes.iter().filter(|&&x| x < ref_point).cloned().collect();
            
            if !gains.is_empty() && !losses.is_empty() {
                let avg_gain = gains.iter().sum::<f64>() / gains.len() as f64 - ref_point;
                let avg_loss = ref_point - losses.iter().sum::<f64>() / losses.len() as f64;
                
                if avg_gain > 0.0 {
                    let empirical_loss_aversion = avg_loss / avg_gain;
                    // Update loss aversion factor based on empirical data
                    self.config.loss_aversion_factor = (self.config.loss_aversion_factor + empirical_loss_aversion) / 2.0;
                }
            }
        }
        
        Ok(())
    }
    
    fn get_metrics(&self) -> QuantumMetrics {
        self.metrics.clone()
    }
}