//! Advanced Cognitive Architecture for PADS
//! 
//! This module implements sophisticated cognitive patterns harvested from pads-core,
//! including Antifragile patterns, Quantum cognitive states, Temporal recursion,
//! Swarm emergence, and Meta-cognitive reflection systems.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

/// Cognitive node in the decision lattice
#[derive(Debug, Clone)]
pub struct CognitiveNode {
    pub id: u64,
    pub archetype: CognitiveArchetype,
    pub activation: f64,
    pub connections: Vec<u64>,
    pub memory: CircularBuffer<CognitiveMemory>,
    pub last_updated: SystemTime,
}

/// Advanced cognitive patterns beyond simple board members
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CognitiveArchetype {
    /// Original Talebian patterns
    Antifragile { 
        convexity: f64,
        gain_from_disorder: f64,
        volatility_threshold: f64,
        adaptation_rate: f64,
    },
    BlackSwan { 
        tail_sensitivity: f64,
        impact_threshold: f64,
        detection_window: usize,
        confidence_threshold: f64,
    },
    Barbell {
        safe_allocation: f64,
        risk_allocation: f64,
        rebalance_threshold: f64,
        asymmetry_factor: f64,
    },
    
    /// Quantum cognitive patterns
    QuantumSuperposition { 
        states: Vec<QuantumCognitiveState>,
        coherence: f64,
        decoherence_rate: f64,
        measurement_probability: f64,
    },
    QuantumEntangler { 
        entanglement_strength: f64,
        bell_pairs: Vec<(u64, u64)>,
        correlation_matrix: Vec<Vec<f64>>,
        quantum_correlation: f64,
    },
    QuantumTunneler {
        barrier_height: f64,
        tunnel_probability: f64,
        energy_threshold: f64,
        escape_velocity: f64,
    },
    
    /// Temporal patterns
    TemporalRecursion { 
        depth: usize,
        memory_window: usize,
        fractal_dimension: f64,
        time_dilation_factor: f64,
        causal_loops: Vec<CausalLoop>,
    },
    ChronoSeer {
        lookahead_horizons: Vec<u64>,
        temporal_resolution: f64,
        prophetic_accuracy: f64,
        timeline_branches: usize,
    },
    
    /// Swarm patterns
    SwarmEmergence { 
        min_agents: usize,
        consensus_threshold: f64,
        stigmergy_strength: f64,
        emergent_properties: Vec<String>,
        collective_intelligence_factor: f64,
    },
    CollectiveIntelligence {
        shared_memory: Arc<RwLock<SharedMemory>>,
        sync_frequency: f64,
        consensus_algorithm: ConsensusAlgorithm,
        distributed_cognition: bool,
    },
    
    /// Chaos patterns
    ChaosAttractor { 
        lyapunov_exponent: f64,
        strange_attractor_dim: f64,
        butterfly_sensitivity: f64,
        phase_space_dimension: usize,
    },
    FractalAnalyzer { 
        dimensions: Vec<f64>,
        self_similarity: f64,
        hausdorff_dimension: f64,
        lacunarity: f64,
    },
    
    /// Meta-cognitive patterns
    SelfReflection { 
        introspection_depth: usize,
        meta_level: u8,
        consciousness_threshold: f64,
        self_awareness_score: f64,
    },
    Metamorphosis {
        transformation_rate: f64,
        adaptation_speed: f64,
        evolution_pressure: f64,
        morphogenic_field: Vec<f64>,
    },
    
    /// Market-specific patterns
    WhaleDetector {
        volume_threshold: f64,
        pattern_memory: Vec<WhalePattern>,
        detection_sensitivity: f64,
        false_positive_rate: f64,
    },
    LiquidityVacuum {
        vacuum_threshold: f64,
        fill_prediction: f64,
        market_impact_model: MarketImpactModel,
        slippage_factor: f64,
    },
    MomentumSurfer {
        wave_detection: f64,
        ride_duration: f64,
        momentum_decay: f64,
        trend_strength_threshold: f64,
    },
}

/// Quantum cognitive state with advanced properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCognitiveState {
    pub amplitude: f64,
    pub phase: f64,
    pub entangled_with: Option<u64>,
    pub spin: Option<f64>,
    pub polarization: Option<f64>,
    pub energy_level: f64,
}

/// Causal loop for temporal recursion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLoop {
    pub loop_id: String,
    pub cause_node: u64,
    pub effect_node: u64,
    pub delay: Duration,
    pub strength: f64,
    pub confidence: f64,
}

/// Consensus algorithm for collective intelligence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusAlgorithm {
    Byzantine,
    ProofOfStake,
    ProofOfWork,
    DelegatedProofOfStake,
    PracticalByzantine,
    RaftConsensus,
}

/// Market impact model for liquidity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactModel {
    pub linear_impact: f64,
    pub square_root_impact: f64,
    pub temporary_impact_factor: f64,
    pub permanent_impact_factor: f64,
}

/// Circular buffer for cognitive memory
#[derive(Debug, Clone)]
pub struct CircularBuffer<T> {
    buffer: Vec<T>,
    head: usize,
    capacity: usize,
    size: usize,
}

/// Cognitive memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMemory {
    pub timestamp: u64,
    pub pattern: String,
    pub outcome: f64,
    pub confidence: f64,
    pub context: HashMap<String, f64>,
    pub emotional_state: EmotionalState,
    pub cognitive_load: f64,
}

/// Emotional state for cognitive processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub fear: f64,
    pub greed: f64,
    pub confidence: f64,
    pub uncertainty: f64,
    pub excitement: f64,
    pub regret: f64,
}

/// Shared memory for collective intelligence
#[derive(Debug, Clone)]
pub struct SharedMemory {
    pub episodic: HashMap<String, Vec<CognitiveMemory>>,
    pub semantic: HashMap<String, f64>,
    pub procedural: HashMap<String, String>, // Serialized procedures
    pub working_memory: HashMap<String, f64>,
    pub long_term_memory: HashMap<String, Vec<u8>>,
}

/// Whale pattern for market analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhalePattern {
    pub volume_spike: f64,
    pub price_impact: f64,
    pub duration: u64,
    pub frequency: f64,
    pub correlation_with_price: f64,
    pub market_cap_impact: f64,
}

/// Cognitive insights for decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveInsights {
    pub dominant_archetype: String,
    pub archetype_confidence: f64,
    pub cognitive_coherence: f64,
    pub pattern_stability: f64,
    pub emergence_score: f64,
    pub meta_cognitive_score: f64,
    pub temporal_alignment: f64,
    pub quantum_advantage: f64,
}

/// Temporal analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalResults {
    pub prediction_horizon: Duration,
    pub temporal_confidence: f64,
    pub fractal_analysis: FractalAnalysis,
    pub causal_chains: Vec<CausalChain>,
    pub timeline_coherence: f64,
    pub recursive_depth_used: usize,
}

/// Fractal analysis for temporal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalAnalysis {
    pub hurst_exponent: f64,
    pub box_counting_dimension: f64,
    pub correlation_dimension: f64,
    pub self_similarity_score: f64,
    pub scaling_exponent: f64,
}

/// Causal chain for temporal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalChain {
    pub chain_id: String,
    pub events: Vec<CausalEvent>,
    pub probability: f64,
    pub impact_magnitude: f64,
    pub temporal_span: Duration,
}

/// Causal event in temporal chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEvent {
    pub event_id: String,
    pub timestamp: SystemTime,
    pub event_type: String,
    pub probability: f64,
    pub impact: f64,
    pub dependencies: Vec<String>,
}

/// Meta-cognitive assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaAssessment {
    pub self_awareness_level: f64,
    pub reflection_depth: usize,
    pub cognitive_flexibility: f64,
    pub pattern_recognition_accuracy: f64,
    pub adaptation_efficiency: f64,
    pub meta_learning_rate: f64,
    pub consciousness_score: f64,
}

/// Feedback for cognitive evolution
#[derive(Debug, Clone)]
pub struct Feedback {
    pub performance: f64,
    pub performance_threshold: f64,
    pub stability_factor: f64,
    pub decision_quality: f64,
    pub coordination_score: f64,
    pub prediction_accuracy: f64,
    pub market_complexity: f64,
    pub volatility_survived: f64,
    pub adaptation_success: f64,
    pub learning_efficiency: f64,
}

impl CognitiveArchetype {
    /// Evolve archetype based on performance feedback with advanced adaptation
    pub fn evolve(&mut self, feedback: &Feedback) {
        match self {
            Self::QuantumSuperposition { states, coherence, decoherence_rate, .. } => {
                // Collapse poorly performing states
                states.retain(|s| s.amplitude > feedback.performance_threshold);
                
                // Adjust coherence based on stability and performance
                *coherence *= feedback.stability_factor() * (1.0 + feedback.performance * 0.1);
                *decoherence_rate *= 1.0 - feedback.adaptation_success * 0.1;
                
                // Add new states if performing exceptionally well
                if feedback.performance > 0.9 && states.len() < 15 {
                    states.push(QuantumCognitiveState {
                        amplitude: 0.1 + feedback.performance * 0.1,
                        phase: rand::random::<f64>() * 2.0 * std::f64::consts::PI,
                        entangled_with: None,
                        spin: Some(rand::random::<f64>() - 0.5),
                        polarization: Some(rand::random::<f64>()),
                        energy_level: feedback.performance,
                    });
                }
            }
            
            Self::Antifragile { convexity, gain_from_disorder, adaptation_rate, .. } => {
                // Increase gain from disorder after surviving volatility
                if feedback.volatility_survived > 0.8 {
                    *gain_from_disorder *= 1.0 + feedback.volatility_survived * 0.2;
                    *convexity = (*convexity * 1.1).min(5.0);
                    *adaptation_rate = (*adaptation_rate * 1.05).min(1.0);
                }
                
                // Enhance antifragility based on market complexity
                let complexity_bonus = feedback.market_complexity * 0.1;
                *gain_from_disorder += complexity_bonus;
            }
            
            Self::SwarmEmergence { consensus_threshold, stigmergy_strength, collective_intelligence_factor, .. } => {
                // Adjust consensus based on decision quality
                if feedback.decision_quality > 0.95 {
                    *consensus_threshold *= 0.9; // Lower threshold for excellent decisions
                } else if feedback.decision_quality < 0.6 {
                    *consensus_threshold *= 1.1; // Raise threshold for poor decisions
                }
                
                // Strengthen stigmergy for coordinated success
                if feedback.coordination_score > 0.85 {
                    *stigmergy_strength = (*stigmergy_strength * 1.15).min(1.0);
                    *collective_intelligence_factor *= 1.1;
                }
            }
            
            Self::TemporalRecursion { depth, fractal_dimension, time_dilation_factor, .. } => {
                // Adjust recursion depth based on prediction accuracy
                if feedback.prediction_accuracy > 0.9 && *depth < 15 {
                    *depth += 1;
                } else if feedback.prediction_accuracy < 0.6 && *depth > 3 {
                    *depth -= 1;
                }
                
                // Update fractal dimension based on market complexity
                *fractal_dimension = feedback.market_complexity * 1.5;
                *time_dilation_factor *= 1.0 + feedback.learning_efficiency * 0.1;
            }
            
            Self::SelfReflection { introspection_depth, consciousness_threshold, self_awareness_score, .. } => {
                // Deepen introspection for better performance
                if feedback.performance > 0.85 && *introspection_depth < 10 {
                    *introspection_depth += 1;
                }
                
                *consciousness_threshold *= feedback.stability_factor();
                *self_awareness_score = (*self_awareness_score + feedback.performance) / 2.0;
            }
            
            _ => {
                // Default evolution for other archetypes
                self.default_evolution(feedback);
            }
        }
    }
    
    /// Default evolution strategy for all archetypes
    fn default_evolution(&mut self, feedback: &Feedback) {
        // Generic adaptation based on performance and learning efficiency
        // Each archetype can override this for specific behavior
    }
    
    /// Compute activation based on market state with advanced logic
    pub fn compute_activation(&self, market: &MarketState) -> f64 {
        match self {
            Self::BlackSwan { tail_sensitivity, impact_threshold, confidence_threshold, .. } => {
                let tail_event_probability = Self::detect_tail_event(market);
                let confidence_factor = if tail_event_probability > *confidence_threshold { 1.2 } else { 1.0 };
                
                if tail_event_probability > *impact_threshold {
                    1.0 * confidence_factor // Maximum activation for black swan events
                } else {
                    tail_event_probability * tail_sensitivity * confidence_factor
                }
            }
            
            Self::MomentumSurfer { wave_detection, momentum_decay, trend_strength_threshold, .. } => {
                let momentum_strength = market.momentum.abs();
                let trend_strength = market.trend.abs();
                
                if momentum_strength > *wave_detection && trend_strength > *trend_strength_threshold {
                    let decay_factor = 1.0 - momentum_decay;
                    (momentum_strength * trend_strength * decay_factor).min(1.0)
                } else {
                    0.0
                }
            }
            
            Self::ChaosAttractor { lyapunov_exponent, butterfly_sensitivity, .. } => {
                // Activate in chaotic market conditions
                let chaos_metric = Self::compute_chaos_metric(market);
                let butterfly_effect = Self::compute_butterfly_effect(market, *butterfly_sensitivity);
                
                (chaos_metric * lyapunov_exponent + butterfly_effect).tanh()
            }
            
            Self::QuantumSuperposition { states, coherence, decoherence_rate, .. } => {
                // Superposition of all states with decoherence
                let total_amplitude: f64 = states.iter()
                    .map(|s| s.amplitude * s.amplitude * s.energy_level)
                    .sum();
                
                let decoherence_factor = 1.0 - decoherence_rate;
                total_amplitude.sqrt() * coherence * decoherence_factor
            }
            
            Self::Antifragile { gain_from_disorder, volatility_threshold, .. } => {
                // Activate more strongly during volatile periods
                if market.volatility > *volatility_threshold {
                    let volatility_bonus = (market.volatility - volatility_threshold) * 2.0;
                    (gain_from_disorder + volatility_bonus).min(1.0)
                } else {
                    gain_from_disorder * 0.5
                }
            }
            
            _ => 0.5, // Default activation for other patterns
        }
    }
    
    /// Detect tail events in market data with enhanced sensitivity
    fn detect_tail_event(market: &MarketState) -> f64 {
        // Multi-factor tail event detection
        let volatility_z = (market.volatility - 0.2) / 0.1;
        let volume_z = (market.volume - 1000.0) / 500.0;
        let price_z = (market.price - 100.0) / 20.0;
        
        let combined_z = (volatility_z.powi(2) + volume_z.powi(2) + price_z.powi(2)).sqrt();
        
        if combined_z > 3.0 {
            1.0 / (1.0 + (-combined_z).exp())
        } else {
            0.0
        }
    }
    
    /// Compute chaos metric from market state
    fn compute_chaos_metric(market: &MarketState) -> f64 {
        // Enhanced chaos detection
        let trend_volatility_divergence = (market.trend - market.volatility).abs();
        let price_volume_decorrelation = 1.0 - (market.price * market.volume).ln().abs() / 100.0;
        let momentum_instability = market.momentum.abs() * market.volatility;
        
        (trend_volatility_divergence + price_volume_decorrelation + momentum_instability) / 3.0
    }
    
    /// Compute butterfly effect sensitivity
    fn compute_butterfly_effect(market: &MarketState, sensitivity: f64) -> f64 {
        // Small changes that could lead to large effects
        let micro_changes = (market.price % 1.0) * (market.volume % 100.0) / 100.0;
        micro_changes * sensitivity
    }
}

impl Feedback {
    pub fn stability_factor(&self) -> f64 {
        1.0 + (self.stability_factor - 0.5) * 0.2
    }
    
    /// Create comprehensive feedback from market performance
    pub fn from_performance(
        performance: f64,
        market_state: &MarketState,
        prediction_accuracy: f64,
        coordination_score: f64,
    ) -> Self {
        Self {
            performance,
            performance_threshold: 0.7,
            stability_factor: 0.8,
            decision_quality: performance * 0.9,
            coordination_score,
            prediction_accuracy,
            market_complexity: market_state.volatility * 2.0 + market_state.trend.abs(),
            volatility_survived: if market_state.volatility > 0.3 { performance } else { 1.0 },
            adaptation_success: performance * coordination_score,
            learning_efficiency: prediction_accuracy * 0.8,
        }
    }
}

impl<T: Clone> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            head: 0,
            capacity,
            size: 0,
        }
    }
    
    pub fn push(&mut self, item: T) {
        if self.size < self.capacity {
            self.buffer.push(item);
            self.size += 1;
        } else {
            self.buffer[self.head] = item;
            self.head = (self.head + 1) % self.capacity;
        }
    }
    
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buffer.iter()
    }
    
    pub fn len(&self) -> usize {
        self.size
    }
    
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl CognitiveNode {
    pub fn new(id: u64, archetype: CognitiveArchetype) -> Self {
        Self {
            id,
            archetype,
            activation: 0.0,
            connections: Vec::new(),
            memory: CircularBuffer::new(1000),
            last_updated: SystemTime::now(),
        }
    }
    
    pub fn add_connection(&mut self, other_id: u64) {
        if !self.connections.contains(&other_id) {
            self.connections.push(other_id);
        }
    }
    
    pub fn update_activation(&mut self, market: &MarketState) {
        self.activation = self.archetype.compute_activation(market);
        self.last_updated = SystemTime::now();
    }
    
    pub fn remember(&mut self, memory: CognitiveMemory) {
        self.memory.push(memory);
    }
    
    pub fn get_recent_memories(&self, count: usize) -> Vec<&CognitiveMemory> {
        self.memory.iter().rev().take(count).collect()
    }
    
    pub fn cognitive_load(&self) -> f64 {
        let memory_load = self.memory.len() as f64 / self.memory.capacity() as f64;
        let connection_load = self.connections.len() as f64 / 50.0; // Assume max 50 connections
        let activation_load = self.activation;
        
        (memory_load + connection_load + activation_load) / 3.0
    }
}

impl SharedMemory {
    pub fn new() -> Self {
        Self {
            episodic: HashMap::new(),
            semantic: HashMap::new(),
            procedural: HashMap::new(),
            working_memory: HashMap::new(),
            long_term_memory: HashMap::new(),
        }
    }
    
    pub fn store_episodic(&mut self, key: String, memory: CognitiveMemory) {
        self.episodic.entry(key).or_insert_with(Vec::new).push(memory);
    }
    
    pub fn store_semantic(&mut self, key: String, value: f64) {
        self.semantic.insert(key, value);
    }
    
    pub fn store_procedural(&mut self, key: String, procedure: String) {
        self.procedural.insert(key, procedure);
    }
    
    pub fn get_episodic(&self, key: &str) -> Option<&Vec<CognitiveMemory>> {
        self.episodic.get(key)
    }
    
    pub fn get_semantic(&self, key: &str) -> Option<&f64> {
        self.semantic.get(key)
    }
    
    pub fn get_procedural(&self, key: &str) -> Option<&String> {
        self.procedural.get(key)
    }
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self {
            fear: 0.3,
            greed: 0.3,
            confidence: 0.5,
            uncertainty: 0.4,
            excitement: 0.2,
            regret: 0.1,
        }
    }
}

impl EmotionalState {
    pub fn emotional_balance(&self) -> f64 {
        let positive = self.confidence + self.excitement;
        let negative = self.fear + self.greed + self.uncertainty + self.regret;
        
        (positive - negative + 2.0) / 4.0 // Normalize to 0-1 range
    }
    
    pub fn update_from_outcome(&mut self, predicted: f64, actual: f64, confidence: f64) {
        let error = (predicted - actual).abs();
        
        if error < 0.1 && confidence > 0.8 {
            // Good prediction
            self.confidence = (self.confidence * 0.9 + 0.9).min(1.0);
            self.excitement = (self.excitement * 0.8 + 0.3).min(1.0);
            self.regret *= 0.9;
        } else {
            // Poor prediction
            self.confidence *= 0.9;
            self.uncertainty = (self.uncertainty * 0.8 + error).min(1.0);
            self.regret = (self.regret * 0.8 + error * 0.5).min(1.0);
        }
    }
}

// Market state definition for compatibility
use crate::types::MarketState;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cognitive_node_creation() {
        let archetype = CognitiveArchetype::Antifragile {
            convexity: 1.5,
            gain_from_disorder: 0.8,
            volatility_threshold: 0.3,
            adaptation_rate: 0.1,
        };
        
        let node = CognitiveNode::new(1, archetype);
        assert_eq!(node.id, 1);
        assert_eq!(node.activation, 0.0);
        assert!(node.connections.is_empty());
    }
    
    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(3);
        
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        buffer.push(4); // Should overwrite 1
        
        assert_eq!(buffer.len(), 3);
        let items: Vec<_> = buffer.iter().cloned().collect();
        assert_eq!(items, vec![4, 2, 3]);
    }
    
    #[test]
    fn test_archetype_evolution() {
        let mut archetype = CognitiveArchetype::QuantumSuperposition {
            states: vec![
                QuantumCognitiveState {
                    amplitude: 0.7,
                    phase: 0.0,
                    entangled_with: None,
                    spin: Some(0.5),
                    polarization: Some(0.8),
                    energy_level: 0.9,
                },
            ],
            coherence: 0.9,
            decoherence_rate: 0.1,
            measurement_probability: 0.5,
        };
        
        let feedback = Feedback {
            performance: 0.95,
            performance_threshold: 0.5,
            stability_factor: 0.8,
            decision_quality: 0.9,
            coordination_score: 0.85,
            prediction_accuracy: 0.9,
            market_complexity: 1.4,
            volatility_survived: 0.9,
            adaptation_success: 0.8,
            learning_efficiency: 0.85,
        };
        
        archetype.evolve(&feedback);
        
        if let CognitiveArchetype::QuantumSuperposition { states, coherence, .. } = archetype {
            assert_eq!(states.len(), 2); // Should add a new state
            assert!(coherence > 0.9); // Coherence should increase
        }
    }
    
    #[test]
    fn test_emotional_state_update() {
        let mut emotional_state = EmotionalState::default();
        
        // Simulate good prediction
        emotional_state.update_from_outcome(0.8, 0.85, 0.9);
        
        assert!(emotional_state.confidence > 0.5);
        assert!(emotional_state.regret < 0.1);
        
        // Simulate bad prediction
        emotional_state.update_from_outcome(0.8, 0.2, 0.9);
        
        assert!(emotional_state.uncertainty > 0.4);
        assert!(emotional_state.regret > 0.1);
    }
    
    #[test]
    fn test_shared_memory() {
        let mut memory = SharedMemory::new();
        
        let cognitive_memory = CognitiveMemory {
            timestamp: 12345,
            pattern: "test_pattern".to_string(),
            outcome: 0.8,
            confidence: 0.9,
            context: HashMap::new(),
            emotional_state: EmotionalState::default(),
            cognitive_load: 0.5,
        };
        
        memory.store_episodic("test_episode".to_string(), cognitive_memory);
        memory.store_semantic("test_concept".to_string(), 0.75);
        
        assert!(memory.get_episodic("test_episode").is_some());
        assert_eq!(memory.get_semantic("test_concept"), Some(&0.75));
    }
}