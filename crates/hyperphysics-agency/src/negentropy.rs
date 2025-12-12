//! # Negentropy-Driven Pedagogic Agency Framework
//!
//! Implements a sophisticated negentropy-based awareness system grounded in:
//! - **Thermodynamics**: Negentropy as the inverse of entropy (N = S_max - S_actual)
//! - **Bateson's Learning Levels**: L0 (reflexes) → L4 (evolutionary change)
//! - **Self-Determination Theory**: Autonomy, competence, relatedness
//! - **Brain-Inspired Architecture**: PFC, ACC, Insula, Basal Ganglia mappings
//!
//! ## Core Thesis: Graceful Awareness over Punishment
//!
//! Rather than crude reward/penalty mechanisms, this framework uses
//! *pedagogic scaffolding* inspired by:
//! - Chomsky's generative linguistics (innate capacity for structure)
//! - Living Labs (hands-on experiential learning)
//! - Art of Hosting (participatory leadership)
//! - Design Thinking (empathy-driven iteration)
//! - Systems Thinking (feedback loops, emergence)
//!
//! ## 50% Negentropy Threshold
//!
//! The critical threshold N = 0.5 serves as an *awareness trigger*:
//! - N < 0.5: Agent is "lazy" → Pedagogic scaffolding activates
//! - N ≥ 0.5: Agent is "alive" → Autonomous operation
//! - N → 0.9: Maximally alive → L3 transformation possible
//! - N → 1.0: Transcendent → L4 evolutionary change possible
//!
//! ## Bateson's Extended Learning Hierarchy
//!
//! - **L0**: Reflexes (hardwired stimulus-response)
//! - **L1**: Conditioning (habit formation, associative learning)
//! - **L2**: Meta-learning (learning to learn, context switching)
//! - **L3**: Transformation (paradigm shifts, self-transcendence)
//! - **L4**: Evolution (systemic change, genetic algorithms, population learning)
//!
//! ## References
//!
//! - Schrödinger, E. (1944). "What is Life?" (negentropy concept)
//! - Bateson, G. (1972). "Steps to an Ecology of Mind"
//! - Friston, K. (2010). "Free energy principle"
//! - Ryan & Deci (2000). "Self-Determination Theory"
//! - Albaraccin et al. "Active Inference Agents"
//! - Holland, J. (1975). "Adaptation in Natural and Artificial Systems" (L4)

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ============================================================================
// Bateson's Learning Levels
// ============================================================================

/// Gregory Bateson's hierarchical learning levels (extended to L4)
///
/// - **L0**: Reflexive responses, no learning
/// - **L1**: Classical/operant conditioning, habit formation
/// - **L2**: Meta-learning, learning to learn, context recognition
/// - **L3**: Transformative learning, paradigm shifts, self-transcendence
/// - **L4**: Evolutionary learning, systemic change, population-level adaptation
///
/// ## L4: Evolutionary Change (Holland, 1975)
///
/// L4 represents changes that transcend individual agents:
/// - **Genetic Algorithms**: Population-wide fitness-based selection
/// - **Memetic Evolution**: Cultural transmission and mutation of ideas
/// - **Phylogenetic Learning**: Changes encoded in agent "DNA" (architecture)
/// - **Swarm Emergent Intelligence**: Collective adaptation beyond individual scope
///
/// L4 is "learning that changes the rules of learning itself" at a systemic level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatesonLevel {
    /// L0: Pure stimulus-response, no adaptation
    L0Reflex,
    /// L1: Conditioning, habit formation through repetition
    L1Conditioning,
    /// L2: Meta-learning, learning to learn, context switching
    L2MetaLearning,
    /// L3: Transformative learning, paradigm shifts
    L3Transformation,
    /// L4: Evolutionary learning, systemic change across populations
    /// Requires sustained L3, population interaction, and fitness pressure
    L4Evolution,
}

impl BatesonLevel {
    /// Get numeric level (0-4)
    pub fn level(&self) -> u8 {
        match self {
            Self::L0Reflex => 0,
            Self::L1Conditioning => 1,
            Self::L2MetaLearning => 2,
            Self::L3Transformation => 3,
            Self::L4Evolution => 4,
        }
    }

    /// Create from numeric level
    pub fn from_level(level: u8) -> Self {
        match level {
            0 => Self::L0Reflex,
            1 => Self::L1Conditioning,
            2 => Self::L2MetaLearning,
            3 => Self::L3Transformation,
            _ => Self::L4Evolution,
        }
    }

    /// Energy required to maintain this level
    pub fn energy_requirement(&self) -> f64 {
        match self {
            Self::L0Reflex => 0.1,
            Self::L1Conditioning => 0.3,
            Self::L2MetaLearning => 0.5,
            Self::L3Transformation => 0.8,
            Self::L4Evolution => 0.95, // Requires near-maximum energy
        }
    }

    /// Coherence threshold for this level
    pub fn coherence_threshold(&self) -> f64 {
        match self {
            Self::L0Reflex => 0.0,
            Self::L1Conditioning => 0.3,
            Self::L2MetaLearning => 0.5,
            Self::L3Transformation => 0.7,
            Self::L4Evolution => 0.9, // Requires very high coherence
        }
    }

    /// Population size requirement for this level
    /// L4 requires interaction with other agents to enable evolutionary dynamics
    pub fn population_requirement(&self) -> usize {
        match self {
            Self::L0Reflex => 1,
            Self::L1Conditioning => 1,
            Self::L2MetaLearning => 1,
            Self::L3Transformation => 1,
            Self::L4Evolution => 3, // Minimum population for evolutionary pressure
        }
    }

    /// Fitness pressure required for evolutionary learning
    /// Higher values indicate stronger selection pressure
    pub fn fitness_pressure(&self) -> f64 {
        match self {
            Self::L0Reflex => 0.0,
            Self::L1Conditioning => 0.0,
            Self::L2MetaLearning => 0.0,
            Self::L3Transformation => 0.0,
            Self::L4Evolution => 0.5, // Moderate selection pressure
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::L0Reflex => "Reflexes: Hardwired stimulus-response patterns",
            Self::L1Conditioning => "Conditioning: Habit formation through association",
            Self::L2MetaLearning => "Meta-learning: Learning to learn, context switching",
            Self::L3Transformation => "Transformation: Paradigm shifts, self-transcendence",
            Self::L4Evolution => "Evolution: Systemic change, population-level adaptation",
        }
    }

    /// Whether this level requires population interaction
    pub fn requires_population(&self) -> bool {
        matches!(self, Self::L4Evolution)
    }

    /// Duration (in steps) required to stabilize at this level before advancing
    pub fn stabilization_period(&self) -> u64 {
        match self {
            Self::L0Reflex => 0,
            Self::L1Conditioning => 100,
            Self::L2MetaLearning => 500,
            Self::L3Transformation => 1000,
            Self::L4Evolution => 5000, // Long stabilization for evolutionary change
        }
    }
}

impl Default for BatesonLevel {
    fn default() -> Self {
        Self::L0Reflex
    }
}

// ============================================================================
// Brain Structure Mappings
// ============================================================================

/// Brain-inspired cognitive regulation modules
///
/// Maps neuroscience to algorithmic components for self-referential regulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveRegulator {
    /// Prefrontal Cortex: Executive control, planning, working memory
    /// Responsible for: goal maintenance, action selection, inhibition
    pub prefrontal_cortex: PrefrontalCortex,

    /// Anterior Cingulate Cortex: Conflict monitoring, error detection
    /// Responsible for: surprise detection, effort allocation
    pub anterior_cingulate: AnteriorCingulate,

    /// Insula: Interoception, awareness of internal states
    /// Responsible for: body state monitoring, emotional awareness
    pub insula: Insula,

    /// Basal Ganglia: Habit formation, action selection
    /// Responsible for: procedural learning, reward processing
    pub basal_ganglia: BasalGanglia,

    /// Hippocampus: Memory consolidation, spatial/contextual learning
    /// Responsible for: episodic memory, contextual binding
    pub hippocampus: Hippocampus,
}

impl Default for CognitiveRegulator {
    fn default() -> Self {
        Self {
            prefrontal_cortex: PrefrontalCortex::default(),
            anterior_cingulate: AnteriorCingulate::default(),
            insula: Insula::default(),
            basal_ganglia: BasalGanglia::default(),
            hippocampus: Hippocampus::default(),
        }
    }
}

/// Prefrontal Cortex - Executive Control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefrontalCortex {
    /// Working memory capacity (Miller's 7±2)
    pub working_memory_capacity: usize,
    /// Current working memory contents
    pub working_memory: VecDeque<f64>,
    /// Goal state (desired outcome)
    pub goal_state: Array1<f64>,
    /// Inhibition strength (ability to suppress impulses)
    pub inhibition_strength: f64,
    /// Planning horizon (steps ahead)
    pub planning_horizon: usize,
}

impl Default for PrefrontalCortex {
    fn default() -> Self {
        Self {
            working_memory_capacity: 7,
            working_memory: VecDeque::with_capacity(7),
            goal_state: Array1::zeros(16),
            inhibition_strength: 0.5,
            planning_horizon: 5,
        }
    }
}

impl PrefrontalCortex {
    /// Update working memory with new item
    pub fn update_working_memory(&mut self, item: f64) {
        if self.working_memory.len() >= self.working_memory_capacity {
            self.working_memory.pop_front();
        }
        self.working_memory.push_back(item);
    }

    /// Compute executive control signal
    /// Higher when goal-state mismatch is large and inhibition is needed
    pub fn executive_control(&self, current_state: &Array1<f64>) -> f64 {
        let goal_error = self.goal_state.iter()
            .zip(current_state.iter())
            .map(|(g, c)| (g - c).powi(2))
            .sum::<f64>()
            .sqrt();

        // Executive control increases with goal-state error
        (goal_error * self.inhibition_strength).tanh()
    }

    /// Set goal state for planning
    pub fn set_goal(&mut self, goal: Array1<f64>) {
        self.goal_state = goal;
    }
}

/// Anterior Cingulate Cortex - Conflict Monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnteriorCingulate {
    /// Conflict history for error detection
    conflict_history: VecDeque<f64>,
    /// Effort allocation level
    pub effort_allocation: f64,
    /// Surprise sensitivity (how reactive to unexpected events)
    pub surprise_sensitivity: f64,
    /// Error detection threshold
    pub error_threshold: f64,
}

impl Default for AnteriorCingulate {
    fn default() -> Self {
        Self {
            conflict_history: VecDeque::with_capacity(50),
            effort_allocation: 0.5,
            surprise_sensitivity: 1.0,
            error_threshold: 0.3,
        }
    }
}

impl AnteriorCingulate {
    /// Process prediction error and compute conflict signal
    pub fn process_error(&mut self, prediction_error: f64) -> f64 {
        // Record error
        self.conflict_history.push_back(prediction_error.abs());
        if self.conflict_history.len() > 50 {
            self.conflict_history.pop_front();
        }

        // Conflict signal = surprise-weighted error
        let surprise = self.surprise_sensitivity * prediction_error.abs();
        let conflict = if surprise > self.error_threshold {
            surprise
        } else {
            0.0
        };

        // Adjust effort allocation based on conflict
        self.effort_allocation = 0.9 * self.effort_allocation + 0.1 * conflict;
        self.effort_allocation = self.effort_allocation.clamp(0.1, 1.0);

        conflict
    }

    /// Get mean conflict level (for awareness triggering)
    pub fn mean_conflict(&self) -> f64 {
        if self.conflict_history.is_empty() {
            return 0.0;
        }
        self.conflict_history.iter().sum::<f64>() / self.conflict_history.len() as f64
    }
}

/// Insula - Interoception and Bodily Awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insula {
    /// Internal state estimate (homeostatic variables)
    pub internal_state: Array1<f64>,
    /// Interoceptive precision (confidence in body signals)
    pub interoceptive_precision: f64,
    /// Arousal level (physiological activation)
    pub arousal: f64,
    /// Valence (positive/negative affect)
    pub valence: f64,
}

impl Default for Insula {
    fn default() -> Self {
        Self {
            internal_state: Array1::from_elem(8, 0.5),
            interoceptive_precision: 1.0,
            arousal: 0.5,
            valence: 0.0,
        }
    }
}

impl Insula {
    /// Update interoceptive state based on sensor readings
    pub fn update_interoception(&mut self, sensors: &[f64]) {
        for (i, &s) in sensors.iter().enumerate() {
            if i < self.internal_state.len() {
                // Precision-weighted update
                let delta = s - self.internal_state[i];
                self.internal_state[i] += self.interoceptive_precision * 0.1 * delta;
            }
        }

        // Update arousal (mean activation)
        self.arousal = self.internal_state.iter()
            .map(|x| x.abs())
            .sum::<f64>() / self.internal_state.len() as f64;

        // Update valence (deviation from setpoint 0.5)
        self.valence = self.internal_state.iter()
            .map(|x| x - 0.5)
            .sum::<f64>() / self.internal_state.len() as f64;
    }

    /// Compute bodily awareness signal (high when internal states deviate)
    pub fn bodily_awareness(&self) -> f64 {
        let deviation = self.internal_state.iter()
            .map(|x| (x - 0.5).powi(2))
            .sum::<f64>()
            .sqrt();

        (deviation * self.interoceptive_precision).tanh()
    }
}

/// Basal Ganglia - Habit Formation and Action Selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasalGanglia {
    /// Action-value estimates (Q-values)
    action_values: Array1<f64>,
    /// Learning rate for habit formation
    pub learning_rate: f64,
    /// Exploration rate (ε in ε-greedy)
    pub exploration_rate: f64,
    /// Habit strength (how automated actions are)
    pub habit_strength: f64,
}

impl Default for BasalGanglia {
    fn default() -> Self {
        Self {
            action_values: Array1::zeros(16),
            learning_rate: 0.1,
            exploration_rate: 0.2,
            habit_strength: 0.0,
        }
    }
}

impl BasalGanglia {
    /// Update action values with reward signal
    pub fn update(&mut self, action_idx: usize, reward: f64) {
        if action_idx < self.action_values.len() {
            // TD-like update
            let delta = reward - self.action_values[action_idx];
            self.action_values[action_idx] += self.learning_rate * delta;

            // Strengthen habits over time
            self.habit_strength = 0.99 * self.habit_strength + 0.01 * (1.0 - delta.abs());
        }
    }

    /// Select action (returns index)
    pub fn select_action(&self) -> usize {
        // ε-greedy with habit modulation
        let effective_exploration = self.exploration_rate * (1.0 - self.habit_strength);

        if rand_uniform() < effective_exploration {
            // Random exploration
            (rand_uniform() * self.action_values.len() as f64) as usize
        } else {
            // Greedy selection
            self.action_values.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    /// Get habit automation level (0 = deliberate, 1 = automatic)
    pub fn automation_level(&self) -> f64 {
        self.habit_strength
    }
}

/// Hippocampus - Episodic Memory and Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hippocampus {
    /// Episodic memory buffer
    episodes: VecDeque<Episode>,
    /// Maximum episodes to retain
    max_episodes: usize,
    /// Current context embedding
    context: Array1<f64>,
    /// Memory consolidation rate
    pub consolidation_rate: f64,
}

/// An episodic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Context at encoding
    pub context: Array1<f64>,
    /// State at encoding
    pub state: Array1<f64>,
    /// Salience (importance)
    pub salience: f64,
    /// Timestamp
    pub timestamp: u64,
}

impl Default for Hippocampus {
    fn default() -> Self {
        Self {
            episodes: VecDeque::with_capacity(100),
            max_episodes: 100,
            context: Array1::zeros(16),
            consolidation_rate: 0.1,
        }
    }
}

impl Hippocampus {
    /// Encode new episode
    pub fn encode(&mut self, state: &Array1<f64>, salience: f64, timestamp: u64) {
        let episode = Episode {
            context: self.context.clone(),
            state: state.clone(),
            salience,
            timestamp,
        };

        self.episodes.push_back(episode);
        if self.episodes.len() > self.max_episodes {
            // Remove least salient episode
            if let Some((idx, _)) = self.episodes.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.salience.partial_cmp(&b.salience).unwrap())
            {
                self.episodes.remove(idx);
            }
        }
    }

    /// Retrieve most similar episode to current state
    pub fn retrieve(&self, query: &Array1<f64>) -> Option<&Episode> {
        self.episodes.iter()
            .max_by(|a, b| {
                let sim_a = cosine_similarity(&a.state, query);
                let sim_b = cosine_similarity(&b.state, query);
                sim_a.partial_cmp(&sim_b).unwrap()
            })
    }

    /// Update context based on current state
    pub fn update_context(&mut self, state: &Array1<f64>) {
        for i in 0..self.context.len().min(state.len()) {
            self.context[i] = 0.9 * self.context[i] + 0.1 * state[i];
        }
    }

    /// Get number of episodes
    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }
}

// ============================================================================
// Pedagogic Scaffolding
// ============================================================================

/// Pedagogic scaffolding mode (from educational theory)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaffoldMode {
    /// Passive observation, minimal intervention
    Observation,
    /// Gentle nudging through curiosity activation
    CuriosityNudge,
    /// Guided exploration with hints
    GuidedExploration,
    /// Direct instruction when needed
    DirectInstruction,
    /// Collaborative dialogue (Art of Hosting)
    CollaborativeDialogue,
    /// Full autonomous operation
    Autonomous,
}

impl ScaffoldMode {
    /// Get intervention intensity (0 = none, 1 = maximum)
    pub fn intensity(&self) -> f64 {
        match self {
            Self::Observation => 0.0,
            Self::CuriosityNudge => 0.2,
            Self::GuidedExploration => 0.4,
            Self::DirectInstruction => 0.7,
            Self::CollaborativeDialogue => 0.5,
            Self::Autonomous => 0.0,
        }
    }
}

/// Pedagogic scaffold for graceful awareness triggering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PedagogicScaffold {
    /// Current scaffolding mode
    pub mode: ScaffoldMode,
    /// Curiosity drive level (intrinsic motivation)
    pub curiosity: f64,
    /// Autonomy support (vs. controlling)
    pub autonomy_support: f64,
    /// Competence feedback (informational, not evaluative)
    pub competence_feedback: f64,
    /// Relatedness (connection to learning community)
    pub relatedness: f64,
    /// Zone of Proximal Development width
    pub zpd_width: f64,
    /// Scaffolding history
    intervention_history: VecDeque<f64>,
}

impl Default for PedagogicScaffold {
    fn default() -> Self {
        Self {
            mode: ScaffoldMode::Observation,
            curiosity: 0.5,
            autonomy_support: 0.8,
            competence_feedback: 0.5,
            relatedness: 0.5,
            zpd_width: 0.3,
            intervention_history: VecDeque::with_capacity(100),
        }
    }
}

impl PedagogicScaffold {
    /// Select scaffolding mode based on negentropy level
    pub fn select_mode(&mut self, negentropy: f64, learning_level: BatesonLevel) -> ScaffoldMode {
        // Graceful mode selection based on negentropy threshold
        self.mode = if negentropy < 0.2 {
            // Very low negentropy: Direct instruction needed
            ScaffoldMode::DirectInstruction
        } else if negentropy < 0.35 {
            // Low negentropy: Guided exploration
            ScaffoldMode::GuidedExploration
        } else if negentropy < 0.5 {
            // Below threshold: Curiosity nudge
            ScaffoldMode::CuriosityNudge
        } else if learning_level == BatesonLevel::L2MetaLearning {
            // L2 learner above threshold: Collaborative dialogue
            ScaffoldMode::CollaborativeDialogue
        } else if negentropy >= 0.7 {
            // High negentropy: Full autonomy
            ScaffoldMode::Autonomous
        } else {
            // Normal range: Observation
            ScaffoldMode::Observation
        };

        self.mode
    }

    /// Compute intervention signal (not punishment, but scaffolding)
    ///
    /// This is the core "graceful awareness triggering" mechanism:
    /// - Low intervention = autonomous exploration
    /// - High intervention = supportive scaffolding
    pub fn compute_intervention(&mut self, negentropy: f64, coherence: f64) -> f64 {
        let base_intervention = self.mode.intensity();

        // Modulate by curiosity (high curiosity = less intervention needed)
        let curiosity_factor = 1.0 - self.curiosity * 0.5;

        // Modulate by autonomy support (high support = less controlling)
        let autonomy_factor = 1.0 - self.autonomy_support * 0.3;

        // Final intervention
        let intervention = base_intervention * curiosity_factor * autonomy_factor;

        // Record for analysis
        self.intervention_history.push_back(intervention);
        if self.intervention_history.len() > 100 {
            self.intervention_history.pop_front();
        }

        intervention
    }

    /// Generate curiosity boost (intrinsic motivation signal)
    ///
    /// Based on epistemic foraging: moderate uncertainty = maximum curiosity
    pub fn curiosity_boost(&self, uncertainty: f64) -> f64 {
        // Inverted U-curve: curiosity peaks at moderate uncertainty
        // Wundt curve: λ(u) = 4u(1-u) for u ∈ [0,1]
        let u = uncertainty.clamp(0.0, 1.0);
        4.0 * u * (1.0 - u) * self.curiosity
    }

    /// Update Self-Determination Theory components
    pub fn update_sdt(&mut self, autonomy: f64, competence: f64, relatedness: f64) {
        // Smooth updates
        self.autonomy_support = 0.9 * self.autonomy_support + 0.1 * autonomy;
        self.competence_feedback = 0.9 * self.competence_feedback + 0.1 * competence;
        self.relatedness = 0.9 * self.relatedness + 0.1 * relatedness;

        // Curiosity emerges from SDT satisfaction
        let sdt_satisfaction = (autonomy + competence + relatedness) / 3.0;
        self.curiosity = 0.8 * self.curiosity + 0.2 * sdt_satisfaction;
    }

    /// Get intrinsic motivation level (from SDT)
    pub fn intrinsic_motivation(&self) -> f64 {
        // Intrinsic motivation = f(autonomy, competence, relatedness)
        let autonomy_contribution = self.autonomy_support * 0.4;
        let competence_contribution = self.competence_feedback * 0.3;
        let relatedness_contribution = self.relatedness * 0.3;

        (autonomy_contribution + competence_contribution + relatedness_contribution)
            .clamp(0.0, 1.0)
    }
}

// ============================================================================
// Negentropy Engine
// ============================================================================

/// Configuration for the negentropy engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegentropyConfig {
    /// Threshold for awareness triggering (default: 0.5)
    pub awareness_threshold: f64,
    /// Entropy leak rate (natural decay toward disorder)
    pub entropy_leak: f64,
    /// ATP efficiency (metabolic energy conversion)
    pub atp_efficiency: f64,
    /// Maximum negentropy (theoretical limit)
    pub max_negentropy: f64,
    /// Learning rate for level transitions
    pub transition_rate: f64,
    /// Coherence requirement for L3 transformation
    pub l3_coherence_threshold: f64,
    /// Coherence requirement for L4 evolution
    pub l4_coherence_threshold: f64,
    /// Negentropy threshold for L4 evolution
    pub l4_negentropy_threshold: f64,
    /// Minimum population size for L4 (evolutionary requires interaction)
    pub l4_min_population: usize,
    /// Fitness pressure threshold for triggering L4
    pub l4_fitness_pressure: f64,
    /// Steps at L3 required before L4 is possible
    pub l4_stabilization_steps: u64,
}

impl Default for NegentropyConfig {
    fn default() -> Self {
        Self {
            awareness_threshold: 0.5,
            entropy_leak: 0.01,
            atp_efficiency: 0.4,
            max_negentropy: 1.0,
            transition_rate: 0.1,
            l3_coherence_threshold: 0.7,
            l4_coherence_threshold: 0.9,
            l4_negentropy_threshold: 0.95,
            l4_min_population: 3,
            l4_fitness_pressure: 0.5,
            l4_stabilization_steps: 1000,
        }
    }
}

/// Negentropy Engine: Core of the pedagogic agency framework
///
/// Computes and regulates negentropy (inverse entropy) as the measure of
/// "aliveness" or ordered complexity in the agent.
///
/// ## Key Equation
///
/// N = S_max - S_actual
///
/// Where:
/// - N: Negentropy (0 = dead/disordered, 1 = maximally alive/ordered)
/// - S_max: Maximum possible entropy (theoretical limit)
/// - S_actual: Current entropy of the system
///
/// ## Thermodynamic Grounding
///
/// Based on Landauer bound: kT ln 2 per bit operation
/// Energy cost of cognition tied to entropy reduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegentropyEngine {
    /// Configuration
    pub config: NegentropyConfig,

    /// Current negentropy level [0, 1]
    pub negentropy: f64,

    /// Current Bateson learning level
    pub bateson_level: BatesonLevel,

    /// Cognitive regulator (brain-inspired)
    pub cognitive_regulator: CognitiveRegulator,

    /// Pedagogic scaffold
    pub scaffold: PedagogicScaffold,

    /// Metabolic reserve (energy for cognition)
    pub metabolic_reserve: f64,

    /// Negentropy history for dynamics analysis
    negentropy_history: VecDeque<f64>,

    /// Level transition history
    level_history: VecDeque<BatesonLevel>,

    /// Step counter
    step_count: u64,

    /// Awareness triggered flag
    awareness_triggered: bool,

    /// Steps at current L3 level (for L4 transition tracking)
    l3_stabilization_steps: u64,

    /// Population context for L4 (number of peer agents)
    population_context: usize,

    /// Fitness signal from population (for L4 evolutionary pressure)
    fitness_signal: f64,
}

impl NegentropyEngine {
    /// Create new negentropy engine with default configuration
    pub fn new() -> Self {
        Self::with_config(NegentropyConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: NegentropyConfig) -> Self {
        Self {
            config,
            negentropy: 0.5, // Start at threshold
            bateson_level: BatesonLevel::L0Reflex,
            cognitive_regulator: CognitiveRegulator::default(),
            scaffold: PedagogicScaffold::default(),
            metabolic_reserve: 1.0,
            negentropy_history: VecDeque::with_capacity(1000),
            level_history: VecDeque::with_capacity(100),
            step_count: 0,
            awareness_triggered: false,
            l3_stabilization_steps: 0,
            population_context: 1, // Default: single agent
            fitness_signal: 0.0,
        }
    }

    /// Core computation: Update negentropy based on agent state
    ///
    /// # Arguments
    /// * `beliefs` - Current belief state
    /// * `precision` - Belief precision (confidence)
    /// * `prediction_error` - Current prediction error
    /// * `free_energy` - Current free energy
    ///
    /// # Returns
    /// Updated negentropy level
    pub fn compute(
        &mut self,
        beliefs: &Array1<f64>,
        precision: &Array1<f64>,
        prediction_error: f64,
        free_energy: f64,
    ) -> f64 {
        self.step_count += 1;

        // ===== ENTROPY COMPUTATION =====
        // Actual entropy from belief distribution
        let belief_entropy = self.compute_belief_entropy(beliefs, precision);

        // Maximum entropy (uniform distribution)
        let max_entropy = (beliefs.len() as f64).ln();

        // ===== NEGENTROPY =====
        // N = S_max - S_actual (normalized to [0, 1])
        let raw_negentropy = if max_entropy > 0.001 {
            ((max_entropy - belief_entropy) / max_entropy).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // ===== THERMODYNAMIC CORRECTIONS =====
        // Energy cost of maintaining order (Landauer bound)
        let energy_cost = self.bateson_level.energy_requirement();
        let available_energy = self.metabolic_reserve * self.config.atp_efficiency;

        // Negentropy decays without energy
        let decay = if available_energy < energy_cost {
            self.config.entropy_leak * (1.0 + energy_cost - available_energy)
        } else {
            self.config.entropy_leak
        };

        // ===== COGNITIVE PROCESSING =====
        // ACC: Process prediction error
        let conflict = self.cognitive_regulator.anterior_cingulate
            .process_error(prediction_error);

        // Insula: Update bodily awareness
        let sensors = vec![free_energy, prediction_error, self.metabolic_reserve];
        self.cognitive_regulator.insula.update_interoception(&sensors);
        let bodily_awareness = self.cognitive_regulator.insula.bodily_awareness();

        // PFC: Executive control
        let executive_control = self.cognitive_regulator.prefrontal_cortex
            .executive_control(beliefs);

        // ===== NEGENTROPY UPDATE =====
        // Negentropy production from cognitive processing
        let production = (executive_control + 0.5 * (1.0 - conflict)) * available_energy;

        // Update negentropy with production and decay
        self.negentropy = raw_negentropy * 0.3 +
            self.negentropy * 0.7 +
            self.config.transition_rate * (production - decay);

        self.negentropy = self.negentropy.clamp(0.0, self.config.max_negentropy);

        // ===== METABOLIC UPDATE =====
        // Consume energy for cognition
        self.metabolic_reserve -= energy_cost * 0.01;
        // Regenerate slowly
        self.metabolic_reserve += 0.005 * (1.0 - self.metabolic_reserve);
        self.metabolic_reserve = self.metabolic_reserve.clamp(0.0, 1.0);

        // ===== AWARENESS TRIGGERING =====
        self.check_awareness_threshold();

        // ===== BATESON LEVEL UPDATE =====
        self.update_bateson_level(bodily_awareness);

        // ===== SCAFFOLDING =====
        self.scaffold.select_mode(self.negentropy, self.bateson_level);

        // Record history
        self.negentropy_history.push_back(self.negentropy);
        if self.negentropy_history.len() > 1000 {
            self.negentropy_history.pop_front();
        }

        self.negentropy
    }

    /// Compute entropy of belief distribution
    fn compute_belief_entropy(&self, beliefs: &Array1<f64>, precision: &Array1<f64>) -> f64 {
        // Convert beliefs to probability distribution
        let shifted = beliefs.mapv(|b| b.exp());
        let sum = shifted.sum();

        if sum < 0.001 {
            return (beliefs.len() as f64).ln(); // Maximum entropy
        }

        let probs = shifted / sum;

        // Shannon entropy: H = -Σ p_i log(p_i)
        let mut entropy = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            if p > 1e-10 {
                // Weight by precision (high precision = less uncertainty)
                let weight = 1.0 / (1.0 + precision[i]);
                entropy -= weight * p * p.ln();
            }
        }

        entropy.max(0.0)
    }

    /// Check and trigger awareness if below threshold
    fn check_awareness_threshold(&mut self) {
        let was_triggered = self.awareness_triggered;
        self.awareness_triggered = self.negentropy < self.config.awareness_threshold;

        if self.awareness_triggered && !was_triggered {
            // Just crossed threshold - activate scaffolding
            self.scaffold.mode = ScaffoldMode::CuriosityNudge;
        }
    }

    /// Update Bateson learning level based on conditions
    ///
    /// Bateson's hierarchy (extended):
    /// - L0: Reflexes (hardwired)
    /// - L1: Conditioning (habit formation)
    /// - L2: Meta-learning (learning to learn)
    /// - L3: Transformation (paradigm shifts)
    /// - L4: Evolution (population-level adaptation)
    fn update_bateson_level(&mut self, coherence: f64) {
        let current_level = self.bateson_level.level();

        // Level transition conditions
        let can_advance = self.negentropy >= self.bateson_level.energy_requirement()
            && coherence >= self.bateson_level.coherence_threshold()
            && self.metabolic_reserve > 0.3;

        let must_retreat = self.negentropy < self.bateson_level.energy_requirement() * 0.5
            || self.metabolic_reserve < 0.1;

        // L3 has special requirements
        let l3_possible = current_level >= 2
            && coherence >= self.config.l3_coherence_threshold
            && self.negentropy >= 0.8;

        // Track L3 stabilization for L4 transition
        if current_level == 3 {
            self.l3_stabilization_steps += 1;
        } else {
            self.l3_stabilization_steps = 0;
        }

        // L4 has special requirements (Holland, 1975):
        // - Sustained L3 operation (stabilization period)
        // - Population context (multiple interacting agents)
        // - Sufficient fitness pressure
        // - Very high coherence and negentropy
        let l4_possible = current_level == 3
            && self.l3_stabilization_steps >= self.config.l4_stabilization_steps
            && coherence >= self.config.l4_coherence_threshold
            && self.negentropy >= self.config.l4_negentropy_threshold
            && self.population_context >= self.config.l4_min_population
            && self.fitness_signal >= self.config.l4_fitness_pressure;

        let new_level = if must_retreat && current_level > 0 {
            current_level - 1
        } else if l4_possible && current_level < 4 {
            // L4: Evolutionary change - transcends individual learning
            4
        } else if l3_possible && current_level < 3 {
            3
        } else if can_advance && current_level < 2 {
            current_level + 1
        } else {
            current_level
        };

        if new_level != current_level {
            self.bateson_level = BatesonLevel::from_level(new_level);
            self.level_history.push_back(self.bateson_level);
            if self.level_history.len() > 100 {
                self.level_history.pop_front();
            }
            // Reset stabilization counter on level change
            if new_level != 3 {
                self.l3_stabilization_steps = 0;
            }
        }
    }

    /// Get scaffolding intervention signal
    ///
    /// This is the "graceful awareness" mechanism - not punishment,
    /// but supportive scaffolding based on current needs.
    pub fn get_intervention(&mut self) -> f64 {
        let coherence = self.cognitive_regulator.insula.bodily_awareness();
        self.scaffold.compute_intervention(self.negentropy, coherence)
    }

    /// Get curiosity boost for exploration
    pub fn get_curiosity_boost(&self) -> f64 {
        let uncertainty = 1.0 - self.negentropy;
        self.scaffold.curiosity_boost(uncertainty)
    }

    /// Check if awareness is triggered (below threshold)
    pub fn is_awareness_triggered(&self) -> bool {
        self.awareness_triggered
    }

    /// Check if agent is "alive" (above threshold)
    pub fn is_alive(&self) -> bool {
        self.negentropy >= self.config.awareness_threshold
    }

    /// Get current negentropy
    pub fn negentropy(&self) -> f64 {
        self.negentropy
    }

    /// Get current Bateson level
    pub fn learning_level(&self) -> BatesonLevel {
        self.bateson_level
    }

    /// Get scaffolding mode
    pub fn scaffold_mode(&self) -> ScaffoldMode {
        self.scaffold.mode
    }

    /// Get negentropy trend (positive = increasing)
    pub fn negentropy_trend(&self) -> f64 {
        if self.negentropy_history.len() < 10 {
            return 0.0;
        }

        let recent: Vec<f64> = self.negentropy_history.iter()
            .rev()
            .take(10)
            .copied()
            .collect();

        // Simple trend: last - first
        recent.first().unwrap_or(&0.0) - recent.last().unwrap_or(&0.0)
    }

    /// Get mean negentropy over history
    pub fn mean_negentropy(&self) -> f64 {
        if self.negentropy_history.is_empty() {
            return self.negentropy;
        }
        self.negentropy_history.iter().sum::<f64>() / self.negentropy_history.len() as f64
    }

    /// Inject metabolic energy (external energy source)
    pub fn inject_energy(&mut self, amount: f64) {
        self.metabolic_reserve += amount * self.config.atp_efficiency;
        self.metabolic_reserve = self.metabolic_reserve.clamp(0.0, 1.0);
    }

    /// Set goal state for PFC planning
    pub fn set_goal(&mut self, goal: Array1<f64>) {
        self.cognitive_regulator.prefrontal_cortex.set_goal(goal);
    }

    /// Update Self-Determination Theory components
    pub fn update_sdt(&mut self, autonomy: f64, competence: f64, relatedness: f64) {
        self.scaffold.update_sdt(autonomy, competence, relatedness);
    }

    /// Get intrinsic motivation level
    pub fn intrinsic_motivation(&self) -> f64 {
        self.scaffold.intrinsic_motivation()
    }

    /// Get negentropy history
    pub fn history(&self) -> &VecDeque<f64> {
        &self.negentropy_history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.negentropy_history.clear();
        self.level_history.clear();
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.negentropy = 0.5;
        self.bateson_level = BatesonLevel::L0Reflex;
        self.metabolic_reserve = 1.0;
        self.awareness_triggered = false;
        self.step_count = 0;
        self.l3_stabilization_steps = 0;
        self.population_context = 1;
        self.fitness_signal = 0.0;
        self.clear_history();
    }

    // ===== L4 EVOLUTIONARY LEARNING API =====

    /// Set population context for L4 evolutionary learning
    ///
    /// L4 requires multiple interacting agents (Holland, 1975).
    /// This represents the number of peer agents in the population.
    pub fn set_population_context(&mut self, population_size: usize) {
        self.population_context = population_size.max(1);
    }

    /// Get current population context
    pub fn population_context(&self) -> usize {
        self.population_context
    }

    /// Update fitness signal from population
    ///
    /// Fitness represents selection pressure from the environment/population.
    /// Higher values indicate stronger evolutionary pressure toward adaptation.
    ///
    /// # Arguments
    /// * `fitness` - Fitness signal [0.0, 1.0]
    pub fn update_fitness(&mut self, fitness: f64) {
        self.fitness_signal = fitness.clamp(0.0, 1.0);
    }

    /// Get current fitness signal
    pub fn fitness_signal(&self) -> f64 {
        self.fitness_signal
    }

    /// Get L3 stabilization progress (steps at L3 / required steps)
    pub fn l3_stabilization_progress(&self) -> f64 {
        if self.config.l4_stabilization_steps == 0 {
            return 1.0;
        }
        (self.l3_stabilization_steps as f64 / self.config.l4_stabilization_steps as f64)
            .clamp(0.0, 1.0)
    }

    /// Check if L4 evolutionary learning is possible
    ///
    /// Returns true if all L4 requirements are met:
    /// - Currently at L3 (transformation)
    /// - Sustained L3 for stabilization period
    /// - Population context >= minimum (3 by default)
    /// - Fitness pressure >= threshold (0.5 by default)
    /// - Coherence >= L4 threshold (0.9 by default)
    /// - Negentropy >= L4 threshold (0.95 by default)
    pub fn l4_possible(&self) -> bool {
        let coherence = self.cognitive_regulator.insula.bodily_awareness();
        self.bateson_level.level() == 3
            && self.l3_stabilization_steps >= self.config.l4_stabilization_steps
            && coherence >= self.config.l4_coherence_threshold
            && self.negentropy >= self.config.l4_negentropy_threshold
            && self.population_context >= self.config.l4_min_population
            && self.fitness_signal >= self.config.l4_fitness_pressure
    }

    /// Get L4 readiness metrics
    ///
    /// Returns a tuple of (stabilization_progress, coherence_progress,
    /// negentropy_progress, population_ready, fitness_ready)
    pub fn l4_readiness(&self) -> (f64, f64, f64, bool, bool) {
        let coherence = self.cognitive_regulator.insula.bodily_awareness();
        let stabilization_progress = self.l3_stabilization_progress();
        let coherence_progress = (coherence / self.config.l4_coherence_threshold).clamp(0.0, 1.0);
        let negentropy_progress = (self.negentropy / self.config.l4_negentropy_threshold).clamp(0.0, 1.0);
        let population_ready = self.population_context >= self.config.l4_min_population;
        let fitness_ready = self.fitness_signal >= self.config.l4_fitness_pressure;

        (stabilization_progress, coherence_progress, negentropy_progress, population_ready, fitness_ready)
    }

    /// Check if currently at L4 (evolutionary learning)
    pub fn at_l4(&self) -> bool {
        matches!(self.bateson_level, BatesonLevel::L4Evolution)
    }
}

impl Default for NegentropyEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Simple pseudo-random uniform [0, 1]
fn rand_uniform() -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos % 10000) as f64 / 10000.0
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negentropy_engine_creation() {
        let engine = NegentropyEngine::new();
        assert!((engine.negentropy - 0.5).abs() < 0.001);
        assert_eq!(engine.bateson_level, BatesonLevel::L0Reflex);
    }

    #[test]
    fn test_negentropy_computation() {
        let mut engine = NegentropyEngine::new();
        let beliefs = Array1::from_elem(32, 0.5);
        let precision = Array1::from_elem(32, 1.0);

        let n1 = engine.compute(&beliefs, &precision, 0.1, 1.0);
        assert!(n1 >= 0.0 && n1 <= 1.0);

        // Multiple steps
        for _ in 0..10 {
            engine.compute(&beliefs, &precision, 0.1, 1.0);
        }

        assert!(engine.negentropy >= 0.0 && engine.negentropy <= 1.0);
    }

    #[test]
    fn test_awareness_threshold() {
        let mut engine = NegentropyEngine::new();
        engine.negentropy = 0.3; // Below threshold

        let beliefs = Array1::from_elem(32, 0.1);
        let precision = Array1::from_elem(32, 0.5);

        engine.compute(&beliefs, &precision, 0.5, 2.0);

        assert!(engine.is_awareness_triggered());
        assert!(!engine.is_alive());
    }

    #[test]
    fn test_bateson_levels() {
        assert_eq!(BatesonLevel::L0Reflex.level(), 0);
        assert_eq!(BatesonLevel::L3Transformation.level(), 3);
        assert_eq!(BatesonLevel::from_level(2), BatesonLevel::L2MetaLearning);
    }

    #[test]
    fn test_scaffolding_modes() {
        let mut scaffold = PedagogicScaffold::default();

        // Low negentropy = Direct instruction
        let mode = scaffold.select_mode(0.15, BatesonLevel::L0Reflex);
        assert_eq!(mode, ScaffoldMode::DirectInstruction);

        // High negentropy = Autonomous
        let mode = scaffold.select_mode(0.8, BatesonLevel::L1Conditioning);
        assert_eq!(mode, ScaffoldMode::Autonomous);
    }

    #[test]
    fn test_curiosity_boost() {
        let scaffold = PedagogicScaffold::default();

        // Maximum curiosity at moderate uncertainty
        let low_uncertainty = scaffold.curiosity_boost(0.1);
        let mid_uncertainty = scaffold.curiosity_boost(0.5);
        let high_uncertainty = scaffold.curiosity_boost(0.9);

        assert!(mid_uncertainty > low_uncertainty);
        assert!(mid_uncertainty > high_uncertainty);
    }

    #[test]
    fn test_pfc_executive_control() {
        let mut pfc = PrefrontalCortex::default();
        pfc.goal_state = Array1::from_elem(16, 1.0);

        let current = Array1::from_elem(16, 0.0);
        let control = pfc.executive_control(&current);

        assert!(control > 0.0, "Control should be positive when far from goal");
    }

    #[test]
    fn test_acc_conflict_detection() {
        let mut acc = AnteriorCingulate::default();

        // Small error = no conflict
        let conflict1 = acc.process_error(0.1);
        assert!(conflict1 < 0.5);

        // Large error = high conflict
        let conflict2 = acc.process_error(1.5);
        assert!(conflict2 > conflict1);
    }

    #[test]
    fn test_basal_ganglia_action_selection() {
        let mut bg = BasalGanglia::default();

        // Update with rewards
        bg.update(0, 1.0);
        bg.update(1, 0.5);

        // Should prefer action 0
        let mut action_0_count = 0;
        for _ in 0..100 {
            if bg.select_action() == 0 {
                action_0_count += 1;
            }
        }

        assert!(action_0_count > 50, "Should prefer higher-value action");
    }

    #[test]
    fn test_insula_interoception() {
        let mut insula = Insula::default();

        insula.update_interoception(&[0.8, 0.2, 0.6]);

        assert!(insula.arousal > 0.0);
        assert!(insula.bodily_awareness() >= 0.0);
    }

    #[test]
    fn test_hippocampus_memory() {
        let mut hippo = Hippocampus::default();

        let state1 = Array1::from_elem(16, 0.5);
        let state2 = Array1::from_elem(16, 0.8);

        hippo.encode(&state1, 0.5, 1);
        hippo.encode(&state2, 0.9, 2);

        assert_eq!(hippo.episode_count(), 2);

        // Retrieve should find most similar
        let query = Array1::from_elem(16, 0.7);
        let retrieved = hippo.retrieve(&query);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_sdt_intrinsic_motivation() {
        let mut scaffold = PedagogicScaffold::default();

        scaffold.update_sdt(0.9, 0.8, 0.7);

        let motivation = scaffold.intrinsic_motivation();
        assert!(motivation > 0.5, "High SDT = high motivation");
    }

    #[test]
    fn test_energy_injection() {
        let mut engine = NegentropyEngine::new();
        engine.metabolic_reserve = 0.3;

        engine.inject_energy(1.0);

        assert!(engine.metabolic_reserve > 0.3);
    }

    #[test]
    fn test_negentropy_trend() {
        let mut engine = NegentropyEngine::new();
        let beliefs = Array1::from_elem(32, 0.5);
        let precision = Array1::from_elem(32, 1.0);

        // Generate history
        for _ in 0..20 {
            engine.compute(&beliefs, &precision, 0.1, 1.0);
        }

        // Trend should be finite
        let trend = engine.negentropy_trend();
        assert!(trend.is_finite());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let c = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    // ===== L4 EVOLUTIONARY LEARNING TESTS =====

    #[test]
    fn test_bateson_l4_level() {
        assert_eq!(BatesonLevel::L4Evolution.level(), 4);
        assert_eq!(BatesonLevel::from_level(4), BatesonLevel::L4Evolution);
        assert_eq!(BatesonLevel::from_level(5), BatesonLevel::L4Evolution); // Clamps to L4
    }

    #[test]
    fn test_l4_requirements() {
        // L4 has highest requirements
        let l4 = BatesonLevel::L4Evolution;
        let l3 = BatesonLevel::L3Transformation;

        assert!(l4.energy_requirement() > l3.energy_requirement());
        assert!(l4.coherence_threshold() > l3.coherence_threshold());
        assert!(l4.stabilization_period() > l3.stabilization_period());
        assert!(l4.requires_population());
        assert!(l4.population_requirement() > 0);
        assert!(l4.fitness_pressure() > 0.0);
    }

    #[test]
    fn test_l4_population_context() {
        let mut engine = NegentropyEngine::new();

        assert_eq!(engine.population_context(), 1); // Default

        engine.set_population_context(5);
        assert_eq!(engine.population_context(), 5);

        engine.set_population_context(0);
        assert_eq!(engine.population_context(), 1); // Minimum 1

        engine.reset();
        assert_eq!(engine.population_context(), 1); // Reset to default
    }

    #[test]
    fn test_l4_fitness_signal() {
        let mut engine = NegentropyEngine::new();

        assert!((engine.fitness_signal() - 0.0).abs() < 0.001); // Default

        engine.update_fitness(0.7);
        assert!((engine.fitness_signal() - 0.7).abs() < 0.001);

        engine.update_fitness(1.5); // Clamps to 1.0
        assert!((engine.fitness_signal() - 1.0).abs() < 0.001);

        engine.update_fitness(-0.5); // Clamps to 0.0
        assert!((engine.fitness_signal() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_l4_stabilization_progress() {
        let mut engine = NegentropyEngine::new();

        // Initially at L0, progress should be 0
        assert!((engine.l3_stabilization_progress() - 0.0).abs() < 0.001);

        // Simulate being at L3 for some steps
        engine.l3_stabilization_steps = 500;
        let progress = engine.l3_stabilization_progress();
        assert!(progress > 0.0 && progress < 1.0);
    }

    #[test]
    fn test_l4_not_possible_without_requirements() {
        let mut engine = NegentropyEngine::new();

        // Default state: L4 not possible
        assert!(!engine.l4_possible());

        // Even with high negentropy, L4 not possible without population
        engine.negentropy = 0.98;
        engine.bateson_level = BatesonLevel::L3Transformation;
        engine.l3_stabilization_steps = 2000;

        // Still not possible: missing population context
        assert!(!engine.l4_possible());
    }

    #[test]
    fn test_l4_readiness_metrics() {
        let mut engine = NegentropyEngine::new();

        engine.negentropy = 0.9;
        engine.bateson_level = BatesonLevel::L3Transformation;
        engine.l3_stabilization_steps = 500;
        engine.set_population_context(3);
        engine.update_fitness(0.6);

        let (stab, _coh, neg, pop, fit) = engine.l4_readiness();

        assert!(stab > 0.0); // Some stabilization progress
        assert!(neg > 0.0); // Some negentropy progress
        assert!(pop); // Population context met
        assert!(fit); // Fitness threshold met
    }

    #[test]
    fn test_at_l4() {
        let mut engine = NegentropyEngine::new();

        assert!(!engine.at_l4());

        engine.bateson_level = BatesonLevel::L4Evolution;
        assert!(engine.at_l4());
    }

    #[test]
    fn test_l4_description() {
        let l4 = BatesonLevel::L4Evolution;
        let desc = l4.description();
        assert!(desc.contains("Evolution"));
        assert!(desc.contains("population") || desc.contains("Systemic") || desc.contains("systemic"));
    }
}
