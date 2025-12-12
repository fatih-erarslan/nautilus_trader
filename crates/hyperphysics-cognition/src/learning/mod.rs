//! Bateson's Learning Levels (hierarchical meta-learning)
//!
//! Implements Gregory Bateson's four levels of learning:
//! - **Level 0 (Proto-learning)**: Reflexive, no adaptation
//! - **Level I (Learning)**: Conditioning, habit formation
//! - **Level II (Deutero-learning)**: Learning to learn, context sensitivity
//! - **Level III (Learning III)**: Paradigm shift, identity transformation
//!
//! ## Scientific Basis
//!
//! - Bateson (1972) "Steps to an Ecology of Mind"
//! - Harries-Jones (1995) "A Recursive Vision: Ecological Understanding"
//! - Wang et al. (2016) "Learning to reinforcement learn" (Meta-RL)
//!
//! ## Learning Hierarchy
//!
//! ```text
//! ┌────────────────────────────────────────────────────────┐
//! │         BATESON'S LEARNING LEVELS                      │
//! │                                                        │
//! │   Level III: Paradigm Shift                           │
//! │   (Identity transformation, rare)                     │
//! │              ▲                                         │
//! │              │ restructure                            │
//! │   ┌──────────┴───────────┐                            │
//! │   │  Level II: Deutero   │                            │
//! │   │  (Learning to learn) │                            │
//! │   │  (Context-sensitive) │                            │
//! │   └──────────┬───────────┘                            │
//! │              │ meta-learning                          │
//! │   ┌──────────▼───────────┐                            │
//! │   │  Level I: Learning   │                            │
//! │   │  (Conditioning)      │                            │
//! │   │  (Habit formation)   │                            │
//! │   └──────────┬───────────┘                            │
//! │              │ adaptation                             │
//! │   ┌──────────▼───────────┐                            │
//! │   │  Level 0: Proto      │                            │
//! │   │  (Reflexive)         │                            │
//! │   │  (No learning)       │                            │
//! │   └──────────────────────┘                            │
//! └────────────────────────────────────────────────────────┘
//! ```

use crate::error::{CognitionError, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, trace};

/// Learning level (Bateson's hierarchy)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LearningLevel {
    /// Level 0: Proto-learning (reflexive, no adaptation)
    ProtoLearning,

    /// Level I: Learning (conditioning, habit formation)
    Learning,

    /// Level II: Deutero-learning (learning to learn, meta-learning)
    DeuteroLearning,

    /// Level III: Learning III (paradigm shift, identity transformation)
    LearningIII,
}

impl LearningLevel {
    /// Get level number
    pub fn level(&self) -> u8 {
        match self {
            Self::ProtoLearning => 0,
            Self::Learning => 1,
            Self::DeuteroLearning => 2,
            Self::LearningIII => 3,
        }
    }

    /// Get level name
    pub fn name(&self) -> &'static str {
        match self {
            Self::ProtoLearning => "Proto-learning",
            Self::Learning => "Learning",
            Self::DeuteroLearning => "Deutero-learning",
            Self::LearningIII => "Learning III",
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            Self::ProtoLearning => "Reflexive responses, no adaptation",
            Self::Learning => "Conditioning and habit formation",
            Self::DeuteroLearning => "Learning to learn, context-sensitive",
            Self::LearningIII => "Paradigm shift, identity transformation",
        }
    }

    /// Can transition to higher level
    pub fn can_ascend(&self) -> bool {
        self.level() < 3
    }

    /// Get next higher level
    pub fn ascend(&self) -> Option<Self> {
        match self {
            Self::ProtoLearning => Some(Self::Learning),
            Self::Learning => Some(Self::DeuteroLearning),
            Self::DeuteroLearning => Some(Self::LearningIII),
            Self::LearningIII => None,
        }
    }
}

/// Proto-learning (Level 0) - Reflexive
#[derive(Debug, Clone)]
pub struct ProtoLearning {
    /// Fixed response mapping
    stimulus_response: HashMap<Vec<u8>, Vec<u8>>,
}

impl ProtoLearning {
    /// Create new proto-learning
    pub fn new() -> Self {
        Self {
            stimulus_response: HashMap::new(),
        }
    }

    /// Add reflex mapping
    pub fn add_reflex(&mut self, stimulus: Vec<u8>, response: Vec<u8>) {
        self.stimulus_response.insert(stimulus, response);
    }

    /// Get reflexive response
    pub fn respond(&self, stimulus: &[u8]) -> Option<&Vec<u8>> {
        self.stimulus_response.get(stimulus)
    }
}

impl Default for ProtoLearning {
    fn default() -> Self {
        Self::new()
    }
}

/// Learning (Level I) - Conditioning
#[derive(Debug, Clone)]
pub struct Learning {
    /// Learning rate
    learning_rate: f64,

    /// Association strengths
    associations: HashMap<(Vec<u8>, Vec<u8>), f64>,
}

impl Learning {
    /// Create new learning
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            associations: HashMap::new(),
        }
    }

    /// Update association (classical conditioning)
    pub fn update(&mut self, stimulus: Vec<u8>, response: Vec<u8>, reinforcement: f64) {
        let key = (stimulus, response);
        let strength = self.associations.entry(key).or_insert(0.0);
        *strength += self.learning_rate * (reinforcement - *strength);
    }

    /// Get association strength
    pub fn strength(&self, stimulus: &[u8], response: &[u8]) -> f64 {
        let key = (stimulus.to_vec(), response.to_vec());
        self.associations.get(&key).copied().unwrap_or(0.0)
    }
}

/// Deutero-learning (Level II) - Meta-learning
#[derive(Debug, Clone)]
pub struct DeuteroLearning {
    /// Meta-learning rate (learning rate for learning rates)
    meta_learning_rate: f64,

    /// Context-specific learning rates
    context_learning_rates: HashMap<Vec<u8>, f64>,

    /// Context sensitivity parameter
    context_sensitivity: f64,
}

impl DeuteroLearning {
    /// Create new deutero-learning
    pub fn new(meta_learning_rate: f64, context_sensitivity: f64) -> Self {
        Self {
            meta_learning_rate,
            context_learning_rates: HashMap::new(),
            context_sensitivity,
        }
    }

    /// Adapt learning rate for context
    pub fn adapt(&mut self, context: Vec<u8>, performance: f64) {
        let lr = self.context_learning_rates.entry(context).or_insert(0.1);

        // Increase learning rate if performance is poor, decrease if good
        let delta = self.meta_learning_rate * (0.5 - performance) * self.context_sensitivity;
        *lr = (*lr + delta).clamp(0.001, 1.0);
    }

    /// Get learning rate for context
    pub fn get_learning_rate(&self, context: &[u8]) -> f64 {
        self.context_learning_rates
            .get(context)
            .copied()
            .unwrap_or(0.1)
    }

    /// Get context sensitivity
    pub fn sensitivity(&self) -> f64 {
        self.context_sensitivity
    }
}

/// Learning III (Level III) - Paradigm shift
#[derive(Debug, Clone)]
pub struct LearningIII {
    /// Restructuring threshold
    restructure_threshold: f64,

    /// Accumulated restructuring pressure
    restructure_pressure: f64,

    /// Number of paradigm shifts
    shift_count: u32,
}

impl LearningIII {
    /// Create new learning III
    pub fn new(restructure_threshold: f64) -> Self {
        Self {
            restructure_threshold,
            restructure_pressure: 0.0,
            shift_count: 0,
        }
    }

    /// Accumulate restructuring pressure
    pub fn accumulate_pressure(&mut self, amount: f64) {
        self.restructure_pressure += amount;
    }

    /// Check if paradigm shift should occur
    pub fn should_shift(&self) -> bool {
        self.restructure_pressure >= self.restructure_threshold
    }

    /// Trigger paradigm shift
    pub fn shift(&mut self) {
        if self.should_shift() {
            self.shift_count += 1;
            self.restructure_pressure = 0.0;
            info!("🌀 Paradigm shift #{} triggered", self.shift_count);
        }
    }

    /// Get shift count
    pub fn shifts(&self) -> u32 {
        self.shift_count
    }
}

/// Learning context (environmental/situational)
#[derive(Debug, Clone)]
pub struct LearningContext {
    /// Context identifier
    pub id: Vec<u8>,

    /// Context features
    pub features: Vec<f64>,

    /// Timestamp
    pub timestamp: u64,
}

impl LearningContext {
    /// Create new learning context
    pub fn new(id: Vec<u8>, features: Vec<f64>) -> Self {
        Self {
            id,
            features,
            timestamp: 0,
        }
    }
}

/// Bateson learner (unified system)
pub struct BatesonLearner {
    /// Current learning level
    level: Arc<RwLock<LearningLevel>>,

    /// Proto-learning (Level 0)
    proto: Arc<RwLock<ProtoLearning>>,

    /// Learning (Level I)
    learning_i: Arc<RwLock<Learning>>,

    /// Deutero-learning (Level II)
    deutero: Arc<RwLock<DeuteroLearning>>,

    /// Learning III (Level III)
    learning_iii: Arc<RwLock<LearningIII>>,
}

impl BatesonLearner {
    /// Create new Bateson learner
    pub fn new() -> Result<Self> {
        debug!("📚 Bateson learner initialized (Level 0: Proto-learning)");

        Ok(Self {
            level: Arc::new(RwLock::new(LearningLevel::ProtoLearning)),
            proto: Arc::new(RwLock::new(ProtoLearning::new())),
            learning_i: Arc::new(RwLock::new(Learning::new(0.1))),
            deutero: Arc::new(RwLock::new(DeuteroLearning::new(0.01, 1.0))),
            learning_iii: Arc::new(RwLock::new(LearningIII::new(10.0))),
        })
    }

    /// Get current learning level
    pub fn level(&self) -> LearningLevel {
        *self.level.read()
    }

    /// Ascend to next learning level
    pub fn ascend(&self) -> Result<LearningLevel> {
        let mut level = self.level.write();

        let new_level = level.ascend()
            .ok_or_else(|| CognitionError::Learning("Already at maximum level".to_string()))?;

        info!("📈 Ascending to {} (Level {})", new_level.name(), new_level.level());
        *level = new_level;

        Ok(new_level)
    }

    /// Learn at current level
    pub fn learn(&self, stimulus: Vec<u8>, response: Vec<u8>, reinforcement: f64, context: LearningContext) {
        let level = self.level();

        match level {
            LearningLevel::ProtoLearning => {
                // No learning, only reflexes
                trace!("Proto-learning: reflex triggered");
            }

            LearningLevel::Learning => {
                let mut learning = self.learning_i.write();
                learning.update(stimulus, response, reinforcement);
                trace!("Level I: association updated");
            }

            LearningLevel::DeuteroLearning => {
                let mut deutero = self.deutero.write();
                deutero.adapt(context.id, reinforcement);
                trace!("Level II: learning rate adapted for context");
            }

            LearningLevel::LearningIII => {
                let mut learning_iii = self.learning_iii.write();
                learning_iii.accumulate_pressure((1.0 - reinforcement).abs());

                if learning_iii.should_shift() {
                    learning_iii.shift();
                }
                trace!("Level III: restructuring pressure accumulated");
            }
        }
    }

    /// Get proto-learning
    pub fn proto(&self) -> ProtoLearning {
        self.proto.read().clone()
    }

    /// Get learning I
    pub fn learning_i(&self) -> Learning {
        self.learning_i.read().clone()
    }

    /// Get deutero-learning
    pub fn deutero(&self) -> DeuteroLearning {
        self.deutero.read().clone()
    }

    /// Get learning III
    pub fn learning_iii(&self) -> LearningIII {
        self.learning_iii.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_levels() {
        assert_eq!(LearningLevel::ProtoLearning.level(), 0);
        assert_eq!(LearningLevel::Learning.level(), 1);
        assert_eq!(LearningLevel::DeuteroLearning.level(), 2);
        assert_eq!(LearningLevel::LearningIII.level(), 3);
    }

    #[test]
    fn test_level_ascension() {
        let mut level = LearningLevel::ProtoLearning;

        level = level.ascend().unwrap();
        assert_eq!(level, LearningLevel::Learning);

        level = level.ascend().unwrap();
        assert_eq!(level, LearningLevel::DeuteroLearning);

        level = level.ascend().unwrap();
        assert_eq!(level, LearningLevel::LearningIII);

        assert!(level.ascend().is_none());
    }

    #[test]
    fn test_proto_learning() {
        let mut proto = ProtoLearning::new();
        proto.add_reflex(vec![1, 2, 3], vec![4, 5, 6]);

        let response = proto.respond(&[1, 2, 3]);
        assert_eq!(response, Some(&vec![4, 5, 6]));
    }

    #[test]
    fn test_learning_i() {
        let mut learning = Learning::new(0.1);

        learning.update(vec![1], vec![2], 1.0);
        let strength = learning.strength(&[1], &[2]);
        assert!(strength > 0.0);

        learning.update(vec![1], vec![2], 1.0);
        let strength2 = learning.strength(&[1], &[2]);
        assert!(strength2 > strength); // Should increase
    }

    #[test]
    fn test_deutero_learning() {
        let mut deutero = DeuteroLearning::new(0.1, 1.0);

        let context = vec![1, 2, 3];

        // Poor performance → increase learning rate
        deutero.adapt(context.clone(), 0.2);
        let lr1 = deutero.get_learning_rate(&context);

        // Good performance → decrease learning rate
        deutero.adapt(context.clone(), 0.8);
        let lr2 = deutero.get_learning_rate(&context);

        assert!(lr1 > lr2);
    }

    #[test]
    fn test_learning_iii() {
        let mut learning_iii = LearningIII::new(10.0);

        assert_eq!(learning_iii.shifts(), 0);
        assert!(!learning_iii.should_shift());

        learning_iii.accumulate_pressure(5.0);
        assert!(!learning_iii.should_shift());

        learning_iii.accumulate_pressure(6.0);
        assert!(learning_iii.should_shift());

        learning_iii.shift();
        assert_eq!(learning_iii.shifts(), 1);
        assert!(!learning_iii.should_shift()); // Pressure reset
    }

    #[test]
    fn test_bateson_learner() {
        let learner = BatesonLearner::new().unwrap();

        assert_eq!(learner.level(), LearningLevel::ProtoLearning);

        learner.ascend().unwrap();
        assert_eq!(learner.level(), LearningLevel::Learning);

        learner.ascend().unwrap();
        assert_eq!(learner.level(), LearningLevel::DeuteroLearning);
    }
}
