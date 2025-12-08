//! # Interoceptive Inference Module
//!
//! Body-state inference and allostatic regulation via predictive coding
//! on hyperbolic manifolds.
//!
//! ## Theoretical Foundation
//!
//! Interoception is the sense of the internal state of the body. Active
//! interoceptive inference extends predictive coding to bodily signals,
//! enabling:
//!
//! - Homeostatic regulation through prediction error minimization
//! - Allostatic anticipation of bodily needs
//! - Emotional inference as interoceptive prediction
//!
//! ## Hyperbolic Extension
//!
//! In hyperbolic space:
//! - Interoceptive signals form hierarchical representations
//! - Set points can shift along geodesics
//! - Allostatic regulation follows hyperbolic gradient descent
//!
//! ## References
//!
//! - Seth & Friston (2016) "Active interoceptive inference and the emotional brain"
//! - Barrett & Simmons (2015) "Interoceptive predictions in the brain"
//! - Paulus & Stein (2010) "Interoception in anxiety and depression"
//! - Craig (2009) "How do you feel — now? The anterior insular and human awareness"
//! - Pezzulo et al. (2015) "Active Inference, homeostatic regulation and adaptive behavioural control"

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::hyperbolic_snn::LorentzVec;
use crate::free_energy::{Precision, PrecisionWeightedError};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for interoceptive inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteroceptionConfig {
    /// Number of hierarchical levels for interoceptive inference
    pub num_levels: usize,
    /// Precision of interoceptive signals (inverse variance)
    pub interoceptive_precision: f64,
    /// Precision of exteroceptive signals
    pub exteroceptive_precision: f64,
    /// Learning rate for belief updates
    pub learning_rate: f64,
    /// Tolerance for homeostatic deviation
    pub homeostatic_tolerance: f64,
    /// Allostatic anticipation horizon (time units)
    pub anticipation_horizon: f64,
    /// Whether to use hyperbolic representations
    pub hyperbolic: bool,
}

impl Default for InteroceptionConfig {
    fn default() -> Self {
        Self {
            num_levels: 3,
            interoceptive_precision: 1.0,
            exteroceptive_precision: 2.0,
            learning_rate: 0.1,
            homeostatic_tolerance: 0.1,
            anticipation_horizon: 10.0,
            hyperbolic: true,
        }
    }
}

// ============================================================================
// Interoceptive States
// ============================================================================

/// Cardiac state representation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CardiacState {
    /// Heart rate (beats per minute)
    pub heart_rate: f64,
    /// Heart rate variability (RMSSD in ms)
    pub hrv: f64,
    /// Systolic blood pressure
    pub systolic_bp: f64,
    /// Diastolic blood pressure
    pub diastolic_bp: f64,
}

impl Default for CardiacState {
    fn default() -> Self {
        Self {
            heart_rate: 70.0,
            hrv: 50.0,
            systolic_bp: 120.0,
            diastolic_bp: 80.0,
        }
    }
}

impl CardiacState {
    /// Convert to feature vector
    pub fn to_features(&self) -> Vec<f64> {
        vec![
            self.heart_rate / 100.0,  // Normalized
            self.hrv / 100.0,
            self.systolic_bp / 200.0,
            self.diastolic_bp / 150.0,
        ]
    }

    /// Compute cardiac arousal (0-1)
    pub fn arousal(&self) -> f64 {
        // Higher HR and lower HRV indicate arousal
        let hr_component = (self.heart_rate - 60.0) / 100.0;
        let hrv_component = 1.0 - (self.hrv / 100.0).min(1.0);
        (hr_component + hrv_component) / 2.0
    }
}

/// Respiratory state representation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RespiratoryState {
    /// Breathing rate (breaths per minute)
    pub breathing_rate: f64,
    /// Tidal volume (liters)
    pub tidal_volume: f64,
    /// Blood oxygen saturation (%)
    pub spo2: f64,
    /// End-tidal CO2 (mmHg)
    pub etco2: f64,
}

impl Default for RespiratoryState {
    fn default() -> Self {
        Self {
            breathing_rate: 14.0,
            tidal_volume: 0.5,
            spo2: 98.0,
            etco2: 40.0,
        }
    }
}

impl RespiratoryState {
    /// Convert to feature vector
    pub fn to_features(&self) -> Vec<f64> {
        vec![
            self.breathing_rate / 30.0,
            self.tidal_volume,
            self.spo2 / 100.0,
            self.etco2 / 60.0,
        ]
    }

    /// Compute respiratory distress (0-1)
    pub fn distress(&self) -> f64 {
        let br_distress = ((self.breathing_rate - 14.0).abs() / 10.0).min(1.0);
        let o2_distress = ((98.0 - self.spo2) / 10.0).max(0.0).min(1.0);
        (br_distress + o2_distress) / 2.0
    }
}

/// Metabolic state representation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MetabolicState {
    /// Blood glucose (mg/dL)
    pub glucose: f64,
    /// Core temperature (Celsius)
    pub temperature: f64,
    /// Hydration level (0-1)
    pub hydration: f64,
    /// Energy availability (0-1)
    pub energy: f64,
}

impl Default for MetabolicState {
    fn default() -> Self {
        Self {
            glucose: 100.0,
            temperature: 37.0,
            hydration: 0.8,
            energy: 0.7,
        }
    }
}

impl MetabolicState {
    /// Convert to feature vector
    pub fn to_features(&self) -> Vec<f64> {
        vec![
            self.glucose / 200.0,
            (self.temperature - 35.0) / 4.0,
            self.hydration,
            self.energy,
        ]
    }

    /// Compute metabolic stress (0-1)
    pub fn stress(&self) -> f64 {
        let glucose_stress = ((self.glucose - 100.0).abs() / 50.0).min(1.0);
        let temp_stress = ((self.temperature - 37.0).abs() / 2.0).min(1.0);
        let resource_stress = 1.0 - (self.hydration + self.energy) / 2.0;
        (glucose_stress + temp_stress + resource_stress) / 3.0
    }
}

/// Complete interoceptive state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteroceptiveState {
    /// Cardiac signals
    pub cardiac: CardiacState,
    /// Respiratory signals
    pub respiratory: RespiratoryState,
    /// Metabolic state
    pub metabolic: MetabolicState,
    /// Overall arousal level (0-1)
    pub arousal: f64,
    /// Valence (-1 to +1)
    pub valence: f64,
    /// Timestamp
    pub time: f64,
    /// Position in hyperbolic space (if using hyperbolic representation)
    pub position: Option<LorentzVec>,
}

impl Default for InteroceptiveState {
    fn default() -> Self {
        Self {
            cardiac: CardiacState::default(),
            respiratory: RespiratoryState::default(),
            metabolic: MetabolicState::default(),
            arousal: 0.3,
            valence: 0.0,
            time: 0.0,
            position: None,
        }
    }
}

impl InteroceptiveState {
    /// Create from subsystem states
    pub fn new(cardiac: CardiacState, respiratory: RespiratoryState, metabolic: MetabolicState) -> Self {
        let arousal = (cardiac.arousal() + respiratory.distress() + metabolic.stress()) / 3.0;
        let valence = 1.0 - 2.0 * arousal; // High arousal = negative valence (simplified)

        Self {
            cardiac,
            respiratory,
            metabolic,
            arousal,
            valence,
            time: 0.0,
            position: None,
        }
    }

    /// Convert to full feature vector
    pub fn to_features(&self) -> Vec<f64> {
        let mut features = Vec::new();
        features.extend(self.cardiac.to_features());
        features.extend(self.respiratory.to_features());
        features.extend(self.metabolic.to_features());
        features.push(self.arousal);
        features.push(self.valence);
        features
    }

    /// Convert to hyperbolic position
    pub fn to_hyperbolic(&self) -> LorentzVec {
        let features = self.to_features();

        // Use first 3 features as spatial coordinates (normalized to Poincaré disk)
        let x = features.get(0).copied().unwrap_or(0.0) * 0.5;
        let y = features.get(1).copied().unwrap_or(0.0) * 0.5;
        let z = features.get(2).copied().unwrap_or(0.0) * 0.5;

        // Project to hyperboloid
        let spatial_sq = x * x + y * y + z * z;
        let t = (1.0 + spatial_sq).sqrt();

        LorentzVec::new(t, x, y, z)
    }

    /// Compute distance from homeostatic set point
    pub fn deviation_from(&self, set_point: &InteroceptiveState) -> f64 {
        let self_features = self.to_features();
        let set_features = set_point.to_features();

        let mut sum_sq = 0.0;
        for (a, b) in self_features.iter().zip(set_features.iter()) {
            sum_sq += (a - b).powi(2);
        }
        sum_sq.sqrt()
    }

    /// Check if state is within homeostatic bounds
    pub fn is_homeostatic(&self, set_point: &InteroceptiveState, tolerance: f64) -> bool {
        self.deviation_from(set_point) <= tolerance
    }
}

// ============================================================================
// Interoceptive Belief
// ============================================================================

/// Hierarchical belief about interoceptive states
#[derive(Debug, Clone)]
pub struct InteroceptiveBelief {
    /// Mean belief at each level
    pub means: Vec<InteroceptiveState>,
    /// Precision at each level
    pub precisions: Vec<f64>,
    /// Prediction errors at each level
    pub errors: Vec<f64>,
}

impl InteroceptiveBelief {
    /// Create new belief with given number of levels
    pub fn new(num_levels: usize) -> Self {
        Self {
            means: (0..num_levels).map(|_| InteroceptiveState::default()).collect(),
            precisions: vec![1.0; num_levels],
            errors: vec![0.0; num_levels],
        }
    }

    /// Update belief from observation
    pub fn update(&mut self, observation: &InteroceptiveState, level: usize, learning_rate: f64) {
        if level >= self.means.len() {
            return;
        }

        // Prediction error
        let error = observation.deviation_from(&self.means[level]);
        self.errors[level] = error;

        // Precision-weighted belief update
        let gain = learning_rate * self.precisions[level];

        // Update mean towards observation
        let obs_features = observation.to_features();
        let mean_features = self.means[level].to_features();

        // Simple linear interpolation update
        let new_arousal = self.means[level].arousal + gain * (observation.arousal - self.means[level].arousal);
        let new_valence = self.means[level].valence + gain * (observation.valence - self.means[level].valence);

        self.means[level].arousal = new_arousal;
        self.means[level].valence = new_valence;
    }

    /// Get top-level belief (most abstract)
    pub fn top_level(&self) -> &InteroceptiveState {
        self.means.last().unwrap_or(&self.means[0])
    }

    /// Get total prediction error
    pub fn total_error(&self) -> f64 {
        self.errors.iter().sum()
    }
}

// ============================================================================
// Allostatic Regulator
// ============================================================================

/// Allostatic regulation policy
#[derive(Debug, Clone)]
pub struct AllostaticRegulator {
    /// Target set point (can shift based on anticipated needs)
    pub set_point: InteroceptiveState,
    /// Baseline set point (true homeostatic target)
    pub baseline: InteroceptiveState,
    /// Current regulatory actions
    pub actions: Vec<RegulatoryAction>,
    /// Anticipation buffer (predicted future needs)
    pub anticipation: VecDeque<InteroceptiveState>,
    /// Regulation gain
    pub gain: f64,
}

/// Regulatory action to restore homeostasis
#[derive(Debug, Clone)]
pub struct RegulatoryAction {
    /// Action type
    pub action_type: RegulatoryActionType,
    /// Intensity (0-1)
    pub intensity: f64,
    /// Target subsystem
    pub target: InteroceptiveSubsystem,
}

/// Types of regulatory actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegulatoryActionType {
    /// Increase activation
    Activate,
    /// Decrease activation
    Inhibit,
    /// Maintain current state
    Maintain,
    /// Shift set point
    Adapt,
}

/// Interoceptive subsystems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteroceptiveSubsystem {
    Cardiac,
    Respiratory,
    Metabolic,
    Autonomic,
    All,
}

impl AllostaticRegulator {
    /// Create new regulator with given baseline
    pub fn new(baseline: InteroceptiveState) -> Self {
        Self {
            set_point: baseline.clone(),
            baseline,
            actions: Vec::new(),
            anticipation: VecDeque::with_capacity(100),
            gain: 0.5,
        }
    }

    /// Regulate based on current state and prediction errors
    pub fn regulate(&mut self, current: &InteroceptiveState, errors: &[f64]) -> Vec<RegulatoryAction> {
        self.actions.clear();

        // Check cardiac deviation
        let cardiac_error = current.cardiac.arousal() - self.set_point.cardiac.arousal();
        if cardiac_error.abs() > 0.1 {
            self.actions.push(RegulatoryAction {
                action_type: if cardiac_error > 0.0 {
                    RegulatoryActionType::Inhibit
                } else {
                    RegulatoryActionType::Activate
                },
                intensity: cardiac_error.abs().min(1.0),
                target: InteroceptiveSubsystem::Cardiac,
            });
        }

        // Check respiratory distress
        let respiratory_error = current.respiratory.distress() - self.set_point.respiratory.distress();
        if respiratory_error.abs() > 0.1 {
            self.actions.push(RegulatoryAction {
                action_type: if respiratory_error > 0.0 {
                    RegulatoryActionType::Inhibit
                } else {
                    RegulatoryActionType::Activate
                },
                intensity: respiratory_error.abs().min(1.0),
                target: InteroceptiveSubsystem::Respiratory,
            });
        }

        // Check metabolic stress
        let metabolic_error = current.metabolic.stress() - self.set_point.metabolic.stress();
        if metabolic_error.abs() > 0.1 {
            self.actions.push(RegulatoryAction {
                action_type: if metabolic_error > 0.0 {
                    RegulatoryActionType::Inhibit
                } else {
                    RegulatoryActionType::Activate
                },
                intensity: metabolic_error.abs().min(1.0),
                target: InteroceptiveSubsystem::Metabolic,
            });
        }

        // If total error is high, consider set point adaptation (allostasis)
        let total_error: f64 = errors.iter().sum();
        if total_error > 0.5 {
            self.actions.push(RegulatoryAction {
                action_type: RegulatoryActionType::Adapt,
                intensity: (total_error - 0.5).min(1.0),
                target: InteroceptiveSubsystem::All,
            });
        }

        self.actions.clone()
    }

    /// Anticipate future needs based on predicted states
    pub fn anticipate(&mut self, predicted: &InteroceptiveState) {
        self.anticipation.push_back(predicted.clone());
        if self.anticipation.len() > 100 {
            self.anticipation.pop_front();
        }

        // Adjust set point based on anticipated needs
        if let Some(future) = self.anticipation.back() {
            // If anticipating high arousal, proactively shift set point
            if future.arousal > self.baseline.arousal + 0.2 {
                self.set_point.arousal = self.baseline.arousal + 0.1 * (future.arousal - self.baseline.arousal);
            } else {
                // Relax set point back towards baseline
                self.set_point.arousal = 0.9 * self.set_point.arousal + 0.1 * self.baseline.arousal;
            }
        }
    }

    /// Reset set point to baseline
    pub fn reset_to_baseline(&mut self) {
        self.set_point = self.baseline.clone();
        self.anticipation.clear();
    }
}

// ============================================================================
// Interoceptive Inference System
// ============================================================================

/// Complete interoceptive inference system
pub struct InteroceptiveInference {
    /// Configuration
    pub config: InteroceptionConfig,
    /// Hierarchical belief about body states
    pub belief: InteroceptiveBelief,
    /// Allostatic regulator
    pub regulator: AllostaticRegulator,
    /// History of states
    state_history: VecDeque<InteroceptiveState>,
    /// Current time
    current_time: f64,
    /// Statistics
    pub stats: InteroceptiveStats,
}

/// Statistics for interoceptive inference
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InteroceptiveStats {
    /// Total observations processed
    pub observations: usize,
    /// Average prediction error
    pub avg_error: f64,
    /// Current arousal
    pub current_arousal: f64,
    /// Current valence
    pub current_valence: f64,
    /// Time in homeostatic range (%)
    pub homeostatic_time: f64,
    /// Regulatory actions taken
    pub regulatory_actions: usize,
    /// Set point adaptations
    pub adaptations: usize,
}

impl InteroceptiveInference {
    /// Create new interoceptive inference system
    pub fn new(config: InteroceptionConfig) -> Self {
        let baseline = InteroceptiveState::default();
        Self {
            belief: InteroceptiveBelief::new(config.num_levels),
            regulator: AllostaticRegulator::new(baseline),
            state_history: VecDeque::with_capacity(1000),
            current_time: 0.0,
            stats: InteroceptiveStats::default(),
            config,
        }
    }

    /// Process interoceptive observation
    pub fn process(&mut self, observation: &InteroceptiveState) -> InteroceptiveResult {
        self.current_time = observation.time;
        self.stats.observations += 1;

        // Store in history
        self.state_history.push_back(observation.clone());
        if self.state_history.len() > 1000 {
            self.state_history.pop_front();
        }

        // Update beliefs at all levels (bottom-up)
        for level in 0..self.config.num_levels {
            let lr = self.config.learning_rate / (level as f64 + 1.0);
            self.belief.update(observation, level, lr);
        }

        // Compute total prediction error
        let total_error = self.belief.total_error();
        self.stats.avg_error = 0.9 * self.stats.avg_error + 0.1 * total_error;

        // Update statistics
        self.stats.current_arousal = observation.arousal;
        self.stats.current_valence = observation.valence;

        // Check homeostasis
        let is_homeostatic = observation.is_homeostatic(&self.regulator.set_point, self.config.homeostatic_tolerance);
        if is_homeostatic {
            self.stats.homeostatic_time = 0.99 * self.stats.homeostatic_time + 0.01;
        } else {
            self.stats.homeostatic_time *= 0.99;
        }

        // Generate regulatory actions if needed
        let actions = if !is_homeostatic {
            let actions = self.regulator.regulate(observation, &self.belief.errors);
            self.stats.regulatory_actions += actions.len();

            // Count adaptations
            for action in &actions {
                if action.action_type == RegulatoryActionType::Adapt {
                    self.stats.adaptations += 1;
                }
            }
            actions
        } else {
            Vec::new()
        };

        InteroceptiveResult {
            belief: self.belief.top_level().clone(),
            prediction_error: total_error,
            is_homeostatic,
            regulatory_actions: actions,
            arousal: observation.arousal,
            valence: observation.valence,
        }
    }

    /// Predict future state for anticipatory regulation
    pub fn predict(&mut self, horizon: f64) -> InteroceptiveState {
        // Simple linear extrapolation from recent history
        if self.state_history.len() < 2 {
            return self.belief.top_level().clone();
        }

        let recent: Vec<_> = self.state_history.iter().rev().take(10).collect();
        let current = recent[0];

        // Estimate rate of change
        let arousal_rate = if recent.len() >= 2 {
            (recent[0].arousal - recent[recent.len() - 1].arousal) / recent.len() as f64
        } else {
            0.0
        };

        let mut predicted = current.clone();
        predicted.arousal = (current.arousal + arousal_rate * horizon).clamp(0.0, 1.0);
        predicted.time = self.current_time + horizon;

        // Anticipate in regulator
        self.regulator.anticipate(&predicted);

        predicted
    }

    /// Get current set point
    pub fn get_set_point(&self) -> &InteroceptiveState {
        &self.regulator.set_point
    }

    /// Manually set the homeostatic set point
    pub fn set_baseline(&mut self, baseline: InteroceptiveState) {
        self.regulator.baseline = baseline.clone();
        self.regulator.set_point = baseline;
    }

    /// Convert current state to hyperbolic position
    pub fn to_hyperbolic_position(&self) -> LorentzVec {
        self.belief.top_level().to_hyperbolic()
    }

    /// Get hyperbolic distance from set point
    pub fn hyperbolic_error(&self) -> f64 {
        let current_pos = self.belief.top_level().to_hyperbolic();
        let setpoint_pos = self.regulator.set_point.to_hyperbolic();
        current_pos.hyperbolic_distance(&setpoint_pos)
    }
}

/// Result of interoceptive inference step
#[derive(Debug, Clone)]
pub struct InteroceptiveResult {
    /// Updated belief about body state
    pub belief: InteroceptiveState,
    /// Total prediction error
    pub prediction_error: f64,
    /// Whether state is within homeostatic bounds
    pub is_homeostatic: bool,
    /// Regulatory actions to take
    pub regulatory_actions: Vec<RegulatoryAction>,
    /// Current arousal
    pub arousal: f64,
    /// Current valence
    pub valence: f64,
}

// ============================================================================
// Russell Circumplex Model of Affect
// ============================================================================
//
// References:
// - Russell (1980) "A circumplex model of affect" J. Personality & Social Psychology
// - Russell & Barrett (1999) "Core affect, prototypical emotional episodes..."
// - Posner, Russell & Peterson (2005) "The circumplex model of affect"
// - Kuppens et al. (2013) "The relation between valence and arousal..."
//
// The circumplex positions emotions in a 2D space:
// - X-axis: Valence (pleasure → displeasure), range [-1, 1]
// - Y-axis: Arousal (activation → deactivation), range [-1, 1]
//
// Emotions are positioned at specific angles in this circular space.

/// Named emotions in the Russell circumplex model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CircumplexEmotion {
    /// High valence, high arousal (45°)
    Excited,
    /// High valence, moderate arousal (30°)
    Happy,
    /// High valence, low arousal (330°)
    Relaxed,
    /// High valence, very low arousal (315°)
    Calm,
    /// Low valence, low arousal (225°)
    Sad,
    /// Low valence, very low arousal (210°)
    Depressed,
    /// Low valence, high arousal (135°)
    Angry,
    /// Low valence, high arousal (150°)
    Afraid,
    /// Moderate valence, high arousal (90°)
    Alert,
    /// Moderate valence, moderate arousal (0°)
    Content,
    /// Low valence, moderate arousal (180°)
    Tense,
    /// Moderate valence, low arousal (270°)
    Tired,
}

impl CircumplexEmotion {
    /// Get the canonical angle (in radians) for this emotion
    pub fn canonical_angle(&self) -> f64 {
        use std::f64::consts::PI;
        match self {
            Self::Excited => PI / 4.0,        // 45°
            Self::Happy => PI / 6.0,          // 30°
            Self::Relaxed => 11.0 * PI / 6.0, // 330°
            Self::Calm => 7.0 * PI / 4.0,     // 315°
            Self::Sad => 5.0 * PI / 4.0,      // 225°
            Self::Depressed => 7.0 * PI / 6.0,// 210°
            Self::Angry => 3.0 * PI / 4.0,    // 135°
            Self::Afraid => 5.0 * PI / 6.0,   // 150°
            Self::Alert => PI / 2.0,          // 90°
            Self::Content => 0.0,             // 0°
            Self::Tense => PI,                // 180°
            Self::Tired => 3.0 * PI / 2.0,    // 270°
        }
    }

    /// Get canonical (valence, arousal) coordinates for this emotion
    pub fn canonical_coords(&self) -> (f64, f64) {
        let angle = self.canonical_angle();
        let valence = angle.cos();
        let arousal = angle.sin();
        (valence, arousal)
    }

    /// Get canonical radius (intensity) - typically 0.7-0.9 for basic emotions
    pub fn canonical_intensity(&self) -> f64 {
        match self {
            Self::Excited | Self::Afraid | Self::Angry | Self::Depressed => 0.85,
            Self::Happy | Self::Sad | Self::Alert | Self::Tense => 0.75,
            Self::Relaxed | Self::Calm | Self::Content | Self::Tired => 0.65,
        }
    }

    /// All emotions in the circumplex
    pub fn all() -> &'static [CircumplexEmotion] {
        &[
            Self::Excited, Self::Happy, Self::Relaxed, Self::Calm,
            Self::Sad, Self::Depressed, Self::Angry, Self::Afraid,
            Self::Alert, Self::Content, Self::Tense, Self::Tired,
        ]
    }
}

/// A point in the Russell circumplex affective space
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CircumplexPoint {
    /// Valence: pleasure-displeasure axis, range [-1, 1]
    pub valence: f64,
    /// Arousal: activation-deactivation axis, range [-1, 1]
    pub arousal: f64,
}

impl CircumplexPoint {
    /// Create new circumplex point
    pub fn new(valence: f64, arousal: f64) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(-1.0, 1.0),
        }
    }

    /// Create from polar coordinates (angle in radians, intensity 0-1)
    pub fn from_polar(angle: f64, intensity: f64) -> Self {
        let r = intensity.clamp(0.0, 1.0);
        Self {
            valence: r * angle.cos(),
            arousal: r * angle.sin(),
        }
    }

    /// Convert to polar coordinates (angle, intensity)
    pub fn to_polar(&self) -> (f64, f64) {
        let intensity = (self.valence * self.valence + self.arousal * self.arousal).sqrt();
        let angle = self.arousal.atan2(self.valence);
        // Normalize angle to [0, 2π)
        let normalized_angle = if angle < 0.0 {
            angle + 2.0 * std::f64::consts::PI
        } else {
            angle
        };
        (normalized_angle, intensity)
    }

    /// Get intensity (distance from origin)
    pub fn intensity(&self) -> f64 {
        (self.valence * self.valence + self.arousal * self.arousal).sqrt()
    }

    /// Euclidean distance to another point
    pub fn distance_to(&self, other: &CircumplexPoint) -> f64 {
        let dv = self.valence - other.valence;
        let da = self.arousal - other.arousal;
        (dv * dv + da * da).sqrt()
    }

    /// Angular distance (in radians) to another point
    pub fn angular_distance_to(&self, other: &CircumplexPoint) -> f64 {
        let (angle1, _) = self.to_polar();
        let (angle2, _) = other.to_polar();
        let diff = (angle1 - angle2).abs();
        if diff > std::f64::consts::PI {
            2.0 * std::f64::consts::PI - diff
        } else {
            diff
        }
    }

    /// Identify the nearest named emotion
    pub fn nearest_emotion(&self) -> (CircumplexEmotion, f64) {
        let mut best_emotion = CircumplexEmotion::Content;
        let mut best_distance = f64::INFINITY;

        for &emotion in CircumplexEmotion::all() {
            let (ev, ea) = emotion.canonical_coords();
            let ei = emotion.canonical_intensity();
            let emotion_point = CircumplexPoint::new(ev * ei, ea * ei);
            let dist = self.distance_to(&emotion_point);

            if dist < best_distance {
                best_distance = dist;
                best_emotion = emotion;
            }
        }

        (best_emotion, best_distance)
    }

    /// Get emotion probabilities (soft classification using RBF kernel)
    pub fn emotion_probabilities(&self, bandwidth: f64) -> Vec<(CircumplexEmotion, f64)> {
        let mut probs = Vec::new();
        let mut total = 0.0;

        for &emotion in CircumplexEmotion::all() {
            let (ev, ea) = emotion.canonical_coords();
            let ei = emotion.canonical_intensity();
            let emotion_point = CircumplexPoint::new(ev * ei, ea * ei);
            let dist = self.distance_to(&emotion_point);

            // RBF kernel: exp(-dist² / (2 * bandwidth²))
            let prob = (-dist * dist / (2.0 * bandwidth * bandwidth)).exp();
            probs.push((emotion, prob));
            total += prob;
        }

        // Normalize to probabilities
        if total > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= total;
            }
        }

        // Sort by probability descending
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        probs
    }

    /// Linear interpolation between two circumplex points
    pub fn lerp(&self, other: &CircumplexPoint, t: f64) -> CircumplexPoint {
        let t = t.clamp(0.0, 1.0);
        CircumplexPoint::new(
            self.valence + t * (other.valence - self.valence),
            self.arousal + t * (other.arousal - self.arousal),
        )
    }

    /// Spherical linear interpolation (follows arc on unit circle)
    pub fn slerp(&self, other: &CircumplexPoint, t: f64) -> CircumplexPoint {
        let (angle1, r1) = self.to_polar();
        let (angle2, r2) = other.to_polar();

        // Handle angle wrapping
        let mut delta_angle = angle2 - angle1;
        if delta_angle > std::f64::consts::PI {
            delta_angle -= 2.0 * std::f64::consts::PI;
        } else if delta_angle < -std::f64::consts::PI {
            delta_angle += 2.0 * std::f64::consts::PI;
        }

        let t = t.clamp(0.0, 1.0);
        let interp_angle = angle1 + t * delta_angle;
        let interp_r = r1 + t * (r2 - r1);

        CircumplexPoint::from_polar(interp_angle, interp_r)
    }
}

impl Default for CircumplexPoint {
    fn default() -> Self {
        Self { valence: 0.0, arousal: 0.0 }
    }
}

/// Russell circumplex model for affect representation
#[derive(Debug, Clone)]
pub struct RussellCircumplex {
    /// Current affective state
    pub current: CircumplexPoint,
    /// History of affective states
    history: VecDeque<(f64, CircumplexPoint)>, // (time, point)
    /// Smoothing factor for updates (0 = no smoothing, 1 = full smoothing)
    smoothing: f64,
    /// Maximum history length
    max_history: usize,
}

impl RussellCircumplex {
    /// Create new circumplex model
    pub fn new() -> Self {
        Self {
            current: CircumplexPoint::default(),
            history: VecDeque::with_capacity(1000),
            smoothing: 0.3,
            max_history: 1000,
        }
    }

    /// Create with custom smoothing factor
    pub fn with_smoothing(smoothing: f64) -> Self {
        Self {
            current: CircumplexPoint::default(),
            history: VecDeque::with_capacity(1000),
            smoothing: smoothing.clamp(0.0, 1.0),
            max_history: 1000,
        }
    }

    /// Update from interoceptive state
    pub fn update_from_interoceptive(&mut self, state: &InteroceptiveState, time: f64) {
        // Map interoceptive signals to circumplex coordinates
        // Arousal is primarily driven by cardiac/respiratory activation
        // Valence is modulated by metabolic state and homeostatic deviation

        let arousal_raw = state.arousal * 2.0 - 1.0; // Map [0,1] to [-1,1]
        let valence_raw = state.valence; // Already in [-1,1]

        let new_point = CircumplexPoint::new(valence_raw, arousal_raw);

        // Exponential smoothing
        let smoothed = CircumplexPoint::new(
            self.smoothing * self.current.valence + (1.0 - self.smoothing) * new_point.valence,
            self.smoothing * self.current.arousal + (1.0 - self.smoothing) * new_point.arousal,
        );

        self.current = smoothed;

        // Store in history
        self.history.push_back((time, self.current));
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Direct update with valence and arousal
    pub fn update(&mut self, valence: f64, arousal: f64, time: f64) {
        let new_point = CircumplexPoint::new(valence, arousal);

        let smoothed = CircumplexPoint::new(
            self.smoothing * self.current.valence + (1.0 - self.smoothing) * new_point.valence,
            self.smoothing * self.current.arousal + (1.0 - self.smoothing) * new_point.arousal,
        );

        self.current = smoothed;
        self.history.push_back((time, self.current));
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Get the current dominant emotion
    pub fn current_emotion(&self) -> (CircumplexEmotion, f64) {
        self.current.nearest_emotion()
    }

    /// Get emotion probability distribution
    pub fn emotion_distribution(&self, bandwidth: f64) -> Vec<(CircumplexEmotion, f64)> {
        self.current.emotion_probabilities(bandwidth)
    }

    /// Get affect trajectory over time window
    pub fn trajectory(&self, window_seconds: f64) -> Vec<(f64, CircumplexPoint)> {
        if self.history.is_empty() {
            return vec![];
        }

        let current_time = self.history.back().map(|(t, _)| *t).unwrap_or(0.0);
        let start_time = current_time - window_seconds;

        self.history
            .iter()
            .filter(|(t, _)| *t >= start_time)
            .cloned()
            .collect()
    }

    /// Compute affect variability (standard deviation of recent positions)
    pub fn affect_variability(&self, window_seconds: f64) -> (f64, f64) {
        let trajectory = self.trajectory(window_seconds);
        if trajectory.len() < 2 {
            return (0.0, 0.0);
        }

        let n = trajectory.len() as f64;
        let mean_v: f64 = trajectory.iter().map(|(_, p)| p.valence).sum::<f64>() / n;
        let mean_a: f64 = trajectory.iter().map(|(_, p)| p.arousal).sum::<f64>() / n;

        let var_v: f64 = trajectory.iter().map(|(_, p)| (p.valence - mean_v).powi(2)).sum::<f64>() / n;
        let var_a: f64 = trajectory.iter().map(|(_, p)| (p.arousal - mean_a).powi(2)).sum::<f64>() / n;

        (var_v.sqrt(), var_a.sqrt())
    }

    /// Compute affect inertia (autocorrelation)
    pub fn affect_inertia(&self, lag: usize) -> (f64, f64) {
        if self.history.len() <= lag {
            return (0.0, 0.0);
        }

        let points: Vec<_> = self.history.iter().map(|(_, p)| p).collect();
        let n = points.len() - lag;

        if n < 2 {
            return (0.0, 0.0);
        }

        // Mean and variance
        let mean_v: f64 = points.iter().map(|p| p.valence).sum::<f64>() / points.len() as f64;
        let mean_a: f64 = points.iter().map(|p| p.arousal).sum::<f64>() / points.len() as f64;

        let var_v: f64 = points.iter().map(|p| (p.valence - mean_v).powi(2)).sum::<f64>() / points.len() as f64;
        let var_a: f64 = points.iter().map(|p| (p.arousal - mean_a).powi(2)).sum::<f64>() / points.len() as f64;

        if var_v < 1e-10 || var_a < 1e-10 {
            return (0.0, 0.0);
        }

        // Autocorrelation
        let mut cov_v = 0.0;
        let mut cov_a = 0.0;
        for i in 0..n {
            cov_v += (points[i].valence - mean_v) * (points[i + lag].valence - mean_v);
            cov_a += (points[i].arousal - mean_a) * (points[i + lag].arousal - mean_a);
        }
        cov_v /= n as f64;
        cov_a /= n as f64;

        (cov_v / var_v, cov_a / var_a)
    }

    /// Map circumplex position to hyperbolic space (Poincaré disk)
    /// The circumplex naturally maps to a disk, making this a natural embedding
    pub fn to_hyperbolic(&self) -> LorentzVec {
        // Circumplex [-1,1]×[-1,1] maps to Poincaré disk with scaling
        let scale = 0.8; // Stay away from boundary
        let x = self.current.valence * scale;
        let y = self.current.arousal * scale;

        // Convert to hyperboloid (z=0 for 2D embedding)
        let r_sq = x * x + y * y;
        let t = (1.0 + r_sq) / (1.0 - r_sq);
        let spatial_scale = 2.0 / (1.0 - r_sq);

        LorentzVec::new(t, x * spatial_scale, y * spatial_scale, 0.0)
    }

    /// Compute geodesic distance between two affect states in hyperbolic embedding
    pub fn hyperbolic_distance(&self, other: &RussellCircumplex) -> f64 {
        let p1 = self.to_hyperbolic();
        let p2 = other.to_hyperbolic();
        p1.hyperbolic_distance(&p2)
    }
}

impl Default for RussellCircumplex {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cardiac_state() {
        let cardiac = CardiacState::default();
        assert!((cardiac.heart_rate - 70.0).abs() < 0.01);

        let arousal = cardiac.arousal();
        assert!(arousal >= 0.0 && arousal <= 1.0);
    }

    #[test]
    fn test_interoceptive_state() {
        let state = InteroceptiveState::default();
        assert!(state.arousal >= 0.0 && state.arousal <= 1.0);
        assert!(state.valence >= -1.0 && state.valence <= 1.0);

        let features = state.to_features();
        assert!(!features.is_empty());
    }

    #[test]
    fn test_hyperbolic_conversion() {
        let state = InteroceptiveState::default();
        let pos = state.to_hyperbolic();

        // Should be on hyperboloid: t² - x² - y² - z² = 1
        let minkowski_norm = pos.t * pos.t - pos.x * pos.x - pos.y * pos.y - pos.z * pos.z;
        assert!((minkowski_norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_interoceptive_inference() {
        let config = InteroceptionConfig::default();
        let mut inference = InteroceptiveInference::new(config);

        let observation = InteroceptiveState::default();
        let result = inference.process(&observation);

        assert!(result.prediction_error >= 0.0);
        assert!(result.arousal >= 0.0);
    }

    #[test]
    fn test_allostatic_regulation() {
        let baseline = InteroceptiveState::default();
        let mut regulator = AllostaticRegulator::new(baseline);

        // Create state with high arousal
        let mut current = InteroceptiveState::default();
        current.cardiac.heart_rate = 120.0; // Elevated
        current.arousal = 0.8;

        let actions = regulator.regulate(&current, &[0.5, 0.3, 0.2]);
        assert!(!actions.is_empty());
    }

    #[test]
    fn test_homeostatic_check() {
        let set_point = InteroceptiveState::default();
        let current = InteroceptiveState::default();

        assert!(current.is_homeostatic(&set_point, 0.1));

        let mut deviated = InteroceptiveState::default();
        deviated.arousal = 0.9;
        assert!(!deviated.is_homeostatic(&set_point, 0.1));
    }

    #[test]
    fn test_prediction() {
        let config = InteroceptionConfig::default();
        let mut inference = InteroceptiveInference::new(config);

        // Add some history
        for i in 0..10 {
            let mut obs = InteroceptiveState::default();
            obs.time = i as f64;
            obs.arousal = 0.3 + 0.01 * i as f64;
            inference.process(&obs);
        }

        let predicted = inference.predict(5.0);
        assert!(predicted.time > inference.current_time);
    }

    // ========================================================================
    // Russell Circumplex Tests
    // ========================================================================

    #[test]
    fn test_circumplex_emotion_angles() {
        use std::f64::consts::PI;

        // Excited is at 45°
        let excited_angle = CircumplexEmotion::Excited.canonical_angle();
        assert!((excited_angle - PI / 4.0).abs() < 1e-10);

        // Alert is at 90°
        let alert_angle = CircumplexEmotion::Alert.canonical_angle();
        assert!((alert_angle - PI / 2.0).abs() < 1e-10);

        // Tense is at 180°
        let tense_angle = CircumplexEmotion::Tense.canonical_angle();
        assert!((tense_angle - PI).abs() < 1e-10);
    }

    #[test]
    fn test_circumplex_point_polar() {
        let point = CircumplexPoint::new(0.5, 0.5);
        let (angle, intensity) = point.to_polar();

        // Should be approximately 45° with intensity ~0.707
        assert!((angle - std::f64::consts::PI / 4.0).abs() < 0.01);
        assert!((intensity - 0.707).abs() < 0.01);

        // Round trip
        let reconstructed = CircumplexPoint::from_polar(angle, intensity);
        assert!((reconstructed.valence - point.valence).abs() < 1e-10);
        assert!((reconstructed.arousal - point.arousal).abs() < 1e-10);
    }

    #[test]
    fn test_circumplex_nearest_emotion() {
        // Point near "Excited" position (high valence, high arousal)
        let excited_point = CircumplexPoint::new(0.6, 0.6);
        let (emotion, distance) = excited_point.nearest_emotion();
        assert_eq!(emotion, CircumplexEmotion::Excited);
        assert!(distance < 0.5);

        // Point near "Sad" position (low valence, low arousal)
        let sad_point = CircumplexPoint::new(-0.5, -0.5);
        let (emotion, _) = sad_point.nearest_emotion();
        assert_eq!(emotion, CircumplexEmotion::Sad);
    }

    #[test]
    fn test_circumplex_emotion_probabilities() {
        let point = CircumplexPoint::new(0.6, 0.6);
        let probs = point.emotion_probabilities(0.3);

        // Should have 12 emotions
        assert_eq!(probs.len(), 12);

        // Probabilities should sum to 1
        let sum: f64 = probs.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Excited should be most likely
        assert_eq!(probs[0].0, CircumplexEmotion::Excited);
    }

    #[test]
    fn test_circumplex_interpolation() {
        let p1 = CircumplexPoint::new(0.0, 0.0);
        let p2 = CircumplexPoint::new(1.0, 0.0);

        // Midpoint via lerp
        let mid = p1.lerp(&p2, 0.5);
        assert!((mid.valence - 0.5).abs() < 1e-10);
        assert!(mid.arousal.abs() < 1e-10);

        // Endpoints
        let start = p1.lerp(&p2, 0.0);
        assert!((start.valence - p1.valence).abs() < 1e-10);

        let end = p1.lerp(&p2, 1.0);
        assert!((end.valence - p2.valence).abs() < 1e-10);
    }

    #[test]
    fn test_russell_circumplex_model() {
        let mut circumplex = RussellCircumplex::new();

        // Update with excited state
        circumplex.update(0.8, 0.8, 0.0);
        let (emotion, _) = circumplex.current_emotion();
        assert_eq!(emotion, CircumplexEmotion::Excited);

        // Update towards sad
        for i in 1..20 {
            circumplex.update(-0.6, -0.6, i as f64);
        }
        let (emotion, _) = circumplex.current_emotion();
        assert_eq!(emotion, CircumplexEmotion::Sad);
    }

    #[test]
    fn test_circumplex_from_interoceptive() {
        let mut circumplex = RussellCircumplex::new();

        let mut state = InteroceptiveState::default();
        state.arousal = 0.9; // High arousal -> positive y in circumplex
        state.valence = 0.5; // Positive valence -> positive x

        circumplex.update_from_interoceptive(&state, 0.0);

        // Should be in upper-right quadrant
        assert!(circumplex.current.valence > 0.0);
        assert!(circumplex.current.arousal > 0.0);
    }

    #[test]
    fn test_circumplex_hyperbolic_embedding() {
        let circumplex = RussellCircumplex::with_smoothing(0.0);
        let pos = circumplex.to_hyperbolic();

        // At origin (neutral affect), should be on hyperboloid
        let norm = pos.t * pos.t - pos.x * pos.x - pos.y * pos.y - pos.z * pos.z;
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_circumplex_affect_dynamics() {
        let mut circumplex = RussellCircumplex::with_smoothing(0.0);

        // Add trajectory
        for i in 0..100 {
            let angle = (i as f64 / 100.0) * 2.0 * std::f64::consts::PI;
            let v = 0.5 * angle.cos();
            let a = 0.5 * angle.sin();
            circumplex.update(v, a, i as f64);
        }

        // Affect variability should be non-zero
        let (var_v, var_a) = circumplex.affect_variability(100.0);
        assert!(var_v > 0.0);
        assert!(var_a > 0.0);
    }
}
