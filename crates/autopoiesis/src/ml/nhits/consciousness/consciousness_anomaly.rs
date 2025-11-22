/// Consciousness Anomaly - Consciousness-Driven Anomaly Detection
///
/// This module implements anomaly detection using consciousness field patterns.
/// It identifies deviations from expected consciousness coherence and detects
/// anomalous patterns that may indicate prediction failures or novel phenomena.

use ndarray::{Array2, Array1};
use nalgebra::{DMatrix, DVector};
use std::collections::{HashMap, VecDeque};
use crate::consciousness::core::ConsciousnessState;

/// Anomaly classification types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnomalyType {
    CoherenceDrop,      // Sudden loss of consciousness coherence
    FieldDisruption,    // Quantum field disturbance
    PatternBreak,       // Break in established patterns
    EnergySpike,        // Unexpected energy surge
    PhaseShift,         // Phase transition in consciousness
    NovelPattern,       // Emergence of new patterns
    SystemFailure,      // Model prediction failure
    Unknown,            // Unclassified anomaly
}

/// Anomaly detection result
#[derive(Clone, Debug)]
pub struct AnomalyDetection {
    pub anomaly_type: AnomalyType,
    pub severity: f64,          // 0.0 to 1.0
    pub confidence: f64,        // Detection confidence
    pub location: Array1<f64>,  // Where in feature space
    pub time_detected: f64,
    pub consciousness_impact: f64,
    pub field_signature: Array1<f64>,
    pub recovery_prediction: Option<f64>, // Predicted recovery time
}

impl AnomalyDetection {
    pub fn new(anomaly_type: AnomalyType, severity: f64, location: Array1<f64>, time: f64) -> Self {
        Self {
            anomaly_type,
            severity: severity.clamp(0.0, 1.0),
            confidence: 0.5,
            location,
            time_detected: time,
            consciousness_impact: 0.0,
            field_signature: Array1::zeros(location.len()),
            recovery_prediction: None,
        }
    }
}

/// Consciousness-aware anomaly detector
pub struct ConsciousnessAnomaly {
    pub normal_patterns: HashMap<String, NormalPattern>,
    pub anomaly_history: VecDeque<AnomalyDetection>,
    pub consciousness_baseline: ConsciousnessBaseline,
    pub field_monitors: Vec<FieldMonitor>,
    pub detection_thresholds: DetectionThresholds,
    pub adaptation_parameters: AdaptationParameters,
    pub input_dimension: usize,
    pub current_time: f64,
}

impl ConsciousnessAnomaly {
    pub fn new(input_dimension: usize) -> Self {
        let num_monitors = 8;
        let mut field_monitors = Vec::with_capacity(num_monitors);
        
        for i in 0..num_monitors {
            field_monitors.push(FieldMonitor::new(i, input_dimension));
        }
        
        Self {
            normal_patterns: HashMap::new(),
            anomaly_history: VecDeque::with_capacity(1000),
            consciousness_baseline: ConsciousnessBaseline::new(),
            field_monitors,
            detection_thresholds: DetectionThresholds::default(),
            adaptation_parameters: AdaptationParameters::default(),
            input_dimension,
            current_time: 0.0,
        }
    }
    
    /// Detect anomalies in consciousness patterns
    pub fn detect_anomalies(&mut self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) -> Vec<AnomalyDetection> {
        self.current_time += 1.0;
        let mut detected_anomalies = Vec::new();
        
        // Update consciousness baseline
        self.consciousness_baseline.update(consciousness);
        
        // Update field monitors
        for monitor in &mut self.field_monitors {
            monitor.update(prediction, consciousness);
        }
        
        // Check for consciousness coherence anomalies
        if let Some(coherence_anomaly) = self.detect_coherence_anomaly(consciousness) {
            detected_anomalies.push(coherence_anomaly);
        }
        
        // Check for field disruption anomalies
        if let Some(field_anomaly) = self.detect_field_disruption(prediction, consciousness) {
            detected_anomalies.push(field_anomaly);
        }
        
        // Check for pattern break anomalies
        if let Some(pattern_anomaly) = self.detect_pattern_break(prediction, consciousness) {
            detected_anomalies.push(pattern_anomaly);
        }
        
        // Check for energy spike anomalies
        if let Some(energy_anomaly) = self.detect_energy_spike(prediction, consciousness) {
            detected_anomalies.push(energy_anomaly);
        }
        
        // Check for phase shift anomalies
        if let Some(phase_anomaly) = self.detect_phase_shift(consciousness) {
            detected_anomalies.push(phase_anomaly);
        }
        
        // Check for novel pattern emergence
        if let Some(novel_anomaly) = self.detect_novel_patterns(prediction, consciousness) {
            detected_anomalies.push(novel_anomaly);
        }
        
        // Store detected anomalies
        for mut anomaly in detected_anomalies.iter().cloned() {
            anomaly.consciousness_impact = self.compute_consciousness_impact(&anomaly, consciousness);
            anomaly.field_signature = self.compute_field_signature(&anomaly, prediction);
            anomaly.recovery_prediction = self.predict_recovery_time(&anomaly);
            
            self.anomaly_history.push_back(anomaly);
        }
        
        // Maintain anomaly history size
        if self.anomaly_history.len() > 1000 {
            self.anomaly_history.pop_front();
        }
        
        // Update normal patterns if no anomalies
        if detected_anomalies.is_empty() {
            self.update_normal_patterns(prediction, consciousness);
        }
        
        // Adapt detection parameters
        self.adapt_detection_parameters(&detected_anomalies, consciousness);
        
        detected_anomalies
    }
    
    /// Detect consciousness coherence anomalies
    fn detect_coherence_anomaly(&self, consciousness: &ConsciousnessState) -> Option<AnomalyDetection> {
        let coherence_deviation = (consciousness.coherence_level - self.consciousness_baseline.mean_coherence).abs();
        
        if coherence_deviation > self.detection_thresholds.coherence_threshold {
            let severity = (coherence_deviation / self.detection_thresholds.coherence_threshold).min(1.0);
            let location = Array1::from_vec(vec![consciousness.coherence_level, consciousness.field_coherence]);
            
            let mut anomaly = AnomalyDetection::new(AnomalyType::CoherenceDrop, severity, location, self.current_time);
            anomaly.confidence = self.compute_detection_confidence(severity, &self.consciousness_baseline.coherence_history);
            
            Some(anomaly)
        } else {
            None
        }
    }
    
    /// Detect field disruption anomalies
    fn detect_field_disruption(&self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) -> Option<AnomalyDetection> {
        let field_disruption_score = self.compute_field_disruption_score(prediction, consciousness);
        
        if field_disruption_score > self.detection_thresholds.field_disruption_threshold {
            let severity = (field_disruption_score / self.detection_thresholds.field_disruption_threshold).min(1.0);
            let location = prediction.clone();
            
            let mut anomaly = AnomalyDetection::new(AnomalyType::FieldDisruption, severity, location, self.current_time);
            anomaly.confidence = self.compute_field_disruption_confidence(field_disruption_score);
            
            Some(anomaly)
        } else {
            None
        }
    }
    
    /// Compute field disruption score
    fn compute_field_disruption_score(&self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) -> f64 {
        let mut disruption_score = 0.0;
        
        // Check field monitors for disruptions
        for monitor in &self.field_monitors {
            let monitor_disruption = monitor.compute_disruption_score(prediction, consciousness);
            disruption_score = disruption_score.max(monitor_disruption);
        }
        
        // Add consciousness field coherence disruption
        let field_coherence_deviation = (consciousness.field_coherence - self.consciousness_baseline.mean_field_coherence).abs();
        disruption_score += field_coherence_deviation * 0.5;
        
        disruption_score
    }
    
    /// Detect pattern break anomalies
    fn detect_pattern_break(&self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) -> Option<AnomalyDetection> {
        let pattern_break_score = self.compute_pattern_break_score(prediction, consciousness);
        
        if pattern_break_score > self.detection_thresholds.pattern_break_threshold {
            let severity = (pattern_break_score / self.detection_thresholds.pattern_break_threshold).min(1.0);
            let location = prediction.clone();
            
            let mut anomaly = AnomalyDetection::new(AnomalyType::PatternBreak, severity, location, self.current_time);
            anomaly.confidence = self.compute_pattern_break_confidence(pattern_break_score);
            
            Some(anomaly)
        } else {
            None
        }
    }
    
    /// Compute pattern break score
    fn compute_pattern_break_score(&self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) -> f64 {
        let mut max_break_score = 0.0;
        
        // Check against known normal patterns
        for (_, normal_pattern) in &self.normal_patterns {
            let pattern_deviation = normal_pattern.compute_deviation(prediction, consciousness);
            max_break_score = max_break_score.max(pattern_deviation);
        }
        
        // If no normal patterns, use baseline deviation
        if self.normal_patterns.is_empty() {
            let baseline_deviation = self.compute_baseline_deviation(prediction);
            max_break_score = baseline_deviation;
        }
        
        max_break_score
    }
    
    /// Detect energy spike anomalies
    fn detect_energy_spike(&self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) -> Option<AnomalyDetection> {
        let energy_level = self.compute_energy_level(prediction, consciousness);
        let energy_spike_score = energy_level - self.consciousness_baseline.mean_energy;
        
        if energy_spike_score > self.detection_thresholds.energy_spike_threshold {
            let severity = (energy_spike_score / self.detection_thresholds.energy_spike_threshold).min(1.0);
            let location = Array1::from_vec(vec![energy_level]);
            
            let mut anomaly = AnomalyDetection::new(AnomalyType::EnergySpike, severity, location, self.current_time);
            anomaly.confidence = self.compute_energy_spike_confidence(energy_spike_score);
            
            Some(anomaly)
        } else {
            None
        }
    }
    
    /// Compute energy level from prediction and consciousness
    fn compute_energy_level(&self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) -> f64 {
        let prediction_energy = prediction.mapv(|x| x * x).sum();
        let consciousness_energy = consciousness.coherence_level * consciousness.field_coherence;
        
        prediction_energy * consciousness_energy
    }
    
    /// Detect phase shift anomalies
    fn detect_phase_shift(&self, consciousness: &ConsciousnessState) -> Option<AnomalyDetection> {
        let phase_shift_score = self.compute_phase_shift_score(consciousness);
        
        if phase_shift_score > self.detection_thresholds.phase_shift_threshold {
            let severity = (phase_shift_score / self.detection_thresholds.phase_shift_threshold).min(1.0);
            let location = Array1::from_vec(vec![consciousness.coherence_level, consciousness.field_coherence]);
            
            let mut anomaly = AnomalyDetection::new(AnomalyType::PhaseShift, severity, location, self.current_time);
            anomaly.confidence = self.compute_phase_shift_confidence(phase_shift_score);
            
            Some(anomaly)
        } else {
            None
        }
    }
    
    /// Compute phase shift score
    fn compute_phase_shift_score(&self, consciousness: &ConsciousnessState) -> f64 {
        if self.consciousness_baseline.coherence_history.len() < 5 {
            return 0.0;
        }
        
        // Compute phase shift based on recent consciousness history
        let recent_coherence: Vec<f64> = self.consciousness_baseline.coherence_history.iter()
            .rev()
            .take(5)
            .cloned()
            .collect();
        
        let mut phase_changes = 0.0;
        for i in 1..recent_coherence.len() {
            let change = (recent_coherence[i] - recent_coherence[i-1]).abs();
            phase_changes += change;
        }
        
        let avg_phase_change = phase_changes / (recent_coherence.len() - 1) as f64;
        let current_change = (consciousness.coherence_level - recent_coherence[0]).abs();
        
        // Phase shift is detected when current change significantly exceeds average
        if avg_phase_change > 0.0 {
            current_change / avg_phase_change - 1.0
        } else {
            current_change
        }
    }
    
    /// Detect novel pattern emergence
    fn detect_novel_patterns(&mut self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) -> Option<AnomalyDetection> {
        let novelty_score = self.compute_novelty_score(prediction, consciousness);
        
        if novelty_score > self.detection_thresholds.novelty_threshold {
            let severity = (novelty_score / self.detection_thresholds.novelty_threshold).min(1.0);
            let location = prediction.clone();
            
            let mut anomaly = AnomalyDetection::new(AnomalyType::NovelPattern, severity, location, self.current_time);
            anomaly.confidence = self.compute_novelty_confidence(novelty_score);
            
            // Learn this as a new normal pattern if confidence is high
            if anomaly.confidence > 0.8 {
                self.add_novel_pattern(prediction, consciousness);
            }
            
            Some(anomaly)
        } else {
            None
        }
    }
    
    /// Compute novelty score for pattern
    fn compute_novelty_score(&self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) -> f64 {
        let mut min_similarity = f64::INFINITY;
        
        // Compare with all known normal patterns
        for (_, normal_pattern) in &self.normal_patterns {
            let similarity = normal_pattern.compute_similarity(prediction, consciousness);
            min_similarity = min_similarity.min(similarity);
        }
        
        // Novelty is inverse of similarity
        if min_similarity.is_finite() {
            1.0 / (1.0 + min_similarity)
        } else {
            1.0 // Completely novel if no patterns exist
        }
    }
    
    /// Add novel pattern to normal patterns
    fn add_novel_pattern(&mut self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) {
        let pattern_id = format!("novel_pattern_{}", self.current_time as u64);
        let normal_pattern = NormalPattern::new(prediction.clone(), consciousness.clone());
        
        self.normal_patterns.insert(pattern_id, normal_pattern);
        
        // Limit number of stored patterns
        if self.normal_patterns.len() > 100 {
            // Remove oldest patterns (simplified removal)
            let keys_to_remove: Vec<String> = self.normal_patterns.keys()
                .take(10)
                .cloned()
                .collect();
            
            for key in keys_to_remove {
                self.normal_patterns.remove(&key);
            }
        }
    }
    
    /// Compute consciousness impact of anomaly
    fn compute_consciousness_impact(&self, anomaly: &AnomalyDetection, consciousness: &ConsciousnessState) -> f64 {
        let impact_factor = match anomaly.anomaly_type {
            AnomalyType::CoherenceDrop => 0.9,
            AnomalyType::FieldDisruption => 0.8,
            AnomalyType::PatternBreak => 0.6,
            AnomalyType::EnergySpike => 0.7,
            AnomalyType::PhaseShift => 0.8,
            AnomalyType::NovelPattern => 0.3,
            AnomalyType::SystemFailure => 1.0,
            AnomalyType::Unknown => 0.5,
        };
        
        anomaly.severity * impact_factor * (1.0 - consciousness.coherence_level * consciousness.field_coherence)
    }
    
    /// Compute field signature of anomaly
    fn compute_field_signature(&self, anomaly: &AnomalyDetection, prediction: &Array1<f64>) -> Array1<f64> {
        let mut signature = Array1::zeros(prediction.len());
        
        // Create signature based on anomaly type and location
        for (i, &val) in prediction.iter().enumerate() {
            let phase = (i as f64 / prediction.len() as f64) * 2.0 * std::f64::consts::PI;
            let anomaly_phase = match anomaly.anomaly_type {
                AnomalyType::CoherenceDrop => phase,
                AnomalyType::FieldDisruption => phase * 2.0,
                AnomalyType::PatternBreak => phase * 0.5,
                AnomalyType::EnergySpike => phase * 3.0,
                AnomalyType::PhaseShift => phase + std::f64::consts::PI/2.0,
                AnomalyType::NovelPattern => phase * 1.5,
                AnomalyType::SystemFailure => phase * 4.0,
                AnomalyType::Unknown => phase,
            };
            
            signature[i] = val * (anomaly_phase * anomaly.severity).sin();
        }
        
        signature
    }
    
    /// Predict recovery time for anomaly
    fn predict_recovery_time(&self, anomaly: &AnomalyDetection) -> Option<f64> {
        // Base recovery time depends on anomaly type and severity
        let base_recovery_time = match anomaly.anomaly_type {
            AnomalyType::CoherenceDrop => 5.0 + anomaly.severity * 10.0,
            AnomalyType::FieldDisruption => 3.0 + anomaly.severity * 8.0,
            AnomalyType::PatternBreak => 2.0 + anomaly.severity * 5.0,
            AnomalyType::EnergySpike => 1.0 + anomaly.severity * 3.0,
            AnomalyType::PhaseShift => 8.0 + anomaly.severity * 15.0,
            AnomalyType::NovelPattern => None.map_or(0.0, |x| x), // Novel patterns may not "recover"
            AnomalyType::SystemFailure => 20.0 + anomaly.severity * 30.0,
            AnomalyType::Unknown => 10.0,
        };
        
        if anomaly.anomaly_type == AnomalyType::NovelPattern {
            None
        } else {
            Some(base_recovery_time)
        }
    }
    
    /// Update normal patterns during non-anomalous periods
    fn update_normal_patterns(&mut self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) {
        // Update existing patterns
        for pattern in self.normal_patterns.values_mut() {
            pattern.update(prediction, consciousness);
        }
        
        // Add new pattern if we have few patterns
        if self.normal_patterns.len() < 10 {
            let pattern_id = format!("normal_pattern_{}", self.current_time as u64);
            let normal_pattern = NormalPattern::new(prediction.clone(), consciousness.clone());
            self.normal_patterns.insert(pattern_id, normal_pattern);
        }
    }
    
    /// Adapt detection parameters based on recent anomalies
    fn adapt_detection_parameters(&mut self, recent_anomalies: &[AnomalyDetection], consciousness: &ConsciousnessState) {
        let adaptation_rate = self.adaptation_parameters.adaptation_rate;
        
        // Adapt thresholds based on anomaly frequency
        if recent_anomalies.len() > 2 {
            // Too many anomalies - raise thresholds
            self.detection_thresholds.coherence_threshold *= 1.0 + adaptation_rate;
            self.detection_thresholds.field_disruption_threshold *= 1.0 + adaptation_rate;
            self.detection_thresholds.pattern_break_threshold *= 1.0 + adaptation_rate;
        } else if recent_anomalies.is_empty() && consciousness.coherence_level > 0.8 {
            // No anomalies and high coherence - lower thresholds for sensitivity
            self.detection_thresholds.coherence_threshold *= 1.0 - adaptation_rate * 0.5;
            self.detection_thresholds.field_disruption_threshold *= 1.0 - adaptation_rate * 0.5;
            self.detection_thresholds.pattern_break_threshold *= 1.0 - adaptation_rate * 0.5;
        }
        
        // Bound thresholds
        self.detection_thresholds.bound_thresholds();
    }
    
    /// Compute detection confidence
    fn compute_detection_confidence(&self, severity: f64, history: &VecDeque<f64>) -> f64 {
        let base_confidence = severity.clamp(0.0, 1.0);
        
        // Increase confidence if we have historical context
        let history_bonus = if history.len() > 10 {
            0.2
        } else {
            0.0
        };
        
        (base_confidence + history_bonus).min(1.0)
    }
    
    /// Compute field disruption confidence
    fn compute_field_disruption_confidence(&self, disruption_score: f64) -> f64 {
        (disruption_score / (1.0 + disruption_score)).clamp(0.0, 1.0)
    }
    
    /// Compute pattern break confidence
    fn compute_pattern_break_confidence(&self, break_score: f64) -> f64 {
        (break_score / (2.0 + break_score)).clamp(0.0, 1.0)
    }
    
    /// Compute energy spike confidence
    fn compute_energy_spike_confidence(&self, spike_score: f64) -> f64 {
        (spike_score.abs() / (1.0 + spike_score.abs())).clamp(0.0, 1.0)
    }
    
    /// Compute phase shift confidence
    fn compute_phase_shift_confidence(&self, shift_score: f64) -> f64 {
        (shift_score / (3.0 + shift_score)).clamp(0.0, 1.0)
    }
    
    /// Compute novelty confidence
    fn compute_novelty_confidence(&self, novelty_score: f64) -> f64 {
        (novelty_score / (1.5 + novelty_score)).clamp(0.0, 1.0)
    }
    
    /// Compute baseline deviation
    fn compute_baseline_deviation(&self, prediction: &Array1<f64>) -> f64 {
        // Simple baseline using prediction magnitude
        let magnitude = prediction.mapv(|x| x.abs()).sum();
        
        if self.consciousness_baseline.mean_prediction_magnitude > 0.0 {
            (magnitude - self.consciousness_baseline.mean_prediction_magnitude).abs() / 
            self.consciousness_baseline.mean_prediction_magnitude
        } else {
            magnitude
        }
    }
    
    /// Get anomaly detection statistics
    pub fn get_anomaly_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("total_anomalies".to_string(), self.anomaly_history.len() as f64);
        stats.insert("normal_patterns".to_string(), self.normal_patterns.len() as f64);
        stats.insert("current_time".to_string(), self.current_time);
        
        // Anomaly type distribution
        let mut type_counts = HashMap::new();
        for anomaly in &self.anomaly_history {
            let type_name = format!("{:?}", anomaly.anomaly_type);
            *type_counts.entry(type_name).or_insert(0) += 1;
        }
        
        for (anomaly_type, count) in type_counts {
            stats.insert(format!("anomalies_{}", anomaly_type.to_lowercase()), count as f64);
        }
        
        // Recent anomaly statistics
        let recent_anomalies: Vec<&AnomalyDetection> = self.anomaly_history.iter()
            .rev()
            .take(10)
            .collect();
        
        if !recent_anomalies.is_empty() {
            let avg_recent_severity: f64 = recent_anomalies.iter()
                .map(|a| a.severity)
                .sum::<f64>() / recent_anomalies.len() as f64;
            
            let avg_recent_confidence: f64 = recent_anomalies.iter()
                .map(|a| a.confidence)
                .sum::<f64>() / recent_anomalies.len() as f64;
            
            stats.insert("avg_recent_severity".to_string(), avg_recent_severity);
            stats.insert("avg_recent_confidence".to_string(), avg_recent_confidence);
        }
        
        // Detection thresholds
        stats.insert("coherence_threshold".to_string(), self.detection_thresholds.coherence_threshold);
        stats.insert("field_disruption_threshold".to_string(), self.detection_thresholds.field_disruption_threshold);
        stats.insert("pattern_break_threshold".to_string(), self.detection_thresholds.pattern_break_threshold);
        
        stats
    }
}

/// Normal pattern storage and matching
#[derive(Clone)]
struct NormalPattern {
    pub pattern_center: Array1<f64>,
    pub consciousness_center: ConsciousnessState,
    pub pattern_variance: Array1<f64>,
    pub update_count: usize,
    pub creation_time: f64,
}

impl NormalPattern {
    fn new(pattern: Array1<f64>, consciousness: ConsciousnessState) -> Self {
        let pattern_variance = Array1::ones(pattern.len()) * 0.1; // Initial variance
        
        Self {
            pattern_center: pattern,
            consciousness_center: consciousness,
            pattern_variance,
            update_count: 1,
            creation_time: 0.0, // Should be set by caller
        }
    }
    
    fn update(&mut self, pattern: &Array1<f64>, consciousness: &ConsciousnessState) {
        let learning_rate = 1.0 / (self.update_count as f64 + 1.0);
        
        // Update pattern center
        self.pattern_center = &self.pattern_center * (1.0 - learning_rate) + pattern * learning_rate;
        
        // Update consciousness center
        self.consciousness_center.coherence_level = 
            self.consciousness_center.coherence_level * (1.0 - learning_rate) + 
            consciousness.coherence_level * learning_rate;
        
        self.consciousness_center.field_coherence = 
            self.consciousness_center.field_coherence * (1.0 - learning_rate) + 
            consciousness.field_coherence * learning_rate;
        
        // Update variance
        let pattern_diff = pattern - &self.pattern_center;
        let squared_diff = pattern_diff.mapv(|x| x * x);
        self.pattern_variance = &self.pattern_variance * (1.0 - learning_rate) + &squared_diff * learning_rate;
        
        self.update_count += 1;
    }
    
    fn compute_deviation(&self, pattern: &Array1<f64>, consciousness: &ConsciousnessState) -> f64 {
        let pattern_deviation = self.compute_pattern_deviation(pattern);
        let consciousness_deviation = self.compute_consciousness_deviation(consciousness);
        
        (pattern_deviation + consciousness_deviation) / 2.0
    }
    
    fn compute_similarity(&self, pattern: &Array1<f64>, consciousness: &ConsciousnessState) -> f64 {
        let deviation = self.compute_deviation(pattern, consciousness);
        1.0 / (1.0 + deviation)
    }
    
    fn compute_pattern_deviation(&self, pattern: &Array1<f64>) -> f64 {
        if pattern.len() != self.pattern_center.len() {
            return f64::INFINITY;
        }
        
        let mut total_deviation = 0.0;
        
        for i in 0..pattern.len() {
            let diff = pattern[i] - self.pattern_center[i];
            let normalized_diff = diff / (self.pattern_variance[i].sqrt() + 1e-8);
            total_deviation += normalized_diff * normalized_diff;
        }
        
        (total_deviation / pattern.len() as f64).sqrt()
    }
    
    fn compute_consciousness_deviation(&self, consciousness: &ConsciousnessState) -> f64 {
        let coherence_diff = (consciousness.coherence_level - self.consciousness_center.coherence_level).abs();
        let field_coherence_diff = (consciousness.field_coherence - self.consciousness_center.field_coherence).abs();
        
        (coherence_diff + field_coherence_diff) / 2.0
    }
}

/// Consciousness baseline tracking
#[derive(Clone)]
struct ConsciousnessBaseline {
    pub mean_coherence: f64,
    pub mean_field_coherence: f64,
    pub mean_energy: f64,
    pub mean_prediction_magnitude: f64,
    pub coherence_history: VecDeque<f64>,
    pub update_count: usize,
}

impl ConsciousnessBaseline {
    fn new() -> Self {
        Self {
            mean_coherence: 0.5,
            mean_field_coherence: 0.5,
            mean_energy: 1.0,
            mean_prediction_magnitude: 1.0,
            coherence_history: VecDeque::with_capacity(100),
            update_count: 0,
        }
    }
    
    fn update(&mut self, consciousness: &ConsciousnessState) {
        let learning_rate = 1.0 / (self.update_count as f64 + 1.0);
        
        self.mean_coherence = self.mean_coherence * (1.0 - learning_rate) + 
                            consciousness.coherence_level * learning_rate;
        
        self.mean_field_coherence = self.mean_field_coherence * (1.0 - learning_rate) + 
                                  consciousness.field_coherence * learning_rate;
        
        self.coherence_history.push_back(consciousness.coherence_level);
        if self.coherence_history.len() > 100 {
            self.coherence_history.pop_front();
        }
        
        self.update_count += 1;
    }
}

/// Field monitor for detecting field-based anomalies
#[derive(Clone)]
struct FieldMonitor {
    pub monitor_id: usize,
    pub sensitivity_pattern: Array1<f64>,
    pub baseline_field: Array1<f64>,
    pub disruption_history: VecDeque<f64>,
    pub adaptation_rate: f64,
}

impl FieldMonitor {
    fn new(monitor_id: usize, dimension: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let sensitivity_pattern = Array1::from_shape_fn(dimension, |i| {
            let phase = (i as f64 / dimension as f64 + monitor_id as f64 / 8.0) * 2.0 * std::f64::consts::PI;
            phase.sin() * 0.1
        });
        
        Self {
            monitor_id,
            sensitivity_pattern,
            baseline_field: Array1::zeros(dimension),
            disruption_history: VecDeque::with_capacity(50),
            adaptation_rate: 0.01,
        }
    }
    
    fn update(&mut self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) {
        // Update baseline field
        let learning_rate = self.adaptation_rate;
        self.baseline_field = &self.baseline_field * (1.0 - learning_rate) + 
                            prediction * learning_rate;
        
        // Compute and store disruption score
        let disruption = self.compute_disruption_score(prediction, consciousness);
        self.disruption_history.push_back(disruption);
        if self.disruption_history.len() > 50 {
            self.disruption_history.pop_front();
        }
    }
    
    fn compute_disruption_score(&self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) -> f64 {
        if prediction.len() != self.baseline_field.len() {
            return 0.0;
        }
        
        // Compute field deviation weighted by sensitivity pattern
        let field_deviation = prediction - &self.baseline_field;
        let weighted_deviation = &field_deviation * &self.sensitivity_pattern;
        let disruption_magnitude = weighted_deviation.mapv(|x| x.abs()).sum();
        
        // Modulate by consciousness coherence
        disruption_magnitude * (1.0 - consciousness.field_coherence * 0.5)
    }
}

/// Detection thresholds for different anomaly types
#[derive(Clone)]
struct DetectionThresholds {
    pub coherence_threshold: f64,
    pub field_disruption_threshold: f64,
    pub pattern_break_threshold: f64,
    pub energy_spike_threshold: f64,
    pub phase_shift_threshold: f64,
    pub novelty_threshold: f64,
}

impl Default for DetectionThresholds {
    fn default() -> Self {
        Self {
            coherence_threshold: 0.2,
            field_disruption_threshold: 0.3,
            pattern_break_threshold: 0.4,
            energy_spike_threshold: 2.0,
            phase_shift_threshold: 0.5,
            novelty_threshold: 0.7,
        }
    }
}

impl DetectionThresholds {
    fn bound_thresholds(&mut self) {
        self.coherence_threshold = self.coherence_threshold.clamp(0.05, 0.8);
        self.field_disruption_threshold = self.field_disruption_threshold.clamp(0.1, 1.0);
        self.pattern_break_threshold = self.pattern_break_threshold.clamp(0.1, 1.0);
        self.energy_spike_threshold = self.energy_spike_threshold.clamp(0.5, 10.0);
        self.phase_shift_threshold = self.phase_shift_threshold.clamp(0.1, 2.0);
        self.novelty_threshold = self.novelty_threshold.clamp(0.3, 0.9);
    }
}

/// Adaptation parameters for anomaly detection
#[derive(Clone)]
struct AdaptationParameters {
    pub adaptation_rate: f64,
    pub sensitivity_decay: f64,
    pub confidence_threshold: f64,
}

impl Default for AdaptationParameters {
    fn default() -> Self {
        Self {
            adaptation_rate: 0.01,
            sensitivity_decay: 0.99,
            confidence_threshold: 0.7,
        }
    }
}