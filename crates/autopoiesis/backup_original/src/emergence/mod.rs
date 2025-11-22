//! Emergence module for detecting and analyzing emergent behaviors in swarm systems
//! 
//! This module provides sophisticated tools for identifying, classifying, and predicting
//! emergent phenomena in complex dynamical systems. It includes emergence detection
//! algorithms and temporal pattern recognition capabilities.

pub mod detector;
pub mod patterns;

// Re-export key types for convenient access
pub use detector::{
    EmergenceDetector,
    DetectionParameters,
    EmergenceState,
    EmergenceType,
    EmergenceAlert,
    EmergenceHistory,
    SystemMetrics,
};

pub use patterns::{
    TemporalPatternRecognizer,
    PatternParameters,
    TemporalPattern,
    PatternType,
    PatternStatistics,
    PatternPrediction,
    PatternHistory,
    PatternOccurrence,
};

/// Integrated emergence analysis system that combines detection and pattern recognition
/// for comprehensive understanding of emergent behaviors
pub struct EmergenceAnalysisSystem {
    /// Emergence detector
    pub detector: EmergenceDetector,
    /// Pattern recognizer
    pub pattern_recognizer: TemporalPatternRecognizer,
    /// Analysis parameters
    params: AnalysisParameters,
    /// Analysis history
    analysis_history: AnalysisHistory,
    /// Current analysis state
    current_analysis: CurrentAnalysis,
}

#[derive(Clone, Debug)]
pub struct AnalysisParameters {
    /// Detection sensitivity
    pub detection_sensitivity: f64,
    /// Pattern recognition depth
    pub pattern_depth: usize,
    /// Prediction horizon
    pub prediction_horizon: usize,
    /// Analysis update frequency
    pub update_frequency: usize,
    /// Memory retention period
    pub memory_retention: usize,
    /// Cross-validation enabled
    pub cross_validation: bool,
}

#[derive(Clone, Debug)]
pub struct AnalysisHistory {
    /// Historical emergence scores
    pub emergence_scores: std::collections::VecDeque<f64>,
    /// Pattern strength evolution
    pub pattern_strengths: std::collections::VecDeque<f64>,
    /// Prediction accuracy tracking
    pub prediction_accuracy: std::collections::VecDeque<f64>,
    /// Alert frequency
    pub alert_frequency: std::collections::VecDeque<usize>,
}

#[derive(Clone, Debug)]
pub struct CurrentAnalysis {
    /// Current emergence score
    pub emergence_score: f64,
    /// Active patterns
    pub active_patterns: Vec<String>,
    /// Confidence level
    pub confidence: f64,
    /// Trend direction
    pub trend: TrendDirection,
    /// Risk assessment
    pub risk_level: RiskLevel,
    /// Recommendations
    pub recommendations: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for AnalysisParameters {
    fn default() -> Self {
        Self {
            detection_sensitivity: 0.7,
            pattern_depth: 5,
            prediction_horizon: 50,
            update_frequency: 10,
            memory_retention: 1000,
            cross_validation: true,
        }
    }
}

impl EmergenceAnalysisSystem {
    /// Create new emergence analysis system
    pub fn new(
        detection_params: DetectionParameters,
        pattern_params: PatternParameters,
        analysis_params: AnalysisParameters,
    ) -> Self {
        let detector = EmergenceDetector::new(detection_params);
        let pattern_recognizer = TemporalPatternRecognizer::new(pattern_params);

        let analysis_history = AnalysisHistory {
            emergence_scores: std::collections::VecDeque::with_capacity(analysis_params.memory_retention),
            pattern_strengths: std::collections::VecDeque::with_capacity(analysis_params.memory_retention),
            prediction_accuracy: std::collections::VecDeque::with_capacity(analysis_params.memory_retention),
            alert_frequency: std::collections::VecDeque::with_capacity(analysis_params.memory_retention),
        };

        let current_analysis = CurrentAnalysis {
            emergence_score: 0.0,
            active_patterns: Vec::new(),
            confidence: 0.0,
            trend: TrendDirection::Unknown,
            risk_level: RiskLevel::Low,
            recommendations: Vec::new(),
        };

        Self {
            detector,
            pattern_recognizer,
            params: analysis_params,
            analysis_history,
            current_analysis,
        }
    }

    /// Perform comprehensive emergence analysis
    pub fn analyze_emergence(&mut self, history: &EmergenceHistory) -> AnalysisResult {
        // Update emergence detection
        let emergence_state = self.detector.get_emergence_state();
        
        // Update pattern recognition
        let discovered_patterns = self.pattern_recognizer.analyze_patterns(history);
        
        // Combine results for comprehensive analysis
        let combined_analysis = self.combine_analysis_results(emergence_state, &discovered_patterns);
        
        // Generate predictions
        let predictions = self.generate_predictions();
        
        // Update analysis history
        self.update_analysis_history(&combined_analysis);
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&combined_analysis);
        
        AnalysisResult {
            emergence_state: emergence_state.clone(),
            patterns: discovered_patterns,
            predictions,
            combined_analysis,
            recommendations,
            confidence: self.calculate_overall_confidence(),
        }
    }

    /// Update analysis from integrated dynamics system
    pub fn update_from_dynamics(&mut self, dynamics: &crate::dynamics::IntegratedDynamicsSystem) {
        let integrated_state = dynamics.get_integrated_state();
        
        // Extract metrics for analysis
        let metrics = SystemMetrics {
            timestamp: integrated_state.time_step as f64,
            system_size: integrated_state.soc_state.grid_snapshot.len(),
            total_energy: integrated_state.soc_state.total_energy + integrated_state.lattice_state.total_energy,
            entropy: 0.0, // Would be calculated from actual data
            information: integrated_state.system_complexity,
            complexity: integrated_state.system_complexity,
            coherence: integrated_state.system_coherence,
            coupling: integrated_state.coupling_strength,
        };

        // Create temporary history for analysis
        let mut temp_history = EmergenceHistory {
            metrics_history: std::collections::VecDeque::new(),
            phase_trajectories: std::collections::VecDeque::new(),
            avalanche_events: std::collections::VecDeque::new(),
            fitness_evolution: std::collections::VecDeque::new(),
            lattice_states: std::collections::VecDeque::new(),
        };
        
        temp_history.metrics_history.push_back(metrics);
        
        // Perform analysis
        self.analyze_emergence(&temp_history);
    }

    /// Combine emergence detection and pattern recognition results
    fn combine_analysis_results(
        &mut self,
        emergence_state: &EmergenceState,
        patterns: &[TemporalPattern],
    ) -> CombinedAnalysis {
        // Calculate pattern strength
        let pattern_strength = if patterns.is_empty() {
            0.0
        } else {
            patterns.iter().map(|p| p.strength).sum::<f64>() / patterns.len() as f64
        };

        // Determine overall emergence level
        let emergence_level = self.classify_emergence_level(
            emergence_state.emergence_score,
            pattern_strength,
        );

        // Identify dominant patterns
        let mut dominant_patterns = patterns.iter()
            .filter(|p| p.strength > 0.7)
            .map(|p| p.id.clone())
            .collect::<Vec<_>>();
        dominant_patterns.sort();

        // Calculate temporal stability
        let temporal_stability = self.calculate_temporal_stability(emergence_state, patterns);

        // Assess criticality
        let criticality = self.assess_criticality(emergence_state, patterns);

        CombinedAnalysis {
            emergence_level,
            pattern_strength,
            dominant_patterns,
            temporal_stability,
            criticality,
            emergence_types: emergence_state.emergence_types.clone(),
            confidence: emergence_state.confidence,
        }
    }

    /// Classify overall emergence level
    fn classify_emergence_level(&self, emergence_score: f64, pattern_strength: f64) -> EmergenceLevel {
        let combined_score = (emergence_score + pattern_strength) / 2.0;
        
        if combined_score > 0.8 {
            EmergenceLevel::High
        } else if combined_score > 0.6 {
            EmergenceLevel::Medium
        } else if combined_score > 0.3 {
            EmergenceLevel::Low
        } else {
            EmergenceLevel::Minimal
        }
    }

    /// Calculate temporal stability
    fn calculate_temporal_stability(&self, emergence_state: &EmergenceState, patterns: &[TemporalPattern]) -> f64 {
        // Stability based on emergence persistence and pattern regularity
        let emergence_stability = emergence_state.temporal_stability;
        
        let pattern_stability = if patterns.is_empty() {
            0.5 // Neutral when no patterns
        } else {
            patterns.iter()
                .filter(|p| matches!(p.pattern_type, PatternType::Periodic | PatternType::QuasiPeriodic))
                .map(|p| p.strength)
                .sum::<f64>() / patterns.len() as f64
        };

        (emergence_stability + pattern_stability) / 2.0
    }

    /// Assess system criticality
    fn assess_criticality(&self, emergence_state: &EmergenceState, patterns: &[TemporalPattern]) -> f64 {
        let mut criticality = 0.0;

        // High emergence score increases criticality
        criticality += emergence_state.emergence_score * 0.4;

        // Critical behavior patterns increase criticality
        let critical_patterns = patterns.iter()
            .filter(|p| matches!(p.pattern_type, PatternType::Critical | PatternType::Chaotic))
            .count();
        criticality += (critical_patterns as f64 / patterns.len().max(1) as f64) * 0.3;

        // Phase transitions increase criticality
        if emergence_state.emergence_types.contains(&EmergenceType::PhaseTransition) {
            criticality += 0.3;
        }

        criticality.min(1.0)
    }

    /// Generate predictions based on current analysis
    fn generate_predictions(&self) -> Vec<EmergencePrediction> {
        let mut predictions = Vec::new();

        // Pattern-based predictions
        let pattern_predictions = self.pattern_recognizer.predict_patterns(self.params.prediction_horizon);
        
        for pattern_pred in pattern_predictions {
            let emergence_prediction = EmergencePrediction {
                prediction_type: PredictionType::Pattern,
                time_horizon: pattern_pred.predicted_occurrence_time as usize,
                confidence: pattern_pred.confidence,
                expected_emergence_score: pattern_pred.expected_strength,
                expected_types: pattern_pred.emergence_types,
                description: format!("Pattern {} expected", pattern_pred.pattern_id),
            };
            predictions.push(emergence_prediction);
        }

        // Trend-based predictions
        if self.analysis_history.emergence_scores.len() > 10 {
            let trend_prediction = self.predict_emergence_trend();
            predictions.push(trend_prediction);
        }

        predictions
    }

    /// Predict emergence trend
    fn predict_emergence_trend(&self) -> EmergencePrediction {
        let recent_scores: Vec<f64> = self.analysis_history.emergence_scores.iter()
            .rev()
            .take(10)
            .cloned()
            .collect();

        if recent_scores.len() < 2 {
            return EmergencePrediction {
                prediction_type: PredictionType::Trend,
                time_horizon: self.params.prediction_horizon,
                confidence: 0.1,
                expected_emergence_score: 0.0,
                expected_types: Vec::new(),
                description: "Insufficient data for trend prediction".to_string(),
            };
        }

        // Simple linear trend calculation
        let x: Vec<f64> = (0..recent_scores.len()).map(|i| i as f64).collect();
        let slope = self.calculate_linear_slope(&x, &recent_scores);
        
        let current_score = recent_scores[0];
        let predicted_score = current_score + slope * self.params.prediction_horizon as f64;
        
        let confidence = if slope.abs() > 0.01 { 0.7 } else { 0.3 };

        EmergencePrediction {
            prediction_type: PredictionType::Trend,
            time_horizon: self.params.prediction_horizon,
            confidence,
            expected_emergence_score: predicted_score.clamp(0.0, 1.0),
            expected_types: vec![EmergenceType::SelfOrganization],
            description: format!("Trend prediction: slope = {:.4}", slope),
        }
    }

    /// Calculate linear slope for trend analysis
    fn calculate_linear_slope(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }

    /// Update analysis history
    fn update_analysis_history(&mut self, analysis: &CombinedAnalysis) {
        // Update emergence scores
        self.analysis_history.emergence_scores.push_back(
            self.detector.get_emergence_state().emergence_score
        );
        if self.analysis_history.emergence_scores.len() > self.params.memory_retention {
            self.analysis_history.emergence_scores.pop_front();
        }

        // Update pattern strengths
        self.analysis_history.pattern_strengths.push_back(analysis.pattern_strength);
        if self.analysis_history.pattern_strengths.len() > self.params.memory_retention {
            self.analysis_history.pattern_strengths.pop_front();
        }

        // Update alert frequency
        let current_alerts = self.detector.get_alerts().len();
        self.analysis_history.alert_frequency.push_back(current_alerts);
        if self.analysis_history.alert_frequency.len() > self.params.memory_retention {
            self.analysis_history.alert_frequency.pop_front();
        }
    }

    /// Generate recommendations based on analysis
    fn generate_recommendations(&mut self, analysis: &CombinedAnalysis) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Recommendations based on emergence level
        match analysis.emergence_level {
            EmergenceLevel::High => {
                recommendations.push("High emergence detected - monitor system closely".to_string());
                recommendations.push("Consider increasing observation frequency".to_string());
            },
            EmergenceLevel::Medium => {
                recommendations.push("Moderate emergence - maintain current monitoring".to_string());
            },
            EmergenceLevel::Low => {
                recommendations.push("Low emergence - routine monitoring sufficient".to_string());
            },
            EmergenceLevel::Minimal => {
                recommendations.push("Minimal emergence - consider system stimulation".to_string());
            },
        }

        // Recommendations based on criticality
        if analysis.criticality > 0.8 {
            recommendations.push("High criticality - prepare for potential phase transitions".to_string());
        }

        // Recommendations based on stability
        if analysis.temporal_stability < 0.3 {
            recommendations.push("Low temporal stability - expect volatile behavior".to_string());
        }

        // Pattern-specific recommendations
        for pattern_id in &analysis.dominant_patterns {
            if pattern_id.contains("critical") {
                recommendations.push("Critical patterns detected - monitor for cascading effects".to_string());
            } else if pattern_id.contains("periodic") {
                recommendations.push("Periodic patterns suggest predictable behavior".to_string());
            }
        }

        // Update current analysis
        self.current_analysis = CurrentAnalysis {
            emergence_score: analysis.confidence,
            active_patterns: analysis.dominant_patterns.clone(),
            confidence: analysis.confidence,
            trend: self.determine_trend(),
            risk_level: self.assess_risk_level(analysis),
            recommendations: recommendations.clone(),
        };

        recommendations
    }

    /// Determine current trend direction
    fn determine_trend(&self) -> TrendDirection {
        if self.analysis_history.emergence_scores.len() < 5 {
            return TrendDirection::Unknown;
        }

        let recent: Vec<f64> = self.analysis_history.emergence_scores.iter()
            .rev()
            .take(5)
            .cloned()
            .collect();

        let x: Vec<f64> = (0..recent.len()).map(|i| i as f64).collect();
        let slope = self.calculate_linear_slope(&x, &recent);

        if slope > 0.01 {
            TrendDirection::Increasing
        } else if slope < -0.01 {
            TrendDirection::Decreasing
        } else {
            // Check for oscillations
            let mut changes = 0;
            for i in 1..recent.len() {
                if (recent[i] - recent[i-1]).abs() > 0.05 {
                    changes += 1;
                }
            }
            
            if changes >= 3 {
                TrendDirection::Oscillating
            } else {
                TrendDirection::Stable
            }
        }
    }

    /// Assess risk level
    fn assess_risk_level(&self, analysis: &CombinedAnalysis) -> RiskLevel {
        let mut risk_score = 0.0;

        // High emergence increases risk
        risk_score += match analysis.emergence_level {
            EmergenceLevel::High => 0.4,
            EmergenceLevel::Medium => 0.2,
            EmergenceLevel::Low => 0.1,
            EmergenceLevel::Minimal => 0.0,
        };

        // High criticality increases risk
        risk_score += analysis.criticality * 0.3;

        // Low stability increases risk
        risk_score += (1.0 - analysis.temporal_stability) * 0.3;

        if risk_score > 0.8 {
            RiskLevel::Critical
        } else if risk_score > 0.6 {
            RiskLevel::High
        } else if risk_score > 0.3 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    /// Calculate overall confidence
    fn calculate_overall_confidence(&self) -> f64 {
        let detection_confidence = self.detector.get_emergence_state().confidence;
        let pattern_confidence = if self.pattern_recognizer.get_patterns().is_empty() {
            0.5
        } else {
            self.pattern_recognizer.get_patterns().iter()
                .map(|p| p.strength)
                .sum::<f64>() / self.pattern_recognizer.get_patterns().len() as f64
        };

        (detection_confidence + pattern_confidence) / 2.0
    }

    /// Get current analysis state
    pub fn get_current_analysis(&self) -> &CurrentAnalysis {
        &self.current_analysis
    }

    /// Get analysis history
    pub fn get_analysis_history(&self) -> &AnalysisHistory {
        &self.analysis_history
    }

    /// Reset analysis system
    pub fn reset(&mut self) {
        self.detector.reset();
        self.analysis_history.emergence_scores.clear();
        self.analysis_history.pattern_strengths.clear();
        self.analysis_history.prediction_accuracy.clear();
        self.analysis_history.alert_frequency.clear();
    }
}

#[derive(Clone, Debug)]
pub struct AnalysisResult {
    pub emergence_state: EmergenceState,
    pub patterns: Vec<TemporalPattern>,
    pub predictions: Vec<EmergencePrediction>,
    pub combined_analysis: CombinedAnalysis,
    pub recommendations: Vec<String>,
    pub confidence: f64,
}

#[derive(Clone, Debug)]
pub struct CombinedAnalysis {
    pub emergence_level: EmergenceLevel,
    pub pattern_strength: f64,
    pub dominant_patterns: Vec<String>,
    pub temporal_stability: f64,
    pub criticality: f64,
    pub emergence_types: Vec<EmergenceType>,
    pub confidence: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EmergenceLevel {
    Minimal,
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug)]
pub struct EmergencePrediction {
    pub prediction_type: PredictionType,
    pub time_horizon: usize,
    pub confidence: f64,
    pub expected_emergence_score: f64,
    pub expected_types: Vec<EmergenceType>,
    pub description: String,
}

#[derive(Clone, Debug)]
pub enum PredictionType {
    Pattern,
    Trend,
    Critical,
    Transition,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emergence_analysis_system_creation() {
        let detection_params = DetectionParameters::default();
        let pattern_params = PatternParameters::default();
        let analysis_params = AnalysisParameters::default();

        let system = EmergenceAnalysisSystem::new(
            detection_params,
            pattern_params,
            analysis_params,
        );

        assert_eq!(system.current_analysis.emergence_score, 0.0);
        assert_eq!(system.current_analysis.trend, TrendDirection::Unknown);
    }

    #[test]
    fn test_emergence_level_classification() {
        let detection_params = DetectionParameters::default();
        let pattern_params = PatternParameters::default();
        let analysis_params = AnalysisParameters::default();

        let system = EmergenceAnalysisSystem::new(
            detection_params,
            pattern_params,
            analysis_params,
        );

        assert_eq!(system.classify_emergence_level(0.9, 0.8), EmergenceLevel::High);
        assert_eq!(system.classify_emergence_level(0.5, 0.6), EmergenceLevel::Medium);
        assert_eq!(system.classify_emergence_level(0.3, 0.2), EmergenceLevel::Low);
        assert_eq!(system.classify_emergence_level(0.1, 0.1), EmergenceLevel::Minimal);
    }

    #[test]
    fn test_trend_determination() {
        let detection_params = DetectionParameters::default();
        let pattern_params = PatternParameters::default();
        let analysis_params = AnalysisParameters::default();

        let mut system = EmergenceAnalysisSystem::new(
            detection_params,
            pattern_params,
            analysis_params,
        );

        // Add increasing trend data
        for i in 0..10 {
            system.analysis_history.emergence_scores.push_back(i as f64 * 0.1);
        }

        let trend = system.determine_trend();
        assert_eq!(trend, TrendDirection::Increasing);
    }
}