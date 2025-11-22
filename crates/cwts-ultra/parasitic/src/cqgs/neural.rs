//! Neural Intelligence for CQGS
//!
//! Machine learning powered pattern recognition and predictive analysis
//! for enhanced quality governance decision making.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

use crate::cqgs::sentinels::{SentinelId, SentinelType};
use crate::cqgs::{CqgsEvent, QualityViolation, ViolationSeverity};

/// Neural pattern recognition and learning system
pub struct NeuralEngine {
    learned_patterns: Arc<RwLock<HashMap<Uuid, LearnedPattern>>>,
    prediction_models: Arc<RwLock<HashMap<SentinelType, PredictionModel>>>,
    training_data: Arc<Mutex<Vec<TrainingExample>>>,
    config: NeuralConfig,
}

/// Neural engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    pub learning_rate: f64,
    pub confidence_threshold: f64,
    pub max_training_examples: usize,
    pub enable_online_learning: bool,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            confidence_threshold: 0.8,
            max_training_examples: 10000,
            enable_online_learning: true,
        }
    }
}

/// Learned pattern from quality violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedPattern {
    pub id: Uuid,
    pub pattern_type: PatternType,
    pub confidence: f64,
    pub occurrences: u32,
    pub predictive_indicators: Vec<String>,
    pub remediation_success_rate: f64,
    pub last_updated: std::time::SystemTime,
}

/// Types of quality patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    RecurringViolation,
    CascadingFailure,
    PerformanceDegradation,
    SecurityThreat,
    TestFlakiness,
    DependencyIssue,
    ConfigurationDrift,
}

/// Prediction model for specific sentinel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionModel {
    pub sentinel_type: SentinelType,
    pub accuracy: f64,
    pub predictions_made: u64,
    pub correct_predictions: u64,
    pub feature_weights: HashMap<String, f64>,
    pub last_training: std::time::SystemTime,
}

/// Training example for neural learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub violation: QualityViolation,
    pub context: ViolationContext,
    pub outcome: RemediationOutcome,
    pub timestamp: std::time::SystemTime,
}

/// Context information for violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationContext {
    pub system_load: f64,
    pub recent_violations: u32,
    pub time_of_day: u8, // 0-23
    pub day_of_week: u8, // 0-6
    pub deployment_recent: bool,
    pub configuration_changes: bool,
    pub external_dependencies_healthy: bool,
}

/// Outcome of remediation attempts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationOutcome {
    pub success: bool,
    pub time_to_resolve: std::time::Duration,
    pub human_intervention_required: bool,
    pub root_cause_identified: bool,
    pub recurrence_prevented: bool,
}

/// Quality prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPrediction {
    pub prediction_id: Uuid,
    pub predicted_violations: Vec<PredictedViolation>,
    pub system_health_forecast: f64,
    pub confidence: f64,
    pub recommendation: String,
    pub created_at: std::time::SystemTime,
}

/// Individual violation prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedViolation {
    pub violation_type: ViolationSeverity,
    pub probability: f64,
    pub expected_time: std::time::Duration, // Time until occurrence
    pub affected_components: Vec<String>,
    pub preventive_actions: Vec<String>,
}

impl NeuralEngine {
    /// Create new neural engine
    pub fn new(config: NeuralConfig) -> Self {
        Self {
            learned_patterns: Arc::new(RwLock::new(HashMap::new())),
            prediction_models: Arc::new(RwLock::new(HashMap::new())),
            training_data: Arc::new(Mutex::new(Vec::new())),
            config,
        }
    }

    /// Initialize prediction models for all sentinel types
    pub async fn initialize_models(&self) {
        let mut models = self.prediction_models.write().await;

        for sentinel_type in [
            SentinelType::Quality,
            SentinelType::Performance,
            SentinelType::Security,
            SentinelType::Coverage,
            SentinelType::Integrity,
            SentinelType::ZeroMock,
            SentinelType::Neural,
            SentinelType::Healing,
        ] {
            models.insert(
                sentinel_type,
                PredictionModel {
                    sentinel_type,
                    accuracy: 0.5, // Start with baseline accuracy
                    predictions_made: 0,
                    correct_predictions: 0,
                    feature_weights: HashMap::new(),
                    last_training: std::time::SystemTime::now(),
                },
            );
        }

        tracing::info!(
            "Initialized neural prediction models for {} sentinel types",
            models.len()
        );
    }

    /// Learn from a quality violation and its resolution
    pub async fn learn_from_violation(
        &self,
        violation: &QualityViolation,
        context: ViolationContext,
        outcome: RemediationOutcome,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Add to training data
        {
            let mut training_data = self.training_data.lock().await;
            training_data.push(TrainingExample {
                violation: violation.clone(),
                context,
                outcome,
                timestamp: std::time::SystemTime::now(),
            });

            // Keep only recent training examples
            if training_data.len() > self.config.max_training_examples {
                training_data.remove(0);
            }
        }

        // Update learned patterns
        self.update_patterns(violation).await?;

        // Retrain relevant model if online learning is enabled
        if self.config.enable_online_learning {
            self.incremental_training(&violation.sentinel_id).await?;
        }

        Ok(())
    }

    /// Update learned patterns based on new violation
    async fn update_patterns(
        &self,
        violation: &QualityViolation,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut patterns = self.learned_patterns.write().await;

        // Look for existing similar patterns
        let mut pattern_found = false;
        for pattern in patterns.values_mut() {
            if self.is_similar_violation(violation, pattern) {
                pattern.occurrences += 1;
                pattern.confidence = (pattern.confidence + 0.1).min(1.0);
                pattern.last_updated = std::time::SystemTime::now();
                pattern_found = true;
                break;
            }
        }

        // Create new pattern if none found
        if !pattern_found {
            let new_pattern = LearnedPattern {
                id: Uuid::new_v4(),
                pattern_type: self.classify_violation(violation),
                confidence: 0.5,
                occurrences: 1,
                predictive_indicators: self.extract_indicators(violation),
                remediation_success_rate: 0.5,
                last_updated: std::time::SystemTime::now(),
            };
            patterns.insert(new_pattern.id, new_pattern);
        }

        Ok(())
    }

    /// Check if violation is similar to existing pattern
    fn is_similar_violation(&self, violation: &QualityViolation, pattern: &LearnedPattern) -> bool {
        // Simple similarity check based on location and message content
        violation
            .location
            .contains(&pattern.predictive_indicators[0])
            || violation
                .message
                .to_lowercase()
                .contains(&pattern.predictive_indicators[0].to_lowercase())
    }

    /// Classify violation into pattern type
    fn classify_violation(&self, violation: &QualityViolation) -> PatternType {
        let message_lower = violation.message.to_lowercase();
        let location_lower = violation.location.to_lowercase();

        if message_lower.contains("performance") || message_lower.contains("slow") {
            PatternType::PerformanceDegradation
        } else if message_lower.contains("security") || message_lower.contains("vulnerability") {
            PatternType::SecurityThreat
        } else if message_lower.contains("test") || message_lower.contains("flaky") {
            PatternType::TestFlakiness
        } else if message_lower.contains("dependency") || message_lower.contains("import") {
            PatternType::DependencyIssue
        } else if message_lower.contains("config") || message_lower.contains("setting") {
            PatternType::ConfigurationDrift
        } else if location_lower.contains("cascade") || message_lower.contains("chain") {
            PatternType::CascadingFailure
        } else {
            PatternType::RecurringViolation
        }
    }

    /// Extract predictive indicators from violation
    fn extract_indicators(&self, violation: &QualityViolation) -> Vec<String> {
        let mut indicators = Vec::new();

        // Extract key words from message
        let words: Vec<&str> = violation
            .message
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .take(3)
            .collect();

        for word in words {
            indicators.push(word.to_lowercase());
        }

        // Add location indicator
        if let Some(file_part) = violation.location.split('/').last() {
            if let Some(filename) = file_part.split(':').next() {
                indicators.push(filename.to_string());
            }
        }

        indicators
    }

    /// Perform incremental training for specific sentinel
    async fn incremental_training(
        &self,
        sentinel_id: &SentinelId,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // In a real implementation, this would perform actual ML training
        // For now, we'll simulate incremental learning by updating model accuracy

        let training_data = self.training_data.lock().await;
        let recent_examples: Vec<_> = training_data
            .iter()
            .filter(|example| example.violation.sentinel_id == *sentinel_id)
            .take(100) // Use last 100 examples
            .collect();

        if recent_examples.len() >= 10 {
            // Calculate success rate from recent examples
            let success_rate = recent_examples
                .iter()
                .map(|example| if example.outcome.success { 1.0 } else { 0.0 })
                .sum::<f64>()
                / recent_examples.len() as f64;

            // Update model accuracy (simplified)
            let mut models = self.prediction_models.write().await;
            for model in models.values_mut() {
                if model
                    .sentinel_type
                    .to_string()
                    .contains(&sentinel_id.to_string())
                {
                    model.accuracy = (model.accuracy * 0.9 + success_rate * 0.1).max(0.1);
                    model.last_training = std::time::SystemTime::now();
                    break;
                }
            }
        }

        Ok(())
    }

    /// Generate quality predictions based on current system state
    pub async fn predict_quality_issues(
        &self,
        context: &ViolationContext,
    ) -> Result<QualityPrediction, Box<dyn std::error::Error + Send + Sync>> {
        let mut predicted_violations = Vec::new();

        // Analyze current patterns and context
        let patterns = self.learned_patterns.read().await;
        for pattern in patterns.values() {
            if pattern.confidence >= self.config.confidence_threshold {
                let probability = self.calculate_violation_probability(pattern, context);

                if probability > 0.3 {
                    predicted_violations.push(PredictedViolation {
                        violation_type: ViolationSeverity::Warning, // Simplified
                        probability,
                        expected_time: std::time::Duration::from_secs(2 * 60 * 60), // 2 hours prediction horizon
                        affected_components: pattern.predictive_indicators.clone(),
                        preventive_actions: vec![
                            "Monitor system metrics closely".to_string(),
                            "Review recent configuration changes".to_string(),
                            "Check dependency health".to_string(),
                        ],
                    });
                }
            }
        }

        // Calculate overall system health forecast
        let health_forecast = if predicted_violations.is_empty() {
            0.95 // High confidence if no issues predicted
        } else {
            let avg_probability: f64 = predicted_violations
                .iter()
                .map(|v| v.probability)
                .sum::<f64>()
                / predicted_violations.len() as f64;
            1.0 - (avg_probability * 0.5) // Reduce health based on predicted issues
        };

        let prediction = QualityPrediction {
            prediction_id: Uuid::new_v4(),
            predicted_violations,
            system_health_forecast: health_forecast,
            confidence: 0.8, // Placeholder confidence
            recommendation: self.generate_recommendation(health_forecast).await,
            created_at: std::time::SystemTime::now(),
        };

        Ok(prediction)
    }

    /// Calculate probability of violation based on pattern and context
    fn calculate_violation_probability(
        &self,
        pattern: &LearnedPattern,
        context: &ViolationContext,
    ) -> f64 {
        let mut base_probability =
            pattern.confidence * (pattern.occurrences as f64 / 100.0).min(1.0);

        // Adjust based on context
        if context.system_load > 0.8 {
            base_probability *= 1.5; // Higher load increases risk
        }

        if context.recent_violations > 5 {
            base_probability *= 1.3; // Recent issues increase risk
        }

        if context.deployment_recent {
            base_probability *= 1.2; // Recent deployments increase risk
        }

        if !context.external_dependencies_healthy {
            base_probability *= 1.4; // Unhealthy dependencies increase risk
        }

        base_probability.min(1.0)
    }

    /// Generate recommendation based on health forecast
    async fn generate_recommendation(&self, health_forecast: f64) -> String {
        if health_forecast > 0.9 {
            "System health is excellent. Continue monitoring.".to_string()
        } else if health_forecast > 0.8 {
            "System health is good. Monitor for potential issues.".to_string()
        } else if health_forecast > 0.7 {
            "System health is concerning. Investigate potential issues.".to_string()
        } else {
            "System health is poor. Immediate intervention recommended.".to_string()
        }
    }

    /// Get neural engine statistics
    pub async fn get_statistics(&self) -> NeuralStatistics {
        let patterns = self.learned_patterns.read().await;
        let models = self.prediction_models.read().await;
        let training_data = self.training_data.lock().await;

        let total_predictions: u64 = models.values().map(|m| m.predictions_made).sum();
        let total_correct: u64 = models.values().map(|m| m.correct_predictions).sum();
        let overall_accuracy = if total_predictions > 0 {
            total_correct as f64 / total_predictions as f64
        } else {
            0.0
        };

        NeuralStatistics {
            learned_patterns_count: patterns.len(),
            training_examples_count: training_data.len(),
            total_predictions,
            overall_accuracy,
            model_accuracies: models
                .iter()
                .map(|(k, v)| (k.clone(), v.accuracy))
                .collect(),
            confidence_threshold: self.config.confidence_threshold,
        }
    }
}

/// Neural engine performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralStatistics {
    pub learned_patterns_count: usize,
    pub training_examples_count: usize,
    pub total_predictions: u64,
    pub overall_accuracy: f64,
    pub model_accuracies: HashMap<SentinelType, f64>,
    pub confidence_threshold: f64,
}

impl std::fmt::Display for SentinelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SentinelType::Quality => write!(f, "Quality"),
            SentinelType::Performance => write!(f, "Performance"),
            SentinelType::Security => write!(f, "Security"),
            SentinelType::Coverage => write!(f, "Coverage"),
            SentinelType::Integrity => write!(f, "Integrity"),
            SentinelType::ZeroMock => write!(f, "ZeroMock"),
            SentinelType::Neural => write!(f, "Neural"),
            SentinelType::Healing => write!(f, "Healing"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cqgs::sentinels::SentinelId;

    #[tokio::test]
    async fn test_neural_engine_creation() {
        let config = NeuralConfig::default();
        let engine = NeuralEngine::new(config);

        engine.initialize_models().await;

        let stats = engine.get_statistics().await;
        assert_eq!(stats.learned_patterns_count, 0);
        assert_eq!(stats.training_examples_count, 0);
    }

    #[tokio::test]
    async fn test_pattern_learning() {
        let engine = NeuralEngine::new(NeuralConfig::default());

        let violation = QualityViolation {
            id: Uuid::new_v4(),
            sentinel_id: SentinelId::new("test".to_string()),
            severity: ViolationSeverity::Warning,
            message: "Performance degradation detected".to_string(),
            location: "src/lib.rs:100".to_string(),
            timestamp: std::time::SystemTime::now(),
            remediation_suggestion: None,
            auto_fixable: false,
            hyperbolic_coordinates: None,
        };

        let context = ViolationContext {
            system_load: 0.7,
            recent_violations: 2,
            time_of_day: 14,
            day_of_week: 3,
            deployment_recent: false,
            configuration_changes: false,
            external_dependencies_healthy: true,
        };

        let outcome = RemediationOutcome {
            success: true,
            time_to_resolve: std::time::Duration::from_secs(300),
            human_intervention_required: false,
            root_cause_identified: true,
            recurrence_prevented: true,
        };

        engine
            .learn_from_violation(&violation, context, outcome)
            .await
            .unwrap();

        let stats = engine.get_statistics().await;
        assert_eq!(stats.learned_patterns_count, 1);
        assert_eq!(stats.training_examples_count, 1);
    }
}
