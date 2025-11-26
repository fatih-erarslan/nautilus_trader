//! Pattern Learning Engine - ReasoningBank self-learning implementation
//!
//! This module implements the core self-learning capabilities:
//! - Experience recording and tracking
//! - Verdict judgment for prediction quality
//! - Memory distillation for successful patterns
//! - Adaptive threshold adjustment
//! - Trajectory building and analysis

use anyhow::{anyhow, Result};
use chrono::Utc;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::metrics::{calculate_sharpe_ratio, calculate_win_rate, calculate_profit_factor};
use crate::types::{
    Adaptation, DistilledPattern, MatchingThresholds,
    PatternExperience, PatternTrajectory, PatternVerdict,
};

/// Mock AgentDB interface for pattern storage
/// In production, this would integrate with the actual AgentDB crate
#[async_trait::async_trait]
pub trait PatternStorage: Send + Sync {
    async fn insert(&self, collection: &str, data: &JsonValue, vector: Option<&[f32]>) -> Result<String>;
    async fn query(&self, collection: &str, filter: &str, limit: usize) -> Result<Vec<JsonValue>>;
    async fn query_similar(&self, collection: &str, vector: &[f32], limit: usize) -> Result<Vec<JsonValue>>;
    async fn update(&self, collection: &str, id: &str, data: &JsonValue) -> Result<()>;
}

/// Pattern Learning Engine - Core self-learning system
pub struct PatternLearningEngine {
    /// Storage backend for pattern data
    storage: Arc<dyn PatternStorage>,

    /// In-memory cache of recent experiences
    experiences: Arc<RwLock<Vec<PatternExperience>>>,

    /// Pattern performance trajectories
    trajectories: Arc<RwLock<HashMap<String, PatternTrajectory>>>,

    /// Current matching thresholds
    thresholds: Arc<RwLock<MatchingThresholds>>,

    /// Maximum experiences to keep in memory
    max_memory_size: usize,
}

impl PatternLearningEngine {
    /// Create a new pattern learning engine
    pub fn new(storage: Arc<dyn PatternStorage>) -> Self {
        Self {
            storage,
            experiences: Arc::new(RwLock::new(Vec::new())),
            trajectories: Arc::new(RwLock::new(HashMap::new())),
            thresholds: Arc::new(RwLock::new(MatchingThresholds::default())),
            max_memory_size: 1000,
        }
    }

    /// Record a new pattern matching experience
    ///
    /// # Arguments
    /// * `experience` - The pattern experience to record
    ///
    /// # Returns
    /// Result indicating success or error
    pub async fn record_experience(&self, experience: PatternExperience) -> Result<()> {
        // Store in memory cache
        let mut experiences = self.experiences.write().await;
        experiences.push(experience.clone());

        // Trim if cache is too large
        if experiences.len() > self.max_memory_size {
            let excess = experiences.len() - self.max_memory_size;
            experiences.drain(0..excess);
        }
        drop(experiences);

        // Serialize and store in AgentDB
        let data = serde_json::to_value(&experience)?;
        self.storage
            .insert("pattern_experiences", &data, Some(&experience.pattern_vector))
            .await?;

        info!(
            "Recorded experience: {} (similarity: {:.2}, confidence: {:.2}, predicted: {:.4})",
            experience.pattern_type,
            experience.similarity,
            experience.confidence,
            experience.predicted_outcome
        );

        Ok(())
    }

    /// Update an experience with the actual outcome
    ///
    /// # Arguments
    /// * `experience_id` - ID of the experience to update
    /// * `actual_outcome` - The realized return
    ///
    /// # Returns
    /// Result with the verdict if successful
    pub async fn update_outcome(
        &self,
        experience_id: &str,
        actual_outcome: f64,
    ) -> Result<PatternVerdict> {
        // Update in memory
        let mut experiences = self.experiences.write().await;
        let exp = experiences
            .iter_mut()
            .find(|e| e.id == experience_id)
            .ok_or_else(|| anyhow!("Experience not found: {}", experience_id))?;

        exp.actual_outcome = Some(actual_outcome);

        // Calculate prediction accuracy
        let prediction_error = (exp.predicted_outcome - actual_outcome).abs();
        let accuracy = if exp.predicted_outcome.abs() > 0.001 {
            1.0 - (prediction_error / exp.predicted_outcome.abs()).min(1.0)
        } else {
            if actual_outcome.abs() < 0.001 { 1.0 } else { 0.0 }
        };

        info!(
            "Updated outcome for {}: predicted={:.4}, actual={:.4}, accuracy={:.2}%",
            experience_id,
            exp.predicted_outcome,
            actual_outcome,
            accuracy * 100.0
        );

        // Judge prediction quality
        let verdict = self.judge_prediction(exp).await?;

        // If high quality, distill to long-term memory
        if verdict.should_learn && verdict.quality_score > 0.8 {
            self.distill_to_memory(exp, &verdict).await?;
        }

        // If low quality, adapt thresholds
        if verdict.should_adapt {
            self.apply_adaptations(&verdict.suggested_changes).await?;
        }

        Ok(verdict)
    }

    /// Judge the quality of a prediction
    ///
    /// # Arguments
    /// * `exp` - The experience to judge
    ///
    /// # Returns
    /// PatternVerdict with quality assessment
    async fn judge_prediction(&self, exp: &PatternExperience) -> Result<PatternVerdict> {
        let actual = exp
            .actual_outcome
            .ok_or_else(|| anyhow!("Outcome not set"))?;
        let predicted = exp.predicted_outcome;

        // Check direction correctness
        let direction_correct = (predicted > 0.0) == (actual > 0.0) ||
                               (predicted.abs() < 0.001 && actual.abs() < 0.001);

        // Calculate magnitude error (normalized)
        let magnitude_error = if actual.abs() > 0.001 {
            ((predicted - actual).abs() / actual.abs()).min(1.0)
        } else {
            if predicted.abs() < 0.001 { 0.0 } else { 1.0 }
        };

        // Quality score calculation (0-1)
        let quality_score = if direction_correct {
            // Correct direction: score based on magnitude accuracy
            0.5 + (0.5 * (1.0 - magnitude_error))
        } else {
            // Wrong direction: heavily penalized
            0.5 * (1.0 - magnitude_error) * 0.5
        };

        // Generate adaptation suggestions
        let suggested_changes = self.suggest_adaptations(exp, quality_score).await?;

        let verdict = PatternVerdict {
            experience_id: exp.id.clone(),
            quality_score,
            direction_correct,
            magnitude_error,
            should_learn: quality_score > 0.7,
            should_adapt: quality_score < 0.4,
            suggested_changes,
        };

        info!(
            "Pattern verdict for {}: score={:.2}, direction={}, magnitude_err={:.2}",
            exp.pattern_type, quality_score, direction_correct, magnitude_error
        );

        Ok(verdict)
    }

    /// Distill a successful pattern to long-term memory
    async fn distill_to_memory(
        &self,
        exp: &PatternExperience,
        verdict: &PatternVerdict,
    ) -> Result<()> {
        let distilled = DistilledPattern {
            pattern_type: exp.pattern_type.clone(),
            pattern_vector: exp.pattern_vector.clone(),
            success_rate: verdict.quality_score,
            avg_return: exp.actual_outcome.unwrap(),
            confidence_threshold: exp.confidence,
            similarity_threshold: exp.similarity,
            market_conditions: exp.market_context.clone(),
            sample_count: 1,
            last_updated: Utc::now(),
        };

        // Store in long-term memory
        let data = serde_json::to_value(&distilled)?;
        self.storage
            .insert("distilled_patterns", &data, Some(&exp.pattern_vector))
            .await?;

        info!(
            "Distilled pattern to memory: {} (quality: {:.2})",
            exp.pattern_type, verdict.quality_score
        );

        Ok(())
    }

    /// Suggest adaptations based on performance
    async fn suggest_adaptations(
        &self,
        exp: &PatternExperience,
        quality_score: f64,
    ) -> Result<Vec<Adaptation>> {
        let mut adaptations = Vec::new();

        // Low quality: tighten thresholds
        if quality_score < 0.5 {
            if exp.similarity < 0.95 {
                adaptations.push(Adaptation {
                    parameter: "similarity_threshold".to_string(),
                    current_value: exp.similarity,
                    suggested_value: (exp.similarity + 0.05).min(0.95),
                    reason: "Low quality predictions - require higher similarity".to_string(),
                });
            }

            if exp.confidence < 0.90 {
                adaptations.push(Adaptation {
                    parameter: "confidence_threshold".to_string(),
                    current_value: exp.confidence,
                    suggested_value: (exp.confidence + 0.05).min(0.90),
                    reason: "Low quality predictions - require higher confidence".to_string(),
                });
            }
        }

        // High quality: can relax thresholds
        if quality_score > 0.85 {
            adaptations.push(Adaptation {
                parameter: "similarity_threshold".to_string(),
                current_value: exp.similarity,
                suggested_value: (exp.similarity - 0.02).max(0.70),
                reason: "High quality predictions - can relax threshold slightly".to_string(),
            });

            adaptations.push(Adaptation {
                parameter: "confidence_threshold".to_string(),
                current_value: exp.confidence,
                suggested_value: (exp.confidence - 0.02).max(0.60),
                reason: "High quality predictions - can relax threshold slightly".to_string(),
            });
        }

        Ok(adaptations)
    }

    /// Apply suggested adaptations to thresholds
    async fn apply_adaptations(&self, adaptations: &[Adaptation]) -> Result<()> {
        if adaptations.is_empty() {
            return Ok(());
        }

        let mut thresholds = self.thresholds.write().await;

        for adaptation in adaptations {
            match adaptation.parameter.as_str() {
                "similarity_threshold" => {
                    info!(
                        "Adapting similarity threshold: {:.2} -> {:.2} ({})",
                        thresholds.similarity_threshold,
                        adaptation.suggested_value,
                        adaptation.reason
                    );
                    thresholds.similarity_threshold = adaptation.suggested_value;
                }
                "confidence_threshold" => {
                    info!(
                        "Adapting confidence threshold: {:.2} -> {:.2} ({})",
                        thresholds.confidence_threshold,
                        adaptation.suggested_value,
                        adaptation.reason
                    );
                    thresholds.confidence_threshold = adaptation.suggested_value;
                }
                _ => {
                    warn!("Unknown adaptation parameter: {}", adaptation.parameter);
                }
            }
        }

        Ok(())
    }

    /// Build performance trajectory for a pattern type
    ///
    /// # Arguments
    /// * `pattern_type` - Type of pattern to analyze
    ///
    /// # Returns
    /// PatternTrajectory with performance metrics
    pub async fn build_trajectory(&self, pattern_type: &str) -> Result<PatternTrajectory> {
        // Get all experiences for this pattern type
        let experiences = self.experiences.read().await;
        let pattern_exps: Vec<PatternExperience> = experiences
            .iter()
            .filter(|e| e.pattern_type == pattern_type && e.actual_outcome.is_some())
            .cloned()
            .collect();

        if pattern_exps.is_empty() {
            return Err(anyhow!(
                "No completed experiences for pattern type: {}",
                pattern_type
            ));
        }

        // Calculate metrics
        let total = pattern_exps.len();
        let returns: Vec<f64> = pattern_exps
            .iter()
            .map(|e| e.actual_outcome.unwrap())
            .collect();

        let success_rate = calculate_win_rate(&returns);
        let avg_return = returns.iter().sum::<f64>() / total as f64;
        let sharpe_ratio = calculate_sharpe_ratio(&returns);
        let best_return = returns
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let worst_return = returns.iter().cloned().fold(f64::INFINITY, f64::min);

        let trajectory = PatternTrajectory {
            pattern_type: pattern_type.to_string(),
            sample_count: total,
            success_rate,
            avg_return,
            sharpe_ratio,
            best_return,
            worst_return,
            experiences: pattern_exps,
            last_updated: Utc::now(),
        };

        info!(
            "Built trajectory for {}: {} samples, {:.1}% win rate, avg return={:.4}, Sharpe={:.2}",
            pattern_type,
            total,
            success_rate * 100.0,
            avg_return,
            sharpe_ratio
        );

        // Cache trajectory
        self.trajectories
            .write()
            .await
            .insert(pattern_type.to_string(), trajectory.clone());

        Ok(trajectory)
    }

    /// Get current matching thresholds
    pub async fn get_thresholds(&self) -> MatchingThresholds {
        self.thresholds.read().await.clone()
    }

    /// Get all cached experiences
    pub async fn get_experiences(&self) -> Vec<PatternExperience> {
        self.experiences.read().await.clone()
    }

    /// Get cached trajectory for a pattern type
    pub async fn get_trajectory(&self, pattern_type: &str) -> Option<PatternTrajectory> {
        self.trajectories.read().await.get(pattern_type).cloned()
    }

    /// Perform adaptive threshold adjustment based on recent performance
    ///
    /// # Arguments
    /// * `lookback` - Number of recent experiences to analyze
    ///
    /// # Returns
    /// Updated thresholds
    pub async fn adapt_thresholds(&self, lookback: usize) -> Result<MatchingThresholds> {
        let experiences = self.experiences.read().await;
        let recent: Vec<&PatternExperience> = experiences
            .iter()
            .rev()
            .take(lookback)
            .filter(|e| e.actual_outcome.is_some())
            .collect();

        if recent.len() < 10 {
            warn!("Not enough data for threshold adaptation ({} samples)", recent.len());
            return Ok(self.get_thresholds().await);
        }

        // Calculate recent performance
        let returns: Vec<f64> = recent
            .iter()
            .map(|e| e.actual_outcome.unwrap())
            .collect();

        let success_rate = calculate_win_rate(&returns);
        let profit_factor = calculate_profit_factor(&returns);

        let mut thresholds = self.thresholds.write().await;

        // Adapt based on performance
        if success_rate < 0.55 || profit_factor < 1.2 {
            // Poor performance - tighten thresholds
            thresholds.similarity_threshold = (thresholds.similarity_threshold + 0.03).min(0.95);
            thresholds.confidence_threshold = (thresholds.confidence_threshold + 0.03).min(0.90);

            info!(
                "Tightening thresholds (win rate={:.1}%, PF={:.2}): similarity={:.2}, confidence={:.2}",
                success_rate * 100.0,
                profit_factor,
                thresholds.similarity_threshold,
                thresholds.confidence_threshold
            );
        } else if success_rate > 0.70 && profit_factor > 1.8 {
            // Good performance - can relax slightly
            thresholds.similarity_threshold = (thresholds.similarity_threshold - 0.02).max(0.70);
            thresholds.confidence_threshold = (thresholds.confidence_threshold - 0.02).max(0.60);

            info!(
                "Relaxing thresholds (win rate={:.1}%, PF={:.2}): similarity={:.2}, confidence={:.2}",
                success_rate * 100.0,
                profit_factor,
                thresholds.similarity_threshold,
                thresholds.confidence_threshold
            );
        }

        Ok(thresholds.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    // Mock storage implementation for testing
    struct MockStorage;

    #[async_trait::async_trait]
    impl PatternStorage for MockStorage {
        async fn insert(&self, _: &str, _: &JsonValue, _: Option<&[f32]>) -> Result<String> {
            Ok(Uuid::new_v4().to_string())
        }

        async fn query(&self, _: &str, _: &str, _: usize) -> Result<Vec<JsonValue>> {
            Ok(vec![])
        }

        async fn query_similar(&self, _: &str, _: &[f32], _: usize) -> Result<Vec<JsonValue>> {
            Ok(vec![])
        }

        async fn update(&self, _: &str, _: &str, _: &JsonValue) -> Result<()> {
            Ok(())
        }
    }

    fn create_test_experience(predicted: f64, actual: Option<f64>) -> PatternExperience {
        PatternExperience {
            id: Uuid::new_v4().to_string(),
            pattern_type: "test_pattern".to_string(),
            pattern_vector: vec![0.1, 0.2, 0.3],
            similarity: 0.85,
            confidence: 0.75,
            predicted_outcome: predicted,
            actual_outcome: actual,
            market_context: MarketContext {
                symbol: "BTC-USD".to_string(),
                timeframe: "1h".to_string(),
                volatility: 0.02,
                volume: 1000000.0,
                trend: "bullish".to_string(),
                sentiment: 0.6,
            },
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_record_experience() {
        let storage = Arc::new(MockStorage);
        let engine = PatternLearningEngine::new(storage);

        let exp = create_test_experience(0.05, None);
        let result = engine.record_experience(exp).await;

        assert!(result.is_ok());
        assert_eq!(engine.get_experiences().await.len(), 1);
    }

    #[tokio::test]
    async fn test_update_outcome_and_verdict() {
        let storage = Arc::new(MockStorage);
        let engine = PatternLearningEngine::new(storage);

        let exp = create_test_experience(0.05, None);
        let exp_id = exp.id.clone();
        engine.record_experience(exp).await.unwrap();

        // Update with good outcome
        let verdict = engine.update_outcome(&exp_id, 0.048).await.unwrap();

        assert!(verdict.quality_score > 0.8);
        assert!(verdict.direction_correct);
        assert!(verdict.should_learn);
    }

    #[tokio::test]
    async fn test_build_trajectory() {
        let storage = Arc::new(MockStorage);
        let engine = PatternLearningEngine::new(storage);

        // Add multiple completed experiences
        for i in 0..10 {
            let predicted = 0.05;
            let actual = 0.045 + (i as f64 * 0.002);
            let mut exp = create_test_experience(predicted, Some(actual));
            engine.record_experience(exp).await.unwrap();
        }

        let trajectory = engine.build_trajectory("test_pattern").await.unwrap();

        assert_eq!(trajectory.sample_count, 10);
        assert!(trajectory.success_rate > 0.0);
        assert!(trajectory.avg_return > 0.0);
    }

    #[tokio::test]
    async fn test_adaptive_thresholds() {
        let storage = Arc::new(MockStorage);
        let engine = PatternLearningEngine::new(storage);

        // Add poor performing experiences
        for _ in 0..20 {
            let mut exp = create_test_experience(0.05, Some(-0.02));
            engine.record_experience(exp).await.unwrap();
        }

        let initial_thresholds = engine.get_thresholds().await;
        let adapted = engine.adapt_thresholds(20).await.unwrap();

        // Should tighten thresholds due to poor performance
        assert!(adapted.similarity_threshold > initial_thresholds.similarity_threshold);
        assert!(adapted.confidence_threshold > initial_thresholds.confidence_threshold);
    }
}
