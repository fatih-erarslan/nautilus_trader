//! ReasoningBank integration for adaptive learning

use crate::error::{Result, SwarmError};
use crate::types::{ReasoningExperience, ReasoningVerdict, AdaptationSuggestion};

use std::collections::VecDeque;
use parking_lot::RwLock;
use std::sync::Arc;

/// Maximum experiences to keep in memory
const MAX_EXPERIENCES: usize = 10000;

/// ReasoningBank client for experience recording and verdict judgment
pub struct ReasoningBankClient {
    /// Stored experiences
    experiences: Arc<RwLock<VecDeque<ReasoningExperience>>>,
    /// Agent performance metrics
    metrics: Arc<RwLock<std::collections::HashMap<String, AgentPerformance>>>,
}

/// Agent performance tracking
#[derive(Debug, Clone, Default)]
struct AgentPerformance {
    /// Total experiences
    total_experiences: u64,
    /// Successful outcomes
    successful_outcomes: u64,
    /// Average outcome value
    avg_outcome: f64,
    /// Average confidence
    avg_confidence: f64,
    /// Recent trend (positive/negative)
    trend: f64,
}

impl ReasoningBankClient {
    /// Create new ReasoningBank client
    pub fn new() -> Self {
        Self {
            experiences: Arc::new(RwLock::new(VecDeque::new())),
            metrics: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Record an experience
    pub async fn record_experience(&self, experience: ReasoningExperience) -> Result<()> {
        let agent_id = experience.agent_id.clone();
        let outcome_value = experience.outcome.value;
        let confidence = experience.outcome.confidence;

        // Store experience
        let mut experiences = self.experiences.write();
        if experiences.len() >= MAX_EXPERIENCES {
            experiences.pop_front();
        }
        experiences.push_back(experience);

        // Update metrics
        let mut metrics = self.metrics.write();
        let perf = metrics.entry(agent_id).or_insert_with(AgentPerformance::default);

        perf.total_experiences += 1;
        if outcome_value > 0.0 {
            perf.successful_outcomes += 1;
        }

        // Update running averages
        let n = perf.total_experiences as f64;
        perf.avg_outcome = (perf.avg_outcome * (n - 1.0) + outcome_value) / n;
        perf.avg_confidence = (perf.avg_confidence * (n - 1.0) + confidence) / n;

        Ok(())
    }

    /// Judge an experience and provide verdict
    pub async fn judge_experience(
        &self,
        agent_id: &str,
        expected: f64,
        actual: f64,
    ) -> Result<ReasoningVerdict> {
        let metrics = self.metrics.read();
        let perf = metrics
            .get(agent_id)
            .ok_or_else(|| SwarmError::AgentNotFound(agent_id.to_string()))?;

        // Calculate prediction error
        let error = (expected - actual).abs();
        let relative_error = error / (actual.abs() + 1e-8);

        // Score based on error and historical performance
        let error_score = 1.0 - relative_error.min(1.0);
        let trend_score = perf.trend.max(-1.0).min(1.0);
        let score = (error_score + trend_score) / 2.0;

        // Determine if adaptation is needed
        let should_adapt = relative_error > 0.2 || perf.avg_outcome < 0.5;

        // Generate suggestions
        let mut suggestions = Vec::new();
        if relative_error > 0.2 {
            suggestions.push(AdaptationSuggestion {
                parameter: "prediction_threshold".to_string(),
                current_value: 0.8,
                suggested_value: 0.85,
                reason: "High prediction error detected".to_string(),
            });
        }

        if perf.avg_confidence < 0.6 {
            suggestions.push(AdaptationSuggestion {
                parameter: "confidence_boost".to_string(),
                current_value: 1.0,
                suggested_value: 1.1,
                reason: "Low confidence scores".to_string(),
            });
        }

        Ok(ReasoningVerdict {
            experience_id: uuid::Uuid::new_v4().to_string(),
            score,
            should_adapt,
            suggested_changes: suggestions,
            confidence: perf.avg_confidence,
        })
    }

    /// Get agent performance metrics
    pub fn get_performance(&self, agent_id: &str) -> Option<AgentPerformance> {
        let metrics = self.metrics.read();
        metrics.get(agent_id).cloned()
    }

    /// Get total number of recorded experiences
    pub fn experience_count(&self) -> usize {
        self.experiences.read().len()
    }

    /// Get experiences for a specific agent
    pub fn get_agent_experiences(&self, agent_id: &str) -> Vec<ReasoningExperience> {
        self.experiences
            .read()
            .iter()
            .filter(|exp| exp.agent_id == agent_id)
            .cloned()
            .collect()
    }
}

impl Default for ReasoningBankClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OutcomeMetrics;

    #[tokio::test]
    async fn test_record_experience() {
        let client = ReasoningBankClient::new();

        let exp = ReasoningExperience {
            message_id: "test-1".to_string(),
            agent_id: "agent-1".to_string(),
            action: "test_action".to_string(),
            outcome: OutcomeMetrics {
                value: 0.8,
                confidence: 0.9,
                latency_ms: 10.0,
                additional: std::collections::HashMap::new(),
            },
            context: serde_json::json!({}),
            timestamp: chrono::Utc::now(),
        };

        assert!(client.record_experience(exp).await.is_ok());
        assert_eq!(client.experience_count(), 1);
    }

    #[tokio::test]
    async fn test_judge_experience() {
        let client = ReasoningBankClient::new();

        // Record some experiences first
        for i in 0..10 {
            let exp = ReasoningExperience {
                message_id: format!("test-{}", i),
                agent_id: "agent-1".to_string(),
                action: "test_action".to_string(),
                outcome: OutcomeMetrics {
                    value: 0.7 + (i as f64 * 0.01),
                    confidence: 0.8,
                    latency_ms: 10.0,
                    additional: std::collections::HashMap::new(),
                },
                context: serde_json::json!({}),
                timestamp: chrono::Utc::now(),
            };
            client.record_experience(exp).await.unwrap();
        }

        let verdict = client.judge_experience("agent-1", 0.75, 0.76).await.unwrap();

        assert!(verdict.score >= 0.0 && verdict.score <= 1.0);
        assert!(verdict.confidence >= 0.0 && verdict.confidence <= 1.0);
    }
}
