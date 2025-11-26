//! Consensus building mechanisms for collective decision-making

use crate::{error::*, types::*};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for consensus building
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Minimum agreement threshold (0.0 to 1.0)
    pub threshold: f64,

    /// Consensus algorithm to use
    pub algorithm: ConsensusAlgorithm,

    /// Timeout for consensus in seconds
    pub timeout_secs: u64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            threshold: 0.67, // 67% agreement required
            algorithm: ConsensusAlgorithm::Majority,
            timeout_secs: 60,
        }
    }
}

/// Consensus algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusAlgorithm {
    /// Simple majority voting
    Majority,

    /// Unanimous agreement required
    Unanimous,

    /// Weighted voting based on agent expertise
    Weighted,

    /// Byzantine fault tolerant consensus
    Byzantine,
}

/// Result of consensus building
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub success: bool,
    pub agreed_result: String,
    pub agreement_level: f64,
    pub participating_agents: usize,
}

/// Consensus builder for collective intelligence
pub struct ConsensusBuilder {
    config: ConsensusConfig,
}

impl ConsensusBuilder {
    /// Create a new consensus builder
    pub fn new(config: ConsensusConfig) -> Result<Self> {
        if config.threshold < 0.0 || config.threshold > 1.0 {
            return Err(HiveMindError::InvalidConfiguration(
                "Threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(Self { config })
    }

    /// Build consensus from multiple task results
    pub async fn build_consensus(&self, results: &[TaskResult]) -> Result<TaskResult> {
        if results.is_empty() {
            return Err(HiveMindError::ConsensusError(
                "No results to build consensus from".to_string(),
            ));
        }

        match self.config.algorithm {
            ConsensusAlgorithm::Majority => self.majority_consensus(results).await,
            ConsensusAlgorithm::Unanimous => self.unanimous_consensus(results).await,
            ConsensusAlgorithm::Weighted => self.weighted_consensus(results).await,
            ConsensusAlgorithm::Byzantine => self.byzantine_consensus(results).await,
        }
    }

    /// Simple majority voting
    async fn majority_consensus(&self, results: &[TaskResult]) -> Result<TaskResult> {
        let mut votes: HashMap<String, usize> = HashMap::new();

        // Count votes for each unique output
        for result in results {
            if result.success {
                *votes.entry(result.output.clone()).or_insert(0) += 1;
            }
        }

        // Find the most voted output
        let total_votes: usize = votes.values().sum();
        let (winning_output, winning_votes) = votes
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .ok_or_else(|| HiveMindError::ConsensusError("No valid votes".to_string()))?;

        let agreement_level = winning_votes as f64 / total_votes as f64;

        if agreement_level >= self.config.threshold {
            Ok(TaskResult {
                task_id: results[0].task_id.clone(),
                success: true,
                output: winning_output,
                agent_id: AgentId::from_string("consensus".to_string()),
                completed_at: chrono::Utc::now(),
            })
        } else {
            Err(HiveMindError::ConsensusError(format!(
                "Consensus threshold not met: {:.2}% < {:.2}%",
                agreement_level * 100.0,
                self.config.threshold * 100.0
            )))
        }
    }

    /// Unanimous consensus (all agents must agree)
    async fn unanimous_consensus(&self, results: &[TaskResult]) -> Result<TaskResult> {
        if results.is_empty() {
            return Err(HiveMindError::ConsensusError("No results".to_string()));
        }

        let first_output = &results[0].output;

        // Check if all outputs are identical
        let unanimous = results.iter().all(|r| r.success && r.output == *first_output);

        if unanimous {
            Ok(TaskResult {
                task_id: results[0].task_id.clone(),
                success: true,
                output: first_output.clone(),
                agent_id: AgentId::from_string("consensus".to_string()),
                completed_at: chrono::Utc::now(),
            })
        } else {
            Err(HiveMindError::ConsensusError(
                "Unanimous agreement not reached".to_string(),
            ))
        }
    }

    /// Weighted consensus based on agent expertise
    async fn weighted_consensus(&self, results: &[TaskResult]) -> Result<TaskResult> {
        // For now, treat all agents equally (implement weighting later)
        self.majority_consensus(results).await
    }

    /// Byzantine fault tolerant consensus
    async fn byzantine_consensus(&self, results: &[TaskResult]) -> Result<TaskResult> {
        // Simplified BFT: require 2f+1 out of 3f+1 nodes to agree
        let n = results.len();
        let f = (n - 1) / 3; // Maximum number of faulty nodes
        let required = 2 * f + 1;

        let mut votes: HashMap<String, usize> = HashMap::new();

        for result in results {
            if result.success {
                *votes.entry(result.output.clone()).or_insert(0) += 1;
            }
        }

        if let Some((winning_output, winning_votes)) =
            votes.into_iter().max_by_key(|(_, count)| *count)
        {
            if winning_votes >= required {
                return Ok(TaskResult {
                    task_id: results[0].task_id.clone(),
                    success: true,
                    output: winning_output,
                    agent_id: AgentId::from_string("consensus-bft".to_string()),
                    completed_at: chrono::Utc::now(),
                });
            }
        }

        Err(HiveMindError::ConsensusError(
            "BFT consensus not reached".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_results(outputs: Vec<&str>) -> Vec<TaskResult> {
        outputs
            .into_iter()
            .map(|output| TaskResult {
                task_id: "test-task".to_string(),
                success: true,
                output: output.to_string(),
                agent_id: AgentId::new(),
                completed_at: chrono::Utc::now(),
            })
            .collect()
    }

    #[tokio::test]
    async fn test_majority_consensus() {
        let mut config = ConsensusConfig::default();
        config.threshold = 0.5; // Lower threshold to 50% for this test
        let builder = ConsensusBuilder::new(config).unwrap();

        let results = create_test_results(vec!["A", "A", "B"]);
        let consensus = builder.build_consensus(&results).await;

        assert!(consensus.is_ok());
        assert_eq!(consensus.unwrap().output, "A");
    }

    #[tokio::test]
    async fn test_unanimous_consensus() {
        let mut config = ConsensusConfig::default();
        config.algorithm = ConsensusAlgorithm::Unanimous;
        let builder = ConsensusBuilder::new(config).unwrap();

        let results = create_test_results(vec!["A", "A", "A"]);
        let consensus = builder.build_consensus(&results).await;

        assert!(consensus.is_ok());
        assert_eq!(consensus.unwrap().output, "A");
    }

    #[tokio::test]
    async fn test_consensus_failure() {
        let config = ConsensusConfig::default();
        let builder = ConsensusBuilder::new(config).unwrap();

        let results = create_test_results(vec!["A", "B", "C"]);
        let consensus = builder.build_consensus(&results).await;

        // With 67% threshold, this should fail (each has 33%)
        assert!(consensus.is_err());
    }
}
