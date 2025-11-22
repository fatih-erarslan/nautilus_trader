//! Result synthesis strategies for ensemble execution.

use crate::backend::{BackendId, ReasoningResult, ResultValue};
use crate::{RouterError, RouterResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Strategy for synthesizing results from multiple backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynthesisStrategy {
    /// Take the result with highest confidence
    HighestConfidence,
    /// Take the result with highest quality
    HighestQuality,
    /// Weighted average based on confidence
    WeightedAverage,
    /// Majority voting (for classification)
    MajorityVote,
    /// Median value (robust to outliers)
    Median,
    /// Fastest result
    Fastest,
    /// Custom ensemble with consensus
    Consensus,
}

/// Result synthesizer for combining outputs from multiple backends
pub struct ResultSynthesizer {
    strategy: SynthesisStrategy,
    /// Minimum agreement threshold for consensus
    consensus_threshold: f64,
}

impl ResultSynthesizer {
    /// Create a new synthesizer
    pub fn new(strategy: SynthesisStrategy) -> Self {
        Self {
            strategy,
            consensus_threshold: 0.6,
        }
    }

    /// Set consensus threshold
    pub fn with_consensus_threshold(mut self, threshold: f64) -> Self {
        self.consensus_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Synthesize multiple results into one
    pub fn synthesize(&self, results: Vec<ReasoningResult>) -> RouterResult<ReasoningResult> {
        if results.is_empty() {
            return Err(RouterError::SynthesisFailed(
                "No results to synthesize".to_string(),
            ));
        }

        if results.len() == 1 {
            return Ok(results.into_iter().next().unwrap());
        }

        match self.strategy {
            SynthesisStrategy::HighestConfidence => self.select_highest_confidence(results),
            SynthesisStrategy::HighestQuality => self.select_highest_quality(results),
            SynthesisStrategy::WeightedAverage => self.weighted_average(results),
            SynthesisStrategy::MajorityVote => self.majority_vote(results),
            SynthesisStrategy::Median => self.median_value(results),
            SynthesisStrategy::Fastest => self.select_fastest(results),
            SynthesisStrategy::Consensus => self.consensus(results),
        }
    }

    /// Select result with highest confidence
    fn select_highest_confidence(
        &self,
        results: Vec<ReasoningResult>,
    ) -> RouterResult<ReasoningResult> {
        results
            .into_iter()
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| RouterError::SynthesisFailed("No results".to_string()))
    }

    /// Select result with highest quality
    fn select_highest_quality(
        &self,
        results: Vec<ReasoningResult>,
    ) -> RouterResult<ReasoningResult> {
        results
            .into_iter()
            .max_by(|a, b| {
                a.quality
                    .partial_cmp(&b.quality)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| RouterError::SynthesisFailed("No results".to_string()))
    }

    /// Weighted average of scalar results
    fn weighted_average(&self, results: Vec<ReasoningResult>) -> RouterResult<ReasoningResult> {
        let total_weight: f64 = results.iter().map(|r| r.confidence).sum();

        if total_weight <= 0.0 {
            return self.select_highest_quality(results);
        }

        // Try to extract scalar values
        let scalar_results: Vec<(f64, f64)> = results
            .iter()
            .filter_map(|r| match &r.value {
                ResultValue::Scalar(v) => Some((*v, r.confidence)),
                ResultValue::Solution { fitness, .. } => Some((*fitness, r.confidence)),
                _ => None,
            })
            .collect();

        if scalar_results.is_empty() {
            // Can't average non-scalar results
            return self.select_highest_quality(results);
        }

        let weighted_sum: f64 = scalar_results.iter().map(|(v, w)| v * w).sum();
        let avg_value = weighted_sum / total_weight;

        let avg_confidence = results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64;
        let avg_quality = results.iter().map(|r| r.quality).sum::<f64>() / results.len() as f64;
        let total_latency: Duration = results.iter().map(|r| r.latency).sum();

        // Create aggregated metadata
        let metadata = serde_json::json!({
            "synthesis_strategy": "weighted_average",
            "num_results": results.len(),
            "backend_ids": results.iter().map(|r| r.backend_id.to_string()).collect::<Vec<_>>()
        });

        Ok(ReasoningResult {
            value: ResultValue::Scalar(avg_value),
            confidence: avg_confidence,
            quality: avg_quality,
            latency: total_latency / results.len() as u32,
            backend_id: BackendId::new("ensemble"),
            metadata,
        })
    }

    /// Majority voting for classification
    fn majority_vote(&self, results: Vec<ReasoningResult>) -> RouterResult<ReasoningResult> {
        let mut votes: HashMap<u32, (usize, f64)> = HashMap::new();

        for result in &results {
            if let ResultValue::Classification { class, .. } = &result.value {
                let entry = votes.entry(*class).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += result.confidence;
            }
        }

        if votes.is_empty() {
            return self.select_highest_quality(results);
        }

        let (winning_class, (vote_count, total_confidence)) = votes
            .into_iter()
            .max_by_key(|(_, (count, _))| *count)
            .unwrap();

        let agreement = vote_count as f64 / results.len() as f64;
        let avg_latency: Duration =
            results.iter().map(|r| r.latency).sum::<Duration>() / results.len() as u32;

        Ok(ReasoningResult {
            value: ResultValue::Classification {
                class: winning_class,
                probabilities: vec![agreement],
            },
            confidence: total_confidence / vote_count as f64,
            quality: agreement,
            latency: avg_latency,
            backend_id: BackendId::new("ensemble-vote"),
            metadata: serde_json::json!({
                "synthesis_strategy": "majority_vote",
                "agreement": agreement,
                "vote_count": vote_count
            }),
        })
    }

    /// Median value (robust to outliers)
    fn median_value(&self, results: Vec<ReasoningResult>) -> RouterResult<ReasoningResult> {
        let mut scalar_values: Vec<(f64, &ReasoningResult)> = results
            .iter()
            .filter_map(|r| match &r.value {
                ResultValue::Scalar(v) => Some((*v, r)),
                ResultValue::Solution { fitness, .. } => Some((*fitness, r)),
                _ => None,
            })
            .collect();

        if scalar_values.is_empty() {
            return self.select_highest_quality(results);
        }

        scalar_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mid = scalar_values.len() / 2;
        let median_value = if scalar_values.len() % 2 == 0 {
            (scalar_values[mid - 1].0 + scalar_values[mid].0) / 2.0
        } else {
            scalar_values[mid].0
        };

        let median_result = scalar_values[mid].1;

        Ok(ReasoningResult {
            value: ResultValue::Scalar(median_value),
            confidence: median_result.confidence,
            quality: median_result.quality,
            latency: median_result.latency,
            backend_id: BackendId::new("ensemble-median"),
            metadata: serde_json::json!({
                "synthesis_strategy": "median",
                "original_backend": median_result.backend_id.to_string()
            }),
        })
    }

    /// Select fastest result
    fn select_fastest(&self, results: Vec<ReasoningResult>) -> RouterResult<ReasoningResult> {
        results
            .into_iter()
            .min_by_key(|r| r.latency)
            .ok_or_else(|| RouterError::SynthesisFailed("No results".to_string()))
    }

    /// Consensus-based synthesis
    fn consensus(&self, results: Vec<ReasoningResult>) -> RouterResult<ReasoningResult> {
        // Extract comparable values
        let values: Vec<f64> = results
            .iter()
            .filter_map(|r| match &r.value {
                ResultValue::Scalar(v) => Some(*v),
                ResultValue::Solution { fitness, .. } => Some(*fitness),
                _ => None,
            })
            .collect();

        if values.is_empty() {
            return self.select_highest_confidence(results);
        }

        // Compute mean and standard deviation
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        // Filter results within consensus range (2 sigma)
        let consensus_range = 2.0 * std_dev;
        let consensus_results: Vec<_> = results
            .into_iter()
            .filter(|r| {
                let v = match &r.value {
                    ResultValue::Scalar(v) => *v,
                    ResultValue::Solution { fitness, .. } => *fitness,
                    _ => mean, // Include non-scalar in consensus
                };
                (v - mean).abs() <= consensus_range
            })
            .collect();

        let agreement = consensus_results.len() as f64 / values.len() as f64;

        if agreement < self.consensus_threshold {
            // No consensus - return highest quality
            return Err(RouterError::SynthesisFailed(format!(
                "No consensus: agreement {:.2} < threshold {:.2}",
                agreement, self.consensus_threshold
            )));
        }

        // Weighted average of consensus results
        self.weighted_average(consensus_results)
    }
}

impl Default for ResultSynthesizer {
    fn default() -> Self {
        Self::new(SynthesisStrategy::WeightedAverage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(value: f64, confidence: f64, quality: f64, backend: &str) -> ReasoningResult {
        ReasoningResult {
            value: ResultValue::Scalar(value),
            confidence,
            quality,
            latency: Duration::from_millis(10),
            backend_id: BackendId::new(backend),
            metadata: serde_json::Value::Null,
        }
    }

    #[test]
    fn test_highest_confidence() {
        let synth = ResultSynthesizer::new(SynthesisStrategy::HighestConfidence);
        let results = vec![
            make_result(1.0, 0.8, 0.9, "a"),
            make_result(2.0, 0.95, 0.7, "b"),
            make_result(3.0, 0.7, 0.85, "c"),
        ];

        let result = synth.synthesize(results).unwrap();
        assert_eq!(result.backend_id.0, "b");
    }

    #[test]
    fn test_highest_quality() {
        let synth = ResultSynthesizer::new(SynthesisStrategy::HighestQuality);
        let results = vec![
            make_result(1.0, 0.8, 0.9, "a"),
            make_result(2.0, 0.95, 0.7, "b"),
            make_result(3.0, 0.7, 0.95, "c"),
        ];

        let result = synth.synthesize(results).unwrap();
        assert_eq!(result.backend_id.0, "c");
    }

    #[test]
    fn test_weighted_average() {
        let synth = ResultSynthesizer::new(SynthesisStrategy::WeightedAverage);
        let results = vec![
            make_result(10.0, 0.5, 0.9, "a"),
            make_result(20.0, 0.5, 0.9, "b"),
        ];

        let result = synth.synthesize(results).unwrap();
        if let ResultValue::Scalar(v) = result.value {
            assert!((v - 15.0).abs() < 0.01);
        } else {
            panic!("Expected scalar result");
        }
    }

    #[test]
    fn test_median() {
        let synth = ResultSynthesizer::new(SynthesisStrategy::Median);
        let results = vec![
            make_result(1.0, 0.9, 0.9, "a"),
            make_result(2.0, 0.9, 0.9, "b"),
            make_result(100.0, 0.9, 0.9, "outlier"), // Outlier
        ];

        let result = synth.synthesize(results).unwrap();
        if let ResultValue::Scalar(v) = result.value {
            assert!((v - 2.0).abs() < 0.01); // Median ignores outlier
        } else {
            panic!("Expected scalar result");
        }
    }

    #[test]
    fn test_fastest() {
        let synth = ResultSynthesizer::new(SynthesisStrategy::Fastest);
        let results = vec![
            ReasoningResult {
                value: ResultValue::Scalar(1.0),
                confidence: 0.9,
                quality: 0.9,
                latency: Duration::from_millis(100),
                backend_id: BackendId::new("slow"),
                metadata: serde_json::Value::Null,
            },
            ReasoningResult {
                value: ResultValue::Scalar(2.0),
                confidence: 0.9,
                quality: 0.9,
                latency: Duration::from_millis(10),
                backend_id: BackendId::new("fast"),
                metadata: serde_json::Value::Null,
            },
        ];

        let result = synth.synthesize(results).unwrap();
        assert_eq!(result.backend_id.0, "fast");
    }

    #[test]
    fn test_single_result() {
        let synth = ResultSynthesizer::new(SynthesisStrategy::WeightedAverage);
        let results = vec![make_result(42.0, 0.9, 0.9, "only")];

        let result = synth.synthesize(results).unwrap();
        assert_eq!(result.backend_id.0, "only");
    }

    #[test]
    fn test_empty_results() {
        let synth = ResultSynthesizer::new(SynthesisStrategy::WeightedAverage);
        let results: Vec<ReasoningResult> = vec![];

        let result = synth.synthesize(results);
        assert!(result.is_err());
    }
}
