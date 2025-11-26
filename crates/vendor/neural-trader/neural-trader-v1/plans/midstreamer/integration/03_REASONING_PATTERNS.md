# ReasoningBank Self-Learning Integration

**Status:** Design Complete
**Priority:** High
**Complexity:** Advanced
**Dependencies:** AgentDB, midstreamer, temporal analysis

---

## Overview

ReasoningBank enables **self-learning** through:
1. **Experience Recording** - Track all pattern matching attempts and outcomes
2. **Trajectory Analysis** - Learn which patterns lead to profitable trades
3. **Verdict Judgment** - Evaluate if pattern predictions matched reality
4. **Memory Distillation** - Extract successful patterns into long-term memory
5. **Adaptive Improvement** - Continuously refine pattern matching thresholds

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│         ReasoningBank Self-Learning Cycle                    │
└──────────────────────────────────────────────────────────────┘

1. EXPERIENCE RECORDING
   ├─ Pattern matched (DTW/LCS)
   ├─ Expected outcome predicted
   ├─ Trade executed based on prediction
   └─ Actual outcome observed
        ↓
2. TRAJECTORY TRACKING
   ├─ Store in AgentDB: [pattern, prediction, outcome]
   ├─ Compute success rate per pattern type
   └─ Identify high-performance patterns
        ↓
3. VERDICT JUDGMENT
   ├─ Compare predicted vs actual outcome
   ├─ Calculate accuracy, Sharpe, drawdown
   └─ Assign quality score (0-1)
        ↓
4. MEMORY DISTILLATION
   ├─ Extract successful patterns (score > 0.8)
   ├─ Store in long-term memory (AgentDB)
   └─ Create pattern clusters by similarity
        ↓
5. ADAPTIVE IMPROVEMENT
   ├─ Adjust similarity thresholds
   ├─ Update confidence calculations
   └─ Refine pattern selection criteria
        ↓
   (Repeat with improved parameters)
```

---

## Implementation

### 1. Experience Recording

```rust
// neural-trader-rust/crates/reasoning/src/pattern_learning.rs

use agentdb::AgentDB;
use chrono::{DateTime, Utc};

pub struct PatternLearningEngine {
    agentdb: Arc<AgentDB>,
    experiences: Arc<RwLock<Vec<PatternExperience>>>,
    trajectories: Arc<RwLock<HashMap<String, PatternTrajectory>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternExperience {
    pub id: String,
    pub pattern_type: String,
    pub pattern_vector: Vec<f32>,
    pub similarity: f64,
    pub confidence: f64,
    pub predicted_outcome: f64,  // Expected return
    pub actual_outcome: Option<f64>,  // Actual return (filled later)
    pub market_context: MarketContext,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    pub symbol: String,
    pub timeframe: String,
    pub volatility: f64,
    pub volume: f64,
    pub trend: String,
    pub sentiment: f64,
}

impl PatternLearningEngine {
    /// Record a new pattern matching experience
    pub async fn record_experience(&self, experience: PatternExperience) -> Result<()> {
        // Store in short-term memory
        self.experiences.write().await.push(experience.clone());

        // Store in AgentDB with vector embedding
        self.agentdb.insert(
            "pattern_experiences",
            &experience,
            Some(&experience.pattern_vector),
        ).await?;

        tracing::info!(
            "Recorded experience: {} (similarity: {:.2}, predicted: {:.4})",
            experience.pattern_type,
            experience.similarity,
            experience.predicted_outcome
        );

        Ok(())
    }

    /// Update experience with actual outcome
    pub async fn update_outcome(&self, experience_id: &str, actual_outcome: f64) -> Result<()> {
        // Update in memory
        let mut experiences = self.experiences.write().await;
        if let Some(exp) = experiences.iter_mut().find(|e| e.id == experience_id) {
            exp.actual_outcome = Some(actual_outcome);

            // Compute accuracy
            let prediction_error = (exp.predicted_outcome - actual_outcome).abs();
            let accuracy = 1.0 - (prediction_error / exp.predicted_outcome.abs().max(0.01));

            tracing::info!(
                "Updated outcome for {}: predicted={:.4}, actual={:.4}, accuracy={:.2}%",
                experience_id,
                exp.predicted_outcome,
                actual_outcome,
                accuracy * 100.0
            );

            // Judge if this was a good prediction
            let verdict = self.judge_prediction(exp).await?;

            // If good prediction, add to long-term memory
            if verdict.quality_score > 0.8 {
                self.distill_to_memory(exp, verdict).await?;
            }
        }

        Ok(())
    }

    /// Judge prediction quality
    async fn judge_prediction(&self, exp: &PatternExperience) -> Result<PatternVerdict> {
        let actual = exp.actual_outcome
            .ok_or_else(|| anyhow!("Outcome not set"))?;

        let predicted = exp.predicted_outcome;

        // Calculate metrics
        let prediction_error = (predicted - actual).abs();
        let direction_correct = (predicted > 0.0) == (actual > 0.0);
        let magnitude_error = (prediction_error / actual.abs().max(0.001)).min(1.0);

        // Quality score (0-1)
        let quality_score = if direction_correct {
            // Correct direction: score based on magnitude accuracy
            0.5 + (0.5 * (1.0 - magnitude_error))
        } else {
            // Wrong direction: score based on how wrong
            0.5 * (1.0 - magnitude_error)
        };

        let verdict = PatternVerdict {
            experience_id: exp.id.clone(),
            quality_score,
            direction_correct,
            magnitude_error,
            should_learn: quality_score > 0.7,
            should_adapt: quality_score < 0.3,
            suggested_changes: self.suggest_adaptations(exp, quality_score).await?,
        };

        tracing::info!(
            "Pattern verdict: score={:.2}, direction={}, magnitude_err={:.2}",
            quality_score,
            direction_correct,
            magnitude_error
        );

        Ok(verdict)
    }

    /// Distill successful pattern to long-term memory
    async fn distill_to_memory(
        &self,
        exp: &PatternExperience,
        verdict: PatternVerdict,
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

        // Store in AgentDB long-term memory
        self.agentdb.insert(
            "distilled_patterns",
            &distilled,
            Some(&exp.pattern_vector),
        ).await?;

        tracing::info!(
            "Distilled pattern to memory: {} (score: {:.2})",
            exp.pattern_type,
            verdict.quality_score
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

        // If quality is low, suggest threshold adjustments
        if quality_score < 0.5 {
            if exp.similarity < 0.9 {
                adaptations.push(Adaptation {
                    parameter: "similarity_threshold".to_string(),
                    current_value: exp.similarity,
                    suggested_value: exp.similarity + 0.05,
                    reason: "Low quality predictions - require higher similarity".to_string(),
                });
            }

            if exp.confidence < 0.9 {
                adaptations.push(Adaptation {
                    parameter: "confidence_threshold".to_string(),
                    current_value: exp.confidence,
                    suggested_value: exp.confidence + 0.05,
                    reason: "Low quality predictions - require higher confidence".to_string(),
                });
            }
        }

        // If quality is high, we can relax thresholds slightly
        if quality_score > 0.9 {
            adaptations.push(Adaptation {
                parameter: "similarity_threshold".to_string(),
                current_value: exp.similarity,
                suggested_value: (exp.similarity - 0.02).max(0.7),
                reason: "High quality predictions - can relax threshold".to_string(),
            });
        }

        Ok(adaptations)
    }

    /// Build trajectory of pattern performance over time
    pub async fn build_trajectory(&self, pattern_type: &str) -> Result<PatternTrajectory> {
        // Query all experiences for this pattern type
        let experiences: Vec<PatternExperience> = self.agentdb.query(
            "pattern_experiences",
            format!("pattern_type = '{}'", pattern_type),
            None,
            1000,
        ).await?;

        // Filter to only completed experiences
        let completed: Vec<_> = experiences.iter()
            .filter(|e| e.actual_outcome.is_some())
            .collect();

        if completed.is_empty() {
            return Err(anyhow!("No completed experiences for pattern type: {}", pattern_type));
        }

        // Calculate metrics
        let total = completed.len();
        let profitable = completed.iter()
            .filter(|e| e.actual_outcome.unwrap() > 0.0)
            .count();

        let success_rate = profitable as f64 / total as f64;

        let avg_return = completed.iter()
            .map(|e| e.actual_outcome.unwrap())
            .sum::<f64>() / total as f64;

        let returns: Vec<f64> = completed.iter()
            .map(|e| e.actual_outcome.unwrap())
            .collect();

        let sharpe_ratio = calculate_sharpe_ratio(&returns);

        let trajectory = PatternTrajectory {
            pattern_type: pattern_type.to_string(),
            sample_count: total,
            success_rate,
            avg_return,
            sharpe_ratio,
            best_return: returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            worst_return: returns.iter().cloned().fold(f64::INFINITY, f64::min),
            experiences: completed.into_iter().cloned().collect(),
            last_updated: Utc::now(),
        };

        tracing::info!(
            "Built trajectory for {}: {} samples, {:.1}% success, Sharpe={:.2}",
            pattern_type,
            total,
            success_rate * 100.0,
            sharpe_ratio
        );

        Ok(trajectory)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternVerdict {
    pub experience_id: String,
    pub quality_score: f64,  // 0-1
    pub direction_correct: bool,
    pub magnitude_error: f64,
    pub should_learn: bool,
    pub should_adapt: bool,
    pub suggested_changes: Vec<Adaptation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adaptation {
    pub parameter: String,
    pub current_value: f64,
    pub suggested_value: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledPattern {
    pub pattern_type: String,
    pub pattern_vector: Vec<f32>,
    pub success_rate: f64,
    pub avg_return: f64,
    pub confidence_threshold: f64,
    pub similarity_threshold: f64,
    pub market_conditions: MarketContext,
    pub sample_count: usize,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTrajectory {
    pub pattern_type: String,
    pub sample_count: usize,
    pub success_rate: f64,
    pub avg_return: f64,
    pub sharpe_ratio: f64,
    pub best_return: f64,
    pub worst_return: f64,
    pub experiences: Vec<PatternExperience>,
    pub last_updated: DateTime<Utc>,
}

fn calculate_sharpe_ratio(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        0.0
    } else {
        mean / std_dev * (252.0_f64).sqrt()  // Annualized Sharpe
    }
}
```

---

### 2. Adaptive Pattern Matching

```rust
// neural-trader-rust/crates/strategies/src/adaptive_pattern_matcher.rs

pub struct AdaptivePatternMatcher {
    learning_engine: Arc<PatternLearningEngine>,
    midstreamer: Arc<MidstreamerClient>,
    thresholds: Arc<RwLock<MatchingThresholds>>,
}

#[derive(Debug, Clone)]
struct MatchingThresholds {
    pub similarity_threshold: f64,
    pub confidence_threshold: f64,
    pub min_sample_size: usize,
}

impl AdaptivePatternMatcher {
    /// Match pattern with adaptive learning
    pub async fn match_with_learning(
        &self,
        current_pattern: &[f32],
        pattern_type: &str,
    ) -> Result<Option<TradingSignal>> {
        // Get current thresholds
        let thresholds = self.thresholds.read().await.clone();

        // Query similar historical patterns from AgentDB
        let historical = self.learning_engine.agentdb.query_similar(
            "distilled_patterns",
            current_pattern,
            10,  // Top 10 similar patterns
        ).await?;

        if historical.is_empty() {
            return Ok(None);
        }

        // Find best match using midstreamer DTW
        let mut best_match: Option<(DistilledPattern, f64)> = None;

        for pattern in historical {
            let similarity = self.midstreamer.compare_dtw(
                current_pattern,
                &pattern.pattern_vector,
                None,
            ).await?;

            if similarity.similarity >= thresholds.similarity_threshold {
                if best_match.is_none() || similarity.similarity > best_match.as_ref().unwrap().1 {
                    best_match = Some((pattern, similarity.similarity));
                }
            }
        }

        if let Some((pattern, similarity)) = best_match {
            // Calculate confidence based on historical success rate
            let confidence = pattern.success_rate * similarity;

            if confidence >= thresholds.confidence_threshold {
                // Record this as a new experience
                let experience = PatternExperience {
                    id: Uuid::new_v4().to_string(),
                    pattern_type: pattern_type.to_string(),
                    pattern_vector: current_pattern.to_vec(),
                    similarity,
                    confidence,
                    predicted_outcome: pattern.avg_return,
                    actual_outcome: None,  // Will be filled later
                    market_context: get_current_market_context().await?,
                    timestamp: Utc::now(),
                };

                self.learning_engine.record_experience(experience.clone()).await?;

                // Generate trading signal
                Ok(Some(TradingSignal {
                    direction: if pattern.avg_return > 0.0 { "LONG" } else { "SHORT" }.to_string(),
                    confidence,
                    expected_return: pattern.avg_return,
                    experience_id: experience.id,
                    pattern_type: pattern_type.to_string(),
                }))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Update thresholds based on recent performance
    pub async fn adapt_thresholds(&self) -> Result<()> {
        // Get recent experiences
        let experiences = self.learning_engine.experiences.read().await;
        let recent: Vec<_> = experiences.iter()
            .rev()
            .take(100)
            .filter(|e| e.actual_outcome.is_some())
            .collect();

        if recent.len() < 20 {
            return Ok(());  // Not enough data
        }

        // Calculate recent performance
        let profitable = recent.iter()
            .filter(|e| e.actual_outcome.unwrap() > 0.0)
            .count();

        let success_rate = profitable as f64 / recent.len() as f64;

        // Adapt thresholds
        let mut thresholds = self.thresholds.write().await;

        if success_rate < 0.55 {
            // Performance is poor - make thresholds stricter
            thresholds.similarity_threshold = (thresholds.similarity_threshold + 0.02).min(0.95);
            thresholds.confidence_threshold = (thresholds.confidence_threshold + 0.02).min(0.90);

            tracing::info!(
                "Adapting thresholds (stricter): similarity={:.2}, confidence={:.2}",
                thresholds.similarity_threshold,
                thresholds.confidence_threshold
            );
        } else if success_rate > 0.75 {
            // Performance is good - can relax thresholds slightly
            thresholds.similarity_threshold = (thresholds.similarity_threshold - 0.01).max(0.70);
            thresholds.confidence_threshold = (thresholds.confidence_threshold - 0.01).max(0.60);

            tracing::info!(
                "Adapting thresholds (relaxed): similarity={:.2}, confidence={:.2}",
                thresholds.similarity_threshold,
                thresholds.confidence_threshold
            );
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub direction: String,
    pub confidence: f64,
    pub expected_return: f64,
    pub experience_id: String,
    pub pattern_type: String,
}
```

---

## Performance Metrics

### Expected Learning Curve

```
Success Rate Over Time (Self-Learning)
────────────────────────────────────────

75% │                                    ╭──────
    │                                ╭───╯
70% │                            ╭───╯
    │                        ╭───╯
65% │                    ╭───╯
    │                ╭───╯
60% │            ╭───╯
    │        ╭───╯
55% │    ╭───╯
    │╭───╯
50% ├────────────────────────────────────────────
    0   100  200  300  400  500  600  700  800
              Number of Experiences

Initial: 50-55% (random)
Week 1:  55-60% (basic learning)
Week 2:  60-65% (pattern recognition)
Month 1: 65-70% (refined thresholds)
Month 3: 70-75% (mature system)
```

---

## Next Steps

1. Implement PatternLearningEngine
2. Add AdaptivePatternMatcher
3. Create experience tracking dashboard
4. Set up automated threshold adaptation
5. Monitor learning progress metrics

---

**Cross-References:**
- [Master Plan](../00_MASTER_PLAN.md)
- [QUIC Coordination](../architecture/02_QUIC_COORDINATION.md)
- [20-Year Evolution](../evolution/04_FUTURE_VISION.md)
