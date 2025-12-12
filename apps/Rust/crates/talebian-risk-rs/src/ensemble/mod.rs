//! Ensemble methods for risk management

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleResult {
    pub consensus_score: f64,
    pub individual_scores: Vec<f64>,
    pub confidence: f64,
}

pub fn ensemble_predict(models: &[f64]) -> EnsembleResult {
    let consensus_score = models.iter().sum::<f64>() / models.len() as f64;

    EnsembleResult {
        consensus_score,
        individual_scores: models.to_vec(),
        confidence: 0.8,
    }
}
