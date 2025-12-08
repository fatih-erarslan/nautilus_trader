use crate::{
    ensemble::ModelPredictor,
    types::{MarketCondition, ModelType},
};
use anyhow::Result;
use std::{collections::HashMap, sync::Arc};
use tracing::debug;

/// Model selector based on market conditions
pub struct ModelSelector {
    /// Model selection rules
    selection_rules: HashMap<MarketCondition, Vec<ModelType>>,
}

impl ModelSelector {
    /// Create new model selector
    pub fn new() -> Self {
        let mut selection_rules = HashMap::new();
        
        // Define model selection rules for each market condition
        selection_rules.insert(
            MarketCondition::Trending,
            vec![
                ModelType::Transformer,  // Good for capturing long-term dependencies
                ModelType::LSTM,         // Excellent for sequential patterns
                ModelType::XGBoost,      // Fast and accurate
                ModelType::LightGBM,     // Fast and accurate
            ],
        );
        
        selection_rules.insert(
            MarketCondition::Ranging,
            vec![
                ModelType::XGBoost,      // Good for non-linear patterns
                ModelType::LightGBM,     // Good for non-linear patterns
                ModelType::LSTM,         // Can capture mean reversion
                ModelType::NBeats,       // Good for stationary series
            ],
        );
        
        selection_rules.insert(
            MarketCondition::HighVolatility,
            vec![
                ModelType::LSTM,         // Handles volatility well
                ModelType::GRU,          // Faster than LSTM, good for volatility
                ModelType::Transformer,  // Attention mechanism for sudden changes
                ModelType::IsolationForest, // Detect outliers
            ],
        );
        
        selection_rules.insert(
            MarketCondition::LowVolatility,
            vec![
                ModelType::XGBoost,      // Efficient for stable patterns
                ModelType::LightGBM,     // Efficient for stable patterns
                ModelType::NHits,        // Good for smooth series
                ModelType::NBeats,       // Good for decomposition
            ],
        );
        
        selection_rules.insert(
            MarketCondition::Breakout,
            vec![
                ModelType::Transformer,  // Quick adaptation to new patterns
                ModelType::LSTM,         // Sequential pattern detection
                ModelType::XGBoost,      // Fast inference
                ModelType::IsolationForest, // Detect unusual patterns
            ],
        );
        
        selection_rules.insert(
            MarketCondition::Reversal,
            vec![
                ModelType::LSTM,         // Good for detecting pattern changes
                ModelType::GRU,          // Efficient pattern change detection
                ModelType::Transformer,  // Attention to key reversal points
                ModelType::XGBoost,      // Non-linear pattern recognition
            ],
        );
        
        selection_rules.insert(
            MarketCondition::Anomalous,
            vec![
                ModelType::IsolationForest, // Primary anomaly detector
                ModelType::XGBoost,      // Robust to outliers
                ModelType::LightGBM,     // Robust to outliers
                ModelType::Transformer,  // Can adapt to unusual patterns
            ],
        );
        
        Self { selection_rules }
    }
    
    /// Select models based on market condition
    pub fn select_models(
        &self,
        market_condition: MarketCondition,
        available_models: &HashMap<ModelType, Arc<dyn ModelPredictor>>,
    ) -> Result<Vec<Arc<dyn ModelPredictor>>> {
        // Get preferred models for this market condition
        let preferred_types = self.selection_rules
            .get(&market_condition)
            .cloned()
            .unwrap_or_else(|| {
                // Default selection if no specific rule
                vec![
                    ModelType::XGBoost,
                    ModelType::LSTM,
                    ModelType::Transformer,
                    ModelType::LightGBM,
                ]
            });
        
        // Select available models in preference order
        let mut selected_models = Vec::new();
        
        for model_type in preferred_types {
            if let Some(model) = available_models.get(&model_type) {
                selected_models.push(Arc::clone(model));
            }
        }
        
        // Ensure we have at least 2 models
        if selected_models.len() < 2 {
            debug!("Only {} models selected, adding more", selected_models.len());
            
            // Add any available models not already selected
            for (model_type, model) in available_models {
                if !selected_models.iter().any(|m| m.model_type() == *model_type) {
                    selected_models.push(Arc::clone(model));
                    if selected_models.len() >= 3 {
                        break;
                    }
                }
            }
        }
        
        if selected_models.is_empty() {
            return Err(anyhow::anyhow!("No models available for selection"));
        }
        
        debug!(
            "Selected {} models for {:?} market condition",
            selected_models.len(),
            market_condition
        );
        
        Ok(selected_models)
    }
    
    /// Update selection rules based on performance
    pub fn update_rules(
        &mut self,
        market_condition: MarketCondition,
        model_performances: &[(ModelType, f64)], // (model_type, performance_score)
    ) {
        // Sort models by performance
        let mut sorted_models: Vec<ModelType> = model_performances
            .iter()
            .filter(|(_, score)| *score > 0.0)
            .map(|(model_type, _)| *model_type)
            .collect();
        
        sorted_models.sort_by(|a, b| {
            let score_a = model_performances.iter()
                .find(|(mt, _)| mt == a)
                .map(|(_, s)| s)
                .unwrap_or(&0.0);
            let score_b = model_performances.iter()
                .find(|(mt, _)| mt == b)
                .map(|(_, s)| s)
                .unwrap_or(&0.0);
            score_b.partial_cmp(score_a).unwrap()
        });
        
        // Update rules with top performing models
        if sorted_models.len() >= 3 {
            self.selection_rules.insert(
                market_condition,
                sorted_models.into_iter().take(4).collect(),
            );
            
            debug!("Updated selection rules for {:?}", market_condition);
        }
    }
}