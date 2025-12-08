//! Parasitic opportunity detection and analysis

use crate::{
    MacchiavelianConfig, MarketData, ParasiticOpportunity, TalebianRiskError, WhaleDetection,
};
use serde::{Deserialize, Serialize};

/// Opportunity analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpportunityAnalysis {
    pub overall_score: f64,
    pub momentum_component: f64,
    pub volatility_component: f64,
    pub whale_alignment_component: f64,
    pub regime_component: f64,
    pub recommended_allocation: f64,
    pub confidence: f64,
}

/// Parasitic opportunity engine
pub struct ParasiticOpportunityEngine {
    config: MacchiavelianConfig,
    opportunity_history: Vec<ParasiticOpportunity>,
}

/// Opportunity status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpportunityStatus {
    pub active_opportunities: usize,
    pub avg_score: f64,
    pub success_rate: f64,
}

impl ParasiticOpportunityEngine {
    pub fn new(config: MacchiavelianConfig) -> Self {
        Self {
            config,
            opportunity_history: Vec::new(),
        }
    }

    pub fn analyze_opportunity(
        &mut self,
        market_data: &MarketData,
        whale_detection: &WhaleDetection,
        antifragility_score: f64,
    ) -> Result<OpportunityAnalysis, TalebianRiskError> {
        // Calculate momentum component
        let momentum_component = if !market_data.returns.is_empty() {
            let recent_momentum = market_data.returns.iter().rev().take(5).sum::<f64>();
            (recent_momentum * 10.0).tanh() // Normalize to [-1, 1] then shift to [0, 1]
        } else {
            0.5
        };

        // Volatility component (higher volatility = more opportunity for antifragile strategies)
        let volatility_component = (market_data.volatility * 10.0).min(1.0);

        // Whale alignment component
        let whale_alignment_component = if whale_detection.detected {
            whale_detection.confidence
        } else {
            0.2
        };

        // Regime component (based on antifragility score)
        let regime_component = antifragility_score;

        // Overall score (weighted average)
        let weights = [0.3, 0.25, 0.25, 0.2]; // momentum, volatility, whale, regime
        let components = [
            momentum_component,
            volatility_component,
            whale_alignment_component,
            regime_component,
        ];
        let overall_score = weights
            .iter()
            .zip(components.iter())
            .map(|(w, c)| w * c)
            .sum::<f64>();

        // Recommended allocation based on score and confidence
        let base_allocation = overall_score * self.config.kelly_fraction;
        let recommended_allocation = base_allocation.min(self.config.kelly_max_fraction);

        // Confidence based on consistency of signals
        let confidence = if overall_score > 0.7 {
            0.8
        } else if overall_score > 0.5 {
            0.6
        } else {
            0.4
        };

        let analysis = OpportunityAnalysis {
            overall_score,
            momentum_component,
            volatility_component,
            whale_alignment_component,
            regime_component,
            recommended_allocation,
            confidence,
        };

        // Store for tracking
        let opportunity = ParasiticOpportunity {
            id: format!("opp_{}", market_data.timestamp_unix),
            expected_return: overall_score * 0.02, // 2% max expected return
            risk_level: 1.0 - confidence,
            time_window: 3600, // 1 hour
            confidence,
            entry_price: market_data.price,
            exit_price: market_data.price * (1.0 + overall_score * 0.02),
            stop_loss: market_data.price * (1.0 - (1.0 - confidence) * 0.05),
            opportunity_score: overall_score,
            momentum_factor: momentum_component,
            volatility_factor: volatility_component,
            whale_alignment: whale_alignment_component,
            regime_factor: regime_component,
            recommended_allocation: overall_score * confidence * 0.1, // 10% max allocation
        };

        self.opportunity_history.push(opportunity);

        // Keep only recent history
        if self.opportunity_history.len() > 1000 {
            self.opportunity_history.drain(0..500);
        }

        Ok(analysis)
    }

    pub fn get_opportunity_status(&self) -> OpportunityStatus {
        let active_opportunities = self
            .opportunity_history
            .iter()
            .filter(|o| o.opportunity_score > self.config.parasitic_opportunity_threshold)
            .count();

        let avg_score = if !self.opportunity_history.is_empty() {
            self.opportunity_history
                .iter()
                .map(|o| o.opportunity_score)
                .sum::<f64>()
                / self.opportunity_history.len() as f64
        } else {
            0.0
        };

        let success_rate = 0.65; // Would be calculated from actual outcomes

        OpportunityStatus {
            active_opportunities,
            avg_score,
            success_rate,
        }
    }
}
