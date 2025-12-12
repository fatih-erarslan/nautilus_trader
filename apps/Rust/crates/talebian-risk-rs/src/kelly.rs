//! Kelly Criterion implementation for optimal position sizing

use crate::{MacchiavelianConfig, MarketData, TalebianRiskError, WhaleDetection};
use serde::{Deserialize, Serialize};

/// Kelly criterion calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellyCalculation {
    pub fraction: f64,
    pub adjusted_fraction: f64,
    pub risk_adjusted_size: f64,
    pub confidence: f64,
    pub expected_return: f64,
    pub variance: f64,
}

/// Kelly criterion engine
pub struct KellyEngine {
    config: MacchiavelianConfig,
    trade_history: Vec<TradeOutcome>,
}

#[derive(Debug, Clone)]
struct TradeOutcome {
    pub return_pct: f64,
    pub was_whale_trade: bool,
    pub momentum_score: f64,
}

impl KellyEngine {
    pub fn new(config: MacchiavelianConfig) -> Self {
        Self {
            config,
            trade_history: Vec::new(),
        }
    }

    pub fn calculate_kelly_fraction(
        &self,
        _market_data: &MarketData,
        _whale_detection: &WhaleDetection,
        expected_return: f64,
        confidence: f64,
    ) -> Result<KellyCalculation, TalebianRiskError> {
        let base_fraction = self.config.kelly_fraction;
        let adjusted_fraction = (base_fraction * confidence).min(self.config.kelly_max_fraction);

        Ok(KellyCalculation {
            fraction: base_fraction,
            adjusted_fraction,
            risk_adjusted_size: adjusted_fraction,
            confidence,
            expected_return,
            variance: 0.04, // Simplified
        })
    }

    pub fn record_trade_outcome(
        &mut self,
        return_pct: f64,
        was_whale_trade: bool,
        momentum_score: f64,
    ) -> Result<(), TalebianRiskError> {
        let outcome = TradeOutcome {
            return_pct,
            was_whale_trade,
            momentum_score,
        };

        self.trade_history.push(outcome);

        // Keep only recent history
        if self.trade_history.len() > 1000 {
            self.trade_history.drain(0..500);
        }

        Ok(())
    }

    pub fn get_kelly_status(&self) -> KellyStatus {
        let total_trades = self.trade_history.len();
        let avg_return = if total_trades == 0 {
            0.0
        } else {
            use crate::safe_math::safe_divide_with_fallback;
            let sum = self.trade_history.iter().map(|t| t.return_pct).sum::<f64>();
            safe_divide_with_fallback(sum, total_trades as f64, 0.0)
        };
        
        KellyStatus {
            total_trades,
            avg_return,
            current_fraction: self.config.kelly_fraction,
        }
    }
}

/// Kelly engine status
#[derive(Debug, Clone)]
pub struct KellyStatus {
    pub total_trades: usize,
    pub avg_return: f64,
    pub current_fraction: f64,
}
