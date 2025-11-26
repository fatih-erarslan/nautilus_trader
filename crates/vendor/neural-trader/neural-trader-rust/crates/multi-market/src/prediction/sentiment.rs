//! Sentiment Analysis for Prediction Markets

use crate::error::{MultiMarketError, Result};
use crate::prediction::polymarket::Market;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// Sentiment score for a market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScore {
    /// Overall sentiment (-1 to 1)
    pub overall: Decimal,
    /// Bullish sentiment (0 to 1)
    pub bullish: Decimal,
    /// Bearish sentiment (0 to 1)
    pub bearish: Decimal,
    /// Confidence level (0 to 1)
    pub confidence: Decimal,
}

/// Market sentiment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSentiment {
    /// Market ID
    pub market_id: String,
    /// Sentiment scores
    pub sentiment: SentimentScore,
    /// Volume trend
    pub volume_trend: VolumeTrend,
    /// Price momentum
    pub price_momentum: Decimal,
    /// Manipulation risk (0 to 1)
    pub manipulation_risk: Decimal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolumeTrend {
    Increasing,
    Decreasing,
    Stable,
}

/// Sentiment analyzer
pub struct SentimentAnalyzer {
    manipulation_threshold: Decimal,
}

impl SentimentAnalyzer {
    pub fn new() -> Self {
        Self {
            manipulation_threshold: dec!(0.7),
        }
    }

    /// Analyze market sentiment
    pub fn analyze(&self, market: &Market) -> Result<MarketSentiment> {
        let sentiment = self.calculate_sentiment(market)?;
        let volume_trend = self.detect_volume_trend(market);
        let price_momentum = self.calculate_momentum(market)?;
        let manipulation_risk = self.detect_manipulation(market)?;

        Ok(MarketSentiment {
            market_id: market.condition_id.clone(),
            sentiment,
            volume_trend,
            price_momentum,
            manipulation_risk,
        })
    }

    fn calculate_sentiment(&self, market: &Market) -> Result<SentimentScore> {
        if market.outcome_prices.is_empty() {
            return Err(MultiMarketError::MarketDataError("No price data".to_string()));
        }

        let prices: Vec<Decimal> = market.outcome_prices.values().copied().collect();
        let avg_price = prices.iter().sum::<Decimal>() / Decimal::from(prices.len());

        let bullish = prices.iter().filter(|&&p| p > avg_price).count() as f64 / prices.len() as f64;
        let bearish = 1.0 - bullish;

        Ok(SentimentScore {
            overall: Decimal::try_from(bullish - bearish).unwrap_or(Decimal::ZERO),
            bullish: Decimal::try_from(bullish).unwrap_or(Decimal::ZERO),
            bearish: Decimal::try_from(bearish).unwrap_or(Decimal::ZERO),
            confidence: dec!(0.8),
        })
    }

    fn detect_volume_trend(&self, market: &Market) -> VolumeTrend {
        // Simplified: would need historical data
        if market.volume > dec!(100000) {
            VolumeTrend::Increasing
        } else if market.volume < dec!(10000) {
            VolumeTrend::Decreasing
        } else {
            VolumeTrend::Stable
        }
    }

    fn calculate_momentum(&self, market: &Market) -> Result<Decimal> {
        // Simplified momentum calculation
        let prices: Vec<Decimal> = market.outcome_prices.values().copied().collect();
        if prices.is_empty() {
            return Ok(Decimal::ZERO);
        }

        let max_price = prices.iter().max().unwrap_or(&Decimal::ZERO);
        let min_price = prices.iter().min().unwrap_or(&Decimal::ZERO);

        if *min_price == Decimal::ZERO {
            return Ok(Decimal::ZERO);
        }

        Ok((*max_price - *min_price) / *min_price)
    }

    fn detect_manipulation(&self, market: &Market) -> Result<Decimal> {
        // Simplified: Check for extreme price movements and low liquidity
        let risk = if market.liquidity < dec!(10000) {
            dec!(0.6)
        } else if market.volume > dec!(1000000) {
            dec!(0.1)
        } else {
            dec!(0.3)
        };

        Ok(risk)
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
