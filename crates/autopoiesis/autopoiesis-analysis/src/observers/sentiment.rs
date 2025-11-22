//! Sentiment observer implementation

use crate::prelude::*;

/// Observer for market sentiment analysis
#[derive(Debug, Clone)]
pub struct SentimentObserver {
    pub sources: Vec<SentimentSource>,
    pub weights: Vec<f64>,
    pub decay_factor: f64,
}

#[derive(Debug, Clone)]
pub enum SentimentSource {
    News,
    Social,
    Technical,
    OrderFlow,
    Volatility,
}

impl SentimentObserver {
    pub fn new(sources: Vec<SentimentSource>, weights: Vec<f64>, decay_factor: f64) -> Self {
        Self { sources, weights, decay_factor }
    }
    
    pub fn observe(&self, market_data: &MarketData) -> SentimentReading {
        let mut total_sentiment = 0.0;
        let mut total_weight = 0.0;
        
        for (source, &weight) in self.sources.iter().zip(self.weights.iter()) {
            let sentiment = self.calculate_source_sentiment(source, market_data);
            total_sentiment += sentiment * weight;
            total_weight += weight;
        }
        
        let normalized_sentiment = if total_weight > 0.0 {
            total_sentiment / total_weight
        } else {
            0.0
        };
        
        SentimentReading {
            overall_sentiment: normalized_sentiment,
            confidence: self.calculate_confidence(market_data),
            sources: self.get_source_breakdown(market_data),
            trend: self.calculate_sentiment_trend(market_data),
        }
    }
    
    fn calculate_source_sentiment(&self, source: &SentimentSource, data: &MarketData) -> f64 {
        match source {
            SentimentSource::News => {
                // Analyze news sentiment (simplified)
                data.news_sentiment.unwrap_or(0.0)
            },
            SentimentSource::Social => {
                // Social media sentiment
                data.social_sentiment.unwrap_or(0.0)
            },
            SentimentSource::Technical => {
                // Technical indicators sentiment
                self.calculate_technical_sentiment(data)
            },
            SentimentSource::OrderFlow => {
                // Order flow sentiment
                self.calculate_order_flow_sentiment(data)
            },
            SentimentSource::Volatility => {
                // Volatility-based sentiment (VIX-like)
                self.calculate_volatility_sentiment(data)
            },
        }
    }
    
    fn calculate_technical_sentiment(&self, data: &MarketData) -> f64 {
        let mut technical_score = 0.0;
        let mut indicator_count = 0;
        
        // RSI sentiment
        if let Some(rsi) = data.rsi {
            technical_score += if rsi > 70.0 { -0.5 } else if rsi < 30.0 { 0.5 } else { 0.0 };
            indicator_count += 1;
        }
        
        // Moving average sentiment
        if let Some(ma_signal) = data.ma_signal {
            technical_score += ma_signal;
            indicator_count += 1;
        }
        
        // MACD sentiment
        if let Some(macd) = data.macd_signal {
            technical_score += macd;
            indicator_count += 1;
        }
        
        if indicator_count > 0 {
            technical_score / indicator_count as f64
        } else {
            0.0
        }
    }
    
    fn calculate_order_flow_sentiment(&self, data: &MarketData) -> f64 {
        let buy_pressure = data.buy_volume.unwrap_or(0.0);
        let sell_pressure = data.sell_volume.unwrap_or(0.0);
        let total_volume = buy_pressure + sell_pressure;
        
        if total_volume > 0.0 {
            (buy_pressure - sell_pressure) / total_volume
        } else {
            0.0
        }
    }
    
    fn calculate_volatility_sentiment(&self, data: &MarketData) -> f64 {
        let current_vol = data.volatility.unwrap_or(0.2);
        let historical_avg = data.historical_volatility.unwrap_or(0.2);
        
        // High volatility typically indicates fear (negative sentiment)
        let vol_ratio = current_vol / historical_avg;
        if vol_ratio > 1.5 {
            -0.7 // High fear
        } else if vol_ratio < 0.7 {
            0.3 // Low fear (complacency)
        } else {
            0.0 // Normal
        }
    }
    
    fn calculate_confidence(&self, data: &MarketData) -> f64 {
        let volume_confidence = data.volume.unwrap_or(0.0) / data.average_volume.unwrap_or(1.0);
        let spread_confidence = 1.0 / (1.0 + data.bid_ask_spread.unwrap_or(0.01));
        let volatility_confidence = 1.0 / (1.0 + data.volatility.unwrap_or(0.2));
        
        (volume_confidence * spread_confidence * volatility_confidence).cbrt().min(1.0)
    }
    
    fn get_source_breakdown(&self, data: &MarketData) -> Vec<(SentimentSource, f64)> {
        self.sources.iter()
            .map(|source| (source.clone(), self.calculate_source_sentiment(source, data)))
            .collect()
    }
    
    fn calculate_sentiment_trend(&self, data: &MarketData) -> f64 {
        // Simple trend calculation based on recent sentiment changes
        let current_sentiment = data.current_sentiment.unwrap_or(0.0);
        let previous_sentiment = data.previous_sentiment.unwrap_or(0.0);
        
        current_sentiment - previous_sentiment
    }
}

#[derive(Debug, Clone)]
pub struct SentimentReading {
    pub overall_sentiment: f64, // -1.0 to 1.0
    pub confidence: f64,        // 0.0 to 1.0
    pub sources: Vec<(SentimentSource, f64)>,
    pub trend: f64,             // sentiment change rate
}

#[derive(Debug, Clone, Default)]
pub struct MarketData {
    pub price: Option<f64>,
    pub volume: Option<f64>,
    pub average_volume: Option<f64>,
    pub volatility: Option<f64>,
    pub historical_volatility: Option<f64>,
    pub bid_ask_spread: Option<f64>,
    pub rsi: Option<f64>,
    pub ma_signal: Option<f64>,
    pub macd_signal: Option<f64>,
    pub buy_volume: Option<f64>,
    pub sell_volume: Option<f64>,
    pub news_sentiment: Option<f64>,
    pub social_sentiment: Option<f64>,
    pub current_sentiment: Option<f64>,
    pub previous_sentiment: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sentiment_observer() {
        let observer = SentimentObserver::new(
            vec![SentimentSource::Technical, SentimentSource::OrderFlow],
            vec![0.6, 0.4],
            0.95
        );
        
        let mut market_data = MarketData::default();
        market_data.rsi = Some(75.0); // Overbought
        market_data.buy_volume = Some(1000.0);
        market_data.sell_volume = Some(800.0);
        
        let sentiment = observer.observe(&market_data);
        assert!(sentiment.overall_sentiment.abs() <= 1.0);
        assert!(sentiment.confidence >= 0.0 && sentiment.confidence <= 1.0);
    }
}