//! Sentiment Analysis Module
//!
//! Advanced sentiment analysis for market factors and quantum-enhanced sentiment detection.

use crate::core::{QarResult, FactorMap, StandardFactors};
use crate::error::QarError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sentiment analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    /// Overall sentiment score (-1.0 to 1.0)
    pub score: f64,
    /// Confidence in sentiment assessment
    pub confidence: f64,
    /// Sentiment components
    pub sentiment_components: SentimentComponents,
    /// Sentiment indicators
    pub sentiment_indicators: SentimentIndicators,
    /// Sentiment momentum
    pub sentiment_momentum: SentimentMomentum,
}

/// Sentiment components breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentComponents {
    /// Market sentiment
    pub market_sentiment: f64,
    /// Fear and greed index
    pub fear_greed_index: f64,
    /// Volatility sentiment
    pub volatility_sentiment: f64,
    /// Volume sentiment
    pub volume_sentiment: f64,
    /// Momentum sentiment
    pub momentum_sentiment: f64,
    /// Risk appetite
    pub risk_appetite: f64,
    /// Liquidity sentiment
    pub liquidity_sentiment: f64,
}

/// Sentiment indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentIndicators {
    /// VIX-like fear index
    pub fear_index: f64,
    /// Put-call ratio equivalent
    pub put_call_ratio: f64,
    /// Sentiment oscillator
    pub sentiment_oscillator: f64,
    /// Bullish percentage
    pub bullish_percentage: f64,
    /// Sentiment extremes
    pub sentiment_extremes: SentimentExtremes,
}

/// Sentiment extremes detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentExtremes {
    /// Extreme bullishness detected
    pub extreme_bullish: bool,
    /// Extreme bearishness detected
    pub extreme_bearish: bool,
    /// Complacency level
    pub complacency_level: f64,
    /// Panic level
    pub panic_level: f64,
}

/// Sentiment momentum analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentMomentum {
    /// Short-term sentiment trend
    pub short_term_trend: SentimentTrend,
    /// Medium-term sentiment trend
    pub medium_term_trend: SentimentTrend,
    /// Long-term sentiment trend
    pub long_term_trend: SentimentTrend,
    /// Sentiment reversal signals
    pub reversal_signals: Vec<ReversalSignal>,
}

/// Sentiment trend enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SentimentTrend {
    StronglyBullish,
    Bullish,
    Neutral,
    Bearish,
    StronglyBearish,
}

/// Sentiment reversal signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReversalSignal {
    /// Signal type
    pub signal_type: ReversalType,
    /// Signal strength
    pub strength: f64,
    /// Time horizon
    pub time_horizon: TimeHorizon,
    /// Description
    pub description: String,
}

/// Reversal signal types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReversalType {
    BullishReversal,
    BearishReversal,
    ContraryIndicator,
    ExtremeReading,
}

/// Time horizon for signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeHorizon {
    Immediate,  // 1-3 periods
    ShortTerm,  // 3-10 periods
    MediumTerm, // 10-30 periods
    LongTerm,   // 30+ periods
}

/// Sentiment analyzer
pub struct SentimentAnalyzer {
    config: super::AnalysisConfig,
    sentiment_params: SentimentParameters,
    sentiment_history: Vec<f64>,
    fear_greed_history: Vec<f64>,
    history: Vec<SentimentResult>,
}

/// Sentiment analysis parameters
#[derive(Debug, Clone)]
pub struct SentimentParameters {
    /// Lookback window for sentiment calculation
    pub lookback_window: usize,
    /// Extreme sentiment thresholds
    pub extreme_bullish_threshold: f64,
    pub extreme_bearish_threshold: f64,
    /// Sentiment smoothing factor
    pub smoothing_factor: f64,
    /// Contrarian signal sensitivity
    pub contrarian_sensitivity: f64,
}

impl Default for SentimentParameters {
    fn default() -> Self {
        Self {
            lookback_window: 20,
            extreme_bullish_threshold: 0.8,
            extreme_bearish_threshold: -0.8,
            smoothing_factor: 0.3,
            contrarian_sensitivity: 0.7,
        }
    }
}

impl SentimentAnalyzer {
    /// Create a new sentiment analyzer
    pub fn new(config: super::AnalysisConfig) -> QarResult<Self> {
        Ok(Self {
            config,
            sentiment_params: SentimentParameters::default(),
            sentiment_history: Vec::new(),
            fear_greed_history: Vec::new(),
            history: Vec::new(),
        })
    }

    /// Analyze sentiment from market factors
    pub async fn analyze(&mut self, factors: &FactorMap) -> QarResult<SentimentResult> {
        // Extract sentiment-relevant data
        let sentiment_data = self.extract_sentiment_data(factors)?;
        
        // Calculate sentiment components
        let sentiment_components = self.calculate_sentiment_components(&sentiment_data)?;
        
        // Calculate sentiment indicators
        let sentiment_indicators = self.calculate_sentiment_indicators(&sentiment_components)?;
        
        // Analyze sentiment momentum
        let sentiment_momentum = self.analyze_sentiment_momentum(&sentiment_components)?;
        
        // Calculate overall score and confidence
        let score = self.calculate_overall_sentiment_score(&sentiment_components);
        let confidence = self.calculate_sentiment_confidence(&sentiment_components, &sentiment_indicators);

        // Update history
        self.update_sentiment_history(score, sentiment_components.fear_greed_index);

        let result = SentimentResult {
            score,
            confidence,
            sentiment_components,
            sentiment_indicators,
            sentiment_momentum,
        };

        // Store in history
        self.add_to_history(result.clone());

        Ok(result)
    }

    /// Extract sentiment-relevant data from factors
    fn extract_sentiment_data(&self, factors: &FactorMap) -> QarResult<SentimentData> {
        Ok(SentimentData {
            sentiment_factor: factors.get_factor(&StandardFactors::Sentiment)?,
            volatility: factors.get_factor(&StandardFactors::Volatility)?,
            volume: factors.get_factor(&StandardFactors::Volume)?,
            momentum: factors.get_factor(&StandardFactors::Momentum)?,
            trend: factors.get_factor(&StandardFactors::Trend)?,
            risk: factors.get_factor(&StandardFactors::Risk)?,
            liquidity: factors.get_factor(&StandardFactors::Liquidity)?,
            efficiency: factors.get_factor(&StandardFactors::Efficiency)?,
        })
    }

    /// Calculate sentiment components
    fn calculate_sentiment_components(&self, data: &SentimentData) -> QarResult<SentimentComponents> {
        Ok(SentimentComponents {
            market_sentiment: self.calculate_market_sentiment(data),
            fear_greed_index: self.calculate_fear_greed_index(data),
            volatility_sentiment: self.calculate_volatility_sentiment(data),
            volume_sentiment: self.calculate_volume_sentiment(data),
            momentum_sentiment: self.calculate_momentum_sentiment(data),
            risk_appetite: self.calculate_risk_appetite(data),
            liquidity_sentiment: self.calculate_liquidity_sentiment(data),
        })
    }

    /// Calculate market sentiment
    fn calculate_market_sentiment(&self, data: &SentimentData) -> f64 {
        // Base market sentiment from sentiment factor
        let base_sentiment = (data.sentiment_factor - 0.5) * 2.0; // Convert to -1 to 1 range
        
        // Adjust based on trend and momentum
        let trend_adjustment = (data.trend - 0.5) * 0.3;
        let momentum_adjustment = (data.momentum - 0.5) * 0.2;
        
        // Apply efficiency factor (higher efficiency = more reliable sentiment)
        let efficiency_weight = 0.5 + data.efficiency * 0.5;
        
        let adjusted_sentiment = (base_sentiment + trend_adjustment + momentum_adjustment) * efficiency_weight;
        
        adjusted_sentiment.max(-1.0).min(1.0)
    }

    /// Calculate fear and greed index
    fn calculate_fear_greed_index(&self, data: &SentimentData) -> f64 {
        // Fear and greed based on multiple factors
        let volatility_component = 1.0 - data.volatility; // Low volatility = greed
        let volume_component = data.volume; // High volume can indicate greed or fear
        let momentum_component = data.momentum; // High momentum = greed
        let risk_component = 1.0 - data.risk; // Low risk = greed
        
        // Weight the components
        let fear_greed = volatility_component * 0.3 +
                        volume_component * 0.2 +
                        momentum_component * 0.3 +
                        risk_component * 0.2;
        
        // Convert to -1 (extreme fear) to 1 (extreme greed) scale
        (fear_greed - 0.5) * 2.0
    }

    /// Calculate volatility sentiment
    fn calculate_volatility_sentiment(&self, data: &SentimentData) -> f64 {
        // High volatility generally indicates fear/negative sentiment
        let vol_sentiment = 1.0 - data.volatility * 2.0; // Convert to -1 to 1 scale
        
        // Adjust for market conditions
        if data.trend > 0.6 && data.volatility > 0.7 {
            // High volatility in uptrend might be positive (excitement)
            vol_sentiment.max(-0.5)
        } else {
            vol_sentiment.max(-1.0).min(1.0)
        }
    }

    /// Calculate volume sentiment
    fn calculate_volume_sentiment(&self, data: &SentimentData) -> f64 {
        // Volume sentiment depends on context
        let base_volume_sentiment = (data.volume - 0.5) * 2.0;
        
        // High volume in uptrend = positive, high volume in downtrend = negative
        let trend_context = if data.trend > 0.5 {
            1.0 // Positive context
        } else {
            -1.0 // Negative context
        };
        
        // Apply trend context if volume is significantly high or low
        if data.volume > 0.7 || data.volume < 0.3 {
            base_volume_sentiment * trend_context * 0.7
        } else {
            base_volume_sentiment * 0.5 // Neutral volume
        }
    }

    /// Calculate momentum sentiment
    fn calculate_momentum_sentiment(&self, data: &SentimentData) -> f64 {
        // Strong momentum in either direction affects sentiment
        let momentum_strength = (data.momentum - 0.5).abs() * 2.0;
        let momentum_direction = if data.momentum > 0.5 { 1.0 } else { -1.0 };
        
        // Strong momentum = strong sentiment
        let sentiment_intensity = momentum_strength;
        
        momentum_direction * sentiment_intensity
    }

    /// Calculate risk appetite
    fn calculate_risk_appetite(&self, data: &SentimentData) -> f64 {
        // Risk appetite based on risk, volatility, and liquidity
        let low_risk_component = 1.0 - data.risk;
        let low_volatility_component = 1.0 - data.volatility;
        let high_liquidity_component = data.liquidity;
        
        // High risk appetite = willingness to take risks
        let risk_appetite = (low_risk_component * 0.4 +
                           low_volatility_component * 0.3 +
                           high_liquidity_component * 0.3);
        
        // Convert to -1 (risk aversion) to 1 (risk seeking) scale
        (risk_appetite - 0.5) * 2.0
    }

    /// Calculate liquidity sentiment
    fn calculate_liquidity_sentiment(&self, data: &SentimentData) -> f64 {
        // High liquidity generally positive for sentiment
        let liquidity_sentiment = (data.liquidity - 0.5) * 2.0;
        
        // Adjust for efficiency (how well liquidity is being utilized)
        let efficiency_adjustment = (data.efficiency - 0.5) * 0.3;
        
        (liquidity_sentiment + efficiency_adjustment).max(-1.0).min(1.0)
    }

    /// Calculate sentiment indicators
    fn calculate_sentiment_indicators(&self, components: &SentimentComponents) -> QarResult<SentimentIndicators> {
        let fear_index = self.calculate_fear_index(components);
        let put_call_ratio = self.calculate_put_call_ratio(components);
        let sentiment_oscillator = self.calculate_sentiment_oscillator(components);
        let bullish_percentage = self.calculate_bullish_percentage(components);
        let sentiment_extremes = self.detect_sentiment_extremes(components);

        Ok(SentimentIndicators {
            fear_index,
            put_call_ratio,
            sentiment_oscillator,
            bullish_percentage,
            sentiment_extremes,
        })
    }

    /// Calculate fear index (VIX-like indicator)
    fn calculate_fear_index(&self, components: &SentimentComponents) -> f64 {
        // Fear index based on volatility sentiment and risk appetite
        let fear_from_volatility = -components.volatility_sentiment;
        let fear_from_risk = -components.risk_appetite;
        let fear_from_fg = -components.fear_greed_index;
        
        let fear_index = (fear_from_volatility * 0.4 +
                         fear_from_risk * 0.3 +
                         fear_from_fg * 0.3);
        
        // Normalize to 0-100 scale (like VIX)
        ((fear_index + 1.0) / 2.0 * 100.0).max(0.0).min(100.0)
    }

    /// Calculate put-call ratio equivalent
    fn calculate_put_call_ratio(&self, components: &SentimentComponents) -> f64 {
        // Put-call ratio based on fear/greed and risk appetite
        let bearish_bias = -components.fear_greed_index; // Fear = more puts
        let risk_aversion = -components.risk_appetite; // Risk aversion = more puts
        
        // Base ratio around 0.7 (neutral), adjust up for bearish sentiment
        let base_ratio = 0.7;
        let adjustment = (bearish_bias + risk_aversion) * 0.3;
        
        (base_ratio + adjustment).max(0.1).min(2.0)
    }

    /// Calculate sentiment oscillator
    fn calculate_sentiment_oscillator(&self, components: &SentimentComponents) -> f64 {
        // Oscillator combining multiple sentiment measures
        let weighted_sentiment = components.market_sentiment * 0.3 +
                               components.momentum_sentiment * 0.25 +
                               components.fear_greed_index * 0.25 +
                               components.volume_sentiment * 0.2;
        
        // Apply smoothing if we have history
        if !self.sentiment_history.is_empty() {
            let prev_oscillator = self.sentiment_history.last().unwrap_or(&0.0);
            let alpha = self.sentiment_params.smoothing_factor;
            alpha * weighted_sentiment + (1.0 - alpha) * prev_oscillator
        } else {
            weighted_sentiment
        }
    }

    /// Calculate bullish percentage
    fn calculate_bullish_percentage(&self, components: &SentimentComponents) -> f64 {
        // Convert overall sentiment to percentage of bullish indicators
        let sentiment_factors = vec![
            components.market_sentiment,
            components.momentum_sentiment,
            components.volume_sentiment,
            components.risk_appetite,
            components.liquidity_sentiment,
        ];
        
        let bullish_count = sentiment_factors.iter().filter(|&&s| s > 0.0).count();
        let total_count = sentiment_factors.len();
        
        bullish_count as f64 / total_count as f64 * 100.0
    }

    /// Detect sentiment extremes
    fn detect_sentiment_extremes(&self, components: &SentimentComponents) -> SentimentExtremes {
        let overall_sentiment = components.market_sentiment;
        let fear_greed = components.fear_greed_index;
        
        let extreme_bullish = overall_sentiment > self.sentiment_params.extreme_bullish_threshold &&
                             fear_greed > self.sentiment_params.extreme_bullish_threshold;
        
        let extreme_bearish = overall_sentiment < self.sentiment_params.extreme_bearish_threshold &&
                             fear_greed < self.sentiment_params.extreme_bearish_threshold;
        
        // Complacency: high sentiment with low volatility
        let complacency_level = if components.volatility_sentiment > 0.5 && components.risk_appetite > 0.5 {
            (components.volatility_sentiment + components.risk_appetite) / 2.0
        } else {
            0.0
        };
        
        // Panic: very negative sentiment with high volatility
        let panic_level = if components.volatility_sentiment < -0.5 && components.fear_greed_index < -0.5 {
            (-components.volatility_sentiment - components.fear_greed_index) / 2.0
        } else {
            0.0
        };

        SentimentExtremes {
            extreme_bullish,
            extreme_bearish,
            complacency_level,
            panic_level,
        }
    }

    /// Analyze sentiment momentum
    fn analyze_sentiment_momentum(&self, components: &SentimentComponents) -> QarResult<SentimentMomentum> {
        let short_term_trend = self.calculate_sentiment_trend(&self.sentiment_history, 5);
        let medium_term_trend = self.calculate_sentiment_trend(&self.sentiment_history, 15);
        let long_term_trend = self.calculate_sentiment_trend(&self.sentiment_history, 30);
        
        let reversal_signals = self.detect_reversal_signals(components, &short_term_trend, &medium_term_trend)?;

        Ok(SentimentMomentum {
            short_term_trend,
            medium_term_trend,
            long_term_trend,
            reversal_signals,
        })
    }

    /// Calculate sentiment trend
    fn calculate_sentiment_trend(&self, history: &[f64], lookback: usize) -> SentimentTrend {
        if history.len() < lookback {
            return SentimentTrend::Neutral;
        }

        let recent_data = &history[history.len() - lookback..];
        let first_half = &recent_data[..recent_data.len() / 2];
        let second_half = &recent_data[recent_data.len() / 2..];
        
        let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;
        
        let change = second_avg - first_avg;
        
        match change {
            x if x > 0.3 => SentimentTrend::StronglyBullish,
            x if x > 0.1 => SentimentTrend::Bullish,
            x if x < -0.3 => SentimentTrend::StronglyBearish,
            x if x < -0.1 => SentimentTrend::Bearish,
            _ => SentimentTrend::Neutral,
        }
    }

    /// Detect reversal signals
    fn detect_reversal_signals(
        &self, 
        components: &SentimentComponents,
        short_trend: &SentimentTrend,
        medium_trend: &SentimentTrend,
    ) -> QarResult<Vec<ReversalSignal>> {
        let mut signals = Vec::new();

        // Extreme reading reversal
        if components.fear_greed_index > 0.8 {
            signals.push(ReversalSignal {
                signal_type: ReversalType::BearishReversal,
                strength: components.fear_greed_index,
                time_horizon: TimeHorizon::ShortTerm,
                description: "Extreme greed reading suggests potential bearish reversal".to_string(),
            });
        } else if components.fear_greed_index < -0.8 {
            signals.push(ReversalSignal {
                signal_type: ReversalType::BullishReversal,
                strength: -components.fear_greed_index,
                time_horizon: TimeHorizon::ShortTerm,
                description: "Extreme fear reading suggests potential bullish reversal".to_string(),
            });
        }

        // Contrarian indicator
        if matches!(short_trend, SentimentTrend::StronglyBullish) && 
           matches!(medium_trend, SentimentTrend::StronglyBullish) {
            signals.push(ReversalSignal {
                signal_type: ReversalType::ContraryIndicator,
                strength: 0.7,
                time_horizon: TimeHorizon::MediumTerm,
                description: "Persistent bullish sentiment may indicate contrarian opportunity".to_string(),
            });
        } else if matches!(short_trend, SentimentTrend::StronglyBearish) && 
                  matches!(medium_trend, SentimentTrend::StronglyBearish) {
            signals.push(ReversalSignal {
                signal_type: ReversalType::ContraryIndicator,
                strength: 0.7,
                time_horizon: TimeHorizon::MediumTerm,
                description: "Persistent bearish sentiment may indicate contrarian opportunity".to_string(),
            });
        }

        // Divergence signals
        if components.momentum_sentiment > 0.5 && components.market_sentiment < -0.2 {
            signals.push(ReversalSignal {
                signal_type: ReversalType::BullishReversal,
                strength: 0.6,
                time_horizon: TimeHorizon::Immediate,
                description: "Positive momentum with negative sentiment suggests bullish divergence".to_string(),
            });
        }

        Ok(signals)
    }

    /// Calculate overall sentiment score
    fn calculate_overall_sentiment_score(&self, components: &SentimentComponents) -> f64 {
        let sentiment_weights = [
            (components.market_sentiment, 0.25),
            (components.fear_greed_index, 0.20),
            (components.momentum_sentiment, 0.20),
            (components.volume_sentiment, 0.15),
            (components.risk_appetite, 0.10),
            (components.liquidity_sentiment, 0.05),
            (components.volatility_sentiment, 0.05),
        ];

        sentiment_weights.iter().map(|(sentiment, weight)| sentiment * weight).sum()
    }

    /// Calculate confidence in sentiment assessment
    fn calculate_sentiment_confidence(&self, components: &SentimentComponents, indicators: &SentimentIndicators) -> f64 {
        let mut confidence_factors = Vec::new();

        // Data availability confidence
        let data_confidence = if self.sentiment_history.len() >= 20 {
            0.9
        } else if self.sentiment_history.len() >= 10 {
            0.7
        } else {
            0.5
        };
        confidence_factors.push(data_confidence);

        // Sentiment consistency confidence
        let sentiment_values = vec![
            components.market_sentiment,
            components.momentum_sentiment,
            components.volume_sentiment,
            components.fear_greed_index,
        ];
        
        let mean_sentiment = sentiment_values.iter().sum::<f64>() / sentiment_values.len() as f64;
        let sentiment_variance = sentiment_values.iter()
            .map(|s| (s - mean_sentiment).powi(2))
            .sum::<f64>() / sentiment_values.len() as f64;
        
        let consistency_confidence = 1.0 - sentiment_variance.sqrt();
        confidence_factors.push(consistency_confidence.max(0.0).min(1.0));

        // Extreme readings confidence (extremes are often more reliable)
        let extreme_confidence = if indicators.sentiment_extremes.extreme_bullish || 
                                   indicators.sentiment_extremes.extreme_bearish {
            0.9
        } else {
            0.6
        };
        confidence_factors.push(extreme_confidence);

        // Calculate overall confidence
        confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64
    }

    /// Update sentiment history
    fn update_sentiment_history(&mut self, sentiment: f64, fear_greed: f64) {
        self.sentiment_history.push(sentiment);
        self.fear_greed_history.push(fear_greed);
        
        // Maintain history size
        let max_history = self.config.max_history;
        if self.sentiment_history.len() > max_history {
            self.sentiment_history.remove(0);
        }
        if self.fear_greed_history.len() > max_history {
            self.fear_greed_history.remove(0);
        }
    }

    fn add_to_history(&mut self, result: SentimentResult) {
        self.history.push(result);
        
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }

    /// Get analysis history
    pub fn get_history(&self) -> &[SentimentResult] {
        &self.history
    }

    /// Get latest analysis
    pub fn get_latest(&self) -> Option<&SentimentResult> {
        self.history.last()
    }

    /// Get sentiment parameters
    pub fn get_parameters(&self) -> &SentimentParameters {
        &self.sentiment_params
    }

    /// Update sentiment parameters
    pub fn update_parameters(&mut self, params: SentimentParameters) {
        self.sentiment_params = params;
    }

    /// Get sentiment history
    pub fn get_sentiment_history(&self) -> &[f64] {
        &self.sentiment_history
    }

    /// Get fear-greed history
    pub fn get_fear_greed_history(&self) -> &[f64] {
        &self.fear_greed_history
    }
}

/// Sentiment data structure
#[derive(Debug, Clone)]
pub struct SentimentData {
    pub sentiment_factor: f64,
    pub volatility: f64,
    pub volume: f64,
    pub momentum: f64,
    pub trend: f64,
    pub risk: f64,
    pub liquidity: f64,
    pub efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;

    #[tokio::test]
    async fn test_sentiment_analyzer() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = SentimentAnalyzer::new(config).unwrap();

        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Sentiment.to_string(), 0.8);
        factors.insert(StandardFactors::Volatility.to_string(), 0.3);
        factors.insert(StandardFactors::Volume.to_string(), 0.7);
        factors.insert(StandardFactors::Momentum.to_string(), 0.6);
        factors.insert(StandardFactors::Trend.to_string(), 0.7);
        factors.insert(StandardFactors::Risk.to_string(), 0.4);
        factors.insert(StandardFactors::Liquidity.to_string(), 0.8);
        factors.insert(StandardFactors::Efficiency.to_string(), 0.6);
        
        let factor_map = FactorMap::new(factors).unwrap();
        let result = analyzer.analyze(&factor_map).await;
        
        assert!(result.is_ok());
        let sentiment_result = result.unwrap();
        assert!(sentiment_result.score >= -1.0 && sentiment_result.score <= 1.0);
        assert!(sentiment_result.confidence >= 0.0 && sentiment_result.confidence <= 1.0);
    }

    #[test]
    fn test_fear_greed_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = SentimentAnalyzer::new(config).unwrap();
        
        // High greed scenario
        let greed_data = SentimentData {
            sentiment_factor: 0.9,
            volatility: 0.2,
            volume: 0.8,
            momentum: 0.9,
            trend: 0.8,
            risk: 0.2,
            liquidity: 0.9,
            efficiency: 0.8,
        };
        
        let fear_greed = analyzer.calculate_fear_greed_index(&greed_data);
        assert!(fear_greed > 0.0); // Should indicate greed
        
        // High fear scenario
        let fear_data = SentimentData {
            sentiment_factor: 0.1,
            volatility: 0.9,
            volume: 0.9,
            momentum: 0.1,
            trend: 0.2,
            risk: 0.9,
            liquidity: 0.2,
            efficiency: 0.3,
        };
        
        let fear_greed = analyzer.calculate_fear_greed_index(&fear_data);
        assert!(fear_greed < 0.0); // Should indicate fear
    }

    #[test]
    fn test_sentiment_trend_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = SentimentAnalyzer::new(config).unwrap();
        
        // Strong upward trend
        let upward_history = vec![-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.8];
        let trend = analyzer.calculate_sentiment_trend(&upward_history, 6);
        assert!(matches!(trend, SentimentTrend::StronglyBullish | SentimentTrend::Bullish));
        
        // Strong downward trend
        let downward_history = vec![0.8, 0.6, 0.4, 0.2, -0.1, -0.3, -0.5, -0.7];
        let trend = analyzer.calculate_sentiment_trend(&downward_history, 6);
        assert!(matches!(trend, SentimentTrend::StronglyBearish | SentimentTrend::Bearish));
        
        // Neutral trend
        let neutral_history = vec![0.1, 0.2, 0.15, 0.18, 0.12, 0.16];
        let trend = analyzer.calculate_sentiment_trend(&neutral_history, 6);
        assert!(matches!(trend, SentimentTrend::Neutral));
    }

    #[test]
    fn test_extreme_sentiment_detection() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = SentimentAnalyzer::new(config).unwrap();
        
        // Extreme bullish components
        let extreme_bullish = SentimentComponents {
            market_sentiment: 0.9,
            fear_greed_index: 0.85,
            volatility_sentiment: 0.8,
            volume_sentiment: 0.7,
            momentum_sentiment: 0.8,
            risk_appetite: 0.9,
            liquidity_sentiment: 0.8,
        };
        
        let extremes = analyzer.detect_sentiment_extremes(&extreme_bullish);
        assert!(extremes.extreme_bullish);
        assert!(!extremes.extreme_bearish);
        
        // Extreme bearish components
        let extreme_bearish = SentimentComponents {
            market_sentiment: -0.9,
            fear_greed_index: -0.85,
            volatility_sentiment: -0.8,
            volume_sentiment: -0.7,
            momentum_sentiment: -0.8,
            risk_appetite: -0.9,
            liquidity_sentiment: -0.8,
        };
        
        let extremes = analyzer.detect_sentiment_extremes(&extreme_bearish);
        assert!(!extremes.extreme_bullish);
        assert!(extremes.extreme_bearish);
    }

    #[test]
    fn test_fear_index_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = SentimentAnalyzer::new(config).unwrap();
        
        // High fear components
        let high_fear = SentimentComponents {
            market_sentiment: -0.7,
            fear_greed_index: -0.8,
            volatility_sentiment: -0.9,
            volume_sentiment: -0.5,
            momentum_sentiment: -0.6,
            risk_appetite: -0.8,
            liquidity_sentiment: -0.4,
        };
        
        let fear_index = analyzer.calculate_fear_index(&high_fear);
        assert!(fear_index > 50.0); // High fear should give high fear index
        assert!(fear_index <= 100.0);
        
        // Low fear components
        let low_fear = SentimentComponents {
            market_sentiment: 0.7,
            fear_greed_index: 0.6,
            volatility_sentiment: 0.8,
            volume_sentiment: 0.5,
            momentum_sentiment: 0.6,
            risk_appetite: 0.8,
            liquidity_sentiment: 0.7,
        };
        
        let fear_index = analyzer.calculate_fear_index(&low_fear);
        assert!(fear_index < 50.0); // Low fear should give low fear index
        assert!(fear_index >= 0.0);
    }

    #[test]
    fn test_bullish_percentage_calculation() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = SentimentAnalyzer::new(config).unwrap();
        
        // All bullish components
        let all_bullish = SentimentComponents {
            market_sentiment: 0.5,
            fear_greed_index: 0.3,
            volatility_sentiment: 0.2,
            volume_sentiment: 0.4,
            momentum_sentiment: 0.6,
            risk_appetite: 0.7,
            liquidity_sentiment: 0.3,
        };
        
        let bullish_pct = analyzer.calculate_bullish_percentage(&all_bullish);
        assert_eq!(bullish_pct, 100.0); // All sentiment factors are positive
        
        // Mixed components
        let mixed = SentimentComponents {
            market_sentiment: 0.5,
            fear_greed_index: -0.3,
            volatility_sentiment: 0.2,
            volume_sentiment: -0.4,
            momentum_sentiment: 0.6,
            risk_appetite: -0.7,
            liquidity_sentiment: 0.3,
        };
        
        let bullish_pct = analyzer.calculate_bullish_percentage(&mixed);
        assert!(bullish_pct > 0.0 && bullish_pct < 100.0); // Mixed should be between 0 and 100
    }

    #[test]
    fn test_put_call_ratio() {
        let config = super::super::AnalysisConfig::default();
        let analyzer = SentimentAnalyzer::new(config).unwrap();
        
        // Bearish sentiment should increase put-call ratio
        let bearish_components = SentimentComponents {
            market_sentiment: -0.6,
            fear_greed_index: -0.7,
            volatility_sentiment: -0.5,
            volume_sentiment: -0.3,
            momentum_sentiment: -0.4,
            risk_appetite: -0.8,
            liquidity_sentiment: -0.2,
        };
        
        let put_call_ratio = analyzer.calculate_put_call_ratio(&bearish_components);
        assert!(put_call_ratio > 0.7); // Should be above neutral
        
        // Bullish sentiment should decrease put-call ratio
        let bullish_components = SentimentComponents {
            market_sentiment: 0.6,
            fear_greed_index: 0.7,
            volatility_sentiment: 0.5,
            volume_sentiment: 0.3,
            momentum_sentiment: 0.4,
            risk_appetite: 0.8,
            liquidity_sentiment: 0.2,
        };
        
        let put_call_ratio = analyzer.calculate_put_call_ratio(&bullish_components);
        assert!(put_call_ratio < 0.7); // Should be below neutral
    }

    #[tokio::test]
    async fn test_sentiment_history_tracking() {
        let config = super::super::AnalysisConfig::default();
        let mut analyzer = SentimentAnalyzer::new(config).unwrap();

        // Add multiple sentiment readings
        for i in 0..10 {
            let sentiment_value = (i as f64 / 10.0) * 2.0 - 1.0; // -1.0 to 1.0
            
            let mut factors = std::collections::HashMap::new();
            factors.insert(StandardFactors::Sentiment.to_string(), (sentiment_value + 1.0) / 2.0);
            factors.insert(StandardFactors::Volatility.to_string(), 0.4);
            factors.insert(StandardFactors::Volume.to_string(), 0.6);
            factors.insert(StandardFactors::Momentum.to_string(), 0.5);
            factors.insert(StandardFactors::Trend.to_string(), 0.6);
            factors.insert(StandardFactors::Risk.to_string(), 0.4);
            factors.insert(StandardFactors::Liquidity.to_string(), 0.7);
            factors.insert(StandardFactors::Efficiency.to_string(), 0.6);
            
            let factor_map = FactorMap::new(factors).unwrap();
            let _ = analyzer.analyze(&factor_map).await.unwrap();
        }

        assert_eq!(analyzer.get_sentiment_history().len(), 10);
        assert_eq!(analyzer.get_fear_greed_history().len(), 10);
        
        // Sentiment should show upward trend
        let first_sentiment = analyzer.get_sentiment_history()[0];
        let last_sentiment = analyzer.get_sentiment_history()[9];
        assert!(last_sentiment > first_sentiment);
    }
}