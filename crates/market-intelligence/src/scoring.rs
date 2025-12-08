use crate::*;
use sentiment_engine::{SentimentScore, SocialSentiment, NewsSentiment, WhaleActivity};
use trend_analyzer::TrendScore;

pub struct ScoringEngine {
    weights: ScoringWeights,
    normalization_params: NormalizationParams,
}

#[derive(Clone)]
struct ScoringWeights {
    trend_weight: f64,
    sentiment_weight: f64,
    volume_weight: f64,
    volatility_weight: f64,
    momentum_weight: f64,
    market_context_weight: f64,
    risk_adjustment: f64,
}

#[derive(Clone)]
struct NormalizationParams {
    trend_range: (f64, f64),
    sentiment_range: (f64, f64),
    volume_range: (f64, f64),
    volatility_range: (f64, f64),
}

impl ScoringEngine {
    pub fn new() -> Self {
        Self {
            weights: ScoringWeights {
                trend_weight: 0.25,
                sentiment_weight: 0.20,
                volume_weight: 0.15,
                volatility_weight: 0.10,
                momentum_weight: 0.20,
                market_context_weight: 0.10,
                risk_adjustment: 0.15,
            },
            normalization_params: NormalizationParams {
                trend_range: (-1.0, 1.0),
                sentiment_range: (0.0, 1.0),
                volume_range: (0.0, 1.0),
                volatility_range: (0.0, 1.0),
            },
        }
    }
    
    pub fn calculate_overall_score(
        &self,
        trend: &TrendScore,
        sentiment: Option<&SentimentScore>,
        market_context: &MarketContext,
        ml_score: f64,
        risk_score: f64,
    ) -> f64 {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        
        // Trend component
        let trend_component = self.calculate_trend_component(trend);
        total_score += trend_component * self.weights.trend_weight;
        total_weight += self.weights.trend_weight;
        
        // Sentiment component
        if let Some(sentiment) = sentiment {
            let sentiment_component = self.calculate_sentiment_component(sentiment);
            total_score += sentiment_component * self.weights.sentiment_weight;
            total_weight += self.weights.sentiment_weight;
        }
        
        // Volume component
        let volume_component = self.calculate_volume_component(trend, market_context);
        total_score += volume_component * self.weights.volume_weight;
        total_weight += self.weights.volume_weight;
        
        // Volatility component
        let volatility_component = self.calculate_volatility_component(trend);
        total_score += volatility_component * self.weights.volatility_weight;
        total_weight += self.weights.volatility_weight;
        
        // Momentum component
        let momentum_component = self.calculate_momentum_component(trend);
        total_score += momentum_component * self.weights.momentum_weight;
        total_weight += self.weights.momentum_weight;
        
        // Market context component
        let context_component = self.calculate_market_context_component(market_context);
        total_score += context_component * self.weights.market_context_weight;
        total_weight += self.weights.market_context_weight;
        
        // ML enhancement
        if ml_score > 0.0 {
            total_score = (total_score + ml_score) / 2.0; // Blend with ML score
        }
        
        // Risk adjustment
        let risk_adjusted_score = total_score * (1.0 - risk_score * self.weights.risk_adjustment);
        
        // Normalize to [0, 1]
        risk_adjusted_score.max(0.0).min(1.0)
    }
    
    fn calculate_trend_component(&self, trend: &TrendScore) -> f64 {
        let mut component = 0.0;
        
        // Trend strength (normalized)
        let trend_strength = (trend.trend_strength + 1.0) / 2.0; // Convert from [-1,1] to [0,1]
        component += trend_strength * 0.4;
        
        // Trend confidence
        component += trend.confidence * 0.3;
        
        // Support/resistance quality
        let sr_quality = self.calculate_support_resistance_quality(trend);
        component += sr_quality * 0.3;
        
        component.min(1.0)
    }
    
    fn calculate_sentiment_component(&self, sentiment: &SentimentScore) -> f64 {
        let mut component = 0.0;
        
        // Overall sentiment score
        component += sentiment.overall_score * 0.4;
        
        // Social sentiment strength
        let social_strength = self.calculate_social_sentiment_strength(&sentiment.social_sentiment);
        component += social_strength * 0.25;
        
        // News sentiment quality
        let news_quality = self.calculate_news_sentiment_quality(&sentiment.news_sentiment);
        component += news_quality * 0.20;
        
        // Whale activity confidence
        let whale_confidence = self.calculate_whale_confidence(&sentiment.whale_activity);
        component += whale_confidence * 0.15;
        
        component.min(1.0)
    }
    
    fn calculate_volume_component(&self, trend: &TrendScore, context: &MarketContext) -> f64 {
        let mut component = 0.0;
        
        // Volume confirmation of trend
        component += trend.volume_confirmation * 0.5;
        
        // Liquidity score from context
        component += context.liquidity_score * 0.3;
        
        // Volume profile quality
        let volume_quality = self.calculate_volume_profile_quality(&context.volume_profile);
        component += volume_quality * 0.2;
        
        component.min(1.0)
    }
    
    fn calculate_volatility_component(&self, trend: &TrendScore) -> f64 {
        // Optimal volatility range for trading
        let optimal_vol_min = 0.02; // 2%
        let optimal_vol_max = 0.08; // 8%
        
        if trend.volatility >= optimal_vol_min && trend.volatility <= optimal_vol_max {
            // Scale from 0.8 to 1.0 in optimal range
            0.8 + 0.2 * (1.0 - (trend.volatility - optimal_vol_min).abs() / (optimal_vol_max - optimal_vol_min))
        } else if trend.volatility < optimal_vol_min {
            // Too low volatility
            trend.volatility / optimal_vol_min * 0.6
        } else {
            // Too high volatility
            0.6 * (1.0 - (trend.volatility - optimal_vol_max) / 0.1).max(0.1)
        }
    }
    
    fn calculate_momentum_component(&self, trend: &TrendScore) -> f64 {
        let mut component = 0.0;
        
        // Momentum strength
        component += trend.momentum_score.abs() * 0.6;
        
        // Momentum persistence (simplified)
        let momentum_consistency = if trend.momentum_score.signum() == trend.trend_strength.signum() {
            1.0 // Momentum aligns with trend
        } else {
            0.3 // Momentum conflicts with trend
        };
        component += momentum_consistency * 0.4;
        
        component.min(1.0)
    }
    
    fn calculate_market_context_component(&self, context: &MarketContext) -> f64 {
        let mut component = 0.0;
        
        // Market regime favorability
        let regime_score = match context.market_regime.as_str() {
            "trending" => 0.9,
            "volatile" => 0.6,
            "ranging" => 0.4,
            "breakout" => 1.0,
            _ => 0.5,
        };
        component += regime_score * 0.3;
        
        // Sector performance
        let sector_normalized = (context.sector_performance + 1.0) / 2.0; // Normalize to [0,1]
        component += sector_normalized * 0.25;
        
        // Market cap tier preference
        let cap_tier_score = match context.market_cap_tier.as_str() {
            "large-cap" => 0.8,
            "mid-cap" => 0.9,
            "small-cap" => 0.7,
            "micro-cap" => 0.5,
            _ => 0.6,
        };
        component += cap_tier_score * 0.2;
        
        // Volatility regime
        let vol_regime_score = match context.volatility_regime.as_str() {
            "low" => 0.6,
            "normal" => 0.9,
            "high" => 0.7,
            "extreme" => 0.3,
            _ => 0.5,
        };
        component += vol_regime_score * 0.25;
        
        component.min(1.0)
    }
    
    // Helper methods for detailed component calculations
    fn calculate_support_resistance_quality(&self, trend: &TrendScore) -> f64 {
        let support_count = trend.support_levels.len();
        let resistance_count = trend.resistance_levels.len();
        
        // Quality based on number and strength of levels
        let level_quality = ((support_count + resistance_count) as f64 / 6.0).min(1.0);
        
        level_quality
    }
    
    fn calculate_social_sentiment_strength(&self, social: &SocialSentiment) -> f64 {
        let mut strength = 0.0;
        
        // Twitter sentiment weight
        if social.twitter_bullish_ratio > 0.6 || social.twitter_bullish_ratio < 0.4 {
            strength += (social.twitter_bullish_ratio - 0.5).abs() * 2.0 * 0.4;
        }
        
        // Reddit sentiment
        strength += social.reddit_sentiment.abs() * 0.3;
        
        // Viral score contribution
        strength += social.viral_score * 0.2;
        
        // Influencer mentions boost
        let influencer_boost = (social.influencer_mentions as f64 / 50.0).min(1.0) * 0.1;
        strength += influencer_boost;
        
        strength.min(1.0)
    }
    
    fn calculate_news_sentiment_quality(&self, news: &NewsSentiment) -> f64 {
        let mut quality = 0.0;
        
        // Headline sentiment strength
        quality += (news.headline_sentiment - 0.5).abs() * 2.0 * 0.4;
        
        // Article sentiment consistency
        let headline_article_consistency = 1.0 - (news.headline_sentiment - news.article_sentiment).abs();
        quality += headline_article_consistency * 0.3;
        
        // News volume factor
        let volume_factor = (news.news_volume as f64 / 50.0).min(1.0);
        quality += volume_factor * 0.2;
        
        // Major outlet coverage bonus
        if news.major_outlet_coverage {
            quality += 0.1;
        }
        
        quality.min(1.0)
    }
    
    fn calculate_whale_confidence(&self, whale: &WhaleActivity) -> f64 {
        let mut confidence = 0.0;
        
        // Accumulation vs distribution
        if whale.accumulation_score > 0.6 {
            confidence += (whale.accumulation_score - 0.5) * 2.0 * 0.4;
        }
        
        // Smart money confidence
        confidence += whale.smart_money_confidence * 0.4;
        
        // Large transaction activity
        let tx_factor = (whale.large_transactions as f64 / 100.0).min(1.0);
        confidence += tx_factor * 0.2;
        
        confidence.min(1.0)
    }
    
    fn calculate_volume_profile_quality(&self, volume: &VolumeProfile) -> f64 {
        let mut quality = 0.0;
        
        // Volume trend (growing volume is positive)
        if volume.volume_trend > 1.0 {
            quality += ((volume.volume_trend - 1.0) / 1.0).min(1.0) * 0.4;
        }
        
        // Institutional vs retail balance
        let institutional_factor = volume.institutional_flow * 0.3;
        let retail_factor = volume.retail_interest * 0.2;
        quality += institutional_factor + retail_factor;
        
        // Volume consistency (lower volatility is better)
        let consistency = 1.0 - volume.volume_volatility.min(1.0);
        quality += consistency * 0.1;
        
        quality.min(1.0)
    }
    
    pub fn calculate_signal_strength(&self, score: f64) -> SignalStrength {
        match score {
            s if s >= 0.8 => SignalStrength::VeryStrong,
            s if s >= 0.65 => SignalStrength::Strong,
            s if s >= 0.55 => SignalStrength::Moderate,
            s if s >= 0.45 => SignalStrength::Weak,
            s if s >= 0.3 => SignalStrength::VeryWeak,
            _ => SignalStrength::NoSignal,
        }
    }
    
    pub fn adjust_weights(&mut self, adjustment: WeightAdjustment) {
        match adjustment {
            WeightAdjustment::IncreaseTrendFocus => {
                self.weights.trend_weight += 0.1;
                self.weights.sentiment_weight -= 0.05;
                self.weights.volume_weight -= 0.05;
            }
            WeightAdjustment::IncreaseSentimentFocus => {
                self.weights.sentiment_weight += 0.1;
                self.weights.trend_weight -= 0.05;
                self.weights.market_context_weight -= 0.05;
            }
            WeightAdjustment::IncreaseRiskAversion => {
                self.weights.risk_adjustment += 0.05;
            }
            WeightAdjustment::DecreaseRiskAversion => {
                self.weights.risk_adjustment -= 0.05;
            }
        }
        
        // Normalize weights to ensure they sum to reasonable values
        self.normalize_weights();
    }
    
    fn normalize_weights(&mut self) {
        let total = self.weights.trend_weight + 
                   self.weights.sentiment_weight +
                   self.weights.volume_weight +
                   self.weights.volatility_weight +
                   self.weights.momentum_weight +
                   self.weights.market_context_weight;
        
        if total > 0.0 {
            self.weights.trend_weight /= total;
            self.weights.sentiment_weight /= total;
            self.weights.volume_weight /= total;
            self.weights.volatility_weight /= total;
            self.weights.momentum_weight /= total;
            self.weights.market_context_weight /= total;
        }
        
        // Ensure risk adjustment stays within bounds
        self.weights.risk_adjustment = self.weights.risk_adjustment.max(0.0).min(0.5);
    }
}

#[derive(Debug, Clone)]
pub enum SignalStrength {
    VeryStrong,
    Strong,
    Moderate,
    Weak,
    VeryWeak,
    NoSignal,
}

#[derive(Debug, Clone)]
pub enum WeightAdjustment {
    IncreaseTrendFocus,
    IncreaseSentimentFocus,
    IncreaseRiskAversion,
    DecreaseRiskAversion,
}

// Advanced scoring strategies
pub struct AdaptiveScoring {
    base_engine: ScoringEngine,
    market_regime_adjustments: HashMap<String, ScoringWeights>,
    performance_history: Vec<ScoringPerformance>,
}

impl AdaptiveScoring {
    pub fn new() -> Self {
        let mut regime_adjustments = HashMap::new();
        
        // Trending market adjustments
        regime_adjustments.insert("trending".to_string(), ScoringWeights {
            trend_weight: 0.35,      // Increase trend importance
            sentiment_weight: 0.15,   // Decrease sentiment
            volume_weight: 0.20,
            volatility_weight: 0.05,  // Decrease volatility importance
            momentum_weight: 0.20,
            market_context_weight: 0.05,
            risk_adjustment: 0.10,    // Lower risk adjustment in trends
        });
        
        // Volatile market adjustments
        regime_adjustments.insert("volatile".to_string(), ScoringWeights {
            trend_weight: 0.15,       // Decrease trend importance
            sentiment_weight: 0.25,   // Increase sentiment importance
            volume_weight: 0.15,
            volatility_weight: 0.20,  // Increase volatility importance
            momentum_weight: 0.15,
            market_context_weight: 0.10,
            risk_adjustment: 0.25,    // Higher risk adjustment
        });
        
        Self {
            base_engine: ScoringEngine::new(),
            market_regime_adjustments: regime_adjustments,
            performance_history: vec![],
        }
    }
    
    pub fn score_with_adaptation(
        &self,
        trend: &TrendScore,
        sentiment: Option<&SentimentScore>,
        market_context: &MarketContext,
        ml_score: f64,
        risk_score: f64,
    ) -> f64 {
        // Adjust scoring based on market regime
        let mut engine = self.base_engine.clone();
        
        if let Some(regime_weights) = self.market_regime_adjustments.get(&market_context.market_regime) {
            engine.weights = regime_weights.clone();
        }
        
        engine.calculate_overall_score(trend, sentiment, market_context, ml_score, risk_score)
    }
}

#[derive(Debug, Clone)]
struct ScoringPerformance {
    timestamp: DateTime<Utc>,
    predicted_score: f64,
    actual_performance: f64,
    market_regime: String,
}