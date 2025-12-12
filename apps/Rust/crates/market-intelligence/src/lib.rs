use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use dashmap::DashMap;
use futures::stream::{self, StreamExt};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

// Import our other crates
use trend_analyzer::{TrendAnalyzer, TrendScore, TrendError};
use sentiment_engine::{SentimentEngine, SentimentScore, SentimentError};

pub mod pair_selector;
pub mod scoring;
pub mod ml_models;
pub mod risk_assessment;
pub mod market_scanner;
pub mod correlation_analysis;

#[derive(Error, Debug)]
pub enum IntelligenceError {
    #[error("Trend analysis error: {0}")]
    TrendError(#[from] TrendError),
    
    #[error("Sentiment analysis error: {0}")]
    SentimentError(#[from] SentimentError),
    
    #[error("Data processing error: {0}")]
    DataError(String),
    
    #[error("ML model error: {0}")]
    ModelError(String),
    
    #[error("API error: {0}")]
    ApiError(#[from] reqwest::Error),
    
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairIntelligence {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub overall_score: f64,
    pub profitability_score: f64,
    pub risk_score: f64,
    pub trend_analysis: TrendScore,
    pub sentiment_analysis: SentimentScore,
    pub market_context: MarketContext,
    pub recommendation: TradingRecommendation,
    pub confidence: f64,
    pub expected_return: f64,
    pub max_drawdown_risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    pub market_regime: String,
    pub volatility_regime: String,
    pub liquidity_score: f64,
    pub correlation_cluster: String,
    pub sector_performance: f64,
    pub market_cap_tier: String,
    pub volume_profile: VolumeProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeProfile {
    pub avg_volume_24h: f64,
    pub volume_trend: f64,
    pub volume_volatility: f64,
    pub institutional_flow: f64,
    pub retail_interest: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingRecommendation {
    pub action: TradingAction,
    pub entry_price_range: (f64, f64),
    pub stop_loss: f64,
    pub take_profit: Vec<f64>,
    pub position_size: f64,
    pub holding_period: Duration,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradingAction {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
    Avoid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

#[derive(Clone)]
pub struct MarketIntelligence {
    trend_analyzer: Arc<TrendAnalyzer>,
    sentiment_engine: Arc<SentimentEngine>,
    pair_selector: Arc<pair_selector::PairSelector>,
    scoring_engine: Arc<scoring::ScoringEngine>,
    ml_models: Arc<ml_models::MLPipeline>,
    risk_assessor: Arc<risk_assessment::RiskAssessor>,
    market_scanner: Arc<market_scanner::MarketScanner>,
    correlation_analyzer: Arc<correlation_analysis::CorrelationAnalyzer>,
    cache: Arc<DashMap<String, CachedIntelligence>>,
    config: IntelligenceConfig,
}

#[derive(Clone)]
pub struct IntelligenceConfig {
    pub max_pairs_to_analyze: usize,
    pub min_volume_threshold: f64,
    pub min_market_cap: f64,
    pub risk_tolerance: f64,
    pub cache_duration_minutes: u64,
    pub enable_ml_scoring: bool,
    pub enable_sentiment_analysis: bool,
    pub enable_correlation_analysis: bool,
}

#[derive(Clone)]
struct CachedIntelligence {
    intelligence: PairIntelligence,
    timestamp: DateTime<Utc>,
}

impl MarketIntelligence {
    pub fn new(
        trend_analyzer: TrendAnalyzer,
        sentiment_engine: SentimentEngine,
        config: IntelligenceConfig,
    ) -> Self {
        let trend_analyzer = Arc::new(trend_analyzer);
        let sentiment_engine = Arc::new(sentiment_engine);
        
        Self {
            pair_selector: Arc::new(pair_selector::PairSelector::new()),
            scoring_engine: Arc::new(scoring::ScoringEngine::new()),
            ml_models: Arc::new(ml_models::MLPipeline::new()),
            risk_assessor: Arc::new(risk_assessment::RiskAssessor::new()),
            market_scanner: Arc::new(market_scanner::MarketScanner::new()),
            correlation_analyzer: Arc::new(correlation_analysis::CorrelationAnalyzer::new()),
            trend_analyzer,
            sentiment_engine,
            cache: Arc::new(DashMap::new()),
            config,
        }
    }
    
    pub async fn analyze_pair(&self, symbol: &str) -> Result<PairIntelligence, IntelligenceError> {
        // Check cache first
        if let Some(cached) = self.get_cached_intelligence(symbol) {
            return Ok(cached);
        }
        
        // Parallel analysis
        let trend_future = self.trend_analyzer.analyze_pair_trends(symbol);
        let sentiment_future = if self.config.enable_sentiment_analysis {
            Some(self.sentiment_engine.analyze_pair_sentiment(symbol))
        } else {
            None
        };
        let market_context_future = self.analyze_market_context(symbol);
        
        // Execute analyses
        let trend_result = trend_future.await?;
        let sentiment_result = if let Some(sentiment_future) = sentiment_future {
            Some(sentiment_future.await?)
        } else {
            None
        };
        let market_context = market_context_future.await?;
        
        // ML-based scoring
        let ml_score = if self.config.enable_ml_scoring {
            self.ml_models.predict_profitability(symbol, &trend_result, sentiment_result.as_ref()).await?
        } else {
            0.5 // Neutral score if ML disabled
        };
        
        // Risk assessment
        let risk_score = self.risk_assessor.calculate_risk_score(
            symbol,
            &trend_result,
            &market_context,
        ).await?;
        
        // Generate comprehensive score
        let overall_score = self.scoring_engine.calculate_overall_score(
            &trend_result,
            sentiment_result.as_ref(),
            &market_context,
            ml_score,
            risk_score,
        );
        
        // Generate trading recommendation
        let recommendation = self.generate_recommendation(
            &trend_result,
            sentiment_result.as_ref(),
            &market_context,
            overall_score,
            risk_score,
        );
        
        let intelligence = PairIntelligence {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            overall_score,
            profitability_score: ml_score,
            risk_score,
            trend_analysis: trend_result,
            sentiment_analysis: sentiment_result.unwrap_or_else(|| {
                // Create default sentiment if not analyzed
                SentimentScore {
                    symbol: symbol.to_string(),
                    timestamp: Utc::now(),
                    overall_score: 0.5,
                    social_sentiment: Default::default(),
                    news_sentiment: Default::default(),
                    whale_activity: Default::default(),
                    defi_metrics: None,
                    confidence: 0.0,
                    sources_count: 0,
                }
            }),
            market_context,
            recommendation,
            confidence: self.calculate_confidence(&trend_result, risk_score),
            expected_return: self.estimate_expected_return(overall_score, risk_score),
            max_drawdown_risk: self.estimate_max_drawdown(risk_score),
        };
        
        // Cache the result
        self.cache_intelligence(symbol, &intelligence);
        
        Ok(intelligence)
    }
    
    pub async fn scan_top_opportunities(&self, limit: usize) -> Result<Vec<PairIntelligence>, IntelligenceError> {
        // Get candidate pairs
        let candidates = self.market_scanner.scan_market_for_opportunities().await?;
        
        // Analyze each pair
        let mut analyses = vec![];
        let semaphore = Arc::new(tokio::sync::Semaphore::new(10)); // Limit concurrent requests
        
        let futures: Vec<_> = candidates.into_iter()
            .take(self.config.max_pairs_to_analyze)
            .map(|symbol| {
                let intelligence = Arc::clone(&self as &Arc<_>);
                let sem = Arc::clone(&semaphore);
                async move {
                    let _permit = sem.acquire().await.unwrap();
                    intelligence.analyze_pair(&symbol).await
                }
            })
            .collect();
        
        // Execute analyses in parallel
        let results = stream::iter(futures)
            .buffer_unordered(10)
            .collect::<Vec<_>>()
            .await;
        
        // Collect successful analyses
        for result in results {
            if let Ok(analysis) = result {
                analyses.push(analysis);
            }
        }
        
        // Sort by overall score
        analyses.sort_by(|a, b| b.overall_score.partial_cmp(&a.overall_score).unwrap());
        
        // Apply additional filtering
        analyses.retain(|analysis| {
            analysis.overall_score > 0.6 && // Minimum score threshold
            analysis.risk_score < self.config.risk_tolerance &&
            analysis.confidence > 0.5
        });
        
        Ok(analyses.into_iter().take(limit).collect())
    }
    
    async fn analyze_market_context(&self, symbol: &str) -> Result<MarketContext, IntelligenceError> {
        // Market regime detection
        let market_regime = self.determine_market_regime(symbol).await?;
        
        // Volatility analysis
        let volatility_regime = self.analyze_volatility_regime(symbol).await?;
        
        // Liquidity scoring
        let liquidity_score = self.calculate_liquidity_score(symbol).await?;
        
        // Correlation analysis
        let correlation_cluster = if self.config.enable_correlation_analysis {
            self.correlation_analyzer.get_correlation_cluster(symbol).await?
        } else {
            "unknown".to_string()
        };
        
        // Sector performance
        let sector_performance = self.analyze_sector_performance(symbol).await?;
        
        // Market cap classification
        let market_cap_tier = self.classify_market_cap(symbol).await?;
        
        // Volume profile
        let volume_profile = self.analyze_volume_profile(symbol).await?;
        
        Ok(MarketContext {
            market_regime,
            volatility_regime,
            liquidity_score,
            correlation_cluster,
            sector_performance,
            market_cap_tier,
            volume_profile,
        })
    }
    
    fn generate_recommendation(
        &self,
        trend: &TrendScore,
        sentiment: Option<&SentimentScore>,
        context: &MarketContext,
        overall_score: f64,
        risk_score: f64,
    ) -> TradingRecommendation {
        // Determine action based on scores
        let action = match overall_score {
            s if s > 0.8 => TradingAction::StrongBuy,
            s if s > 0.65 => TradingAction::Buy,
            s if s > 0.45 => TradingAction::Hold,
            s if s > 0.3 => TradingAction::Sell,
            s if s > 0.15 => TradingAction::StrongSell,
            _ => TradingAction::Avoid,
        };
        
        // Risk level assessment
        let risk_level = match risk_score {
            r if r < 0.2 => RiskLevel::VeryLow,
            r if r < 0.4 => RiskLevel::Low,
            r if r < 0.6 => RiskLevel::Medium,
            r if r < 0.8 => RiskLevel::High,
            _ => RiskLevel::VeryHigh,
        };
        
        // Calculate entry range based on current price and volatility
        let current_price = trend.current_price;
        let volatility = trend.volatility;
        let entry_range = (
            current_price * (1.0 - volatility * 0.02),
            current_price * (1.0 + volatility * 0.02),
        );
        
        // Stop loss based on risk tolerance
        let stop_loss = current_price * (1.0 - risk_score * 0.1);
        
        // Take profit levels
        let take_profit = vec![
            current_price * (1.0 + overall_score * 0.05),
            current_price * (1.0 + overall_score * 0.10),
            current_price * (1.0 + overall_score * 0.15),
        ];
        
        // Position size based on risk
        let position_size = match risk_level {
            RiskLevel::VeryLow => 0.10,
            RiskLevel::Low => 0.08,
            RiskLevel::Medium => 0.05,
            RiskLevel::High => 0.03,
            RiskLevel::VeryHigh => 0.01,
        };
        
        // Holding period based on trend strength
        let holding_period = if trend.trend_strength.abs() > 0.7 {
            Duration::days(7) // Strong trend = longer hold
        } else {
            Duration::days(3) // Weak trend = shorter hold
        };
        
        TradingRecommendation {
            action,
            entry_price_range: entry_range,
            stop_loss,
            take_profit,
            position_size,
            holding_period,
            risk_level,
        }
    }
    
    // Helper methods for market context analysis
    async fn determine_market_regime(&self, _symbol: &str) -> Result<String, IntelligenceError> {
        // Simplified market regime detection
        Ok("trending".to_string())
    }
    
    async fn analyze_volatility_regime(&self, _symbol: &str) -> Result<String, IntelligenceError> {
        Ok("normal".to_string())
    }
    
    async fn calculate_liquidity_score(&self, _symbol: &str) -> Result<f64, IntelligenceError> {
        Ok(0.7) // Mock liquidity score
    }
    
    async fn analyze_sector_performance(&self, _symbol: &str) -> Result<f64, IntelligenceError> {
        Ok(0.6) // Mock sector performance
    }
    
    async fn classify_market_cap(&self, _symbol: &str) -> Result<String, IntelligenceError> {
        Ok("large-cap".to_string())
    }
    
    async fn analyze_volume_profile(&self, _symbol: &str) -> Result<VolumeProfile, IntelligenceError> {
        Ok(VolumeProfile {
            avg_volume_24h: 1000000.0,
            volume_trend: 1.2,
            volume_volatility: 0.3,
            institutional_flow: 0.6,
            retail_interest: 0.7,
        })
    }
    
    fn calculate_confidence(&self, trend: &TrendScore, risk_score: f64) -> f64 {
        let trend_confidence = trend.confidence;
        let risk_confidence = 1.0 - risk_score;
        
        (trend_confidence + risk_confidence) / 2.0
    }
    
    fn estimate_expected_return(&self, overall_score: f64, risk_score: f64) -> f64 {
        // Simple model: higher score = higher expected return, adjusted for risk
        let base_return = overall_score * 0.2; // Max 20% expected return
        let risk_adjustment = 1.0 - (risk_score * 0.5);
        
        base_return * risk_adjustment
    }
    
    fn estimate_max_drawdown(&self, risk_score: f64) -> f64 {
        // Higher risk = higher potential drawdown
        risk_score * 0.3 // Max 30% drawdown for highest risk
    }
    
    fn get_cached_intelligence(&self, symbol: &str) -> Option<PairIntelligence> {
        self.cache.get(symbol).and_then(|entry| {
            let cached = entry.value();
            if (Utc::now() - cached.timestamp).num_minutes() < self.config.cache_duration_minutes as i64 {
                Some(cached.intelligence.clone())
            } else {
                None
            }
        })
    }
    
    fn cache_intelligence(&self, symbol: &str, intelligence: &PairIntelligence) {
        self.cache.insert(symbol.to_string(), CachedIntelligence {
            intelligence: intelligence.clone(),
            timestamp: Utc::now(),
        });
    }
}

// Default implementations for structs that need them
impl Default for sentiment_engine::SocialSentiment {
    fn default() -> Self {
        Self {
            twitter_bullish_ratio: 0.5,
            reddit_sentiment: 0.5,
            engagement_velocity: 1.0,
            influencer_mentions: 0,
            viral_score: 0.0,
            community_size_growth: 1.0,
        }
    }
}

impl Default for sentiment_engine::NewsSentiment {
    fn default() -> Self {
        Self {
            headline_sentiment: 0.5,
            article_sentiment: 0.5,
            news_volume: 0,
            news_velocity: 0.0,
            major_outlet_coverage: false,
            fud_score: 0.0,
            hype_score: 0.0,
        }
    }
}

impl Default for sentiment_engine::WhaleActivity {
    fn default() -> Self {
        Self {
            accumulation_score: 0.5,
            distribution_score: 0.5,
            exchange_netflow: 0.0,
            large_transactions: 0,
            whale_count_change: 0.0,
            smart_money_confidence: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trend_analyzer::TrendConfig;
    use sentiment_engine::SentimentConfig;

    #[tokio::test]
    async fn test_pair_analysis() {
        let trend_analyzer = TrendAnalyzer::new(TrendConfig::default());
        let sentiment_engine = SentimentEngine::new(SentimentConfig {
            twitter_api_key: None,
            reddit_client_id: None,
            reddit_client_secret: None,
            news_api_keys: vec![],
            web3_endpoints: vec![],
            cache_duration_minutes: 5,
            max_concurrent_requests: 10,
        });
        
        let config = IntelligenceConfig {
            max_pairs_to_analyze: 10,
            min_volume_threshold: 100000.0,
            min_market_cap: 1000000.0,
            risk_tolerance: 0.7,
            cache_duration_minutes: 5,
            enable_ml_scoring: false,
            enable_sentiment_analysis: false,
            enable_correlation_analysis: false,
        };
        
        let intelligence = MarketIntelligence::new(trend_analyzer, sentiment_engine, config);
        let result = intelligence.analyze_pair("BTC/USDT").await;
        
        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert_eq!(analysis.symbol, "BTC/USDT");
        assert!(analysis.overall_score >= 0.0 && analysis.overall_score <= 1.0);
    }
    
    #[tokio::test]
    async fn test_top_opportunities_scan() {
        let trend_analyzer = TrendAnalyzer::new(TrendConfig::default());
        let sentiment_engine = SentimentEngine::new(SentimentConfig {
            twitter_api_key: None,
            reddit_client_id: None,
            reddit_client_secret: None,
            news_api_keys: vec![],
            web3_endpoints: vec![],
            cache_duration_minutes: 5,
            max_concurrent_requests: 10,
        });
        
        let config = IntelligenceConfig {
            max_pairs_to_analyze: 5,
            min_volume_threshold: 100000.0,
            min_market_cap: 1000000.0,
            risk_tolerance: 0.8,
            cache_duration_minutes: 5,
            enable_ml_scoring: false,
            enable_sentiment_analysis: false,
            enable_correlation_analysis: false,
        };
        
        let intelligence = MarketIntelligence::new(trend_analyzer, sentiment_engine, config);
        let opportunities = intelligence.scan_top_opportunities(3).await;
        
        assert!(opportunities.is_ok());
        let results = opportunities.unwrap();
        assert!(results.len() <= 3);
    }
}