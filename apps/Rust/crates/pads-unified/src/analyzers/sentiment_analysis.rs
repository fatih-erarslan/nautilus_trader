//! # Advanced Multi-Dimensional Sentiment Analysis 📊
//!
//! Comprehensive sentiment analysis system with 6 dimensions and real-time
//! market sentiment correlation. Enhanced from Python PADS with advanced
//! lexicon-based and transformer-based analysis.
//!
//! ## Sentiment Dimensions
//! - **Polarity**: Positive vs Negative sentiment
//! - **Fear**: Fear vs Greed indicators
//! - **Confidence**: Certainty vs Uncertainty
//! - **Volatility**: Stability vs Turbulence expectations
//! - **Momentum**: Trend continuation vs Reversal
//! - **Conviction**: Strength of belief in position

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

use crate::narrative_forecasting::MarketContext;

#[derive(Error, Debug)]
pub enum SentimentAnalysisError {
    #[error("Lexicon analysis error: {0}")]
    LexiconError(String),
    
    #[error("Transformer model error: {0}")]
    TransformerError(String),
    
    #[error("Data processing error: {0}")]
    ProcessingError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
}

/// Six-dimensional sentiment analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    pub overall_sentiment: f64,
    pub confidence: f64,
    pub dimensions: SentimentDimensions,
    pub key_phrases: Vec<String>,
    pub sentiment_momentum: f64,
    pub sentiment_volatility: f64,
    pub market_alignment: f64,
    pub conviction_strength: f64,
    pub analysis_timestamp: DateTime<Utc>,
    pub source_count: u32,
}

/// Six core sentiment dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentDimensions {
    pub polarity: f64,      // -1.0 (negative) to 1.0 (positive)
    pub fear: f64,          // -1.0 (greed) to 1.0 (fear)
    pub confidence: f64,    // -1.0 (uncertain) to 1.0 (confident)
    pub volatility: f64,    // -1.0 (stable) to 1.0 (volatile)
    pub momentum: f64,      // -1.0 (reversal) to 1.0 (continuation)
    pub conviction: f64,    // -1.0 (weak) to 1.0 (strong)
}

/// Configuration for sentiment analysis
#[derive(Clone, Debug)]
pub struct SentimentConfig {
    pub enable_lexicon_analysis: bool,
    pub enable_transformer_analysis: bool,
    pub enable_social_media_analysis: bool,
    pub enable_news_analysis: bool,
    pub enable_on_chain_analysis: bool,
    pub lexicon_weights: HashMap<String, f64>,
    pub transformer_model: String,
    pub social_media_sources: Vec<String>,
    pub news_sources: Vec<String>,
    pub update_frequency_minutes: u64,
    pub cache_duration_minutes: u64,
}

impl Default for SentimentConfig {
    fn default() -> Self {
        Self {
            enable_lexicon_analysis: true,
            enable_transformer_analysis: true,
            enable_social_media_analysis: true,
            enable_news_analysis: true,
            enable_on_chain_analysis: true,
            lexicon_weights: HashMap::new(),
            transformer_model: "distilbert-base-uncased-finetuned-sst-2-english".to_string(),
            social_media_sources: vec!["twitter".to_string(), "reddit".to_string()],
            news_sources: vec!["coindesk".to_string(), "cointelegraph".to_string()],
            update_frequency_minutes: 5,
            cache_duration_minutes: 15,
        }
    }
}

impl Default for SentimentResult {
    fn default() -> Self {
        Self {
            overall_sentiment: 0.0,
            confidence: 0.0,
            dimensions: SentimentDimensions::default(),
            key_phrases: Vec::new(),
            sentiment_momentum: 0.0,
            sentiment_volatility: 0.0,
            market_alignment: 0.0,
            conviction_strength: 0.0,
            analysis_timestamp: Utc::now(),
            source_count: 0,
        }
    }
}

impl Default for SentimentDimensions {
    fn default() -> Self {
        Self {
            polarity: 0.0,
            fear: 0.0,
            confidence: 0.0,
            volatility: 0.0,
            momentum: 0.0,
            conviction: 0.0,
        }
    }
}

/// Advanced multi-dimensional sentiment analyzer
pub struct SentimentAnalyzer {
    config: SentimentConfig,
    lexicons: HashMap<String, HashMap<String, f64>>,
    transformer_client: Option<Arc<TransformerClient>>,
    social_media_client: Option<Arc<SocialMediaClient>>,
    news_client: Option<Arc<NewsClient>>,
    on_chain_client: Option<Arc<OnChainClient>>,
    sentiment_history: Arc<RwLock<Vec<SentimentHistoryEntry>>>,
    performance_metrics: Arc<RwLock<SentimentMetrics>>,
}

#[derive(Debug, Clone)]
struct SentimentHistoryEntry {
    timestamp: DateTime<Utc>,
    symbol: String,
    sentiment: SentimentResult,
    market_price: f64,
}

#[derive(Debug, Clone)]
struct SentimentMetrics {
    total_analyses: u64,
    average_confidence: f64,
    average_conviction: f64,
    dimension_accuracy: HashMap<String, f64>,
    processing_time_ms: f64,
}

impl SentimentAnalyzer {
    pub fn new(config: SentimentConfig) -> Self {
        let lexicons = Self::build_sentiment_lexicons();
        let transformer_client = if config.enable_transformer_analysis {
            Some(Arc::new(TransformerClient::new(config.transformer_model.clone())))
        } else {
            None
        };
        let social_media_client = if config.enable_social_media_analysis {
            Some(Arc::new(SocialMediaClient::new(config.social_media_sources.clone())))
        } else {
            None
        };
        let news_client = if config.enable_news_analysis {
            Some(Arc::new(NewsClient::new(config.news_sources.clone())))
        } else {
            None
        };
        let on_chain_client = if config.enable_on_chain_analysis {
            Some(Arc::new(OnChainClient::new()))
        } else {
            None
        };

        Self {
            config,
            lexicons,
            transformer_client,
            social_media_client,
            news_client,
            on_chain_client,
            sentiment_history: Arc::new(RwLock::new(Vec::new())),
            performance_metrics: Arc::new(RwLock::new(SentimentMetrics::default())),
        }
    }

    /// Analyze multi-dimensional sentiment for a symbol
    pub async fn analyze_multi_dimensional_sentiment(
        &self,
        symbol: &str,
        context: &MarketContext,
    ) -> Result<SentimentResult, SentimentAnalysisError> {
        let start_time = std::time::Instant::now();
        
        // Collect sentiment data from multiple sources
        let lexicon_future = self.analyze_lexicon_sentiment(symbol, context);
        let transformer_future = self.analyze_transformer_sentiment(symbol, context);
        let social_future = self.analyze_social_media_sentiment(symbol, context);
        let news_future = self.analyze_news_sentiment(symbol, context);
        let on_chain_future = self.analyze_on_chain_sentiment(symbol, context);

        let (lexicon_result, transformer_result, social_result, news_result, on_chain_result) = tokio::join!(
            lexicon_future,
            transformer_future,
            social_future,
            news_future,
            on_chain_future
        );

        // Process results
        let lexicon_sentiment = lexicon_result?;
        let transformer_sentiment = transformer_result.unwrap_or_default();
        let social_sentiment = social_result.unwrap_or_default();
        let news_sentiment = news_result.unwrap_or_default();
        let on_chain_sentiment = on_chain_result.unwrap_or_default();

        // Combine all sentiment sources
        let combined_sentiment = self.combine_sentiment_sources(
            &lexicon_sentiment,
            &transformer_sentiment,
            &social_sentiment,
            &news_sentiment,
            &on_chain_sentiment,
        );

        // Calculate derived metrics
        let sentiment_momentum = self.calculate_sentiment_momentum(symbol).await?;
        let sentiment_volatility = self.calculate_sentiment_volatility(symbol).await?;
        let market_alignment = self.calculate_market_alignment(&combined_sentiment, context);
        let conviction_strength = self.calculate_conviction_strength(&combined_sentiment);

        // Extract key phrases
        let key_phrases = self.extract_key_sentiment_phrases(&combined_sentiment);

        let processing_time = start_time.elapsed().as_millis() as f64;

        let result = SentimentResult {
            overall_sentiment: combined_sentiment.overall_sentiment,
            confidence: combined_sentiment.confidence,
            dimensions: combined_sentiment.dimensions,
            key_phrases,
            sentiment_momentum,
            sentiment_volatility,
            market_alignment,
            conviction_strength,
            analysis_timestamp: Utc::now(),
            source_count: combined_sentiment.source_count,
        };

        // Store in history
        self.store_sentiment_history(symbol, &result, context.current_price).await;

        // Update performance metrics
        self.update_performance_metrics(&result, processing_time).await;

        Ok(result)
    }

    /// Analyze sentiment using lexicon-based approach
    async fn analyze_lexicon_sentiment(
        &self,
        symbol: &str,
        context: &MarketContext,
    ) -> Result<SentimentResult, SentimentAnalysisError> {
        if !self.config.enable_lexicon_analysis {
            return Ok(SentimentResult::default());
        }

        let text_sources = self.collect_text_sources(symbol, context).await?;
        let mut dimensions = SentimentDimensions::default();
        let mut key_phrases = Vec::new();
        let mut source_count = 0;

        for (source, text) in text_sources {
            let text_lower = text.to_lowercase();
            source_count += 1;

            // Analyze each dimension
            dimensions.polarity += self.analyze_dimension(&text_lower, "polarity");
            dimensions.fear += self.analyze_dimension(&text_lower, "fear");
            dimensions.confidence += self.analyze_dimension(&text_lower, "confidence");
            dimensions.volatility += self.analyze_dimension(&text_lower, "volatility");
            dimensions.momentum += self.analyze_dimension(&text_lower, "momentum");
            dimensions.conviction += self.analyze_dimension(&text_lower, "conviction");

            // Extract key phrases
            key_phrases.extend(self.extract_key_phrases_from_text(&text, &source));
        }

        if source_count > 0 {
            dimensions.polarity /= source_count as f64;
            dimensions.fear /= source_count as f64;
            dimensions.confidence /= source_count as f64;
            dimensions.volatility /= source_count as f64;
            dimensions.momentum /= source_count as f64;
            dimensions.conviction /= source_count as f64;
        }

        // Calculate overall sentiment
        let overall_sentiment = self.calculate_overall_sentiment(&dimensions);
        
        // Calculate confidence
        let confidence = self.calculate_analysis_confidence(source_count, &key_phrases);

        Ok(SentimentResult {
            overall_sentiment,
            confidence,
            dimensions,
            key_phrases: key_phrases.into_iter().take(10).collect(),
            sentiment_momentum: 0.0,
            sentiment_volatility: 0.0,
            market_alignment: 0.0,
            conviction_strength: dimensions.conviction,
            analysis_timestamp: Utc::now(),
            source_count,
        })
    }

    /// Analyze sentiment using transformer models
    async fn analyze_transformer_sentiment(
        &self,
        symbol: &str,
        context: &MarketContext,
    ) -> Result<SentimentResult, SentimentAnalysisError> {
        if let Some(transformer_client) = &self.transformer_client {
            transformer_client.analyze_sentiment(symbol, context).await
        } else {
            Ok(SentimentResult::default())
        }
    }

    /// Analyze social media sentiment
    async fn analyze_social_media_sentiment(
        &self,
        symbol: &str,
        context: &MarketContext,
    ) -> Result<SentimentResult, SentimentAnalysisError> {
        if let Some(social_client) = &self.social_media_client {
            social_client.analyze_sentiment(symbol, context).await
        } else {
            Ok(SentimentResult::default())
        }
    }

    /// Analyze news sentiment
    async fn analyze_news_sentiment(
        &self,
        symbol: &str,
        context: &MarketContext,
    ) -> Result<SentimentResult, SentimentAnalysisError> {
        if let Some(news_client) = &self.news_client {
            news_client.analyze_sentiment(symbol, context).await
        } else {
            Ok(SentimentResult::default())
        }
    }

    /// Analyze on-chain sentiment
    async fn analyze_on_chain_sentiment(
        &self,
        symbol: &str,
        context: &MarketContext,
    ) -> Result<SentimentResult, SentimentAnalysisError> {
        if let Some(on_chain_client) = &self.on_chain_client {
            on_chain_client.analyze_sentiment(symbol, context).await
        } else {
            Ok(SentimentResult::default())
        }
    }

    /// Build comprehensive sentiment lexicons
    fn build_sentiment_lexicons() -> HashMap<String, HashMap<String, f64>> {
        let mut lexicons = HashMap::new();

        // Polarity lexicon
        let mut polarity_lexicon = HashMap::new();
        polarity_lexicon.insert("bullish".to_string(), 0.8);
        polarity_lexicon.insert("bearish".to_string(), -0.8);
        polarity_lexicon.insert("positive".to_string(), 0.7);
        polarity_lexicon.insert("negative".to_string(), -0.7);
        polarity_lexicon.insert("optimistic".to_string(), 0.6);
        polarity_lexicon.insert("pessimistic".to_string(), -0.6);
        polarity_lexicon.insert("rally".to_string(), 0.9);
        polarity_lexicon.insert("crash".to_string(), -0.9);
        polarity_lexicon.insert("surge".to_string(), 0.8);
        polarity_lexicon.insert("dump".to_string(), -0.8);
        lexicons.insert("polarity".to_string(), polarity_lexicon);

        // Fear lexicon
        let mut fear_lexicon = HashMap::new();
        fear_lexicon.insert("fear".to_string(), 0.8);
        fear_lexicon.insert("panic".to_string(), 0.9);
        fear_lexicon.insert("anxiety".to_string(), 0.7);
        fear_lexicon.insert("worry".to_string(), 0.6);
        fear_lexicon.insert("greed".to_string(), -0.8);
        fear_lexicon.insert("fomo".to_string(), -0.7);
        fear_lexicon.insert("euphoria".to_string(), -0.9);
        fear_lexicon.insert("complacency".to_string(), -0.6);
        fear_lexicon.insert("cautious".to_string(), 0.5);
        fear_lexicon.insert("aggressive".to_string(), -0.5);
        lexicons.insert("fear".to_string(), fear_lexicon);

        // Confidence lexicon
        let mut confidence_lexicon = HashMap::new();
        confidence_lexicon.insert("confident".to_string(), 0.8);
        confidence_lexicon.insert("certain".to_string(), 0.9);
        confidence_lexicon.insert("sure".to_string(), 0.7);
        confidence_lexicon.insert("convinced".to_string(), 0.8);
        confidence_lexicon.insert("uncertain".to_string(), -0.7);
        confidence_lexicon.insert("doubtful".to_string(), -0.6);
        confidence_lexicon.insert("unsure".to_string(), -0.7);
        confidence_lexicon.insert("hesitant".to_string(), -0.5);
        confidence_lexicon.insert("decisive".to_string(), 0.6);
        confidence_lexicon.insert("confused".to_string(), -0.8);
        lexicons.insert("confidence".to_string(), confidence_lexicon);

        // Volatility lexicon
        let mut volatility_lexicon = HashMap::new();
        volatility_lexicon.insert("volatile".to_string(), 0.8);
        volatility_lexicon.insert("turbulent".to_string(), 0.9);
        volatility_lexicon.insert("chaotic".to_string(), 0.9);
        volatility_lexicon.insert("unstable".to_string(), 0.7);
        volatility_lexicon.insert("stable".to_string(), -0.8);
        volatility_lexicon.insert("calm".to_string(), -0.7);
        volatility_lexicon.insert("steady".to_string(), -0.6);
        volatility_lexicon.insert("consistent".to_string(), -0.7);
        volatility_lexicon.insert("erratic".to_string(), 0.8);
        volatility_lexicon.insert("predictable".to_string(), -0.6);
        lexicons.insert("volatility".to_string(), volatility_lexicon);

        // Momentum lexicon
        let mut momentum_lexicon = HashMap::new();
        momentum_lexicon.insert("momentum".to_string(), 0.7);
        momentum_lexicon.insert("trending".to_string(), 0.6);
        momentum_lexicon.insert("accelerating".to_string(), 0.8);
        momentum_lexicon.insert("slowing".to_string(), -0.6);
        momentum_lexicon.insert("reversal".to_string(), -0.8);
        momentum_lexicon.insert("continuation".to_string(), 0.7);
        momentum_lexicon.insert("breakout".to_string(), 0.8);
        momentum_lexicon.insert("breakdown".to_string(), -0.8);
        momentum_lexicon.insert("consolidation".to_string(), -0.3);
        momentum_lexicon.insert("stagnant".to_string(), -0.5);
        lexicons.insert("momentum".to_string(), momentum_lexicon);

        // Conviction lexicon
        let mut conviction_lexicon = HashMap::new();
        conviction_lexicon.insert("strong".to_string(), 0.8);
        conviction_lexicon.insert("weak".to_string(), -0.8);
        conviction_lexicon.insert("firm".to_string(), 0.7);
        conviction_lexicon.insert("solid".to_string(), 0.7);
        conviction_lexicon.insert("shaky".to_string(), -0.7);
        conviction_lexicon.insert("robust".to_string(), 0.8);
        conviction_lexicon.insert("fragile".to_string(), -0.7);
        conviction_lexicon.insert("resilient".to_string(), 0.6);
        conviction_lexicon.insert("vulnerable".to_string(), -0.6);
        conviction_lexicon.insert("unwavering".to_string(), 0.9);
        lexicons.insert("conviction".to_string(), conviction_lexicon);

        lexicons
    }

    /// Analyze specific sentiment dimension
    fn analyze_dimension(&self, text: &str, dimension: &str) -> f64 {
        if let Some(lexicon) = self.lexicons.get(dimension) {
            let mut total_score = 0.0;
            let mut match_count = 0;

            for (term, score) in lexicon {
                if text.contains(term) {
                    total_score += score;
                    match_count += 1;
                }
            }

            if match_count > 0 {
                total_score / match_count as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Collect text sources for analysis
    async fn collect_text_sources(&self, symbol: &str, context: &MarketContext) -> Result<Vec<(String, String)>, SentimentAnalysisError> {
        let mut sources = Vec::new();
        
        // Mock implementation - in production, would collect from various sources
        sources.push(("market_summary".to_string(), format!("Market analysis for {} shows mixed sentiment", symbol)));
        sources.push(("news_headline".to_string(), format!("{} trading at ${:.2} with high volatility", symbol, context.current_price)));
        
        Ok(sources)
    }

    /// Extract key phrases from text
    fn extract_key_phrases_from_text(&self, text: &str, source: &str) -> Vec<String> {
        let mut phrases = Vec::new();
        
        // Extract sentiment-rich phrases using regex
        let phrase_patterns = vec![
            r"\b(bullish|bearish|positive|negative|optimistic|pessimistic)\b",
            r"\b(fear|greed|panic|euphoria|anxiety|confidence)\b",
            r"\b(rally|crash|surge|dump|breakout|breakdown)\b",
            r"\b(volatile|stable|turbulent|calm|chaotic)\b",
            r"\b(momentum|trending|reversal|continuation)\b",
            r"\b(strong|weak|robust|fragile|solid|shaky)\b",
        ];

        for pattern in phrase_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                for mat in regex.find_iter(text) {
                    phrases.push(mat.as_str().to_string());
                }
            }
        }

        phrases
    }

    /// Calculate overall sentiment from dimensions
    fn calculate_overall_sentiment(&self, dimensions: &SentimentDimensions) -> f64 {
        let weights = [
            ("polarity", 0.3, dimensions.polarity),
            ("fear", 0.2, -dimensions.fear), // Invert fear (fear is negative)
            ("confidence", 0.2, dimensions.confidence),
            ("volatility", 0.1, -dimensions.volatility), // Invert volatility
            ("momentum", 0.1, dimensions.momentum),
            ("conviction", 0.1, dimensions.conviction),
        ];

        let weighted_sum: f64 = weights.iter()
            .map(|(_, weight, value)| weight * value)
            .sum();

        weighted_sum.clamp(-1.0, 1.0)
    }

    /// Calculate analysis confidence
    fn calculate_analysis_confidence(&self, source_count: u32, key_phrases: &[String]) -> f64 {
        let source_confidence = (source_count as f64 / 10.0).min(1.0);
        let phrase_confidence = (key_phrases.len() as f64 / 20.0).min(1.0);
        
        (source_confidence + phrase_confidence) / 2.0
    }

    /// Combine sentiment from multiple sources
    fn combine_sentiment_sources(
        &self,
        lexicon: &SentimentResult,
        transformer: &SentimentResult,
        social: &SentimentResult,
        news: &SentimentResult,
        on_chain: &SentimentResult,
    ) -> SentimentResult {
        let weights = [
            ("lexicon", 0.25, lexicon),
            ("transformer", 0.25, transformer),
            ("social", 0.20, social),
            ("news", 0.20, news),
            ("on_chain", 0.10, on_chain),
        ];

        let mut combined_dimensions = SentimentDimensions::default();
        let mut combined_confidence = 0.0;
        let mut combined_overall = 0.0;
        let mut combined_key_phrases = Vec::new();
        let mut total_sources = 0;

        for (_, weight, result) in &weights {
            if result.source_count > 0 {
                combined_dimensions.polarity += weight * result.dimensions.polarity;
                combined_dimensions.fear += weight * result.dimensions.fear;
                combined_dimensions.confidence += weight * result.dimensions.confidence;
                combined_dimensions.volatility += weight * result.dimensions.volatility;
                combined_dimensions.momentum += weight * result.dimensions.momentum;
                combined_dimensions.conviction += weight * result.dimensions.conviction;
                
                combined_confidence += weight * result.confidence;
                combined_overall += weight * result.overall_sentiment;
                combined_key_phrases.extend(result.key_phrases.clone());
                total_sources += result.source_count;
            }
        }

        // Remove duplicates and limit key phrases
        combined_key_phrases.sort();
        combined_key_phrases.dedup();
        combined_key_phrases.truncate(15);

        SentimentResult {
            overall_sentiment: combined_overall,
            confidence: combined_confidence,
            dimensions: combined_dimensions,
            key_phrases: combined_key_phrases,
            sentiment_momentum: 0.0,
            sentiment_volatility: 0.0,
            market_alignment: 0.0,
            conviction_strength: combined_dimensions.conviction,
            analysis_timestamp: Utc::now(),
            source_count: total_sources,
        }
    }

    /// Calculate sentiment momentum
    async fn calculate_sentiment_momentum(&self, symbol: &str) -> Result<f64, SentimentAnalysisError> {
        let history = self.sentiment_history.read().await;
        let recent_entries: Vec<_> = history.iter()
            .filter(|entry| entry.symbol == symbol)
            .rev()
            .take(10)
            .collect();

        if recent_entries.len() < 2 {
            return Ok(0.0);
        }

        let recent_sentiment = recent_entries[0].sentiment.overall_sentiment;
        let older_sentiment = recent_entries.last().unwrap().sentiment.overall_sentiment;
        
        Ok(recent_sentiment - older_sentiment)
    }

    /// Calculate sentiment volatility
    async fn calculate_sentiment_volatility(&self, symbol: &str) -> Result<f64, SentimentAnalysisError> {
        let history = self.sentiment_history.read().await;
        let recent_entries: Vec<_> = history.iter()
            .filter(|entry| entry.symbol == symbol)
            .rev()
            .take(20)
            .collect();

        if recent_entries.len() < 2 {
            return Ok(0.0);
        }

        let sentiments: Vec<f64> = recent_entries.iter()
            .map(|entry| entry.sentiment.overall_sentiment)
            .collect();

        let mean = sentiments.iter().sum::<f64>() / sentiments.len() as f64;
        let variance = sentiments.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>() / sentiments.len() as f64;

        Ok(variance.sqrt())
    }

    /// Calculate market alignment
    fn calculate_market_alignment(&self, sentiment: &SentimentResult, context: &MarketContext) -> f64 {
        let price_momentum = context.momentum_score;
        let sentiment_momentum = sentiment.sentiment_momentum;
        
        // Calculate alignment between sentiment and price momentum
        let momentum_alignment = 1.0 - (price_momentum - sentiment_momentum).abs();
        
        // Factor in market regime alignment
        let regime_alignment = match context.market_regime.as_str() {
            "bull" => if sentiment.overall_sentiment > 0.0 { 1.0 } else { 0.0 },
            "bear" => if sentiment.overall_sentiment < 0.0 { 1.0 } else { 0.0 },
            _ => 0.5,
        };

        (momentum_alignment + regime_alignment) / 2.0
    }

    /// Calculate conviction strength
    fn calculate_conviction_strength(&self, sentiment: &SentimentResult) -> f64 {
        let dimension_strength = (
            sentiment.dimensions.conviction.abs() +
            sentiment.dimensions.confidence.abs() +
            sentiment.dimensions.polarity.abs()
        ) / 3.0;

        dimension_strength * sentiment.confidence
    }

    /// Extract key sentiment phrases
    fn extract_key_sentiment_phrases(&self, sentiment: &SentimentResult) -> Vec<String> {
        sentiment.key_phrases.clone()
    }

    /// Store sentiment history
    async fn store_sentiment_history(&self, symbol: &str, sentiment: &SentimentResult, market_price: f64) {
        let mut history = self.sentiment_history.write().await;
        
        history.push(SentimentHistoryEntry {
            timestamp: Utc::now(),
            symbol: symbol.to_string(),
            sentiment: sentiment.clone(),
            market_price,
        });

        // Limit history size
        if history.len() > 1000 {
            history.drain(0..history.len() - 1000);
        }
    }

    /// Update performance metrics
    async fn update_performance_metrics(&self, result: &SentimentResult, processing_time: f64) {
        let mut metrics = self.performance_metrics.write().await;
        
        metrics.total_analyses += 1;
        metrics.average_confidence = (metrics.average_confidence * (metrics.total_analyses - 1) as f64 + result.confidence) / metrics.total_analyses as f64;
        metrics.average_conviction = (metrics.average_conviction * (metrics.total_analyses - 1) as f64 + result.conviction_strength) / metrics.total_analyses as f64;
        metrics.processing_time_ms = (metrics.processing_time_ms * (metrics.total_analyses - 1) as f64 + processing_time) / metrics.total_analyses as f64;
    }
}

impl Default for SentimentMetrics {
    fn default() -> Self {
        Self {
            total_analyses: 0,
            average_confidence: 0.0,
            average_conviction: 0.0,
            dimension_accuracy: HashMap::new(),
            processing_time_ms: 0.0,
        }
    }
}

// Client implementations for different sentiment sources
pub struct TransformerClient {
    model_name: String,
    client: reqwest::Client,
}

impl TransformerClient {
    pub fn new(model_name: String) -> Self {
        Self {
            model_name,
            client: reqwest::Client::new(),
        }
    }

    pub async fn analyze_sentiment(&self, symbol: &str, context: &MarketContext) -> Result<SentimentResult, SentimentAnalysisError> {
        // Placeholder implementation
        Ok(SentimentResult::default())
    }
}

pub struct SocialMediaClient {
    sources: Vec<String>,
    client: reqwest::Client,
}

impl SocialMediaClient {
    pub fn new(sources: Vec<String>) -> Self {
        Self {
            sources,
            client: reqwest::Client::new(),
        }
    }

    pub async fn analyze_sentiment(&self, symbol: &str, context: &MarketContext) -> Result<SentimentResult, SentimentAnalysisError> {
        // Placeholder implementation
        Ok(SentimentResult::default())
    }
}

pub struct NewsClient {
    sources: Vec<String>,
    client: reqwest::Client,
}

impl NewsClient {
    pub fn new(sources: Vec<String>) -> Self {
        Self {
            sources,
            client: reqwest::Client::new(),
        }
    }

    pub async fn analyze_sentiment(&self, symbol: &str, context: &MarketContext) -> Result<SentimentResult, SentimentAnalysisError> {
        // Placeholder implementation
        Ok(SentimentResult::default())
    }
}

pub struct OnChainClient {
    client: reqwest::Client,
}

impl OnChainClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    pub async fn analyze_sentiment(&self, symbol: &str, context: &MarketContext) -> Result<SentimentResult, SentimentAnalysisError> {
        // Placeholder implementation
        Ok(SentimentResult::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sentiment_analyzer_creation() {
        let config = SentimentConfig::default();
        let analyzer = SentimentAnalyzer::new(config);
        
        // Test lexicon creation
        assert!(analyzer.lexicons.contains_key("polarity"));
        assert!(analyzer.lexicons.contains_key("fear"));
        assert!(analyzer.lexicons.contains_key("confidence"));
        assert!(analyzer.lexicons.contains_key("volatility"));
        assert!(analyzer.lexicons.contains_key("momentum"));
        assert!(analyzer.lexicons.contains_key("conviction"));
    }

    #[tokio::test]
    async fn test_dimension_analysis() {
        let config = SentimentConfig::default();
        let analyzer = SentimentAnalyzer::new(config);
        
        let bullish_text = "the market is very bullish and positive with strong momentum";
        let bearish_text = "the market is bearish and negative with weak sentiment";
        
        let bullish_score = analyzer.analyze_dimension(bullish_text, "polarity");
        let bearish_score = analyzer.analyze_dimension(bearish_text, "polarity");
        
        assert!(bullish_score > 0.0);
        assert!(bearish_score < 0.0);
    }

    #[test]
    fn test_overall_sentiment_calculation() {
        let config = SentimentConfig::default();
        let analyzer = SentimentAnalyzer::new(config);
        
        let dimensions = SentimentDimensions {
            polarity: 0.8,
            fear: -0.2,
            confidence: 0.7,
            volatility: 0.3,
            momentum: 0.6,
            conviction: 0.9,
        };
        
        let overall = analyzer.calculate_overall_sentiment(&dimensions);
        assert!(overall > 0.0);
        assert!(overall <= 1.0);
    }
}