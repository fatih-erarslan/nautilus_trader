use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use dashmap::DashMap;
use futures::stream::{self, StreamExt};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

pub mod llm_client;
pub mod sentiment_analyzer;
pub mod narrative_builder;
pub mod prediction_extractor;
pub mod claude_client;

use llm_client::{LLMClient, LLMProvider, LLMConfig};
use sentiment_analyzer::{SentimentAnalyzer, SentimentDimension, SentimentResult};
use narrative_builder::NarrativeBuilder;
use prediction_extractor::PredictionExtractor;

#[derive(Error, Debug)]
pub enum NarrativeError {
    #[error("LLM API error: {0}")]
    LLMError(String),
    
    #[error("Sentiment analysis error: {0}")]
    SentimentError(String),
    
    #[error("Prediction extraction error: {0}")]
    ExtractionError(String),
    
    #[error("Cache error: {0}")]
    CacheError(String),
    
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeForecast {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub current_price: f64,
    pub narrative: String,
    pub price_prediction: f64,
    pub confidence_score: f64,
    pub timeframe: String,
    pub sentiment_analysis: SentimentResult,
    pub key_factors: Vec<String>,
    pub execution_time_ms: f64,
    pub llm_provider: String,
    pub model_used: String,
    pub cache_hit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    pub volume: f64,
    pub support_level: f64,
    pub resistance_level: f64,
    pub additional_context: HashMap<String, String>,
}

#[derive(Clone)]
pub struct NarrativeForecaster {
    llm_client: Arc<dyn LLMClient + Send + Sync>,
    sentiment_analyzer: Arc<SentimentAnalyzer>,
    narrative_builder: Arc<NarrativeBuilder>,
    prediction_extractor: Arc<PredictionExtractor>,
    cache: Arc<DashMap<String, CachedForecast>>,
    config: ForecasterConfig,
    prediction_history: Arc<RwLock<Vec<PredictionEntry>>>,
}

#[derive(Clone)]
pub struct ForecasterConfig {
    pub cache_duration_minutes: u64,
    pub rate_limit_delay_ms: u64,
    pub max_retries: u32,
    pub timeout_seconds: u64,
    pub enable_sentiment_analysis: bool,
    pub enable_caching: bool,
    pub max_history_size: usize,
}

impl Default for ForecasterConfig {
    fn default() -> Self {
        Self {
            cache_duration_minutes: 15,
            rate_limit_delay_ms: 1000,
            max_retries: 3,
            timeout_seconds: 30,
            enable_sentiment_analysis: true,
            enable_caching: true,
            max_history_size: 1000,
        }
    }
}

#[derive(Clone)]
struct CachedForecast {
    forecast: NarrativeForecast,
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct PredictionEntry {
    timestamp: DateTime<Utc>,
    symbol: String,
    current_price: f64,
    predicted_price: f64,
    confidence: f64,
    actual_price: Option<f64>,
    sentiment_score: f64,
}

impl NarrativeForecaster {
    pub fn new(llm_config: LLMConfig, config: ForecasterConfig) -> Result<Self, NarrativeError> {
        // Create LLM client based on provider
        let llm_client: Arc<dyn LLMClient + Send + Sync> = match llm_config.provider {
            LLMProvider::Claude => {
                Arc::new(claude_client::ClaudeClient::new(llm_config)?)
            }
            LLMProvider::OpenAI => {
                Arc::new(llm_client::OpenAIClient::new(llm_config)?)
            }
            LLMProvider::Ollama => {
                Arc::new(llm_client::OllamaClient::new(llm_config)?)
            }
            LLMProvider::LMStudio => {
                Arc::new(llm_client::LMStudioClient::new(llm_config)?)
            }
        };
        
        Ok(Self {
            llm_client,
            sentiment_analyzer: Arc::new(SentimentAnalyzer::new()),
            narrative_builder: Arc::new(NarrativeBuilder::new()),
            prediction_extractor: Arc::new(PredictionExtractor::new()),
            cache: Arc::new(DashMap::new()),
            config,
            prediction_history: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    pub async fn generate_narrative_forecast(
        &self,
        symbol: &str,
        context: &MarketContext,
    ) -> Result<NarrativeForecast, NarrativeError> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        if self.config.enable_caching {
            if let Some(cached) = self.get_cached_forecast(symbol, context).await {
                let mut forecast = cached.forecast;
                forecast.cache_hit = true;
                return Ok(forecast);
            }
        }
        
        // Rate limiting
        self.apply_rate_limiting().await;
        
        // Build narrative prompt
        let prompt = self.narrative_builder.build_future_retrospective_prompt(
            symbol,
            context,
        )?;
        
        // Generate narrative using LLM
        let narrative_response = self.llm_client.generate_response(&prompt).await?;
        
        // Extract predictions from narrative
        let prediction_data = self.prediction_extractor.extract_prediction_data(
            &narrative_response,
            context.support_level,
        )?;
        
        // Perform sentiment analysis if enabled
        let sentiment_analysis = if self.config.enable_sentiment_analysis {
            self.sentiment_analyzer.analyze_comprehensive_sentiment(
                &narrative_response,
                symbol,
            ).await?
        } else {
            SentimentResult::default()
        };
        
        // Extract key factors
        let key_factors = self.extract_key_factors(&narrative_response);
        
        let execution_time = start_time.elapsed().as_millis() as f64;
        
        // Build final forecast
        let forecast = NarrativeForecast {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            current_price: context.support_level, // Using support as current price for now
            narrative: narrative_response,
            price_prediction: prediction_data.price_prediction,
            confidence_score: prediction_data.confidence_score,
            timeframe: prediction_data.timeframe,
            sentiment_analysis,
            key_factors,
            execution_time_ms: execution_time,
            llm_provider: self.llm_client.provider_name(),
            model_used: self.llm_client.model_name(),
            cache_hit: false,
        };
        
        // Cache the result
        if self.config.enable_caching {
            self.cache_forecast(symbol, context, &forecast).await;
        }
        
        // Store in prediction history
        self.store_prediction_history(&forecast).await;
        
        Ok(forecast)
    }
    
    pub async fn analyze_batch_symbols(
        &self,
        symbols_with_context: Vec<(String, MarketContext)>,
    ) -> Result<Vec<NarrativeForecast>, NarrativeError> {
        let futures: Vec<_> = symbols_with_context.into_iter()
            .map(|(symbol, context)| {
                let forecaster = self.clone();
                async move {
                    forecaster.generate_narrative_forecast(&symbol, &context).await
                }
            })
            .collect();
        
        // Process in parallel with concurrency limit
        let results = stream::iter(futures)
            .buffer_unordered(5) // Limit concurrent requests
            .collect::<Vec<_>>()
            .await;
        
        // Collect successful results
        let mut forecasts = Vec::new();
        for result in results {
            match result {
                Ok(forecast) => forecasts.push(forecast),
                Err(e) => log::warn!("Failed to generate forecast: {}", e),
            }
        }
        
        Ok(forecasts)
    }
    
    pub async fn update_prediction_accuracy(
        &self,
        symbol: &str,
        timestamp: DateTime<Utc>,
        actual_price: f64,
    ) -> Result<(), NarrativeError> {
        let mut history = self.prediction_history.write().await;
        
        // Find matching prediction and update with actual price
        for entry in history.iter_mut() {
            if entry.symbol == symbol && 
               (timestamp - entry.timestamp).num_minutes().abs() < 60 { // Within 1 hour
                entry.actual_price = Some(actual_price);
                break;
            }
        }
        
        Ok(())
    }
    
    pub async fn get_accuracy_metrics(&self) -> Result<AccuracyMetrics, NarrativeError> {
        let history = self.prediction_history.read().await;
        
        let completed_predictions: Vec<_> = history.iter()
            .filter(|entry| entry.actual_price.is_some())
            .collect();
        
        if completed_predictions.is_empty() {
            return Ok(AccuracyMetrics::default());
        }
        
        // Calculate MAPE (Mean Absolute Percentage Error)
        let mape = completed_predictions.iter()
            .map(|entry| {
                let actual = entry.actual_price.unwrap();
                let predicted = entry.predicted_price;
                ((actual - predicted).abs() / actual) * 100.0
            })
            .sum::<f64>() / completed_predictions.len() as f64;
        
        // Calculate RMSE (Root Mean Square Error)
        let mse = completed_predictions.iter()
            .map(|entry| {
                let actual = entry.actual_price.unwrap();
                let predicted = entry.predicted_price;
                (actual - predicted).powi(2)
            })
            .sum::<f64>() / completed_predictions.len() as f64;
        let rmse = mse.sqrt();
        
        // Calculate directional accuracy
        let correct_direction = completed_predictions.iter()
            .filter(|entry| {
                let actual = entry.actual_price.unwrap();
                let predicted = entry.predicted_price;
                let current = entry.current_price;
                
                let actual_direction = actual > current;
                let predicted_direction = predicted > current;
                
                actual_direction == predicted_direction
            })
            .count();
        
        let directional_accuracy = (correct_direction as f64 / completed_predictions.len() as f64) * 100.0;
        
        Ok(AccuracyMetrics {
            mape,
            rmse,
            directional_accuracy,
            total_predictions: completed_predictions.len(),
            average_confidence: completed_predictions.iter()
                .map(|entry| entry.confidence)
                .sum::<f64>() / completed_predictions.len() as f64,
        })
    }
    
    async fn get_cached_forecast(
        &self,
        symbol: &str,
        context: &MarketContext,
    ) -> Option<CachedForecast> {
        let cache_key = self.generate_cache_key(symbol, context);
        
        self.cache.get(&cache_key).and_then(|entry| {
            let cached = entry.value();
            if (Utc::now() - cached.timestamp).num_minutes() < self.config.cache_duration_minutes as i64 {
                Some(cached.clone())
            } else {
                None
            }
        })
    }
    
    async fn cache_forecast(
        &self,
        symbol: &str,
        context: &MarketContext,
        forecast: &NarrativeForecast,
    ) {
        let cache_key = self.generate_cache_key(symbol, context);
        
        self.cache.insert(cache_key, CachedForecast {
            forecast: forecast.clone(),
            timestamp: Utc::now(),
        });
        
        // Clean old cache entries periodically
        if self.cache.len() > 1000 {
            self.clean_cache().await;
        }
    }
    
    async fn clean_cache(&self) {
        let cutoff = Utc::now() - Duration::minutes(self.config.cache_duration_minutes as i64);
        
        self.cache.retain(|_, cached| {
            cached.timestamp > cutoff
        });
    }
    
    fn generate_cache_key(&self, symbol: &str, context: &MarketContext) -> String {
        // Create a hash-based cache key
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        symbol.hash(&mut hasher);
        context.volume.to_bits().hash(&mut hasher);
        context.support_level.to_bits().hash(&mut hasher);
        context.resistance_level.to_bits().hash(&mut hasher);
        
        format!("{}_{:x}", symbol, hasher.finish())
    }
    
    async fn apply_rate_limiting(&self) {
        if self.config.rate_limit_delay_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(
                self.config.rate_limit_delay_ms
            )).await;
        }
    }
    
    async fn store_prediction_history(&self, forecast: &NarrativeForecast) {
        let mut history = self.prediction_history.write().await;
        
        history.push(PredictionEntry {
            timestamp: forecast.timestamp,
            symbol: forecast.symbol.clone(),
            current_price: forecast.current_price,
            predicted_price: forecast.price_prediction,
            confidence: forecast.confidence_score,
            actual_price: None,
            sentiment_score: forecast.sentiment_analysis.overall_sentiment,
        });
        
        // Limit history size
        if history.len() > self.config.max_history_size {
            history.drain(0..history.len() - self.config.max_history_size);
        }
    }
    
    fn extract_key_factors(&self, narrative: &str) -> Vec<String> {
        // Extract key factors from narrative using regex patterns
        let patterns = vec![
            r"(?i)key factors?:?\s*(.+?)(?:\n|$)",
            r"(?i)factors?:?\s*(.+?)(?:\n|$)",
            r"(?i)drivers?:?\s*(.+?)(?:\n|$)",
            r"(?i)influenced by:?\s*(.+?)(?:\n|$)",
        ];
        
        for pattern in patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if let Some(captures) = regex.captures(narrative) {
                    if let Some(factors_text) = captures.get(1) {
                        return factors_text.as_str()
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .take(5) // Limit to 5 factors
                            .collect();
                    }
                }
            }
        }
        
        // Fallback: extract common trading terms
        let trading_terms = vec![
            "support", "resistance", "volume", "momentum", "breakout",
            "trend", "sentiment", "volatility", "pressure", "demand",
        ];
        
        trading_terms.into_iter()
            .filter(|term| narrative.to_lowercase().contains(term))
            .map(|s| s.to_string())
            .take(3)
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub mape: f64,                    // Mean Absolute Percentage Error
    pub rmse: f64,                    // Root Mean Square Error
    pub directional_accuracy: f64,    // Percentage of correct direction predictions
    pub total_predictions: usize,     // Total number of predictions evaluated
    pub average_confidence: f64,      // Average confidence score
}

impl Default for AccuracyMetrics {
    fn default() -> Self {
        Self {
            mape: 0.0,
            rmse: 0.0,
            directional_accuracy: 0.0,
            total_predictions: 0,
            average_confidence: 0.0,
        }
    }
}

// Factory function for easy initialization
pub fn create_claude_forecaster(api_key: String) -> Result<NarrativeForecaster, NarrativeError> {
    let llm_config = LLMConfig {
        provider: LLMProvider::Claude,
        api_key: Some(api_key),
        model: "claude-sonnet-4-20250514".to_string(), // Claude Sonnet 4 - same model as Claude Max
        base_url: Some("https://api.anthropic.com/v1/messages".to_string()),
        max_tokens: 1500, // Increased for better reasoning depth
        temperature: 0.3, // Slightly lower for more consistent reasoning
        timeout_seconds: 45, // Increased timeout for more complex reasoning
    };
    
    let config = ForecasterConfig {
        cache_duration_minutes: 10, // Shorter cache for Sonnet 4's dynamic reasoning
        rate_limit_delay_ms: 800,   // Slightly faster rate for premium model
        max_retries: 3,
        timeout_seconds: 45,
        enable_sentiment_analysis: true,
        enable_caching: true,
        max_history_size: 1500, // Larger history for better pattern recognition
    };
    
    NarrativeForecaster::new(llm_config, config)
}

pub fn create_fallback_forecaster() -> Result<NarrativeForecaster, NarrativeError> {
    // Try LMStudio first, then Ollama
    let llm_config = LLMConfig {
        provider: LLMProvider::LMStudio,
        api_key: None,
        model: "llama-3.1-8b-instruct".to_string(),
        base_url: Some("http://localhost:1234/v1/chat/completions".to_string()),
        max_tokens: 1000,
        temperature: 0.4,
        timeout_seconds: 30,
    };
    
    let config = ForecasterConfig::default();
    
    NarrativeForecaster::new(llm_config, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_narrative_forecaster_creation() {
        let config = ForecasterConfig::default();
        let llm_config = LLMConfig {
            provider: LLMProvider::Claude,
            api_key: Some("test-key".to_string()),
            model: "claude-sonnet-4-20250514".to_string(), // Updated to Sonnet 4
            base_url: Some("https://api.anthropic.com/v1/messages".to_string()),
            max_tokens: 1500,
            temperature: 0.3,
            timeout_seconds: 45,
        };
        
        let forecaster = NarrativeForecaster::new(llm_config, config);
        assert!(forecaster.is_ok());
    }
    
    #[tokio::test]
    async fn test_cache_key_generation() {
        let config = ForecasterConfig::default();
        let llm_config = LLMConfig {
            provider: LLMProvider::Claude,
            api_key: Some("test-key".to_string()),
            model: "claude-sonnet-4-20250514".to_string(), // Updated to Sonnet 4
            base_url: Some("https://api.anthropic.com/v1/messages".to_string()),
            max_tokens: 1500,
            temperature: 0.3,
            timeout_seconds: 45,
        };
        
        let forecaster = NarrativeForecaster::new(llm_config, config).unwrap();
        
        let context = MarketContext {
            volume: 1000000.0,
            support_level: 50000.0,
            resistance_level: 55000.0,
            additional_context: HashMap::new(),
        };
        
        let key1 = forecaster.generate_cache_key("BTC/USDT", &context);
        let key2 = forecaster.generate_cache_key("BTC/USDT", &context);
        let key3 = forecaster.generate_cache_key("ETH/USDT", &context);
        
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}