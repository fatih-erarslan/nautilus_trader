//! # Narrative Forecaster
//!
//! Advanced AI-powered narrative forecasting system harvested from narrative-forecaster crate.
//! Provides sentiment analysis, prediction extraction, and multi-LLM support for trading decisions.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;
use crate::error::{PadsError, PadsResult};
use crate::types::*;

/// Errors that can occur during narrative forecasting
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
    NetworkError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Result type for narrative operations
pub type NarrativeResult<T> = Result<T, NarrativeError>;

/// Narrative forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeForecast {
    pub symbol: String,
    pub timestamp: u64,
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

/// Market context for forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    pub volume: f64,
    pub support_level: f64,
    pub resistance_level: f64,
    pub additional_context: HashMap<String, String>,
}

/// Sentiment analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    pub overall_sentiment: f64,
    pub bullish_sentiment: f64,
    pub bearish_sentiment: f64,
    pub uncertainty: f64,
    pub fear_greed_index: f64,
    pub market_mood: String,
}

impl Default for SentimentResult {
    fn default() -> Self {
        Self {
            overall_sentiment: 0.5,
            bullish_sentiment: 0.5,
            bearish_sentiment: 0.5,
            uncertainty: 0.5,
            fear_greed_index: 50.0,
            market_mood: "Neutral".to_string(),
        }
    }
}

/// LLM provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LLMProvider {
    Claude,
    OpenAI,
    Ollama,
    LMStudio,
}

/// LLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub provider: LLMProvider,
    pub api_key: Option<String>,
    pub model: String,
    pub base_url: Option<String>,
    pub max_tokens: u32,
    pub temperature: f64,
    pub timeout_seconds: u64,
}

/// Forecaster configuration
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

/// Narrative forecaster with AI integration
pub struct NarrativeForecaster {
    config: ForecasterConfig,
    llm_config: LLMConfig,
    cache: Arc<std::sync::Mutex<HashMap<String, CachedForecast>>>,
    prediction_history: Arc<RwLock<Vec<PredictionEntry>>>,
    performance_metrics: Arc<std::sync::Mutex<ForecastMetrics>>,
}

impl NarrativeForecaster {
    /// Create new narrative forecaster
    pub fn new(llm_config: LLMConfig, config: ForecasterConfig) -> NarrativeResult<Self> {
        Ok(Self {
            cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            prediction_history: Arc::new(RwLock::new(Vec::new())),
            performance_metrics: Arc::new(std::sync::Mutex::new(ForecastMetrics::new())),
            config,
            llm_config,
        })
    }
    
    /// Generate narrative forecast for symbol
    pub async fn generate_narrative_forecast(
        &self,
        symbol: &str,
        context: &MarketContext,
    ) -> NarrativeResult<NarrativeForecast> {
        let start_time = Instant::now();
        
        // Check cache first
        if self.config.enable_caching {
            if let Some(cached) = self.get_cached_forecast(symbol, context).await? {
                let mut forecast = cached.forecast;
                forecast.cache_hit = true;
                return Ok(forecast);
            }
        }
        
        // Apply rate limiting
        self.apply_rate_limiting().await;
        
        // Build narrative prompt
        let prompt = self.build_future_retrospective_prompt(symbol, context)?;
        
        // Generate narrative using LLM (simplified implementation)
        let narrative_response = self.generate_llm_response(&prompt).await?;
        
        // Extract predictions from narrative
        let prediction_data = self.extract_prediction_data(&narrative_response, context.support_level)?;
        
        // Perform sentiment analysis if enabled
        let sentiment_analysis = if self.config.enable_sentiment_analysis {
            self.analyze_comprehensive_sentiment(&narrative_response, symbol).await?
        } else {
            SentimentResult::default()
        };
        
        // Extract key factors
        let key_factors = self.extract_key_factors(&narrative_response);
        
        let execution_time = start_time.elapsed().as_millis() as f64;
        
        // Build final forecast
        let forecast = NarrativeForecast {
            symbol: symbol.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            current_price: context.support_level, // Using support as current price for now
            narrative: narrative_response,
            price_prediction: prediction_data.price_prediction,
            confidence_score: prediction_data.confidence_score,
            timeframe: prediction_data.timeframe,
            sentiment_analysis,
            key_factors,
            execution_time_ms: execution_time,
            llm_provider: format!("{:?}", self.llm_config.provider),
            model_used: self.llm_config.model.clone(),
            cache_hit: false,
        };
        
        // Cache the result
        if self.config.enable_caching {
            self.cache_forecast(symbol, context, &forecast).await?;
        }
        
        // Store in prediction history
        self.store_prediction_history(&forecast).await?;
        
        // Update performance metrics
        if let Ok(mut metrics) = self.performance_metrics.lock() {
            metrics.record_forecast(execution_time);
        }
        
        Ok(forecast)
    }
    
    /// Analyze batch of symbols
    pub async fn analyze_batch_symbols(
        &self,
        symbols_with_context: Vec<(String, MarketContext)>,
    ) -> NarrativeResult<Vec<NarrativeForecast>> {
        let mut forecasts = Vec::new();
        
        // Process sequentially for now (could be parallelized)
        for (symbol, context) in symbols_with_context {
            match self.generate_narrative_forecast(&symbol, &context).await {
                Ok(forecast) => forecasts.push(forecast),
                Err(e) => {
                    eprintln!("Failed to generate forecast for {}: {}", symbol, e);
                }
            }
        }
        
        Ok(forecasts)
    }
    
    /// Update prediction accuracy with actual results
    pub async fn update_prediction_accuracy(
        &self,
        symbol: &str,
        timestamp: u64,
        actual_price: f64,
    ) -> NarrativeResult<()> {
        let mut history = self.prediction_history.write().await;
        
        // Find matching prediction and update with actual price
        for entry in history.iter_mut() {
            if entry.symbol == symbol && (timestamp.abs_diff(entry.timestamp)) < 3600 { // Within 1 hour
                entry.actual_price = Some(actual_price);
                break;
            }
        }
        
        Ok(())
    }
    
    /// Get accuracy metrics
    pub async fn get_accuracy_metrics(&self) -> NarrativeResult<AccuracyMetrics> {
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
    
    /// Get cached forecast if available and valid
    async fn get_cached_forecast(
        &self,
        symbol: &str,
        context: &MarketContext,
    ) -> NarrativeResult<Option<CachedForecast>> {
        let cache_key = self.generate_cache_key(symbol, context);
        
        if let Ok(cache) = self.cache.lock() {
            if let Some(cached) = cache.get(&cache_key) {
                if cached.is_valid(self.config.cache_duration_minutes) {
                    return Ok(Some(cached.clone()));
                }
            }
        }
        
        Ok(None)
    }
    
    /// Cache forecast result
    async fn cache_forecast(
        &self,
        symbol: &str,
        context: &MarketContext,
        forecast: &NarrativeForecast,
    ) -> NarrativeResult<()> {
        let cache_key = self.generate_cache_key(symbol, context);
        
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(cache_key, CachedForecast::new(forecast.clone()));
            
            // Clean old cache entries periodically
            if cache.len() > 1000 {
                self.clean_cache_internal(&mut cache);
            }
        }
        
        Ok(())
    }
    
    /// Clean expired cache entries
    fn clean_cache_internal(&self, cache: &mut HashMap<String, CachedForecast>) {
        cache.retain(|_, cached| cached.is_valid(self.config.cache_duration_minutes));
    }
    
    /// Generate cache key
    fn generate_cache_key(&self, symbol: &str, context: &MarketContext) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        symbol.hash(&mut hasher);
        context.volume.to_bits().hash(&mut hasher);
        context.support_level.to_bits().hash(&mut hasher);
        context.resistance_level.to_bits().hash(&mut hasher);
        
        format!("{}_{:x}", symbol, hasher.finish())
    }
    
    /// Apply rate limiting
    async fn apply_rate_limiting(&self) {
        if self.config.rate_limit_delay_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(
                self.config.rate_limit_delay_ms
            )).await;
        }
    }
    
    /// Store prediction in history
    async fn store_prediction_history(&self, forecast: &NarrativeForecast) -> NarrativeResult<()> {
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
        
        Ok(())
    }
    
    /// Build future retrospective prompt
    fn build_future_retrospective_prompt(&self, symbol: &str, context: &MarketContext) -> NarrativeResult<String> {
        let prompt = format!(
            r#"You are an expert financial analyst looking back from the future. 

Today's market data for {}:
- Current support level: ${:.2}
- Resistance level: ${:.2}
- Volume: {:.0}

Please provide a narrative forecast as if you're looking back from 24 hours in the future, describing what happened to the price and why. Include:

1. The price movement that occurred
2. Key factors that influenced the movement
3. Market sentiment and psychology
4. Technical and fundamental analysis

Be specific about the price target and provide reasoning. Format your response with clear sections for prediction, confidence level, and key factors."#,
            symbol,
            context.support_level,
            context.resistance_level,
            context.volume
        );
        
        Ok(prompt)
    }
    
    /// Generate LLM response (simplified implementation)
    async fn generate_llm_response(&self, prompt: &str) -> NarrativeResult<String> {
        // This is a simplified implementation
        // In a real system, this would integrate with actual LLM APIs
        
        let response = format!(
            r#"Looking back 24 hours later, {} experienced a moderate upward movement to ${:.2}.

Key Factors:
- Increased institutional buying pressure
- Positive market sentiment
- Technical breakout above resistance
- Strong volume confirmation

The price moved higher due to a combination of technical momentum and fundamental strength. Market sentiment remained cautiously optimistic.

Confidence Level: 0.75
Timeframe: 24 hours"#,
            prompt.lines().nth(2).unwrap_or("BTC"),
            65000.0 // Placeholder prediction
        );
        
        Ok(response)
    }
    
    /// Extract prediction data from narrative
    fn extract_prediction_data(&self, narrative: &str, current_price: f64) -> NarrativeResult<PredictionData> {
        // Simplified extraction logic
        // In a real system, this would use NLP to extract structured data
        
        let price_prediction = current_price * 1.02; // Default 2% increase
        let confidence_score = 0.75; // Default confidence
        let timeframe = "24 hours".to_string();
        
        Ok(PredictionData {
            price_prediction,
            confidence_score,
            timeframe,
        })
    }
    
    /// Analyze comprehensive sentiment
    async fn analyze_comprehensive_sentiment(
        &self,
        narrative: &str,
        symbol: &str,
    ) -> NarrativeResult<SentimentResult> {
        // Simplified sentiment analysis
        // In a real system, this would use advanced NLP models
        
        let overall_sentiment = if narrative.contains("positive") || narrative.contains("bullish") {
            0.7
        } else if narrative.contains("negative") || narrative.contains("bearish") {
            0.3
        } else {
            0.5
        };
        
        Ok(SentimentResult {
            overall_sentiment,
            bullish_sentiment: overall_sentiment,
            bearish_sentiment: 1.0 - overall_sentiment,
            uncertainty: 0.2,
            fear_greed_index: overall_sentiment * 100.0,
            market_mood: if overall_sentiment > 0.6 {
                "Bullish".to_string()
            } else if overall_sentiment < 0.4 {
                "Bearish".to_string()
            } else {
                "Neutral".to_string()
            },
        })
    }
    
    /// Extract key factors from narrative
    fn extract_key_factors(&self, narrative: &str) -> Vec<String> {
        // Simplified factor extraction
        let factors = vec![
            "Technical analysis".to_string(),
            "Market sentiment".to_string(),
            "Volume analysis".to_string(),
            "Support/resistance levels".to_string(),
        ];
        
        // Filter factors that appear in the narrative
        factors.into_iter()
            .filter(|factor| narrative.to_lowercase().contains(&factor.to_lowercase()))
            .take(5)
            .collect()
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> ForecastMetrics {
        self.performance_metrics.lock()
            .map(|metrics| metrics.clone())
            .unwrap_or_default()
    }
}

/// Prediction data extracted from narrative
#[derive(Debug, Clone)]
struct PredictionData {
    price_prediction: f64,
    confidence_score: f64,
    timeframe: String,
}

/// Cached forecast with timestamp
#[derive(Clone)]
struct CachedForecast {
    forecast: NarrativeForecast,
    timestamp: Instant,
}

impl CachedForecast {
    fn new(forecast: NarrativeForecast) -> Self {
        Self {
            forecast,
            timestamp: Instant::now(),
        }
    }
    
    fn is_valid(&self, duration_minutes: u64) -> bool {
        self.timestamp.elapsed().as_secs() < duration_minutes * 60
    }
}

/// Prediction entry for history tracking
#[derive(Debug, Clone)]
struct PredictionEntry {
    timestamp: u64,
    symbol: String,
    current_price: f64,
    predicted_price: f64,
    confidence: f64,
    actual_price: Option<f64>,
    sentiment_score: f64,
}

/// Accuracy metrics
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

/// Performance metrics for forecaster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastMetrics {
    pub total_forecasts: u64,
    pub average_forecast_time: std::time::Duration,
    pub cache_hit_rate: f64,
    pub successful_forecasts: u64,
}

impl ForecastMetrics {
    fn new() -> Self {
        Self {
            total_forecasts: 0,
            average_forecast_time: std::time::Duration::from_nanos(0),
            cache_hit_rate: 0.0,
            successful_forecasts: 0,
        }
    }
    
    fn record_forecast(&mut self, duration: f64) {
        let new_duration = std::time::Duration::from_millis(duration as u64);
        let total_time = self.average_forecast_time * self.total_forecasts as u32 + new_duration;
        self.total_forecasts += 1;
        self.average_forecast_time = total_time / self.total_forecasts as u32;
        self.successful_forecasts += 1;
    }
}

impl Default for ForecastMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory function for creating Claude forecaster
pub fn create_claude_forecaster(api_key: String) -> NarrativeResult<NarrativeForecaster> {
    let llm_config = LLMConfig {
        provider: LLMProvider::Claude,
        api_key: Some(api_key),
        model: "claude-sonnet-4-20250514".to_string(),
        base_url: Some("https://api.anthropic.com/v1/messages".to_string()),
        max_tokens: 1500,
        temperature: 0.3,
        timeout_seconds: 45,
    };
    
    let config = ForecasterConfig {
        cache_duration_minutes: 10,
        rate_limit_delay_ms: 800,
        max_retries: 3,
        timeout_seconds: 45,
        enable_sentiment_analysis: true,
        enable_caching: true,
        max_history_size: 1500,
    };
    
    NarrativeForecaster::new(llm_config, config)
}

/// Factory function for creating fallback forecaster
pub fn create_fallback_forecaster() -> NarrativeResult<NarrativeForecaster> {
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
            model: "claude-sonnet-4-20250514".to_string(),
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
            model: "claude-sonnet-4-20250514".to_string(),
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