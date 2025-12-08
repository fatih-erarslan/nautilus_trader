//! Advanced NLP sentiment analysis with transformer models

use crate::{config::{SentimentConfig, ModelPrecision}, error::{SentimentError, SentimentResult as SentimentErrorResult}, ComponentHealth};
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, Module};
// use candle_transformers::models::bert::{BertModel, BertConfig};
#[cfg(feature = "nlp")]
use tokenizers::{Tokenizer, Encoding};
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, error, instrument};
use anyhow::Result;
use rayon::prelude::*;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Advanced sentiment analyzer with transformer models
pub struct SentimentAnalyzer {
    config: Arc<SentimentConfig>,
    models: Arc<RwLock<HashMap<String, ModelWrapper>>>,
    #[cfg(feature = "nlp")]
    tokenizer: Arc<Tokenizer>,
    device: Device,
    cache: Arc<Mutex<SentimentCache>>,
    preprocessor: Arc<TextPreprocessor>,
    postprocessor: Arc<SentimentPostprocessor>,
    metrics: Arc<RwLock<SentimentMetrics>>,
    batch_processor: Arc<BatchProcessor>,
}

/// Model wrapper for different sentiment models
#[derive(Clone)]
pub struct ModelWrapper {
    pub model_name: String,
    pub precision: ModelPrecision,
    pub last_used: std::time::Instant,
}


/// Sentiment analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    pub text: String,
    pub sentiment: SentimentLabel,
    pub confidence: f64,
    pub scores: SentimentScores,
    pub emotions: Option<EmotionScores>,
    pub aspects: Option<Vec<AspectSentiment>>,
    pub language: Option<String>,
    pub processing_time: Duration,
    pub model_used: String,
    pub metadata: SentimentMetadata,
}

/// Sentiment label
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SentimentLabel {
    Positive,
    Negative,
    Neutral,
    Mixed,
    Unknown,
}

/// Sentiment scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScores {
    pub positive: f64,
    pub negative: f64,
    pub neutral: f64,
    pub compound: f64,
}

/// Emotion scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionScores {
    pub joy: f64,
    pub sadness: f64,
    pub anger: f64,
    pub fear: f64,
    pub surprise: f64,
    pub disgust: f64,
    pub trust: f64,
    pub anticipation: f64,
}

/// Aspect-based sentiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AspectSentiment {
    pub aspect: String,
    pub sentiment: SentimentLabel,
    pub confidence: f64,
    pub span: (usize, usize),
}

/// Sentiment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentMetadata {
    pub text_length: usize,
    pub token_count: usize,
    pub preprocessing_applied: Vec<String>,
    pub model_version: String,
    pub cache_hit: bool,
    pub batch_size: usize,
    pub gpu_used: bool,
}

/// Text preprocessor
pub struct TextPreprocessor {
    config: Arc<SentimentConfig>,
    stop_words: Arc<HashMap<String, Vec<String>>>,
    emoji_mappings: Arc<HashMap<String, String>>,
}

impl TextPreprocessor {
    pub fn new(config: Arc<SentimentConfig>) -> Self {
        let stop_words = Arc::new(Self::load_stop_words());
        let emoji_mappings = Arc::new(Self::load_emoji_mappings());
        
        Self {
            config,
            stop_words,
            emoji_mappings,
        }
    }

    pub fn preprocess(&self, text: &str) -> SentimentResult<String> {
        let mut processed = text.to_string();
        let mut applied_steps = Vec::new();
        
        // Remove URLs
        if self.config.preprocessing.remove_urls {
            processed = self.remove_urls(&processed);
            applied_steps.push("remove_urls".to_string());
        }
        
        // Remove mentions
        if self.config.preprocessing.remove_mentions {
            processed = self.remove_mentions(&processed);
            applied_steps.push("remove_mentions".to_string());
        }
        
        // Remove hashtags
        if self.config.preprocessing.remove_hashtags {
            processed = self.remove_hashtags(&processed);
            applied_steps.push("remove_hashtags".to_string());
        }
        
        // Handle emojis
        if self.config.preprocessing.remove_emojis {
            processed = self.remove_emojis(&processed);
            applied_steps.push("remove_emojis".to_string());
        } else {
            processed = self.convert_emojis(&processed);
            applied_steps.push("convert_emojis".to_string());
        }
        
        // Convert to lowercase
        if self.config.preprocessing.lowercase {
            processed = processed.to_lowercase();
            applied_steps.push("lowercase".to_string());
        }
        
        // Remove punctuation
        if self.config.preprocessing.remove_punctuation {
            processed = self.remove_punctuation(&processed);
            applied_steps.push("remove_punctuation".to_string());
        }
        
        // Remove stop words
        if self.config.preprocessing.remove_stopwords {
            processed = self.remove_stopwords(&processed);
            applied_steps.push("remove_stopwords".to_string());
        }
        
        // Normalize whitespace
        if self.config.preprocessing.normalize_whitespace {
            processed = self.normalize_whitespace(&processed);
            applied_steps.push("normalize_whitespace".to_string());
        }
        
        // Stemming
        if self.config.preprocessing.stem_words {
            processed = self.stem_words(&processed);
            applied_steps.push("stem_words".to_string());
        }
        
        // Lemmatization
        if self.config.preprocessing.lemmatize {
            processed = self.lemmatize(&processed);
            applied_steps.push("lemmatize".to_string());
        }
        
        Ok(processed)
    }

    fn remove_urls(&self, text: &str) -> String {
        let url_regex = regex::Regex::new(r"https?://[^\s]+").unwrap();
        url_regex.replace_all(text, "").to_string()
    }

    fn remove_mentions(&self, text: &str) -> String {
        let mention_regex = regex::Regex::new(r"@[^\s]+").unwrap();
        mention_regex.replace_all(text, "").to_string()
    }

    fn remove_hashtags(&self, text: &str) -> String {
        let hashtag_regex = regex::Regex::new(r"#[^\s]+").unwrap();
        hashtag_regex.replace_all(text, "").to_string()
    }

    fn remove_emojis(&self, text: &str) -> String {
        // Simple emoji removal - in practice, use a proper emoji regex
        let emoji_regex = regex::Regex::new(r"[\u{1f600}-\u{1f64f}]|[\u{1f300}-\u{1f5ff}]|[\u{1f680}-\u{1f6ff}]|[\u{1f1e0}-\u{1f1ff}]|[\u{2600}-\u{26ff}]|[\u{2700}-\u{27bf}]").unwrap();
        emoji_regex.replace_all(text, "").to_string()
    }

    fn convert_emojis(&self, text: &str) -> String {
        // Convert emojis to text descriptions
        let mut result = text.to_string();
        for (emoji, description) in self.emoji_mappings.iter() {
            result = result.replace(emoji, description);
        }
        result
    }

    fn remove_punctuation(&self, text: &str) -> String {
        let punct_regex = regex::Regex::new(r"[^\w\s]").unwrap();
        punct_regex.replace_all(text, "").to_string()
    }

    fn remove_stopwords(&self, text: &str) -> String {
        // Simple English stop words removal
        let words: Vec<&str> = text.split_whitespace().collect();
        let stop_words = self.stop_words.get("en").unwrap();
        
        words.into_iter()
            .filter(|word| !stop_words.contains(&word.to_lowercase()))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn normalize_whitespace(&self, text: &str) -> String {
        let whitespace_regex = regex::Regex::new(r"\s+").unwrap();
        whitespace_regex.replace_all(text.trim(), " ").to_string()
    }

    fn stem_words(&self, text: &str) -> String {
        // Simple stemming - in practice, use a proper stemmer
        text.split_whitespace()
            .map(|word| {
                if word.ends_with("ing") {
                    &word[..word.len()-3]
                } else if word.ends_with("ed") {
                    &word[..word.len()-2]
                } else if word.ends_with("s") && word.len() > 1 {
                    &word[..word.len()-1]
                } else {
                    word
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn lemmatize(&self, text: &str) -> String {
        // Placeholder for lemmatization - in practice, use a proper lemmatizer
        text.to_string()
    }

    fn load_stop_words() -> HashMap<String, Vec<String>> {
        let mut stop_words = HashMap::new();
        
        // English stop words
        let en_stop_words = vec![
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with"
        ].into_iter().map(|s| s.to_string()).collect();
        
        stop_words.insert("en".to_string(), en_stop_words);
        stop_words
    }

    fn load_emoji_mappings() -> HashMap<String, String> {
        let mut mappings = HashMap::new();
        
        // Common emoji mappings
        mappings.insert("😊".to_string(), " happy ".to_string());
        mappings.insert("😢".to_string(), " sad ".to_string());
        mappings.insert("😡".to_string(), " angry ".to_string());
        mappings.insert("😍".to_string(), " love ".to_string());
        mappings.insert("👍".to_string(), " good ".to_string());
        mappings.insert("👎".to_string(), " bad ".to_string());
        mappings.insert("🚀".to_string(), " rocket ".to_string());
        mappings.insert("📈".to_string(), " up ".to_string());
        mappings.insert("📉".to_string(), " down ".to_string());
        
        mappings
    }
}

/// Sentiment postprocessor
pub struct SentimentPostprocessor {
    config: Arc<SentimentConfig>,
}

impl SentimentPostprocessor {
    pub fn new(config: Arc<SentimentConfig>) -> Self {
        Self { config }
    }

    pub fn postprocess(&self, raw_scores: &[f64], text: &str) -> SentimentResult<SentimentResult> {
        let scores = self.normalize_scores(raw_scores)?;
        let sentiment = self.classify_sentiment(&scores)?;
        let confidence = self.calculate_confidence(&scores)?;
        
        Ok(SentimentResult {
            text: text.to_string(),
            sentiment,
            confidence,
            scores,
            emotions: None, // TODO: Implement emotion analysis
            aspects: None, // TODO: Implement aspect-based sentiment
            language: None, // TODO: Implement language detection
            processing_time: Duration::from_millis(0), // Will be set by caller
            model_used: "bert-base-uncased".to_string(),
            metadata: SentimentMetadata {
                text_length: text.len(),
                token_count: 0, // Will be set by caller
                preprocessing_applied: vec![],
                model_version: "1.0".to_string(),
                cache_hit: false,
                batch_size: 1,
                gpu_used: false,
            },
        })
    }

    fn normalize_scores(&self, raw_scores: &[f64]) -> SentimentResult<SentimentScores> {
        if raw_scores.len() < 3 {
            return Err(SentimentError::Inference("Insufficient scores".to_string()));
        }
        
        let sum: f64 = raw_scores.iter().sum();
        if sum == 0.0 {
            return Err(SentimentError::Inference("All scores are zero".to_string()));
        }
        
        let negative = raw_scores[0] / sum;
        let neutral = raw_scores[1] / sum;
        let positive = raw_scores[2] / sum;
        
        let compound = positive - negative;
        
        Ok(SentimentScores {
            positive,
            negative,
            neutral,
            compound,
        })
    }

    fn classify_sentiment(&self, scores: &SentimentScores) -> SentimentResult<SentimentLabel> {
        let threshold = self.config.sentiment_threshold as f64;
        
        if scores.compound >= threshold {
            Ok(SentimentLabel::Positive)
        } else if scores.compound <= -threshold {
            Ok(SentimentLabel::Negative)
        } else if scores.neutral > scores.positive && scores.neutral > scores.negative {
            Ok(SentimentLabel::Neutral)
        } else if (scores.positive - scores.negative).abs() < 0.1 {
            Ok(SentimentLabel::Mixed)
        } else {
            Ok(SentimentLabel::Unknown)
        }
    }

    fn calculate_confidence(&self, scores: &SentimentScores) -> SentimentResult<f64> {
        let max_score = scores.positive.max(scores.negative.max(scores.neutral));
        let min_score = scores.positive.min(scores.negative.min(scores.neutral));
        
        Ok(max_score - min_score)
    }
}

/// Batch processor for efficient processing
pub struct BatchProcessor {
    config: Arc<SentimentConfig>,
    batch_queue: Arc<Mutex<Vec<BatchItem>>>,
}

#[derive(Debug, Clone)]
pub struct BatchItem {
    pub id: String,
    pub text: String,
    pub callback: Option<tokio::sync::oneshot::Sender<SentimentResult>>,
}

impl BatchProcessor {
    pub fn new(config: Arc<SentimentConfig>) -> Self {
        Self {
            config,
            batch_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn add_to_batch(&self, item: BatchItem) -> Result<()> {
        let mut queue = self.batch_queue.lock().await;
        queue.push(item);
        
        if queue.len() >= self.config.batch_size {
            self.process_batch(&mut queue).await?;
        }
        
        Ok(())
    }

    async fn process_batch(&self, batch: &mut Vec<BatchItem>) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        
        debug!("Processing batch of {} items", batch.len());
        
        // Process all items in the batch
        let results: Vec<_> = batch.par_iter()
            .map(|item| {
                // Placeholder for actual batch processing
                SentimentResult {
                    text: item.text.clone(),
                    sentiment: SentimentLabel::Neutral,
                    confidence: 0.5,
                    scores: SentimentScores {
                        positive: 0.33,
                        negative: 0.33,
                        neutral: 0.34,
                        compound: 0.0,
                    },
                    emotions: None,
                    aspects: None,
                    language: None,
                    processing_time: Duration::from_millis(10),
                    model_used: "bert-base-uncased".to_string(),
                    metadata: SentimentMetadata {
                        text_length: item.text.len(),
                        token_count: 0,
                        preprocessing_applied: vec![],
                        model_version: "1.0".to_string(),
                        cache_hit: false,
                        batch_size: batch.len(),
                        gpu_used: false,
                    },
                }
            })
            .collect();
        
        // Send results back to callers
        for (item, result) in batch.iter().zip(results.iter()) {
            if let Some(callback) = &item.callback {
                let _ = callback.send(result.clone());
            }
        }
        
        batch.clear();
        Ok(())
    }
}

/// Sentiment cache for performance optimization
pub struct SentimentCache {
    cache: HashMap<String, (SentimentResult, Instant)>,
    max_size: usize,
    ttl: Duration,
}

impl SentimentCache {
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            ttl,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<SentimentResult> {
        if let Some((result, timestamp)) = self.cache.get(key) {
            if timestamp.elapsed() < self.ttl {
                return Some(result.clone());
            } else {
                self.cache.remove(key);
            }
        }
        None
    }

    pub fn put(&mut self, key: String, result: SentimentResult) {
        if self.cache.len() >= self.max_size {
            self.evict_old_entries();
        }
        
        self.cache.insert(key, (result, Instant::now()));
    }

    fn evict_old_entries(&mut self) {
        let now = Instant::now();
        let expired_keys: Vec<_> = self.cache.iter()
            .filter(|(_, (_, timestamp))| now.duration_since(*timestamp) > self.ttl)
            .map(|(key, _)| key.clone())
            .collect();
        
        for key in expired_keys {
            self.cache.remove(&key);
        }
        
        // If still too many entries, remove oldest
        if self.cache.len() >= self.max_size {
            let oldest_key = self.cache.iter()
                .min_by_key(|(_, (_, timestamp))| timestamp)
                .map(|(key, _)| key.clone());
            
            if let Some(key) = oldest_key {
                self.cache.remove(&key);
            }
        }
    }
}

/// Sentiment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentMetrics {
    pub total_analyzed: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_processing_time: Duration,
    pub gpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub batch_processing_count: u64,
    pub error_count: u64,
    pub model_switches: u64,
    pub last_reset: chrono::DateTime<chrono::Utc>,
}

impl Default for SentimentMetrics {
    fn default() -> Self {
        Self {
            total_analyzed: 0,
            cache_hits: 0,
            cache_misses: 0,
            average_processing_time: Duration::from_millis(0),
            gpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            batch_processing_count: 0,
            error_count: 0,
            model_switches: 0,
            last_reset: chrono::Utc::now(),
        }
    }
}


impl SentimentAnalyzer {
    /// Create a new sentiment analyzer
    pub async fn new(config: Arc<SentimentConfig>) -> Result<Self> {
        info!("Initializing sentiment analyzer with config: {:?}", config);
        
        // Initialize device
        let device = if config.enable_gpu {
            Device::new_cuda(config.gpu_device_id)?
        } else {
            Device::Cpu
        };
        
        // Load tokenizer
        #[cfg(feature = "nlp")]
        let tokenizer = Arc::new(Tokenizer::from_file(&config.tokenizer_path)?);
        
        // Initialize components
        let models = Arc::new(RwLock::new(HashMap::new()));
        let cache = Arc::new(Mutex::new(SentimentCache::new(
            config.cache_size,
            Duration::from_secs(3600), // 1 hour TTL
        )));
        let preprocessor = Arc::new(TextPreprocessor::new(config.clone()));
        let postprocessor = Arc::new(SentimentPostprocessor::new(config.clone()));
        let metrics = Arc::new(RwLock::new(SentimentMetrics::default()));
        let batch_processor = Arc::new(BatchProcessor::new(config.clone()));
        
        // Load models
        let mut models_map = models.write().await;
        for model_name in &config.language_models {
            let model = Self::load_model(&config.model_path, model_name, &device, config.precision.clone()).await?;
            models_map.insert(model_name.clone(), model);
        }
        drop(models_map);
        
        Ok(Self {
            config,
            models,
            #[cfg(feature = "nlp")]
            tokenizer,
            device,
            cache,
            preprocessor,
            postprocessor,
            metrics,
            batch_processor,
        })
    }

    /// Load a sentiment model
    async fn load_model(
        model_path: &str,
        model_name: &str,
        device: &Device,
        precision: ModelPrecision,
    ) -> Result<ModelWrapper> {
        info!("Loading model: {}", model_name);
        
        // Simplified model loading - in a real implementation, this would load actual model weights
        Ok(ModelWrapper {
            model_name: model_name.to_string(),
            precision,
            last_used: Instant::now(),
        })
    }

    /// Start the sentiment analyzer
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting sentiment analyzer");
        
        // Start batch processing task
        let batch_processor = Arc::clone(&self.batch_processor);
        let config = Arc::clone(&self.config);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            
            loop {
                interval.tick().await;
                
                // Process any pending batches
                let mut queue = batch_processor.batch_queue.lock().await;
                if !queue.is_empty() {
                    if let Err(e) = batch_processor.process_batch(&mut queue).await {
                        error!("Error processing batch: {}", e);
                    }
                }
            }
        });
        
        info!("Sentiment analyzer started successfully");
        Ok(())
    }

    /// Stop the sentiment analyzer
    #[instrument(skip(self))]
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping sentiment analyzer");
        
        // Process any remaining batches
        let mut queue = self.batch_processor.batch_queue.lock().await;
        if !queue.is_empty() {
            if let Err(e) = self.batch_processor.process_batch(&mut queue).await {
                error!("Error processing final batch: {}", e);
            }
        }
        
        info!("Sentiment analyzer stopped successfully");
        Ok(())
    }

    /// Analyze sentiment of text
    #[instrument(skip(self, text))]
    pub async fn analyze(&self, text: &str) -> SentimentErrorResult<SentimentResult> {
        let start_time = Instant::now();
        
        // Check cache first
        let cache_key = format!("{:x}", md5::compute(text));
        {
            let mut cache = self.cache.lock().await;
            if let Some(mut result) = cache.get(&cache_key) {
                result.metadata.cache_hit = true;
                
                // Update metrics
                {
                    let mut metrics = self.metrics.write().await;
                    metrics.cache_hits += 1;
                }
                
                return Ok(result);
            }
        }
        
        // Preprocess text
        let processed_text = self.preprocessor.preprocess(text)?;
        
        // Tokenize
        let encoding = self.tokenizer.encode(processed_text.clone(), true)
            .map_err(|e| SentimentError::Tokenization(e.to_string()))?;
        
        // Check sequence length
        if encoding.len() > self.config.max_sequence_length {
            return Err(SentimentError::TextTooLong(format!(
                "Text length {} exceeds maximum {}",
                encoding.len(),
                self.config.max_sequence_length
            )));
        }
        
        // Get model
        let model_name = &self.config.language_models[0]; // Use first model by default
        let model = {
            let models = self.models.read().await;
            models.get(model_name).cloned()
                .ok_or_else(|| SentimentError::ModelLoading(format!("Model {} not found", model_name)))?
        };
        
        // Perform inference
        let raw_scores = self.perform_inference(&model, &encoding).await?;
        
        // Postprocess results
        let mut result = self.postprocessor.postprocess(&raw_scores, text)?;
        result.processing_time = start_time.elapsed();
        result.metadata.token_count = encoding.len();
        result.metadata.gpu_used = self.config.enable_gpu;
        result.model_used = model_name.clone();
        
        // Cache result
        {
            let mut cache = self.cache.lock().await;
            cache.put(cache_key, result.clone());
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_analyzed += 1;
            metrics.cache_misses += 1;
            metrics.average_processing_time = Duration::from_millis(
                ((metrics.average_processing_time.as_millis() as u64 * (metrics.total_analyzed - 1)) + 
                 result.processing_time.as_millis() as u64) / metrics.total_analyzed
            );
        }
        
        Ok(result)
    }

    /// Perform inference with the model
    async fn perform_inference(&self, model: &ModelWrapper, encoding: &Encoding) -> SentimentResult<Vec<f64>> {
        // Simplified inference - in a real implementation, this would use actual model inference
        let token_count = encoding.len();
        
        // Mock sentiment scores based on token count and model name
        let scores = if model.model_name.contains("positive") {
            vec![0.1, 0.2, 0.7] // negative, neutral, positive
        } else if token_count > 50 {
            vec![0.3, 0.4, 0.3] // neutral for long text
        } else {
            vec![0.2, 0.3, 0.5] // slightly positive for short text
        };
        
        Ok(scores)
    }

    /// Analyze sentiment in batch
    pub async fn analyze_batch(&self, texts: &[String]) -> SentimentErrorResult<Vec<SentimentResult>> {
        let mut results = Vec::new();
        
        for text in texts {
            let result = self.analyze(text).await?;
            results.push(result);
        }
        
        Ok(results)
    }

    /// Health check
    pub async fn health_check(&self) -> Result<ComponentHealth> {
        // Check if models are loaded
        let models = self.models.read().await;
        if models.is_empty() {
            return Ok(ComponentHealth::Unhealthy);
        }
        
        // Check device availability
        if self.config.enable_gpu {
            match self.device {
                Device::Cuda(_) => Ok(ComponentHealth::Healthy),
                _ => Ok(ComponentHealth::Degraded),
            }
        } else {
            Ok(ComponentHealth::Healthy)
        }
    }

    /// Get metrics
    pub async fn get_metrics(&self) -> SentimentMetrics {
        self.metrics.read().await.clone()
    }

    /// Reset analyzer
    pub async fn reset(&self) -> Result<()> {
        info!("Resetting sentiment analyzer");
        
        // Clear cache
        {
            let mut cache = self.cache.lock().await;
            cache.cache.clear();
        }
        
        // Reset metrics
        {
            let mut metrics = self.metrics.write().await;
            *metrics = SentimentMetrics::default();
        }
        
        // Clear batch queue
        {
            let mut queue = self.batch_processor.batch_queue.lock().await;
            queue.clear();
        }
        
        info!("Sentiment analyzer reset successfully");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_text_preprocessor() {
        let config = Arc::new(SentimentConfig::default());
        let preprocessor = TextPreprocessor::new(config);
        
        let text = "Hello @user! Check out https://example.com #awesome 😊";
        let result = preprocessor.preprocess(text).unwrap();
        
        assert!(!result.contains("@user"));
        assert!(!result.contains("https://example.com"));
        assert!(!result.contains("#awesome"));
    }

    #[test]
    async fn test_sentiment_cache() {
        let mut cache = SentimentCache::new(10, Duration::from_secs(60));
        
        let result = SentimentResult {
            text: "test".to_string(),
            sentiment: SentimentLabel::Positive,
            confidence: 0.8,
            scores: SentimentScores {
                positive: 0.8,
                negative: 0.1,
                neutral: 0.1,
                compound: 0.7,
            },
            emotions: None,
            aspects: None,
            language: None,
            processing_time: Duration::from_millis(10),
            model_used: "test".to_string(),
            metadata: SentimentMetadata {
                text_length: 4,
                token_count: 1,
                preprocessing_applied: vec![],
                model_version: "1.0".to_string(),
                cache_hit: false,
                batch_size: 1,
                gpu_used: false,
            },
        };
        
        cache.put("test_key".to_string(), result.clone());
        let cached_result = cache.get("test_key").unwrap();
        
        assert_eq!(cached_result.sentiment, SentimentLabel::Positive);
        assert_eq!(cached_result.confidence, 0.8);
    }

    #[test]
    async fn test_sentiment_postprocessor() {
        let config = Arc::new(SentimentConfig::default());
        let postprocessor = SentimentPostprocessor::new(config);
        
        let raw_scores = vec![0.1, 0.2, 0.7]; // negative, neutral, positive
        let result = postprocessor.postprocess(&raw_scores, "test text").unwrap();
        
        assert_eq!(result.sentiment, SentimentLabel::Positive);
        assert!(result.confidence > 0.0);
        assert!(result.scores.positive > result.scores.negative);
    }
}