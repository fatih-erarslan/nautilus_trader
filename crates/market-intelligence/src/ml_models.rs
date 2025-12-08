use crate::*;
use ndarray::{Array1, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressor;
use smartcore::model_selection::train_test_split;
use smartcore::metrics::mean_squared_error;

pub struct MLPipeline {
    models: HashMap<String, Box<dyn PredictiveModel + Send + Sync>>,
    feature_extractors: HashMap<String, Box<dyn FeatureExtractor + Send + Sync>>,
    model_cache: Arc<RwLock<HashMap<String, CachedPrediction>>>,
    training_data: Arc<RwLock<TrainingDataset>>,
}

#[derive(Clone)]
struct CachedPrediction {
    prediction: f64,
    confidence: f64,
    timestamp: DateTime<Utc>,
}

struct TrainingDataset {
    features: Vec<Vec<f64>>,
    targets: Vec<f64>,
    symbols: Vec<String>,
    timestamps: Vec<DateTime<Utc>>,
}

trait PredictiveModel {
    fn predict(&self, features: &[f64]) -> Result<f64, IntelligenceError>;
    fn predict_with_confidence(&self, features: &[f64]) -> Result<(f64, f64), IntelligenceError>;
    fn retrain(&mut self, features: &[Vec<f64>], targets: &[f64]) -> Result<(), IntelligenceError>;
    fn get_feature_importance(&self) -> Option<Vec<f64>>;
}

trait FeatureExtractor {
    fn extract_features(
        &self,
        symbol: &str,
        trend: &TrendScore,
        sentiment: Option<&SentimentScore>,
    ) -> Result<Vec<f64>, IntelligenceError>;
    fn get_feature_names(&self) -> Vec<String>;
}

impl MLPipeline {
    pub fn new() -> Self {
        let mut models: HashMap<String, Box<dyn PredictiveModel + Send + Sync>> = HashMap::new();
        let mut extractors: HashMap<String, Box<dyn FeatureExtractor + Send + Sync>> = HashMap::new();
        
        // Initialize models
        models.insert("profitability".to_string(), Box::new(ProfitabilityModel::new()));
        models.insert("volatility".to_string(), Box::new(VolatilityModel::new()));
        models.insert("trend_continuation".to_string(), Box::new(TrendContinuationModel::new()));
        
        // Initialize feature extractors
        extractors.insert("technical".to_string(), Box::new(TechnicalFeatureExtractor::new()));
        extractors.insert("sentiment".to_string(), Box::new(SentimentFeatureExtractor::new()));
        extractors.insert("market".to_string(), Box::new(MarketFeatureExtractor::new()));
        
        Self {
            models,
            feature_extractors: extractors,
            model_cache: Arc::new(RwLock::new(HashMap::new())),
            training_data: Arc::new(RwLock::new(TrainingDataset {
                features: vec![],
                targets: vec![],
                symbols: vec![],
                timestamps: vec![],
            })),
        }
    }
    
    pub async fn predict_profitability(
        &self,
        symbol: &str,
        trend: &TrendScore,
        sentiment: Option<&SentimentScore>,
    ) -> Result<f64, IntelligenceError> {
        // Check cache first
        if let Some(cached) = self.get_cached_prediction(symbol).await {
            return Ok(cached.prediction);
        }
        
        // Extract features
        let features = self.extract_all_features(symbol, trend, sentiment).await?;
        
        // Get ensemble prediction
        let prediction = self.predict_ensemble(&features).await?;
        
        // Cache the result
        self.cache_prediction(symbol, prediction, 0.8).await;
        
        Ok(prediction)
    }
    
    async fn extract_all_features(
        &self,
        symbol: &str,
        trend: &TrendScore,
        sentiment: Option<&SentimentScore>,
    ) -> Result<Vec<f64>, IntelligenceError> {
        let mut all_features = vec![];
        
        // Extract technical features
        if let Some(extractor) = self.feature_extractors.get("technical") {
            let tech_features = extractor.extract_features(symbol, trend, sentiment)?;
            all_features.extend(tech_features);
        }
        
        // Extract sentiment features if available
        if sentiment.is_some() {
            if let Some(extractor) = self.feature_extractors.get("sentiment") {
                let sent_features = extractor.extract_features(symbol, trend, sentiment)?;
                all_features.extend(sent_features);
            }
        }
        
        // Extract market features
        if let Some(extractor) = self.feature_extractors.get("market") {
            let market_features = extractor.extract_features(symbol, trend, sentiment)?;
            all_features.extend(market_features);
        }
        
        Ok(all_features)
    }
    
    async fn predict_ensemble(&self, features: &[f64]) -> Result<f64, IntelligenceError> {
        let mut predictions = vec![];
        let mut weights = vec![];
        
        // Get predictions from all models
        for (model_name, model) in &self.models {
            if let Ok((prediction, confidence)) = model.predict_with_confidence(features) {
                predictions.push(prediction);
                weights.push(confidence);
            }
        }
        
        if predictions.is_empty() {
            return Err(IntelligenceError::ModelError("No models produced predictions".to_string()));
        }
        
        // Weighted average ensemble
        let total_weight: f64 = weights.iter().sum();
        if total_weight > 0.0 {
            let weighted_sum: f64 = predictions.iter()
                .zip(weights.iter())
                .map(|(pred, weight)| pred * weight)
                .sum();
            Ok(weighted_sum / total_weight)
        } else {
            // Simple average if no weights
            Ok(predictions.iter().sum::<f64>() / predictions.len() as f64)
        }
    }
    
    pub async fn add_training_data(
        &self,
        symbol: &str,
        trend: &TrendScore,
        sentiment: Option<&SentimentScore>,
        actual_return: f64,
    ) -> Result<(), IntelligenceError> {
        let features = self.extract_all_features(symbol, trend, sentiment).await?;
        
        let mut training_data = self.training_data.write().await;
        training_data.features.push(features);
        training_data.targets.push(actual_return);
        training_data.symbols.push(symbol.to_string());
        training_data.timestamps.push(Utc::now());
        
        // Limit training data size
        const MAX_TRAINING_SIZE: usize = 10000;
        if training_data.features.len() > MAX_TRAINING_SIZE {
            let keep_from = training_data.features.len() - MAX_TRAINING_SIZE;
            training_data.features.drain(0..keep_from);
            training_data.targets.drain(0..keep_from);
            training_data.symbols.drain(0..keep_from);
            training_data.timestamps.drain(0..keep_from);
        }
        
        Ok(())
    }
    
    pub async fn retrain_models(&mut self) -> Result<(), IntelligenceError> {
        let training_data = self.training_data.read().await;
        
        if training_data.features.len() < 100 {
            return Err(IntelligenceError::InsufficientData("Not enough training data".to_string()));
        }
        
        // Retrain each model
        for (model_name, model) in &mut self.models {
            if let Err(e) = model.retrain(&training_data.features, &training_data.targets) {
                log::warn!("Failed to retrain model {}: {}", model_name, e);
            }
        }
        
        Ok(())
    }
    
    async fn get_cached_prediction(&self, symbol: &str) -> Option<CachedPrediction> {
        let cache = self.model_cache.read().await;
        cache.get(symbol).and_then(|cached| {
            if (Utc::now() - cached.timestamp).num_minutes() < 15 {
                Some(cached.clone())
            } else {
                None
            }
        })
    }
    
    async fn cache_prediction(&self, symbol: &str, prediction: f64, confidence: f64) {
        let mut cache = self.model_cache.write().await;
        cache.insert(symbol.to_string(), CachedPrediction {
            prediction,
            confidence,
            timestamp: Utc::now(),
        });
    }
}

// Specific model implementations
struct ProfitabilityModel {
    model: Option<RandomForestRegressor<f64>>,
    feature_importance: Vec<f64>,
}

impl ProfitabilityModel {
    fn new() -> Self {
        Self {
            model: None,
            feature_importance: vec![],
        }
    }
}

impl PredictiveModel for ProfitabilityModel {
    fn predict(&self, features: &[f64]) -> Result<f64, IntelligenceError> {
        if let Some(model) = &self.model {
            let feature_matrix = DenseMatrix::from_2d_array(&[features]);
            let predictions = model.predict(&feature_matrix)
                .map_err(|e| IntelligenceError::ModelError(format!("Prediction failed: {}", e)))?;
            
            Ok(predictions[0])
        } else {
            // Fallback: simple heuristic
            Ok(self.heuristic_prediction(features))
        }
    }
    
    fn predict_with_confidence(&self, features: &[f64]) -> Result<(f64, f64), IntelligenceError> {
        let prediction = self.predict(features)?;
        
        // Simple confidence based on feature quality
        let confidence = if features.iter().any(|&f| f.is_nan() || f.is_infinite()) {
            0.3 // Low confidence for bad features
        } else {
            0.7 // Default confidence
        };
        
        Ok((prediction, confidence))
    }
    
    fn retrain(&mut self, features: &[Vec<f64>], targets: &[f64]) -> Result<(), IntelligenceError> {
        if features.is_empty() || targets.is_empty() {
            return Err(IntelligenceError::InsufficientData("Empty training data".to_string()));
        }
        
        // Convert to DenseMatrix format
        let feature_matrix = DenseMatrix::from_2d_vec(&features);
        let target_vector = targets.to_vec();
        
        // Train Random Forest
        let model = RandomForestRegressor::fit(&feature_matrix, &target_vector, Default::default())
            .map_err(|e| IntelligenceError::ModelError(format!("Training failed: {}", e)))?;
        
        self.model = Some(model);
        
        // Calculate feature importance (simplified)
        self.feature_importance = vec![1.0 / features[0].len() as f64; features[0].len()];
        
        Ok(())
    }
    
    fn get_feature_importance(&self) -> Option<Vec<f64>> {
        if self.feature_importance.is_empty() {
            None
        } else {
            Some(self.feature_importance.clone())
        }
    }
}

impl ProfitabilityModel {
    fn heuristic_prediction(&self, features: &[f64]) -> f64 {
        // Simple heuristic based on first few features
        if features.len() < 5 {
            return 0.5;
        }
        
        let trend_strength = features[0];
        let momentum = features[1];
        let volume_confirmation = features[2];
        let volatility = features[3];
        let sentiment = features.get(4).copied().unwrap_or(0.5);
        
        // Simple weighted combination
        let prediction = (trend_strength * 0.3 + 
                         momentum * 0.25 + 
                         volume_confirmation * 0.2 + 
                         (1.0 - volatility) * 0.1 + 
                         sentiment * 0.15).max(0.0).min(1.0);
        
        prediction
    }
}

struct VolatilityModel {
    model: Option<LinearRegression<f64>>,
}

impl VolatilityModel {
    fn new() -> Self {
        Self { model: None }
    }
}

impl PredictiveModel for VolatilityModel {
    fn predict(&self, features: &[f64]) -> Result<f64, IntelligenceError> {
        if let Some(model) = &self.model {
            let feature_matrix = DenseMatrix::from_2d_array(&[features]);
            let predictions = model.predict(&feature_matrix)
                .map_err(|e| IntelligenceError::ModelError(format!("Prediction failed: {}", e)))?;
            
            Ok(predictions[0].max(0.0).min(1.0))
        } else {
            // Fallback heuristic
            Ok(features.get(3).copied().unwrap_or(0.5))
        }
    }
    
    fn predict_with_confidence(&self, features: &[f64]) -> Result<(f64, f64), IntelligenceError> {
        let prediction = self.predict(features)?;
        Ok((prediction, 0.6))
    }
    
    fn retrain(&mut self, features: &[Vec<f64>], targets: &[f64]) -> Result<(), IntelligenceError> {
        let feature_matrix = DenseMatrix::from_2d_vec(&features);
        let target_vector = targets.to_vec();
        
        let model = LinearRegression::fit(&feature_matrix, &target_vector, Default::default())
            .map_err(|e| IntelligenceError::ModelError(format!("Training failed: {}", e)))?;
        
        self.model = Some(model);
        Ok(())
    }
    
    fn get_feature_importance(&self) -> Option<Vec<f64>> {
        None // Linear regression doesn't provide feature importance easily
    }
}

struct TrendContinuationModel {
    model: Option<DecisionTreeRegressor<f64>>,
}

impl TrendContinuationModel {
    fn new() -> Self {
        Self { model: None }
    }
}

impl PredictiveModel for TrendContinuationModel {
    fn predict(&self, features: &[f64]) -> Result<f64, IntelligenceError> {
        if let Some(model) = &self.model {
            let feature_matrix = DenseMatrix::from_2d_array(&[features]);
            let predictions = model.predict(&feature_matrix)
                .map_err(|e| IntelligenceError::ModelError(format!("Prediction failed: {}", e)))?;
            
            Ok(predictions[0].max(0.0).min(1.0))
        } else {
            // Trend continuation heuristic
            let trend_strength = features.get(0).copied().unwrap_or(0.0);
            let momentum = features.get(1).copied().unwrap_or(0.0);
            
            let continuation_prob = if trend_strength.signum() == momentum.signum() {
                0.6 + (trend_strength.abs() * 0.3)
            } else {
                0.4 - (trend_strength.abs() * 0.2)
            };
            
            Ok(continuation_prob.max(0.0).min(1.0))
        }
    }
    
    fn predict_with_confidence(&self, features: &[f64]) -> Result<(f64, f64), IntelligenceError> {
        let prediction = self.predict(features)?;
        let confidence = 0.5 + (prediction - 0.5).abs(); // Higher confidence for extreme predictions
        Ok((prediction, confidence))
    }
    
    fn retrain(&mut self, features: &[Vec<f64>], targets: &[f64]) -> Result<(), IntelligenceError> {
        let feature_matrix = DenseMatrix::from_2d_vec(&features);
        let target_vector = targets.to_vec();
        
        let model = DecisionTreeRegressor::fit(&feature_matrix, &target_vector, Default::default())
            .map_err(|e| IntelligenceError::ModelError(format!("Training failed: {}", e)))?;
        
        self.model = Some(model);
        Ok(())
    }
    
    fn get_feature_importance(&self) -> Option<Vec<f64>> {
        None // Would need to implement feature importance calculation
    }
}

// Feature extractors
struct TechnicalFeatureExtractor;

impl TechnicalFeatureExtractor {
    fn new() -> Self {
        Self
    }
}

impl FeatureExtractor for TechnicalFeatureExtractor {
    fn extract_features(
        &self,
        _symbol: &str,
        trend: &TrendScore,
        _sentiment: Option<&SentimentScore>,
    ) -> Result<Vec<f64>, IntelligenceError> {
        let mut features = vec![];
        
        // Basic trend features
        features.push(trend.trend_strength);
        features.push(trend.momentum_score);
        features.push(trend.volume_confirmation);
        features.push(trend.volatility);
        features.push(trend.confidence);
        features.push(trend.current_price);
        
        // Support/resistance features
        features.push(trend.support_levels.len() as f64);
        features.push(trend.resistance_levels.len() as f64);
        
        // Price position relative to support/resistance
        if !trend.support_levels.is_empty() {
            let nearest_support = trend.support_levels.iter()
                .min_by(|&a, &b| (trend.current_price - a).abs().partial_cmp(&(trend.current_price - b).abs()).unwrap())
                .unwrap();
            features.push((trend.current_price - nearest_support) / trend.current_price);
        } else {
            features.push(0.0);
        }
        
        if !trend.resistance_levels.is_empty() {
            let nearest_resistance = trend.resistance_levels.iter()
                .min_by(|&a, &b| (trend.current_price - a).abs().partial_cmp(&(trend.current_price - b).abs()).unwrap())
                .unwrap();
            features.push((nearest_resistance - trend.current_price) / trend.current_price);
        } else {
            features.push(0.0);
        }
        
        Ok(features)
    }
    
    fn get_feature_names(&self) -> Vec<String> {
        vec![
            "trend_strength".to_string(),
            "momentum_score".to_string(),
            "volume_confirmation".to_string(),
            "volatility".to_string(),
            "confidence".to_string(),
            "current_price".to_string(),
            "support_count".to_string(),
            "resistance_count".to_string(),
            "support_distance".to_string(),
            "resistance_distance".to_string(),
        ]
    }
}

struct SentimentFeatureExtractor;

impl SentimentFeatureExtractor {
    fn new() -> Self {
        Self
    }
}

impl FeatureExtractor for SentimentFeatureExtractor {
    fn extract_features(
        &self,
        _symbol: &str,
        _trend: &TrendScore,
        sentiment: Option<&SentimentScore>,
    ) -> Result<Vec<f64>, IntelligenceError> {
        let mut features = vec![];
        
        if let Some(sentiment) = sentiment {
            features.push(sentiment.overall_score);
            features.push(sentiment.confidence);
            features.push(sentiment.sources_count as f64);
            
            // Social sentiment features
            features.push(sentiment.social_sentiment.twitter_bullish_ratio);
            features.push(sentiment.social_sentiment.reddit_sentiment);
            features.push(sentiment.social_sentiment.viral_score);
            features.push(sentiment.social_sentiment.influencer_mentions as f64);
            
            // News sentiment features
            features.push(sentiment.news_sentiment.headline_sentiment);
            features.push(sentiment.news_sentiment.article_sentiment);
            features.push(sentiment.news_sentiment.news_volume as f64);
            features.push(if sentiment.news_sentiment.major_outlet_coverage { 1.0 } else { 0.0 });
            
            // Whale activity features
            features.push(sentiment.whale_activity.accumulation_score);
            features.push(sentiment.whale_activity.smart_money_confidence);
            features.push(sentiment.whale_activity.large_transactions as f64);
        } else {
            // Fill with neutral values if no sentiment data
            features.extend(vec![0.5; 13]);
        }
        
        Ok(features)
    }
    
    fn get_feature_names(&self) -> Vec<String> {
        vec![
            "overall_sentiment".to_string(),
            "sentiment_confidence".to_string(),
            "sources_count".to_string(),
            "twitter_bullish".to_string(),
            "reddit_sentiment".to_string(),
            "viral_score".to_string(),
            "influencer_mentions".to_string(),
            "headline_sentiment".to_string(),
            "article_sentiment".to_string(),
            "news_volume".to_string(),
            "major_outlet_coverage".to_string(),
            "whale_accumulation".to_string(),
            "smart_money_confidence".to_string(),
            "large_transactions".to_string(),
        ]
    }
}

struct MarketFeatureExtractor;

impl MarketFeatureExtractor {
    fn new() -> Self {
        Self
    }
}

impl FeatureExtractor for MarketFeatureExtractor {
    fn extract_features(
        &self,
        symbol: &str,
        _trend: &TrendScore,
        _sentiment: Option<&SentimentScore>,
    ) -> Result<Vec<f64>, IntelligenceError> {
        let mut features = vec![];
        
        // Market cap tier (simplified)
        let market_cap_tier = if symbol.contains("BTC") || symbol.contains("ETH") {
            1.0 // Large cap
        } else if symbol.contains("BNB") || symbol.contains("ADA") || symbol.contains("SOL") {
            0.7 // Mid cap
        } else {
            0.3 // Small cap
        };
        features.push(market_cap_tier);
        
        // Time-based features
        let now = Utc::now();
        let hour = now.hour() as f64 / 24.0;
        let day_of_week = now.weekday().num_days_from_monday() as f64 / 7.0;
        features.push(hour);
        features.push(day_of_week);
        
        // Seasonal features (simplified)
        let day_of_year = now.ordinal() as f64 / 365.0;
        features.push((day_of_year * 2.0 * std::f64::consts::PI).sin()); // Seasonal cycle
        features.push((day_of_year * 2.0 * std::f64::consts::PI).cos());
        
        Ok(features)
    }
    
    fn get_feature_names(&self) -> Vec<String> {
        vec![
            "market_cap_tier".to_string(),
            "hour_of_day".to_string(),
            "day_of_week".to_string(),
            "seasonal_sin".to_string(),
            "seasonal_cos".to_string(),
        ]
    }
}