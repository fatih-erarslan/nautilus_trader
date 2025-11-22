//! Real-time market regime detection system
//!
//! This module implements advanced market regime detection using multiple algorithms:
//! - Hidden Markov Models for regime classification
//! - Volatility clustering analysis
//! - Correlation regime detection
//! - Machine learning-based regime prediction

use std::sync::Arc;
use std::collections::VecDeque;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::config::MarketReadinessConfig;
use crate::types::ValidationResult;
use crate::{MarketRegime, ValidationStatus};
use crate::error::MarketReadinessError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeState {
    pub current_regime: MarketRegime,
    pub confidence: f64,
    pub transition_probability: f64,
    pub time_in_regime: chrono::Duration,
    pub regime_history: Vec<RegimeTransition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeTransition {
    pub from_regime: MarketRegime,
    pub to_regime: MarketRegime,
    pub timestamp: DateTime<Utc>,
    pub confidence: f64,
    pub trigger_factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketIndicators {
    pub volatility: f64,
    pub volume: f64,
    pub price_change: f64,
    pub correlation: f64,
    pub liquidity: f64,
    pub momentum: f64,
    pub vix_level: f64,
    pub news_sentiment: f64,
}

#[derive(Debug)]
pub struct RegimeDetector {
    config: Arc<MarketReadinessConfig>,
    current_state: Arc<RwLock<RegimeState>>,
    hmm_model: Arc<RwLock<HiddenMarkovModel>>,
    volatility_analyzer: Arc<VolatilityAnalyzer>,
    correlation_analyzer: Arc<CorrelationAnalyzer>,
    ml_predictor: Arc<MLPredictor>,
    indicator_buffer: Arc<RwLock<VecDeque<MarketIndicators>>>,
}

impl RegimeDetector {
    pub async fn new(config: Arc<MarketReadinessConfig>) -> Result<Self> {
        let hmm_model = Arc::new(RwLock::new(HiddenMarkovModel::new()));
        let volatility_analyzer = Arc::new(VolatilityAnalyzer::new());
        let correlation_analyzer = Arc::new(CorrelationAnalyzer::new());
        let ml_predictor = Arc::new(MLPredictor::new());
        
        let current_state = Arc::new(RwLock::new(RegimeState {
            current_regime: MarketRegime::Ranging,
            confidence: 0.5,
            transition_probability: 0.0,
            time_in_regime: chrono::Duration::zero(),
            regime_history: Vec::new(),
        }));
        
        Ok(Self {
            config,
            current_state,
            hmm_model,
            volatility_analyzer,
            correlation_analyzer,
            ml_predictor,
            indicator_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing regime detector...");
        
        // Initialize HMM model
        self.hmm_model.write().await.initialize().await?;
        
        // Initialize volatility analyzer
        self.volatility_analyzer.initialize().await?;
        
        // Initialize correlation analyzer
        self.correlation_analyzer.initialize().await?;
        
        // Initialize ML predictor
        self.ml_predictor.initialize().await?;
        
        // Start regime detection loop
        self.start_regime_detection().await?;
        
        info!("Regime detector initialized successfully");
        Ok(())
    }

    pub async fn detect_regime(&self) -> Result<ValidationResult> {
        let start_time = std::time::Instant::now();
        
        // Get current market indicators
        let indicators = self.collect_market_indicators().await?;
        
        // Update indicator buffer
        {
            let mut buffer = self.indicator_buffer.write().await;
            buffer.push_back(indicators.clone());
            if buffer.len() > 1000 {
                buffer.pop_front();
            }
        }
        
        // Run regime detection algorithms
        let hmm_result = self.hmm_model.read().await.detect_regime(&indicators).await?;
        let volatility_result = self.volatility_analyzer.analyze_regime(&indicators).await?;
        let correlation_result = self.correlation_analyzer.analyze_regime(&indicators).await?;
        let ml_result = self.ml_predictor.predict_regime(&indicators).await?;
        
        // Combine results using ensemble approach
        let regime_prediction = self.ensemble_prediction(
            &hmm_result,
            &volatility_result,
            &correlation_result,
            &ml_result,
        ).await?;
        
        // Update current state
        let mut current_state = self.current_state.write().await;
        let previous_regime = current_state.current_regime.clone();
        
        // Check for regime transition
        if regime_prediction.regime != previous_regime {
            let transition = RegimeTransition {
                from_regime: previous_regime,
                to_regime: regime_prediction.regime.clone(),
                timestamp: Utc::now(),
                confidence: regime_prediction.confidence,
                trigger_factors: regime_prediction.trigger_factors.clone(),
            };
            
            current_state.regime_history.push(transition);
            current_state.current_regime = regime_prediction.regime.clone();
            current_state.time_in_regime = chrono::Duration::zero();
            
            info!("Regime transition detected: {:?} -> {:?}", previous_regime, regime_prediction.regime);
        }
        
        current_state.confidence = regime_prediction.confidence;
        current_state.transition_probability = regime_prediction.transition_probability;
        
        let duration = start_time.elapsed().as_millis() as u64;
        
        // Validate regime detection quality
        let validation_result = if regime_prediction.confidence > 0.8 {
            ValidationResult {
                status: ValidationStatus::Passed,
                message: format!("Regime detection successful: {:?}", regime_prediction.regime),
                details: Some(serde_json::json!({
                    "regime": regime_prediction.regime,
                    "confidence": regime_prediction.confidence,
                    "transition_probability": regime_prediction.transition_probability,
                    "time_in_regime": current_state.time_in_regime.num_seconds(),
                    "indicators": indicators,
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: regime_prediction.confidence,
            }
        } else if regime_prediction.confidence > 0.6 {
            ValidationResult {
                status: ValidationStatus::Warning,
                message: format!("Regime detection uncertain: {:?}", regime_prediction.regime),
                details: Some(serde_json::json!({
                    "regime": regime_prediction.regime,
                    "confidence": regime_prediction.confidence,
                    "warning": "Low confidence in regime detection",
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: regime_prediction.confidence,
            }
        } else {
            ValidationResult {
                status: ValidationStatus::Failed,
                message: "Regime detection failed due to low confidence".to_string(),
                details: Some(serde_json::json!({
                    "confidence": regime_prediction.confidence,
                    "threshold": 0.6,
                })),
                timestamp: Utc::now(),
                duration_ms: duration,
                confidence: regime_prediction.confidence,
            }
        };
        
        Ok(validation_result)
    }

    pub async fn get_current_regime(&self) -> Result<MarketRegime> {
        let state = self.current_state.read().await;
        Ok(state.current_regime.clone())
    }

    async fn collect_market_indicators(&self) -> Result<MarketIndicators> {
        // Collect real-time market indicators
        // This would integrate with actual market data feeds
        Ok(MarketIndicators {
            volatility: self.calculate_volatility().await?,
            volume: self.calculate_volume().await?,
            price_change: self.calculate_price_change().await?,
            correlation: self.calculate_correlation().await?,
            liquidity: self.calculate_liquidity().await?,
            momentum: self.calculate_momentum().await?,
            vix_level: self.get_vix_level().await?,
            news_sentiment: self.get_news_sentiment().await?,
        })
    }

    async fn ensemble_prediction(
        &self,
        hmm: &RegimePrediction,
        volatility: &RegimePrediction,
        correlation: &RegimePrediction,
        ml: &RegimePrediction,
    ) -> Result<RegimePrediction> {
        // Weighted ensemble of predictions
        let weights = [0.3, 0.25, 0.25, 0.2]; // HMM, Volatility, Correlation, ML
        let predictions = [hmm, volatility, correlation, ml];
        
        // Calculate weighted confidence for each regime
        let mut regime_scores = std::collections::HashMap::new();
        
        for (i, prediction) in predictions.iter().enumerate() {
            let score = regime_scores.entry(prediction.regime.clone())
                .or_insert(0.0);
            *score += weights[i] * prediction.confidence;
        }
        
        // Find the regime with highest weighted score
        let (best_regime, best_score) = regime_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(regime, score)| (regime.clone(), *score))
            .unwrap_or((MarketRegime::Ranging, 0.5));
        
        // Collect trigger factors from all predictions
        let mut trigger_factors = Vec::new();
        for prediction in predictions {
            trigger_factors.extend(prediction.trigger_factors.clone());
        }
        
        Ok(RegimePrediction {
            regime: best_regime,
            confidence: best_score,
            transition_probability: self.calculate_transition_probability().await?,
            trigger_factors,
        })
    }

    async fn start_regime_detection(&self) -> Result<()> {
        let current_state = self.current_state.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                // Update time in regime
                let mut state = current_state.write().await;
                state.time_in_regime = state.time_in_regime + chrono::Duration::minutes(1);
            }
        });
        
        Ok(())
    }

    // Market indicator calculation methods
    async fn calculate_volatility(&self) -> Result<f64> {
        // Calculate rolling volatility
        Ok(0.15) // 15% annualized volatility
    }

    async fn calculate_volume(&self) -> Result<f64> {
        // Calculate normalized volume
        Ok(1.2) // 20% above average
    }

    async fn calculate_price_change(&self) -> Result<f64> {
        // Calculate price change percentage
        Ok(0.02) // 2% change
    }

    async fn calculate_correlation(&self) -> Result<f64> {
        // Calculate market correlation
        Ok(0.7) // 70% correlation
    }

    async fn calculate_liquidity(&self) -> Result<f64> {
        // Calculate liquidity measure
        Ok(0.8) // 80% liquidity
    }

    async fn calculate_momentum(&self) -> Result<f64> {
        // Calculate momentum indicator
        Ok(0.6) // 60% momentum
    }

    async fn get_vix_level(&self) -> Result<f64> {
        // Get VIX level
        Ok(20.0) // VIX at 20
    }

    async fn get_news_sentiment(&self) -> Result<f64> {
        // Get news sentiment score
        Ok(0.3) // Slightly positive sentiment
    }

    async fn calculate_transition_probability(&self) -> Result<f64> {
        // Calculate probability of regime transition
        Ok(0.15) // 15% probability
    }
}

#[derive(Debug, Clone)]
struct RegimePrediction {
    regime: MarketRegime,
    confidence: f64,
    transition_probability: f64,
    trigger_factors: Vec<String>,
}

#[derive(Debug)]
struct HiddenMarkovModel {
    states: Vec<MarketRegime>,
    transition_matrix: Vec<Vec<f64>>,
    emission_matrix: Vec<Vec<f64>>,
    initial_state: Vec<f64>,
}

impl HiddenMarkovModel {
    fn new() -> Self {
        Self {
            states: vec![
                MarketRegime::Trending,
                MarketRegime::Ranging,
                MarketRegime::Volatile,
                MarketRegime::Calm,
                MarketRegime::Crisis,
                MarketRegime::Recovery,
            ],
            transition_matrix: Vec::new(),
            emission_matrix: Vec::new(),
            initial_state: Vec::new(),
        }
    }

    async fn initialize(&mut self) -> Result<()> {
        // Initialize transition matrix
        let state_count = self.states.len();
        self.transition_matrix = vec![vec![0.1; state_count]; state_count];
        
        // Set higher probabilities for staying in same state
        for i in 0..state_count {
            self.transition_matrix[i][i] = 0.7;
        }
        
        // Initialize emission matrix
        self.emission_matrix = vec![vec![0.2; 8]; state_count]; // 8 indicators
        
        // Initialize initial state probabilities
        self.initial_state = vec![1.0 / state_count as f64; state_count];
        
        Ok(())
    }

    async fn detect_regime(&self, indicators: &MarketIndicators) -> Result<RegimePrediction> {
        // Simplified HMM inference
        let observation = self.indicators_to_observation(indicators);
        
        // Calculate likelihood for each state
        let mut likelihoods = Vec::new();
        for (i, state) in self.states.iter().enumerate() {
            let likelihood = self.calculate_likelihood(i, &observation);
            likelihoods.push(likelihood);
        }
        
        // Find most likely state
        let (best_idx, best_likelihood) = likelihoods.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        
        let trigger_factors = self.identify_trigger_factors(indicators);
        
        Ok(RegimePrediction {
            regime: self.states[best_idx].clone(),
            confidence: *best_likelihood,
            transition_probability: 0.2,
            trigger_factors,
        })
    }

    fn indicators_to_observation(&self, indicators: &MarketIndicators) -> Vec<f64> {
        vec![
            indicators.volatility,
            indicators.volume,
            indicators.price_change,
            indicators.correlation,
            indicators.liquidity,
            indicators.momentum,
            indicators.vix_level / 100.0, // Normalize VIX
            indicators.news_sentiment,
        ]
    }

    fn calculate_likelihood(&self, state_idx: usize, observation: &[f64]) -> f64 {
        // Simplified likelihood calculation
        let mut likelihood = 1.0;
        for (i, &obs) in observation.iter().enumerate() {
            likelihood *= self.emission_matrix[state_idx][i] * obs;
        }
        likelihood.max(0.001) // Avoid zero likelihood
    }

    fn identify_trigger_factors(&self, indicators: &MarketIndicators) -> Vec<String> {
        let mut factors = Vec::new();
        
        if indicators.volatility > 0.3 {
            factors.push("High volatility".to_string());
        }
        if indicators.volume > 2.0 {
            factors.push("High volume".to_string());
        }
        if indicators.price_change.abs() > 0.05 {
            factors.push("Large price movement".to_string());
        }
        if indicators.vix_level > 30.0 {
            factors.push("High VIX".to_string());
        }
        
        factors
    }
}

#[derive(Debug)]
struct VolatilityAnalyzer {
    volatility_history: Arc<RwLock<VecDeque<f64>>>,
}

impl VolatilityAnalyzer {
    fn new() -> Self {
        Self {
            volatility_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
        }
    }

    async fn initialize(&self) -> Result<()> {
        Ok(())
    }

    async fn analyze_regime(&self, indicators: &MarketIndicators) -> Result<RegimePrediction> {
        let mut history = self.volatility_history.write().await;
        history.push_back(indicators.volatility);
        if history.len() > 100 {
            history.pop_front();
        }
        
        let avg_volatility = history.iter().sum::<f64>() / history.len() as f64;
        let volatility_ratio = indicators.volatility / avg_volatility;
        
        let (regime, confidence) = if volatility_ratio > 2.0 {
            (MarketRegime::Crisis, 0.9)
        } else if volatility_ratio > 1.5 {
            (MarketRegime::Volatile, 0.8)
        } else if volatility_ratio < 0.5 {
            (MarketRegime::Calm, 0.7)
        } else {
            (MarketRegime::Ranging, 0.6)
        };
        
        Ok(RegimePrediction {
            regime,
            confidence,
            transition_probability: 0.1,
            trigger_factors: vec!["Volatility analysis".to_string()],
        })
    }
}

#[derive(Debug)]
struct CorrelationAnalyzer {
    correlation_history: Arc<RwLock<VecDeque<f64>>>,
}

impl CorrelationAnalyzer {
    fn new() -> Self {
        Self {
            correlation_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
        }
    }

    async fn initialize(&self) -> Result<()> {
        Ok(())
    }

    async fn analyze_regime(&self, indicators: &MarketIndicators) -> Result<RegimePrediction> {
        let mut history = self.correlation_history.write().await;
        history.push_back(indicators.correlation);
        if history.len() > 100 {
            history.pop_front();
        }
        
        let (regime, confidence) = if indicators.correlation > 0.8 {
            (MarketRegime::Crisis, 0.85)
        } else if indicators.correlation > 0.6 {
            (MarketRegime::Trending, 0.75)
        } else if indicators.correlation < 0.3 {
            (MarketRegime::Recovery, 0.7)
        } else {
            (MarketRegime::Ranging, 0.65)
        };
        
        Ok(RegimePrediction {
            regime,
            confidence,
            transition_probability: 0.1,
            trigger_factors: vec!["Correlation analysis".to_string()],
        })
    }
}

#[derive(Debug)]
struct MLPredictor {
    model_weights: Vec<f64>,
}

impl MLPredictor {
    fn new() -> Self {
        Self {
            model_weights: vec![0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1], // 8 features
        }
    }

    async fn initialize(&self) -> Result<()> {
        Ok(())
    }

    async fn predict_regime(&self, indicators: &MarketIndicators) -> Result<RegimePrediction> {
        let features = vec![
            indicators.volatility,
            indicators.volume,
            indicators.price_change,
            indicators.correlation,
            indicators.liquidity,
            indicators.momentum,
            indicators.vix_level / 100.0,
            indicators.news_sentiment,
        ];
        
        // Simple linear combination
        let score: f64 = features.iter()
            .zip(self.model_weights.iter())
            .map(|(f, w)| f * w)
            .sum();
        
        let (regime, confidence) = if score > 0.7 {
            (MarketRegime::Trending, 0.8)
        } else if score > 0.5 {
            (MarketRegime::Volatile, 0.7)
        } else if score > 0.3 {
            (MarketRegime::Ranging, 0.6)
        } else {
            (MarketRegime::Calm, 0.65)
        };
        
        Ok(RegimePrediction {
            regime,
            confidence,
            transition_probability: 0.15,
            trigger_factors: vec!["ML prediction".to_string()],
        })
    }
}