//! Whale Defense ML Integration with QAR Decision Loop
//!
//! This module integrates the whale defense ML ensemble directly into the QAR
//! decision pipeline, replacing simulation with actual ML-based whale detection.

use crate::core::{QarResult, FactorMap};
use crate::error::QarError;
use crate::decision::{DecisionContext, RiskConstraints};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use whale_defense_ml::{
    WhaleDetector, WhaleDetectorBuilder, PredictionResult as WhalePrediction,
    MarketTick, WhaleAlert
};

/// Configuration for whale defense integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleDefenseConfig {
    /// Enable whale defense analysis
    pub enabled: bool,
    /// Maximum inference time in microseconds
    pub max_inference_time_us: u64,
    /// Whale probability threshold for alerts
    pub alert_threshold: f32,
    /// Threat level threshold for defensive measures
    pub defensive_threshold: u8,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
    /// Model weights path (optional)
    pub weights_path: Option<String>,
}

impl Default for WhaleDefenseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_inference_time_us: 500, // Sub-millisecond target
            alert_threshold: 0.5,
            defensive_threshold: 3,
            use_gpu: true,
            weights_path: None,
        }
    }
}

/// Whale defense analysis result for QAR integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleDefenseAnalysis {
    /// Whale detection probability (0.0 to 1.0)
    pub whale_probability: f32,
    /// Threat level (1-5)
    pub threat_level: u8,
    /// Detection confidence
    pub confidence: f32,
    /// Inference time in microseconds
    pub inference_time_us: u64,
    /// Interpretability information
    pub interpretability: WhaleInterpretability,
    /// Risk factors
    pub risk_factors: WhaleRiskFactors,
    /// Defensive recommendations
    pub defensive_recommendations: DefensiveRecommendations,
}

/// Whale detection interpretability for QAR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleInterpretability {
    /// Top contributing features
    pub top_features: Vec<(String, f32)>,
    /// Anomaly score
    pub anomaly_score: f32,
    /// Behavioral classification
    pub behavioral_classification: String,
    /// Pattern confidence
    pub pattern_confidence: f32,
}

/// Risk factors from whale analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleRiskFactors {
    /// Market impact risk
    pub market_impact_risk: f32,
    /// Timing risk (execution timing vulnerability)
    pub timing_risk: f32,
    /// Information leakage risk
    pub information_leakage_risk: f32,
    /// Counter-party risk
    pub counter_party_risk: f32,
}

/// Defensive recommendations for QAR execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefensiveRecommendations {
    /// Use order fragmentation
    pub fragment_orders: bool,
    /// Apply timing randomization
    pub randomize_timing: bool,
    /// Enable steganographic execution
    pub use_steganography: bool,
    /// Reduce position size
    pub reduce_position_size: f32, // Multiplier (0.5 = 50% reduction)
    /// Delay execution (microseconds)
    pub execution_delay_us: u64,
}

/// Integrated whale defense for QAR decision loop
pub struct QARWhaleDefense {
    /// Configuration
    config: WhaleDefenseConfig,
    /// Whale detector instance
    whale_detector: Arc<WhaleDetector>,
    /// Market data buffer for sequence processing
    market_buffer: Arc<RwLock<Vec<MarketTick>>>,
    /// Performance metrics
    performance_metrics: Arc<RwLock<WhaleDefenseMetrics>>,
}

/// Performance metrics for whale defense
#[derive(Debug, Clone, Default)]
pub struct WhaleDefenseMetrics {
    pub total_analyses: u64,
    pub average_inference_time_us: u64,
    pub whale_detections: u64,
    pub false_positives: u64,
    pub performance_violations: u64, // Times exceeding max_inference_time
}

impl QARWhaleDefense {
    /// Create new whale defense integration
    pub async fn new(config: WhaleDefenseConfig) -> QarResult<Self> {
        // Build whale detector with optimal configuration
        let mut builder = WhaleDetectorBuilder::new();
        
        if config.use_gpu {
            if let Ok(gpu_device) = candle_core::Device::new_cuda(0) {
                builder = builder.device(gpu_device);
            }
        }
        
        if let Some(ref weights_path) = config.weights_path {
            builder = builder.weights_path(weights_path);
        }
        
        let whale_detector = builder.build()
            .map_err(|e| QarError::WhaleDefense(format!("Failed to create whale detector: {}", e)))?;
        
        Ok(Self {
            config,
            whale_detector: Arc::new(whale_detector),
            market_buffer: Arc::new(RwLock::new(Vec::with_capacity(60))),
            performance_metrics: Arc::new(RwLock::new(WhaleDefenseMetrics::default())),
        })
    }
    
    /// Analyze market data for whale activity
    pub async fn analyze_whale_activity(
        &self,
        context: &DecisionContext,
        current_price: f32,
        current_volume: f32,
        bid: Option<f32>,
        ask: Option<f32>,
    ) -> QarResult<WhaleDefenseAnalysis> {
        if !self.config.enabled {
            return Ok(Self::create_neutral_analysis());
        }
        
        let start_time = Instant::now();
        
        // Process current market tick
        let whale_prediction = self.whale_detector
            .process_tick(current_price, current_volume, bid, ask)
            .await
            .map_err(|e| QarError::WhaleDefense(format!("Whale detection failed: {}", e)))?;
        
        let analysis = match whale_prediction {
            Some(prediction) => {
                self.process_whale_prediction(prediction, context).await?
            },
            None => {
                // Not enough data for analysis yet
                Self::create_neutral_analysis()
            }
        };
        
        // Update performance metrics
        let inference_time_us = start_time.elapsed().as_micros() as u64;
        self.update_metrics(&analysis, inference_time_us).await;
        
        // Check performance constraints
        if inference_time_us > self.config.max_inference_time_us {
            tracing::warn!(
                "Whale defense inference time {}μs exceeds {}μs target",
                inference_time_us, self.config.max_inference_time_us
            );
            
            if let Ok(mut metrics) = self.performance_metrics.write().await {
                metrics.performance_violations += 1;
            }
        }
        
        Ok(analysis)
    }
    
    /// Process whale prediction result
    async fn process_whale_prediction(
        &self,
        prediction: WhalePrediction,
        context: &DecisionContext,
    ) -> QarResult<WhaleDefenseAnalysis> {
        // Extract interpretability information
        let interpretability = WhaleInterpretability {
            top_features: prediction.interpretability.top_features.clone(),
            anomaly_score: prediction.interpretability.anomaly_score,
            behavioral_classification: self.classify_whale_behavior(&prediction),
            pattern_confidence: prediction.confidence,
        };
        
        // Calculate risk factors based on market context
        let risk_factors = self.calculate_risk_factors(&prediction, context).await?;
        
        // Generate defensive recommendations
        let defensive_recommendations = self.generate_defensive_recommendations(
            &prediction, &risk_factors, context
        ).await?;
        
        Ok(WhaleDefenseAnalysis {
            whale_probability: prediction.whale_probability,
            threat_level: prediction.threat_level,
            confidence: prediction.confidence,
            inference_time_us: prediction.inference_time_us,
            interpretability,
            risk_factors,
            defensive_recommendations,
        })
    }
    
    /// Classify whale behavior based on prediction
    fn classify_whale_behavior(&self, prediction: &WhalePrediction) -> String {
        match (prediction.whale_probability, prediction.threat_level) {
            (p, _) if p < 0.3 => "Normal Market Activity".to_string(),
            (p, t) if p < 0.5 && t <= 2 => "Minor Institutional Activity".to_string(),
            (p, t) if p < 0.7 && t <= 3 => "Moderate Whale Activity".to_string(),
            (p, t) if p < 0.9 && t <= 4 => "Significant Whale Presence".to_string(),
            (_, _) => "Critical Whale Activity Detected".to_string(),
        }
    }
    
    /// Calculate risk factors from whale analysis
    async fn calculate_risk_factors(
        &self,
        prediction: &WhalePrediction,
        context: &DecisionContext,
    ) -> QarResult<WhaleRiskFactors> {
        // Extract market factors for risk calculation
        let volatility = context.factors.get_factor(&crate::core::StandardFactors::Volatility)?;
        let liquidity = context.factors.get_factor(&crate::core::StandardFactors::Liquidity)?;
        let volume = context.factors.get_factor(&crate::core::StandardFactors::Volume)?;
        
        // Market impact risk based on whale probability and liquidity
        let market_impact_risk = prediction.whale_probability * (1.0 - liquidity as f32);
        
        // Timing risk based on whale activity and volatility
        let timing_risk = prediction.whale_probability * volatility as f32 * 
            (prediction.threat_level as f32 / 5.0);
        
        // Information leakage risk based on whale confidence and volume
        let information_leakage_risk = prediction.confidence * volume as f32 *
            (prediction.whale_probability * 0.8);
        
        // Counter-party risk based on threat level and market conditions
        let counter_party_risk = (prediction.threat_level as f32 / 5.0) * 
            (1.0 - liquidity as f32) * prediction.confidence;
        
        Ok(WhaleRiskFactors {
            market_impact_risk,
            timing_risk,
            information_leakage_risk,
            counter_party_risk,
        })
    }
    
    /// Generate defensive recommendations
    async fn generate_defensive_recommendations(
        &self,
        prediction: &WhalePrediction,
        risk_factors: &WhaleRiskFactors,
        context: &DecisionContext,
    ) -> QarResult<DefensiveRecommendations> {
        let whale_prob = prediction.whale_probability;
        let threat_level = prediction.threat_level;
        
        // Fragment orders for high whale probability or market impact risk
        let fragment_orders = whale_prob > 0.6 || risk_factors.market_impact_risk > 0.5;
        
        // Randomize timing for moderate threat levels
        let randomize_timing = threat_level >= 3 || risk_factors.timing_risk > 0.4;
        
        // Use steganography for high threat levels
        let use_steganography = threat_level >= 4 || risk_factors.information_leakage_risk > 0.6;
        
        // Reduce position size based on combined risk
        let combined_risk = (risk_factors.market_impact_risk + 
                           risk_factors.timing_risk + 
                           risk_factors.information_leakage_risk + 
                           risk_factors.counter_party_risk) / 4.0;
        let reduce_position_size = if combined_risk > 0.7 {
            0.5 // 50% reduction
        } else if combined_risk > 0.5 {
            0.7 // 30% reduction  
        } else if combined_risk > 0.3 {
            0.85 // 15% reduction
        } else {
            1.0 // No reduction
        };
        
        // Execution delay based on threat level
        let execution_delay_us = match threat_level {
            5 => 1000, // 1ms delay for critical threats
            4 => 500,  // 500μs delay for high threats
            3 => 200,  // 200μs delay for medium threats
            _ => 0,    // No delay for low threats
        };
        
        Ok(DefensiveRecommendations {
            fragment_orders,
            randomize_timing,
            use_steganography,
            reduce_position_size,
            execution_delay_us,
        })
    }
    
    /// Create neutral analysis when whale defense is disabled
    fn create_neutral_analysis() -> WhaleDefenseAnalysis {
        WhaleDefenseAnalysis {
            whale_probability: 0.0,
            threat_level: 1,
            confidence: 1.0,
            inference_time_us: 0,
            interpretability: WhaleInterpretability {
                top_features: vec![],
                anomaly_score: 0.0,
                behavioral_classification: "Whale Defense Disabled".to_string(),
                pattern_confidence: 1.0,
            },
            risk_factors: WhaleRiskFactors {
                market_impact_risk: 0.0,
                timing_risk: 0.0,
                information_leakage_risk: 0.0,
                counter_party_risk: 0.0,
            },
            defensive_recommendations: DefensiveRecommendations {
                fragment_orders: false,
                randomize_timing: false,
                use_steganography: false,
                reduce_position_size: 1.0,
                execution_delay_us: 0,
            },
        }
    }
    
    /// Update performance metrics
    async fn update_metrics(&self, analysis: &WhaleDefenseAnalysis, inference_time_us: u64) {
        if let Ok(mut metrics) = self.performance_metrics.write().await {
            metrics.total_analyses += 1;
            
            // Update average inference time
            metrics.average_inference_time_us = 
                (metrics.average_inference_time_us * (metrics.total_analyses - 1) + inference_time_us) /
                metrics.total_analyses;
            
            // Count whale detections
            if analysis.whale_probability > self.config.alert_threshold {
                metrics.whale_detections += 1;
            }
        }
    }
    
    /// Get performance metrics
    pub async fn get_metrics(&self) -> WhaleDefenseMetrics {
        if let Ok(metrics) = self.performance_metrics.read().await {
            metrics.clone()
        } else {
            WhaleDefenseMetrics::default()
        }
    }
    
    /// Update false positive count (called when actual outcome is known)
    pub async fn update_false_positive(&self, was_false_positive: bool) {
        if was_false_positive {
            if let Ok(mut metrics) = self.performance_metrics.write().await {
                metrics.false_positives += 1;
            }
        }
    }
}

/// Extension trait for DecisionContext to include whale defense
pub trait DecisionContextWhaleExt {
    /// Add whale defense analysis to decision context
    fn with_whale_defense(self, whale_analysis: WhaleDefenseAnalysis) -> Self;
    
    /// Get whale defense analysis from context
    fn get_whale_defense(&self) -> Option<&WhaleDefenseAnalysis>;
}

// Note: This would require modifying DecisionContext in the actual implementation
// For now, we'll use a wrapper approach

/// Enhanced decision context with whale defense
#[derive(Debug, Clone)]
pub struct EnhancedDecisionContext {
    pub base_context: DecisionContext,
    pub whale_defense: Option<WhaleDefenseAnalysis>,
}

impl EnhancedDecisionContext {
    /// Create enhanced context from base context
    pub fn from_base(base_context: DecisionContext) -> Self {
        Self {
            base_context,
            whale_defense: None,
        }
    }
    
    /// Add whale defense analysis
    pub fn with_whale_defense(mut self, whale_analysis: WhaleDefenseAnalysis) -> Self {
        self.whale_defense = Some(whale_analysis);
        self
    }
    
    /// Get whale threat adjustment for risk calculations
    pub fn get_whale_threat_adjustment(&self) -> f32 {
        if let Some(ref whale_analysis) = self.whale_defense {
            whale_analysis.whale_probability * (whale_analysis.threat_level as f32 / 5.0)
        } else {
            0.0
        }
    }
    
    /// Get combined risk factors for decision adjustment
    pub fn get_combined_risk_adjustment(&self) -> f32 {
        if let Some(ref whale_analysis) = self.whale_defense {
            let risk_factors = &whale_analysis.risk_factors;
            (risk_factors.market_impact_risk + 
             risk_factors.timing_risk + 
             risk_factors.information_leakage_risk + 
             risk_factors.counter_party_risk) / 4.0
        } else {
            0.0
        }
    }
    
    /// Check if defensive measures are recommended
    pub fn requires_defensive_measures(&self) -> bool {
        if let Some(ref whale_analysis) = self.whale_defense {
            whale_analysis.threat_level >= 3 || whale_analysis.whale_probability > 0.6
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{AnalysisResult, TrendDirection, VolatilityLevel, MarketRegime};
    use std::collections::HashMap;

    fn create_test_decision_context() -> DecisionContext {
        let mut factors = std::collections::HashMap::new();
        factors.insert("trend".to_string(), 0.7);
        factors.insert("volatility".to_string(), 0.4);
        factors.insert("liquidity".to_string(), 0.8);
        factors.insert("volume".to_string(), 0.6);
        
        let factor_map = FactorMap::new(factors).unwrap();
        
        let analysis = AnalysisResult {
            timestamp: chrono::Utc::now(),
            trend: TrendDirection::Bullish,
            trend_strength: 0.8,
            volatility: VolatilityLevel::Medium,
            regime: MarketRegime::Bull,
            confidence: 0.9,
            metrics: HashMap::new(),
        };
        
        let risk_constraints = RiskConstraints {
            max_position_size: 0.1,
            max_risk_per_trade: 0.05,
            max_portfolio_risk: 0.2,
            min_confidence: 0.6,
        };
        
        DecisionContext {
            factors: factor_map,
            analysis,
            historical_performance: None,
            risk_constraints,
        }
    }

    #[tokio::test]
    async fn test_whale_defense_creation() {
        let config = WhaleDefenseConfig::default();
        let whale_defense = QARWhaleDefense::new(config).await;
        assert!(whale_defense.is_ok());
    }

    #[tokio::test]
    async fn test_whale_analysis() {
        let config = WhaleDefenseConfig {
            enabled: true,
            use_gpu: false, // Use CPU for testing
            ..Default::default()
        };
        
        let whale_defense = QARWhaleDefense::new(config).await.unwrap();
        let context = create_test_decision_context();
        
        // Process multiple ticks to build sequence
        for i in 0..60 {
            let price = 50000.0 + (i as f32 * 10.0);
            let volume = 1_000_000.0;
            
            let analysis = whale_defense.analyze_whale_activity(
                &context, price, volume, None, None
            ).await;
            
            assert!(analysis.is_ok());
            let analysis = analysis.unwrap();
            
            // Verify analysis structure
            assert!(analysis.whale_probability >= 0.0 && analysis.whale_probability <= 1.0);
            assert!(analysis.threat_level >= 1 && analysis.threat_level <= 5);
            assert!(analysis.confidence >= 0.0 && analysis.confidence <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_enhanced_decision_context() {
        let base_context = create_test_decision_context();
        let whale_analysis = WhaleDefenseAnalysis {
            whale_probability: 0.7,
            threat_level: 4,
            confidence: 0.9,
            inference_time_us: 300,
            interpretability: WhaleInterpretability {
                top_features: vec![("volume".to_string(), 0.8)],
                anomaly_score: 0.6,
                behavioral_classification: "Significant Whale Activity".to_string(),
                pattern_confidence: 0.9,
            },
            risk_factors: WhaleRiskFactors {
                market_impact_risk: 0.6,
                timing_risk: 0.5,
                information_leakage_risk: 0.4,
                counter_party_risk: 0.3,
            },
            defensive_recommendations: DefensiveRecommendations {
                fragment_orders: true,
                randomize_timing: true,
                use_steganography: false,
                reduce_position_size: 0.7,
                execution_delay_us: 500,
            },
        };
        
        let enhanced_context = EnhancedDecisionContext::from_base(base_context)
            .with_whale_defense(whale_analysis);
        
        // Test whale threat adjustment
        let threat_adjustment = enhanced_context.get_whale_threat_adjustment();
        assert!(threat_adjustment > 0.0);
        assert!(threat_adjustment <= 1.0);
        
        // Test combined risk adjustment
        let risk_adjustment = enhanced_context.get_combined_risk_adjustment();
        assert!(risk_adjustment > 0.0);
        assert!(risk_adjustment <= 1.0);
        
        // Test defensive measures requirement
        assert!(enhanced_context.requires_defensive_measures());
    }

    #[test]
    fn test_whale_behavior_classification() {
        let config = WhaleDefenseConfig::default();
        let whale_defense = QARWhaleDefense {
            config,
            whale_detector: Arc::new(WhaleDetector::new().unwrap()),
            market_buffer: Arc::new(RwLock::new(Vec::new())),
            performance_metrics: Arc::new(RwLock::new(WhaleDefenseMetrics::default())),
        };
        
        // Test different threat levels
        let prediction_low = WhalePrediction {
            whale_probability: 0.2,
            threat_level: 1,
            confidence: 0.8,
            inference_time_us: 200,
            model_predictions: HashMap::new(),
            interpretability: whale_defense_ml::InterpretabilityInfo {
                top_features: vec![],
                attention_weights: None,
                feature_importance: HashMap::new(),
                anomaly_score: 0.1,
            },
        };
        
        let classification = whale_defense.classify_whale_behavior(&prediction_low);
        assert_eq!(classification, "Normal Market Activity");
        
        let prediction_high = WhalePrediction {
            whale_probability: 0.95,
            threat_level: 5,
            confidence: 0.9,
            inference_time_us: 200,
            model_predictions: HashMap::new(),
            interpretability: whale_defense_ml::InterpretabilityInfo {
                top_features: vec![],
                attention_weights: None,
                feature_importance: HashMap::new(),
                anomaly_score: 0.9,
            },
        };
        
        let classification = whale_defense.classify_whale_behavior(&prediction_high);
        assert_eq!(classification, "Critical Whale Activity Detected");
    }
}