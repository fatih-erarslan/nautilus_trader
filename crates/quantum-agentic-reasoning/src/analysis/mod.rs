//! Market Analysis Module
//!
//! This module provides quantum-enhanced market analysis capabilities including
//! trend analysis, volatility estimation, correlation analysis, and regime detection.

pub mod market_analysis;
pub mod trend_analysis;
pub mod volatility_analysis;
pub mod correlation_analysis;
pub mod regime_detection;
pub mod pattern_analysis;
pub mod risk_analysis;
pub mod sentiment_analysis;

use crate::core::{QarResult, StandardFactors, FactorMap};
use crate::quantum::QuantumState;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Market analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Analysis window size
    pub window_size: usize,
    /// Update frequency in seconds
    pub update_frequency: u64,
    /// Confidence threshold for decisions
    pub confidence_threshold: f64,
    /// Use quantum analysis
    pub use_quantum: bool,
    /// Maximum history to maintain
    pub max_history: usize,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            update_frequency: 60,
            confidence_threshold: 0.7,
            use_quantum: true,
            max_history: 1000,
        }
    }
}

/// Market analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Analysis timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Market trend direction
    pub trend: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub trend_strength: f64,
    /// Volatility level
    pub volatility: VolatilityLevel,
    /// Market regime
    pub regime: MarketRegime,
    /// Confidence in analysis
    pub confidence: f64,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Trend direction enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Bullish,
    Bearish,
    Sideways,
    Unknown,
}

/// Volatility level enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VolatilityLevel {
    Low,
    Medium,
    High,
    Extreme,
}

/// Market regime enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MarketRegime {
    Bull,
    Bear,
    Consolidation,
    Transition,
    Crisis,
}

/// Main market analyzer
#[derive(Debug)]
pub struct MarketAnalyzer {
    config: AnalysisConfig,
    trend_analyzer: trend_analysis::TrendAnalyzer,
    volatility_analyzer: volatility_analysis::VolatilityAnalyzer,
    correlation_analyzer: correlation_analysis::CorrelationAnalyzer,
    regime_detector: regime_detection::RegimeDetector,
    pattern_analyzer: pattern_analysis::PatternAnalyzer,
    risk_analyzer: risk_analysis::RiskAnalyzer,
    sentiment_analyzer: sentiment_analysis::SentimentAnalyzer,
    history: Vec<AnalysisResult>,
}

impl MarketAnalyzer {
    /// Create a new market analyzer
    pub fn new(config: AnalysisConfig) -> QarResult<Self> {
        Ok(Self {
            trend_analyzer: trend_analysis::TrendAnalyzer::new(config.clone())?,
            volatility_analyzer: volatility_analysis::VolatilityAnalyzer::new(config.clone())?,
            correlation_analyzer: correlation_analysis::CorrelationAnalyzer::new(config.clone())?,
            regime_detector: regime_detection::RegimeDetector::new(config.clone())?,
            pattern_analyzer: pattern_analysis::PatternAnalyzer::new(config.clone())?,
            risk_analyzer: risk_analysis::RiskAnalyzer::new(config.clone())?,
            sentiment_analyzer: sentiment_analysis::SentimentAnalyzer::new(config.clone())?,
            config,
            history: Vec::new(),
        })
    }

    /// Perform comprehensive market analysis
    pub async fn analyze(&mut self, factors: &FactorMap) -> QarResult<AnalysisResult> {
        let timestamp = chrono::Utc::now();

        // Perform individual analyses
        let trend_result = self.trend_analyzer.analyze(factors).await?;
        let volatility_result = self.volatility_analyzer.analyze(factors).await?;
        let correlation_result = self.correlation_analyzer.analyze(factors).await?;
        let regime_result = self.regime_detector.detect(factors).await?;
        let pattern_result = self.pattern_analyzer.analyze(factors).await?;
        let risk_result = self.risk_analyzer.analyze(factors).await?;
        let sentiment_result = self.sentiment_analyzer.analyze(factors).await?;

        // Combine results
        let trend = self.determine_trend(&trend_result, &regime_result);
        let trend_strength = trend_result.strength;
        let volatility = self.determine_volatility(&volatility_result);
        let regime = regime_result.regime;
        
        // Calculate overall confidence
        let confidence = self.calculate_confidence(&[
            trend_result.confidence,
            volatility_result.confidence,
            correlation_result.confidence,
            regime_result.confidence,
            pattern_result.confidence,
            risk_result.confidence,
            sentiment_result.confidence,
        ]);

        // Compile metrics
        let mut metrics = HashMap::new();
        metrics.insert("trend_strength".to_string(), trend_strength);
        metrics.insert("volatility_score".to_string(), volatility_result.score);
        metrics.insert("correlation_score".to_string(), correlation_result.score);
        metrics.insert("regime_confidence".to_string(), regime_result.confidence);
        metrics.insert("pattern_score".to_string(), pattern_result.score);
        metrics.insert("risk_score".to_string(), risk_result.score);
        metrics.insert("sentiment_score".to_string(), sentiment_result.score);

        let result = AnalysisResult {
            timestamp,
            trend,
            trend_strength,
            volatility,
            regime,
            confidence,
            metrics,
        };

        // Store in history
        self.add_to_history(result.clone());

        Ok(result)
    }

    /// Get analysis history
    pub fn get_history(&self) -> &[AnalysisResult] {
        &self.history
    }

    /// Get latest analysis result
    pub fn get_latest(&self) -> Option<&AnalysisResult> {
        self.history.last()
    }

    /// Clear analysis history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Add result to history
    fn add_to_history(&mut self, result: AnalysisResult) {
        self.history.push(result);
        
        // Maintain maximum history size
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }

    /// Determine overall trend
    fn determine_trend(
        &self,
        trend_result: &trend_analysis::TrendResult,
        regime_result: &regime_detection::RegimeResult,
    ) -> TrendDirection {
        match (trend_result.direction.clone(), &regime_result.regime) {
            (trend_analysis::TrendDirection::Bullish, MarketRegime::Bull) => TrendDirection::Bullish,
            (trend_analysis::TrendDirection::Bearish, MarketRegime::Bear) => TrendDirection::Bearish,
            (trend_analysis::TrendDirection::Sideways, _) => TrendDirection::Sideways,
            (_, MarketRegime::Consolidation) => TrendDirection::Sideways,
            (_, MarketRegime::Transition) => TrendDirection::Unknown,
            _ => trend_result.direction.into(),
        }
    }

    /// Determine volatility level
    fn determine_volatility(&self, volatility_result: &volatility_analysis::VolatilityResult) -> VolatilityLevel {
        match volatility_result.level {
            volatility_analysis::VolatilityLevel::Low => VolatilityLevel::Low,
            volatility_analysis::VolatilityLevel::Medium => VolatilityLevel::Medium,
            volatility_analysis::VolatilityLevel::High => VolatilityLevel::High,
            volatility_analysis::VolatilityLevel::Extreme => VolatilityLevel::Extreme,
        }
    }

    /// Calculate overall confidence
    fn calculate_confidence(&self, confidences: &[f64]) -> f64 {
        if confidences.is_empty() {
            return 0.0;
        }

        // Use harmonic mean for conservative confidence estimation
        let sum_reciprocals: f64 = confidences.iter().map(|&c| if c > 0.0 { 1.0 / c } else { f64::INFINITY }).sum();
        
        if sum_reciprocals.is_infinite() {
            0.0
        } else {
            confidences.len() as f64 / sum_reciprocals
        }
    }
}

/// Convert between internal and external trend types
impl From<trend_analysis::TrendDirection> for TrendDirection {
    fn from(direction: trend_analysis::TrendDirection) -> Self {
        match direction {
            trend_analysis::TrendDirection::Bullish => TrendDirection::Bullish,
            trend_analysis::TrendDirection::Bearish => TrendDirection::Bearish,
            trend_analysis::TrendDirection::Sideways => TrendDirection::Sideways,
            trend_analysis::TrendDirection::Unknown => TrendDirection::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;

    #[tokio::test]
    async fn test_market_analyzer_creation() {
        let config = AnalysisConfig::default();
        let analyzer = MarketAnalyzer::new(config);
        assert!(analyzer.is_ok());
    }

    #[tokio::test]
    async fn test_analysis_with_sample_data() {
        let config = AnalysisConfig::default();
        let mut analyzer = MarketAnalyzer::new(config).unwrap();

        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Trend.to_string(), 0.7);
        factors.insert(StandardFactors::Volatility.to_string(), 0.3);
        factors.insert(StandardFactors::Momentum.to_string(), 0.8);
        factors.insert(StandardFactors::Sentiment.to_string(), 0.6);
        
        let factor_map = FactorMap::new(factors).unwrap();
        let result = analyzer.analyze(&factor_map).await;
        
        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert!(analysis.confidence >= 0.0 && analysis.confidence <= 1.0);
        assert!(analysis.trend_strength >= 0.0 && analysis.trend_strength <= 1.0);
    }

    #[test]
    fn test_confidence_calculation() {
        let config = AnalysisConfig::default();
        let analyzer = MarketAnalyzer::new(config).unwrap();
        
        let confidences = vec![0.8, 0.7, 0.9, 0.6];
        let overall_confidence = analyzer.calculate_confidence(&confidences);
        
        assert!(overall_confidence > 0.0);
        assert!(overall_confidence <= 1.0);
    }

    #[test]
    fn test_history_management() {
        let config = AnalysisConfig {
            max_history: 2,
            ..Default::default()
        };
        let mut analyzer = MarketAnalyzer::new(config).unwrap();

        let result1 = AnalysisResult {
            timestamp: chrono::Utc::now(),
            trend: TrendDirection::Bullish,
            trend_strength: 0.8,
            volatility: VolatilityLevel::Medium,
            regime: MarketRegime::Bull,
            confidence: 0.9,
            metrics: HashMap::new(),
        };

        let result2 = result1.clone();
        let result3 = result1.clone();

        analyzer.add_to_history(result1);
        analyzer.add_to_history(result2);
        analyzer.add_to_history(result3);

        // Should only keep the last 2 results
        assert_eq!(analyzer.get_history().len(), 2);
    }
}