//! Antifragility-Enhanced Trading Strategy Integration
//! 
//! This module integrates the CDFA-Antifragility-Analyzer with the main trading strategy
//! to provide real-time antifragility scoring, dynamic position sizing, and stress testing.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::antifragility_integration::*;
use cdfa_antifragility_analyzer::*;

/// Errors specific to trading strategy integration
#[derive(Error, Debug)]
pub enum TradingStrategyError {
    #[error("Integration error: {0}")]
    IntegrationError(#[from] IntegrationError),
    
    #[error("Position sizing error: {message}")]
    PositionSizingError { message: String },
    
    #[error("Real-time analysis error: {message}")]
    RealTimeError { message: String },
    
    #[error("Risk management error: {message}")]
    RiskManagementError { message: String },
}

/// Result type for trading strategy operations
pub type TradingStrategyResult<T> = Result<T, TradingStrategyError>;

/// Antifragility-Enhanced Trading Strategy
#[derive(Debug)]
pub struct AntifragilityTradingStrategy {
    /// Core antifragility framework
    framework: Arc<AntifragilityTDDFramework>,
    
    /// Real-time analysis state
    real_time_state: Arc<RwLock<RealTimeAnalysisState>>,
    
    /// Position sizing manager
    position_manager: Arc<RwLock<PositionSizingManager>>,
    
    /// Risk management system
    risk_manager: Arc<RwLock<RiskManager>>,
    
    /// Configuration
    config: TradingStrategyConfig,
}

impl AntifragilityTradingStrategy {
    /// Create new antifragility-enhanced trading strategy
    pub fn new(config: TradingStrategyConfig) -> Self {
        let antifragility_params = AntifragilityParameters {
            enable_simd: config.enable_simd,
            enable_parallel: config.enable_parallel,
            cache_size: config.cache_size,
            min_data_points: config.min_data_points,
            vol_period: config.volatility_period,
            perf_period: config.performance_period,
            corr_window: config.correlation_window,
            ..Default::default()
        };
        
        let framework = Arc::new(AntifragilityTDDFramework::new(antifragility_params));
        
        Self {
            framework,
            real_time_state: Arc::new(RwLock::new(RealTimeAnalysisState::new(config.buffer_size))),
            position_manager: Arc::new(RwLock::new(PositionSizingManager::new(config.position_config.clone()))),
            risk_manager: Arc::new(RwLock::new(RiskManager::new(config.risk_config.clone()))),
            config,
        }
    }
    
    /// Process new market data point
    pub fn process_market_data(&self, market_tick: MarketTick) -> TradingStrategyResult<TradingSignal> {
        // Add data to real-time state
        self.update_real_time_state(market_tick)?;
        
        // Check if we should perform analysis
        let should_analyze = {
            let state = self.real_time_state.read().unwrap();
            state.should_analyze()
        };
        
        if should_analyze {
            // Perform antifragility analysis
            let analysis_result = self.perform_real_time_analysis()?;
            
            // Update position sizing
            self.update_position_sizing(&analysis_result)?;
            
            // Generate trading signal
            let signal = self.generate_trading_signal(&analysis_result, market_tick)?;
            
            // Apply risk management
            let final_signal = self.apply_risk_management(signal)?;
            
            Ok(final_signal)
        } else {
            // No analysis needed, return neutral signal
            Ok(TradingSignal::neutral(market_tick.symbol))
        }
    }
    
    /// Update real-time analysis state
    fn update_real_time_state(&self, tick: MarketTick) -> TradingStrategyResult<()> {
        let mut state = self.real_time_state.write().unwrap();
        state.add_tick(tick);
        Ok(())
    }
    
    /// Perform real-time antifragility analysis
    fn perform_real_time_analysis(&self) -> TradingStrategyResult<AnalysisResult> {
        let state = self.real_time_state.read().unwrap();
        let market_data = state.get_market_data();
        
        if market_data.prices.len() < self.config.min_data_points {
            return Err(TradingStrategyError::RealTimeError {
                message: format!("Insufficient data points: {} < {}", 
                               market_data.prices.len(), self.config.min_data_points),
            });
        }
        
        let analysis = self.framework.analyzer.analyze_prices(&market_data.prices, &market_data.volumes)
            .map_err(|e| TradingStrategyError::IntegrationError(IntegrationError::AnalysisError(e)))?;
        
        Ok(analysis)
    }
    
    /// Update position sizing based on antifragility analysis
    fn update_position_sizing(&self, analysis: &AnalysisResult) -> TradingStrategyResult<()> {
        let mut position_manager = self.position_manager.write().unwrap();
        position_manager.update_antifragility_score(analysis.antifragility_index)?;
        Ok(())
    }
    
    /// Generate trading signal based on antifragility analysis
    fn generate_trading_signal(&self, analysis: &AnalysisResult, tick: MarketTick) -> TradingStrategyResult<TradingSignal> {
        let position_manager = self.position_manager.read().unwrap();
        let base_position_size = position_manager.calculate_position_size(tick.price)?;
        
        // Determine signal direction based on antifragility components
        let signal_direction = self.determine_signal_direction(analysis)?;
        
        // Calculate signal strength
        let signal_strength = self.calculate_signal_strength(analysis)?;
        
        // Adjust position size based on antifragility
        let adjusted_size = base_position_size * signal_strength;
        
        Ok(TradingSignal {
            symbol: tick.symbol,
            direction: signal_direction,
            size: adjusted_size,
            confidence: analysis.antifragility_index,
            timestamp: tick.timestamp,
            analysis_summary: format!(
                "Antifragility: {:.3}, Convexity: {:.3}, Recovery: {:.3}, Benefit: {:.3}",
                analysis.antifragility_index,
                analysis.convexity_score,
                analysis.recovery_score,
                analysis.benefit_ratio_score
            ),
        })
    }
    
    /// Determine signal direction based on antifragility analysis
    fn determine_signal_direction(&self, analysis: &AnalysisResult) -> TradingStrategyResult<SignalDirection> {
        // Combined scoring approach
        let antifragility_score = analysis.antifragility_index;
        let convexity_score = analysis.convexity_score;
        let recovery_score = analysis.recovery_score;
        let benefit_ratio_score = analysis.benefit_ratio_score;
        
        // Weighted decision making
        let bullish_score = 
            antifragility_score * 0.3 +
            convexity_score * 0.3 +
            recovery_score * 0.2 +
            benefit_ratio_score * 0.2;
        
        let bearish_threshold = 0.4;
        let bullish_threshold = 0.6;
        
        if bullish_score > bullish_threshold {
            Ok(SignalDirection::Long)
        } else if bullish_score < bearish_threshold {
            Ok(SignalDirection::Short)
        } else {
            Ok(SignalDirection::Neutral)
        }
    }
    
    /// Calculate signal strength based on antifragility components
    fn calculate_signal_strength(&self, analysis: &AnalysisResult) -> TradingStrategyResult<f64> {
        // Signal strength based on antifragility confidence
        let base_strength = analysis.antifragility_index;
        
        // Boost strength if all components align
        let component_alignment = self.calculate_component_alignment(analysis)?;
        let aligned_strength = base_strength * (1.0 + component_alignment * 0.5);
        
        // Reduce strength if fragility is high
        let fragility_penalty = 1.0 - (analysis.fragility_score * 0.3);
        let final_strength = aligned_strength * fragility_penalty;
        
        Ok(final_strength.clamp(0.1, 1.0))
    }
    
    /// Calculate component alignment score
    fn calculate_component_alignment(&self, analysis: &AnalysisResult) -> TradingStrategyResult<f64> {
        let components = vec![
            analysis.convexity_score,
            analysis.asymmetry_score,
            analysis.recovery_score,
            analysis.benefit_ratio_score,
        ];
        
        let mean = components.iter().sum::<f64>() / components.len() as f64;
        let variance = components.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / components.len() as f64;
        
        // High alignment = low variance
        let alignment = 1.0 - variance.sqrt();
        Ok(alignment.clamp(0.0, 1.0))
    }
    
    /// Apply risk management to trading signal
    fn apply_risk_management(&self, signal: TradingSignal) -> TradingStrategyResult<TradingSignal> {
        let mut risk_manager = self.risk_manager.write().unwrap();
        risk_manager.evaluate_signal(signal)
    }
    
    /// Perform stress testing on current portfolio
    pub fn perform_stress_test(&self, scenarios: &[StressScenario]) -> TradingStrategyResult<StressTestResult> {
        let mut results = Vec::new();
        
        for scenario in scenarios {
            let scenario_result = self.analyze_stress_scenario(scenario)?;
            results.push(scenario_result);
        }
        
        let overall_resilience = self.calculate_overall_resilience(&results)?;
        
        Ok(StressTestResult {
            scenario_results: results,
            overall_resilience,
            recommendations: self.generate_stress_test_recommendations(&results)?,
        })
    }
    
    /// Analyze single stress scenario
    fn analyze_stress_scenario(&self, scenario: &StressScenario) -> TradingStrategyResult<ScenarioResult> {
        let market_data = MarketData {
            prices: scenario.price_path.clone(),
            volumes: scenario.volume_path.clone(),
            timestamps: scenario.timestamps.clone(),
        };
        
        let analysis = self.framework.analyzer.analyze_prices(&market_data.prices, &market_data.volumes)
            .map_err(|e| TradingStrategyError::IntegrationError(IntegrationError::AnalysisError(e)))?;
        
        // Calculate scenario-specific metrics
        let drawdown = self.calculate_max_drawdown(&scenario.price_path)?;
        let volatility = self.calculate_volatility(&scenario.price_path)?;
        let recovery_time = self.calculate_recovery_time(&scenario.price_path)?;
        
        Ok(ScenarioResult {
            scenario_name: scenario.name.clone(),
            antifragility_index: analysis.antifragility_index,
            max_drawdown: drawdown,
            volatility,
            recovery_time,
            resilience_score: self.calculate_resilience_score(&analysis, drawdown, recovery_time)?,
        })
    }
    
    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, prices: &[f64]) -> TradingStrategyResult<f64> {
        if prices.is_empty() {
            return Ok(0.0);
        }
        
        let mut peak = prices[0];
        let mut max_drawdown = 0.0;
        
        for &price in prices {
            if price > peak {
                peak = price;
            }
            
            let drawdown = (peak - price) / peak;
            max_drawdown = max_drawdown.max(drawdown);
        }
        
        Ok(max_drawdown)
    }
    
    /// Calculate volatility
    fn calculate_volatility(&self, prices: &[f64]) -> TradingStrategyResult<f64> {
        if prices.len() < 2 {
            return Ok(0.0);
        }
        
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0] - 1.0).ln())
            .collect();
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        Ok(variance.sqrt())
    }
    
    /// Calculate recovery time
    fn calculate_recovery_time(&self, prices: &[f64]) -> TradingStrategyResult<f64> {
        if prices.is_empty() {
            return Ok(0.0);
        }
        
        let mut peak = prices[0];
        let mut trough_idx = 0;
        let mut recovery_time = 0.0;
        
        for (i, &price) in prices.iter().enumerate() {
            if price > peak {
                peak = price;
            } else if price < prices[trough_idx] {
                trough_idx = i;
            }
            
            // Check if we've recovered from trough
            if i > trough_idx && price >= peak * 0.95 {
                recovery_time = (i - trough_idx) as f64;
                break;
            }
        }
        
        Ok(recovery_time)
    }
    
    /// Calculate resilience score
    fn calculate_resilience_score(&self, analysis: &AnalysisResult, drawdown: f64, recovery_time: f64) -> TradingStrategyResult<f64> {
        let antifragility_component = analysis.antifragility_index;
        let drawdown_component = 1.0 - drawdown.min(1.0);
        let recovery_component = if recovery_time > 0.0 {
            1.0 / (1.0 + recovery_time * 0.1)
        } else {
            1.0
        };
        
        let resilience = (antifragility_component * 0.5 + drawdown_component * 0.3 + recovery_component * 0.2).clamp(0.0, 1.0);
        Ok(resilience)
    }
    
    /// Calculate overall resilience
    fn calculate_overall_resilience(&self, results: &[ScenarioResult]) -> TradingStrategyResult<f64> {
        if results.is_empty() {
            return Ok(0.0);
        }
        
        let avg_resilience = results.iter()
            .map(|r| r.resilience_score)
            .sum::<f64>() / results.len() as f64;
        
        Ok(avg_resilience)
    }
    
    /// Generate stress test recommendations
    fn generate_stress_test_recommendations(&self, results: &[ScenarioResult]) -> TradingStrategyResult<Vec<String>> {
        let mut recommendations = Vec::new();
        
        let avg_resilience = results.iter()
            .map(|r| r.resilience_score)
            .sum::<f64>() / results.len() as f64;
        
        if avg_resilience < 0.6 {
            recommendations.push("Consider reducing position sizes during high volatility periods".to_string());
        }
        
        let high_drawdown_scenarios = results.iter()
            .filter(|r| r.max_drawdown > 0.2)
            .count();
        
        if high_drawdown_scenarios > results.len() / 2 {
            recommendations.push("Implement stronger stop-loss mechanisms".to_string());
        }
        
        let slow_recovery_scenarios = results.iter()
            .filter(|r| r.recovery_time > 50.0)
            .count();
        
        if slow_recovery_scenarios > results.len() / 3 {
            recommendations.push("Consider more aggressive rebalancing during recovery phases".to_string());
        }
        
        Ok(recommendations)
    }
    
    /// Get current antifragility metrics
    pub fn get_current_metrics(&self) -> TradingStrategyResult<AntifragilityMetrics> {
        let state = self.real_time_state.read().unwrap();
        let market_data = state.get_market_data();
        
        if market_data.prices.len() < self.config.min_data_points {
            return Ok(AntifragilityMetrics::default());
        }
        
        let analysis = self.framework.analyzer.analyze_prices(&market_data.prices, &market_data.volumes)
            .map_err(|e| TradingStrategyError::IntegrationError(IntegrationError::AnalysisError(e)))?;
        
        let position_manager = self.position_manager.read().unwrap();
        let risk_manager = self.risk_manager.read().unwrap();
        
        Ok(AntifragilityMetrics {
            antifragility_index: analysis.antifragility_index,
            fragility_score: analysis.fragility_score,
            convexity_score: analysis.convexity_score,
            asymmetry_score: analysis.asymmetry_score,
            recovery_score: analysis.recovery_score,
            benefit_ratio_score: analysis.benefit_ratio_score,
            position_size_multiplier: position_manager.get_current_multiplier(),
            risk_level: risk_manager.get_current_risk_level(),
            data_points: market_data.prices.len(),
        })
    }
    
    /// Get performance benchmarks
    pub fn get_performance_benchmarks(&self) -> PerformanceBenchmarks {
        self.framework.get_benchmarks()
    }
}

/// Trading strategy configuration
#[derive(Debug, Clone)]
pub struct TradingStrategyConfig {
    pub buffer_size: usize,
    pub min_data_points: usize,
    pub volatility_period: usize,
    pub performance_period: usize,
    pub correlation_window: usize,
    pub enable_simd: bool,
    pub enable_parallel: bool,
    pub cache_size: usize,
    pub position_config: PositionSizingConfig,
    pub risk_config: RiskManagementConfig,
}

impl Default for TradingStrategyConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            min_data_points: 100,
            volatility_period: 21,
            performance_period: 63,
            correlation_window: 42,
            enable_simd: true,
            enable_parallel: true,
            cache_size: 500,
            position_config: PositionSizingConfig::default(),
            risk_config: RiskManagementConfig::default(),
        }
    }
}

/// Real-time analysis state
#[derive(Debug)]
pub struct RealTimeAnalysisState {
    ticks: Vec<MarketTick>,
    buffer_size: usize,
    last_analysis: Option<Instant>,
    analysis_interval: Duration,
}

impl RealTimeAnalysisState {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            ticks: Vec::with_capacity(buffer_size),
            buffer_size,
            last_analysis: None,
            analysis_interval: Duration::from_secs(60), // Analyze every minute
        }
    }
    
    pub fn add_tick(&mut self, tick: MarketTick) {
        self.ticks.push(tick);
        if self.ticks.len() > self.buffer_size {
            self.ticks.remove(0);
        }
    }
    
    pub fn should_analyze(&self) -> bool {
        if self.ticks.len() < 100 {
            return false;
        }
        
        match self.last_analysis {
            Some(last) => last.elapsed() >= self.analysis_interval,
            None => true,
        }
    }
    
    pub fn get_market_data(&self) -> MarketData {
        let prices: Vec<f64> = self.ticks.iter().map(|t| t.price).collect();
        let volumes: Vec<f64> = self.ticks.iter().map(|t| t.volume).collect();
        let timestamps: Vec<u64> = self.ticks.iter().map(|t| t.timestamp).collect();
        
        MarketData {
            prices,
            volumes,
            timestamps,
        }
    }
}

/// Market tick data
#[derive(Debug, Clone)]
pub struct MarketTick {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub timestamp: u64,
}

/// Trading signal
#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub symbol: String,
    pub direction: SignalDirection,
    pub size: f64,
    pub confidence: f64,
    pub timestamp: u64,
    pub analysis_summary: String,
}

impl TradingSignal {
    pub fn neutral(symbol: String) -> Self {
        Self {
            symbol,
            direction: SignalDirection::Neutral,
            size: 0.0,
            confidence: 0.0,
            timestamp: 0,
            analysis_summary: "Neutral signal".to_string(),
        }
    }
}

/// Signal direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    Long,
    Short,
    Neutral,
}

/// Position sizing manager
#[derive(Debug)]
pub struct PositionSizingManager {
    config: PositionSizingConfig,
    current_antifragility: f64,
    position_multiplier: f64,
}

impl PositionSizingManager {
    pub fn new(config: PositionSizingConfig) -> Self {
        Self {
            config,
            current_antifragility: 0.5,
            position_multiplier: 1.0,
        }
    }
    
    pub fn update_antifragility_score(&mut self, score: f64) -> TradingStrategyResult<()> {
        self.current_antifragility = score;
        
        // Adjust position multiplier based on antifragility
        self.position_multiplier = if score > 0.7 {
            1.5 // Increase positions for highly antifragile assets
        } else if score < 0.3 {
            0.5 // Reduce positions for fragile assets
        } else {
            1.0
        };
        
        Ok(())
    }
    
    pub fn calculate_position_size(&self, price: f64) -> TradingStrategyResult<f64> {
        let base_size = self.config.base_position_size;
        let antifragility_adjustment = self.current_antifragility * self.config.antifragility_multiplier;
        
        let adjusted_size = base_size * self.position_multiplier * (1.0 + antifragility_adjustment);
        
        Ok(adjusted_size.clamp(self.config.min_position_size, self.config.max_position_size))
    }
    
    pub fn get_current_multiplier(&self) -> f64 {
        self.position_multiplier
    }
}

/// Position sizing configuration
#[derive(Debug, Clone)]
pub struct PositionSizingConfig {
    pub base_position_size: f64,
    pub min_position_size: f64,
    pub max_position_size: f64,
    pub antifragility_multiplier: f64,
}

impl Default for PositionSizingConfig {
    fn default() -> Self {
        Self {
            base_position_size: 0.1,
            min_position_size: 0.01,
            max_position_size: 0.5,
            antifragility_multiplier: 0.3,
        }
    }
}

/// Risk manager
#[derive(Debug)]
pub struct RiskManager {
    config: RiskManagementConfig,
    current_risk_level: f64,
}

impl RiskManager {
    pub fn new(config: RiskManagementConfig) -> Self {
        Self {
            config,
            current_risk_level: 0.5,
        }
    }
    
    pub fn evaluate_signal(&mut self, signal: TradingSignal) -> TradingStrategyResult<TradingSignal> {
        let mut adjusted_signal = signal;
        
        // Apply risk-based position sizing
        if self.current_risk_level > self.config.high_risk_threshold {
            adjusted_signal.size *= self.config.high_risk_multiplier;
        } else if self.current_risk_level < self.config.low_risk_threshold {
            adjusted_signal.size *= self.config.low_risk_multiplier;
        }
        
        // Apply maximum position limits
        adjusted_signal.size = adjusted_signal.size.min(self.config.max_position_size);
        
        Ok(adjusted_signal)
    }
    
    pub fn get_current_risk_level(&self) -> f64 {
        self.current_risk_level
    }
}

/// Risk management configuration
#[derive(Debug, Clone)]
pub struct RiskManagementConfig {
    pub high_risk_threshold: f64,
    pub low_risk_threshold: f64,
    pub high_risk_multiplier: f64,
    pub low_risk_multiplier: f64,
    pub max_position_size: f64,
}

impl Default for RiskManagementConfig {
    fn default() -> Self {
        Self {
            high_risk_threshold: 0.8,
            low_risk_threshold: 0.2,
            high_risk_multiplier: 0.5,
            low_risk_multiplier: 1.2,
            max_position_size: 0.3,
        }
    }
}

/// Stress test scenario
#[derive(Debug, Clone)]
pub struct StressScenario {
    pub name: String,
    pub price_path: Vec<f64>,
    pub volume_path: Vec<f64>,
    pub timestamps: Vec<u64>,
}

/// Stress test result
#[derive(Debug, Clone)]
pub struct StressTestResult {
    pub scenario_results: Vec<ScenarioResult>,
    pub overall_resilience: f64,
    pub recommendations: Vec<String>,
}

/// Scenario result
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    pub scenario_name: String,
    pub antifragility_index: f64,
    pub max_drawdown: f64,
    pub volatility: f64,
    pub recovery_time: f64,
    pub resilience_score: f64,
}

/// Antifragility metrics
#[derive(Debug, Clone, Default)]
pub struct AntifragilityMetrics {
    pub antifragility_index: f64,
    pub fragility_score: f64,
    pub convexity_score: f64,
    pub asymmetry_score: f64,
    pub recovery_score: f64,
    pub benefit_ratio_score: f64,
    pub position_size_multiplier: f64,
    pub risk_level: f64,
    pub data_points: usize,
}

impl AntifragilityMetrics {
    pub fn summary(&self) -> String {
        format!(
            "Antifragility Metrics:\n\
             - Antifragility Index: {:.3}\n\
             - Fragility Score: {:.3}\n\
             - Convexity: {:.3}\n\
             - Asymmetry: {:.3}\n\
             - Recovery: {:.3}\n\
             - Benefit Ratio: {:.3}\n\
             - Position Multiplier: {:.3}\n\
             - Risk Level: {:.3}\n\
             - Data Points: {}",
            self.antifragility_index,
            self.fragility_score,
            self.convexity_score,
            self.asymmetry_score,
            self.recovery_score,
            self.benefit_ratio_score,
            self.position_size_multiplier,
            self.risk_level,
            self.data_points
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_strategy() -> AntifragilityTradingStrategy {
        let config = TradingStrategyConfig::default();
        AntifragilityTradingStrategy::new(config)
    }
    
    fn create_test_tick(price: f64, volume: f64) -> MarketTick {
        MarketTick {
            symbol: "BTCUSD".to_string(),
            price,
            volume,
            timestamp: 1234567890,
        }
    }
    
    #[test]
    fn test_strategy_creation() {
        let strategy = create_test_strategy();
        let metrics = strategy.get_current_metrics().unwrap();
        assert_eq!(metrics.data_points, 0);
    }
    
    #[test]
    fn test_market_data_processing() {
        let strategy = create_test_strategy();
        
        // Add some test data
        for i in 0..200 {
            let tick = create_test_tick(100.0 + i as f64 * 0.1, 1000.0);
            let result = strategy.process_market_data(tick);
            assert!(result.is_ok());
        }
        
        let metrics = strategy.get_current_metrics().unwrap();
        assert_eq!(metrics.data_points, 200);
    }
    
    #[test]
    fn test_stress_testing() {
        let strategy = create_test_strategy();
        
        // Create a stress scenario
        let scenario = StressScenario {
            name: "Test Crash".to_string(),
            price_path: (0..100).map(|i| 100.0 - i as f64 * 0.5).collect(),
            volume_path: vec![1000.0; 100],
            timestamps: (0..100).map(|i| i as u64).collect(),
        };
        
        let result = strategy.perform_stress_test(&[scenario]);
        assert!(result.is_ok());
        
        let stress_result = result.unwrap();
        assert_eq!(stress_result.scenario_results.len(), 1);
        assert!(!stress_result.recommendations.is_empty());
    }
    
    #[test]
    fn test_position_sizing() {
        let config = PositionSizingConfig::default();
        let mut manager = PositionSizingManager::new(config);
        
        // Test with high antifragility
        manager.update_antifragility_score(0.8).unwrap();
        let position_size = manager.calculate_position_size(100.0).unwrap();
        assert!(position_size > 0.1); // Should be larger than base
        
        // Test with low antifragility
        manager.update_antifragility_score(0.2).unwrap();
        let position_size = manager.calculate_position_size(100.0).unwrap();
        assert!(position_size < 0.1); // Should be smaller than base
    }
    
    #[test]
    fn test_risk_management() {
        let config = RiskManagementConfig::default();
        let mut manager = RiskManager::new(config);
        
        let signal = TradingSignal {
            symbol: "BTCUSD".to_string(),
            direction: SignalDirection::Long,
            size: 0.2,
            confidence: 0.8,
            timestamp: 1234567890,
            analysis_summary: "Test signal".to_string(),
        };
        
        let adjusted_signal = manager.evaluate_signal(signal).unwrap();
        assert!(adjusted_signal.size <= 0.3); // Should respect max position size
    }
}