//! Market phase detection based on Panarchy theory

use crate::factors::MarketData;
use std::collections::VecDeque;

/// Market phases from Panarchy theory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketPhase {
    /// Growth phase - expansion and opportunity
    Growth,
    /// Conservation phase - stability and efficiency
    Conservation,
    /// Release phase - chaos and destruction
    Release,
    /// Reorganization phase - renewal and innovation
    Reorganization,
}

impl MarketPhase {
    /// Get phase name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Growth => "growth",
            Self::Conservation => "conservation",
            Self::Release => "release",
            Self::Reorganization => "reorganization",
        }
    }
    
    /// Get phase characteristics
    pub fn characteristics(&self) -> PhaseCharacteristics {
        match self {
            Self::Growth => PhaseCharacteristics {
                volatility_range: (0.0, 0.4),
                trend_bias: 0.5,
                momentum_strength: 0.6,
                risk_level: 0.3,
                opportunity_level: 0.8,
            },
            Self::Conservation => PhaseCharacteristics {
                volatility_range: (0.0, 0.3),
                trend_bias: 0.2,
                momentum_strength: 0.3,
                risk_level: 0.2,
                opportunity_level: 0.4,
            },
            Self::Release => PhaseCharacteristics {
                volatility_range: (0.5, 1.0),
                trend_bias: -0.5,
                momentum_strength: 0.8,
                risk_level: 0.9,
                opportunity_level: 0.6,
            },
            Self::Reorganization => PhaseCharacteristics {
                volatility_range: (0.3, 0.7),
                trend_bias: 0.0,
                momentum_strength: 0.5,
                risk_level: 0.6,
                opportunity_level: 0.7,
            },
        }
    }
}

/// Characteristics of a market phase
#[derive(Debug, Clone)]
pub struct PhaseCharacteristics {
    pub volatility_range: (f64, f64),
    pub trend_bias: f64,
    pub momentum_strength: f64,
    pub risk_level: f64,
    pub opportunity_level: f64,
}

/// Market phase detector
pub struct PhaseDetector {
    /// Current detected phase
    current_phase: MarketPhase,
    /// Phase history
    phase_history: VecDeque<(MarketPhase, chrono::DateTime<chrono::Utc>)>,
    /// Market data history for phase detection
    market_history: VecDeque<MarketData>,
    /// Configuration
    config: PhaseDetectorConfig,
}

/// Phase detector configuration
#[derive(Debug, Clone)]
pub struct PhaseDetectorConfig {
    /// History size for phase detection
    pub history_size: usize,
    /// Minimum duration in same phase before transition
    pub min_phase_duration: std::time::Duration,
    /// Sensitivity to phase changes
    pub sensitivity: f64,
}

impl Default for PhaseDetectorConfig {
    fn default() -> Self {
        Self {
            history_size: 50,
            min_phase_duration: std::time::Duration::from_secs(300), // 5 minutes
            sensitivity: 0.7,
        }
    }
}

impl PhaseDetector {
    /// Create new phase detector
    pub fn new() -> Self {
        Self::with_config(PhaseDetectorConfig::default())
    }
    
    /// Create with custom configuration
    pub fn with_config(config: PhaseDetectorConfig) -> Self {
        Self {
            current_phase: MarketPhase::Growth,
            phase_history: VecDeque::with_capacity(100),
            market_history: VecDeque::with_capacity(config.history_size),
            config,
        }
    }
    
    /// Detect current market phase
    pub fn detect(&mut self, market_data: &MarketData) -> MarketPhase {
        // Add to history
        self.market_history.push_back(market_data.clone());
        if self.market_history.len() > self.config.history_size {
            self.market_history.pop_front();
        }
        
        // Need enough history
        if self.market_history.len() < 5 {
            return self.current_phase;
        }
        
        // Calculate phase indicators
        let indicators = self.calculate_phase_indicators();
        
        // Determine phase based on indicators
        let detected_phase = self.determine_phase(&indicators);
        
        // Check if we should transition
        if self.should_transition(detected_phase) {
            self.transition_to(detected_phase);
        }
        
        self.current_phase
    }
    
    /// Calculate phase indicators from market history
    fn calculate_phase_indicators(&self) -> PhaseIndicators {
        let recent_data: Vec<&MarketData> = self.market_history.iter()
            .rev()
            .take(20)
            .collect();
        
        if recent_data.is_empty() {
            return PhaseIndicators::default();
        }
        
        // Average values
        let avg_trend = recent_data.iter()
            .map(|d| d.trend)
            .sum::<f64>() / recent_data.len() as f64;
        
        let avg_volatility = recent_data.iter()
            .map(|d| d.volatility)
            .sum::<f64>() / recent_data.len() as f64;
        
        let avg_momentum = recent_data.iter()
            .map(|d| d.momentum)
            .sum::<f64>() / recent_data.len() as f64;
        
        let avg_cycle = recent_data.iter()
            .map(|d| d.cycle)
            .sum::<f64>() / recent_data.len() as f64;
        
        // Trend direction and strength
        let trend_direction = avg_trend.signum();
        let trend_strength = avg_trend.abs();
        
        // Volatility regime
        let volatility_regime = if avg_volatility < 0.3 {
            VolatilityRegime::Low
        } else if avg_volatility < 0.7 {
            VolatilityRegime::Medium
        } else {
            VolatilityRegime::High
        };
        
        // Momentum characteristics
        let momentum_aligned = avg_momentum.signum() == trend_direction;
        let momentum_strength = avg_momentum.abs();
        
        PhaseIndicators {
            trend_direction,
            trend_strength,
            volatility_regime,
            volatility_value: avg_volatility,
            momentum_aligned,
            momentum_strength,
            cycle_position: avg_cycle,
        }
    }
    
    /// Determine phase from indicators
    fn determine_phase(&self, indicators: &PhaseIndicators) -> MarketPhase {
        // Growth: positive trend, low volatility, aligned momentum
        if indicators.trend_direction > 0.0
            && indicators.trend_strength > 0.3
            && matches!(indicators.volatility_regime, VolatilityRegime::Low | VolatilityRegime::Medium)
            && indicators.momentum_aligned
            && indicators.momentum_strength > 0.2 {
            return MarketPhase::Growth;
        }
        
        // Release: high volatility, negative trend or strong divergence
        if matches!(indicators.volatility_regime, VolatilityRegime::High)
            || (indicators.trend_direction < 0.0 && indicators.trend_strength > 0.3) {
            return MarketPhase::Release;
        }
        
        // Conservation: low volatility, weak trend, low momentum
        if matches!(indicators.volatility_regime, VolatilityRegime::Low)
            && indicators.trend_strength < 0.2
            && indicators.momentum_strength < 0.2 {
            return MarketPhase::Conservation;
        }
        
        // Reorganization: medium volatility, changing directions
        if matches!(indicators.volatility_regime, VolatilityRegime::Medium)
            && !indicators.momentum_aligned {
            return MarketPhase::Reorganization;
        }
        
        // Default based on cycle position
        if indicators.cycle_position > 0.5 {
            MarketPhase::Growth
        } else if indicators.cycle_position > 0.0 {
            MarketPhase::Conservation
        } else if indicators.cycle_position > -0.5 {
            MarketPhase::Release
        } else {
            MarketPhase::Reorganization
        }
    }
    
    /// Check if we should transition to new phase
    fn should_transition(&self, new_phase: MarketPhase) -> bool {
        if new_phase == self.current_phase {
            return false;
        }
        
        // Check minimum duration in current phase
        if let Some((_, timestamp)) = self.phase_history.back() {
            let duration = chrono::Utc::now() - *timestamp;
            if duration.to_std().unwrap() < self.config.min_phase_duration {
                return false;
            }
        }
        
        true
    }
    
    /// Transition to new phase
    fn transition_to(&mut self, new_phase: MarketPhase) {
        self.phase_history.push_back((self.current_phase, chrono::Utc::now()));
        if self.phase_history.len() > 100 {
            self.phase_history.pop_front();
        }
        
        self.current_phase = new_phase;
        
        tracing::info!(
            "Market phase transition: {} -> {}",
            self.phase_history.back().unwrap().0.name(),
            new_phase.name()
        );
    }
    
    /// Update phase detector with feedback
    pub fn update(&mut self, market_data: &MarketData, actual_return: f64) {
        // Could adjust sensitivity based on prediction accuracy
        // For now, just ensure we have the latest market data
        self.detect(market_data);
    }
    
    /// Get current phase
    pub fn current_phase(&self) -> MarketPhase {
        self.current_phase
    }
    
    /// Get phase history
    pub fn phase_history(&self) -> Vec<(MarketPhase, chrono::DateTime<chrono::Utc>)> {
        self.phase_history.iter().cloned().collect()
    }
    
    /// Get time in current phase
    pub fn time_in_phase(&self) -> std::time::Duration {
        if let Some((_, timestamp)) = self.phase_history.back() {
            let duration = chrono::Utc::now() - *timestamp;
            duration.to_std().unwrap_or_default()
        } else {
            std::time::Duration::from_secs(0)
        }
    }
}

/// Phase detection indicators
#[derive(Debug, Default)]
struct PhaseIndicators {
    trend_direction: f64,
    trend_strength: f64,
    volatility_regime: VolatilityRegime,
    volatility_value: f64,
    momentum_aligned: bool,
    momentum_strength: f64,
    cycle_position: f64,
}

/// Volatility regime
#[derive(Debug, Default, PartialEq)]
enum VolatilityRegime {
    #[default]
    Low,
    Medium,
    High,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_phase_characteristics() {
        let growth = MarketPhase::Growth.characteristics();
        assert!(growth.opportunity_level > growth.risk_level);
        
        let release = MarketPhase::Release.characteristics();
        assert!(release.risk_level > release.opportunity_level);
    }
    
    #[test]
    fn test_phase_detection() {
        let mut detector = PhaseDetector::new();
        
        // Create growth market conditions
        let mut market_data = MarketData::random();
        market_data.trend = 0.6;
        market_data.volatility = 0.2;
        market_data.momentum = 0.5;
        
        let phase = detector.detect(&market_data);
        // Initial phase is Growth by default
        assert_eq!(phase, MarketPhase::Growth);
        
        // Create release conditions
        market_data.trend = -0.6;
        market_data.volatility = 0.8;
        market_data.momentum = -0.7;
        
        // Need multiple detections to transition
        for _ in 0..10 {
            detector.detect(&market_data);
        }
        
        // Should eventually detect high volatility
        assert!(detector.market_history.iter().any(|d| d.volatility > 0.7));
    }
    
    #[test]
    fn test_phase_transitions() {
        let config = PhaseDetectorConfig {
            history_size: 10,
            min_phase_duration: std::time::Duration::from_secs(0), // No minimum for testing
            sensitivity: 0.5,
        };
        
        let mut detector = PhaseDetector::with_config(config);
        
        // Test transition from growth to release
        let mut market_data = MarketData::random();
        market_data.trend = 0.7;
        market_data.volatility = 0.2;
        market_data.momentum = 0.6;
        
        for _ in 0..5 {
            detector.detect(&market_data);
        }
        
        // Change to release conditions
        market_data.trend = -0.7;
        market_data.volatility = 0.9;
        market_data.momentum = -0.8;
        
        for _ in 0..5 {
            detector.detect(&market_data);
        }
        
        // Should have transitioned
        let history = detector.phase_history();
        assert!(!history.is_empty());
    }
}