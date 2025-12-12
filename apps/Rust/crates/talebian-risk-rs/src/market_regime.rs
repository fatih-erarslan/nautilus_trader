//! # Market Regime Detection and Classification
//!
//! Advanced market regime detection for context-aware risk management.
//! Identifies bull/bear markets, volatility regimes, and special conditions
//! like whale activity periods and high-opportunity windows.

use crate::{MacchiavelianConfig, MarketData, TalebianRiskError, WhaleDetection};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[cfg(feature = "simd")]
use wide::f64x4;

/// Market regime classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MarketRegime {
    Bull {
        strength: f64,
        duration: u32,
        confidence: f64,
    },
    Bear {
        strength: f64,
        duration: u32,
        confidence: f64,
    },
    Sideways {
        volatility_level: VolatilityLevel,
        duration: u32,
    },
    Transition {
        from: Box<MarketRegime>,
        to: Box<MarketRegime>,
        progress: f64,
    },
    WhaleActive {
        whale_type: WhaleRegimeType,
        intensity: f64,
        duration: u32,
    },
    HighOpportunity {
        opportunity_factors: Vec<OpportunityFactor>,
        intensity: f64,
        estimated_duration: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VolatilityLevel {
    Low,     // < 1% daily
    Normal,  // 1-3% daily
    High,    // 3-5% daily
    Extreme, // > 5% daily
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WhaleRegimeType {
    Accumulation, // Stealth buying
    Distribution, // Stealth selling
    Momentum,     // Following trends
    Contrarian,   // Against trends
    Manipulation, // Price manipulation
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OpportunityFactor {
    HighVolatility,
    WhaleActivity,
    MomentumAlignment,
    LowCorrelation,
    VolatilityBreakout,
    RegimeTransition,
}

/// Comprehensive regime analysis
#[derive(Debug, Clone)]
pub struct RegimeAnalysis {
    pub current_regime: MarketRegime,
    pub regime_probability: f64,
    pub transition_probability: f64,
    pub regime_stability: f64,
    pub opportunity_score: f64,
    pub risk_level: f64,
    pub recommended_strategy: TradingStrategy,
    pub regime_duration: u32,
    pub next_regime_prediction: Option<MarketRegime>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TradingStrategy {
    Aggressive,    // High risk, high reward
    Moderate,      // Balanced approach
    Conservative,  // Risk-focused
    Defensive,     // Capital preservation
    Opportunistic, // Wait for better opportunities
    Parasitic,     // Follow whale movements
}

/// Market regime detection engine
pub struct MarketRegimeEngine {
    config: MacchiavelianConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    volatility_history: VecDeque<f64>,
    returns_history: VecDeque<f64>,
    regime_history: VecDeque<RegimeAnalysis>,
    current_regime: Option<MarketRegime>,
    regime_start_time: Option<i64>,
    whale_activity_periods: VecDeque<WhaleActivityPeriod>,
}

#[derive(Debug, Clone)]
struct WhaleActivityPeriod {
    start_time: i64,
    end_time: Option<i64>,
    whale_type: WhaleRegimeType,
    intensity: f64,
    cumulative_impact: f64,
}

impl MarketRegimeEngine {
    /// Create new market regime detection engine
    pub fn new(config: MacchiavelianConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::with_capacity(1000),
            volume_history: VecDeque::with_capacity(1000),
            volatility_history: VecDeque::with_capacity(1000),
            returns_history: VecDeque::with_capacity(1000),
            regime_history: VecDeque::with_capacity(100),
            current_regime: None,
            regime_start_time: None,
            whale_activity_periods: VecDeque::with_capacity(50),
        }
    }

    /// Detect and analyze current market regime
    pub fn analyze_regime(
        &mut self,
        market_data: &MarketData,
        whale_detection: &WhaleDetection,
    ) -> Result<RegimeAnalysis, TalebianRiskError> {
        // Update internal state
        self.update_history(market_data)?;
        self.update_whale_activity(market_data, whale_detection)?;

        // Detect current regime
        let detected_regime = self.detect_regime(market_data, whale_detection)?;

        // Calculate regime probabilities and stability
        let regime_probability = self.calculate_regime_probability(&detected_regime)?;
        let transition_probability = self.calculate_transition_probability(&detected_regime)?;
        let regime_stability = self.calculate_regime_stability(&detected_regime)?;

        // Calculate opportunity and risk scores
        let opportunity_score =
            self.calculate_opportunity_score(&detected_regime, whale_detection)?;
        let risk_level = self.calculate_risk_level(&detected_regime, market_data)?;

        // Determine recommended strategy
        let recommended_strategy =
            self.determine_trading_strategy(&detected_regime, opportunity_score, risk_level)?;

        // Calculate regime duration
        let regime_duration = self.calculate_average_regime_duration() as u32;

        // Predict next regime
        let next_regime_prediction = self.predict_next_regime(&detected_regime)?;

        // Update current regime
        if let Some(ref current) = self.current_regime {
            if !self.regimes_match(current, &detected_regime) {
                self.regime_start_time = Some(market_data.timestamp.timestamp());
            }
        } else {
            self.regime_start_time = Some(market_data.timestamp.timestamp());
        }
        self.current_regime = Some(detected_regime.clone());

        let analysis = RegimeAnalysis {
            current_regime: detected_regime,
            regime_probability,
            transition_probability,
            regime_stability,
            opportunity_score,
            risk_level,
            recommended_strategy,
            regime_duration,
            next_regime_prediction,
        };

        // Store analysis
        self.regime_history.push_back(analysis.clone());
        while self.regime_history.len() > 100 {
            self.regime_history.pop_front();
        }

        Ok(analysis)
    }

    /// Detect current market regime using multiple indicators
    fn detect_regime(
        &self,
        market_data: &MarketData,
        whale_detection: &WhaleDetection,
    ) -> Result<MarketRegime, TalebianRiskError> {
        // Priority 1: Check for whale activity regime
        if let Some(whale_regime) = self.detect_whale_regime(whale_detection)? {
            return Ok(whale_regime);
        }

        // Priority 2: Check for high opportunity conditions
        if let Some(opportunity_regime) =
            self.detect_opportunity_regime(market_data, whale_detection)?
        {
            return Ok(opportunity_regime);
        }

        // Priority 3: Detect trend-based regimes
        if let Some(trend_regime) = self.detect_trend_regime()? {
            return Ok(trend_regime);
        }

        // Default: Sideways market with volatility classification
        let volatility_level = self.classify_volatility(market_data.volatility);
        let duration = self.estimate_sideways_duration()?;

        Ok(MarketRegime::Sideways {
            volatility_level,
            duration,
        })
    }

    /// Detect whale-driven market regimes
    fn detect_whale_regime(
        &self,
        whale_detection: &WhaleDetection,
    ) -> Result<Option<MarketRegime>, TalebianRiskError> {
        if !whale_detection.is_whale_detected || whale_detection.confidence < 0.7 {
            return Ok(None);
        }

        // Determine whale regime type
        let whale_type = if whale_detection.confidence > 0.8
            && whale_detection.volume_spike < 0.5
        {
            WhaleRegimeType::Accumulation // Stealth accumulation
        } else if whale_detection.confidence > 0.8 && whale_detection.volume_spike < 0.5 {
            WhaleRegimeType::Distribution // Stealth distribution
        } else if whale_detection.volume_spike > 1.0 {
            // High volume whale activity
            match whale_detection.direction {
                crate::WhaleDirection::Buying => {
                    if self.is_momentum_aligned(true)? {
                        WhaleRegimeType::Momentum
                    } else {
                        WhaleRegimeType::Contrarian
                    }
                }
                crate::WhaleDirection::Selling => {
                    if self.is_momentum_aligned(false)? {
                        WhaleRegimeType::Momentum
                    } else {
                        WhaleRegimeType::Contrarian
                    }
                }
                crate::WhaleDirection::Neutral => WhaleRegimeType::Manipulation,
                crate::WhaleDirection::Mixed => WhaleRegimeType::Manipulation,
            }
        } else {
            WhaleRegimeType::Manipulation
        };

        let intensity = whale_detection.confidence * (1.0 + whale_detection.volume_spike);
        let duration = self.estimate_whale_duration(&whale_type)?;

        Ok(Some(MarketRegime::WhaleActive {
            whale_type,
            intensity,
            duration,
        }))
    }

    /// Detect high opportunity regimes
    fn detect_opportunity_regime(
        &self,
        market_data: &MarketData,
        whale_detection: &WhaleDetection,
    ) -> Result<Option<MarketRegime>, TalebianRiskError> {
        let mut opportunity_factors = Vec::new();
        let mut intensity_sum = 0.0;

        // Check for high volatility
        if market_data.volatility > 0.03 {
            opportunity_factors.push(OpportunityFactor::HighVolatility);
            intensity_sum += market_data.volatility * 10.0; // Scale to [0, 1]
        }

        // Check for whale activity
        if whale_detection.is_whale_detected && whale_detection.confidence > 0.6 {
            opportunity_factors.push(OpportunityFactor::WhaleActivity);
            intensity_sum += whale_detection.confidence;
        }

        // Check for momentum alignment
        if self.has_momentum_alignment()? {
            opportunity_factors.push(OpportunityFactor::MomentumAlignment);
            intensity_sum += 0.8;
        }

        // Check for volatility breakout
        if self.has_volatility_breakout(market_data)? {
            opportunity_factors.push(OpportunityFactor::VolatilityBreakout);
            intensity_sum += 0.9;
        }

        // Check for regime transition
        if self.is_regime_transition()? {
            opportunity_factors.push(OpportunityFactor::RegimeTransition);
            intensity_sum += 0.7;
        }

        // Require at least 2 opportunity factors for high opportunity regime
        if opportunity_factors.len() >= 2 {
            let intensity = intensity_sum / opportunity_factors.len() as f64;
            let estimated_duration = self.estimate_opportunity_duration(&opportunity_factors)?;

            Ok(Some(MarketRegime::HighOpportunity {
                opportunity_factors,
                intensity,
                estimated_duration,
            }))
        } else {
            Ok(None)
        }
    }

    /// Detect trend-based regimes (bull/bear)
    fn detect_trend_regime(&self) -> Result<Option<MarketRegime>, TalebianRiskError> {
        if self.returns_history.len() < 20 {
            return Ok(None);
        }

        // Calculate trend metrics
        let short_term_trend: f64 = self.returns_history.iter().rev().take(5).sum();
        let medium_term_trend: f64 = self.returns_history.iter().rev().take(20).sum();
        let long_term_trend: f64 = if self.returns_history.len() >= 50 {
            self.returns_history.iter().rev().take(50).sum()
        } else {
            medium_term_trend
        };

        // Calculate trend strength and consistency
        let trend_strength = (short_term_trend + medium_term_trend + long_term_trend).abs() / 3.0;
        let trend_consistency = self.calculate_trend_consistency()?;
        let trend_confidence = trend_strength * trend_consistency;

        // Determine trend direction
        let trend_direction = (short_term_trend + medium_term_trend + long_term_trend) / 3.0;

        // Trend thresholds (aggressive settings favor trend detection)
        let trend_threshold = 0.02; // 2% cumulative return
        let confidence_threshold = 0.6;

        if trend_direction > trend_threshold && trend_confidence > confidence_threshold {
            // Bull market
            let duration = self.estimate_trend_duration(true)?;
            Ok(Some(MarketRegime::Bull {
                strength: trend_strength,
                duration,
                confidence: trend_confidence,
            }))
        } else if trend_direction < -trend_threshold && trend_confidence > confidence_threshold {
            // Bear market
            let duration = self.estimate_trend_duration(false)?;
            Ok(Some(MarketRegime::Bear {
                strength: trend_strength,
                duration,
                confidence: trend_confidence,
            }))
        } else {
            Ok(None)
        }
    }

    /// Calculate regime probability
    fn calculate_regime_probability(
        &self,
        regime: &MarketRegime,
    ) -> Result<f64, TalebianRiskError> {
        let base_probability = match regime {
            MarketRegime::Bull { confidence, .. } => *confidence,
            MarketRegime::Bear { confidence, .. } => *confidence,
            MarketRegime::Sideways { .. } => 0.8, // High confidence in sideways detection
            MarketRegime::WhaleActive { intensity, .. } => intensity.min(0.95),
            MarketRegime::HighOpportunity { intensity, .. } => intensity.min(0.9),
            MarketRegime::Transition { progress, .. } => 0.5 + progress * 0.4,
        };

        // Adjust based on historical accuracy
        let historical_adjustment = self.calculate_historical_accuracy()?;
        let adjusted_probability = base_probability * historical_adjustment;

        Ok(adjusted_probability.max(0.1).min(0.95))
    }

    /// Calculate transition probability
    fn calculate_transition_probability(
        &self,
        current_regime: &MarketRegime,
    ) -> Result<f64, TalebianRiskError> {
        if self.regime_history.len() < 5 {
            return Ok(0.2); // Default moderate transition probability
        }

        // Analyze recent regime changes
        let _recent_transitions = self.count_recent_transitions()?;
        let regime_age = self.calculate_regime_age(current_regime)?;

        // Base transition probability
        let base_transition = match current_regime {
            MarketRegime::Bull { .. } | MarketRegime::Bear { .. } => {
                0.1 + (regime_age as f64 / 100.0).min(0.3) // Increases with age
            }
            MarketRegime::Sideways { .. } => 0.3, // Sideways markets are unstable
            MarketRegime::WhaleActive { .. } => 0.6, // Whale regimes are temporary
            MarketRegime::HighOpportunity { .. } => 0.8, // Opportunity windows are brief
            MarketRegime::Transition { progress, .. } => 1.0 - progress, // Decreases as transition completes
        };

        // Adjust for recent volatility
        let volatility_adjustment = if self.volatility_history.len() >= 10 {
            let recent_vol: f64 = self.volatility_history.iter().rev().take(10).sum::<f64>() / 10.0;
            (recent_vol / 0.02).min(2.0) // Higher volatility = higher transition probability
        } else {
            1.0
        };

        let transition_probability = base_transition * volatility_adjustment;

        Ok(transition_probability.max(0.05).min(0.95))
    }

    /// Calculate regime stability
    fn calculate_regime_stability(&self, regime: &MarketRegime) -> Result<f64, TalebianRiskError> {
        let base_stability = match regime {
            MarketRegime::Bull {
                strength,
                confidence,
                ..
            } => strength * confidence,
            MarketRegime::Bear {
                strength,
                confidence,
                ..
            } => strength * confidence,
            MarketRegime::Sideways {
                volatility_level, ..
            } => match volatility_level {
                VolatilityLevel::Low => 0.9,
                VolatilityLevel::Normal => 0.7,
                VolatilityLevel::High => 0.4,
                VolatilityLevel::Extreme => 0.2,
            },
            MarketRegime::WhaleActive { intensity, .. } => 0.8 - intensity * 0.3, // High intensity = lower stability
            MarketRegime::HighOpportunity { .. } => 0.3, // Opportunity regimes are unstable
            MarketRegime::Transition { progress, .. } => progress * 0.5, // Stability increases as transition completes
        };

        // Adjust for market conditions
        let volatility_penalty = if self.volatility_history.len() >= 5 {
            let avg_vol: f64 = self.volatility_history.iter().rev().take(5).sum::<f64>() / 5.0;
            1.0 - (avg_vol / 0.05).min(0.5) // Up to 50% penalty for high volatility
        } else {
            1.0
        };

        let stability = base_stability * volatility_penalty;

        Ok(stability.max(0.1).min(0.95))
    }

    /// Calculate opportunity score for current regime
    fn calculate_opportunity_score(
        &self,
        regime: &MarketRegime,
        whale_detection: &WhaleDetection,
    ) -> Result<f64, TalebianRiskError> {
        let base_score = match regime {
            MarketRegime::Bull { strength, .. } => 0.7 + strength * 0.2,
            MarketRegime::Bear { strength, .. } => 0.4 + strength * 0.2, // Bear markets have opportunities too
            MarketRegime::Sideways {
                volatility_level, ..
            } => {
                match volatility_level {
                    VolatilityLevel::Low => 0.2,
                    VolatilityLevel::Normal => 0.4,
                    VolatilityLevel::High => 0.7, // High volatility = opportunity
                    VolatilityLevel::Extreme => 0.9,
                }
            }
            MarketRegime::WhaleActive { intensity, .. } => 0.8 + intensity * 0.2,
            MarketRegime::HighOpportunity { intensity, .. } => 0.9 + intensity * 0.1,
            MarketRegime::Transition { .. } => 0.6, // Transitions create opportunities
        };

        // Whale detection bonus
        let whale_bonus = if whale_detection.is_whale_detected {
            whale_detection.confidence * 0.2
        } else {
            0.0
        };

        let opportunity_score = (base_score + whale_bonus).min(1.0);

        Ok(opportunity_score)
    }

    /// Calculate risk level for current regime
    fn calculate_risk_level(
        &self,
        regime: &MarketRegime,
        market_data: &MarketData,
    ) -> Result<f64, TalebianRiskError> {
        let base_risk = match regime {
            MarketRegime::Bull { .. } => 0.3, // Bull markets have moderate risk
            MarketRegime::Bear { .. } => 0.7, // Bear markets are risky
            MarketRegime::Sideways {
                volatility_level, ..
            } => match volatility_level {
                VolatilityLevel::Low => 0.2,
                VolatilityLevel::Normal => 0.4,
                VolatilityLevel::High => 0.6,
                VolatilityLevel::Extreme => 0.8,
            },
            MarketRegime::WhaleActive { .. } => 0.5, // Moderate risk with whales
            MarketRegime::HighOpportunity { .. } => 0.6, // High opportunity = higher risk
            MarketRegime::Transition { .. } => 0.8,  // Transitions are risky
        };

        // Volatility adjustment  
        let volatility_multiplier = (market_data.volatility / 0.02_f64).min(2.0_f64);
        let risk_level = (base_risk * volatility_multiplier).min(1.0_f64);

        Ok(risk_level)
    }

    /// Determine recommended trading strategy
    fn determine_trading_strategy(
        &self,
        regime: &MarketRegime,
        opportunity_score: f64,
        risk_level: f64,
    ) -> Result<TradingStrategy, TalebianRiskError> {
        let strategy = match regime {
            MarketRegime::Bull { confidence, .. } => {
                if *confidence > 0.8 && opportunity_score > 0.7 {
                    TradingStrategy::Aggressive
                } else {
                    TradingStrategy::Moderate
                }
            }
            MarketRegime::Bear { confidence, .. } => {
                if *confidence > 0.8 {
                    TradingStrategy::Defensive
                } else {
                    TradingStrategy::Conservative
                }
            }
            MarketRegime::Sideways {
                volatility_level, ..
            } => match volatility_level {
                VolatilityLevel::Low => TradingStrategy::Opportunistic,
                VolatilityLevel::Normal => TradingStrategy::Conservative,
                VolatilityLevel::High => TradingStrategy::Moderate,
                VolatilityLevel::Extreme => TradingStrategy::Aggressive,
            },
            MarketRegime::WhaleActive { whale_type, .. } => match whale_type {
                WhaleRegimeType::Accumulation | WhaleRegimeType::Distribution => {
                    TradingStrategy::Parasitic
                }
                WhaleRegimeType::Momentum => TradingStrategy::Aggressive,
                WhaleRegimeType::Contrarian => TradingStrategy::Moderate,
                WhaleRegimeType::Manipulation => TradingStrategy::Conservative,
            },
            MarketRegime::HighOpportunity { .. } => {
                if risk_level < 0.5 {
                    TradingStrategy::Aggressive
                } else {
                    TradingStrategy::Moderate
                }
            }
            MarketRegime::Transition { .. } => TradingStrategy::Conservative,
        };

        Ok(strategy)
    }

    // Helper methods

    fn update_history(&mut self, market_data: &MarketData) -> Result<(), TalebianRiskError> {
        self.price_history.push_back(market_data.price);
        self.volume_history.push_back(market_data.volume);
        self.volatility_history.push_back(market_data.volatility);

        if let Some(latest_return) = market_data.returns.last() {
            self.returns_history.push_back(*latest_return);
        }

        // Maintain history sizes
        while self.price_history.len() > 1000 {
            self.price_history.pop_front();
        }
        while self.volume_history.len() > 1000 {
            self.volume_history.pop_front();
        }
        while self.volatility_history.len() > 1000 {
            self.volatility_history.pop_front();
        }
        while self.returns_history.len() > 1000 {
            self.returns_history.pop_front();
        }

        Ok(())
    }

    fn update_whale_activity(
        &mut self,
        market_data: &MarketData,
        whale_detection: &WhaleDetection,
    ) -> Result<(), TalebianRiskError> {
        if whale_detection.is_whale_detected {
            // Check if we're continuing an existing whale period or starting a new one
            let whale_type = self.classify_whale_activity(whale_detection)?;

            let whale_types_compatible =
                if let Some(last_period) = self.whale_activity_periods.back() {
                    last_period.end_time.is_none()
                        && self.whale_types_compatible(&last_period.whale_type, &whale_type)
                } else {
                    false
                };

            if let Some(last_period) = self.whale_activity_periods.back_mut() {
                if whale_types_compatible {
                    // Continue existing period
                    last_period.intensity =
                        (last_period.intensity + whale_detection.confidence) / 2.0;
                    last_period.cumulative_impact += whale_detection.confidence * 0.1;
                } else {
                    // End previous period and start new one
                    last_period.end_time = Some(market_data.timestamp.timestamp());
                    self.start_new_whale_period(
                        market_data.timestamp.timestamp(),
                        whale_type,
                        whale_detection.confidence,
                    );
                }
            } else {
                // Start first whale period
                self.start_new_whale_period(
                    market_data.timestamp.timestamp(),
                    whale_type,
                    whale_detection.confidence,
                );
            }
        } else {
            // End current whale period if active
            if let Some(last_period) = self.whale_activity_periods.back_mut() {
                if last_period.end_time.is_none() {
                    last_period.end_time = Some(market_data.timestamp.timestamp());
                }
            }
        }

        // Maintain whale activity history
        while self.whale_activity_periods.len() > 50 {
            self.whale_activity_periods.pop_front();
        }

        Ok(())
    }

    fn start_new_whale_period(
        &mut self,
        timestamp: i64,
        whale_type: WhaleRegimeType,
        intensity: f64,
    ) {
        let period = WhaleActivityPeriod {
            start_time: timestamp,
            end_time: None,
            whale_type,
            intensity,
            cumulative_impact: intensity * 0.1,
        };
        self.whale_activity_periods.push_back(period);
    }

    fn classify_whale_activity(
        &self,
        whale_detection: &WhaleDetection,
    ) -> Result<WhaleRegimeType, TalebianRiskError> {
        if whale_detection.confidence > 0.8 && whale_detection.volume_spike < 0.5 {
            Ok(WhaleRegimeType::Accumulation)
        } else if whale_detection.confidence > 0.8 && whale_detection.volume_spike < 0.5 {
            Ok(WhaleRegimeType::Distribution)
        } else if whale_detection.volume_spike > 1.0 {
            match whale_detection.direction {
                crate::WhaleDirection::Buying | crate::WhaleDirection::Selling => {
                    Ok(WhaleRegimeType::Momentum)
                }
                crate::WhaleDirection::Neutral => Ok(WhaleRegimeType::Manipulation),
                crate::WhaleDirection::Mixed => Ok(WhaleRegimeType::Manipulation),
            }
        } else {
            Ok(WhaleRegimeType::Contrarian)
        }
    }

    fn whale_types_compatible(&self, type1: &WhaleRegimeType, type2: &WhaleRegimeType) -> bool {
        type1 == type2
            || (matches!(
                type1,
                WhaleRegimeType::Accumulation | WhaleRegimeType::Distribution
            ) && matches!(
                type2,
                WhaleRegimeType::Accumulation | WhaleRegimeType::Distribution
            ))
    }

    fn classify_volatility(&self, volatility: f64) -> VolatilityLevel {
        if volatility < 0.01 {
            VolatilityLevel::Low
        } else if volatility < 0.03 {
            VolatilityLevel::Normal
        } else if volatility < 0.05 {
            VolatilityLevel::High
        } else {
            VolatilityLevel::Extreme
        }
    }

    fn regimes_match(&self, regime1: &MarketRegime, regime2: &MarketRegime) -> bool {
        match (regime1, regime2) {
            (MarketRegime::Bull { .. }, MarketRegime::Bull { .. }) => true,
            (MarketRegime::Bear { .. }, MarketRegime::Bear { .. }) => true,
            (MarketRegime::Sideways { .. }, MarketRegime::Sideways { .. }) => true,
            (
                MarketRegime::WhaleActive { whale_type: t1, .. },
                MarketRegime::WhaleActive { whale_type: t2, .. },
            ) => self.whale_types_compatible(t1, t2),
            (MarketRegime::HighOpportunity { .. }, MarketRegime::HighOpportunity { .. }) => true,
            _ => false,
        }
    }

    // Additional helper methods for regime detection

    fn is_momentum_aligned(&self, bullish: bool) -> Result<bool, TalebianRiskError> {
        if self.returns_history.len() < 10 {
            return Ok(false);
        }

        let recent_returns: Vec<f64> = self
            .returns_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        let momentum = recent_returns.iter().sum::<f64>();

        if bullish {
            Ok(momentum > 0.01) // 1% positive momentum
        } else {
            Ok(momentum < -0.01) // 1% negative momentum
        }
    }

    fn has_momentum_alignment(&self) -> Result<bool, TalebianRiskError> {
        if self.returns_history.len() < 20 {
            return Ok(false);
        }

        let short_momentum: f64 = self.returns_history.iter().rev().take(5).sum();
        let medium_momentum: f64 = self.returns_history.iter().rev().take(15).sum();

        Ok((short_momentum > 0.0 && medium_momentum > 0.0)
            || (short_momentum < 0.0 && medium_momentum < 0.0))
    }

    fn has_volatility_breakout(&self, market_data: &MarketData) -> Result<bool, TalebianRiskError> {
        if self.volatility_history.len() < 20 {
            return Ok(false);
        }

        let avg_volatility: f64 =
            self.volatility_history.iter().sum::<f64>() / self.volatility_history.len() as f64;
        Ok(market_data.volatility > avg_volatility * 2.0)
    }

    fn is_regime_transition(&self) -> Result<bool, TalebianRiskError> {
        if self.regime_history.len() < 3 {
            return Ok(false);
        }

        let recent_regimes: Vec<&MarketRegime> = self
            .regime_history
            .iter()
            .rev()
            .take(3)
            .map(|analysis| &analysis.current_regime)
            .collect();

        // Check if regimes are different
        let all_different = !self.regimes_match(recent_regimes[0], recent_regimes[1])
            || !self.regimes_match(recent_regimes[1], recent_regimes[2]);

        Ok(all_different)
    }

    fn calculate_trend_consistency(&self) -> Result<f64, TalebianRiskError> {
        if self.returns_history.len() < 10 {
            return Ok(0.5);
        }

        let recent_returns: Vec<f64> = self
            .returns_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        let positive_count = recent_returns.iter().filter(|&&r| r > 0.0).count() as f64;
        let consistency = (positive_count / 10.0 - 0.5).abs() * 2.0;

        Ok(consistency)
    }

    fn calculate_historical_accuracy(&self) -> Result<f64, TalebianRiskError> {
        if self.regime_history.len() < 10 {
            return Ok(0.8); // Default moderate accuracy
        }

        // Simplified accuracy calculation
        // In a real implementation, this would compare predictions to actual outcomes
        let avg_confidence: f64 = self
            .regime_history
            .iter()
            .map(|analysis| analysis.regime_probability)
            .sum::<f64>()
            / self.regime_history.len() as f64;

        Ok(avg_confidence)
    }

    fn count_recent_transitions(&self) -> Result<u32, TalebianRiskError> {
        if self.regime_history.len() < 5 {
            return Ok(0);
        }

        let mut transitions = 0;
        let recent_regimes: Vec<&MarketRegime> = self
            .regime_history
            .iter()
            .rev()
            .take(5)
            .map(|analysis| &analysis.current_regime)
            .collect();

        for i in 1..recent_regimes.len() {
            if !self.regimes_match(recent_regimes[i - 1], recent_regimes[i]) {
                transitions += 1;
            }
        }

        Ok(transitions)
    }

    fn calculate_regime_age(&self, regime: &MarketRegime) -> Result<u32, TalebianRiskError> {
        match regime {
            MarketRegime::Bull { duration, .. }
            | MarketRegime::Bear { duration, .. }
            | MarketRegime::Sideways { duration, .. }
            | MarketRegime::WhaleActive { duration, .. }
            | MarketRegime::HighOpportunity {
                estimated_duration: duration,
                ..
            } => Ok(*duration),
            MarketRegime::Transition { .. } => Ok(1), // Transitions are always young
        }
    }

    // Duration estimation methods

    fn estimate_sideways_duration(&self) -> Result<u32, TalebianRiskError> {
        // Estimate based on volatility level and recent history
        Ok(20) // Default 20 periods
    }

    fn estimate_whale_duration(
        &self,
        whale_type: &WhaleRegimeType,
    ) -> Result<u32, TalebianRiskError> {
        let duration = match whale_type {
            WhaleRegimeType::Accumulation | WhaleRegimeType::Distribution => 50, // Longer stealth operations
            WhaleRegimeType::Momentum => 15, // Quick momentum following
            WhaleRegimeType::Contrarian => 10, // Brief contrarian moves
            WhaleRegimeType::Manipulation => 5, // Short manipulation periods
        };
        Ok(duration)
    }

    fn estimate_opportunity_duration(
        &self,
        factors: &[OpportunityFactor],
    ) -> Result<u32, TalebianRiskError> {
        let base_duration = 10;
        let mut duration_multiplier = 1.0;

        for factor in factors {
            match factor {
                OpportunityFactor::HighVolatility => duration_multiplier *= 1.5,
                OpportunityFactor::WhaleActivity => duration_multiplier *= 1.3,
                OpportunityFactor::MomentumAlignment => duration_multiplier *= 1.2,
                OpportunityFactor::LowCorrelation => duration_multiplier *= 0.8,
                OpportunityFactor::VolatilityBreakout => duration_multiplier *= 0.7,
                OpportunityFactor::RegimeTransition => duration_multiplier *= 0.6,
            }
        }

        Ok((base_duration as f64 * duration_multiplier) as u32)
    }

    fn estimate_trend_duration(&self, is_bull: bool) -> Result<u32, TalebianRiskError> {
        // Estimate based on trend strength and historical patterns
        let base_duration = if is_bull { 100 } else { 60 }; // Bull markets typically last longer

        // Adjust based on volatility
        let volatility_adjustment = if self.volatility_history.len() >= 10 {
            let avg_vol: f64 = self.volatility_history.iter().rev().take(10).sum::<f64>() / 10.0;
            1.0 / (1.0 + avg_vol * 5.0) // Higher volatility = shorter duration
        } else {
            1.0
        };

        Ok((base_duration as f64 * volatility_adjustment) as u32)
    }

    fn predict_next_regime(
        &self,
        current_regime: &MarketRegime,
    ) -> Result<Option<MarketRegime>, TalebianRiskError> {
        // Simplified next regime prediction
        match current_regime {
            MarketRegime::Bull { .. } => {
                if self.volatility_history.len() >= 5 {
                    let recent_vol: f64 =
                        self.volatility_history.iter().rev().take(5).sum::<f64>() / 5.0;
                    if recent_vol > 0.04 {
                        Ok(Some(MarketRegime::Sideways {
                            volatility_level: VolatilityLevel::High,
                            duration: 30,
                        }))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            MarketRegime::Bear { .. } => Ok(Some(MarketRegime::Sideways {
                volatility_level: VolatilityLevel::Normal,
                duration: 40,
            })),
            MarketRegime::Sideways { .. } => {
                if self.has_momentum_alignment()? {
                    Ok(Some(MarketRegime::Bull {
                        strength: 0.6,
                        duration: 80,
                        confidence: 0.7,
                    }))
                } else {
                    Ok(None)
                }
            }
            MarketRegime::WhaleActive { .. } => Ok(Some(MarketRegime::Sideways {
                volatility_level: VolatilityLevel::Normal,
                duration: 25,
            })),
            MarketRegime::HighOpportunity { .. } => Ok(Some(MarketRegime::Sideways {
                volatility_level: VolatilityLevel::High,
                duration: 15,
            })),
            MarketRegime::Transition { to, .. } => Ok(Some((**to).clone())),
        }
    }

    /// Get current regime analysis summary
    pub fn get_regime_summary(&self) -> RegimeSummary {
        RegimeSummary {
            current_regime: self.current_regime.clone(),
            regime_start_time: self.regime_start_time,
            recent_regimes: self.regime_history.iter().rev().take(5).cloned().collect(),
            whale_activity_periods: self
                .whale_activity_periods
                .iter()
                .rev()
                .take(3)
                .cloned()
                .collect(),
            total_regime_changes: self.regime_history.len(),
            average_regime_duration: self.calculate_average_regime_duration(),
        }
    }

    fn calculate_average_regime_duration(&self) -> f64 {
        if self.regime_history.len() < 2 {
            return 0.0;
        }

        let total_duration: u32 = self
            .regime_history
            .iter()
            .map(|analysis| analysis.regime_duration)
            .sum();

        total_duration as f64 / self.regime_history.len() as f64
    }
}

/// Summary of regime analysis
#[derive(Debug, Clone)]
pub struct RegimeSummary {
    pub current_regime: Option<MarketRegime>,
    pub regime_start_time: Option<i64>,
    pub recent_regimes: Vec<RegimeAnalysis>,
    pub whale_activity_periods: Vec<WhaleActivityPeriod>,
    pub total_regime_changes: usize,
    pub average_regime_duration: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MacchiavelianConfig, WhaleDirection};

    #[test]
    fn test_regime_engine_creation() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = MarketRegimeEngine::new(config);

        assert!(engine.current_regime.is_none());
        assert!(engine.regime_start_time.is_none());
    }

    #[test]
    fn test_volatility_classification() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = MarketRegimeEngine::new(config);

        assert_eq!(engine.classify_volatility(0.005), VolatilityLevel::Low);
        assert_eq!(engine.classify_volatility(0.02), VolatilityLevel::Normal);
        assert_eq!(engine.classify_volatility(0.04), VolatilityLevel::High);
        assert_eq!(engine.classify_volatility(0.08), VolatilityLevel::Extreme);
    }

    #[test]
    fn test_whale_regime_detection() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = MarketRegimeEngine::new(config);

        let whale_detection = WhaleDetection {
            is_whale_detected: true,
            confidence: 0.9,
            volume_anomaly: 0.3, // Low volume anomaly
            price_impact: 0.5,
            order_book_imbalance: 0.6,
            smart_money_flow: 0.9, // High smart money flow
            direction: WhaleDirection::Bullish,
        };

        let whale_regime = engine.detect_whale_regime(&whale_detection).unwrap();
        assert!(whale_regime.is_some());

        if let Some(MarketRegime::WhaleActive { whale_type, .. }) = whale_regime {
            assert_eq!(whale_type, WhaleRegimeType::Accumulation);
        }
    }

    #[test]
    fn test_regime_matching() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = MarketRegimeEngine::new(config);

        let bull1 = MarketRegime::Bull {
            strength: 0.8,
            duration: 50,
            confidence: 0.9,
        };
        let bull2 = MarketRegime::Bull {
            strength: 0.6,
            duration: 30,
            confidence: 0.7,
        };
        let bear = MarketRegime::Bear {
            strength: 0.7,
            duration: 40,
            confidence: 0.8,
        };

        assert!(engine.regimes_match(&bull1, &bull2));
        assert!(!engine.regimes_match(&bull1, &bear));
    }

    #[test]
    fn test_opportunity_regime_detection() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let mut engine = MarketRegimeEngine::new(config);

        // Add some momentum history
        for i in 0..25 {
            engine.returns_history.push_back(0.01); // Positive momentum
        }

        let market_data = MarketData {
            timestamp: 0,
            price: 100.0,
            volume: 1000.0,
            bid: 99.5,
            ask: 100.5,
            bid_volume: 500.0,
            ask_volume: 500.0,
            volatility: 0.04, // High volatility
            returns: vec![0.01],
            volume_history: vec![],
        };

        let whale_detection = WhaleDetection {
            is_whale_detected: true,
            confidence: 0.8,
            volume_anomaly: 0.7,
            price_impact: 0.6,
            order_book_imbalance: 0.6,
            smart_money_flow: 0.7,
            direction: WhaleDirection::Bullish,
        };

        let opportunity_regime = engine
            .detect_opportunity_regime(&market_data, &whale_detection)
            .unwrap();
        assert!(opportunity_regime.is_some());

        if let Some(MarketRegime::HighOpportunity {
            opportunity_factors,
            ..
        }) = opportunity_regime
        {
            assert!(opportunity_factors.contains(&OpportunityFactor::HighVolatility));
            assert!(opportunity_factors.contains(&OpportunityFactor::WhaleActivity));
        }
    }

    #[test]
    fn test_trading_strategy_determination() {
        let config = MacchiavelianConfig::aggressive_defaults();
        let engine = MarketRegimeEngine::new(config);

        // Test bull market strategy
        let bull_regime = MarketRegime::Bull {
            strength: 0.8,
            duration: 50,
            confidence: 0.9,
        };
        let strategy = engine
            .determine_trading_strategy(&bull_regime, 0.8, 0.3)
            .unwrap();
        assert_eq!(strategy, TradingStrategy::Aggressive);

        // Test whale accumulation strategy
        let whale_regime = MarketRegime::WhaleActive {
            whale_type: WhaleRegimeType::Accumulation,
            intensity: 0.8,
            duration: 30,
        };
        let strategy = engine
            .determine_trading_strategy(&whale_regime, 0.7, 0.4)
            .unwrap();
        assert_eq!(strategy, TradingStrategy::Parasitic);

        // Test high volatility sideways strategy
        let sideways_regime = MarketRegime::Sideways {
            volatility_level: VolatilityLevel::Extreme,
            duration: 20,
        };
        let strategy = engine
            .determine_trading_strategy(&sideways_regime, 0.6, 0.7)
            .unwrap();
        assert_eq!(strategy, TradingStrategy::Aggressive);
    }
}
