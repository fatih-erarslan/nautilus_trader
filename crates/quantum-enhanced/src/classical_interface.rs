//! Classical interface for quantum pattern results

use crate::types::*;
use crate::Result;

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use tracing::{debug, info};

/// Classical interface that converts quantum signals to trading-ready format
#[derive(Clone)]
pub struct ClassicalInterface {
    /// Configuration
    config: QuantumConfig,
    /// Signal conversion parameters
    conversion_params: ConversionParams,
}

#[derive(Debug, Clone)]
struct ConversionParams {
    /// Minimum confidence threshold for trading signals
    min_confidence_threshold: f64,
    /// Minimum coherence threshold for trading signals
    min_coherence_threshold: f64,
    /// Signal strength scaling factor
    signal_scaling_factor: f64,
    /// Maximum position size multiplier
    max_position_multiplier: f64,
}

/// Classical trading signal converted from quantum signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalTradingSignal {
    /// Signal ID
    pub id: String,
    /// Trading direction (Long/Short)
    pub direction: TradingDirection,
    /// Signal strength (0.0 to 1.0)
    pub strength: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Recommended position size multiplier
    pub position_size_multiplier: f64,
    /// Target instruments
    pub instruments: Vec<String>,
    /// Entry conditions
    pub entry_conditions: EntryConditions,
    /// Risk management parameters
    pub risk_parameters: RiskParameters,
    /// Expected hold time
    pub expected_hold_time_minutes: u32,
    /// Quantum metadata for analysis
    pub quantum_metadata: QuantumMetadata,
}

/// Entry conditions for the trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryConditions {
    /// Immediate entry (market order)
    pub immediate_entry: bool,
    /// Limit price offset from current price (percentage)
    pub limit_price_offset: Option<f64>,
    /// Maximum wait time for entry (minutes)
    pub max_wait_time_minutes: u32,
    /// Required volume confirmation
    pub volume_confirmation: bool,
}

/// Risk management parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskParameters {
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Maximum drawdown allowed
    pub max_drawdown_pct: f64,
    /// Position timeout (minutes)
    pub position_timeout_minutes: u32,
    /// Trailing stop enabled
    pub trailing_stop_enabled: bool,
}

/// Quantum metadata for signal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetadata {
    /// Original quantum pattern type
    pub pattern_type: String,
    /// Quantum coherence level
    pub coherence: f64,
    /// Entanglement strength
    pub entanglement_strength: f64,
    /// Dominant frequencies
    pub dominant_frequencies: Vec<f64>,
    /// Quantum validation score
    pub validation_score: f64,
}

impl ClassicalInterface {
    /// Create a new classical interface
    pub async fn new(config: QuantumConfig) -> Result<Self> {
        info!("Initializing Classical Interface for quantum signal conversion");

        let conversion_params = ConversionParams {
            min_confidence_threshold: config.coherence_threshold,
            min_coherence_threshold: config.coherence_threshold * 0.8,
            signal_scaling_factor: 1.0,
            max_position_multiplier: 2.0,
        };

        Ok(Self {
            config,
            conversion_params,
        })
    }

    /// Convert quantum signal to classical trading signal
    pub async fn convert_to_trading_signal(
        &self,
        quantum_signal: &QuantumSignal,
        current_market_data: &MarketData,
    ) -> Result<Option<ClassicalTradingSignal>> {
        
        debug!("Converting quantum signal {} to classical trading signal", quantum_signal.id);

        // Check if signal meets minimum thresholds
        if !self.meets_trading_thresholds(quantum_signal).await? {
            debug!("Quantum signal {} does not meet trading thresholds", quantum_signal.id);
            return Ok(None);
        }

        // Determine trading direction
        let direction = match quantum_signal.trading_direction() {
            Some(dir) => dir,
            None => {
                debug!("Quantum signal {} has no clear trading direction", quantum_signal.id);
                return Ok(None);
            }
        };

        // Calculate position size multiplier
        let position_size_multiplier = self.calculate_position_size_multiplier(quantum_signal).await?;

        // Generate entry conditions
        let entry_conditions = self.generate_entry_conditions(quantum_signal, current_market_data).await?;

        // Calculate risk parameters
        let risk_parameters = self.calculate_risk_parameters(quantum_signal, current_market_data).await?;

        // Estimate hold time
        let expected_hold_time_minutes = self.estimate_hold_time(quantum_signal).await?;

        // Create quantum metadata
        let quantum_metadata = QuantumMetadata {
            pattern_type: quantum_signal.pattern_type.to_string(),
            coherence: quantum_signal.coherence,
            entanglement_strength: quantum_signal.entanglement_map.values().sum::<f64>() / quantum_signal.entanglement_map.len().max(1) as f64,
            dominant_frequencies: quantum_signal.frequency_signature.to_vec(),
            validation_score: quantum_signal.confidence * quantum_signal.coherence,
        };

        let trading_signal = ClassicalTradingSignal {
            id: quantum_signal.id.to_string(),
            direction,
            strength: quantum_signal.strength.abs(),
            confidence: quantum_signal.confidence,
            position_size_multiplier,
            instruments: quantum_signal.affected_instruments.clone(),
            entry_conditions,
            risk_parameters,
            expected_hold_time_minutes,
            quantum_metadata,
        };

        info!("Generated classical trading signal: {} {} with strength {:.3}, confidence {:.3}",
              trading_signal.direction.to_string(), 
              trading_signal.instruments.join(","),
              trading_signal.strength, 
              trading_signal.confidence);

        Ok(Some(trading_signal))
    }

    /// Convert multiple quantum signals to portfolio allocation
    pub async fn convert_to_portfolio_allocation(
        &self,
        quantum_signals: &[QuantumSignal],
        current_market_data: &MarketData,
        total_capital: f64,
    ) -> Result<PortfolioAllocation> {
        
        let mut allocations = HashMap::new();
        let mut total_weight = 0.0;

        // Convert each quantum signal and calculate weights
        for quantum_signal in quantum_signals {
            if let Some(trading_signal) = self.convert_to_trading_signal(quantum_signal, current_market_data).await? {
                let weight = trading_signal.strength * trading_signal.confidence * trading_signal.position_size_multiplier;
                
                for instrument in &trading_signal.instruments {
                    let allocation = allocations.entry(instrument.clone()).or_insert(InstrumentAllocation {
                        instrument: instrument.clone(),
                        direction: trading_signal.direction,
                        weight: 0.0,
                        capital_allocation: 0.0,
                        risk_parameters: trading_signal.risk_parameters.clone(),
                        quantum_signals: Vec::new(),
                    });
                    
                    allocation.weight += weight;
                    allocation.quantum_signals.push(quantum_signal.id.to_string());
                    total_weight += weight;
                }
            }
        }

        // Normalize weights and calculate capital allocations
        if total_weight > 0.0 {
            for allocation in allocations.values_mut() {
                allocation.weight /= total_weight;
                allocation.capital_allocation = allocation.weight * total_capital;
            }
        }

        Ok(PortfolioAllocation {
            total_capital,
            allocations: allocations.into_values().collect(),
            rebalance_threshold: 0.05, // 5% threshold
            max_concentration: 0.25,    // 25% max per instrument
        })
    }

    /// Validate quantum signal for trading readiness
    pub async fn validate_signal_for_trading(
        &self,
        quantum_signal: &QuantumSignal,
        market_conditions: &MarketConditions,
    ) -> Result<SignalValidation> {
        
        let mut validation_issues = Vec::new();
        let mut validation_score = 1.0;

        // Check confidence threshold
        if quantum_signal.confidence < self.conversion_params.min_confidence_threshold {
            validation_issues.push(format!(
                "Confidence {:.3} below threshold {:.3}",
                quantum_signal.confidence,
                self.conversion_params.min_confidence_threshold
            ));
            validation_score *= 0.5;
        }

        // Check coherence threshold
        if quantum_signal.coherence < self.conversion_params.min_coherence_threshold {
            validation_issues.push(format!(
                "Coherence {:.3} below threshold {:.3}",
                quantum_signal.coherence,
                self.conversion_params.min_coherence_threshold
            ));
            validation_score *= 0.7;
        }

        // Check market conditions compatibility
        if market_conditions.volatility > 0.5 && quantum_signal.pattern_type == QuantumPatternType::CoherentOscillation {
            validation_issues.push("High volatility incompatible with coherent oscillation pattern".to_string());
            validation_score *= 0.8;
        }

        // Check instrument availability
        for instrument in &quantum_signal.affected_instruments {
            if !market_conditions.available_instruments.contains(instrument) {
                validation_issues.push(format!("Instrument {} not available for trading", instrument));
                validation_score *= 0.6;
            }
        }

        let is_valid = validation_score > 0.5 && validation_issues.len() < 3;

        Ok(SignalValidation {
            is_valid,
            validation_score,
            issues: validation_issues,
            recommendations: self.generate_validation_recommendations(quantum_signal, &validation_issues).await?,
        })
    }

    // Private helper methods

    async fn meets_trading_thresholds(&self, quantum_signal: &QuantumSignal) -> Result<bool> {
        Ok(quantum_signal.is_tradeable(
            self.conversion_params.min_confidence_threshold,
            self.conversion_params.min_coherence_threshold,
        ))
    }

    async fn calculate_position_size_multiplier(&self, quantum_signal: &QuantumSignal) -> Result<f64> {
        // Base multiplier from signal strength and confidence
        let base_multiplier = quantum_signal.strength.abs() * quantum_signal.confidence;
        
        // Adjust for quantum coherence
        let coherence_adjustment = 0.5 + 0.5 * quantum_signal.coherence;
        
        // Adjust for pattern type
        let pattern_adjustment = match quantum_signal.pattern_type {
            QuantumPatternType::SuperpositionMomentum => 1.2,
            QuantumPatternType::EntangledCorrelation => 1.1,
            QuantumPatternType::QuantumTunneling => 1.3,
            QuantumPatternType::QuantumResonance => 1.0,
            _ => 0.9,
        };

        let multiplier = base_multiplier * coherence_adjustment * pattern_adjustment;
        
        Ok(multiplier.min(self.conversion_params.max_position_multiplier).max(0.1))
    }

    async fn generate_entry_conditions(
        &self,
        quantum_signal: &QuantumSignal,
        market_data: &MarketData,
    ) -> Result<EntryConditions> {
        
        // Determine entry urgency based on coherence time
        let immediate_entry = quantum_signal.coherence > 0.8;
        
        // Calculate limit price offset based on signal strength
        let limit_price_offset = if immediate_entry {
            None
        } else {
            Some(quantum_signal.strength * 0.002) // 0.2% max offset
        };

        // Calculate max wait time based on pattern persistence
        let max_wait_time_minutes = match quantum_signal.pattern_type {
            QuantumPatternType::QuantumTunneling => 5,   // Fast execution
            QuantumPatternType::SuperpositionMomentum => 15,
            QuantumPatternType::EntangledCorrelation => 30,
            _ => 60,
        };

        // Volume confirmation for stronger signals
        let volume_confirmation = quantum_signal.confidence > 0.8;

        Ok(EntryConditions {
            immediate_entry,
            limit_price_offset,
            max_wait_time_minutes,
            volume_confirmation,
        })
    }

    async fn calculate_risk_parameters(
        &self,
        quantum_signal: &QuantumSignal,
        _market_data: &MarketData,
    ) -> Result<RiskParameters> {
        
        // Base risk parameters
        let base_stop_loss = 0.02; // 2%
        let base_take_profit = 0.04; // 4%

        // Adjust based on signal confidence
        let confidence_factor = quantum_signal.confidence;
        let stop_loss_pct = base_stop_loss * (2.0 - confidence_factor);
        let take_profit_pct = base_take_profit * (1.0 + confidence_factor);

        // Adjust based on coherence
        let coherence_factor = quantum_signal.coherence;
        let max_drawdown_pct = 0.05 * (2.0 - coherence_factor);

        // Pattern-specific adjustments
        let (timeout_minutes, trailing_stop) = match quantum_signal.pattern_type {
            QuantumPatternType::QuantumTunneling => (30, false),    // Quick reversal
            QuantumPatternType::SuperpositionMomentum => (120, true), // Momentum following
            QuantumPatternType::EntangledCorrelation => (240, true),  // Longer trends
            QuantumPatternType::CoherentOscillation => (60, false),   // Range-bound
            _ => (180, true),
        };

        Ok(RiskParameters {
            stop_loss_pct,
            take_profit_pct,
            max_drawdown_pct,
            position_timeout_minutes: timeout_minutes,
            trailing_stop_enabled: trailing_stop,
        })
    }

    async fn estimate_hold_time(&self, quantum_signal: &QuantumSignal) -> Result<u32> {
        // Base hold time from coherence (higher coherence = longer persistence)
        let base_hold_minutes = (quantum_signal.coherence * 240.0) as u32; // Up to 4 hours

        // Adjust for pattern type
        let pattern_multiplier = match quantum_signal.pattern_type {
            QuantumPatternType::QuantumTunneling => 0.5,        // Quick reversals
            QuantumPatternType::SuperpositionMomentum => 1.5,   // Momentum persists
            QuantumPatternType::EntangledCorrelation => 2.0,    // Long-term relationships
            QuantumPatternType::CoherentOscillation => 0.8,     // Oscillating patterns
            QuantumPatternType::QuantumResonance => 1.2,        // Resonance effects
            _ => 1.0,
        };

        let estimated_minutes = (base_hold_minutes as f64 * pattern_multiplier) as u32;
        
        Ok(estimated_minutes.max(5).min(480)) // Min 5 minutes, max 8 hours
    }

    async fn generate_validation_recommendations(
        &self,
        quantum_signal: &QuantumSignal,
        issues: &[String],
    ) -> Result<Vec<String>> {
        
        let mut recommendations = Vec::new();

        if quantum_signal.confidence < self.conversion_params.min_confidence_threshold {
            recommendations.push("Wait for higher confidence quantum signal or reduce position size".to_string());
        }

        if quantum_signal.coherence < self.conversion_params.min_coherence_threshold {
            recommendations.push("Consider ensemble approach with multiple quantum models".to_string());
        }

        if issues.iter().any(|i| i.contains("volatility")) {
            recommendations.push("Apply volatility-adjusted position sizing".to_string());
        }

        if issues.iter().any(|i| i.contains("not available")) {
            recommendations.push("Use alternative highly correlated instruments".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Signal meets all validation criteria - proceed with trading".to_string());
        }

        Ok(recommendations)
    }
}

/// Portfolio allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioAllocation {
    pub total_capital: f64,
    pub allocations: Vec<InstrumentAllocation>,
    pub rebalance_threshold: f64,
    pub max_concentration: f64,
}

/// Individual instrument allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentAllocation {
    pub instrument: String,
    pub direction: TradingDirection,
    pub weight: f64,
    pub capital_allocation: f64,
    pub risk_parameters: RiskParameters,
    pub quantum_signals: Vec<String>,
}

/// Market conditions for signal validation
#[derive(Debug, Clone)]
pub struct MarketConditions {
    pub volatility: f64,
    pub liquidity: f64,
    pub available_instruments: Vec<String>,
    pub market_regime: String,
}

/// Signal validation result
#[derive(Debug, Clone)]
pub struct SignalValidation {
    pub is_valid: bool,
    pub validation_score: f64,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

impl std::fmt::Display for TradingDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradingDirection::Long => write!(f, "LONG"),
            TradingDirection::Short => write!(f, "SHORT"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use chrono::Utc;
    use ndarray::Array1;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_signal_conversion() {
        let config = QuantumConfig::default();
        let interface = ClassicalInterface::new(config).await.unwrap();

        let quantum_signal = QuantumSignal {
            id: Uuid::new_v4(),
            strength: 0.8,
            confidence: 0.9,
            pattern_type: QuantumPatternType::SuperpositionMomentum,
            coherence: 0.85,
            entanglement_map: HashMap::new(),
            frequency_signature: Array1::zeros(0),
            timestamp: Utc::now(),
            execution_time_us: 50,
            affected_instruments: vec!["BTCUSDT".to_string()],
            metadata: HashMap::new(),
        };

        let market_data = MarketData {
            price_history: HashMap::new(),
            volume_data: HashMap::new(),
            timestamps: vec![Utc::now()],
            features: ndarray::Array2::zeros((1, 1)),
            regime_indicators: Array1::zeros(1),
        };

        let trading_signal = interface.convert_to_trading_signal(&quantum_signal, &market_data).await.unwrap();
        
        assert!(trading_signal.is_some());
        let signal = trading_signal.unwrap();
        assert_eq!(signal.direction, TradingDirection::Long);
        assert!(signal.confidence > 0.0);
        assert!(signal.strength > 0.0);
    }

    #[tokio::test]
    async fn test_portfolio_allocation() {
        let config = QuantumConfig::default();
        let interface = ClassicalInterface::new(config).await.unwrap();

        let quantum_signals = vec![
            QuantumSignal::new(0.5, 0.8, QuantumPatternType::SuperpositionMomentum, 0.7),
            QuantumSignal::new(-0.3, 0.75, QuantumPatternType::EntangledCorrelation, 0.8),
        ];

        let market_data = MarketData {
            price_history: HashMap::new(),
            volume_data: HashMap::new(),
            timestamps: vec![Utc::now()],
            features: ndarray::Array2::zeros((1, 1)),
            regime_indicators: Array1::zeros(1),
        };

        let allocation = interface.convert_to_portfolio_allocation(
            &quantum_signals, 
            &market_data, 
            100000.0
        ).await.unwrap();

        assert_eq!(allocation.total_capital, 100000.0);
        assert!(!allocation.allocations.is_empty());
    }

    #[tokio::test]
    async fn test_signal_validation() {
        let config = QuantumConfig::default();
        let interface = ClassicalInterface::new(config).await.unwrap();

        let mut quantum_signal = QuantumSignal::new(0.8, 0.9, QuantumPatternType::SuperpositionMomentum, 0.85);
        quantum_signal.affected_instruments = vec!["BTCUSDT".to_string()];

        let market_conditions = MarketConditions {
            volatility: 0.3,
            liquidity: 0.8,
            available_instruments: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            market_regime: "trending".to_string(),
        };

        let validation = interface.validate_signal_for_trading(&quantum_signal, &market_conditions).await.unwrap();
        
        assert!(validation.is_valid);
        assert!(validation.validation_score > 0.5);
        assert!(!validation.recommendations.is_empty());
    }
}