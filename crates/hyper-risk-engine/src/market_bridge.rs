//! Market Data Bridge - Integration layer between hyperphysics-market and hyper-risk-engine
//!
//! This module provides a production-ready bridge that connects market data providers
//! to the regime detection agent for real-time market regime classification.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   Market Data Bridge                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │  MarketDataProvider (any) ──► MarketDataBridge             │
//! │                                      │                      │
//! │                                      ▼                      │
//! │                           Data Quality Validation           │
//! │                                      │                      │
//! │                                      ▼                      │
//! │                           Bar → Observation                 │
//! │                                      │                      │
//! │                                      ▼                      │
//! │                         RegimeDetectionAgent                │
//! │                                      │                      │
//! │                                      ▼                      │
//! │                              MarketRegime                   │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **Zero-copy data conversion**: Efficient Bar to observation transformation
//! - **Data quality validation**: Automatic detection of invalid/stale data
//! - **Multiple provider support**: Works with any `MarketDataProvider` implementation
//! - **Async/await**: Non-blocking integration suitable for high-frequency trading
//! - **Comprehensive error handling**: Production-grade error types and recovery
//!
//! # Example
//!
//! ```rust,no_run
//! use hyper_risk_engine::market_bridge::MarketDataBridge;
//! use hyper_risk_engine::agents::RegimeDetectionAgent;
//! use hyperphysics_market::providers::{AlpacaProvider, MarketDataProvider};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create market data provider
//! let provider = AlpacaProvider::new("API_KEY".to_string(), "API_SECRET".to_string(), true);
//!
//! // Create regime detection agent
//! let regime_agent = RegimeDetectionAgent::new();
//!
//! // Create bridge
//! let mut bridge = MarketDataBridge::new(Box::new(provider), regime_agent);
//!
//! // Fetch latest data and detect regime
//! let regime = bridge.update_regime("AAPL").await?;
//! println!("Current market regime: {:?}", regime);
//! # Ok(())
//! # }
//! ```
//!
//! # Scientific References
//!
//! - Hamilton (1989): "A New Approach to the Economic Analysis of Nonstationary Time Series"
//! - Engle & Sheppard (2001): "Theoretical and Empirical Properties of DCC"
//! - Ang & Bekaert (2002): "Regime Switches in Interest Rates"

use chrono::{Duration, Utc};
use thiserror::Error;

use crate::agents::RegimeDetectionAgent;
use crate::core::MarketRegime;
use hyperphysics_market::data::Bar;
use hyperphysics_market::providers::MarketDataProvider;

// ============================================================================
// Agent Configuration
// ============================================================================

/// Re-export regime detection configuration from agents module
pub use crate::agents::regime_detection::RegimeDetectionConfig;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur in the market data bridge
#[derive(Error, Debug)]
pub enum BridgeError {
    /// Market data provider error
    #[error("Market data provider error: {0}")]
    ProviderError(String),

    /// Invalid data detected (zero/negative prices, etc.)
    #[error("Invalid market data: {0}")]
    InvalidData(String),

    /// Stale data detected (data older than threshold)
    #[error("Stale market data: {0}")]
    StaleData(String),

    /// Regime detection failure
    #[error("Regime detection failed: {0}")]
    RegimeDetectionError(String),

    /// Insufficient data for regime detection
    #[error("Insufficient data: need at least {0} observations")]
    InsufficientData(usize),

    /// Symbol not supported by provider
    #[error("Symbol not supported: {0}")]
    UnsupportedSymbol(String),
}

/// Type alias for Result with BridgeError
pub type BridgeResult<T> = Result<T, BridgeError>;

// ============================================================================
// Data Quality Validation
// ============================================================================

/// Data quality validator for market data
#[derive(Debug, Clone)]
pub struct DataQualityValidator {
    /// Maximum allowed age for data (seconds)
    max_data_age_secs: i64,
    /// Minimum allowed price (to detect invalid data)
    min_price: f64,
    /// Maximum allowed price change percentage (to detect anomalies)
    max_price_change_pct: f64,
}

impl Default for DataQualityValidator {
    fn default() -> Self {
        Self {
            max_data_age_secs: 300,  // 5 minutes
            min_price: 0.0001,       // Minimum valid price
            max_price_change_pct: 50.0, // 50% max change
        }
    }
}

impl DataQualityValidator {
    /// Create new validator with custom parameters
    pub fn new(max_data_age_secs: i64, min_price: f64, max_price_change_pct: f64) -> Self {
        Self {
            max_data_age_secs,
            min_price,
            max_price_change_pct,
        }
    }

    /// Validate a single bar
    pub fn validate_bar(&self, bar: &Bar) -> BridgeResult<()> {
        // Check for valid prices
        if bar.open <= 0.0 || bar.high <= 0.0 || bar.low <= 0.0 || bar.close <= 0.0 {
            return Err(BridgeError::InvalidData(
                format!("Non-positive prices detected in bar: {:?}", bar)
            ));
        }

        // Check minimum price threshold
        if bar.low < self.min_price {
            return Err(BridgeError::InvalidData(
                format!("Price below minimum threshold: {} < {}", bar.low, self.min_price)
            ));
        }

        // Check OHLC consistency (high must be highest, low must be lowest)
        if bar.high < bar.low || bar.high < bar.open || bar.high < bar.close {
            return Err(BridgeError::InvalidData(
                format!("Invalid OHLC relationship: high={}, low={}, open={}, close={}",
                    bar.high, bar.low, bar.open, bar.close)
            ));
        }

        if bar.low > bar.open || bar.low > bar.close {
            return Err(BridgeError::InvalidData(
                format!("Low price exceeds open/close: low={}, open={}, close={}",
                    bar.low, bar.open, bar.close)
            ));
        }

        // Volume is u64, so it's always non-negative
        // Just validate it's not suspiciously large or zero for liquid markets
        if bar.volume == 0 {
            // Zero volume is suspicious for active trading periods but may be valid
            // for low liquidity or after-hours trading
            tracing::warn!("Zero volume detected for bar at {:?}", bar.timestamp);
        }

        // Check data freshness
        let now = Utc::now();
        let age = now.signed_duration_since(bar.timestamp);
        if age.num_seconds() > self.max_data_age_secs {
            return Err(BridgeError::StaleData(
                format!("Data is {} seconds old (max: {})",
                    age.num_seconds(), self.max_data_age_secs)
            ));
        }

        Ok(())
    }

    /// Validate a sequence of bars
    pub fn validate_bars(&self, bars: &[Bar]) -> BridgeResult<()> {
        if bars.is_empty() {
            return Err(BridgeError::InsufficientData(1));
        }

        // Validate each bar individually
        for bar in bars {
            self.validate_bar(bar)?;
        }

        // Check for chronological ordering
        for window in bars.windows(2) {
            if window[0].timestamp > window[1].timestamp {
                return Err(BridgeError::InvalidData(
                    "Bars are not in chronological order".to_string()
                ));
            }
        }

        // Check for suspicious price changes
        for window in bars.windows(2) {
            let prev = &window[0];
            let curr = &window[1];
            let change_pct = ((curr.close - prev.close) / prev.close).abs() * 100.0;

            if change_pct > self.max_price_change_pct {
                return Err(BridgeError::InvalidData(
                    format!("Suspicious price change: {:.2}% (max: {:.2}%)",
                        change_pct, self.max_price_change_pct)
                ));
            }
        }

        Ok(())
    }
}

// ============================================================================
// Market Data Bridge
// ============================================================================

/// Market data bridge connecting providers to regime detection
///
/// This is the main integration component that:
/// 1. Fetches data from any `MarketDataProvider`
/// 2. Validates data quality
/// 3. Converts bars to observations
/// 4. Feeds to `RegimeDetectionAgent`
/// 5. Returns detected market regime
pub struct MarketDataBridge {
    /// Market data provider
    provider: Box<dyn MarketDataProvider>,
    /// Regime detection agent
    regime_agent: RegimeDetectionAgent,
    /// Data quality validator
    validator: DataQualityValidator,
    /// Minimum number of observations needed for regime detection
    min_observations: usize,
}

impl MarketDataBridge {
    /// Create new market data bridge
    ///
    /// # Arguments
    ///
    /// * `provider` - Market data provider implementation
    /// * `regime_agent` - Regime detection agent
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use hyper_risk_engine::market_bridge::{MarketDataBridge, RegimeDetectionConfig};
    /// use hyper_risk_engine::agents::RegimeDetectionAgent;
    /// use hyperphysics_market::providers::AlpacaProvider;
    ///
    /// let provider = AlpacaProvider::new("key".to_string(), "secret".to_string(), true);
    /// let config = RegimeDetectionConfig::default();
    /// let agent = RegimeDetectionAgent::new(config);
    /// let bridge = MarketDataBridge::new(Box::new(provider), agent);
    /// ```
    pub fn new(
        provider: Box<dyn MarketDataProvider>,
        regime_agent: RegimeDetectionAgent,
    ) -> Self {
        Self {
            provider,
            regime_agent,
            validator: DataQualityValidator::default(),
            min_observations: 20, // Minimum for HMM stability
        }
    }

    /// Create new bridge with default regime detection configuration
    pub fn with_default_config(provider: Box<dyn MarketDataProvider>) -> Self {
        let config = RegimeDetectionConfig::default();
        let regime_agent = RegimeDetectionAgent::new(config);
        Self::new(provider, regime_agent)
    }

    /// Create bridge with custom validator
    pub fn with_validator(
        provider: Box<dyn MarketDataProvider>,
        regime_agent: RegimeDetectionAgent,
        validator: DataQualityValidator,
    ) -> Self {
        Self {
            provider,
            regime_agent,
            validator,
            min_observations: 20,
        }
    }

    /// Set minimum observations required for regime detection
    pub fn set_min_observations(&mut self, min: usize) {
        self.min_observations = min.max(5); // Enforce absolute minimum
    }

    /// Get provider name
    pub fn provider_name(&self) -> &str {
        self.provider.provider_name()
    }

    /// Check if symbol is supported
    pub async fn supports_symbol(&self, symbol: &str) -> BridgeResult<bool> {
        self.provider
            .supports_symbol(symbol)
            .await
            .map_err(|e| BridgeError::ProviderError(e.to_string()))
    }

    /// Fetch latest bar and update regime
    ///
    /// This is a convenience method for real-time regime detection
    /// using the most recent market data.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol
    ///
    /// # Returns
    ///
    /// Current market regime
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use hyper_risk_engine::market_bridge::MarketDataBridge;
    /// # async fn example(mut bridge: MarketDataBridge) -> Result<(), Box<dyn std::error::Error>> {
    /// let regime = bridge.update_regime("BTCUSD").await?;
    /// println!("Regime: {:?}", regime);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn update_regime(&mut self, symbol: &str) -> BridgeResult<MarketRegime> {
        // Check symbol support
        if !self.supports_symbol(symbol).await? {
            return Err(BridgeError::UnsupportedSymbol(symbol.to_string()));
        }

        // Fetch latest bar
        let bar = self.provider
            .fetch_latest_bar(symbol)
            .await
            .map_err(|e| BridgeError::ProviderError(e.to_string()))?;

        // Validate data quality
        self.validator.validate_bar(&bar)?;

        // Convert bar to observation and feed to agent
        let (log_return, volatility) = self.bar_to_observation_components(&bar);

        // Add observation to regime agent
        self.regime_agent.add_observation(log_return, volatility);

        // Get current regime
        let regime = self.regime_agent.get_regime();

        Ok(regime)
    }

    /// Fetch historical bars and perform regime detection
    ///
    /// This method fetches historical data, validates it, and uses it
    /// to initialize/update the regime detection agent.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol
    /// * `lookback_days` - Number of days of historical data
    ///
    /// # Returns
    ///
    /// Vector of detected regimes (one per bar)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use hyper_risk_engine::market_bridge::MarketDataBridge;
    /// # async fn example(mut bridge: MarketDataBridge) -> Result<(), Box<dyn std::error::Error>> {
    /// let regimes = bridge.detect_regime_history("AAPL", 30).await?;
    /// println!("Detected {} regime changes", regimes.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn detect_regime_history(
        &mut self,
        symbol: &str,
        lookback_days: i64,
    ) -> BridgeResult<Vec<MarketRegime>> {
        // Check symbol support
        if !self.supports_symbol(symbol).await? {
            return Err(BridgeError::UnsupportedSymbol(symbol.to_string()));
        }

        // Calculate time range
        let end = Utc::now();
        let start = end - Duration::days(lookback_days);

        // Fetch historical bars (using 1-hour timeframe for better granularity)
        let bars = self.provider
            .fetch_bars(
                symbol,
                hyperphysics_market::data::Timeframe::Hour1,
                start,
                end,
            )
            .await
            .map_err(|e| BridgeError::ProviderError(e.to_string()))?;

        // Validate sufficient data
        if bars.len() < self.min_observations {
            return Err(BridgeError::InsufficientData(self.min_observations));
        }

        // Validate data quality
        self.validator.validate_bars(&bars)?;

        // Feed all bars to regime agent and collect regimes
        let mut regimes = Vec::with_capacity(bars.len());
        for bar in &bars {
            let (log_return, volatility) = self.bar_to_observation_components(bar);
            self.regime_agent.add_observation(log_return, volatility);
            let regime = self.regime_agent.get_regime();
            regimes.push(regime);
        }

        Ok(regimes)
    }

    /// Convert Bar to observation components for regime detection
    ///
    /// Extracts key features from market bar:
    /// - Log return (close-to-open)
    /// - Volatility proxy (high-low range)
    ///
    /// # Arguments
    ///
    /// * `bar` - Market bar data
    ///
    /// # Returns
    ///
    /// Tuple of (log_return, volatility)
    fn bar_to_observation_components(&self, bar: &Bar) -> (f64, f64) {
        // Calculate log return (more stable than linear return)
        let log_return = (bar.close / bar.open).ln();

        // Calculate normalized range as volatility proxy (Parkinson estimator)
        // Using high-low range provides better volatility estimate than close-to-close
        let range = (bar.high / bar.low).ln();
        let parkinson_vol = range / (4.0 * (2.0_f64.ln()).sqrt());

        (log_return, parkinson_vol)
    }

    /// Get current regime without updating
    pub fn current_regime(&self) -> MarketRegime {
        self.regime_agent.get_regime()
    }

    /// Get current regime probabilities
    pub fn current_probabilities(&self) -> crate::agents::regime_detection::RegimeProbabilities {
        self.regime_agent.get_probabilities()
    }

    /// Reset regime detection state
    pub fn reset(&mut self) {
        // Create new regime agent with default configuration
        let config = RegimeDetectionConfig::default();
        self.regime_agent = RegimeDetectionAgent::new(config);
    }
}

// ============================================================================
// Async Batch Processing
// ============================================================================

/// Batch process multiple symbols concurrently
///
/// This is a utility function for efficient multi-symbol regime detection.
///
/// # Arguments
///
/// * `bridges` - Vector of market data bridges (one per symbol)
/// * `symbols` - Symbols to process
///
/// # Returns
///
/// Vector of (symbol, regime) pairs
///
/// # Example
///
/// ```rust,no_run
/// use hyper_risk_engine::market_bridge::{MarketDataBridge, batch_detect_regimes};
///
/// # async fn example(bridge: MarketDataBridge) -> Result<(), Box<dyn std::error::Error>> {
/// let symbols = vec!["AAPL", "MSFT", "GOOGL"];
/// let bridges = vec![bridge]; // In practice, create one per symbol
///
/// let results = batch_detect_regimes(bridges, &symbols).await;
/// for (symbol, regime) in results {
///     println!("{}: {:?}", symbol, regime);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn batch_detect_regimes(
    mut bridges: Vec<MarketDataBridge>,
    symbols: &[&str],
) -> Vec<(String, BridgeResult<MarketRegime>)> {
    use futures::future::join_all;

    // Create futures for each symbol
    let futures = symbols
        .iter()
        .zip(bridges.iter_mut())
        .map(|(symbol, bridge)| async move {
            let regime = bridge.update_regime(symbol).await;
            (symbol.to_string(), regime)
        });

    // Execute all concurrently
    join_all(futures).await
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_quality_validator() {
        let validator = DataQualityValidator::default();

        // Valid bar
        let valid_bar = Bar {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: 1000,
            vwap: None,
            trade_count: None,
        };
        assert!(validator.validate_bar(&valid_bar).is_ok());

        // Invalid: negative price
        let invalid_bar = Bar {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            open: -100.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: 1000,
            vwap: None,
            trade_count: None,
        };
        assert!(validator.validate_bar(&invalid_bar).is_err());

        // Invalid: OHLC inconsistency
        let invalid_bar = Bar {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            open: 100.0,
            high: 95.0, // High less than open
            low: 99.0,
            close: 103.0,
            volume: 1000,
            vwap: None,
            trade_count: None,
        };
        assert!(validator.validate_bar(&invalid_bar).is_err());
    }

    #[test]
    fn test_bar_to_observation() {
        use chrono::Utc;

        let bar = Bar {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: 1000,
            vwap: None,
            trade_count: None,
        };

        // Create a minimal mock for testing
        use crate::agents::RegimeDetectionAgent;
        use async_trait::async_trait;

        // Mock provider is not needed for this test
        struct MockProvider;

        #[async_trait]
        impl MarketDataProvider for MockProvider {
            async fn fetch_bars(
                &self,
                _symbol: &str,
                _timeframe: hyperphysics_market::data::Timeframe,
                _start: chrono::DateTime<chrono::Utc>,
                _end: chrono::DateTime<chrono::Utc>,
            ) -> hyperphysics_market::error::MarketResult<Vec<Bar>> {
                unimplemented!()
            }

            async fn fetch_latest_bar(&self, _symbol: &str) -> hyperphysics_market::error::MarketResult<Bar> {
                unimplemented!()
            }

            fn provider_name(&self) -> &str {
                "mock"
            }

            async fn supports_symbol(&self, _symbol: &str) -> hyperphysics_market::error::MarketResult<bool> {
                Ok(true)
            }
        }

        let config = RegimeDetectionConfig::default();
        let bridge = MarketDataBridge::new(
            Box::new(MockProvider),
            RegimeDetectionAgent::new(config),
        );

        let (log_return, volatility) = bridge.bar_to_observation_components(&bar);

        // Log return should be reasonable
        assert!(log_return.abs() < 1.0); // Less than 100% return

        // Volatility should be positive
        assert!(volatility > 0.0);
    }
}
