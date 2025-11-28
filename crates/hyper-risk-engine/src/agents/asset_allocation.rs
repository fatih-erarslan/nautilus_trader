//! Asset allocation agent for portfolio optimization.
//!
//! Operates in the medium path (<1ms) to compute optimal asset allocations
//! based on risk-return objectives and market conditions.
//!
//! ## ML-Enhanced Allocation (requires `ml` feature)
//!
//! When the `ml` feature is enabled, this agent can use neural forecasters:
//! - **N-HiTS**: Hierarchical interpolation for return forecasts
//! - **DeepAR**: Probabilistic autoregressive model for volatility forecasts with uncertainty
//!
//! ## Scientific References
//!
//! - Challu et al. (2022): "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting"
//! - Salinas et al. (2020): "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"
//! - Markowitz (1952): "Portfolio Selection" Journal of Finance

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::core::types::{
    MarketRegime, Portfolio, RiskDecision, Symbol, Timestamp,
};
use crate::core::error::Result;

use super::base::{Agent, AgentConfig, AgentId, AgentStats, AgentStatus};

// Conditional imports for ML-enhanced allocation
#[cfg(feature = "ml")]
use hyperphysics_ml::prelude::{
    Backend, Device, Tensor, TensorOps,
    NHits, NHitsConfig,
    DeepAR, DeepARConfig,
};
#[cfg(feature = "ml")]
use hyperphysics_ml::models::{Forecaster, ForecastOutput};

/// Configuration for the asset allocation agent.
#[derive(Debug, Clone)]
pub struct AssetAllocationConfig {
    /// Base agent configuration.
    pub base: AgentConfig,
    /// Target volatility for the portfolio.
    pub target_volatility: f64,
    /// Maximum weight for any single asset.
    pub max_single_weight: f64,
    /// Minimum weight for any single asset.
    pub min_single_weight: f64,
    /// Rebalancing threshold (deviation from target).
    pub rebalance_threshold: f64,
    /// Risk-free rate for Sharpe calculations.
    pub risk_free_rate: f64,
    // ============================================================================
    // ML Forecasting Configuration (requires `ml` feature)
    // ============================================================================
    /// Enable ML-based return forecasting.
    pub use_ml_forecasts: bool,
    /// Forecast horizon in periods.
    pub forecast_horizon: usize,
    /// Input sequence length for forecasters.
    pub forecast_input_length: usize,
    /// Confidence level for volatility intervals (0.0 to 1.0).
    pub forecast_confidence: f32,
    /// Weight given to ML forecasts vs historical (0.0 = historical only, 1.0 = ML only).
    pub ml_forecast_weight: f64,
}

impl Default for AssetAllocationConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "asset_allocation_agent".to_string(),
                enabled: true,
                priority: 2,
                max_latency_us: 1000, // 1ms
                verbose: false,
            },
            target_volatility: 0.15,
            max_single_weight: 0.25,
            min_single_weight: 0.0,
            rebalance_threshold: 0.05,
            risk_free_rate: 0.05,
            // ML forecasting defaults
            use_ml_forecasts: true,
            forecast_horizon: 24,        // 24 periods ahead
            forecast_input_length: 96,   // 96 periods lookback
            forecast_confidence: 0.95,   // 95% confidence intervals
            ml_forecast_weight: 0.7,     // 70% ML, 30% historical
        }
    }
}

/// Target allocation for an asset.
#[derive(Debug, Clone)]
pub struct AssetTarget {
    /// Symbol of the asset.
    pub symbol: Symbol,
    /// Target weight (0.0 to 1.0).
    pub target_weight: f64,
    /// Current weight.
    pub current_weight: f64,
    /// Deviation from target.
    pub deviation: f64,
    /// Recommended action.
    pub action: AllocationAction,
}

/// Recommended allocation action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationAction {
    /// Increase position.
    Increase,
    /// Decrease position.
    Decrease,
    /// Hold current position.
    Hold,
    /// Close position entirely.
    Close,
}

/// Asset allocation recommendation.
#[derive(Debug, Clone)]
pub struct AllocationRecommendation {
    /// Individual asset targets.
    pub targets: Vec<AssetTarget>,
    /// Portfolio-level metrics.
    pub expected_return: f64,
    /// Expected portfolio volatility.
    pub expected_volatility: f64,
    /// Expected Sharpe ratio.
    pub expected_sharpe: f64,
    /// Recommendation timestamp.
    pub generated_at: Timestamp,
    /// Whether rebalancing is recommended.
    pub needs_rebalance: bool,
}

/// Asset return and risk statistics.
#[derive(Debug, Clone)]
pub struct AssetStats {
    /// Expected return.
    pub expected_return: f64,
    /// Volatility (annualized).
    pub volatility: f64,
    /// Correlation with other assets (simplified).
    pub beta: f64,
}

/// ML-forecasted return and volatility with uncertainty.
///
/// Combines N-HiTS point forecasts with DeepAR uncertainty estimates.
#[cfg(feature = "ml")]
#[derive(Debug, Clone)]
pub struct MLForecast {
    /// Symbol being forecast.
    pub symbol: Symbol,
    /// Point forecast of expected return (from N-HiTS).
    pub expected_return: f64,
    /// Point forecast of volatility (from DeepAR mean).
    pub volatility: f64,
    /// Lower bound of volatility at configured confidence level.
    pub volatility_lower: f64,
    /// Upper bound of volatility at configured confidence level.
    pub volatility_upper: f64,
    /// Forecast horizon used.
    pub horizon: usize,
    /// Timestamp of forecast generation.
    pub generated_at: Timestamp,
}

/// Historical price data for ML forecasting.
#[cfg(feature = "ml")]
#[derive(Debug, Clone)]
pub struct PriceHistory {
    /// Symbol.
    pub symbol: Symbol,
    /// OHLCV data: [open, high, low, close, volume] per period.
    pub ohlcv: Vec<[f64; 5]>,
    /// Timestamps for each period.
    pub timestamps: Vec<Timestamp>,
}

/// Asset allocation agent.
pub struct AssetAllocationAgent {
    config: AssetAllocationConfig,
    status: AtomicU8,
    stats: AgentStats,
    /// Asset statistics.
    asset_stats: RwLock<HashMap<Symbol, AssetStats>>,
    /// Current target allocation.
    current_targets: RwLock<HashMap<Symbol, f64>>,
    /// Latest recommendation.
    latest_recommendation: RwLock<Option<AllocationRecommendation>>,
    // ============================================================================
    // ML Forecasting Components (requires `ml` feature)
    // ============================================================================
    /// N-HiTS model for return forecasting.
    #[cfg(feature = "ml")]
    nhits_model: RwLock<Option<NHits>>,
    /// DeepAR model for volatility forecasting with uncertainty.
    #[cfg(feature = "ml")]
    deepar_model: RwLock<Option<DeepAR>>,
    /// Cached ML forecasts per symbol.
    #[cfg(feature = "ml")]
    ml_forecasts: RwLock<HashMap<Symbol, MLForecast>>,
    /// Price history for ML input.
    #[cfg(feature = "ml")]
    price_history: RwLock<HashMap<Symbol, PriceHistory>>,
    /// ML compute device.
    #[cfg(feature = "ml")]
    ml_device: Device,
}

// Manual Debug impl since NHits and DeepAR don't derive Debug properly
impl std::fmt::Debug for AssetAllocationAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssetAllocationAgent")
            .field("config", &self.config)
            .field("status", &self.status)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl AssetAllocationAgent {
    /// Create a new asset allocation agent.
    pub fn new(config: AssetAllocationConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            stats: AgentStats::new(),
            asset_stats: RwLock::new(HashMap::new()),
            current_targets: RwLock::new(HashMap::new()),
            latest_recommendation: RwLock::new(None),
            #[cfg(feature = "ml")]
            nhits_model: RwLock::new(None),
            #[cfg(feature = "ml")]
            deepar_model: RwLock::new(None),
            #[cfg(feature = "ml")]
            ml_forecasts: RwLock::new(HashMap::new()),
            #[cfg(feature = "ml")]
            price_history: RwLock::new(HashMap::new()),
            #[cfg(feature = "ml")]
            ml_device: Device::Cpu,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(AssetAllocationConfig::default())
    }

    // ============================================================================
    // ML Forecasting Methods (requires `ml` feature)
    // ============================================================================

    /// Initialize ML forecasters (N-HiTS for returns, DeepAR for volatility).
    ///
    /// This must be called before using ML-enhanced allocation.
    /// Models are initialized on CPU by default; use `set_ml_device()` to change.
    #[cfg(feature = "ml")]
    pub fn init_ml_forecasters(&self) -> Result<()> {
        // Initialize N-HiTS for return forecasting
        let nhits_config = NHitsConfig::new(
            self.config.forecast_input_length,
            self.config.forecast_horizon,
            5, // OHLCV features
        );

        let nhits = NHits::new(nhits_config, &self.ml_device)
            .map_err(|e| crate::core::error::RiskError::ConfigurationError(
                format!("Failed to create N-HiTS model: {}", e)
            ))?;

        *self.nhits_model.write() = Some(nhits);

        // Initialize DeepAR for volatility forecasting with uncertainty
        let deepar_config = DeepARConfig::new(
            self.config.forecast_input_length,
            self.config.forecast_horizon,
        ).with_hidden_size(128)
         .with_num_layers(2);

        let deepar = DeepAR::new(deepar_config, &self.ml_device)
            .map_err(|e| crate::core::error::RiskError::ConfigurationError(
                format!("Failed to create DeepAR model: {}", e)
            ))?;

        *self.deepar_model.write() = Some(deepar);

        tracing::info!(
            horizon = self.config.forecast_horizon,
            input_length = self.config.forecast_input_length,
            "ML forecasters initialized (N-HiTS + DeepAR)"
        );

        Ok(())
    }

    /// Check if ML forecasters are initialized.
    #[cfg(feature = "ml")]
    pub fn is_ml_initialized(&self) -> bool {
        self.nhits_model.read().is_some() && self.deepar_model.read().is_some()
    }

    /// Update price history for a symbol.
    ///
    /// The agent maintains a rolling window of price data for ML forecasting.
    #[cfg(feature = "ml")]
    pub fn update_price_history(&self, history: PriceHistory) {
        self.price_history.write().insert(history.symbol.clone(), history);
    }

    /// Generate ML forecasts for a symbol using N-HiTS (returns) and DeepAR (volatility).
    ///
    /// Returns forecasted return and volatility with uncertainty bounds.
    #[cfg(feature = "ml")]
    pub fn forecast_ml(&self, symbol: &Symbol) -> Result<Option<MLForecast>> {
        if !self.config.use_ml_forecasts || !self.is_ml_initialized() {
            return Ok(None);
        }

        let history_guard = self.price_history.read();
        let history = match history_guard.get(symbol) {
            Some(h) if h.ohlcv.len() >= self.config.forecast_input_length => h,
            _ => return Ok(None), // Not enough data
        };

        // Prepare input tensor [1, seq_len, 5] for OHLCV
        let seq_len = self.config.forecast_input_length;
        let start_idx = history.ohlcv.len().saturating_sub(seq_len);
        let input_data: Vec<f32> = history.ohlcv[start_idx..]
            .iter()
            .flat_map(|ohlcv| ohlcv.iter().map(|&v| v as f32))
            .collect();

        let input = Tensor::from_slice(&input_data, vec![1, seq_len, 5], &self.ml_device)
            .map_err(|e| crate::core::error::RiskError::ConfigurationError(
                format!("Failed to create input tensor: {}", e)
            ))?;

        // N-HiTS for return forecast
        let return_forecast = {
            let nhits_guard = self.nhits_model.read();
            let nhits = nhits_guard.as_ref().unwrap();
            nhits.forecast(&input)
                .map_err(|e| crate::core::error::RiskError::ConfigurationError(
                    format!("N-HiTS forecast failed: {}", e)
                ))?
        };

        // DeepAR for volatility with uncertainty
        let volatility_forecast = {
            let deepar_guard = self.deepar_model.read();
            let deepar = deepar_guard.as_ref().unwrap();
            deepar.forecast_with_intervals(&input, self.config.forecast_confidence)
                .map_err(|e| crate::core::error::RiskError::ConfigurationError(
                    format!("DeepAR forecast failed: {}", e)
                ))?
        };

        // Extract scalar values from forecasts
        let expected_return = self.extract_mean_forecast(&return_forecast)?;
        let (volatility, vol_lower, vol_upper) = self.extract_volatility_forecast(&volatility_forecast)?;

        let forecast = MLForecast {
            symbol: symbol.clone(),
            expected_return,
            volatility,
            volatility_lower: vol_lower,
            volatility_upper: vol_upper,
            horizon: self.config.forecast_horizon,
            generated_at: Timestamp::now(),
        };

        // Cache the forecast
        self.ml_forecasts.write().insert(symbol.clone(), forecast.clone());

        Ok(Some(forecast))
    }

    /// Extract mean forecast value from N-HiTS output.
    #[cfg(feature = "ml")]
    fn extract_mean_forecast(&self, forecast: &Tensor) -> Result<f64> {
        if let Some(data) = forecast.as_slice() {
            // Average over horizon for expected return
            let sum: f32 = data.iter().sum();
            let mean = sum / data.len().max(1) as f32;
            // Convert to annualized return (simplified)
            Ok(mean as f64 * 252.0 / self.config.forecast_horizon as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Extract volatility forecast with bounds from DeepAR output.
    #[cfg(feature = "ml")]
    fn extract_volatility_forecast(&self, forecast: &ForecastOutput) -> Result<(f64, f64, f64)> {
        let point = if let Some(data) = forecast.point.as_slice() {
            let sum: f32 = data.iter().sum();
            let mean = sum / data.len().max(1) as f32;
            // Annualize volatility
            (mean.abs() as f64 * (252.0_f64).sqrt())
        } else {
            self.config.target_volatility // Fallback
        };

        let lower = if let Some(ref lower_tensor) = forecast.lower {
            if let Some(data) = lower_tensor.as_slice() {
                let sum: f32 = data.iter().sum();
                let mean = sum / data.len().max(1) as f32;
                (mean.abs() as f64 * (252.0_f64).sqrt())
            } else {
                point * 0.8
            }
        } else {
            point * 0.8 // 20% lower bound if not available
        };

        let upper = if let Some(ref upper_tensor) = forecast.upper {
            if let Some(data) = upper_tensor.as_slice() {
                let sum: f32 = data.iter().sum();
                let mean = sum / data.len().max(1) as f32;
                (mean.abs() as f64 * (252.0_f64).sqrt())
            } else {
                point * 1.2
            }
        } else {
            point * 1.2 // 20% upper bound if not available
        };

        Ok((point, lower, upper))
    }

    /// Get cached ML forecast for a symbol.
    #[cfg(feature = "ml")]
    pub fn get_ml_forecast(&self, symbol: &Symbol) -> Option<MLForecast> {
        self.ml_forecasts.read().get(symbol).cloned()
    }

    /// Blend historical stats with ML forecasts.
    ///
    /// Uses configured `ml_forecast_weight` to combine historical and ML estimates.
    #[cfg(feature = "ml")]
    pub fn blend_forecasts(&self, symbol: &Symbol) -> Option<AssetStats> {
        let historical = self.asset_stats.read().get(symbol).cloned()?;

        let ml_forecast = match self.ml_forecasts.read().get(symbol) {
            Some(f) => f.clone(),
            None => return Some(historical),
        };

        let weight = self.config.ml_forecast_weight;

        Some(AssetStats {
            expected_return: weight * ml_forecast.expected_return
                           + (1.0 - weight) * historical.expected_return,
            volatility: weight * ml_forecast.volatility
                      + (1.0 - weight) * historical.volatility,
            beta: historical.beta, // Keep historical beta
        })
    }

    /// Update asset statistics.
    pub fn update_asset_stats(&self, symbol: Symbol, stats: AssetStats) {
        self.asset_stats.write().insert(symbol, stats);
    }

    /// Set target allocation for an asset.
    pub fn set_target(&self, symbol: Symbol, weight: f64) {
        let clamped = weight.clamp(self.config.min_single_weight, self.config.max_single_weight);
        self.current_targets.write().insert(symbol, clamped);
    }

    /// Get the latest allocation recommendation.
    pub fn get_recommendation(&self) -> Option<AllocationRecommendation> {
        self.latest_recommendation.read().clone()
    }

    /// Compute optimal allocation based on current portfolio and targets.
    fn compute_allocation(&self, portfolio: &Portfolio) -> AllocationRecommendation {
        let targets = self.current_targets.read();
        let asset_stats = self.asset_stats.read();

        // Calculate current weights
        let total_value = portfolio.total_value;
        let mut current_weights: HashMap<Symbol, f64> = HashMap::new();

        if total_value > 0.0 {
            for position in &portfolio.positions {
                let position_value = position.market_value();
                current_weights.insert(position.symbol.clone(), position_value / total_value);
            }
        }

        // Build allocation targets
        let mut allocation_targets = Vec::new();
        let mut needs_rebalance = false;

        for (symbol, &target_weight) in targets.iter() {
            let current_weight = current_weights.get(symbol).copied().unwrap_or(0.0);
            let deviation = current_weight - target_weight;

            let action = if deviation.abs() > self.config.rebalance_threshold {
                needs_rebalance = true;
                if deviation > 0.0 {
                    AllocationAction::Decrease
                } else {
                    AllocationAction::Increase
                }
            } else if target_weight == 0.0 && current_weight > 0.0 {
                needs_rebalance = true;
                AllocationAction::Close
            } else {
                AllocationAction::Hold
            };

            allocation_targets.push(AssetTarget {
                symbol: symbol.clone(),
                target_weight,
                current_weight,
                deviation,
                action,
            });
        }

        // Calculate expected portfolio metrics
        let (expected_return, expected_volatility) = self.calculate_portfolio_metrics(&targets, &asset_stats);
        let expected_sharpe = if expected_volatility > 0.0 {
            (expected_return - self.config.risk_free_rate) / expected_volatility
        } else {
            0.0
        };

        AllocationRecommendation {
            targets: allocation_targets,
            expected_return,
            expected_volatility,
            expected_sharpe,
            generated_at: Timestamp::now(),
            needs_rebalance,
        }
    }

    /// Calculate expected portfolio return and volatility.
    fn calculate_portfolio_metrics(
        &self,
        weights: &HashMap<Symbol, f64>,
        asset_stats: &HashMap<Symbol, AssetStats>,
    ) -> (f64, f64) {
        let mut expected_return = 0.0;
        let mut variance = 0.0;

        for (symbol, &weight) in weights.iter() {
            if let Some(stats) = asset_stats.get(symbol) {
                expected_return += weight * stats.expected_return;
                // Simplified variance calculation (ignoring correlations)
                variance += (weight * stats.volatility).powi(2);
            }
        }

        (expected_return, variance.sqrt())
    }

    /// Convert u8 to AgentStatus.
    fn status_from_u8(value: u8) -> AgentStatus {
        match value {
            0 => AgentStatus::Idle,
            1 => AgentStatus::Processing,
            2 => AgentStatus::Paused,
            3 => AgentStatus::Error,
            4 => AgentStatus::ShuttingDown,
            _ => AgentStatus::Error,
        }
    }
}

impl Agent for AssetAllocationAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, portfolio: &Portfolio, _regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        // Compute allocation recommendation
        let recommendation = self.compute_allocation(portfolio);
        *self.latest_recommendation.write() = Some(recommendation);

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_cycle(latency_ns);
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);

        Ok(None)
    }

    fn start(&self) -> Result<()> {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        self.status.store(AgentStatus::ShuttingDown as u8, Ordering::Relaxed);
        Ok(())
    }

    fn pause(&self) {
        self.status.store(AgentStatus::Paused as u8, Ordering::Relaxed);
    }

    fn resume(&self) {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
    }

    fn process_count(&self) -> u64 {
        self.stats.cycles.load(Ordering::Relaxed)
    }

    fn avg_latency_ns(&self) -> u64 {
        self.stats.avg_latency_ns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_allocation_agent_creation() {
        let agent = AssetAllocationAgent::with_defaults();
        assert_eq!(agent.status(), AgentStatus::Idle);
        assert_eq!(agent.process_count(), 0);
    }

    #[test]
    fn test_target_setting() {
        let agent = AssetAllocationAgent::with_defaults();

        let symbol = Symbol::new("AAPL");
        agent.set_target(symbol.clone(), 0.20);

        let targets = agent.current_targets.read();
        assert_eq!(targets.get(&symbol), Some(&0.20));
    }

    #[test]
    fn test_weight_clamping() {
        let agent = AssetAllocationAgent::with_defaults();

        let symbol = Symbol::new("AAPL");
        // Try to set weight above max (0.25)
        agent.set_target(symbol.clone(), 0.50);

        let targets = agent.current_targets.read();
        assert_eq!(targets.get(&symbol), Some(&0.25));
    }

    #[test]
    fn test_allocation_computation() {
        let agent = AssetAllocationAgent::with_defaults();
        agent.start().unwrap();

        let symbol = Symbol::new("AAPL");
        agent.set_target(symbol.clone(), 0.20);
        agent.update_asset_stats(symbol, AssetStats {
            expected_return: 0.10,
            volatility: 0.20,
            beta: 1.0,
        });

        let portfolio = Portfolio::default();
        agent.process(&portfolio, MarketRegime::SidewaysLow).unwrap();

        let recommendation = agent.get_recommendation();
        assert!(recommendation.is_some());
    }

    #[test]
    fn test_agent_lifecycle() {
        let agent = AssetAllocationAgent::with_defaults();

        agent.start().unwrap();
        assert_eq!(agent.status(), AgentStatus::Idle);

        agent.pause();
        assert_eq!(agent.status(), AgentStatus::Paused);

        agent.resume();
        assert_eq!(agent.status(), AgentStatus::Idle);

        agent.stop().unwrap();
        assert_eq!(agent.status(), AgentStatus::ShuttingDown);
    }

    #[test]
    fn test_config_ml_defaults() {
        let config = AssetAllocationConfig::default();
        assert!(config.use_ml_forecasts);
        assert_eq!(config.forecast_horizon, 24);
        assert_eq!(config.forecast_input_length, 96);
        assert!((config.forecast_confidence - 0.95).abs() < 0.01);
        assert!((config.ml_forecast_weight - 0.7).abs() < 0.01);
    }

    #[cfg(feature = "ml")]
    #[test]
    fn test_ml_forecasters_not_initialized() {
        let agent = AssetAllocationAgent::with_defaults();
        assert!(!agent.is_ml_initialized());
    }

    #[cfg(feature = "ml")]
    #[test]
    fn test_price_history_update() {
        let agent = AssetAllocationAgent::with_defaults();
        let symbol = Symbol::new("AAPL");

        let history = PriceHistory {
            symbol: symbol.clone(),
            ohlcv: vec![[100.0, 101.0, 99.0, 100.5, 1000.0]; 100],
            timestamps: (0..100).map(|i| Timestamp::from_nanos(i * 1_000_000_000)).collect(),
        };

        agent.update_price_history(history);

        let stored = agent.price_history.read();
        assert!(stored.contains_key(&symbol));
        assert_eq!(stored.get(&symbol).unwrap().ohlcv.len(), 100);
    }

    #[cfg(feature = "ml")]
    #[test]
    fn test_blend_forecasts_no_ml() {
        let agent = AssetAllocationAgent::with_defaults();
        let symbol = Symbol::new("AAPL");

        agent.update_asset_stats(symbol.clone(), AssetStats {
            expected_return: 0.10,
            volatility: 0.20,
            beta: 1.0,
        });

        // Without ML forecast, should return historical stats
        let blended = agent.blend_forecasts(&symbol);
        assert!(blended.is_some());
        let stats = blended.unwrap();
        assert!((stats.expected_return - 0.10).abs() < 0.01);
    }
}
