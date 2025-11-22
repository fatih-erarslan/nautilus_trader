//! Signal generation and technical analysis for Tengri trading strategy
//! 
//! Provides comprehensive technical analysis indicators, pattern recognition,
//! and signal generation using multiple timeframes and cross-asset analysis.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};
use statrs::statistics::Statistics;

use crate::{Result, TengriError};
use crate::config::{StrategyParameters, TrendFollowingParams, MeanReversionParams, MomentumParams};
use crate::types::{TradingSignal, SignalType, PriceData, MarketState, MarketRegime, VolatilityMetrics};

/// Signal generator for technical analysis and trading signals
pub struct SignalGenerator {
    config: StrategyParameters,
    price_history: Arc<RwLock<HashMap<String, VecDeque<PriceData>>>>,
    signal_cache: Arc<RwLock<HashMap<String, TradingSignal>>>,
    market_state: Arc<RwLock<MarketState>>,
    volatility_cache: Arc<RwLock<HashMap<String, VolatilityMetrics>>>,
}

/// Technical indicator calculations
pub struct TechnicalIndicators;

/// Pattern recognition system
pub struct PatternRecognizer;

/// Multi-timeframe analysis
pub struct MultiTimeframeAnalyzer {
    timeframes: Vec<u64>,
    lookback_periods: HashMap<u64, usize>,
}

/// Cross-asset correlation analyzer
pub struct CorrelationAnalyzer {
    correlation_window: usize,
    correlation_matrix: Arc<RwLock<Array2<f64>>>,
    asset_returns: Arc<RwLock<HashMap<String, VecDeque<f64>>>>,
}

impl SignalGenerator {
    /// Create new signal generator with configuration
    pub fn new(config: StrategyParameters) -> Self {
        Self {
            config,
            price_history: Arc::new(RwLock::new(HashMap::new())),
            signal_cache: Arc::new(RwLock::new(HashMap::new())),
            market_state: Arc::new(RwLock::new(MarketState::default())),
            volatility_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Update price history and generate new signals
    pub async fn update_price(&mut self, price_data: PriceData) -> Result<Option<TradingSignal>> {
        // Update price history
        {
            let mut history = self.price_history.write().await;
            let symbol_history = history
                .entry(price_data.symbol.clone())
                .or_insert_with(VecDeque::new);
            
            symbol_history.push_back(price_data.clone());
            
            // Limit history size for memory management
            while symbol_history.len() > self.config.lookback_period as usize * 2 {
                symbol_history.pop_front();
            }
        }

        // Generate signal for this symbol
        let signal = self.generate_signal(&price_data.symbol).await?;
        
        // Cache the signal if generated
        if let Some(ref signal) = signal {
            let mut cache = self.signal_cache.write().await;
            cache.insert(price_data.symbol.clone(), signal.clone());
        }

        Ok(signal)
    }

    /// Generate trading signal for a specific symbol
    pub async fn generate_signal(&self, symbol: &str) -> Result<Option<TradingSignal>> {
        let history = self.price_history.read().await;
        let symbol_history = match history.get(symbol) {
            Some(history) => history,
            None => return Ok(None),
        };

        if symbol_history.len() < self.config.lookback_period as usize {
            return Ok(None);
        }

        // Extract price data for analysis
        let prices: Vec<f64> = symbol_history.iter().map(|p| p.price).collect();
        let timestamps: Vec<DateTime<Utc>> = symbol_history.iter().map(|p| p.timestamp).collect();

        // Calculate technical indicators
        let mut signal_strength = 0.0;
        let mut signal_type = SignalType::Hold;
        let mut confidence = 0.0;
        let mut metadata = HashMap::new();

        // Trend following signals
        if self.config.trend_following.enabled {
            let trend_signal = self.calculate_trend_following_signal(&prices).await?;
            signal_strength += trend_signal.strength * 0.4; // 40% weight
            confidence += trend_signal.confidence * 0.4;
            metadata.insert("trend_strength".to_string(), trend_signal.strength);
            metadata.insert("trend_confidence".to_string(), trend_signal.confidence);
            
            if trend_signal.strength.abs() > signal_strength.abs() {
                signal_type = trend_signal.signal_type;
            }
        }

        // Mean reversion signals
        if self.config.mean_reversion.enabled {
            let mean_reversion_signal = self.calculate_mean_reversion_signal(&prices).await?;
            signal_strength += mean_reversion_signal.strength * 0.3; // 30% weight
            confidence += mean_reversion_signal.confidence * 0.3;
            metadata.insert("mean_reversion_strength".to_string(), mean_reversion_signal.strength);
            metadata.insert("mean_reversion_confidence".to_string(), mean_reversion_signal.confidence);
        }

        // Momentum signals
        if self.config.momentum.enabled {
            let momentum_signal = self.calculate_momentum_signal(&prices).await?;
            signal_strength += momentum_signal.strength * 0.3; // 30% weight
            confidence += momentum_signal.confidence * 0.3;
            metadata.insert("momentum_strength".to_string(), momentum_signal.strength);
            metadata.insert("momentum_confidence".to_string(), momentum_signal.confidence);
        }

        // Volatility adjustment
        let current_volatility = TechnicalIndicators::calculate_volatility(&prices, 20)?;
        let volatility_adjustment = if current_volatility > self.config.volatility_factor {
            0.8 // Reduce signal strength in high volatility
        } else {
            1.0
        };
        
        signal_strength *= volatility_adjustment;
        confidence *= volatility_adjustment;
        metadata.insert("volatility".to_string(), current_volatility);
        metadata.insert("volatility_adjustment".to_string(), volatility_adjustment);

        // Determine final signal type based on strength and threshold
        if signal_strength > self.config.signal_threshold {
            signal_type = if signal_strength > self.config.signal_threshold * 1.5 {
                SignalType::StrongBuy
            } else {
                SignalType::Buy
            };
        } else if signal_strength < -self.config.signal_threshold {
            signal_type = if signal_strength < -self.config.signal_threshold * 1.5 {
                SignalType::StrongSell
            } else {
                SignalType::Sell
            };
        } else {
            signal_type = SignalType::Hold;
        }

        // Only generate signal if confidence is above minimum threshold
        if confidence < 0.5 {
            return Ok(None);
        }

        let signal = TradingSignal {
            id: uuid::Uuid::new_v4(),
            symbol: symbol.to_string(),
            signal_type,
            strength: signal_strength,
            confidence,
            timestamp: Utc::now(),
            source: "tengri_signals".to_string(),
            metadata,
            expires_at: Some(Utc::now() + chrono::Duration::minutes(15)), // 15-minute expiry
        };

        Ok(Some(signal))
    }

    /// Calculate trend following signal
    async fn calculate_trend_following_signal(&self, prices: &[f64]) -> Result<SignalComponents> {
        let params = &self.config.trend_following;
        
        // Calculate EMAs
        let ema_fast = TechnicalIndicators::calculate_ema(prices, params.ema_fast as usize)?;
        let ema_slow = TechnicalIndicators::calculate_ema(prices, params.ema_slow as usize)?;
        
        // Calculate ATR for volatility adjustment
        let atr = TechnicalIndicators::calculate_atr(prices, params.atr_period as usize)?;
        
        // Current values
        let current_fast = ema_fast.last().copied().unwrap_or(0.0);
        let current_slow = ema_slow.last().copied().unwrap_or(0.0);
        let current_price = prices.last().copied().unwrap_or(0.0);
        let current_atr = atr.last().copied().unwrap_or(0.0);
        
        // Trend strength based on EMA crossover
        let ema_diff = (current_fast - current_slow) / current_slow;
        
        // Breakout detection
        let breakout_threshold = params.breakout_threshold;
        let price_movement = (current_price - current_slow) / current_slow;
        
        let mut strength = 0.0;
        let mut confidence: f64 = 0.0;
        
        // EMA crossover signal
        if ema_diff > 0.001 { // Fast EMA above slow EMA
            strength += 0.5;
            confidence += 0.6;
        } else if ema_diff < -0.001 { // Fast EMA below slow EMA
            strength -= 0.5;
            confidence += 0.6;
        }
        
        // Breakout signal
        if price_movement > breakout_threshold {
            strength += 0.3;
            confidence += 0.4;
        } else if price_movement < -breakout_threshold {
            strength -= 0.3;
            confidence += 0.4;
        }
        
        // ATR-based volatility filter
        let volatility_filter = if current_atr / current_price > 0.02 {
            0.8 // High volatility, reduce confidence
        } else {
            1.0
        };
        
        confidence *= volatility_filter;
        
        let signal_type = if strength > 0.3 {
            SignalType::Buy
        } else if strength < -0.3 {
            SignalType::Sell
        } else {
            SignalType::Hold
        };
        
        Ok(SignalComponents {
            signal_type,
            strength,
            confidence: confidence.min(1.0),
        })
    }

    /// Calculate mean reversion signal
    async fn calculate_mean_reversion_signal(&self, prices: &[f64]) -> Result<SignalComponents> {
        let params = &self.config.mean_reversion;
        
        // Calculate Bollinger Bands
        let (upper_band, lower_band, middle_band) = TechnicalIndicators::calculate_bollinger_bands(
            prices, 
            params.bollinger_period as usize, 
            params.bollinger_std
        )?;
        
        // Calculate RSI
        let rsi = TechnicalIndicators::calculate_rsi(prices, params.rsi_period as usize)?;
        
        let current_price = prices.last().copied().unwrap_or(0.0);
        let current_upper = upper_band.last().copied().unwrap_or(0.0);
        let current_lower = lower_band.last().copied().unwrap_or(0.0);
        let current_middle = middle_band.last().copied().unwrap_or(0.0);
        let current_rsi = rsi.last().copied().unwrap_or(50.0);
        
        let mut strength = 0.0;
        let mut confidence: f64 = 0.0;
        
        // Bollinger Band signals
        if current_price > current_upper {
            strength -= 0.4; // Overbought, sell signal
            confidence += 0.5;
        } else if current_price < current_lower {
            strength += 0.4; // Oversold, buy signal
            confidence += 0.5;
        }
        
        // RSI signals
        if current_rsi > params.rsi_overbought {
            strength -= 0.3; // Overbought
            confidence += 0.4;
        } else if current_rsi < params.rsi_oversold {
            strength += 0.3; // Oversold
            confidence += 0.4;
        }
        
        // Distance from middle band (mean reversion tendency)
        let distance_from_mean = (current_price - current_middle) / current_middle;
        if distance_from_mean.abs() > 0.02 {
            strength += -distance_from_mean.signum() * 0.2;
            confidence += 0.3;
        }
        
        let signal_type = if strength > 0.3 {
            SignalType::Buy
        } else if strength < -0.3 {
            SignalType::Sell
        } else {
            SignalType::Hold
        };
        
        Ok(SignalComponents {
            signal_type,
            strength,
            confidence: confidence.min(1.0),
        })
    }

    /// Calculate momentum signal
    async fn calculate_momentum_signal(&self, prices: &[f64]) -> Result<SignalComponents> {
        let params = &self.config.momentum;
        
        // Calculate MACD
        let (macd_line, signal_line, histogram) = TechnicalIndicators::calculate_macd(
            prices,
            params.macd_fast as usize,
            params.macd_slow as usize,
            params.macd_signal as usize,
        )?;
        
        // Calculate momentum
        let momentum = TechnicalIndicators::calculate_momentum(prices, params.momentum_period as usize)?;
        
        let current_macd = macd_line.last().copied().unwrap_or(0.0);
        let current_signal = signal_line.last().copied().unwrap_or(0.0);
        let current_histogram = histogram.last().copied().unwrap_or(0.0);
        let current_momentum = momentum.last().copied().unwrap_or(0.0);
        
        let mut strength = 0.0;
        let mut confidence: f64 = 0.0;
        
        // MACD signals
        if current_macd > current_signal && current_histogram > 0.0 {
            strength += 0.4; // Bullish MACD crossover
            confidence += 0.6;
        } else if current_macd < current_signal && current_histogram < 0.0 {
            strength -= 0.4; // Bearish MACD crossover
            confidence += 0.6;
        }
        
        // Momentum signals
        if current_momentum > params.momentum_threshold {
            strength += 0.3; // Strong upward momentum
            confidence += 0.4;
        } else if current_momentum < -params.momentum_threshold {
            strength -= 0.3; // Strong downward momentum
            confidence += 0.4;
        }
        
        let signal_type = if strength > 0.3 {
            SignalType::Buy
        } else if strength < -0.3 {
            SignalType::Sell
        } else {
            SignalType::Hold
        };
        
        Ok(SignalComponents {
            signal_type,
            strength,
            confidence: confidence.min(1.0),
        })
    }

    /// Get latest signal for symbol
    pub async fn get_latest_signal(&self, symbol: &str) -> Option<TradingSignal> {
        let cache = self.signal_cache.read().await;
        cache.get(symbol).cloned()
    }

    /// Update market state based on overall market conditions
    pub async fn update_market_state(&mut self, symbols: &[String]) -> Result<()> {
        let history = self.price_history.read().await;
        
        let mut all_returns = Vec::new();
        let mut volatilities = Vec::new();
        
        for symbol in symbols {
            if let Some(prices) = history.get(symbol) {
                let price_vec: Vec<f64> = prices.iter().map(|p| p.price).collect();
                if price_vec.len() >= 20 {
                    // Calculate returns
                    let returns = TechnicalIndicators::calculate_returns(&price_vec)?;
                    all_returns.extend(returns);
                    
                    // Calculate volatility
                    let vol = TechnicalIndicators::calculate_volatility(&price_vec, 20)?;
                    volatilities.push(vol);
                }
            }
        }
        
        if all_returns.is_empty() {
            return Ok(());
        }
        
        // Analyze market regime
        let avg_return = all_returns.iter().sum::<f64>() / all_returns.len() as f64;
        let avg_volatility = volatilities.iter().sum::<f64>() / volatilities.len() as f64;
        
        let regime = if avg_volatility > 0.03 {
            if avg_return > 0.001 {
                MarketRegime::BullMarket
            } else if avg_return < -0.001 {
                MarketRegime::BearMarket
            } else {
                MarketRegime::Volatile
            }
        } else if avg_volatility < 0.01 {
            MarketRegime::LowVolatility
        } else {
            if avg_return.abs() < 0.0005 {
                MarketRegime::Sideways
            } else {
                MarketRegime::Trending
            }
        };
        
        let trend_strength = avg_return * 100.0; // Convert to percentage
        let sentiment = trend_strength.tanh(); // Normalize between -1 and 1
        
        let mut market_state = self.market_state.write().await;
        market_state.regime = regime;
        market_state.trend_strength = trend_strength;
        market_state.volatility_regime = if avg_volatility > 0.03 {
            "high".to_string()
        } else if avg_volatility < 0.01 {
            "low".to_string()
        } else {
            "medium".to_string()
        };
        market_state.sentiment = sentiment;
        market_state.liquidity_score = 0.5; // Would need order book data for accurate calculation
        market_state.timestamp = Utc::now();
        
        Ok(())
    }

    /// Get current market state
    pub async fn get_market_state(&self) -> MarketState {
        self.market_state.read().await.clone()
    }
}

/// Components of a trading signal
#[derive(Debug, Clone)]
struct SignalComponents {
    signal_type: SignalType,
    strength: f64,
    confidence: f64,
}

impl TechnicalIndicators {
    /// Calculate Exponential Moving Average
    pub fn calculate_ema(prices: &[f64], period: usize) -> Result<Vec<f64>> {
        if prices.len() < period {
            return Err(TengriError::Strategy("Insufficient data for EMA calculation".to_string()));
        }
        
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema = Vec::with_capacity(prices.len());
        
        // Initialize with SMA
        let sma = prices[..period].iter().sum::<f64>() / period as f64;
        ema.push(sma);
        
        for i in period..prices.len() {
            let new_ema = alpha * prices[i] + (1.0 - alpha) * ema[i - period];
            ema.push(new_ema);
        }
        
        Ok(ema)
    }

    /// Calculate Simple Moving Average
    pub fn calculate_sma(prices: &[f64], period: usize) -> Result<Vec<f64>> {
        if prices.len() < period {
            return Err(TengriError::Strategy("Insufficient data for SMA calculation".to_string()));
        }
        
        let mut sma = Vec::new();
        for i in period..=prices.len() {
            let avg = prices[i-period..i].iter().sum::<f64>() / period as f64;
            sma.push(avg);
        }
        
        Ok(sma)
    }

    /// Calculate Bollinger Bands
    pub fn calculate_bollinger_bands(prices: &[f64], period: usize, std_dev: f64) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let sma = Self::calculate_sma(prices, period)?;
        let mut upper_band = Vec::new();
        let mut lower_band = Vec::new();
        
        for i in period..=prices.len() {
            let window = &prices[i-period..i];
            let mean = sma[i-period];
            let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();
            
            upper_band.push(mean + std_dev * std);
            lower_band.push(mean - std_dev * std);
        }
        
        Ok((upper_band, lower_band, sma))
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn calculate_rsi(prices: &[f64], period: usize) -> Result<Vec<f64>> {
        if prices.len() < period + 1 {
            return Err(TengriError::Strategy("Insufficient data for RSI calculation".to_string()));
        }
        
        let mut gains = Vec::new();
        let mut losses = Vec::new();
        
        // Calculate price changes
        for i in 1..prices.len() {
            let change = prices[i] - prices[i-1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }
        
        let mut rsi = Vec::new();
        
        // Initial averages
        let mut avg_gain = gains[..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss = losses[..period].iter().sum::<f64>() / period as f64;
        
        if avg_loss == 0.0 {
            rsi.push(100.0);
        } else {
            let rs = avg_gain / avg_loss;
            rsi.push(100.0 - (100.0 / (1.0 + rs)));
        }
        
        // Subsequent RSI values using smoothed averages
        for i in period..gains.len() {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            
            if avg_loss == 0.0 {
                rsi.push(100.0);
            } else {
                let rs = avg_gain / avg_loss;
                rsi.push(100.0 - (100.0 / (1.0 + rs)));
            }
        }
        
        Ok(rsi)
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn calculate_macd(prices: &[f64], fast_period: usize, slow_period: usize, signal_period: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let ema_fast = Self::calculate_ema(prices, fast_period)?;
        let ema_slow = Self::calculate_ema(prices, slow_period)?;
        
        // Calculate MACD line
        let start_idx = slow_period - fast_period;
        let mut macd_line = Vec::new();
        for i in start_idx..ema_fast.len() {
            macd_line.push(ema_fast[i] - ema_slow[i - start_idx]);
        }
        
        // Calculate signal line (EMA of MACD)
        let signal_line = Self::calculate_ema(&macd_line, signal_period)?;
        
        // Calculate histogram
        let mut histogram = Vec::new();
        let start_histogram = signal_period;
        for i in start_histogram..macd_line.len() {
            histogram.push(macd_line[i] - signal_line[i - start_histogram]);
        }
        
        Ok((macd_line, signal_line, histogram))
    }

    /// Calculate ATR (Average True Range)
    pub fn calculate_atr(prices: &[f64], period: usize) -> Result<Vec<f64>> {
        if prices.len() < period + 1 {
            return Err(TengriError::Strategy("Insufficient data for ATR calculation".to_string()));
        }
        
        let mut true_ranges = Vec::new();
        
        for i in 1..prices.len() {
            let high = prices[i];
            let low = prices[i];
            let prev_close = prices[i-1];
            
            let tr1 = high - low;
            let tr2 = (high - prev_close).abs();
            let tr3 = (low - prev_close).abs();
            
            true_ranges.push(tr1.max(tr2).max(tr3));
        }
        
        // Calculate ATR as SMA of true ranges
        Self::calculate_sma(&true_ranges, period)
    }

    /// Calculate price momentum
    pub fn calculate_momentum(prices: &[f64], period: usize) -> Result<Vec<f64>> {
        if prices.len() < period + 1 {
            return Err(TengriError::Strategy("Insufficient data for momentum calculation".to_string()));
        }
        
        let mut momentum = Vec::new();
        for i in period..prices.len() {
            let current = prices[i];
            let past = prices[i - period];
            momentum.push((current - past) / past);
        }
        
        Ok(momentum)
    }

    /// Calculate volatility (standard deviation of returns)
    pub fn calculate_volatility(prices: &[f64], period: usize) -> Result<f64> {
        if prices.len() < period {
            return Err(TengriError::Strategy("Insufficient data for volatility calculation".to_string()));
        }
        
        let returns = Self::calculate_returns(prices)?;
        if returns.len() < period {
            return Ok(0.0);
        }
        
        let recent_returns = &returns[returns.len()-period..];
        let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let variance = recent_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent_returns.len() as f64;
        
        Ok(variance.sqrt())
    }

    /// Calculate price returns
    pub fn calculate_returns(prices: &[f64]) -> Result<Vec<f64>> {
        if prices.len() < 2 {
            return Ok(Vec::new());
        }
        
        let mut returns = Vec::new();
        for i in 1..prices.len() {
            if prices[i-1] != 0.0 {
                returns.push((prices[i] - prices[i-1]) / prices[i-1]);
            }
        }
        
        Ok(returns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_calculation() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = TechnicalIndicators::calculate_sma(&prices, 3).unwrap();
        assert_eq!(sma, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ema_calculation() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = TechnicalIndicators::calculate_ema(&prices, 3).unwrap();
        assert!(ema.len() == 3);
        assert!(ema[0] == 2.0); // First value should be SMA
    }

    #[test]
    fn test_returns_calculation() {
        let prices = vec![100.0, 110.0, 105.0, 115.0];
        let returns = TechnicalIndicators::calculate_returns(&prices).unwrap();
        assert!((returns[0] - 0.1).abs() < 1e-10); // 10% return
        assert!((returns[1] - (-0.045454545)).abs() < 1e-6); // -4.55% return approximately
    }

    #[tokio::test]
    async fn test_signal_generator_creation() {
        let config = StrategyParameters::default();
        let generator = SignalGenerator::new(config);
        assert!(generator.price_history.read().await.is_empty());
    }
}