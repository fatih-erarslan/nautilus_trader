//! Technical analysis implementation

use crate::prelude::*;
use crate::models::MarketData;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::VecDeque;

/// Technical analysis engine for market data
#[derive(Debug, Clone)]
pub struct TechnicalAnalysis {
    /// Configuration for technical indicators
    config: TechnicalAnalysisConfig,
    
    /// Price history buffer
    price_history: VecDeque<PricePoint>,
    
    /// Volume history buffer
    volume_history: VecDeque<VolumePoint>,
    
    /// Calculated indicators cache
    indicators_cache: IndicatorsCache,
}

#[derive(Debug, Clone)]
pub struct TechnicalAnalysisConfig {
    /// Maximum history size
    pub max_history_size: usize,
    
    /// Moving average periods
    pub ma_periods: Vec<usize>,
    
    /// RSI period
    pub rsi_period: usize,
    
    /// MACD parameters
    pub macd_config: MacdConfig,
    
    /// Bollinger Bands parameters
    pub bollinger_config: BollingerConfig,
    
    /// Stochastic parameters
    pub stochastic_config: StochasticConfig,
}

#[derive(Debug, Clone)]
pub struct MacdConfig {
    pub fast_period: usize,
    pub slow_period: usize,
    pub signal_period: usize,
}

#[derive(Debug, Clone)]
pub struct BollingerConfig {
    pub period: usize,
    pub std_dev_multiplier: f64,
}

#[derive(Debug, Clone)]
pub struct StochasticConfig {
    pub k_period: usize,
    pub d_period: usize,
    pub smooth_k: usize,
}

#[derive(Debug, Clone)]
struct PricePoint {
    timestamp: DateTime<Utc>,
    open: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
}

#[derive(Debug, Clone)]
struct VolumePoint {
    timestamp: DateTime<Utc>,
    volume: Decimal,
}

#[derive(Debug, Clone, Default)]
struct IndicatorsCache {
    moving_averages: std::collections::HashMap<usize, Decimal>,
    rsi: Option<f64>,
    macd: Option<MacdIndicator>,
    bollinger_bands: Option<BollingerBands>,
    stochastic: Option<StochasticOscillator>,
    last_update: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct TechnicalIndicators {
    pub timestamp: DateTime<Utc>,
    pub moving_averages: std::collections::HashMap<usize, Decimal>,
    pub rsi: Option<f64>,
    pub macd: Option<MacdIndicator>,
    pub bollinger_bands: Option<BollingerBands>,
    pub stochastic: Option<StochasticOscillator>,
    pub volume_indicators: VolumeIndicators,
    pub support_resistance: SupportResistance,
}

#[derive(Debug, Clone)]
pub struct MacdIndicator {
    pub macd_line: f64,
    pub signal_line: f64,
    pub histogram: f64,
}

#[derive(Debug, Clone)]
pub struct BollingerBands {
    pub upper_band: Decimal,
    pub middle_band: Decimal,
    pub lower_band: Decimal,
    pub bandwidth: f64,
    pub percent_b: f64,
}

#[derive(Debug, Clone)]
pub struct StochasticOscillator {
    pub k_percent: f64,
    pub d_percent: f64,
    pub signal: StochasticSignal,
}

#[derive(Debug, Clone, Default)]
pub enum StochasticSignal {
    Overbought,
    Oversold,
    #[default]
    Neutral,
}

#[derive(Debug, Clone)]
pub struct VolumeIndicators {
    pub volume_ma: Decimal,
    pub volume_ratio: f64,
    pub on_balance_volume: Decimal,
    pub volume_weighted_average_price: Decimal,
}

#[derive(Debug, Clone)]
pub struct SupportResistance {
    pub support_levels: Vec<Decimal>,
    pub resistance_levels: Vec<Decimal>,
    pub pivot_point: Decimal,
}

impl Default for TechnicalAnalysisConfig {
    fn default() -> Self {
        Self {
            max_history_size: 1000,
            ma_periods: vec![5, 10, 20, 50, 100, 200],
            rsi_period: 14,
            macd_config: MacdConfig {
                fast_period: 12,
                slow_period: 26,
                signal_period: 9,
            },
            bollinger_config: BollingerConfig {
                period: 20,
                std_dev_multiplier: 2.0,
            },
            stochastic_config: StochasticConfig {
                k_period: 14,
                d_period: 3,
                smooth_k: 3,
            },
        }
    }
}

impl TechnicalAnalysis {
    /// Create a new technical analysis engine
    pub fn new(config: TechnicalAnalysisConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            indicators_cache: IndicatorsCache::default(),
        }
    }

    /// Add market data and calculate indicators
    pub async fn add_data(&mut self, market_data: &MarketData) -> Result<TechnicalIndicators> {
        // Add to price history
        self.add_price_data(market_data);
        
        // Add to volume history
        self.add_volume_data(market_data);
        
        // Calculate all indicators
        self.calculate_indicators().await
    }

    /// Calculate technical indicators for current data
    pub async fn calculate_indicators(&mut self) -> Result<TechnicalIndicators> {
        let timestamp = Utc::now();

        // Calculate moving averages
        let moving_averages = self.calculate_moving_averages()?;

        // Calculate RSI
        let rsi = self.calculate_rsi()?;

        // Calculate MACD
        let macd = self.calculate_macd()?;

        // Calculate Bollinger Bands
        let bollinger_bands = self.calculate_bollinger_bands()?;

        // Calculate Stochastic Oscillator
        let stochastic = self.calculate_stochastic()?;

        // Calculate Volume Indicators
        let volume_indicators = self.calculate_volume_indicators()?;

        // Calculate Support/Resistance
        let support_resistance = self.calculate_support_resistance()?;

        // Update cache
        self.indicators_cache = IndicatorsCache {
            moving_averages: moving_averages.clone(),
            rsi,
            macd: macd.clone(),
            bollinger_bands: bollinger_bands.clone(),
            stochastic: stochastic.clone(),
            last_update: Some(timestamp),
        };

        Ok(TechnicalIndicators {
            timestamp,
            moving_averages,
            rsi,
            macd,
            bollinger_bands,
            stochastic,
            volume_indicators,
            support_resistance,
        })
    }

    /// Get cached indicators (without recalculation)
    pub fn get_cached_indicators(&self) -> Option<TechnicalIndicators> {
        if let Some(last_update) = self.indicators_cache.last_update {
            Some(TechnicalIndicators {
                timestamp: last_update,
                moving_averages: self.indicators_cache.moving_averages.clone(),
                rsi: self.indicators_cache.rsi,
                macd: self.indicators_cache.macd.clone(),
                bollinger_bands: self.indicators_cache.bollinger_bands.clone(),
                stochastic: self.indicators_cache.stochastic.clone(),
                volume_indicators: VolumeIndicators {
                    volume_ma: Decimal::ZERO,
                    volume_ratio: 1.0,
                    on_balance_volume: Decimal::ZERO,
                    volume_weighted_average_price: Decimal::ZERO,
                },
                support_resistance: SupportResistance {
                    support_levels: vec![],
                    resistance_levels: vec![],
                    pivot_point: Decimal::ZERO,
                },
            })
        } else {
            None
        }
    }

    /// Generate trading signals based on technical indicators
    pub async fn generate_signals(&self) -> Result<TechnicalSignals> {
        let indicators = self.get_cached_indicators()
            .ok_or_else(|| Error::Analysis("No indicators available".to_string()))?;

        let mut signals = TechnicalSignals::default();

        // Moving Average Crossover Signals
        if let (Some(ma_short), Some(ma_long)) = (
            indicators.moving_averages.get(&self.config.ma_periods[0]),
            indicators.moving_averages.get(&self.config.ma_periods[1])
        ) {
            if ma_short > ma_long {
                signals.ma_signal = MovingAverageSignal::Bullish;
            } else if ma_short < ma_long {
                signals.ma_signal = MovingAverageSignal::Bearish;
            } else {
                signals.ma_signal = MovingAverageSignal::Neutral;
            }
        }

        // RSI Signals
        if let Some(rsi) = indicators.rsi {
            signals.rsi_signal = if rsi > 70.0 {
                RsiSignal::Overbought
            } else if rsi < 30.0 {
                RsiSignal::Oversold
            } else {
                RsiSignal::Neutral
            };
        }

        // MACD Signals
        if let Some(macd) = &indicators.macd {
            signals.macd_signal = if macd.macd_line > macd.signal_line && macd.histogram > 0.0 {
                MacdSignal::Bullish
            } else if macd.macd_line < macd.signal_line && macd.histogram < 0.0 {
                MacdSignal::Bearish
            } else {
                MacdSignal::Neutral
            };
        }

        // Bollinger Bands Signals
        if let Some(bb) = &indicators.bollinger_bands {
            if let Some(current_price) = self.get_current_price() {
                signals.bollinger_signal = if current_price >= bb.upper_band {
                    BollingerSignal::Overbought
                } else if current_price <= bb.lower_band {
                    BollingerSignal::Oversold
                } else {
                    BollingerSignal::Neutral
                };
            }
        }

        // Stochastic Signals
        if let Some(stoch) = &indicators.stochastic {
            signals.stochastic_signal = stoch.signal.clone();
        }

        // Overall Signal
        signals.overall_signal = self.calculate_overall_signal(&signals);

        Ok(signals)
    }

    fn add_price_data(&mut self, market_data: &MarketData) {
        let price_point = PricePoint {
            timestamp: market_data.timestamp,
            open: market_data.last, // Simplified - in real implementation would track OHLC properly
            high: market_data.ask,
            low: market_data.bid,
            close: market_data.last,
        };

        self.price_history.push_back(price_point);

        // Maintain buffer size
        while self.price_history.len() > self.config.max_history_size {
            self.price_history.pop_front();
        }
    }

    fn add_volume_data(&mut self, market_data: &MarketData) {
        let volume_point = VolumePoint {
            timestamp: market_data.timestamp,
            volume: market_data.volume_24h,
        };

        self.volume_history.push_back(volume_point);

        // Maintain buffer size
        while self.volume_history.len() > self.config.max_history_size {
            self.volume_history.pop_front();
        }
    }

    fn calculate_moving_averages(&self) -> Result<std::collections::HashMap<usize, Decimal>> {
        let mut mas = std::collections::HashMap::new();

        for &period in &self.config.ma_periods {
            if self.price_history.len() >= period {
                let sum: Decimal = self.price_history
                    .iter()
                    .rev()
                    .take(period)
                    .map(|p| p.close)
                    .sum();
                let ma = sum / Decimal::from(period);
                mas.insert(period, ma);
            }
        }

        Ok(mas)
    }

    fn calculate_rsi(&self) -> Result<Option<f64>> {
        let period = self.config.rsi_period;
        if self.price_history.len() < period + 1 {
            return Ok(None);
        }

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        for i in 1..=period {
            let current = self.price_history[self.price_history.len() - i].close;
            let previous = self.price_history[self.price_history.len() - i - 1].close;
            let change = current - previous;

            if change > Decimal::ZERO {
                gains.push(change.to_f64().unwrap_or(0.0));
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push((-change).to_f64().unwrap_or(0.0));
            }
        }

        let avg_gain = gains.iter().sum::<f64>() / period as f64;
        let avg_loss = losses.iter().sum::<f64>() / period as f64;

        if avg_loss == 0.0 {
            Ok(Some(100.0))
        } else {
            let rs = avg_gain / avg_loss;
            let rsi = 100.0 - (100.0 / (1.0 + rs));
            Ok(Some(rsi))
        }
    }

    fn calculate_macd(&self) -> Result<Option<MacdIndicator>> {
        let config = &self.config.macd_config;
        
        if self.price_history.len() < config.slow_period {
            return Ok(None);
        }

        // Calculate EMAs
        let fast_ema = self.calculate_ema(config.fast_period)?;
        let slow_ema = self.calculate_ema(config.slow_period)?;
        
        if fast_ema.is_none() || slow_ema.is_none() {
            return Ok(None);
        }

        let macd_line = fast_ema.unwrap() - slow_ema.unwrap();
        
        // For simplicity, using SMA for signal line (should be EMA in real implementation)
        let signal_line = macd_line; // Simplified
        let histogram = macd_line - signal_line;

        Ok(Some(MacdIndicator {
            macd_line,
            signal_line,
            histogram,
        }))
    }

    fn calculate_ema(&self, period: usize) -> Result<Option<f64>> {
        if self.price_history.len() < period {
            return Ok(None);
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = self.price_history[self.price_history.len() - period].close.to_f64().unwrap_or(0.0);

        for i in (self.price_history.len() - period + 1)..self.price_history.len() {
            let price = self.price_history[i].close.to_f64().unwrap_or(0.0);
            ema = (price * multiplier) + (ema * (1.0 - multiplier));
        }

        Ok(Some(ema))
    }

    fn calculate_bollinger_bands(&self) -> Result<Option<BollingerBands>> {
        let config = &self.config.bollinger_config;
        
        if self.price_history.len() < config.period {
            return Ok(None);
        }

        // Calculate SMA (middle band)
        let prices: Vec<f64> = self.price_history
            .iter()
            .rev()
            .take(config.period)
            .map(|p| p.close.to_f64().unwrap_or(0.0))
            .collect();

        let sma = prices.iter().sum::<f64>() / config.period as f64;
        
        // Calculate standard deviation
        let variance = prices.iter()
            .map(|price| (price - sma).powi(2))
            .sum::<f64>() / config.period as f64;
        let std_dev = variance.sqrt();

        let upper_band = Decimal::from_f64_retain(sma + (std_dev * config.std_dev_multiplier)).unwrap_or_default();
        let middle_band = Decimal::from_f64_retain(sma).unwrap_or_default();
        let lower_band = Decimal::from_f64_retain(sma - (std_dev * config.std_dev_multiplier)).unwrap_or_default();

        let bandwidth = ((upper_band - lower_band) / middle_band).to_f64().unwrap_or(0.0);
        
        let current_price = self.get_current_price().unwrap_or_default();
        let percent_b = if upper_band != lower_band {
            ((current_price - lower_band) / (upper_band - lower_band)).to_f64().unwrap_or(0.0)
        } else {
            0.5
        };

        Ok(Some(BollingerBands {
            upper_band,
            middle_band,
            lower_band,
            bandwidth,
            percent_b,
        }))
    }

    fn calculate_stochastic(&self) -> Result<Option<StochasticOscillator>> {
        let config = &self.config.stochastic_config;
        
        if self.price_history.len() < config.k_period {
            return Ok(None);
        }

        // Calculate %K
        let recent_prices: Vec<&PricePoint> = self.price_history
            .iter()
            .rev()
            .take(config.k_period)
            .collect();

        let highest_high = recent_prices.iter()
            .map(|p| p.high)
            .max()
            .unwrap_or_default();

        let lowest_low = recent_prices.iter()
            .map(|p| p.low)
            .min()
            .unwrap_or_default();

        let current_close = self.get_current_price().unwrap_or_default();

        let k_percent = if highest_high != lowest_low {
            ((current_close - lowest_low) / (highest_high - lowest_low)).to_f64().unwrap_or(0.0) * 100.0
        } else {
            50.0
        };

        // Calculate %D (simplified as SMA of %K)
        let d_percent = k_percent; // Simplified

        let signal = if k_percent > 80.0 {
            StochasticSignal::Overbought
        } else if k_percent < 20.0 {
            StochasticSignal::Oversold
        } else {
            StochasticSignal::Neutral
        };

        Ok(Some(StochasticOscillator {
            k_percent,
            d_percent,
            signal,
        }))
    }

    fn calculate_volume_indicators(&self) -> Result<VolumeIndicators> {
        if self.volume_history.is_empty() {
            return Ok(VolumeIndicators {
                volume_ma: Decimal::ZERO,
                volume_ratio: 1.0,
                on_balance_volume: Decimal::ZERO,
                volume_weighted_average_price: Decimal::ZERO,
            });
        }

        // Volume moving average
        let volume_ma = if self.volume_history.len() >= 20 {
            let sum: Decimal = self.volume_history.iter().rev().take(20).map(|v| v.volume).sum();
            sum / Decimal::from(20)
        } else {
            let sum: Decimal = self.volume_history.iter().map(|v| v.volume).sum();
            sum / Decimal::from(self.volume_history.len())
        };

        // Volume ratio (current vs average)
        let current_volume = self.volume_history.back().map(|v| v.volume).unwrap_or_default();
        let volume_ratio = if volume_ma > Decimal::ZERO {
            (current_volume / volume_ma).to_f64().unwrap_or(1.0)
        } else {
            1.0
        };

        // Simplified OBV and VWAP calculations
        let on_balance_volume = current_volume; // Simplified
        let volume_weighted_average_price = self.get_current_price().unwrap_or_default(); // Simplified

        Ok(VolumeIndicators {
            volume_ma,
            volume_ratio,
            on_balance_volume,
            volume_weighted_average_price,
        })
    }

    fn calculate_support_resistance(&self) -> Result<SupportResistance> {
        if self.price_history.len() < 50 {
            return Ok(SupportResistance {
                support_levels: vec![],
                resistance_levels: vec![],
                pivot_point: Decimal::ZERO,
            });
        }

        // Simplified support/resistance calculation
        let prices: Vec<Decimal> = self.price_history.iter().map(|p| p.close).collect();
        let mut sorted_prices = prices.clone();
        sorted_prices.sort();

        let len = sorted_prices.len();
        let pivot_point = sorted_prices[len / 2]; // Median as pivot

        // Support levels (below pivot)
        let support_levels = vec![
            sorted_prices[len / 4],     // 25th percentile
            sorted_prices[len / 8],     // 12.5th percentile
        ];

        // Resistance levels (above pivot)
        let resistance_levels = vec![
            sorted_prices[len * 3 / 4], // 75th percentile
            sorted_prices[len * 7 / 8], // 87.5th percentile
        ];

        Ok(SupportResistance {
            support_levels,
            resistance_levels,
            pivot_point,
        })
    }

    fn get_current_price(&self) -> Option<Decimal> {
        self.price_history.back().map(|p| p.close)
    }

    fn calculate_overall_signal(&self, signals: &TechnicalSignals) -> OverallSignal {
        let mut bullish_count = 0;
        let mut bearish_count = 0;

        // Count bullish signals
        match signals.ma_signal {
            MovingAverageSignal::Bullish => bullish_count += 1,
            MovingAverageSignal::Bearish => bearish_count += 1,
            _ => {}
        }

        match signals.rsi_signal {
            RsiSignal::Oversold => bullish_count += 1,
            RsiSignal::Overbought => bearish_count += 1,
            _ => {}
        }

        match signals.macd_signal {
            MacdSignal::Bullish => bullish_count += 1,
            MacdSignal::Bearish => bearish_count += 1,
            _ => {}
        }

        match signals.bollinger_signal {
            BollingerSignal::Oversold => bullish_count += 1,
            BollingerSignal::Overbought => bearish_count += 1,
            _ => {}
        }

        match signals.stochastic_signal {
            StochasticSignal::Oversold => bullish_count += 1,
            StochasticSignal::Overbought => bearish_count += 1,
            _ => {}
        }

        // Determine overall signal
        if bullish_count > bearish_count + 1 {
            OverallSignal::StrongBuy
        } else if bullish_count > bearish_count {
            OverallSignal::Buy
        } else if bearish_count > bullish_count + 1 {
            OverallSignal::StrongSell
        } else if bearish_count > bullish_count {
            OverallSignal::Sell
        } else {
            OverallSignal::Hold
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TechnicalSignals {
    pub ma_signal: MovingAverageSignal,
    pub rsi_signal: RsiSignal,
    pub macd_signal: MacdSignal,
    pub bollinger_signal: BollingerSignal,
    pub stochastic_signal: StochasticSignal,
    pub overall_signal: OverallSignal,
}

#[derive(Debug, Clone, Default)]
pub enum MovingAverageSignal {
    Bullish,
    Bearish,
    #[default]
    Neutral,
}

#[derive(Debug, Clone, Default)]
pub enum RsiSignal {
    Overbought,
    Oversold,
    #[default]
    Neutral,
}

#[derive(Debug, Clone, Default)]
pub enum MacdSignal {
    Bullish,
    Bearish,
    #[default]
    Neutral,
}

#[derive(Debug, Clone, Default)]
pub enum BollingerSignal {
    Overbought,
    Oversold,
    #[default]
    Neutral,
}

#[derive(Debug, Clone, Default)]
pub enum OverallSignal {
    StrongBuy,
    Buy,
    #[default]
    Hold,
    Sell,
    StrongSell,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_technical_analysis_creation() {
        let config = TechnicalAnalysisConfig::default();
        let ta = TechnicalAnalysis::new(config);
        
        assert_eq!(ta.price_history.len(), 0);
        assert_eq!(ta.volume_history.len(), 0);
    }

    #[tokio::test]
    async fn test_add_market_data() {
        let config = TechnicalAnalysisConfig::default();
        let mut ta = TechnicalAnalysis::new(config);
        
        let market_data = MarketData {
            symbol: "BTC/USD".to_string(),
            timestamp: Utc::now(),
            bid: dec!(50000),
            ask: dec!(50001),
            mid: dec!(50000.5),
            last: dec!(50000),
            volume_24h: dec!(1000),
            bid_size: dec!(10),
            ask_size: dec!(10),
        };

        let result = ta.add_data(&market_data).await;
        assert!(result.is_ok());
        assert_eq!(ta.price_history.len(), 1);
        assert_eq!(ta.volume_history.len(), 1);
    }
}