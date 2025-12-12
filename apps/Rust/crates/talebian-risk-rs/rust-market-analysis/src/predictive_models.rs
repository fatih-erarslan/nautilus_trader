//! Predictive modeling module for market forecasting
//! 
//! Implements advanced time series forecasting models including ARIMA, LSTM,
//! GARCH, and ensemble methods for price and volatility prediction.

use crate::{
    types::*,
    config::Config,
    error::{AnalysisError, Result},
    utils::{statistical, ml, time_series},
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use statrs::statistics::Statistics;
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc, Duration};
use rayon::prelude::*;
use tracing::{info, debug, warn};

/// Predictive modeling engine for market forecasting
#[derive(Debug, Clone)]
pub struct PredictiveEngine {
    config: PredictiveConfig,
    arima_model: ArimaModel,
    lstm_model: LstmModel,
    garch_model: GarchModel,
    ensemble_model: EnsembleModel,
    feature_store: FeatureStore,
    model_cache: HashMap<String, CachedPrediction>,
}

#[derive(Debug, Clone)]
pub struct PredictiveConfig {
    pub prediction_horizons: Vec<u32>,        // Minutes into the future
    pub ensemble_weights: EnsembleWeights,
    pub feature_count: usize,
    pub validation_split: f64,
    pub retraining_threshold: f64,            // Model accuracy threshold for retraining
    pub confidence_levels: Vec<f64>,          // For prediction intervals
    pub max_lookback_periods: usize,
    pub online_learning_rate: f64,
}

impl Default for PredictiveConfig {
    fn default() -> Self {
        Self {
            prediction_horizons: vec![1, 5, 15, 30, 60],
            ensemble_weights: EnsembleWeights::default(),
            feature_count: 30,
            validation_split: 0.2,
            retraining_threshold: 0.7,
            confidence_levels: vec![0.80, 0.95],
            max_lookback_periods: 500,
            online_learning_rate: 0.01,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnsembleWeights {
    pub arima: f64,
    pub lstm: f64,
    pub garch: f64,
    pub linear_regression: f64,
}

impl Default for EnsembleWeights {
    fn default() -> Self {
        Self {
            arima: 0.25,
            lstm: 0.35,
            garch: 0.25,
            linear_regression: 0.15,
        }
    }
}

impl PredictiveEngine {
    pub fn new(config: &Config) -> Result<Self> {
        let predictive_config = PredictiveConfig::default();
        
        Ok(Self {
            config: predictive_config.clone(),
            arima_model: ArimaModel::new(&predictive_config)?,
            lstm_model: LstmModel::new(&predictive_config)?,
            garch_model: GarchModel::new(&predictive_config)?,
            ensemble_model: EnsembleModel::new(&predictive_config)?,
            feature_store: FeatureStore::new()?,
            model_cache: HashMap::new(),
        })
    }
    
    /// Generate comprehensive market predictions
    pub async fn generate_predictions(&self, data: &MarketData) -> Result<Predictions> {
        let start_time = std::time::Instant::now();
        debug!("Starting prediction generation for {}", data.symbol);
        
        // Extract and prepare features
        let features = self.extract_prediction_features(data).await?;
        
        // Generate predictions from individual models
        let (arima_predictions, lstm_predictions, garch_predictions, volatility_forecast) = tokio::try_join!(
            self.generate_arima_predictions(data, &features),
            self.generate_lstm_predictions(data, &features),
            self.generate_price_predictions_from_garch(data, &features),
            self.generate_volatility_forecast(data, &features)
        )?;
        
        // Combine predictions using ensemble
        let ensemble_predictions = self.combine_predictions(
            arima_predictions,
            lstm_predictions,
            garch_predictions,
            &features
        ).await?;
        
        // Generate trend probabilities
        let trend_probability = self.calculate_trend_probability(data, &features).await?;
        
        // Generate liquidity forecast
        let liquidity_forecast = self.generate_liquidity_forecast(data, &features).await?;
        
        // Split predictions by time horizon
        let (short_term, medium_term) = self.split_predictions_by_horizon(ensemble_predictions)?;
        
        let predictions = Predictions {
            short_term,
            medium_term,
            volatility_forecast,
            trend_probability,
            liquidity_forecast,
        };
        
        let processing_time = start_time.elapsed();
        debug!("Prediction generation completed in {:?}", processing_time);
        
        Ok(predictions)
    }
    
    /// Extract features for prediction models
    async fn extract_prediction_features(&self, data: &MarketData) -> Result<PredictionFeatures> {
        // Price-based features
        let price_features = self.extract_price_features(data)?;
        
        // Volume-based features
        let volume_features = self.extract_volume_features(data)?;
        
        // Technical indicator features
        let technical_features = self.extract_technical_indicator_features(data)?;
        
        // Statistical features
        let statistical_features = self.extract_statistical_features(data)?;
        
        // Market microstructure features
        let microstructure_features = self.extract_microstructure_features(data)?;
        
        Ok(PredictionFeatures {
            price_features,
            volume_features,
            technical_features,
            statistical_features,
            microstructure_features,
            timestamp: Utc::now(),
        })
    }
    
    /// Extract price-based features
    fn extract_price_features(&self, data: &MarketData) -> Result<PricePredictionFeatures> {
        if data.prices.is_empty() {
            return Err(AnalysisError::insufficient_data("Empty price data"));
        }
        
        let prices = &data.prices;
        let returns = time_series::calculate_returns(prices, time_series::ReturnType::Log)?;
        
        // Moving averages
        let sma_5 = self.calculate_sma(prices, 5)?;
        let sma_20 = self.calculate_sma(prices, 20)?;
        let sma_50 = self.calculate_sma(prices, 50)?;
        let ema_12 = self.calculate_ema(prices, 12)?;
        let ema_26 = self.calculate_ema(prices, 26)?;
        
        // Price momentum features
        let momentum_1 = self.calculate_momentum(prices, 1)?;
        let momentum_3 = self.calculate_momentum(prices, 3)?;
        let momentum_7 = self.calculate_momentum(prices, 7)?;
        
        // Price volatility
        let volatility_5 = if returns.len() >= 5 { returns[returns.len()-5..].std_dev() } else { 0.0 };
        let volatility_20 = if returns.len() >= 20 { returns[returns.len()-20..].std_dev() } else { 0.0 };
        
        // Price range features
        let high_low_ratio = self.calculate_high_low_ratio(prices)?;
        let price_position = self.calculate_price_position(prices)?;
        
        Ok(PricePredictionFeatures {
            current_price: prices[prices.len() - 1],
            returns_1d: if !returns.is_empty() { returns[returns.len() - 1] } else { 0.0 },
            sma_5,
            sma_20,
            sma_50,
            ema_12,
            ema_26,
            momentum_1,
            momentum_3,
            momentum_7,
            volatility_5,
            volatility_20,
            high_low_ratio,
            price_position,
        })
    }
    
    /// Extract volume-based features
    fn extract_volume_features(&self, data: &MarketData) -> Result<VolumePredictionFeatures> {
        if data.volumes.is_empty() {
            return Ok(VolumePredictionFeatures::default());
        }
        
        let volumes = &data.volumes;
        
        let current_volume = volumes[volumes.len() - 1];
        let avg_volume_20 = if volumes.len() >= 20 { 
            volumes[volumes.len()-20..].mean() 
        } else { 
            volumes.mean() 
        };
        
        let volume_ratio = if avg_volume_20 > 0.0 { current_volume / avg_volume_20 } else { 1.0 };
        let volume_trend = self.calculate_volume_trend(volumes)?;
        let volume_volatility = volumes.std_dev();
        
        Ok(VolumePredictionFeatures {
            current_volume,
            avg_volume_20,
            volume_ratio,
            volume_trend,
            volume_volatility,
        })
    }
    
    /// Extract technical indicator features
    fn extract_technical_indicator_features(&self, data: &MarketData) -> Result<TechnicalPredictionFeatures> {
        let prices = &data.prices;
        
        let rsi = self.calculate_rsi(prices, 14)?;
        let (macd, macd_signal, macd_histogram) = self.calculate_macd(prices)?;
        let bollinger_position = self.calculate_bollinger_position(prices)?;
        let stochastic = self.calculate_stochastic(data)?;
        let atr = self.calculate_atr(data)?;
        
        Ok(TechnicalPredictionFeatures {
            rsi,
            macd,
            macd_signal,
            macd_histogram,
            bollinger_position,
            stochastic,
            atr,
        })
    }
    
    /// Extract statistical features
    fn extract_statistical_features(&self, data: &MarketData) -> Result<StatisticalPredictionFeatures> {
        let prices = &data.prices;
        let returns = time_series::calculate_returns(prices, time_series::ReturnType::Log)?;
        
        let skewness = if returns.len() >= 10 { statistical::skewness(&returns)? } else { 0.0 };
        let kurtosis = if returns.len() >= 10 { statistical::kurtosis(&returns)? } else { 0.0 };
        let hurst_exponent = if returns.len() >= 20 { statistical::hurst_exponent(&returns)? } else { 0.5 };
        
        // Autocorrelation
        let autocorr = if returns.len() >= 10 { 
            statistical::autocorrelation(&returns, 5)?
        } else { 
            vec![0.0; 5] 
        };
        
        Ok(StatisticalPredictionFeatures {
            skewness,
            kurtosis,
            hurst_exponent,
            autocorr_1: autocorr.get(0).copied().unwrap_or(0.0),
            autocorr_5: autocorr.get(4).copied().unwrap_or(0.0),
        })
    }
    
    /// Extract market microstructure features
    fn extract_microstructure_features(&self, data: &MarketData) -> Result<MicrostructurePredictionFeatures> {
        let bid_ask_spread = if let Some(ref order_book) = data.order_book {
            if !order_book.asks.is_empty() && !order_book.bids.is_empty() {
                order_book.asks[0].price - order_book.bids[0].price
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        let trade_intensity = data.trades.len() as f64;
        let order_flow_imbalance = self.calculate_order_flow_imbalance(data)?;
        
        Ok(MicrostructurePredictionFeatures {
            bid_ask_spread,
            trade_intensity,
            order_flow_imbalance,
        })
    }
    
    /// Generate ARIMA predictions
    async fn generate_arima_predictions(&self, data: &MarketData, features: &PredictionFeatures) -> Result<Vec<PricePrediction>> {
        self.arima_model.predict(data, features).await
    }
    
    /// Generate LSTM predictions
    async fn generate_lstm_predictions(&self, data: &MarketData, features: &PredictionFeatures) -> Result<Vec<PricePrediction>> {
        self.lstm_model.predict(data, features).await
    }
    
    /// Generate price predictions from GARCH model
    async fn generate_price_predictions_from_garch(&self, data: &MarketData, features: &PredictionFeatures) -> Result<Vec<PricePrediction>> {
        // GARCH primarily predicts volatility, but we can derive price predictions
        let volatility_predictions = self.garch_model.predict_volatility(data, features).await?;
        
        let current_price = data.prices.last().copied().unwrap_or(0.0);
        let mut price_predictions = Vec::new();
        
        for (i, volatility) in volatility_predictions.iter().enumerate() {
            let horizon = Duration::minutes(self.config.prediction_horizons[i.min(self.config.prediction_horizons.len()-1)] as i64);
            
            // Simple random walk with volatility scaling
            let predicted_price = current_price;
            let confidence_interval = (
                current_price * (1.0 - 2.0 * volatility),
                current_price * (1.0 + 2.0 * volatility)
            );
            
            price_predictions.push(PricePrediction {
                horizon,
                predicted_price,
                confidence_interval,
                probability_distribution: vec![(predicted_price, 1.0)],
                model_uncertainty: *volatility,
            });
        }
        
        Ok(price_predictions)
    }
    
    /// Generate volatility forecast
    async fn generate_volatility_forecast(&self, data: &MarketData, features: &PredictionFeatures) -> Result<VolatilityForecast> {
        let current_volatility = self.calculate_current_volatility(data)?;
        let forecasted_volatility = self.garch_model.predict_volatility(data, features).await?;
        
        let forecasted_volatility_with_horizons: Vec<(Duration, f64)> = self.config.prediction_horizons
            .iter()
            .zip(forecasted_volatility.iter())
            .map(|(&horizon_minutes, &vol)| (Duration::minutes(horizon_minutes as i64), vol))
            .collect();
        
        let volatility_regime_change_probability = self.calculate_volatility_regime_change_probability(data)?;
        let garch_parameters = self.garch_model.get_parameters();
        
        Ok(VolatilityForecast {
            current_volatility,
            forecasted_volatility: forecasted_volatility_with_horizons,
            volatility_regime_change_probability,
            garch_parameters: Some(garch_parameters),
        })
    }
    
    /// Combine predictions using ensemble methods
    async fn combine_predictions(
        &self,
        arima_predictions: Vec<PricePrediction>,
        lstm_predictions: Vec<PricePrediction>,
        garch_predictions: Vec<PricePrediction>,
        features: &PredictionFeatures
    ) -> Result<Vec<PricePrediction>> {
        self.ensemble_model.combine_predictions(
            arima_predictions,
            lstm_predictions,
            garch_predictions,
            &self.config.ensemble_weights
        ).await
    }
    
    /// Calculate trend probability
    async fn calculate_trend_probability(&self, data: &MarketData, features: &PredictionFeatures) -> Result<TrendProbability> {
        let prices = &data.prices;
        let returns = time_series::calculate_returns(prices, time_series::ReturnType::Log)?;
        
        // Calculate momentum indicators
        let momentum_indicators = MomentumIndicators {
            rsi: features.technical_features.rsi,
            macd: features.technical_features.macd,
            macd_signal: features.technical_features.macd_signal,
            stochastic: features.technical_features.stochastic,
            williams_r: 100.0 - features.technical_features.stochastic,
        };
        
        // Calculate trend probabilities based on technical indicators
        let mut uptrend_score = 0.0;
        let mut downtrend_score = 0.0;
        let mut total_indicators = 0.0;
        
        // RSI contribution
        if momentum_indicators.rsi > 50.0 {
            uptrend_score += (momentum_indicators.rsi - 50.0) / 50.0;
        } else {
            downtrend_score += (50.0 - momentum_indicators.rsi) / 50.0;
        }
        total_indicators += 1.0;
        
        // MACD contribution
        if momentum_indicators.macd > momentum_indicators.macd_signal {
            uptrend_score += 1.0;
        } else {
            downtrend_score += 1.0;
        }
        total_indicators += 1.0;
        
        // Stochastic contribution
        if momentum_indicators.stochastic > 50.0 {
            uptrend_score += (momentum_indicators.stochastic - 50.0) / 50.0;
        } else {
            downtrend_score += (50.0 - momentum_indicators.stochastic) / 50.0;
        }
        total_indicators += 1.0;
        
        // Normalize probabilities
        let total_score = uptrend_score + downtrend_score;
        let uptrend_probability = if total_score > 0.0 { uptrend_score / total_score } else { 0.33 };
        let downtrend_probability = if total_score > 0.0 { downtrend_score / total_score } else { 0.33 };
        let sideways_probability = 1.0 - uptrend_probability - downtrend_probability;
        
        // Calculate trend strength based on recent price movements
        let trend_strength = if returns.len() >= 10 {
            let recent_returns = &returns[returns.len()-10..];
            let positive_returns = recent_returns.iter().filter(|&&r| r > 0.0).count() as f64;
            let trend_consistency = positive_returns / recent_returns.len() as f64;
            (trend_consistency - 0.5).abs() * 2.0 // 0 = no trend, 1 = strong trend
        } else {
            0.0
        };
        
        Ok(TrendProbability {
            uptrend_probability,
            downtrend_probability,
            sideways_probability,
            trend_strength,
            momentum_indicators,
        })
    }
    
    /// Generate liquidity forecast
    async fn generate_liquidity_forecast(&self, data: &MarketData, features: &PredictionFeatures) -> Result<LiquidityForecast> {
        let current_liquidity = self.calculate_current_liquidity(data)?;
        
        // Forecast liquidity based on volume trends and market microstructure
        let volume_trend_factor = features.volume_features.volume_trend;
        let trade_intensity_factor = features.microstructure_features.trade_intensity / 100.0; // Normalize
        
        let forecasted_liquidity: Vec<(Duration, f64)> = self.config.prediction_horizons
            .iter()
            .map(|&horizon_minutes| {
                let horizon = Duration::minutes(horizon_minutes as i64);
                let decay_factor = (-horizon_minutes as f64 / 60.0).exp(); // Exponential decay
                let liquidity = current_liquidity * (1.0 + volume_trend_factor * 0.1) * decay_factor;
                (horizon, liquidity.max(0.0))
            })
            .collect();
        
        let market_impact_cost = 1.0 / current_liquidity.max(1.0); // Inverse relationship
        let optimal_execution_window = if current_liquidity > 1000.0 {
            Some(Duration::minutes(5))
        } else {
            Some(Duration::minutes(15))
        };
        
        Ok(LiquidityForecast {
            current_liquidity,
            forecasted_liquidity,
            market_impact_cost,
            optimal_execution_window,
        })
    }
    
    /// Split predictions by time horizon
    fn split_predictions_by_horizon(&self, predictions: Vec<PricePrediction>) -> Result<(Vec<PricePrediction>, Vec<PricePrediction>)> {
        let mut short_term = Vec::new();
        let mut medium_term = Vec::new();
        
        for prediction in predictions {
            if prediction.horizon <= Duration::minutes(60) {
                short_term.push(prediction);
            } else {
                medium_term.push(prediction);
            }
        }
        
        Ok((short_term, medium_term))
    }
    
    /// Retrain models based on recent performance
    pub async fn retrain(&mut self, feedback: &PredictionFeedback) -> Result<()> {
        info!("Retraining predictive models with feedback");
        
        // Update ensemble weights based on individual model performance
        let total_accuracy = feedback.price_prediction_accuracy 
            + feedback.volatility_prediction_accuracy 
            + feedback.directional_accuracy;
            
        if total_accuracy > 0.0 {
            // Adjust weights based on relative performance
            let price_weight = feedback.price_prediction_accuracy / total_accuracy;
            let vol_weight = feedback.volatility_prediction_accuracy / total_accuracy;
            let dir_weight = feedback.directional_accuracy / total_accuracy;
            
            // Update ensemble weights with learning rate
            let lr = self.config.online_learning_rate;
            self.config.ensemble_weights.arima = 
                self.config.ensemble_weights.arima * (1.0 - lr) + price_weight * lr;
            self.config.ensemble_weights.garch = 
                self.config.ensemble_weights.garch * (1.0 - lr) + vol_weight * lr;
            self.config.ensemble_weights.lstm = 
                self.config.ensemble_weights.lstm * (1.0 - lr) + dir_weight * lr;
        }
        
        // Retrain individual models
        self.arima_model.update_parameters(feedback).await?;
        self.lstm_model.retrain(feedback).await?;
        self.garch_model.update_parameters(feedback).await?;
        
        // Update model performance metrics
        for (metric_name, value) in &feedback.model_performance_metrics {
            info!("Model performance - {}: {:.4}", metric_name, value);
        }
        
        info!("Model retraining completed");
        Ok(())
    }
    
    // Helper calculation methods
    
    fn calculate_sma(&self, prices: &[f64], period: usize) -> Result<f64> {
        if prices.len() < period {
            return Ok(prices.mean());
        }
        Ok(prices[prices.len()-period..].mean())
    }
    
    fn calculate_ema(&self, prices: &[f64], period: usize) -> Result<f64> {
        if prices.len() < period {
            return Ok(prices.mean());
        }
        
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];
        
        for &price in &prices[1..] {
            ema = alpha * price + (1.0 - alpha) * ema;
        }
        
        Ok(ema)
    }
    
    fn calculate_momentum(&self, prices: &[f64], periods: usize) -> Result<f64> {
        if prices.len() <= periods {
            return Ok(0.0);
        }
        
        let current = prices[prices.len() - 1];
        let previous = prices[prices.len() - 1 - periods];
        
        if previous != 0.0 {
            Ok((current - previous) / previous)
        } else {
            Ok(0.0)
        }
    }
    
    fn calculate_high_low_ratio(&self, prices: &[f64]) -> Result<f64> {
        if prices.is_empty() {
            return Ok(1.0);
        }
        
        let high = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let low = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if low > 0.0 {
            Ok(high / low)
        } else {
            Ok(1.0)
        }
    }
    
    fn calculate_price_position(&self, prices: &[f64]) -> Result<f64> {
        if prices.is_empty() {
            return Ok(0.5);
        }
        
        let current_price = prices[prices.len() - 1];
        let high = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let low = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if high == low {
            Ok(0.5)
        } else {
            Ok((current_price - low) / (high - low))
        }
    }
    
    fn calculate_volume_trend(&self, volumes: &[f64]) -> Result<f64> {
        if volumes.len() < 2 {
            return Ok(0.0);
        }
        
        let recent_avg = if volumes.len() >= 10 {
            volumes[volumes.len()-5..].mean()
        } else {
            volumes[volumes.len()/2..].mean()
        };
        
        let historical_avg = if volumes.len() >= 10 {
            volumes[volumes.len()-10..volumes.len()-5].mean()
        } else {
            volumes[..volumes.len()/2].mean()
        };
        
        if historical_avg > 0.0 {
            Ok((recent_avg - historical_avg) / historical_avg)
        } else {
            Ok(0.0)
        }
    }
    
    fn calculate_rsi(&self, prices: &[f64], period: usize) -> Result<f64> {
        if prices.len() < period + 1 {
            return Ok(50.0);
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in prices.len()-period..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        
        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;
        
        if avg_loss == 0.0 {
            return Ok(100.0);
        }
        
        let rs = avg_gain / avg_loss;
        Ok(100.0 - (100.0 / (1.0 + rs)))
    }
    
    fn calculate_macd(&self, prices: &[f64]) -> Result<(f64, f64, f64)> {
        let ema12 = self.calculate_ema(prices, 12)?;
        let ema26 = self.calculate_ema(prices, 26)?;
        let macd_line = ema12 - ema26;
        
        // Simplified signal line (would need historical MACD values for proper EMA)
        let signal_line = macd_line * 0.9; // Approximation
        let histogram = macd_line - signal_line;
        
        Ok((macd_line, signal_line, histogram))
    }
    
    fn calculate_bollinger_position(&self, prices: &[f64]) -> Result<f64> {
        if prices.len() < 20 {
            return Ok(0.5);
        }
        
        let period = 20;
        let recent_prices = &prices[prices.len()-period..];
        let sma = recent_prices.mean();
        let std_dev = recent_prices.std_dev();
        let current_price = prices[prices.len() - 1];
        
        if std_dev == 0.0 {
            return Ok(0.5);
        }
        
        let upper_band = sma + (2.0 * std_dev);
        let lower_band = sma - (2.0 * std_dev);
        
        Ok((current_price - lower_band) / (upper_band - lower_band))
    }
    
    fn calculate_stochastic(&self, data: &MarketData) -> Result<f64> {
        if data.prices.len() < 14 {
            return Ok(50.0);
        }
        
        let period = 14;
        let recent_prices = &data.prices[data.prices.len()-period..];
        let current_price = data.prices[data.prices.len() - 1];
        let lowest_low = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let highest_high = recent_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if highest_high == lowest_low {
            return Ok(50.0);
        }
        
        Ok(((current_price - lowest_low) / (highest_high - lowest_low)) * 100.0)
    }
    
    fn calculate_atr(&self, data: &MarketData) -> Result<f64> {
        if data.prices.len() < 2 {
            return Ok(0.0);
        }
        
        // Simplified ATR using price changes
        let price_changes: Vec<f64> = data.prices.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
            
        if price_changes.is_empty() {
            return Ok(0.0);
        }
        
        Ok(price_changes.mean())
    }
    
    fn calculate_order_flow_imbalance(&self, data: &MarketData) -> Result<f64> {
        if data.trades.is_empty() {
            return Ok(0.0);
        }
        
        let (buy_volume, sell_volume) = data.trades.iter().fold((0.0, 0.0), |(buy, sell), trade| {
            match trade.side {
                TradeSide::Buy => (buy + trade.quantity, sell),
                TradeSide::Sell => (buy, sell + trade.quantity),
            }
        });
        
        let total_volume = buy_volume + sell_volume;
        if total_volume > 0.0 {
            Ok((buy_volume - sell_volume) / total_volume)
        } else {
            Ok(0.0)
        }
    }
    
    fn calculate_current_volatility(&self, data: &MarketData) -> Result<f64> {
        let returns = time_series::calculate_returns(&data.prices, time_series::ReturnType::Log)?;
        
        if returns.len() >= 20 {
            Ok(returns[returns.len()-20..].std_dev())
        } else {
            Ok(returns.std_dev())
        }
    }
    
    fn calculate_volatility_regime_change_probability(&self, data: &MarketData) -> Result<f64> {
        let returns = time_series::calculate_returns(&data.prices, time_series::ReturnType::Log)?;
        
        if returns.len() < 40 {
            return Ok(0.5);
        }
        
        let recent_vol = returns[returns.len()-20..].std_dev();
        let historical_vol = returns[returns.len()-40..returns.len()-20].std_dev();
        
        if historical_vol == 0.0 {
            return Ok(0.5);
        }
        
        let vol_ratio = recent_vol / historical_vol;
        
        // Higher ratio indicates higher probability of regime change
        Ok((vol_ratio - 1.0).abs().min(1.0))
    }
    
    fn calculate_current_liquidity(&self, data: &MarketData) -> Result<f64> {
        if let Some(ref order_book) = data.order_book {
            let bid_depth: f64 = order_book.bids.iter().map(|level| level.quantity).sum();
            let ask_depth: f64 = order_book.asks.iter().map(|level| level.quantity).sum();
            Ok(bid_depth + ask_depth)
        } else {
            // Approximate liquidity using volume
            Ok(data.volumes.iter().sum::<f64>() / data.volumes.len() as f64)
        }
    }
}

// Individual Model Implementations

/// ARIMA model for time series forecasting
#[derive(Debug, Clone)]
struct ArimaModel {
    order: (usize, usize, usize), // (p, d, q)
    parameters: Option<ArimaParameters>,
    is_trained: bool,
}

#[derive(Debug, Clone)]
struct ArimaParameters {
    ar_coefficients: Vec<f64>,
    ma_coefficients: Vec<f64>,
    residual_variance: f64,
}

impl ArimaModel {
    fn new(config: &PredictiveConfig) -> Result<Self> {
        Ok(Self {
            order: (1, 1, 1), // Simple ARIMA(1,1,1)
            parameters: None,
            is_trained: false,
        })
    }
    
    async fn predict(&self, data: &MarketData, features: &PredictionFeatures) -> Result<Vec<PricePrediction>> {
        // Simplified ARIMA prediction
        let current_price = data.prices.last().copied().unwrap_or(0.0);
        let returns = time_series::calculate_returns(&data.prices, time_series::ReturnType::Log)?;
        
        let mut predictions = Vec::new();
        
        for &horizon_minutes in &[1, 5, 15, 30, 60] {
            let horizon = Duration::minutes(horizon_minutes as i64);
            
            // Simple AR(1) approximation
            let recent_return = returns.last().copied().unwrap_or(0.0);
            let predicted_return = recent_return * 0.1; // Mean reversion
            let predicted_price = current_price * (1.0 + predicted_return);
            
            let volatility = if returns.len() >= 20 { 
                returns[returns.len()-20..].std_dev() 
            } else { 
                returns.std_dev() 
            };
            
            let confidence_interval = (
                predicted_price * (1.0 - 2.0 * volatility),
                predicted_price * (1.0 + 2.0 * volatility)
            );
            
            predictions.push(PricePrediction {
                horizon,
                predicted_price,
                confidence_interval,
                probability_distribution: vec![(predicted_price, 1.0)],
                model_uncertainty: volatility,
            });
        }
        
        Ok(predictions)
    }
    
    async fn update_parameters(&mut self, feedback: &PredictionFeedback) -> Result<()> {
        // Update ARIMA parameters based on feedback
        info!("Updating ARIMA model parameters");
        Ok(())
    }
}

/// LSTM neural network model
#[derive(Debug, Clone)]
struct LstmModel {
    sequence_length: usize,
    hidden_size: usize,
    is_trained: bool,
}

impl LstmModel {
    fn new(config: &PredictiveConfig) -> Result<Self> {
        Ok(Self {
            sequence_length: 60,
            hidden_size: 128,
            is_trained: false,
        })
    }
    
    async fn predict(&self, data: &MarketData, features: &PredictionFeatures) -> Result<Vec<PricePrediction>> {
        // Simplified LSTM prediction
        let current_price = data.prices.last().copied().unwrap_or(0.0);
        let returns = time_series::calculate_returns(&data.prices, time_series::ReturnType::Log)?;
        
        let mut predictions = Vec::new();
        
        for &horizon_minutes in &[1, 5, 15, 30, 60] {
            let horizon = Duration::minutes(horizon_minutes as i64);
            
            // Simple neural network approximation using features
            let price_momentum = features.price_features.momentum_1;
            let volume_factor = features.volume_features.volume_ratio;
            let technical_signal = (features.technical_features.rsi - 50.0) / 50.0;
            
            let predicted_return = (price_momentum * 0.3 + 
                                   (volume_factor - 1.0) * 0.2 + 
                                   technical_signal * 0.5) * 0.01;
                                   
            let predicted_price = current_price * (1.0 + predicted_return);
            
            let volatility = if returns.len() >= 10 { 
                returns[returns.len()-10..].std_dev() 
            } else { 
                0.02 
            };
            
            let confidence_interval = (
                predicted_price * (1.0 - 1.96 * volatility),
                predicted_price * (1.0 + 1.96 * volatility)
            );
            
            predictions.push(PricePrediction {
                horizon,
                predicted_price,
                confidence_interval,
                probability_distribution: vec![(predicted_price, 1.0)],
                model_uncertainty: volatility,
            });
        }
        
        Ok(predictions)
    }
    
    async fn retrain(&mut self, feedback: &PredictionFeedback) -> Result<()> {
        info!("Retraining LSTM model");
        // Implement LSTM retraining logic
        Ok(())
    }
}

/// GARCH model for volatility prediction
#[derive(Debug, Clone)]
struct GarchModel {
    parameters: GarchParameters,
    is_fitted: bool,
}

impl GarchModel {
    fn new(config: &PredictiveConfig) -> Result<Self> {
        Ok(Self {
            parameters: GarchParameters {
                omega: 0.000001,
                alpha: 0.1,
                beta: 0.85,
            },
            is_fitted: false,
        })
    }
    
    async fn predict_volatility(&self, data: &MarketData, features: &PredictionFeatures) -> Result<Vec<f64>> {
        let returns = time_series::calculate_returns(&data.prices, time_series::ReturnType::Log)?;
        
        if returns.is_empty() {
            return Ok(vec![0.02; 5]); // Default volatility
        }
        
        // Calculate GARCH volatility forecast
        let current_return = returns.last().copied().unwrap_or(0.0);
        let current_variance = if returns.len() >= 20 {
            returns[returns.len()-20..].variance()
        } else {
            returns.variance()
        };
        
        let mut forecasts = Vec::new();
        let mut variance = current_variance;
        
        for _ in 0..5 {
            // GARCH(1,1) forecast
            variance = self.parameters.omega + 
                      self.parameters.alpha * current_return.powi(2) + 
                      self.parameters.beta * variance;
            forecasts.push(variance.sqrt());
        }
        
        Ok(forecasts)
    }
    
    fn get_parameters(&self) -> GarchParameters {
        self.parameters.clone()
    }
    
    async fn update_parameters(&mut self, feedback: &PredictionFeedback) -> Result<()> {
        info!("Updating GARCH model parameters");
        
        // Adjust parameters based on volatility prediction accuracy
        if feedback.volatility_prediction_accuracy < 0.7 {
            // Increase mean reversion
            self.parameters.alpha = (self.parameters.alpha * 0.9).max(0.01);
            self.parameters.beta = (self.parameters.beta * 1.01).min(0.95);
        }
        
        Ok(())
    }
}

/// Ensemble model for combining predictions
#[derive(Debug, Clone)]
struct EnsembleModel {
    meta_learner: Option<MetaLearner>,
}

#[derive(Debug, Clone)]
struct MetaLearner {
    weights: Vec<f64>,
    bias: f64,
}

impl EnsembleModel {
    fn new(config: &PredictiveConfig) -> Result<Self> {
        Ok(Self {
            meta_learner: Some(MetaLearner {
                weights: vec![0.25, 0.35, 0.25, 0.15],
                bias: 0.0,
            }),
        })
    }
    
    async fn combine_predictions(
        &self,
        arima_predictions: Vec<PricePrediction>,
        lstm_predictions: Vec<PricePrediction>,
        garch_predictions: Vec<PricePrediction>,
        weights: &EnsembleWeights
    ) -> Result<Vec<PricePrediction>> {
        let mut combined_predictions = Vec::new();
        
        let max_len = arima_predictions.len()
            .max(lstm_predictions.len())
            .max(garch_predictions.len());
        
        for i in 0..max_len {
            let arima_pred = arima_predictions.get(i);
            let lstm_pred = lstm_predictions.get(i);
            let garch_pred = garch_predictions.get(i);
            
            if let (Some(arima), Some(lstm)) = (arima_pred, lstm_pred) {
                let combined_price = 
                    arima.predicted_price * weights.arima +
                    lstm.predicted_price * weights.lstm +
                    garch_pred.map(|g| g.predicted_price * weights.garch).unwrap_or(0.0);
                
                let combined_uncertainty = (
                    arima.model_uncertainty.powi(2) * weights.arima.powi(2) +
                    lstm.model_uncertainty.powi(2) * weights.lstm.powi(2) +
                    garch_pred.map(|g| g.model_uncertainty.powi(2) * weights.garch.powi(2)).unwrap_or(0.0)
                ).sqrt();
                
                let confidence_interval = (
                    combined_price * (1.0 - 2.0 * combined_uncertainty),
                    combined_price * (1.0 + 2.0 * combined_uncertainty)
                );
                
                combined_predictions.push(PricePrediction {
                    horizon: arima.horizon,
                    predicted_price: combined_price,
                    confidence_interval,
                    probability_distribution: vec![(combined_price, 1.0)],
                    model_uncertainty: combined_uncertainty,
                });
            }
        }
        
        Ok(combined_predictions)
    }
}

/// Feature store for caching and managing prediction features
#[derive(Debug, Clone)]
struct FeatureStore {
    cached_features: HashMap<String, (PredictionFeatures, DateTime<Utc>)>,
    ttl: Duration,
}

impl FeatureStore {
    fn new() -> Result<Self> {
        Ok(Self {
            cached_features: HashMap::new(),
            ttl: Duration::minutes(5),
        })
    }
}

/// Prediction features structure
#[derive(Debug, Clone)]
struct PredictionFeatures {
    price_features: PricePredictionFeatures,
    volume_features: VolumePredictionFeatures,
    technical_features: TechnicalPredictionFeatures,
    statistical_features: StatisticalPredictionFeatures,
    microstructure_features: MicrostructurePredictionFeatures,
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct PricePredictionFeatures {
    current_price: f64,
    returns_1d: f64,
    sma_5: f64,
    sma_20: f64,
    sma_50: f64,
    ema_12: f64,
    ema_26: f64,
    momentum_1: f64,
    momentum_3: f64,
    momentum_7: f64,
    volatility_5: f64,
    volatility_20: f64,
    high_low_ratio: f64,
    price_position: f64,
}

#[derive(Debug, Clone, Default)]
struct VolumePredictionFeatures {
    current_volume: f64,
    avg_volume_20: f64,
    volume_ratio: f64,
    volume_trend: f64,
    volume_volatility: f64,
}

#[derive(Debug, Clone)]
struct TechnicalPredictionFeatures {
    rsi: f64,
    macd: f64,
    macd_signal: f64,
    macd_histogram: f64,
    bollinger_position: f64,
    stochastic: f64,
    atr: f64,
}

#[derive(Debug, Clone)]
struct StatisticalPredictionFeatures {
    skewness: f64,
    kurtosis: f64,
    hurst_exponent: f64,
    autocorr_1: f64,
    autocorr_5: f64,
}

#[derive(Debug, Clone)]
struct MicrostructurePredictionFeatures {
    bid_ask_spread: f64,
    trade_intensity: f64,
    order_flow_imbalance: f64,
}

/// Cached prediction result
#[derive(Debug, Clone)]
struct CachedPrediction {
    predictions: Predictions,
    timestamp: DateTime<Utc>,
    ttl: Duration,
}

impl CachedPrediction {
    fn is_expired(&self) -> bool {
        Utc::now() - self.timestamp > self.ttl
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    
    #[tokio::test]
    async fn test_predictive_engine_creation() {
        let config = Config::default();
        let engine = PredictiveEngine::new(&config);
        assert!(engine.is_ok());
    }
    
    #[tokio::test]
    async fn test_prediction_generation() {
        let config = Config::default();
        let engine = PredictiveEngine::new(&config).unwrap();
        let market_data = MarketData::mock_data();
        
        let predictions = engine.generate_predictions(&market_data).await;
        assert!(predictions.is_ok());
        
        let pred = predictions.unwrap();
        assert!(!pred.short_term.is_empty() || !pred.medium_term.is_empty());
    }
    
    #[test]
    fn test_feature_extraction() {
        let config = Config::default();
        let engine = PredictiveEngine::new(&config).unwrap();
        let market_data = MarketData::mock_data();
        
        let price_features = engine.extract_price_features(&market_data);
        assert!(price_features.is_ok());
        
        let features = price_features.unwrap();
        assert!(features.current_price > 0.0);
    }
    
    #[test]
    fn test_technical_indicators() {
        let config = Config::default();
        let engine = PredictiveEngine::new(&config).unwrap();
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        
        let sma = engine.calculate_sma(&prices, 3).unwrap();
        assert!((sma - 104.0).abs() < 1e-10);
        
        let momentum = engine.calculate_momentum(&prices, 1).unwrap();
        assert!(momentum > 0.0); // Upward momentum
    }
}