# Examples Guide

Comprehensive collection of examples demonstrating NHITS capabilities across different domains and use cases.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Financial Time Series](#financial-time-series)
- [Energy & Utilities](#energy--utilities)
- [IoT & Sensor Data](#iot--sensor-data)
- [Weather Forecasting](#weather-forecasting)
- [Supply Chain & Demand](#supply-chain--demand)
- [Healthcare & Biomedical](#healthcare--biomedical)
- [Advanced Examples](#advanced-examples)
- [Integration Examples](#integration-examples)

## Basic Examples

### Simple Time Series Forecasting

```rust
use std::sync::Arc;
use autopoiesis::ml::nhits::prelude::*;
use autopoiesis::consciousness::ConsciousnessField;
use autopoiesis::core::autopoiesis::AutopoieticSystem;
use ndarray::{Array1, Array3};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize systems
    let consciousness = Arc::new(ConsciousnessField::new());
    let autopoietic = Arc::new(AutopoieticSystem::new());
    
    // Create simple configuration
    let config = NHITSConfigBuilder::new()
        .with_lookback(48)      // 48 hours of history
        .with_horizon(12)       // 12 hours forecast
        .with_features(1, 1)    // Single feature
        .build()?;
    
    let mut model = NHITS::new(config, consciousness, autopoietic);
    
    // Generate sample data (hourly temperature)
    let data = generate_temperature_data(200)?;
    let input = prepare_batch_data(&data, 48, 32)?;
    
    // Train model
    println!("Training model...");
    let history = model.train(&input, None, 50)?;
    println!("Final loss: {:.4}", history.train_losses.last().unwrap());
    
    // Generate forecast
    let latest_data = input.slice(s![0..1, -48.., ..]).to_owned();
    let forecast = model.forward(&latest_data, 48, 12)?;
    
    println!("12-hour Temperature Forecast:");
    for i in 0..12 {
        println!("  Hour {}: {:.1}°C", i + 1, forecast[[0, i, 0]]);
    }
    
    Ok(())
}

fn generate_temperature_data(hours: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut data = Vec::with_capacity(hours);
    let mut temp = 20.0; // Starting temperature
    
    for hour in 0..hours {
        // Daily cycle + weekly pattern + noise
        let daily_cycle = 5.0 * (hour as f64 * 2.0 * std::f64::consts::PI / 24.0).sin();
        let weekly_cycle = 2.0 * (hour as f64 * 2.0 * std::f64::consts::PI / (24.0 * 7.0)).sin();
        let noise = (rand::random::<f64>() - 0.5) * 2.0;
        
        temp = 20.0 + daily_cycle + weekly_cycle + noise;
        data.push(temp);
    }
    
    Ok(data)
}

fn prepare_batch_data(
    data: &[f64], 
    lookback: usize, 
    batch_size: usize
) -> Result<Array3<f64>, Box<dyn std::error::Error>> {
    let sequences = data.len() - lookback;
    let mut batch = Array3::zeros((batch_size, sequences.min(batch_size), 1));
    
    for i in 0..batch_size.min(sequences) {
        for j in 0..lookback {
            if i + j < data.len() {
                batch[[i, j, 0]] = data[i + j];
            }
        }
    }
    
    Ok(batch)
}
```

### Online Learning Example

```rust
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let consciousness = Arc::new(ConsciousnessField::new());
    let autopoietic = Arc::new(AutopoieticSystem::new());
    
    let config = NHITSConfig::for_use_case(UseCase::ShortTermForecasting);
    let mut model = NHITS::new(config, consciousness, autopoietic);
    
    // Simulate streaming data
    let mut data_stream = create_data_stream();
    let mut observations = Vec::new();
    
    while let Some(new_point) = data_stream.next().await {
        observations.push(new_point);
        
        // Wait for enough data
        if observations.len() < 100 {
            continue;
        }
        
        // Prepare training data
        let input = Array1::from_vec(observations[observations.len()-50..].to_vec());
        let target = Array1::from_vec(vec![new_point]);
        
        // Update model online
        model.update_online(&[input], &[target])?;
        
        // Generate forecast every 10 observations
        if observations.len() % 10 == 0 {
            let forecast_input = Array1::from_vec(
                observations[observations.len()-50..].to_vec()
            );
            let forecast = model.predict(&forecast_input, 5)?;
            
            println!("New observation: {:.2}", new_point);
            println!("5-step forecast: {:?}", forecast.to_vec());
        }
        
        // Keep sliding window
        if observations.len() > 1000 {
            observations.remove(0);
        }
    }
    
    Ok(())
}

async fn create_data_stream() -> impl tokio_stream::Stream<Item = f64> {
    tokio_stream::iter(0..1000)
        .then(|i| async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            // Simulate sensor reading
            20.0 + 5.0 * (i as f64 * 0.1).sin() + rand::random::<f64>() * 2.0 - 1.0
        })
}
```

## Financial Time Series

### Stock Price Prediction

```rust
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Deserialize, Serialize)]
struct StockData {
    timestamp: DateTime<Utc>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load historical stock data
    let stock_data = load_stock_data("AAPL", "2023-01-01", "2024-01-01").await?;
    
    // Initialize NHITS with financial configuration
    let consciousness = Arc::new(ConsciousnessField::new());
    let autopoietic = Arc::new(AutopoieticSystem::new());
    
    let config = NHITSConfigBuilder::new()
        .with_lookback(60)      // 60 days of history
        .with_horizon(5)        // 5 days forecast
        .with_features(5, 1)    // OHLCV input, Close output
        .with_consciousness(true, 0.15) // Enable consciousness for market awareness
        .with_blocks(vec![
            BlockConfig {
                hidden_size: 256,
                num_basis: 8,
                pooling_factor: 2,
                activation: ActivationType::GELU,
                dropout_rate: 0.2,
                ..Default::default()
            },
            BlockConfig {
                hidden_size: 128,
                num_basis: 6,
                pooling_factor: 2,
                activation: ActivationType::GELU,
                dropout_rate: 0.1,
                ..Default::default()
            }
        ])
        .build()?;
    
    let mut model = NHITS::new(config, consciousness, autopoietic);
    
    // Prepare data
    let (features, targets) = prepare_financial_data(&stock_data)?;
    let (train_data, val_data) = split_data(&features, 0.8)?;
    
    println!("Training on {} samples, validating on {} samples", 
             train_data.shape()[0], val_data.shape()[0]);
    
    // Train model
    let history = model.train(&train_data, Some(&val_data), 100)?;
    
    // Evaluate performance
    let test_input = val_data.slice(s![0..10, .., ..]).to_owned();
    let predictions = model.forward(&test_input, 60, 5)?;
    
    // Calculate financial metrics
    let metrics = calculate_financial_metrics(&predictions, &val_data)?;
    
    println!("Training Results:");
    println!("  Final train loss: {:.6}", history.train_losses.last().unwrap());
    println!("  Final val loss: {:.6}", history.val_losses.last().unwrap());
    println!("  Best epoch: {}", history.best_epoch);
    
    println!("Financial Metrics:");
    println!("  Directional Accuracy: {:.2}%", metrics.directional_accuracy * 100.0);
    println!("  Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    
    // Generate current forecast
    let latest_data = features.slice(s![-1.., .., ..]).to_owned();
    let current_forecast = model.forward(&latest_data, 60, 5)?;
    
    println!("5-Day Stock Price Forecast:");
    for i in 0..5 {
        println!("  Day {}: ${:.2}", i + 1, current_forecast[[0, i, 0]]);
    }
    
    Ok(())
}

async fn load_stock_data(
    symbol: &str, 
    start_date: &str, 
    end_date: &str
) -> Result<Vec<StockData>, Box<dyn std::error::Error>> {
    // In practice, load from API like Alpha Vantage, Yahoo Finance, etc.
    // For demo, generate synthetic data
    generate_synthetic_stock_data(365)
}

fn prepare_financial_data(
    data: &[StockData]
) -> Result<(Array3<f64>, Array3<f64>), Box<dyn std::error::Error>> {
    let n_samples = data.len() - 60 - 5; // lookback - horizon
    let mut features = Array3::zeros((n_samples, 60, 5)); // OHLCV
    let mut targets = Array3::zeros((n_samples, 5, 1));   // Close prices
    
    for i in 0..n_samples {
        // Features: 60 days of OHLCV
        for j in 0..60 {
            let idx = i + j;
            features[[i, j, 0]] = data[idx].open;
            features[[i, j, 1]] = data[idx].high;
            features[[i, j, 2]] = data[idx].low;
            features[[i, j, 3]] = data[idx].close;
            features[[i, j, 4]] = data[idx].volume;
        }
        
        // Targets: next 5 days close prices
        for j in 0..5 {
            let idx = i + 60 + j;
            targets[[i, j, 0]] = data[idx].close;
        }
    }
    
    // Normalize features
    let normalized_features = normalize_financial_features(features)?;
    
    Ok((normalized_features, targets))
}

fn normalize_financial_features(
    mut features: Array3<f64>
) -> Result<Array3<f64>, Box<dyn std::error::Error>> {
    let shape = features.shape();
    
    // Normalize each feature separately
    for feature_idx in 0..shape[2] {
        let mut feature_data: Vec<f64> = features
            .slice(s![.., .., feature_idx])
            .iter()
            .cloned()
            .collect();
        
        if feature_idx == 4 { // Volume - use log transformation
            feature_data = feature_data.iter().map(|&x| (x + 1.0).ln()).collect();
        }
        
        // Z-score normalization
        let mean: f64 = feature_data.iter().sum::<f64>() / feature_data.len() as f64;
        let std: f64 = (feature_data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / feature_data.len() as f64).sqrt();
        
        let mut idx = 0;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                features[[i, j, feature_idx]] = (feature_data[idx] - mean) / std;
                idx += 1;
            }
        }
    }
    
    Ok(features)
}

#[derive(Debug)]
struct FinancialMetrics {
    directional_accuracy: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    volatility: f64,
}

fn calculate_financial_metrics(
    predictions: &Array3<f64>,
    actuals: &Array3<f64>,
) -> Result<FinancialMetrics, Box<dyn std::error::Error>> {
    let n_samples = predictions.shape()[0];
    let n_horizons = predictions.shape()[1];
    
    let mut correct_directions = 0;
    let mut total_predictions = 0;
    let mut returns = Vec::new();
    
    for i in 0..n_samples {
        for h in 1..n_horizons {
            let pred_return = predictions[[i, h, 0]] - predictions[[i, h-1, 0]];
            let actual_return = actuals[[i, h, 0]] - actuals[[i, h-1, 0]];
            
            if pred_return * actual_return > 0.0 {
                correct_directions += 1;
            }
            total_predictions += 1;
            
            returns.push(actual_return);
        }
    }
    
    let directional_accuracy = correct_directions as f64 / total_predictions as f64;
    
    // Calculate Sharpe ratio
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let return_std = (returns.iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>() / returns.len() as f64).sqrt();
    let sharpe_ratio = mean_return / return_std * (252.0_f64).sqrt(); // Annualized
    
    // Calculate max drawdown
    let mut peak = 0.0;
    let mut max_drawdown = 0.0;
    let mut cumulative_return = 0.0;
    
    for ret in &returns {
        cumulative_return += ret;
        if cumulative_return > peak {
            peak = cumulative_return;
        }
        let drawdown = (peak - cumulative_return) / peak;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }
    
    Ok(FinancialMetrics {
        directional_accuracy,
        sharpe_ratio,
        max_drawdown,
        volatility: return_std * (252.0_f64).sqrt(),
    })
}

fn generate_synthetic_stock_data(days: usize) -> Result<Vec<StockData>, Box<dyn std::error::Error>> {
    let mut data = Vec::with_capacity(days);
    let mut price = 150.0; // Starting price
    let start_date = Utc::now() - chrono::Duration::days(days as i64);
    
    for i in 0..days {
        let timestamp = start_date + chrono::Duration::days(i as i64);
        
        // Generate realistic OHLCV data
        let daily_return = (rand::random::<f64>() - 0.5) * 0.04; // ±2% daily volatility
        let new_price = price * (1.0 + daily_return);
        
        let high = new_price * (1.0 + rand::random::<f64>() * 0.02);
        let low = new_price * (1.0 - rand::random::<f64>() * 0.02);
        let open = price;
        let close = new_price;
        let volume = 1000000.0 + rand::random::<f64>() * 500000.0;
        
        data.push(StockData {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });
        
        price = new_price;
    }
    
    Ok(data)
}
```

### Cryptocurrency Trading Bot

```rust
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;

#[derive(Debug)]
struct TradingBot {
    model: NHITS,
    position_size: f64,
    current_position: f64,
    portfolio_value: f64,
    price_history: VecDeque<f64>,
    feature_history: VecDeque<Array1<f64>>,
}

impl TradingBot {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let consciousness = Arc::new(ConsciousnessField::new());
        let autopoietic = Arc::new(AutopoieticSystem::new());
        
        let config = NHITSConfigBuilder::new()
            .with_lookback(100)     // 100 price points
            .with_horizon(1)        // Next price prediction
            .with_features(4, 1)    // Price, volume, volatility, momentum
            .with_consciousness(true, 0.2) // High consciousness for market volatility
            .build()?;
        
        let model = NHITS::new(config, consciousness, autopoietic);
        
        Ok(Self {
            model,
            position_size: 0.1, // 10% of portfolio per trade
            current_position: 0.0,
            portfolio_value: 10000.0, // $10k starting capital
            price_history: VecDeque::with_capacity(1000),
            feature_history: VecDeque::with_capacity(1000),
        })
    }
    
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Connect to Binance WebSocket
        let (ws_stream, _) = connect_async("wss://stream.binance.com:9443/ws/btcusdt@trade").await?;
        let (mut write, mut read) = ws_stream.split();
        
        println!("Connected to Binance WebSocket. Starting trading bot...");
        
        while let Some(msg) = read.next().await {
            match msg? {
                Message::Text(text) => {
                    if let Ok(trade_data) = serde_json::from_str::<Value>(&text) {
                        let price = trade_data["p"].as_str()
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(0.0);
                        
                        let quantity = trade_data["q"].as_str()
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(0.0);
                        
                        self.process_market_data(price, quantity).await?;
                    }
                }
                Message::Ping(payload) => {
                    write.send(Message::Pong(payload)).await?;
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    async fn process_market_data(&mut self, price: f64, volume: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.price_history.push_back(price);
        
        // Keep sliding window
        if self.price_history.len() > 1000 {
            self.price_history.pop_front();
        }
        
        // Need enough data to make predictions
        if self.price_history.len() < 100 {
            return Ok(());
        }
        
        // Calculate features
        let features = self.calculate_features()?;
        self.feature_history.push_back(features.clone());
        
        if self.feature_history.len() > 1000 {
            self.feature_history.pop_front();
        }
        
        // Update model with new data (online learning)
        if self.feature_history.len() >= 2 {
            let prev_features = &self.feature_history[self.feature_history.len() - 2];
            let current_price = Array1::from_vec(vec![price]);
            
            self.model.update_online(&[prev_features.clone()], &[current_price])?;
        }
        
        // Generate prediction
        if self.feature_history.len() >= 100 {
            let recent_features: Vec<f64> = self.feature_history
                .iter()
                .rev()
                .take(100)
                .flat_map(|f| f.iter().cloned())
                .collect();
            
            let input = Array1::from_vec(recent_features);
            let prediction = self.model.predict(&input, 1)?;
            let predicted_price = prediction[0];
            
            // Trading decision
            let signal = self.generate_trading_signal(price, predicted_price);
            
            match signal {
                TradingSignal::Buy => self.execute_buy(price).await?,
                TradingSignal::Sell => self.execute_sell(price).await?,
                TradingSignal::Hold => {}
            }
            
            // Log status
            if rand::random::<f64>() < 0.01 { // Log 1% of the time
                println!("Price: ${:.2}, Predicted: ${:.2}, Position: {:.4}, Portfolio: ${:.2}",
                         price, predicted_price, self.current_position, self.portfolio_value);
            }
        }
        
        Ok(())
    }
    
    fn calculate_features(&self) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let prices: Vec<f64> = self.price_history.iter().cloned().collect();
        let n = prices.len();
        
        if n < 20 {
            return Ok(Array1::zeros(4));
        }
        
        // Current price (normalized)
        let current_price = prices[n - 1];
        let price_ma20 = prices[n-20..].iter().sum::<f64>() / 20.0;
        let normalized_price = (current_price - price_ma20) / price_ma20;
        
        // Volatility (20-period standard deviation)
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        let recent_returns = &returns[returns.len().saturating_sub(20)..];
        let mean_return = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let volatility = (recent_returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / recent_returns.len() as f64).sqrt();
        
        // Momentum (5-period vs 20-period MA)
        let price_ma5 = prices[n-5..].iter().sum::<f64>() / 5.0;
        let momentum = (price_ma5 - price_ma20) / price_ma20;
        
        // RSI (14-period)
        let rsi = self.calculate_rsi(&prices, 14);
        
        Ok(Array1::from_vec(vec![normalized_price, volatility, momentum, rsi]))
    }
    
    fn calculate_rsi(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0; // Neutral RSI
        }
        
        let mut gains = Vec::new();
        let mut losses = Vec::new();
        
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
        
        let recent_gains = &gains[gains.len().saturating_sub(period)..];
        let recent_losses = &losses[losses.len().saturating_sub(period)..];
        
        let avg_gain = recent_gains.iter().sum::<f64>() / period as f64;
        let avg_loss = recent_losses.iter().sum::<f64>() / period as f64;
        
        if avg_loss == 0.0 {
            return 100.0;
        }
        
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
    
    fn generate_trading_signal(&self, current_price: f64, predicted_price: f64) -> TradingSignal {
        let price_change = (predicted_price - current_price) / current_price;
        let threshold = 0.002; // 0.2% threshold
        
        if price_change > threshold && self.current_position < self.position_size {
            TradingSignal::Buy
        } else if price_change < -threshold && self.current_position > -self.position_size {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        }
    }
    
    async fn execute_buy(&mut self, price: f64) -> Result<(), Box<dyn std::error::Error>> {
        let trade_amount = self.portfolio_value * 0.01; // 1% per trade
        let quantity = trade_amount / price;
        
        self.current_position += quantity;
        self.portfolio_value -= trade_amount;
        
        println!("BUY: {:.6} BTC at ${:.2}, Portfolio: ${:.2}", quantity, price, self.portfolio_value);
        Ok(())
    }
    
    async fn execute_sell(&mut self, price: f64) -> Result<(), Box<dyn std::error::Error>> {
        let trade_amount = self.portfolio_value * 0.01;
        let quantity = trade_amount / price;
        
        self.current_position -= quantity;
        self.portfolio_value += trade_amount;
        
        println!("SELL: {:.6} BTC at ${:.2}, Portfolio: ${:.2}", quantity, price, self.portfolio_value);
        Ok(())
    }
}

#[derive(Debug)]
enum TradingSignal {
    Buy,
    Sell,
    Hold,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut bot = TradingBot::new()?;
    bot.run().await?;
    Ok(())
}
```

## Energy & Utilities

### Smart Grid Load Forecasting

```rust
use chrono::{DateTime, Utc, Timelike, Weekday};

#[derive(Debug, Clone)]
struct EnergyData {
    timestamp: DateTime<Utc>,
    load_mw: f64,
    temperature: f64,
    humidity: f64,
    wind_speed: f64,
    solar_irradiance: f64,
    day_of_week: u8,
    hour_of_day: u8,
    is_holiday: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load historical energy data
    let energy_data = load_energy_data("grid_data.csv").await?;
    
    // Configure NHITS for energy forecasting
    let consciousness = Arc::new(ConsciousnessField::new());
    let autopoietic = Arc::new(AutopoieticSystem::new());
    
    let config = NHITSConfigBuilder::new()
        .with_lookback(168)     // One week of hourly data
        .with_horizon(24)       // Next 24 hours
        .with_features(8, 1)    // Multiple weather + time features → load
        .with_consciousness(true, 0.1)
        .with_blocks(vec![
            BlockConfig {
                hidden_size: 512,
                num_basis: 12,
                pooling_factor: 2,
                activation: ActivationType::GELU,
                dropout_rate: 0.15,
                ..Default::default()
            },
            BlockConfig {
                hidden_size: 256,
                num_basis: 8,
                pooling_factor: 2,
                activation: ActivationType::GELU,
                dropout_rate: 0.1,
                ..Default::default()
            },
            BlockConfig {
                hidden_size: 128,
                num_basis: 6,
                pooling_factor: 2,
                activation: ActivationType::GELU,
                dropout_rate: 0.05,
                ..Default::default()
            }
        ])
        .build()?;
    
    let mut model = NHITS::new(config, consciousness, autopoietic);
    
    // Prepare data with weather and calendar features
    let (features, targets) = prepare_energy_features(&energy_data)?;
    let (train_data, val_data) = split_temporal_data(&features, &targets, 0.8)?;
    
    println!("Training energy load forecasting model...");
    println!("  Training samples: {}", train_data.shape()[0]);
    println!("  Validation samples: {}", val_data.shape()[0]);
    
    // Train with early stopping
    let history = model.train(&train_data, Some(&val_data), 200)?;
    
    // Evaluate performance
    let test_predictions = model.forward(&val_data, 168, 24)?;
    let metrics = calculate_energy_metrics(&test_predictions, &val_data)?;
    
    println!("Training Results:");
    println!("  Final train loss: {:.6}", history.train_losses.last().unwrap());
    println!("  Final val loss: {:.6}", history.val_losses.last().unwrap());
    println!("  Best epoch: {}", history.best_epoch);
    
    println!("Energy Forecasting Metrics:");
    println!("  MAPE: {:.2}%", metrics.mape * 100.0);
    println!("  RMSE: {:.2} MW", metrics.rmse);
    println!("  Peak Error: {:.2}%", metrics.peak_error * 100.0);
    println!("  Load Factor Accuracy: {:.2}%", metrics.load_factor_accuracy * 100.0);
    
    // Generate 24-hour forecast with weather predictions
    let weather_forecast = get_weather_forecast(24).await?;
    let current_features = prepare_forecast_features(&energy_data, &weather_forecast)?;
    let load_forecast = model.forward(&current_features, 168, 24)?;
    
    println!("24-Hour Load Forecast:");
    let mut total_energy = 0.0;
    for i in 0..24 {
        let hour = (Utc::now().hour() + i as u32) % 24;
        let load = load_forecast[[0, i, 0]];
        total_energy += load;
        println!("  {:02}:00 - {:.1} MW", hour, load);
    }
    println!("  Total Energy: {:.1} MWh", total_energy);
    
    // Identify peak demand period
    let (peak_hour, peak_load) = find_peak_demand(&load_forecast);
    println!("  Peak Demand: {:.1} MW at hour {}", peak_load, peak_hour);
    
    // Generate load management recommendations
    let recommendations = generate_load_recommendations(&load_forecast, &weather_forecast)?;
    println!("Load Management Recommendations:");
    for rec in recommendations {
        println!("  - {}", rec);
    }
    
    Ok(())
}

fn prepare_energy_features(
    data: &[EnergyData]
) -> Result<(Array3<f64>, Array3<f64>), Box<dyn std::error::Error>> {
    let lookback = 168;
    let horizon = 24;
    let n_samples = data.len() - lookback - horizon;
    
    let mut features = Array3::zeros((n_samples, lookback, 8));
    let mut targets = Array3::zeros((n_samples, horizon, 1));
    
    for i in 0..n_samples {
        // Historical features (168 hours)
        for j in 0..lookback {
            let idx = i + j;
            features[[i, j, 0]] = data[idx].load_mw;
            features[[i, j, 1]] = data[idx].temperature;
            features[[i, j, 2]] = data[idx].humidity;
            features[[i, j, 3]] = data[idx].wind_speed;
            features[[i, j, 4]] = data[idx].solar_irradiance;
            features[[i, j, 5]] = data[idx].day_of_week as f64;
            features[[i, j, 6]] = data[idx].hour_of_day as f64;
            features[[i, j, 7]] = if data[idx].is_holiday { 1.0 } else { 0.0 };
        }
        
        // Target load (next 24 hours)
        for j in 0..horizon {
            let idx = i + lookback + j;
            targets[[i, j, 0]] = data[idx].load_mw;
        }
    }
    
    // Normalize features
    normalize_energy_features(&mut features)?;
    
    Ok((features, targets))
}

fn normalize_energy_features(features: &mut Array3<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let shape = features.shape();
    
    // Normalize each feature type
    for feature_idx in 0..shape[2] {
        let mut values: Vec<f64> = features
            .slice(s![.., .., feature_idx])
            .iter()
            .cloned()
            .collect();
        
        match feature_idx {
            0 => { // Load - use min-max normalization
                let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = max_val - min_val;
                
                if range > 0.0 {
                    let mut idx = 0;
                    for i in 0..shape[0] {
                        for j in 0..shape[1] {
                            features[[i, j, feature_idx]] = (values[idx] - min_val) / range;
                            idx += 1;
                        }
                    }
                }
            },
            1..=4 => { // Weather features - z-score normalization
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let std = (values.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / values.len() as f64).sqrt();
                
                if std > 0.0 {
                    let mut idx = 0;
                    for i in 0..shape[0] {
                        for j in 0..shape[1] {
                            features[[i, j, feature_idx]] = (values[idx] - mean) / std;
                            idx += 1;
                        }
                    }
                }
            },
            5 => { // Day of week - cyclical encoding
                let mut idx = 0;
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        let day = values[idx];
                        features[[i, j, feature_idx]] = (day * 2.0 * std::f64::consts::PI / 7.0).sin();
                        idx += 1;
                    }
                }
            },
            6 => { // Hour of day - cyclical encoding
                let mut idx = 0;
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        let hour = values[idx];
                        features[[i, j, feature_idx]] = (hour * 2.0 * std::f64::consts::PI / 24.0).sin();
                        idx += 1;
                    }
                }
            },
            _ => {} // Holiday - keep as binary
        }
    }
    
    Ok(())
}

#[derive(Debug)]
struct EnergyMetrics {
    mape: f64,
    rmse: f64,
    peak_error: f64,
    load_factor_accuracy: f64,
}

fn calculate_energy_metrics(
    predictions: &Array3<f64>,
    actuals: &Array3<f64>,
) -> Result<EnergyMetrics, Box<dyn std::error::Error>> {
    let n_samples = predictions.shape()[0];
    let n_hours = predictions.shape()[1];
    
    let mut absolute_errors = Vec::new();
    let mut squared_errors = Vec::new();
    let mut percentage_errors = Vec::new();
    let mut peak_errors = Vec::new();
    
    for i in 0..n_samples {
        let mut sample_peak_pred = 0.0;
        let mut sample_peak_actual = 0.0;
        
        for j in 0..n_hours {
            let pred = predictions[[i, j, 0]];
            let actual = actuals[[i, j, 0]];
            
            absolute_errors.push((pred - actual).abs());
            squared_errors.push((pred - actual).powi(2));
            
            if actual != 0.0 {
                percentage_errors.push(((pred - actual) / actual).abs());
            }
            
            if pred > sample_peak_pred { sample_peak_pred = pred; }
            if actual > sample_peak_actual { sample_peak_actual = actual; }
        }
        
        if sample_peak_actual > 0.0 {
            peak_errors.push(((sample_peak_pred - sample_peak_actual) / sample_peak_actual).abs());
        }
    }
    
    let mape = percentage_errors.iter().sum::<f64>() / percentage_errors.len() as f64;
    let rmse = (squared_errors.iter().sum::<f64>() / squared_errors.len() as f64).sqrt();
    let peak_error = peak_errors.iter().sum::<f64>() / peak_errors.len() as f64;
    
    // Calculate load factor accuracy (ratio of average to peak)
    let mut load_factor_errors = Vec::new();
    for i in 0..n_samples {
        let pred_avg = (0..n_hours).map(|j| predictions[[i, j, 0]]).sum::<f64>() / n_hours as f64;
        let pred_peak = (0..n_hours).map(|j| predictions[[i, j, 0]]).fold(0.0, f64::max);
        let actual_avg = (0..n_hours).map(|j| actuals[[i, j, 0]]).sum::<f64>() / n_hours as f64;
        let actual_peak = (0..n_hours).map(|j| actuals[[i, j, 0]]).fold(0.0, f64::max);
        
        if pred_peak > 0.0 && actual_peak > 0.0 {
            let pred_lf = pred_avg / pred_peak;
            let actual_lf = actual_avg / actual_peak;
            load_factor_errors.push((pred_lf - actual_lf).abs());
        }
    }
    
    let load_factor_accuracy = 1.0 - (load_factor_errors.iter().sum::<f64>() / load_factor_errors.len() as f64);
    
    Ok(EnergyMetrics {
        mape,
        rmse,
        peak_error,
        load_factor_accuracy,
    })
}

async fn get_weather_forecast(hours: usize) -> Result<Vec<WeatherForecast>, Box<dyn std::error::Error>> {
    // In practice, integrate with weather API like OpenWeatherMap
    // For demo, generate realistic weather progression
    let mut forecast = Vec::with_capacity(hours);
    let mut temp = 22.0; // Starting temperature
    
    for i in 0..hours {
        let hour = (Utc::now().hour() + i as u32) % 24;
        
        // Realistic temperature progression
        let daily_cycle = 5.0 * ((hour as f64 - 14.0) * std::f64::consts::PI / 12.0).cos();
        temp = 22.0 + daily_cycle + (rand::random::<f64>() - 0.5) * 2.0;
        
        forecast.push(WeatherForecast {
            hour: hour as u8,
            temperature: temp,
            humidity: 60.0 + (rand::random::<f64>() - 0.5) * 20.0,
            wind_speed: 5.0 + rand::random::<f64>() * 10.0,
            solar_irradiance: if hour >= 6 && hour <= 18 {
                800.0 * (1.0 - ((hour as f64 - 12.0) / 6.0).powi(2)).max(0.0)
            } else {
                0.0
            },
        });
    }
    
    Ok(forecast)
}

#[derive(Debug)]
struct WeatherForecast {
    hour: u8,
    temperature: f64,
    humidity: f64,
    wind_speed: f64,
    solar_irradiance: f64,
}

fn find_peak_demand(forecast: &Array3<f64>) -> (usize, f64) {
    let mut peak_hour = 0;
    let mut peak_load = 0.0;
    
    for i in 0..forecast.shape()[1] {
        let load = forecast[[0, i, 0]];
        if load > peak_load {
            peak_load = load;
            peak_hour = i;
        }
    }
    
    (peak_hour, peak_load)
}

fn generate_load_recommendations(
    load_forecast: &Array3<f64>,
    weather_forecast: &[WeatherForecast],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut recommendations = Vec::new();
    
    // Find peak demand period
    let (peak_hour, peak_load) = find_peak_demand(load_forecast);
    
    // High load recommendations
    if peak_load > 1000.0 { // Assuming MW
        recommendations.push(format!(
            "High peak demand expected at hour {} ({:.1} MW). Consider demand response activation.",
            peak_hour, peak_load
        ));
    }
    
    // Weather-based recommendations
    for (i, weather) in weather_forecast.iter().enumerate() {
        let load = load_forecast[[0, i, 0]];
        
        if weather.temperature > 30.0 && load > 800.0 {
            recommendations.push(format!(
                "Hour {}: High temperature ({:.1}°C) driving cooling load. Consider pre-cooling strategies.",
                i, weather.temperature
            ));
        }
        
        if weather.solar_irradiance > 600.0 && load > 700.0 {
            recommendations.push(format!(
                "Hour {}: High solar generation potential ({:.0} W/m²). Coordinate with renewable sources.",
                i, weather.solar_irradiance
            ));
        }
        
        if weather.wind_speed > 12.0 {
            recommendations.push(format!(
                "Hour {}: High wind speeds ({:.1} m/s). Increase wind generation forecast.",
                i, weather.wind_speed
            ));
        }
    }
    
    // Load management strategies
    let avg_load = (0..24).map(|i| load_forecast[[0, i, 0]]).sum::<f64>() / 24.0;
    let load_variance = (0..24)
        .map(|i| (load_forecast[[0, i, 0]] - avg_load).powi(2))
        .sum::<f64>() / 24.0;
    
    if load_variance > 10000.0 { // High variance
        recommendations.push(
            "High load variability detected. Consider energy storage deployment for peak shaving.".to_string()
        );
    }
    
    Ok(recommendations)
}

async fn load_energy_data(filename: &str) -> Result<Vec<EnergyData>, Box<dyn std::error::Error>> {
    // In practice, load from CSV file or database
    // For demo, generate synthetic energy data
    generate_synthetic_energy_data(365 * 24) // One year of hourly data
}

fn generate_synthetic_energy_data(hours: usize) -> Result<Vec<EnergyData>, Box<dyn std::error::Error>> {
    let mut data = Vec::with_capacity(hours);
    let start_time = Utc::now() - chrono::Duration::hours(hours as i64);
    
    for i in 0..hours {
        let timestamp = start_time + chrono::Duration::hours(i as i64);
        let hour = timestamp.hour();
        let day_of_week = timestamp.weekday() as u8;
        let is_weekend = matches!(timestamp.weekday(), Weekday::Sat | Weekday::Sun);
        
        // Temperature cycle
        let temp = 20.0 + 10.0 * ((hour as f64 - 14.0) * std::f64::consts::PI / 12.0).cos()
                   + (rand::random::<f64>() - 0.5) * 5.0;
        
        // Load pattern based on hour and day type
        let base_load = if is_weekend { 600.0 } else { 800.0 };
        let hourly_pattern = match hour {
            0..=5 => 0.7,    // Night
            6..=8 => 0.9,    // Morning ramp
            9..=17 => 1.0,   // Day
            18..=21 => 1.1,  // Evening peak
            _ => 0.8,        // Late evening
        };
        
        // Temperature effect (cooling load)
        let temp_effect = if temp > 25.0 {
            1.0 + (temp - 25.0) * 0.02 // 2% increase per degree above 25°C
        } else if temp < 15.0 {
            1.0 + (15.0 - temp) * 0.01 // 1% increase per degree below 15°C
        } else {
            1.0
        };
        
        let load = base_load * hourly_pattern * temp_effect + (rand::random::<f64>() - 0.5) * 50.0;
        
        data.push(EnergyData {
            timestamp,
            load_mw: load.max(0.0),
            temperature: temp,
            humidity: 50.0 + (rand::random::<f64>() - 0.5) * 30.0,
            wind_speed: 5.0 + rand::random::<f64>() * 15.0,
            solar_irradiance: if hour >= 6 && hour <= 18 {
                800.0 * (1.0 - ((hour as f64 - 12.0) / 6.0).powi(2)).max(0.0)
            } else {
                0.0
            },
            day_of_week,
            hour_of_day: hour as u8,
            is_holiday: false, // Simplified
        });
    }
    
    Ok(data)
}
```

## IoT & Sensor Data

### Industrial Equipment Monitoring

```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::time::{interval, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SensorReading {
    sensor_id: String,
    timestamp: DateTime<Utc>,
    temperature: f64,
    vibration: f64,
    pressure: f64,
    current: f64,
    voltage: f64,
    rpm: f64,
}

#[derive(Debug, Clone)]
struct EquipmentMonitor {
    model: NHITS,
    sensor_history: HashMap<String, VecDeque<SensorReading>>,
    alert_thresholds: AlertThresholds,
    maintenance_predictor: MaintenancePredictor,
}

#[derive(Debug, Clone)]
struct AlertThresholds {
    temperature_max: f64,
    vibration_max: f64,
    pressure_min: f64,
    pressure_max: f64,
    anomaly_score_threshold: f64,
}

impl EquipmentMonitor {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let consciousness = Arc::new(ConsciousnessField::new());
        let autopoietic = Arc::new(AutopoieticSystem::new());
        
        // Configure for anomaly detection and predictive maintenance
        let config = NHITSConfigBuilder::new()
            .with_lookback(288)     // 24 hours of 5-minute readings
            .with_horizon(12)       // 1 hour ahead predictions
            .with_features(6, 6)    // All sensor readings → future readings
            .with_consciousness(true, 0.25) // High consciousness for anomaly detection
            .with_adaptation(AdaptationStrategy::ConsciousnessGuided, 0.05)
            .with_blocks(vec![
                BlockConfig {
                    hidden_size: 256,
                    num_basis: 10,
                    pooling_factor: 3,
                    activation: ActivationType::GELU,
                    dropout_rate: 0.2,
                    ..Default::default()
                },
                BlockConfig {
                    hidden_size: 128,
                    num_basis: 6,
                    pooling_factor: 2,
                    activation: ActivationType::GELU,
                    dropout_rate: 0.1,
                    ..Default::default()
                }
            ])
            .build()?;
        
        let model = NHITS::new(config, consciousness, autopoietic);
        
        let alert_thresholds = AlertThresholds {
            temperature_max: 80.0,
            vibration_max: 10.0,
            pressure_min: 2.0,
            pressure_max: 8.0,
            anomaly_score_threshold: 3.0,
        };
        
        Ok(Self {
            model,
            sensor_history: HashMap::new(),
            alert_thresholds,
            maintenance_predictor: MaintenancePredictor::new()?,
        })
    }
    
    pub async fn start_monitoring(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut data_interval = interval(Duration::from_secs(300)); // 5-minute intervals
        let mut analysis_interval = interval(Duration::from_secs(3600)); // Hourly analysis
        
        println!("Starting equipment monitoring system...");
        
        loop {
            tokio::select! {
                _ = data_interval.tick() => {
                    let readings = self.collect_sensor_data().await?;
                    for reading in readings {
                        self.process_sensor_reading(reading).await?;
                    }
                }
                _ = analysis_interval.tick() => {
                    self.perform_comprehensive_analysis().await?;
                }
            }
        }
    }
    
    async fn collect_sensor_data(&self) -> Result<Vec<SensorReading>, Box<dyn std::error::Error>> {
        // In practice, collect from actual IoT sensors via MQTT, HTTP, etc.
        // For demo, generate realistic sensor data
        let sensor_ids = vec!["pump_01", "motor_02", "compressor_03", "turbine_04"];
        let mut readings = Vec::new();
        
        for sensor_id in sensor_ids {
            let reading = self.generate_realistic_sensor_data(sensor_id).await;
            readings.push(reading);
        }
        
        Ok(readings)
    }
    
    async fn generate_realistic_sensor_data(&self, sensor_id: &str) -> SensorReading {
        let now = Utc::now();
        
        // Simulate different equipment types
        let (base_temp, base_vib, base_pressure, base_current, base_voltage, base_rpm) = match sensor_id {
            "pump_01" => (45.0, 2.0, 5.0, 15.0, 220.0, 1750.0),
            "motor_02" => (60.0, 1.5, 0.0, 25.0, 380.0, 3000.0),
            "compressor_03" => (70.0, 3.0, 7.0, 40.0, 480.0, 1200.0),
            "turbine_04" => (550.0, 5.0, 12.0, 100.0, 6600.0, 1800.0),
            _ => (50.0, 2.0, 5.0, 20.0, 220.0, 2000.0),
        };
        
        // Add realistic variations and potential degradation
        let age_factor = self.get_equipment_age_factor(sensor_id);
        let load_factor = self.get_current_load_factor();
        
        // Temperature increases with age and load
        let temperature = base_temp + 
                         (rand::random::<f64>() - 0.5) * 10.0 + 
                         age_factor * 15.0 + 
                         load_factor * 10.0;
        
        // Vibration increases with wear
        let vibration = base_vib + 
                       (rand::random::<f64>() - 0.5) * 1.0 + 
                       age_factor * 2.0;
        
        // Pressure can vary with load
        let pressure = base_pressure + 
                      (rand::random::<f64>() - 0.5) * 2.0 + 
                      load_factor * 1.0;
        
        // Current varies with load
        let current = base_current * (0.8 + load_factor * 0.4) + 
                     (rand::random::<f64>() - 0.5) * 2.0;
        
        // Voltage should be relatively stable
        let voltage = base_voltage + (rand::random::<f64>() - 0.5) * 10.0;
        
        // RPM varies with load
        let rpm = base_rpm * (0.9 + load_factor * 0.2) + 
                 (rand::random::<f64>() - 0.5) * 50.0;
        
        SensorReading {
            sensor_id: sensor_id.to_string(),
            timestamp: now,
            temperature,
            vibration,
            pressure,
            current,
            voltage,
            rpm,
        }
    }
    
    fn get_equipment_age_factor(&self, sensor_id: &str) -> f64 {
        // Simulate equipment aging (0.0 = new, 1.0 = old)
        match sensor_id {
            "pump_01" => 0.3,
            "motor_02" => 0.7,
            "compressor_03" => 0.2,
            "turbine_04" => 0.5,
            _ => 0.4,
        }
    }
    
    fn get_current_load_factor(&self) -> f64 {
        // Simulate current load (0.0 = idle, 1.0 = full load)
        let hour = Utc::now().hour();
        match hour {
            8..=18 => 0.8 + rand::random::<f64>() * 0.2, // High load during work hours
            19..=23 => 0.4 + rand::random::<f64>() * 0.3, // Medium load evening
            _ => 0.1 + rand::random::<f64>() * 0.2,       // Low load at night
        }
    }
    
    async fn process_sensor_reading(&mut self, reading: SensorReading) -> Result<(), Box<dyn std::error::Error>> {
        let sensor_id = reading.sensor_id.clone();
        
        // Update sensor history
        self.sensor_history
            .entry(sensor_id.clone())
            .or_insert_with(|| VecDeque::with_capacity(500))
            .push_back(reading.clone());
        
        // Keep only recent readings
        if let Some(history) = self.sensor_history.get_mut(&sensor_id) {
            while history.len() > 500 {
                history.pop_front();
            }
        }
        
        // Immediate threshold alerts
        self.check_immediate_alerts(&reading).await?;
        
        // Anomaly detection if we have enough history
        if self.sensor_history[&sensor_id].len() >= 288 {
            let anomaly_score = self.detect_anomaly(&sensor_id).await?;
            
            if anomaly_score > self.alert_thresholds.anomaly_score_threshold {
                self.handle_anomaly_alert(&sensor_id, anomaly_score, &reading).await?;
            }
        }
        
        // Update model with new data
        self.update_model_online(&sensor_id).await?;
        
        Ok(())
    }
    
    async fn check_immediate_alerts(&self, reading: &SensorReading) -> Result<(), Box<dyn std::error::Error>> {
        let mut alerts = Vec::new();
        
        if reading.temperature > self.alert_thresholds.temperature_max {
            alerts.push(format!("HIGH TEMPERATURE: {:.1}°C", reading.temperature));
        }
        
        if reading.vibration > self.alert_thresholds.vibration_max {
            alerts.push(format!("HIGH VIBRATION: {:.2} mm/s", reading.vibration));
        }
        
        if reading.pressure < self.alert_thresholds.pressure_min || 
           reading.pressure > self.alert_thresholds.pressure_max {
            alerts.push(format!("PRESSURE OUT OF RANGE: {:.1} bar", reading.pressure));
        }
        
        if !alerts.is_empty() {
            println!("🚨 IMMEDIATE ALERT - {}: {}", 
                     reading.sensor_id, alerts.join(", "));
            
            // In practice, send to monitoring system, SMS, email, etc.
            self.send_immediate_alert(&reading.sensor_id, &alerts).await?;
        }
        
        Ok(())
    }
    
    async fn detect_anomaly(&mut self, sensor_id: &str) -> Result<f64, Box<dyn std::error::Error>> {
        let history = &self.sensor_history[sensor_id];
        if history.len() < 288 {
            return Ok(0.0);
        }
        
        // Prepare recent data for anomaly detection
        let recent_data = self.prepare_sensor_features(sensor_id)?;
        let prediction = self.model.forward(&recent_data, 288, 12)?;
        
        // Compare prediction with recent actual values
        let recent_actuals: Vec<f64> = history.iter()
            .rev()
            .take(12)
            .map(|r| r.temperature) // Focus on temperature for simplicity
            .collect();
        
        if recent_actuals.len() < 12 {
            return Ok(0.0);
        }
        
        // Calculate anomaly score as normalized deviation
        let mut total_deviation = 0.0;
        for i in 0..12.min(recent_actuals.len()) {
            let predicted = prediction[[0, i, 0]]; // Assuming temperature is first output
            let actual = recent_actuals[11 - i]; // Reverse order
            total_deviation += (predicted - actual).abs();
        }
        
        let average_deviation = total_deviation / 12.0;
        let anomaly_score = average_deviation / 10.0; // Normalize to typical scale
        
        Ok(anomaly_score)
    }
    
    fn prepare_sensor_features(&self, sensor_id: &str) -> Result<Array3<f64>, Box<dyn std::error::Error>> {
        let history = &self.sensor_history[sensor_id];
        let lookback = 288;
        
        if history.len() < lookback {
            return Err("Insufficient sensor history".into());
        }
        
        let mut features = Array3::zeros((1, lookback, 6));
        
        for (i, reading) in history.iter().rev().take(lookback).enumerate() {
            let j = lookback - 1 - i; // Reverse chronological order
            features[[0, j, 0]] = reading.temperature;
            features[[0, j, 1]] = reading.vibration;
            features[[0, j, 2]] = reading.pressure;
            features[[0, j, 3]] = reading.current;
            features[[0, j, 4]] = reading.voltage;
            features[[0, j, 5]] = reading.rpm;
        }
        
        Ok(features)
    }
    
    async fn update_model_online(&mut self, sensor_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let history = &self.sensor_history[sensor_id];
        if history.len() < 300 { // Need some history for training
            return Ok(());
        }
        
        // Create training sample from recent history
        let input_data = self.prepare_sensor_features(sensor_id)?;
        
        // Use next reading as target (simplified)
        let latest_reading = history.back().unwrap();
        let target = Array1::from_vec(vec![
            latest_reading.temperature,
            latest_reading.vibration,
            latest_reading.pressure,
            latest_reading.current,
            latest_reading.voltage,
            latest_reading.rpm,
        ]);
        
        // Online update
        self.model.update_online(
            &[input_data.slice(s![0, .., ..]).to_owned().into_dimensionality().unwrap()],
            &[target]
        )?;
        
        Ok(())
    }
    
    async fn handle_anomaly_alert(
        &self, 
        sensor_id: &str, 
        anomaly_score: f64, 
        reading: &SensorReading
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 ANOMALY DETECTED - {}: Score {:.2}", sensor_id, anomaly_score);
        println!("   Current readings - Temp: {:.1}°C, Vibration: {:.2} mm/s, Pressure: {:.1} bar",
                 reading.temperature, reading.vibration, reading.pressure);
        
        // Generate maintenance recommendations
        let recommendations = self.maintenance_predictor
            .generate_recommendations(sensor_id, anomaly_score, reading)
            .await?;
        
        println!("   Recommendations:");
        for rec in &recommendations {
            println!("   - {}", rec);
        }
        
        // In practice, integrate with CMMS, work order system, etc.
        self.create_maintenance_work_order(sensor_id, anomaly_score, &recommendations).await?;
        
        Ok(())
    }
    
    async fn perform_comprehensive_analysis(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔧 Performing comprehensive equipment analysis...");
        
        for (sensor_id, history) in &self.sensor_history {
            if history.len() < 288 {
                continue;
            }
            
            // Generate detailed predictions
            let features = self.prepare_sensor_features(sensor_id)?;
            let predictions = self.model.forward(&features, 288, 12)?;
            
            // Analyze trends
            let trends = self.analyze_equipment_trends(sensor_id, &predictions).await?;
            
            // Estimate remaining useful life
            let rul = self.maintenance_predictor.estimate_remaining_life(sensor_id, &trends).await?;
            
            println!("📊 {} Analysis:", sensor_id);
            println!("   Temperature trend: {:.2}°C/hour", trends.temperature_trend);
            println!("   Vibration trend: {:.3} mm/s/hour", trends.vibration_trend);
            println!("   Estimated RUL: {} hours", rul);
            
            // Check if maintenance is needed soon
            if rul < 168 { // Less than 1 week
                println!("   ⚠️  Maintenance recommended within {} hours", rul);
                self.schedule_preventive_maintenance(sensor_id, rul).await?;
            }
        }
        
        Ok(())
    }
    
    async fn analyze_equipment_trends(
        &self, 
        sensor_id: &str, 
        predictions: &Array3<f64>
    ) -> Result<EquipmentTrends, Box<dyn std::error::Error>> {
        let history = &self.sensor_history[sensor_id];
        let recent_readings: Vec<&SensorReading> = history.iter().rev().take(24).collect();
        
        if recent_readings.len() < 2 {
            return Ok(EquipmentTrends::default());
        }
        
        // Calculate trends from recent history
        let temp_trend = self.calculate_trend(
            &recent_readings.iter().map(|r| r.temperature).collect::<Vec<f64>>()
        );
        let vibration_trend = self.calculate_trend(
            &recent_readings.iter().map(|r| r.vibration).collect::<Vec<f64>>()
        );
        let pressure_trend = self.calculate_trend(
            &recent_readings.iter().map(|r| r.pressure).collect::<Vec<f64>>()
        );
        
        Ok(EquipmentTrends {
            temperature_trend: temp_trend,
            vibration_trend,
            pressure_trend,
        })
    }
    
    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        // Simple linear trend calculation
        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for (i, &value) in values.iter().enumerate() {
            let x_diff = i as f64 - x_mean;
            numerator += x_diff * (value - y_mean);
            denominator += x_diff * x_diff;
        }
        
        if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    async fn send_immediate_alert(&self, sensor_id: &str, alerts: &[String]) -> Result<(), Box<dyn std::error::Error>> {
        // In practice, integrate with notification systems
        println!("📧 Sending alert for {}: {}", sensor_id, alerts.join(", "));
        Ok(())
    }
    
    async fn create_maintenance_work_order(
        &self, 
        sensor_id: &str, 
        anomaly_score: f64, 
        recommendations: &[String]
    ) -> Result<(), Box<dyn std::error::Error>> {
        // In practice, integrate with CMMS/ERP systems
        println!("📋 Creating work order for {} (Priority: {})", 
                 sensor_id, 
                 if anomaly_score > 5.0 { "HIGH" } else { "MEDIUM" });
        Ok(())
    }
    
    async fn schedule_preventive_maintenance(&self, sensor_id: &str, rul_hours: u64) -> Result<(), Box<dyn std::error::Error>> {
        // In practice, integrate with maintenance scheduling systems
        println!("📅 Scheduling preventive maintenance for {} in {} hours", sensor_id, rul_hours);
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
struct EquipmentTrends {
    temperature_trend: f64,
    vibration_trend: f64,
    pressure_trend: f64,
}

struct MaintenancePredictor;

impl MaintenancePredictor {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self)
    }
    
    async fn generate_recommendations(
        &self, 
        sensor_id: &str, 
        anomaly_score: f64, 
        reading: &SensorReading
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut recommendations = Vec::new();
        
        // Temperature-based recommendations
        if reading.temperature > 70.0 {
            recommendations.push("Check cooling system and air filters".to_string());
            recommendations.push("Verify lubricant levels and quality".to_string());
        }
        
        // Vibration-based recommendations
        if reading.vibration > 5.0 {
            recommendations.push("Inspect bearings and alignment".to_string());
            recommendations.push("Check for loose bolts or worn couplings".to_string());
        }
        
        // Pressure-based recommendations
        if reading.pressure < 3.0 {
            recommendations.push("Check for system leaks".to_string());
            recommendations.push("Verify pump/compressor performance".to_string());
        } else if reading.pressure > 7.0 {
            recommendations.push("Check discharge valves and filters".to_string());
        }
        
        // General anomaly recommendations
        if anomaly_score > 4.0 {
            recommendations.push("Schedule immediate inspection".to_string());
            recommendations.push("Consider temporary load reduction".to_string());
        }
        
        Ok(recommendations)
    }
    
    async fn estimate_remaining_life(
        &self, 
        sensor_id: &str, 
        trends: &EquipmentTrends
    ) -> Result<u64, Box<dyn std::error::Error>> {
        // Simplified RUL estimation based on trends
        let mut rul_hours = 8760; // Default: 1 year
        
        // Reduce RUL based on negative trends
        if trends.temperature_trend > 0.1 {
            rul_hours = (rul_hours as f64 * 0.8) as u64;
        }
        
        if trends.vibration_trend > 0.05 {
            rul_hours = (rul_hours as f64 * 0.7) as u64;
        }
        
        // Equipment-specific adjustments
        match sensor_id {
            s if s.contains("motor") => rul_hours = (rul_hours as f64 * 0.9) as u64,
            s if s.contains("pump") => rul_hours = (rul_hours as f64 * 1.1) as u64,
            s if s.contains("turbine") => rul_hours = (rul_hours as f64 * 0.8) as u64,
            _ => {}
        }
        
        Ok(rul_hours.max(24)) // Minimum 24 hours
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut monitor = EquipmentMonitor::new()?;
    monitor.start_monitoring().await?;
    Ok(())
}
```

This examples guide provides comprehensive, production-ready examples across different domains, showcasing the flexibility and power of the NHITS system. Each example includes proper error handling, realistic data simulation, and practical integration patterns that can be adapted for real-world use cases.