//! Real Market Data Validation Tests
//!
//! Test forecasting accuracy on actual financial time series data

use neuro_divergent::*;
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
struct MarketDataPoint {
    timestamp: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

/// Load real market data from CSV
fn load_market_data(symbol: &str) -> Result<Vec<f64>> {
    let data_path = format!(
        "{}/test-data/market/{}.csv",
        env!("CARGO_MANIFEST_DIR"),
        symbol
    );

    let contents = fs::read_to_string(&data_path)
        .map_err(|e| NeuroDivergentError::IoError(e))?;

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(contents.as_bytes());

    let close_prices: Vec<f64> = reader
        .deserialize()
        .filter_map(|result: std::result::Result<MarketDataPoint, _>| {
            result.ok().map(|point| point.close)
        })
        .collect();

    if close_prices.is_empty() {
        return Err(NeuroDivergentError::DataError(
            format!("No data loaded for symbol {}", symbol)
        ));
    }

    Ok(close_prices)
}

#[test]
#[ignore] // Requires market data files
fn test_sp500_forecasting_accuracy() {
    // Load S&P 500 historical data
    let data = load_market_data("SPY").expect("Failed to load SPY data");

    // Use 80% for training, 20% for testing
    let split_idx = (data.len() as f64 * 0.8) as usize;
    let train_data = &data[..split_idx];
    let test_data = &data[split_idx..];

    let horizon = test_data.len().min(30); // Forecast up to 30 days

    // TODO: Implement model testing
    // let mut model = NHITSModel::new(config);
    // model.fit(train_data).expect("Training on SPY failed");
    //
    // let prediction = model.predict(horizon).expect("Prediction failed");
    //
    // // Calculate metrics
    // let mape = compute_mape(&prediction.mean, &test_data[..horizon]);
    // let rmse = compute_rmse(&prediction.mean, &test_data[..horizon]);
    //
    // println!("S&P 500 Forecasting Results:");
    // println!("  MAPE: {:.2}%", mape * 100.0);
    // println!("  RMSE: {:.4}", rmse);
    //
    // // Financial forecasting is challenging - expect reasonable but not perfect accuracy
    // assert!(mape < 0.20, "MAPE too high: {:.2}%", mape * 100.0);

    println!("S&P 500 test placeholder - target MAPE: <20%");
}

#[test]
#[ignore]
fn test_bitcoin_volatility_forecasting() {
    let data = load_market_data("BTC-USD").expect("Failed to load BTC data");

    let split_idx = (data.len() as f64 * 0.8) as usize;
    let train_data = &data[..split_idx];
    let test_data = &data[split_idx..];

    let horizon = 7; // 1 week forecast

    // TODO: Test with high-volatility asset
    // let mut model = TFTModel::new(config); // TFT handles volatility well
    // model.fit(train_data).expect("Training on BTC failed");
    //
    // let prediction = model.predict(horizon).expect("Prediction failed");
    //
    // // Check prediction intervals capture actual values
    // let interval_95 = prediction.intervals.iter()
    //     .find(|i| (i.level - 0.95).abs() < 0.01)
    //     .unwrap();
    //
    // let mut coverage_count = 0;
    // for i in 0..horizon {
    //     if test_data[i] >= interval_95.lower[i] && test_data[i] <= interval_95.upper[i] {
    //         coverage_count += 1;
    //     }
    // }
    //
    // let coverage = coverage_count as f64 / horizon as f64;
    // println!("Bitcoin 95% Interval Coverage: {:.2}%", coverage * 100.0);
    //
    // // 95% interval should capture ~95% of actual values
    // assert!(coverage >= 0.85, "Interval coverage too low: {:.2}%", coverage * 100.0);

    println!("Bitcoin volatility test placeholder");
}

#[test]
#[ignore]
fn test_forex_intraday_prediction() {
    let data = load_market_data("EURUSD").expect("Failed to load EUR/USD data");

    // Use recent data for intraday pattern
    let recent_data = &data[data.len().saturating_sub(1000)..];

    let split_idx = (recent_data.len() as f64 * 0.8) as usize;
    let train_data = &recent_data[..split_idx];
    let test_data = &recent_data[split_idx..];

    let horizon = 24; // Next 24 hours

    // TODO: Test intraday patterns
    println!("EUR/USD intraday test placeholder");
}

#[test]
#[ignore]
fn test_commodity_seasonality() {
    let data = load_market_data("GLD").expect("Failed to load Gold data");

    // TODO: Test seasonal pattern detection
    // let mut model = NBEATSModel::new(config);
    // model.fit(&data).expect("Training on Gold failed");
    //
    // // NBEATS should decompose into trend and seasonality
    // let decomposition = model.decompose(&data).expect("Decomposition failed");
    //
    // assert!(decomposition.seasonality.iter().any(|&x| x.abs() > 0.01));

    println!("Gold seasonality test placeholder");
}

#[test]
#[ignore]
fn test_multi_asset_ensemble() {
    let symbols = vec!["SPY", "BTC-USD", "EURUSD", "GLD"];
    let mut all_predictions = Vec::new();

    for symbol in symbols {
        let data = load_market_data(symbol)
            .expect(&format!("Failed to load {}", symbol));

        // TODO: Train model per asset
        // let mut model = NHITSModel::new(config);
        // model.fit(&data).expect("Training failed");
        // let pred = model.predict(7).expect("Prediction failed");
        // all_predictions.push(pred);
    }

    // TODO: Create cross-asset ensemble
    println!("Multi-asset ensemble test placeholder");
}

#[test]
#[ignore]
fn test_crisis_period_robustness() {
    // Test on 2008 financial crisis period
    let data = load_market_data("SPY_2008").expect("Failed to load crisis data");

    // TODO: Test model robustness during extreme volatility
    // Models should still produce valid predictions even during crises
    println!("Crisis period robustness test placeholder");
}

// Helper functions
fn compute_mape(predictions: &[f64], actuals: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(actuals.iter())
        .map(|(&pred, &actual)| ((actual - pred) / actual).abs())
        .sum::<f64>()
        / predictions.len() as f64
}

fn compute_rmse(predictions: &[f64], actuals: &[f64]) -> f64 {
    let mse = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(&pred, &actual)| (actual - pred).powi(2))
        .sum::<f64>()
        / predictions.len() as f64;

    mse.sqrt()
}

fn compute_directional_accuracy(predictions: &[f64], actuals: &[f64]) -> f64 {
    let mut correct = 0;

    for i in 1..predictions.len() {
        let pred_direction = predictions[i] > predictions[i - 1];
        let actual_direction = actuals[i] > actuals[i - 1];

        if pred_direction == actual_direction {
            correct += 1;
        }
    }

    correct as f64 / (predictions.len() - 1) as f64
}

#[test]
fn test_metric_calculations() {
    let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let actuals = vec![1.1, 1.9, 3.2, 3.8, 5.1];

    let mape = compute_mape(&predictions, &actuals);
    let rmse = compute_rmse(&predictions, &actuals);
    let dir_acc = compute_directional_accuracy(&predictions, &actuals);

    assert!(mape >= 0.0 && mape <= 1.0);
    assert!(rmse >= 0.0);
    assert!(dir_acc >= 0.0 && dir_acc <= 1.0);

    println!("Metrics: MAPE={:.4}, RMSE={:.4}, Dir Acc={:.4}", mape, rmse, dir_acc);
}
