// Unit tests for trading strategies
use rust_decimal::Decimal;
use std::str::FromStr;

#[test]
fn test_moving_average_calculation() {
    let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0];
    let period = 3;

    // Calculate simple moving average
    let ma: f64 = prices.iter().skip(prices.len() - period).sum::<f64>() / period as f64;

    // MA of last 3 prices: (101 + 103 + 105) / 3 = 103
    assert!((ma - 103.0).abs() < 0.01);
}

#[test]
fn test_exponential_moving_average() {
    let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0];
    let period = 3;
    let multiplier = 2.0 / (period as f64 + 1.0);

    // Calculate EMA
    let mut ema = prices[0];
    for &price in &prices[1..] {
        ema = (price - ema) * multiplier + ema;
    }

    assert!(ema > 100.0 && ema < 106.0);
    assert!(ema > prices.iter().sum::<f64>() / prices.len() as f64); // EMA reacts faster
}

#[test]
fn test_rsi_calculation() {
    let prices = vec![44.0, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08];

    // Calculate price changes
    let mut gains = 0.0;
    let mut losses = 0.0;

    for i in 1..prices.len() {
        let change = prices[i] - prices[i-1];
        if change > 0.0 {
            gains += change;
        } else {
            losses += change.abs();
        }
    }

    let avg_gain = gains / (prices.len() - 1) as f64;
    let avg_loss = losses / (prices.len() - 1) as f64;

    let rs = avg_gain / avg_loss;
    let rsi = 100.0 - (100.0 / (1.0 + rs));

    // RSI should be between 0 and 100
    assert!(rsi >= 0.0 && rsi <= 100.0);
}

#[test]
fn test_macd_signal() {
    // MACD = 12-period EMA - 26-period EMA
    let ema_12 = 50.5;
    let ema_26 = 49.8;

    let macd = ema_12 - ema_26;

    // Positive MACD suggests bullish momentum
    assert!(macd > 0.0);
}

#[test]
fn test_bollinger_bands() {
    let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0];
    let period = 5;
    let std_dev_multiplier = 2.0;

    // Calculate SMA
    let sma: f64 = prices.iter().skip(prices.len() - period).sum::<f64>() / period as f64;

    // Calculate standard deviation
    let recent_prices = &prices[prices.len() - period..];
    let variance: f64 = recent_prices.iter()
        .map(|&p| (p - sma).powi(2))
        .sum::<f64>() / period as f64;
    let std_dev = variance.sqrt();

    let upper_band = sma + (std_dev * std_dev_multiplier);
    let lower_band = sma - (std_dev * std_dev_multiplier);

    // Verify bands encompass price
    assert!(upper_band > sma);
    assert!(lower_band < sma);
    assert!(upper_band > prices[prices.len() - 1]);
}

#[test]
fn test_pairs_trading_spread() {
    let asset_a_price = 100.0;
    let asset_b_price = 98.0;
    let hedge_ratio = 1.0;

    let spread = asset_a_price - (hedge_ratio * asset_b_price);

    // Spread represents price differential
    assert_eq!(spread, 2.0);
}

#[test]
fn test_mean_reversion_signal() {
    let current_price = 95.0;
    let moving_average = 100.0;
    let std_dev = 5.0;

    // Calculate z-score
    let z_score = (current_price - moving_average) / std_dev;

    // Z-score of -1 suggests price is 1 std dev below mean
    assert_eq!(z_score, -1.0);

    // Signal: Buy when z-score < -2, Sell when > 2
    let entry_threshold = -2.0;
    assert!(z_score > entry_threshold); // Not yet oversold
}

#[test]
fn test_momentum_strategy() {
    let prices = vec![100.0, 105.0, 110.0, 115.0, 120.0];
    let lookback = 4;

    // Calculate momentum
    let momentum = prices[prices.len() - 1] / prices[prices.len() - lookback] - 1.0;

    // 20% gain over 4 periods
    assert!((momentum - 0.20).abs() < 0.01);
}

#[test]
fn test_volatility_breakout() {
    let high_prices = vec![102.0, 103.0, 105.0, 104.0, 108.0];
    let low_prices = vec![98.0, 99.0, 101.0, 100.0, 103.0];

    // Calculate ATR (Average True Range)
    let mut true_ranges = Vec::new();
    for i in 0..high_prices.len() {
        let tr = high_prices[i] - low_prices[i];
        true_ranges.push(tr);
    }

    let atr: f64 = true_ranges.iter().sum::<f64>() / true_ranges.len() as f64;

    // Higher ATR suggests higher volatility
    assert!(atr > 0.0);
    assert!(atr < 10.0);
}

#[test]
fn test_signal_generation() {
    let buy_signal = true;
    let sell_signal = false;

    assert!(buy_signal != sell_signal);
}
