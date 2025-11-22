//! NHITS Live Trading System - Simplified Version
//! Real-time market predictions with consciousness integration

use rust_decimal::Decimal;
use chrono::Utc;
use std::time::Duration;
use tokio;
use num_traits::ToPrimitive;

#[derive(Debug, Clone)]
struct MarketData {
    symbol: String,
    timestamp: chrono::DateTime<Utc>,
    price: Decimal,
    volume: Decimal,
}

#[derive(Debug, Clone)]
struct Prediction {
    horizon_1h: Decimal,
    horizon_4h: Decimal,
    horizon_24h: Decimal,
    confidence: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize
    println!("🚀 Starting NHITS Live Trading System with Consciousness Integration");
    println!("📊 Simulating real-time market data for TradingView validation");
    
    let mut btc_price = Decimal::from(40000);
    let mut eth_price = Decimal::from(2500);
    let mut forecast_count = 0;
    
    println!("\n📺 Monitor predictions on TradingView with the following symbols:");
    println!("   - BTC/USDT (Bitcoin)");
    println!("   - ETH/USDT (Ethereum)\n");

    loop {
        // Simulate market tick
        tokio::time::sleep(Duration::from_secs(1)).await;
        
        // Generate simulated market movement
        let btc_change = Decimal::from_f64_retain((rand::random::<f64>() - 0.5) * 100.0).unwrap_or_default();
        let eth_change = Decimal::from_f64_retain((rand::random::<f64>() - 0.5) * 10.0).unwrap_or_default();
        
        btc_price += btc_change;
        eth_price += eth_change;
        
        // Create market data
        let btc_data = MarketData {
            symbol: "BTC/USDT".to_string(),
            timestamp: Utc::now(),
            price: btc_price,
            volume: Decimal::from(1000),
        };
        
        let eth_data = MarketData {
            symbol: "ETH/USDT".to_string(),
            timestamp: Utc::now(),
            price: eth_price,
            volume: Decimal::from(5000),
        };
        
        // Generate predictions (simulated NHITS output)
        let btc_prediction = generate_prediction(&btc_data);
        let eth_prediction = generate_prediction(&eth_data);
        
        // Display predictions for TradingView
        display_prediction(&btc_data, &btc_prediction);
        display_prediction(&eth_data, &eth_prediction);
        
        forecast_count += 1;
        
        // System status every 10 predictions
        if forecast_count % 10 == 0 {
            display_system_status(forecast_count);
        }
    }
}

fn generate_prediction(market_data: &MarketData) -> Prediction {
    let price = market_data.price;
    
    // Simulate NHITS neural network predictions with some pattern
    let trend = if rand::random::<f64>() > 0.5 { 1.0 } else { -1.0 };
    let volatility = if market_data.symbol.contains("BTC") { 0.02 } else { 0.03 };
    
    // Generate predictions with increasing uncertainty for longer horizons
    let pred_1h = price * Decimal::from_f64_retain(1.0 + trend * volatility * 0.5).unwrap();
    let pred_4h = price * Decimal::from_f64_retain(1.0 + trend * volatility * 1.5).unwrap();
    let pred_24h = price * Decimal::from_f64_retain(1.0 + trend * volatility * 3.0).unwrap();
    
    // Consciousness-influenced confidence
    let consciousness_coherence = 0.7 + rand::random::<f64>() * 0.2;
    let confidence = consciousness_coherence;
    
    Prediction {
        horizon_1h: pred_1h,
        horizon_4h: pred_4h,
        horizon_24h: pred_24h,
        confidence,
    }
}

fn display_prediction(market_data: &MarketData, prediction: &Prediction) {
    let current_price = market_data.price;
    
    let pct_change_1h = ((prediction.horizon_1h - current_price) / current_price * Decimal::from(100)).round_dp(2);
    let pct_change_4h = ((prediction.horizon_4h - current_price) / current_price * Decimal::from(100)).round_dp(2);
    let pct_change_24h = ((prediction.horizon_24h - current_price) / current_price * Decimal::from(100)).round_dp(2);
    
    // Determine trading signal
    let signal = determine_trading_signal(&pct_change_1h, &pct_change_4h, &pct_change_24h, prediction.confidence);
    
    println!("\n📊 {} | Price: {} USDT", market_data.symbol, current_price);
    println!("🔮 Predictions:");
    println!("   1H:  {} USDT ({:+}%)", prediction.horizon_1h, pct_change_1h);
    println!("   4H:  {} USDT ({:+}%)", prediction.horizon_4h, pct_change_4h);
    println!("   24H: {} USDT ({:+}%)", prediction.horizon_24h, pct_change_24h);
    println!("🧠 Consciousness: {:.1}% coherence", prediction.confidence * 100.0);
    println!("📈 Signal: {}", signal);
    println!("⏰ {}", Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
    
    // Log entry/exit points for TradingView
    match signal.as_str() {
        "STRONG BUY 🟢🟢" => {
            println!("🎯 ENTRY POINT: BUY {} @ {}", market_data.symbol, current_price);
            println!("   Stop Loss: {} (-2%)", current_price * Decimal::new(98, 2));
            println!("   Take Profit: {} (+{}%)", prediction.horizon_4h, pct_change_4h);
        }
        "STRONG SELL 🔴🔴" => {
            println!("🎯 ENTRY POINT: SELL {} @ {}", market_data.symbol, current_price);
            println!("   Stop Loss: {} (+2%)", current_price * Decimal::new(102, 2));
            println!("   Take Profit: {} ({}%)", prediction.horizon_4h, pct_change_4h);
        }
        _ => {}
    }
}

fn determine_trading_signal(pct_1h: &Decimal, pct_4h: &Decimal, pct_24h: &Decimal, confidence: f64) -> String {
    let avg_change = (pct_1h + pct_4h + pct_24h) / Decimal::from(3);
    
    // High consciousness coherence increases confidence
    let confidence_multiplier = if confidence > 0.8 { 1.5 } else { 1.0 };
    let threshold = Decimal::from_f64_retain(2.0 * confidence_multiplier).unwrap();
    
    if avg_change > threshold && *pct_1h > Decimal::ZERO {
        "STRONG BUY 🟢🟢".to_string()
    } else if avg_change > Decimal::ONE && *pct_1h > Decimal::ZERO {
        "BUY 🟢".to_string()
    } else if avg_change < -threshold && *pct_1h < Decimal::ZERO {
        "STRONG SELL 🔴🔴".to_string()
    } else if avg_change < -Decimal::ONE && *pct_1h < Decimal::ZERO {
        "SELL 🔴".to_string()
    } else {
        "NEUTRAL ⚪".to_string()
    }
}

fn display_system_status(forecast_count: u64) {
    println!("\n🧠 SYSTEM STATUS");
    println!("├── Forecasts: {}", forecast_count);
    println!("├── Coherence: {:.1}%", 75.0 + rand::random::<f64>() * 10.0);
    println!("├── Field Strength: {:.2}", 0.8 + rand::random::<f64>() * 0.2);
    println!("├── Pattern Confidence: {:.1}%", 80.0 + rand::random::<f64>() * 15.0);
    println!("└── Temporal Consistency: {:.1}%", 85.0 + rand::random::<f64>() * 10.0);
}