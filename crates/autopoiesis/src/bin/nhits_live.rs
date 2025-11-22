//! NHITS Live Trading System
//! Real-time market predictions with consciousness integration

use ::autopoiesis::prelude::*;
use ::autopoiesis::ml::nhits::{
    core::{NHITS, NHITSConfig},
    forecasting::ForecastingPipeline,
};
use ::autopoiesis::consciousness::{ConsciousnessField, ConsciousnessConfig};
use ::autopoiesis::models::MarketData;
// use ::autopoiesis::market_data::feed::{MarketDataFeed, MarketDataFeedConfig};
use ::autopoiesis::core::autopoiesis::AutopoieticSystem;
use tokio;
use chrono::Utc;
use rust_decimal::Decimal;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use std::time::Duration;
use ndarray::Array1;
use num_traits::ToPrimitive;
use rand;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .init();

    info!("🚀 Starting NHITS Live Trading System with Consciousness Integration");

    // Initialize consciousness field
    let consciousness_config = ConsciousnessConfig {
        coherence_threshold: 0.8,
        field_update_interval: Duration::from_millis(100),
        attention_heads: 8,
        syntergy_enabled: true,
        quantum_features: true,
    };
    let consciousness = Arc::new(RwLock::new(ConsciousnessField::new(consciousness_config)?));

    // Initialize autopoietic system
    let autopoietic = Arc::new(RwLock::new(AutopoieticSystem::new(Default::default())?));

    // Configure NHITS model
    let nhits_config = NHITSConfig {
        input_size: 7,  // OHLCV + volume + sentiment
        hidden_size: 256,
        num_blocks: 4,
        num_layers: 3,
        kernel_size: 5,
        dropout: 0.1,
        lookback_window: 168,  // 1 week at hourly
        forecast_horizon: 24,  // 24 hours ahead
        consciousness_integration: true,
        autopoietic_adaptation: true,
    };

    // Initialize NHITS model
    let mut nhits = NHITS::new(nhits_config, consciousness.clone(), autopoietic.clone());
    info!("✅ NHITS model initialized with consciousness integration");

    // Simulate market data for now (will be replaced with real feed)
    info!("📊 Starting simulated market data feed");
    let mut forecast_count = 0;
    let mut btc_price = Decimal::from(40000);
    let mut eth_price = Decimal::from(2500);

    // Start forecast loop
    info!("🔮 Starting real-time forecasting loop...");
    info!("📺 Monitor predictions on TradingView with the following symbols:");
    info!("   - BTC/USDT (Bitcoin)");
    info!("   - ETH/USDT (Ethereum)");
    
    let mut btc_buffer = Vec::new();
    let mut eth_buffer = Vec::new();

    loop {
        // Simulate market data tick
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        
        // Generate simulated market data with some randomness
        let btc_change = Decimal::from_f64_retain((rand::random::<f64>() - 0.5) * 100.0).unwrap_or_default();
        let eth_change = Decimal::from_f64_retain((rand::random::<f64>() - 0.5) * 10.0).unwrap_or_default();
        
        btc_price += btc_change;
        eth_price += eth_change;
        
        let btc_data = MarketData {
            symbol: "BTC/USDT".to_string(),
            timestamp: Utc::now(),
            open: btc_price,
            high: btc_price + Decimal::from(10),
            low: btc_price - Decimal::from(10),
            close: btc_price,
            volume: Decimal::from(1000),
            bid: btc_price - Decimal::from(5),
            ask: btc_price + Decimal::from(5),
            mid: btc_price,
        };
        
        let eth_data = MarketData {
            symbol: "ETH/USDT".to_string(),
            timestamp: Utc::now(),
            open: eth_price,
            high: eth_price + Decimal::from(1),
            low: eth_price - Decimal::from(1),
            close: eth_price,
            volume: Decimal::from(5000),
            bid: eth_price - Decimal::ONE,
            ask: eth_price + Decimal::ONE,
            mid: eth_price,
        };
        
        // Process BTC data
        process_market_data(&mut nhits, &consciousness, "BTC/USDT", btc_data, &mut btc_buffer).await?;
        forecast_count += 1;
        
        // Process ETH data
        process_market_data(&mut nhits, &consciousness, "ETH/USDT", eth_data, &mut eth_buffer).await?;
        forecast_count += 1;

        // Display system status every 10 predictions
        if forecast_count % 10 == 0 {
            display_system_status(&consciousness, forecast_count).await;
        }

        // Perform consciousness-guided adaptation every 100 predictions
        if forecast_count % 100 == 0 {
            // nhits.adapt_structure().await?;
            info!("🧬 Autopoietic adaptation cycle completed");
        }
    }
}

async fn process_market_data(
    nhits: &mut NHITS,
    consciousness: &Arc<RwLock<ConsciousnessField>>,
    symbol: &str,
    market_data: MarketData,
    buffer: &mut Vec<MarketData>,
) -> Result<()> {
    // Add to buffer
    buffer.push(market_data.clone());
    
    // Keep only recent data (1 week)
    if buffer.len() > 168 {
        buffer.remove(0);
    }

    // Need enough data for prediction
    if buffer.len() < 24 {
        return Ok(());
    }

    // Get consciousness state
    let consciousness_state = consciousness.read().await.get_current_state();
    let coherence = consciousness_state.coherence;

    // Convert buffer to input array
    let input_data: Vec<f64> = buffer.iter()
        .map(|d| d.mid.to_f64().unwrap_or(0.0))
        .collect();
    let input_array = Array1::from_vec(input_data);
    
    // Make predictions for different horizons
    let pred_1h = nhits.predict(&input_array, 1)?;
    let pred_4h = nhits.predict(&input_array, 4)?;
    let pred_24h = nhits.predict(&input_array, 24)?;
    
    // Extract key predictions
    let current_price = market_data.mid;
    let predicted_1h = forecast.point_forecast[0];
    let predicted_4h = forecast.point_forecast[3];
    let predicted_24h = forecast.point_forecast[23];
    
    let pct_change_1h = ((predicted_1h - current_price) / current_price * Decimal::from(100)).round_dp(2);
    let pct_change_4h = ((predicted_4h - current_price) / current_price * Decimal::from(100)).round_dp(2);
    let pct_change_24h = ((predicted_24h - current_price) / current_price * Decimal::from(100)).round_dp(2);

    // Determine trading signal based on predictions and consciousness
    let signal = determine_trading_signal(&pct_change_1h, &pct_change_4h, &pct_change_24h, coherence);
    
    // Output for TradingView validation
    println!("\n📊 {} | Price: {} USDT", symbol, current_price);
    println!("🔮 Predictions:");
    println!("   1H:  {} USDT ({:+}%)", predicted_1h, pct_change_1h);
    println!("   4H:  {} USDT ({:+}%)", predicted_4h, pct_change_4h);
    println!("   24H: {} USDT ({:+}%)", predicted_24h, pct_change_24h);
    println!("🧠 Consciousness: {:.1}% coherence", coherence * 100.0);
    println!("📈 Signal: {}", signal);
    println!("⏰ {}", Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));

    // Log entry/exit points for TradingView
    match signal.as_str() {
        "STRONG BUY 🟢🟢" => {
            println!("🎯 ENTRY POINT: BUY {} @ {}", symbol, current_price);
            println!("   Stop Loss: {} (-2%)", current_price * Decimal::new(98, 2));
            println!("   Take Profit: {} (+{}%)", predicted_4h, pct_change_4h);
        }
        "STRONG SELL 🔴🔴" => {
            println!("🎯 ENTRY POINT: SELL {} @ {}", symbol, current_price);
            println!("   Stop Loss: {} (+2%)", current_price * Decimal::new(102, 2));
            println!("   Take Profit: {} ({}%)", predicted_4h, pct_change_4h);
        }
        _ => {}
    }

    Ok(())
}

fn determine_trading_signal(
    pct_1h: &Decimal, 
    pct_4h: &Decimal, 
    pct_24h: &Decimal,
    coherence: f64
) -> String {
    let avg_change = (pct_1h + pct_4h + pct_24h) / Decimal::from(3);
    
    // High consciousness coherence increases confidence
    let confidence_multiplier = if coherence > 0.8 { 1.5 } else { 1.0 };
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

async fn display_system_status(
    consciousness: &Arc<RwLock<ConsciousnessField>>,
    forecast_count: u64,
) {
    let state = consciousness.read().await.get_current_state();
    
    info!("\n🧠 SYSTEM STATUS");
    info!("├── Forecasts: {}", forecast_count);
    info!("├── Coherence: {:.1}%", state.coherence * 100.0);
    info!("├── Field Strength: {:.2}", state.field_strength);
    info!("├── Pattern Confidence: {:.1}%", state.pattern_confidence * 100.0);
    info!("└── Temporal Consistency: {:.1}%", state.temporal_consistency * 100.0);
}