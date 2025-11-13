//! Example: Fetch market data from Alpaca Markets API
//!
//! This example demonstrates real API integration with:
//! - Historical bar data fetching
//! - Latest bar retrieval
//! - Symbol validation
//! - Rate limiting
//! - Error handling
//!
//! # Prerequisites
//!
//! Set environment variables:
//! - ALPACA_API_KEY: Your Alpaca API key
//! - ALPACA_API_SECRET: Your Alpaca API secret
//!
//! # Usage
//!
//! ```bash
//! export ALPACA_API_KEY="your_key_here"
//! export ALPACA_API_SECRET="your_secret_here"
//! cargo run --example alpaca_fetch_bars
//! ```

use chrono::{Duration, Utc};
use hyperphysics_market::data::Timeframe;
use hyperphysics_market::providers::{AlpacaProvider, MarketDataProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt::init();

    println!("=== Alpaca Markets API Integration Example ===\n");

    // Read API credentials from environment
    let api_key = std::env::var("ALPACA_API_KEY")
        .expect("ALPACA_API_KEY environment variable not set");
    let api_secret = std::env::var("ALPACA_API_SECRET")
        .expect("ALPACA_API_SECRET environment variable not set");

    // Create provider instance (paper trading mode)
    let provider = AlpacaProvider::new(api_key, api_secret, true);

    println!("Provider: {}", provider.provider_name());
    println!("Real-time support: {}\n", provider.supports_realtime());

    // Example 1: Validate symbols
    println!("--- Symbol Validation ---");
    let symbols = vec!["AAPL", "INVALID_SYMBOL", "TSLA"];

    for symbol in &symbols {
        match provider.supports_symbol(symbol).await {
            Ok(supported) => {
                println!(
                    "{}: {}",
                    symbol,
                    if supported { "Valid" } else { "Invalid" }
                );
            }
            Err(e) => println!("{}: Error - {}", symbol, e),
        }
    }
    println!();

    // Example 2: Fetch latest bar
    println!("--- Latest Bar Data ---");
    match provider.fetch_latest_bar("AAPL").await {
        Ok(bar) => {
            println!("Symbol: {}", bar.symbol);
            println!("Timestamp: {}", bar.timestamp);
            println!("Open: ${:.2}", bar.open);
            println!("High: ${:.2}", bar.high);
            println!("Low: ${:.2}", bar.low);
            println!("Close: ${:.2}", bar.close);
            println!("Volume: {}", bar.volume);
            if let Some(vwap) = bar.vwap {
                println!("VWAP: ${:.2}", vwap);
            }
            println!();
        }
        Err(e) => {
            eprintln!("Error fetching latest bar: {}", e);
        }
    }

    // Example 3: Fetch historical bars
    println!("--- Historical Bar Data (Last 7 Days) ---");
    let end = Utc::now();
    let start = end - Duration::days(7);

    match provider
        .fetch_bars("AAPL", Timeframe::Day1, start, end)
        .await
    {
        Ok(bars) => {
            println!("Fetched {} daily bars", bars.len());
            println!("\nFirst 3 bars:");

            for bar in bars.iter().take(3) {
                println!(
                    "{} | O: ${:.2} H: ${:.2} L: ${:.2} C: ${:.2} V: {}",
                    bar.timestamp.format("%Y-%m-%d"),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume
                );
            }

            if bars.len() > 3 {
                println!("... ({} more bars)", bars.len() - 3);
            }
            println!();
        }
        Err(e) => {
            eprintln!("Error fetching historical bars: {}", e);
        }
    }

    // Example 4: Fetch intraday data
    println!("--- Intraday Data (Last Hour, 5-Minute Bars) ---");
    let end = Utc::now();
    let start = end - Duration::hours(1);

    match provider
        .fetch_bars("SPY", Timeframe::Minute5, start, end)
        .await
    {
        Ok(bars) => {
            println!("Fetched {} 5-minute bars for SPY", bars.len());

            if !bars.is_empty() {
                let latest = &bars[bars.len() - 1];
                println!(
                    "\nMost recent bar: {} | Close: ${:.2} | Volume: {}",
                    latest.timestamp.format("%H:%M"),
                    latest.close,
                    latest.volume
                );
            }
            println!();
        }
        Err(e) => {
            eprintln!("Error fetching intraday data: {}", e);
        }
    }

    // Example 5: Calculate basic statistics
    println!("--- Price Statistics (Last 30 Days) ---");
    let end = Utc::now();
    let start = end - Duration::days(30);

    match provider
        .fetch_bars("AAPL", Timeframe::Day1, start, end)
        .await
    {
        Ok(bars) if !bars.is_empty() => {
            let high = bars.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
            let low = bars.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
            let avg_close = bars.iter().map(|b| b.close).sum::<f64>() / bars.len() as f64;
            let total_volume: u64 = bars.iter().map(|b| b.volume).sum();

            println!("30-Day Statistics for AAPL:");
            println!("Highest price: ${:.2}", high);
            println!("Lowest price: ${:.2}", low);
            println!("Average close: ${:.2}", avg_close);
            println!("Total volume: {}", total_volume);
            println!("Trading days: {}", bars.len());
        }
        Ok(_) => println!("No data available"),
        Err(e) => {
            eprintln!("Error calculating statistics: {}", e);
        }
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
