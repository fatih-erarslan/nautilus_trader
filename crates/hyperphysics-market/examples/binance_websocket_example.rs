//! Binance WebSocket Client Example
//!
//! This example demonstrates how to use the production-ready Binance WebSocket client
//! to receive real-time market data with automatic reconnection and error handling.
//!
//! Run with:
//! ```bash
//! cargo run --example binance_websocket_example --features="market"
//! ```

use hyperphysics_market::providers::{BinanceWebSocketClient, binance_websocket::BinanceStreamMessage};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn, error};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting Binance WebSocket client example");

    // Create client (use testnet=false for production)
    let mut client = BinanceWebSocketClient::new(true)?;

    // Connect to WebSocket
    info!("Connecting to Binance WebSocket...");
    client.connect().await?;
    info!("Connected successfully!");

    // Subscribe to multiple streams
    info!("Subscribing to market data streams...");

    // Subscribe to BTC/USDT trades
    client.subscribe_trades("btcusdt").await?;
    info!("Subscribed to BTCUSDT trades");

    // Subscribe to ETH/USDT 1-minute klines
    client.subscribe_klines("ethusdt", "1m").await?;
    info!("Subscribed to ETHUSDT 1m klines");

    // Subscribe to BNB/USDT orderbook depth
    client.subscribe_depth("bnbusdt").await?;
    info!("Subscribed to BNBUSDT depth");

    // Process messages for 30 seconds
    info!("Receiving market data for 30 seconds...");
    let start = tokio::time::Instant::now();
    let duration = Duration::from_secs(30);

    let mut trade_count = 0;
    let mut kline_count = 0;
    let mut depth_count = 0;

    while start.elapsed() < duration {
        match client.next_message().await {
            Ok(Some(msg)) => {
                match msg {
                    BinanceStreamMessage::Trade(trade) => {
                        trade_count += 1;
                        info!(
                            "TRADE [{}/{}] {} @ {} x {} (maker: {})",
                            trade.symbol,
                            trade.trade_id,
                            if trade.is_buyer_maker { "SELL" } else { "BUY" },
                            trade.price,
                            trade.quantity,
                            trade.is_buyer_maker
                        );
                    }
                    BinanceStreamMessage::Kline(kline) => {
                        kline_count += 1;
                        info!(
                            "KLINE [{}] {} O:{} H:{} L:{} C:{} V:{} Trades:{} Closed:{}",
                            kline.symbol,
                            kline.kline.interval,
                            kline.kline.open,
                            kline.kline.high,
                            kline.kline.low,
                            kline.kline.close,
                            kline.kline.volume,
                            kline.kline.num_trades,
                            kline.kline.is_closed
                        );
                    }
                    BinanceStreamMessage::DepthUpdate(depth) => {
                        depth_count += 1;
                        let best_bid = depth.bids.first().map(|(p, _)| p.as_str()).unwrap_or("N/A");
                        let best_ask = depth.asks.first().map(|(p, _)| p.as_str()).unwrap_or("N/A");

                        info!(
                            "DEPTH [{}] Updates:{}-{} BestBid:{} BestAsk:{}",
                            depth.symbol,
                            depth.first_update_id,
                            depth.final_update_id,
                            best_bid,
                            best_ask
                        );
                    }
                    BinanceStreamMessage::Ticker24hr(ticker) => {
                        info!(
                            "TICKER [{}] Price:{} Change:{} ({}%)",
                            ticker.symbol,
                            ticker.last_price,
                            ticker.price_change,
                            ticker.price_change_percent
                        );
                    }
                }
            }
            Ok(None) => {
                sleep(Duration::from_millis(100)).await;
            }
            Err(e) => {
                warn!("Error receiving message: {}", e);

                // Attempt reconnection on error
                error!("Connection error, attempting to reconnect...");
                match client.reconnect().await {
                    Ok(_) => info!("Reconnected successfully"),
                    Err(e) => {
                        error!("Reconnection failed: {}", e);
                        break;
                    }
                }
            }
        }
    }

    // Summary
    info!("\n=== Session Summary ===");
    info!("Total trades received: {}", trade_count);
    info!("Total klines received: {}", kline_count);
    info!("Total depth updates received: {}", depth_count);
    info!("Session duration: {:?}", start.elapsed());

    // Graceful shutdown
    info!("Disconnecting from Binance WebSocket...");
    client.disconnect().await?;
    info!("Disconnected successfully");

    Ok(())
}
