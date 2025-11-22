// Kraken Ultra Example - Real-world usage examples
use crate::exchange::kraken_ultra::*;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn, error};

/// Example: Basic Kraken connection and market data
pub async fn basic_market_data_example() -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting Kraken basic market data example");
    
    // Create Kraken instance for spot trading
    let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
    
    // Connect to public WebSocket
    kraken.connect_websocket(false).await?;
    info!("Connected to Kraken WebSocket");
    
    // Subscribe to multiple data streams
    kraken.subscribe_order_book(vec!["BTC/USD".to_string(), "ETH/USD".to_string()], Some(10)).await?;
    kraken.subscribe_trades(vec!["BTC/USD".to_string(), "ETH/USD".to_string()]).await?;
    kraken.subscribe_ticker(vec!["BTC/USD".to_string(), "ETH/USD".to_string()]).await?;
    kraken.subscribe_ohlc(vec!["BTC/USD".to_string()], 1).await?; // 1 minute candles
    
    info!("Subscribed to market data streams");
    
    // Start message processing
    kraken.start_message_loop().await?;
    
    // Let it run for 30 seconds to collect data
    sleep(Duration::from_secs(30)).await;
    
    // Print some data
    if let Some(order_book) = kraken.get_order_book("BTC/USD") {
        info!("BTC/USD Order Book - Best bid: {}, Best ask: {}", 
            order_book.bids.first().map(|b| &b.price).unwrap_or(&"N/A".to_string()),
            order_book.asks.first().map(|a| &a.price).unwrap_or(&"N/A".to_string())
        );
    }
    
    if let Some(ticker) = kraken.get_ticker("BTC/USD") {
        info!("BTC/USD Ticker - Last: {}, Volume: {}", ticker.last, ticker.volume);
    }
    
    let recent_trades = kraken.get_recent_trades("BTC/USD", 5);
    info!("Recent BTC/USD trades: {}", recent_trades.len());
    
    // Disconnect
    kraken.disconnect().await?;
    info!("Disconnected from Kraken WebSocket");
    
    Ok(())
}

/// Example: Authenticated operations with real trading
pub async fn authenticated_trading_example() -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting Kraken authenticated trading example");
    
    // Create Kraken instance
    let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
    
    // Set credentials (use environment variables in real applications)
    let api_key = std::env::var("KRAKEN_API_KEY").unwrap_or_else(|_| "your_api_key".to_string());
    let api_secret = std::env::var("KRAKEN_API_SECRET").unwrap_or_else(|_| "your_api_secret".to_string());
    
    if api_key == "your_api_key" {
        warn!("Using dummy credentials - set KRAKEN_API_KEY and KRAKEN_API_SECRET for real trading");
        return Ok(());
    }
    
    kraken.set_credentials(api_key, api_secret, None);
    
    // Get account information
    match kraken.get_balance().await {
        Ok(balance) => {
            info!("Account balance: {:?}", balance);
        }
        Err(e) => {
            error!("Failed to get balance: {}", e);
            return Err(e);
        }
    }
    
    // Get trade balance
    match kraken.get_trade_balance(Some("USD")).await {
        Ok(trade_balance) => {
            info!("Trade balance: {:?}", trade_balance);
        }
        Err(e) => {
            error!("Failed to get trade balance: {}", e);
        }
    }
    
    // Connect to authenticated WebSocket
    kraken.connect_websocket(true).await?;
    info!("Connected to authenticated WebSocket");
    
    // Subscribe to private data
    kraken.subscribe_own_trades().await?;
    kraken.subscribe_open_orders().await?;
    
    // Start message processing
    kraken.start_message_loop().await?;
    
    // Example: Place a small limit order (TESTING ONLY - WILL PLACE REAL ORDER!)
    let small_order_result = kraken.place_limit_order(
        "BTCUSD",
        "buy",
        "0.001", // Very small amount for testing
        "30000.0", // Price well below market
        None, // No leverage
        None, // Standard time in force
    ).await;
    
    match small_order_result {
        Ok(order_response) => {
            info!("Order placed successfully: {:?}", order_response);
            
            // Wait a moment then cancel the order
            sleep(Duration::from_secs(2)).await;
            
            if let Some(txid) = order_response.txid.first() {
                match kraken.cancel_order(txid).await {
                    Ok(cancel_response) => {
                        info!("Order cancelled: {:?}", cancel_response);
                    }
                    Err(e) => {
                        error!("Failed to cancel order: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            error!("Failed to place order: {}", e);
        }
    }
    
    // Get open orders
    match kraken.get_open_orders(Some(true), None).await {
        Ok(orders) => {
            info!("Open orders: {:?}", orders);
        }
        Err(e) => {
            error!("Failed to get open orders: {}", e);
        }
    }
    
    // Let it run for a bit to see private data updates
    sleep(Duration::from_secs(10)).await;
    
    // Disconnect
    kraken.disconnect().await?;
    info!("Disconnected from authenticated WebSocket");
    
    Ok(())
}

/// Example: Market making strategy simulation
pub async fn market_making_example() -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting Kraken market making example");
    
    let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
    
    // Set credentials
    let api_key = std::env::var("KRAKEN_API_KEY").unwrap_or_else(|_| "your_api_key".to_string());
    let api_secret = std::env::var("KRAKEN_API_SECRET").unwrap_or_else(|_| "your_api_secret".to_string());
    
    if api_key == "your_api_key" {
        warn!("Simulating market making without real orders");
    } else {
        kraken.set_credentials(api_key, api_secret, None);
    }
    
    // Connect to WebSocket
    kraken.connect_websocket(false).await?;
    
    // Subscribe to order book for BTC/USD
    kraken.subscribe_order_book(vec!["BTC/USD".to_string()], Some(20)).await?;
    kraken.subscribe_trades(vec!["BTC/USD".to_string()]).await?;
    
    // Start message processing
    kraken.start_message_loop().await?;
    
    // Wait for initial data
    sleep(Duration::from_secs(5)).await;
    
    // Market making simulation
    for i in 0..10 {
        if let Some(order_book) = kraken.get_order_book("BTC/USD") {
            if let (Some(best_bid), Some(best_ask)) = (order_book.bids.first(), order_book.asks.first()) {
                let bid_price: f64 = best_bid.price.parse().unwrap_or(0.0);
                let ask_price: f64 = best_ask.price.parse().unwrap_or(0.0);
                let mid_price = (bid_price + ask_price) / 2.0;
                let spread = ask_price - bid_price;
                
                info!("Cycle {}: Mid: {:.2}, Spread: {:.2}", i + 1, mid_price, spread);
                
                // Simulate placing orders with tighter spread
                let our_bid = mid_price - spread * 0.25;
                let our_ask = mid_price + spread * 0.25;
                
                info!("Would place: BID {:.2} @ 0.001 BTC, ASK {:.2} @ 0.001 BTC", our_bid, our_ask);
                
                // In real implementation, you would:
                // 1. Check if we have existing orders to cancel
                // 2. Cancel old orders if price moved too much
                // 3. Place new bid and ask orders
                // 4. Monitor fills and adjust position
                
                if kraken.credentials.is_some() {
                    // Example of actual order placement (commented out for safety)
                    /*
                    let bid_order = kraken.place_limit_order(
                        "BTCUSD",
                        "buy",
                        "0.001",
                        &format!("{:.2}", our_bid),
                        None,
                        Some("IOC") // Immediate or cancel
                    ).await;
                    
                    let ask_order = kraken.place_limit_order(
                        "BTCUSD",
                        "sell",
                        "0.001",
                        &format!("{:.2}", our_ask),
                        None,
                        Some("IOC")
                    ).await;
                    */
                }
            }
        }
        
        sleep(Duration::from_secs(10)).await;
    }
    
    kraken.disconnect().await?;
    info!("Market making simulation completed");
    
    Ok(())
}

/// Example: Risk management and position monitoring
pub async fn risk_management_example() -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting Kraken risk management example");
    
    let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
    
    // Set credentials
    let api_key = std::env::var("KRAKEN_API_KEY").unwrap_or_else(|_| "your_api_key".to_string());
    let api_secret = std::env::var("KRAKEN_API_SECRET").unwrap_or_else(|_| "your_api_secret".to_string());
    
    if api_key == "your_api_key" {
        warn!("Using dummy credentials for risk management simulation");
        return Ok(());
    }
    
    kraken.set_credentials(api_key, api_secret, None);
    
    // Connect to authenticated WebSocket
    kraken.connect_websocket(true).await?;
    
    // Subscribe to account updates
    kraken.subscribe_own_trades().await?;
    kraken.subscribe_open_orders().await?;
    
    // Subscribe to market data for position monitoring
    kraken.connect_websocket(false).await?;
    kraken.subscribe_ticker(vec!["BTC/USD".to_string(), "ETH/USD".to_string()]).await?;
    
    kraken.start_message_loop().await?;
    
    // Risk monitoring loop
    for i in 0..20 {
        // Get current balances
        match kraken.get_balance().await {
            Ok(balance_response) => {
                info!("Balance check {}: {:?}", i + 1, balance_response);
                
                // Extract balance info and check exposure
                if let Some(result) = balance_response.get("result") {
                    // Check BTC exposure
                    if let Some(btc_balance) = result.get("XXBT") {
                        let btc_amount: f64 = btc_balance.as_str().unwrap_or("0").parse().unwrap_or(0.0);
                        
                        // Get current BTC price
                        if let Some(ticker) = kraken.get_ticker("BTC/USD") {
                            let btc_price: f64 = ticker.last.parse().unwrap_or(0.0);
                            let btc_exposure_usd = btc_amount * btc_price;
                            
                            info!("BTC Position: {:.6} BTC (~${:.2} USD)", btc_amount, btc_exposure_usd);
                            
                            // Risk check: if exposure > $10,000, consider reducing
                            if btc_exposure_usd > 10000.0 {
                                warn!("HIGH BTC EXPOSURE: ${:.2} - Consider reducing position", btc_exposure_usd);
                            }
                            
                            // Risk check: if price dropped more than 5% from some reference
                            // (In real implementation, you'd track entry prices)
                            let reference_price = 50000.0; // Example reference
                            let price_change_pct = (btc_price - reference_price) / reference_price * 100.0;
                            
                            if price_change_pct < -5.0 {
                                warn!("BTC down {:.1}% from reference - Consider stop loss", price_change_pct);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to get balance: {}", e);
            }
        }
        
        // Check open orders for stuck orders
        match kraken.get_open_orders(Some(false), None).await {
            Ok(orders_response) => {
                if let Some(result) = orders_response.get("result") {
                    if let Some(open_orders) = result.get("open").and_then(|o| o.as_object()) {
                        info!("Open orders count: {}", open_orders.len());
                        
                        for (order_id, order_data) in open_orders {
                            if let Some(open_time) = order_data.get("opentm").and_then(|t| t.as_f64()) {
                                let current_time = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as f64;
                                
                                let age_minutes = (current_time - open_time) / 60.0;
                                
                                // Cancel orders older than 30 minutes
                                if age_minutes > 30.0 {
                                    warn!("Canceling old order {} (age: {:.1} min)", order_id, age_minutes);
                                    
                                    match kraken.cancel_order(order_id).await {
                                        Ok(_) => info!("Cancelled old order {}", order_id),
                                        Err(e) => error!("Failed to cancel order {}: {}", order_id, e),
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to get open orders: {}", e);
            }
        }
        
        sleep(Duration::from_secs(30)).await;
    }
    
    kraken.disconnect().await?;
    info!("Risk management monitoring completed");
    
    Ok(())
}

/// Example: Multi-timeframe analysis
pub async fn multi_timeframe_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting Kraken multi-timeframe analysis example");
    
    let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
    
    // Connect to WebSocket
    kraken.connect_websocket(false).await?;
    
    // Subscribe to multiple timeframes
    kraken.subscribe_ohlc(vec!["BTC/USD".to_string()], 1).await?;    // 1 minute
    kraken.subscribe_ohlc(vec!["BTC/USD".to_string()], 5).await?;    // 5 minute
    kraken.subscribe_ohlc(vec!["BTC/USD".to_string()], 60).await?;   // 1 hour
    kraken.subscribe_ohlc(vec!["BTC/USD".to_string()], 1440).await?; // 1 day
    
    // Subscribe to real-time data
    kraken.subscribe_trades(vec!["BTC/USD".to_string()]).await?;
    kraken.subscribe_ticker(vec!["BTC/USD".to_string()]).await?;
    
    kraken.start_message_loop().await?;
    
    // Let it collect data
    sleep(Duration::from_secs(60)).await;
    
    // Analysis loop
    for i in 0..10 {
        info!("--- Analysis cycle {} ---", i + 1);
        
        // Get OHLC data for different timeframes
        let ohlc_1m = kraken.get_ohlc_data("BTC/USD", 1, 20);
        let ohlc_5m = kraken.get_ohlc_data("BTC/USD", 5, 20);
        let ohlc_1h = kraken.get_ohlc_data("BTC/USD", 60, 20);
        
        info!("OHLC Data available - 1m: {}, 5m: {}, 1h: {}", 
              ohlc_1m.len(), ohlc_5m.len(), ohlc_1h.len());
        
        // Simple trend analysis
        if !ohlc_1h.is_empty() && ohlc_1h.len() >= 5 {
            let recent_closes: Vec<f64> = ohlc_1h.iter()
                .take(5)
                .map(|candle| candle.close.parse().unwrap_or(0.0))
                .collect();
            
            if recent_closes.len() >= 2 {
                let trend = if recent_closes[0] > recent_closes[recent_closes.len() - 1] {
                    "UPTREND"
                } else {
                    "DOWNTREND"
                };
                
                info!("1H Trend: {} (Current: {:.2}, 5 candles ago: {:.2})", 
                      trend, recent_closes[0], recent_closes[recent_closes.len() - 1]);
            }
        }
        
        // Volume analysis
        if !ohlc_5m.is_empty() {
            let avg_volume: f64 = ohlc_5m.iter()
                .take(10)
                .map(|candle| candle.volume.parse().unwrap_or(0.0))
                .sum::<f64>() / ohlc_5m.len().min(10) as f64;
            
            if let Some(latest) = ohlc_5m.first() {
                let latest_volume: f64 = latest.volume.parse().unwrap_or(0.0);
                let volume_ratio = latest_volume / avg_volume;
                
                if volume_ratio > 2.0 {
                    info!("HIGH VOLUME ALERT: {:.2}x average (Latest: {:.2}, Avg: {:.2})", 
                          volume_ratio, latest_volume, avg_volume);
                }
            }
        }
        
        // Price momentum
        let recent_trades = kraken.get_recent_trades("BTC/USD", 50);
        if !recent_trades.is_empty() {
            let buy_trades = recent_trades.iter().filter(|t| t.side == "buy").count();
            let sell_trades = recent_trades.iter().filter(|t| t.side == "sell").count();
            let buy_ratio = buy_trades as f64 / recent_trades.len() as f64;
            
            info!("Recent trade sentiment: {:.1}% buys, {:.1}% sells", 
                  buy_ratio * 100.0, (1.0 - buy_ratio) * 100.0);
            
            if buy_ratio > 0.7 {
                info!("BULLISH MOMENTUM: {:.1}% recent trades are buys", buy_ratio * 100.0);
            } else if buy_ratio < 0.3 {
                info!("BEARISH MOMENTUM: {:.1}% recent trades are sells", (1.0 - buy_ratio) * 100.0);
            }
        }
        
        sleep(Duration::from_secs(30)).await;
    }
    
    kraken.disconnect().await?;
    info!("Multi-timeframe analysis completed");
    
    Ok(())
}

/// Example: Error handling and reconnection
pub async fn error_handling_example() -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting Kraken error handling example");
    
    let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
    
    // Test connection with retry logic
    let mut connection_attempts = 0;
    let max_attempts = 5;
    
    while connection_attempts < max_attempts {
        connection_attempts += 1;
        info!("Connection attempt {}/{}", connection_attempts, max_attempts);
        
        match kraken.connect_websocket(false).await {
            Ok(_) => {
                info!("Connected successfully on attempt {}", connection_attempts);
                break;
            }
            Err(e) => {
                error!("Connection attempt {} failed: {}", connection_attempts, e);
                if connection_attempts >= max_attempts {
                    return Err(format!("Failed to connect after {} attempts", max_attempts).into());
                }
                
                // Exponential backoff
                let delay = Duration::from_secs(2_u64.pow(connection_attempts - 1));
                info!("Retrying in {:?}", delay);
                sleep(delay).await;
            }
        }
    }
    
    // Subscribe with error handling
    let symbols = vec!["BTC/USD".to_string(), "ETH/USD".to_string(), "INVALID/PAIR".to_string()];
    
    for symbol in &symbols {
        match kraken.subscribe_ticker(vec![symbol.clone()]).await {
            Ok(_) => info!("Successfully subscribed to {}", symbol),
            Err(e) => warn!("Failed to subscribe to {}: {}", symbol, e),
        }
    }
    
    // Start message loop with error handling
    kraken.start_message_loop().await?;
    
    // Simulate connection issues and recovery
    for cycle in 0..10 {
        info!("Monitoring cycle {}", cycle + 1);
        
        // Check connection status
        if !kraken.is_connected() {
            warn!("Connection lost, attempting to reconnect...");
            
            match kraken.reconnect_websocket(false, 1).await {
                Ok(_) => info!("Reconnected successfully"),
                Err(e) => error!("Reconnection failed: {}", e),
            }
        }
        
        // Test API endpoint with rate limiting
        match kraken.get_system_status().await {
            Ok(status) => {
                if let Some(result) = status.get("result") {
                    info!("System status: {:?}", result.get("status"));
                }
            }
            Err(e) => {
                warn!("API call failed: {}", e);
                // Could implement circuit breaker pattern here
            }
        }
        
        sleep(Duration::from_secs(5)).await;
    }
    
    kraken.disconnect().await?;
    info!("Error handling example completed");
    
    Ok(())
}

/// Run all examples
pub async fn run_all_kraken_examples() -> Result<(), Box<dyn std::error::Error>> {
    info!("Running all Kraken examples");
    
    // Run examples (some may be skipped based on credentials)
    if let Err(e) = basic_market_data_example().await {
        error!("Basic market data example failed: {}", e);
    }
    
    sleep(Duration::from_secs(2)).await;
    
    if let Err(e) = authenticated_trading_example().await {
        error!("Authenticated trading example failed: {}", e);
    }
    
    sleep(Duration::from_secs(2)).await;
    
    if let Err(e) = market_making_example().await {
        error!("Market making example failed: {}", e);
    }
    
    sleep(Duration::from_secs(2)).await;
    
    if let Err(e) = risk_management_example().await {
        error!("Risk management example failed: {}", e);
    }
    
    sleep(Duration::from_secs(2)).await;
    
    if let Err(e) = multi_timeframe_analysis_example().await {
        error!("Multi-timeframe analysis example failed: {}", e);
    }
    
    sleep(Duration::from_secs(2)).await;
    
    if let Err(e) = error_handling_example().await {
        error!("Error handling example failed: {}", e);
    }
    
    info!("All Kraken examples completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Integration test
    async fn test_basic_example() {
        let _ = basic_market_data_example().await;
    }

    #[tokio::test]
    #[ignore] // Requires credentials
    async fn test_authenticated_example() {
        let _ = authenticated_trading_example().await;
    }
}