// Bybit Ultra Usage Example
use crate::exchange::bybit_ultra::{BybitUltra, BybitMarketType};
use tokio::time::{sleep, Duration};
use tracing::{info, warn, error};

/// Example demonstrating Bybit Ultra integration
pub async fn bybit_example() -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting Bybit Ultra Example");

    // 1. Create Bybit exchange instance for Linear (Perpetual) trading
    let mut bybit = BybitUltra::new(BybitMarketType::Linear);
    info!("Created Bybit Ultra instance for Linear markets");

    // 2. Set API credentials (replace with real credentials for live trading)
    bybit.set_credentials(
        "YOUR_API_KEY".to_string(),
        "YOUR_API_SECRET".to_string(),
        true // Use testnet
    );
    info!("Set API credentials for testnet");

    // 3. Connect to public WebSocket for market data
    match bybit.connect_websocket(false).await {
        Ok(_) => {
            info!("Connected to Bybit public WebSocket");
            
            // 4. Subscribe to order book updates
            if let Err(e) = bybit.subscribe_orderbook("BTCUSDT").await {
                error!("Failed to subscribe to orderbook: {}", e);
                return Err(e);
            }
            info!("Subscribed to BTCUSDT orderbook");

            // 5. Subscribe to trade stream
            if let Err(e) = bybit.subscribe_trades("BTCUSDT").await {
                error!("Failed to subscribe to trades: {}", e);
                return Err(e);
            }
            info!("Subscribed to BTCUSDT trades");

            // 6. Subscribe to kline data (1m interval)
            if let Err(e) = bybit.subscribe_klines("BTCUSDT", "1").await {
                error!("Failed to subscribe to klines: {}", e);
                return Err(e);
            }
            info!("Subscribed to BTCUSDT 1m klines");

            // 7. Start message processing loop
            if let Err(e) = bybit.start_message_loop().await {
                error!("Failed to start message loop: {}", e);
                return Err(e);
            }
            info!("Started message processing loop");

            // 8. Let it run for 10 seconds to collect data
            info!("Collecting market data for 10 seconds...");
            sleep(Duration::from_secs(10)).await;

            // 9. Check received order book data
            if let Some(orderbook) = bybit.get_orderbook("BTCUSDT") {
                info!("Current BTCUSDT orderbook:");
                info!("  Best bid: {} @ {}", orderbook.b[0][1], orderbook.b[0][0]);
                info!("  Best ask: {} @ {}", orderbook.a[0][1], orderbook.a[0][0]);
                info!("  Update ID: {}", orderbook.u);
            } else {
                warn!("No orderbook data received");
            }

            // 10. Disconnect from public WebSocket
            if let Err(e) = bybit.disconnect().await {
                error!("Failed to disconnect: {}", e);
                return Err(e);
            }
            info!("Disconnected from public WebSocket");

        }
        Err(e) => {
            error!("Failed to connect to WebSocket: {}", e);
            return Err(e);
        }
    }

    // 11. Connect to private WebSocket for account data
    match bybit.connect_websocket(true).await {
        Ok(_) => {
            info!("Connected to Bybit private WebSocket");
            
            // 12. Subscribe to position updates
            if let Err(e) = bybit.subscribe_positions().await {
                error!("Failed to subscribe to positions: {}", e);
                return Err(e);
            }
            info!("Subscribed to position updates");

            // 13. Subscribe to order updates
            if let Err(e) = bybit.subscribe_orders().await {
                error!("Failed to subscribe to orders: {}", e);
                return Err(e);
            }
            info!("Subscribed to order updates");

            // 14. Subscribe to wallet updates
            if let Err(e) = bybit.subscribe_wallet().await {
                error!("Failed to subscribe to wallet: {}", e);
                return Err(e);
            }
            info!("Subscribed to wallet updates");

            // 15. Start message processing for private data
            if let Err(e) = bybit.start_message_loop().await {
                error!("Failed to start private message loop: {}", e);
                return Err(e);
            }
            info!("Started private message processing");

            // 16. Let it run for 5 seconds to receive account data
            info!("Collecting account data for 5 seconds...");
            sleep(Duration::from_secs(5)).await;

            // 17. Check current positions
            let positions = bybit.get_current_positions();
            if !positions.is_empty() {
                info!("Current positions:");
                for (key, position) in positions.iter() {
                    info!("  {}: {} {} @ {}", 
                        key, position.side, position.size, position.avg_price);
                }
            } else {
                info!("No open positions");
            }

            // 18. Check current orders
            let orders = bybit.get_current_orders();
            if !orders.is_empty() {
                info!("Current orders:");
                for (order_id, order) in orders.iter() {
                    info!("  {}: {} {} {} @ {}", 
                        order_id, order.symbol, order.side, order.qty, order.price);
                }
            } else {
                info!("No open orders");
            }

            // 19. Check wallet information
            if let Some(wallet) = bybit.get_current_wallet() {
                info!("Current wallet:");
                info!("  Total equity: {}", wallet.total_equity);
                info!("  Available balance: {}", wallet.total_available_balance);
                info!("  Account type: {}", wallet.account_type);
            } else {
                info!("No wallet data received");
            }

            // 20. Disconnect from private WebSocket
            if let Err(e) = bybit.disconnect().await {
                error!("Failed to disconnect from private WebSocket: {}", e);
                return Err(e);
            }
            info!("Disconnected from private WebSocket");

        }
        Err(e) => {
            error!("Failed to connect to private WebSocket: {}", e);
            return Err(e);
        }
    }

    info!("Bybit Ultra Example completed successfully");
    Ok(())
}

/// Example demonstrating REST API trading operations
pub async fn bybit_trading_example() -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting Bybit Trading Example");

    let mut bybit = BybitUltra::new(BybitMarketType::Linear);
    
    // Set credentials (replace with real credentials)
    bybit.set_credentials(
        "YOUR_API_KEY".to_string(),
        "YOUR_API_SECRET".to_string(),
        true // testnet
    );

    // 1. Get account information
    match bybit.get_account_info().await {
        Ok(account_info) => {
            info!("Account info: {}", account_info);
        }
        Err(e) => {
            error!("Failed to get account info: {}", e);
            return Err(e);
        }
    }

    // 2. Get wallet balance
    match bybit.get_wallet_balance("UNIFIED").await {
        Ok(wallet) => {
            info!("Wallet balance: {}", wallet);
        }
        Err(e) => {
            error!("Failed to get wallet balance: {}", e);
            return Err(e);
        }
    }

    // 3. Get current positions
    match bybit.get_positions(Some("BTCUSDT")).await {
        Ok(positions) => {
            info!("Positions: {}", positions);
        }
        Err(e) => {
            error!("Failed to get positions: {}", e);
            return Err(e);
        }
    }

    // 4. Get open orders
    match bybit.get_open_orders(Some("BTCUSDT")).await {
        Ok(orders) => {
            info!("Open orders: {}", orders);
        }
        Err(e) => {
            error!("Failed to get open orders: {}", e);
            return Err(e);
        }
    }

    // 5. Place a limit order (example - be careful with real trading!)
    match bybit.place_order(
        "BTCUSDT",
        "Buy",
        "Limit",
        "0.001",
        Some("30000.0"),
        Some("GTC"),
        None,
        None
    ).await {
        Ok(order_result) => {
            info!("Order placed: {}", order_result);
        }
        Err(e) => {
            warn!("Order placement failed (expected on testnet without funds): {}", e);
        }
    }

    info!("Bybit Trading Example completed");
    Ok(())
}

/// Example showing error handling and reconnection logic
pub async fn bybit_resilience_example() -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting Bybit Resilience Example");

    let mut bybit = BybitUltra::new(BybitMarketType::Linear);
    bybit.set_credentials(
        "YOUR_API_KEY".to_string(),
        "YOUR_API_SECRET".to_string(),
        true
    );

    // Demonstrate reconnection logic
    for attempt in 1..=3 {
        info!("Connection attempt {}", attempt);
        
        match bybit.connect_websocket(false).await {
            Ok(_) => {
                info!("Connected successfully on attempt {}", attempt);
                break;
            }
            Err(e) => {
                warn!("Connection attempt {} failed: {}", attempt, e);
                
                if attempt < 3 {
                    // Use built-in reconnection logic
                    match bybit.reconnect_websocket(false, attempt).await {
                        Ok(_) => {
                            info!("Reconnected successfully");
                            break;
                        }
                        Err(reconnect_error) => {
                            error!("Reconnection failed: {}", reconnect_error);
                        }
                    }
                } else {
                    error!("All connection attempts failed");
                    return Err(e);
                }
            }
        }
    }

    info!("Bybit Resilience Example completed");
    Ok(())
}

/// Performance monitoring example
pub async fn bybit_performance_example() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    
    info!("Starting Bybit Performance Example");

    let mut bybit = BybitUltra::new(BybitMarketType::Linear);
    
    // Measure connection time
    let start = Instant::now();
    match bybit.connect_websocket(false).await {
        Ok(_) => {
            let connection_time = start.elapsed();
            info!("WebSocket connection established in {:?}", connection_time);
            
            // Measure subscription time
            let sub_start = Instant::now();
            if let Ok(_) = bybit.subscribe_orderbook("BTCUSDT").await {
                let subscription_time = sub_start.elapsed();
                info!("Orderbook subscription completed in {:?}", subscription_time);
            }
            
            // Measure message processing performance
            bybit.start_message_loop().await?;
            
            let data_start = Instant::now();
            sleep(Duration::from_secs(5)).await;
            let data_time = data_start.elapsed();
            
            if let Some(_orderbook) = bybit.get_orderbook("BTCUSDT") {
                info!("Received orderbook data in {:?}", data_time);
            }
            
            bybit.disconnect().await?;
        }
        Err(e) => {
            error!("Performance test failed: {}", e);
            return Err(e);
        }
    }

    info!("Bybit Performance Example completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    #[ignore = "requires network connection"]
    async fn test_bybit_example() {
        let result = bybit_example().await;
        // Don't assert success since this requires real network/API access
        match result {
            Ok(_) => println!("Bybit example completed successfully"),
            Err(e) => println!("Bybit example failed (expected without real credentials): {}", e),
        }
    }
}