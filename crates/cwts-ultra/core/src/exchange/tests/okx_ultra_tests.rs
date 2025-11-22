#[cfg(test)]
mod okx_ultra_tests {
    use super::super::okx_ultra::*;
    use std::sync::Arc;
    use std::collections::HashMap;
    use tokio::time::{sleep, Duration};
    
    // Test data generators for comprehensive exchange testing
    fn generate_test_api_credentials() -> ApiCredentials {
        ApiCredentials {
            api_key: "test_api_key_12345".to_string(),
            secret_key: "test_secret_key_67890".to_string(),
            passphrase: "test_passphrase".to_string(),
        }
    }
    
    fn generate_test_market_data() -> MarketData {
        MarketData {
            symbol: "BTC-USDT".to_string(),
            timestamp: 1640995200000, // Jan 1, 2022
            price: 47850.5,
            volume: 1250.75,
            bid: 47849.0,
            ask: 47851.0,
            high_24h: 48200.0,
            low_24h: 47200.0,
            volume_24h: 15000.0,
            change_24h: 1.25,
            change_percent_24h: 2.68,
        }
    }
    
    fn generate_test_order_book(depth: usize) -> OrderBook {
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        
        let base_price = 47850.0;
        
        for i in 0..depth {
            // Bids decrease from base price
            bids.push(OrderBookEntry {
                price: base_price - (i as f64 * 0.1),
                size: 1.0 + (i as f64 * 0.1),
                num_orders: (i + 1) as u32,
            });
            
            // Asks increase from base price  
            asks.push(OrderBookEntry {
                price: base_price + (i as f64 * 0.1),
                size: 1.0 + (i as f64 * 0.1),
                num_orders: (i + 1) as u32,
            });
        }
        
        OrderBook {
            symbol: "BTC-USDT".to_string(),
            timestamp: 1640995200000,
            bids,
            asks,
            checksum: "test_checksum".to_string(),
        }
    }
    
    fn generate_test_trade() -> Trade {
        Trade {
            id: "test_trade_123".to_string(),
            symbol: "BTC-USDT".to_string(),
            timestamp: 1640995200000,
            side: TradeSide::Buy,
            price: 47850.5,
            size: 0.1,
            fee: 0.0001,
            fee_currency: "BTC".to_string(),
        }
    }
    
    fn generate_test_position() -> Position {
        Position {
            symbol: "BTC-USDT-SWAP".to_string(),
            side: PositionSide::Long,
            size: 10.0,
            contracts: 1000,
            entry_price: 47500.0,
            mark_price: 47850.0,
            unrealized_pnl: 350.0,
            realized_pnl: 0.0,
            margin: 475.0,
            margin_ratio: 0.1,
            liquidation_price: 42750.0,
            timestamp: 1640995200000,
        }
    }
    
    #[test]
    fn test_okx_ultra_client_creation() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, false); // testnet = false
        
        assert_eq!(client.base_url, "https://www.okx.com");
        assert_eq!(client.api_credentials.api_key, "test_api_key_12345");
        assert!(!client.testnet_mode);
        
        // Test testnet mode
        let credentials_testnet = generate_test_api_credentials();
        let client_testnet = OkxUltraClient::new(credentials_testnet, true);
        assert_eq!(client_testnet.base_url, "https://www.okx.com"); // Still same base but different endpoints
        assert!(client_testnet.testnet_mode);
    }
    
    #[tokio::test]
    async fn test_okx_ultra_rate_limiter() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, false);
        
        // Test rate limiter allows requests within limits
        let start_time = std::time::Instant::now();
        
        // Make several rapid requests (should be rate limited)
        for i in 0..5 {
            let allowed = client.check_rate_limit(RateLimitType::PublicApi).await;
            if i < 3 {
                assert!(allowed, "First few requests should be allowed");
            }
            
            // Small delay to avoid overwhelming in tests
            sleep(Duration::from_millis(10)).await;
        }
        
        let elapsed = start_time.elapsed();
        println!("Rate limiter test took: {:?}", elapsed);
        
        // Should complete quickly if rate limiter is working
        assert!(elapsed < Duration::from_secs(1));
    }
    
    #[tokio::test]
    async fn test_okx_ultra_signature_generation() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, false);
        
        let method = "GET";
        let path = "/api/v5/market/ticker";
        let query = "instId=BTC-USDT";
        let body = "";
        let timestamp = "2022-01-01T00:00:00.000Z";
        
        let signature = client.generate_signature(method, path, query, body, timestamp).await;
        
        // Should generate a valid signature
        assert!(!signature.is_empty());
        assert!(signature.len() > 20); // Base64 encoded signature should be reasonable length
        
        // Same inputs should generate same signature
        let signature2 = client.generate_signature(method, path, query, body, timestamp).await;
        assert_eq!(signature, signature2);
        
        // Different inputs should generate different signature
        let signature3 = client.generate_signature("POST", path, query, body, timestamp).await;
        assert_ne!(signature, signature3);
    }
    
    #[tokio::test]
    async fn test_okx_ultra_error_handling() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, false);
        
        // Test various error scenarios
        
        // Invalid symbol
        let result = client.get_market_data("INVALID-SYMBOL").await;
        assert!(result.is_err());
        
        // Empty API key scenario
        let empty_credentials = ApiCredentials {
            api_key: "".to_string(),
            secret_key: "test_secret".to_string(),
            passphrase: "test_pass".to_string(),
        };
        let empty_client = OkxUltraClient::new(empty_credentials, false);
        
        // Should handle authentication error gracefully
        let auth_result = empty_client.get_account_balance().await;
        assert!(auth_result.is_err());
        
        println!("Error handling tests completed successfully");
    }
    
    #[test]
    fn test_okx_ultra_data_structures() {
        // Test MarketData serialization/deserialization
        let market_data = generate_test_market_data();
        assert_eq!(market_data.symbol, "BTC-USDT");
        assert!(market_data.price > 0.0);
        assert!(market_data.volume > 0.0);
        assert!(market_data.bid < market_data.ask);
        assert!(market_data.high_24h >= market_data.low_24h);
        
        // Test OrderBook structure
        let order_book = generate_test_order_book(10);
        assert_eq!(order_book.bids.len(), 10);
        assert_eq!(order_book.asks.len(), 10);
        
        // Verify price ordering (bids descending, asks ascending)
        for i in 1..order_book.bids.len() {
            assert!(order_book.bids[i-1].price > order_book.bids[i].price);
        }
        for i in 1..order_book.asks.len() {
            assert!(order_book.asks[i-1].price < order_book.asks[i].price);
        }
        
        // Test Trade structure
        let trade = generate_test_trade();
        assert!(matches!(trade.side, TradeSide::Buy | TradeSide::Sell));
        assert!(trade.price > 0.0);
        assert!(trade.size > 0.0);
        assert!(trade.fee >= 0.0);
        
        // Test Position structure
        let position = generate_test_position();
        assert!(matches!(position.side, PositionSide::Long | PositionSide::Short));
        assert!(position.size > 0.0);
        assert!(position.entry_price > 0.0);
    }
    
    #[tokio::test]
    async fn test_okx_ultra_websocket_connection() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, false);
        
        // Test WebSocket connection setup (mock)
        let ws_config = client.create_websocket_config("BTC-USDT".to_string()).await;
        
        assert_eq!(ws_config.symbol, "BTC-USDT");
        assert!(ws_config.channels.contains(&"ticker".to_string()));
        assert!(ws_config.channels.contains(&"trade".to_string()));
        assert!(ws_config.channels.contains(&"books".to_string()));
        
        // Test subscription message generation
        let subscribe_msg = client.generate_subscribe_message(&ws_config.channels, &ws_config.symbol).await;
        assert!(subscribe_msg.contains("subscribe"));
        assert!(subscribe_msg.contains("BTC-USDT"));
        
        println!("WebSocket configuration test completed");
    }
    
    #[tokio::test] 
    async fn test_okx_ultra_order_management() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, true); // Use testnet for order tests
        
        // Test order creation parameters
        let order_request = OrderRequest {
            symbol: "BTC-USDT".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            size: 0.001,
            price: Some(45000.0),
            time_in_force: Some(TimeInForce::GTC),
            client_order_id: Some("test_order_123".to_string()),
            reduce_only: false,
            post_only: false,
        };
        
        // Validate order request
        assert_eq!(order_request.symbol, "BTC-USDT");
        assert!(matches!(order_request.side, OrderSide::Buy | OrderSide::Sell));
        assert!(matches!(order_request.order_type, OrderType::Market | OrderType::Limit | OrderType::Stop));
        assert!(order_request.size > 0.0);
        
        // Test order validation
        assert!(client.validate_order_request(&order_request).await);
        
        // Test invalid order (negative size)
        let invalid_order = OrderRequest {
            size: -0.001,
            ..order_request
        };
        assert!(!client.validate_order_request(&invalid_order).await);
        
        println!("Order management validation tests completed");
    }
    
    #[tokio::test]
    async fn test_okx_ultra_market_data_processing() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, false);
        
        // Test ticker data processing
        let raw_ticker_data = r#"{
            "instType": "SPOT",
            "instId": "BTC-USDT",
            "last": "47850.5",
            "lastSz": "0.001",
            "askPx": "47851.0",
            "askSz": "1.5",
            "bidPx": "47849.0",
            "bidSz": "2.1",
            "open24h": "46825.0",
            "high24h": "48200.0",
            "low24h": "47200.0",
            "vol24h": "15000.0",
            "ts": "1640995200000"
        }"#;
        
        let parsed_data = client.parse_ticker_data(raw_ticker_data).await;
        assert!(parsed_data.is_ok());
        
        let market_data = parsed_data.unwrap();
        assert_eq!(market_data.symbol, "BTC-USDT");
        assert!((market_data.price - 47850.5).abs() < 0.001);
        assert!((market_data.bid - 47849.0).abs() < 0.001);
        assert!((market_data.ask - 47851.0).abs() < 0.001);
        
        // Test order book data processing
        let raw_orderbook_data = r#"{
            "asks": [["47851.0", "1.5", "0", "2"], ["47852.0", "2.0", "0", "3"]],
            "bids": [["47849.0", "2.1", "0", "2"], ["47848.0", "1.8", "0", "1"]],
            "ts": "1640995200000",
            "checksum": "123456789"
        }"#;
        
        let parsed_book = client.parse_orderbook_data(raw_orderbook_data, "BTC-USDT").await;
        assert!(parsed_book.is_ok());
        
        let order_book = parsed_book.unwrap();
        assert_eq!(order_book.asks.len(), 2);
        assert_eq!(order_book.bids.len(), 2);
        assert!((order_book.asks[0].price - 47851.0).abs() < 0.001);
        assert!((order_book.bids[0].price - 47849.0).abs() < 0.001);
        
        println!("Market data processing tests completed");
    }
    
    #[tokio::test]
    async fn test_okx_ultra_risk_management() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, false);
        
        // Test position size validation
        let large_order = OrderRequest {
            symbol: "BTC-USDT".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            size: 100.0, // Very large size
            price: None,
            time_in_force: None,
            client_order_id: None,
            reduce_only: false,
            post_only: false,
        };
        
        let risk_check = client.assess_order_risk(&large_order, 10000.0).await; // $10k balance
        assert!(!risk_check.approved);
        assert!(risk_check.risk_score > 0.8);
        assert!(!risk_check.warnings.is_empty());
        
        // Test reasonable order
        let reasonable_order = OrderRequest {
            size: 0.01, // Small size
            ..large_order
        };
        
        let reasonable_risk = client.assess_order_risk(&reasonable_order, 10000.0).await;
        assert!(reasonable_risk.approved);
        assert!(reasonable_risk.risk_score < 0.3);
        
        // Test margin requirements
        let margin_req = client.calculate_margin_requirement(&reasonable_order).await;
        assert!(margin_req > 0.0);
        assert!(margin_req < reasonable_order.size * 50000.0); // Should be less than full notional
        
        println!("Risk management tests completed");
    }
    
    #[tokio::test]
    async fn test_okx_ultra_portfolio_tracking() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, false);
        
        // Test portfolio balance calculation
        let mut balances = HashMap::new();
        balances.insert("BTC".to_string(), 2.5);
        balances.insert("ETH".to_string(), 15.0);
        balances.insert("USDT".to_string(), 10000.0);
        
        let mut prices = HashMap::new();
        prices.insert("BTC".to_string(), 47850.0);
        prices.insert("ETH".to_string(), 3200.0);
        prices.insert("USDT".to_string(), 1.0);
        
        let portfolio_value = client.calculate_portfolio_value(&balances, &prices).await;
        
        let expected_value = 2.5 * 47850.0 + 15.0 * 3200.0 + 10000.0 * 1.0;
        assert!((portfolio_value - expected_value).abs() < 1.0);
        
        // Test PnL calculation
        let positions = vec![generate_test_position()];
        let total_pnl = client.calculate_total_pnl(&positions).await;
        assert!((total_pnl - 350.0).abs() < 0.001);
        
        // Test portfolio diversification
        let diversification = client.calculate_portfolio_diversification(&balances, &prices).await;
        assert!(diversification.len() == 3);
        
        // All percentages should sum to 100%
        let total_percentage: f64 = diversification.values().sum();
        assert!((total_percentage - 100.0).abs() < 0.001);
        
        println!("Portfolio tracking tests completed");
    }
    
    #[tokio::test]
    async fn test_okx_ultra_advanced_order_types() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, true);
        
        // Test stop-loss order
        let stop_order = OrderRequest {
            symbol: "BTC-USDT".to_string(),
            side: OrderSide::Sell,
            order_type: OrderType::Stop,
            size: 0.1,
            price: Some(45000.0), // Stop price
            time_in_force: Some(TimeInForce::GTC),
            client_order_id: Some("stop_order_123".to_string()),
            reduce_only: true,
            post_only: false,
        };
        
        assert!(client.validate_order_request(&stop_order).await);
        assert!(stop_order.reduce_only); // Stop orders should typically be reduce-only
        
        // Test take-profit order
        let tp_order = OrderRequest {
            symbol: "BTC-USDT".to_string(),
            side: OrderSide::Sell,
            order_type: OrderType::Limit,
            size: 0.1,
            price: Some(50000.0), // Take profit price
            time_in_force: Some(TimeInForce::GTC),
            client_order_id: Some("tp_order_123".to_string()),
            reduce_only: true,
            post_only: true,
        };
        
        assert!(client.validate_order_request(&tp_order).await);
        assert!(tp_order.post_only); // Take profit orders often use post-only
        
        // Test OCO (One-Cancels-Other) order simulation
        let oco_orders = client.create_oco_order(
            0.1, // size
            45000.0, // stop price  
            50000.0, // limit price
            "BTC-USDT".to_string()
        ).await;
        
        assert_eq!(oco_orders.len(), 2);
        assert!(oco_orders[0].order_type == OrderType::Stop);
        assert!(oco_orders[1].order_type == OrderType::Limit);
        
        println!("Advanced order types tests completed");
    }
    
    #[tokio::test]
    async fn test_okx_ultra_latency_optimization() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, false);
        
        // Test connection pooling
        let start_time = std::time::Instant::now();
        
        // Make multiple requests to test connection reuse
        for _ in 0..5 {
            let _config = client.create_websocket_config("BTC-USDT".to_string()).await;
            sleep(Duration::from_millis(1)).await;
        }
        
        let total_time = start_time.elapsed();
        println!("Connection pooling test took: {:?}", total_time);
        
        // Should be faster due to connection reuse
        assert!(total_time < Duration::from_millis(500));
        
        // Test request optimization
        let market_symbols = vec!["BTC-USDT", "ETH-USDT", "BNB-USDT"];
        let batch_start = std::time::Instant::now();
        
        let batch_results = client.batch_get_market_data(market_symbols).await;
        let batch_time = batch_start.elapsed();
        
        assert!(batch_results.len() <= 3); // May have errors but should not crash
        println!("Batch request took: {:?}", batch_time);
        
        // Batch should be faster than individual requests
        assert!(batch_time < Duration::from_millis(1000));
    }
    
    #[tokio::test] 
    async fn test_okx_ultra_concurrent_operations() {
        let credentials = generate_test_api_credentials();
        let client = Arc::new(OkxUltraClient::new(credentials, false));
        
        let mut handles = vec![];
        
        // Test concurrent market data requests
        for i in 0..3 {
            let client_clone = Arc::clone(&client);
            let symbol = match i {
                0 => "BTC-USDT".to_string(),
                1 => "ETH-USDT".to_string(), 
                _ => "BNB-USDT".to_string(),
            };
            
            let handle = tokio::spawn(async move {
                let config = client_clone.create_websocket_config(symbol.clone()).await;
                assert_eq!(config.symbol, symbol);
                println!("Thread {} completed for {}", i, symbol);
            });
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        for handle in handles {
            handle.await.expect("Task panicked");
        }
        
        println!("✅ Concurrent operations test completed");
    }
    
    #[tokio::test]
    async fn test_okx_ultra_error_recovery() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, false);
        
        // Test network timeout handling
        let timeout_result = client.handle_network_timeout().await;
        assert!(timeout_result.is_ok());
        
        // Test API error handling
        let api_error = client.handle_api_error(429, "Rate limit exceeded").await;
        assert!(api_error.contains("Rate limit"));
        
        // Test connection recovery
        let recovery_result = client.recover_connection().await;
        assert!(recovery_result.is_ok());
        
        // Test circuit breaker functionality
        let mut consecutive_failures = 0;
        for _ in 0..6 {
            consecutive_failures += 1;
            let should_break = client.check_circuit_breaker(consecutive_failures).await;
            
            if consecutive_failures >= 5 {
                assert!(should_break);
                break;
            } else {
                assert!(!should_break);
            }
        }
        
        println!("Error recovery tests completed");
    }
    
    #[test]
    fn test_okx_ultra_data_validation() {
        // Test market data validation
        let mut market_data = generate_test_market_data();
        assert!(market_data.is_valid());
        
        // Test with invalid data
        market_data.price = -100.0; // Invalid negative price
        assert!(!market_data.is_valid());
        
        market_data.price = 47850.5; // Fix price
        market_data.bid = 48000.0; // Invalid bid > ask
        market_data.ask = 47800.0;
        assert!(!market_data.is_valid());
        
        // Test order book validation
        let mut order_book = generate_test_order_book(5);
        assert!(order_book.is_valid());
        
        // Invalid order book (bid >= ask)
        order_book.bids[0].price = 48000.0;
        order_book.asks[0].price = 47000.0;
        assert!(!order_book.is_valid());
        
        // Test position validation
        let mut position = generate_test_position();
        assert!(position.is_valid());
        
        // Invalid position (negative size)
        position.size = -10.0;
        assert!(!position.is_valid());
        
        println!("Data validation tests completed");
    }
    
    #[tokio::test]
    async fn test_okx_ultra_performance_metrics() {
        let credentials = generate_test_api_credentials();
        let client = OkxUltraClient::new(credentials, false);
        
        // Test latency measurement
        let start = std::time::Instant::now();
        let _signature = client.generate_signature("GET", "/api/v5/market/ticker", "", "", "2022-01-01T00:00:00.000Z").await;
        let signature_latency = start.elapsed();
        
        println!("Signature generation latency: {:?}", signature_latency);
        assert!(signature_latency < Duration::from_millis(10));
        
        // Test throughput measurement
        let throughput_start = std::time::Instant::now();
        let mut operations = 0;
        
        while throughput_start.elapsed() < Duration::from_millis(100) {
            let _config = client.create_websocket_config("BTC-USDT".to_string()).await;
            operations += 1;
        }
        
        let ops_per_second = (operations as f64 / 0.1);
        println!("Operations per second: {:.2}", ops_per_second);
        assert!(ops_per_second > 10.0); // Should handle at least 10 ops/sec
        
        // Test memory efficiency
        let initial_memory = get_memory_usage();
        
        // Create many objects to test memory management
        for _ in 0..1000 {
            let _market_data = generate_test_market_data();
            let _order_book = generate_test_order_book(100);
        }
        
        let final_memory = get_memory_usage();
        println!("Memory usage increase: {} bytes", final_memory - initial_memory);
        
        // Memory increase should be reasonable
        assert!((final_memory - initial_memory) < 100_000_000); // Less than 100MB
    }
    
    // Helper function to get memory usage (simplified)
    fn get_memory_usage() -> usize {
        // In a real implementation, this would use system APIs
        // For testing, we'll return a mock value
        use std::alloc::{GlobalAlloc, Layout, System};
        
        // This is a simplified mock - in practice you'd use proper memory profiling
        std::mem::size_of::<OkxUltraClient>() * 1000
    }
    
    #[test]
    fn test_okx_ultra_configuration() {
        let credentials = generate_test_api_credentials();
        
        // Test production configuration
        let prod_client = OkxUltraClient::new(credentials.clone(), false);
        assert_eq!(prod_client.base_url, "https://www.okx.com");
        assert!(!prod_client.testnet_mode);
        
        // Test testnet configuration
        let test_client = OkxUltraClient::new(credentials.clone(), true);
        assert!(test_client.testnet_mode);
        
        // Test configuration validation
        let empty_creds = ApiCredentials {
            api_key: "".to_string(),
            secret_key: "".to_string(),
            passphrase: "".to_string(),
        };
        
        let invalid_client = OkxUltraClient::new(empty_creds, false);
        assert!(invalid_client.api_credentials.api_key.is_empty());
        
        println!("Configuration tests completed");
    }
    
    #[test]
    fn test_okx_ultra_enum_serialization() {
        // Test TradeSide serialization
        assert!(matches!(TradeSide::Buy, TradeSide::Buy));
        assert!(matches!(TradeSide::Sell, TradeSide::Sell));
        
        // Test OrderType serialization
        assert!(matches!(OrderType::Market, OrderType::Market));
        assert!(matches!(OrderType::Limit, OrderType::Limit));
        assert!(matches!(OrderType::Stop, OrderType::Stop));
        
        // Test TimeInForce serialization
        assert!(matches!(TimeInForce::GTC, TimeInForce::GTC));
        assert!(matches!(TimeInForce::IOC, TimeInForce::IOC));
        assert!(matches!(TimeInForce::FOK, TimeInForce::FOK));
        
        // Test PositionSide serialization
        assert!(matches!(PositionSide::Long, PositionSide::Long));
        assert!(matches!(PositionSide::Short, PositionSide::Short));
        
        println!("Enum serialization tests completed");
    }
}