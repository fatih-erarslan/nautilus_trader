// Kraken Ultra Tests - Comprehensive testing suite
#[cfg(test)]
mod tests {
    use super::super::kraken_ultra::*;
    use tokio;
    use std::time::Duration;

    #[tokio::test]
    async fn test_kraken_ultra_creation() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        assert_eq!(kraken.market_type, KrakenMarketType::Spot);
        assert!(!kraken.is_connected());
        assert!(kraken.credentials.is_none());
    }

    #[tokio::test]
    async fn test_kraken_ultra_futures_creation() {
        let kraken = KrakenUltra::new(KrakenMarketType::Futures);
        assert_eq!(kraken.market_type, KrakenMarketType::Futures);
        assert!(!kraken.is_connected());
    }

    #[tokio::test]
    async fn test_credential_setting() {
        let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
        kraken.set_credentials(
            "test_api_key".to_string(),
            "test_api_secret".to_string(),
            Some("test_passphrase".to_string())
        );
        
        assert!(kraken.credentials.is_some());
        let creds = kraken.credentials.unwrap();
        assert_eq!(creds.api_key, "test_api_key");
        assert_eq!(creds.api_secret, "test_api_secret");
        assert_eq!(creds.api_passphrase, Some("test_passphrase".to_string()));
    }

    #[tokio::test]
    async fn test_nonce_generation() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        let nonce1 = kraken.generate_nonce();
        tokio::time::sleep(Duration::from_millis(1)).await;
        let nonce2 = kraken.generate_nonce();
        
        assert!(nonce2 > nonce1);
        assert!(nonce1 > 0);
    }

    #[tokio::test]
    async fn test_multiple_nonce_generation() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        let mut nonces = Vec::new();
        
        for _ in 0..10 {
            nonces.push(kraken.generate_nonce());
        }
        
        // All nonces should be unique and increasing
        for i in 1..nonces.len() {
            assert!(nonces[i] > nonces[i-1]);
        }
    }

    #[tokio::test]
    async fn test_websocket_url_selection() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        assert_eq!(kraken.get_ws_url(false), KRAKEN_WS_PUBLIC);
        assert_eq!(kraken.get_ws_url(true), KRAKEN_WS_AUTH);
    }

    #[tokio::test]
    async fn test_signature_generation() {
        let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
        kraken.set_credentials(
            "test_key".to_string(),
            "a2VybmVsL3NlY3JldA==".to_string(), // base64 encoded "kernel/secret"
            None
        );
        
        let endpoint = "/0/private/Balance";
        let nonce = 1234567890u64;
        let post_data = format!("nonce={}", nonce);
        
        let signature = kraken.generate_signature(endpoint, nonce, &post_data);
        assert!(signature.is_ok());
        
        let sig = signature.unwrap();
        assert!(!sig.is_empty());
        assert!(base64::decode(&sig).is_ok());
        
        // Test with different parameters
        let signature2 = kraken.generate_signature(endpoint, nonce + 1, &post_data);
        assert!(signature2.is_ok());
        assert_ne!(sig, signature2.unwrap());
    }

    #[tokio::test]
    async fn test_signature_generation_invalid_secret() {
        let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
        kraken.set_credentials(
            "test_key".to_string(),
            "invalid_base64!".to_string(),
            None
        );
        
        let endpoint = "/0/private/Balance";
        let nonce = 1234567890u64;
        let post_data = format!("nonce={}", nonce);
        
        let signature = kraken.generate_signature(endpoint, nonce, &post_data);
        assert!(signature.is_err());
    }

    #[tokio::test]
    async fn test_signature_generation_no_credentials() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        
        let endpoint = "/0/private/Balance";
        let nonce = 1234567890u64;
        let post_data = format!("nonce={}", nonce);
        
        let signature = kraken.generate_signature(endpoint, nonce, &post_data);
        assert!(signature.is_err());
        assert_eq!(signature.unwrap_err().to_string(), "No credentials set");
    }

    #[tokio::test]
    async fn test_rate_limiter_basic() {
        let limiter = RateLimiter::new(3, Duration::from_secs(1));
        
        // First three requests should be immediate
        let start = std::time::SystemTime::now();
        limiter.acquire().await.unwrap();
        limiter.acquire().await.unwrap();
        limiter.acquire().await.unwrap();
        
        let elapsed = start.elapsed().unwrap();
        assert!(elapsed < Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_rate_limiter_with_delay() {
        let limiter = RateLimiter::new(2, Duration::from_secs(1));
        
        // First two requests should be immediate
        limiter.acquire().await.unwrap();
        limiter.acquire().await.unwrap();
        
        // Third request should be delayed
        let start = std::time::SystemTime::now();
        limiter.acquire().await.unwrap();
        let elapsed = start.elapsed().unwrap();
        assert!(elapsed >= Duration::from_millis(900));
    }

    #[tokio::test]
    async fn test_rate_limiter_window_reset() {
        let limiter = RateLimiter::new(1, Duration::from_millis(100));
        
        // First request
        limiter.acquire().await.unwrap();
        
        // Wait for window to reset
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Second request should be immediate
        let start = std::time::SystemTime::now();
        limiter.acquire().await.unwrap();
        let elapsed = start.elapsed().unwrap();
        assert!(elapsed < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn test_kraken_message_parsing() {
        let json_msg = r#"{
            "channel": "book10",
            "type": "snapshot",
            "data": [
                {
                    "symbol": "BTC/USD",
                    "bids": [{"price": "50000.0", "qty": "1.0"}],
                    "asks": [{"price": "50100.0", "qty": "0.5"}],
                    "timestamp": "2024-01-01T00:00:00.000Z"
                }
            ],
            "sequence": 12345,
            "timestamp": "2024-01-01T00:00:00.000Z"
        }"#;
        
        let msg: Result<KrakenMessage, _> = serde_json::from_str(json_msg);
        assert!(msg.is_ok());
        
        let parsed = msg.unwrap();
        assert_eq!(parsed.channel, Some("book10".to_string()));
        assert_eq!(parsed.r#type, Some("snapshot".to_string()));
        assert_eq!(parsed.sequence, Some(12345));
        assert!(parsed.data.is_some());
    }

    #[tokio::test]
    async fn test_subscription_serialization() {
        let subscription = KrakenSubscription {
            method: "subscribe".to_string(),
            params: KrakenSubscriptionParams {
                channel: "book10".to_string(),
                symbol: Some(vec!["BTC/USD".to_string(), "ETH/USD".to_string()]),
                snapshot: Some(true),
                event_trigger: None,
                ratecounter: None,
            },
            req_id: Some(1000),
        };
        
        let serialized = serde_json::to_string(&subscription);
        assert!(serialized.is_ok());
        
        let json_str = serialized.unwrap();
        assert!(json_str.contains("subscribe"));
        assert!(json_str.contains("book10"));
        assert!(json_str.contains("BTC/USD"));
        assert!(json_str.contains("ETH/USD"));
    }

    #[tokio::test]
    async fn test_order_request_serialization() {
        let order = KrakenOrderRequest {
            ordertype: "limit".to_string(),
            r#type: "buy".to_string(),
            volume: "1.0".to_string(),
            pair: "BTCUSD".to_string(),
            price: Some("50000.0".to_string()),
            price2: None,
            leverage: Some("2:1".to_string()),
            oflags: Some("fciq".to_string()),
            starttm: None,
            expiretm: None,
            userref: Some("12345".to_string()),
            validate: Some(true),
            close: None,
            trading_agreement: Some("agree".to_string()),
        };
        
        let serialized = serde_urlencoded::to_string(&order);
        assert!(serialized.is_ok());
        
        let data = serialized.unwrap();
        assert!(data.contains("ordertype=limit"));
        assert!(data.contains("type=buy"));
        assert!(data.contains("volume=1.0"));
        assert!(data.contains("pair=BTCUSD"));
        assert!(data.contains("price=50000.0"));
        assert!(data.contains("leverage=2%3A1")); // URL encoded "2:1"
        assert!(data.contains("oflags=fciq"));
        assert!(data.contains("userref=12345"));
        assert!(data.contains("validate=true"));
        assert!(data.contains("trading_agreement=agree"));
    }

    #[tokio::test]
    async fn test_order_request_minimal() {
        let order = KrakenOrderRequest {
            ordertype: "market".to_string(),
            r#type: "sell".to_string(),
            volume: "0.5".to_string(),
            pair: "ETHUSD".to_string(),
            price: None,
            price2: None,
            leverage: None,
            oflags: None,
            starttm: None,
            expiretm: None,
            userref: None,
            validate: None,
            close: None,
            trading_agreement: Some("agree".to_string()),
        };
        
        let serialized = serde_urlencoded::to_string(&order);
        assert!(serialized.is_ok());
        
        let data = serialized.unwrap();
        assert!(data.contains("ordertype=market"));
        assert!(data.contains("type=sell"));
        assert!(data.contains("volume=0.5"));
        assert!(data.contains("pair=ETHUSD"));
        assert!(!data.contains("price="));
        assert!(!data.contains("leverage="));
    }

    #[tokio::test]
    async fn test_data_structures() {
        // Test order book level
        let level = KrakenOrderBookLevel {
            price: "50000.0".to_string(),
            qty: "1.5".to_string(),
            timestamp: Some("2024-01-01T00:00:00.000Z".to_string()),
        };
        
        let serialized = serde_json::to_string(&level);
        assert!(serialized.is_ok());
        
        // Test trade
        let trade = KrakenTrade {
            symbol: "BTC/USD".to_string(),
            side: "buy".to_string(),
            price: "50000.0".to_string(),
            qty: "0.1".to_string(),
            ord_type: "limit".to_string(),
            trade_id: 123456789,
            timestamp: "2024-01-01T00:00:00.000Z".to_string(),
        };
        
        let serialized = serde_json::to_string(&trade);
        assert!(serialized.is_ok());
        
        // Test ticker
        let ticker = KrakenTicker {
            symbol: "BTC/USD".to_string(),
            bid: "49950.0".to_string(),
            bid_qty: "2.0".to_string(),
            ask: "50050.0".to_string(),
            ask_qty: "1.5".to_string(),
            last: "50000.0".to_string(),
            volume: "1500.0".to_string(),
            vwap: "49975.0".to_string(),
            low: "49500.0".to_string(),
            high: "50500.0".to_string(),
            change: "500.0".to_string(),
            change_pct: "1.01".to_string(),
        };
        
        let serialized = serde_json::to_string(&ticker);
        assert!(serialized.is_ok());
    }

    #[tokio::test]
    async fn test_order_book_operations() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        
        // Initially no order book
        assert!(kraken.get_order_book("BTC/USD").is_none());
        
        // Simulate order book update
        let order_book = KrakenOrderBook {
            symbol: "BTC/USD".to_string(),
            bids: vec![
                KrakenOrderBookLevel {
                    price: "50000.0".to_string(),
                    qty: "1.0".to_string(),
                    timestamp: Some("2024-01-01T00:00:00.000Z".to_string()),
                }
            ],
            asks: vec![
                KrakenOrderBookLevel {
                    price: "50100.0".to_string(),
                    qty: "0.5".to_string(),
                    timestamp: Some("2024-01-01T00:00:00.000Z".to_string()),
                }
            ],
            checksum: Some(123456),
            timestamp: "2024-01-01T00:00:00.000Z".to_string(),
        };
        
        kraken.order_books.write().insert("BTC/USD".to_string(), order_book.clone());
        
        let retrieved = kraken.get_order_book("BTC/USD");
        assert!(retrieved.is_some());
        let ob = retrieved.unwrap();
        assert_eq!(ob.symbol, "BTC/USD");
        assert_eq!(ob.bids.len(), 1);
        assert_eq!(ob.asks.len(), 1);
        assert_eq!(ob.checksum, Some(123456));
    }

    #[tokio::test]
    async fn test_trade_operations() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        
        // Initially no trades
        assert_eq!(kraken.get_recent_trades("BTC/USD", 10).len(), 0);
        
        // Add some trades
        let trades = vec![
            KrakenTrade {
                symbol: "BTC/USD".to_string(),
                side: "buy".to_string(),
                price: "50000.0".to_string(),
                qty: "0.1".to_string(),
                ord_type: "market".to_string(),
                trade_id: 1,
                timestamp: "2024-01-01T00:00:00.000Z".to_string(),
            },
            KrakenTrade {
                symbol: "ETH/USD".to_string(),
                side: "sell".to_string(),
                price: "3000.0".to_string(),
                qty: "1.0".to_string(),
                ord_type: "limit".to_string(),
                trade_id: 2,
                timestamp: "2024-01-01T00:00:01.000Z".to_string(),
            },
            KrakenTrade {
                symbol: "BTC/USD".to_string(),
                side: "sell".to_string(),
                price: "49950.0".to_string(),
                qty: "0.2".to_string(),
                ord_type: "limit".to_string(),
                trade_id: 3,
                timestamp: "2024-01-01T00:00:02.000Z".to_string(),
            },
        ];
        
        *kraken.trades.write() = trades;
        
        let btc_trades = kraken.get_recent_trades("BTC/USD", 10);
        assert_eq!(btc_trades.len(), 2);
        assert_eq!(btc_trades[0].symbol, "BTC/USD");
        assert_eq!(btc_trades[1].symbol, "BTC/USD");
        
        let eth_trades = kraken.get_recent_trades("ETH/USD", 10);
        assert_eq!(eth_trades.len(), 1);
        assert_eq!(eth_trades[0].symbol, "ETH/USD");
    }

    #[tokio::test]
    async fn test_ticker_operations() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        
        // Initially no ticker
        assert!(kraken.get_ticker("BTC/USD").is_none());
        
        // Add ticker
        let ticker = KrakenTicker {
            symbol: "BTC/USD".to_string(),
            bid: "49950.0".to_string(),
            bid_qty: "2.0".to_string(),
            ask: "50050.0".to_string(),
            ask_qty: "1.5".to_string(),
            last: "50000.0".to_string(),
            volume: "1500.0".to_string(),
            vwap: "49975.0".to_string(),
            low: "49500.0".to_string(),
            high: "50500.0".to_string(),
            change: "500.0".to_string(),
            change_pct: "1.01".to_string(),
        };
        
        kraken.tickers.write().insert("BTC/USD".to_string(), ticker.clone());
        
        let retrieved = kraken.get_ticker("BTC/USD");
        assert!(retrieved.is_some());
        let t = retrieved.unwrap();
        assert_eq!(t.symbol, "BTC/USD");
        assert_eq!(t.bid, "49950.0");
        assert_eq!(t.ask, "50050.0");
        assert_eq!(t.last, "50000.0");
    }

    #[tokio::test]
    async fn test_ohlc_operations() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        
        // Initially no OHLC data
        assert_eq!(kraken.get_ohlc_data("BTC/USD", 1, 10).len(), 0);
        
        // Add OHLC data
        let ohlc_data = vec![
            KrakenOHLC {
                symbol: "BTC/USD".to_string(),
                open: "49000.0".to_string(),
                high: "50500.0".to_string(),
                low: "48500.0".to_string(),
                close: "50000.0".to_string(),
                volume: "150.0".to_string(),
                vwap: "49750.0".to_string(),
                trades: 500,
                interval: 1,
                timestamp: "2024-01-01T00:00:00.000Z".to_string(),
            },
            KrakenOHLC {
                symbol: "BTC/USD".to_string(),
                open: "50000.0".to_string(),
                high: "50800.0".to_string(),
                low: "49800.0".to_string(),
                close: "50200.0".to_string(),
                volume: "120.0".to_string(),
                vwap: "50300.0".to_string(),
                trades: 400,
                interval: 1,
                timestamp: "2024-01-01T00:01:00.000Z".to_string(),
            },
        ];
        
        kraken.ohlc_data.write().insert("BTC/USD-1".to_string(), ohlc_data);
        
        let retrieved = kraken.get_ohlc_data("BTC/USD", 1, 5);
        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved[0].close, "50200.0"); // Most recent first
        assert_eq!(retrieved[1].close, "50000.0");
    }

    #[tokio::test]
    async fn test_balance_operations() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        
        // Initially no balances
        assert!(kraken.get_current_balances().is_empty());
        
        // Add balances
        let mut balances = std::collections::HashMap::new();
        balances.insert("BTC".to_string(), KrakenBalance {
            asset: "BTC".to_string(),
            balance: "2.5".to_string(),
            hold: Some("0.1".to_string()),
        });
        balances.insert("USD".to_string(), KrakenBalance {
            asset: "USD".to_string(),
            balance: "50000.0".to_string(),
            hold: Some("1000.0".to_string()),
        });
        
        *kraken.balances.write() = balances;
        
        let retrieved = kraken.get_current_balances();
        assert_eq!(retrieved.len(), 2);
        assert!(retrieved.contains_key("BTC"));
        assert!(retrieved.contains_key("USD"));
        assert_eq!(retrieved["BTC"].balance, "2.5");
        assert_eq!(retrieved["USD"].balance, "50000.0");
    }

    #[tokio::test]
    async fn test_connection_status() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        
        // Initially not connected
        assert!(!kraken.is_connected());
        assert!(kraken.get_connection_id().is_none());
        
        // Simulate connection ID
        *kraken.connection_id.write() = Some("test-connection-123".to_string());
        
        let conn_id = kraken.get_connection_id();
        assert!(conn_id.is_some());
        assert_eq!(conn_id.unwrap(), "test-connection-123");
    }

    #[tokio::test]
    #[ignore] // Integration test - requires network access
    async fn test_public_endpoints() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        
        // Test system status
        let status = kraken.get_system_status().await;
        if status.is_ok() {
            let result = status.unwrap();
            println!("System status: {:?}", result);
        }
        
        // Test server time
        let time = kraken.get_server_time().await;
        if time.is_ok() {
            let result = time.unwrap();
            println!("Server time: {:?}", result);
        }
        
        // Test asset info
        let assets = kraken.get_asset_info(None, None).await;
        if assets.is_ok() {
            let result = assets.unwrap();
            println!("Assets: {:?}", result);
        }
    }

    #[tokio::test]
    #[ignore] // Integration test - requires API credentials
    async fn test_authenticated_endpoints() {
        let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
        kraken.set_credentials(
            std::env::var("KRAKEN_API_KEY").unwrap_or_default(),
            std::env::var("KRAKEN_API_SECRET").unwrap_or_default(),
            None
        );
        
        if kraken.credentials.as_ref().unwrap().api_key.is_empty() {
            println!("Skipping authenticated tests - no credentials");
            return;
        }
        
        // Test balance
        let balance = kraken.get_balance().await;
        if balance.is_ok() {
            println!("Balance: {:?}", balance.unwrap());
        }
        
        // Test trade balance
        let trade_balance = kraken.get_trade_balance(None).await;
        if trade_balance.is_ok() {
            println!("Trade balance: {:?}", trade_balance.unwrap());
        }
    }

    #[tokio::test]
    #[ignore] // Integration test - requires WebSocket connection
    async fn test_websocket_connection() {
        let mut kraken = KrakenUltra::new(KrakenMarketType::Spot);
        
        let connection_result = kraken.connect_websocket(false).await;
        match connection_result {
            Ok(_) => {
                println!("Kraken WebSocket connected successfully");
                
                // Test subscription
                let sub_result = kraken.subscribe_ticker(vec!["BTC/USD".to_string()]).await;
                match sub_result {
                    Ok(_) => println!("Subscribed to ticker successfully"),
                    Err(e) => println!("Subscription failed: {}", e),
                }
                
                // Start message loop
                let loop_result = kraken.start_message_loop().await;
                if loop_result.is_ok() {
                    println!("Message loop started");
                    
                    // Wait a bit to receive some messages
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
                
                // Disconnect
                let disconnect_result = kraken.disconnect().await;
                if disconnect_result.is_ok() {
                    println!("Disconnected successfully");
                }
            }
            Err(e) => println!("Kraken WebSocket connection failed: {}", e),
        }
    }

    #[tokio::test]
    async fn test_error_handling() {
        let kraken = KrakenUltra::new(KrakenMarketType::Spot);
        
        // Test request without credentials
        let result = kraken.get_balance().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No credentials set"));
        
        // Test invalid signature
        let mut kraken_invalid = KrakenUltra::new(KrakenMarketType::Spot);
        kraken_invalid.set_credentials(
            "invalid".to_string(),
            "invalid_base64!".to_string(),
            None
        );
        
        let result = kraken_invalid.get_balance().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        let kraken = Arc::new(KrakenUltra::new(KrakenMarketType::Spot));
        
        let mut handles = vec![];
        
        // Spawn multiple tasks that operate on the same instance
        for i in 0..10 {
            let kraken_clone = Arc::clone(&kraken);
            let handle = tokio::spawn(async move {
                let nonce = kraken_clone.generate_nonce();
                tokio::time::sleep(Duration::from_millis(i * 10)).await;
                nonce
            });
            handles.push(handle);
        }
        
        let mut nonces = vec![];
        for handle in handles {
            nonces.push(handle.await.unwrap());
        }
        
        // All nonces should be unique
        nonces.sort();
        nonces.dedup();
        assert_eq!(nonces.len(), 10);
    }

    use std::sync::Arc;
}