// Bybit Ultra Tests - Comprehensive test suite
use crate::exchange::bybit_ultra::*;
use tokio;
use std::time::Duration;

#[tokio::test]
async fn test_bybit_ultra_initialization() {
    // Test different market types
    let spot_exchange = BybitUltra::new(BybitMarketType::Spot);
    let linear_exchange = BybitUltra::new(BybitMarketType::Linear);
    let inverse_exchange = BybitUltra::new(BybitMarketType::Inverse);
    let option_exchange = BybitUltra::new(BybitMarketType::Option);

    assert_eq!(spot_exchange.market_type, BybitMarketType::Spot);
    assert_eq!(linear_exchange.market_type, BybitMarketType::Linear);
    assert_eq!(inverse_exchange.market_type, BybitMarketType::Inverse);
    assert_eq!(option_exchange.market_type, BybitMarketType::Option);
    
    // All should start disconnected
    assert!(!spot_exchange.is_connected());
    assert!(!linear_exchange.is_connected());
    assert!(!inverse_exchange.is_connected());
    assert!(!option_exchange.is_connected());
}

#[tokio::test]
async fn test_credentials_management() {
    let mut exchange = BybitUltra::new(BybitMarketType::Linear);
    
    // Initially no credentials
    assert!(exchange.credentials.is_none());
    
    // Set credentials
    exchange.set_credentials(
        "test_api_key".to_string(),
        "test_api_secret".to_string(),
        true // testnet
    );
    
    // Verify credentials are set
    assert!(exchange.credentials.is_some());
    let creds = exchange.credentials.as_ref().unwrap();
    assert_eq!(creds.api_key, "test_api_key");
    assert_eq!(creds.api_secret, "test_api_secret");
    assert!(creds.testnet);
}

#[tokio::test]
async fn test_websocket_url_selection() {
    let spot_exchange = BybitUltra::new(BybitMarketType::Spot);
    let linear_exchange = BybitUltra::new(BybitMarketType::Linear);
    let inverse_exchange = BybitUltra::new(BybitMarketType::Inverse);
    let option_exchange = BybitUltra::new(BybitMarketType::Option);

    // Test public URLs
    assert_eq!(spot_exchange.get_ws_url(false), "wss://stream.bybit.com/v5/public/spot");
    assert_eq!(linear_exchange.get_ws_url(false), "wss://stream.bybit.com/v5/public/linear");
    assert_eq!(inverse_exchange.get_ws_url(false), "wss://stream.bybit.com/v5/public/linear");
    assert_eq!(option_exchange.get_ws_url(false), "wss://stream.bybit.com/v5/public/option");

    // Test private URLs (should be same for all market types)
    assert_eq!(spot_exchange.get_ws_url(true), "wss://stream.bybit.com/v5/private");
    assert_eq!(linear_exchange.get_ws_url(true), "wss://stream.bybit.com/v5/private");
    assert_eq!(inverse_exchange.get_ws_url(true), "wss://stream.bybit.com/v5/private");
    assert_eq!(option_exchange.get_ws_url(true), "wss://stream.bybit.com/v5/private");
}

#[tokio::test]
async fn test_hmac_signature_generation() {
    let mut exchange = BybitUltra::new(BybitMarketType::Linear);
    exchange.set_credentials(
        "test_api_key".to_string(),
        "test_secret_key".to_string(),
        false
    );

    let timestamp = 1234567890u64;
    let params = r#"{"symbol":"BTCUSDT","side":"Buy","orderType":"Market","qty":"0.001"}"#;
    
    let signature_result = exchange.generate_signature(timestamp, params);
    assert!(signature_result.is_ok());
    
    let signature = signature_result.unwrap();
    // HMAC-SHA256 signature should be 64 characters (32 bytes in hex)
    assert_eq!(signature.len(), 64);
    
    // Same input should produce same signature (deterministic)
    let signature2 = exchange.generate_signature(timestamp, params).unwrap();
    assert_eq!(signature, signature2);
    
    // Different timestamp should produce different signature
    let signature3 = exchange.generate_signature(timestamp + 1, params).unwrap();
    assert_ne!(signature, signature3);
}

#[tokio::test]
async fn test_message_deserialization() {
    // Test connection confirmation message
    let connection_msg = r#"{
        "success": true,
        "ret_msg": "connected",
        "conn_id": "clntavdrmcnqvhve1cru-2zjh8",
        "op": "subscribe"
    }"#;
    
    let parsed: Result<BybitMessage, _> = serde_json::from_str(connection_msg);
    assert!(parsed.is_ok());
    
    let msg = parsed.unwrap();
    assert_eq!(msg.success, Some(true));
    assert_eq!(msg.ret_msg, Some("connected".to_string()));
    assert_eq!(msg.conn_id, Some("clntavdrmcnqvhve1cru-2zjh8".to_string()));
    assert_eq!(msg.op, Some("subscribe".to_string()));

    // Test subscription response message
    let subscription_msg = r#"{
        "success": true,
        "ret_msg": "subscribe",
        "op": "subscribe",
        "req_id": "sub_1234567890"
    }"#;
    
    let parsed2: Result<BybitMessage, _> = serde_json::from_str(subscription_msg);
    assert!(parsed2.is_ok());
    
    let msg2 = parsed2.unwrap();
    assert_eq!(msg2.success, Some(true));
    assert_eq!(msg2.ret_msg, Some("subscribe".to_string()));
    assert_eq!(msg2.req_id, Some("sub_1234567890".to_string()));
}

#[tokio::test]
async fn test_orderbook_data_deserialization() {
    let orderbook_msg = r#"{
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 1672304486868,
        "data": [{
            "s": "BTCUSDT",
            "b": [
                ["16493.50", "0.006"],
                ["16493.00", "0.100"]
            ],
            "a": [
                ["16611.00", "0.029"],
                ["16612.00", "0.213"]
            ],
            "u": 18521288,
            "seq": 7961638396,
            "cts": 1672304486867
        }]
    }"#;
    
    let parsed: Result<BybitMessage, _> = serde_json::from_str(orderbook_msg);
    assert!(parsed.is_ok());
    
    let msg = parsed.unwrap();
    assert_eq!(msg.topic, Some("orderbook.50.BTCUSDT".to_string()));
    assert_eq!(msg.message_type, Some("snapshot".to_string()));
    assert_eq!(msg.ts, Some(1672304486868));
    assert!(msg.data.is_some());
    
    // Test parsing the order book data
    if let Some(data) = msg.data {
        let orderbook_data: Result<Vec<BybitOrderBook>, _> = serde_json::from_value(data);
        assert!(orderbook_data.is_ok());
        
        let orderbooks = orderbook_data.unwrap();
        assert_eq!(orderbooks.len(), 1);
        
        let orderbook = &orderbooks[0];
        assert_eq!(orderbook.s, "BTCUSDT");
        assert_eq!(orderbook.u, 18521288);
        assert_eq!(orderbook.seq, 7961638396);
        assert_eq!(orderbook.cts, 1672304486867);
        assert_eq!(orderbook.b.len(), 2);
        assert_eq!(orderbook.a.len(), 2);
        
        // Check bid data
        assert_eq!(orderbook.b[0][0], "16493.50");
        assert_eq!(orderbook.b[0][1], "0.006");
        
        // Check ask data
        assert_eq!(orderbook.a[0][0], "16611.00");
        assert_eq!(orderbook.a[0][1], "0.029");
    }
}

#[tokio::test]
async fn test_trade_data_deserialization() {
    let trade_msg = r#"{
        "topic": "publicTrade.BTCUSDT",
        "type": "snapshot",
        "ts": 1672304486865,
        "data": [{
            "T": 1672304486865,
            "s": "BTCUSDT",
            "S": "Buy",
            "v": "0.001",
            "p": "16618.00",
            "L": "PlusTick",
            "i": "20f43950-d8dd-5b31-9112-a178eb6023af",
            "BT": false
        }]
    }"#;
    
    let parsed: Result<BybitMessage, _> = serde_json::from_str(trade_msg);
    assert!(parsed.is_ok());
    
    let msg = parsed.unwrap();
    if let Some(data) = msg.data {
        let trade_data: Result<Vec<BybitTrade>, _> = serde_json::from_value(data);
        assert!(trade_data.is_ok());
        
        let trades = trade_data.unwrap();
        assert_eq!(trades.len(), 1);
        
        let trade = &trades[0];
        assert_eq!(trade.T, 1672304486865);
        assert_eq!(trade.s, "BTCUSDT");
        assert_eq!(trade.S, "Buy");
        assert_eq!(trade.v, "0.001");
        assert_eq!(trade.p, "16618.00");
        assert_eq!(trade.L, "PlusTick");
        assert!(!trade.BT);
    }
}

#[tokio::test]
async fn test_position_data_deserialization() {
    let position_msg = r#"{
        "topic": "position",
        "id": "592324803b2785-26fa-4214-9963-bdd4727f07be",
        "creationTime": 1672364262474,
        "data": [{
            "positionIdx": 0,
            "tradeMode": 0,
            "symbol": "BTCUSDT",
            "side": "Buy",
            "size": "0.001",
            "positionValue": "16.618",
            "avgPrice": "16618.00000000",
            "unrealisedPnl": "-0.00000005",
            "markPrice": "16618.00",
            "liqPrice": "",
            "bustPrice": "",
            "positionMM": "0.01661800",
            "positionIM": "0.33236000",
            "tpslMode": "Full",
            "takeProfit": "0.00",
            "stopLoss": "0.00",
            "trailingStop": "0.00",
            "sessionAvgPrice": "",
            "createdTime": "1672121182216",
            "updatedTime": "1672364262473",
            "seq": 5723621632
        }]
    }"#;
    
    let parsed: Result<BybitMessage, _> = serde_json::from_str(position_msg);
    assert!(parsed.is_ok());
    
    let msg = parsed.unwrap();
    if let Some(data) = msg.data {
        let position_data: Result<Vec<BybitPosition>, _> = serde_json::from_value(data);
        assert!(position_data.is_ok());
        
        let positions = position_data.unwrap();
        assert_eq!(positions.len(), 1);
        
        let position = &positions[0];
        assert_eq!(position.position_idx, 0);
        assert_eq!(position.trade_mode, 0);
        assert_eq!(position.symbol, "BTCUSDT");
        assert_eq!(position.side, "Buy");
        assert_eq!(position.size, "0.001");
        assert_eq!(position.position_value, "16.618");
        assert_eq!(position.avg_price, "16618.00000000");
        assert_eq!(position.unrealised_pnl, "-0.00000005");
        assert_eq!(position.seq, 5723621632);
    }
}

#[tokio::test]
async fn test_order_data_deserialization() {
    let order_msg = r#"{
        "topic": "order",
        "id": "592324803b2785-26fa-4214-9963-bdd4727f07be",
        "creationTime": 1672364262474,
        "data": [{
            "orderId": "fd4300ae-7847-404e-b947-b46980a4d140",
            "orderLinkId": "test-000005",
            "blockTradeId": "",
            "symbol": "ETHUSDT",
            "price": "1200.00",
            "qty": "1.00",
            "side": "Buy",
            "isLeverage": "",
            "positionIdx": 1,
            "orderStatus": "New",
            "cancelType": "UNKNOWN",
            "rejectReason": "EC_NoError",
            "avgPrice": "0",
            "leavesQty": "1.00",
            "leavesValue": "1200",
            "cumExecQty": "0.00",
            "cumExecValue": "0",
            "cumExecFee": "0",
            "timeInForce": "GTC",
            "orderType": "Limit",
            "stopOrderType": "UNKNOWN",
            "orderIv": "",
            "marketUnit": "",
            "triggerPrice": "0.00",
            "takeProfit": "0.00",
            "stopLoss": "0.00",
            "tpslMode": "",
            "tpLimitPrice": "",
            "slLimitPrice": "",
            "tpTriggerBy": "",
            "slTriggerBy": "",
            "triggerDirection": 0,
            "triggerBy": "",
            "lastPriceOnCreated": "",
            "reduceOnly": false,
            "closeOnTrigger": false,
            "placeType": "",
            "smpType": "None",
            "smpGroup": 0,
            "smpOrderId": "",
            "createdTime": "1672364262444",
            "updatedTime": "1672364262457"
        }]
    }"#;
    
    let parsed: Result<BybitMessage, _> = serde_json::from_str(order_msg);
    assert!(parsed.is_ok());
    
    let msg = parsed.unwrap();
    if let Some(data) = msg.data {
        let order_data: Result<Vec<BybitOrder>, _> = serde_json::from_value(data);
        assert!(order_data.is_ok());
        
        let orders = order_data.unwrap();
        assert_eq!(orders.len(), 1);
        
        let order = &orders[0];
        assert_eq!(order.order_id, "fd4300ae-7847-404e-b947-b46980a4d140");
        assert_eq!(order.order_link_id, "test-000005");
        assert_eq!(order.symbol, "ETHUSDT");
        assert_eq!(order.price, "1200.00");
        assert_eq!(order.qty, "1.00");
        assert_eq!(order.side, "Buy");
        assert_eq!(order.position_idx, 1);
        assert_eq!(order.order_status, "New");
        assert_eq!(order.order_type, "Limit");
        assert_eq!(order.time_in_force, "GTC");
        assert!(!order.reduce_only);
        assert!(!order.close_on_trigger);
    }
}

#[tokio::test]
async fn test_subscription_serialization() {
    // Test subscription message creation
    let subscription_arg = SubscriptionArg {
        op: "subscribe".to_string(),
        args: vec!["orderbook.50.BTCUSDT".to_string()],
        req_id: Some("sub_123456".to_string()),
    };
    
    let json = serde_json::to_string(&subscription_arg).unwrap();
    assert!(json.contains("subscribe"));
    assert!(json.contains("orderbook.50.BTCUSDT"));
    assert!(json.contains("sub_123456"));
    
    // Test deserialization back
    let parsed: SubscriptionArg = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.op, "subscribe");
    assert_eq!(parsed.args.len(), 1);
    assert_eq!(parsed.args[0], "orderbook.50.BTCUSDT");
    assert_eq!(parsed.req_id, Some("sub_123456".to_string()));
}

#[tokio::test]
async fn test_data_structures_memory_layout() {
    // Test that critical structures have expected memory layout
    use std::mem::size_of;
    
    // These structures should be reasonably sized for performance
    assert!(size_of::<BybitOrderBook>() < 2000); // Should be under 2KB
    assert!(size_of::<BybitTrade>() < 500);      // Should be under 500 bytes
    assert!(size_of::<BybitPosition>() < 2000);  // Should be under 2KB
    assert!(size_of::<BybitOrder>() < 2000);     // Should be under 2KB
    
    // Message structure should be compact
    assert!(size_of::<BybitMessage>() < 200);    // Should be under 200 bytes
}

#[tokio::test]
async fn test_market_type_categorization() {
    // Test correct category strings for different market types
    let spot = BybitUltra::new(BybitMarketType::Spot);
    let linear = BybitUltra::new(BybitMarketType::Linear);
    let inverse = BybitUltra::new(BybitMarketType::Inverse);
    let option = BybitUltra::new(BybitMarketType::Option);
    
    // These would be used in API calls
    let spot_category = match spot.market_type {
        BybitMarketType::Spot => "spot",
        BybitMarketType::Linear => "linear",
        BybitMarketType::Inverse => "inverse",
        BybitMarketType::Option => "option",
    };
    assert_eq!(spot_category, "spot");
    
    let linear_category = match linear.market_type {
        BybitMarketType::Spot => "spot",
        BybitMarketType::Linear => "linear",
        BybitMarketType::Inverse => "inverse",
        BybitMarketType::Option => "option",
    };
    assert_eq!(linear_category, "linear");
    
    let inverse_category = match inverse.market_type {
        BybitMarketType::Spot => "spot",
        BybitMarketType::Linear => "linear",
        BybitMarketType::Inverse => "inverse",
        BybitMarketType::Option => "option",
    };
    assert_eq!(inverse_category, "inverse");
    
    let option_category = match option.market_type {
        BybitMarketType::Spot => "spot",
        BybitMarketType::Linear => "linear",
        BybitMarketType::Inverse => "inverse",
        BybitMarketType::Option => "option",
    };
    assert_eq!(option_category, "option");
}

#[tokio::test]
async fn test_concurrent_access_patterns() {
    use std::sync::Arc;
    use tokio::sync::RwLock;
    
    let exchange = Arc::new(RwLock::new(BybitUltra::new(BybitMarketType::Linear)));
    
    // Test concurrent read access to order books
    let exchange1 = Arc::clone(&exchange);
    let exchange2 = Arc::clone(&exchange);
    
    let handle1 = tokio::spawn(async move {
        let ex = exchange1.read().await;
        ex.get_orderbook("BTCUSDT")
    });
    
    let handle2 = tokio::spawn(async move {
        let ex = exchange2.read().await;
        ex.get_current_positions()
    });
    
    let (result1, result2) = tokio::join!(handle1, handle2);
    
    // Both operations should complete successfully
    assert!(result1.is_ok());
    assert!(result2.is_ok());
    
    // Results should be None/empty since no data was populated
    assert!(result1.unwrap().is_none());
    assert!(result2.unwrap().is_empty());
}

/// Integration test that would require real API connection
/// This test is ignored by default but can be run manually
#[tokio::test]
#[ignore = "requires real API connection"]
async fn test_live_api_connection() {
    let mut exchange = BybitUltra::new(BybitMarketType::Linear);
    
    // Note: This would require real API credentials
    // exchange.set_credentials("real_key".to_string(), "real_secret".to_string(), true);
    
    // Test WebSocket connection
    let connection_result = exchange.connect_websocket(false).await;
    match connection_result {
        Ok(_) => {
            println!("Connected to Bybit WebSocket successfully");
            
            // Test subscription
            let sub_result = exchange.subscribe_orderbook("BTCUSDT").await;
            assert!(sub_result.is_ok());
            
            // Let it run for a few seconds to receive data
            tokio::time::sleep(Duration::from_secs(3)).await;
            
            // Check if we received any order book data
            let orderbook = exchange.get_orderbook("BTCUSDT");
            if let Some(ob) = orderbook {
                println!("Received order book data: {} bids, {} asks", ob.b.len(), ob.a.len());
            }
            
            // Clean disconnect
            let disconnect_result = exchange.disconnect().await;
            assert!(disconnect_result.is_ok());
        }
        Err(e) => {
            eprintln!("Connection failed (expected if no internet): {}", e);
        }
    }
}

/// Performance test for message processing
#[tokio::test]
async fn test_message_processing_performance() {
    use std::time::Instant;
    
    let orderbook_msg = r#"{
        "topic": "orderbook.50.BTCUSDT",
        "type": "snapshot",
        "ts": 1672304486868,
        "data": [{
            "s": "BTCUSDT",
            "b": [["16493.50", "0.006"], ["16493.00", "0.100"]],
            "a": [["16611.00", "0.029"], ["16612.00", "0.213"]],
            "u": 18521288,
            "seq": 7961638396,
            "cts": 1672304486867
        }]
    }"#;
    
    let start = Instant::now();
    let iterations = 1000;
    
    for _ in 0..iterations {
        let parsed: Result<BybitMessage, _> = serde_json::from_str(orderbook_msg);
        assert!(parsed.is_ok());
        
        let msg = parsed.unwrap();
        if let Some(data) = msg.data {
            let orderbook_data: Result<Vec<BybitOrderBook>, _> = serde_json::from_value(data);
            assert!(orderbook_data.is_ok());
        }
    }
    
    let duration = start.elapsed();
    println!("Processed {} messages in {:?}", iterations, duration);
    println!("Average time per message: {:?}", duration / iterations);
    
    // Should be able to process at least 1000 messages per second
    assert!(duration.as_millis() < 1000);
}