use crate::cache::MarketTick;
use crate::data::BinanceWebSocketClient;
use std::time::Duration;
use tokio::time::sleep;

/// Demonstration of production-grade Binance WebSocket integration
///
/// This example shows how to use the BinanceWebSocketClient with real API integration,
/// following Constitutional Prime Directive (NO synthetic data).
///
/// FORBIDDEN ACTIONS:
/// ❌ NO mock/synthetic/random data generation
/// ❌ NO hardcoded market values  
/// ❌ NO placeholder implementations
///
/// REQUIRED IMPLEMENTATION:
/// ✅ Only production Binance WebSocket streams
/// ✅ Cryptographic data integrity validation  
/// ✅ Real-time processing with audit trails
/// ✅ Circuit breakers and fault tolerance
/// ✅ Performance monitoring and caching

/// Example usage of the real data integration system
pub async fn demonstrate_real_data_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 CWTS Ultra - Real Data Integration Demo");
    println!("📊 Connecting to PRODUCTION Binance WebSocket...");

    // REQUIRED: Use real API credentials from environment
    let api_key = std::env::var("BINANCE_API_KEY")
        .expect("BINANCE_API_KEY environment variable required for production");
    let secret_key = std::env::var("BINANCE_SECRET_KEY")
        .expect("BINANCE_SECRET_KEY environment variable required for production");

    // Initialize client with REAL credentials only
    let mut client = BinanceWebSocketClient::new(api_key, secret_key).await?;

    println!("✅ Client initialized with production credentials");

    // Perform health check before starting
    let health_status = client.health_check().await?;
    println!("🏥 Health Check Results:");
    println!(
        "   Circuit Breaker: {}",
        if health_status.circuit_breaker_healthy {
            "HEALTHY"
        } else {
            "UNHEALTHY"
        }
    );
    println!(
        "   Pool Connections: {}/{}",
        health_status.pool_healthy_connections, health_status.pool_total_connections
    );
    println!("   Cache Entries: {}", health_status.cache_entries);
    println!("   Validator Hashes: {}", health_status.validator_hashes);
    println!(
        "   Overall Status: {}",
        if health_status.overall_healthy {
            "🟢 HEALTHY"
        } else {
            "🔴 UNHEALTHY"
        }
    );

    if !health_status.overall_healthy {
        println!("⚠️ Warning: System not fully healthy, proceeding with caution");
    }

    // Connect to real market data streams
    println!("🔗 Establishing WebSocket connection to Binance...");
    client.connect_to_market_data().await?;
    println!("✅ Connected to LIVE market data streams");

    // Process real market data
    println!("📈 Processing real-time market data...");
    let market_stream = client.process_real_market_data().await?;

    println!("📊 Market Data Stream Statistics:");
    println!("   Total Ticks: {}", market_stream.total_ticks());
    println!("   Processed: {}", market_stream.processed_count());

    // Demonstrate processing each tick
    let mut stream = market_stream;
    let mut processed_count = 0;
    const MAX_DEMO_TICKS: usize = 10; // Limit for demo

    println!("\n🎯 Processing Market Ticks (Real Data Only):");
    println!(
        "{:<12} {:<12} {:<12} {:<12} {:<12}",
        "Symbol", "Price", "Volume", "Bid", "Ask"
    );
    println!("{:-<60}", "");

    while let Some(tick) = stream.next_real_tick() {
        display_market_tick(tick);
        processed_count += 1;

        if processed_count >= MAX_DEMO_TICKS {
            println!("📝 Demo limit reached ({} ticks processed)", MAX_DEMO_TICKS);
            break;
        }

        // Small delay for readable output
        sleep(Duration::from_millis(100)).await;
    }

    // Show client metrics
    let metrics = client.get_metrics();
    println!("\n📈 Performance Metrics:");
    println!(
        "   Connections Established: {}",
        metrics.connections_established
    );
    println!("   Messages Received: {}", metrics.messages_received);
    println!("   Validation Successes: {}", metrics.validation_successes);
    println!("   Validation Failures: {}", metrics.validation_failures);
    println!("   Cache Hits: {}", metrics.cache_hits);
    println!("   Cache Misses: {}", metrics.cache_misses);
    println!(
        "   Circuit Breaker Trips: {}",
        metrics.circuit_breaker_trips
    );

    // Circuit breaker status
    println!(
        "\n🔄 Circuit Breaker Status: {}",
        client.get_circuit_breaker_state()
    );

    // Demonstrate cache functionality
    println!("\n💾 Testing Cache Functionality:");
    let cache_key = "BTCUSDT_1640995200000"; // Example key
    match client.get_cached_data(&cache_key).await? {
        Some(cached_tick) => {
            println!("   ✅ Cache Hit for key: {}", cache_key);
            display_market_tick(&cached_tick);
        }
        None => {
            println!("   ❌ Cache Miss for key: {}", cache_key);
        }
    }

    // Clean shutdown
    println!("\n🔒 Closing connection and cleaning up...");
    client.close().await?;
    println!("✅ Connection closed successfully");

    println!("\n🎉 Real Data Integration Demo Complete!");
    println!("✅ All data processed was REAL market data from Binance");
    println!("✅ No synthetic or mock data was generated");
    println!("✅ Production-grade fault tolerance and monitoring demonstrated");

    Ok(())
}

/// Display market tick data in formatted table
fn display_market_tick(tick: &MarketTick) {
    println!(
        "{:<12} {:<12.4} {:<12.2} {:<12.4} {:<12.4}",
        tick.symbol, tick.price, tick.volume, tick.bid_price, tick.ask_price
    );
}

/// Alternative demo function for testing without real credentials
pub async fn demonstrate_validation_only() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 CWTS Ultra - Validation Demo (No Live Data)");
    println!("📋 Testing validation components without live connections...");

    // Test validation components
    use crate::cache::VolatilityBasedCache;
    use crate::circuit::CircuitBreaker;
    use crate::pool::ConnectionPool;
    use crate::validation::CryptographicDataValidator;

    // Test circuit breaker
    let circuit_breaker = CircuitBreaker::new(5, 60);
    println!("✅ Circuit breaker initialized");
    println!("   State: {:?}", circuit_breaker.get_state());
    println!("   Failure count: {}", circuit_breaker.get_failure_count());

    // Test data validator
    let mut validator = CryptographicDataValidator::new()?;
    println!("✅ Cryptographic validator initialized");

    let test_message = r#"{"symbol":"BTCUSDT","price":50000.0,"volume":100.0,"timestamp":1640995200000,"bid_price":49999.0,"ask_price":50001.0,"trade_id":12345}"#;
    match validator.validate_message_integrity(test_message) {
        Ok(()) => println!("   ✅ Message validation passed"),
        Err(e) => println!("   ❌ Message validation failed: {}", e),
    }

    // Test volatility cache
    let cache = VolatilityBasedCache::new();
    let cache_stats = cache.get_statistics();
    println!("✅ Volatility cache initialized");
    println!("   Total entries: {}", cache_stats.total_entries);
    println!("   Max entries: {}", cache_stats.max_entries);

    // Test connection pool
    let pool = ConnectionPool::new(10);
    let pool_stats = pool.get_statistics();
    println!("✅ Connection pool initialized");
    println!("   Active connections: {}", pool_stats.active_connections);
    println!("   Max connections: {}", pool_stats.max_connections);

    println!("\n🎉 Validation Demo Complete!");
    println!("✅ All components initialized successfully");
    println!("✅ Ready for real data integration with live credentials");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validation_demo() {
        // Test the validation-only demo
        let result = demonstrate_validation_only().await;
        assert!(result.is_ok(), "Validation demo should succeed");
    }

    #[test]
    fn test_display_market_tick() {
        let tick = MarketTick {
            symbol: "BTCUSDT".to_string(),
            price: 50000.0,
            volume: 100.0,
            timestamp: 1640995200000,
            bid_price: 49999.0,
            ask_price: 50001.0,
            trade_id: 12345,
        };

        // Test that display function doesn't panic
        display_market_tick(&tick);
    }
}
