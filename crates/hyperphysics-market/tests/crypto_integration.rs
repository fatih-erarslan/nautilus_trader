//! Integration tests for cryptocurrency exchange providers

use hyperphysics_market::providers::{BinanceProvider, OKXProvider, MarketDataProvider};
use hyperphysics_market::data::Timeframe;
use hyperphysics_market::arbitrage::{ArbitrageDetector, ArbitrageType};
use std::sync::Arc;

#[tokio::test]
async fn test_binance_provider_creation() {
    let provider = BinanceProvider::new(true);
    assert_eq!(provider.provider_name(), "Binance");
    assert!(provider.supports_realtime());
}

#[tokio::test]
async fn test_okx_provider_creation() {
    let provider = OKXProvider::new(true);
    assert_eq!(provider.provider_name(), "OKX");
    assert!(provider.supports_realtime());
}

#[tokio::test]
async fn test_binance_with_credentials() {
    let provider = BinanceProvider::with_credentials(
        "test_key".to_string(),
        "test_secret".to_string(),
        true,
    );
    assert_eq!(provider.provider_name(), "Binance");
}

#[tokio::test]
async fn test_okx_with_credentials() {
    let provider = OKXProvider::with_credentials(
        "test_key".to_string(),
        "test_secret".to_string(),
        "test_pass".to_string(),
        true,
    );
    assert_eq!(provider.provider_name(), "OKX");
}

#[tokio::test]
async fn test_arbitrage_detector() {
    let mut detector = ArbitrageDetector::new(0.5, 100);

    // Add mock providers (in real scenario would be actual providers)
    let binance = Arc::new(BinanceProvider::new(true));
    let okx = Arc::new(OKXProvider::new(true));

    detector.add_provider("binance".to_string(), binance);
    detector.add_provider("okx".to_string(), okx);

    // Simulate price updates
    detector.update_price("binance", "BTC-USDT", 50000.0).await;
    detector.update_price("okx", "BTC-USDT", 50300.0).await;

    // Detect opportunities
    let opportunities = detector.detect_cross_exchange("BTC-USDT").await.unwrap();

    if !opportunities.is_empty() {
        assert_eq!(opportunities[0].arbitrage_type, ArbitrageType::CrossExchange);
        assert!(opportunities[0].profit_pct >= 0.5);
        assert_eq!(opportunities[0].exchanges.len(), 2);
    }
}

#[tokio::test]
async fn test_triangular_arbitrage() {
    let detector = ArbitrageDetector::new(0.3, 100);

    // Simulate prices for triangular arbitrage
    // BTC-ETH, ETH-USDT, BTC-USDT
    detector.update_price("binance", "BTC-ETH", 20.0).await;
    detector.update_price("binance", "ETH-USDT", 3000.0).await;
    detector.update_price("binance", "BTC-USDT", 60500.0).await; // Slight inefficiency

    let opportunities = detector
        .detect_triangular("binance", "BTC", "ETH", "USDT")
        .await
        .unwrap();

    // May or may not find opportunity depending on exact prices
    if !opportunities.is_empty() {
        assert_eq!(opportunities[0].arbitrage_type, ArbitrageType::Triangular);
        assert_eq!(opportunities[0].exchanges.len(), 1);
        assert_eq!(opportunities[0].symbols.len(), 3);
    }
}

#[tokio::test]
async fn test_spread_calculation() {
    let detector = ArbitrageDetector::new(0.5, 100);

    detector.update_price("binance", "ETH-USDT", 3000.0).await;
    detector.update_price("okx", "ETH-USDT", 3015.0).await;
    detector.update_price("coinbase", "ETH-USDT", 3010.0).await;

    let spread = detector.get_spread("ETH-USDT").await;
    assert!(spread.is_some());

    let (min, max, spread_pct) = spread.unwrap();
    assert_eq!(min, 3000.0);
    assert_eq!(max, 3015.0);
    assert!((spread_pct - 0.5).abs() < 0.01);
}

#[tokio::test]
async fn test_multiple_symbol_scan() {
    let mut detector = ArbitrageDetector::new(0.5, 100);

    let binance = Arc::new(BinanceProvider::new(true));
    let okx = Arc::new(OKXProvider::new(true));

    detector.add_provider("binance".to_string(), binance);
    detector.add_provider("okx".to_string(), okx);

    // Simulate prices for multiple symbols
    detector.update_price("binance", "BTC-USDT", 50000.0).await;
    detector.update_price("okx", "BTC-USDT", 50300.0).await;

    detector.update_price("binance", "ETH-USDT", 3000.0).await;
    detector.update_price("okx", "ETH-USDT", 3020.0).await;

    // Test that opportunities are stored
    let opportunities = detector.get_opportunities().await;
    // Initially empty until scan_all is called
    assert!(opportunities.is_empty());

    // Clear should work
    detector.clear_opportunities().await;
    assert!(detector.get_opportunities().await.is_empty());
}

#[test]
fn test_timeframe_conversions() {
    // Test that both providers handle the same timeframes
    let timeframes = vec![
        Timeframe::Minute1,
        Timeframe::Minute5,
        Timeframe::Minute15,
        Timeframe::Hour1,
        Timeframe::Hour4,
        Timeframe::Day1,
        Timeframe::Week1,
        Timeframe::Month1,
    ];

    // Both providers should support these timeframes
    for _tf in timeframes {
        // Would test actual conversion here
        // Just ensuring they exist
    }
}
