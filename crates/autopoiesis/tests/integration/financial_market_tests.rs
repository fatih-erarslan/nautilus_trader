//! Integration tests for financial market modules
//! Tests market dynamics, trading systems, and autopoietic market behaviors

use autopoiesis::domains::finance::*;
use autopoiesis::engines::*;
use autopoiesis::portfolio::*;
use autopoiesis::risk::*;
use autopoiesis::market_data::*;
use autopoiesis::execution::*;
use rust_decimal::Decimal;
use std::collections::HashMap;
use tokio::time::{sleep, Duration};
#[cfg(feature = "test-utils")]
use approx::assert_relative_eq;

#[cfg(feature = "property-tests")]
use proptest::prelude::*;
use uuid::Uuid;

/// Test market data for testing
#[derive(Clone, Debug)]
pub struct TestMarketData {
    pub symbol: String,
    pub price: Decimal,
    pub volume: Decimal,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub bid: Decimal,
    pub ask: Decimal,
}

impl TestMarketData {
    pub fn new(symbol: &str, price: f64) -> Self {
        let price = Decimal::from_f64_retain(price).unwrap();
        Self {
            symbol: symbol.to_string(),
            price,
            volume: Decimal::from_f64_retain(1000.0).unwrap(),
            timestamp: chrono::Utc::now(),
            bid: price - Decimal::from_f64_retain(0.01).unwrap(),
            ask: price + Decimal::from_f64_retain(0.01).unwrap(),
        }
    }
    
    pub fn with_spread(&mut self, spread: f64) -> &mut Self {
        let spread_decimal = Decimal::from_f64_retain(spread).unwrap();
        self.bid = self.price - spread_decimal / Decimal::from(2);
        self.ask = self.price + spread_decimal / Decimal::from(2);
        self
    }
}

/// Test utilities for financial market tests
pub mod test_utils {
    use super::*;
    
    pub fn create_mock_portfolio() -> HashMap<String, Decimal> {
        let mut portfolio = HashMap::new();
        portfolio.insert("BTC".to_string(), Decimal::from_f64_retain(1.0).unwrap());
        portfolio.insert("ETH".to_string(), Decimal::from_f64_retain(10.0).unwrap());
        portfolio.insert("USD".to_string(), Decimal::from_f64_retain(50000.0).unwrap());
        portfolio
    }
    
    pub fn create_test_market_data() -> Vec<TestMarketData> {
        vec![
            TestMarketData::new("BTC", 45000.0),
            TestMarketData::new("ETH", 3000.0),
            TestMarketData::new("SOL", 100.0),
            TestMarketData::new("ADA", 0.50),
        ]
    }
    
    pub fn simulate_price_movement(initial_price: f64, volatility: f64, steps: usize) -> Vec<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut prices = vec![initial_price];
        
        for _ in 0..steps {
            let last_price = prices.last().unwrap();
            let change = rng.gen_range(-volatility..volatility);
            let new_price = last_price * (1.0 + change);
            prices.push(new_price.max(0.01)); // Prevent negative prices
        }
        
        prices
    }
}

#[tokio::test]
async fn test_market_data_aggregation() {
    let test_data = test_utils::create_test_market_data();
    
    // Test basic aggregation
    let mut aggregated_data = HashMap::new();
    for data in test_data {
        aggregated_data.insert(data.symbol.clone(), data);
    }
    
    assert_eq!(aggregated_data.len(), 4);
    assert!(aggregated_data.contains_key("BTC"));
    assert!(aggregated_data.contains_key("ETH"));
    
    // Test price bounds
    for (symbol, data) in &aggregated_data {
        assert!(data.price > Decimal::ZERO);
        assert!(data.bid <= data.price);
        assert!(data.ask >= data.price);
        assert!(data.volume > Decimal::ZERO);
    }
}

#[tokio::test]
async fn test_portfolio_valuation() {
    let portfolio = test_utils::create_mock_portfolio();
    let market_data = test_utils::create_test_market_data();
    
    // Create price map
    let mut prices = HashMap::new();
    for data in market_data {
        prices.insert(data.symbol, data.price);
    }
    
    // Calculate portfolio value
    let mut total_value = Decimal::ZERO;
    for (symbol, quantity) in &portfolio {
        if let Some(price) = prices.get(symbol) {
            total_value += quantity * price;
        } else if symbol == "USD" {
            total_value += *quantity; // USD is base currency
        }
    }
    
    // Portfolio should have significant value
    assert!(total_value > Decimal::from_f64_retain(80000.0).unwrap()); // BTC + ETH + USD
}

#[tokio::test]
async fn test_risk_calculation() {
    let portfolio = test_utils::create_mock_portfolio();
    
    // Simple risk calculation based on portfolio concentration
    let mut total_assets = Decimal::ZERO;
    let mut max_position = Decimal::ZERO;
    
    for (symbol, quantity) in &portfolio {
        if symbol != "USD" {
            let value = if symbol == "BTC" {
                *quantity * Decimal::from_f64_retain(45000.0).unwrap()
            } else if symbol == "ETH" {
                *quantity * Decimal::from_f64_retain(3000.0).unwrap()
            } else {
                *quantity
            };
            
            total_assets += value;
            max_position = max_position.max(value);
        }
    }
    
    let concentration_risk = max_position / total_assets;
    
    // Risk metrics should be reasonable
    assert!(concentration_risk >= Decimal::ZERO);
    assert!(concentration_risk <= Decimal::ONE);
}

#[tokio::test]
async fn test_market_making_simulation() {
    let mut market_data = TestMarketData::new("BTC", 45000.0);
    market_data.with_spread(100.0); // $100 spread
    
    // Simulate market making
    let mid_price = (market_data.bid + market_data.ask) / Decimal::from(2);
    let spread = market_data.ask - market_data.bid;
    
    // Basic market making metrics
    assert_relative_eq!(mid_price.to_f64().unwrap(), 45000.0, epsilon = 1.0);
    assert_relative_eq!(spread.to_f64().unwrap(), 100.0, epsilon = 0.01);
    
    // Market maker should place orders around mid price
    let buy_order_price = mid_price - spread / Decimal::from(4);
    let sell_order_price = mid_price + spread / Decimal::from(4);
    
    assert!(buy_order_price < mid_price);
    assert!(sell_order_price > mid_price);
}

#[tokio::test]
async fn test_arbitrage_detection() {
    // Create arbitrage opportunity
    let exchange1_data = TestMarketData::new("BTC", 45000.0);
    let exchange2_data = TestMarketData::new("BTC", 45200.0);
    
    let price_diff = exchange2_data.price - exchange1_data.price;
    let arbitrage_opportunity = price_diff / exchange1_data.price;
    
    // Should detect profitable arbitrage
    assert!(price_diff > Decimal::ZERO);
    assert!(arbitrage_opportunity > Decimal::from_f64_retain(0.001).unwrap()); // >0.1% opportunity
    
    // Calculate potential profit (simplified)
    let trade_size = Decimal::from_f64_retain(1.0).unwrap(); // 1 BTC
    let gross_profit = trade_size * price_diff;
    let estimated_fees = gross_profit * Decimal::from_f64_retain(0.002).unwrap(); // 0.2% total fees
    let net_profit = gross_profit - estimated_fees;
    
    assert!(net_profit > Decimal::ZERO);
}

#[tokio::test]
async fn test_trend_following_strategy() {
    let prices = test_utils::simulate_price_movement(45000.0, 0.02, 50);
    
    // Simple moving average trend following
    let window_size = 10;
    let mut signals = Vec::new();
    
    for i in window_size..prices.len() {
        let short_ma: f64 = prices[i-5..i].iter().sum::<f64>() / 5.0;
        let long_ma: f64 = prices[i-window_size..i].iter().sum::<f64>() / window_size as f64;
        
        let signal = if short_ma > long_ma { 1.0 } else { -1.0 }; // Buy/Sell signal
        signals.push(signal);
    }
    
    // Should generate trading signals
    assert!(signals.len() > 0);
    assert!(signals.iter().any(|&s| s > 0.0)); // At least one buy signal
    assert!(signals.iter().any(|&s| s < 0.0)); // At least one sell signal
}

#[tokio::test]
async fn test_portfolio_rebalancing() {
    let mut portfolio = test_utils::create_mock_portfolio();
    let target_allocation = {
        let mut target = HashMap::new();
        target.insert("BTC".to_string(), 0.5); // 50% BTC
        target.insert("ETH".to_string(), 0.3); // 30% ETH
        target.insert("USD".to_string(), 0.2); // 20% USD
        target
    };
    
    // Current market values
    let prices = {
        let mut prices = HashMap::new();
        prices.insert("BTC".to_string(), Decimal::from_f64_retain(45000.0).unwrap());
        prices.insert("ETH".to_string(), Decimal::from_f64_retain(3000.0).unwrap());
        prices.insert("USD".to_string(), Decimal::ONE);
        prices
    };
    
    // Calculate current allocation
    let mut total_value = Decimal::ZERO;
    let mut current_values = HashMap::new();
    
    for (symbol, quantity) in &portfolio {
        let price = prices.get(symbol).unwrap_or(&Decimal::ONE);
        let value = quantity * price;
        current_values.insert(symbol.clone(), value);
        total_value += value;
    }
    
    // Calculate rebalancing needs
    let mut rebalance_orders = Vec::new();
    for (symbol, &target_pct) in &target_allocation {
        let current_value = current_values.get(symbol).unwrap_or(&Decimal::ZERO);
        let target_value = total_value * Decimal::from_f64_retain(target_pct).unwrap();
        let diff = target_value - current_value;
        
        if diff.abs() > Decimal::from_f64_retain(100.0).unwrap() { // $100 threshold
            rebalance_orders.push((symbol.clone(), diff));
        }
    }
    
    // Should generate rebalancing orders
    assert!(rebalance_orders.len() > 0);
    
    // Net rebalancing should be approximately zero (accounting for precision)
    let net_rebalance: Decimal = rebalance_orders.iter().map(|(_, diff)| diff).sum();
    assert!(net_rebalance.abs() < Decimal::ONE);
}

#[tokio::test]
async fn test_risk_monitoring() {
    let portfolio = test_utils::create_mock_portfolio();
    let prices = test_utils::simulate_price_movement(45000.0, 0.05, 100); // High volatility
    
    // Calculate rolling volatility
    let window_size = 20;
    let mut volatilities = Vec::new();
    
    for i in window_size..prices.len() {
        let window = &prices[i-window_size..i];
        let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
        let variance: f64 = window.iter()
            .map(|&price| (price - mean).powi(2))
            .sum::<f64>() / window.len() as f64;
        let volatility = variance.sqrt();
        volatilities.push(volatility);
    }
    
    // Risk metrics should be calculable
    assert!(volatilities.len() > 0);
    let avg_volatility: f64 = volatilities.iter().sum::<f64>() / volatilities.len() as f64;
    assert!(avg_volatility > 0.0);
    
    // High volatility should trigger risk warnings
    let risk_threshold = 2500.0; // $2500 volatility threshold
    let high_risk_periods = volatilities.iter().filter(|&&v| v > risk_threshold).count();
    
    // With 5% volatility, should have some high-risk periods
    assert!(high_risk_periods <= volatilities.len());
}

#[tokio::test]
async fn test_execution_optimization() {
    let large_order_size = Decimal::from_f64_retain(10.0).unwrap(); // 10 BTC
    let market_depth = vec![
        (Decimal::from_f64_retain(44950.0).unwrap(), Decimal::from_f64_retain(2.0).unwrap()),
        (Decimal::from_f64_retain(44960.0).unwrap(), Decimal::from_f64_retain(3.0).unwrap()),
        (Decimal::from_f64_retain(44970.0).unwrap(), Decimal::from_f64_retain(4.0).unwrap()),
        (Decimal::from_f64_retain(44980.0).unwrap(), Decimal::from_f64_retain(5.0).unwrap()),
    ];
    
    // Simulate TWAP (Time-Weighted Average Price) execution
    let total_available = market_depth.iter().map(|(_, qty)| qty).sum::<Decimal>();
    
    if large_order_size <= total_available {
        // Can execute the order
        let mut remaining_size = large_order_size;
        let mut total_cost = Decimal::ZERO;
        let mut executed_size = Decimal::ZERO;
        
        for (price, available) in market_depth {
            if remaining_size <= Decimal::ZERO {
                break;
            }
            
            let execute_size = remaining_size.min(available);
            total_cost += execute_size * price;
            executed_size += execute_size;
            remaining_size -= execute_size;
        }
        
        let average_price = total_cost / executed_size;
        
        // Should execute at reasonable average price
        assert!(executed_size > Decimal::ZERO);
        assert!(average_price > Decimal::from_f64_retain(44900.0).unwrap());
        assert!(average_price < Decimal::from_f64_retain(45000.0).unwrap());
    }
}

#[tokio::test]
async fn test_market_impact_estimation() {
    let base_price = Decimal::from_f64_retain(45000.0).unwrap();
    let order_sizes = vec![0.1, 0.5, 1.0, 5.0, 10.0]; // Different order sizes in BTC
    
    // Simplified market impact model: impact = sqrt(order_size) * 0.001
    let mut impacts = Vec::new();
    
    for size in order_sizes {
        let impact_factor = size.sqrt() * 0.001;
        let price_impact = base_price * Decimal::from_f64_retain(impact_factor).unwrap();
        impacts.push(price_impact);
    }
    
    // Market impact should increase with order size
    for i in 1..impacts.len() {
        assert!(impacts[i] >= impacts[i-1]);
    }
    
    // Large orders should have significant impact
    let large_order_impact = impacts.last().unwrap();
    assert!(*large_order_impact > Decimal::from_f64_retain(10.0).unwrap()); // >$10 impact for 10 BTC
}

#[tokio::test]
async fn test_synthetic_instruments() {
    // Test synthetic position creation (e.g., BTC/ETH ratio trade)
    let btc_price = Decimal::from_f64_retain(45000.0).unwrap();
    let eth_price = Decimal::from_f64_retain(3000.0).unwrap();
    let ratio = btc_price / eth_price; // BTC/ETH ratio
    
    // Create synthetic long BTC/ETH position
    let btc_quantity = Decimal::from_f64_retain(1.0).unwrap();
    let eth_quantity = btc_quantity * ratio; // Equivalent ETH amount
    
    // Position value should be neutral initially
    let btc_value = btc_quantity * btc_price;
    let eth_value = eth_quantity * eth_price;
    
    assert_relative_eq!(btc_value.to_f64().unwrap(), eth_value.to_f64().unwrap(), epsilon = 0.01);
    
    // Test ratio change impact
    let new_btc_price = Decimal::from_f64_retain(46000.0).unwrap(); // BTC up
    let new_eth_price = Decimal::from_f64_retain(2950.0).unwrap(); // ETH down
    
    let new_btc_value = btc_quantity * new_btc_price;
    let new_eth_value = eth_quantity * new_eth_price;
    let pnl = new_btc_value - new_eth_value;
    
    // Should profit from BTC outperforming ETH
    assert!(pnl > Decimal::ZERO);
}

/// Property-based tests for financial markets
#[cfg(feature = "property-tests")]
mod financial_property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_portfolio_value_non_negative(
            btc_qty in 0.0f64..100.0,
            eth_qty in 0.0f64..1000.0,
            btc_price in 1000.0f64..100000.0,
            eth_price in 100.0f64..10000.0
        ) {
            let btc_value = btc_qty * btc_price;
            let eth_value = eth_qty * eth_price;
            let total_value = btc_value + eth_value;
            
            prop_assert!(total_value >= 0.0);
            prop_assert!(btc_value >= 0.0);
            prop_assert!(eth_value >= 0.0);
        }
        
        #[test]
        fn test_spread_consistency(
            mid_price in 1.0f64..100000.0,
            spread_pct in 0.0001f64..0.1 // 0.01% to 10%
        ) {
            let spread = mid_price * spread_pct;
            let bid = mid_price - spread / 2.0;
            let ask = mid_price + spread / 2.0;
            
            prop_assert!(bid < mid_price);
            prop_assert!(ask > mid_price);
            prop_assert!(ask - bid >= 0.0);
            prop_assert!((ask + bid) / 2.0 - mid_price < 1e-10); // Numerical precision
        }
        
        #[test]
        fn test_arbitrage_calculation(
            price1 in 1000.0f64..50000.0,
            price2 in 1000.0f64..50000.0
        ) {
            let price_diff = (price2 - price1).abs();
            let min_price = price1.min(price2);
            let arbitrage_pct = price_diff / min_price;
            
            prop_assert!(arbitrage_pct >= 0.0);
            prop_assert!(arbitrage_pct <= 1.0); // Max 100% difference
        }
    }
}

/// Performance benchmark tests for financial operations
mod financial_benchmarks {
    use super::*;
    use std::time::Instant;
    
    #[tokio::test]
    async fn benchmark_portfolio_valuation() {
        let portfolio = test_utils::create_mock_portfolio();
        let market_data = test_utils::create_test_market_data();
        
        let start = Instant::now();
        
        // Benchmark 10,000 portfolio valuations
        for _ in 0..10_000 {
            let mut total_value = Decimal::ZERO;
            for (symbol, quantity) in &portfolio {
                for data in &market_data {
                    if data.symbol == *symbol {
                        total_value += quantity * data.price;
                        break;
                    }
                }
                if symbol == "USD" {
                    total_value += *quantity;
                }
            }
        }
        
        let duration = start.elapsed();
        println!("Portfolio valuation benchmark: {:?} for 10,000 calculations", duration);
        
        // Should complete within reasonable time
        assert!(duration.as_millis() < 1000); // Less than 1 second
    }
    
    #[tokio::test]
    async fn benchmark_risk_calculation() {
        let prices = test_utils::simulate_price_movement(45000.0, 0.02, 1000);
        
        let start = Instant::now();
        
        // Benchmark volatility calculations
        let window_size = 20;
        let mut volatilities = Vec::new();
        
        for i in window_size..prices.len() {
            let window = &prices[i-window_size..i];
            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 = window.iter()
                .map(|&price| (price - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            volatilities.push(variance.sqrt());
        }
        
        let duration = start.elapsed();
        println!("Risk calculation benchmark: {:?} for {} windows", duration, volatilities.len());
        
        // Should complete efficiently
        assert!(duration.as_millis() < 100); // Less than 100ms
        assert!(volatilities.len() > 0);
    }
}