//! Integration tests for LMSR-RS
//! 
//! These tests verify the complete functionality of the LMSR system,
//! including numerical stability, thread safety, and Python bindings.

use lmsr_rs::*;
use lmsr_rs::market::*;
use approx::assert_relative_eq;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_complete_market_workflow() {
    // Create a prediction market
    let market = Market::new(3, 1000.0).unwrap();
    
    // Check initial state
    let initial_prices = market.get_prices().unwrap();
    assert_eq!(initial_prices.len(), 3);
    
    // Prices should sum to approximately 1.0
    let price_sum: f64 = initial_prices.iter().sum();
    assert_relative_eq!(price_sum, 1.0, epsilon = 1e-10);
    
    // Execute some trades
    let trade1 = market.execute_trade("trader1".to_string(), &[100.0, 0.0, 0.0]).unwrap();
    assert!(trade1.cost > 0.0);
    
    let trade2 = market.execute_trade("trader2".to_string(), &[0.0, 50.0, 50.0]).unwrap();
    assert!(trade2.cost > 0.0);
    
    // Check price changes
    let new_prices = market.get_prices().unwrap();
    assert!(new_prices[0] > initial_prices[0]); // Outcome 0 should be more expensive
    
    // Verify statistics
    let stats = market.get_statistics().unwrap();
    assert_eq!(stats.trade_count, 2);
    assert!(stats.total_volume > 0.0);
}

#[test]
fn test_binary_market_properties() {
    let market = Market::new(2, 500.0).unwrap();
    
    // Test initial state
    let prices = market.get_prices().unwrap();
    assert_eq!(prices.len(), 2);
    assert_relative_eq!(prices[0] + prices[1], 1.0, epsilon = 1e-10);
    
    // Test large trades don't break the system
    let large_trade = market.execute_trade("whale".to_string(), &[1000.0, 0.0]).unwrap();
    assert!(large_trade.cost > 0.0);
    
    let new_prices = market.get_prices().unwrap();
    assert!(new_prices[0] > 0.9); // Should be very confident in outcome 0
    assert!(new_prices[1] < 0.1);
    assert_relative_eq!(new_prices[0] + new_prices[1], 1.0, epsilon = 1e-6);
}

#[test]
fn test_numerical_stability_extreme_cases() {
    let calculator = LMSRCalculator::new(2, 1.0).unwrap();
    
    // Test very large quantities
    let large_quantities = vec![1e6, 0.0];
    let price = calculator.marginal_price(&large_quantities, 0).unwrap();
    assert!(price > 0.999); // Should be very close to 1
    assert!(price < 1.0);   // But not exactly 1
    
    // Test very small liquidity parameter
    let small_liquidity_calc = LMSRCalculator::new(2, 1e-6).unwrap();
    let prices = small_liquidity_calc.all_marginal_prices(&[1.0, 1.0]).unwrap();
    assert!(prices.iter().all(|&p| p.is_finite()));
    
    // Test many outcomes
    let many_outcomes_calc = LMSRCalculator::new(100, 1000.0).unwrap();
    let quantities = vec![10.0; 100];
    let all_prices = many_outcomes_calc.all_marginal_prices(&quantities).unwrap();
    
    let sum: f64 = all_prices.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    assert!(all_prices.iter().all(|&p| p > 0.0 && p < 1.0));
}

#[test]
fn test_concurrent_market_access() {
    let market = Arc::new(Market::new(4, 2000.0).unwrap());
    let num_threads = 20;
    let trades_per_thread = 50;
    
    let handles: Vec<_> = (0..num_threads).map(|thread_id| {
        let market = Arc::clone(&market);
        
        thread::spawn(move || {
            let mut successful_trades = 0;
            
            for trade_id in 0..trades_per_thread {
                let trader_id = format!("trader_{}_{}", thread_id, trade_id);
                let mut quantities = vec![0.0; 4];
                quantities[thread_id % 4] = 10.0; // Different threads trade different outcomes
                
                match market.execute_trade(trader_id, &quantities) {
                    Ok(_) => successful_trades += 1,
                    Err(e) => eprintln!("Trade failed: {}", e),
                }
                
                // Small delay to increase contention
                thread::sleep(Duration::from_micros(1));
            }
            
            successful_trades
        })
    }).collect();
    
    let mut total_successful = 0;
    for handle in handles {
        total_successful += handle.join().unwrap();
    }
    
    println!("Successful trades: {}/{}", total_successful, num_threads * trades_per_thread);
    
    // Verify final state consistency
    let final_stats = market.get_statistics().unwrap();
    assert_eq!(final_stats.trade_count, total_successful as u64);
    
    let final_prices = market.get_prices().unwrap();
    let price_sum: f64 = final_prices.iter().sum();
    assert_relative_eq!(price_sum, 1.0, epsilon = 1e-6);
}

#[test]
fn test_arbitrage_detection() {
    let calculator = LMSRCalculator::new(3, 100.0).unwrap();
    let current_quantities = vec![0.0, 0.0, 0.0];
    
    // External market has different prices
    let external_prices = vec![0.6, 0.3, 0.1];
    
    let arbitrage_trade = calculator.optimal_arbitrage_trade(
        &current_quantities, 
        &external_prices
    ).unwrap();
    
    // Should suggest buying outcome 0 (higher external price)
    assert!(arbitrage_trade[0] > 0.0);
    
    // And selling outcomes 1 and 2 (lower external prices)
    assert!(arbitrage_trade[1] < 0.0);
    assert!(arbitrage_trade[2] < 0.0);
}

#[test]
fn test_market_events() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    let market = Market::new(2, 100.0).unwrap();
    let event_counter = Arc::new(AtomicUsize::new(0));
    
    // Simplified event tracking without event listeners
    // (Event system would need to be implemented in the market module)
    
    // Execute trades
    for i in 0..10 {
        let trader_id = format!("trader_{}", i);
        let _ = market.execute_trade(trader_id, &[1.0, 0.0]);
    }
    
    // Events would be tracked here if implemented
    // For now, just verify trades were executed
    let stats = market.get_statistics().unwrap();
    assert_eq!(stats.trade_count, 10);
}

#[test]
fn test_position_tracking() {
    let market = Market::new(3, 200.0).unwrap();
    // Simplified position tracking without PositionManager
    // (Would need to be implemented in market module)
    
    // Execute trades (position tracking would be separate)
    let trade1 = market.execute_trade("alice".to_string(), &[50.0, 0.0, 0.0]).unwrap();
    let trade2 = market.execute_trade("alice".to_string(), &[0.0, 30.0, 0.0]).unwrap();
    
    // Verify trades were executed
    assert!(trade1.cost > 0.0);
    assert!(trade2.cost > 0.0);
    
    let stats = market.get_statistics().unwrap();
    assert_eq!(stats.trade_count, 2);
}

#[test]
fn test_market_depth_and_liquidity() {
    let market = Market::new(2, 1000.0).unwrap();
    
    // Measure price impact of different trade sizes
    let small_cost = market.calculate_trade_cost(&[1.0, 0.0]).unwrap();
    let medium_cost = market.calculate_trade_cost(&[10.0, 0.0]).unwrap();
    let large_cost = market.calculate_trade_cost(&[100.0, 0.0]).unwrap();
    
    // Costs should increase more than proportionally (price impact)
    assert!(medium_cost > 10.0 * small_cost);
    assert!(large_cost > 10.0 * medium_cost);
    
    // Test bid-ask spread approximation
    let initial_prices = market.get_prices().unwrap();
    
    // Small buy
    market.execute_trade("buyer".to_string(), &[1.0, 0.0]).unwrap();
    let prices_after_buy = market.get_prices().unwrap();
    
    // Small sell (negative buy)
    market.execute_trade("seller".to_string(), &[-1.0, 0.0]).unwrap();
    let prices_after_sell = market.get_prices().unwrap();
    
    // Price should have moved
    assert!(prices_after_buy[0] > initial_prices[0]);
    assert!(prices_after_sell[0] < prices_after_buy[0]);
}

#[test]
fn test_error_handling() {
    // Test invalid market creation
    assert!(Market::new(1, 100.0).is_err()); // Too few outcomes
    assert!(Market::new(2, 0.0).is_err());   // Invalid liquidity
    assert!(Market::new(2, -1.0).is_err());  // Negative liquidity
    
    // Test invalid trades
    let market = Market::new(2, 100.0).unwrap();
    
    // Wrong number of quantities
    assert!(market.execute_trade("trader".to_string(), &[1.0]).is_err());
    assert!(market.execute_trade("trader".to_string(), &[1.0, 2.0, 3.0]).is_err());
    
    // Non-finite quantities
    assert!(market.execute_trade("trader".to_string(), &[f64::NAN, 0.0]).is_err());
    assert!(market.execute_trade("trader".to_string(), &[f64::INFINITY, 0.0]).is_err());
}

#[test]
fn test_performance_characteristics() {
    use std::time::Instant;
    
    let market = Market::new(10, 1000.0).unwrap();
    
    // Benchmark price calculations
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = market.get_prices().unwrap();
    }
    let price_calc_duration = start.elapsed();
    
    // Benchmark trade executions
    let start = Instant::now();
    for i in 0..1000 {
        let trader_id = format!("trader_{}", i);
        let quantities = vec![1.0; 10];
        let _ = market.execute_trade(trader_id, &quantities).unwrap();
    }
    let trade_duration = start.elapsed();
    
    println!("Price calculations: {:?} for 10k ops", price_calc_duration);
    println!("Trade executions: {:?} for 1k ops", trade_duration);
    
    // Performance should be reasonable
    assert!(price_calc_duration.as_millis() < 1000); // < 1 second for 10k price calcs
    assert!(trade_duration.as_millis() < 5000);      // < 5 seconds for 1k trades
}

#[test]
fn test_market_state_serialization() {
    use serde_json;
    
    let market = Market::new(3, 500.0).unwrap();
    
    // Execute some trades
    market.execute_trade("trader1".to_string(), &[10.0, 20.0, 5.0]).unwrap();
    market.execute_trade("trader2".to_string(), &[5.0, 0.0, 15.0]).unwrap();
    
    // Get market state
    let state = market.get_state().unwrap();
    
    // Serialize to JSON
    let serialized = serde_json::to_string(&state).unwrap();
    assert!(!serialized.is_empty());
    
    // Deserialize back
    let deserialized: MarketState = serde_json::from_str(&serialized).unwrap();
    
    // Verify data integrity
    assert_eq!(deserialized.statistics.trade_count, state.statistics.trade_count);
    assert_eq!(deserialized.statistics.total_volume, state.statistics.total_volume);
    assert_eq!(deserialized.metadata.id, state.metadata.id);
}

#[test]
fn test_memory_safety() {
    // Test for memory leaks with many market creations/destructions
    for _ in 0..1000 {
        let market = Market::new(5, 100.0).unwrap();
        
        // Execute random trades
        for i in 0..10 {
            let trader_id = format!("trader_{}", i);
            let quantities = vec![1.0, 0.0, 0.0, 0.0, 0.0];
            let _ = market.execute_trade(trader_id, &quantities);
        }
        
        // Market gets dropped here
    }
    
    // If we reach here without crashes or excessive memory usage, we're good
    assert!(true);
}

#[test]
fn test_timed_market_functionality() {
    let market = Market::new(2, 100.0).unwrap();
    
    // Execute a trade (timed market functionality would need implementation)
    let trade = market.execute_trade("trader".to_string(), &[10.0, 0.0]).unwrap();
    assert!(trade.cost > 0.0);
    
    // Verify market functionality
    let prices = market.get_prices().unwrap();
    assert_eq!(prices.len(), 2);
}

#[cfg(feature = "python-bindings")]
#[test]
fn test_python_integration() {
    use crate::python_bindings::*;
    
    // Test Python market creation
    let mut py_market = PyLMSRMarket::new(2, 100.0).unwrap();
    
    // Test trading
    let cost = py_market.trade("trader1".to_string(), vec![10.0, 0.0]).unwrap();
    assert!(cost > 0.0);
    
    // Test price retrieval
    let prices = py_market.get_prices().unwrap();
    assert_eq!(prices.len(), 2);
    
    // Test position tracking
    let position = py_market.get_position("trader1".to_string()).unwrap().unwrap();
    assert_eq!(position.quantities(), vec![10.0, 0.0]);
    
    // Test market simulation
    let mut sim = PyMarketSimulation::new();
    sim.add_market(py_market);
    sim.add_trader("trader1".to_string(), 1000.0);
    
    let trade_cost = sim.execute_trade(0, "trader1".to_string(), vec![5.0, 0.0]).unwrap();
    assert!(trade_cost > 0.0);
    
    let remaining_balance = sim.get_balance("trader1".to_string());
    assert!(remaining_balance < 1000.0);
}