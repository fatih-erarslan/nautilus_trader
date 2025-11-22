//! LMSR Trading Simulation Example
//! 
//! This example demonstrates how to use the LMSR-RS crate for market making
//! and trading simulation in a realistic financial scenario.

use lmsr_rs::*;
use lmsr_rs::market::MarketFactory;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use rand::prelude::*;

fn main() -> Result<()> {
    println!("🚀 LMSR Trading Simulation Starting...\n");
    
    // Create a prediction market for election outcomes
    let market = MarketFactory::create_categorical_market(
        "2024 Presidential Election".to_string(),
        "Who will win the 2024 Presidential Election?".to_string(),
        vec![
            "Candidate A".to_string(),
            "Candidate B".to_string(), 
            "Candidate C".to_string(),
            "Other".to_string()
        ],
        10000.0 // $10,000 liquidity parameter
    )?;
    
    println!("📊 Market Created: {}", market.get_metadata().name);
    println!("🎯 Outcomes: {:?}", market.get_metadata().outcomes);
    println!("💰 Liquidity Parameter: $10,000\n");
    
    // Initial market state
    let initial_prices = market.get_prices()?;
    println!("📈 Initial Prices: {:?}", initial_prices);
    println!("💡 Initial probabilities sum: {:.6}\n", initial_prices.iter().sum::<f64>());
    
    // Set up position tracking
    let position_manager = crate::market::PositionManager::new();
    
    // Create traders with different strategies
    let mut traders = HashMap::new();
    traders.insert("Institutional Investor".to_string(), TraderStrategy::Conservative);
    traders.insert("Hedge Fund".to_string(), TraderStrategy::Aggressive);
    traders.insert("Retail Trader 1".to_string(), TraderStrategy::Random);
    traders.insert("Retail Trader 2".to_string(), TraderStrategy::Random);
    traders.insert("Arbitrageur".to_string(), TraderStrategy::Arbitrage);
    
    // Simulation parameters
    let num_rounds = 50;
    let mut rng = thread_rng();
    
    println!("🎮 Starting {} rounds of trading simulation...\n", num_rounds);
    
    let start_time = Instant::now();
    
    for round in 1..=num_rounds {
        println!("--- Round {} ---", round);
        
        // Each trader makes trades based on their strategy
        for (trader_name, strategy) in &traders {
            let trade_quantities = generate_trade(strategy, &market, &mut rng)?;
            
            if trade_quantities.iter().any(|&x| x != 0.0) {
                match market.execute_trade(trader_name.clone(), &trade_quantities) {
                    Ok(trade_record) => {
                        // Update position
                        position_manager.update_position(
                            trader_name.clone(),
                            &trade_quantities,
                            trade_record.cost
                        )?;
                        
                        println!("💸 {} traded {:?} for ${:.2}", 
                               trader_name, trade_quantities, trade_record.cost);
                    }
                    Err(e) => {
                        eprintln!("❌ Trade failed for {}: {}", trader_name, e);
                    }
                }
            }
        }
        
        // Show current market state
        let current_prices = market.get_prices()?;
        let stats = market.get_statistics()?;
        
        println!("📊 Current Prices: {:?}", current_prices);
        println!("📈 Total Volume: ${:.2}, Trades: {}", stats.total_volume, stats.trade_count);
        
        // Show most profitable outcome
        let max_price_idx = current_prices.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        println!("🏆 Leading: {} ({:.1}%)", 
               market.get_metadata().outcomes[max_price_idx],
               current_prices[max_price_idx] * 100.0);
        
        println!();
        
        // Simulate time passing
        std::thread::sleep(Duration::from_millis(10));
    }
    
    let simulation_duration = start_time.elapsed();
    
    println!("🎯 Simulation Complete!\n");
    
    // Final analysis
    let final_stats = market.get_statistics()?;
    let final_prices = market.get_prices()?;
    
    println!("📊 FINAL MARKET STATE");
    println!("====================");
    for (i, outcome) in market.get_metadata().outcomes.iter().enumerate() {
        println!("{}: ${:.4} ({:.1}%)", outcome, final_prices[i], final_prices[i] * 100.0);
    }
    
    println!("\n📈 TRADING STATISTICS");
    println!("====================");
    println!("Total Trades: {}", final_stats.trade_count);
    println!("Total Volume: ${:.2}", final_stats.total_volume);
    println!("Average Trade Size: ${:.2}", final_stats.total_volume / final_stats.trade_count as f64);
    println!("Simulation Time: {:?}", simulation_duration);
    println!("Trades per Second: {:.2}", final_stats.trade_count as f64 / simulation_duration.as_secs_f64());
    
    println!("\n💰 TRADER POSITIONS");
    println!("==================");
    for trader_name in traders.keys() {
        if let Some(position) = position_manager.get_position(trader_name) {
            let position_value = position_manager.calculate_position_value(trader_name, &final_prices)?;
            let pnl = position_value - position.total_invested;
            
            println!("{}", trader_name);
            println!("  Shares: {:?}", position.quantities);
            println!("  Invested: ${:.2}", position.total_invested);
            println!("  Current Value: ${:.2}", position_value);
            println!("  P&L: ${:.2} ({:.1}%)", pnl, (pnl / position.total_invested) * 100.0);
            println!();
        }
    }
    
    // Performance benchmarks
    println!("⚡ PERFORMANCE BENCHMARKS");
    println!("========================");
    
    let benchmark_start = Instant::now();
    for _ in 0..10000 {
        let _ = market.get_prices()?;
    }
    let price_calc_time = benchmark_start.elapsed();
    
    let benchmark_start = Instant::now();
    for i in 0..1000 {
        let trader_id = format!("benchmark_trader_{}", i);
        let quantities = vec![0.1, 0.0, 0.0, 0.0];
        let _ = market.execute_trade(trader_id, &quantities)?;
    }
    let trade_time = benchmark_start.elapsed();
    
    println!("Price Calculations: 10,000 ops in {:?} ({:.2} μs/op)", 
           price_calc_time, price_calc_time.as_micros() as f64 / 10000.0);
    println!("Trade Executions: 1,000 ops in {:?} ({:.2} μs/op)", 
           trade_time, trade_time.as_micros() as f64 / 1000.0);
    
    // Test numerical stability
    println!("\n🔬 NUMERICAL STABILITY TEST");
    println!("===========================");
    
    let test_market = Market::new(4, 1.0)?; // Small liquidity parameter
    
    // Extreme trade sizes
    let extreme_trade = test_market.execute_trade(
        "extreme_trader".to_string(), 
        &[1000000.0, 0.0, 0.0, 0.0]
    )?;
    
    let extreme_prices = test_market.get_prices()?;
    println!("Extreme trade cost: ${:.2}", extreme_trade.cost);
    println!("Resulting prices: {:?}", extreme_prices);
    println!("Prices sum: {:.10}", extreme_prices.iter().sum::<f64>());
    println!("All prices finite: {}", extreme_prices.iter().all(|p| p.is_finite()));
    
    println!("\n✅ Simulation completed successfully!");
    
    Ok(())
}

#[derive(Debug, Clone)]
enum TraderStrategy {
    Conservative,  // Small, safe trades
    Aggressive,    // Large, risky trades
    Random,        // Random trades
    Arbitrage,     // Exploit price inefficiencies
}

fn generate_trade(
    strategy: &TraderStrategy, 
    market: &Market, 
    rng: &mut ThreadRng
) -> Result<Vec<f64>> {
    let num_outcomes = market.get_metadata().outcomes.len();
    let current_prices = market.get_prices()?;
    
    match strategy {
        TraderStrategy::Conservative => {
            // Small trades on undervalued outcomes
            let mut quantities = vec![0.0; num_outcomes];
            if rng.gen_bool(0.3) { // 30% chance to trade
                let outcome = rng.gen_range(0..num_outcomes);
                quantities[outcome] = rng.gen_range(1.0..5.0);
            }
            Ok(quantities)
        }
        
        TraderStrategy::Aggressive => {
            // Large trades on high-conviction outcomes
            let mut quantities = vec![0.0; num_outcomes];
            if rng.gen_bool(0.2) { // 20% chance to trade, but large size
                let outcome = rng.gen_range(0..num_outcomes);
                quantities[outcome] = rng.gen_range(10.0..50.0);
            }
            Ok(quantities)
        }
        
        TraderStrategy::Random => {
            // Completely random trades
            let mut quantities = vec![0.0; num_outcomes];
            if rng.gen_bool(0.4) { // 40% chance to trade
                for i in 0..num_outcomes {
                    if rng.gen_bool(0.3) {
                        quantities[i] = rng.gen_range(-2.0..5.0); // Can sell too
                    }
                }
            }
            Ok(quantities)
        }
        
        TraderStrategy::Arbitrage => {
            // Look for arbitrage opportunities (simplified)
            let mut quantities = vec![0.0; num_outcomes];
            
            // Find most undervalued outcome (highest expected return)
            let expected_prices = vec![0.3, 0.4, 0.2, 0.1]; // Arbitrageur's beliefs
            
            let mut best_opportunity = None;
            let mut best_return = 0.0;
            
            for i in 0..num_outcomes {
                let expected_return = expected_prices[i] / current_prices[i];
                if expected_return > best_return && expected_return > 1.1 {
                    best_return = expected_return;
                    best_opportunity = Some(i);
                }
            }
            
            if let Some(outcome) = best_opportunity {
                quantities[outcome] = rng.gen_range(5.0..15.0);
            }
            
            Ok(quantities)
        }
    }
}