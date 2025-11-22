//! Electric Eel Shocker demonstration
//! Shows the basic functionality of the Electric Eel organism

use chrono::Utc;
use parasitic::organisms::ElectricEelShocker;
use parasitic::traits::{MarketData, Organism};

fn main() -> parasitic::Result<()> {
    println!("🔌 Electric Eel Shocker Demo");
    println!("============================");

    // Create an Electric Eel Shocker
    let shocker = ElectricEelShocker::new()?;

    println!("✓ Electric Eel Shocker created");
    println!("  Name: {}", shocker.name());
    println!("  Type: {}", shocker.organism_type());
    println!("  Active: {}", shocker.is_active());
    println!(
        "  Initial charge: {:.2}%",
        shocker.get_charge_level() * 100.0
    );

    // Create test market data
    let market_data = MarketData {
        symbol: "BTC_USD".to_string(),
        timestamp: Utc::now(),
        price: 50000.0,
        volume: 1000.0,
        volatility: 0.15, // 15% volatility
        bid: 49950.0,
        ask: 50050.0,
        spread_percent: 0.002,
        market_cap: Some(1_000_000_000.0),
        liquidity_score: 0.7, // 70% liquidity
    };

    println!("\n📊 Market Data:");
    println!("  Symbol: {}", market_data.symbol);
    println!("  Price: ${:.2}", market_data.price);
    println!("  Volatility: {:.1}%", market_data.volatility * 100.0);
    println!("  Liquidity: {:.1}%", market_data.liquidity_score * 100.0);

    // Generate bioelectric shock
    println!("\n⚡ Generating bioelectric shock...");
    let shock_intensity = 0.7; // 70% intensity

    match shocker.generate_bioelectric_shock(&market_data, shock_intensity) {
        Ok(result) => {
            println!("✅ Shock generated successfully!");
            println!(
                "  Shock intensity: {:.1}%",
                result.disruption_result.shock_intensity * 100.0
            );
            println!(
                "  Voltage generated: {:.0}V",
                result.disruption_result.voltage_generated
            );
            println!(
                "  Disruption radius: {:.4}",
                result.disruption_result.disruption_radius
            );
            println!(
                "  Price levels affected: {}",
                result.disruption_result.affected_price_levels.len()
            );
            println!(
                "  Charge remaining: {:.1}%",
                result.bioelectric_charge_remaining * 100.0
            );
            println!("  Information gain: {:.3}", result.information_gain);

            // Check for hidden liquidity pools discovered
            if !result.hidden_liquidity_pools.is_empty() {
                println!("\n🔍 Hidden Liquidity Pools Discovered:");
                for (i, pool) in result.hidden_liquidity_pools.iter().enumerate() {
                    println!(
                        "  Pool {}: ${:.2} ({:.0} volume, {:.1}% confidence)",
                        i + 1,
                        pool.price_level,
                        pool.estimated_volume,
                        pool.confidence_score * 100.0
                    );
                }
            } else {
                println!("\n🔍 No hidden liquidity pools discovered");
            }

            // Show next optimal timing
            println!("\n⏰ Next Optimal Shock Window:");
            println!("  Duration: {}ms", result.next_optimal_window.duration_ms);
            println!(
                "  Expected effectiveness: {:.1}%",
                result.next_optimal_window.expected_effectiveness * 100.0
            );
            println!(
                "  Market condition: {:?}",
                result.next_optimal_window.market_condition
            );
        }
        Err(e) => {
            println!("❌ Shock generation failed: {}", e);
            return Err(e);
        }
    }

    // Show final metrics
    println!("\n📈 Organism Metrics:");
    match shocker.get_metrics() {
        Ok(metrics) => {
            println!("  Total operations: {}", metrics.total_operations);
            println!("  Success rate: {:.1}%", metrics.accuracy_rate * 100.0);
            println!(
                "  Avg processing time: {}ns",
                metrics.average_processing_time_ns
            );
            println!("  Memory usage: {}KB", metrics.memory_usage_bytes / 1024);

            if let Some(charge) = metrics.custom_metrics.get("bioelectric_charge_level") {
                println!("  Bioelectric charge: {:.1}%", charge * 100.0);
            }
        }
        Err(e) => {
            println!("  Failed to get metrics: {}", e);
        }
    }

    println!("\n🎯 Demo completed successfully!");
    Ok(())
}
