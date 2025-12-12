//! Example: Creating and managing an antifragile portfolio
//!
//! This example demonstrates how to use the Talebian Risk Management library
//! to create and manage a portfolio with antifragile characteristics.

use talebian_risk::prelude::*;
use chrono::Utc;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦾 Talebian Risk Management - Antifragile Portfolio Example");
    println!("============================================================");
    
    // 1. Create risk configuration
    let risk_config = RiskConfig {
        confidence_level: 0.95,
        lookback_days: 252,
        min_observations: 50,
        max_position_size: 0.25, // 25% maximum position
        black_swan_threshold: 0.01,
        antifragility_params: AntifragilityParams {
            volatility_threshold: 0.02,
            convexity_sensitivity: 1.5,
            hormesis_window: 21,
            min_stress_level: 0.05,
            max_stress_level: 0.5,
        },
        fat_tail_adjustment: true,
        time_weighted: true,
    };
    
    println!("📊 Risk Configuration:");
    println!("  • Confidence Level: {:.1}%", risk_config.confidence_level * 100.0);
    println!("  • Max Position Size: {:.1}%", risk_config.max_position_size * 100.0);
    println!("  • Black Swan Threshold: {:.2}%", risk_config.black_swan_threshold * 100.0);
    println!();
    
    // 2. Create antifragile portfolio
    let mut portfolio = AntifragilePortfolio::new("antifragile_portfolio", risk_config);
    
    // 3. Add assets with different risk characteristics
    println!("🏗️  Building Antifragile Portfolio:");
    
    // Safe assets (60% allocation)
    portfolio.add_asset("US_TREASURY_BONDS", 0.4, AssetType::Safe)?;
    portfolio.add_asset("CASH_EQUIVALENT", 0.2, AssetType::Safe)?;
    
    // Antifragile assets (25% allocation)
    portfolio.add_asset("GOLD", 0.15, AssetType::Antifragile)?;
    portfolio.add_asset("REAL_ESTATE", 0.1, AssetType::Antifragile)?;
    
    // Volatile assets with upside convexity (15% allocation)
    portfolio.add_asset("CRYPTOCURRENCY", 0.05, AssetType::Volatile)?;
    portfolio.add_asset("TECH_STOCKS", 0.05, AssetType::Volatile)?;
    portfolio.add_asset("CALL_OPTIONS", 0.05, AssetType::Derivative)?;
    
    println!("  ✓ Added 7 assets across different risk categories");
    println!("  ✓ Total allocation: 100%");
    println!();
    
    // 4. Simulate price updates
    println!("💹 Updating Asset Prices:");
    let price_updates = vec![
        ("US_TREASURY_BONDS", 102.5),
        ("CASH_EQUIVALENT", 100.0),
        ("GOLD", 1850.0),
        ("REAL_ESTATE", 250.0),
        ("CRYPTOCURRENCY", 45000.0),
        ("TECH_STOCKS", 150.0),
        ("CALL_OPTIONS", 25.0),
    ];
    
    for (asset, price) in price_updates {
        portfolio.update_price(asset, price)?;
        println!("  ✓ Updated {} price to ${:.2}", asset, price);
    }
    println!();
    
    // 5. Calculate portfolio metrics
    println!("📈 Portfolio Risk Metrics:");
    let risk_metrics = portfolio.calculate_risk_metrics()?;
    
    println!("  • Safe Allocation: {:.1}%", risk_metrics.safe_allocation * 100.0);
    println!("  • Risky Allocation: {:.1}%", risk_metrics.risky_allocation * 100.0);
    println!("  • Antifragile Allocation: {:.1}%", risk_metrics.antifragile_allocation * 100.0);
    println!("  • Effective Positions: {:.1}", risk_metrics.effective_positions);
    println!("  • Diversification Ratio: {:.3}", risk_metrics.diversification_ratio);
    println!("  • Tail Risk: {:.3}", risk_metrics.tail_risk);
    println!();
    
    // 6. Measure antifragility
    let antifragility_score = portfolio.measure_antifragility()?;
    println!("🦾 Antifragility Analysis:");
    println!("  • Portfolio Antifragility Score: {:.3}", antifragility_score);
    
    let antifragility_level = match antifragility_score {
        score if score > 0.7 => "Highly Antifragile",
        score if score > 0.4 => "Moderately Antifragile", 
        score if score > 0.1 => "Slightly Antifragile",
        _ => "Robust/Fragile",
    };
    println!("  • Antifragility Level: {}", antifragility_level);
    println!();
    
    // 7. Create antifragility measurer for detailed analysis
    let antifragility_params = AntifragilityParams {
        volatility_threshold: 0.02,
        convexity_sensitivity: 1.5,
        hormesis_window: 21,
        min_stress_level: 0.05,
        max_stress_level: 0.5,
    };
    
    let mut antifragility_measurer = AntifragilityMeasurer::new(
        "portfolio_measurer", 
        antifragility_params
    );
    
    // 8. Generate sample portfolio returns for analysis
    println!("📊 Generating Portfolio Return Analysis:");
    let portfolio_returns = generate_sample_portfolio_returns();
    
    let measurement = antifragility_measurer.measure_antifragility(&portfolio_returns)?;
    
    println!("  • Overall Score: {:.3}", measurement.overall_score);
    println!("  • Convexity: {:.3}", measurement.convexity);
    println!("  • Volatility Benefit: {:.3}", measurement.volatility_benefit);
    println!("  • Stress Response: {:.3}", measurement.stress_response);
    println!("  • Hormesis Effect: {:.3}", measurement.hormesis_effect);
    println!("  • Tail Benefit: {:.3}", measurement.tail_benefit);
    println!("  • Regime Adaptation: {:.3}", measurement.regime_adaptation);
    println!("  • Level: {}", measurement.level_description());
    println!();
    
    // 9. Set up black swan detection
    println!("🦢 Black Swan Detection Setup:");
    let black_swan_params = BlackSwanParams {
        min_std_devs: 2.5,
        probability_threshold: 0.01,
        lookback_period: 252,
        min_impact: 0.03,
        tail_thickness: 2.0,
        cluster_window: 21,
        correlation_breakdown: 0.3,
        volatility_spike: 3.0,
        regime_sensitivity: 0.5,
    };
    
    let mut black_swan_detector = BlackSwanDetector::new("portfolio_detector", black_swan_params);
    
    // Add some market observations
    for day in 0..10 {
        let observation = generate_market_observation(day);
        black_swan_detector.add_observation(observation)?;
    }
    
    let current_probability = black_swan_detector.get_current_probability();
    let alert_state = black_swan_detector.get_alert_state();
    
    println!("  • Current Black Swan Probability: {:.4}%", current_probability * 100.0);
    println!("  • Alert State: {:?}", alert_state);
    println!();
    
    // 10. Create barbell strategy
    println!("⚖️  Barbell Strategy Implementation:");
    let strategy_config = StrategyConfig {
        max_position_size: 0.25,
        risk_budget: 0.15,
        rebalancing_frequency: 30,
        min_position_size: 0.01,
        transaction_costs: 0.001,
        risk_aversion: 2.0,
        strategy_params: HashMap::new(),
    };
    
    let barbell_params = BarbellParams {
        safe_target: 0.8,      // 80% safe assets
        risky_target: 0.2,     // 20% risky assets
        max_safe_allocation: 0.95,
        max_risky_allocation: 0.3,
        min_safe_allocation: 0.6,
        min_risky_allocation: 0.05,
        safe_volatility_threshold: 0.05,
        risky_return_threshold: 0.12,
        rebalancing_tolerance: 0.05,
        adjustment_factor: 0.1,
        convexity_bias: 1.5,
    };
    
    let barbell_strategy = BarbellStrategy::new("portfolio_barbell", strategy_config, barbell_params)?;
    
    // Generate market data for strategy analysis
    let market_data = generate_comprehensive_market_data();
    
    let suitable = barbell_strategy.is_suitable(&market_data)?;
    println!("  • Strategy Suitable: {}", if suitable { "Yes" } else { "No" });
    
    if suitable {
        let expected_return = barbell_strategy.expected_return(&market_data)?;
        let strategy_risk_metrics = barbell_strategy.risk_metrics(&market_data)?;
        let barbell_metrics = barbell_strategy.get_barbell_metrics();
        
        println!("  • Expected Return: {:.2}%", expected_return * 100.0);
        println!("  • Strategy Volatility: {:.2}%", strategy_risk_metrics.volatility * 100.0);
        println!("  • Sharpe Ratio: {:.3}", strategy_risk_metrics.sortino_ratio);
        println!("  • Max Drawdown: {:.2}%", strategy_risk_metrics.max_drawdown * 100.0);
        println!("  • Safe Allocation: {:.1}%", barbell_metrics.safe_allocation * 100.0);
        println!("  • Risky Allocation: {:.1}%", barbell_metrics.risky_allocation * 100.0);
        println!("  • Barbell Ratio: {:.2}", barbell_metrics.barbell_ratio);
        println!("  • Convexity Exposure: {:.3}", barbell_metrics.convexity_exposure);
        println!("  • Safety Score: {:.3}", barbell_metrics.safety_score);
    }
    println!();
    
    // 11. Portfolio rebalancing simulation
    println!("🔄 Portfolio Rebalancing Simulation:");
    let mut target_weights = HashMap::new();
    target_weights.insert("US_TREASURY_BONDS".to_string(), 0.35);
    target_weights.insert("CASH_EQUIVALENT".to_string(), 0.25);
    target_weights.insert("GOLD".to_string(), 0.20);
    target_weights.insert("REAL_ESTATE".to_string(), 0.10);
    target_weights.insert("CRYPTOCURRENCY".to_string(), 0.05);
    target_weights.insert("TECH_STOCKS".to_string(), 0.03);
    target_weights.insert("CALL_OPTIONS".to_string(), 0.02);
    
    let rebalance_actions = portfolio.rebalance(target_weights)?;
    
    println!("  • Rebalancing Actions Required: {}", rebalance_actions.len());
    for action in &rebalance_actions {
        println!("    - {:?} {}: {:.1}% → {:.1}% (Δ {:.1}%)", 
                action.action_type,
                action.asset,
                action.current_weight * 100.0,
                action.target_weight * 100.0,
                action.weight_change * 100.0);
    }
    
    let total_cost: f64 = rebalance_actions.iter().map(|a| a.estimated_cost).sum();
    println!("  • Total Estimated Transaction Cost: {:.4}%", total_cost * 100.0);
    println!();
    
    // 12. Summary and recommendations
    println!("📋 Portfolio Summary & Recommendations:");
    println!("  ✓ Portfolio successfully configured with antifragile characteristics");
    println!("  ✓ Diversified across {} asset types with proper risk allocation", 
             portfolio.get_positions().len());
    println!("  ✓ Antifragility score of {:.3} indicates {} portfolio", 
             antifragility_score, antifragility_level.to_lowercase());
    println!("  ✓ Black swan monitoring active with {:.4}% current probability", 
             current_probability * 100.0);
    println!("  ✓ Barbell strategy provides asymmetric risk-return profile");
    
    if antifragility_score < 0.3 {
        println!("  ⚠️  Consider increasing allocation to antifragile assets");
    }
    
    if current_probability > 0.05 {
        println!("  ⚠️  Elevated black swan probability - consider risk reduction");
    }
    
    if risk_metrics.tail_risk > 1.0 {
        println!("  ⚠️  High tail risk detected - review risky asset allocation");
    }
    
    println!();
    println!("🎯 Next Steps:");
    println!("  1. Monitor antifragility metrics regularly");
    println!("  2. Adjust allocations based on market regime changes");
    println!("  3. Rebalance when drift exceeds tolerance thresholds");
    println!("  4. Maintain black swan detection and response protocols");
    println!("  5. Review and update risk parameters quarterly");
    
    Ok(())
}

/// Generate sample portfolio returns for antifragility analysis
fn generate_sample_portfolio_returns() -> Vec<f64> {
    // Simulate returns with some antifragile characteristics
    let mut returns = Vec::new();
    
    // Normal market periods
    for i in 0..80 {
        let base_return = 0.0008; // ~20% annualized
        let noise = (i as f64 * 0.1).sin() * 0.002;
        returns.push(base_return + noise);
    }
    
    // Stress period 1 - portfolio benefits from volatility
    for i in 0..10 {
        let stress_return = -0.01 + (i as f64 * 0.3).sin() * 0.02; // High volatility
        returns.push(stress_return);
    }
    
    // Recovery period
    for i in 0..10 {
        let recovery_return = 0.005 + (i as f64 * 0.2).cos() * 0.001;
        returns.push(recovery_return);
    }
    
    returns
}

/// Generate market observation for black swan detection
fn generate_market_observation(day: usize) -> MarketObservation {
    use std::collections::HashMap;
    use ndarray::Array2;
    
    let mut returns = HashMap::new();
    let mut volatilities = HashMap::new();
    let mut volumes = HashMap::new();
    
    // Generate realistic market data
    let market_stress = if day == 5 { 1.0 } else { 0.0 }; // Stress event on day 5
    
    returns.insert("SPY".to_string(), -0.02 * market_stress + 0.001);
    returns.insert("TLT".to_string(), 0.01 * market_stress + 0.0005);
    returns.insert("GLD".to_string(), 0.015 * market_stress + 0.0002);
    returns.insert("BTC".to_string(), -0.05 * market_stress + 0.002);
    
    volatilities.insert("SPY".to_string(), 0.02 + 0.03 * market_stress);
    volatilities.insert("TLT".to_string(), 0.01 + 0.01 * market_stress);
    volatilities.insert("GLD".to_string(), 0.015 + 0.02 * market_stress);
    volatilities.insert("BTC".to_string(), 0.04 + 0.06 * market_stress);
    
    volumes.insert("SPY".to_string(), 1000000.0 * (1.0 + market_stress));
    volumes.insert("TLT".to_string(), 500000.0 * (1.0 + 0.5 * market_stress));
    volumes.insert("GLD".to_string(), 300000.0 * (1.0 + 0.3 * market_stress));
    volumes.insert("BTC".to_string(), 200000.0 * (1.0 + 2.0 * market_stress));
    
    let correlations = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 0.3 + 0.4 * market_stress, 0.1, 0.5 + 0.3 * market_stress,
            0.3 + 0.4 * market_stress, 1.0, 0.2, 0.1,
            0.1, 0.2, 1.0, 0.3,
            0.5 + 0.3 * market_stress, 0.1, 0.3, 1.0,
        ]
    ).unwrap();
    
    MarketObservation {
        timestamp: Utc::now(),
        returns,
        volatilities,
        correlations,
        volumes,
        regime: if market_stress > 0.5 { 
            MarketRegime::Crisis 
        } else { 
            MarketRegime::Normal 
        },
    }
}

/// Generate comprehensive market data for strategy analysis
fn generate_comprehensive_market_data() -> MarketData {
    let mut prices = HashMap::new();
    let mut returns = HashMap::new();
    let mut volatilities = HashMap::new();
    let mut correlations = HashMap::new();
    let mut volumes = HashMap::new();
    let mut asset_types = HashMap::new();
    
    // Safe assets
    prices.insert("US_TREASURY_BONDS".to_string(), 102.5);
    returns.insert("US_TREASURY_BONDS".to_string(), vec![0.0005, 0.0003, 0.0008, 0.0002, 0.0006]);
    volatilities.insert("US_TREASURY_BONDS".to_string(), 0.015);
    volumes.insert("US_TREASURY_BONDS".to_string(), 5000000.0);
    asset_types.insert("US_TREASURY_BONDS".to_string(), AssetType::Safe);
    
    prices.insert("CASH_EQUIVALENT".to_string(), 100.0);
    returns.insert("CASH_EQUIVALENT".to_string(), vec![0.0001, 0.0001, 0.0001, 0.0001, 0.0001]);
    volatilities.insert("CASH_EQUIVALENT".to_string(), 0.001);
    volumes.insert("CASH_EQUIVALENT".to_string(), 10000000.0);
    asset_types.insert("CASH_EQUIVALENT".to_string(), AssetType::Safe);
    
    // Antifragile assets
    prices.insert("GOLD".to_string(), 1850.0);
    returns.insert("GOLD".to_string(), vec![0.002, -0.001, 0.003, 0.001, 0.002]);
    volatilities.insert("GOLD".to_string(), 0.018);
    volumes.insert("GOLD".to_string(), 2000000.0);
    asset_types.insert("GOLD".to_string(), AssetType::Antifragile);
    
    // Risky assets
    prices.insert("TECH_STOCKS".to_string(), 150.0);
    returns.insert("TECH_STOCKS".to_string(), vec![0.02, -0.01, 0.03, 0.01, 0.025]);
    volatilities.insert("TECH_STOCKS".to_string(), 0.25);
    volumes.insert("TECH_STOCKS".to_string(), 3000000.0);
    asset_types.insert("TECH_STOCKS".to_string(), AssetType::Volatile);
    
    prices.insert("CRYPTOCURRENCY".to_string(), 45000.0);
    returns.insert("CRYPTOCURRENCY".to_string(), vec![0.05, -0.03, 0.08, -0.02, 0.06]);
    volatilities.insert("CRYPTOCURRENCY".to_string(), 0.60);
    volumes.insert("CRYPTOCURRENCY".to_string(), 1000000.0);
    asset_types.insert("CRYPTOCURRENCY".to_string(), AssetType::Volatile);
    
    // Add correlations
    correlations.insert(("US_TREASURY_BONDS".to_string(), "TECH_STOCKS".to_string()), -0.2);
    correlations.insert(("GOLD".to_string(), "CRYPTOCURRENCY".to_string()), 0.1);
    correlations.insert(("CASH_EQUIVALENT".to_string(), "GOLD".to_string()), 0.05);
    
    MarketData {
        prices,
        returns,
        volatilities,
        correlations,
        volumes,
        asset_types,
        timestamp: Utc::now(),
        regime: MarketRegime::Normal,
    }
}