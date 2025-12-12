use risk_management::*;
use std::collections::HashMap;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 TENGRI Trading Swarm - Quantum-Enhanced Risk Management Demo");
    println!("================================================================");

    // Initialize the risk management system
    let config = RiskConfig::default();
    let risk_manager = RiskManager::new(config).await?;
    
    println!("✅ Risk management system initialized");

    // Create a sample portfolio
    let portfolio = create_sample_portfolio().await;
    println!("📊 Sample portfolio created with {} positions", portfolio.positions.len());
    println!("   Total Value: ${:.2}", portfolio.total_value);

    // 1. Calculate Value-at-Risk (VaR)
    println!("\n🔍 Calculating Value-at-Risk (VaR)...");
    let var_result = risk_manager.calculate_var(&portfolio, 0.05).await?;
    println!("   VaR (95% confidence): {:.2}%", var_result.var_values.get("5%").unwrap_or(&0.0) * 100.0);
    println!("   Calculation method: {:?}", var_result.method);
    println!("   Calculation time: {:?}", var_result.calculation_duration);

    // 2. Calculate Conditional VaR (Expected Shortfall)
    println!("\n📉 Calculating Conditional VaR (CVaR)...");
    let cvar_result = risk_manager.calculate_cvar(&portfolio, 0.05).await?;
    println!("   CVaR (95% confidence): {:.2}%", cvar_result.cvar_values.get("5%").unwrap_or(&0.0) * 100.0);

    // 3. Run stress tests
    println!("\n⚡ Running stress test scenarios...");
    let stress_scenarios = create_stress_scenarios().await;
    let stress_results = risk_manager.run_stress_tests(&portfolio, &stress_scenarios).await?;
    
    println!("   Stress tests completed in {:?}", stress_results.computation_time);
    println!("   Worst-case loss: ${:.2}", stress_results.overall_impact.worst_case_loss);
    println!("   Portfolio resilience score: {:.1}/100", stress_results.overall_impact.resilience_score);
    
    for scenario in &stress_results.scenario_results {
        println!("   📊 {}: {:.2}% P&L", scenario.scenario_name, scenario.portfolio_pnl_percent);
    }

    // 4. Get quantum risk metrics
    println!("\n🔬 Calculating quantum-enhanced risk metrics...");
    let quantum_metrics = risk_manager.get_quantum_risk_metrics(&portfolio).await?;
    println!("   Quantum VaR: {:.3}%", quantum_metrics.quantum_var * 100.0);
    println!("   Quantum CVaR: {:.3}%", quantum_metrics.quantum_cvar * 100.0);
    println!("   Quantum advantage: {:.3}", quantum_metrics.quantum_advantage);
    println!("   Quantum circuit fidelity: {:.3}", quantum_metrics.quantum_fidelity);

    // 5. Real-time monitoring
    println!("\n⏱️  Real-time risk monitoring...");
    let real_time_metrics = risk_manager.get_real_time_metrics().await?;
    println!("   Current drawdown: {:.2}%", real_time_metrics.current_drawdown * 100.0);
    println!("   Portfolio volatility: {:.2}%", real_time_metrics.portfolio_volatility * 100.0);
    println!("   Sharpe ratio: {:.2}", real_time_metrics.sharpe_ratio);
    println!("   Concentration risk: {:.2}%", real_time_metrics.concentration_risk * 100.0);

    // 6. Monte Carlo simulation
    println!("\n🎲 Running Monte Carlo simulation...");
    let time_horizon = std::time::Duration::from_secs(30 * 24 * 3600); // 30 days
    let mc_results = risk_manager.run_monte_carlo_simulation(&portfolio, 50_000, time_horizon).await?;
    println!("   Simulations: {}", mc_results.returns.len());
    println!("   Probability of loss: {:.2}%", mc_results.probability_of_loss * 100.0);
    println!("   Expected shortfall: {:.3}%", mc_results.expected_shortfall * 100.0);

    // 7. Comprehensive risk report
    println!("\n📋 Generating comprehensive risk report...");
    let comprehensive_report = risk_manager.get_comprehensive_risk_report(&portfolio).await?;
    println!("   Report generated in {:?}", comprehensive_report.generation_time);
    println!("   Quantum advantage: {:.3}", comprehensive_report.quantum_metrics.quantum_advantage);
    println!("   Compliance status: {}", if comprehensive_report.compliance_report.is_compliant { "✅ Compliant" } else { "❌ Non-compliant" });

    // 8. Risk-adjusted performance metrics
    println!("\n📈 Risk-adjusted performance metrics...");
    let benchmark_returns = vec![0.0008; portfolio.returns.len()]; // 20% annual benchmark
    let returns_array = ndarray::Array1::from_vec(portfolio.returns.clone());
    let benchmark_array = ndarray::Array1::from_vec(benchmark_returns);
    
    let risk_adjusted_metrics = risk_manager.calculate_risk_adjusted_metrics(&returns_array, &benchmark_array).await?;
    println!("   Sharpe ratio: {:.2}", risk_adjusted_metrics.sharpe_ratio);
    println!("   Sortino ratio: {:.2}", risk_adjusted_metrics.sortino_ratio);
    println!("   Calmar ratio: {:.2}", risk_adjusted_metrics.calmar_ratio);
    println!("   Maximum drawdown: {:.2}%", risk_adjusted_metrics.max_drawdown * 100.0);

    // 9. Correlation analysis
    println!("\n🔗 Correlation risk analysis...");
    let correlation_analysis = risk_manager.analyze_correlation_risk(&portfolio.assets, &portfolio.market_data).await?;
    println!("   Correlation stability: {:.2}", correlation_analysis.correlation_stability);
    println!("   Diversification ratio: {:.2}", correlation_analysis.diversification_ratio);

    // 10. Performance summary
    println!("\n⚡ Performance Summary");
    println!("=====================");
    println!("   VaR calculation: {:?}", var_result.calculation_duration);
    println!("   CVaR calculation: {:?}", cvar_result.calculation_duration);
    println!("   Stress testing: {:?}", stress_results.computation_time);
    println!("   Comprehensive report: {:?}", comprehensive_report.generation_time);

    // Check if real-time constraints are met
    if var_result.calculation_duration < std::time::Duration::from_micros(10) {
        println!("   ✅ VaR meets <10μs real-time constraint");
    } else {
        println!("   ⚠️  VaR calculation: {:?} (target: <10μs)", var_result.calculation_duration);
    }

    if real_time_metrics.timestamp.timestamp_millis() > 0 {
        println!("   ✅ Real-time monitoring active");
    }

    println!("\n🎉 Risk management demo completed successfully!");
    println!("   System ready for ultra-high frequency trading operations");

    Ok(())
}

async fn create_sample_portfolio() -> Portfolio {
    let mut portfolio = Portfolio::default();
    
    // Create realistic trading positions
    portfolio.positions = vec![
        Position {
            symbol: "AAPL".to_string(),
            quantity: 1000.0,
            price: 150.0,
            market_value: 150_000.0,
            weight: 0.25,
            pnl: 5_000.0,
            entry_price: 145.0,
            entry_time: chrono::Utc::now() - chrono::Duration::days(30),
        },
        Position {
            symbol: "GOOGL".to_string(),
            quantity: 200.0,
            price: 2500.0,
            market_value: 500_000.0,
            weight: 0.40,
            pnl: -10_000.0,
            entry_price: 2550.0,
            entry_time: chrono::Utc::now() - chrono::Duration::days(15),
        },
        Position {
            symbol: "TSLA".to_string(),
            quantity: 500.0,
            price: 400.0,
            market_value: 200_000.0,
            weight: 0.20,
            pnl: 25_000.0,
            entry_price: 350.0,
            entry_time: chrono::Utc::now() - chrono::Duration::days(60),
        },
        Position {
            symbol: "NVDA".to_string(),
            quantity: 300.0,
            price: 500.0,
            market_value: 150_000.0,
            weight: 0.15,
            pnl: 15_000.0,
            entry_price: 450.0,
            entry_time: chrono::Utc::now() - chrono::Duration::days(45),
        },
    ];
    
    // Add asset information
    portfolio.assets = vec![
        Asset {
            symbol: "AAPL".to_string(),
            name: "Apple Inc.".to_string(),
            asset_class: AssetClass::Equity,
            price: 150.0,
            volatility: 0.25,
            beta: 1.2,
            expected_return: 0.12,
            liquidity_score: 0.95,
        },
        Asset {
            symbol: "GOOGL".to_string(),
            name: "Alphabet Inc.".to_string(),
            asset_class: AssetClass::Equity,
            price: 2500.0,
            volatility: 0.30,
            beta: 1.1,
            expected_return: 0.15,
            liquidity_score: 0.90,
        },
        Asset {
            symbol: "TSLA".to_string(),
            name: "Tesla Inc.".to_string(),
            asset_class: AssetClass::Equity,
            price: 400.0,
            volatility: 0.45,
            beta: 1.8,
            expected_return: 0.20,
            liquidity_score: 0.85,
        },
        Asset {
            symbol: "NVDA".to_string(),
            name: "NVIDIA Corporation".to_string(),
            asset_class: AssetClass::Equity,
            price: 500.0,
            volatility: 0.40,
            beta: 1.6,
            expected_return: 0.25,
            liquidity_score: 0.88,
        },
    ];
    
    // Generate historical returns (1 year of daily returns)
    use rand::prelude::*;
    use rand_distr::Normal;
    
    let mut rng = thread_rng();
    let mut returns = Vec::new();
    
    // Simulate realistic market returns with autocorrelation
    let mut previous_return = 0.0;
    for _ in 0..252 {
        let base_return = Normal::new(0.0005, 0.02).unwrap().sample(&mut rng);
        let momentum = previous_return * 0.1; // Small momentum effect
        let current_return = base_return + momentum;
        returns.push(current_return);
        previous_return = current_return;
    }
    
    portfolio.returns = returns;
    portfolio.targets = vec![0.001; 252]; // 0.1% daily target
    
    // Update portfolio value
    portfolio.total_value = portfolio.calculate_value();
    portfolio.cash = 50_000.0; // $50k cash
    
    portfolio
}

async fn create_stress_scenarios() -> Vec<StressScenario> {
    use ndarray::Array2;
    
    vec![
        // Black Monday style crash
        StressScenario {
            name: "Market Crash 2024".to_string(),
            description: "Severe market downturn (-30% equity markets)".to_string(),
            asset_shocks: {
                let mut shocks = HashMap::new();
                shocks.insert("AAPL".to_string(), -0.25);
                shocks.insert("GOOGL".to_string(), -0.30);
                shocks.insert("TSLA".to_string(), -0.40);
                shocks.insert("NVDA".to_string(), -0.35);
                shocks
            },
            volatility_multipliers: {
                let mut multipliers = HashMap::new();
                multipliers.insert("AAPL".to_string(), 2.5);
                multipliers.insert("GOOGL".to_string(), 2.0);
                multipliers.insert("TSLA".to_string(), 3.0);
                multipliers.insert("NVDA".to_string(), 2.8);
                multipliers
            },
            correlation_shifts: Array2::zeros((0, 0)),
            liquidity_impacts: {
                let mut impacts = HashMap::new();
                impacts.insert("AAPL".to_string(), 0.2);
                impacts.insert("GOOGL".to_string(), 0.25);
                impacts.insert("TSLA".to_string(), 0.4);
                impacts.insert("NVDA".to_string(), 0.3);
                impacts
            },
            probability: 0.02,
        },
        
        // Tech sector rotation
        StressScenario {
            name: "Tech Rotation".to_string(),
            description: "Rotation out of growth stocks into value".to_string(),
            asset_shocks: {
                let mut shocks = HashMap::new();
                shocks.insert("AAPL".to_string(), -0.15);
                shocks.insert("GOOGL".to_string(), -0.20);
                shocks.insert("TSLA".to_string(), -0.25);
                shocks.insert("NVDA".to_string(), -0.30);
                shocks
            },
            volatility_multipliers: {
                let mut multipliers = HashMap::new();
                multipliers.insert("AAPL".to_string(), 1.5);
                multipliers.insert("GOOGL".to_string(), 1.3);
                multipliers.insert("TSLA".to_string(), 1.8);
                multipliers.insert("NVDA".to_string(), 2.0);
                multipliers
            },
            correlation_shifts: Array2::zeros((0, 0)),
            liquidity_impacts: {
                let mut impacts = HashMap::new();
                impacts.insert("AAPL".to_string(), 0.1);
                impacts.insert("GOOGL".to_string(), 0.15);
                impacts.insert("TSLA".to_string(), 0.2);
                impacts.insert("NVDA".to_string(), 0.25);
                impacts
            },
            probability: 0.15,
        },
        
        // Interest rate shock
        StressScenario {
            name: "Rate Shock".to_string(),
            description: "Federal Reserve emergency rate hike (+200bp)".to_string(),
            asset_shocks: {
                let mut shocks = HashMap::new();
                shocks.insert("AAPL".to_string(), -0.12);
                shocks.insert("GOOGL".to_string(), -0.10);
                shocks.insert("TSLA".to_string(), -0.18);
                shocks.insert("NVDA".to_string(), -0.15);
                shocks
            },
            volatility_multipliers: {
                let mut multipliers = HashMap::new();
                multipliers.insert("AAPL".to_string(), 1.3);
                multipliers.insert("GOOGL".to_string(), 1.2);
                multipliers.insert("TSLA".to_string(), 1.6);
                multipliers.insert("NVDA".to_string(), 1.4);
                multipliers
            },
            correlation_shifts: Array2::zeros((0, 0)),
            liquidity_impacts: {
                let mut impacts = HashMap::new();
                impacts.insert("AAPL".to_string(), 0.05);
                impacts.insert("GOOGL".to_string(), 0.05);
                impacts.insert("TSLA".to_string(), 0.1);
                impacts.insert("NVDA".to_string(), 0.08);
                impacts
            },
            probability: 0.10,
        },
    ]
}