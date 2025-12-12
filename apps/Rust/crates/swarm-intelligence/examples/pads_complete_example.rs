//! # Complete PADS Example
//!
//! Comprehensive example demonstrating the Panarchy Adaptive Decision System (PADS)
//! for hyperbolic trading systems with hierarchical decision-making.

use std::collections::HashMap;
use std::time::Duration;
use tokio::time::{sleep, interval};
use tracing::{info, warn, error, Level};
use tracing_subscriber;

use swarm_intelligence::pads::{
    PadsSystem, PadsConfig, DecisionContext, DecisionLayer, AdaptiveCyclePhase,
    init_pads_with_config
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🚀 Starting PADS Complete Example for Hyperbolic Trading System");

    // Create enterprise-grade PADS configuration
    let config = create_trading_pads_config();
    
    // Initialize PADS system
    let mut pads = PadsSystem::new(config).await?;
    
    info!("✅ PADS system created successfully");
    
    // Start the PADS system
    pads.start().await?;
    info!("✅ PADS system started and operational");
    
    // Demonstrate different use cases
    demonstrate_tactical_decisions(&pads).await?;
    demonstrate_operational_decisions(&pads).await?;
    demonstrate_strategic_decisions(&pads).await?;
    demonstrate_meta_strategic_decisions(&pads).await?;
    
    // Demonstrate adaptive cycle awareness
    demonstrate_adaptive_cycle_decision_making(&pads).await?;
    
    // Demonstrate real-time decision stream
    demonstrate_real_time_decision_stream(&pads).await?;
    
    // Show system metrics and health
    demonstrate_system_monitoring(&pads).await?;
    
    // Graceful shutdown
    pads.stop().await?;
    info!("✅ PADS system stopped gracefully");
    
    Ok(())
}

/// Create enterprise-grade PADS configuration for trading systems
fn create_trading_pads_config() -> PadsConfig {
    PadsConfig::builder()
        .with_system_id("hyperbolic-trading-pads".to_string())
        .with_decision_layers(4) // All layers enabled
        .with_adaptive_cycles(true) // Enable panarchy framework
        .with_real_time_monitoring(true) // Enable monitoring
        .with_thread_pool_size(8) // High-performance threading
        .build()
}

/// Demonstrate tactical-level decisions (microseconds to seconds)
async fn demonstrate_tactical_decisions(pads: &PadsSystem) -> Result<(), Box<dyn std::error::Error>> {
    info!("🎯 Demonstrating Tactical Decision Making (Market Execution)");
    
    // Simulate rapid market execution decisions
    for i in 1..=5 {
        let mut context = DecisionContext::new(
            format!("tactical-execution-{:03}", i),
            DecisionLayer::Tactical,
            AdaptiveCyclePhase::Growth,
        );
        
        // Add market constraints
        context.constraints.insert("max_slippage".to_string(), 0.001); // 0.1%
        context.constraints.insert("max_latency_ms".to_string(), 50.0);
        context.urgency = 0.9; // High urgency for tactical decisions
        
        // Add market environment factors
        context.environment.insert("volatility".to_string(), 0.02);
        context.environment.insert("liquidity".to_string(), 0.8);
        context.environment.insert("spread_bps".to_string(), 2.5);
        
        let response = pads.make_decision(context).await?;
        
        info!("  ⚡ Tactical Decision {}: {} (confidence: {:.2}%)", 
              i, response.action, response.confidence * 100.0);
        
        // Simulate rapid execution
        sleep(Duration::from_millis(100)).await;
    }
    
    info!("✅ Tactical decisions completed\n");
    Ok(())
}

/// Demonstrate operational-level decisions (seconds to minutes)
async fn demonstrate_operational_decisions(pads: &PadsSystem) -> Result<(), Box<dyn std::error::Error>> {
    info!("⚙️ Demonstrating Operational Decision Making (Portfolio Optimization)");
    
    let operational_scenarios = vec![
        ("position-sizing", "Optimize position sizes for current portfolio"),
        ("risk-rebalancing", "Rebalance portfolio for optimal risk distribution"),
        ("hedge-adjustment", "Adjust hedging strategies based on market conditions"),
    ];
    
    for (scenario, description) in operational_scenarios {
        let mut context = DecisionContext::new(
            format!("operational-{}", scenario),
            DecisionLayer::Operational,
            AdaptiveCyclePhase::Conservation, // Optimization phase
        );
        
        // Add operational constraints
        context.constraints.insert("max_portfolio_risk".to_string(), 0.02); // 2% VaR
        context.constraints.insert("min_diversification".to_string(), 0.7);
        context.constraints.insert("max_concentration".to_string(), 0.1); // 10% max position
        context.urgency = 0.6; // Medium urgency
        
        // Add market environment
        context.environment.insert("market_regime".to_string(), 0.3); // Trending market
        context.environment.insert("correlation_regime".to_string(), 0.4);
        
        let response = pads.make_decision(context).await?;
        
        info!("  🔧 Operational Decision - {}: {}", description, response.action);
        info!("     Confidence: {:.1}%, Reasoning: {:?}", 
              response.confidence * 100.0, response.reasoning.get(0).unwrap_or(&"N/A".to_string()));
        
        sleep(Duration::from_millis(500)).await;
    }
    
    info!("✅ Operational decisions completed\n");
    Ok(())
}

/// Demonstrate strategic-level decisions (minutes to hours)
async fn demonstrate_strategic_decisions(pads: &PadsSystem) -> Result<(), Box<dyn std::error::Error>> {
    info!("🎯 Demonstrating Strategic Decision Making (Strategy Allocation)");
    
    let strategic_scenarios = vec![
        ("strategy-allocation", AdaptiveCyclePhase::Growth, "Allocate capital across trading strategies"),
        ("market-regime-adaptation", AdaptiveCyclePhase::Release, "Adapt to changing market regime"),
        ("alpha-source-optimization", AdaptiveCyclePhase::Conservation, "Optimize alpha generation sources"),
    ];
    
    for (scenario, phase, description) in strategic_scenarios {
        let mut context = DecisionContext::new(
            format!("strategic-{}", scenario),
            DecisionLayer::Strategic,
            phase,
        );
        
        // Add strategic constraints
        context.constraints.insert("max_strategy_allocation".to_string(), 0.25); // 25% max per strategy
        context.constraints.insert("min_sharpe_ratio".to_string(), 1.5);
        context.constraints.insert("max_drawdown".to_string(), 0.05); // 5% max drawdown
        context.urgency = 0.4; // Lower urgency for strategic decisions
        
        // Add strategic environment factors
        context.environment.insert("market_outlook".to_string(), 0.6); // Bullish
        context.environment.insert("macro_uncertainty".to_string(), 0.3);
        context.environment.insert("regulatory_environment".to_string(), 0.8);
        
        let response = pads.make_decision(context).await?;
        
        info!("  📊 Strategic Decision - {}: {}", description, response.action);
        info!("     Phase: {:?}, Confidence: {:.1}%", phase, response.confidence * 100.0);
        
        // Show reasoning for strategic decisions
        for (i, reason) in response.reasoning.iter().enumerate() {
            if i < 2 { // Show first 2 reasoning points
                info!("     Reasoning {}: {}", i + 1, reason);
            }
        }
        
        sleep(Duration::from_millis(800)).await;
    }
    
    info!("✅ Strategic decisions completed\n");
    Ok(())
}

/// Demonstrate meta-strategic decisions (hours to days)
async fn demonstrate_meta_strategic_decisions(pads: &PadsSystem) -> Result<(), Box<dyn std::error::Error>> {
    info!("🌟 Demonstrating Meta-Strategic Decision Making (System Evolution)");
    
    let meta_scenarios = vec![
        ("system-architecture", "Evolve trading system architecture"),
        ("capability-development", "Develop new trading capabilities"),
        ("paradigm-shift", "Adapt to fundamental market paradigm shift"),
    ];
    
    for (scenario, description) in meta_scenarios {
        let mut context = DecisionContext::new(
            format!("meta-strategic-{}", scenario),
            DecisionLayer::MetaStrategic,
            AdaptiveCyclePhase::Reorganization, // Transformation phase
        );
        
        // Add meta-strategic constraints
        context.constraints.insert("transformation_capacity".to_string(), 0.3);
        context.constraints.insert("system_stability".to_string(), 0.8);
        context.constraints.insert("innovation_risk".to_string(), 0.4);
        context.urgency = 0.2; // Lowest urgency but highest impact
        
        // Add long-term environment factors
        context.environment.insert("technology_evolution".to_string(), 0.7);
        context.environment.insert("competitive_landscape".to_string(), 0.5);
        context.environment.insert("regulatory_evolution".to_string(), 0.4);
        
        let response = pads.make_decision(context).await?;
        
        info!("  🚀 Meta-Strategic Decision - {}: {}", description, response.action);
        info!("     Transformation Focus, Confidence: {:.1}%", response.confidence * 100.0);
        
        // Meta-strategic decisions have comprehensive reasoning
        info!("     Comprehensive Analysis:");
        for (i, reason) in response.reasoning.iter().enumerate() {
            info!("       • {}", reason);
        }
        
        sleep(Duration::from_millis(1000)).await;
    }
    
    info!("✅ Meta-strategic decisions completed\n");
    Ok(())
}

/// Demonstrate adaptive cycle awareness in decision making
async fn demonstrate_adaptive_cycle_decision_making(pads: &PadsSystem) -> Result<(), Box<dyn std::error::Error>> {
    info!("🔄 Demonstrating Adaptive Cycle Awareness");
    
    let phases = vec![
        (AdaptiveCyclePhase::Growth, "Exploit growth opportunities"),
        (AdaptiveCyclePhase::Conservation, "Optimize and consolidate gains"),
        (AdaptiveCyclePhase::Release, "Creative destruction and innovation"),
        (AdaptiveCyclePhase::Reorganization, "Renewal and transformation"),
    ];
    
    for (phase, description) in phases {
        let mut context = DecisionContext::new(
            format!("adaptive-cycle-{:?}", phase),
            DecisionLayer::Operational,
            phase,
        );
        
        // Adjust context based on phase characteristics
        match phase {
            AdaptiveCyclePhase::Growth => {
                context.urgency = 0.7;
                context.risk_tolerance = 0.6;
                context.environment.insert("opportunity_level".to_string(), 0.8);
            }
            AdaptiveCyclePhase::Conservation => {
                context.urgency = 0.4;
                context.risk_tolerance = 0.3;
                context.environment.insert("efficiency_focus".to_string(), 0.9);
            }
            AdaptiveCyclePhase::Release => {
                context.urgency = 0.8;
                context.risk_tolerance = 0.8;
                context.environment.insert("innovation_pressure".to_string(), 0.9);
            }
            AdaptiveCyclePhase::Reorganization => {
                context.urgency = 0.6;
                context.risk_tolerance = 0.5;
                context.environment.insert("transformation_need".to_string(), 0.7);
            }
        }
        
        let response = pads.make_decision(context).await?;
        
        info!("  🌊 Phase {:?}: {}", phase, description);
        info!("     Decision: {} (confidence: {:.1}%)", response.action, response.confidence * 100.0);
        
        sleep(Duration::from_millis(600)).await;
    }
    
    info!("✅ Adaptive cycle demonstration completed\n");
    Ok(())
}

/// Demonstrate real-time decision stream
async fn demonstrate_real_time_decision_stream(pads: &PadsSystem) -> Result<(), Box<dyn std::error::Error>> {
    info!("📡 Demonstrating Real-Time Decision Stream (10 seconds)");
    
    let mut decision_count = 0;
    let mut interval_timer = interval(Duration::from_millis(500));
    let start_time = std::time::Instant::now();
    
    while start_time.elapsed() < Duration::from_secs(10) {
        interval_timer.tick().await;
        decision_count += 1;
        
        // Alternate between different decision types
        let (layer, phase) = match decision_count % 4 {
            0 => (DecisionLayer::Tactical, AdaptiveCyclePhase::Growth),
            1 => (DecisionLayer::Operational, AdaptiveCyclePhase::Conservation),
            2 => (DecisionLayer::Strategic, AdaptiveCyclePhase::Release),
            3 => (DecisionLayer::MetaStrategic, AdaptiveCyclePhase::Reorganization),
            _ => unreachable!(),
        };
        
        let mut context = DecisionContext::new(
            format!("realtime-{:03}", decision_count),
            layer,
            phase,
        );
        
        // Add real-time market simulation data
        context.environment.insert("price_momentum".to_string(), (decision_count as f64 % 10.0) / 10.0);
        context.environment.insert("volume_spike".to_string(), if decision_count % 3 == 0 { 0.8 } else { 0.3 });
        context.urgency = 0.5 + (decision_count as f64 % 5.0) / 10.0;
        
        match pads.make_decision(context).await {
            Ok(response) => {
                info!("  ⚡ RT-{:03} [{:?}]: {} ({:.1}%)", 
                      decision_count, layer, response.action, response.confidence * 100.0);
            }
            Err(e) => {
                warn!("  ❌ RT-{:03} failed: {}", decision_count, e);
            }
        }
    }
    
    info!("✅ Real-time stream completed: {} decisions processed\n", decision_count);
    Ok(())
}

/// Demonstrate system monitoring and health assessment
async fn demonstrate_system_monitoring(pads: &PadsSystem) -> Result<(), Box<dyn std::error::Error>> {
    info!("📊 System Monitoring and Health Assessment");
    
    // Check system health
    let health = pads.is_healthy().await;
    info!("  🏥 System Health: {}", if health { "HEALTHY ✅" } else { "UNHEALTHY ❌" });
    
    // Get system state
    let state = pads.get_state().await;
    info!("  📈 System State:");
    info!("     Active Layer: {:?}", state.active_layer);
    info!("     Cycle Phase: {:?}", state.cycle_phase);
    info!("     Health Status: {:?}", state.health);
    info!("     Version: {}", state.version);
    
    // Get performance metrics
    let metrics = pads.get_metrics().await;
    info!("  📊 Performance Metrics:");
    
    for (metric, value) in metrics.iter() {
        match metric.as_str() {
            "decisions_processed" => {
                info!("     Decisions Processed: {:.0}", value);
            }
            "avg_processing_time_ms" => {
                info!("     Average Processing Time: {:.1}ms", value);
            }
            "avg_confidence" => {
                info!("     Average Confidence: {:.1}%", value * 100.0);
            }
            "error_count" => {
                info!("     Error Count: {:.0}", value);
            }
            "uptime_seconds" => {
                info!("     Uptime: {:.0} seconds", value);
            }
            "system_health" => {
                info!("     Health Score: {:.1}%", value * 100.0);
            }
            _ => {
                info!("     {}: {:.3}", metric, value);
            }
        }
    }
    
    // Performance summary
    if let (Some(decisions), Some(avg_time), Some(avg_conf)) = (
        metrics.get("decisions_processed"),
        metrics.get("avg_processing_time_ms"),
        metrics.get("avg_confidence")
    ) {
        info!("  📋 Performance Summary:");
        info!("     Throughput: {:.1} decisions/second", decisions / 10.0); // Approximate
        info!("     Quality: {:.1}% average confidence", avg_conf * 100.0);
        info!("     Efficiency: {:.1}ms average latency", avg_time);
        
        // Calculate performance grade
        let performance_grade = if *avg_time < 50.0 && *avg_conf > 0.8 {
            "EXCELLENT 🌟"
        } else if *avg_time < 100.0 && *avg_conf > 0.7 {
            "GOOD ✅"
        } else if *avg_time < 200.0 && *avg_conf > 0.6 {
            "ACCEPTABLE 👍"
        } else {
            "NEEDS IMPROVEMENT ⚠️"
        };
        
        info!("     Overall Grade: {}", performance_grade);
    }
    
    info!("✅ System monitoring completed\n");
    Ok(())
}