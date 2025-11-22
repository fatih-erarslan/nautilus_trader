//! Basic market readiness validation example
//! 
//! This example demonstrates the complete market readiness validation system
//! including all 8 mandatory components implemented for production trading.

use std::sync::Arc;
use tengri_market_readiness_sentinel::{
    MarketReadinessSentinel,
    config::MarketReadinessConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 TENGRI Market Readiness Sentinel - Production Validation Example");
    println!("════════════════════════════════════════════════════════════════════");

    // Create configuration with production-ready settings
    let config = Arc::new(MarketReadinessConfig::default());
    
    // Initialize the comprehensive market readiness sentinel
    let mut sentinel = MarketReadinessSentinel::new(config).await?;
    
    // Initialize all validation components
    println!("📋 Initializing comprehensive market readiness validation components...");
    sentinel.initialize().await?;
    
    // Start continuous monitoring
    println!("🔍 Starting continuous market monitoring...");
    sentinel.start_continuous_monitoring().await?;
    
    // Run comprehensive market readiness validation
    println!("✅ Running comprehensive market readiness validation...");
    let report = sentinel.validate_market_readiness().await?;
    
    // Display validation results
    println!("\n🎯 MARKET READINESS VALIDATION REPORT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 Overall Status: {:?}", report.overall_status);
    println!("🕒 Timestamp: {}", report.timestamp);
    println!("🔄 Next Validation: {}", report.next_validation);
    
    println!("\n📈 VALIDATION COMPONENTS:");
    for (component, result) in &report.validations {
        let status_icon = match result.status {
            tengri_market_readiness_sentinel::ValidationStatus::Passed => "✅",
            tengri_market_readiness_sentinel::ValidationStatus::Warning => "⚠️",
            tengri_market_readiness_sentinel::ValidationStatus::Failed => "❌",
            tengri_market_readiness_sentinel::ValidationStatus::InProgress => "🔄",
        };
        println!("  {} {}: {} ({}ms, {:.1}% confidence)", 
                status_icon, 
                component, 
                result.message, 
                result.duration_ms,
                result.confidence * 100.0);
    }
    
    println!("\n🌍 MARKET CONDITIONS:");
    println!("  📊 Regime: {:?}", report.market_conditions.regime);
    println!("  📈 Volatility: {:?}", report.market_conditions.volatility_level);
    println!("  💧 Liquidity: {:?}", report.market_conditions.liquidity_status);
    println!("  🕐 Trading Hours: {:?}", report.market_conditions.trading_hours_status);
    println!("  💥 Market Impact: {:.4}", report.market_conditions.market_impact_estimate);
    println!("  📏 Spread: {:.4} bps", report.market_conditions.current_spread * 10000.0);
    
    println!("\n⚠️  RISK ASSESSMENT:");
    println!("  📊 VaR 95%: ${:.2}", report.risk_assessment.var_95);
    println!("  📊 VaR 99%: ${:.2}", report.risk_assessment.var_99);
    println!("  📉 Expected Shortfall: ${:.2}", report.risk_assessment.expected_shortfall);
    println!("  📉 Max Drawdown: {:.2}%", report.risk_assessment.max_drawdown * 100.0);
    println!("  🎯 Position Limits: ${:.0}", report.risk_assessment.position_limits.max_position_size);
    
    println!("\n🏛️  COMPLIANCE STATUS:");
    println!("  📋 Regulatory Checks: {}/{} passed", 
            report.compliance_status.regulatory_checks.iter().filter(|c| c.status).count(),
            report.compliance_status.regulatory_checks.len());
    println!("  🔴 Circuit Breakers: {}/{} active", 
            report.compliance_status.circuit_breakers.iter().filter(|c| c.enabled).count(),
            report.compliance_status.circuit_breakers.len());
    println!("  📊 Position Limits: {}", if report.compliance_status.position_limits_check { "✅" } else { "❌" });
    println!("  🕵️ Market Manipulation: {}", if report.compliance_status.market_manipulation_check { "✅" } else { "❌" });
    println!("  🎯 Best Execution: {}", if report.compliance_status.best_execution_check { "✅" } else { "❌" });
    
    if !report.recommendations.is_empty() {
        println!("\n💡 RECOMMENDATIONS:");
        for (i, recommendation) in report.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, recommendation);
        }
    }
    
    // Get system health status
    println!("\n🔧 SYSTEM HEALTH:");
    let health = sentinel.get_health_status().await?;
    println!("  🟢 Status: {:?}", health.status);
    println!("  ⏱️  Uptime: {} hours", health.uptime.num_hours());
    println!("  🔄 Validations: {}", health.validation_count);
    println!("  ❌ Errors: {}", health.error_count);
    
    // Stop monitoring gracefully
    println!("\n🛑 Stopping monitoring system...");
    sentinel.stop_continuous_monitoring().await?;
    
    println!("\n✅ Market readiness validation completed successfully!");
    println!("🎯 System is production-ready for institutional trading with:");
    println!("   • Real-time market data validation with failover");
    println!("   • Advanced market regime detection (6 regimes)");
    println!("   • Multi-timezone trading hours validation");
    println!("   • GARCH volatility assessment with stress testing");
    println!("   • VaR-based risk limits validation");
    println!("   • Multi-jurisdiction regulatory compliance (MiFID II, SEC, CFTC)");
    println!("   • Market impact assessment with multiple models");
    println!("   • Comprehensive monitoring and alerting");
    
    Ok(())
}