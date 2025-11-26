//! Demonstration of the 20 new MCP tools

use mcp_server::tools::{account, neural_extended, risk, config};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Neural Trader MCP Tools Demo ===\n");

    // Trading Operations (8 tools)
    println!("📊 TRADING OPERATIONS\n");

    println!("1. get_account_info:");
    let account_info = account::get_account_info(json!({
        "broker": "alpaca",
        "include_positions": true
    })).await;
    println!("   Account Status: {}", account_info["status"]);
    println!("   Portfolio Value: ${}\n", account_info["balances"]["portfolio_value"]);

    println!("2. get_positions:");
    let positions = account::get_positions(json!({})).await;
    println!("   Total Positions: {}\n", positions["total_count"]);

    println!("3. get_orders:");
    let orders = account::get_orders(json!({"status": "all", "limit": 10})).await;
    println!("   Total Orders: {}\n", orders["total_count"]);

    println!("4. get_portfolio_value:");
    let portfolio = account::get_portfolio_value(json!({
        "include_history": false,
        "include_breakdown": true
    })).await;
    println!("   Total Value: ${}", portfolio["total_value"]);
    println!("   Day Change: {}%\n", portfolio["performance"]["day_change_percent"]);

    println!("5. get_market_status:");
    let market = account::get_market_status(json!({"market": "US"})).await;
    println!("   Market Open: {}", market["is_open"]);
    println!("   Status: {}\n", market["current_status"]);

    // Neural Network Training (5 tools)
    println!("🧠 NEURAL NETWORK TRAINING\n");

    println!("6. neural_train_model:");
    let training = neural_extended::neural_train_model(json!({
        "model_type": "lstm",
        "dataset": "stock_prices",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "use_gpu": true
    })).await;
    println!("   Training ID: {}", training["training_id"]);
    println!("   Status: {}\n", training["status"]);

    println!("7. neural_get_status:");
    let status = neural_extended::neural_get_status(json!({"training_id": "latest"})).await;
    println!("   Progress: {}%", status["progress"]["completion_percentage"]);
    println!("   Current Loss: {}\n", status["current_metrics"]["train_loss"]);

    println!("8. neural_save_model:");
    let saved = neural_extended::neural_save_model(json!({
        "model_id": "lstm_v1",
        "include_optimizer": true
    })).await;
    println!("   Model Saved: {}", saved["save_path"]);
    println!("   Size: {} MB\n", saved["checkpoint_info"]["model_size_mb"]);

    // Risk Management (4 tools)
    println!("⚠️  RISK MANAGEMENT\n");

    println!("9. calculate_position_size:");
    let position_size = risk::calculate_position_size(json!({
        "bankroll": 100000.0,
        "win_probability": 0.6,
        "win_loss_ratio": 2.0,
        "risk_fraction": 0.5
    })).await;
    println!("   Kelly Fraction: {}", position_size["results"]["kelly_fraction"]);
    println!("   Recommended Size: ${}\n", position_size["results"]["recommended_position_size"]);

    println!("10. check_risk_limits:");
    let risk_check = risk::check_risk_limits(json!({
        "symbol": "AAPL",
        "quantity": 100.0,
        "price": 180.0,
        "side": "buy",
        "portfolio_value": 100000.0
    })).await;
    println!("   Passed: {}", risk_check["passed"]);
    println!("   Position %: {}%\n", risk_check["trade"]["portfolio_percentage"]);

    println!("11. get_portfolio_risk:");
    let portfolio_risk = risk::get_portfolio_risk(json!({
        "confidence_level": 0.95,
        "time_horizon_days": 1,
        "use_monte_carlo": true,
        "use_gpu": true
    })).await;
    println!("   VaR (95%): ${}", portfolio_risk["var_metrics"]["monte_carlo_var"]);
    println!("   CVaR: ${}\n", portfolio_risk["var_metrics"]["cvar"]);

    println!("12. stress_test_portfolio:");
    let stress = risk::stress_test_portfolio(json!({
        "scenarios": ["market_crash", "volatility_spike"],
        "portfolio_value": 125340.50,
        "use_gpu": true
    })).await;
    println!("   Scenarios Tested: {}", stress["scenarios_tested"]);
    println!("   Worst Case Loss: ${}\n", stress["aggregate_metrics"]["worst_case_loss"]);

    // System Configuration (3 tools)
    println!("⚙️  SYSTEM CONFIGURATION\n");

    println!("13. get_config:");
    let config_data = config::get_config(json!({"section": "risk"})).await;
    println!("   VaR Confidence: {}", config_data["risk"]["var_confidence_level"]);
    println!("   Kelly Fraction: {}\n", config_data["risk"]["kelly_fraction"]);

    println!("14. set_config:");
    let updated = config::set_config(json!({
        "section": "risk",
        "updates": {
            "kelly_fraction": 0.25
        }
    })).await;
    println!("   Status: {}", updated["status"]);
    println!("   Updates Applied: {}\n", updated["updates_applied"].as_array().unwrap().len());

    println!("15. health_check:");
    let health = config::health_check(json!({"detailed": false})).await;
    println!("   Overall Status: {}", health["overall_status"]);
    println!("   Healthy Components: {}/{}\n",
        health["summary"]["healthy"],
        health["summary"]["total_components"]);

    // Summary
    println!("✅ Demo Complete!");
    println!("\n📊 Tool Summary:");
    println!("   • Trading Operations: 8 tools");
    println!("   • Neural Training: 5 tools");
    println!("   • Risk Management: 4 tools");
    println!("   • System Config: 3 tools");
    println!("   ────────────────────────");
    println!("   Total: 20 new critical MCP tools");

    Ok(())
}
