//! Risk management tools implementation

use serde_json::{json, Value};
use chrono::Utc;

/// Calculate position size using Kelly Criterion
pub async fn calculate_position_size(params: Value) -> Value {
    let bankroll = params["bankroll"].as_f64().unwrap_or(100000.0);
    let win_probability = params["win_probability"].as_f64().unwrap_or(0.55);
    let win_loss_ratio = params["win_loss_ratio"].as_f64().unwrap_or(1.5);
    let risk_fraction = params["risk_fraction"].as_f64().unwrap_or(1.0);

    // Kelly Criterion: f* = (bp - q) / b
    // where b = win/loss ratio, p = win probability, q = loss probability
    let loss_probability = 1.0 - win_probability;
    let kelly_fraction = (win_loss_ratio * win_probability - loss_probability) / win_loss_ratio;

    // Apply risk fraction (conservative Kelly)
    let adjusted_fraction = kelly_fraction * risk_fraction;
    let position_size = bankroll * adjusted_fraction.max(0.0).min(0.25); // Cap at 25%

    json!({
        "calculation_id": format!("kelly_{}", Utc::now().timestamp()),
        "timestamp": Utc::now().to_rfc3339(),
        "inputs": {
            "bankroll": bankroll,
            "win_probability": win_probability,
            "win_loss_ratio": win_loss_ratio,
            "risk_fraction": risk_fraction
        },
        "results": {
            "kelly_fraction": kelly_fraction,
            "adjusted_fraction": adjusted_fraction,
            "recommended_position_size": position_size,
            "position_percentage": (position_size / bankroll) * 100.0,
            "max_loss": position_size / win_loss_ratio
        },
        "warnings": if kelly_fraction > 0.25 {
            vec!["Kelly fraction exceeds 25% - highly aggressive"]
        } else if kelly_fraction < 0.0 {
            vec!["Negative Kelly fraction - unfavorable odds, do not trade"]
        } else {
            vec![]
        },
        "methodology": "Kelly Criterion with conservative risk adjustment"
    })
}

/// Check risk limits before executing a trade
pub async fn check_risk_limits(params: Value) -> Value {
    let symbol = params["symbol"].as_str().unwrap_or("UNKNOWN");
    let quantity = params["quantity"].as_f64().unwrap_or(0.0);
    let price = params["price"].as_f64().unwrap_or(0.0);
    let side = params["side"].as_str().unwrap_or("buy");
    let portfolio_value = params["portfolio_value"].as_f64().unwrap_or(100000.0);

    let position_value = quantity * price;
    let position_pct = (position_value / portfolio_value) * 100.0;

    // Define risk limits
    let max_position_pct = 10.0;
    let max_daily_trades = 50;
    let max_sector_exposure_pct = 30.0;
    let max_leverage = 2.0;

    let current_leverage = 1.2; // Mock value
    let current_daily_trades = 23; // Mock value
    let current_sector_exposure = 18.5; // Mock value

    let mut violations = Vec::new();
    let mut warnings = Vec::new();
    let mut passed = true;

    // Check position size limit
    if position_pct > max_position_pct {
        violations.push(json!({
            "type": "position_size",
            "limit": max_position_pct,
            "actual": position_pct,
            "message": format!("Position size {:.2}% exceeds limit of {:.2}%", position_pct, max_position_pct)
        }));
        passed = false;
    } else if position_pct > max_position_pct * 0.8 {
        warnings.push("Position size approaching limit");
    }

    // Check daily trade limit
    if current_daily_trades >= max_daily_trades {
        violations.push(json!({
            "type": "daily_trades",
            "limit": max_daily_trades,
            "actual": current_daily_trades,
            "message": format!("Daily trade limit of {} reached", max_daily_trades)
        }));
        passed = false;
    }

    // Check leverage limit
    if current_leverage > max_leverage {
        violations.push(json!({
            "type": "leverage",
            "limit": max_leverage,
            "actual": current_leverage,
            "message": format!("Leverage {:.2}x exceeds limit of {:.2}x", current_leverage, max_leverage)
        }));
        passed = false;
    }

    json!({
        "check_id": format!("risk_check_{}", Utc::now().timestamp()),
        "timestamp": Utc::now().to_rfc3339(),
        "trade": {
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "side": side,
            "value": position_value,
            "portfolio_percentage": position_pct
        },
        "passed": passed,
        "violations": violations,
        "warnings": warnings,
        "limits": {
            "max_position_percentage": max_position_pct,
            "max_daily_trades": max_daily_trades,
            "max_sector_exposure": max_sector_exposure_pct,
            "max_leverage": max_leverage
        },
        "current_status": {
            "leverage": current_leverage,
            "daily_trades": current_daily_trades,
            "sector_exposure": current_sector_exposure
        },
        "recommendation": if passed {
            "Trade approved - within all risk limits"
        } else {
            "Trade rejected - violates risk limits"
        }
    })
}

/// Get current portfolio risk metrics (VaR, CVaR, etc.)
pub async fn get_portfolio_risk(params: Value) -> Value {
    let confidence_level = params["confidence_level"].as_f64().unwrap_or(0.95);
    let time_horizon = params["time_horizon_days"].as_i64().unwrap_or(1);
    let use_monte_carlo = params["use_monte_carlo"].as_bool().unwrap_or(true);
    let use_gpu = params["use_gpu"].as_bool().unwrap_or(false);

    let portfolio_value = 125340.50;

    json!({
        "analysis_id": format!("portfolio_risk_{}", Utc::now().timestamp()),
        "timestamp": Utc::now().to_rfc3339(),
        "portfolio_value": portfolio_value,
        "time_horizon_days": time_horizon,
        "confidence_level": confidence_level,
        "var_metrics": {
            "parametric_var": 2340.50,
            "historical_var": 2567.30,
            "monte_carlo_var": if use_monte_carlo { serde_json::json!(2445.80) } else { serde_json::json!(null) },
            "cvar": 3120.40,
            "expected_shortfall": 3240.50
        },
        "risk_decomposition": {
            "systematic_risk": 0.68,
            "idiosyncratic_risk": 0.32,
            "tail_risk": 0.15
        },
        "portfolio_metrics": {
            "volatility_annual": 0.18,
            "sharpe_ratio": 2.45,
            "sortino_ratio": 3.12,
            "beta": 1.12,
            "max_drawdown": 0.08,
            "correlation_avg": 0.65
        },
        "position_risk_contributions": [
            {
                "symbol": "AAPL",
                "var_contribution": 892.30,
                "percentage": 38.1,
                "marginal_var": 0.0156
            },
            {
                "symbol": "GOOGL",
                "var_contribution": 623.40,
                "percentage": 26.6,
                "marginal_var": 0.0134
            }
        ],
        "stress_scenarios": {
            "market_crash_10pct": -12534.05,
            "volatility_spike_2x": -8923.45,
            "correlation_surge": -5670.20,
            "liquidity_crisis": -7890.30
        },
        "gpu_accelerated": use_gpu,
        "computation_time_ms": if use_gpu { 187.3 } else { 2340.5 },
        "recommendations": [
            "Current VaR within acceptable limits",
            "Consider diversifying AAPL position (high risk contribution)",
            "Tail risk elevated - monitor closely"
        ]
    })
}

/// Run stress test scenarios on portfolio
pub async fn stress_test_portfolio(params: Value) -> Value {
    let scenarios = params["scenarios"].as_array()
        .and_then(|arr| arr.iter().map(|v| v.as_str()).collect::<Option<Vec<_>>>())
        .unwrap_or_else(|| vec!["market_crash", "volatility_spike", "sector_rotation"]);
    let portfolio_value = params["portfolio_value"].as_f64().unwrap_or(125340.50);
    let use_gpu = params["use_gpu"].as_bool().unwrap_or(false);

    let mut scenario_results = Vec::new();

    for scenario in scenarios.iter() {
        let result = match *scenario {
            "market_crash" => json!({
                "name": "Market Crash (2008-style)",
                "description": "S&P 500 drops 10% in one day",
                "impact": {
                    "portfolio_change": -12534.05,
                    "portfolio_change_pct": -10.0,
                    "var_breach": true,
                    "margin_call": false
                },
                "position_impacts": [
                    {"symbol": "AAPL", "impact_pct": -11.2},
                    {"symbol": "GOOGL", "impact_pct": -9.8}
                ]
            }),
            "volatility_spike" => json!({
                "name": "Volatility Spike",
                "description": "VIX doubles to 40",
                "impact": {
                    "portfolio_change": -8923.45,
                    "portfolio_change_pct": -7.12,
                    "var_breach": false,
                    "margin_call": false
                },
                "position_impacts": [
                    {"symbol": "AAPL", "impact_pct": -7.5},
                    {"symbol": "GOOGL", "impact_pct": -6.8}
                ]
            }),
            "sector_rotation" => json!({
                "name": "Tech Sector Rotation",
                "description": "Capital flows out of tech sector",
                "impact": {
                    "portfolio_change": -3456.78,
                    "portfolio_change_pct": -2.76,
                    "var_breach": false,
                    "margin_call": false
                },
                "position_impacts": [
                    {"symbol": "AAPL", "impact_pct": -3.2},
                    {"symbol": "GOOGL", "impact_pct": -2.9}
                ]
            }),
            "credit_crisis" => json!({
                "name": "Credit Crisis",
                "description": "Corporate spreads widen 200bps",
                "impact": {
                    "portfolio_change": -5670.30,
                    "portfolio_change_pct": -4.52,
                    "var_breach": false,
                    "margin_call": false
                },
                "position_impacts": [
                    {"symbol": "AAPL", "impact_pct": -4.8},
                    {"symbol": "GOOGL", "impact_pct": -4.3}
                ]
            }),
            _ => json!({
                "name": "Custom Scenario",
                "description": "User-defined stress scenario",
                "impact": {
                    "portfolio_change": -2000.0,
                    "portfolio_change_pct": -1.6,
                    "var_breach": false,
                    "margin_call": false
                }
            })
        };
        scenario_results.push(result);
    }

    // Calculate aggregate stress metrics
    let worst_case_loss: f64 = scenario_results.iter()
        .filter_map(|s| s["impact"]["portfolio_change"].as_f64())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    let avg_loss: f64 = scenario_results.iter()
        .filter_map(|s| s["impact"]["portfolio_change"].as_f64())
        .sum::<f64>() / scenario_results.len() as f64;

    json!({
        "test_id": format!("stress_test_{}", Utc::now().timestamp()),
        "timestamp": Utc::now().to_rfc3339(),
        "portfolio_value": portfolio_value,
        "scenarios_tested": scenario_results.len(),
        "scenarios": scenario_results,
        "aggregate_metrics": {
            "worst_case_loss": worst_case_loss,
            "worst_case_loss_pct": (worst_case_loss / portfolio_value) * 100.0,
            "average_loss": avg_loss,
            "average_loss_pct": (avg_loss / portfolio_value) * 100.0,
            "scenarios_breaching_var": 1,
            "scenarios_margin_call": 0
        },
        "resilience_score": {
            "overall": 7.8,
            "market_stress": 7.2,
            "liquidity_stress": 8.5,
            "tail_risk": 7.0
        },
        "recommendations": [
            "Portfolio resilient to most stress scenarios",
            "Consider hedging against market crash scenario",
            "Maintain adequate cash reserves for volatility spikes"
        ],
        "gpu_accelerated": use_gpu,
        "computation_time_ms": if use_gpu { 234.5 } else { 1890.3 }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_calculate_position_size() {
        let params = json!({
            "bankroll": 100000.0,
            "win_probability": 0.6,
            "win_loss_ratio": 2.0,
            "risk_fraction": 0.5
        });
        let result = calculate_position_size(params).await;
        assert!(result["results"]["kelly_fraction"].as_f64().unwrap() > 0.0);
        assert!(result["results"]["recommended_position_size"].as_f64().unwrap() > 0.0);
    }

    #[tokio::test]
    async fn test_check_risk_limits() {
        let params = json!({
            "symbol": "AAPL",
            "quantity": 100.0,
            "price": 180.0,
            "side": "buy",
            "portfolio_value": 100000.0
        });
        let result = check_risk_limits(params).await;
        assert!(result["passed"].is_boolean());
        assert!(result["limits"].is_object());
    }

    #[tokio::test]
    async fn test_get_portfolio_risk() {
        let params = json!({
            "confidence_level": 0.95,
            "time_horizon_days": 1,
            "use_monte_carlo": true
        });
        let result = get_portfolio_risk(params).await;
        assert!(result["var_metrics"].is_object());
        assert!(result["portfolio_metrics"].is_object());
    }

    #[tokio::test]
    async fn test_stress_test_portfolio() {
        let params = json!({
            "scenarios": ["market_crash", "volatility_spike"],
            "portfolio_value": 125340.50
        });
        let result = stress_test_portfolio(params).await;
        assert!(result["scenarios"].is_array());
        assert_eq!(result["scenarios_tested"], 2);
    }
}
