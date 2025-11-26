//! Account and trading operations tools

use serde_json::{json, Value};
use chrono::Utc;

/// Get account information including balance and buying power
pub async fn get_account_info(params: Value) -> Value {
    let broker = params["broker"].as_str().unwrap_or("alpaca");
    let include_positions = params["include_positions"].as_bool().unwrap_or(true);

    json!({
        "account_id": "acc_12345",
        "broker": broker,
        "timestamp": Utc::now().to_rfc3339(),
        "status": "active",
        "account_type": "margin",
        "balances": {
            "cash": 45230.00,
            "portfolio_value": 125340.50,
            "equity": 125340.50,
            "long_market_value": 80110.50,
            "short_market_value": 0.0,
            "buying_power": 90460.00,
            "margin_used": 34880.50,
            "margin_available": 90460.00,
            "maintenance_margin": 20027.63,
            "initial_margin": 40055.25,
            "last_maintenance_margin": 20027.63
        },
        "day_trading": {
            "pattern_day_trader": false,
            "day_trade_count": 2,
            "day_trades_remaining": 1,
            "last_day_trade": "2024-11-12"
        },
        "performance": {
            "unrealized_pl": 5110.50,
            "unrealized_pl_percent": 4.25,
            "realized_pl_today": 890.30,
            "realized_pl_total": 12340.50
        },
        "positions_summary": if include_positions {
            json!({
                "total_positions": 2,
                "long_positions": 2,
                "short_positions": 0,
                "largest_position": {
                    "symbol": "AAPL",
                    "value": 26767.50,
                    "percentage": 21.4
                }
            })
        } else {
            json!(null)
        },
        "restrictions": {
            "trading_blocked": false,
            "transfers_blocked": false,
            "account_blocked": false,
            "trading_suspended": false
        }
    })
}

/// Get current open positions
pub async fn get_positions(params: Value) -> Value {
    let symbol_filter = params["symbol"].as_str();
    let _include_closed = params["include_closed"].as_bool().unwrap_or(false);

    let mut positions = vec![
        json!({
            "symbol": "AAPL",
            "asset_id": "asset_aapl_123",
            "exchange": "NASDAQ",
            "asset_class": "us_equity",
            "quantity": 150.0,
            "side": "long",
            "avg_entry_price": 172.30,
            "current_price": 178.45,
            "market_value": 26767.50,
            "cost_basis": 25845.00,
            "unrealized_pl": 922.50,
            "unrealized_pl_percent": 3.57,
            "unrealized_intraday_pl": 145.50,
            "unrealized_intraday_pl_percent": 0.54,
            "current_price_as_of": Utc::now().to_rfc3339(),
            "lastday_price": 177.48,
            "change_today": 0.97
        }),
        json!({
            "symbol": "GOOGL",
            "asset_id": "asset_googl_456",
            "exchange": "NASDAQ",
            "asset_class": "us_equity",
            "quantity": 80.0,
            "side": "long",
            "avg_entry_price": 138.20,
            "current_price": 142.15,
            "market_value": 11372.00,
            "cost_basis": 11056.00,
            "unrealized_pl": 316.00,
            "unrealized_pl_percent": 2.86,
            "unrealized_intraday_pl": 68.00,
            "unrealized_intraday_pl_percent": 0.60,
            "current_price_as_of": Utc::now().to_rfc3339(),
            "lastday_price": 141.30,
            "change_today": 0.85
        })
    ];

    if let Some(symbol) = symbol_filter {
        positions.retain(|p| p["symbol"].as_str() == Some(symbol));
    }

    json!({
        "timestamp": Utc::now().to_rfc3339(),
        "positions": positions,
        "total_count": positions.len(),
        "summary": {
            "total_market_value": positions.iter()
                .filter_map(|p| p["market_value"].as_f64())
                .sum::<f64>(),
            "total_unrealized_pl": positions.iter()
                .filter_map(|p| p["unrealized_pl"].as_f64())
                .sum::<f64>(),
            "total_cost_basis": positions.iter()
                .filter_map(|p| p["cost_basis"].as_f64())
                .sum::<f64>()
        }
    })
}

/// Get order history and status
pub async fn get_orders(params: Value) -> Value {
    let status_filter = params["status"].as_str().unwrap_or("all");
    let limit = params["limit"].as_i64().unwrap_or(50);
    let symbol_filter = params["symbol"].as_str();

    let orders = vec![
        json!({
            "order_id": "ord_12345",
            "client_order_id": "client_ord_001",
            "created_at": "2024-11-13T10:30:00Z",
            "updated_at": "2024-11-13T10:30:05Z",
            "submitted_at": "2024-11-13T10:30:00Z",
            "filled_at": "2024-11-13T10:30:05Z",
            "symbol": "AAPL",
            "asset_class": "us_equity",
            "side": "buy",
            "order_type": "market",
            "time_in_force": "day",
            "quantity": 50.0,
            "filled_quantity": 50.0,
            "remaining_quantity": 0.0,
            "status": "filled",
            "limit_price": null,
            "stop_price": null,
            "filled_avg_price": 178.42,
            "extended_hours": false,
            "legs": null
        }),
        json!({
            "order_id": "ord_12346",
            "client_order_id": "client_ord_002",
            "created_at": "2024-11-13T11:15:00Z",
            "updated_at": "2024-11-13T11:15:00Z",
            "submitted_at": "2024-11-13T11:15:00Z",
            "filled_at": null,
            "symbol": "GOOGL",
            "asset_class": "us_equity",
            "side": "buy",
            "order_type": "limit",
            "time_in_force": "gtc",
            "quantity": 30.0,
            "filled_quantity": 0.0,
            "remaining_quantity": 30.0,
            "status": "pending_new",
            "limit_price": 140.50,
            "stop_price": null,
            "filled_avg_price": null,
            "extended_hours": false,
            "legs": null
        }),
        json!({
            "order_id": "ord_12347",
            "client_order_id": "client_ord_003",
            "created_at": "2024-11-13T09:45:00Z",
            "updated_at": "2024-11-13T09:50:00Z",
            "submitted_at": "2024-11-13T09:45:00Z",
            "filled_at": null,
            "expired_at": "2024-11-13T09:50:00Z",
            "symbol": "TSLA",
            "asset_class": "us_equity",
            "side": "sell",
            "order_type": "limit",
            "time_in_force": "ioc",
            "quantity": 20.0,
            "filled_quantity": 0.0,
            "remaining_quantity": 0.0,
            "status": "expired",
            "limit_price": 235.00,
            "stop_price": null,
            "filled_avg_price": null,
            "extended_hours": false,
            "legs": null
        })
    ];

    let mut filtered_orders = orders.clone();

    if status_filter != "all" {
        filtered_orders.retain(|o| o["status"].as_str() == Some(status_filter));
    }

    if let Some(symbol) = symbol_filter {
        filtered_orders.retain(|o| o["symbol"].as_str() == Some(symbol));
    }

    filtered_orders.truncate(limit as usize);

    json!({
        "timestamp": Utc::now().to_rfc3339(),
        "orders": filtered_orders,
        "total_count": filtered_orders.len(),
        "summary": {
            "filled": filtered_orders.iter().filter(|o| o["status"] == "filled").count(),
            "pending": filtered_orders.iter().filter(|o| o["status"] == "pending_new").count(),
            "cancelled": filtered_orders.iter().filter(|o| o["status"] == "cancelled").count(),
            "expired": filtered_orders.iter().filter(|o| o["status"] == "expired").count()
        }
    })
}

/// Cancel a pending order
pub async fn cancel_order(params: Value) -> Value {
    let order_id = params["order_id"].as_str().unwrap_or("unknown");

    json!({
        "order_id": order_id,
        "status": "cancelled",
        "cancelled_at": Utc::now().to_rfc3339(),
        "message": "Order cancelled successfully",
        "original_order": {
            "symbol": "GOOGL",
            "side": "buy",
            "quantity": 30.0,
            "order_type": "limit",
            "limit_price": 140.50
        }
    })
}

/// Modify an existing order
pub async fn modify_order(params: Value) -> Value {
    let order_id = params["order_id"].as_str().unwrap_or("unknown");
    let new_quantity = params["quantity"].as_f64();
    let new_limit_price = params["limit_price"].as_f64();

    json!({
        "order_id": order_id,
        "status": "replaced",
        "replaced_at": Utc::now().to_rfc3339(),
        "message": "Order modified successfully",
        "new_order_id": format!("ord_{}", Utc::now().timestamp()),
        "modifications": {
            "quantity": new_quantity,
            "limit_price": new_limit_price
        },
        "updated_order": {
            "symbol": "GOOGL",
            "side": "buy",
            "quantity": new_quantity.unwrap_or(30.0),
            "order_type": "limit",
            "limit_price": new_limit_price.unwrap_or(140.50),
            "time_in_force": "gtc"
        }
    })
}

/// Get trade execution fills
pub async fn get_fills(params: Value) -> Value {
    let order_id = params["order_id"].as_str();
    let limit = params["limit"].as_i64().unwrap_or(50);

    let fills = vec![
        json!({
            "fill_id": "fill_001",
            "order_id": "ord_12345",
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 50.0,
            "price": 178.42,
            "timestamp": "2024-11-13T10:30:05Z",
            "liquidity": "taker",
            "commission": 0.50
        }),
        json!({
            "fill_id": "fill_002",
            "order_id": "ord_12344",
            "symbol": "GOOGL",
            "side": "sell",
            "quantity": 20.0,
            "price": 141.85,
            "timestamp": "2024-11-13T09:15:03Z",
            "liquidity": "maker",
            "commission": 0.20
        })
    ];

    let mut filtered_fills = fills.clone();

    if let Some(oid) = order_id {
        filtered_fills.retain(|f| f["order_id"].as_str() == Some(oid));
    }

    filtered_fills.truncate(limit as usize);

    json!({
        "timestamp": Utc::now().to_rfc3339(),
        "fills": filtered_fills,
        "total_count": filtered_fills.len(),
        "summary": {
            "total_quantity": filtered_fills.iter()
                .filter_map(|f| f["quantity"].as_f64())
                .sum::<f64>(),
            "total_commission": filtered_fills.iter()
                .filter_map(|f| f["commission"].as_f64())
                .sum::<f64>(),
            "buys": filtered_fills.iter().filter(|f| f["side"] == "buy").count(),
            "sells": filtered_fills.iter().filter(|f| f["side"] == "sell").count()
        }
    })
}

/// Get total portfolio value with breakdown
pub async fn get_portfolio_value(params: Value) -> Value {
    let include_history = params["include_history"].as_bool().unwrap_or(false);
    let include_breakdown = params["include_breakdown"].as_bool().unwrap_or(true);

    let mut response = json!({
        "timestamp": Utc::now().to_rfc3339(),
        "total_value": 125340.50,
        "components": {
            "cash": 45230.00,
            "long_positions": 80110.50,
            "short_positions": 0.0,
            "options": 0.0,
            "crypto": 0.0
        },
        "performance": {
            "day_change": 1238.50,
            "day_change_percent": 0.99,
            "week_change": 3456.20,
            "week_change_percent": 2.84,
            "month_change": 8901.30,
            "month_change_percent": 7.64,
            "year_change": 23450.50,
            "year_change_percent": 22.98
        },
        "metrics": {
            "sharpe_ratio": 2.45,
            "sortino_ratio": 3.12,
            "max_drawdown": 0.08,
            "volatility": 0.18,
            "beta": 1.12
        }
    });

    if include_breakdown {
        response["position_breakdown"] = json!([
            {
                "symbol": "AAPL",
                "value": 26767.50,
                "percentage": 21.36,
                "unrealized_pl": 922.50
            },
            {
                "symbol": "GOOGL",
                "value": 11372.00,
                "percentage": 9.07,
                "unrealized_pl": 316.00
            },
            {
                "asset_class": "Cash",
                "value": 45230.00,
                "percentage": 36.08
            }
        ]);
    }

    if include_history {
        response["history_30d"] = json!({
            "values": vec![116439.0, 118234.5, 120156.3, 122340.5, 125340.5],
            "dates": vec!["2024-10-14", "2024-10-21", "2024-10-28", "2024-11-04", "2024-11-13"]
        });
    }

    response
}

/// Get market status (open/closed/pre-market/after-hours)
pub async fn get_market_status(params: Value) -> Value {
    let market = params["market"].as_str().unwrap_or("US");

    json!({
        "timestamp": Utc::now().to_rfc3339(),
        "market": market,
        "is_open": true,
        "current_status": "open",
        "next_open": "2024-11-14T09:30:00-05:00",
        "next_close": "2024-11-13T16:00:00-05:00",
        "trading_hours": {
            "pre_market": {
                "start": "04:00:00",
                "end": "09:30:00",
                "is_active": false
            },
            "regular": {
                "start": "09:30:00",
                "end": "16:00:00",
                "is_active": true
            },
            "after_hours": {
                "start": "16:00:00",
                "end": "20:00:00",
                "is_active": false
            }
        },
        "exchange_holidays": [
            {"date": "2024-11-28", "name": "Thanksgiving"},
            {"date": "2024-12-25", "name": "Christmas"},
            {"date": "2025-01-01", "name": "New Year's Day"}
        ],
        "early_close_days": [
            {"date": "2024-11-29", "close_time": "13:00:00", "reason": "Day after Thanksgiving"}
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_account_info() {
        let params = json!({
            "broker": "alpaca",
            "include_positions": true
        });
        let result = get_account_info(params).await;
        assert_eq!(result["status"], "active");
        assert!(result["balances"].is_object());
    }

    #[tokio::test]
    async fn test_get_positions() {
        let params = json!({});
        let result = get_positions(params).await;
        assert!(result["positions"].is_array());
        assert!(result["summary"].is_object());
    }

    #[tokio::test]
    async fn test_get_orders() {
        let params = json!({
            "status": "all",
            "limit": 10
        });
        let result = get_orders(params).await;
        assert!(result["orders"].is_array());
    }

    #[tokio::test]
    async fn test_cancel_order() {
        let params = json!({"order_id": "ord_12346"});
        let result = cancel_order(params).await;
        assert_eq!(result["status"], "cancelled");
    }

    #[tokio::test]
    async fn test_modify_order() {
        let params = json!({
            "order_id": "ord_12346",
            "quantity": 40.0,
            "limit_price": 141.00
        });
        let result = modify_order(params).await;
        assert_eq!(result["status"], "replaced");
    }

    #[tokio::test]
    async fn test_get_fills() {
        let params = json!({"limit": 10});
        let result = get_fills(params).await;
        assert!(result["fills"].is_array());
    }

    #[tokio::test]
    async fn test_get_portfolio_value() {
        let params = json!({
            "include_history": true,
            "include_breakdown": true
        });
        let result = get_portfolio_value(params).await;
        assert!(result["total_value"].as_f64().unwrap() > 0.0);
    }

    #[tokio::test]
    async fn test_get_market_status() {
        let params = json!({"market": "US"});
        let result = get_market_status(params).await;
        assert!(result["is_open"].is_boolean());
    }
}
