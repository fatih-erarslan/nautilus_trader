//! Polymarket API Client
//!
//! Integration with Polymarket CLOB (Central Limit Order Book) API

use crate::error::{MultiMarketError, Result};
use chrono::{DateTime, Utc};
use reqwest::{Client, StatusCode};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, error, info};

const API_BASE_URL: &str = "https://clob.polymarket.com";
const GAMMA_API_URL: &str = "https://gamma-api.polymarket.com";

/// Market on Polymarket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    /// Condition ID
    pub condition_id: String,
    /// Market question
    pub question: String,
    /// Market description
    pub description: Option<String>,
    /// Market outcomes
    pub outcomes: Vec<String>,
    /// Outcome prices (probabilities)
    pub outcome_prices: HashMap<String, Decimal>,
    /// Market volume
    pub volume: Decimal,
    /// Market liquidity
    pub liquidity: Decimal,
    /// Market active status
    pub active: bool,
    /// Market closed status
    pub closed: bool,
    /// End date
    pub end_date: Option<DateTime<Utc>>,
    /// Creation date
    pub created_at: DateTime<Utc>,
}

/// Order on Polymarket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Order ID
    pub id: String,
    /// Market ID
    pub market_id: String,
    /// Outcome index
    pub outcome_index: usize,
    /// Side (BUY or SELL)
    pub side: OrderSide,
    /// Size
    pub size: Decimal,
    /// Price
    pub price: Decimal,
    /// Order status
    pub status: OrderStatus,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Updated at
    pub updated_at: DateTime<Utc>,
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum OrderStatus {
    Open,
    Matched,
    Cancelled,
    Expired,
}

/// Position on Polymarket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Market ID
    pub market_id: String,
    /// Outcome index
    pub outcome_index: usize,
    /// Size
    pub size: Decimal,
    /// Average price
    pub avg_price: Decimal,
    /// Current value
    pub current_value: Decimal,
    /// Unrealized PnL
    pub unrealized_pnl: Decimal,
}

/// Trade on Polymarket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Market ID
    pub market_id: String,
    /// Outcome index
    pub outcome_index: usize,
    /// Side
    pub side: OrderSide,
    /// Size
    pub size: Decimal,
    /// Price
    pub price: Decimal,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Polymarket client
pub struct PolymarketClient {
    api_key: String,
    client: Client,
}

impl PolymarketClient {
    /// Create new Polymarket client
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: Client::new(),
        }
    }

    /// Get all active markets
    pub async fn get_markets(&self) -> Result<Vec<Market>> {
        let url = format!("{}/markets", GAMMA_API_URL);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            return Err(MultiMarketError::ApiError(format!(
                "Failed to get markets: {}",
                response.status()
            )));
        }

        let data: Vec<serde_json::Value> = response.json().await?;
        let mut markets = Vec::new();

        for market_data in data {
            if let Some(market) = self.parse_market(market_data) {
                markets.push(market);
            }
        }

        info!("Retrieved {} markets", markets.len());
        Ok(markets)
    }

    /// Get market by condition ID
    pub async fn get_market(&self, condition_id: &str) -> Result<Option<Market>> {
        let url = format!("{}/markets/{}", GAMMA_API_URL, condition_id);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }

        if response.status() != StatusCode::OK {
            return Err(MultiMarketError::ApiError(format!(
                "Failed to get market: {}",
                response.status()
            )));
        }

        let market_data: serde_json::Value = response.json().await?;
        Ok(self.parse_market(market_data))
    }

    /// Get orderbook for a market
    pub async fn get_orderbook(&self, market_id: &str) -> Result<serde_json::Value> {
        let url = format!("{}/book", API_BASE_URL);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .query(&[("market", market_id)])
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            return Err(MultiMarketError::ApiError(format!(
                "Failed to get orderbook: {}",
                response.status()
            )));
        }

        let orderbook = response.json().await?;
        Ok(orderbook)
    }

    /// Place order
    pub async fn place_order(
        &self,
        market_id: &str,
        outcome_index: usize,
        side: OrderSide,
        size: Decimal,
        price: Decimal,
    ) -> Result<Order> {
        let url = format!("{}/order", API_BASE_URL);

        let order_request = serde_json::json!({
            "market": market_id,
            "outcome": outcome_index,
            "side": side,
            "size": size.to_string(),
            "price": price.to_string(),
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&order_request)
            .send()
            .await?;

        if response.status() != StatusCode::OK && response.status() != StatusCode::CREATED {
            return Err(MultiMarketError::OrderError(format!(
                "Failed to place order: {}",
                response.status()
            )));
        }

        let order_data: serde_json::Value = response.json().await?;
        self.parse_order(order_data)
            .ok_or_else(|| MultiMarketError::OrderError("Failed to parse order".to_string()))
    }

    /// Cancel order
    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        let url = format!("{}/order/{}", API_BASE_URL, order_id);

        let response = self
            .client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if response.status() != StatusCode::OK && response.status() != StatusCode::NO_CONTENT {
            return Err(MultiMarketError::OrderError(format!(
                "Failed to cancel order: {}",
                response.status()
            )));
        }

        info!("Cancelled order: {}", order_id);
        Ok(())
    }

    /// Get user positions
    pub async fn get_positions(&self) -> Result<Vec<Position>> {
        let url = format!("{}/positions", API_BASE_URL);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            return Err(MultiMarketError::ApiError(format!(
                "Failed to get positions: {}",
                response.status()
            )));
        }

        let data: Vec<serde_json::Value> = response.json().await?;
        let positions = data
            .into_iter()
            .filter_map(|p| self.parse_position(p))
            .collect();

        Ok(positions)
    }

    /// Get recent trades for a market
    pub async fn get_trades(&self, market_id: &str, limit: usize) -> Result<Vec<Trade>> {
        let url = format!("{}/trades", API_BASE_URL);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .query(&[("market", market_id), ("limit", &limit.to_string())])
            .send()
            .await?;

        if response.status() != StatusCode::OK {
            return Err(MultiMarketError::ApiError(format!(
                "Failed to get trades: {}",
                response.status()
            )));
        }

        let data: Vec<serde_json::Value> = response.json().await?;
        let trades = data
            .into_iter()
            .filter_map(|t| self.parse_trade(t))
            .collect();

        Ok(trades)
    }

    fn parse_market(&self, data: serde_json::Value) -> Option<Market> {
        let condition_id = data.get("condition_id")?.as_str()?.to_string();
        let question = data.get("question")?.as_str()?.to_string();
        let description = data.get("description").and_then(|d| d.as_str()).map(String::from);

        let outcomes: Vec<String> = data
            .get("outcomes")?
            .as_array()?
            .iter()
            .filter_map(|o| o.as_str().map(String::from))
            .collect();

        let outcome_prices: HashMap<String, Decimal> = data
            .get("outcome_prices")?
            .as_object()?
            .iter()
            .filter_map(|(k, v)| {
                v.as_str()
                    .and_then(|s| s.parse::<Decimal>().ok())
                    .map(|d| (k.clone(), d))
            })
            .collect();

        let volume = data
            .get("volume")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or(Decimal::ZERO);

        let liquidity = data
            .get("liquidity")
            .and_then(|l| l.as_str())
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or(Decimal::ZERO);

        let active = data.get("active")?.as_bool().unwrap_or(true);
        let closed = data.get("closed")?.as_bool().unwrap_or(false);

        let end_date = data
            .get("end_date")
            .and_then(|d| d.as_str())
            .and_then(|s| s.parse::<DateTime<Utc>>().ok());

        let created_at = data
            .get("created_at")
            .and_then(|d| d.as_str())
            .and_then(|s| s.parse::<DateTime<Utc>>().ok())
            .unwrap_or_else(Utc::now);

        Some(Market {
            condition_id,
            question,
            description,
            outcomes,
            outcome_prices,
            volume,
            liquidity,
            active,
            closed,
            end_date,
            created_at,
        })
    }

    fn parse_order(&self, data: serde_json::Value) -> Option<Order> {
        Some(Order {
            id: data.get("id")?.as_str()?.to_string(),
            market_id: data.get("market")?.as_str()?.to_string(),
            outcome_index: data.get("outcome")?.as_u64()? as usize,
            side: serde_json::from_value(data.get("side")?.clone()).ok()?,
            size: data
                .get("size")
                .and_then(|s| s.as_str())
                .and_then(|s| s.parse::<Decimal>().ok())?,
            price: data
                .get("price")
                .and_then(|p| p.as_str())
                .and_then(|s| s.parse::<Decimal>().ok())?,
            status: serde_json::from_value(data.get("status")?.clone()).ok()?,
            created_at: data
                .get("created_at")
                .and_then(|d| d.as_str())
                .and_then(|s| s.parse::<DateTime<Utc>>().ok())
                .unwrap_or_else(Utc::now),
            updated_at: data
                .get("updated_at")
                .and_then(|d| d.as_str())
                .and_then(|s| s.parse::<DateTime<Utc>>().ok())
                .unwrap_or_else(Utc::now),
        })
    }

    fn parse_position(&self, data: serde_json::Value) -> Option<Position> {
        Some(Position {
            market_id: data.get("market")?.as_str()?.to_string(),
            outcome_index: data.get("outcome")?.as_u64()? as usize,
            size: data
                .get("size")
                .and_then(|s| s.as_str())
                .and_then(|s| s.parse::<Decimal>().ok())?,
            avg_price: data
                .get("avg_price")
                .and_then(|p| p.as_str())
                .and_then(|s| s.parse::<Decimal>().ok())?,
            current_value: data
                .get("current_value")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<Decimal>().ok())?,
            unrealized_pnl: data
                .get("unrealized_pnl")
                .and_then(|p| p.as_str())
                .and_then(|s| s.parse::<Decimal>().ok())
                .unwrap_or(Decimal::ZERO),
        })
    }

    fn parse_trade(&self, data: serde_json::Value) -> Option<Trade> {
        Some(Trade {
            id: data.get("id")?.as_str()?.to_string(),
            market_id: data.get("market")?.as_str()?.to_string(),
            outcome_index: data.get("outcome")?.as_u64()? as usize,
            side: serde_json::from_value(data.get("side")?.clone()).ok()?,
            size: data
                .get("size")
                .and_then(|s| s.as_str())
                .and_then(|s| s.parse::<Decimal>().ok())?,
            price: data
                .get("price")
                .and_then(|p| p.as_str())
                .and_then(|s| s.parse::<Decimal>().ok())?,
            timestamp: data
                .get("timestamp")
                .and_then(|t| t.as_str())
                .and_then(|s| s.parse::<DateTime<Utc>>().ok())
                .unwrap_or_else(Utc::now),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = PolymarketClient::new("test_api_key");
        assert_eq!(client.api_key, "test_api_key");
    }

    #[test]
    fn test_order_side_serialization() {
        let buy = OrderSide::Buy;
        let sell = OrderSide::Sell;

        let buy_json = serde_json::to_string(&buy).unwrap();
        let sell_json = serde_json::to_string(&sell).unwrap();

        assert_eq!(buy_json, r#""BUY""#);
        assert_eq!(sell_json, r#""SELL""#);
    }
}
