//! News analysis and sentiment tracking
//!
//! Provides NAPI bindings for news collection and analysis

use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::error::NeuralTraderError;

/// Analyze news sentiment
#[napi]
pub async fn analyze_news(
    symbol: String,
    lookback_hours: Option<u32>,
) -> Result<NewsSentiment> {
    // Validate symbol
    if symbol.is_empty() {
        return Err(NeuralTraderError::News(
            "Symbol cannot be empty for news analysis".to_string()
        ).into());
    }

    // Validate lookback hours
    let hours = lookback_hours.unwrap_or(24);
    if hours == 0 {
        return Err(NeuralTraderError::News(
            "Lookback hours must be greater than 0".to_string()
        ).into());
    }

    if hours > 720 { // 30 days
        return Err(NeuralTraderError::News(
            format!("Lookback hours {} exceeds maximum of 720 (30 days)", hours)
        ).into());
    }

    tracing::info!(
        "Analyzing news sentiment for '{}' over last {} hours",
        symbol, hours
    );

    // TODO: Implement actual news sentiment analysis
    Ok(NewsSentiment {
        symbol,
        sentiment_score: 0.65,
        article_count: 42,
        positive_ratio: 0.68,
        negative_ratio: 0.32,
    })
}

/// News sentiment result
#[napi(object)]
pub struct NewsSentiment {
    pub symbol: String,
    pub sentiment_score: f64,
    pub article_count: u32,
    pub positive_ratio: f64,
    pub negative_ratio: f64,
}

/// Control news collection
#[napi]
pub async fn control_news_collection(
    action: String,
    symbols: Option<Vec<String>>,
) -> Result<String> {
    // Validate action
    let valid_actions = ["start", "stop", "pause", "resume", "status"];
    if !valid_actions.contains(&action.to_lowercase().as_str()) {
        return Err(NeuralTraderError::News(
            format!("Invalid action '{}'. Valid actions: {}", action, valid_actions.join(", "))
        ).into());
    }

    // Validate symbols if provided
    if let Some(ref syms) = symbols {
        if syms.is_empty() {
            return Err(NeuralTraderError::News(
                "Symbol list cannot be empty when provided".to_string()
            ).into());
        }

        for sym in syms {
            if sym.is_empty() {
                return Err(NeuralTraderError::News(
                    "Symbol list contains empty string".to_string()
                ).into());
            }
        }
    }

    tracing::info!(
        "News collection control: {} for symbols: {:?}",
        action, symbols
    );

    Ok(format!("News collection {} for symbols: {:?}", action, symbols))
}
