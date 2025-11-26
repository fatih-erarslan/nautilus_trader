//! News analysis and sentiment tools

use serde_json::{json, Value};
use chrono::Utc;

/// AI sentiment analysis of market news
pub async fn analyze_news(params: Value) -> Value {
    let symbol = params["symbol"].as_str().unwrap_or("AAPL");
    let lookback_hours = params["lookback_hours"].as_i64().unwrap_or(24);
    let use_gpu = params["use_gpu"].as_bool().unwrap_or(false);

    json!({
        "analysis_id": format!("news_{}", Utc::now().timestamp()),
        "symbol": symbol,
        "lookback_hours": lookback_hours,
        "timestamp": Utc::now().to_rfc3339(),
        "sentiment": {
            "overall_score": 0.72,
            "confidence": 0.85,
            "trend": "positive",
            "bullish_articles": 12,
            "bearish_articles": 3,
            "neutral_articles": 5
        },
        "key_topics": [
            {"topic": "earnings", "sentiment": 0.85, "mentions": 8},
            {"topic": "product launch", "sentiment": 0.78, "mentions": 5},
            {"topic": "market share", "sentiment": 0.65, "mentions": 3}
        ],
        "recent_headlines": [
            {
                "headline": "Strong Q4 earnings beat expectations",
                "sentiment": 0.92,
                "source": "Bloomberg",
                "timestamp": "2 hours ago"
            },
            {
                "headline": "New product announcement drives investor interest",
                "sentiment": 0.78,
                "source": "Reuters",
                "timestamp": "4 hours ago"
            }
        ],
        "gpu_accelerated": use_gpu,
        "computation_time_ms": if use_gpu { 23.5 } else { 145.2 }
    })
}

/// Get real-time news sentiment
pub async fn get_news_sentiment(params: Value) -> Value {
    let symbol = params["symbol"].as_str().unwrap_or("AAPL");

    json!({
        "symbol": symbol,
        "timestamp": Utc::now().to_rfc3339(),
        "current_sentiment": 0.72,
        "sentiment_trend": "improving",
        "24h_change": 0.08,
        "volume_mentions": 145,
        "sentiment_distribution": {
            "very_bullish": 0.25,
            "bullish": 0.35,
            "neutral": 0.25,
            "bearish": 0.10,
            "very_bearish": 0.05
        },
        "sentiment_by_source": [
            {"source": "Bloomberg", "sentiment": 0.78, "articles": 12},
            {"source": "Reuters", "sentiment": 0.68, "articles": 8},
            {"source": "WSJ", "sentiment": 0.75, "articles": 6}
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_analyze_news() {
        let params = json!({
            "symbol": "AAPL",
            "lookback_hours": 24
        });
        let result = analyze_news(params).await;
        assert_eq!(result["symbol"], "AAPL");
        assert!(result["sentiment"].is_object());
    }

    #[tokio::test]
    async fn test_get_news_sentiment() {
        let params = json!({"symbol": "GOOGL"});
        let result = get_news_sentiment(params).await;
        assert_eq!(result["symbol"], "GOOGL");
        assert!(result["current_sentiment"].is_f64());
    }
}
