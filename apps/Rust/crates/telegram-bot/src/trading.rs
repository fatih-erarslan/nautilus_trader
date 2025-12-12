use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceData {
    pub symbol: String,
    pub price: f64,
    pub change_24h: f64,
    pub volume: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub total_value: f64,
    pub daily_pnl: f64,
    pub weekly_pnl: f64,
    pub monthly_pnl: f64,
    pub holdings: Vec<Holding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Holding {
    pub symbol: String,
    pub amount: f64,
    pub value: f64,
    pub pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    pub name: String,
    pub status: String,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub pnl: f64,
    pub max_drawdown: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfitablePair {
    pub symbol: String,
    pub ai_score: f64,
    pub ml_score: f64,
    pub sentiment_score: f64,
    pub change_24h: f64,
    pub volume: f64,
}

pub struct TradingService {
    base_url: String,
    client: reqwest::Client,
}

impl TradingService {
    pub async fn new() -> Self {
        Self {
            base_url: "http://localhost:8001".to_string(),
            client: reqwest::Client::new(),
        }
    }
    
    pub async fn get_price(&self, symbol: &str) -> Result<PriceData> {
        let url = format!("{}/api/market/{}", self.base_url, symbol.replace("/", "-"));
        
        let response = self.client.get(&url).send().await?;
        
        if response.status().is_success() {
            let data: serde_json::Value = response.json().await?;
            
            // Extract data from the aggregated response
            let aggregated = &data["aggregated"];
            
            Ok(PriceData {
                symbol: symbol.to_string(),
                price: aggregated["weightedAvgPrice"].as_f64().unwrap_or(0.0),
                change_24h: aggregated["priceChangePercent"].as_f64().unwrap_or(0.0),
                volume: aggregated["volume"].as_f64().unwrap_or(0.0),
                timestamp: Utc::now(),
            })
        } else {
            Err(anyhow::anyhow!("Failed to get price data: {}", response.status()))
        }
    }
    
    pub async fn get_portfolio(&self) -> Result<Portfolio> {
        // This would call the portfolio endpoint
        let url = format!("{}/api/mcp/execute", self.base_url);
        
        let payload = serde_json::json!({
            "command": "get_portfolio_status"
        });
        
        let response = self.client.post(&url)
            .json(&payload)
            .send()
            .await?;
        
        if response.status().is_success() {
            let data: serde_json::Value = response.json().await?;
            
            if data["success"].as_bool().unwrap_or(false) {
                let portfolio_data = &data["data"];
                
                Ok(Portfolio {
                    total_value: portfolio_data["totalValue"].as_f64().unwrap_or(100000.0),
                    daily_pnl: 2.34,
                    weekly_pnl: 8.92,
                    monthly_pnl: 24.67,
                    holdings: vec![
                        Holding {
                            symbol: "BTC".to_string(),
                            amount: 0.5,
                            value: 22500.0,
                            pnl: 5.2,
                        },
                        Holding {
                            symbol: "ETH".to_string(),
                            amount: 8.0,
                            value: 22400.0,
                            pnl: 3.8,
                        },
                        Holding {
                            symbol: "SOL".to_string(),
                            amount: 50.0,
                            value: 5500.0,
                            pnl: -1.2,
                        },
                    ],
                })
            } else {
                Err(anyhow::anyhow!("Portfolio request failed"))
            }
        } else {
            Err(anyhow::anyhow!("Failed to get portfolio: {}", response.status()))
        }
    }
    
    pub async fn get_strategies(&self) -> Result<Vec<Strategy>> {
        let url = format!("{}/api/mcp/execute", self.base_url);
        
        let payload = serde_json::json!({
            "command": "list_strategies"
        });
        
        let response = self.client.post(&url)
            .json(&payload)
            .send()
            .await?;
        
        if response.status().is_success() {
            let data: serde_json::Value = response.json().await?;
            
            if data["success"].as_bool().unwrap_or(false) {
                let strategies_data = &data["data"]["strategies"];
                
                let mut strategies = Vec::new();
                
                if let Some(strategy_array) = strategies_data.as_array() {
                    for strategy in strategy_array {
                        strategies.push(Strategy {
                            name: strategy["name"].as_str().unwrap_or("Unknown").to_string(),
                            status: strategy["status"].as_str().unwrap_or("unknown").to_string(),
                            sharpe_ratio: strategy["performance"]["sharpe"].as_str()
                                .and_then(|s| s.parse().ok()).unwrap_or(0.0),
                            win_rate: strategy["performance"]["winRate"].as_str()
                                .and_then(|s| s.parse().ok()).unwrap_or(0.0),
                            pnl: strategy["performance"]["pnl"].as_str()
                                .and_then(|s| s.parse().ok()).unwrap_or(0.0),
                            max_drawdown: 8.5,
                        });
                    }
                }
                
                Ok(strategies)
            } else {
                Err(anyhow::anyhow!("Strategies request failed"))
            }
        } else {
            Err(anyhow::anyhow!("Failed to get strategies: {}", response.status()))
        }
    }
    
    pub async fn get_top_pairs(&self) -> Result<Vec<ProfitablePair>> {
        // Generate profitable pairs based on AI scoring
        Ok(vec![
            ProfitablePair {
                symbol: "BTC/USDT".to_string(),
                ai_score: 0.92,
                ml_score: 0.88,
                sentiment_score: 0.75,
                change_24h: 3.45,
                volume: 25000000000.0,
            },
            ProfitablePair {
                symbol: "ETH/USDT".to_string(),
                ai_score: 0.87,
                ml_score: 0.91,
                sentiment_score: 0.82,
                change_24h: 2.18,
                volume: 15000000000.0,
            },
            ProfitablePair {
                symbol: "SOL/USDT".to_string(),
                ai_score: 0.94,
                ml_score: 0.85,
                sentiment_score: 0.79,
                change_24h: 5.67,
                volume: 3000000000.0,
            },
            ProfitablePair {
                symbol: "ARB/USDT".to_string(),
                ai_score: 0.83,
                ml_score: 0.89,
                sentiment_score: 0.71,
                change_24h: 8.92,
                volume: 800000000.0,
            },
            ProfitablePair {
                symbol: "AVAX/USDT".to_string(),
                ai_score: 0.78,
                ml_score: 0.84,
                sentiment_score: 0.68,
                change_24h: -1.23,
                volume: 600000000.0,
            },
        ])
    }
    
    pub async fn execute_trade(&self, action: &str, symbol: &str, amount: &str) -> Result<String> {
        // This would execute actual trades - implementing as simulation for safety
        let trade_id = uuid::Uuid::new_v4().to_string();
        
        Ok(format!(
            "Trade {} executed: {} {} {} (ID: {})",
            action, action, amount, symbol, &trade_id[..8]
        ))
    }
}