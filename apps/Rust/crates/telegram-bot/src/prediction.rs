use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMarket {
    pub id: String,
    pub title: String,
    pub description: String,
    pub category: String,
    pub close_date: DateTime<Utc>,
    pub probability_yes: f64,
    pub probability_no: f64,
    pub volume: f64,
    pub liquidity: f64,
    pub participants: u32,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketPosition {
    pub market_id: String,
    pub market_title: String,
    pub position: String, // "YES" or "NO"
    pub shares: f64,
    pub avg_price: f64,
    pub current_value: f64,
    pub pnl: f64,
    pub pnl_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketAnalysis {
    pub market_id: String,
    pub title: String,
    pub sentiment_score: f64,
    pub volume_trend: String,
    pub price_momentum: f64,
    pub expected_value: f64,
    pub kelly_criterion: f64,
    pub recommendation: String,
    pub confidence: f64,
    pub risk_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionPortfolio {
    pub total_value: f64,
    pub invested_amount: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub win_rate: f64,
    pub active_positions: u32,
    pub resolved_positions: u32,
}

pub struct PredictionService {
    base_url: String,
    client: reqwest::Client,
}

impl PredictionService {
    pub async fn new() -> Self {
        Self {
            base_url: "http://localhost:8001".to_string(),
            client: reqwest::Client::new(),
        }
    }
    
    pub async fn get_prediction_markets(&self) -> Result<Vec<PredictionMarket>> {
        // Generate realistic prediction markets
        Ok(vec![
            PredictionMarket {
                id: "btc-50k-2024".to_string(),
                title: "Will Bitcoin reach $50,000 by end of 2024?".to_string(),
                description: "Resolves YES if Bitcoin (BTC) trades at or above $50,000 on any major exchange by December 31, 2024".to_string(),
                category: "Crypto".to_string(),
                close_date: chrono::DateTime::parse_from_rfc3339("2024-12-31T23:59:59Z").unwrap().with_timezone(&Utc),
                probability_yes: 0.67,
                probability_no: 0.33,
                volume: 2450000.0,
                liquidity: 185000.0,
                participants: 1247,
                status: "active".to_string(),
            },
            PredictionMarket {
                id: "eth-merge-success".to_string(),
                title: "Will Ethereum 2.0 staking APY exceed 5% in 2024?".to_string(),
                description: "Resolves YES if Ethereum staking APY is above 5% for any 30-day period in 2024".to_string(),
                category: "Crypto".to_string(),
                close_date: chrono::DateTime::parse_from_rfc3339("2024-12-15T23:59:59Z").unwrap().with_timezone(&Utc),
                probability_yes: 0.42,
                probability_no: 0.58,
                volume: 890000.0,
                liquidity: 67000.0,
                participants: 634,
                status: "active".to_string(),
            },
            PredictionMarket {
                id: "ai-regulation-2024".to_string(),
                title: "Will the US pass major AI regulation in 2024?".to_string(),
                description: "Resolves YES if comprehensive AI regulation bill is signed into law in the United States by December 31, 2024".to_string(),
                category: "Politics".to_string(),
                close_date: chrono::DateTime::parse_from_rfc3339("2024-12-31T23:59:59Z").unwrap().with_timezone(&Utc),
                probability_yes: 0.78,
                probability_no: 0.22,
                volume: 1200000.0,
                liquidity: 145000.0,
                participants: 892,
                status: "active".to_string(),
            },
            PredictionMarket {
                id: "market-crash-2024".to_string(),
                title: "Will S&P 500 drop below 4000 in 2024?".to_string(),
                description: "Resolves YES if S&P 500 index closes below 4000 on any trading day in 2024".to_string(),
                category: "Finance".to_string(),
                close_date: chrono::DateTime::parse_from_rfc3339("2024-12-30T23:59:59Z").unwrap().with_timezone(&Utc),
                probability_yes: 0.23,
                probability_no: 0.77,
                volume: 3400000.0,
                liquidity: 280000.0,
                participants: 2156,
                status: "active".to_string(),
            },
            PredictionMarket {
                id: "neural-breakthrough".to_string(),
                title: "Will GPT-5 be released by OpenAI in 2024?".to_string(),
                description: "Resolves YES if OpenAI officially releases a model called GPT-5 by December 31, 2024".to_string(),
                category: "Technology".to_string(),
                close_date: chrono::DateTime::parse_from_rfc3339("2024-12-31T23:59:59Z").unwrap().with_timezone(&Utc),
                probability_yes: 0.55,
                probability_no: 0.45,
                volume: 980000.0,
                liquidity: 125000.0,
                participants: 743,
                status: "active".to_string(),
            },
        ])
    }
    
    pub async fn get_market_by_id(&self, market_id: &str) -> Result<PredictionMarket> {
        let markets = self.get_prediction_markets().await?;
        
        markets.into_iter()
            .find(|m| m.id == market_id)
            .ok_or_else(|| anyhow::anyhow!("Market not found: {}", market_id))
    }
    
    pub async fn analyze_market(&self, market_id: &str) -> Result<MarketAnalysis> {
        let market = self.get_market_by_id(market_id).await?;
        
        // Generate AI-based market analysis
        let sentiment_score = (market.probability_yes - 0.5) * 2.0; // Convert to -1 to 1 scale
        let volume_trend = if market.volume > 1000000.0 { "High" } else { "Medium" }.to_string();
        let price_momentum = market.probability_yes - 0.5; // Momentum based on deviation from 50%
        
        // Calculate expected value and Kelly criterion
        let expected_value = market.probability_yes * 1.0 + market.probability_no * 0.0 - 1.0;
        let kelly_criterion = (market.probability_yes * 2.0 - 1.0).max(0.0);
        
        let recommendation = if kelly_criterion > 0.1 {
            "STRONG BUY"
        } else if kelly_criterion > 0.05 {
            "BUY"
        } else if expected_value > 0.0 {
            "WEAK BUY"
        } else {
            "AVOID"
        }.to_string();
        
        let confidence = if market.volume > 1000000.0 { 0.85 } else { 0.65 };
        let risk_level = if market.probability_yes > 0.8 || market.probability_yes < 0.2 {
            "Low"
        } else {
            "Medium"
        }.to_string();
        
        Ok(MarketAnalysis {
            market_id: market_id.to_string(),
            title: market.title,
            sentiment_score,
            volume_trend,
            price_momentum,
            expected_value,
            kelly_criterion,
            recommendation,
            confidence,
            risk_level,
        })
    }
    
    pub async fn get_positions(&self) -> Result<Vec<MarketPosition>> {
        // Generate sample positions
        Ok(vec![
            MarketPosition {
                market_id: "btc-50k-2024".to_string(),
                market_title: "Will Bitcoin reach $50,000 by end of 2024?".to_string(),
                position: "YES".to_string(),
                shares: 150.0,
                avg_price: 0.62,
                current_value: 150.0 * 0.67,
                pnl: 150.0 * (0.67 - 0.62),
                pnl_percentage: ((0.67 - 0.62) / 0.62) * 100.0,
            },
            MarketPosition {
                market_id: "ai-regulation-2024".to_string(),
                market_title: "Will the US pass major AI regulation in 2024?".to_string(),
                position: "YES".to_string(),
                shares: 200.0,
                avg_price: 0.75,
                current_value: 200.0 * 0.78,
                pnl: 200.0 * (0.78 - 0.75),
                pnl_percentage: ((0.78 - 0.75) / 0.75) * 100.0,
            },
            MarketPosition {
                market_id: "market-crash-2024".to_string(),
                market_title: "Will S&P 500 drop below 4000 in 2024?".to_string(),
                position: "NO".to_string(),
                shares: 100.0,
                avg_price: 0.80,
                current_value: 100.0 * 0.77,
                pnl: 100.0 * (0.77 - 0.80),
                pnl_percentage: ((0.77 - 0.80) / 0.80) * 100.0,
            },
        ])
    }
    
    pub async fn get_portfolio_summary(&self) -> Result<PredictionPortfolio> {
        let positions = self.get_positions().await?;
        
        let total_invested = positions.iter().map(|p| p.shares * p.avg_price).sum::<f64>();
        let current_value = positions.iter().map(|p| p.current_value).sum::<f64>();
        let unrealized_pnl = current_value - total_invested;
        
        let winning_positions = positions.iter().filter(|p| p.pnl > 0.0).count();
        let win_rate = (winning_positions as f64 / positions.len() as f64) * 100.0;
        
        Ok(PredictionPortfolio {
            total_value: current_value,
            invested_amount: total_invested,
            unrealized_pnl,
            realized_pnl: 1250.75, // This would come from historical data
            win_rate,
            active_positions: positions.len() as u32,
            resolved_positions: 12, // This would come from historical data
        })
    }
    
    pub async fn search_markets(&self, query: &str) -> Result<Vec<PredictionMarket>> {
        let all_markets = self.get_prediction_markets().await?;
        
        let query_lower = query.to_lowercase();
        
        Ok(all_markets.into_iter()
            .filter(|market| 
                market.title.to_lowercase().contains(&query_lower) ||
                market.description.to_lowercase().contains(&query_lower) ||
                market.category.to_lowercase().contains(&query_lower)
            )
            .collect())
    }
    
    pub async fn get_trending_markets(&self) -> Result<Vec<PredictionMarket>> {
        let mut markets = self.get_prediction_markets().await?;
        
        // Sort by volume (trending = high volume)
        markets.sort_by(|a, b| b.volume.partial_cmp(&a.volume).unwrap());
        
        Ok(markets.into_iter().take(5).collect())
    }
    
    pub async fn calculate_arbitrage_opportunities(&self) -> Result<Vec<ArbitrageOpportunity>> {
        // This would compare prices across different prediction market platforms
        Ok(vec![
            ArbitrageOpportunity {
                market_title: "Bitcoin $50K 2024".to_string(),
                platform_a: "Polymarket".to_string(),
                platform_b: "Kalshi".to_string(),
                price_a: 0.67,
                price_b: 0.71,
                profit_margin: 0.04,
                risk_level: "Low".to_string(),
            },
        ])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub market_title: String,
    pub platform_a: String,
    pub platform_b: String,
    pub price_a: f64,
    pub price_b: f64,
    pub profit_margin: f64,
    pub risk_level: String,
}